"""Render a fixed shallow depth-of-field look (no interactive refocusing).

Flow:
  1. Start from a local image (png/jpg/jpeg) treated as the original capture.
  2. Depth Anything V2 provides per-pixel relative depth, from which a pseudo
     CoC map (focused near the image center) is built.
  3. RendererNet (6->3) is fed the original image, a fixed f-stop of 1.2 and the
     pseudo CoC map, producing a strongly defocused render.
  4. The render is blended back onto the original using a CoC-derived weight:
     where the CoC is close to 0 (in focus) the NN has no effect and the output
     equals the original png; where the CoC grows the NN render takes over.

There is no point selection / refocusing: the image is simply rendered at the
configured f-stop. The three outputs (original, pseudo CoC map, blended render)
are shown together.
"""

import io
import os
import boto3

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from skimage.transform import resize
from skimage.filters import gaussian

from models.RendererNet import RendererNet
from data_preprocessing import (
    F_STOP_MAX,
    FOCAL_LENGTH_MM_MAX,
    COC_PX_MAX,
    TARGET_SIZE,
)
from depth_anything_inference import (
    load_depth_anything,
    predict_relative_depth,
    generate_pseudo_coc_from_relative_depth,
)

# ------------------
# Config
# ------------------

S3_BUCKET = "tejas-blender-bucket"
RENDERER_KEY = "defocus-checkpoints/renderer-net/best_renderer.pth"

# The png/jpg/jpeg we start from (treated as the original capture).
INPUT_IMAGE_PATH = "cache/IMG_4171.jpg"

# Directory where rendered images are written.
OUTPUT_DIR = "outputs"

# Fixed shallow depth-of-field. f/1.2 is passed straight into the renderer so it
# blurs the out-of-focus regions aggressively.
F_STOP = 1.2
FOCAL_LENGTH_MM = 6.765

# Depth Anything V2 config (used to build the pseudo CoC map).
DEPTH_ENCODER = "vitb"
DEPTH_CHECKPOINT = "checkpoints/depth_anything_v2_vitb.pth"

# Cap the pseudo CoC at the same maximum (in px) the renderer was trained
# against so it lives in a consistent range.
COC_MAX_PX = 4.0

# CoC-weighted blend: below COC_FOCUS_THRESHOLD_PX the render is treated as fully
# in focus and the original png is kept untouched (NN weight 0). The weight then
# smoothly ramps to 1 by COC_MAX_PX, letting the NN render dominate where the
# scene is most defocused.
COC_FOCUS_THRESHOLD_PX = 0.4

# Non-NN baseline: a flat Gaussian blur applied to the background. Every pixel
# whose CoC exceeds GAUSSIAN_COC_THRESHOLD_PX is replaced with a uniformly
# blurred (GAUSSIAN_SIGMA_PX) version; everything else keeps the original.
GAUSSIAN_COC_THRESHOLD_PX = 1.0
GAUSSIAN_SIGMA_PX = 12.0

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# ------------------
# Model loading (weights from S3)
# ------------------

def load_checkpoint_from_s3(s3_key, map_location):
    s3 = boto3.client("s3")
    buffer = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, s3_key, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location=map_location)


def load_model(model, s3_key):
    checkpoint = load_checkpoint_from_s3(s3_key, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"Loaded weights from s3://{S3_BUCKET}/{s3_key}")
    return model


renderer_net = load_model(
    RendererNet(in_channels=6, out_channels=3).to(device), RENDERER_KEY
)

# Monocular depth estimator (provides per-pixel depth for the pseudo CoC).
depth_model, depth_device = load_depth_anything(
    encoder=DEPTH_ENCODER, checkpoint_path=DEPTH_CHECKPOINT
)
print(f"Loaded Depth Anything V2 ({DEPTH_ENCODER}) on {depth_device}")


# ------------------
# Preprocessing helpers
# ------------------

def make_param_maps(size):
    # Constant f-stop / focal-length planes, normalized exactly as in training.
    fstop_map = np.ones((1, size, size), dtype=np.float32) * (F_STOP / F_STOP_MAX)
    focal_map = np.ones((1, size, size), dtype=np.float32) * (
        FOCAL_LENGTH_MM / FOCAL_LENGTH_MM_MAX
    )
    return fstop_map, focal_map


def to_chw_resized(rgb, size):
    rs = resize(
        rgb, (size, size), anti_aliasing=True, preserve_range=True
    ).astype(np.float32)
    return np.transpose(rs, (2, 0, 1))


@torch.no_grad()
def run_rgb_model(model, rgb, coc_norm_512, out_size):
    # RendererNet input:
    # [RGB (3), f-stop (1), focal (1), CoC (1)] = 6 channels.
    chw = to_chw_resized(rgb, TARGET_SIZE)
    fstop_map, focal_map = make_param_maps(TARGET_SIZE)
    coc_channel = coc_norm_512[None, :, :]

    x = np.concatenate(
        [chw, fstop_map, focal_map, coc_channel], axis=0
    )[None].astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    x = torch.from_numpy(x).to(device)

    out = model(x)[0].cpu().numpy()
    out = np.transpose(out, (1, 2, 0))
    out = np.clip(out, 0, 1)

    out = resize(
        out, out_size, anti_aliasing=True, preserve_range=True
    ).astype(np.float32)
    return out


def coc_px_to_norm_512(coc_px):
    coc_norm = np.clip(coc_px, 0, COC_PX_MAX) / COC_PX_MAX
    coc_norm = resize(
        coc_norm,
        (TARGET_SIZE, TARGET_SIZE),
        order=1,
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)
    return coc_norm


# ------------------
# Input image + monocular relative depth
# ------------------

input_rgb = np.array(
    Image.open(INPUT_IMAGE_PATH).convert("RGB")
).astype(np.float32) / 255.0

H0, W0 = input_rgb.shape[:2]

# Monocular relative depth of the input image, at the displayed resolution.
rel_depth = predict_relative_depth(INPUT_IMAGE_PATH, depth_model, normalize=True)
rel_depth = resize(
    rel_depth, (H0, W0), order=1, anti_aliasing=True, preserve_range=True
).astype(np.float32)


def pseudo_coc_px(focus_y, focus_x):
    # Pseudo CoC map (in px, at the display resolution) focused on a pixel,
    # built from relative depth and scaled to span [0, COC_MAX_PX].
    coc_px = generate_pseudo_coc_from_relative_depth(
        rel_depth,
        focus_y,
        focus_x,
        coc_max_px=COC_MAX_PX,
        blur_strength=1.0,
    )
    return np.clip(coc_px, 0, COC_MAX_PX).astype(np.float32)


def coc_blend_weight(coc_px):
    # NN weight derived from the CoC: 0 where the scene is in focus (CoC near 0)
    # so the original png is preserved, smoothly ramping to 1 as blur grows.
    span = max(COC_MAX_PX - COC_FOCUS_THRESHOLD_PX, 1e-6)
    t = np.clip((coc_px - COC_FOCUS_THRESHOLD_PX) / span, 0.0, 1.0)
    # Smoothstep for a gentle in-focus -> out-of-focus transition.
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def render_fixed_fstop(focus_y, focus_x):
    # Render the original at the configured f-stop, then blend back onto the
    # original using the CoC weight so in-focus regions are untouched.
    coc_px = pseudo_coc_px(focus_y, focus_x)
    coc_norm_512 = coc_px_to_norm_512(coc_px)

    nn_render = run_rgb_model(renderer_net, input_rgb, coc_norm_512, (H0, W0))

    weight = coc_blend_weight(coc_px)[:, :, None]
    blended = (1.0 - weight) * input_rgb + weight * nn_render
    return np.clip(blended, 0, 1).astype(np.float32), coc_px


def gaussian_background(coc_px):
    # Non-NN baseline: blur the whole image uniformly, then keep that blur only
    # where CoC exceeds the threshold (the "background"); elsewhere stay sharp.
    blurred = gaussian(
        input_rgb, sigma=GAUSSIAN_SIGMA_PX, channel_axis=-1, preserve_range=True
    ).astype(np.float32)

    mask = (coc_px > GAUSSIAN_COC_THRESHOLD_PX)[:, :, None].astype(np.float32)
    composite = (1.0 - mask) * input_rgb + mask * blurred
    return np.clip(composite, 0, 1).astype(np.float32)


def save_render(rgb, suffix):
    # Persist a rendered float [0, 1] image, matching the input dimensions.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if rgb.shape[:2] != (H0, W0):
        rgb = resize(
            rgb, (H0, W0), anti_aliasing=True, preserve_range=True
        ).astype(np.float32)

    out_uint8 = (np.clip(rgb, 0, 1) * 255.0).round().astype(np.uint8)

    base = os.path.splitext(os.path.basename(INPUT_IMAGE_PATH))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base}_{suffix}.png")
    Image.fromarray(out_uint8).save(out_path)
    print(f"Saved rendered image ({W0}x{H0}) to {out_path}")
    return out_path


# Focus near the image center (no point selection); render once at f/1.2.
focus_y, focus_x = H0 // 2, W0 // 2
rendered, coc_px_display = render_fixed_fstop(focus_y, focus_x)
fstop_tag = str(F_STOP).replace(".", "_")
save_render(rendered, f"rendered_f{fstop_tag}")

# Non-NN baseline: flat Gaussian blur on the background (CoC > 1).
gaussian_render = gaussian_background(coc_px_display)
save_render(gaussian_render, "gaussian_bg")


# ------------------
# Display: 1x4 grid.
#   [Original, Pseudo CoC, Rendered @ f/1.2, Gaussian background]
# ------------------

fig, axes = plt.subplots(1, 4, figsize=(24, 6))
ax_orig, ax_coc, ax_render, ax_gauss = axes

ax_orig.imshow(input_rgb)
ax_orig.set_title("Original")
ax_orig.axis("off")

coc_im = ax_coc.imshow(
    coc_px_display, cmap="inferno", vmin=0, vmax=COC_MAX_PX
)
ax_coc.set_title("Pseudo CoC (Depth Anything, px)")
ax_coc.axis("off")
fig.colorbar(coc_im, ax=ax_coc, fraction=0.046, pad=0.04)

ax_render.imshow(rendered)
ax_render.set_title(f"Rendered @ f/{F_STOP}")
ax_render.axis("off")

ax_gauss.imshow(gaussian_render)
ax_gauss.set_title(f"Gaussian bg (CoC > {GAUSSIAN_COC_THRESHOLD_PX})")
ax_gauss.axis("off")

plt.tight_layout()
print(f"Rendered at f/{F_STOP}. Close the window to exit.")
plt.show()
