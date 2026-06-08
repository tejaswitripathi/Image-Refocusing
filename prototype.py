"""End-to-end prototype of the refocusing pipeline.

Flow:
  1. Start from a local image (png/jpg/jpeg) + camera f-stop + focal length.
  2. Depth Anything V2 provides per-pixel relative depth, from which a pseudo
     CoC map is built.
  3. SharpenerNet (RendererNet 6->3) turns the (defocused) image + pseudo CoC
     into a sharp image.
  4. The sharp image is shown; click anywhere to refocus to that point. A pseudo
     CoC map focused on the clicked pixel is built and RendererNet (6->3)
     renders a new defocused image.

The three outputs (pseudo CoC map, sharp image, refocused defocused image) are
shown together in an interactive figure.
"""

import io
import boto3

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from skimage.transform import resize

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
SHARPENER_KEY = "defocus-checkpoints/sharpener-net/best_sharpener.pth"
RENDERER_KEY = "defocus-checkpoints/renderer-net/best_renderer.pth"

# The png/jpg/jpeg we start from (treated as a defocused capture).
INPUT_IMAGE_PATH = "cache/P1250745.jpg"

# Camera parameters for the capture.
F_STOP = 5.6
FOCAL_LENGTH_MM = 42.0

# Depth Anything V2 config (used to refocus to an arbitrary clicked point).
DEPTH_ENCODER = "vitb"
DEPTH_CHECKPOINT = "checkpoints/depth_anything_v2_vitb.pth"

# Cap both CoC sources (NN-estimated and Depth-Anything pseudo) at the same
# maximum (in px) so they live in the same range before being fed to the
# sharpener / renderer. The CoC NN tops out around 4 px, so the pseudo CoC is
# scaled to span [0, COC_MAX_PX] as well.
COC_MAX_PX = 4.0

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


sharpener_net = load_model(
    RendererNet(in_channels=6, out_channels=3).to(device), SHARPENER_KEY
)
renderer_net = load_model(
    RendererNet(in_channels=6, out_channels=3).to(device), RENDERER_KEY
)

# Monocular depth estimator (provides per-pixel depth for click-to-refocus).
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
    # RendererNet / SharpenerNet input:
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


def pseudo_coc_norm(focus_y, focus_x):
    # Pseudo CoC map (normalized 512) focused on a pixel, from relative depth.
    # Scaled to span [0, COC_MAX_PX] px so it matches the CoC NN's range.
    pseudo_coc_px = generate_pseudo_coc_from_relative_depth(
        rel_depth,
        focus_y,
        focus_x,
        coc_max_px=COC_MAX_PX,
        blur_strength=1.0,
    )
    pseudo_coc_px = np.clip(pseudo_coc_px, 0, COC_MAX_PX)
    return coc_px_to_norm_512(pseudo_coc_px)


def to_px_display(coc_norm_512):
    return resize(
        coc_norm_512 * COC_PX_MAX,
        (H0, W0),
        order=1,
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)


# Assume the original capture is focused near the image center; this focus
# point is used to build the Depth-Anything pseudo CoC for sharpening.
init_y, init_x = H0 // 2, W0 // 2

# ------------------
# Pseudo CoC (Depth Anything) feeding the sharpener
# ------------------

coc_pseudo_norm = pseudo_coc_norm(init_y, init_x)

# Sharpen the (defocused) input with the pseudo CoC.
sharp = run_rgb_model(sharpener_net, input_rgb, coc_pseudo_norm, (H0, W0))


def refocus(focus_y, focus_x):
    # Refocus uses a Depth-Anything pseudo CoC focused on the clicked pixel,
    # applied to the sharp image via RendererNet.
    coc_norm_new = pseudo_coc_norm(focus_y, focus_x)
    return run_rgb_model(renderer_net, sharp, coc_norm_new, (H0, W0))


# Initial refocus on the image center, so the refocus panel is populated.
refocused = refocus(init_y, init_x)

# ------------------
# Display: 1x3 grid (Depth Anything pseudo CoC path).
#   [pseudo CoC, Sharp, Refocused]
# Click the Sharp panel to refocus.
# ------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
ax_coc, ax_sharp, ax_refocus = axes

coc_im = ax_coc.imshow(
    to_px_display(coc_pseudo_norm), cmap="inferno", vmin=0, vmax=COC_MAX_PX
)
ax_coc.set_title("Pseudo CoC (Depth Anything, px)")
ax_coc.axis("off")
fig.colorbar(coc_im, ax=ax_coc, fraction=0.046, pad=0.04)

ax_sharp.imshow(sharp)
ax_sharp.set_title("Sharp via Pseudo CoC (click to refocus)")
ax_sharp.axis("off")

refocus_im = ax_refocus.imshow(refocused)
ax_refocus.set_title(f"Refocused via Pseudo CoC @ ({init_x}, {init_y})")
ax_refocus.axis("off")


def onclick(event):
    if event.inaxes is not ax_sharp:
        return
    if event.xdata is None or event.ydata is None:
        return

    x = int(np.clip(round(event.xdata), 0, W0 - 1))
    y = int(np.clip(round(event.ydata), 0, H0 - 1))

    print(f"Clicked ({x}, {y}) -> relative depth {rel_depth[y, x]:.3f}")

    new_refocused = refocus(y, x)

    refocus_im.set_data(new_refocused)
    ax_refocus.set_title(f"Refocused via Pseudo CoC @ ({x}, {y})")

    fig.canvas.draw_idle()


cid = fig.canvas.mpl_connect("button_press_event", onclick)

plt.tight_layout()
print("Click the sharp image to refocus. Close the window to exit.")
plt.show()
