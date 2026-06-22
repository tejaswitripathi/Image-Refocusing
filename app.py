"""Gradio demo: fixed shallow depth-of-field rendering.

Mirrors `prototype.py` but as a Hugging Face Spaces app:
  1. Depth Anything V2 (via `transformers`) estimates relative depth.
  2. A pseudo CoC map (focused near the image center) is built from that depth.
  3. RendererNet renders the image at a chosen f-stop, then the render is blended
     back onto the original using a CoC weight, so in-focus regions (CoC ~ 0) are
     left untouched.
  4. A non-NN baseline (flat Gaussian blur on the background, CoC > threshold) is
     produced for comparison.

The Depth Anything checkpoint is pulled from the Hub by `transformers`, so the
`external/` code and the local `.pth` are not required. The RendererNet weights
are resolved (in order) from a local file, the Hugging Face Hub, or S3 -- see
`load_renderer_weights`.
"""

import os

import numpy as np
from PIL import Image
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize
from skimage.filters import gaussian

import gradio as gr
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


# ------------------
# Config (env-overridable so the Space can be configured without code edits)
# ------------------

# Depth Anything V2 checkpoint on the Hub. Swap to "...-Small-hf" for a faster
# (lower quality) model on CPU Spaces, or "...-Large-hf" for best quality.
DEPTH_MODEL_ID = os.environ.get(
    "DEPTH_MODEL_ID", "depth-anything/Depth-Anything-V2-Base-hf"
)

# RendererNet weights. Provide ONE of:
#   * RENDERER_LOCAL_PATH        -> a local .pth bundled with the Space
#   * RENDERER_REPO_ID (+FILE)   -> a Hugging Face model repo (uses HF_TOKEN if private)
#   * S3 env creds + bucket/key  -> fall back to the original S3 location
RENDERER_LOCAL_PATH = os.environ.get("RENDERER_LOCAL_PATH", "best_renderer.pth")
RENDERER_REPO_ID = os.environ.get("RENDERER_REPO_ID", "")
RENDERER_FILENAME = os.environ.get("RENDERER_FILENAME", "best_renderer.pth")
S3_BUCKET = os.environ.get("RENDERER_S3_BUCKET", "tejas-blender-bucket")
S3_KEY = os.environ.get(
    "RENDERER_S3_KEY", "defocus-checkpoints/renderer-net/best_renderer.pth"
)

# Training-time normalization constants (must match how RendererNet was trained).
F_STOP_MAX = 22.0
FOCAL_LENGTH_MM_MAX = 200.0
COC_PX_NORM = 25.0  # CoC channel was trained as clip(coc_px, 0, 25) / 25
TARGET_SIZE = 512  # spatial size RendererNet runs at

# Pseudo CoC cap (px) -- the depth-derived CoC spans [0, COC_MAX_PX].
COC_MAX_PX = 4.0

# Cap the working resolution so CPU inference stays responsive.
MAX_SIDE = 1024

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# ------------------
# RendererNet (inlined so the Space is self-contained)
# ------------------

def double_convolution(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


class RendererNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_convolution_1 = double_convolution(in_channels, 64)
        self.down_convolution_2 = double_convolution(64, 128)
        self.down_convolution_3 = double_convolution(128, 256)
        self.down_convolution_4 = double_convolution(256, 512)
        self.down_convolution_5 = double_convolution(512, 1024)

        self.up_transpose_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_convolution_1 = double_convolution(1024, 512)
        self.up_transpose_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_convolution_2 = double_convolution(512, 256)
        self.up_transpose_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_convolution_3 = double_convolution(256, 128)
        self.up_transpose_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_convolution_4 = double_convolution(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        down_1 = self.down_convolution_1(x)
        down_2 = self.max_pool2d(down_1)
        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool2d(down_3)
        down_5 = self.down_convolution_3(down_4)
        down_6 = self.max_pool2d(down_5)
        down_7 = self.down_convolution_4(down_6)
        down_8 = self.max_pool2d(down_7)
        down_9 = self.down_convolution_5(down_8)

        up_1 = self.up_transpose_1(down_9)
        x = self.up_convolution_1(torch.cat([down_7, up_1], 1))
        up_2 = self.up_transpose_2(x)
        x = self.up_convolution_2(torch.cat([down_5, up_2], 1))
        up_3 = self.up_transpose_3(x)
        x = self.up_convolution_3(torch.cat([down_3, up_3], 1))
        up_4 = self.up_transpose_4(x)
        x = self.up_convolution_4(torch.cat([down_1, up_4], 1))
        return torch.sigmoid(self.out(x))


# ------------------
# Weight loading
# ------------------

def _strip_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def load_renderer_weights(map_location):
    # 1) Local file bundled with the Space.
    if RENDERER_LOCAL_PATH and os.path.exists(RENDERER_LOCAL_PATH):
        print(f"Loading RendererNet weights from local file {RENDERER_LOCAL_PATH}")
        return _strip_state_dict(
            torch.load(RENDERER_LOCAL_PATH, map_location=map_location)
        )

    # 2) Hugging Face Hub model repo.
    if RENDERER_REPO_ID:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=RENDERER_REPO_ID,
            filename=RENDERER_FILENAME,
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"Loading RendererNet weights from hub {RENDERER_REPO_ID}/{RENDERER_FILENAME}")
        return _strip_state_dict(torch.load(path, map_location=map_location))

    # 3) S3 (requires AWS credentials in the environment).
    try:
        import io
        import boto3

        s3 = boto3.client("s3")
        buffer = io.BytesIO()
        s3.download_fileobj(S3_BUCKET, S3_KEY, buffer)
        buffer.seek(0)
        print(f"Loading RendererNet weights from s3://{S3_BUCKET}/{S3_KEY}")
        return _strip_state_dict(torch.load(buffer, map_location=map_location))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not locate RendererNet weights. Set RENDERER_LOCAL_PATH, "
            "RENDERER_REPO_ID (+RENDERER_FILENAME), or provide S3 credentials."
        ) from exc


print("Loading RendererNet ...")
renderer_net = RendererNet(in_channels=6, out_channels=3).to(device)
renderer_net.load_state_dict(load_renderer_weights(device))
renderer_net.eval()

print(f"Loading Depth Anything V2 ({DEPTH_MODEL_ID}) ...")
depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(device)
depth_model.eval()
print("Models ready.")


# ------------------
# Pipeline helpers
# ------------------

def fit_to_max_side(rgb, max_side):
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return rgb
    scale = max_side / float(longest)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    return resize(
        rgb, (new_h, new_w), anti_aliasing=True, preserve_range=True
    ).astype(np.float32)


@torch.no_grad()
def predict_relative_depth(rgb):
    # rgb: float [0,1] HxWx3. Returns normalized [0,1] relative depth at HxW.
    h, w = rgb.shape[:2]
    pil = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))

    inputs = depth_processor(images=pil, return_tensors="pt").to(device)
    depth = depth_model(**inputs).predicted_depth  # [1, h', w']

    depth = F.interpolate(
        depth.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
    )[0, 0].cpu().numpy().astype(np.float32)

    depth -= depth.min()
    depth /= depth.max() + 1e-8
    return depth


def pseudo_coc_px(rel_depth, focus_y, focus_x):
    focus_depth = rel_depth[focus_y, focus_x]
    coc = np.abs(rel_depth - focus_depth)
    coc -= coc.min()
    coc /= coc.max() + 1e-8
    coc = coc * COC_MAX_PX
    return np.clip(coc, 0, COC_MAX_PX).astype(np.float32)


def coc_px_to_norm_512(coc_px):
    coc_norm = np.clip(coc_px, 0, COC_PX_NORM) / COC_PX_NORM
    return resize(
        coc_norm,
        (TARGET_SIZE, TARGET_SIZE),
        order=1,
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)


def coc_blend_weight(coc_px, focus_threshold_px):
    span = max(COC_MAX_PX - focus_threshold_px, 1e-6)
    t = np.clip((coc_px - focus_threshold_px) / span, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def make_param_maps(size, f_stop, focal_length_mm):
    fstop_map = np.ones((1, size, size), dtype=np.float32) * (f_stop / F_STOP_MAX)
    focal_map = np.ones((1, size, size), dtype=np.float32) * (
        focal_length_mm / FOCAL_LENGTH_MM_MAX
    )
    return fstop_map, focal_map


@torch.no_grad()
def run_renderer(rgb, coc_norm_512, f_stop, focal_length_mm, out_size):
    rs = resize(
        rgb, (TARGET_SIZE, TARGET_SIZE), anti_aliasing=True, preserve_range=True
    ).astype(np.float32)
    chw = np.transpose(rs, (2, 0, 1))

    fstop_map, focal_map = make_param_maps(TARGET_SIZE, f_stop, focal_length_mm)
    coc_channel = coc_norm_512[None, :, :]

    x = np.concatenate([chw, fstop_map, focal_map, coc_channel], axis=0)[None]
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    x = torch.from_numpy(x).to(device)

    out = renderer_net(x)[0].cpu().numpy()
    out = np.clip(np.transpose(out, (1, 2, 0)), 0, 1)
    return resize(
        out, out_size, anti_aliasing=True, preserve_range=True
    ).astype(np.float32)


def colorize_coc(coc_px):
    norm = np.clip(coc_px / COC_MAX_PX, 0, 1)
    rgba = cm.inferno(norm)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def to_uint8(rgb):
    return (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)


# ------------------
# Main inference entrypoint
# ------------------

def render(
    image,
    f_stop,
    focal_length_mm,
    focus_threshold_px,
    gaussian_threshold_px,
    gaussian_sigma_px,
):
    if image is None:
        return None, None, None

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    rgb = fit_to_max_side(rgb, MAX_SIDE)
    h, w = rgb.shape[:2]

    rel_depth = predict_relative_depth(rgb)

    # Focus near the image center (no point selection, like prototype.py).
    focus_y, focus_x = h // 2, w // 2
    coc_px = pseudo_coc_px(rel_depth, focus_y, focus_x)

    # NN render blended back onto the original by CoC weight.
    coc_norm_512 = coc_px_to_norm_512(coc_px)
    nn_render = run_renderer(rgb, coc_norm_512, f_stop, focal_length_mm, (h, w))
    weight = coc_blend_weight(coc_px, focus_threshold_px)[:, :, None]
    blended = np.clip((1.0 - weight) * rgb + weight * nn_render, 0, 1)

    # Non-NN baseline: flat Gaussian blur where CoC exceeds the threshold.
    blurred = gaussian(
        rgb, sigma=gaussian_sigma_px, channel_axis=-1, preserve_range=True
    ).astype(np.float32)
    mask = (coc_px > gaussian_threshold_px)[:, :, None].astype(np.float32)
    gaussian_render = np.clip((1.0 - mask) * rgb + mask * blurred, 0, 1)

    return (
        Image.fromarray(to_uint8(blended)),
        Image.fromarray(to_uint8(gaussian_render)),
        Image.fromarray(colorize_coc(coc_px)),
    )


# ------------------
# Gradio UI
# ------------------

with gr.Blocks(title="Shallow Depth-of-Field Renderer") as demo:
    gr.Markdown(
        "# Physically-Based Portrait Mode Engine\n"
        "Upload an image and render it at a chosen f-stop. Depth Anything V2 "
        "builds a pseudo circle-of-confusion (CoC) map (focused at the image "
        "center); RendererNet blurs the out-of-focus regions while in-focus "
        "areas (CoC near 0) stay untouched. A flat-Gaussian baseline is shown "
        "for comparison."
    )

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="pil", label="Input image")
            f_stop = gr.Slider(0.95, 22.0, value=1.2, step=0.05, label="f-stop")
            focal_length = gr.Slider(
                4.0, 200.0, value=24.0, step=0.5, label="Focal length (mm)"
            )
            focus_threshold = gr.Slider(
                0.0, COC_MAX_PX, value=0.4, step=0.05,
                label="In-focus CoC threshold (px) - below this the NN is suppressed",
            )
            with gr.Accordion("Gaussian baseline", open=False):
                gaussian_threshold = gr.Slider(
                    0.0, COC_MAX_PX, value=1.0, step=0.05,
                    label="Background CoC threshold (px)",
                )
                gaussian_sigma = gr.Slider(
                    1.0, 30.0, value=12.0, step=1.0, label="Gaussian sigma (px)"
                )
            run_btn = gr.Button("Render", variant="primary")

        with gr.Column(scale=2):
            out_render = gr.Image(label="Rendered (NN)")
            with gr.Row():
                out_gaussian = gr.Image(label="Gaussian background baseline")
                out_coc = gr.Image(label="Pseudo CoC map")

    run_btn.click(
        fn=render,
        inputs=[
            inp,
            f_stop,
            focal_length,
            focus_threshold,
            gaussian_threshold,
            gaussian_sigma,
        ],
        outputs=[out_render, out_gaussian, out_coc],
    )

    _example_dir = "cache"
    if os.path.isdir(_example_dir):
        _examples = [
            os.path.join(_example_dir, f)
            for f in sorted(os.listdir(_example_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ][:4]
        if _examples:
            gr.Examples(examples=_examples, inputs=inp)


if __name__ == "__main__":
    demo.launch()
