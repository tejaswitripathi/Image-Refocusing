import io
import os
import boto3

from coc_map import getDepth, generate_coc_map, getMetadata
from scipy.signal import fftconvolve
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.metrics import mean_squared_error

import torch
from skimage.transform import resize
from models.RendererNet import RendererNet
from data_preprocessing import (
    F_STOP_MAX,
    FOCAL_LENGTH_MM_MAX,
    COC_PX_MAX,
    TARGET_SIZE,
)

# ------------------
# S3 checkpoint config (matches train.py)
# ------------------

S3_BUCKET = "tejas-blender-bucket"
S3_CHECKPOINT_PREFIX = "defocus-checkpoints/renderer-net"
S3_BEST_KEY = f"{S3_CHECKPOINT_PREFIX}/best_renderer.pth"


def load_checkpoint_from_s3(s3_key, map_location):
    # Pull the best checkpoint straight from S3 into memory (no local file).
    s3 = boto3.client("s3")
    buffer = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, s3_key, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location=map_location)


def download_sample_from_s3(s3_prefix, local_cache_dir="cache"):
    # Download all files for one sample (sharp.png, defocused.png, depth_*.exr,
    # metadata.json, ...) so getMetadata/generate_coc_map can read them on disk.
    # Returns a local datadir (with a trailing slash) mirroring the S3 layout.
    s3 = boto3.client("s3")

    s3_prefix = s3_prefix.rstrip("/") + "/"
    sample_name = s3_prefix.rstrip("/").split("/")[-1]

    local_folder = os.path.join(local_cache_dir, sample_name)
    os.makedirs(local_folder, exist_ok=True)

    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_prefix)

    for obj in response.get("Contents", []):
        key = obj["Key"]
        filename = key.split("/")[-1]

        if not filename:
            continue

        local_path = os.path.join(local_folder, filename)

        if not os.path.exists(local_path):
            s3.download_file(S3_BUCKET, key, local_path)

    return local_folder + "/"

def disk_kernel(radius):
    radius = max(float(radius), 0.5)
    r = int(np.ceil(radius))
    y, x = np.ogrid[-r:r+1, -r:r+1]
    mask = x*x + y*y <= radius*radius
    kernel = mask.astype(np.float32)
    kernel /= kernel.sum()
    return kernel

def apply_disk_blur(img, radius):
    k = disk_kernel(radius)
    out = np.zeros_like(img)

    for c in range(3):
        out[..., c] = fftconvolve(img[..., c], k, mode="same")

    return out

def evaluate_metrics(recreated, target):
    recreated_eval = np.clip(recreated, 0, 1).astype(np.float32)
    target_eval = np.clip(target, 0, 1).astype(np.float32)

    psnr = peak_signal_noise_ratio(
        target_eval,
        recreated_eval,
        data_range=1.0
    )

    ssim = structural_similarity(
        target_eval,
        recreated_eval,
        channel_axis=-1,
        data_range=1.0
    )

    return psnr, ssim

def display_sharp_recreated_target(sharp, recreated, target, coc_px=None):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(sharp)
    plt.title("Sharp")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(recreated)
    plt.title("NN Defocused")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(target)
    plt.title("Ground Truth Defocused")
    plt.axis("off")

    # CoC heatmap underneath the NN defocused panel.
    if coc_px is not None:
        ax = plt.subplot(2, 3, 5)
        im = ax.imshow(coc_px, cmap="inferno")
        ax.set_title("CoC Map (px)")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

def plot_metrics(scales, psnr_data, ssim_data):
    fig, ax = plt.subplots()
    ax.scatter(scales, psnr_data, color='blue', label='PSNR trend')
    ax.scatter(scales, ssim_data, color='red', marker='s', label='SSIM trend')

    ax.set_title("CoC Radii Scales vs. PSNR & SSIM")
    ax.legend()
    plt.show()

def recreate_with_disk_blur(sharp, radius_map, num_bins=96):
    bins = np.linspace(0, radius_map.max(), num_bins + 1)
    out = np.zeros_like(sharp)

    for i in range(num_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (radius_map >= lo) & (radius_map < hi)

        if not np.any(mask):
            continue

        radius = (lo + hi) / 2.0

        if radius < 0.5:
            blurred = sharp
        else:
            blurred = apply_disk_blur(sharp, radius)

        out[mask] = blurred[mask]

    return np.clip(out, 0, 1)

def recreate_defocused_with_model(datadir, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    metadata = getMetadata(datadir)

    # ----------------------------
    # Sharp input RGB
    # ----------------------------

    sharp = np.array(
        Image.open(datadir + "sharp.png").convert("RGB")
    ).astype(np.float32) / 255.0

    original_h, original_w = sharp.shape[:2]

    target_size = TARGET_SIZE

    sharp_rs = resize(
        sharp,
        (target_size, target_size),
        anti_aliasing=True,
        preserve_range=True
    ).astype(np.float32)

    sharp_chw = np.transpose(sharp_rs, (2, 0, 1))  # [3, H, W]

    # ----------------------------
    # CoC map (computed analytically from depth + metadata)
    # ----------------------------

    coc_px = generate_coc_map(metadata).astype(np.float32)
    coc_norm = np.clip(coc_px, 0, COC_PX_MAX) / COC_PX_MAX

    coc_norm = resize(
        coc_norm,
        (target_size, target_size),
        order=1,
        anti_aliasing=True,
        preserve_range=True
    ).astype(np.float32)

    # CoC in pixels at the original resolution, for visualization.
    coc_px_display = resize(
        coc_px,
        (original_h, original_w),
        order=1,
        anti_aliasing=True,
        preserve_range=True
    ).astype(np.float32)

    # ----------------------------
    # Conditioning maps (same normalization as training)
    # ----------------------------

    f_stop = metadata["f_stop"] / F_STOP_MAX
    fstop_map = np.ones((1, target_size, target_size), dtype=np.float32) * f_stop

    # getMetadata returns focal length in meters; convert back to mm.
    focal_length = (metadata["focal_length_m"] * 1000.0) / FOCAL_LENGTH_MM_MAX
    focal_map = np.ones((1, target_size, target_size), dtype=np.float32) * focal_length

    coc_channel = coc_norm[None, :, :]

    # ----------------------------
    # Input tensor: sharp RGB (3) + f-stop (1) + focal (1) + CoC (1) = 6 ch
    # ----------------------------

    x = np.concatenate(
        [sharp_chw, fstop_map, focal_map, coc_channel],
        axis=0
    ).astype(np.float32)

    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    x = torch.from_numpy(x)[None, ...].to(device)  # [1, 6, 512, 512]

    # ----------------------------
    # Model + best params from S3
    # ----------------------------

    model = RendererNet(in_channels=6, out_channels=3).to(device)
    checkpoint = load_checkpoint_from_s3(S3_BEST_KEY, map_location=device)

    # supports either raw state_dict or full checkpoint dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Loaded best checkpoint from epoch {checkpoint.get('epoch')} "
            f"(val MSE: {checkpoint.get('val_loss')})"
        )
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    with torch.no_grad():
        pred = model(x)

    pred = pred.cpu().numpy()[0]            # [3, H, W]
    pred = np.transpose(pred, (1, 2, 0))    # [H, W, 3]
    pred = np.clip(pred, 0, 1)

    # resize back to the original resolution for comparison/display
    pred = resize(
        pred,
        (original_h, original_w),
        anti_aliasing=True,
        preserve_range=True
    ).astype(np.float32)

    return sharp, pred, coc_px_display


S3_SAMPLE_PREFIX = "defocus-dataset/bedroom/dataset/img_00000_f1.2_fl35.0_fd2.77/"

datadir = download_sample_from_s3(S3_SAMPLE_PREFIX)

sharp, nn_defocused, coc_px = recreate_defocused_with_model(datadir)

gt_defocused = np.array(
    Image.open(datadir + "defocused.png").convert("RGB")
).astype(np.float32) / 255.0

# ----------------------------
# Error metrics (NN defocused vs. ground-truth defocused)
# ----------------------------

mse = mean_squared_error(
    np.clip(gt_defocused, 0, 1),
    np.clip(nn_defocused, 0, 1)
)
psnr, ssim = evaluate_metrics(nn_defocused, gt_defocused)

print(f"MSE:  {mse:.6f}")
print(f"RMSE: {np.sqrt(mse):.6f}")
print(f"PSNR: {psnr:.3f} dB")
print(f"SSIM: {ssim:.4f}")

display_sharp_recreated_target(sharp, nn_defocused, gt_defocused, coc_px)

# num_bins = 96

# sharp_filepath = datadir + "sharp.png"
# defocused_filepath = datadir + "defocused.png"

# sharp = np.array(Image.open(sharp_filepath).convert("RGB")).astype(np.float32) / 255.0
# target = np.array(Image.open(defocused_filepath).convert("RGB")).astype(np.float32) / 255.0

# # scales = np.linspace(1.95, 2.05, 11)
# # psnr_data = []
# # ssim_data = []

# data = []

# # for scale in scales:
# radius_map = coc_px / 2.01
# radius_map = np.clip(radius_map, 0, 25)

# recreated = recreate_with_disk_blur(sharp, radius_map)

# psnr, ssim = evaluate_metrics(recreated, target)
# print(f"PSNR: {psnr:.3f}")
# print(f"SSIM: {ssim:.3f}")

# display_sharp_recreated_target(sharp, recreated, target)