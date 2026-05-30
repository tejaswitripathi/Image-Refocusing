from coc_map import getDepth, generate_coc_map, getMetadata
from scipy.signal import fftconvolve
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.metrics import mean_squared_error

import torch
from skimage.transform import resize
from models.UNet import UNet

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

def display_sharp_recreated_target(sharp, recreated, target):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(sharp)
    plt.title("Sharp")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(recreated)
    plt.title("Recreated Defocus")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(target)
    plt.title("Blender Defocused")
    plt.axis("off")

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

def predict_coc_with_model(datadir, checkpoint_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    metadata = getMetadata(datadir)

    defocused_filepath = datadir + "defocused.png"
    defocused = np.array(Image.open(defocused_filepath).convert("RGB")).astype(np.float32) / 255.0

    original_h, original_w = defocused.shape[:2]

    # model was trained on 512x512 full-frame resized images
    target_size = 512

    rgb = resize(
        defocused,
        (target_size, target_size),
        anti_aliasing=True,
        preserve_range=True
    ).astype(np.float32)

    rgb = np.transpose(rgb, (2, 0, 1))  # [3, H, W]

    f_stop = metadata["f_stop"] / 8.0
    fstop_map = np.ones((1, target_size, target_size), dtype=np.float32) * f_stop

    focal_length = metadata["focal_length_m"] * 1000 / 135.0
    focal_map = np.ones((1, target_size, target_size), dtype=np.float32) * focal_length

    x = np.concatenate([rgb, fstop_map, focal_map], axis=0).astype(np.float32)
    x = torch.from_numpy(x)[None, ...].to(device)  # [1, 5, 512, 512]

    model = UNet(in_channels=5, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # supports either raw state_dict or checkpoint dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    with torch.no_grad():
        pred_norm = model(x)

    pred_norm = pred_norm.cpu().numpy()[0, 0]
    pred_norm = np.clip(pred_norm, 0, 1)

    # convert normalized CoC back to pixels
    pred_coc_px_512 = pred_norm * 25.0

    # resize back to original resolution
    pred_coc_px = resize(
        pred_coc_px_512,
        (original_h, original_w),
        order=1,
        anti_aliasing=True,
        preserve_range=True
    ).astype(np.float32)

    return pred_coc_px

datadir = "bedroom/dataset/img_00000_f1.2_fl35_fd3.00/"
checkpoint_path = "models/unet-params/best_unet_coc_2.pth"
metadata = getMetadata(datadir)
coc_px_nn = predict_coc_with_model(datadir, checkpoint_path)
coc_px_def = generate_coc_map(metadata)

diff = coc_px_nn - coc_px_def
abs_diff = np.abs(diff)

print("MSE:", np.mean(diff ** 2))
print("MAE:", np.mean(abs_diff))
print("Max abs diff:", np.max(abs_diff))
print("Median abs diff:", np.median(abs_diff))
print("95th percentile abs diff:", np.percentile(abs_diff, 95))

plt.figure(figsize=(18, 5))

rel_diff = np.abs(coc_px_nn - coc_px_def) / (coc_px_def + 1e-3)

plt.imshow(
    np.clip(rel_diff, 0, 1),
    cmap="inferno"
)
plt.title("Relative Error")
plt.colorbar()
plt.axis("off")
plt.show()

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