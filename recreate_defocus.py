from coc_map import getDepth, generate_coc_map, getMetadata
from scipy.signal import fftconvolve
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

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

datadir = "cafe/dataset/img_00000_f1.2_fl50_fd5.61/"
metadata = getMetadata(datadir)
coc_px = generate_coc_map(metadata)
num_bins = 96

sharp_filepath = datadir + "sharp.png"
defocused_filepath = datadir + "defocused.png"

sharp = np.array(Image.open(sharp_filepath).convert("RGB")).astype(np.float32) / 255.0
target = np.array(Image.open(defocused_filepath).convert("RGB")).astype(np.float32) / 255.0

# scales = np.linspace(1.95, 2.05, 11)
# psnr_data = []
# ssim_data = []

# for scale in scales:
radius_map = coc_px / 2.01
radius_map = np.clip(radius_map, 0, 25)
bins = np.linspace(0, radius_map.max(), num_bins + 1)

recreated = np.zeros_like(sharp)

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

    recreated[mask] = blurred[mask]

recreated = np.clip(recreated, 0, 1)

psnr, ssim = evaluate_metrics(recreated, target)
print(f"PSNR: {psnr:.3f}")
print(f"SSIM: {ssim:.3f}")

display_sharp_recreated_target(sharp, recreated, target)
