import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image
from scipy.ndimage import gaussian_filter

# datadir = "cafe/dataset/img_00000_f1.2_fl50_fd5.61/"

    


def getDepth(datadir):
    
    filepath = datadir + "depth_.exr"

    exr = OpenEXR.InputFile(filepath)

    header = exr.header()

    # print(header["channels"].keys())

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    depth_str = exr.channel("depth.V", FLOAT)

    dw = header["dataWindow"]

    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    depth = np.frombuffer(depth_str, dtype=np.float32)
    depth = depth.reshape((height, width))
    
    return depth, width, height

# print(depth.min(), depth.max())
def getMetadata(datadir):
    depth, width, height = getDepth(datadir)

    metadata_path = datadir + "metadata.json"
    with open(metadata_path, 'r') as file:
        metadata = json.load(file)

    f_stop = metadata["f_stop"]
    focal_length_m = metadata["focal_length_mm"] / 1000
    sensor_width_m = metadata["sensor_width_mm"] / 1000
    focus_distance_m = metadata["focus_distance_m"]

    A = focal_length_m / f_stop

    return {
        "f_stop": f_stop,
        "focal_length_m": focal_length_m,
        "sensor_width_m": sensor_width_m,
        "focus_distance_m": focus_distance_m,
        "A": A,
        "depth": depth,
        "width": width,
        "height": height
    }

# depth_vis = depth.copy()

# depth_vis = np.clip(depth, 0, 30)
# depth_vis -= depth_vis.min()
# depth_vis /= depth_vis.max()
# depth_vis = 1.0 - depth_vis


# depth_vis = (255 * depth_vis).astype(np.uint8)
def generate_coc_map(metadata):

    depth = metadata["depth"]
    width = metadata["width"]

    focus_distance_m = metadata["focus_distance_m"]
    focal_length_m = metadata["focal_length_m"]
    sensor_width_m = metadata["sensor_width_m"]
    A = metadata["A"]

    z = depth.astype(np.float32)

    eps = 1e-6
    z = np.maximum(z, eps)

    coc_m = (
        A
        * np.abs((z - focus_distance_m) / z)
        * (focal_length_m / (focus_distance_m - focal_length_m))
    )

    coc_px = coc_m / sensor_width_m * width

    return coc_px

# coc_px = generate_coc_map(datadir)
# coc_vis = np.clip(coc_px, 0, np.percentile(coc_px, 99))

# coc_vis -= coc_vis.min()
# coc_vis /= coc_vis.max()

# plt.imshow(coc_vis, cmap="magma")
# plt.colorbar(label="CoC radius/diameter px")
# plt.axis("off")
# plt.show()


# sharp_filepath = datadir + "sharp.png"
# defocused_filepath = datadir + "defocused.png"

# sharp = np.array(Image.open(sharp_filepath).convert("RGB")).astype(np.float32) / 255.0
# target = np.array(Image.open(defocused_filepath).convert("RGB")).astype(np.float32) / 255.0

# # CoC is diameter-ish; use radius-ish/sigma-ish approximation
# sigma_map = coc_px / 2.0

# # prevent insane blur outliers
# sigma_map = np.clip(sigma_map, 0, 20)

# # quantize blur levels
# num_bins = 16
# bins = np.linspace(0, sigma_map.max(), num_bins + 1)

# recreated = np.zeros_like(sharp)

# for i in range(num_bins):
#     lo, hi = bins[i], bins[i + 1]
#     mask = (sigma_map >= lo) & (sigma_map < hi)

#     if not np.any(mask):
#         continue

#     sigma = (lo + hi) / 2.0

#     blurred = gaussian_filter(
#         sharp,
#         sigma=(sigma, sigma, 0)
#     )

#     recreated[mask] = blurred[mask]

# recreated = np.clip(recreated, 0, 1)

# plt.figure(figsize=(15, 5))

# plt.subplot(1, 3, 1)
# plt.imshow(sharp)
# plt.title("Sharp")
# plt.axis("off")

# plt.subplot(1, 3, 2)
# plt.imshow(recreated)
# plt.title("Recreated Defocus")
# plt.axis("off")

# plt.subplot(1, 3, 3)
# plt.imshow(target)
# plt.title("Blender Defocused")
# plt.axis("off")

# plt.show()