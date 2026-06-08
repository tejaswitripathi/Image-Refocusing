# depth_anything_inference.py

import sys
import cv2
import torch
import numpy as np

sys.path.append("external/Depth-Anything-V2")

from depth_anything_v2.dpt import DepthAnythingV2


def load_depth_anything(encoder="vitb", checkpoint_path="checkpoints/depth_anything_v2_vitb.pth"):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device).eval()

    return model, device


def predict_relative_depth(image_path, model, normalize=True):
    raw_img = cv2.imread(image_path)

    if raw_img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    depth = model.infer_image(raw_img)  # H x W numpy array

    depth = depth.astype(np.float32)

    if normalize:
        depth -= depth.min()
        depth /= depth.max() + 1e-8

    return depth

def generate_pseudo_coc_from_relative_depth(
    rel_depth,
    focus_y,
    focus_x,
    coc_max_px=25.0,
    blur_strength=1.0
):
    focus_depth = rel_depth[focus_y, focus_x]

    pseudo_coc = np.abs(rel_depth - focus_depth)

    pseudo_coc -= pseudo_coc.min()
    pseudo_coc /= pseudo_coc.max() + 1e-8

    pseudo_coc = pseudo_coc * coc_max_px * blur_strength

    return pseudo_coc.astype(np.float32)