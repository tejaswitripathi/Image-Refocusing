import os
import json
import boto3
import numpy as np
import torch

from torch.utils.data import Dataset
from skimage import io
from skimage.transform import resize


from coc_map import getMetadata

class DefocusDataset(Dataset):

    def __init__(
        self,
        scenes=("cafe", "grass", "bedroom"),
        bucket_name="tejas-blender-bucket",
        s3_prefix="defocus-dataset",
        local_cache_dir="cache"
    ):

        self.bucket_name = bucket_name
        self.s3_prefix = s3_prefix
        self.local_cache_dir = local_cache_dir

        self.s3 = boto3.client("s3")

        self.samples = []

        # Build sample index
        for scene in scenes:

            prefix = f"{s3_prefix}/{scene}/"

            response = self.s3.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix,
                Delimiter="/"
            )

            if "CommonPrefixes" not in response:
                continue

            for obj in response["CommonPrefixes"]:

                folder_prefix = obj["Prefix"]

                # Example:
                # defocus-dataset/cafe/img_0001/

                sample_name = folder_prefix.rstrip("/").split("/")[-1]

                self.samples.append({
                    "scene": scene,
                    "prefix": folder_prefix,
                    "sample_name": sample_name
                })

        print(f"Found {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def _download_if_missing(self, s3_key, local_path):

        if os.path.exists(local_path):
            return

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # print(f"Downloading {s3_key}")

        self.s3.download_file(
            self.bucket_name,
            s3_key,
            local_path
        )

    def __getitem__(self, idx):

        sample = self.samples[idx]

        scene = sample["scene"]
        prefix = sample["prefix"]
        sample_name = sample["sample_name"]

        local_folder = os.path.join(
            self.local_cache_dir,
            scene,
            sample_name
        )

        os.makedirs(local_folder, exist_ok=True)

        # ----------------------------
        # File paths
        # ----------------------------

        files = {
            "defocused": "defocused.png",
            "metadata": "metadata.json",
            "coc": "coc.json"
        }

        local_paths = {}

        for key, filename in files.items():

            s3_key = prefix + filename

            local_path = os.path.join(
                local_folder,
                filename
            )

            local_paths[key] = local_path

            self._download_if_missing(
                s3_key,
                local_path
            )

        # ----------------------------
        # Load RGB
        # ----------------------------

        defocused_img = io.imread(
            local_paths["defocused"]
        ).astype(np.float32)

        rgb = defocused_img[:, :, :3] / 255.0

        # ----------------------------
        # Load CoC
        # ----------------------------

        with open(local_paths["coc"], "r") as f:
            coc_px = np.array(
                json.load(f),
                dtype=np.float32
            )

        coc_px = np.clip(coc_px, 0, 25) / 25.0

        # ----------------------------
        # Resize full image
        # ----------------------------

        target_size = 512

        rgb = resize(
            rgb,
            (target_size, target_size),
            anti_aliasing=True,
            preserve_range=True
        ).astype(np.float32)

        coc_px = resize(
            coc_px,
            (target_size, target_size),
            order=1,
            anti_aliasing=True,
            preserve_range=True
        ).astype(np.float32)

        # ----------------------------
        # Convert RGB to CHW
        # ----------------------------

        rgb = np.transpose(rgb, (2, 0, 1))

        H = target_size
        W = target_size

        # ----------------------------
        # Metadata
        # ----------------------------

        with open(local_paths["metadata"], "r") as f:
            metadata = json.load(f)

        f_stop = metadata["f_stop"] / 8.0

        fstop_map = np.ones(
            (1, H, W),
            dtype=np.float32
        ) * f_stop

        focal_length = (
            metadata["focal_length_mm"]
        ) / 135.0

        focal_map = np.ones(
            (1, H, W),
            dtype=np.float32
        ) * focal_length

        # ----------------------------
        # Input tensor
        # ----------------------------

        x = np.concatenate(
            [
                rgb,
                fstop_map,
                focal_map
            ],
            axis=0
        ).astype(np.float32)

        # ----------------------------
        # Target tensor
        # ----------------------------

        y = coc_px[None, :, :].astype(np.float32)

        # ----------------------------
        # NaN protection
        # ----------------------------

        x = np.nan_to_num(
            x,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        y = np.nan_to_num(
            y,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        # ----------------------------
        # Torch tensors
        # ----------------------------

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        return x, y

# def load_data(scenes=["cafe", "grass", "bedroom"]):
#     X = []
#     Y = []

#     for scene in scenes:
#         datadir = f"{scene}/dataset/"
#         contents = sorted(os.listdir(datadir))

#         for folder in contents:
#             filepath = os.path.join(datadir, folder)

#             if not os.path.isdir(filepath):
#                 continue

#             defocused_path = os.path.join(filepath, "defocused.png")
#             coc_path = os.path.join(filepath, "coc.json")

#             if not os.path.exists(defocused_path) or not os.path.exists(coc_path):
#                 continue

#             defocused_img = io.imread(defocused_path).astype(np.float32)

#             # RGB: H x W x 3 -> 3 x H x W
#             rgb = defocused_img[:, :, :3] / 255.0
#             rgb = np.transpose(rgb, (2, 0, 1))

#             metadata = getMetadata(filepath + "/")

#             H = metadata["height"]
#             W = metadata["width"]

#             f_stop = metadata["f_stop"] / 8.0
#             fstop_map = np.ones((1, H, W), dtype=np.float32) * f_stop

#             focal_length = (metadata["focal_length_m"] * 1000.0) / 135.0
#             focal_map = np.ones((1, H, W), dtype=np.float32) * focal_length

#             x = np.concatenate(
#                 [rgb, fstop_map, focal_map],
#                 axis=0
#             ).astype(np.float32)

#             with open(coc_path, "r") as f:
#                 coc_px = np.array(json.load(f), dtype=np.float32)

#             y = np.clip(coc_px, 0, 25) / 25.0
#             y = y[None, :, :].astype(np.float32)

#             X.append(x)
#             Y.append(y)

#     X = np.stack(X, axis=0)
#     Y = np.stack(Y, axis=0)

#     return X, Y


# print(f"R channel: {r.shape}")
# print(f"G channel: {g.shape}")
# print(f"B channel: {b.shape}")
# print(f"f-stop channel: {fstop_map.shape}")
# print(f"focal length channel: {focal_map.shape}")
# print(f"coc channel: {coc_px.shape}")

# print(r[:10])