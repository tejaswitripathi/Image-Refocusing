import os
import numpy as np
from skimage import io
from coc_map import getMetadata
import json
import boto3

class DefocusDataset(dataset):

    def __getitem__(self, idx):

        # check local cache
        # if missing:
        #   download from S3

        # load image
        # load metadata
        # load CoC

        s3 = boto3.client('s3')

        bucket_name = 'your-bucket-name'
        object_key = 'folder/filename.ext'  # S3 "path"
        local_file_path = 'local-filename.ext'

        # Execute the download
        s3.download_file(bucket_name, object_key, local_file_path)

        return x, y

def load_data(scenes=["cafe", "grass", "bedroom"]):
    X = []
    Y = []

    for scene in scenes:
        datadir = f"{scene}/dataset/"
        contents = sorted(os.listdir(datadir))

        for folder in contents:
            filepath = os.path.join(datadir, folder)

            if not os.path.isdir(filepath):
                continue

            defocused_path = os.path.join(filepath, "defocused.png")
            coc_path = os.path.join(filepath, "coc.json")

            if not os.path.exists(defocused_path) or not os.path.exists(coc_path):
                continue

            defocused_img = io.imread(defocused_path).astype(np.float32)

            # RGB: H x W x 3 -> 3 x H x W
            rgb = defocused_img[:, :, :3] / 255.0
            rgb = np.transpose(rgb, (2, 0, 1))

            metadata = getMetadata(filepath + "/")

            H = metadata["height"]
            W = metadata["width"]

            f_stop = metadata["f_stop"] / 8.0
            fstop_map = np.ones((1, H, W), dtype=np.float32) * f_stop

            focal_length = (metadata["focal_length_m"] * 1000.0) / 135.0
            focal_map = np.ones((1, H, W), dtype=np.float32) * focal_length

            x = np.concatenate(
                [rgb, fstop_map, focal_map],
                axis=0
            ).astype(np.float32)

            with open(coc_path, "r") as f:
                coc_px = np.array(json.load(f), dtype=np.float32)

            y = np.clip(coc_px, 0, 25) / 25.0
            y = y[None, :, :].astype(np.float32)

            X.append(x)
            Y.append(y)

    X = np.stack(X, axis=0)
    Y = np.stack(Y, axis=0)

    return X, Y


# print(f"R channel: {r.shape}")
# print(f"G channel: {g.shape}")
# print(f"B channel: {b.shape}")
# print(f"f-stop channel: {fstop_map.shape}")
# print(f"focal length channel: {focal_map.shape}")
# print(f"coc channel: {coc_px.shape}")

# print(r[:10])