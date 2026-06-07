import io
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler

from models.UNet import UNet
from models.RendererNet import RendererNet
from data_preprocessing import DefocusDataset

import boto3


def upload_checkpoint_to_s3(checkpoint, s3_key):
    # Serialize the checkpoint to an in-memory buffer and stream it straight
    # to S3 so nothing is written to local disk.
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)

    s3.upload_fileobj(buffer, S3_BUCKET, s3_key)
    print(f"Uploaded to s3://{S3_BUCKET}/{s3_key}")


def download_checkpoint_from_s3(s3_key, map_location):
    # Pull the checkpoint from S3 into memory and load it without touching disk.
    buffer = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, s3_key, buffer)
    buffer.seek(0)

    return torch.load(buffer, map_location=map_location)


def split_render_batch(batch_X, batch_y):
    # DefocusDataset (direction="render") packs the input as:
    #   x = [sharp RGB (0:3), f-stop (3), focal (4), CoC_gt (5)]
    #   y = defocused RGB
    # We unpack those pieces here so we can re-route them through the pipeline.
    sharp_rgb = batch_X[:, 0:3]
    fstop_map = batch_X[:, 3:4]
    focal_map = batch_X[:, 4:5]
    coc_gt = batch_X[:, 5:6]
    defocused_gt = batch_y
    return sharp_rgb, fstop_map, focal_map, coc_gt, defocused_gt


def run_pipeline(coc_net, renderer_net, batch_X, batch_y):
    # COCNet: estimate the CoC map from the defocused image + camera metadata.
    #   input  = [defocused RGB (3), f-stop (1), focal (1)] -> 5 channels
    #   output = CoC map (1 channel)
    # RendererNet (frozen): re-render the defocused image from the sharp image,
    # camera metadata, and the *predicted* CoC map.
    #   input  = [sharp RGB (3), f-stop (1), focal (1), CoC_pred (1)] -> 6 ch
    #   output = defocused RGB (3 channels)
    sharp_rgb, fstop_map, focal_map, coc_gt, defocused_gt = split_render_batch(
        batch_X, batch_y
    )

    coc_input = torch.cat([defocused_gt, fstop_map, focal_map], dim=1)
    coc_pred = coc_net(coc_input)

    renderer_input = torch.cat([sharp_rgb, fstop_map, focal_map, coc_pred], dim=1)
    defocused_pred = renderer_net(renderer_input)

    return coc_pred, coc_gt, defocused_pred, defocused_gt

# ------------------
# Config
# ------------------

learning_rate = 1e-4
batch_size = 4
num_epochs = 100
val_split = 0.15
COC_LOSS_WEIGHT = 0.2

s3 = boto3.client("s3")

S3_BUCKET = "tejas-blender-bucket"

# Pretrained (frozen) renderer used as a differentiable image formation model.
S3_RENDERER_BEST_KEY = "defocus-checkpoints/renderer-net/best_renderer.pth"

# Where this CoC estimator's best checkpoint is stored.
S3_CHECKPOINT_PREFIX = "defocus-checkpoints/coc-net"
S3_BEST_KEY = f"{S3_CHECKPOINT_PREFIX}/best_coc.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ------------------
# Load data
# ------------------

dataset = DefocusDataset(direction="render")

val_size = max(1, int(len(dataset) * val_split))
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# ------------------
# Models
# ------------------

# CoC estimator (the network we are training).
coc_net = UNet(in_channels=5, out_channels=1).to(device)

# Frozen renderer: provides the image-space gradient signal but is not trained.
renderer_net = RendererNet(in_channels=6, out_channels=3).to(device)

renderer_ckpt = download_checkpoint_from_s3(
    S3_RENDERER_BEST_KEY,
    map_location=device
)

if isinstance(renderer_ckpt, dict) and "model_state_dict" in renderer_ckpt:
    renderer_net.load_state_dict(renderer_ckpt["model_state_dict"])
else:
    renderer_net.load_state_dict(renderer_ckpt)

renderer_net.eval()
for p in renderer_net.parameters():
    p.requires_grad = False

print("Loaded frozen RendererNet from "
      f"s3://{S3_BUCKET}/{S3_RENDERER_BEST_KEY}")

# ------------------
# Loss / optim
# ------------------

image_loss_fn = nn.MSELoss()
coc_loss_fn = nn.MSELoss()

optimizer = optim.AdamW(coc_net.parameters(), lr=learning_rate, weight_decay=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

scaler = GradScaler("cuda", enabled=(device == "cuda"))

best_val_loss = float("inf")

# ------------------
# Training loop
# ------------------

start_epoch = 0

try:
    checkpoint = download_checkpoint_from_s3(S3_BEST_KEY, map_location=device)

    coc_net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"]
    best_val_loss = checkpoint.get("val_loss", float("inf"))

    print(f"Resuming from best checkpoint at epoch {start_epoch} "
          f"(best val loss: {best_val_loss:.6f})")

except Exception as e:
    print(f"No best checkpoint found in S3. Starting fresh. Reason: {e}")

for epoch in range(start_epoch, num_epochs):
    print(f"Starting epoch {epoch+1}")
    coc_net.train()
    train_loss = 0.0

    for batch_idx, (batch_X, batch_y) in enumerate(train_loader):

        if torch.isnan(batch_X).any():
            print(f"NaNs found in batch_X at batch {batch_idx}")
            raise RuntimeError("Stopping training due to NaNs in inputs")

        if torch.isnan(batch_y).any():
            print(f"NaNs found in batch_y at batch {batch_idx}")
            raise RuntimeError("Stopping training due to NaNs in targets")

        if torch.isinf(batch_X).any():
            print(f"Infs found in batch_X at batch {batch_idx}")
            raise RuntimeError("Stopping training due to Infs in inputs")

        if torch.isinf(batch_y).any():
            print(f"Infs found in batch_y at batch {batch_idx}")
            raise RuntimeError("Stopping training due to Infs in targets")

        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=(device == "cuda")):
            coc_pred, coc_gt, defocused_pred, defocused_gt = run_pipeline(
                coc_net, renderer_net, batch_X, batch_y
            )

            image_loss = image_loss_fn(defocused_pred, defocused_gt)
            coc_loss = coc_loss_fn(coc_pred, coc_gt)
            loss = image_loss + COC_LOSS_WEIGHT * coc_loss

        # ----------------------------
        # Loss sanity check
        # ----------------------------

        if torch.isnan(loss):
            print(f"NaN loss detected at batch {batch_idx}")

            print("batch_X range:",
                batch_X.min().item(),
                batch_X.max().item())

            print("batch_y range:",
                batch_y.min().item(),
                batch_y.max().item())

            print("coc_pred range:",
                coc_pred.min().item(),
                coc_pred.max().item())

            print("defocused_pred range:",
                defocused_pred.min().item(),
                defocused_pred.max().item())

            raise RuntimeError("Stopping training due to NaN loss")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # ------------------
    # Validation
    # ------------------

    coc_net.eval()
    val_loss = 0.0
    val_image_loss = 0.0
    val_coc_loss = 0.0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            coc_pred, coc_gt, defocused_pred, defocused_gt = run_pipeline(
                coc_net, renderer_net, batch_X, batch_y
            )

            image_loss = image_loss_fn(defocused_pred, defocused_gt)
            coc_loss = coc_loss_fn(coc_pred, coc_gt)
            loss = image_loss + COC_LOSS_WEIGHT * coc_loss

            val_loss += loss.item()
            val_image_loss += image_loss.item()
            val_coc_loss += coc_loss.item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_image_loss = val_image_loss / len(val_loader)
    avg_val_coc_loss = val_coc_loss / len(val_loader)

    scheduler.step(avg_val_loss)

    print(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"Train loss: {avg_train_loss:.6f} | "
        f"Val loss: {avg_val_loss:.6f} "
        f"(img: {avg_val_image_loss:.6f}, coc: {avg_val_coc_loss:.6f}) | "
        f"LR: {optimizer.param_groups[0]['lr']:.2e}"
    )

    # Only upload on a new best. The checkpoint carries everything needed to
    # resume (model + optimizer + scheduler + epoch), so if training dies we
    # can just pull this from S3 and continue.
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": coc_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": avg_val_loss,
        }

        try:
            upload_checkpoint_to_s3(checkpoint, S3_BEST_KEY)
            print(f"New best val loss: {best_val_loss:.6f} (epoch {epoch+1})")
        except Exception as e:
            print("Checkpoint upload failed:", e)

print("Training complete!")
