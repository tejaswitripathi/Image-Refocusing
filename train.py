import os
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler

from models.UNet import UNet
from data_preprocessing import DefocusDataset

import boto3

def upload_checkpoint(local_path, s3_key):
    s3.upload_file(
        local_path,
        S3_BUCKET,
        s3_key
    )
    print(f"Uploaded to s3://{S3_BUCKET}/{s3_key}")

def gradient_loss(pred, target):
    pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])

    pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

    return (
        torch.abs(pred_dx - target_dx).mean()
        + torch.abs(pred_dy - target_dy).mean()
    )

def coc_loss(pred, target):
    l1 = torch.abs(pred - target)
    weight = 1.0 + 4.0 * target
    weighted_l1 = (weight * l1).mean()

    grad = gradient_loss(pred, target)

    return weighted_l1 + 0.2 * grad

# ------------------
# Config
# ------------------

learning_rate = 1e-4
batch_size = 4
num_epochs = 100
val_split = 0.15
checkpoint_dir = "checkpoints"

s3 = boto3.client("s3")

S3_BUCKET = "tejas-blender-bucket"
S3_CHECKPOINT_PREFIX = "defocus-checkpoints/unet-coc"

os.makedirs(checkpoint_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ------------------
# Load data
# ------------------

dataset = DefocusDataset()

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
# Model
# ------------------

model = UNet(in_channels=5, out_channels=1).to(device)

criterion = coc_loss
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

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
latest_path = os.path.join(checkpoint_dir, "latest.pth")

try:
    checkpoint = torch.load(latest_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"]
    best_val_loss = checkpoint.get("val_loss", float("inf"))

    print(f"Resuming from epoch {start_epoch}")

except Exception as e:
    print(f"No local checkpoint found. Starting fresh. Reason: {e}")

for epoch in range(start_epoch, num_epochs):
    print(f"Starting epoch {epoch+1}")
    model.train()
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
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

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

            print("predictions range:",
                predictions.min().item(),
                predictions.max().item())

            raise RuntimeError("Stopping training due to NaN loss")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # ------------------
    # Validation
    # ------------------

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    scheduler.step(avg_val_loss)

    print(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"Train L1: {avg_train_loss:.6f} | "
        f"Val L1: {avg_val_loss:.6f} | "
        f"LR: {optimizer.param_groups[0]['lr']:.2e}"
    )

    # save latest checkpoint
    epoch_path = os.path.join(
        checkpoint_dir,
        f"epoch_{epoch+1}.pth"
    )

    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": avg_val_loss,
    }, epoch_path)

    upload_checkpoint(
        epoch_path,
        f"{S3_CHECKPOINT_PREFIX}/epoch_{epoch+1}.pth"
    )

    latest_path = os.path.join(checkpoint_dir, "latest.pth")

    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": avg_val_loss,
    }, latest_path)

    upload_checkpoint(
        latest_path,
        f"{S3_CHECKPOINT_PREFIX}/latest.pth"
    )

    # save best checkpoint
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss

        best_path = os.path.join(checkpoint_dir, "best_unet_coc.pth")
        torch.save(model.state_dict(), best_path)

        upload_checkpoint(
            best_path,
            f"{S3_CHECKPOINT_PREFIX}/best_unet_coc.pth"
        )

        print(f"Saved new best model: {best_path}")

print("Training complete!")