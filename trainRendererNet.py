import io
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler

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

# ------------------
# Config
# ------------------

learning_rate = 1e-4
batch_size = 4
num_epochs = 100
val_split = 0.15

s3 = boto3.client("s3")

S3_BUCKET = "tejas-blender-bucket"
S3_CHECKPOINT_PREFIX = "defocus-checkpoints/renderer-net"
S3_BEST_KEY = f"{S3_CHECKPOINT_PREFIX}/best_renderer.pth"

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

model = RendererNet(in_channels=6, out_channels=3).to(device)

criterion = nn.MSELoss()
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

try:
    checkpoint = download_checkpoint_from_s3(S3_BEST_KEY, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"]
    best_val_loss = checkpoint.get("val_loss", float("inf"))

    print(f"Resuming from best checkpoint at epoch {start_epoch} "
          f"(best val MSE: {best_val_loss:.6f})")

except Exception as e:
    print(f"No best checkpoint found in S3. Starting fresh. Reason: {e}")

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
        f"Train MSE: {avg_train_loss:.6f} | "
        f"Val MSE: {avg_val_loss:.6f} | "
        f"LR: {optimizer.param_groups[0]['lr']:.2e}"
    )

    # Only upload on a new best. The checkpoint carries everything needed to
    # resume (model + optimizer + scheduler + epoch), so if training dies we
    # can just pull this from S3 and continue.
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": avg_val_loss,
        }

        try:
            upload_checkpoint_to_s3(checkpoint, S3_BEST_KEY)
            print(f"New best val MSE: {best_val_loss:.6f} (epoch {epoch+1})")
        except Exception as e:
            print("Checkpoint upload failed:", e)

print("Training complete!")