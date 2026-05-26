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

# ------------------
# Config
# ------------------

learning_rate = 1e-4
batch_size = 2
num_epochs = 50
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

# X, Y = load_data()
# X = torch.from_numpy(X).float()
# Y = torch.from_numpy(Y).float()

# print("X:", X.shape)  # [N, 5, H, W]
# print("Y:", Y.shape)  # [N, 1, H, W]

# dataset = TensorDataset(X, Y)

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

criterion = nn.L1Loss()
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
    model.train()
    train_loss = 0.0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=(device == "cuda")):
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

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