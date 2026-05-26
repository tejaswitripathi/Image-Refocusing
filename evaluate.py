import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import load_data
from models.UNet import UNet

input_size = 1024*1024
batch_size = 1

X, Y = load_data()
X = torch.from_numpy(X).float()
Y = torch.from_numpy(Y).float()

dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(in_channels=5, out_channels=1).to(device)
model.load_state_dict(torch.load('unet_coc.pth', weights_only=True))

def check_accuracy(loader, model):
    model.eval()

    total_mse = 0.0
    total_l1 = 0.0
    num_batches = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            mse = F.mse_loss(pred, y)
            l1 = F.l1_loss(pred, y)

            total_mse += mse.item()
            total_l1 += l1.item()

            num_batches += 1

        print(f"Average MSE: {total_mse / num_batches:.6f}")
        print(f"Average L1 : {total_l1 / num_batches:.6f}")

        # visualize one sample
        x, y = next(iter(loader))

        x = x.to(device)
        pred = model(x[:1]).cpu().numpy()[0, 0]
        gt = y[:1].numpy()[0, 0]

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(gt, cmap="magma")
        plt.title("GT CoC")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(pred, cmap="magma")
        plt.title("Pred CoC")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(np.abs(gt - pred), cmap="inferno")
        plt.title("Absolute Error")
        plt.axis("off")

        plt.show()

    # model.train()

# Final accuracy check on training and test sets
check_accuracy(dataloader, model)