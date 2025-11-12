import torch
from torch.utils.data import DataLoader, random_split
from dataset import SRDataset
from model import SRCNN
from utils import download_combined_dataset, psnr
import os
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt

def train(project_dir, epochs=10, lr=1e-3):
    device = "cpu"
    print("Training with 512x512 images...")

    data_dir = download_combined_dataset(project_dir)

    full_set = SRDataset(data_dir)

    t = int(0.6 * len(full_set))
    v = int(0.2 * len(full_set))
    tr, val, _ = random_split(full_set, [t, v, len(full_set) - t - v])

    tr_loader = DataLoader(tr, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=1, num_workers=0)

    model = SRCNN().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    best_psnr = -1
    save_path = os.path.join(project_dir, "checkpoints", "best_model.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # CSV logging
    metrics_file = os.path.join(project_dir, "results", "metrics.csv")
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_psnr", "val_psnr"])

    train_losses = []
    val_losses = []
    train_psnr_list = []
    val_psnr_list = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        running_loss = 0
        running_psnr = 0

        for lr_img, hr_img in tqdm(tr_loader):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)

            optim.zero_grad()
            out = model(lr_img)
            loss = loss_fn(out, hr_img)
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_psnr += psnr(out, hr_img)

        avg_train_loss = running_loss / len(tr_loader)
        avg_train_psnr = running_psnr / len(tr_loader)
        print(f"Train Loss: {avg_train_loss:.4f}, PSNR: {avg_train_psnr:.4f}")

        model.eval()
        val_loss = 0
        val_ps = 0
        with torch.no_grad():
            for lr_img, hr_img in val_loader:
                lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                out = model(lr_img)
                val_loss += loss_fn(out, hr_img).item()
                val_ps += psnr(out, hr_img)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = val_ps / len(val_loader)

        print(f"Val Loss: {avg_val_loss:.4f}, PSNR: {avg_val_psnr:.4f}")

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_psnr_list.append(float(avg_train_psnr))
        val_psnr_list.append(float(avg_val_psnr))

        with open(metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                avg_train_loss,
                avg_val_loss,
                float(avg_train_psnr),
                float(avg_val_psnr),
            ])

        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            torch.save(model.state_dict(), save_path)
            print("Model saved:", save_path)

    plt.figure(figsize=(12,5))

    # Loss subplot
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_psnr_list, label="Train PSNR")
    plt.plot(val_psnr_list, label="Val PSNR")
    plt.axhline(best_psnr, color="red", linestyle="--", label=f"Best: {best_psnr:.2f} dB")
    plt.title("Training and Validation PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.legend()

    graph_path = os.path.join(project_dir, "results", "training_history.png")
    plt.tight_layout()
    plt.savefig(graph_path)

    print(f"Training history plot saved as: {graph_path}")

if __name__ == "__main__":
    train("SRCNN")
