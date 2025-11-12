import torch
from torch.utils.data import DataLoader, random_split
from dataset import SRDataset
from model import SRCNN
from utils import psnr
import os

def test():
    device = 'cpu'
    print("Testing.")

    data_dir = "./data/combined_400"
    full_set = SRDataset(data_dir)

    t = int(0.6 * len(full_set))
    v = int(0.2 * len(full_set))
    _, _, test_set = random_split(full_set, [t, v, len(full_set) - t - v])

    loader = DataLoader(test_set, batch_size=1)

    ckpt = "./checkpoints/best_model.pth"
    if not os.path.exists(ckpt):
        print("No trained model found. Train first!")
        return

    model = SRCNN().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    total_psnr = 0
    with torch.no_grad():
        for lr_img, hr_img in loader:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            out = model(lr_img)
            total_psnr += psnr(out, hr_img)

    print("Test PSNR:", total_psnr / len(loader))

if __name__ == "__main__":
    test()