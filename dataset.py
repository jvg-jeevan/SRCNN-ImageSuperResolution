import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class SRDataset(Dataset):
    def __init__(self, image_dir):
        self.dir = image_dir

        self.files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        self.to_tensor = T.ToTensor()
        print("Dataset initialized with:", len(self.files), "images")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.dir, self.files[idx])

        hr_img = Image.open(path).convert("RGB")

        hr_img = hr_img.resize((512, 512), Image.BICUBIC)

        lr_small = hr_img.resize((256, 256), Image.BICUBIC)

        lr_up = lr_small.resize((512, 512), Image.BICUBIC)

        return self.to_tensor(lr_up), self.to_tensor(hr_img)