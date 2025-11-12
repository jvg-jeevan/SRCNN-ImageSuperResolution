import urllib.request
import zipfile
import os
import shutil
import tarfile
from PIL import Image
import torch
import torch.nn.functional as F

def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def download_combined_dataset(project_dir):
    dataset_dir = os.path.join(project_dir, "data")
    combined_dir = os.path.join(dataset_dir, "combined_400")

    if os.path.exists(combined_dir) and len(os.listdir(combined_dir)) > 0:
        print("Dataset exists:", combined_dir)
        return combined_dir

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)

    print("Preparing combined dataset (400+ images)...")

    try:
        print("Downloading DIV2K (100 images)...")
        url1 = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
        zip1 = os.path.join(dataset_dir, "div2k_val.zip")

        urllib.request.urlretrieve(url1, zip1)
        with zipfile.ZipFile(zip1, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(dataset_dir, "temp1"))

        temp_dir1 = os.path.join(dataset_dir, "temp1", "DIV2K_valid_HR")
        for img in os.listdir(temp_dir1):
            if img.endswith(('.png', '.jpg', '.jpeg')):
                shutil.copy(os.path.join(temp_dir1, img), os.path.join(combined_dir, img))

        os.remove(zip1)
        shutil.rmtree(os.path.join(dataset_dir, "temp1"))
        print("Added 100 images")
    except Exception as e:
        print("DIV2K download failed:", e)

    # ------------------- 2. Urban100 ------------------
    try:
        print("Downloading Urban100 (~100 images)...")
        base_url = "https://github.com/jbhuang0604/SelfExSR/raw/master/data/Urban100/image_SRF_4/"

        for i in range(1, 101):
            img_name = f"img_{i:03d}_SRF_4_HR.png"
            img_url = base_url + img_name
            try:
                urllib.request.urlretrieve(img_url, os.path.join(combined_dir, f"urban_{i:03d}.png"))
            except:
                continue

        print("Added ~100 images")
    except Exception as e:
        print("Urban100 download failed:", e)

    # ------------------- 3. BSD300 ------------------
    try:
        print("Downloading BSD300 (300 images)...")
        url3 = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        tgz_path = os.path.join(dataset_dir, "bsd300.tgz")

        urllib.request.urlretrieve(url3, tgz_path)
        with tarfile.open(tgz_path, 'r:gz') as tar_ref:
            tar_ref.extractall(os.path.join(dataset_dir, "temp3"))

        for subdir in ['train', 'test']:
            bsd_dir = os.path.join(dataset_dir, "temp3", "BSDS300", "images", subdir)
            if os.path.exists(bsd_dir):
                for img in os.listdir(bsd_dir):
                    if img.endswith(('.jpg', '.png')):
                        shutil.copy(os.path.join(bsd_dir, img), os.path.join(combined_dir, f"bsd_{subdir}_{img}"))

        os.remove(tgz_path)
        shutil.rmtree(os.path.join(dataset_dir, "temp3"))
        print("Added 300 images")
    except Exception as e:
        print("BSD300 download failed:", e)

    print("====================================")
    print("DATASET READY!")
    print("Location:", combined_dir)
    print("Total images:", len(os.listdir(combined_dir)))
    print("====================================")

    return combined_dir