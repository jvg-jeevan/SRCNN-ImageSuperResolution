import torch
from PIL import Image
import torchvision.transforms as T
from model import SRCNN
from utils import preprocess_image, create_lr_image
import os
import argparse

def super_resolve(input_path, output_path, model_path):
    device = 'cpu'

    if not os.path.exists(input_path):
        print("Input image not found:", input_path)
        return

    img = preprocess_image(input_path, size=128)
    lr_img = create_lr_image(img, scale=2)

    to_tensor = T.ToTensor()
    lr_tensor = to_tensor(lr_img).unsqueeze(0).to(device)

    model = SRCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        sr_tensor = model(lr_tensor).squeeze(0)

    sr_np = sr_tensor.permute(1, 2, 0).cpu().numpy().clip(0, 1)
    sr_np = (sr_np * 255).astype('uint8')
    sr_image = Image.fromarray(sr_np)

    sr_image.save(output_path)
    print("Super-resolved image saved as:", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to low-res image")
    parser.add_argument("--output", default="results/output.png", help="Where to save result")
    parser.add_argument("--model", default="checkpoints/best_model.pth", help="Trained model path")
    args = parser.parse_args()

    super_resolve(args.input, args.output, args.model)