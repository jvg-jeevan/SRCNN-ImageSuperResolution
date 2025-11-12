import gradio as gr
from PIL import Image
import torch
import torchvision.transforms as T
from model import SRCNN
import os

device = "cpu"

model_path = "checkpoints/best_model.pth"
if not os.path.exists(model_path):
    raise ValueError("Model not found. Training not Done!")

model = SRCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def super_resolve(img):

    img = img.convert("RGB")
    w, h = img.size

    lr_small = img.resize((w // 2, h // 2), Image.BICUBIC)

    lr_up = lr_small.resize((w, h), Image.BICUBIC)

    to_tensor = T.ToTensor()
    lr_tensor = to_tensor(lr_up).unsqueeze(0).to(device)

    with torch.no_grad():
        sr = model(lr_tensor).squeeze(0)

    sr_np = sr.permute(1, 2, 0).cpu().numpy().clip(0, 1)
    sr_img = Image.fromarray((sr_np * 255).astype("uint8"))

    os.makedirs("results", exist_ok=True)
    output_path = "results/gradio_output.png"
    sr_img.save(output_path)

    return sr_img, output_path

demo = gr.Interface(
    fn=super_resolve,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(type="pil", label="Super-Resolved Output"),
        gr.File(label="Download Output")
    ],
    title="SRCNN Super Resolution",
    description="Upload an image and get a higher resolution output",
)

if __name__ == "__main__":
    demo.launch()