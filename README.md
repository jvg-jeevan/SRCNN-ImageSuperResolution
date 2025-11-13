
<h1 align="center">🔍 Image Super-Resolution using Convolutional Neural Networks</h1>
<p align="center">
  Deep Learning based Super-Resolution implemented in PyTorch  
  <br>Upscale low-resolution images into high-resolution outputs
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red" />
  <img src="https://img.shields.io/badge/PSNR-30.6 dB-green" />
  <img src="https://img.shields.io/badge/Compute-CPU Only-blue" />
  <img src="https://img.shields.io/badge/UI-Gradio" />
</p>

## ✅ Introduction

Super-resolution enhances the resolution and clarity of images.  
While bicubic interpolation simply stretches pixels, **SRCNN** learns image textures and edges using convolutional layers, producing sharper and more realistic reconstructions.

This project re-implements the original **SRCNN architecture** proposed by:

> **Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang**  
> *Image Super-Resolution Using Deep Convolutional Networks (IEEE TPAMI, 2015)*  
> https://arxiv.org/abs/1501.00092

---

## ✅ Key Features

✔ Trained from scratch using ~500 high-res images  
✔ Average **Test PSNR > 30 dB** (significantly higher than bicubic)  
✔ Works entirely on CPU  
✔ CLI inference, Gradio Web UI  
✔ Automatic dataset downloader  
✔ Saves training curves & metrics

---

## 🧠 Model Architecture (SRCNN)

<img width="1533" height="569" alt="image" src="https://github.com/user-attachments/assets/97d215da-e46f-429a-9e78-8ba8ac3bd752" />


| Layer | Kernel | Channels | Description             | 
|-------|--------|----------|-------------------------|
| Conv1 | 9×9    | 64       | Feature extraction      |
| Conv2 | 1×1    | 32       | Non-linear mapping      |
| Conv3 | 5×5    | 3        | HR image reconstruction |

Loss: **MSE**  
Optimizer: **Adam**  
Metric: **PSNR**


---

## 📂 Repository Structure

```

SRCNN/
│── checkpoints/          
│── data/                 
│   └── combined_400/
│── results/              # output images, metrics.csv, training plots
│── dataset.py            
│── model.py              
│── train.py             
│── test.py              
│── inference.py         
│── gradio_app.py               
│── utils.py

````

---

## ✅ Installation

### 1️⃣ Create environment
```bash
python -m venv .venv
.venv\Scripts\activate
````

### 2️⃣ Install dependencies

```bash
pip install torch torchvision pillow tqdm matplotlib gradio
```

(optional)

```bash
pip install opencv-python
```

---

## 📥 Dataset

No manual download required.
First run of training downloads:

| Dataset            | Count |
| ------------------ | ----- |
| DIV2K (Validation) | ~100  |
| Urban100           | ~100  |
| BSD300             | ~300  |

Saved here:

```
data/combined_400/
```

---

## 🏋️ Training

```bash
python train.py
```

Outputs:
✔ `checkpoints/best_model.pth`
✔ `results/metrics.csv`
✔ `results/training_history.png`

The training plot shows:

* Train vs Validation Loss
* Train vs Validation PSNR
* Best PSNR marker

---

## ✅ Evaluation

```bash
python test.py
```

Example:

```
Dataset initialized with 500 images
✅ Test PSNR: 30.60 dB
```

---

## 🖼 Inference (Single Image)

```bash
python inference.py --input myphoto.jpg --output results/sr_output.png
```

---

## 🌐 Web App (Gradio)

```bash
python gradio_app.py
```

<img width="1919" height="959" alt="image" src="https://github.com/user-attachments/assets/7658b9ed-8486-4fb4-8e0a-22867b47a51a" />


✔ Upload image
✔ Processed HR image shown
✔ Download button included

<img width="1919" height="960" alt="adf" src="https://github.com/user-attachments/assets/9dab45c2-b571-4488-bfce-60c8bad4cf84" />


---



---

## ✅ Results

| Method           | PSNR           |
| ---------------- | --------------- |
| Bicubic          | ~23 dB          |
| **SRCNN (Ours)** | **30.60 dB ✅** |



---

## 📊 Training Curves

Saved at:

```
results/training_history.png
```

Include:
✅ Loss curve
✅ PSNR curve
✅ Best model checkpoint

---

## ⚠️ Limitations

* CPU-only → slower for large images
* SRCNN improves detail but cannot hallucinate unseen textures
* Advanced models (VDSR, EDSR, ESRGAN) perform better

---

## 📚 Reference

This implementation is based on:

> Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang.
> **Image Super-Resolution Using Deep Convolutional Networks.**
> IEEE Transactions on Pattern Analysis and Machine Intelligence, 2015.

Paper: [https://arxiv.org/abs/1501.00092](https://arxiv.org/abs/1501.00092)

---

## 👨‍💻 Author

**Jeevan V Gowda**<br>
M.Tech – Signal Processing and Machine Learning<br>
National Institute of Technology, Karnataka
[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jvg-jeevan)  
---
## 📝 License

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

This project is licensed under the **MIT License**.
---
