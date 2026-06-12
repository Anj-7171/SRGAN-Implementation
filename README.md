# SRGAN Implementation

A PyTorch implementation of **Super-Resolution Generative Adversarial Network (SRGAN)** for single image super-resolution. This project aims to reconstruct high-resolution images from low-resolution inputs while preserving realistic textures and fine details.

## Overview

Traditional super-resolution methods often optimize for pixel-wise accuracy, resulting in blurry outputs. SRGAN introduces a Generative Adversarial Network (GAN) framework that produces visually realistic and perceptually convincing high-resolution images by combining:

- **Generator Network** (SRResNet-based architecture)
- **Discriminator Network**
- **Perceptual Loss (Content Loss + Adversarial Loss)**

The original SRGAN paper demonstrated significant improvements in visual quality compared to conventional CNN-based super-resolution methods. :contentReference[oaicite:0]{index=0}

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/976e0d62-882a-4ba4-aa9f-704627aab679" />


---

## Features

- Image super-resolution using SRGAN
- Generator based on residual blocks
- Adversarial training with discriminator network
- Perceptual loss using feature extraction
- Model training and inference support
- Performance evaluation using image quality metrics

---

## Project Structure

```text
SRGAN-Implementation/
│
├── data/                   # Dataset
├── models/                 # Generator and Discriminator
├── training/               # Training scripts
├── utils/                  # Utility functions
├── outputs/                # Generated images
├── checkpoints/            # Saved model weights
├── train.py                # Training script
├── inference.py            # Super-resolution inference
├── requirements.txt
└── README.md
```

---

## SRGAN Architecture

### Generator

The generator receives a low-resolution image and generates a high-resolution version through:

1. Initial Convolution Layer
2. Residual Blocks
3. Skip Connection
4. Upsampling Layers (Pixel Shuffle)
5. Final Reconstruction Layer

### Discriminator

The discriminator learns to distinguish between:

- Real high-resolution images
- Generated super-resolved images

This adversarial training encourages the generator to produce realistic textures.

---

## Dataset

The model can be trained on super-resolution datasets such as:

- DIV2K
- CelebA
- ImageNet subsets
- Custom datasets

Training images are converted into:

- High Resolution (HR)
- Low Resolution (LR) using bicubic downsampling

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Anj-7171/SRGAN-Implementation.git
cd SRGAN-Implementation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Start training using:

```bash
python train.py
```

Training process:

1. Train Generator (SRResNet)
2. Train Discriminator
3. Compute Content Loss
4. Compute Adversarial Loss
5. Update Generator and Discriminator weights

---

## Inference

Run super-resolution on an image:

```bash
python inference.py --image path/to/image.jpg
```

Output images will be saved in the output directory.

---

## Loss Functions

### Content Loss

Measures similarity between generated and ground truth images in feature space.

### Adversarial Loss

Encourages the generator to create images that fool the discriminator.

### Total Generator Loss

```math
L_{SR} = L_{Content} + \lambda L_{Adversarial}
```

---

## Results

| Input (LR) | Output (SR) | Ground Truth (HR) |
|------------|-------------|-------------------|
| Low Resolution Image | Super Resolved Image | Original High Resolution Image |

The SRGAN model produces sharper textures and more realistic details than traditional CNN-based super-resolution methods. :contentReference[oaicite:1]{index=1}

---

## Evaluation Metrics

- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Perceptual Quality Comparison

> Note: SRGAN may achieve lower PSNR than pixel-wise optimized methods while generating more visually realistic images. :contentReference[oaicite:2]{index=2}

---

## Applications

- Image enhancement
- Face restoration
- Medical image super-resolution
- Satellite image processing
- Video upscaling
- Old photograph restoration

---

## Future Improvements

- ESRGAN implementation
- Real-ESRGAN integration
- Multiple scaling factors (×2, ×4, ×8)
- Lightweight deployment models
- Web-based inference interface

---

## References

1. Christian Ledig et al., *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (SRGAN)*, 2016. :contentReference[oaicite:3]{index=3}
2. Xintao Wang et al., *ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks*, 2018. :contentReference[oaicite:4]{index=4}

---

## Author

**Anjana Nair**

GitHub: https://github.com/Anj-7171
