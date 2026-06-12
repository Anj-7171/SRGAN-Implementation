# degradation.py
from PIL import Image, ImageFilter
import torchvision.transforms as T
import random
import numpy as np

def degrade_image(img, severity="medium"):
    """
    Apply realistic degradations to simulate a low-quality image.
    img: PIL Image (HR)
    """
    w, h = img.size

    # ── 1. Downscale 4x (bicubic) ──────────────────────────────
    img = img.resize((w//4, h//4), Image.BICUBIC)

    # ── 2. Gaussian Blur ───────────────────────────────────────
    blur_radius = {"low": 0.5, "medium": 1.2, "high": 2.0}[severity]
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # ── 3. Gaussian Noise ──────────────────────────────────────
    noise_std = {"low": 5, "medium": 15, "high": 30}[severity]
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, noise_std, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # ── 4. JPEG Compression artifacts ─────────────────────────
    import io
    quality = {"low": 85, "medium": 50, "high": 20}[severity]
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    img = Image.open(buffer).copy()

    return img


if __name__ == "__main__":
    img = Image.open("data/train/0005.png").convert("RGB") 

    low    = degrade_image(img, severity="low")
    medium = degrade_image(img, severity="medium")
    high   = degrade_image(img, severity="high")

    low.save("degraded_low.png")
    medium.save("degraded_medium.png")
    high.save("degraded_high.png")

    print("Saved: degraded_low.png, degraded_medium.png, degraded_high.png")