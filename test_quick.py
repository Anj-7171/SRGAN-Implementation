# test_quick.py  — create this file
from PIL import Image
import torchvision.transforms as T
from degradation import degrade_image

# Take one HR image and downscale it 4x to simulate LR input
img = Image.open("data/train/0005.png").convert("RGB")  # change filename
lr = degrade_image(img, severity="medium")
lr.save("test_lr.png")
print(f"LR size: {lr.size}, HR size: {img.size}")