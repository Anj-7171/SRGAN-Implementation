# test_quick.py  — create this file
from PIL import Image
import torchvision.transforms as T

# Take one HR image and downscale it 4x to simulate LR input
img = Image.open("data/train/0001.png").convert("RGB")  # change filename
w, h = img.size
lr = img.resize((w//4, h//4), Image.BICUBIC)
lr.save("test_lr.png")
print(f"LR size: {lr.size}, HR size: {img.size}")