import torch
from PIL import Image
import torchvision.transforms as T
import config
from models.generator import Generator

def upscale(image_path: str, output_path: str = "output_sr.png"):
    gen = Generator().to(config.DEVICE)
    gen.load_state_dict(torch.load(config.CHECKPOINT_GEN, map_location=config.DEVICE))
    gen.eval()
    img = Image.open(image_path).convert("RGB")
    to_tensor = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    lr = to_tensor(img).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        sr = gen(lr).squeeze(0).cpu()

    # Denormalize → [0,1] → PIL
    sr = (sr * 0.5 + 0.5).clamp(0, 1)
    T.ToPILImage()(sr).save(output_path)
    print(f"Saved SR image → {output_path}")

if __name__ == "__main__":
    upscale("test_lr.png")