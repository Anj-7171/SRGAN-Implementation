import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from degradation import degrade_image

class SRDataset(Dataset):
    def __init__(self, root_dir, high_res=96, degradation_severity="medium"):
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                      if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        self.high_res = high_res
        self.severity = degradation_severity

        self.hr_transform = T.Compose([
            T.RandomCrop(high_res),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])
        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")

        # Step 1: crop HR patch first
        hr = T.RandomCrop(self.high_res)(img)

        # Step 2: apply full degradation pipeline to get LR
        lr = degrade_image(hr, severity=self.severity)

        # Step 3: convert both to tensors
        hr = self.to_tensor(hr)
        lr = self.to_tensor(lr)

        return lr, hr