# config.py
# config.py
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")         # or "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
BATCH_SIZE = 16
NUM_WORKERS = 4
HIGH_RES = 96             # HR patch size
LOW_RES = HIGH_RES // 4   # 4x upscaling → LR = 24
IMG_CHANNELS = 3

# Paths
TRAIN_DIR = "data/train"  # folder of HR images
CHECKPOINT_GEN = "gen.pth"
CHECKPOINT_DISC = "disc.pth"


DEGRADATION_SEVERITY = "medium"