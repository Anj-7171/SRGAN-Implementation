import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from models.generator import Generator
from models.discriminator import Discriminator
from models.feature_extractor import VGGFeatureExtractor
from data.dataset import SRDataset

def train():
    gen  = Generator().to(config.DEVICE)
    disc = Discriminator().to(config.DEVICE)
    vgg  = VGGFeatureExtractor().to(config.DEVICE)

    opt_gen  = torch.optim.Adam(gen.parameters(),  lr=config.LEARNING_RATE)
    opt_disc = torch.optim.Adam(disc.parameters(), lr=config.LEARNING_RATE)

    bce = nn.BCEWithLogitsLoss()
    l1  = nn.L1Loss()

    dataset    = SRDataset(config.TRAIN_DIR, config.HIGH_RES)
    dataloader = DataLoader(dataset, config.BATCH_SIZE, shuffle=True,
                            num_workers=config.NUM_WORKERS, pin_memory=True)
    writer = SummaryWriter("logs")

    for epoch in range(config.NUM_EPOCHS):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")

        for lr_imgs, hr_imgs in loop:
            lr_imgs = lr_imgs.to(config.DEVICE)
            hr_imgs = hr_imgs.to(config.DEVICE)

            # ── Discriminator ───────────────────────────────────────────
            fake_hr = gen(lr_imgs).detach()
            d_real  = disc(hr_imgs)
            d_fake  = disc(fake_hr)
            loss_d  = bce(d_real, torch.ones_like(d_real)) + \
                      bce(d_fake, torch.zeros_like(d_fake))
            opt_disc.zero_grad(); loss_d.backward(); opt_disc.step()

            # ── Generator ────────────────────────────────────────────────
            fake_hr     = gen(lr_imgs)
            d_fake      = disc(fake_hr)
            adv_loss    = 1e-3 * bce(d_fake, torch.ones_like(d_fake))
            perc_loss   = l1(vgg(fake_hr), vgg(hr_imgs))
            pixel_loss  = l1(fake_hr, hr_imgs)
            loss_g      = adv_loss + perc_loss + pixel_loss

            opt_gen.zero_grad(); loss_g.backward(); opt_gen.step()

            loop.set_postfix(D=loss_d.item(), G=loss_g.item())

        writer.add_scalar("Loss/Discriminator", loss_d.item(), epoch)
        writer.add_scalar("Loss/Generator",     loss_g.item(), epoch)

    torch.save(gen.state_dict(),  config.CHECKPOINT_GEN)
    torch.save(disc.state_dict(), config.CHECKPOINT_DISC)
    print("✅ Checkpoints saved.")

if __name__ == "__main__":
    train()