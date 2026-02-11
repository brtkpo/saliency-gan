import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SaliconDataset
from model import Generator, Discriminator
from tqdm import tqdm
import pandas as pd

def tv_loss(x):
    h_variation = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    w_variation = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return h_variation + w_variation

epoch_results = []
DATA_DIR = "data"
SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
IMG_SIZE = (224, 224)
BATCH_SIZE = 4
LR = 2e-4
NUM_EPOCHS = 25
lambda_l1 = 100
lambda_tv = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

EARLY_STOP_PATIENCE = 5
best_val_l1 = float("inf")
epochs_no_improve = 0

train_dataset = SaliconDataset(split=SPLIT_TRAIN, data_dir=DATA_DIR, img_size=IMG_SIZE, augment=True)
val_dataset = SaliconDataset(split=SPLIT_VAL, data_dir=DATA_DIR, img_size=IMG_SIZE, augment=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

gen = Generator(in_channels=3, out_channels=1).to(DEVICE)
disc = Discriminator(in_channels=4).to(DEVICE)

criterion_GAN = nn.MSELoss()
criterion_L1 = nn.L1Loss()

optimizer_G = optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_D = optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))

def save_checkpoint(epoch, val_l1):
    torch.save({
        "gen_state_dict": gen.state_dict(),
        "disc_state_dict": disc.state_dict(),
        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D": optimizer_D.state_dict(),
        "epoch": epoch,
        "val_L1": val_l1
    }, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}_valL1_{val_l1:.4f}.pth"))
    print(f"[Checkpoint] Saved epoch {epoch} | Val L1: {val_l1:.4f}")

def evaluate(val_loader):
    gen.eval()
    total_l1 = 0.0
    with torch.no_grad():
        for _, images, sal_maps in val_loader:
            images = images.to(DEVICE)
            sal_maps = sal_maps.to(DEVICE)
            preds = gen(images)
            total_l1 += criterion_L1(preds, sal_maps).item() * images.size(0)
    gen.train()
    return total_l1 / len(val_loader.dataset)

for epoch in range(1, NUM_EPOCHS + 1):
    gen.train()
    disc.train()
    epoch_loss_G = 0.0
    epoch_loss_D = 0.0

    for img_names, images, sal_maps in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
        images = images.to(DEVICE)
        sal_maps = sal_maps.to(DEVICE)

        optimizer_D.zero_grad()

        fake_maps = gen(images)
        real_input = torch.cat([images, sal_maps], dim=1)   # [B,4,H,W]
        fake_input = torch.cat([images, fake_maps.detach()], dim=1)

        pred_real = disc(real_input)
        pred_fake = disc(fake_input)

        real_labels = torch.ones_like(pred_real, device=DEVICE)
        fake_labels = torch.zeros_like(pred_fake, device=DEVICE)

        loss_D_real = criterion_GAN(pred_real, real_labels)
        loss_D_fake = criterion_GAN(pred_fake, fake_labels)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        fake_input = torch.cat([images, fake_maps], dim=1)
        pred_fake_for_G = disc(fake_input)
        real_labels_for_G = torch.ones_like(pred_fake_for_G, device=DEVICE)

        loss_G_GAN = criterion_GAN(pred_fake_for_G, real_labels_for_G)
        loss_G_L1 = criterion_L1(fake_maps, sal_maps)
        #loss_G = loss_G_GAN + 100 * loss_G_L1
        loss_G = loss_G_GAN + lambda_l1 * loss_G_L1 + lambda_tv * tv_loss(fake_maps)
        loss_G.backward()
        optimizer_G.step()

        epoch_loss_G += loss_G.item() * images.size(0)
        epoch_loss_D += loss_D.item() * images.size(0)

    val_l1 = evaluate(val_loader)

    print(f"Epoch [{epoch}/{NUM_EPOCHS}] | "
          f"Loss G: {epoch_loss_G/len(train_dataset):.4f} | "
          f"Loss D: {epoch_loss_D/len(train_dataset):.4f} | "
          f"Val L1: {val_l1:.4f}")

    save_checkpoint(epoch, val_l1)

    if val_l1 < best_val_l1:
        best_val_l1 = val_l1
        epochs_no_improve = 0
        save_checkpoint(epoch, val_l1)
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s)")

    if epochs_no_improve >= EARLY_STOP_PATIENCE:
        print(f"Early stopping triggered at epoch {epoch}")
        break

    epoch_results.append({
        "epoch": epoch,
        "loss_G": epoch_loss_G / len(train_dataset),
        "loss_D": epoch_loss_D / len(train_dataset),
        "val_L1": val_l1
    })

    df = pd.DataFrame(epoch_results)
    df.to_csv("training_results.csv", index=False)