import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import torch.amp

from .dataset import SaliconDataset
from .model import Generator, Discriminator

def tv_loss(x):
    """Total variation loss for smoothness"""
    h_variation = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    w_variation = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return h_variation + w_variation

def kl_divergence_loss(pred, target, eps=1e-7):
    """KL divergence for saliency maps"""
    pred_norm = pred / (torch.sum(pred, dim=(2, 3), keepdim=True) + eps)
    target_norm = target / (torch.sum(target, dim=(2, 3), keepdim=True) + eps)
    loss = target_norm * torch.log((target_norm + eps) / (pred_norm + eps))
    return torch.sum(loss, dim=(2, 3)).mean()

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)

def train_model(
    data_dir,
    checkpoint_dir,
    img_size,
    batch_size,
    lr,
    num_epochs,
    early_stop_patience,
    lambda_l1,
    lambda_kld,
    lambda_tv,
    device
):

    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Loading Datasets...")
    train_dataset = SaliconDataset(split="train", data_dir=data_dir, img_size=img_size, augment=True)
    val_dataset = SaliconDataset(split="val", data_dir=data_dir, img_size=img_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    gen = Generator(in_channels=3, out_channels=1).to(device)
    disc = Discriminator(in_channels=4).to(device)
    init_weights(gen)
    init_weights(disc)

    criterion_GAN = nn.MSELoss()
    criterion_L1 = nn.L1Loss()

    optimizer_G = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=3, verbose=True)
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=3, verbose=True)

    scaler = torch.amp.GradScaler('cuda')

    def save_checkpoint(epoch, val_loss):
        torch.save({
            "gen_state_dict": gen.state_dict(),
            "disc_state_dict": disc.state_dict(),
            "optimizer_G": optimizer_G.state_dict(),
            "optimizer_D": optimizer_D.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss
        }, os.path.join(checkpoint_dir, "best_model.pth"))

    def evaluate(val_loader):
        gen.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for _, images, sal_maps in val_loader:
                images = images.to(device)
                sal_maps = sal_maps.to(device)

                with torch.amp.autocast('cuda'):
                    preds = gen(images)
                    l1 = criterion_L1(preds, sal_maps)
                    kld = kl_divergence_loss(preds, sal_maps)
                    loss = l1 + kld

                total_val_loss += loss.item() * images.size(0)
        gen.train()
        return total_val_loss / len(val_loader.dataset)

    epoch_results = []
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        gen.train()
        disc.train()
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0

        loop = tqdm(train_loader, leave=True)
        loop.set_description(f"Epoch [{epoch}/{num_epochs}]")

        for img_names, images, sal_maps in loop:
            images = images.to(device)
            sal_maps = sal_maps.to(device)

            optimizer_D.zero_grad()

            with torch.amp.autocast('cuda'):
                fake_maps = gen(images)
                real_input = torch.cat([images, sal_maps], dim=1)
                fake_input = torch.cat([images, fake_maps.detach()], dim=1)

                pred_real = disc(real_input)
                pred_fake = disc(fake_input)

                loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
                loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
                loss_D = (loss_D_real + loss_D_fake) * 0.5

            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)

            optimizer_G.zero_grad()

            with torch.amp.autocast('cuda'):
                fake_input = torch.cat([images, fake_maps], dim=1)
                pred_fake_for_G = disc(fake_input)

                loss_G = (criterion_GAN(pred_fake_for_G, torch.ones_like(pred_fake_for_G)) +
                          lambda_l1 * criterion_L1(fake_maps, sal_maps) +
                          lambda_kld * kl_divergence_loss(fake_maps, sal_maps) +
                          lambda_tv * tv_loss(fake_maps))

            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)
            scaler.update()

            epoch_loss_G += loss_G.item() * images.size(0)
            epoch_loss_D += loss_D.item() * images.size(0)

            loop.set_postfix(Loss_G=loss_G.item(), Loss_D=loss_D.item())

        avg_loss_G = epoch_loss_G / len(train_dataset)
        avg_loss_D = epoch_loss_D / len(train_dataset)
        val_loss = evaluate(val_loader)

        print(f"Epoch {epoch} Results | Loss G: {avg_loss_G:.4f} | Loss D: {avg_loss_D:.4f} | Val Loss: {val_loss:.4f}")

        scheduler_G.step(val_loss)
        scheduler_D.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(epoch, val_loss)
            print("New Best Model Saved")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= early_stop_patience:
            print("Early stopping triggered. Training complete.")
            break

        epoch_results.append({
            "epoch": epoch,
            "loss_G": avg_loss_G,
            "loss_D": avg_loss_D,
            "val_loss": val_loss
        })
        pd.DataFrame(epoch_results).to_csv("training_results.csv", index=False)