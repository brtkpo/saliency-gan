import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from dataset import SaliconDataset
from model import Generator

DATA_DIR = "data"
RESULTS_DIR = "results"
CSV_PATH = os.path.join(RESULTS_DIR, "results_val_full.csv")
CHECKPOINT_PATH = "checkpoints/best_model.pth"
IMG_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BEST_DIR = os.path.join(RESULTS_DIR, "visuals_best")
WORST_DIR = os.path.join(RESULTS_DIR, "visuals_worst")
os.makedirs(BEST_DIR, exist_ok=True)
os.makedirs(WORST_DIR, exist_ok=True)

gen = Generator(in_channels=3, out_channels=1).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
gen.load_state_dict(checkpoint["gen_state_dict"])
gen.eval()

dataset = SaliconDataset(split="val", data_dir=DATA_DIR, img_size=IMG_SIZE)
df = pd.read_csv(CSV_PATH)


def save_comparison(img_name, auc_val, save_folder, prefix):
    try:
        idx = dataset.images.index(img_name)
    except ValueError:
        print(f"Didn't find {img_name} in validation set.")
        return

    _, image_tensor, gt_tensor = dataset[idx]
    image_input = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = gen(image_input)

    pred_np = pred.squeeze().cpu().numpy()
    pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)

    gt_np = gt_tensor.squeeze().cpu().numpy()
    gt_np = (gt_np - gt_np.min()) / (gt_np.max() - gt_np.min() + 1e-8)

    img_np = image_tensor.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.title(f"Original\n{img_name}")
    plt.imshow(img_np)
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Ground Truth (GT)")
    plt.imshow(gt_np, cmap='jet')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title(f"Predicted (AUC: {auc_val:.4f})")
    plt.imshow(pred_np, cmap='jet')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("Overlay")
    plt.imshow(img_np)
    plt.imshow(pred_np, cmap='jet', alpha=0.4)
    plt.axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_folder, f"{prefix}_{img_name}")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

best_5 = df.sort_values("AUC", ascending=False).head(5)
worst_5 = df.sort_values("AUC", ascending=True).head(5)

print("\nProcessing BEST images...")
for _, row in best_5.iterrows():
    save_comparison(row['image'], row['AUC'], BEST_DIR, "best")

print("\nProcessing WORST images...")
for _, row in worst_5.iterrows():
    save_comparison(row['image'], row['AUC'], WORST_DIR, "worst")