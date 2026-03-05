import os
import torch
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from .dataset import SaliconDataset
from .model import Generator


def visualize_results(
        data_dir="../data",
        results_dir="../results",
        csv_path=None,
        checkpoint_path="../checkpoints/best_model.pth",
        img_size=(224, 224),
        device=None,
):

    if csv_path is None:
        csv_path = os.path.join(results_dir, "results_val_full.csv")

    best_dir = os.path.join(results_dir, "visuals_best")
    worst_dir = os.path.join(results_dir, "visuals_worst")
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(worst_dir, exist_ok=True)

    gen = Generator(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(checkpoint["gen_state_dict"])
    gen.eval()

    dataset = SaliconDataset(split="val", data_dir=data_dir, img_size=img_size)
    df = pd.read_csv(csv_path)

    def save_comparison(img_name, auc_val, save_folder, prefix):
        try:
            idx = dataset.images.index(img_name)
        except ValueError:
            print(f"Didn't find {img_name} in validation set.")
            return

        _, image_tensor, gt_tensor = dataset[idx]
        image_input = image_tensor.unsqueeze(0).to(device)

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
        save_comparison(row['image'], row['AUC'], best_dir, "best")

    print("\nProcessing WORST images...")
    for _, row in worst_5.iterrows():
        save_comparison(row['image'], row['AUC'], worst_dir, "worst")
