import os
import torch
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from .dataset import SaliconDataset
from .model import Generator
from .utils import prepare_tensors_for_visualization


def visualize_results(
    data_dir: str,
    results_dir: str,
    checkpoint_path: str,
    img_size: tuple[int, int],
    device: torch.device,
) -> None:
    """
    Visualize the top and bottom predicted saliency maps based on AUC.

    Parameters
    ----------
    data_dir : str, optional
        Path to the dataset directory (default is "../data").
    results_dir : str, optional
        Path to the results directory containing CSV and to save visuals (default is "../results").
    checkpoint_path : str, optional
        Path to the generator model checkpoint (default is "../checkpoints/best_model.pth").
    img_size : tuple of int, optional
        Target image size as (height, width) (default is (224, 224)).
    device : torch.device or None, optional
        Torch device to use. If None, will use CUDA if available, otherwise CPU.

    Returns
    -------
    None
        Saves visualizations of top/bottom images to the results directory.
    """

    csv_path = os.path.join(results_dir, "results_val_full.csv")

    best_dir = os.path.join(results_dir, "visuals_best")
    worst_dir = os.path.join(results_dir, "visuals_worst")
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(worst_dir, exist_ok=True)

    gen = Generator(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(checkpoint["gen_state_dict"])
    gen.eval()

    dataset = SaliconDataset(split="val", data_dir=data_dir, img_size=img_size, augment=False)
    df = pd.read_csv(csv_path)

    def save_comparison(img_name: str, auc_val: float, save_folder: str, prefix: str) -> None:
        """
        Save a comparison plot of original image, ground truth, prediction, and overlay.

        Parameters
        ----------
        img_name : str
            Name of the image file.
        auc_val : float
            AUC score of the prediction.
        save_folder : str
            Directory to save the visualization.
        prefix : str
            Prefix for the saved file name.

        Returns
        -------
        None
        """
        try:
            idx = dataset.images.index(img_name)
        except ValueError:
            print(f"Didn't find {img_name} in validation set.")
            return

        dataset_item = dataset[idx]
        image_np, gt_map, saliency, img_name = prepare_tensors_for_visualization(dataset_item, gen, device)

        plt.figure(figsize=(20, 5))
        plt.suptitle(f"Saliency Visualization for {img_name})", fontsize=16)

        plt.subplot(1, 4, 1)
        plt.title(f"Original")
        plt.imshow(image_np)
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.title("Ground Truth (GT)")
        plt.imshow(gt_map, cmap='jet')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.title(f"Predicted (AUC: {auc_val:.4f})")
        plt.imshow(saliency, cmap='jet')
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.title("Overlay")
        plt.imshow(image_np)
        plt.imshow(saliency, cmap='jet', alpha=0.4)
        plt.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
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
