from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np


def visualize_image(
    image_np: np.ndarray,
    gt_map: np.ndarray,
    saliency: np.ndarray,
    img_name: str,
    save_folder: Path,
    prefix: Optional[str] = None,
    nss_val: Optional[float] = None,
) -> None:
    """
    Generate and save a comparison plot of original image, ground truth, prediction, and overlay.

    Parameters
    ----------
    image_np : np.ndarray
        Original RGB image.
    gt_map : np.ndarray
        Ground truth saliency map.
    saliency : np.ndarray
        Predicted saliency map.
    img_name : str
        Image filename (used in title and saved file name).
    save_folder : Path
        Directory to save the visualization.
    prefix : str, optional
        Prefix for saved file name.
    nss_val : float, optional
        NSS value to show in the title of predicted map.

    Returns
    -------
    None
    """
    save_folder.mkdir(exist_ok=True, parents=True)

    plt.figure(figsize=(20, 5))
    plt.suptitle(f"Saliency Visualization for {Path(img_name).name}", fontsize=16)

    plt.subplot(1, 4, 1)
    plt.title("Original")
    plt.imshow(image_np)
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("Ground Truth (GT)")
    plt.imshow(gt_map, cmap="jet")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    title_pred = f"Predicted"
    if nss_val is not None:
        title_pred += f" (NSS: {nss_val:.4f})"
    plt.title(title_pred)
    plt.imshow(saliency, cmap="jet")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Overlay")
    plt.imshow(image_np)
    plt.imshow(saliency, cmap="jet", alpha=0.4)
    plt.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    filename = f"{prefix}_{Path(img_name).name}" if prefix else Path(img_name).name
    save_path = save_folder / filename
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")