import torch
import matplotlib.pyplot as plt

from .dataset import SaliconDataset
from .model import Generator
from .utils import prepare_tensors_for_visualization

def run_inference_visualize(
    data_dir: str,
    checkpoint_path: str,
    img_size: tuple[int, int],
    split: str,
    idx: int,
    device: torch.device,
) -> None:
    """
    Run inference on a single image and visualize the original image,
    ground truth saliency map, predicted saliency map, and overlay.

    Parameters
    ----------
    data_dir : str
        Path to dataset folder.
    checkpoint_path : str
        Path to the trained Generator model checkpoint (.pth).
    img_size : tuple of int, default=(224, 224)
        Size to which images and maps will be resized.
    split : str, default="val"
        Dataset split to use ('train' or 'val').
    idx : int, default=5
        Index of the image in the dataset to visualize.
    device : torch.device
        Device to run the model on.

    Returns
    -------
    None
        Displays matplotlib plots of the original image, GT, prediction, and overlay.
    """

    gen = Generator(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(checkpoint["gen_state_dict"])
    gen.eval()

    dataset = SaliconDataset(
        split=split,
        data_dir=data_dir,
        img_size=img_size,
        augment=False,
    )

    dataset_item = dataset[idx]
    image_np, gt_map, saliency, img_name = prepare_tensors_for_visualization(dataset_item, gen, device)

    plt.figure(figsize=(16,4))

    plt.subplot(1,4,1)
    plt.title("Original image")
    plt.imshow(image_np)
    plt.axis("off")

    plt.subplot(1,4,2)
    plt.title("Ground Truth (GT)")
    plt.imshow(gt_map, cmap="hot")
    plt.axis("off")

    plt.subplot(1,4,3)
    plt.title("Predicted saliency")
    plt.imshow(saliency, cmap="hot")
    plt.axis("off")

    plt.subplot(1,4,4)
    plt.title("Overlay (Pred)")
    plt.imshow(image_np)
    plt.imshow(saliency, cmap="hot", alpha=0.5)
    plt.axis("off")

    plt.tight_layout()
    plt.show()