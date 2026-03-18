import torch
from pathlib import Path

from .dataset import SaliconDataset
from .model import Generator
from .utils import prepare_tensors_for_visualization
from .config import Config
from .visualize_image import visualize_image


def visualize_single(
    cfg: Config,
    device: torch.device,
) -> None:
    """
    Run inference on a single image and visualize the original image,
    ground truth saliency map, predicted saliency map, and overlay.

    Parameters
    ----------
    cfg : Config
        Application configuration containing meta, model, and visualization settings.
    device : torch.device
        Device to run the model on.

    Returns
    -------
    None
        Displays matplotlib plots of the original image, GT, prediction, and overlay.
    """
    meta = cfg.meta
    model = cfg.model
    vis = cfg.vis

    data_dir = meta.data_dir
    results_dir = Path(meta.results_dir)
    checkpoint_path = Path(meta.checkpoint_dir) / "best_model.pth"

    split = getattr(meta, "split", "val")
    img_size: tuple[int, int] = model.img_size
    idx: int = vis.visualize_single

    gen = Generator(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(checkpoint["gen_state_dict"])
    gen.eval()

    dataset = SaliconDataset(
        split=split,
        data_dir=Path(data_dir),
        img_size=img_size,
        augment=False,
    )

    dataset_item = dataset[idx]
    image_np, gt_map, saliency, img_name = prepare_tensors_for_visualization(dataset_item, gen, device)

    save_folder = results_dir / "visuals_single"
    visualize_image(image_np, gt_map, saliency, img_name, save_folder)
