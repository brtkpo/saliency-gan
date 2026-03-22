import torch
import pandas as pd
from pathlib import Path

from .dataset import SaliconDataset
from .model import Generator
from .utils import prepare_tensors_for_visualization
from .config import Config
from .visualize_image import visualize_image
from .load_model import load_model


def visualize_best_worst(cfg: Config, device: torch.device) -> None:
    """
    Visualize the top and bottom predicted saliency maps based on AUC.

    Parameters
    ----------
    cfg : Config
        Application configuration containing meta, model, and visualization settings.
    device : torch.device
        Torch device to use for model inference.

    Returns
    -------
    None
        Saves visualizations of top/bottom images to the results directory.
    """

    meta = cfg.meta
    model = cfg.model

    data_dir = Path(meta.data_dir)
    results_dir = Path(meta.results_dir)
    #checkpoint_path = Path(meta.checkpoint_dir) / "best_model.pth"
    checkpoint_path = load_model(cfg)
    #load_model(cfg, device)

    best_dir = results_dir / "visuals_best"
    worst_dir = results_dir / "visuals_worst"
    best_dir.mkdir(exist_ok=True)
    worst_dir.mkdir(exist_ok=True)

    gen = Generator(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(checkpoint["gen_state_dict"])
    gen.eval()

    dataset = SaliconDataset(
        split="val", data_dir=data_dir, img_size=model.img_size, augment=False
    )
    df = pd.read_csv(results_dir / "results_val_full.csv")

    best_5 = df.sort_values("NSS", ascending=False).head(5)
    worst_5 = df.sort_values("NSS", ascending=True).head(5)

    print("\nProcessing BEST images...")
    for _, row in best_5.iterrows():
        idx = [p.name for p in dataset.images].index(Path(row["image"]).name)
        dataset_item = dataset[idx]
        image_np, gt_map, saliency, img_name = prepare_tensors_for_visualization(dataset_item, gen, device)
        visualize_image(image_np, gt_map, saliency, img_name, best_dir, prefix="best", nss_val=row["NSS"])

    print("\nProcessing WORST images...")
    for _, row in worst_5.iterrows():
        idx = [p.name for p in dataset.images].index(Path(row["image"]).name)
        dataset_item = dataset[idx]
        image_np, gt_map, saliency, img_name = prepare_tensors_for_visualization(dataset_item, gen, device)
        visualize_image(image_np, gt_map, saliency, img_name, worst_dir, prefix="worst", nss_val=row["NSS"])
