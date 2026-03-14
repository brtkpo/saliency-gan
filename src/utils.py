import torch
import numpy as np


def prepare_tensors_for_visualization(
    dataset_item: tuple[str, torch.Tensor, torch.Tensor],
    gen: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Prepare original image, ground truth map, and predicted saliency map
    as NumPy arrays for visualization.

    Parameters
    ----------
    dataset_item : tuple
        Tuple containing (image_name, image_tensor, gt_tensor) from SaliconDataset.
    gen : torch.nn.Module
        Trained Generator model.
    device : torch.device
        Device to run the prediction on.

    Returns
    -------
    image_np : np.ndarray
        Original image normalized to [0, 1].
    gt_np : np.ndarray
        Ground truth saliency map normalized to [0, 1].
    pred_np : np.ndarray
        Predicted saliency map normalized to [0, 1].
    img_name : str
        Name of the image.
    """
    img_name, image, gt_map = dataset_item
    image_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = gen(image_tensor)

    # Convert tensors to NumPy arrays and normalize to [0, 1]
    pred_np = pred.squeeze().cpu().numpy()
    pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)

    gt_np = gt_map.squeeze().cpu().numpy()
    gt_np = (gt_np - gt_np.min()) / (gt_np.max() - gt_np.min() + 1e-8)

    image_np = image.squeeze().cpu().permute(1, 2, 0).numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)

    return image_np, gt_np, pred_np, img_name
