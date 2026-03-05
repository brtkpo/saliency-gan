import os
import torch
import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from tqdm import tqdm

from .model import Generator
from .dataset import SaliconDataset

# METRICS
def normalize_map(m):
    """Normalize map to sum to 1"""
    m = m.astype(np.float32)
    return m / (m.sum() + 1e-7)

def compute_metrics(pred_map, gt_map, fixations, orig_w, orig_h):
    """Compute AUC, NSS, CC, KLDiv, SIM for one image"""
    h, w = pred_map.shape
    fix_mask = np.zeros((h, w), dtype=np.uint8)

    scale_x = w / orig_w
    scale_y = h / orig_h

    for fix in fixations:
        try:
            x, y = fix[0], fix[1]
        except IndexError:
            x, y = fix['x'][0][0], fix['y'][0][0]

        scaled_x = int(x * scale_x)
        scaled_y = int(y * scale_y)

        scaled_x = np.clip(scaled_x, 0, w - 1)
        scaled_y = np.clip(scaled_y, 0, h - 1)

        fix_mask[scaled_y, scaled_x] = 1

    if fix_mask.sum() == 0:
        return None

    # AUC
    try:
        auc = roc_auc_score(fix_mask.flatten(), pred_map.flatten())
    except ValueError:
        auc = 0.5

    # NSS
    mu, std = pred_map.mean(), pred_map.std()
    nss_map = (pred_map - mu) / (std + 1e-7)
    nss = np.mean(nss_map[fix_mask == 1])

    # CC
    cc, _ = pearsonr(pred_map.flatten(), gt_map.flatten())

    # KL
    p = normalize_map(pred_map)
    g = normalize_map(gt_map)
    kl = np.sum(g * np.log((g + 1e-7) / (p + 1e-7)))

    # SIM
    p_sim = (pred_map - pred_map.min()) / (pred_map.max() - pred_map.min() + 1e-7)
    g_sim = (gt_map - gt_map.min()) / (gt_map.max() - gt_map.min() + 1e-7)

    p_sim = p_sim / p_sim.sum()
    g_sim = g_sim / g_sim.sum()
    sim = np.sum(np.minimum(p_sim, g_sim))

    return {
        "AUC": auc,
        "NSS": nss,
        "CC": cc,
        "KLDiv": kl,
        "SIM": sim
    }

def run_evaluation(
    data_dir: str,
    checkpoint_path: str,
    results_dir: str,
    split: str = "val",
    img_size: tuple = (224, 224),
    device: torch.device = None,
):
    """Evaluate saliency model on dataset and save results"""

    os.makedirs(results_dir, exist_ok=True)

    # Load dataset
    dataset = SaliconDataset(split=split, data_dir=data_dir, img_size=img_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load model
    gen = Generator(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(checkpoint["gen_state_dict"])
    gen.eval()

    # Fixations folder
    fix_dir = os.path.join(data_dir, "fixations", split)

    results = []

    for img_name, img_tensor, gt_tensor in tqdm(loader, desc="Evaluating"):
        img_tensor = img_tensor.to(device)
        img_filename = img_name[0]

        with torch.no_grad():
            pred = gen(img_tensor).cpu().numpy()[0, 0]

        pred = pred.astype(np.float32)
        gt = gt_tensor.numpy()[0, 0].astype(np.float32)

        # Original image size
        orig_img_path = os.path.join(data_dir, "images", split, img_filename)
        with Image.open(orig_img_path) as orig_img:
            orig_w, orig_h = orig_img.size

        # Load fixation file
        base = os.path.splitext(img_filename)[0]
        mat_path = os.path.join(fix_dir, base + ".mat")
        if not os.path.exists(mat_path):
            continue

        mat = sio.loadmat(mat_path)
        fixations = []

        if "gaze" in mat:
            for subj in mat["gaze"][0]:
                if "fixations" in subj.dtype.names:
                    fix = subj["fixations"]
                    if fix.size > 0:
                        for f in fix:
                            fixations.append(f)

        if len(fixations) == 0:
            continue

        metrics = compute_metrics(pred, gt, fixations, orig_w, orig_h)
        if metrics is None:
            continue

        metrics["image"] = img_filename
        results.append(metrics)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(results_dir, f"results_{split}_full.csv"), index=False)
    summary = df.mean(numeric_only=True).to_frame(name="Mean").T
    summary.to_csv(os.path.join(results_dir, f"results_{split}_summary.csv"), index=False)

    print("\n=== RESULTS SUMMARY ===")
    print(summary)
    return df, summary


# if __name__ == "__main__":
#     run_evaluation(
#         data_dir="../data",
#         checkpoint_path="../checkpoints/best_model.pth",
#         results_dir="../results",
#         split="val",
#         img_size=(224, 224),
#     )