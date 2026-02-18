import os
import torch
import numpy as np
import pandas as pd
import cv2
import scipy.io as sio
from PIL import Image

from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from tqdm import tqdm

from model import Generator
from dataset import SaliconDataset

DATA_DIR = "data"
SPLIT = "val"

fix_dir = os.path.join(DATA_DIR, "fixations", "val")
dataset = SaliconDataset(split="val", data_dir=DATA_DIR, img_size=224)

IMG_SIZE = (224, 224)
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "checkpoints/best_model.pth"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# METRICS
def normalize_map(m):
    m = m.astype(np.float32)
    return m / (m.sum() + 1e-7)

def compute_metrics(pred_map, gt_map, fixations, orig_w, orig_h):
    h, w = pred_map.shape

    # fixation mask
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
    pred_flat = pred_map.flatten()
    fix_flat = fix_mask.flatten()

    try:
        auc = roc_auc_score(fix_flat, pred_flat)
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

# LOAD MODEL
gen = Generator(in_channels=3, out_channels=1).to(DEVICE)
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
gen.load_state_dict(ckpt["gen_state_dict"])
gen.eval()

# DATASET
img_dir = os.path.join(DATA_DIR, "images", SPLIT)
#map_dir = os.path.join(DATA_DIR, "maps", "val")   # GT maps tylko dla val
fix_dir = os.path.join(DATA_DIR, "fixations", SPLIT)

dataset = SaliconDataset(split="val", data_dir=DATA_DIR, img_size=IMG_SIZE)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

results = []

# EVALUATION LOOP
for img_name, img_tensor, gt_tensor in tqdm(loader, desc="Evaluating"):
    img_tensor = img_tensor.to(DEVICE)
    img_filename = img_name[0]

    with torch.no_grad():
        pred = gen(img_tensor).cpu().numpy()[0, 0]

    pred = pred.astype(np.float32)
    gt = gt_tensor.numpy()[0, 0].astype(np.float32)

    orig_img_path = os.path.join(img_dir, img_filename)
    with Image.open(orig_img_path) as orig_img:
        orig_w, orig_h = orig_img.size

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

# SAVE RESULTS
df = pd.DataFrame(results)
df.to_csv(os.path.join(RESULTS_DIR, f"results_{SPLIT}_full.csv"), index=False)

summary = df.mean(numeric_only=True).to_frame(name="Mean").T
summary.to_csv(os.path.join(RESULTS_DIR, f"results_{SPLIT}_summary.csv"), index=False)

print("\n=== RESULTS SUMMARY ===")
print(summary)