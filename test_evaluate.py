import os
import torch
import numpy as np
import pandas as pd
import cv2
import scipy.io as sio

from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from tqdm import tqdm

from model import Generator
from dataset import SaliconDataset

# CONFIG
DATA_DIR = "data"
SPLIT = "val"

fix_dir = os.path.join(DATA_DIR, "fixations", "val")
dataset = SaliconDataset(split="val", data_dir=DATA_DIR, img_size=224)

IMG_SIZE = (224, 224)
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CHECKPOINT_PATH = "checkpoints/checkpoint_epoch_15_valL1_0.0713.pth"
CHECKPOINT_PATH = "checkpoints/checkpoint_epoch_20_valL1_0.0665.pth"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# METRICS
def normalize_map(m):
    m = m.astype(np.float32)
    return m / (m.sum() + 1e-7)

def compute_metrics(pred_map, gt_map, fixations):
    h, w = pred_map.shape

    # fixation mask
    fix_mask = np.zeros((h, w), dtype=np.uint8)
    for x, y in fixations:
        x, y = int(x), int(y)
        if 0 <= x < w and 0 <= y < h:
            fix_mask[y, x] = 1

    if fix_mask.sum() == 0:
        return None

    # AUC
    auc = roc_auc_score(fix_mask.flatten(), pred_map.flatten())

    # NSS
    mu, std = pred_map.mean(), pred_map.std()
    nss_map = (pred_map - mu) / (std + 1e-7)
    nss = nss_map[fix_mask == 1].mean()

    # CC
    cc, _ = pearsonr(pred_map.flatten(), gt_map.flatten())

    # KL
    p = normalize_map(pred_map)
    g = normalize_map(gt_map)
    kl = np.sum(g * np.log((g + 1e-7) / (p + 1e-7)))

    # SIM
    sim = np.sum(np.minimum(p, g))

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
map_dir = os.path.join(DATA_DIR, "maps", "val")   # GT maps tylko dla val
fix_dir = os.path.join(DATA_DIR, "fixations", SPLIT)

dataset = SaliconDataset(split="val", data_dir=DATA_DIR, img_size=IMG_SIZE)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

results = []

# EVALUATION LOOP
for img_name, img_tensor, gt_tensor in tqdm(loader, desc="Evaluating"):
    img_tensor = img_tensor.to(DEVICE)

    # predict saliency
    with torch.no_grad():
        pred = gen(img_tensor).cpu().numpy()[0, 0]

    pred = cv2.resize(pred, IMG_SIZE)
    pred = normalize_map(pred)

    gt = gt_tensor.numpy()[0, 0]
    gt = normalize_map(gt)

    # load fixations
    base = os.path.splitext(img_name[0])[0]
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
                    fixations.extend(fix)

    if len(fixations) == 0:
        continue

    metrics = compute_metrics(pred, gt, fixations)
    if metrics is None:
        continue

    metrics["image"] = img_name[0]
    results.append(metrics)

# SAVE RESULTS
df = pd.DataFrame(results)
df.to_csv(os.path.join(RESULTS_DIR, f"results_{SPLIT}_full.csv"), index=False)

summary = df.mean(numeric_only=True).to_frame(name="Mean").T
summary.to_csv(os.path.join(RESULTS_DIR, f"results_{SPLIT}_summary.csv"), index=False)

print("\n=== RESULTS SUMMARY ===")
print(summary)
