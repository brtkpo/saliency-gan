import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataset import SaliconDataset
from model import Generator
import utils

CHECKPOINT_PATH = "checkpoints/checkpoint_epoch_20_valL1_0.0665.pth"
DATA_DIR = "data"
IMG_SIZE = (224, 224)
BATCH_SIZE = 1  # 1 dla dokładnych metryk per obraz

def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Dataset (bez augmentacji!)
    val_dataset = SaliconDataset(
        split="val",
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        augment=False
    )

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model = Generator
    model = Generator(in_channels=3, out_channels=1).to(device)

    # Wczytanie checkpointu
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["gen_state_dict"])
    model.eval()

    all_metrics = []

    print(f"Calculating metrics for {len(val_dataset)} images...")

    with torch.no_grad():
        for img_names, images, sal_maps in val_loader:
            images = images.to(device)
            sal_maps = sal_maps.to(device)

            preds = model(images)

            # Metryki
            m_auc = utils.auc_judd(preds, sal_maps).item()
            m_nss = utils.nss(preds, sal_maps).item()
            m_cc  = utils.correlationCoefficient(preds, sal_maps).item()
            m_kl  = utils.klDivergence(preds, sal_maps).item()
            m_sim = utils.similarity(preds, sal_maps).item()

            all_metrics.append({
                "image": img_names[0],
                "AUC": m_auc,
                "NSS": m_nss,
                "CC": m_cc,
                "KLDiv": m_kl,
                "SIM": m_sim
            })

    df = pd.DataFrame(all_metrics)

    print("\n=== BASIC STATS ===")
    print(df.describe())

    print("\n=== MEAN METRICS ===")
    print(df.mean(numeric_only=True))

    # zapis do pliku
    df.to_csv("evaluation_results.csv", index=False)
    print("\nSaved to evaluation_results.csv")


if __name__ == "__main__":
    run_evaluation()
