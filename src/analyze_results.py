import pandas as pd
import matplotlib
from pathlib import Path

matplotlib.use("TkAgg")

from .config import Config


def analyze_results(cfg: Config) -> None:
    """
    Analyze saliency model results from a CSV file.

    Parameters
    ----------
    cfg : Config
        Application configuration containing meta, model, and visualization settings.

    Returns
    -------
    None
    """
    meta = cfg.meta

    results_dir = Path(meta.results_dir)
    split = getattr(meta, "split", "val")

    csv_name = f"results_{split}_full.csv"
    csv_path = results_dir / csv_name

    df = pd.read_csv(csv_path)

    print("\n=== FIRST ROWS ===")
    print(df.head())

    print("\n=== BASIC STATS ===")
    print(df.describe())

    mean_metrics = df.mean(numeric_only=True)
    print("\n=== MEAN METRICS ===")
    print(mean_metrics)

    print("\n=== TOP 5 IMAGES (AUC) ===")
    print(df.sort_values("AUC", ascending=False)[["image", "AUC"]].head())

    print("\n=== WORST 5 IMAGES (AUC) ===")
    print(df.sort_values("AUC", ascending=True)[["image", "AUC"]].head())
