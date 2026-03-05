import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import os

def analyze_results(results_dir: str = "results", csv_name: str = "results_val_full.csv") -> None:
    """
    Analyze saliency model results from a CSV file.

    Parameters
    ----------
    results_dir : str, optional
        Path to the directory containing the results CSV file. Default is "results".
    csv_name : str, optional
        Name of the CSV file to analyze. Default is "results_val_full.csv".

    Returns
    -------
    None
    """
    csv_path = os.path.join(results_dir, csv_name)
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
