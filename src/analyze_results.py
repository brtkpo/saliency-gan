import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import os

def analyze_results(results_dir="results", csv_name="results_val_full.csv"):

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
