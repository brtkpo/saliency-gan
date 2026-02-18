import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "results"
CSV_PATH = os.path.join(RESULTS_DIR, "results_val_full.csv")

df = pd.read_csv(CSV_PATH)

print("\n=== FIRST ROWS ===")
print(df.head())

print("\n=== BASIC STATS ===")
print(df.describe())

mean_metrics = df.mean(numeric_only=True)
print("\n=== MEAN METRICS ===")
print(mean_metrics)

plt.figure(figsize=(10, 5))
df[["AUC", "NSS", "CC", "SIM"]].boxplot()
plt.title("Metric distributions (validation set)")
plt.ylabel("Value")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n=== TOP 5 IMAGES (AUC) ===")
print(df.sort_values("AUC", ascending=False)[["image", "AUC"]].head())

print("\n=== WORST 5 IMAGES (AUC) ===")
print(df.sort_values("AUC", ascending=True)[["image", "AUC"]].head())
