import argparse
import torch
from pathlib import Path

from src.config import load_config
from src.train import train_model
from src.test_evaluate import run_evaluation
from src.analyze_results import analyze_results
from src.visualize_single import visualize_single
from src.visualize_best_worst import visualize_best_worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the JSON configuration file",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = cfg.meta
    model = cfg.model
    vis = cfg.vis

    data_dir = Path(meta.data_dir)
    results_dir = Path(meta.results_dir)
    checkpoint_dir = Path(meta.checkpoint_dir)

    results_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    if meta.mode == "train":
        print("=== TRAINING ===")
        train_model(cfg, device)

    if meta.mode in ["train", "evaluate"]:
        print("=== EVALUATION ===")
        df, summary = run_evaluation(cfg, device)

        print("=== ANALYSIS ===")
        analyze_results(cfg)

    if vis.visualize_single is not None:
        print(f"=== VISUALIZING SINGLE IMAGE: {vis.visualize_single} ===")
        visualize_single(cfg, device)

    if vis.visualize_results:
        print("=== VISUALIZING TOP/BOTTOM RESULTS ===")
        visualize_best_worst(cfg, device)


if __name__ == "__main__":
    main()
