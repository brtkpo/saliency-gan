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
    """
    Main entry point for the Saliency GAN workflow.

    This script orchestrates the full pipeline: training, evaluation, result analysis,
    and visualization of saliency predictions, depending on the configuration.

    Parameters
    ----------
    None

    Behavior
    --------
    - If `mode` in configuration is "train", trains the GAN model.
    - If `mode` is "evaluate" or after training, evaluates the model and analyzes results.
    - If `vis.visualize_single` is set, visualizes the prediction for a single image.
    - If `vis.visualize_results` is True, visualizes the top and bottom images based on AUC.

    Notes
    -----
    Device selection (CPU or CUDA) is automatic.
    All outputs (results, checkpoints, visualizations) are saved to directories
    specified in the configuration file.

    Example
    -------
    python main.py --config path/to/config.json
    """
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
    vis = cfg.vis

    results_dir = Path(meta.results_dir)
    checkpoint_dir = Path(meta.checkpoint_dir)

    results_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    if meta.mode == "train":
        print("=== TRAINING ===")
        train_model(cfg, device)

    if meta.mode in ["train", "evaluate"]:
        print("=== EVALUATION ===")
        run_evaluation(cfg, device)

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
