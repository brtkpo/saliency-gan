import argparse
import json
import torch
from pathlib import Path

from src.train import train_model
from src.test_evaluate import run_evaluation
from src.analyze_results import analyze_results
from src.inference_visualize import run_inference_visualize
from src.visualize_results import visualize_results


def main():
    parser = argparse.ArgumentParser(description="Process a JSON config file.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the JSON configuration file",
    )
    args = parser.parse_args()
    config_path = Path(args.config)

    with open(config_path, "r") as f:
        cfg = json.load(f)
        # print("success")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = cfg.get("meta", {})
    mode = meta.get("mode", "train")
    data_dir = Path(meta.get("data_dir", "data"))
    results_dir = Path(meta.get("results_dir", "results"))
    checkpoint_dir = Path(meta.get("checkpoint_dir", "checkpoints"))
    results_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    model_cfg = cfg.get("model_config", {})
    img_size: tuple[int, int] = tuple(
        int(x) for x in model_cfg.get("img_size", (224, 224))
    )
    batch_size = model_cfg.get("batch_size", 4)
    lr = model_cfg.get("lr", 2e-4)
    num_epochs = model_cfg.get("num_epochs", 35)
    early_stop_patience = model_cfg.get("early_stop_patience", 7)
    lambda_l1 = model_cfg.get("lambda_l1", 10.0)
    lambda_kld = model_cfg.get("lambda_kld", 5.0)
    lambda_tv = model_cfg.get("lambda_tv", 0.001)

    vis_cfg = cfg.get("visualization", {})
    visualize_single = vis_cfg.get("visualize_single", None)
    visualize_results_flag = vis_cfg.get("visualize_results", False)

    if mode == "train":
        print("=== TRAINING ===")
        train_model(
            data_dir=data_dir,
            checkpoint_dir=checkpoint_dir,
            img_size=img_size,
            batch_size=batch_size,
            lr=lr,
            num_epochs=num_epochs,
            early_stop_patience=early_stop_patience,
            lambda_l1=lambda_l1,
            lambda_kld=lambda_kld,
            lambda_tv=lambda_tv,
            device=device,
        )

    if mode in ["train", "evaluate"]:
        print("=== EVALUATION ===")
        df, summary = run_evaluation(
            data_dir=str(data_dir),
            checkpoint_path=str(checkpoint_dir / "best_model.pth"),
            results_dir=str(results_dir),
            img_size=img_size,
            split="val",
            device=device,
        )

        print("=== ANALYSIS ===")
        analyze_results(results_dir=str(results_dir))

    if visualize_single is not None:
        print(f"=== VISUALIZING SINGLE IMAGE: {visualize_single} ===")
        run_inference_visualize(
            data_dir=str(data_dir),
            checkpoint_path=str(checkpoint_dir / "best_model.pth"),
            img_size=img_size,
            idx=visualize_single,
            device=device,
            split="val",
        )

    if visualize_results_flag:
        print("=== VISUALIZING TOP/BOTTOM RESULTS ===")
        visualize_results(
            data_dir=str(data_dir),
            results_dir=str(results_dir),
            checkpoint_path=str(checkpoint_dir / "best_model.pth"),
            img_size=img_size,
            device=device,
        )


if __name__ == "__main__":
    main()
