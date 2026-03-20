from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class MetaConfig:
    """
    General configuration related to experiment setup and file paths.

    Parameters
    ----------
    mode : str, default="train"
        Execution mode of the pipeline (e.g., "train", "evaluate").
    data_dir : str, default="../data"
        Path to the dataset directory.
    results_dir : str, default="../results"
        Path where evaluation results and visualizations will be saved.
    checkpoint_dir : str, default="../checkpoints"
        Path where model checkpoints will be stored.
    """
    mode: str = "train"
    data_dir: str = "../data"
    results_dir: str = "../results"
    checkpoint_dir: str = "../checkpoints"


@dataclass
class ModelConfig:
    """
    Configuration of model architecture and training hyperparameters.

    Parameters
    ----------
    img_size : tuple[int, int], default=(224, 224)
        Input image resolution (height, width).
    batch_size : int, default=4
        Number of samples per training batch.
    lr : float, default=2e-4
        Learning rate for optimizers.
    num_epochs : int, default=35
        Maximum number of training epochs.
    early_stop_patience : int, default=7
        Number of epochs without improvement before early stopping.
    lambda_l1 : float, default=10.0
        Weight for L1 reconstruction loss.
    lambda_kld : float, default=5.0
        Weight for KL divergence loss.
    lambda_tv : float, default=0.001
        Weight for total variation loss.
    """
    img_size: tuple[int, int] = (224, 224)
    batch_size: int = 4
    lr: float = 2e-4
    num_epochs: int = 35
    early_stop_patience: int = 7
    lambda_l1: float = 10.0
    lambda_kld: float = 5.0
    lambda_tv: float = 0.001


@dataclass
class VisualizationConfig:
    """
    Configuration for visualization and inference settings.

    Parameters
    ----------
    visualize_results : bool, default=False
        Whether to generate visualizations for best and worst results.
    visualize_single : int or None, default=None
        Index of a single image to visualize from the dataset.
    """
    visualize_results: bool = False
    visualize_single: int | None = None


@dataclass
class Config:
    """
    Main configuration object grouping all sub-configurations.

    Parameters
    ----------
    meta : MetaConfig
        General experiment configuration.
    model : ModelConfig
        Model and training hyperparameters.
    vis : VisualizationConfig
        Visualization-related settings.
    """
    meta: MetaConfig
    model: ModelConfig
    vis: VisualizationConfig


def load_config(path: Path) -> Config:
    """
    Load configuration from a JSON file and convert it into Config dataclasses.

    Parameters
    ----------
    path : Path
        Path to the JSON configuration file.

    Returns
    -------
    Config
        Configuration object containing:
        - meta : MetaConfig with general paths and execution mode
        - model : ModelConfig with training hyperparameters
        - vis : VisualizationConfig with visualization settings
    """
    with open(path) as f:
        raw = json.load(f)

    meta = MetaConfig(**raw.get("meta", {}))
    model = ModelConfig(**raw.get("model_config", {}))
    vis = VisualizationConfig(**raw.get("visualization", {}))

    return Config(meta=meta, model=model, vis=vis)
