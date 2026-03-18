from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class MetaConfig:
    mode: str = "train"
    data_dir: str = "../data"
    results_dir: str = "../results"
    checkpoint_dir: str = "../checkpoints"


@dataclass
class ModelConfig:
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
    visualize_mode: str | None = None
    visualize_results: bool = False
    visualize_single: int | None = None


@dataclass
class Config:
    meta: MetaConfig
    model: ModelConfig
    vis: VisualizationConfig

def load_config(path: Path) -> Config:
    with open(path) as f:
        raw = json.load(f)

    meta = MetaConfig(**raw.get("meta", {}))
    model = ModelConfig(**raw.get("model_config", {}))
    vis = VisualizationConfig(**raw.get("visualization", {}))

    return Config(meta=meta, model=model, vis=vis)