from pathlib import Path
from shutil import copy
from huggingface_hub import hf_hub_download
from .config import Config

def load_model(cfg: Config) -> Path:
    """
    Returns the path to the best model checkpoint.
    Downloads from HF if not present locally.

    Parameters
    ----------
    cfg : Config

    Returns
    -------
    Path
        Path to checkpoint in local checkpoint_dir.
    """
    meta = cfg.meta
    checkpoint_dir = Path(meta.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    local_path = checkpoint_dir / "best_model.pth"

    if local_path.exists():
        return local_path
    else:
        print("Checkpoint not found locally. Downloading from Hugging Face...")
        hf_path = hf_hub_download(
            repo_id="brtkpo/Saliency_GAN",
            filename="best_model.pth"
        )
        copy(hf_path, local_path)
        print(f"Downloaded and saved to {local_path}")
        return local_path