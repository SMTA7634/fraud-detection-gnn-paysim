"""Load YAML config files from the configs/ directory."""

from pathlib import Path
import yaml

from fraud_detection.common.paths import CONFIGS_DIR


def load_config(filename: str) -> dict:
    """Load a YAML config file by filename (e.g. 'data.yaml').

    Returns a plain dict. Raises FileNotFoundError if the file doesn't exist.
    """
    path = CONFIGS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)
