"""Shared helpers: config loading, paths, logging."""
import logging
import random
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str = "configs/config.yaml") -> dict:
    """Load the YAML configuration relative to the project root."""
    with open(PROJECT_ROOT / path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(config: dict, key: str) -> Path:
    """Return an absolute, existing directory for a configured path key."""
    directory = PROJECT_ROOT / config["paths"][key]
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs."""
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str) -> logging.Logger:
    """Return a console logger with a consistent format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)
