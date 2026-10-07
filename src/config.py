"""Small config loader shared by every phase script."""
from __future__ import annotations

import pathlib
import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_config(path: str | pathlib.Path = None) -> dict:
    if path is None:
        path = PROJECT_ROOT / "config" / "pipeline_config.yaml"
    path = pathlib.Path(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve(path_str: str) -> pathlib.Path:
    """Resolve a config path (which may be relative) against the project root."""
    p = pathlib.Path(path_str)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p
