from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config, supporting `_base_` inheritance."""

    path = Path(path)
    data = _read_yaml(path)

    base_files = data.pop("_base_", [])
    if isinstance(base_files, (str, Path)):
        base_files = [base_files]

    merged: Dict[str, Any] = {}
    for base in base_files or []:
        base_path = path.parent / Path(base)
        merged = _merge_dict(merged, load_config(base_path))

    merged = _merge_dict(merged, data)
    return merged


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


__all__ = ["load_config"]
