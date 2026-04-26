from __future__ import annotations

from pathlib import Path
import pickle
from typing import Dict, Any

import numpy as np


def load_feature_split(path: str | Path, modalities: tuple[str, ...] = ("grouprec", "convnext")) -> Dict[str, Any]:
    """Load pre-extracted features from a split pkl file.

    Returns a dict with raw modality arrays and labels.
    """
    p = Path(path)
    with p.open("rb") as f:
        data = pickle.load(f)

    out: Dict[str, Any] = {"labels": np.asarray(data["labels"], dtype=np.int64)}
    for m in modalities:
        if m not in data:
            raise KeyError(f"Missing modality '{m}' in {p}")
        out[m] = np.asarray(data[m], dtype=np.float32)

    n = out["labels"].shape[0]
    for m in modalities:
        if out[m].shape[0] != n:
            raise ValueError(f"Sample count mismatch for modality '{m}': {out[m].shape[0]} vs {n}")

    return out
