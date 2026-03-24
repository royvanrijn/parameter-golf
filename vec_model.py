#!/usr/bin/env python3
"""Utilities for loading vector embedding artifacts produced by ``train_vec.py``."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


def load_vec_artifact(path: str | Path) -> dict[str, Any]:
    """Load a vector artifact file and return its raw payload dictionary."""
    artifact_path = Path(path)
    with artifact_path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected vec artifact payload type: {type(payload)!r}")
    if "embeddings" not in payload:
        raise KeyError("vec artifact missing required key: 'embeddings'")
    return payload


def get_vec_table(payload: dict[str, Any]) -> np.ndarray:
    """Extract the [vocab_size, vec_dim] embedding table as float32."""
    table = np.asarray(payload["embeddings"], dtype=np.float32)
    if table.ndim != 2:
        raise ValueError(f"Expected 2D embedding table, got shape={table.shape}")
    return table
