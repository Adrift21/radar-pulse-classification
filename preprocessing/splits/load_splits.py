"""
load_splits.py — Load pre-computed train/val/test indices.

This is the utility that Module C training scripts will call. It reads
the .npy files produced by make_splits.py and (optionally) verifies the
dataset hash to catch the "splits are stale because dataset.h5 was
regenerated" failure mode early.

Usage:
    from preprocessing.splits.load_splits import load_splits

    train_idx, val_idx, test_idx = load_splits(
        splits_dir="data_generation/synthetic_samples/splits",
        h5_path="data_generation/synthetic_samples/dataset.h5",
        verify_hash=True,
    )

    # Pass to RadarPulseDataset via the `indices` argument (or a Subset wrapper)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def _compute_hash(h5_path: Path) -> str:
    """Recompute the (labels, snr_db) SHA256 — must match make_splits.py."""
    with h5py.File(h5_path, "r") as f:
        labels = np.asarray(f["labels"][:]).ravel().astype(np.uint8)
        snr_db = np.asarray(f["snr_db"][:]).ravel().astype(np.float32)
    h = hashlib.sha256()
    h.update(labels.tobytes())
    h.update(snr_db.tobytes())
    return h.hexdigest()


def load_splits(
    splits_dir: Path | str,
    h5_path: Path | str | None = None,
    verify_hash: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load train/val/test indices from .npy files.

    Parameters
    ----------
    splits_dir : path to the directory containing the .npy files
    h5_path    : path to dataset.h5 — required if verify_hash=True
    verify_hash : if True, recompute the dataset hash and compare against
        split_metadata.json. Raises RuntimeError on mismatch.

    Returns
    -------
    train_idx, val_idx, test_idx : 1-D uint32 arrays (sorted ascending)
    """
    splits_dir = Path(splits_dir)

    train_idx = np.load(splits_dir / "train_idx.npy")
    val_idx = np.load(splits_dir / "val_idx.npy")
    test_idx = np.load(splits_dir / "test_idx.npy")

    if verify_hash:
        if h5_path is None:
            raise ValueError("verify_hash=True requires h5_path")
        metadata_path = splits_dir / "split_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Cannot verify hash: {metadata_path} not found")
        with open(metadata_path) as f:
            metadata = json.load(f)
        expected_hash = metadata["dataset_hash_sha256"]
        actual_hash = _compute_hash(Path(h5_path))
        if expected_hash != actual_hash:
            raise RuntimeError(
                "Dataset hash mismatch — splits are stale.\n"
                f"  expected (from {metadata_path.name}): {expected_hash[:16]}...\n"
                f"  actual   (from {h5_path}):           {actual_hash[:16]}...\n"
                "Re-run preprocessing/splits/make_splits.py to regenerate."
            )

    return train_idx, val_idx, test_idx


def load_split_metadata(splits_dir: Path | str) -> dict:
    """Load the split_metadata.json file as a dict (for logging)."""
    splits_dir = Path(splits_dir)
    with open(splits_dir / "split_metadata.json") as f:
        return json.load(f)
