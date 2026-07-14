"""Frozen train/val/test split: single source of truth for loading and verifying it.

All nine experiments read the same frozen split (``configs/splits.npz``, produced by
``scripts/make_splits.py``). This module owns both the fingerprint definition and the
load-time verification, so the generator and the consumers can never drift apart.

The fingerprint is a cheap content hash of the dataset's ``(labels, snr_db)`` vectors --
not the heavy signal array. That is enough to detect "the dataset was regenerated but the
split file is stale", which would otherwise silently produce results on a mismatched split.

HDF5 reading convention: dataset.h5 is written by MATLAB ``save('-v7.3')``, which stores
axes in column-major order, so the 1-D vectors are shaped (1, N) and must be ravel()'d.
See docs/dataset.md.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np


def read_labels_and_snr(h5_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read (labels, snr_db) from dataset.h5, handling the (1, N) MATLAB layout."""
    with h5py.File(str(h5_path), "r") as f:
        labels = np.asarray(f["labels"][:]).ravel().astype(np.int64)
        snr_db = np.asarray(f["snr_db"][:]).ravel().astype(np.float32)
    if labels.shape != snr_db.shape:
        raise ValueError(
            f"labels {labels.shape} and snr_db {snr_db.shape} length mismatch"
        )
    return labels, snr_db


def dataset_fingerprint(labels: np.ndarray, snr_db: np.ndarray) -> str:
    """Content hash of the dataset's label/SNR vectors (16 hex chars)."""
    h = hashlib.sha256()
    h.update(np.asarray(labels).astype(np.int64).tobytes())
    # Round SNR before hashing so float noise does not change the hash across platforms.
    h.update(np.round(np.asarray(snr_db), 6).astype(np.float32).tobytes())
    return h.hexdigest()[:16]


def load_splits(
    splits_path: str | Path,
    dataset_path: str | Path | None = None,
    verify_fingerprint: bool = True,
) -> Dict[str, np.ndarray]:
    """Load the frozen split, optionally verifying it matches the dataset.

    Parameters
    ----------
    splits_path : path to ``configs/splits.npz``.
    dataset_path : path to ``dataset.h5``. Required for verification.
    verify_fingerprint : if True and the split file carries a ``dataset_fingerprint``,
        recompute it from the dataset and raise on mismatch.

    Raises
    ------
    RuntimeError
        If the dataset's fingerprint does not match the one recorded in the split file --
        i.e. the dataset was regenerated but the split was not.
    """
    data = np.load(str(splits_path), allow_pickle=True)
    splits = {
        "train": data["train_idx"],
        "val": data["val_idx"],
        "test": data["test_idx"],
    }

    if not verify_fingerprint or dataset_path is None:
        return splits
    if "dataset_fingerprint" not in data:
        return splits  # older split files carry no fingerprint; nothing to check

    expected = str(np.asarray(data["dataset_fingerprint"]).ravel()[0])
    labels, snr_db = read_labels_and_snr(dataset_path)
    actual = dataset_fingerprint(labels, snr_db)
    if actual != expected:
        raise RuntimeError(
            f"Dataset fingerprint mismatch: {dataset_path} hashes to {actual!r}, but "
            f"{splits_path} was built against {expected!r}. The dataset was regenerated "
            f"without regenerating the split. Re-run: "
            f"python scripts/make_splits.py --out {splits_path}"
        )
    return splits
