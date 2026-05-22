"""Generate frozen train/val/test index splits for the radar pulse dataset.

This script reads ONLY the label and SNR vectors from dataset.h5 (not the
signals themselves), produces a 70/15/15 split that is stratified jointly
on (class, SNR), and writes the index arrays to a single .npz file that
every experiment (Module C) and the SNR-robustness analysis (Module D)
will read. Freezing the split guarantees all 9 experiments and Module D
see the exact same test set -- the cleanest setup for fair, reproducible
academic comparison (decisions.md, 2026-05-04 Train/Val/Test entry).

Stratification
--------------
We stratify on the joint key (label, snr_db). With 8 classes x 16 SNR
points = 128 groups and ~312 samples per group, even the smallest split
(15% test ~= 47 samples/group) is comfortable for sklearn's stratified
splitter. This guarantees the test set contains every (class, SNR) cell,
which Module D's SNR-stratified accuracy curves depend on.

HDF5 reading convention
-----------------------
dataset.h5 was written by MATLAB save('-v7.3'), which stores axes in
column-major order. The 1-D vectors are therefore shaped (1, N) and must
be ravel()'d. See project_context.md "HDF5 Storage Convention".

Usage
-----
    python scripts/make_splits.py \
        --dataset data_generation/synthetic_samples/dataset.h5 \
        --out configs/splits.npz

    # defaults match the project layout, so usually just:
    python scripts/make_splits.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

# --- Project defaults (decisions.md) ------------------------------------
DEFAULT_DATASET = "data_generation/synthetic_samples/dataset.h5"
DEFAULT_OUT = "configs/splits.npz"
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
MASTER_SEED = 42


# ---------------------------------------------------------------------
# HDF5 reading (column-major convention)
# ---------------------------------------------------------------------
def read_labels_and_snr(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read (labels, snr_db) from dataset.h5, handling the (1, N) layout."""
    with h5py.File(h5_path, "r") as f:
        labels = np.asarray(f["labels"][:]).ravel().astype(np.int64)
        snr_db = np.asarray(f["snr_db"][:]).ravel().astype(np.float32)
    if labels.shape != snr_db.shape:
        raise ValueError(
            f"labels {labels.shape} and snr_db {snr_db.shape} length mismatch"
        )
    return labels, snr_db


def dataset_fingerprint(labels: np.ndarray, snr_db: np.ndarray) -> str:
    """A cheap content hash so we can detect if the dataset changed.

    Hashing labels + snr (not the heavy signals) is enough to catch a
    different dataset while staying fast.
    """
    h = hashlib.sha256()
    h.update(labels.tobytes())
    # round SNR to avoid float noise across platforms before hashing
    h.update(np.round(snr_db, 6).astype(np.float32).tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------
def make_stratified_splits(
    labels: np.ndarray,
    snr_db: np.ndarray,
    seed: int = MASTER_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-stage stratified split on the joint (label, snr) key.

    Returns sorted train/val/test index arrays (disjoint, covering all N).
    """
    n = labels.shape[0]
    all_idx = np.arange(n)

    # Joint stratification key. Encode (label, snr) into a single integer.
    # SNR values are on a discrete 2 dB grid; map them to integer bins.
    snr_unique = np.unique(snr_db)
    snr_to_bin = {v: i for i, v in enumerate(snr_unique)}
    snr_bin = np.array([snr_to_bin[v] for v in snr_db], dtype=np.int64)
    strat_key = labels * len(snr_unique) + snr_bin  # unique per (label, snr)

    # Stage 1: carve out TEST (15%) from the whole set.
    train_val_idx, test_idx = train_test_split(
        all_idx,
        test_size=TEST_FRAC,
        random_state=seed,
        shuffle=True,
        stratify=strat_key,
    )

    # Stage 2: split the remaining 85% into TRAIN and VAL.
    # val proportion *within* the remaining set:
    val_within = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)  # 0.15 / 0.85
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_within,
        random_state=seed,
        shuffle=True,
        stratify=strat_key[train_val_idx],
    )

    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


# ---------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------
def verify_splits(
    labels: np.ndarray,
    snr_db: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    """Assert disjointness, full coverage, and balanced distributions."""
    n = labels.shape[0]

    # 1) Disjoint + complete coverage
    union = np.concatenate([train_idx, val_idx, test_idx])
    assert union.size == n, f"Coverage: {union.size} != {n}"
    assert np.unique(union).size == n, "Splits overlap (duplicate indices)"

    # 2) Fractions roughly match targets
    fr = np.array([train_idx.size, val_idx.size, test_idx.size]) / n
    targets = np.array([TRAIN_FRAC, VAL_FRAC, TEST_FRAC])
    assert np.allclose(fr, targets, atol=0.01), f"Fractions off: {fr}"

    # 3) Per-class balance preserved in every split (within +/- 1%)
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        cls_counts = np.bincount(labels[idx], minlength=8)
        cls_frac = cls_counts / cls_counts.sum()
        assert np.allclose(cls_frac, 1 / 8, atol=0.01), (
            f"{name} class imbalance: {cls_frac}"
        )

    # 4) SNR distribution preserved in the test set (Module D depends on it)
    snr_unique = np.unique(snr_db)
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        counts = np.array([np.sum(snr_db[idx] == s) for s in snr_unique])
        frac = counts / counts.sum()
        assert np.allclose(frac, 1 / snr_unique.size, atol=0.02), (
            f"{name} SNR imbalance: {frac}"
        )

    print("  [verify] disjoint + full coverage .......... OK")
    print(
        f"  [verify] fractions (train/val/test) ........ "
        f"{fr[0]:.3f}/{fr[1]:.3f}/{fr[2]:.3f}"
    )
    print("  [verify] per-class balance (all splits) .... OK")
    print("  [verify] per-SNR balance (all splits) ...... OK")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    args = parser.parse_args()

    h5_path = Path(args.dataset)
    out_path = Path(args.out)

    if not h5_path.exists():
        print(f"ERROR: dataset not found: {h5_path}", file=sys.stderr)
        return 1

    print(f"Reading labels + snr from: {h5_path}")
    labels, snr_db = read_labels_and_snr(h5_path)
    n = labels.size
    fp = dataset_fingerprint(labels, snr_db)
    print(
        f"  N = {n}, classes = {np.unique(labels).size}, "
        f"snr points = {np.unique(snr_db).size}, fingerprint = {fp}"
    )

    print(
        f"Splitting {TRAIN_FRAC:.0%}/{VAL_FRAC:.0%}/{TEST_FRAC:.0%} "
        f"(stratified on (class, snr), seed={args.seed}) ..."
    )
    train_idx, val_idx, test_idx = make_stratified_splits(
        labels, snr_db, seed=args.seed
    )
    print(f"  train={train_idx.size}, val={val_idx.size}, test={test_idx.size}")

    print("Verifying ...")
    verify_splits(labels, snr_db, train_idx, val_idx, test_idx)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        # metadata for reproducibility / provenance
        master_seed=np.array([args.seed]),
        train_frac=np.array([TRAIN_FRAC]),
        val_frac=np.array([VAL_FRAC]),
        test_frac=np.array([TEST_FRAC]),
        n_total=np.array([n]),
        dataset_fingerprint=np.array([fp], dtype=object),
        generated_at=np.array([dt.datetime.now().isoformat()], dtype=object),
        stratification=np.array(["joint_label_snr"], dtype=object),
    )
    print(f"Saved splits -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
