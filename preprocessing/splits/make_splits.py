"""
make_splits.py — Generate train/val/test split indices for the radar dataset.

Strategy: Joint (class, snr_bin) stratification — each of the 8 classes ×
16 SNR levels = 128 groups is split independently at 70/15/15. This
guarantees every (class, snr) cell has proportional representation in
every split, which is critical for Module D's SNR-stratified evaluation.

The split is computed ONCE and reused across all 9 experiments in
Module C (3 architectures × 3 TF representations) to ensure
apples-to-apples comparison.

Outputs (under output_dir, default `data_generation/synthetic_samples/splits/`):
    train_idx.npy        : (N_train,) uint32, indices into dataset.h5
    val_idx.npy          : (N_val,)   uint32
    test_idx.npy         : (N_test,)  uint32
    split_metadata.json  : seed, ratios, dataset hash, generation date,
                           per-(class, snr) counts for verification

Usage:
    python -m preprocessing.splits.make_splits
    python -m preprocessing.splits.make_splits --h5 path/to/dataset.h5 --seed 42

The dataset hash is computed from (labels, snr_db) — any regeneration of
the dataset will produce a different hash, signalling that splits must
be regenerated too.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# Defaults (decisions.md 2026-05-04 entries)
# -----------------------------------------------------------------------------
DEFAULT_MASTER_SEED = 42
DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_VAL_FRAC = 0.15
DEFAULT_TEST_FRAC = 0.15

# SNR grid: -10..+20 dB, 2 dB step → 16 bins (decisions.md 2026-05-04)
SNR_MIN_DB = -10.0
SNR_STEP_DB = 2.0
SNR_NUM_BINS = 16


# -----------------------------------------------------------------------------
# Dataset I/O — follows MATLAB column-major HDF5 convention
# (project_context.md "HDF5 okuma her zaman transpose ile")
# -----------------------------------------------------------------------------
def load_labels_and_snr(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read labels and snr_db arrays from dataset.h5, applying ravel().

    Returns
    -------
    labels : (N,) uint8     — class index 0..7
    snr_db : (N,) float32   — intended SNR per sample
    """
    with h5py.File(h5_path, "r") as f:
        labels = np.asarray(f["labels"][:]).ravel().astype(np.uint8)
        snr_db = np.asarray(f["snr_db"][:]).ravel().astype(np.float32)
    return labels, snr_db


def compute_dataset_hash(labels: np.ndarray, snr_db: np.ndarray) -> str:
    """SHA256 hash of (labels, snr_db) — fingerprint of the dataset
    contents relevant for splitting. Regeneration of dataset.h5 will
    produce a different hash, alerting downstream code that splits are
    stale.
    """
    h = hashlib.sha256()
    h.update(labels.tobytes())
    h.update(snr_db.tobytes())
    return h.hexdigest()


def snr_db_to_bin(snr_db: np.ndarray) -> np.ndarray:
    """Map SNR values in dB to integer bin indices 0..15.

    bin = round((snr_db - SNR_MIN_DB) / SNR_STEP_DB)

    Values produced by Module A are already on a 2 dB grid, so rounding
    is purely a defensive cast. We assert all bins fall in [0, 15].
    """
    bins = np.round((snr_db - SNR_MIN_DB) / SNR_STEP_DB).astype(np.int32)
    if bins.min() < 0 or bins.max() >= SNR_NUM_BINS:
        raise ValueError(
            f"SNR bin out of range [0, {SNR_NUM_BINS - 1}]: "
            f"min={bins.min()}, max={bins.max()}"
        )
    return bins


# -----------------------------------------------------------------------------
# Core split logic
# -----------------------------------------------------------------------------
def make_joint_stratified_split(
    labels: np.ndarray,
    snr_bins: np.ndarray,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Produce train/val/test indices using joint (class, snr_bin) strata.

    For each of the 8 × 16 = 128 (class, snr_bin) groups, split its
    members independently at the specified ratios. Two-step procedure:
      1) Group → train vs (val+test)  with ratio train : (1 - train)
      2) (val+test) → val vs test     with ratio val/(val+test) : test/(val+test)

    Both sklearn calls use seeded RNGs derived from the master seed, so
    the result is fully reproducible.

    Returns
    -------
    train_idx, val_idx, test_idx : 1-D uint32 arrays, sorted ascending
                                   (sorting helps HDF5 chunked reads)
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError(
            f"train+val+test must sum to 1.0, got {train_frac + val_frac + test_frac}"
        )

    n_total = labels.shape[0]
    all_idx = np.arange(n_total, dtype=np.uint32)

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    n_classes = int(labels.max()) + 1
    rng = np.random.default_rng(seed)

    for c in range(n_classes):
        for b in range(SNR_NUM_BINS):
            mask = (labels == c) & (snr_bins == b)
            group_idx = all_idx[mask]
            n_group = group_idx.shape[0]

            if n_group < 3:
                # Need at least 3 to put one sample in each split.
                # Should not occur with our balanced dataset
                # (~5000/16 ≈ 312 per (class, snr)).
                raise RuntimeError(
                    f"(class={c}, snr_bin={b}) has only {n_group} samples; "
                    "cannot split 70/15/15."
                )

            # Per-group seeds: derived but distinct, so each (c, b) cell
            # has its own RNG stream. This means changing samples_per_class
            # later wouldn't reshuffle ALL groups — only the larger ones.
            seed_a = int(rng.integers(0, 2**31 - 1))
            seed_b = int(rng.integers(0, 2**31 - 1))

            # Step 1: train vs (val + test)
            train_g, valtest_g = train_test_split(
                group_idx,
                train_size=train_frac,
                random_state=seed_a,
                shuffle=True,
            )

            # Step 2: split (val + test) into val and test
            # val proportion within (val+test) = val / (val + test)
            val_within = val_frac / (val_frac + test_frac)
            val_g, test_g = train_test_split(
                valtest_g,
                train_size=val_within,
                random_state=seed_b,
                shuffle=True,
            )

            train_parts.append(train_g)
            val_parts.append(val_g)
            test_parts.append(test_g)

    train_idx = np.sort(np.concatenate(train_parts)).astype(np.uint32)
    val_idx = np.sort(np.concatenate(val_parts)).astype(np.uint32)
    test_idx = np.sort(np.concatenate(test_parts)).astype(np.uint32)

    return train_idx, val_idx, test_idx


# -----------------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------------
def verify_split(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    labels: np.ndarray,
    snr_bins: np.ndarray,
) -> dict:
    """Sanity checks + return summary statistics for metadata.json."""
    n_total = labels.shape[0]

    # 1) No overlap
    assert len(np.intersect1d(train_idx, val_idx)) == 0, "train/val overlap"
    assert len(np.intersect1d(train_idx, test_idx)) == 0, "train/test overlap"
    assert len(np.intersect1d(val_idx, test_idx)) == 0, "val/test overlap"

    # 2) Full coverage
    union = np.union1d(np.union1d(train_idx, val_idx), test_idx)
    assert union.shape[0] == n_total, (
        f"Coverage gap: {n_total - union.shape[0]} samples missing"
    )

    # 3) Per-split class counts (should match train_frac/val_frac/test_frac
    #    of each class's total within rounding)
    n_classes = int(labels.max()) + 1
    per_split_class_counts: dict[str, list[int]] = {}
    for split_name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        counts = [int(np.sum(labels[idx] == c)) for c in range(n_classes)]
        per_split_class_counts[split_name] = counts

    # 4) Per-split (class, snr_bin) minimum — must be >= 1 for every cell
    min_per_cell = {}
    for split_name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        sub_labels = labels[idx]
        sub_bins = snr_bins[idx]
        cell_counts = np.zeros((n_classes, SNR_NUM_BINS), dtype=np.int32)
        for c in range(n_classes):
            for b in range(SNR_NUM_BINS):
                cell_counts[c, b] = int(np.sum((sub_labels == c) & (sub_bins == b)))
        min_per_cell[split_name] = int(cell_counts.min())
        assert cell_counts.min() >= 1, (
            f"{split_name} split has empty (class, snr_bin) cell"
        )

    return {
        "total_samples": int(n_total),
        "split_sizes": {
            "train": int(train_idx.shape[0]),
            "val": int(val_idx.shape[0]),
            "test": int(test_idx.shape[0]),
        },
        "per_split_class_counts": per_split_class_counts,
        "min_samples_per_cell": min_per_cell,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(
    h5_path: Path,
    output_dir: Path,
    seed: int = DEFAULT_MASTER_SEED,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
) -> None:
    print(f"[make_splits] Reading {h5_path}")
    labels, snr_db = load_labels_and_snr(h5_path)
    print(f"[make_splits] Loaded {labels.shape[0]} samples")

    snr_bins = snr_db_to_bin(snr_db)
    dataset_hash = compute_dataset_hash(labels, snr_db)
    print(f"[make_splits] Dataset hash: {dataset_hash[:16]}...")

    print(
        f"[make_splits] Splitting (train={train_frac}, val={val_frac}, "
        f"test={test_frac}, seed={seed})"
    )
    train_idx, val_idx, test_idx = make_joint_stratified_split(
        labels, snr_bins, train_frac, val_frac, test_frac, seed
    )

    print("[make_splits] Verifying split integrity")
    stats = verify_split(train_idx, val_idx, test_idx, labels, snr_bins)
    print(
        f"[make_splits] OK — train: {stats['split_sizes']['train']}, "
        f"val: {stats['split_sizes']['val']}, "
        f"test: {stats['split_sizes']['test']}"
    )
    print(
        f"[make_splits] Min samples per (class, snr) cell: "
        f"train={stats['min_samples_per_cell']['train']}, "
        f"val={stats['min_samples_per_cell']['val']}, "
        f"test={stats['min_samples_per_cell']['test']}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "train_idx.npy", train_idx)
    np.save(output_dir / "val_idx.npy", val_idx)
    np.save(output_dir / "test_idx.npy", test_idx)

    metadata = {
        "master_seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "dataset_hash_sha256": dataset_hash,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "h5_source_path": str(h5_path),
        "stratification": "joint (class, snr_bin)",
        "snr_bin_definition": {
            "snr_min_db": SNR_MIN_DB,
            "snr_step_db": SNR_STEP_DB,
            "num_bins": SNR_NUM_BINS,
        },
        "statistics": stats,
    }
    with open(output_dir / "split_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[make_splits] Wrote splits to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5",
        type=Path,
        default=Path("data_generation/synthetic_samples/dataset.h5"),
        help="Path to dataset.h5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_generation/synthetic_samples/splits"),
        help="Output directory for split index files",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC)
    parser.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    parser.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        h5_path=args.h5,
        output_dir=args.output_dir,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )
