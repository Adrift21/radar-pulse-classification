"""
test_splits.py — Unit tests for preprocessing/splits/make_splits.py.

Tests cover:
  - SNR bin mapping (boundary, off-grid, out-of-range)
  - Joint stratification core: ratios, no overlap, full coverage
  - Per-(class, snr_bin) cell coverage
  - Reproducibility (same seed → same output, different seed → different)
  - Class balance preservation within each split
  - SHA256 hash stability and sensitivity
  - Edge cases: tiny groups, ratio rounding

Run with:
    pytest preprocessing/splits/tests/test_splits.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from preprocessing.splits.load_splits import load_split_metadata, load_splits
from preprocessing.splits.make_splits import (
    SNR_NUM_BINS,
    compute_dataset_hash,
    load_labels_and_snr,
    make_joint_stratified_split,
    snr_db_to_bin,
    verify_split,
)
from preprocessing.splits.make_splits import (
    main as make_splits_main,
)


# -----------------------------------------------------------------------------
# Fixtures: synthetic dataset mirroring the real one
# -----------------------------------------------------------------------------
@pytest.fixture
def synthetic_labels_snr() -> tuple[np.ndarray, np.ndarray]:
    """8 classes × 5000 samples, each with a random SNR from the 16-point grid.

    Matches the structure of the actual dataset.h5 produced by Module A.
    """
    rng = np.random.default_rng(42)
    n_classes = 8
    per_class = 5000
    snr_grid = np.arange(-10, 22, 2, dtype=np.float32)

    labels_parts = []
    snr_db_parts = []
    for c in range(n_classes):
        labels_parts.append(np.full(per_class, c, dtype=np.uint8))
        snr_db_parts.append(rng.choice(snr_grid, size=per_class).astype(np.float32))

    labels = np.concatenate(labels_parts)
    snr_db = np.concatenate(snr_db_parts)
    return labels, snr_db


@pytest.fixture
def synthetic_h5(tmp_path, synthetic_labels_snr):
    """Write a minimal dataset.h5 in MATLAB column-major convention,
    so we can exercise load_labels_and_snr end-to-end.

    The actual file has a /signals dataset too, but make_splits.py only
    reads labels and snr_db, so we omit it.
    """
    labels, snr_db = synthetic_labels_snr
    h5_path = tmp_path / "dataset.h5"
    with h5py.File(h5_path, "w") as f:
        # MATLAB convention: 1-D arrays stored as row vectors (1, N)
        f.create_dataset("labels", data=labels.reshape(1, -1))
        f.create_dataset("snr_db", data=snr_db.reshape(1, -1))
    return h5_path


# -----------------------------------------------------------------------------
# snr_db_to_bin
# -----------------------------------------------------------------------------
class TestSnrDbToBin:
    def test_exact_grid_values(self):
        """Each on-grid SNR value should map to its sequential bin index."""
        snr_db = np.arange(-10, 22, 2, dtype=np.float32)  # [-10, -8, ..., 20]
        bins = snr_db_to_bin(snr_db)
        np.testing.assert_array_equal(bins, np.arange(16))

    def test_boundary_values(self):
        """Min and max SNR map to bin 0 and bin 15 respectively."""
        assert snr_db_to_bin(np.array([-10.0]))[0] == 0
        assert snr_db_to_bin(np.array([20.0]))[0] == SNR_NUM_BINS - 1

    def test_below_range_raises(self):
        with pytest.raises(ValueError, match="bin out of range"):
            snr_db_to_bin(np.array([-12.0]))

    def test_above_range_raises(self):
        with pytest.raises(ValueError, match="bin out of range"):
            snr_db_to_bin(np.array([22.0]))

    def test_slightly_off_grid_rounds(self):
        """Values like -9.9 dB (FP drift) should round to nearest bin."""
        # -9.9 → round(0.05) = 0; 19.9 → round(14.95) = 15
        bins = snr_db_to_bin(np.array([-9.9, 19.9], dtype=np.float32))
        assert bins[0] == 0
        assert bins[1] == 15


# -----------------------------------------------------------------------------
# make_joint_stratified_split — core logic
# -----------------------------------------------------------------------------
class TestStratifiedSplit:
    def test_ratios_approximate(self, synthetic_labels_snr):
        labels, snr_db = synthetic_labels_snr
        snr_bins = snr_db_to_bin(snr_db)
        n = labels.shape[0]

        train, val, test = make_joint_stratified_split(
            labels, snr_bins, 0.70, 0.15, 0.15, seed=42
        )
        # Allow up to 1% drift from target due to per-cell rounding.
        assert abs(train.shape[0] / n - 0.70) < 0.01
        assert abs(val.shape[0] / n - 0.15) < 0.01
        assert abs(test.shape[0] / n - 0.15) < 0.01

    def test_no_overlap(self, synthetic_labels_snr):
        labels, snr_db = synthetic_labels_snr
        snr_bins = snr_db_to_bin(snr_db)
        train, val, test = make_joint_stratified_split(
            labels, snr_bins, 0.70, 0.15, 0.15, seed=42
        )
        assert np.intersect1d(train, val).size == 0
        assert np.intersect1d(train, test).size == 0
        assert np.intersect1d(val, test).size == 0

    def test_full_coverage(self, synthetic_labels_snr):
        labels, snr_db = synthetic_labels_snr
        snr_bins = snr_db_to_bin(snr_db)
        train, val, test = make_joint_stratified_split(
            labels, snr_bins, 0.70, 0.15, 0.15, seed=42
        )
        union = np.union1d(np.union1d(train, val), test)
        assert union.shape[0] == labels.shape[0]

    def test_class_balance_preserved(self, synthetic_labels_snr):
        """Every split should contain approximately equal samples per class."""
        labels, snr_db = synthetic_labels_snr
        snr_bins = snr_db_to_bin(snr_db)
        n_classes = int(labels.max()) + 1
        train, val, test = make_joint_stratified_split(
            labels, snr_bins, 0.70, 0.15, 0.15, seed=42
        )
        for split_idx in (train, val, test):
            counts = [int(np.sum(labels[split_idx] == c)) for c in range(n_classes)]
            # Spread should be small — at most a handful of samples
            assert max(counts) - min(counts) <= 5

    def test_per_cell_coverage(self, synthetic_labels_snr):
        """Every (class, snr_bin) cell must appear in every split."""
        labels, snr_db = synthetic_labels_snr
        snr_bins = snr_db_to_bin(snr_db)
        n_classes = int(labels.max()) + 1
        train, val, test = make_joint_stratified_split(
            labels, snr_bins, 0.70, 0.15, 0.15, seed=42
        )
        for split_name, idx in (("train", train), ("val", val), ("test", test)):
            sub_labels = labels[idx]
            sub_bins = snr_bins[idx]
            for c in range(n_classes):
                for b in range(SNR_NUM_BINS):
                    cell_count = np.sum((sub_labels == c) & (sub_bins == b))
                    assert cell_count >= 1, (
                        f"{split_name}: empty cell (class={c}, snr_bin={b})"
                    )

    def test_indices_sorted_ascending(self, synthetic_labels_snr):
        """Sorted indices help HDF5 chunked reads."""
        labels, snr_db = synthetic_labels_snr
        snr_bins = snr_db_to_bin(snr_db)
        train, val, test = make_joint_stratified_split(
            labels, snr_bins, 0.70, 0.15, 0.15, seed=42
        )
        for arr in (train, val, test):
            assert np.all(np.diff(arr) > 0)  # strictly increasing (no dupes)

    def test_indices_are_uint32(self, synthetic_labels_snr):
        labels, snr_db = synthetic_labels_snr
        snr_bins = snr_db_to_bin(snr_db)
        train, val, test = make_joint_stratified_split(
            labels, snr_bins, 0.70, 0.15, 0.15, seed=42
        )
        for arr in (train, val, test):
            assert arr.dtype == np.uint32

    def test_invalid_ratios_raise(self, synthetic_labels_snr):
        labels, snr_db = synthetic_labels_snr
        snr_bins = snr_db_to_bin(snr_db)
        with pytest.raises(ValueError, match="must sum to 1.0"):
            make_joint_stratified_split(
                labels,
                snr_bins,
                0.7,
                0.2,
                0.2,
                seed=42,  # sums to 1.1
            )

    def test_tiny_group_raises(self):
        """If any (class, snr_bin) cell has < 3 samples, splitting fails."""
        # 2 samples only — cannot place one in each of train/val/test
        labels = np.array([0, 0], dtype=np.uint8)
        snr_bins = np.array([0, 0], dtype=np.int32)
        with pytest.raises(RuntimeError, match="only 2 samples"):
            make_joint_stratified_split(labels, snr_bins, 0.7, 0.15, 0.15, seed=42)


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
class TestReproducibility:
    def test_same_seed_same_output(self, synthetic_labels_snr):
        labels, snr_db = synthetic_labels_snr
        snr_bins = snr_db_to_bin(snr_db)
        a = make_joint_stratified_split(labels, snr_bins, 0.70, 0.15, 0.15, seed=42)
        b = make_joint_stratified_split(labels, snr_bins, 0.70, 0.15, 0.15, seed=42)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_different_seed_different_output(self, synthetic_labels_snr):
        labels, snr_db = synthetic_labels_snr
        snr_bins = snr_db_to_bin(snr_db)
        a_train, _, _ = make_joint_stratified_split(
            labels, snr_bins, 0.70, 0.15, 0.15, seed=42
        )
        b_train, _, _ = make_joint_stratified_split(
            labels, snr_bins, 0.70, 0.15, 0.15, seed=123
        )
        assert not np.array_equal(a_train, b_train)


# -----------------------------------------------------------------------------
# compute_dataset_hash
# -----------------------------------------------------------------------------
class TestDatasetHash:
    def test_hash_is_deterministic(self, synthetic_labels_snr):
        labels, snr_db = synthetic_labels_snr
        h1 = compute_dataset_hash(labels, snr_db)
        h2 = compute_dataset_hash(labels, snr_db)
        assert h1 == h2

    def test_hash_sensitive_to_labels(self, synthetic_labels_snr):
        labels, snr_db = synthetic_labels_snr
        labels2 = labels.copy()
        labels2[0] = (labels2[0] + 1) % 8
        assert compute_dataset_hash(labels, snr_db) != compute_dataset_hash(
            labels2, snr_db
        )

    def test_hash_sensitive_to_snr(self, synthetic_labels_snr):
        labels, snr_db = synthetic_labels_snr
        snr_db2 = snr_db.copy()
        snr_db2[0] += 2.0  # bump one SNR
        assert compute_dataset_hash(labels, snr_db) != compute_dataset_hash(
            labels, snr_db2
        )

    def test_hash_length(self, synthetic_labels_snr):
        labels, snr_db = synthetic_labels_snr
        h = compute_dataset_hash(labels, snr_db)
        assert len(h) == 64  # SHA256 hex


# -----------------------------------------------------------------------------
# verify_split
# -----------------------------------------------------------------------------
class TestVerifySplit:
    def test_passes_for_valid_split(self, synthetic_labels_snr):
        labels, snr_db = synthetic_labels_snr
        snr_bins = snr_db_to_bin(snr_db)
        train, val, test = make_joint_stratified_split(
            labels, snr_bins, 0.70, 0.15, 0.15, seed=42
        )
        stats = verify_split(train, val, test, labels, snr_bins)
        assert stats["split_sizes"]["train"] == train.shape[0]
        assert stats["split_sizes"]["val"] == val.shape[0]
        assert stats["split_sizes"]["test"] == test.shape[0]
        assert all(v >= 1 for v in stats["min_samples_per_cell"].values())

    def test_detects_overlap(self, synthetic_labels_snr):
        """Manually inject an overlap → verify_split should AssertionError."""
        labels, snr_db = synthetic_labels_snr
        snr_bins = snr_db_to_bin(snr_db)
        train, val, test = make_joint_stratified_split(
            labels, snr_bins, 0.70, 0.15, 0.15, seed=42
        )
        # Force train[0] into val too
        val_bad = np.sort(np.append(val, train[0])).astype(np.uint32)
        with pytest.raises(AssertionError, match="train/val overlap"):
            verify_split(train, val_bad, test, labels, snr_bins)


# -----------------------------------------------------------------------------
# End-to-end through main() with synthetic HDF5
# -----------------------------------------------------------------------------
class TestEndToEnd:
    def test_main_writes_expected_files(self, tmp_path, synthetic_h5):
        out_dir = tmp_path / "splits"
        make_splits_main(
            h5_path=synthetic_h5,
            output_dir=out_dir,
            seed=42,
        )
        assert (out_dir / "train_idx.npy").exists()
        assert (out_dir / "val_idx.npy").exists()
        assert (out_dir / "test_idx.npy").exists()
        assert (out_dir / "split_metadata.json").exists()

    def test_main_metadata_is_complete(self, tmp_path, synthetic_h5):
        out_dir = tmp_path / "splits"
        make_splits_main(h5_path=synthetic_h5, output_dir=out_dir, seed=42)
        with open(out_dir / "split_metadata.json") as f:
            meta = json.load(f)
        required = {
            "master_seed",
            "train_frac",
            "val_frac",
            "test_frac",
            "dataset_hash_sha256",
            "generated_at_utc",
            "statistics",
        }
        assert required.issubset(meta.keys())
        assert meta["master_seed"] == 42
        assert len(meta["dataset_hash_sha256"]) == 64

    def test_load_splits_roundtrip(self, tmp_path, synthetic_h5):
        out_dir = tmp_path / "splits"
        make_splits_main(h5_path=synthetic_h5, output_dir=out_dir, seed=42)

        train, val, test = load_splits(
            splits_dir=out_dir, h5_path=synthetic_h5, verify_hash=True
        )
        # Sanity: sizes match metadata
        meta = load_split_metadata(out_dir)
        assert train.shape[0] == meta["statistics"]["split_sizes"]["train"]
        assert val.shape[0] == meta["statistics"]["split_sizes"]["val"]
        assert test.shape[0] == meta["statistics"]["split_sizes"]["test"]

    def test_load_splits_detects_stale_hash(self, tmp_path, synthetic_h5):
        """If dataset.h5 changes after splits are generated, hash check
        must fail loudly rather than silently proceed."""
        out_dir = tmp_path / "splits"
        make_splits_main(h5_path=synthetic_h5, output_dir=out_dir, seed=42)

        # Corrupt the source dataset by flipping one label
        with h5py.File(synthetic_h5, "r+") as f:
            data = f["labels"][:]
            data[0, 0] = (int(data[0, 0]) + 1) % 8
            f["labels"][...] = data

        with pytest.raises(RuntimeError, match="Dataset hash mismatch"):
            load_splits(splits_dir=out_dir, h5_path=synthetic_h5, verify_hash=True)

    def test_load_splits_can_skip_hash(self, tmp_path, synthetic_h5):
        """verify_hash=False should load without touching the HDF5."""
        out_dir = tmp_path / "splits"
        make_splits_main(h5_path=synthetic_h5, output_dir=out_dir, seed=42)
        # Even if h5_path is None, this should work
        train, val, test = load_splits(
            splits_dir=out_dir, h5_path=None, verify_hash=False
        )
        assert train.shape[0] > 0


# -----------------------------------------------------------------------------
# HDF5 reading: MATLAB column-major convention
# -----------------------------------------------------------------------------
class TestHdf5Reading:
    def test_load_labels_snr_shapes(self, synthetic_h5, synthetic_labels_snr):
        """load_labels_and_snr must ravel (1, N) row-vectors to (N,)."""
        labels, snr_db = load_labels_and_snr(synthetic_h5)
        expected_labels, expected_snr = synthetic_labels_snr
        assert labels.ndim == 1
        assert snr_db.ndim == 1
        assert labels.shape == expected_labels.shape
        assert snr_db.shape == expected_snr.shape
        np.testing.assert_array_equal(labels, expected_labels)
        np.testing.assert_array_equal(snr_db, expected_snr)
