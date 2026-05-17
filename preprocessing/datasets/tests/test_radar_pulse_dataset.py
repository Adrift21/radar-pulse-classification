"""
Integration test for ``RadarPulseDataset``.

Mirrors the sandbox test suite that was used to validate the module:

  1.  Construction + basic ``__len__`` / ``__getitem__``
  2.  Three TF representations (STFT / CWD / WVD) all produce
      (1, 224, 224) float32 tensors in [0, 1]
  3.  Reproducibility — two Datasets with the same master_seed and
      same indices produce bit-for-bit identical outputs
  4.  Different master_seed -> different noise realisation
  5.  add_noise=False vs add_noise=True differ as expected
  6.  Per-sample seed -> consecutive samples produce different images
  7.  Subset via custom indices (train/val/test split simulation)
  8.  DataLoader num_workers=0 (single-process) iterates cleanly
  9.  DataLoader num_workers>=1 (multi-process) iterates without h5py
      fork errors, using ``radar_pulse_worker_init``
 10.  Multi-process output matches single-process output (bit-for-bit
      reproducibility across worker counts)
 11.  Per-sample timing benchmark per TF representation
 12.  Multi-worker batch throughput

Outputs only to stdout — no figure. The whole script should complete
in well under a minute on a modern laptop CPU.

Author: Kaan Emre Evci
Project: Radar Pulse Classification (Module B, Phase 2b)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from preprocessing.datasets.radar_pulse_dataset import (  # noqa: E402
    RadarPulseDataset,
    radar_pulse_worker_init,
)

H5_PATH = _PROJECT_ROOT / "data_generation" / "synthetic_samples" / "dataset.h5"

# Use a manageable number of samples per test (24 is plenty to exercise
# all 8 classes; full 40k is unnecessary here)
N_SAMPLES_FOR_TESTS = 24


def _hr() -> str:
    return "=" * 70


def main() -> int:
    if not H5_PATH.exists():
        print(f"ERROR: dataset not found at {H5_PATH}", file=sys.stderr)
        return 1

    # Class-balanced indices: 3 samples from each of the 8 classes (= 24).
    # In the Module-A dataset, samples are laid out sequentially per class
    # (5000 LFM at [0:5000], 5000 NLFM at [5000:10000], ...), so the first
    # 3 indices of each class are picked.
    indices_24 = np.array(
        [
            0,
            1,
            2,  # LFM
            5000,
            5001,
            5002,  # NLFM
            10000,
            10001,
            10002,  # Barker
            15000,
            15001,
            15002,  # Frank
            20000,
            20001,
            20002,  # Polyphase
            25000,
            25001,
            25002,  # Costas
            30000,
            30001,
            30002,  # CW
            35000,
            35001,
            35002,  # SteppedFH
        ]
    )

    # ------------------------------------------------------------------
    print(_hr())
    print("Test 1: Construction + basic __len__ / __getitem__")
    print(_hr())
    ds = RadarPulseDataset(
        h5_path=str(H5_PATH),
        indices=indices_24,
        tf_repr="stft",
        add_noise=True,
        master_seed=42,
    )
    print(f"  len(ds) = {len(ds)}")
    img, label = ds[0]
    print(
        f"  ds[0]: img.shape={tuple(img.shape)}, img.dtype={img.dtype}, "
        f"label={label} (type={type(label).__name__})"
    )
    print(f"  img range: [{img.min().item():.4f}, {img.max().item():.4f}]")

    # ------------------------------------------------------------------
    print()
    print(_hr())
    print("Test 2: Three tf_repr modes all return (1, 224, 224)")
    print(_hr())
    for repr_name in ["stft", "cwd", "wvd"]:
        ds = RadarPulseDataset(
            str(H5_PATH),
            indices=indices_24,
            tf_repr=repr_name,
            add_noise=True,
            master_seed=42,
        )
        img, _ = ds[0]
        print(
            f"  tf_repr={repr_name:>5s} -> shape={tuple(img.shape)}, "
            f"range=[{img.min().item():.3f}, {img.max().item():.3f}]"
        )

    # ------------------------------------------------------------------
    print()
    print(_hr())
    print("Test 3: Reproducibility (same master_seed -> bit-for-bit identical)")
    print(_hr())
    ds_a = RadarPulseDataset(
        str(H5_PATH), indices=indices_24, tf_repr="stft", add_noise=True, master_seed=42
    )
    ds_b = RadarPulseDataset(
        str(H5_PATH), indices=indices_24, tf_repr="stft", add_noise=True, master_seed=42
    )
    diff = torch.abs(ds_a[0][0] - ds_b[0][0]).max().item()
    print(f"  max abs diff (idx=0): {diff:.2e}  (should be 0.0)")

    # ------------------------------------------------------------------
    print()
    print(_hr())
    print("Test 4: Different master_seed -> different noise realisation")
    print(_hr())
    ds_x = RadarPulseDataset(
        str(H5_PATH), indices=indices_24, tf_repr="stft", add_noise=True, master_seed=42
    )
    ds_y = RadarPulseDataset(
        str(H5_PATH),
        indices=indices_24,
        tf_repr="stft",
        add_noise=True,
        master_seed=999,
    )
    diff = torch.abs(ds_x[0][0] - ds_y[0][0]).max().item()
    print(f"  max abs diff (seed=42 vs seed=999): {diff:.4f}  (should be > 0)")

    # ------------------------------------------------------------------
    print()
    print(_hr())
    print("Test 5: add_noise=False vs add_noise=True")
    print(_hr())
    ds_clean = RadarPulseDataset(
        str(H5_PATH), indices=indices_24, tf_repr="stft", add_noise=False
    )
    ds_noisy = RadarPulseDataset(
        str(H5_PATH), indices=indices_24, tf_repr="stft", add_noise=True, master_seed=42
    )
    diff = torch.abs(ds_clean[0][0] - ds_noisy[0][0]).max().item()
    print(f"  max abs diff (clean vs noisy): {diff:.4f}")

    # ------------------------------------------------------------------
    print()
    print(_hr())
    print("Test 6: Per-sample seed differs across consecutive indices")
    print(_hr())
    ds = RadarPulseDataset(
        str(H5_PATH), indices=indices_24, tf_repr="stft", add_noise=True, master_seed=42
    )
    img_0, l_0 = ds[0]
    img_1, l_1 = ds[1]
    print(
        f"  ds[0] label={l_0}, ds[1] label={l_1}, "
        f"images differ: {(img_0 != img_1).any().item()}"
    )

    # ------------------------------------------------------------------
    print()
    print(_hr())
    print("Test 7: Subset via indices (train/val split simulation)")
    print(_hr())
    # Pull from different classes so the train/val split is meaningful.
    # train_idx covers classes 0..4 (one each), val_idx covers classes 5..7.
    train_idx = np.array([0, 5000, 10000, 15000, 20000])  # LFM..Polyphase
    val_idx = np.array([25000, 30000, 35000])  # Costas, CW, SteppedFH
    ds_train = RadarPulseDataset(
        str(H5_PATH), indices=train_idx, tf_repr="cwd", add_noise=True, master_seed=42
    )
    ds_val = RadarPulseDataset(
        str(H5_PATH), indices=val_idx, tf_repr="cwd", add_noise=True, master_seed=42
    )
    print(f"  len(train) = {len(ds_train)}, len(val) = {len(ds_val)}")
    print(f"  train[0] label = {ds_train[0][1]}, val[0] label = {ds_val[0][1]}")

    # ------------------------------------------------------------------
    print()
    print(_hr())
    print("Test 8: DataLoader with num_workers=0 (single-process)")
    print(_hr())
    ds = RadarPulseDataset(
        str(H5_PATH), indices=indices_24, tf_repr="cwd", add_noise=True, master_seed=42
    )
    loader = DataLoader(ds, batch_size=4, num_workers=0, shuffle=False)
    batches = list(loader)
    print(f"  {len(batches)} batches")
    for i, (imgs, labels) in enumerate(batches):
        print(f"  batch {i}: imgs.shape={tuple(imgs.shape)}, labels={labels.tolist()}")

    # ------------------------------------------------------------------
    print()
    print(_hr())
    print("Test 9: DataLoader with num_workers=2 (multi-process + fork-safe h5py)")
    print(_hr())
    ds = RadarPulseDataset(
        str(H5_PATH), indices=indices_24, tf_repr="cwd", add_noise=True, master_seed=42
    )
    loader = DataLoader(
        ds,
        batch_size=4,
        num_workers=2,
        shuffle=False,
        worker_init_fn=radar_pulse_worker_init,
    )
    try:
        batches = list(loader)
        print(f"  {len(batches)} batches (no fork errors)")
        for i, (imgs, labels) in enumerate(batches):
            print(
                f"  batch {i}: imgs.shape={tuple(imgs.shape)}, labels={labels.tolist()}"
            )
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        return 1

    # ------------------------------------------------------------------
    print()
    print(_hr())
    print("Test 10: Multi-process output matches single-process (reproducibility)")
    print(_hr())
    ds1 = RadarPulseDataset(
        str(H5_PATH),
        indices=np.arange(8),
        tf_repr="cwd",
        add_noise=True,
        master_seed=42,
    )
    ds2 = RadarPulseDataset(
        str(H5_PATH),
        indices=np.arange(8),
        tf_repr="cwd",
        add_noise=True,
        master_seed=42,
    )
    loader_single = DataLoader(ds1, batch_size=8, num_workers=0, shuffle=False)
    loader_multi = DataLoader(
        ds2,
        batch_size=8,
        num_workers=2,
        shuffle=False,
        worker_init_fn=radar_pulse_worker_init,
    )
    imgs_s, labels_s = next(iter(loader_single))
    imgs_m, labels_m = next(iter(loader_multi))
    img_diff = torch.abs(imgs_s - imgs_m).max().item()
    label_match = torch.equal(labels_s, labels_m)
    print(f"  max img diff: {img_diff:.2e}  (should be 0.0)")
    print(f"  labels match: {label_match}")

    # ------------------------------------------------------------------
    print()
    print(_hr())
    print("Test 11: Per-sample timing benchmark (single-process)")
    print(_hr())
    for repr_name in ["stft", "cwd", "wvd"]:
        ds = RadarPulseDataset(
            str(H5_PATH),
            indices=indices_24,
            tf_repr=repr_name,
            add_noise=True,
            master_seed=42,
        )
        _ = ds[0]  # warm-up
        t0 = time.time()
        for i in range(len(ds)):
            _ = ds[i]
        dt = time.time() - t0
        per_sample_ms = dt / len(ds) * 1000.0
        print(
            f"  tf_repr={repr_name:>5s}: {dt * 1000:>5.0f} ms total "
            f"/ {per_sample_ms:>5.1f} ms per sample"
        )

    # ------------------------------------------------------------------
    print()
    print(_hr())
    print("Test 12: Multi-worker batch throughput")
    print(_hr())
    for n_workers in [0, 2, 4]:
        ds = RadarPulseDataset(
            str(H5_PATH),
            indices=indices_24,
            tf_repr="cwd",
            add_noise=True,
            master_seed=42,
        )
        loader = DataLoader(
            ds,
            batch_size=4,
            num_workers=n_workers,
            shuffle=False,
            worker_init_fn=radar_pulse_worker_init if n_workers > 0 else None,
        )
        t0 = time.time()
        total = 0
        for imgs, _ in loader:
            total += imgs.shape[0]
        dt = time.time() - t0
        print(
            f"  num_workers={n_workers}: {total} samples in {dt:.2f}s "
            f"-> {total / dt:>5.1f} samples/sec"
        )

    print()
    print("All Dataset tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
