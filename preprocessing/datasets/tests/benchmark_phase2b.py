"""
Phase 2b realistic throughput benchmark.

Measures end-to-end DataLoader throughput on a class-balanced subset of
the project's HDF5 dataset, across:

  - 3 time-frequency representations : STFT, CWD, WVD
  - 2 num_workers settings           : 0 (single-process), 4 (multi-process)
  - 2 batch sizes                    : 32, 64

Total = 12 configurations, run sequentially on the same 200-sample
subset (25 per class x 8 classes).

The results inform Module C's DataLoader configuration choice. Output is
a printed table sortable by samples/sec, plus an inline recommendation.

Outputs only to stdout. Total wall time ~5-8 min on a RTX 5050 laptop.

Author: Kaan Emre Evci
Project: Radar Pulse Classification (Module B, Phase 2b)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

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

# Module A dataset layout: 5000 samples per class, classes laid out
# sequentially starting at idx = class * 5000.
SAMPLES_PER_CLASS = 25
NUM_CLASSES = 8
TOTAL_SAMPLES = SAMPLES_PER_CLASS * NUM_CLASSES  # 200

TF_REPRS = ("stft", "cwd", "wvd")
NUM_WORKERS_GRID = (0, 4)
BATCH_SIZE_GRID = (32, 64)


@dataclass
class BenchResult:
    tf_repr: str
    num_workers: int
    batch_size: int
    total_samples: int
    wall_time_s: float
    samples_per_sec: float
    avg_batch_ms: float


def _build_balanced_indices() -> np.ndarray:
    """200 indices: 25 from each of 8 classes."""
    indices = []
    for cls in range(NUM_CLASSES):
        start = cls * 5000   # Module A layout
        indices.extend(range(start, start + SAMPLES_PER_CLASS))
    return np.array(indices, dtype=np.int64)


def _run_single_config(
    indices: np.ndarray,
    tf_repr: str,
    num_workers: int,
    batch_size: int,
) -> BenchResult:
    """Run one config end-to-end, return timing."""
    ds = RadarPulseDataset(
        h5_path=str(H5_PATH),
        indices=indices,
        tf_repr=tf_repr,
        add_noise=True,
        master_seed=42,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        worker_init_fn=radar_pulse_worker_init if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )

    # Warm-up : one full pass so worker processes are spawned and the
    # h5py file handles are open. We discard this timing.
    for batch in loader:
        _ = batch
        break  # one batch is enough as warm-up signal

    # Actual measurement : full pass through the 200-sample loader
    t0 = time.time()
    total = 0
    n_batches = 0
    for imgs, _labels in loader:
        total += imgs.shape[0]
        n_batches += 1
    wall = time.time() - t0

    return BenchResult(
        tf_repr=tf_repr,
        num_workers=num_workers,
        batch_size=batch_size,
        total_samples=total,
        wall_time_s=wall,
        samples_per_sec=total / wall if wall > 0 else float("inf"),
        avg_batch_ms=wall / n_batches * 1000.0 if n_batches > 0 else 0.0,
    )


def _print_results_table(results: List[BenchResult]) -> None:
    print()
    print("=" * 88)
    print(f"  {'TF':>5s}  {'workers':>8s}  {'batch':>6s}  "
          f"{'samples':>8s}  {'wall(s)':>9s}  "
          f"{'samp/s':>10s}  {'batch(ms)':>10s}")
    print("=" * 88)
    for r in results:
        print(f"  {r.tf_repr:>5s}  {r.num_workers:>8d}  {r.batch_size:>6d}  "
              f"{r.total_samples:>8d}  {r.wall_time_s:>9.2f}  "
              f"{r.samples_per_sec:>10.1f}  {r.avg_batch_ms:>10.1f}")
    print("=" * 88)


def _print_recommendations(results: List[BenchResult]) -> None:
    """Per-TF-representation, which config maximises throughput?"""
    print()
    print("Recommended config per TF representation (max samples/sec):")
    print("-" * 70)
    for repr_name in TF_REPRS:
        sub = [r for r in results if r.tf_repr == repr_name]
        best = max(sub, key=lambda r: r.samples_per_sec)
        print(f"  {repr_name:>5s}: num_workers={best.num_workers}, "
              f"batch_size={best.batch_size}  ->  "
              f"{best.samples_per_sec:.1f} samples/sec  "
              f"({best.avg_batch_ms:.1f} ms/batch)")
    print("-" * 70)

    # Cross-cutting summary
    fastest = max(results, key=lambda r: r.samples_per_sec)
    slowest = min(results, key=lambda r: r.samples_per_sec)
    print()
    print(f"  Fastest overall: {fastest.tf_repr.upper()} "
          f"(workers={fastest.num_workers}, batch={fastest.batch_size}) "
          f"-> {fastest.samples_per_sec:.1f} samples/sec")
    print(f"  Slowest overall: {slowest.tf_repr.upper()} "
          f"(workers={slowest.num_workers}, batch={slowest.batch_size}) "
          f"-> {slowest.samples_per_sec:.1f} samples/sec")

    # Training-time projections (50 epochs, 28000 train samples)
    print()
    print("Module C epoch-time projection (50 epochs, 28000 train samples):")
    print("-" * 70)
    for repr_name in TF_REPRS:
        sub = [r for r in results if r.tf_repr == repr_name]
        best = max(sub, key=lambda r: r.samples_per_sec)
        epoch_s = 28000 / best.samples_per_sec
        total_h = epoch_s * 50 / 3600
        print(f"  {repr_name:>5s} (best config): "
              f"{epoch_s/60:.1f} min/epoch  ->  {total_h:.1f} hours for 50 epochs")
    print("-" * 70)


def main() -> int:
    if not H5_PATH.exists():
        print(f"ERROR: dataset not found at {H5_PATH}", file=sys.stderr)
        return 1

    print("Phase 2b realistic throughput benchmark")
    print(f"  dataset: {H5_PATH.name}")
    print(f"  samples: {TOTAL_SAMPLES} ({SAMPLES_PER_CLASS} per class x {NUM_CLASSES} classes)")
    print(f"  configs: {len(TF_REPRS)} TF x {len(NUM_WORKERS_GRID)} workers x {len(BATCH_SIZE_GRID)} batch")
    print(f"         = {len(TF_REPRS) * len(NUM_WORKERS_GRID) * len(BATCH_SIZE_GRID)} total")
    print()

    indices = _build_balanced_indices()

    results: List[BenchResult] = []
    config_n = 0
    total_configs = len(TF_REPRS) * len(NUM_WORKERS_GRID) * len(BATCH_SIZE_GRID)

    for tf_repr in TF_REPRS:
        for num_workers in NUM_WORKERS_GRID:
            for batch_size in BATCH_SIZE_GRID:
                config_n += 1
                tag = f"[{config_n:>2d}/{total_configs}]"
                print(f"  {tag} tf_repr={tf_repr:>5s}, "
                      f"workers={num_workers}, batch={batch_size} ... ",
                      end="", flush=True)
                t0 = time.time()
                res = _run_single_config(indices, tf_repr, num_workers, batch_size)
                elapsed = time.time() - t0
                print(f"done in {elapsed:.1f}s "
                      f"({res.samples_per_sec:.1f} samp/s)")
                results.append(res)

    _print_results_table(results)
    _print_recommendations(results)

    print()
    print("Benchmark complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
