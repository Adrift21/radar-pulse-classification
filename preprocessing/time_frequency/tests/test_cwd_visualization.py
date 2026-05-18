"""
CWD visualization test for all 8 radar pulse classes.

Mirrors the Phase-1 `test_stft_visualization.py` design:
  1. Load the project's HDF5 dataset.
  2. For each of the 8 classes, pick the *longest-pulse* clean sample
     (i.e. argmax over pulse_widths_us within that class) — this maximizes
     the visible TF signature and gives the most readable reference figure.
  3. Compute the Choi-Williams Distribution at sigma=1.0 with the
     project-standard downsampling (time_step=32, n_freq=256).
  4. Save a 2x4 grid figure suitable for the academic manuscript
     ("Methods" section, Figure candidate next to the STFT reference).
  5. Print a per-class statistics table to verify shape / value range /
     NaN-Inf-free / pulse-width distribution.

Outputs
-------
  preprocessing/time_frequency/tests/outputs/cwd_clean_8classes.png

Not part of the production code path. Used for Phase-2 sign-off and as a
diagnostic when revisiting CWD parameters.

Author: Kaan Emre Evci
Project: Radar Pulse Classification (Module B, Phase 2)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Make sure the project root is on sys.path so that `preprocessing.*` is
# importable when running the script directly. This mirrors what
# test_stft_visualization.py does in Phase 1.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[
    3
]  # tests/ -> time_frequency/ -> preprocessing/ -> project root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from preprocessing.time_frequency.cwd import compute_cwd  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_PATH = _PROJECT_ROOT / "data_generation" / "synthetic_samples" / "dataset.h5"
OUTPUT_DIR = _THIS_FILE.parent / "outputs"
OUTPUT_FIG = OUTPUT_DIR / "cwd_clean_8classes.png"


# ---------------------------------------------------------------------------
# CWD parameters (decisions.md, 2026-05-?? CWD entry)
# ---------------------------------------------------------------------------
CWD_SIGMA = 1.0
CWD_TIME_STEP = 32
CWD_N_FREQ = 256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_dataset(
    h5_path: Path,
) -> Tuple[
    h5py.File,
    np.ndarray,  # labels (N,) uint8
    np.ndarray,  # snr_db (N,) float32
    np.ndarray,  # pulse_widths_us (N,) float32
    List[str],  # class_names (8,)
    float,  # fs
    int,  # signal_length
]:
    """Open the HDF5 dataset and load 1-D metadata arrays.

    Note: signals are NOT loaded eagerly; we'll fetch single samples
    on demand via `f['signals'][:, :, idx].T` to stay memory-friendly.
    The returned file handle must be closed by the caller.
    """
    f = h5py.File(str(h5_path), "r")

    labels = np.asarray(f["labels"][:]).ravel().astype(np.uint8)
    snr_db = np.asarray(f["snr_db"][:]).ravel().astype(np.float32)
    pulse_widths_us = np.asarray(f["pulse_widths_us"][:]).ravel().astype(np.float32)

    class_names_raw = np.asarray(f["class_names"][:])
    class_names = [
        s.decode() if isinstance(s, (bytes, np.bytes_)) else str(s)
        for s in class_names_raw
    ]

    fs = float(np.asarray(f.attrs["sample_rate_hz"]).ravel()[0])
    signal_length = int(np.asarray(f.attrs["signal_length"]).ravel()[0])

    return f, labels, snr_db, pulse_widths_us, class_names, fs, signal_length


def _read_signal(f: h5py.File, idx: int) -> np.ndarray:
    """Read a single complex baseband signal from the dataset.

    Handles the MATLAB column-major layout: on-disk shape is
    (2, 2048, 40000) where last axis is the sample index.
    """
    ri = np.asarray(f["signals"][:, :, idx]).T  # (2048, 2) float32
    return (ri[:, 0] + 1j * ri[:, 1]).astype(np.complex64)


def _to_db(
    magnitude: np.ndarray, eps: float = 1e-12, db_floor: float = -60.0
) -> np.ndarray:
    """Convert magnitude to dB, with a numerical floor for display."""
    peak = float(magnitude.max())
    if peak <= 0:
        return np.full_like(magnitude, db_floor, dtype=np.float64)
    db = 20.0 * np.log10(np.maximum(magnitude, eps) / peak)
    return np.clip(db, db_floor, 0.0)


# ---------------------------------------------------------------------------
# Per-class selection: longest pulse within each class
# ---------------------------------------------------------------------------
def _pick_longest_per_class(
    labels: np.ndarray,
    pulse_widths_us: np.ndarray,
    num_classes: int,
) -> List[int]:
    """For each class index, return the global sample index whose
    pulse_widths_us is maximal within that class. Matches Phase-1
    longest-pulse selection methodology.
    """
    picks: List[int] = []
    for cls in range(num_classes):
        cls_mask = labels == cls
        cls_indices = np.where(cls_mask)[0]
        if cls_indices.size == 0:
            raise RuntimeError(f"No samples found for class index {cls}.")
        cls_pws = pulse_widths_us[cls_indices]
        best_local = int(np.argmax(cls_pws))
        picks.append(int(cls_indices[best_local]))
    return picks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    if not DATASET_PATH.exists():
        print(f"ERROR: dataset not found at {DATASET_PATH}", file=sys.stderr)
        return 1

    print(f"Loading dataset: {DATASET_PATH}")
    f, labels, snr_db, pulse_widths_us, class_names, fs, signal_length = _load_dataset(
        DATASET_PATH
    )
    num_classes = len(class_names)
    print(f"  num_classes   : {num_classes}")
    print(f"  total samples : {labels.size}")
    print(f"  sample_rate   : {fs:g} Hz")
    print(f"  signal_length : {signal_length}")
    print()

    # ----- 1. Pick representative sample per class --------------------
    picks = _pick_longest_per_class(labels, pulse_widths_us, num_classes)

    print("Per-class longest-pulse picks:")
    print("-" * 78)
    print(f"  {'class':<12} {'idx':>7} {'PW (us)':>10} {'SNR (dB)':>10}")
    print("-" * 78)
    for cls, idx in enumerate(picks):
        print(
            f"  {class_names[cls]:<12} {idx:>7d} {pulse_widths_us[idx]:>10.2f} {snr_db[idx]:>10.1f}"
        )
    print("-" * 78)
    print()

    # ----- 2. Compute CWD per pick and gather stats -------------------
    print(
        f"Computing CWD (sigma={CWD_SIGMA}, time_step={CWD_TIME_STEP}, n_freq={CWD_N_FREQ})..."
    )
    cwds: List[np.ndarray] = []
    t_axes: List[np.ndarray] = []
    f_axes: List[np.ndarray] = []
    stats_rows: List[Tuple[str, Tuple[int, int], float, float, bool, bool]] = []

    for cls, idx in enumerate(picks):
        sig = _read_signal(f, idx)
        cwd_mag, t_axis, f_axis = compute_cwd(
            sig,
            fs=fs,
            sigma=CWD_SIGMA,
            time_step=CWD_TIME_STEP,
            n_freq=CWD_N_FREQ,
        )
        cwds.append(cwd_mag)
        t_axes.append(t_axis)
        f_axes.append(f_axis)
        stats_rows.append(
            (
                class_names[cls],
                tuple(cwd_mag.shape),
                float(cwd_mag.min()),
                float(cwd_mag.max()),
                bool(np.any(np.isnan(cwd_mag))),
                bool(np.any(np.isinf(cwd_mag))),
            )
        )

    f.close()

    # ----- 3. Statistics table ----------------------------------------
    print()
    print("CWD statistics per class:")
    print("-" * 78)
    print(
        f"  {'class':<12} {'shape':>15} {'min':>12} {'max':>12} {'NaN?':>6} {'Inf?':>6}"
    )
    print("-" * 78)
    for row in stats_rows:
        cls_name, shape, mn, mx, has_nan, has_inf = row
        print(
            f"  {cls_name:<12} {str(shape):>15} {mn:>12.3e} {mx:>12.3e} {str(has_nan):>6} {str(has_inf):>6}"
        )
    print("-" * 78)
    print()

    any_nan = any(r[4] for r in stats_rows)
    any_inf = any(r[5] for r in stats_rows)
    if any_nan or any_inf:
        print("WARN: NaN or Inf detected in CWD outputs — investigate.")
    else:
        print("OK: all 8 CWDs are finite.")
    print()

    # ----- 4. Plot 2x4 grid ------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building figure ({OUTPUT_FIG.name})...")

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    axes_flat = axes.ravel()

    for cls in range(num_classes):
        ax = axes_flat[cls]
        cwd_db = _to_db(cwds[cls])
        # x-axis : time in us;  y-axis : frequency in MHz
        t_us = t_axes[cls] * 1e6
        f_mhz = f_axes[cls] / 1e6
        im = ax.imshow(
            cwd_db,
            origin="lower",
            aspect="auto",
            extent=(float(t_us[0]), float(t_us[-1]), float(f_mhz[0]), float(f_mhz[-1])),
            cmap="viridis",
            vmin=-60.0,
            vmax=0.0,
            interpolation="nearest",
        )
        pw = pulse_widths_us[picks[cls]]
        ax.set_title(f"[{cls}] {class_names[cls]} (PW={pw:.1f} us)", fontsize=10)
        ax.set_xlabel("Time [us]", fontsize=9)
        ax.set_ylabel("Frequency [MHz]", fontsize=9)
        ax.tick_params(labelsize=8)

    cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
    cbar.set_label("dB (relative to peak)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Choi-Williams Distribution (sigma={CWD_SIGMA}) — clean baseline, 8 classes",
        fontsize=12,
    )

    fig.savefig(OUTPUT_FIG, dpi=150)
    plt.close(fig)
    print(f"Saved figure -> {OUTPUT_FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
