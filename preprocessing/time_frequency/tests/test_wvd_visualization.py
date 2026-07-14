"""
WVD visualization test for all 8 radar pulse classes.

Mirrors the Phase-1 ``test_stft_visualization.py`` and Phase-2
``test_cwd_visualization.py`` design:
  1. Load the project's HDF5 dataset.
  2. For each of the 8 classes, pick the *longest-pulse* clean sample.
  3. Compute the Wigner-Ville Distribution with the project-standard
     downsampling (time_step=32, n_freq=256). This calls
     ``compute_cwd(sigma=np.inf)`` under the hood.
  4. Save a 2x4 grid figure suitable for the academic manuscript.
  5. Print a per-class statistics table.

Outputs
-------
  preprocessing/time_frequency/tests/outputs/wvd_clean_8classes.png

Author: Kaan Emre Evci
Project: Radar Pulse Classification
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from preprocessing.time_frequency.wvd import compute_wvd  # noqa: E402


DATASET_PATH = _PROJECT_ROOT / "data_generation" / "synthetic_samples" / "dataset.h5"
OUTPUT_DIR = _THIS_FILE.parent / "outputs"
OUTPUT_FIG = OUTPUT_DIR / "wvd_clean_8classes.png"

WVD_TIME_STEP = 32
WVD_N_FREQ = 256


def _load_dataset(h5_path: Path) -> Tuple[
    h5py.File, np.ndarray, np.ndarray, np.ndarray, List[str], float, int
]:
    """Open the HDF5 dataset and load 1-D metadata arrays.

    Signals are NOT loaded eagerly; we fetch single samples on demand
    via ``f['signals'][:, :, idx].T`` (MATLAB column-major layout).
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
    """Read a single complex baseband signal (MATLAB column-major fix)."""
    ri = np.asarray(f["signals"][:, :, idx]).T
    return (ri[:, 0] + 1j * ri[:, 1]).astype(np.complex64)


def _to_db(magnitude: np.ndarray, eps: float = 1e-12, db_floor: float = -60.0) -> np.ndarray:
    """Convert magnitude to dB, with a numerical floor for display."""
    peak = float(magnitude.max())
    if peak <= 0:
        return np.full_like(magnitude, db_floor, dtype=np.float64)
    db = 20.0 * np.log10(np.maximum(magnitude, eps) / peak)
    return np.clip(db, db_floor, 0.0)


def _pick_longest_per_class(
    labels: np.ndarray,
    pulse_widths_us: np.ndarray,
    num_classes: int,
) -> List[int]:
    """For each class, return the global sample index with max pulse width."""
    picks: List[int] = []
    for cls in range(num_classes):
        cls_indices = np.where(labels == cls)[0]
        if cls_indices.size == 0:
            raise RuntimeError(f"No samples found for class index {cls}.")
        cls_pws = pulse_widths_us[cls_indices]
        best_local = int(np.argmax(cls_pws))
        picks.append(int(cls_indices[best_local]))
    return picks


def main() -> int:
    if not DATASET_PATH.exists():
        print(f"ERROR: dataset not found at {DATASET_PATH}", file=sys.stderr)
        return 1

    print(f"Loading dataset: {DATASET_PATH}")
    f, labels, snr_db, pulse_widths_us, class_names, fs, signal_length = _load_dataset(DATASET_PATH)
    num_classes = len(class_names)
    print(f"  num_classes   : {num_classes}")
    print(f"  total samples : {labels.size}")
    print(f"  sample_rate   : {fs:g} Hz")
    print(f"  signal_length : {signal_length}")
    print()

    picks = _pick_longest_per_class(labels, pulse_widths_us, num_classes)

    print("Per-class longest-pulse picks:")
    print("-" * 78)
    print(f"  {'class':<12} {'idx':>7} {'PW (us)':>10} {'SNR (dB)':>10}")
    print("-" * 78)
    for cls, idx in enumerate(picks):
        print(f"  {class_names[cls]:<12} {idx:>7d} {pulse_widths_us[idx]:>10.2f} {snr_db[idx]:>10.1f}")
    print("-" * 78)
    print()

    print(f"Computing WVD (time_step={WVD_TIME_STEP}, n_freq={WVD_N_FREQ})...")
    wvds: List[np.ndarray] = []
    t_axes: List[np.ndarray] = []
    f_axes: List[np.ndarray] = []
    stats_rows: List[Tuple[str, Tuple[int, int], float, float, bool, bool]] = []

    for cls, idx in enumerate(picks):
        sig = _read_signal(f, idx)
        wvd_mag, t_axis, f_axis = compute_wvd(
            sig,
            fs=fs,
            time_step=WVD_TIME_STEP,
            n_freq=WVD_N_FREQ,
        )
        wvds.append(wvd_mag)
        t_axes.append(t_axis)
        f_axes.append(f_axis)
        shape_tuple: Tuple[int, int] = (int(wvd_mag.shape[0]), int(wvd_mag.shape[1]))
        stats_rows.append(
            (
                class_names[cls],
                shape_tuple,
                float(wvd_mag.min()),
                float(wvd_mag.max()),
                bool(np.any(np.isnan(wvd_mag))),
                bool(np.any(np.isinf(wvd_mag))),
            )
        )

    f.close()

    print()
    print("WVD statistics per class:")
    print("-" * 78)
    print(f"  {'class':<12} {'shape':>15} {'min':>12} {'max':>12} {'NaN?':>6} {'Inf?':>6}")
    print("-" * 78)
    for row in stats_rows:
        cls_name, shape, mn, mx, has_nan, has_inf = row
        print(f"  {cls_name:<12} {str(shape):>15} {mn:>12.3e} {mx:>12.3e} {str(has_nan):>6} {str(has_inf):>6}")
    print("-" * 78)
    print()

    any_nan = any(r[4] for r in stats_rows)
    any_inf = any(r[5] for r in stats_rows)
    if any_nan or any_inf:
        print("WARN: NaN or Inf detected in WVD outputs — investigate.")
    else:
        print("OK: all 8 WVDs are finite.")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building figure ({OUTPUT_FIG.name})...")

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    axes_flat = axes.ravel()

    for cls in range(num_classes):
        ax = axes_flat[cls]
        wvd_db = _to_db(wvds[cls])
        t_us = t_axes[cls] * 1e6
        f_mhz = f_axes[cls] / 1e6
        im = ax.imshow(
            wvd_db,
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
        "Wigner-Ville Distribution — clean baseline, 8 classes",
        fontsize=12,
    )

    fig.savefig(OUTPUT_FIG, dpi=150)
    plt.close(fig)
    print(f"Saved figure -> {OUTPUT_FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
