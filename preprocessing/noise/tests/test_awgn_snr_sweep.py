"""
AWGN visualization test — SNR sweep on a single LFM sample.

Picks one clean LFM signal from the dataset, adds AWGN at six SNR
levels (-10, -5, 0, +5, +10, +20 dB), computes STFT for each, and
saves a 2x3 grid figure showing how noise degrades the TF signature
as SNR drops.

This is the AWGN module's standalone sign-off: confirms
  - ``add_awgn`` integrates cleanly with the Phase-1 STFT pipeline,
  - empirical SNR matches the requested SNR,
  - the visual signature degrades smoothly from "crystal clear" at
    +20 dB to "barely visible" at -10 dB, which is the canonical SNR
    range the rest of the project will benchmark against.

Outputs
-------
  preprocessing/noise/tests/outputs/awgn_lfm_snr_sweep.png

Author: Kaan Emre Evci
Project: Radar Pulse Classification
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from preprocessing.noise.awgn import add_awgn  # noqa: E402
from preprocessing.time_frequency.stft import compute_stft  # noqa: E402

DATASET_PATH = _PROJECT_ROOT / "data_generation" / "synthetic_samples" / "dataset.h5"
OUTPUT_DIR = _THIS_FILE.parent / "outputs"
OUTPUT_FIG = OUTPUT_DIR / "awgn_lfm_snr_sweep.png"

# SNR levels to sweep (in dB). Matches the project's SNR grid extremes
# and a few interior points.
SNR_GRID_DB = [-10.0, -5.0, 0.0, 5.0, 10.0, 20.0]

# Seed for reproducibility — independent of the dataset's per-sample seed
AWGN_SEED = 1234

# Target class for the sweep (0=LFM, see class_names in dataset)
TARGET_CLASS = 0


def _load_metadata(
    h5_path: Path,
) -> Tuple[h5py.File, np.ndarray, np.ndarray, List[str], float]:
    """Open dataset and load 1-D metadata arrays (signals fetched on demand)."""
    f = h5py.File(str(h5_path), "r")

    labels = np.asarray(f["labels"][:]).ravel().astype(np.uint8)
    pulse_widths_us = np.asarray(f["pulse_widths_us"][:]).ravel().astype(np.float32)

    class_names_raw = np.asarray(f["class_names"][:])
    class_names = [
        s.decode() if isinstance(s, (bytes, np.bytes_)) else str(s)
        for s in class_names_raw
    ]

    fs = float(np.asarray(f.attrs["sample_rate_hz"]).ravel()[0])

    return f, labels, pulse_widths_us, class_names, fs


def _read_signal(f: h5py.File, idx: int) -> np.ndarray:
    """Read a single complex baseband signal (MATLAB column-major fix)."""
    ri = np.asarray(f["signals"][:, :, idx]).T
    return (ri[:, 0] + 1j * ri[:, 1]).astype(np.complex64)


def _detect_active_region(
    signal: np.ndarray, threshold: float = 1e-6
) -> Tuple[int, int]:
    """Detect the active (non-zero) region of a padded signal.

    Returns (start_idx, stop_idx) inclusive — matching MATLAB convention.
    """
    mag = np.abs(signal)
    active = np.where(mag > threshold)[0]
    if active.size == 0:
        raise RuntimeError("No active samples detected (all zeros?)")
    return int(active[0]), int(active[-1])


def _to_db(
    magnitude: np.ndarray, eps: float = 1e-12, db_floor: float = -60.0
) -> np.ndarray:
    """Convert magnitude to dB, with a numerical floor for display."""
    peak = float(magnitude.max())
    if peak <= 0:
        return np.full_like(magnitude, db_floor, dtype=np.float64)
    db = 20.0 * np.log10(np.maximum(magnitude, eps) / peak)
    return np.clip(db, db_floor, 0.0)


def main() -> int:
    if not DATASET_PATH.exists():
        print(f"ERROR: dataset not found at {DATASET_PATH}", file=sys.stderr)
        return 1

    print(f"Loading dataset: {DATASET_PATH}")
    f, labels, pulse_widths_us, class_names, fs = _load_metadata(DATASET_PATH)
    print(f"  sample_rate : {fs:g} Hz")
    print(f"  target class: {TARGET_CLASS} ({class_names[TARGET_CLASS]})")
    print()

    # ----- 1. Pick the longest-pulse LFM ------------------------------
    cls_indices = np.where(labels == TARGET_CLASS)[0]
    cls_pws = pulse_widths_us[cls_indices]
    best_local = int(np.argmax(cls_pws))
    pick_idx = int(cls_indices[best_local])
    pick_pw = pulse_widths_us[pick_idx]
    print(f"Picked sample idx={pick_idx}, PW={pick_pw:.2f} us")
    print()

    # ----- 2. Read clean signal + detect active region ----------------
    clean = _read_signal(f, pick_idx)
    f.close()
    active_idx = _detect_active_region(clean)
    print(f"Active region: samples [{active_idx[0]}, {active_idx[1]}]")
    print(f"  active length : {active_idx[1] - active_idx[0] + 1} samples")
    print(f"  active duration: {(active_idx[1] - active_idx[0] + 1) / fs * 1e6:.2f} us")
    print()

    # ----- 3. AWGN at each SNR level + STFT ---------------------------
    rng = np.random.default_rng(AWGN_SEED)
    stfts: List[np.ndarray] = []
    t_axes: List[np.ndarray] = []
    f_axes: List[np.ndarray] = []
    emp_snrs: List[float] = []

    print(f"Sweeping SNR levels {SNR_GRID_DB} (seed={AWGN_SEED})...")
    print("-" * 70)
    print(f"  {'target (dB)':>14} {'empirical (dB)':>18} {'STFT shape':>15}")
    print("-" * 70)
    for snr_db in SNR_GRID_DB:
        noisy = add_awgn(clean, snr_db, active_idx=active_idx, rng=rng)
        # Verify empirical SNR
        clean_pwr = float(
            np.mean(np.abs(clean[active_idx[0] : active_idx[1] + 1]) ** 2)
        )
        noise_only = noisy - clean
        noise_pwr = float(
            np.mean(np.abs(noise_only[active_idx[0] : active_idx[1] + 1]) ** 2)
        )
        emp_snr = 10.0 * np.log10(clean_pwr / noise_pwr)

        # STFT of the noisy signal
        f_ax, t_ax, stft_complex = compute_stft(noisy, fs=fs)
        stft_mag = np.abs(stft_complex)
        stfts.append(stft_mag)
        t_axes.append(t_ax)
        f_axes.append(f_ax)
        emp_snrs.append(emp_snr)

        print(f"  {snr_db:>+12.1f}   {emp_snr:>+15.2f}    {str(stft_mag.shape):>15}")
    print("-" * 70)
    print()

    # ----- 4. Plot 2x3 grid -------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building figure ({OUTPUT_FIG.name})...")

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    axes_flat = axes.ravel()

    for i, snr_db in enumerate(SNR_GRID_DB):
        ax = axes_flat[i]
        stft_db = _to_db(stfts[i])
        t_us = t_axes[i] * 1e6
        f_mhz = f_axes[i] / 1e6
        im = ax.imshow(
            stft_db,
            origin="lower",
            aspect="auto",
            extent=(float(t_us[0]), float(t_us[-1]), float(f_mhz[0]), float(f_mhz[-1])),
            cmap="viridis",
            vmin=-60.0,
            vmax=0.0,
            interpolation="nearest",
        )
        ax.set_title(
            f"SNR = {snr_db:+.0f} dB  (empirical: {emp_snrs[i]:+.2f} dB)",
            fontsize=10,
        )
        ax.set_xlabel("Time [us]", fontsize=9)
        ax.set_ylabel("Frequency [MHz]", fontsize=9)
        ax.tick_params(labelsize=8)

    cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
    cbar.set_label("dB (relative to peak)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"AWGN SNR sweep on a single {class_names[TARGET_CLASS]} sample "
        f"(idx={pick_idx}, PW={pick_pw:.1f} us) — STFT magnitude",
        fontsize=12,
    )

    fig.savefig(OUTPUT_FIG, dpi=150)
    plt.close(fig)
    print(f"Saved figure -> {OUTPUT_FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
