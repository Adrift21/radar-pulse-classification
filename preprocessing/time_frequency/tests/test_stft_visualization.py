"""
STFT visualization test for radar pulse dataset.

Reads the project HDF5 dataset, picks the sample with the LONGEST pulse width
from each of the 8 classes (no AWGN added), computes the complex STFT with
the project's default parameters, and renders an 8-panel figure showing
dB-scale magnitude spectrograms.

Why longest pulse: longer pulses fill more of the 2048-sample frame, so each
class's TF signature is easier to read in a single visualization. Picking
the longest also doubles as a sanity check — each class's longest sample
should be near the upper bound (20 µs) if the generator's uniform [1, 20] µs draw
is working correctly.

Also prints a per-class pulse-width statistics table (min / median / max /
count) to the terminal. This validates that the generator's pulse_width_us draw
matches the documented uniform [1, 20] µs (or [4, 20] µs for Frank/Polyphase).

Output:
  - figure  : preprocessing/time_frequency/tests/outputs/stft_clean_8classes.png
  - stdout  : per-class pulse-width statistics table

Run from project root:
    python -m preprocessing.time_frequency.tests.test_stft_visualization
or directly:
    python preprocessing/time_frequency/tests/test_stft_visualization.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path so we can import the package when this script
# is run directly (not as a module). This keeps the script runnable both ways.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from preprocessing.time_frequency.stft import compute_stft  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_PATH = _PROJECT_ROOT / "data_generation" / "synthetic_samples" / "dataset.h5"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "stft_clean_8classes.png"

# STFT parameters (locked in decisions.md, 2026-05-05)
WIN_LENGTH = 256
HOP_LENGTH = 32
N_FFT = 256
WINDOW = "hann"

# Visualization parameters
DB_DYNAMIC_RANGE = 60.0  # clip dB-scale magnitude to top 60 dB
DB_FLOOR_EPS = 1e-12  # guard for log10 of near-zero magnitude


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_longest_per_class(
    h5_path: Path,
) -> tuple[np.ndarray, list[str], float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one sample per class — the one with the longest pulse width.

    Returns
    -------
    signals : np.ndarray, shape (num_classes, signal_length), complex64
        Clean complex baseband signals, one per class.
    class_names : list[str]
        Class label strings in dataset order.
    fs : float
        Sample rate in Hz.
    selected_pulse_widths_us : np.ndarray, shape (num_classes,), float32
        The pulse width (in µs) of the selected sample per class.
    labels : np.ndarray, shape (N,), uint8
        The full labels array (used for per-class stats).
    all_pulse_widths_us : np.ndarray, shape (N,), float32
        The full pulse_widths_us array (used for per-class stats).
    """
    with h5py.File(h5_path, "r") as f:
        # MATLAB writes HDF5 with reversed axis order (column-major).
        # Documented intent: signals (N, L, 2). Actual on-disk layout: (2, L, N).
        # 1-D arrays (labels, snr_db, pulse_widths_us) are stored as (1, N).
        labels = np.asarray(f["labels"][:]).ravel()  # type: ignore[index]
        all_pulse_widths_us = (
            np.asarray(f["pulse_widths_us"][:]).ravel().astype(np.float32)
        )  # type: ignore[index]
        class_names_raw = np.asarray(f["class_names"][:])  # type: ignore[index]
        class_names = [
            s.decode() if isinstance(s, bytes) else str(s) for s in class_names_raw
        ]
        fs = float(np.asarray(f.attrs["sample_rate_hz"]).ravel()[0])
        num_classes = int(np.asarray(f.attrs["num_classes"]).ravel()[0])
        signal_length = int(np.asarray(f.attrs["signal_length"]).ravel()[0])

        # For each class, pick the sample with the largest pulse width.
        # We read each chosen sample individually (no fancy indexing).
        sig_ri = np.empty((num_classes, signal_length, 2), dtype=np.float32)
        selected_pulse_widths_us = np.empty(num_classes, dtype=np.float32)
        for c in range(num_classes):
            class_indices = np.where(labels == c)[0]
            if class_indices.size == 0:
                raise RuntimeError(f"No samples found for class {c} ({class_names[c]})")
            class_pws = all_pulse_widths_us[class_indices]
            local_argmax = int(np.argmax(class_pws))
            global_idx = int(class_indices[local_argmax])
            sample = np.asarray(f["signals"][:, :, global_idx])  # type: ignore[index]  # (2, L)
            sig_ri[c] = sample.T  # -> (L, 2)
            selected_pulse_widths_us[c] = class_pws[local_argmax]

    signals = (sig_ri[..., 0] + 1j * sig_ri[..., 1]).astype(np.complex64)
    return (
        signals,
        class_names,
        fs,
        selected_pulse_widths_us,
        labels,
        all_pulse_widths_us,
    )


def print_pulse_width_table(
    class_names: list[str],
    labels: np.ndarray,
    all_pulse_widths_us: np.ndarray,
    selected_pulse_widths_us: np.ndarray,
) -> None:
    """
    Print a per-class pulse-width statistics table to validate the generator's
    uniform [1, 20] µs (or [4, 20] µs for Frank/Polyphase) draw.
    """
    header = f"{'Class':<12} {'Count':>7} {'Min':>8} {'Median':>8} {'Max':>8} {'Selected':>10}"
    sep = "-" * len(header)
    print()
    print("Per-class pulse-width statistics (µs):")
    print(sep)
    print(header)
    print(sep)
    for c, name in enumerate(class_names):
        mask = labels == c
        pws = all_pulse_widths_us[mask]
        if pws.size == 0:
            print(f"{name:<12} {'(empty)':>7}")
            continue
        print(
            f"{name:<12} {pws.size:>7d} "
            f"{pws.min():>8.2f} {float(np.median(pws)):>8.2f} {pws.max():>8.2f} "
            f"{selected_pulse_widths_us[c]:>10.2f}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# STFT image
# ---------------------------------------------------------------------------
def stft_to_db_image(Zxx: np.ndarray, dynamic_range_db: float) -> np.ndarray:
    """
    Convert complex STFT to dB-scale magnitude clipped to a dynamic range.

    The output is normalized so that 0 dB corresponds to the per-sample
    maximum magnitude bin, and -dynamic_range_db corresponds to the floor.
    """
    mag = np.abs(Zxx).astype(np.float32)
    mag_db = 20.0 * np.log10(mag + DB_FLOOR_EPS)
    mag_db -= mag_db.max()  # normalize peak to 0 dB
    return np.clip(mag_db, -dynamic_range_db, 0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    if not DATASET_PATH.exists():
        print(f"[ERROR] Dataset not found: {DATASET_PATH}", file=sys.stderr)
        print(
            "Generate it first via the MATLAB generator "
            "(data_generation/matlab/main_generate_dataset.m).",
            file=sys.stderr,
        )
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading longest-pulse sample of each class from {DATASET_PATH.name} ...")
    (signals, class_names, fs, selected_pws, labels, all_pws) = load_longest_per_class(
        DATASET_PATH
    )
    num_classes = len(class_names)
    print(
        f"  Loaded {num_classes} signals, fs = {fs / 1e6:.1f} MHz, "
        f"signal length = {signals.shape[1]} samples"
    )

    # Sanity-check table: pulse-width distribution per class.
    print_pulse_width_table(class_names, labels, all_pws, selected_pws)

    # Compute STFT for each class.
    print()
    print(
        f"Computing STFTs (win={WIN_LENGTH}, hop={HOP_LENGTH}, "
        f"n_fft={N_FFT}, {WINDOW}) ..."
    )
    spectrograms = []
    for c in range(num_classes):
        f_axis, t_axis, Zxx = compute_stft(
            signals[c],
            fs=fs,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT,
            window=WINDOW,
        )
        spectrograms.append((f_axis, t_axis, Zxx))

    # Render 2x4 grid.
    print("Rendering 2x4 figure ...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    fig.suptitle(
        f"STFT magnitude (dB) | win={WIN_LENGTH}, hop={HOP_LENGTH}, "
        f"n_fft={N_FFT}, {WINDOW} | longest-pulse sample per class (no AWGN)",
        fontsize=12,
    )

    for c, ax in enumerate(axes.flat):
        f_axis, t_axis, Zxx = spectrograms[c]
        img = stft_to_db_image(Zxx, DB_DYNAMIC_RANGE)
        extent = (t_axis[0] * 1e6, t_axis[-1] * 1e6, f_axis[0] / 1e6, f_axis[-1] / 1e6)
        im = ax.imshow(
            img,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=-DB_DYNAMIC_RANGE,
            vmax=0.0,
        )
        ax.set_title(
            f"[{c}] {class_names[c]}  (PW={selected_pws[c]:.1f} µs)", fontsize=10
        )
        ax.set_xlabel("Time (µs)", fontsize=9)
        ax.set_ylabel("Frequency (MHz)", fontsize=9)
        ax.tick_params(labelsize=8)

    cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.8, pad=0.02)
    cbar.set_label("Magnitude (dB)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure -> {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
