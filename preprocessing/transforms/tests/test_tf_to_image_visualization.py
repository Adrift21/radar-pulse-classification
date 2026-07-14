"""
TF-to-image visualization test — 8 classes x 3 representations.

For each of the 8 radar pulse classes, picks the longest-pulse clean
sample, computes STFT, CWD, and WVD, runs each through ``tf_to_image``
to get a (224, 224) float32 image in [0, 1], and lays them out in a
3 x 8 grid:

      row 0 : STFT input images
      row 1 : CWD  input images
      row 2 : WVD  input images
      cols  : LFM | NLFM | Barker | Frank | Polyphase | Costas | CW | SteppedFH

This is exactly what is fed to the CNN / ResNet / ViT models.
The figure is the academic-paper "model input" centrepiece — it shows
the data flowing from raw signal -> TF representation -> normalized
image at the model's first conv layer.

Outputs
-------
  preprocessing/transforms/tests/outputs/model_input_8classes_3repr.png

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

from preprocessing.time_frequency.stft import compute_stft  # noqa: E402
from preprocessing.time_frequency.cwd import compute_cwd  # noqa: E402
from preprocessing.time_frequency.wvd import compute_wvd  # noqa: E402
from preprocessing.transforms.tf_to_image import tf_to_image  # noqa: E402


DATASET_PATH = _PROJECT_ROOT / "data_generation" / "synthetic_samples" / "dataset.h5"
OUTPUT_DIR = _THIS_FILE.parent / "outputs"
OUTPUT_FIG = OUTPUT_DIR / "model_input_8classes_3repr.png"


def _load_metadata(h5_path: Path) -> Tuple[
    h5py.File, np.ndarray, np.ndarray, List[str], float
]:
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
    f, labels, pulse_widths_us, class_names, fs = _load_metadata(DATASET_PATH)
    num_classes = len(class_names)
    print(f"  num_classes : {num_classes}")
    print(f"  sample_rate : {fs:g} Hz")
    print()

    # ----- Pick longest-pulse per class --------------------------------
    picks = _pick_longest_per_class(labels, pulse_widths_us, num_classes)
    print("Per-class longest-pulse picks:")
    for cls, idx in enumerate(picks):
        print(f"  [{cls}] {class_names[cls]:<12} idx={idx:>6d}  PW={pulse_widths_us[idx]:.2f} us")
    print()

    # ----- Compute STFT + CWD + WVD + tf_to_image for each --------------
    # Layout: stft_imgs[cls], cwd_imgs[cls], wvd_imgs[cls]
    stft_imgs: List[np.ndarray] = []
    cwd_imgs: List[np.ndarray] = []
    wvd_imgs: List[np.ndarray] = []

    print("Computing TF representations and converting to model-input images...")
    for cls, idx in enumerate(picks):
        sig = _read_signal(f, idx)

        # STFT  -> note: project API returns (f, t, Zxx)
        _, _, stft_complex = compute_stft(sig, fs=fs)
        stft_mag = np.abs(stft_complex)
        stft_imgs.append(tf_to_image(stft_mag))

        # CWD
        cwd_mag, _, _ = compute_cwd(sig, fs=fs)
        cwd_imgs.append(tf_to_image(cwd_mag))

        # WVD
        wvd_mag, _, _ = compute_wvd(sig, fs=fs)
        wvd_imgs.append(tf_to_image(wvd_mag))

        print(f"  [{cls}] {class_names[cls]:<12} STFT {stft_imgs[-1].shape} CWD {cwd_imgs[-1].shape} WVD {wvd_imgs[-1].shape}")

    f.close()
    print()

    # ----- Stats per representation row --------------------------------
    print("Per-representation aggregate stats:")
    print("-" * 70)
    for name, imgs in [("STFT", stft_imgs), ("CWD", cwd_imgs), ("WVD", wvd_imgs)]:
        arr = np.stack(imgs)   # (8, 224, 224)
        print(f"  {name:<5s} shape={arr.shape} dtype={arr.dtype} "
              f"min={arr.min():.4f} max={arr.max():.4f} "
              f"mean={arr.mean():.4f} std={arr.std():.4f}")
    print("-" * 70)
    print()

    # ----- Plot 3 x 8 grid ---------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building figure ({OUTPUT_FIG.name})...")

    fig, axes = plt.subplots(3, num_classes, figsize=(18, 7.5), constrained_layout=True)

    row_labels = ["STFT", "CWD", "WVD"]
    row_data = [stft_imgs, cwd_imgs, wvd_imgs]

    for row_i, (row_name, imgs) in enumerate(zip(row_labels, row_data)):
        for cls in range(num_classes):
            ax = axes[row_i, cls]
            im = ax.imshow(
                imgs[cls],
                origin="lower",
                aspect="auto",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            if row_i == 0:
                ax.set_title(class_names[cls], fontsize=10)
            if cls == 0:
                ax.set_ylabel(row_name, fontsize=11, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("normalized intensity [0, 1]", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Model input images (224 x 224, dB-clipped to [-60, 0] dB, "
        "per-sample max-normalized)",
        fontsize=12,
    )

    fig.savefig(OUTPUT_FIG, dpi=150)
    plt.close(fig)
    print(f"Saved figure -> {OUTPUT_FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
