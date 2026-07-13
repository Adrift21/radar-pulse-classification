#!/usr/bin/env python
"""Qualitative TF illustration: the same signal under STFT / CWD / WVD across SNR.

This is the visual backbone of the cross-term argument (docs/results_summary.md §3.3):
for a single radar pulse, show how each time-frequency representation degrades as SNR
drops. WVD's high-SNR sharpness turns into a noise-driven cross-term speckle that
drowns the signature at -10 dB, while STFT/CWD stay readable — exactly the mechanism
behind WVD's low-SNR accuracy collapse.

Rows = STFT / CWD / WVD (project-standard params). Columns = SNR levels. One figure
per class. AWGN is added on-the-fly with a fixed seed (reproducible).

Run from repo root:  .venv/Scripts/python.exe analysis/qualitative_tf_illustration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from preprocessing.noise.awgn import add_awgn  # noqa: E402
from preprocessing.time_frequency.cwd import compute_cwd  # noqa: E402
from preprocessing.time_frequency.stft import compute_stft  # noqa: E402
from preprocessing.time_frequency.wvd import compute_wvd  # noqa: E402

DATASET = REPO / "data_generation" / "synthetic_samples" / "dataset.h5"
OUT = REPO / "analysis"

SNRS = [20, 0, -6, -10]          # columns, high -> low
CLASSES = ["LFM", "Costas"]      # one figure each (chirp vs frequency-agile)
TIME_STEP, N_FREQ = 32, 256
DB_FLOOR = -60.0
SEED = 42


def load_meta(f):
    labels = np.asarray(f["labels"][:]).ravel().astype(int)
    pw = np.asarray(f["pulse_widths_us"][:]).ravel().astype(np.float32)
    names = [s.decode() if isinstance(s, (bytes, np.bytes_)) else str(s)
             for s in np.asarray(f["class_names"][:])]
    fs = float(np.asarray(f.attrs["sample_rate_hz"]).ravel()[0])
    return labels, pw, names, fs


def read_signal(f, idx: int) -> np.ndarray:
    ri = np.asarray(f["signals"][:, :, idx]).T
    return (ri[:, 0] + 1j * ri[:, 1]).astype(np.complex64)


def active_idx(sig: np.ndarray) -> tuple[int, int]:
    nz = np.nonzero(np.abs(sig) > 0)[0]
    return (int(nz[0]), int(nz[-1])) if nz.size else (0, sig.size - 1)


def to_db(mag: np.ndarray) -> np.ndarray:
    peak = float(mag.max())
    if peak <= 0:
        return np.full_like(mag, DB_FLOOR, dtype=np.float64)
    db = 20.0 * np.log10(np.maximum(mag, 1e-12) / peak)
    return np.clip(db, DB_FLOOR, 0.0)


def tf_images(sig: np.ndarray, fs: float):
    """Return {name: (db_image, t_us, f_mhz)} for the three representations."""
    fS, tS, Zxx = compute_stft(sig, fs=fs)
    cwd_mag, tC, fC = compute_cwd(sig, fs=fs, time_step=TIME_STEP, n_freq=N_FREQ)
    wvd_mag, tW, fW = compute_wvd(sig, fs=fs, time_step=TIME_STEP, n_freq=N_FREQ)
    return {
        "STFT": (to_db(np.abs(Zxx)), tS * 1e6, fS / 1e6),
        "CWD": (to_db(cwd_mag), tC * 1e6, fC / 1e6),
        "WVD": (to_db(wvd_mag), tW * 1e6, fW / 1e6),
    }


def build_figure(cls_name: str, sig: np.ndarray, fs: float, pw: float) -> Path:
    rng = np.random.default_rng(SEED)
    aidx = active_idx(sig)
    rows = ["STFT", "CWD", "WVD"]
    fig, axes = plt.subplots(3, len(SNRS), figsize=(3.1 * len(SNRS), 8.6),
                             constrained_layout=True)
    im = None
    for c, snr in enumerate(SNRS):
        noisy = add_awgn(sig, float(snr), active_idx=aidx, rng=rng)
        imgs = tf_images(noisy, fs)
        for r, name in enumerate(rows):
            ax = axes[r, c]
            db, t_us, f_mhz = imgs[name]
            im = ax.imshow(db, origin="lower", aspect="auto",
                           extent=(float(t_us[0]), float(t_us[-1]),
                                   float(f_mhz[0]), float(f_mhz[-1])),
                           cmap="viridis", vmin=DB_FLOOR, vmax=0.0,
                           interpolation="nearest")
            if r == 0:
                ax.set_title(f"{snr:+d} dB", fontsize=13, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{name}\nFreq [MHz]", fontsize=11)
            else:
                ax.set_ylabel("")
            if r == 2:
                ax.set_xlabel("Time [µs]", fontsize=9)
            ax.tick_params(labelsize=7)
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.015)
    cbar.set_label("dB (rel. peak)", fontsize=9)
    fig.suptitle(f"TF representations vs SNR — {cls_name} (PW={pw:.1f} µs)\n"
                 f"WVD sharp at high SNR, drowns in noise-driven cross-terms at -10 dB",
                 fontsize=13)
    out = OUT / f"qualitative_tf_{cls_name.lower()}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    if not DATASET.exists():
        print(f"ERROR: dataset not found: {DATASET}", file=sys.stderr)
        return 1
    with h5py.File(str(DATASET), "r") as f:
        labels, pw, names, fs = load_meta(f)
        for cls_name in CLASSES:
            cls = names.index(cls_name)
            cls_idx = np.where(labels == cls)[0]
            idx = int(cls_idx[int(np.argmax(pw[cls_idx]))])  # longest pulse = fills frame
            sig = read_signal(f, idx)
            out = build_figure(cls_name, sig, fs, float(pw[idx]))
            print(f"{cls_name:<8} idx={idx:<6} PW={pw[idx]:.1f} us  ->  {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
