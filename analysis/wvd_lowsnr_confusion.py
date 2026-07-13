#!/usr/bin/env python
"""Low-SNR confusion analysis for the WVD models.

WVD is the representation that collapses at low SNR (see docs/results_summary.md).
This script uses the per-sample arrays saved by evaluate.py
(``experiments/results/wvd_<arch>/eval_arrays.npz``: labels, preds, snr) to show
*which* classes WVD confuses at -10/-8/-6 dB, and prints the dominant confusion
pairs. STFT/CWD cannot be included here: their checkpoints were pruned (the .pth
files are gitignored/large) so their per-sample predictions are unavailable
without retraining; only the aggregate confusion_matrix.png remains for them.

Run from repo root:  .venv/Scripts/python.exe analysis/wvd_lowsnr_confusion.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "experiments" / "results"
OUT = REPO / "analysis"

ARCHS = ["custom_cnn", "resnet50", "vit"]
ARCH_LABEL = {"custom_cnn": "Custom-CNN", "resnet50": "ResNet-50", "vit": "ViT-Small"}
LOW_SNRS = [-10, -8, -6]


def load_arrays(arch: str) -> dict:
    d = np.load(RESULTS / f"wvd_{arch}" / "eval_arrays.npz", allow_pickle=True)
    return {
        "labels": d["labels"], "preds": d["preds"], "snr": d["snr"],
        "class_names": [str(c) for c in d["class_names"]],
    }


def confusion_at(labels, preds, mask, n) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(labels[mask], preds[mask]):
        cm[t, p] += 1
    return cm


def top_pairs(cm: np.ndarray, names: list[str], k: int = 6) -> list[tuple]:
    """Top off-diagonal (true -> predicted) confusion pairs as row-normalised rates."""
    row = cm.sum(axis=1, keepdims=True)
    rate = np.divide(cm, np.maximum(row, 1))
    pairs = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append((rate[i, j], cm[i, j], names[i], names[j]))
    pairs.sort(reverse=True)
    return pairs[:k]


def figure_confusion_grid(arch: str, arrays: dict) -> None:
    names = arrays["class_names"]
    n = len(names)
    fig, axes = plt.subplots(1, len(LOW_SNRS), figsize=(4.6 * len(LOW_SNRS), 4.4))
    for ax, snr in zip(axes, LOW_SNRS):
        mask = np.isclose(arrays["snr"], snr)
        cm = confusion_at(arrays["labels"], arrays["preds"], mask, n)
        row = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, np.maximum(row, 1))
        im = ax.imshow(norm, cmap="magma", vmin=0, vmax=1)
        acc = np.trace(cm) / max(cm.sum(), 1) * 100
        ax.set_title(f"{snr:+d} dB  (acc {acc:.0f}%)", fontsize=11)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=90, fontsize=7)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("Predicted", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("True", fontsize=9)
        for i in range(n):
            for j in range(n):
                v = norm[i, j]
                if v >= 0.08:
                    ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                            fontsize=6, color="white" if v < 0.6 else "black")
    fig.suptitle(f"WVD · {ARCH_LABEL[arch]} — row-normalised confusion at low SNR", fontsize=12)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="row fraction")
    fig.savefig(OUT / f"wvd_lowsnr_confusion_{arch}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    data = {a: load_arrays(a) for a in ARCHS}
    names = data["custom_cnn"]["class_names"]
    n = len(names)

    # Figure for the best WVD model.
    figure_confusion_grid("custom_cnn", data["custom_cnn"])

    # Text: dominant confusion pairs per SNR, pooled across the 3 WVD archs
    # (more robust than a single model) at each low SNR.
    print("WVD low-SNR confusion — dominant (true -> predicted) pairs, pooled over 3 archs")
    print("=" * 74)
    for snr in LOW_SNRS:
        lab = np.concatenate([data[a]["labels"][np.isclose(data[a]["snr"], snr)] for a in ARCHS])
        prd = np.concatenate([data[a]["preds"][np.isclose(data[a]["snr"], snr)] for a in ARCHS])
        cm = confusion_at(lab, prd, np.ones(len(lab), bool), n)
        acc = np.trace(cm) / cm.sum() * 100
        print(f"\n{snr:+d} dB  (pooled acc {acc:.1f}%, {cm.sum()} samples)")
        for rate, cnt, ti, pj in top_pairs(cm, names, k=6):
            print(f"    {ti:>9s} -> {pj:<9s}  {rate*100:4.0f}%  (n={cnt})")

    # Per-class survival at -10 dB: which classes stay above chance?
    print("\n" + "=" * 74)
    print("Per-class recall at -10 dB (pooled over 3 archs):")
    lab = np.concatenate([data[a]["labels"][np.isclose(data[a]["snr"], -10)] for a in ARCHS])
    prd = np.concatenate([data[a]["preds"][np.isclose(data[a]["snr"], -10)] for a in ARCHS])
    cm = confusion_at(lab, prd, np.ones(len(lab), bool), n)
    recalls = np.divide(np.diag(cm), np.maximum(cm.sum(axis=1), 1))
    for i in np.argsort(recalls):
        print(f"    {names[i]:>9s}  recall {recalls[i]*100:5.1f}%")
    print(f"\nSaved figure -> {OUT / 'wvd_lowsnr_confusion_custom_cnn.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
