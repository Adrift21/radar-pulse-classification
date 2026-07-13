#!/usr/bin/env python
"""Aggregate the 9 Module-C experiments (3 TF representations x 3 architectures)
into publication-ready comparison figures + a summary table.

Reads each ``experiments/results/<tf>_<arch>/test_metrics.json`` and produces:

  analysis/all9_snr_robustness.png      -- every model, grouped by TF (colour) + arch (style)
  analysis/tf_family_snr_robustness.png -- 3 TF families, mean line + min/max band over archs
  analysis/summary_table.csv            -- test acc / macro-F1 / low-SNR accuracy per model
  (also prints the summary table as GitHub-flavoured Markdown to stdout)

Run from the repo root:  .venv/Scripts/python.exe analysis/compare_all_experiments.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "experiments" / "results"
OUT = REPO / "analysis"

# Fixed presentation order.
TFS = ["stft", "cwd", "wvd"]
ARCHS = ["custom_cnn", "resnet50", "vit"]
TF_LABEL = {"stft": "STFT", "cwd": "CWD", "wvd": "WVD"}
ARCH_LABEL = {"custom_cnn": "Custom-CNN", "resnet50": "ResNet-50", "vit": "ViT-Small"}

# Colour-blind-safe (Okabe-Ito): one hue per TF representation.
TF_COLOR = {"stft": "#0072B2", "cwd": "#009E73", "wvd": "#D55E00"}
# One line style per architecture.
ARCH_STYLE = {"custom_cnn": "-", "resnet50": "--", "vit": ":"}
ARCH_MARKER = {"custom_cnn": "o", "resnet50": "s", "vit": "^"}


def load(tf: str, arch: str) -> dict:
    path = RESULTS / f"{tf}_{arch}" / "test_metrics.json"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def snr_series(metrics: dict) -> tuple[np.ndarray, np.ndarray]:
    d = metrics["per_snr_accuracy"]
    snrs = sorted(int(k) for k in d)
    acc = np.array([d[str(s)] * 100.0 for s in snrs])
    return np.array(snrs), acc


def figure_all9(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for tf in TFS:
        for arch in ARCHS:
            snrs, acc = snr_series(data[(tf, arch)])
            ax.plot(
                snrs, acc,
                color=TF_COLOR[tf], linestyle=ARCH_STYLE[arch],
                marker=ARCH_MARKER[arch], markersize=4, linewidth=1.6,
                label=f"{TF_LABEL[tf]} · {ARCH_LABEL[arch]}",
            )
    ax.axhline(12.5, color="0.6", linewidth=0.8, linestyle=(0, (2, 3)))
    ax.text(-9.7, 13.6, "chance (8 cls)", fontsize=7, color="0.5", ha="left")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("SNR robustness — 9 models (3 TF representations × 3 architectures)")
    ax.set_xticks(range(-10, 21, 2))
    ax.set_ylim(0, 102)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=3, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "all9_snr_robustness.png", dpi=150)
    plt.close(fig)


def figure_tf_family(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for tf in TFS:
        stack = np.array([snr_series(data[(tf, a)])[1] for a in ARCHS])
        snrs = snr_series(data[(tf, ARCHS[0])])[0]
        mean = stack.mean(axis=0)
        lo, hi = stack.min(axis=0), stack.max(axis=0)
        ax.fill_between(snrs, lo, hi, color=TF_COLOR[tf], alpha=0.15)
        ax.plot(snrs, mean, color=TF_COLOR[tf], linewidth=2.4, marker="o",
                markersize=5, label=f"{TF_LABEL[tf]} (mean of 3 archs)")
    ax.axhline(12.5, color="0.6", linewidth=0.8, linestyle=(0, (2, 3)))
    ax.text(-9.7, 13.6, "chance (8 cls)", fontsize=7, color="0.5", ha="left")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("TF representation robustness — mean ± architecture spread")
    ax.set_xticks(range(-10, 21, 2))
    ax.set_ylim(0, 102)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "tf_family_snr_robustness.png", dpi=150)
    plt.close(fig)


def summary_table(data: dict) -> list[dict]:
    rows = []
    for tf in TFS:
        for arch in ARCHS:
            m = data[(tf, arch)]
            per = m["per_snr_accuracy"]
            rows.append({
                "tf": TF_LABEL[tf],
                "arch": ARCH_LABEL[arch],
                "test_acc": m["test_accuracy"] * 100,
                "macro_f1": m["macro_f1"] * 100,
                "acc_-10dB": per["-10"] * 100,
                "acc_-8dB": per["-8"] * 100,
                "acc_-6dB": per["-6"] * 100,
                "acc_ge6dB": np.mean([per[str(s)] for s in range(6, 21, 2)]) * 100,
            })
    return rows


def write_csv(rows: list[dict]) -> None:
    with open(OUT / "summary_table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in r.items()})


def print_markdown(rows: list[dict]) -> None:
    hdr = ["TF", "Arch", "Test Acc", "Macro-F1", "-10 dB", "-8 dB", "-6 dB", "≥+6 dB"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        print("| {tf} | {arch} | {test_acc:.2f} | {macro_f1:.2f} | {a10:.1f} | {a8:.1f} | {a6:.1f} | {age:.1f} |".format(
            tf=r["tf"], arch=r["arch"], test_acc=r["test_acc"], macro_f1=r["macro_f1"],
            a10=r["acc_-10dB"], a8=r["acc_-8dB"], a6=r["acc_-6dB"], age=r["acc_ge6dB"]))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp125x
    except AttributeError:
        pass
    data = {(tf, arch): load(tf, arch) for tf in TFS for arch in ARCHS}
    figure_all9(data)
    figure_tf_family(data)
    rows = summary_table(data)
    write_csv(rows)
    print_markdown(rows)
    print(f"\nSaved: all9_snr_robustness.png, tf_family_snr_robustness.png, summary_table.csv -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
