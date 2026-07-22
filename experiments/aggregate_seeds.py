#!/usr/bin/env python
"""Aggregate multi-seed runs into mean +/- std over seeds.

Companion to run_multiseed.py. Where docs/results_summary.md §5 reports confidence intervals
for test-set sampling error only, this reports the spread that actually matters for the
fine-grained claims: variation across training seeds (weight init + data ordering). Each
experiment's single number becomes a mean +/- std over the available seeds.

It reads:
  * seed 42  -> the canonical experiments/results/<base>       (falls back to <base>_seed42)
  * seed 43+ -> experiments/results/<base>_seed<k>

It runs safely at any stage: with only the canonical runs present it reports n=1 (no std yet),
and it fills in as run_multiseed.py completes more seeds.

Outputs (written to analysis/):
  seed_summary.csv                 per-experiment mean/std/n of test accuracy
  multiseed_accuracy.png           overall accuracy, mean +/- std error bars (when n>=2)

Usage:  python experiments/aggregate_seeds.py [--seeds 42 43 44]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "experiments" / "results"
OUT = REPO / "analysis"

TFS = ["stft", "cwd", "wvd"]
ARCHS = ["custom_cnn", "resnet50", "vit"]
TF_LABEL = {"stft": "STFT", "cwd": "CWD", "wvd": "WVD"}
ARCH_LABEL = {"custom_cnn": "Custom-CNN", "resnet50": "ResNet-50", "vit": "ViT-Small"}
TF_COLOR = {"stft": "#0072B2", "cwd": "#009E73", "wvd": "#D55E00"}
DEFAULT_SEEDS = [42, 43, 44]


def results_dir(base: str, seed: int) -> Path | None:
    """Where seed <seed> of experiment <base> lives, or None if absent.

    The canonical (seed-42) runs are at results/<base>; explicit per-seed runs at
    results/<base>_seed<k>. Prefer the explicit dir if present.
    """
    explicit = RESULTS / f"{base}_seed{seed}"
    if (explicit / "test_metrics.json").exists():
        return explicit
    if seed == 42 and (RESULTS / base / "test_metrics.json").exists():
        return RESULTS / base
    return None


def load_metrics(base: str, seed: int) -> dict | None:
    d = results_dir(base, seed)
    if d is None:
        return None
    with open(d / "test_metrics.json", encoding="utf-8") as fh:
        return json.load(fh)


def collect(seeds: list[int]) -> dict[tuple[str, str], dict]:
    """For each experiment gather test accuracy and per-SNR accuracy across seeds."""
    out = {}
    snr_keys = [str(s) for s in range(-10, 21, 2)]
    for tf in TFS:
        for arch in ARCHS:
            base = f"{tf}_{arch}"
            accs, per_snr, used = [], [], []
            for s in seeds:
                m = load_metrics(base, s)
                if m is None:
                    continue
                accs.append(m["test_accuracy"] * 100)
                per_snr.append([m["per_snr_accuracy"][k] * 100 for k in snr_keys])
                used.append(s)
            out[(tf, arch)] = {
                "acc": np.array(accs),
                "per_snr": np.array(per_snr) if per_snr else np.empty((0, len(snr_keys))),
                "seeds": used,
            }
    return out, snr_keys


def write_csv(data: dict) -> None:
    with open(OUT / "seed_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tf", "arch", "n_seeds", "seeds", "acc_mean", "acc_std", "acc_min", "acc_max"])
        for tf in TFS:
            for arch in ARCHS:
                a = data[(tf, arch)]["acc"]
                seeds = data[(tf, arch)]["seeds"]
                if a.size == 0:
                    w.writerow([TF_LABEL[tf], ARCH_LABEL[arch], 0, "", "", "", "", ""])
                    continue
                w.writerow([TF_LABEL[tf], ARCH_LABEL[arch], a.size,
                            "|".join(map(str, seeds)),
                            f"{a.mean():.3f}", f"{a.std(ddof=1) if a.size > 1 else 0:.3f}",
                            f"{a.min():.3f}", f"{a.max():.3f}"])


def print_table(data: dict) -> int:
    print("Per-experiment test accuracy, mean +/- std over seeds")
    print("| TF | Arch | n | Mean | Std | Seeds |")
    print("|---|---|---|---|---|---|")
    max_n = 0
    for tf in TFS:
        for arch in ARCHS:
            a = data[(tf, arch)]["acc"]
            seeds = data[(tf, arch)]["seeds"]
            max_n = max(max_n, a.size)
            if a.size == 0:
                print(f"| {TF_LABEL[tf]} | {ARCH_LABEL[arch]} | 0 | - | - | (none) |")
            else:
                std = a.std(ddof=1) if a.size > 1 else 0.0
                print(f"| {TF_LABEL[tf]} | {ARCH_LABEL[arch]} | {a.size} | {a.mean():.2f} "
                      f"| {std:.2f} | {','.join(map(str, seeds))} |")
    return max_n


def representation_summary(data: dict) -> None:
    print("\nRepresentation means (pooled over architectures x seeds)")
    print("| TF | n runs | Mean | Std |")
    print("|---|---|---|---|")
    pooled = {}
    for tf in TFS:
        vals = np.concatenate([data[(tf, a)]["acc"] for a in ARCHS]) if any(
            data[(tf, a)]["acc"].size for a in ARCHS) else np.array([])
        pooled[tf] = vals
        if vals.size:
            std = vals.std(ddof=1) if vals.size > 1 else 0.0
            print(f"| {TF_LABEL[tf]} | {vals.size} | {vals.mean():.2f} | {std:.2f} |")
        else:
            print(f"| {TF_LABEL[tf]} | 0 | - | - |")
    # A quick, honest read on STFT vs CWD once >=2 seeds exist per representation.
    if pooled["stft"].size >= 2 and pooled["cwd"].size >= 2:
        d = pooled["stft"].mean() - pooled["cwd"].mean()
        spread = pooled["stft"].std(ddof=1) + pooled["cwd"].std(ddof=1)
        verdict = "separable" if abs(d) > spread else "within combined std"
        print(f"\nSTFT vs CWD: Δmean = {d:+.2f} pt, combined std = {spread:.2f} pt -> {verdict}.")


def figure(data: dict) -> bool:
    """Overall accuracy with mean +/- std error bars. Returns True if drawn."""
    if not any(data[(tf, a)]["acc"].size >= 2 for tf in TFS for a in ARCHS):
        return False  # nothing with a real std yet
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    x, xticks, xlabels = 0, [], []
    for tf in TFS:
        for arch in ARCHS:
            a = data[(tf, arch)]["acc"]
            if a.size:
                std = a.std(ddof=1) if a.size > 1 else 0.0
                ax.errorbar(x, a.mean(), yerr=std, fmt="o", color=TF_COLOR[tf],
                            capsize=4, markersize=7)
            xticks.append(x)
            xlabels.append(ARCH_LABEL[arch].replace("-", "-\n"))
            x += 1
        x += 0.6
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=8)
    for i, tf in enumerate(TFS):
        ax.text(np.mean(xticks[i * 3:(i + 1) * 3]), ax.get_ylim()[1],
                TF_LABEL[tf], ha="center", va="bottom", fontweight="bold",
                color=TF_COLOR[tf], fontsize=11)
    ax.set_ylabel("Overall test accuracy (%)")
    ax.set_title("Overall accuracy, mean ± std over seeds")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "multiseed_accuracy.png", dpi=150)
    plt.close(fig)
    return True


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = p.parse_args()

    data, _ = collect(args.seeds)
    max_n = print_table(data)
    representation_summary(data)
    write_csv(data)
    drew = figure(data)

    print(f"\nWrote analysis/seed_summary.csv" + (" + multiseed_accuracy.png" if drew else ""))
    if max_n < 2:
        print("\nNote: only 1 seed available so far (no std yet). Run more seeds with:")
        print("  python experiments/run_multiseed.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
