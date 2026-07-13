#!/usr/bin/env python
"""Statistical significance for the 9-experiment matrix.

What the available data allows (per-sample predictions exist ONLY for WVD; the
STFT/CWD checkpoints were pruned, so only their aggregate test_metrics.json remain):

  A. Wilson 95% CIs on overall test accuracy for all 9 models — analytic, needs only
     (n, accuracy). n_total = 6000, fixed SNR-stratified split shared by all experiments.
  B. Bootstrap validation on WVD (per-sample available): percentile-bootstrap CI vs the
     analytic Wilson CI — they should agree, which justifies using Wilson for STFT/CWD.
  C. Representation effect with architecture fixed (custom_cnn): unpaired two-proportion
     z-test between STFT/CWD/WVD. Unpaired because we lack paired per-sample preds for
     STFT/CWD; this is conservative (a paired McNemar would only be more significant).
  D. Architecture effect with representation fixed (WVD, per-sample available): PAIRED
     McNemar between the 3 WVD architectures (identical test-sample ordering verified).
  E. Figure: 9-model overall accuracy with 95% Wilson error bars, grouped by TF.

Run from repo root:  .venv/Scripts/python.exe analysis/statistical_significance.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, norm

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "experiments" / "results"
OUT = REPO / "analysis"

TFS = ["stft", "cwd", "wvd"]
ARCHS = ["custom_cnn", "resnet50", "vit"]
TF_LABEL = {"stft": "STFT", "cwd": "CWD", "wvd": "WVD"}
ARCH_LABEL = {"custom_cnn": "Custom-CNN", "resnet50": "ResNet-50", "vit": "ViT-Small"}
TF_COLOR = {"stft": "#0072B2", "cwd": "#009E73", "wvd": "#D55E00"}
N_TOTAL = 6000
Z = 1.959964  # 95%


def load_acc(tf: str, arch: str) -> float:
    with open(RESULTS / f"{tf}_{arch}" / "test_metrics.json", encoding="utf-8") as fh:
        return json.load(fh)["test_accuracy"]


def wilson(p: float, n: int, z: float = Z) -> tuple[float, float]:
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    pval = 2 * norm.sf(abs(z))
    return z, pval


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[int, int, float]:
    n01 = int(np.sum(correct_a & ~correct_b))
    n10 = int(np.sum(~correct_a & correct_b))
    res = binomtest(min(n01, n10), n01 + n10, 0.5, alternative="two-sided")
    return n01, n10, res.pvalue


def wvd_correct(arch: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(RESULTS / f"wvd_{arch}" / "eval_arrays.npz", allow_pickle=True)
    return (d["labels"], d["preds"], (d["preds"] == d["labels"]))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    rng = np.random.default_rng(42)
    acc = {(tf, a): load_acc(tf, a) for tf in TFS for a in ARCHS}

    # ---- A. Wilson CIs, all 9 -------------------------------------------------
    print("A. Overall test accuracy with Wilson 95% CI (n = 6000)")
    print("| TF | Arch | Acc % | 95% CI |")
    print("|---|---|---|---|")
    for tf in TFS:
        for a in ARCHS:
            p = acc[(tf, a)]
            lo, hi = wilson(p, N_TOTAL)
            print(f"| {TF_LABEL[tf]} | {ARCH_LABEL[a]} | {p*100:.2f} | [{lo*100:.2f}, {hi*100:.2f}] |")

    # ---- B. Bootstrap validation on WVD --------------------------------------
    print("\nB. Bootstrap vs Wilson (validation, WVD)")
    for a in ARCHS:
        _, _, corr = wvd_correct(a)
        boot = np.array([rng.choice(corr, size=corr.size, replace=True).mean()
                         for _ in range(5000)])
        blo, bhi = np.percentile(boot, [2.5, 97.5])
        wlo, whi = wilson(corr.mean(), corr.size)
        print(f"  WVD/{ARCH_LABEL[a]:<10}  bootstrap [{blo*100:.2f}, {bhi*100:.2f}]  "
              f"Wilson [{wlo*100:.2f}, {whi*100:.2f}]")

    # ---- C. Representation effect (arch fixed = custom_cnn) -------------------
    print("\nC. Representation effect, architecture fixed (Custom-CNN)")
    print("   Unpaired two-proportion z-test (conservative; paired would be stronger)")
    for tf1, tf2 in [("stft", "cwd"), ("stft", "wvd"), ("cwd", "wvd")]:
        k1 = round(acc[(tf1, "custom_cnn")] * N_TOTAL)
        k2 = round(acc[(tf2, "custom_cnn")] * N_TOTAL)
        z, pv = two_proportion_z(k1, N_TOTAL, k2, N_TOTAL)
        diff = (k1 - k2) / N_TOTAL * 100
        sig = "significant" if pv < 0.05 else "NOT significant"
        print(f"  {TF_LABEL[tf1]} vs {TF_LABEL[tf2]:<4}  Δ={diff:+5.2f} pt  z={z:6.2f}  "
              f"p={pv:.2e}  -> {sig}")

    # ---- D. Architecture effect (repr fixed = WVD, paired McNemar) ------------
    print("\nD. Architecture effect, representation fixed (WVD) — paired McNemar")
    corr = {a: wvd_correct(a)[2] for a in ARCHS}
    for a1, a2 in [("custom_cnn", "resnet50"), ("custom_cnn", "vit"), ("resnet50", "vit")]:
        n01, n10, pv = mcnemar_exact(corr[a1], corr[a2])
        sig = "significant" if pv < 0.05 else "NOT significant"
        print(f"  {ARCH_LABEL[a1]:<10} vs {ARCH_LABEL[a2]:<10}  discordant {n01}/{n10}  "
              f"p={pv:.3f}  -> {sig}")

    # ---- E. Figure: overall acc with Wilson error bars -----------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    x = 0
    xticks, xlabels = [], []
    for tf in TFS:
        for a in ARCHS:
            p = acc[(tf, a)]
            lo, hi = wilson(p, N_TOTAL)
            ax.errorbar(x, p * 100, yerr=[[(p - lo) * 100], [(hi - p) * 100]],
                        fmt="o", color=TF_COLOR[tf], capsize=4, markersize=7)
            xticks.append(x)
            xlabels.append(ARCH_LABEL[a].replace("-", "-\n"))
            x += 1
        x += 0.6
    # TF group labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=8)
    for i, tf in enumerate(TFS):
        centre = np.mean(xticks[i * 3:(i + 1) * 3])
        ax.text(centre, 100.8, TF_LABEL[tf], ha="center", fontweight="bold",
                color=TF_COLOR[tf], fontsize=11)
    ax.set_ylabel("Overall test accuracy (%)")
    ax.set_title("Overall accuracy with 95% Wilson CI (n = 6000)")
    ax.set_ylim(86, 101.5)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "overall_accuracy_ci.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved figure -> {OUT / 'overall_accuracy_ci.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
