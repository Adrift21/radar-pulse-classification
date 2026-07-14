"""Test-set evaluation for the trained experiments.

Evaluates a trained checkpoint on the frozen TEST split and produces the
full metric suite the paper's Results section needs, in a standard format
that the analysis scripts later read across all 9 experiments.

What it computes
----------------
- Overall test accuracy and loss
- 8x8 confusion matrix (raw counts + row-normalised) -> figure
- Per-class precision / recall / F1 (sklearn classification report)
- Per-SNR accuracy over the 16-point grid -> the x-axis of the
  "SNR robustness" curve -> figure
- Per-(class x SNR) accuracy matrix (8 x 16) -> shows which class
  collapses at which SNR (e.g. Costas / CW at low SNR, predicted in
  decisions.md) -> figure

Determinism
-----------
The test Dataset is built with add_noise=True and a FIXED master_seed
(same convention as validation in train.py), so the per-sample noise is
fixed -> evaluation is bit-for-bit reproducible. The analysis scripts rely on this.

Output layout (experiments/results/<name>/)
-------------------------------------------
    test_metrics.json          # scalar + per-class + per-SNR summary
    eval_arrays.npz            # raw arrays for cross-experiment analysis
                               #   labels, preds, snr, confusion, per_snr_acc,
                               #   class_snr_acc, class_names, snr_grid
    confusion_matrix.png
    per_snr_accuracy.png
    class_snr_accuracy.png     # heatmap

Usage
-----
    python -m experiments.evaluate --config configs/stft_custom_cnn.yaml
    python -m experiments.evaluate --config ... --checkpoint experiments/checkpoints/stft_custom_cnn/best.pth
    python -m experiments.evaluate --config ... --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from experiments.config import Config, load_config
from experiments.splits import load_splits
from models.registry import build_model
from preprocessing.datasets.radar_pulse_dataset import (
    RadarPulseDataset,
    radar_pulse_worker_init,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def read_snr_and_classnames(
    h5_path: str | Path, test_idx: np.ndarray
) -> Tuple[np.ndarray, List[str]]:
    """Read per-sample SNR (for the test indices) and class name strings.

    Follows the MATLAB column-major convention (1-D arrays stored (1, N),
    ravel()'d). See docs/dataset.md.
    """
    with h5py.File(h5_path, "r") as f:
        snr_all = np.asarray(f["snr_db"][:]).ravel().astype(np.float32)
        class_names = [
            s.decode() if isinstance(s, (bytes, bytearray)) else str(s)
            for s in np.asarray(f["class_names"][:])
        ]
    return snr_all[test_idx], class_names


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run the model over the loader; return (labels, preds, mean_loss).

    The loader MUST be built with shuffle=False so the returned order
    matches the test index order (needed to align with the SNR array).
    """
    model.eval()
    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    total_loss, total_n = 0.0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total_n += labels.size(0)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(logits.argmax(1).cpu().numpy())

    labels_arr = np.concatenate(all_labels)
    preds_arr = np.concatenate(all_preds)
    return labels_arr, preds_arr, total_loss / total_n


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def per_snr_accuracy(
    labels: np.ndarray, preds: np.ndarray, snr: np.ndarray, snr_grid: np.ndarray
) -> np.ndarray:
    """Accuracy at each SNR point. NaN where a point has no samples."""
    acc = np.full(snr_grid.size, np.nan, dtype=np.float64)
    correct = labels == preds
    for i, s in enumerate(snr_grid):
        mask = snr == s
        if mask.any():
            acc[i] = correct[mask].mean()
    return acc


def class_snr_accuracy(
    labels: np.ndarray,
    preds: np.ndarray,
    snr: np.ndarray,
    snr_grid: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """(num_classes x len(snr_grid)) accuracy matrix. NaN where empty."""
    mat = np.full((num_classes, snr_grid.size), np.nan, dtype=np.float64)
    correct = labels == preds
    for c in range(num_classes):
        for j, s in enumerate(snr_grid):
            mask = (labels == c) & (snr == s)
            if mask.any():
                mat[c, j] = correct[mask].mean()
    return mat


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------
def plot_confusion(cm: np.ndarray, class_names: List[str], out_path: Path) -> None:
    cm_norm = cm.astype(np.float64) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalised)")
    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm_norm[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                color="white" if v > 0.5 else "black",
                fontsize=7,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_snr(snr_grid: np.ndarray, acc: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(snr_grid, acc * 100, marker="o", lw=2)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Test Accuracy vs SNR")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(snr_grid)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_class_snr(
    mat: np.ndarray, class_names: List[str], snr_grid: np.ndarray, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(mat * 100, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(snr_grid.size))
    ax.set_xticklabels([f"{int(s)}" for s in snr_grid])
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Class")
    ax.set_title("Per-class Accuracy across SNR (%)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def evaluate(cfg: Config, checkpoint: Path, device: torch.device) -> Dict:
    res_dir = cfg.results_dir
    res_dir.mkdir(parents=True, exist_ok=True)

    # --- Data: TEST split, deterministic noise -----------------------
    test_idx = load_splits(cfg.data.splits_path, cfg.data.dataset_path)["test"]
    snr_test, class_names = read_snr_and_classnames(cfg.data.dataset_path, test_idx)
    snr_grid = np.arange(-10, 21, 2, dtype=np.float32)

    test_ds = RadarPulseDataset(
        h5_path=cfg.data.dataset_path,
        indices=test_idx,
        tf_repr=cfg.data.tf_repr,
        add_noise=cfg.data.add_noise,
        master_seed=cfg.data.master_seed,  # FIXED -> deterministic
        output_size=cfg.data.output_size,
        db_floor=cfg.data.db_floor,
    )
    persistent = cfg.loader.persistent_workers and cfg.loader.num_workers > 0
    loader_kwargs = dict(
        batch_size=cfg.loader.batch_size,
        shuffle=False,  # MUST be False to align with snr
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        worker_init_fn=radar_pulse_worker_init,
    )
    if cfg.loader.num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent
        loader_kwargs["prefetch_factor"] = cfg.loader.prefetch_factor
    test_loader = DataLoader(test_ds, **loader_kwargs)

    # --- Model from checkpoint ---------------------------------------
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    print(
        f"Loaded checkpoint: {checkpoint} "
        f"(epoch {ckpt.get('epoch', '?')}, val_acc {ckpt.get('val_acc', '?')})"
    )

    criterion = nn.CrossEntropyLoss()

    # --- Predictions --------------------------------------------------
    labels, preds, test_loss = collect_predictions(
        model, test_loader, criterion, device
    )
    assert labels.size == test_idx.size, "prediction/index count mismatch"

    # --- Metrics ------------------------------------------------------
    overall_acc = float((labels == preds).mean())
    cm = confusion_matrix(labels, preds, labels=list(range(cfg.data.num_classes)))
    report = classification_report(
        labels,
        preds,
        labels=list(range(cfg.data.num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    snr_acc = per_snr_accuracy(labels, preds, snr_test, snr_grid)
    cs_acc = class_snr_accuracy(labels, preds, snr_test, snr_grid, cfg.data.num_classes)

    print(f"\nTest accuracy: {overall_acc * 100:.2f}%  |  test loss: {test_loss:.4f}")
    print("Per-SNR accuracy (%):")
    for s, a in zip(snr_grid, snr_acc):
        print(f"  {int(s):+3d} dB : {a * 100:5.1f}")

    # --- Figures ------------------------------------------------------
    plot_confusion(cm, class_names, res_dir / "confusion_matrix.png")
    plot_per_snr(snr_grid, snr_acc, res_dir / "per_snr_accuracy.png")
    plot_class_snr(cs_acc, class_names, snr_grid, res_dir / "class_snr_accuracy.png")

    # --- Save metrics + raw arrays -----------------------------------
    metrics = {
        "experiment": cfg.experiment.name,
        "tf_repr": cfg.data.tf_repr,
        "model": cfg.model.name,
        "checkpoint": str(checkpoint),
        "test_accuracy": overall_acc,
        "test_loss": test_loss,
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in class_names
        },
        "macro_f1": report["macro avg"]["f1-score"],
        "per_snr_accuracy": {
            int(s): (None if np.isnan(a) else float(a))
            for s, a in zip(snr_grid, snr_acc)
        },
    }
    with open(res_dir / "test_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    np.savez(
        res_dir / "eval_arrays.npz",
        labels=labels,
        preds=preds,
        snr=snr_test,
        confusion=cm,
        per_snr_acc=snr_acc,
        class_snr_acc=cs_acc,
        snr_grid=snr_grid,
        class_names=np.array(class_names, dtype=object),
        test_accuracy=np.array([overall_acc]),
        tf_repr=np.array([cfg.data.tf_repr], dtype=object),
        model=np.array([cfg.model.name], dtype=object),
    )

    print(f"\nSaved metrics + figures + arrays to {res_dir}")
    return metrics


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=str)
    p.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="Path to checkpoint (default: <ckpt_dir>/best.pth).",
    )
    p.add_argument("--device", default=None, type=str)
    args = p.parse_args()

    cfg = load_config(args.config)
    checkpoint = (
        Path(args.checkpoint) if args.checkpoint else cfg.checkpoint_dir / "best.pth"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluate(cfg, checkpoint, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
