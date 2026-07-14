"""Training entry point for the experiments.

One config -> one training run. Architecture-agnostic: the model comes
from the registry, the dataset/representation from the config. The same
script trains all 9 experiments (3 architectures x 3 TF representations).

Pipeline
--------
    load config -> set seeds -> read frozen splits (configs/splits.npz)
    -> build train/val RadarPulseDataset + DataLoader
    -> build model (registry) + AdamW + cosine-with-warmup + AMP
    -> train loop with per-epoch noise reseeding (train only),
       validation, early stopping, checkpointing, TensorBoard logging

Per-epoch noise reseeding (decisions.md, 2026-05-18 training entry)
------------------------------------------------------------------
The Dataset fixes each sample's noise to ``master_seed + global_idx``.
For TRAINING we want a *different* noise realisation each epoch (genuine
on-the-fly augmentation, the stated reason for the runtime AWGN design).
We achieve this by bumping the train Dataset's ``master_seed`` every
epoch: ``base_seed + epoch * EPOCH_SEED_STRIDE``. VAL/TEST keep a fixed
seed -> deterministic, reproducible evaluation (the SNR-robustness analysis depends on it).

Because ``persistent_workers=True`` would cache stale Dataset copies in
the worker processes, the train DataLoader is rebuilt each epoch (cheap;
the heavy data stays on disk). The val loader is built once.

Usage
-----
    python experiments/train.py --config configs/stft_custom_cnn.yaml
    python experiments/train.py --config configs/stft_custom_cnn.yaml --epochs 2  # quick test
    python experiments/train.py --config ... --device cpu --no-amp
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from experiments.config import Config, load_config, save_config
from experiments.splits import load_splits
from models.registry import build_model
from preprocessing.datasets.radar_pulse_dataset import (
    RadarPulseDataset,
    radar_pulse_worker_init,
)

# Large stride so per-epoch seeds never collide with per-sample indices
# (indices are < 40000; stride is far larger).
EPOCH_SEED_STRIDE = 1_000_003  # prime


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def set_global_seeds(seed: int) -> None:
    """Seed Python/NumPy/torch + deterministic cuDNN (decisions.md seed)."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
def make_dataset(
    cfg: Config, indices: np.ndarray, master_seed: int
) -> RadarPulseDataset:
    return RadarPulseDataset(
        h5_path=cfg.data.dataset_path,
        indices=indices,
        tf_repr=cfg.data.tf_repr,
        add_noise=cfg.data.add_noise,
        master_seed=master_seed,
        output_size=cfg.data.output_size,
        db_floor=cfg.data.db_floor,
    )


def make_loader(
    cfg: Config,
    dataset: RadarPulseDataset,
    shuffle: bool,
) -> DataLoader:
    # persistent_workers only valid when num_workers > 0
    persistent = cfg.loader.persistent_workers and cfg.loader.num_workers > 0
    kwargs = dict(
        batch_size=cfg.loader.batch_size,
        shuffle=shuffle,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        worker_init_fn=radar_pulse_worker_init,
        drop_last=False,
    )
    if cfg.loader.num_workers > 0:
        kwargs["persistent_workers"] = persistent
        kwargs["prefetch_factor"] = cfg.loader.prefetch_factor
    return DataLoader(dataset, **kwargs)


# ---------------------------------------------------------------------
# Optim / schedule
# ---------------------------------------------------------------------
def build_optimizer(cfg: Config, model: nn.Module) -> torch.optim.Optimizer:
    if cfg.optim.optimizer.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {cfg.optim.optimizer}")


def lr_at_epoch(cfg: Config, epoch: int) -> float:
    """Linear warmup then cosine decay to min_lr. ``epoch`` is 0-based."""
    base, warm, total, lo = (
        cfg.optim.lr,
        cfg.optim.warmup_epochs,
        cfg.optim.epochs,
        cfg.optim.min_lr,
    )
    if cfg.optim.scheduler == "none":
        return base
    if warm > 0 and epoch < warm:
        return base * float(epoch + 1) / float(warm)
    if total <= warm:
        return base
    progress = (epoch - warm) / float(total - warm)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lo + (base - lo) * cosine


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


# ---------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: "torch.amp.GradScaler | None" = None,
    grad_clip: float = 0.0,
    amp: bool = False,
) -> Tuple[float, float]:
    """Run one epoch. If ``optimizer`` is None -> evaluation (no grad)."""
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, total_correct, total_n = 0.0, 0, 0
    use_amp = amp and device.type == "cuda"

    torch.set_grad_enabled(train_mode)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if train_mode:
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_n += bs

    torch.set_grad_enabled(True)
    return total_loss / total_n, total_correct / total_n


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def train(cfg: Config, device: torch.device, max_epochs: int | None = None) -> None:
    set_global_seeds(cfg.experiment.seed)

    ckpt_dir = cfg.checkpoint_dir
    res_dir = cfg.results_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # Persist the exact config used (run provenance).
    save_config(cfg, res_dir / "config.yaml")

    writer = SummaryWriter(log_dir=str(res_dir / "tb"))

    # --- Data ---------------------------------------------------------
    splits = load_splits(cfg.data.splits_path, cfg.data.dataset_path)
    print(
        f"Splits: train={splits['train'].size}, val={splits['val'].size}, "
        f"test={splits['test'].size}"
    )

    # Val dataset: fixed seed -> deterministic evaluation. Built once.
    val_ds = make_dataset(cfg, splits["val"], master_seed=cfg.data.master_seed)
    val_loader = make_loader(cfg, val_ds, shuffle=False)

    # --- Model / optim ------------------------------------------------
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg.model.name} ({n_params:,} params) on {device}")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.optim.label_smoothing)
    optimizer = build_optimizer(cfg, model)
    use_amp = cfg.optim.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    total_epochs = max_epochs if max_epochs is not None else cfg.optim.epochs

    # --- Training loop ------------------------------------------------
    best_val_loss = math.inf
    best_epoch = -1
    epochs_no_improve = 0
    history = []

    for epoch in range(total_epochs):
        t0 = time.time()

        # Per-epoch noise reseeding for TRAIN only. Rebuild train loader
        # so persistent workers pick up the new seed.
        train_seed = cfg.data.master_seed + (epoch + 1) * EPOCH_SEED_STRIDE
        train_ds = make_dataset(cfg, splits["train"], master_seed=train_seed)
        train_loader = make_loader(cfg, train_ds, shuffle=True)

        lr = lr_at_epoch(cfg, epoch)
        set_lr(optimizer, lr)

        tr_loss, tr_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=cfg.optim.grad_clip_norm,
            amp=use_amp,
        )
        va_loss, va_acc = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            amp=use_amp,
        )

        dt = time.time() - t0
        print(
            f"epoch {epoch + 1:3d}/{total_epochs} | lr {lr:.2e} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} | {dt:.1f}s"
        )

        writer.add_scalar("lr", lr, epoch)
        writer.add_scalars("loss", {"train": tr_loss, "val": va_loss}, epoch)
        writer.add_scalars("acc", {"train": tr_acc, "val": va_acc}, epoch)
        history.append(
            {
                "epoch": epoch + 1,
                "lr": lr,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": va_acc,
                "time_s": dt,
            }
        )

        # --- Checkpoint best (by val loss) ---------------------------
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": va_loss,
                    "val_acc": va_acc,
                    "config": cfg.to_dict(),
                },
                ckpt_dir / "best.pth",
            )
        else:
            epochs_no_improve += 1

        # --- Early stopping ------------------------------------------
        if cfg.optim.early_stopping and epochs_no_improve >= cfg.optim.patience:
            print(
                f"Early stopping at epoch {epoch + 1} "
                f"(no val improvement for {cfg.optim.patience} epochs)."
            )
            break

    # --- Save history + final ----------------------------------------
    with open(res_dir / "history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    torch.save({"model_state": model.state_dict()}, ckpt_dir / "last.pth")
    writer.close()

    print(
        f"\nDone. Best val loss {best_val_loss:.4f} @ epoch {best_epoch}. "
        f"Checkpoints in {ckpt_dir}, logs in {res_dir}."
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=str)
    p.add_argument(
        "--device", default=None, type=str, help="cuda / cpu (default: auto-detect)"
    )
    p.add_argument(
        "--epochs",
        default=None,
        type=int,
        help="Override epoch count (e.g. for a quick test).",
    )
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.no_amp:
        cfg.optim.amp = False

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(cfg, device, max_epochs=args.epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
