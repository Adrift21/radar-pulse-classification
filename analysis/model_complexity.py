#!/usr/bin/env python
"""Parameter count + FLOPs for the 3 architectures (Module C resource-efficiency table).

Instantiates each architecture via the project's own model registry (num_classes=8,
in_channels=1, 224x224 input) and reports total/trainable parameters and forward
FLOPs. `pretrained` is forced to False here: ImageNet weights change no shape, so the
counts are identical while avoiding a network download.

FLOPs are measured with torch's built-in FlopCounterMode (counts conv, linear, and
attention matmuls). We report both GFLOPs (torch convention, 1 MAC = 2 FLOPs) and
GMACs = GFLOPs / 2 for readers who prefer MACs.

Run from repo root:  .venv/Scripts/python.exe analysis/model_complexity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.config import load_config  # noqa: E402
from models.registry import build_model  # noqa: E402

OUT = REPO / "analysis"
# One representative config per architecture (the model is TF-representation agnostic).
CONFIGS = {
    "Custom-CNN": "configs/stft_custom_cnn.yaml",
    "ResNet-50": "configs/stft_resnet50.yaml",
    "ViT-Small": "configs/stft_vit.yaml",
}
INPUT = (1, 1, 224, 224)


def count_params(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_flops(model) -> int:
    x = torch.zeros(INPUT)
    fcm = FlopCounterMode(display=False)
    with torch.no_grad(), fcm:
        model(x)
    return fcm.get_total_flops()


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    torch.manual_seed(0)
    rows = []
    for label, cfg_path in CONFIGS.items():
        cfg = load_config(REPO / cfg_path)
        kwargs = dict(cfg.model.kwargs or {})
        if "pretrained" in kwargs:
            kwargs["pretrained"] = False  # counts are weight-independent; skip download
        model = build_model(_CfgShim(cfg, kwargs)).eval()
        total, trainable = count_params(model)
        flops = count_flops(model)
        rows.append((label, cfg.model.name, total, trainable, flops))

    # Markdown table
    print(f"{'Model':<12} {'name':<11} {'Params (M)':>11} {'Trainable (M)':>14} {'GFLOPs':>9} {'GMACs':>9}")
    print("-" * 72)
    for label, name, total, trainable, flops in rows:
        print(f"{label:<12} {name:<11} {total/1e6:>11.2f} {trainable/1e6:>14.2f} "
              f"{flops/1e9:>9.2f} {flops/2e9:>9.2f}")

    print("\nMarkdown:")
    print("| Model | Params (M) | GFLOPs @224² | GMACs |")
    print("|---|---|---|---|")
    for label, name, total, trainable, flops in rows:
        print(f"| {label} | {total/1e6:.2f} | {flops/1e9:.2f} | {flops/2e9:.2f} |")
    return 0


class _CfgShim:
    """Wrap a loaded Config so build_model sees overridden model.kwargs."""

    def __init__(self, cfg, kwargs):
        self._cfg = cfg
        self.data = cfg.data

        class _M:
            pass

        self.model = _M()
        self.model.name = cfg.model.name
        self.model.kwargs = kwargs


if __name__ == "__main__":
    raise SystemExit(main())
