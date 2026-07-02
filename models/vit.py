"""ViT-Small wrapper for radar pulse classification (Module C).

The third architecture in the 3-architecture x 3-representation matrix,
and the Transformer counterpart to the CNN models (CustomCNN, ResNet-50).

Design (decisions.md, 2026-05-24 entry):
    - Input  : (B, 1, 224, 224) float32  -- single-channel TF image
    - Output : (B, 8) logits
    - Backbone: timm "vit_small_patch16_224", ImageNet-1k pretrained
    - ~21.5M parameters -- deliberately matched to ResNet-50 (~23.5M) so
      the comparison isolates the architectural PARADIGM (self-attention
      vs convolution) from raw model size. ViT-Base (86M) was rejected:
      it would confound capacity with paradigm and is hard to fine-tune
      reliably on 28k training samples.
    - 1-channel adaptation: timm's ``in_chans=1`` adapts the 16x16 patch
      embedding's pretrained 3-channel weights to a single channel.
    - Head: timm replaces the classifier with ``num_classes=8``.

Why a lower learning rate (decisions.md): ViTs lack the convolutional
inductive bias (locality / translation equivariance), so on small/medium
data they are more sensitive to the optimisation regime and benefit from
a lower LR than CNNs. We use 5e-5 (vs ResNet's 1e-4, CustomCNN's 3e-4),
set in configs/stft_vit.yaml.

Notes
-----
- Pretrained weights are fetched by timm on first use (network / HF cache).
- 4 GB VRAM (RTX 5050): batch=64 is the target; if OOM, drop to 32 via
  the config loader knob. This changes only the training batch, not the
  frozen test split, so the comparison stays fair.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class ViTSmall(nn.Module):
    """ImageNet-pretrained ViT-Small adapted to 1-channel, 8-class input.

    Parameters
    ----------
    num_classes : int, default 8
    in_channels : int, default 1
        Single-channel TF images. Passed to timm as ``in_chans``.
    pretrained : bool, default True
        Load ImageNet-1k pretrained weights.
    drop_rate : float, default 0.0
        Classifier dropout (timm arg).
    model_name : str, default "vit_small_patch16_224"
        timm model identifier; exposed so a Swin/other variant can be
        swapped from the config without touching this file.
    """

    def __init__(
        self,
        num_classes: int = 8,
        in_channels: int = 1,
        pretrained: bool = True,
        drop_rate: float = 0.0,
        model_name: str = "vit_small_patch16_224",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.model_name = model_name

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


if __name__ == "__main__":
    # Smoke check (pretrained=False to avoid a network fetch here).
    model = ViTSmall(pretrained=False)
    dummy = torch.randn(2, 1, 224, 224)
    out = model(dummy)
    n = sum(p.numel() for p in model.parameters())
    print(f"Output shape : {tuple(out.shape)}")
    print(f"Parameters   : {n:,}")
    pe = model.backbone.patch_embed.proj
    print(
        f"Patch embed  : in={pe.in_channels}, out={pe.out_channels}, "
        f"patch={pe.kernel_size}"
    )
