"""ResNet-50 wrapper for radar pulse classification (Module C).

The second architecture in the 3-architecture x 3-representation matrix.
A standard ImageNet-pretrained ResNet-50 backbone (timm), adapted to the
project's single-channel time-frequency images and 8 output classes.

Design (decisions.md, 2026-05-19 entry):
    - Input  : (B, 1, 224, 224) float32  -- single-channel TF image
    - Output : (B, 8) logits
    - Backbone: timm "resnet50", ImageNet-1k pretrained
    - 1-channel adaptation: handled by timm's ``in_chans=1`` argument,
      which sums/averages the pretrained 3-channel stem weights onto a
      single input channel (information-preserving, the standard recipe).
    - Head: timm replaces the final FC with ``num_classes=8`` automatically.
    - ~23.5M parameters (vs CustomCNN's 1.77M) -> the "strong model" the
      baseline is measured against.

Why pretrained (decisions.md): transfer learning is standard practice in
radar TF-classification literature and reflects the realistic
"fine-tune a strong backbone vs train a small custom net" question.
Reported explicitly in the paper's Methods section.

Notes
-----
- Pretrained weights are fetched by timm on first use (needs network /
  HF Hub cache). For fully offline runs, set ``pretrained=False`` via the
  config kwargs (scratch ResNet-50; expect lower low-SNR accuracy).
- ``in_chans`` is fixed to 1 to match the dataset; ``num_classes`` and the
  build kwargs come from the registry / config.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class ResNet50(nn.Module):
    """ImageNet-pretrained ResNet-50 adapted to 1-channel, 8-class input.

    Parameters
    ----------
    num_classes : int, default 8
    in_channels : int, default 1
        Single-channel TF images. Passed to timm as ``in_chans``.
    pretrained : bool, default True
        Load ImageNet-1k pretrained weights (timm handles the 3->1
        channel-weight adaptation automatically).
    drop_rate : float, default 0.0
        Dropout before the classifier (timm arg). Kept 0 by default;
        ResNet relies on BatchNorm rather than heavy dropout.
    """

    def __init__(
        self,
        num_classes: int = 8,
        in_channels: int = 1,
        pretrained: bool = True,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.pretrained = pretrained

        self.backbone = timm.create_model(
            "resnet50",
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


if __name__ == "__main__":
    # Smoke check (uses pretrained=False to avoid a network fetch here).
    model = ResNet50(pretrained=False)
    dummy = torch.randn(2, 1, 224, 224)
    out = model(dummy)
    n = sum(p.numel() for p in model.parameters())
    print(f"Output shape : {tuple(out.shape)}")
    print(f"Parameters   : {n:,}")
    # Confirm the stem accepts 1 channel
    stem = model.backbone.conv1
    print(
        f"Stem conv1   : in={stem.in_channels}, out={stem.out_channels}, "
        f"k={stem.kernel_size}"
    )
