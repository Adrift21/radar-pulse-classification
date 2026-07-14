"""Custom CNN baseline for radar pulse classification.

A compact VGG-style convolutional network used as the *baseline*
architecture in the 3-architecture x 3-representation (STFT/CWD/WVD)
experiment matrix. Its role is to establish a fair, reproducible lower
reference against which ResNet-50 and ViT/Swin gains are measured.

Design (decisions.md, 2026-05-18 entry):
    - Input  : (B, 1, 224, 224) float32  -- single-channel TF image
               (output of preprocessing.transforms.tf_to_image)
    - Output : (B, 8) logits             -- 8 radar pulse classes
    - 5 conv blocks (double conv for blocks 1-4, single conv for block 5)
    - BatchNorm after every conv, ReLU activation
    - MaxPool(2x2) after each block: 224 -> 112 -> 56 -> 28 -> 14 -> 7
    - Global Average Pooling -> Dropout(0.5) -> Linear(256, 8)
    - ~1.8M parameters; fits comfortably on RTX 5050 (4 GB) at batch=64

Notes
-----
- Logits are returned (no softmax); use nn.CrossEntropyLoss in training,
  which applies log-softmax internally.
- in_channels and num_classes are parameterised for flexibility, but the
  project defaults (1 and 8) match the dataset / tf_to_image pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    """Compact VGG-style CNN baseline.

    Parameters
    ----------
    num_classes : int, default 8
        Number of output classes (radar pulse types).
    in_channels : int, default 1
        Number of input channels. tf_to_image produces 1-channel images.
    dropout : float, default 0.5
        Dropout probability applied before the final linear layer.
    """

    def __init__(
        self,
        num_classes: int = 8,
        in_channels: int = 1,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels

        # --- Feature extractor -------------------------------------------
        # Block 1: in_channels -> 32   (224 -> 112)
        # Block 2: 32 -> 64            (112 -> 56)
        # Block 3: 64 -> 128           (56 -> 28)
        # Block 4: 128 -> 256          (28 -> 14)
        # Block 5: 256 -> 256 (single) (14 -> 7)
        self.features = nn.Sequential(
            *self._double_block(in_channels, 32),
            *self._double_block(32, 64),
            *self._double_block(64, 128),
            *self._double_block(128, 256),
            *self._single_block(256, 256),
        )

        # --- Classifier head ---------------------------------------------
        # GAP collapses (B, 256, 7, 7) -> (B, 256, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(256, num_classes)

        self._init_weights()

    # ---------------------------------------------------------------------
    # Block builders
    # ---------------------------------------------------------------------
    @staticmethod
    def _double_block(in_ch: int, out_ch: int) -> list[nn.Module]:
        """Conv-BN-ReLU x2 followed by MaxPool(2x2)."""
        return [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]

    @staticmethod
    def _single_block(in_ch: int, out_ch: int) -> list[nn.Module]:
        """Conv-BN-ReLU x1 followed by MaxPool(2x2)."""
        return [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]

    # ---------------------------------------------------------------------
    # Weight initialisation (Kaiming for conv, standard for BN/linear)
    # ---------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (B, in_channels, 224, 224).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, num_classes).
        """
        x = self.features(x)  # (B, 256, 7, 7)
        x = self.global_pool(x)  # (B, 256, 1, 1)
        x = torch.flatten(x, start_dim=1)  # (B, 256)
        x = self.dropout(x)
        x = self.classifier(x)  # (B, num_classes)
        return x


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count (trainable) parameters of a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Quick manual smoke check
    model = CustomCNN()
    dummy = torch.randn(2, 1, 224, 224)
    out = model(dummy)
    print(f"Output shape : {tuple(out.shape)}")
    print(f"Parameters   : {count_parameters(model):,}")
