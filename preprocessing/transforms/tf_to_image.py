"""
Time-frequency representation -> model input image.

A single function, :func:`tf_to_image`, converts a raw TF magnitude
matrix (output of ``compute_stft``, ``compute_cwd``, or ``compute_wvd``)
into the float-tensor model input the networks consume.

Pipeline
--------
    raw magnitude  ->  dB (peak-relative)  ->  clip to [db_floor, 0]
                  ->  linear remap to [0, 1]  ->  resize to (224, 224)
                  ->  float32 single channel

Design choices (decisions.md, 2026-05-17 TF-to-image entry)
-----------------------------------------------------------
- **Per-sample max-normalize.** dB is taken relative to the per-sample
  peak. After clipping to ``[db_floor, 0]`` the result is linearly
  remapped to ``[0, 1]``. This makes the transform invariant to the
  absolute signal power, so adding AWGN at different SNR levels does
  not shift the distribution of model inputs in a way that depends on
  the un-normalized peak. (Global z-score normalisation was considered
  and rejected because adding AWGN changes per-sample mean/std and
  would create train/test inconsistencies. See decisions.md entry.)
- **Single channel.** Output is one channel; the model's first conv
  layer (``Conv2d(in_channels=1, ...)``) is responsible for expanding.
  This keeps disk/RAM small and avoids redundant RGB triplication.
- **224 x 224 output.** Standard ResNet / ViT input size; lets all
  three TF representations (STFT (256, 57), CWD (256, 64), WVD
  (256, 64)) feed the *same* model architecture for an apples-to-apples
  comparison across the nine experiments.
- **Linear interpolation by default.** Time axis upsamples (~57-64 ->
  224), frequency axis downsamples (256 -> 224); ``cv2.INTER_LINEAR``
  is well-behaved in both directions and is the de facto standard for
  spectrogram-like images.

Author: Kaan Emre Evci
Project: Radar Pulse Classification
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import cv2

__all__ = ["tf_to_image", "DB_FLOOR_DEFAULT"]


DB_FLOOR_DEFAULT: float = -60.0


def tf_to_image(
    tf_magnitude: np.ndarray,
    output_size: Tuple[int, int] = (224, 224),
    db_floor: float = DB_FLOOR_DEFAULT,
    eps: float = 1e-12,
    interpolation: str = "linear",
) -> np.ndarray:
    """
    Convert a TF magnitude matrix to a model-ready image.

    Parameters
    ----------
    tf_magnitude : (n_freq, n_time) array_like
        Non-negative magnitude of a time-frequency representation.
        Real-valued; complex inputs should be magnitudized by the caller
        (``np.abs(z)``) before being passed in.
    output_size : (H, W), default (224, 224)
        Output image height and width.
    db_floor : float, default -60.0
        Lower bound (in dB, relative to per-sample peak) for the
        dB-scale clipping. Values below this floor are clipped to it.
    eps : float, default 1e-12
        Small constant added inside ``log10`` to avoid ``log(0)``.
    interpolation : str, default "linear"
        Interpolation mode for resizing. One of:
          - ``"linear"`` : ``cv2.INTER_LINEAR`` (default, recommended).
          - ``"area"``   : ``cv2.INTER_AREA`` (best for pure downsample).
          - ``"nearest"``: ``cv2.INTER_NEAREST`` (debug / sharp edges).

    Returns
    -------
    image : (H, W) float32
        Single-channel float image in [0, 1]. Shape exactly equals
        ``output_size``; dtype is ``float32`` (PyTorch tensor friendly).

    Raises
    ------
    ValueError
        If ``tf_magnitude`` is not 2-D or has zero size, or if
        ``interpolation`` is not one of the supported modes.

    Notes
    -----
    Per-sample peak normalization means each output covers the full
    [0, 1] range whenever the input has any non-zero content. A pure
    zero input is returned as an all-zero image (no division by zero).
    """
    arr = np.asarray(tf_magnitude)
    if arr.ndim != 2:
        raise ValueError(
            f"tf_to_image: expected a 2-D magnitude array, got shape "
            f"{arr.shape!r}."
        )
    if arr.size == 0:
        raise ValueError("tf_to_image: input array is empty.")

    # ----- 1. dB-scale (peak-relative) -----------------------------------
    arr64 = arr.astype(np.float64, copy=False)
    peak = float(arr64.max())
    if peak <= 0.0:
        # Pure-zero input: return a zero image, no division.
        return np.zeros(output_size, dtype=np.float32)

    # 20 * log10(x / peak) — peak maps to 0 dB.
    db = 20.0 * np.log10(np.maximum(arr64, eps) / peak)

    # ----- 2. Clip to [db_floor, 0] -------------------------------------
    db_clipped = np.clip(db, db_floor, 0.0)

    # ----- 3. Linear remap [db_floor, 0] -> [0, 1] ----------------------
    # (db - db_floor) / (0 - db_floor) = (db - db_floor) / (-db_floor)
    if db_floor >= 0.0:
        raise ValueError(
            f"tf_to_image: db_floor must be negative (got {db_floor})."
        )
    norm = (db_clipped - db_floor) / (-db_floor)
    norm = np.clip(norm, 0.0, 1.0)   # belt-and-braces

    # ----- 4. Resize to output_size ------------------------------------
    interp_map = {
        "linear":  cv2.INTER_LINEAR,
        "area":    cv2.INTER_AREA,
        "nearest": cv2.INTER_NEAREST,
    }
    if interpolation not in interp_map:
        raise ValueError(
            f"tf_to_image: unknown interpolation {interpolation!r}; "
            f"expected one of {list(interp_map.keys())}."
        )
    # cv2.resize takes (W, H) order, not (H, W)
    H, W = output_size
    resized = cv2.resize(
        norm.astype(np.float32, copy=False),
        dsize=(W, H),
        interpolation=interp_map[interpolation],
    )

    return resized.astype(np.float32, copy=False)
