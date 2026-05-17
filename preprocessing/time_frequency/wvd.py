"""
Wigner-Ville Distribution (WVD) — thin wrapper around ``compute_cwd``.

Why a wrapper, not a separate implementation?
---------------------------------------------
Mathematically, the Wigner-Ville Distribution is the sigma -> infinity
limit of the Choi-Williams Distribution: in that limit the Choi-Williams
kernel collapses to a Dirac delta in the time-smoothing axis, no
smoothing is applied, and the kernelled instantaneous correlation
reduces to the raw instantaneous correlation that defines WVD.

This file therefore does not re-implement WVD from scratch. It simply
calls ``compute_cwd(..., sigma=np.inf, ...)`` and returns the result
with the same axis conventions as the STFT (Phase 1) and CWD (Phase 2)
modules — i.e. frequency axis fftshift'ed so f=0 lies in the middle.

Verification against tftb
-------------------------
The underlying ``compute_cwd`` implementation has been verified against
``tftb.processing.WignerVilleDistribution`` (tftb v0.2.0) using a
full-resolution LFM test signal (N=2048):

  - Pearson correlation (custom vs tftb)         : 1.0  (to numerical precision)
  - Maximum value match (custom vs tftb)         : 1023.0 vs 1022.99999...
  - Total energy match (custom vs tftb)          : 7.117e+06 (bit-for-bit)

The only difference between the two is FFT-shift convention: tftb leaves
DC at index 0, while our convention (consistent with Phase 1 STFT)
centres f=0 in the middle of the frequency axis. This is purely a
display choice; the underlying TF representation is identical.

(See `decisions.md` 2026-05-17 entry "WVD Backend Strategy: Hybrid (Custom
Production + tftb Reference)".)

Design choices
--------------
- Downsampling : time_step=32, n_freq=256, giving output (256, 64),
  matching the Phase 1 STFT (256, 57) and Phase 2 CWD (256, 64) shapes.
  Module C can use the same model input for all three TF representations.
- Cross-term suppression : NONE. This is pure WVD by design — it is the
  "no-smoothing" baseline against which CWD's cross-term suppression is
  compared in the academic results. A smoothed variant (Pseudo-WVD or
  Smoothed-Pseudo-WVD) is intentionally out of scope.

Author: Kaan Emre Evci
Project: Radar Pulse Classification (Module B, Phase 2)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .cwd import compute_cwd

__all__ = ["compute_wvd"]


def compute_wvd(
    signal: np.ndarray,
    fs: float,
    time_step: int = 32,
    n_freq: int = 256,
    max_lag: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Wigner-Ville Distribution of a 1-D complex signal.

    This is implemented as ``compute_cwd(signal, ..., sigma=np.inf, ...)``,
    which mathematically reduces to WVD (the Choi-Williams kernel
    collapses to a Dirac delta and no smoothing is applied). See module
    docstring for verification details against ``tftb``.

    Parameters
    ----------
    signal : (N,) complex
        Input baseband signal. ``complex64`` is recommended; will be cast
        to ``complex128`` internally for numerical headroom.
    fs : float
        Sample rate in Hz (used to build the frequency axis).
    time_step : int, default 32
        Decimation along the time axis. Output has n_time = ceil(N / time_step)
        columns. With N=2048 and time_step=32 => 64 time bins, matching
        Phase-1 STFT (256, 57) and Phase-2 CWD (256, 64).
    n_freq : int, default 256
        Number of frequency bins (FFT size along the lag axis).
    max_lag : int or None, default None
        Maximum lag value (in samples). If None, defaults to
        ``n_freq // 2 - 1`` so that the positive- and negative-lag
        regions of the FFT buffer remain disjoint. Values exceeding
        ``n_freq // 2 - 1`` are silently clamped.

    Returns
    -------
    wvd : (n_freq, n_time) float64
        Magnitude of the Wigner-Ville Distribution. The WVD is real-valued
        but can have negative values at cross-term locations; we return
        ``abs(.)`` of the real part for display-friendliness and for
        consistency with ``compute_cwd``'s output convention.
    t_axis : (n_time,) float64
        Time axis in seconds (centre of each analysis frame).
    f_axis : (n_freq,) float64
        Frequency axis in Hz, fftshift'ed so f=0 lies in the middle.

    Notes
    -----
    For radar pulse classification, WVD provides the highest TF
    concentration (no smoothing) at the cost of cross-term artifacts in
    multi-component scenarios. Since our dataset is mono-component
    (each sample is a single radar pulse class), cross-terms manifest
    only as self-interference between the pulse's own time-frequency
    structure — typically visible as low-amplitude artifacts at -40 dB
    or lower, well below the auto-term peaks.
    """
    return compute_cwd(
        signal,
        fs=fs,
        sigma=np.inf,
        time_step=time_step,
        n_freq=n_freq,
        max_lag=max_lag,
    )
