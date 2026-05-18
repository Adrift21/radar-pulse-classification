"""
Short-Time Fourier Transform (STFT) for radar pulse signals.

This module provides a thin wrapper around scipy.signal.stft, configured with
the project's chosen STFT parameters (Hann window, 256-sample window length,
hop length 32, no zero-padding). The wrapper returns the complex STFT so that
downstream code can compute magnitude, phase, or power spectrograms as needed.

Reference parameters (decided in decisions.md, 2026-05-05):
    win_length  = 256       (~12.5% of 2048-sample signal)
    window      = 'hann'    (-32 dB side lobe, academic standard)
    hop_length  = 32        (87.5% overlap, 57 time bins for L=2048)
    n_fft       = 256       (no zero-padding, true frequency resolution)

Output shape for an L-sample input is (n_fft // 2 + 1, n_frames), where
n_frames = (L - win_length) // hop_length + 1. For our default L=2048 this
yields (129, 57).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import stft as scipy_stft


def compute_stft(
    signal: np.ndarray,
    fs: float,
    win_length: int = 256,
    hop_length: int = 32,
    n_fft: int = 256,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the complex Short-Time Fourier Transform of a 1D signal.

    Parameters
    ----------
    signal : np.ndarray
        1D input signal, real or complex. Shape (L,).
    fs : float
        Sample rate in Hz (e.g. 100e6 for the project dataset).
    win_length : int, default 256
        Window length in samples. Must satisfy win_length <= n_fft.
    hop_length : int, default 32
        Number of samples between successive frames (= win_length - noverlap).
    n_fft : int, default 256
        FFT size. Equal to win_length means no zero-padding.
    window : str, default 'hann'
        Window function name passed to scipy.signal.get_window.

    Returns
    -------
    f : np.ndarray
        Frequency bin centers in Hz. Shape (n_fft // 2 + 1,) for real input,
        or (n_fft,) for complex input with return_onesided=False.
    t : np.ndarray
        Frame center times in seconds. Shape (n_frames,).
    Zxx : np.ndarray
        Complex STFT coefficients. Shape (n_freq_bins, n_frames), dtype complex.

    Notes
    -----
    For complex baseband signals (the project default), the spectrum is
    two-sided and centered around DC. We use return_onesided=False so that
    negative frequencies are preserved. scipy returns frequencies in
    "fftshift" order (0, +f, ..., -f), so we apply np.fft.fftshift along
    the frequency axis to obtain a monotonic [-fs/2, +fs/2] layout suitable
    for visualization.
    """
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1D, got shape {signal.shape}")
    if win_length > n_fft:
        raise ValueError(f"win_length ({win_length}) must be <= n_fft ({n_fft})")
    if hop_length <= 0 or hop_length > win_length:
        raise ValueError(
            f"hop_length ({hop_length}) must be in (0, win_length={win_length}]"
        )

    is_complex = np.iscomplexobj(signal)
    noverlap = win_length - hop_length

    f, t, Zxx = scipy_stft(
        signal,
        fs=fs,
        window=window,
        nperseg=win_length,
        noverlap=noverlap,
        nfft=n_fft,
        return_onesided=not is_complex,
        boundary=None,  # type: ignore[arg-type]  # scipy stubs incorrectly type as str-only
        padded=False,
    )

    # For complex (two-sided) input, scipy returns frequencies in fft order.
    # Shift to monotonic [-fs/2, +fs/2] for clean visualization.
    if is_complex:
        f = np.fft.fftshift(f)
        Zxx = np.fft.fftshift(Zxx, axes=0)

    return f, t, Zxx
