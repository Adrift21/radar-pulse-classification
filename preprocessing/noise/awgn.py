"""
Additive White Gaussian Noise (AWGN) — runtime helper.

This module exposes a single function, :func:`add_awgn`, that adds complex
AWGN to a (typically already padded) baseband signal at a target SNR.
It is the Python-side counterpart of ``data_generation/matlab/utils/add_awgn.m``
in MATLAB; the two implementations follow exactly the same convention
so that a sample's "intended SNR" attribute in ``dataset.h5`` is honoured
at runtime when the AWGN is actually realised.

Convention
----------
- Signal power is computed over the *active region* only (i.e. the
  non-zero pulse, not the padding tails). This way, padding zeros do
  not artificially deflate the apparent signal power and inflate the
  noise needed for a target SNR.
- AWGN is *complex*: real and imaginary parts are independent
  ``N(0, noise_power/2)``, so the total complex variance is
  ``noise_power = sig_power / 10**(snr_db/10)``.
- Reproducibility is via an explicit ``rng`` parameter (a
  ``numpy.random.Generator``). Caller-controlled RNG is mandatory for
  PyTorch ``DataLoader`` workers: each worker must have its own seeded
  generator so that AWGN realisations are deterministic given a master
  seed (see ``decisions.md`` 2026-05-04 seed entry and the upcoming
  ``RadarPulseDataset.__getitem__`` design).

See also
--------
- ``data_generation/matlab/utils/add_awgn.m`` — MATLAB reference impl.
- ``decisions.md`` 2026-05-05 — "AWGN Stratejisi (pre-compute vs
  on-the-fly)" — chose on-the-fly in DataLoader.

Author: Kaan Emre Evci
Project: Radar Pulse Classification
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["add_awgn"]


def add_awgn(
    signal: np.ndarray,
    snr_db: float,
    active_idx: Optional[Tuple[int, int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Add complex AWGN at a specified SNR (active-region power).

    Parameters
    ----------
    signal : (N,) complex
        Input signal, typically already zero-padded to the frame length.
        ``complex64`` is recommended but ``complex128`` also works.
    snr_db : float
        Desired SNR in dB, relative to the signal power on the active
        region. Common project range: ``[-10, +20] dB``.
    active_idx : tuple of (int, int) or None, default None
        ``(start_idx, stop_idx)`` defining the inclusive active region
        used for signal-power estimation. If ``None``, power is computed
        over the entire signal (use this only if there is no padding).
        Indices follow Python convention (``signal[start_idx:stop_idx+1]``
        is the active slice), matching the MATLAB ``add_awgn.m`` semantics.
    rng : numpy.random.Generator or None, default None
        Random number generator for reproducibility. If ``None``, uses
        ``np.random.default_rng()`` (non-deterministic). Callers in the
        DataLoader path MUST pass an explicitly seeded generator so that
        worker subprocesses produce reproducible noise realisations.

    Returns
    -------
    noisy : (N,) complex
        ``signal + complex_AWGN``. Same dtype and length as ``signal``.

    Raises
    ------
    ValueError
        If the signal has zero power on the active region (cannot set SNR).

    Notes
    -----
    For complex AWGN, ``real`` and ``imag`` parts are independently drawn
    from ``N(0, noise_power/2)`` so the total complex variance equals
    ``noise_power``. This is the standard convention; some references
    define AWGN by per-component variance instead — we deliberately use
    the *total* variance definition to match MATLAB's ``randn`` behaviour
    on complex signals as used in ``add_awgn.m``.
    """
    sig = np.asarray(signal).ravel()
    n_samples = sig.size
    if n_samples == 0:
        raise ValueError("add_awgn: input signal is empty.")

    # ----- 1. Signal power on the active region -----------------------
    if active_idx is None:
        s_active = sig
    else:
        start_idx, stop_idx = active_idx
        # stop_idx is INCLUSIVE (matching MATLAB convention in add_awgn.m).
        # Convert to Python's half-open slice.
        s_active = sig[start_idx : stop_idx + 1]

    sig_power = float(np.mean(np.abs(s_active) ** 2))
    if sig_power <= 0.0:
        raise ValueError(
            "add_awgn: signal has zero power on the active region; "
            "cannot set SNR."
        )

    # ----- 2. Noise power from target SNR -----------------------------
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear        # total complex variance

    # ----- 3. Draw complex AWGN ---------------------------------------
    if rng is None:
        rng = np.random.default_rng()

    sigma_per_component = float(np.sqrt(noise_power / 2.0))
    noise_real = rng.standard_normal(n_samples) * sigma_per_component
    noise_imag = rng.standard_normal(n_samples) * sigma_per_component

    # Preserve input dtype (complex64 vs complex128) where possible
    if np.issubdtype(sig.dtype, np.complexfloating):
        noise = (noise_real + 1j * noise_imag).astype(sig.dtype, copy=False)
    else:
        # If somehow given a real signal, return real noise + real signal
        noise = noise_real.astype(sig.dtype, copy=False)

    return sig + noise
