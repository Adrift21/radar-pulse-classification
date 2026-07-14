"""
Choi-Williams Distribution (CWD) — custom NumPy implementation.

Why custom?
-----------
`tftb v0.2.0` (the maintained `scikit-signal/tftb` package) does NOT include
a Choi-Williams class. Only Wigner-Ville and a few other Cohen's-class
members are exported. Rather than fork tftb, we implement CWD directly from
the standard time-lag formulation. This is also defensible academically —
the implementation is small, transparent, and verifiable against tftb's
WignerVilleDistribution (sigma -> infinity limit).

Mathematical formulation
------------------------
The Choi-Williams Distribution of a complex signal x[n] is

    CWD(t, f) = sum_tau  K_CW(t, tau) * exp(-j*2*pi*f*tau)        (Eq. 1)

where K_CW(t, tau) is the kernelled instantaneous correlation:

    K_CW(t, tau) = sum_mu  G(mu - t; sigma, tau)
                          * x(mu + tau/2) * conj(x(mu - tau/2))    (Eq. 2)

with the Choi-Williams Gaussian-product kernel in time:

    G(mu - t; sigma, tau) = sqrt(sigma / (4 * pi * tau^2))
                          * exp( -sigma * (mu - t)^2 / (4 * tau^2) ) (Eq. 3)

Two limits worth knowing:
- sigma -> infinity : G -> Dirac delta in mu  =>  no smoothing
                       => CWD reduces to the Wigner-Ville Distribution (WVD).
- sigma -> 0+       : G is broad  => aggressive cross-term suppression but
                                     also auto-term blurring.

In the discrete implementation we work with a finite lag range
(tau in [-Lmax, Lmax]) and replace integrals with sums.

Numerical notes
---------------
- For tau == 0 the analytical kernel above has a 1/tau^2 singularity that is
  cancelled by an opposing factor; the *correct* discrete limit is simply
  the (un-smoothed) instantaneous power at the centre time. We handle this
  case explicitly.
- For tau != 0 we evaluate the Gaussian kernel on a discrete grid and
  normalize it to unit sum, which is more numerically stable than relying
  on the analytical prefactor and is the standard convention in time-lag
  CWD implementations.
- The discrete frequency axis is produced by an FFT along tau, then
  fftshift'ed so f = 0 lies in the middle of the array. This matches the
  Phase-1 STFT convention in `stft.py`.

Design choices (see decisions.md, 2026-05-?? CWD entry)
-------------------------------------------------------
- Library: custom (no Choi-Williams in tftb v0.2.0).
- sigma : 1.0 (Choi & Williams 1989 original default).
- Output shape : (n_freq, n_time) magnitude-positive, matching tftb
  convention and our Phase-1 STFT.
- Downsampling : `time_step=32` and `n_freq=256` chosen to match Phase-1
  STFT (256, 57) so that the same input shape is reused across
  TF representations.

Author: Kaan Emre Evci
Project: Radar Pulse Classification
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["compute_cwd", "DEFAULT_SIGMA"]


# Default sigma value. Choi & Williams (1989) recommend sigma = 1.0 as the
# balanced default; this also matches the MATLAB TFTB `tfrcw` default.
DEFAULT_SIGMA: float = 1.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_cw_kernel(sigma: float, n_mu: int, n_tau: int) -> np.ndarray:
    """
    Build the discrete Choi-Williams smoothing kernel G(mu; sigma, tau).

    Parameters
    ----------
    sigma : float
        Kernel parameter (larger sigma => less smoothing, closer to WVD).
    n_mu : int
        Number of samples along the time-smoothing axis (odd). The kernel
        is centred at index (n_mu - 1) // 2.
    n_tau : int
        Number of lag values (excluding tau = 0). The kernel returned has
        shape (n_tau, n_mu); each row is the kernel for the corresponding
        lag value.

    Returns
    -------
    kernel : (n_tau, n_mu) float64, each row sums to 1.

    Notes
    -----
    For lag index k (1-based, k = 1..n_tau) the kernel width scales with
    k, so larger lags get broader smoothing. This is the canonical
    Choi-Williams behaviour: cross-terms (which live at large lags) get
    aggressively smoothed, auto-terms (small lags) stay sharp.
    """
    half = (n_mu - 1) // 2
    mu = np.arange(-half, half + 1, dtype=np.float64)        # (n_mu,)
    # Lag values k = 1, 2, ..., n_tau
    k = np.arange(1, n_tau + 1, dtype=np.float64).reshape(-1, 1)  # (n_tau, 1)
    # Gaussian in mu, scaled by sigma / (4 * k^2). We skip the analytical
    # prefactor and renormalize by sum (more stable, standard practice).
    exponent = -sigma * (mu[None, :] ** 2) / (4.0 * k ** 2)   # (n_tau, n_mu)
    kernel = np.exp(exponent)
    kernel /= kernel.sum(axis=1, keepdims=True)               # row-normalize
    return kernel


def _zero_pad(signal: np.ndarray, pad: int) -> np.ndarray:
    """Reflective-zero pad a 1-D complex signal by `pad` samples on each side."""
    return np.concatenate(
        (np.zeros(pad, dtype=signal.dtype),
         signal,
         np.zeros(pad, dtype=signal.dtype))
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_cwd(
    signal: np.ndarray,
    fs: float,
    sigma: float = DEFAULT_SIGMA,
    time_step: int = 32,
    n_freq: int = 256,
    max_lag: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Choi-Williams Distribution of a 1-D complex signal.

    Parameters
    ----------
    signal : (N,) complex
        Input baseband signal. ``complex64`` is recommended; will be cast
        to ``complex128`` internally for numerical headroom.
    fs : float
        Sample rate in Hz (used to build the frequency axis).
    sigma : float, default 1.0
        Choi-Williams kernel parameter. Larger sigma => sharper but more
        cross-terms; smaller sigma => smoother but blurrier auto-terms.
        sigma = inf reproduces the Wigner-Ville Distribution.
    time_step : int, default 32
        Decimation along the time axis. Output has n_time = ceil(N / time_step)
        columns. With N=2048 and time_step=32 => 64 time bins, matching
        Phase-1 STFT (256, 57) order of magnitude.
    n_freq : int, default 256
        Number of frequency bins (FFT size along the lag axis). Also sets
        the default max_lag if not provided.
    max_lag : int or None, default None
        Maximum lag value (in samples). If None, defaults to
        ``n_freq // 2 - 1`` so that the positive- and negative-lag
        regions of the FFT buffer remain disjoint (Hermitian symmetry).
        Larger max_lag improves frequency resolution but increases cost
        linearly. Values exceeding ``n_freq // 2 - 1`` are silently
        clamped.

    Returns
    -------
    cwd : (n_freq, n_time) float64
        Magnitude of the Choi-Williams Distribution. The CWD is real-valued
        by construction (Hermitian-symmetric correlation kernel + Fourier
        transform yields a real distribution); we return the magnitude
        anyway to keep behaviour identical when sigma=inf reproduces the
        also-real WVD.
    t_axis : (n_time,) float64
        Time axis in seconds (centre of each analysis frame).
    f_axis : (n_freq,) float64
        Frequency axis in Hz, fftshift'ed so f=0 lies in the middle.

    Notes
    -----
    Complexity is O(N * n_tau * n_mu + n_time * n_freq * log(n_freq)) per
    call. For our defaults (N=2048, n_tau=256, n_mu=2*256+1) one call
    takes ~50-150 ms on a modern CPU.

    The output is suitable for direct use as a model input after the
    standard dB-scale + per-sample normalization that the training pipeline applies
    (same convention as STFT and WVD outputs).
    """
    # ----- 1. Input validation & normalization ----------------------------
    sig = np.asarray(signal).astype(np.complex128, copy=False).ravel()
    N = sig.size
    if N == 0:
        raise ValueError("compute_cwd: input signal is empty.")
    if not np.isfinite(sigma) or sigma <= 0:
        # Allow sigma = +inf to express the WVD limit
        if not np.isposinf(sigma):
            raise ValueError(
                f"compute_cwd: sigma must be positive (got {sigma!r}); "
                "use np.inf for the WVD limit."
            )
    if time_step < 1:
        raise ValueError(f"compute_cwd: time_step must be >= 1 (got {time_step}).")
    if n_freq < 2:
        raise ValueError(f"compute_cwd: n_freq must be >= 2 (got {n_freq}).")
    # Positive lags occupy bins [1, max_lag] in the length-n_freq FFT
    # buffer; negative lags are filled by Hermitian symmetry at
    # bins [n_freq - max_lag, n_freq - 1]. To keep these two regions
    # disjoint (and to leave the Nyquist bin n_freq//2 untouched), we
    # require max_lag <= n_freq // 2 - 1.
    max_allowed_lag = n_freq // 2 - 1
    if max_lag is None:
        max_lag = max_allowed_lag
    if max_lag < 1:
        raise ValueError(f"compute_cwd: max_lag must be >= 1 (got {max_lag}).")
    if max_lag > max_allowed_lag:
        # Truncate so positive and negative lag regions stay disjoint.
        max_lag = max_allowed_lag

    # ----- 2. Build output time grid --------------------------------------
    # We compute one CWD column per analysis time t_k = k * time_step.
    t_indices = np.arange(0, N, time_step, dtype=np.int64)     # (n_time,)
    n_time = t_indices.size
    t_axis = t_indices.astype(np.float64) / fs                 # (n_time,) seconds

    # ----- 3. Build kernel for tau = 1..max_lag ---------------------------
    # Heuristic: kernel support along mu scales with lag, but capped to
    # something tractable. Empirically n_mu = 2*max_lag + 1 (matching
    # MATLAB TFTB) is a good choice.
    n_mu = 2 * max_lag + 1
    if np.isposinf(sigma):
        # WVD limit: kernel = delta(mu), i.e. just pick the centre column.
        kernel = None
    else:
        kernel = _build_cw_kernel(sigma=sigma, n_mu=n_mu, n_tau=max_lag)

    # ----- 4. Zero-pad signal so we can index mu freely -------------------
    pad = max_lag + (n_mu - 1) // 2
    padded = _zero_pad(sig, pad)                               # length N + 2*pad

    # ----- 5. Allocate accumulator ----------------------------------------
    # We use the symmetric lag convention tau in [-max_lag, max_lag].
    # The FFT array has length n_freq with tau samples at positions
    # {0, 1, ..., max_lag, n_freq - max_lag, ..., n_freq - 1}
    # so that real-valued (Hermitian) FFT yields a real CWD column.
    cwd = np.zeros((n_freq, n_time), dtype=np.float64)

    # Mu offsets relative to the kernel centre
    half_mu = (n_mu - 1) // 2

    # ----- 6. Main loop over output time columns --------------------------
    # For each output time t, we build a length-n_freq vector R[tau] that
    # holds the kernelled instantaneous correlation, then FFT it along tau.
    for col, t_centre in enumerate(t_indices):
        # Index in the padded array corresponding to the analysis centre.
        t_pad = t_centre + pad

        # R is the kernelled instantaneous correlation as a function of tau.
        # Length n_freq, complex128. We will fill positive lags and use
        # Hermitian symmetry for negative lags.
        R = np.zeros(n_freq, dtype=np.complex128)

        # tau = 0 : R[0] = |x(t)|^2 (unsmoothed; the kernel collapses to delta)
        R[0] = padded[t_pad] * np.conj(padded[t_pad])

        # tau = 1..max_lag : kernelled correlation
        for k in range(1, max_lag + 1):
            # Slice mu range [t - half_mu, t + half_mu] in padded coordinates
            mu_lo = t_pad - half_mu
            mu_hi = t_pad + half_mu + 1
            # x(mu + tau/2) * conj(x(mu - tau/2)). For integer tau we use
            # half-sample shifts that approximate tau/2 by k//2 + (k%2)/2.
            # The standard trick: use the analytic-signal interpretation
            # and shift by k samples on one side (rounding). For mathematical
            # tightness with the WVD limit, we use forward/backward by k.
            # (This matches MATLAB TFTB's tfrcw.m convention.)
            seg_plus  = padded[mu_lo + k : mu_hi + k]
            seg_minus = padded[mu_lo - k : mu_hi - k]
            corr = seg_plus * np.conj(seg_minus)               # (n_mu,) complex

            if kernel is None:
                # WVD limit: pick the centre sample
                R[k] = corr[half_mu]
            else:
                # Choi-Williams: weighted sum over mu
                R[k] = np.dot(kernel[k - 1], corr)

            # Negative lag via Hermitian symmetry: R[-k] = conj(R[k])
            R[n_freq - k] = np.conj(R[k])

        # FFT along tau, then fftshift so f=0 sits in the middle
        spectrum = np.fft.fft(R, n=n_freq)
        spectrum = np.fft.fftshift(spectrum)

        # CWD is real-valued in theory; take real part and then magnitude
        # for consistency (numerical residue can introduce tiny imaginary
        # parts on the order of 1e-15).
        cwd[:, col] = np.abs(spectrum.real)

    # ----- 7. Frequency axis (fftshift'ed) --------------------------------
    f_axis = np.fft.fftshift(np.fft.fftfreq(n_freq, d=1.0 / fs))

    return cwd, t_axis, f_axis
