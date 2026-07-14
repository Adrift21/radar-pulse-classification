"""Regression tests for the scientific core: AWGN, the TF transforms, and tf_to_image.

These tests deliberately require **no dataset**: every signal is synthesised in-process, so
the suite runs on a fresh clone (and in CI) without the 260 MB dataset.h5.

They pin the correctness claims the paper's methodology rests on:
  * AWGN achieves the requested SNR, measured against active-region power (not the padding).
  * The WVD is exactly the CWD in the sigma -> infinity limit (the wrapper's core claim).
  * The three TF transforms return the documented shapes and stay finite.
  * tf_to_image produces the (224, 224) float32 image in [0, 1] that every model consumes.
  * The split fingerprint is deterministic and actually sensitive to dataset changes.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.splits import dataset_fingerprint
from preprocessing.noise.awgn import add_awgn
from preprocessing.time_frequency.cwd import compute_cwd
from preprocessing.time_frequency.stft import compute_stft
from preprocessing.time_frequency.wvd import compute_wvd
from preprocessing.transforms.tf_to_image import tf_to_image

FS = 100e6
N = 2048


def make_lfm(pulse_len: int = 1200, start: int = 400, b_hz: float = 10e6) -> tuple[np.ndarray, tuple[int, int]]:
    """A zero-padded complex LFM chirp, mirroring the real dataset's layout."""
    t = np.arange(pulse_len) / FS
    k = b_hz / (pulse_len / FS)          # chirp rate
    pulse = np.exp(1j * np.pi * k * t**2).astype(np.complex64)
    sig = np.zeros(N, dtype=np.complex64)
    sig[start:start + pulse_len] = pulse
    return sig, (start, start + pulse_len - 1)


def make_tone(freq_hz: float = 5e6) -> np.ndarray:
    t = np.arange(N) / FS
    return np.exp(2j * np.pi * freq_hz * t).astype(np.complex64)


# ---------------------------------------------------------------- AWGN


@pytest.mark.parametrize("target_snr", [-10.0, -4.0, 0.0, 6.0, 20.0])
def test_awgn_achieves_target_snr_on_active_region(target_snr):
    """The empirical SNR must match the requested one, measured on the pulse only."""
    sig, active = make_lfm()
    rng = np.random.default_rng(0)
    noisy = add_awgn(sig, target_snr, active_idx=active, rng=rng)

    noise = noisy - sig
    lo, hi = active
    sig_p = np.mean(np.abs(sig[lo:hi + 1]) ** 2)
    noise_p = np.mean(np.abs(noise[lo:hi + 1]) ** 2)
    empirical = 10 * np.log10(sig_p / noise_p)

    assert empirical == pytest.approx(target_snr, abs=0.5)


def test_awgn_ignoring_active_region_biases_the_snr():
    """Guard the reason active_idx exists: padding would otherwise skew the SNR.

    Measuring power over the whole zero-padded frame understates the signal power, so the
    achieved SNR on the pulse comes out biased. This test pins that the bias is real (and
    therefore that passing active_idx is not optional).
    """
    sig, active = make_lfm()
    lo, hi = active
    rng = np.random.default_rng(0)

    noisy = add_awgn(sig, 0.0, active_idx=None, rng=rng)  # deliberately wrong usage
    noise = noisy - sig
    sig_p = np.mean(np.abs(sig[lo:hi + 1]) ** 2)
    noise_p = np.mean(np.abs(noise[lo:hi + 1]) ** 2)
    empirical = 10 * np.log10(sig_p / noise_p)

    assert abs(empirical) > 1.0, "expected a clear bias when active_idx is omitted"


def test_awgn_is_reproducible_and_seed_sensitive():
    sig, active = make_lfm()
    a = add_awgn(sig, 0.0, active_idx=active, rng=np.random.default_rng(42))
    b = add_awgn(sig, 0.0, active_idx=active, rng=np.random.default_rng(42))
    c = add_awgn(sig, 0.0, active_idx=active, rng=np.random.default_rng(43))

    assert np.array_equal(a, b), "same seed must give bit-identical noise"
    assert not np.array_equal(a, c), "a different seed must give different noise"


def test_awgn_noise_is_complex_and_preserves_shape_and_dtype():
    sig, active = make_lfm()
    noisy = add_awgn(sig, 0.0, active_idx=active, rng=np.random.default_rng(0))
    noise = noisy - sig

    assert noisy.shape == sig.shape
    assert np.iscomplexobj(noisy)
    # Real and imaginary parts each carry half the noise power.
    assert np.var(noise.real) == pytest.approx(np.var(noise.imag), rel=0.2)


# ------------------------------------------------- TF representations


def test_wvd_equals_cwd_in_the_infinite_sigma_limit():
    """The WVD is implemented as CWD(sigma -> inf); pin that identity exactly."""
    sig, _ = make_lfm()
    wvd, t_w, f_w = compute_wvd(sig, fs=FS)
    cwd_inf, t_c, f_c = compute_cwd(sig, fs=FS, sigma=np.inf)

    assert np.allclose(wvd, cwd_inf, rtol=1e-6, atol=1e-8)
    assert np.array_equal(t_w, t_c) and np.array_equal(f_w, f_c)


def test_cwd_kernel_suppresses_relative_to_wvd():
    """The Choi-Williams kernel must actually damp the plane relative to the raw WVD.

    This is the mechanism the paper's central claim rests on, so it is worth pinning:
    at sigma = 1.0 the distribution must not be identical to the sigma -> inf (WVD) case.
    """
    sig, _ = make_lfm()
    wvd, _, _ = compute_wvd(sig, fs=FS)
    cwd, _, _ = compute_cwd(sig, fs=FS, sigma=1.0)

    assert not np.allclose(cwd, wvd, rtol=1e-3), "sigma=1.0 must differ from the WVD limit"


@pytest.mark.parametrize("fn,kwargs", [
    (compute_cwd, {"time_step": 32, "n_freq": 256}),
    (compute_wvd, {"time_step": 32, "n_freq": 256}),
])
def test_quadratic_transforms_shape_and_finiteness(fn, kwargs):
    sig, _ = make_lfm()
    tf, t_axis, f_axis = fn(sig, fs=FS, **kwargs)

    assert tf.shape == (256, N // 32), f"unexpected shape {tf.shape}"
    assert tf.shape == (f_axis.size, t_axis.size)
    assert np.all(np.isfinite(tf)), "TF output must contain no NaN/Inf"
    assert np.all(tf >= 0), "magnitude output must be non-negative"


def test_stft_shape_finiteness_and_centered_frequency_axis():
    sig, _ = make_lfm()
    f, t, Zxx = compute_stft(sig, fs=FS)

    assert Zxx.shape == (f.size, t.size)
    assert np.all(np.isfinite(Zxx))
    # Complex input -> two-sided, fftshifted to a monotonic [-fs/2, +fs/2] axis.
    assert np.all(np.diff(f) > 0), "frequency axis must be monotonic (fftshifted)"
    assert f[0] < 0 < f[-1], "two-sided spectrum must straddle DC"


def test_stft_localises_a_pure_tone_at_its_frequency():
    """Sanity check that the transform is physically correct, not just well-shaped."""
    freq = 5e6
    f, _, Zxx = compute_stft(make_tone(freq), fs=FS)
    power = np.abs(Zxx).mean(axis=1)
    peak_hz = f[int(np.argmax(power))]

    # Within one frequency bin of the true tone.
    assert abs(peak_hz - freq) <= (FS / 256)


def test_stft_of_a_chirp_has_a_rising_frequency_ridge():
    """An LFM must show a monotonically rising ridge -- the diagonal the models key on."""
    sig, (lo, hi) = make_lfm(b_hz=20e6)
    f, t, Zxx = compute_stft(sig, fs=FS)

    # Restrict to frames inside the pulse, then track the peak frequency per frame.
    inside = (t >= lo / FS) & (t <= hi / FS)
    ridge = f[np.argmax(np.abs(Zxx[:, inside]), axis=0)]

    # Compare the first and last thirds rather than requiring strict monotonicity,
    # which would be brittle to a single noisy frame.
    third = max(1, ridge.size // 3)
    assert ridge[-third:].mean() > ridge[:third].mean(), "chirp ridge must rise over time"


# ------------------------------------------------------- tf_to_image


def test_tf_to_image_contract():
    """Every model consumes exactly this: (224, 224) float32 in [0, 1]."""
    sig, _ = make_lfm()
    cwd, _, _ = compute_cwd(sig, fs=FS)
    img = tf_to_image(cwd)

    assert img.shape == (224, 224)
    assert img.dtype == np.float32
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert np.all(np.isfinite(img))


def test_tf_to_image_is_deterministic_and_peak_normalised():
    sig, _ = make_lfm()
    cwd, _, _ = compute_cwd(sig, fs=FS)

    assert np.array_equal(tf_to_image(cwd), tf_to_image(cwd))
    # Per-sample max-normalisation: scaling the input must not change the image.
    assert np.allclose(tf_to_image(cwd), tf_to_image(cwd * 1000.0), atol=1e-6)


def test_tf_to_image_handles_an_all_zero_input():
    """A silent frame must not produce NaNs (log of zero)."""
    img = tf_to_image(np.zeros((256, 64), dtype=np.float32))
    assert np.all(np.isfinite(img))


# ------------------------------------------------------------ splits


def test_dataset_fingerprint_is_deterministic_and_change_sensitive():
    labels = np.arange(100) % 8
    snr = np.tile(np.arange(-10, 22, 2, dtype=np.float32), 100 // 16 + 1)[:100]

    assert dataset_fingerprint(labels, snr) == dataset_fingerprint(labels, snr)

    tampered = labels.copy()
    tampered[0] = (tampered[0] + 1) % 8
    assert dataset_fingerprint(tampered, snr) != dataset_fingerprint(labels, snr), \
        "fingerprint must detect a changed dataset"
