# Dataset and Data Pipeline

Technical reference for the synthetic radar pulse dataset, its on-disk layout, and the
frozen train/validation/test split used by every experiment.

---

## 1. Overview

| Property | Value |
|---|---|
| Total samples | 40,000 (8 classes × 5,000, perfectly balanced) |
| Signal type | Complex baseband, zero-padded |
| Signal length | 2,048 samples (20.48 µs) |
| Sample rate | 100 MHz |
| Pulse width | Uniform in [1, 20] µs (Frank and Polyphase: [4, 20] µs) |
| Target SNR | 16-point grid, −10 dB to +20 dB in 2 dB steps (one value per sample) |
| Noise | **Not stored.** AWGN is applied at training time (see §5) |
| Reproducibility | Per-sample seed = `master_seed (42) + global_sample_index` |
| File | `data_generation/synthetic_samples/dataset.h5` (~260 MB, gitignored) |

The dataset is generated in MATLAB (`data_generation/matlab/main_generate_dataset.m`) and
consumed from Python via `h5py`.

---

## 2. Signal Classes

Labels are **0-based** and `class_names` is stored in this order.

| Index | Class | Parameters |
|---|---|---|
| 0 | LFM | Up/down chirp, B ∈ [5, 20] MHz |
| 1 | NLFM | Quadratic (60%) + sinusoidal (40%) variants |
| 2 | Barker | B7 / B11 / B13, equal probability, rectangular chip |
| 3 | Frank | N ∈ {4, 6, 8}, f_c = 0 |
| 4 | Polyphase | P1 / P2 / P3 / P4, equal probability, N ∈ {4, 6, 8}, f_c = 0 |
| 5 | Costas | N ∈ {5, 6, 7, 8}, 2 canonical sequences per N, symmetric baseband |
| 6 | CW | Single tone, random f_c, random initial phase |
| 7 | SteppedFH | N ∈ {5, 6, 7, 8}, monotonic up/down frequency staircase |

Frank and Polyphase use a wider minimum pulse width ([4, 20] µs) so that the N² chips stay
adequately oversampled at 100 MHz.

---

## 3. HDF5 Layout

> ⚠️ **MATLAB column-major storage convention.** MATLAB's `save -v7.3` writes HDF5 with axes
> reversed relative to what a row-major (NumPy) reader expects. Every Python reader must
> account for this. The naive `f['signals'][idx]` is **wrong**.

```
dataset.h5
├── /signals          (2, 2048, 40000) float32   [real/imag, time, sample]  <- reversed
├── /labels           (1, 40000)       uint8     [class index 0..7, row vector]
├── /snr_db           (1, 40000)       float32   [target SNR per sample, row vector]
├── /pulse_widths_us  (1, 40000)       float32   [active pulse width, row vector]
├── /class_names      (8,)             object    [class label strings]
└── root attributes (each stored as a 1-element array, not a scalar):
    sample_rate_hz, signal_length, master_seed, generation_date,
    dataset_version, samples_per_class, num_classes,
    snr_db_min, snr_db_max, snr_db_step,
    pulse_width_us_min, pulse_width_us_max, storage_convention
```

### Correct Python read pattern

```python
import h5py
import numpy as np

with h5py.File("data_generation/synthetic_samples/dataset.h5", "r") as f:
    # 1-D vectors: stored as row vectors -> ravel()
    labels = np.asarray(f["labels"][:]).ravel()                    # (40000,) uint8
    snr_db = np.asarray(f["snr_db"][:]).ravel()                    # (40000,) float32
    pulse_widths_us = np.asarray(f["pulse_widths_us"][:]).ravel()  # (40000,) float32
    class_names = [s.decode() if isinstance(s, bytes) else str(s)
                   for s in np.asarray(f["class_names"][:])]

    # Attributes: 1-element arrays -> ravel()[0]
    fs = float(np.asarray(f.attrs["sample_rate_hz"]).ravel()[0])

    # A single complex signal: transpose, then combine real/imag
    ri = np.asarray(f["signals"][:, :, idx]).T                     # (2048, 2)
    signal = (ri[:, 0] + 1j * ri[:, 1]).astype(np.complex64)       # (2048,) complex64
```

---

## 4. Frozen Split

All nine experiments share one frozen split so that representation and architecture
comparisons are apples-to-apples.

| | Samples | Fraction |
|---|---|---|
| Train | 28,000 | 70% |
| Validation | 6,000 | 15% |
| Test | 6,000 | 15% |

- **File:** `configs/splits.npz` (git-tracked; keys `train_idx`, `val_idx`, `test_idx` plus
  provenance metadata: `master_seed`, `dataset_fingerprint`, `generated_at`, `stratification`).
- **Generator:** `scripts/make_splits.py`
- **Stratification:** joint on `(class, SNR)`, so every class and every SNR level is
  proportionally represented in each partition.
- **Seed:** 42.

Per-SNR test counts range from 361 to 385 samples (16 SNR levels × ~375), which is what the
per-SNR accuracy and confidence intervals in `results_summary.md` are computed against.

To regenerate:

```bash
python scripts/make_splits.py --out configs/splits.npz
```

---

## 5. Noise (AWGN) Strategy

AWGN is **not** baked into the stored signals. It is added at training time, inside the data
loader, by `preprocessing/noise/awgn.py`:

- Signal power is measured over the **active region only** (the non-zero pulse), not over the
  zero padding. Using the full frame would bias the achieved SNR (measured deviation of ≈ +3 dB
  at a 0 dB target).
- Noise is complex: real and imaginary parts each drawn from `N(0, noise_power / 2)`, matching
  the MATLAB `add_awgn.m` convention.
- The RNG is passed in explicitly so that DataLoader worker processes stay reproducible.

Keeping noise out of the stored data means the same clean signal can be paired with different
noise realisations across epochs, which prevents the model from overfitting to one fixed noise
pattern.

---

## 6. Runtime Pipeline

Each `__getitem__` call performs:

```
HDF5 read (clean signal)
  -> add AWGN at the sample's target SNR
  -> time-frequency transform (STFT | CWD | WVD)
  -> tf_to_image: dB clip, per-sample normalise, resize to 224x224
  -> float32 tensor (1, 224, 224) + integer label
```

Implemented in `preprocessing/datasets/radar_pulse_dataset.py`. The transform is chosen once at
construction, so the same dataset class serves all three representations.
