# Radar Pulse Classification

[![tests](https://github.com/Adrift21/radar-pulse-classification/actions/workflows/tests.yml/badge.svg)](https://github.com/Adrift21/radar-pulse-classification/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

Deep-learning classification of radar pulse waveforms from their time–frequency
representations, with a focus on robustness under low signal-to-noise ratio (SNR)
conditions.

## Overview

This research project studies the classification of eight radar pulse waveform
families — LFM, NLFM, Barker, Frank, polyphase (P1–P4), Costas, CW, and
stepped/frequency-hopping — using deep neural networks trained on time–frequency
distributions. Three representations are compared: the Short-Time Fourier
Transform (STFT), the Choi–Williams Distribution (CWD), and the Wigner–Ville
Distribution (WVD).

The central research question is **noise robustness**: every model is trained and
evaluated across a wide SNR range (−10 dB to +20 dB), and performance is analysed
as a function of SNR rather than at a single operating point. Three architectures
— a compact custom CNN, ResNet-50, and a Vision Transformer (ViT-Small) — are
benchmarked on an identical, frozen data split to isolate the effect of
representation and architecture.

**Status:** All nine experiments (3 representations × 3 architectures) are complete
and analysed. Manuscript in preparation.

## Key Findings

**1. The Wigner–Ville Distribution collapses at low SNR.** This is the headline
result. Averaged over architectures, WVD trails STFT/CWD by ~9 points overall — but
that average hides the mechanism. The entire gap lives in the low-SNR regime:

| SNR | STFT | CWD | WVD |
|---|---|---|---|
| −10 dB | 84.8% | 84.1% | **24.8%** |
| −6 dB | 97.6% | — | 76.2% |
| 0 dB | 98.1% | — | 96.2% |
| ≥ +6 dB | 99.7% | 99.6% | 99.3% |

At −10 dB the WVD is barely above the 12.5% chance level for 8 classes, while above
+2 dB all three representations are practically indistinguishable. The cause is
cross-term interference: the WVD is a quadratic distribution, so additive noise
generates noise×signal and noise×noise cross-terms that spread across the entire
time–frequency plane and drown the signature. The STFT (linear) and the CWD (whose
Choi–Williams kernel suppresses cross-terms) are far more resilient. This is visible
directly in `analysis/qualitative_tf_lfm.png`.

**2. A compact CNN beats ImageNet-scale backbones.** The 1.77M-parameter custom CNN
achieves the highest accuracy of all nine models while using ~13× fewer parameters
and ~1.5× fewer FLOPs than ResNet-50 or ViT-Small. An ImageNet-scale backbone is
unnecessary for this task.

**3. Representation matters far more than architecture.** Architectures differ by
≤ 1 point within a representation family; representations differ by up to 60 points
at −10 dB.

## Results

Overall accuracy on the noisy test set (all SNRs, −10 dB to +20 dB), on the shared
frozen split:

| Architecture | Params | STFT | CWD | WVD |
|---|---|---|---|---|
| Custom CNN | 1.77M | **98.22%** | 97.57% | 89.57% |
| ResNet-50 | 23.52M | 98.00% | 97.55% | 88.43% |
| ViT-Small | 21.47M | 97.45% | 97.15% | 88.85% |

Full per-SNR breakdowns, confusion matrices, statistical significance testing, and
the complete interpretation are in [`docs/results_summary.md`](./docs/results_summary.md).
Per-experiment artefacts live under `experiments/results/`.

## Dataset

The dataset is generated synthetically in MATLAB and consists of 40,000 complex
baseband signals (8 classes × 5,000 samples), each 2,048 samples long at a
100 MHz sample rate. Each signal is assigned a target SNR drawn from a 16-point
grid spanning −10 dB to +20 dB in 2 dB steps.

Additive white Gaussian noise is **not** baked into the stored signals. Instead,
AWGN is applied at training time inside the data loader, computed against the
active-region signal power, so a given clean signal is paired with a different noise
realisation each epoch. This avoids overfitting to a fixed noise pattern.

All nine experiments share one frozen 70/15/15 split (`configs/splits.npz`,
28,000 / 6,000 / 6,000, jointly stratified on class and SNR), which is what makes the
comparison apples-to-apples.

Full details — including the MATLAB column-major HDF5 storage convention that every
reader must account for — are in [`docs/dataset.md`](./docs/dataset.md).

## Repository Structure

```
radar-pulse-classification/
├── data_generation/   # MATLAB scripts for synthetic radar signal generation
├── preprocessing/     # Time-frequency transforms, AWGN, and the dataset class
├── models/            # Custom CNN, ResNet-50, and ViT-Small architectures
├── experiments/       # Training/evaluation pipeline and per-experiment results
├── configs/           # Experiment configs (YAML) and the frozen split (splits.npz)
├── analysis/          # Comparison figures, significance tests, complexity table
├── scripts/           # Split generation
└── docs/              # Dataset reference, decision log, results summary
```

## Requirements

- Python 3.11 or newer
- PyTorch 2.1+ (a CUDA build is strongly recommended for training)
- MATLAB R2025b or newer (synthetic data generation only)
- All Python dependencies are pinned in `requirements.txt`

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Adrift21/radar-pulse-classification.git
cd radar-pulse-classification

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

Generate the dataset in MATLAB (writes `data_generation/synthetic_samples/dataset.h5`):

```matlab
cd data_generation/matlab
addpath(genpath(pwd))
main_generate_dataset
```

Regenerate the frozen split (optional — `configs/splits.npz` is committed):

```bash
python scripts/make_splits.py --out configs/splits.npz
```

Train and evaluate an experiment (run from the repository root):

```bash
# Train — e.g. STFT × custom CNN
python experiments/train.py --config configs/stft_custom_cnn.yaml

# Evaluate the trained checkpoint on the frozen test split
python experiments/evaluate.py --config configs/stft_custom_cnn.yaml
```

Reproduce the cross-experiment analysis (reads the committed `test_metrics.json` files,
so it runs without any checkpoints):

```bash
python analysis/compare_all_experiments.py     # SNR robustness figures + summary table
python analysis/statistical_significance.py    # Wilson CIs, bootstrap, z-test, McNemar
python analysis/model_complexity.py            # parameter and FLOP counts
python analysis/qualitative_tf_illustration.py # STFT/CWD/WVD vs SNR illustration
```

## Tests

The automated test suite is **dataset-free by design**, so it runs on a bare clone
(and in CI) without the 260 MB `dataset.h5`:

```bash
pytest
```

It pins the correctness claims the methodology rests on: that AWGN achieves the
requested SNR against active-region power, that the WVD is exactly the CWD in the
σ→∞ limit, that the three transforms return the documented shapes and stay finite,
and that `tf_to_image` yields the (224, 224) float32 image in [0, 1] the models consume.

> Note: the `test_*.py` files under `preprocessing/*/tests/` are **not** unit tests
> despite the name — they are manual validation/figure scripts that require the
> dataset. They are excluded from collection; run them directly if you want the figures.

## Documentation

- [`docs/dataset.md`](./docs/dataset.md) — dataset layout, HDF5 conventions, split, noise strategy
- [`docs/results_summary.md`](./docs/results_summary.md) — full results, figures, and interpretation
- [`docs/decisions.md`](./docs/decisions.md) — chronological design-decision log (the source for
  the manuscript's Methods section)

## Citation

If this work contributes to your research, please cite:

```bibtex
@misc{radar_pulse_classification_2026,
  author = {Kaan Emre Evci},
  title  = {Radar Pulse Classification with Deep Learning},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file
for details.

## Acknowledgments

This work is conducted as an independent research project in electronic-warfare
signal processing.
