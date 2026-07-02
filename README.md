# Radar Pulse Classification

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

**Status:** 🚧 Active development. STFT and CWD experiments complete; WVD
experiments in progress.

## Key Contributions

- Controlled comparison of three time–frequency representations (STFT, CWD, WVD)
  for radar pulse classification, on a shared frozen train/val/test split.
- Benchmark of three architectures (custom CNN, ResNet-50, ViT-Small) under
  identical training and evaluation conditions.
- SNR robustness analysis across −10 dB to +20 dB, including per-class and
  per-SNR accuracy breakdowns.
- Grad-CAM based explainability of what the models attend to in the
  time–frequency plane.

## Dataset

The dataset is generated synthetically in MATLAB and consists of 40,000 complex
baseband signals (8 classes × 5,000 samples), each 2,048 samples long at a
100 MHz sample rate. Each signal is assigned a target SNR drawn from a 16-point
grid spanning −10 dB to +20 dB in 2 dB steps.

Additive white Gaussian noise is **not** baked into the stored signals. Instead,
AWGN is applied at training time inside the data loader, computed against the
active-region signal power, so a given clean signal can be augmented with
different noise realisations across epochs. This avoids overfitting to a fixed
noise pattern and keeps the evaluation methodologically honest.

## Results

Overall accuracy on the noisy test set (all SNRs, −10 dB to +20 dB), on the
shared frozen split:

| Architecture | STFT | CWD | WVD |
|---|---|---|---|
| Custom CNN | 98.22% | 97.57% | 🚧 in progress |
| ResNet-50 | 98.00% | 97.55% | 🚧 in progress |
| ViT-Small | 97.45% | 97.15% | 🚧 in progress |

Per-SNR robustness curves, confusion matrices, and per-class/per-SNR breakdowns
for each experiment are stored under `experiments/results/`.

## Repository Structure

```
radar-pulse-classification/
├── data_generation/   # MATLAB scripts for synthetic radar signal generation
├── preprocessing/     # Time-frequency extraction, dataset, and frozen splits
├── models/            # Custom CNN, ResNet-50, and ViT-Small architectures
├── experiments/       # Training/evaluation pipeline, configs, and results
├── configs/           # Experiment configuration files (YAML)
├── analysis/          # SNR robustness and comparison figures
├── scripts/           # Helper scripts (splits, setup, etc.)
├── tests/             # Unit tests
└── docs/              # Project context and design-decision log
```

Additional `notebooks/` and `paper/` directories will be added during the
write-up phase.

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

Train and evaluate an experiment (run from the repository root):

```bash
# Train — e.g. STFT × custom CNN
python -m experiments.train --config configs/stft_custom_cnn.yaml

# Evaluate the trained checkpoint on the frozen test split
python -m experiments.evaluate --config configs/stft_custom_cnn.yaml
```

More detailed setup, data-generation, and training notes live in [`docs/`](./docs/).

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
