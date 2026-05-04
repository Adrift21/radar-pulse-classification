\# Radar Pulse Classification



Deep learning based classification of radar pulse signals using time-frequency representations, with a focus on robustness under low signal-to-noise ratio (SNR) conditions.



\## Overview



This research project investigates the classification of radar pulse waveforms (LFM, NLFM, Barker, Frank, Costas, polyphase codes, etc.) using deep neural networks trained on time-frequency distributions (STFT, Choi-Williams, Wigner-Ville). The study evaluates classification performance across a wide SNR range (-10 dB to +20 dB) and compares CNN, ResNet, and Vision Transformer architectures.



\*\*Status:\*\* 🚧 Active development — research in progress.



\## Key Contributions



\- Comparative evaluation of three time-frequency representations (STFT, CWD, WVD) for radar pulse classification

\- Benchmark of CNN, ResNet-50, and Vision Transformer architectures across the same dataset

\- SNR robustness analysis from -10 dB to +20 dB

\- Grad-CAM based explainability analysis of classification decisions



\## Repository Structure



```

radar-pulse-classification/

├── data\_generation/        # MATLAB scripts for synthetic radar signal generation

├── preprocessing/          # Time-frequency representation extraction (Python)

├── models/                 # CNN, ResNet, ViT model architectures

├── experiments/            # Training scripts, configurations, and results

├── analysis/               # SNR robustness and explainability analysis

├── notebooks/              # Tutorial and visualization notebooks

├── paper/                  # LaTeX manuscript and figures

├── configs/                # Experiment configuration files (YAML)

├── scripts/                # Helper scripts (data download, setup, etc.)

├── tests/                  # Unit tests

└── docs/                   # Detailed documentation

```



\## Requirements



\- Python 3.11+

\- PyTorch 2.6+ (CUDA recommended)

\- MATLAB R2022a+ (for synthetic data generation)

\- See `requirements.txt` for full Python dependencies



\## Quick Start



```bash

\# Clone the repository

git clone https://github.com/Adrift21/radar-pulse-classification.git

cd radar-pulse-classification



\# Create virtual environment

python -m venv venv

.\\venv\\Scripts\\activate  # Windows

\# source venv/bin/activate  # Linux/macOS



\# Install dependencies

pip install -r requirements.txt

```



Detailed setup, data generation, and training instructions can be found in \[`docs/`](./docs/).



\## Citation



If this work contributes to your research, please cite:



```bibtex

@misc{radar\_pulse\_classification\_2026,

&#x20; author       = {Kaan Emre Evci},

&#x20; title        = {Radar Pulse Classification with Deep Learning},

&#x20; year         = {2026},

&#x20; note         = {Manuscript in preparation}

}

```



\## License



This project is licensed under the MIT License — see the \[LICENSE](LICENSE) file for details.



\## Acknowledgments



This work is conducted as an independent research project in the field of electronic warfare signal processing.

