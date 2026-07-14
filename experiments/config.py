"""Experiment configuration: dataclasses + YAML loader.

A single YAML file fully describes one experiment in the
3-architecture x 3-representation (= 9) matrix. Every training run reads
exactly one config, so the experiment is reproducible from that file
alone (decisions.md, 2026-05-17 benchmark entry: "tek config uc egitime
de uyar").

The config is organised into nested sections:

    experiment : name, output dirs, seed
    data       : dataset path, splits path, tf_repr, noise, image size
    model      : architecture name + kwargs
    optim      : optimizer / scheduler / epochs / early stopping
    loader     : DataLoader knobs (num_workers, batch_size, ...)

Loading merges the YAML on top of these dataclass defaults, so a YAML
only needs to specify what differs from the defaults.

Usage
-----
    from experiments.config import load_config
    cfg = load_config("configs/stft_custom_cnn.yaml")
    print(cfg.model.name, cfg.data.tf_repr)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


# ---------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------
@dataclass
class ExperimentCfg:
    name: str = "stft_custom_cnn"
    seed: int = 42
    # Output roots; per-experiment subdirs are created as <root>/<name>/
    checkpoint_root: str = "experiments/checkpoints"
    results_root: str = "experiments/results"


@dataclass
class DataCfg:
    dataset_path: str = "data_generation/synthetic_samples/dataset.h5"
    splits_path: str = "configs/splits.npz"
    tf_repr: str = "stft"  # {"stft", "cwd", "wvd"}
    add_noise: bool = True  # noisy training & evaluation
    master_seed: int = 42  # per-sample base noise seed
    output_size: Tuple[int, int] = (224, 224)
    db_floor: float = -60.0
    num_classes: int = 8


@dataclass
class ModelCfg:
    name: str = "custom_cnn"  # registry key
    # Architecture-specific kwargs passed to the constructor.
    kwargs: Dict[str, Any] = field(default_factory=lambda: {"dropout": 0.5})


@dataclass
class OptimCfg:
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    warmup_epochs: int = 3  # linear warmup before cosine
    scheduler: str = "cosine"  # {"cosine", "none"}
    min_lr: float = 1e-6  # cosine floor
    # Early stopping on validation loss.
    early_stopping: bool = True
    patience: int = 10
    # Mixed precision (RTX 5050, 4 GB).
    amp: bool = True
    # Gradient clipping (0 disables).
    grad_clip_norm: float = 0.0
    label_smoothing: float = 0.0


@dataclass
class LoaderCfg:
    batch_size: int = 64  # throughput-benchmark optimum
    num_workers: int = 4  # throughput-benchmark optimum
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2


@dataclass
class Config:
    experiment: ExperimentCfg = field(default_factory=ExperimentCfg)
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    loader: LoaderCfg = field(default_factory=LoaderCfg)

    # -- convenience -------------------------------------------------
    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.experiment.checkpoint_root) / self.experiment.name

    @property
    def results_dir(self) -> Path:
        return Path(self.experiment.results_root) / self.experiment.name

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert to a plain dict (for logging / saving)."""
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------
# YAML loading / merging
# ---------------------------------------------------------------------
_SECTION_TYPES = {
    "experiment": ExperimentCfg,
    "data": DataCfg,
    "model": ModelCfg,
    "optim": OptimCfg,
    "loader": LoaderCfg,
}


def _merge_section(section_cls, overrides: Dict[str, Any]):
    """Build a section dataclass from defaults + YAML overrides."""
    if overrides is None:
        return section_cls()
    valid = {f.name for f in dataclasses.fields(section_cls)}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(
            f"Unknown keys for [{section_cls.__name__}]: {sorted(unknown)}. "
            f"Valid keys: {sorted(valid)}."
        )
    # Special-case tuple fields (YAML gives lists).
    clean = dict(overrides)
    if "output_size" in clean and clean["output_size"] is not None:
        clean["output_size"] = tuple(clean["output_size"])
    return section_cls(**clean)


def load_config(path: str | Path) -> Config:
    """Load a Config from a YAML file, merging onto defaults."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Top-level YAML must be a mapping, got {type(raw)}.")

    unknown = set(raw) - set(_SECTION_TYPES)
    if unknown:
        raise ValueError(
            f"Unknown config sections: {sorted(unknown)}. "
            f"Valid sections: {sorted(_SECTION_TYPES)}."
        )

    sections = {
        key: _merge_section(cls, raw.get(key)) for key, cls in _SECTION_TYPES.items()
    }
    return Config(**sections)


def save_config(cfg: Config, path: str | Path) -> None:
    """Dump a Config back to YAML (for run provenance)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False, default_flow_style=False)


if __name__ == "__main__":
    # Smoke check: default config round-trips through YAML.
    import tempfile

    cfg = Config()
    print("Default experiment name:", cfg.experiment.name)
    print("Default tf_repr        :", cfg.data.tf_repr)
    print("Default epochs         :", cfg.optim.epochs)
    print("checkpoint_dir         :", cfg.checkpoint_dir)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
        save_config(cfg, tf.name)
        reloaded = load_config(tf.name)
    assert reloaded.to_dict() == cfg.to_dict()
    print("Round-trip OK")
