#!/usr/bin/env python
"""Multi-seed runner: repeat every experiment with additional random seeds.

WHY. The confidence intervals in docs/results_summary.md §5 quantify test-set sampling
error only, not training-time randomness (weight initialisation and data ordering). The
fine-grained claims -- "STFT > CWD" (~0.5 pt) and "the Custom-CNN is the best architecture"
-- are small enough that seed variance could threaten them. Repeating each experiment with
several seeds turns each single number into a mean +/- std over seeds, which is what makes
those claims defensible. (The headline WVD collapse is far too large to need this.)

PROTOCOL. Each run varies ``experiment.seed`` -- which seeds weight init, data ordering, and
cuDNN -- while ``data.master_seed`` is held FIXED at its base value. So the validation/test
sets (and their AWGN realisations) are IDENTICAL across seeds, and the only thing that varies
is training randomness. That is the standard "repeat with N seeds" protocol and keeps the
comparison clean.

The canonical results already on disk (``experiments/results/<base>``) ARE the seed-42 run
(they were produced with experiment.seed = master_seed = 42). So by default this runner only
adds the EXTRA seeds {43, 44}, saving a third of the compute. aggregate_seeds.py then reads
the canonical dir as seed 42 plus the ``<base>_seed43`` / ``<base>_seed44`` dirs.

Each (experiment, seed) run:
  1. derives a config with name=<base>_seed<k>, seed=<k> (written to configs/generated/)
  2. trains  -> experiments/checkpoints/<base>_seed<k>/
  3. evaluates-> experiments/results/<base>_seed<k>/{test_metrics.json, eval_arrays.npz, ...}

RESUMABLE. A run whose test_metrics.json already exists is skipped, so the runner can be
stopped and restarted freely across the many hours/days it takes on a single GPU.

Usage (from the repo root):
    python experiments/run_multiseed.py                      # seeds 43,44 x all 9 experiments
    python experiments/run_multiseed.py --seeds 43 44 45     # more seeds
    python experiments/run_multiseed.py --experiments stft_custom_cnn cwd_vit
    python experiments/run_multiseed.py --dry-run            # list what would run
    python experiments/run_multiseed.py --epochs 1 --seeds 99 --experiments stft_custom_cnn
                                                             # quick end-to-end smoke test
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from experiments.config import load_config, save_config  # noqa: E402

# The 9-experiment matrix, in a fixed order (representation x architecture).
EXPERIMENTS = [
    "stft_custom_cnn", "stft_resnet50", "stft_vit",
    "cwd_custom_cnn", "cwd_resnet50", "cwd_vit",
    "wvd_custom_cnn", "wvd_resnet50", "wvd_vit",
]
# seed 42 == the canonical results/<base> runs already on disk.
DEFAULT_EXTRA_SEEDS = [43, 44]

CONFIG_DIR = REPO / "configs"
GEN_DIR = CONFIG_DIR / "generated"
RESULTS_ROOT = REPO / "experiments" / "results"


def stamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def derived_config_path(base: str, seed: int) -> Path:
    return GEN_DIR / f"{base}_seed{seed}.yaml"


def make_derived_config(base: str, seed: int) -> Path:
    """Write a per-seed config derived from configs/<base>.yaml. Returns its path."""
    base_path = CONFIG_DIR / f"{base}.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")
    cfg = load_config(base_path)
    cfg.experiment.name = f"{base}_seed{seed}"
    cfg.experiment.seed = seed
    # master_seed is intentionally left at its base value so val/test stay identical.
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    out = derived_config_path(base, seed)
    save_config(cfg, out)
    return out


def is_done(base: str, seed: int) -> bool:
    return (RESULTS_ROOT / f"{base}_seed{seed}" / "test_metrics.json").exists()


def run(cmd: list[str], log_path: Path) -> int:
    """Run a subprocess, teeing stdout+stderr to a log file. Returns the exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {"PYTHONPATH": str(REPO)}
    import os
    full_env = {**os.environ, **env}
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n===== [{stamp()}] $ {' '.join(cmd)} =====\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(REPO), env=full_env,
                              stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_EXTRA_SEEDS,
                   help=f"Seeds to run (default: {DEFAULT_EXTRA_SEEDS}; seed 42 is the "
                        f"existing canonical results).")
    p.add_argument("--experiments", nargs="+", default=EXPERIMENTS,
                   help="Subset of experiments to run (default: all 9).")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override epoch count (for a quick smoke test).")
    p.add_argument("--dry-run", action="store_true",
                   help="List what would run, without training.")
    args = p.parse_args()

    py = sys.executable
    log_dir = REPO / "experiments" / "logs"
    plan = [(e, s) for s in args.seeds for e in args.experiments]

    print(f"[{stamp()}] Multi-seed plan: {len(plan)} runs "
          f"({len(args.experiments)} experiments x {len(args.seeds)} seeds)")
    todo = [(e, s) for e, s in plan if not is_done(e, s)]
    skipped = len(plan) - len(todo)
    print(f"  already complete (skipping): {skipped}   |   to run: {len(todo)}")

    if args.dry_run:
        for e, s in plan:
            print(f"    {'DONE ' if is_done(e, s) else 'RUN  '} {e}_seed{s}")
        return 0

    for i, (base, seed) in enumerate(todo, 1):
        name = f"{base}_seed{seed}"
        print(f"\n[{stamp()}] ({i}/{len(todo)}) === {name} ===")
        cfg_path = make_derived_config(base, seed)
        log = log_dir / f"{name}.log"

        train_cmd = [py, "experiments/train.py", "--config", str(cfg_path)]
        if args.epochs is not None:
            train_cmd += ["--epochs", str(args.epochs)]
        rc = run(train_cmd, log)
        if rc != 0:
            print(f"  WARNING: training failed (exit {rc}); see {log}. Skipping eval.")
            continue

        rc = run([py, "experiments/evaluate.py", "--config", str(cfg_path)], log)
        if rc != 0:
            print(f"  WARNING: evaluation failed (exit {rc}); see {log}.")
            continue
        print(f"  done -> experiments/results/{name}/")

    print(f"\n[{stamp()}] Multi-seed runner finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
