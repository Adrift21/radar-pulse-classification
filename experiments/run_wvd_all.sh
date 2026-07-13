#!/usr/bin/env bash
# Chained runner for the 3 full WVD trainings (Module C, 7th-9th of 9 experiments).
# Runs sequentially so the single RTX 5050 GPU is not oversubscribed.
# Each run writes its own results/checkpoints; logs go to experiments/logs/<name>.log.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=.venv/Scripts/python.exe
STAMP() { date '+%Y-%m-%d %H:%M:%S'; }

for cfg in wvd_custom_cnn wvd_resnet50 wvd_vit; do
  log="experiments/logs/${cfg}.log"
  echo "===== [$(STAMP)] START ${cfg} =====" | tee -a "experiments/logs/wvd_all.log"
  "$PY" experiments/train.py --config "configs/${cfg}.yaml" > "$log" 2>&1
  rc=$?
  echo "===== [$(STAMP)] TRAIN END ${cfg} (exit ${rc}) =====" | tee -a "experiments/logs/wvd_all.log"
  if [ $rc -ne 0 ]; then
    echo "WARNING: ${cfg} train exited with ${rc}; skipping eval, continuing to next." | tee -a "experiments/logs/wvd_all.log"
    continue
  fi
  # Evaluate immediately so test_metrics.json + figures are never left missing.
  "$PY" experiments/evaluate.py --config "configs/${cfg}.yaml" >> "$log" 2>&1
  erc=$?
  echo "===== [$(STAMP)] EVAL END ${cfg} (exit ${erc}) =====" | tee -a "experiments/logs/wvd_all.log"
  if [ $erc -ne 0 ]; then
    echo "WARNING: ${cfg} eval exited with ${erc}; continuing to next." | tee -a "experiments/logs/wvd_all.log"
  fi
done
echo "===== [$(STAMP)] ALL WVD RUNS DONE =====" | tee -a "experiments/logs/wvd_all.log"
