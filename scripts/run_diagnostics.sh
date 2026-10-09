#!/usr/bin/env bash
# Per-horizon training diagnostics (seed 2023): per-epoch loss and gradient norms of every horizon, logged by
# utils/diagnostics.py (logging only). Solar-Energy and, as a control, ETTh1, both heads.
# Output: results/diagnostics/diag_<dataset>_<head>_seed2023.csv, diag_windows.csv (test metrics on the per-horizon
# windows of the paper's tables) and diag_shared.csv (shared windows).
# Test metrics are accumulated per batch (--stream_metrics), which gives the same MSE/MAE with far less host memory.
set -eu
cd "$(dirname "$0")/.."
DIAG=${DIAG_DIR:-results/diagnostics}
mkdir -p logs "$DIAG"
for head in mlp linear; do
  for ds in solar etth1; do
    echo "=== diagnostics $head / $ds  $(date '+%F %T')"
    ITR=1 DES=diag bash "scripts/AnyHorizonTS/$head/$ds.sh" --stream_metrics --diag_dir "$DIAG" \
      --result_file "$DIAG/diag_shared.csv" --window_result_file "$DIAG/diag_windows.csv" \
      > "logs/diag_${head}_${ds}.log" 2>&1
  done
done
