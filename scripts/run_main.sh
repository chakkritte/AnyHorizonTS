#!/usr/bin/env bash
# Main results: both heads on the six datasets, full horizon set, five seeds.
# Every horizon is scored on the test windows of a loader built for that horizon (the protocol of the paper's
# tables). The metrics of every seed are appended to result_windows.csv; summarize them with
# tools/summarize_results.py. The shared-window rows of test() go to result_shared.csv. Test metrics are accumulated
# per batch (--stream_metrics), which gives the same MSE/MAE with far less host memory. Logs go to logs/.
# Restrict with HEADS=... DATASETS=...
set -eu
cd "$(dirname "$0")/.."
mkdir -p logs
for head in ${HEADS:-linear mlp}; do
  for ds in ${DATASETS:-etth1 etth2 electricity solar traffic weather}; do
    echo "=== $head / $ds  $(date '+%F %T')"
    bash "scripts/AnyHorizonTS/$head/$ds.sh" --stream_metrics "$@" > "logs/main_${head}_${ds}.log" 2>&1
  done
done
