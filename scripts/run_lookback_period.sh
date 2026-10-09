#!/usr/bin/env bash
# Look-back sweep (L = 96 and 336; L = 720 is the main run) for both heads on the six datasets, and period sweep
# (P = 4, 6, 12, 24, 48) for the Linear head on ETTh1 and Weather. The per-horizon test windows of the paper's tables
# go to results/lookback_windows.csv and results/period_windows.csv, the shared-window rows of test() to
# results/lookback_shared.csv and results/period_shared.csv. Test metrics are accumulated per batch (--stream_metrics),
# which gives the same MSE/MAE with far less host memory.
set -eu
cd "$(dirname "$0")/.."
mkdir -p logs results
for L in 96 336; do
  for head in linear mlp; do
    for ds in ${DATASETS:-etth1 etth2 electricity solar traffic weather}; do
      echo "=== L=$L $head / $ds  $(date '+%F %T')"
      SEQ_LEN=$L bash "scripts/AnyHorizonTS/$head/$ds.sh" --stream_metrics --result_file results/lookback_shared.csv \
        --window_result_file results/lookback_windows.csv > "logs/lookback_L${L}_${head}_${ds}.log" 2>&1
    done
  done
done
for ds in etth1 weather; do
  for P in 4 6 12 24 48; do
    echo "=== P=$P linear / $ds  $(date '+%F %T')"
    PERIOD_LEN=$P DES=P$P bash "scripts/AnyHorizonTS/linear/$ds.sh" --stream_metrics \
      --result_file results/period_shared.csv --window_result_file results/period_windows.csv \
      > "logs/period_P${P}_${ds}.log" 2>&1
  done
done
