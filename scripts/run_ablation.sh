#!/usr/bin/env bash
# Horizon-set ablation: the three reduced training sets (the full set 720 336 192 96 comes from run_main.sh).
# Every set is evaluated at all four horizons on the per-horizon test windows. Results are appended to
# result_windows.csv (shared-window rows of test() to result_shared.csv); test metrics are accumulated per batch
# (--stream_metrics).
set -eu
cd "$(dirname "$0")/.."
mkdir -p logs
for set in "720 336 96" "720 192 96" "720 96"; do
  for head in ${HEADS:-linear mlp}; do
    for ds in ${DATASETS:-etth1 etth2 electricity solar traffic weather}; do
      tag=$(echo "$set" | tr ' ' '_')
      echo "=== H={$set} $head / $ds  $(date '+%F %T')"
      HORIZONS="$set" bash "scripts/AnyHorizonTS/$head/$ds.sh" --stream_metrics "$@" \
        > "logs/ablation_${tag}_${head}_${ds}.log" 2>&1
    done
  done
done
