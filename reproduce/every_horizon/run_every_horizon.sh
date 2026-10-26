#!/usr/bin/env bash
# Score every saved checkpoint at every horizon h = 1..720 on the per-horizon test windows (score_every_horizon.py), then summarize
# (summarize_every_horizon.py). No model is retrained. Run from the repository root after the training runs:
#   AnyHorizonTS, full horizon set (scripts/run_main.sh) and the model trained at 720 alone (HORIZONS=720, scripts/run_ablation.sh), both
#   in ./checkpoints (the run names differ by horizon set); SparseTSF (reproduce/matched_sparsetsf/run_sparsetsf.sh) with TAG=official
#   (FLAGS=""), TAG=G1 and TAG=G2 (HORIZONS="24 48 144 480") in STSF_DIR/checkpoints_<TAG>.
# Usage: STSF_DIR=../SparseTSF bash reproduce/every_horizon/run_every_horizon.sh
# Environment: AHTS_CKPT (default ./checkpoints), STSF_DIR (required), OUT (default reproduce/out/every_horizon_scores), DATASETS
# (default all six), DEVICE (default cpu: hours, e.g. about 5 h on 4 threads for the AnyHorizonTS, {720} and official arms of all datasets;
# cuda: minutes),
# PYTHON (default python). The outputs of a previous run in OUT are appended to, so start with an empty OUT.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
STSF_DIR=${STSF_DIR:?set STSF_DIR to the patched SparseTSF checkout with checkpoints_official, checkpoints_G1, checkpoints_G2}
AHTS_CKPT=${AHTS_CKPT:-./checkpoints}
OUT=${OUT:-$HERE/../out/every_horizon_scores}
DATASETS=${DATASETS:-ETTh1 ETTh2 Weather Electricity Solar-Energy Traffic}
DEVICE=${DEVICE:-cpu}
PYTHON=${PYTHON:-python}
SC=$HERE/score_every_horizon.py
mkdir -p "$OUT"
for ds in $DATASETS; do
  echo "=== $ds  $(date '+%F %T')"
  $PYTHON "$SC" --kind ahts --checkpoints "$AHTS_CKPT" --set 720_336_192_96 --check 24 50 250 --dataset "$ds" --device "$DEVICE" --out "$OUT/ahts_main.csv"
  $PYTHON "$SC" --kind ahts --checkpoints "$AHTS_CKPT" --set 720 --dataset "$ds" --device "$DEVICE" --out "$OUT/ahts_h720.csv"
  $PYTHON "$SC" --kind stsf --code "$STSF_DIR" --checkpoints "$STSF_DIR/checkpoints_official" --trained 96 192 336 720 --dataset "$ds" \
    --device "$DEVICE" --out "$OUT/stsf_official.csv"
  $PYTHON "$SC" --kind stsf --code "$STSF_DIR" --checkpoints "$STSF_DIR/checkpoints_G1" --trained 96 192 336 720 --dataset "$ds" \
    --device "$DEVICE" --out "$OUT/stsf_matched_g1.csv"
  $PYTHON "$SC" --kind stsf --code "$STSF_DIR" --checkpoints "$STSF_DIR/checkpoints_G2" --trained 24 48 144 480 --dataset "$ds" \
    --device "$DEVICE" --out "$OUT/stsf_matched_g2.csv"
done
$PYTHON "$HERE/summarize_every_horizon.py" --run "$OUT" --out "$HERE/../out/every_horizon"
