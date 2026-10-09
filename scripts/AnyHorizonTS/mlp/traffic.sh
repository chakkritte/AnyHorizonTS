#!/usr/bin/env bash
# AnyHorizonTS/MLP on Traffic: one model trained on a horizon set and
# evaluated at H = 96, 192, 336 and 720 from the same weights. Five seeds (2023-2027).
# Every horizon is scored on the test windows of a loader built for it and appended to result_windows.csv
# (summarize: python tools/summarize_results.py).
# Hyperparameters as in the paper. Optional overrides (environment): HORIZONS, SEQ_LEN, PERIOD_LEN, ITR, DES,
# PYTHON. Extra command-line arguments are passed to run_longExp.py (e.g. --stream_metrics, --window_result_file).
# Hidden width 512, as in the SparseTSF/MLP script for this dataset (128 on the other four datasets).
set -eu
cd "$(dirname "$0")/../../.."
PYTHON=${PYTHON:-python}
HORIZONS=${HORIZONS:-"720 336 192 96"}
SEQ_LEN=${SEQ_LEN:-720}

$PYTHON -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_${SEQ_LEN}_720 \
  --model SparseTSF \
  --data custom \
  --features M \
  --seq_len "$SEQ_LEN" \
  --pred_len 720 \
  --model_type mlp \
  --d_model 512 \
  --period_len "${PERIOD_LEN:-24}" \
  --enc_in 862 \
  --des "${DES:-test}" \
  --train_epochs 30 \
  --patience 5 \
  --itr "${ITR:-5}" \
  --batch_size 128 \
  --learning_rate 0.01 \
  --num_workers 0 \
  --horizons $HORIZONS "$@"
