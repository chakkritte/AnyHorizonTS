#!/usr/bin/env bash
# AnyHorizonTS/MLP on Weather: one model trained on a horizon set and
# evaluated at H = 96, 192, 336 and 720 from the same weights. Five seeds (2023-2027).
# Every horizon is scored on the test windows of a loader built for it and appended to result_windows.csv
# (summarize: python tools/summarize_results.py).
# Hyperparameters as in the paper. Optional overrides (environment): HORIZONS, SEQ_LEN, PERIOD_LEN, ITR, DES,
# PYTHON. Extra command-line arguments are passed to run_longExp.py (e.g. --stream_metrics, --window_result_file).
set -eu
cd "$(dirname "$0")/../../.."
PYTHON=${PYTHON:-python}
HORIZONS=${HORIZONS:-"720 336 192 96"}
SEQ_LEN=${SEQ_LEN:-720}

$PYTHON -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_${SEQ_LEN}_720 \
  --model SparseTSF \
  --data custom \
  --features M \
  --seq_len "$SEQ_LEN" \
  --pred_len 720 \
  --model_type mlp \
  --d_model 128 \
  --period_len "${PERIOD_LEN:-4}" \
  --enc_in 21 \
  --des "${DES:-test}" \
  --train_epochs 30 \
  --patience 5 \
  --itr "${ITR:-5}" \
  --batch_size 256 \
  --learning_rate 0.05 \
  --horizons $HORIZONS "$@"
