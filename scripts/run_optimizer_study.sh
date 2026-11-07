#!/usr/bin/env bash
# Optimizer study: the full horizon set {720, 336, 192, 96} against {720} alone (read at every horizon by truncation) under other
# optimizers than Adam. Datasets ETTh1, ETTh2, Weather; Linear and MLP heads; seeds 2023-2027. Every other hyperparameter is that of
# scripts/AnyHorizonTS/<head>/<ds>.sh; --stream_metrics. The Adam reference is the main runs (scripts/run_main.sh) and the {720} runs
# (scripts/run_ablation.sh).
#
# PART=eopt (shared learning rate):
#   A  AdamW (PyTorch default weight decay 0.01)        --optim adamw                     DES Aadamw_<full|h720>_<head>_<ds>
#   B  Adam + global-norm gradient clipping at 1.0      --clip_grad_norm 1.0              DES Bclip1_<full|h720>_<head>_<ds>
#   C  SGD, momentum 0.9. Learning rate from {0.01, 0.03, 0.1, 0.3, 1, 3}, chosen per dataset and head on seed 2023 of the FULL set by
#      the lowest best-epoch validation MSE (grid DES Cgrid<lr>_<head>_<ds>, tools/pick_lr.py), then used for both sets
#      (DES Csgd_<full|h720>_<head>_<ds>)
#   D  update-norm log of W_max (--update_norm_log), seed 2023, ETTh1, both heads, full set, Adam (script learning rate) and SGD
#      (chosen learning rate)                                                                DES Dadam_<head>_etth1, Dsgd_<head>_etth1
# PART=eoptb ({720} with its own SGD learning rate; needs nothing from PART=eopt):
#   the same grid on seed 2023 of the {720} set (DES Cgridown<lr>_<head>_<ds>); if 3 is chosen, 10 is added and the choice repeated;
#   then seeds 2023-2027 of {720} at the chosen learning rate (DES Csgdown_h720_<head>_<ds>). The full set keeps the runs of
#   PART=eopt at the learning rate chosen for it (Csgd_full_*).
# PART="eopt eoptb" (default) runs both, in that order.
#
# Outputs in OUT (default results/optimizer_study): result_windows.csv / result.csv (A, B, C), grid_*, grid_vali.csv, lr_<head>_<ds>.txt,
# diag_*, update_norms_{adam,sgd}_<head>_etth1.csv (PART=eopt); gridown_*, gridown_vali.csv, lrown_<head>_<ds>.txt, own_* (PART=eoptb).
# Per-job logs in OUT/logs (the learning-rate choice reads them). Statistics: reproduce/statistics/optimizer_study.py
# (input file names: reproduce/DATA_FORMAT.md). The runs of the paper took about 4 h (PART=eopt) and 1.5 h (PART=eoptb) on one RTX 3090.
# Environment: PART, DATASETS (default etth1 etth2 weather), HEADS (default linear mlp), OUT, PYTHON. A failed job is logged
# (OUT/runner.log) and the queue continues. Extra arguments are passed to every run_longExp.py call.
set -u
cd "$(dirname "$0")/.."
export PYTHON=${PYTHON:-python}
PART=${PART:-eopt eoptb}
OUT=${OUT:-results/optimizer_study}
mkdir -p "$OUT/logs"
OUT=$(cd "$OUT" && pwd)
LOG=$OUT/runner.log
GRID="0.01 0.03 0.1 0.3 1 3"
FULL="720 336 192 96"
EXTRA=("$@")

# run_job <des> <head> <ds> <itr> <result-file prefix> <horizons> [options of run_longExp.py]
run_job() {
  local tag=$1 head=$2 ds=$3 itr=$4 res=$5 hs=$6
  shift 6
  echo "  START $tag $(date '+%F %T')" >> "$LOG"
  DES=$tag ITR=$itr HORIZONS="$hs" bash "scripts/AnyHorizonTS/$head/$ds.sh" --stream_metrics "$@" "${EXTRA[@]}" \
    --result_file "$OUT/${res}result.csv" --window_result_file "$OUT/${res}result_windows.csv" > "$OUT/logs/$tag.log" 2>&1
  echo "  DONE  $tag rc=$? tracebacks=$(grep -c 'Traceback (most recent call last)' "$OUT/logs/$tag.log") $(date '+%F %T')" >> "$LOG"
}

for part in $PART; do
  echo "START $part $(date '+%F %T')" >> "$LOG"
  for ds in ${DATASETS:-etth1 etth2 weather}; do
    for head in ${HEADS:-linear mlp}; do
      cell=${head}_${ds}
      if [ "$part" = eopt ]; then
        for lr in $GRID; do
          run_job "Cgrid${lr}_${cell}" "$head" "$ds" 1 grid_ "$FULL" --optim sgd --momentum 0.9 --learning_rate "$lr"
        done
        rm -f "$OUT/lr_${cell}.txt"
        $PYTHON tools/pick_lr.py "$OUT/lr_${cell}.txt" "$OUT/logs/Cgrid{lr}_${cell}.log" $GRID >> "$OUT/grid_vali.csv" 2>> "$LOG"
        lr=$(cat "$OUT/lr_${cell}.txt" 2>/dev/null)
        echo "  CHOSEN $cell lr=${lr:-none}" >> "$LOG"
        for hs in full h720; do
          if [ $hs = full ]; then H="$FULL"; else H="720"; fi
          run_job "Aadamw_${hs}_${cell}" "$head" "$ds" 5 "" "$H" --optim adamw
          run_job "Bclip1_${hs}_${cell}" "$head" "$ds" 5 "" "$H" --clip_grad_norm 1.0
          if [ -n "$lr" ]; then
            run_job "Csgd_${hs}_${cell}" "$head" "$ds" 5 "" "$H" --optim sgd --momentum 0.9 --learning_rate "$lr"
          fi
        done
      else
        for lr in $GRID; do
          run_job "Cgridown${lr}_${cell}" "$head" "$ds" 1 gridown_ "720" --optim sgd --momentum 0.9 --learning_rate "$lr"
        done
        rm -f "$OUT/lrown_${cell}.txt"
        $PYTHON tools/pick_lr.py "$OUT/lrown_${cell}.txt" "$OUT/logs/Cgridown{lr}_${cell}.log" $GRID >> "$OUT/gridown_vali.csv" 2>> "$LOG"
        lr=$(cat "$OUT/lrown_${cell}.txt" 2>/dev/null)
        if [ "$lr" = "3" ]; then
          run_job "Cgridown10_${cell}" "$head" "$ds" 1 gridown_ "720" --optim sgd --momentum 0.9 --learning_rate 10
          $PYTHON tools/pick_lr.py "$OUT/lrown_${cell}.txt" "$OUT/logs/Cgridown{lr}_${cell}.log" $GRID 10 >> "$OUT/gridown_vali.csv" 2>> "$LOG"
          lr=$(cat "$OUT/lrown_${cell}.txt" 2>/dev/null)
        fi
        echo "  CHOSEN own $cell lr=${lr:-none}" >> "$LOG"
        if [ -n "$lr" ]; then
          run_job "Csgdown_h720_${cell}" "$head" "$ds" 5 own_ "720" --optim sgd --momentum 0.9 --learning_rate "$lr"
        fi
      fi
    done
    if [ "$part" = eopt ] && [ "$ds" = etth1 ]; then   # D: update-norm log
      for head in ${HEADS:-linear mlp}; do
        cell=${head}_etth1
        lr=$(cat "$OUT/lr_${cell}.txt" 2>/dev/null)
        run_job "Dadam_${cell}" "$head" etth1 1 diag_ "$FULL" --update_norm_log "$OUT/update_norms_adam_${cell}.csv"
        if [ -n "$lr" ]; then
          run_job "Dsgd_${cell}" "$head" etth1 1 diag_ "$FULL" --optim sgd --momentum 0.9 --learning_rate "$lr" \
            --update_norm_log "$OUT/update_norms_sgd_${cell}.csv"
        fi
      done
    fi
  done
  echo "DONE $part $(date '+%F %T')" >> "$LOG"
done
