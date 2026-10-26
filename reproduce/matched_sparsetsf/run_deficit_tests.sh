#!/usr/bin/env bash
# Deficit tests of the paper: the arms behind the table that explains why AnyHorizonTS/MLP trails the official SparseTSF/MLP on ETTh1
# (H = 96, 192) and ETTh2 (H = 96), and the optimizer-settings tests of both heads on ETTh1 and ETTh2. Five seeds per arm.
# Usage (from the repository root): STSF_DIR=../SparseTSF bash reproduce/matched_sparsetsf/run_deficit_tests.sh [ARM ...]
# Arms (default: all; the references are the main AnyHorizonTS runs, scripts/run_main.sh, and the official SparseTSF, run_sparsetsf.sh
# with FLAGS=""):
#   stsf_ahts_settings      SparseTSF, --ahts_optimizer_state; both heads, ETTh1 and ETTh2, H in {96, 192, 336, 720}
#   stsf_hmax_windows       SparseTSF, --train_window_horizon 720 (the 7,201 oldest windows); MLP: ETTh1 96, 192 and ETTh2 96;
#                           Linear: ETTh1 and ETTh2 at 96 and 192
#   stsf_both               SparseTSF, both options; MLP: ETTh1 96, 192 and ETTh2 96
#   stsf_newest_windows     SparseTSF, --train_window_horizon 720 --train_window_drop oldest; MLP: ETTh1 96, 192 and ETTh2 96
#   ahts_official_settings  AnyHorizonTS, --official_first_epoch_lr; both heads, ETTh1 and ETTh2
#   ahts_h96_selection      AnyHorizonTS, --early_stop_horizons 96; MLP, ETTh1 and ETTh2
# SparseTSF arms write STSF_DIR/scores_<arm>_<dataset>_<head>.csv; AnyHorizonTS arms append to results/deficit_<arm>_windows.csv
# (per-horizon rows, as result_windows.csv). PYTHON selects the interpreter.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
ARMS=${*:-stsf_ahts_settings stsf_hmax_windows stsf_both stsf_newest_windows ahts_official_settings ahts_h96_selection}

stsf() {  # stsf <arm> <head> <dataset> <horizons> <flags>
  TAG="$1_$3_$2" HEADS=$2 DATASETS=$3 HORIZONS=$4 FLAGS=$5 bash "$HERE/run_sparsetsf.sh"
}
ahts() {  # ahts <arm> <head> <dataset> <flags>
  mkdir -p "$REPO/results" "$REPO/logs"
  bash "$REPO/scripts/AnyHorizonTS/$2/$3.sh" --stream_metrics $4 --result_file "results/deficit_$1_shared.csv" \
    --window_result_file "results/deficit_$1_windows.csv" > "$REPO/logs/deficit_$1_$2_$3.log" 2>&1
}
for arm in $ARMS; do
  echo "=== $arm  $(date '+%F %T')"
  case $arm in
    stsf_ahts_settings)
      for head in linear mlp; do for ds in etth1 etth2; do stsf $arm $head $ds "96 192 336 720" "--ahts_optimizer_state"; done; done ;;
    stsf_hmax_windows)
      stsf $arm mlp etth1 "96 192" "--train_window_horizon 720"
      stsf $arm mlp etth2 "96" "--train_window_horizon 720"
      for ds in etth1 etth2; do stsf $arm linear $ds "96 192" "--train_window_horizon 720"; done ;;
    stsf_both)
      stsf $arm mlp etth1 "96 192" "--train_window_horizon 720 --ahts_optimizer_state"
      stsf $arm mlp etth2 "96" "--train_window_horizon 720 --ahts_optimizer_state" ;;
    stsf_newest_windows)
      stsf $arm mlp etth1 "96 192" "--train_window_horizon 720 --train_window_drop oldest"
      stsf $arm mlp etth2 "96" "--train_window_horizon 720 --train_window_drop oldest" ;;
    ahts_official_settings)
      for head in linear mlp; do for ds in etth1 etth2; do ahts $arm $head $ds "--official_first_epoch_lr"; done; done ;;
    ahts_h96_selection)
      for ds in etth1 etth2; do ahts $arm mlp $ds "--early_stop_horizons 96"; done ;;
    *) echo "unknown arm $arm" >&2; exit 1 ;;
  esac
done
