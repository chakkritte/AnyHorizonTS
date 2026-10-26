#!/usr/bin/env bash
# Train SparseTSF with its official code and scripts (the checkout made by setup_sparsetsf.sh), five seeds (2023-2027) per horizon, then
# score every checkpoint on the per-horizon test windows (eval_ckpt.py) and join the facts of the training logs (make_scores.py).
#
#   Matched SparseTSF, the main comparison of the paper (group G1, H in {96, 192, 336, 720}; default):
#     STSF_DIR=../SparseTSF bash reproduce/matched_sparsetsf/run_sparsetsf.sh
#   Matched SparseTSF trained at horizons outside the training set of AnyHorizonTS (group G2, every-horizon evaluation):
#     STSF_DIR=../SparseTSF TAG=G2 HORIZONS="24 48 144 480" bash reproduce/matched_sparsetsf/run_sparsetsf.sh
#   Official SparseTSF (no option):
#     STSF_DIR=../SparseTSF TAG=official FLAGS="" bash reproduce/matched_sparsetsf/run_sparsetsf.sh
#
# Environment: STSF_DIR (required), TAG (name of the run; default G1), FLAGS (options of run_longExp.py; default
# "--train_window_horizon 720 --ahts_optimizer_state" = matched: the training windows of the loader built at H_max = 720 and the
# optimizer settings of AnyHorizonTS), HORIZONS (default 96 192 336 720), DATASETS (default etth1 etth2 weather electricity solar
# traffic), HEADS (default linear mlp), ITR (seeds, default 5), EPOCH_TEST_ON (see below; default etth1), PYTHON (default python),
# DEVICE (scoring; default cuda if available).
# Outputs in STSF_DIR: checkpoints_<TAG>/, logs_<TAG>/, result_<TAG>.txt (test() output), scores_<TAG>_eval.csv (eval_ckpt.py),
# scores_<TAG>.csv (with the log facts; for G1 this is the input stsf_matched_G1_per_seed.csv of the statistics, see reproduce/DATA_FORMAT.md).
#
# Conventions of the runs of the paper, kept here so that a seed reproduces the same run:
#   - the per-epoch test-loss pass of train() (printed only) runs on the datasets of EPOCH_TEST_ON (default: ETTh1) and is skipped
#     elsewhere (SKIP_EPOCH_TEST=1). It draws from the global random number generator, so skipping it changes the batch order of the
#     later epochs (an equally valid realization). The day-ahead runs (H = 24 and 144) skipped it on every dataset (EPOCH_TEST_ON="");
#   - test() after training is skipped on Electricity, Solar-Energy and Traffic (SKIP_TEST=1; it keeps every prediction in host
#     memory, about 45-50 GB for Traffic at H = 720); every checkpoint is scored by eval_ckpt.py instead (streaming float64);
#   - Traffic runs with --num_workers 4 (the default of 10 workers exceeds a 24 GB /dev/shm at H >= 480; the results do not change).
# GPU runs of the official code are not bit-reproducible on the larger datasets (Weather and up), so reruns there differ slightly.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
STSF_DIR=$(cd "${STSF_DIR:?set STSF_DIR to the patched SparseTSF checkout (setup_sparsetsf.sh)}" && pwd)
TAG=${TAG:-G1}
if [ -z "${FLAGS+x}" ]; then FLAGS="--train_window_horizon 720 --ahts_optimizer_state"; fi
HORIZONS_LIST=${HORIZONS:-96 192 336 720}
DATASETS=${DATASETS:-etth1 etth2 weather electricity solar traffic}
HEADS=${HEADS:-linear mlp}
EPOCH_TEST_ON=${EPOCH_TEST_ON-etth1}
export PYTHON=${PYTHON:-python} ITR=${ITR:-5}
CKPT=$STSF_DIR/checkpoints_$TAG
LOGD=$STSF_DIR/logs_$TAG
mkdir -p "$LOGD"
cd "$STSF_DIR"
declare -A NAME=([etth1]=ETTh1 [etth2]=ETTh2 [weather]=Weather [electricity]=Electricity [solar]=Solar-Energy [traffic]=Traffic)
# the official test() appends to result.txt; the lines of this run are kept in result_<TAG>.txt for the cross-check of eval_ckpt.py
touch result.txt
n0=$(wc -l < result.txt)
cells=""
for ds in $DATASETS; do
  case " $EPOCH_TEST_ON " in *" $ds "*) unset SKIP_EPOCH_TEST ;; *) export SKIP_EPOCH_TEST=1 ;; esac
  case $ds in electricity|solar|traffic) export SKIP_TEST=1 ;; *) export SKIP_TEST=0 ;; esac
  extra=""
  if [ "$ds" = traffic ]; then extra="--num_workers 4"; fi
  for head in $HEADS; do
    for H in $HORIZONS_LIST; do
      echo "=== $TAG $head / $ds / H=$H ($FLAGS)  $(date '+%F %T')"
      # shellcheck disable=SC2086
      HORIZONS=$H bash "scripts/SparseTSF/$head/$ds.sh" $FLAGS --checkpoints "$CKPT/" $extra > "$LOGD/${head}_${ds}_h$H.log" 2>&1
    done
    cells="$cells;${NAME[$ds]}:$head:$(echo $HORIZONS_LIST | tr ' ' ',')"
  done
done
cells=${cells#;}
tail -n +$((n0 + 1)) result.txt > "result_$TAG.txt"
SEEDS=$(seq 2023 $((2022 + ITR)) | tr '\n' ' ')
# shellcheck disable=SC2086
$PYTHON "$HERE/eval_ckpt.py" --code "$STSF_DIR" --checkpoints "$CKPT" --cells "$cells" --seeds $SEEDS \
  --result_txt "$STSF_DIR/result_$TAG.txt" --out "$STSF_DIR/scores_${TAG}_eval.csv" ${DEVICE:+--device $DEVICE}
$PYTHON "$HERE/make_scores.py" "$STSF_DIR/scores_${TAG}_eval.csv" "$STSF_DIR/scores_$TAG.csv" "$LOGD"/*.log
