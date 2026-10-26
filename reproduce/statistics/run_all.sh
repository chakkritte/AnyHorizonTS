#!/usr/bin/env bash
# Run every statistics script of the paper on the per-seed CSVs of the runs in DATA (file names and formats: reproduce/DATA_FORMAT.md);
# outputs and logs go to reproduce/out (OUT=...). The every-horizon summary reads the scores of run_every_horizon.sh
# (EVERY_HORIZON_RUN=..., default reproduce/out/every_horizon_scores).
# Usage: DATA=/path/to/per_seed_csvs bash reproduce/statistics/run_all.sh
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=${DATA:?set DATA to the folder with the per-seed CSVs of the runs (reproduce/DATA_FORMAT.md)}
OUT=${OUT:-$HERE/../out}
PYTHON=${PYTHON:-python}
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
for s in matched_comparison deficit_tests window_recency families; do
  $PYTHON "$HERE/$s.py" --data "$DATA" --out "$OUT" > "$OUT/$s.log"
done
$PYTHON "$HERE/../every_horizon/summarize_every_horizon.py" --run "${EVERY_HORIZON_RUN:-$HERE/../out/every_horizon_scores}" --out "$OUT/every_horizon" \
  > "$OUT/every_horizon.log"
echo "Logs and CSVs in $OUT. Headlines:"
grep -E '^== (main|official) ' "$OUT/matched_comparison.log"
sed -n '2,8p' "$OUT/deficit_tests.log"
grep -E '^== dedicated' "$OUT/every_horizon.log"
