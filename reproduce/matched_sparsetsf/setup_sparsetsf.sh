#!/usr/bin/env bash
# Clone the official SparseTSF code (github.com/lss-1138/SparseTSF, Apache-2.0) at commit b8c2740 and apply the patches of this folder:
#   0001-official-runs.diff                 np.Inf -> np.inf (NumPy 2); SKIP_EPOCH_TEST=1 skips the per-epoch test-loss pass of train()
#                                           (printed only); SKIP_TEST=1 skips test() after training (it keeps every prediction in memory)
#   0002-windows-and-optimizer-options.diff --train_window_horizon H0 (train on the window origins of the H0 loader only) and
#                                           --ahts_optimizer_state (epoch-1 learning rate = --learning_rate, Adam betas (0.9, 0.999))
#   0003-drop-oldest-windows-option.diff    --train_window_drop {recent,oldest} (which origins --train_window_horizon drops)
#   0004-scripts.diff                       scripts/SparseTSF/{linear,mlp}/<dataset>.sh: horizons from $HORIZONS, --itr ${ITR:-5},
#                                           interpreter ${PYTHON:-python}, extra arguments passed on ("$@"), --model_type linear (Linear head)
# Every option is off by default, so without options the patched code trains exactly as the official code.
# Usage: bash reproduce/matched_sparsetsf/setup_sparsetsf.sh [TARGET_DIR]       (default: ./SparseTSF)
#   SPARSETSF_URL=<url or path of a local clone> overrides the source (default: https://github.com/lss-1138/SparseTSF.git)
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
TARGET=${1:-SparseTSF}
URL=${SPARSETSF_URL:-https://github.com/lss-1138/SparseTSF.git}
git clone "$URL" "$TARGET"
cd "$TARGET"
git checkout -q -b anyhorizonts-comparison b8c2740
for p in "$HERE"/patches/000*.diff; do
  git apply --check "$p"
  git apply "$p"
  echo "applied $(basename "$p")"
done
git status --short
echo "Patched SparseTSF in $(pwd). Put the six dataset files in $(pwd)/dataset/ (see dataset/README.md of AnyHorizonTS)."
