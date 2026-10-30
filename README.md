# AnyHorizonTS

**A Unified Multi-Horizon Learning Framework for Time Series Forecasting**

> 📌 **Status:** this repository accompanies a paper currently under review.

## Overview

Long-term forecasting benchmarks are evaluated at several horizons (96, 192, 336, 720), and the usual pipeline trains
a separate model for each one. AnyHorizonTS trains a single model that serves any horizon within a predefined range at
inference time, with no retraining and no horizon-specific checkpoints.

The backbone is SparseTSF (Linear or MLP head). Its output projection is kept at the largest horizon $`H_{\max}`$ as one
matrix $`\mathbf{W}_{\max}`$, and a horizon $H$ uses the first rows of that matrix. Training accumulates the gradients
of all horizons of the training set in every step (multi-horizon gradient accumulation), so one set of weights is
optimized for every horizon. Because a shorter horizon reads a row prefix of $`\mathbf{W}_{\max}`$, one forward pass at
$`H_{\max}`$ returns the forecasts of every shorter horizon at once.

## Repository layout

```
run_longExp.py                 entry point: trains one model on a horizon set, then evaluates it at H = 96, 192, 336, 720
exp/exp_main.py                training loop with multi-horizon gradient accumulation, evaluation, result CSVs
models/SparseTSF.py            SparseTSF backbone with dynamic (sliceable) layers (models/modules/dynamic_ops.py)
data_provider/, utils/         data loaders, metrics, learning-rate schedule, early stopping, training diagnostics
scripts/AnyHorizonTS/{linear,mlp}/<dataset>.sh   one script per head and dataset, with the hyperparameters of the paper
scripts/run_main.sh            main results: both heads, six datasets, five seeds
scripts/run_ablation.sh        training horizon-set ablation
scripts/run_lookback_period.sh look-back (L) and period (P) sweeps
scripts/run_diagnostics.sh     per-horizon loss and gradient-norm logging during training
tools/summarize_results.py     mean and standard deviation over seeds per dataset, head and horizon (result_windows.csv)
tools/benchmark_efficiency.py  parameters, MACs, peak memory and time per iteration
reproduce/                     matched SparseTSF patches, deficit tests, every-horizon scorer, statistics, per-seed scores
dataset/                       put the benchmark files here (see dataset/README.md)
```

Only the SparseTSF backbone is included. The horizon-specific SparseTSF models that AnyHorizonTS is compared with in
the paper are trained with the official SparseTSF code ([lss-1138/SparseTSF](https://github.com/lss-1138/SparseTSF),
commit b8c2740) and its scripts, five seeds per horizon (see [Official SparseTSF baseline](#official-sparsetsf-baseline)).

## Setup

```bash
conda create -n anyhorizonts python=3.12
conda activate anyhorizonts
pip install -r requirements.txt   # tested with PyTorch 2.12.0 (CUDA 13.0)
```

Download the datasets as described in [dataset/README.md](dataset/README.md): ETTh1, ETTh2, Electricity, Traffic and
Weather from the Autoformer Google Drive, and Solar-Energy from the LSTNet data repository.

## Quick start

```bash
bash scripts/AnyHorizonTS/linear/etth1.sh      # one model, horizon set {720, 336, 192, 96}, seeds 2023-2027
python tools/summarize_results.py              # mean and s.d. over seeds per dataset, head and horizon
```

Each seed trains one model at $`H_{\max} = 720`$ and evaluates the same weights at 96, 192, 336 and 720. Every horizon
is scored on the test windows of a data loader built for that horizon, the protocol of the paper's tables and of the
single-horizon baselines (see *Test windows* below). After training, each seed appends one row per horizon and window
set (`setting, H, windows, n_windows, mse, mae`) to `result_windows.csv`, and the summary reads its `per_horizon` rows.

## Reproducing the experiments

| Experiment | Command |
|---|---|
| Main results and seed standard deviations | `bash scripts/run_main.sh`, then `python tools/summarize_results.py` |
| Horizon-specific SparseTSF baselines | official code at commit b8c2740, see [Official SparseTSF baseline](#official-sparsetsf-baseline) |
| Training horizon-set ablation | `bash scripts/run_ablation.sh` (the reduced sets and {720}; the full set comes from `run_main.sh`) |
| Matched SparseTSF, deficit tests, every-horizon evaluation, Holm and equivalence | see [Matched baseline, deficit tests and statistics](#matched-baseline-deficit-tests-and-statistics) |
| Look-back and period sweeps | `bash scripts/run_lookback_period.sh` (writes `results/lookback_windows.csv`, `results/period_windows.csv`) |
| Per-horizon training diagnostics | `bash scripts/run_diagnostics.sh` (per-epoch losses and gradient norms in `results/diagnostics/`) |
| Efficiency (parameters, MACs, memory, time) | `python tools/benchmark_efficiency.py` |

Summaries: `python tools/summarize_results.py` prints the mean and sample standard deviation over seeds of the
`per_horizon` rows of `result_windows.csv`, per dataset, head, horizon set and horizon. Pass another file (for example
`results/lookback_windows.csv`), `--hset 720_336_192_96` to select one training horizon set, `--decimals 3` for the
three decimals of the tables, `--csv FILE` to save the table, or `--windows shared` for the shared windows (see
*Test windows* below).

### Official SparseTSF baseline

The horizon-specific baselines of the paper (SparseTSF/Linear and SparseTSF/MLP, one model per horizon) are trained
with the official code at commit b8c2740 and its scripts, with two changes: five seeds (`--itr 5`, seeds 2023 to 2027;
the scripts have `--itr 1`) and, for the Linear head, `--model_type linear`. At b8c2740, `run_longExp.py` defaults to
`--model_type mlp` and `scripts/SparseTSF/linear/*.sh` do not set it, so the unmodified Linear scripts train the MLP
head. The MLP scripts set `--model_type 'mlp'` and the hidden width (`--d_model` 128, and 512 on Traffic and
Solar-Energy) themselves.

```bash
git clone https://github.com/lss-1138/SparseTSF.git && cd SparseTSF
git checkout b8c2740                           # put the datasets in ./dataset/, as for AnyHorizonTS
sed -i 's/--itr 1/--itr 5/' scripts/SparseTSF/linear/*.sh scripts/SparseTSF/mlp/*.sh
sed -i 's/--model \$model_name/--model $model_name --model_type linear/' scripts/SparseTSF/linear/*.sh
bash scripts/SparseTSF/linear/etth1.sh         # SparseTSF/Linear; likewise etth2, electricity, traffic, weather, solar
bash scripts/SparseTSF/mlp/etth1.sh            # SparseTSF/MLP; likewise the other five datasets
```

Each script trains one model per horizon (`--pred_len` 96, 192, 336, 720), five seeds each. The official `test()`
builds the test loader for that `--pred_len` ($N - L - H + 1$ windows, `drop_last=False`) and scores all of its
windows, so every horizon is scored on the windows of a loader built for that horizon. These are the `per_horizon`
windows of `result_windows.csv` (see *Test windows* below), so AnyHorizonTS and the baselines are compared on the same
forecast origins. `test()` appends `mse` and `mae` of every seed to `result.txt`, and the baseline entries of the
paper are the means over the five seeds of each horizon. The official `test()` keeps all predictions in host memory
(Traffic at $H = 720$ needs about 45 to 50 GB).

The official code also constructs `OneCycleLR(max_lr=learning_rate)` and never steps it under `--lradj type3`. The
constructor alone runs epoch 1 at `learning_rate / 25` and sets Adam's beta1 to 0.95 for the whole run. AnyHorizonTS
keeps the full rate and beta1 = 0.9; `--official_first_epoch_lr` (see *Notes*) reproduces the official state.

The five-seed baseline of the paper was run with the small patch `reproduce/matched_sparsetsf/patches/0001-official-runs.diff`
(NumPy 2, and switches that skip the per-epoch test-loss pass and the memory-hungry `test()`), whose conventions affect the
batch order of a seed; `reproduce/matched_sparsetsf/run_sparsetsf.sh` with `FLAGS=""` reruns it with the conventions of the paper (see
[Matched baseline, deficit tests and statistics](#matched-baseline-deficit-tests-and-statistics)).

### Notes

- **Seeds and batch size.** `--itr 5` trains seeds 2023 to 2027, each with the `--batch_size` of the script (256, or
  128 for Electricity and Traffic and for the MLP head on ETTh2). Evaluation uses `--test_batch_size` (64), which
  does not affect the metrics.
- **Memory.** `test()` at $H = 720$ on Solar-Energy and Traffic stores all predictions in host memory (tens of
  GB). `--stream_metrics` accumulates MSE, MAE and RSE per batch in float64 instead and gives the same values up
  to floating-point rounding; `scripts/run_*.sh` pass it. The per-horizon scoring always streams.
- **Configuration matched to SparseTSF.** The Linear head has no output bias, so it has the 925 parameters of
  SparseTSF/Linear at $L = H = 720$ (`--linear_bias 1` restores the layer of the June 2026 runs). The optimizer is
  Adam, as in SparseTSF (`--optim adamw` restores AdamW). The MLP head uses a hidden width of 512 on Traffic and
  Solar-Energy, as the SparseTSF scripts do, and 128 on the other datasets.
- **Test windows.** Every horizon $H$ is scored on the windows of a data loader built for that horizon
  ($N - L - H + 1$ origins), as the single-horizon baselines are. These `per_horizon` rows of `result_windows.csv`
  (`--window_result_file`, written by default, `none` switches it off) are the numbers of every table of the paper.
  The same file holds the `shared` rows: every horizon on the windows of a loader built once at $`H_{\max}`$
  ($`N - L - H_{\max} + 1`$ origins, fewer windows below $H = 720$). `test()` scores the shared windows as well and
  appends `Setting, MSE, MAE, RSE` rows to `result_shared.csv` (`--result_file`). These rows are secondary and do not
  match the tables (for example MSE 0.284 instead of 0.341 for the Linear head on ETTh2 at $H = 192$). The windows
  are cut from the test partition in chunks of `--test_batch_size` and the errors are accumulated in float64.
- **Horizon set.** `--horizons` is a subset of `720 336 192 96 24` (the data loader provides targets for these
  horizons); the model is built at `--pred_len 720`. The old option names `--sandynamic` and `--AnyTimeTS` are
  accepted.
- **Diagnostics.** `--diag_dir DIR` writes per-epoch, per-horizon losses and gradient norms to `DIR`. It only reads
  the gradient buffers, so the optimization is unchanged.
- **Evaluation only.** `--is_training 0` loads the checkpoint of each seed from `--checkpoints` and runs the same
  evaluations as after training. `--use_gpu 0` runs on the CPU.
- **Opt-in diagnostics (all off by default; with them off the code reproduces the paper runs bit for bit).**
  - `--official_first_epoch_lr` constructs the scheduler of the official SparseTSF code, `OneCycleLR(max_lr=--learning_rate)`,
    instead of `CosineAnnealingLR`. Neither is stepped with `--lradj type3`, but the `OneCycleLR` constructor sets the
    learning rate of epoch 1 to `--learning_rate / 25` and Adam's beta1 to 0.95 for the whole run; from epoch 2 on the
    `type3` schedule is the same. The log prints the epoch-1 learning rate and the Adam betas.
  - `--early_stop_on_all_horizons` monitors the mean over `--horizons` of the per-horizon validation MSE (each on the first
    H steps of the windows of the H_max validation loader) for early stopping and checkpoint selection, instead of the
    H_max validation loss alone. Training is unchanged; the log prints the four per-horizon validation MSEs of every epoch.
  - `--early_stop_horizons H [H ...]` monitors the mean over the listed horizons (a subset of `--horizons`) of the same
    per-horizon validation MSEs; `--early_stop_horizons 96` selects the checkpoint on the H = 96 validation loss alone (one of
    the deficit tests of the paper), and the four horizons together equal `--early_stop_on_all_horizons`. Same validation pass,
    training unchanged.
  - `--score_horizons H [H ...]` (with `--is_training 0`) scores saved checkpoints at other horizons than 96, 192, 336 and
    720, each a multiple of `--period_len` and at most `--pred_len`, on the per-horizon windows of `--window_result_file`.
    Example, day-ahead on hourly data: `--is_training 0 --score_horizons 24 96 720 --checkpoints <dir with the trained
    settings> --window_result_file day_ahead_windows.csv` plus the options of the training script of the dataset and head.
  - `--single_pass` trains with one forward and backward pass at H_max per batch instead of one pass per horizon, with the
    loss sum over `--horizons` of the MSE on the first H steps of that pass (the forecast at H is a prefix of the forecast at
    H_max). This gives the same gradient as the per-horizon passes up to floating-point rounding (about 1e-14 relative in
    float64, not bit-identical in float32) and makes a Traffic epoch about 1.65 to 1.70 times faster (epoch-time protocol of
    the paper: 55.9 s to 33.8 s for the Linear head, 62.4 s to 36.6 s for the MLP head). Validation and early stopping are
    unchanged; it cannot be combined with `--diag_dir`.
- **Efficiency.** `tools/benchmark_efficiency.py` counts parameters and MACs and times synthetic GPU-resident
  batches. The epoch times of the paper are the `Epoch: k cost time` lines of actual training runs, which cover the
  training batches of an epoch (data loading included, validation excluded).

## Matched baseline, deficit tests and statistics

The folder `reproduce/` holds what the comparisons of the paper need beyond the training scripts above: patches to the
official SparseTSF code for the matched baseline and the deficit tests, the every-horizon scorer, and the statistics. No scores
or other results are included: the statistics run on the per-seed scores that the runs below write (file names and formats in
`reproduce/DATA_FORMAT.md`).

The results of the runs of the paper are a separate download, [AnyHorizonTS_results.zip](https://drive.google.com/file/d/1H85rWnB0lqQBfkiy181H_JUcZycN246P/view?usp=sharing) (103 MB): the per-seed
scores, the scores at every horizon, the checkpoints and the training logs, with a README that maps each file to the tables and
figures of the paper.

```
reproduce/matched_sparsetsf/  patches/000[1-4]-*.diff (against SparseTSF b8c2740), setup_sparsetsf.sh, run_sparsetsf.sh,
                              run_deficit_tests.sh, eval_ckpt.py (per-horizon test windows), make_scores.py (facts of the logs)
reproduce/every_horizon/      score_every_horizon.py (every checkpoint at every h = 1..720), summarize_every_horizon.py,
                              run_every_horizon.sh
reproduce/statistics/         stats_lib.py (Welch, Holm, TOST, smallest margin, detectable difference), one script per
                              comparison, run_all.sh
reproduce/DATA_FORMAT.md      file names and formats of the per-seed scores that the statistics read
```

### Statistics

```bash
DATA=/path/to/per_seed_csvs bash reproduce/statistics/run_all.sh   # about a minute on a CPU; logs and CSVs in reproduce/out/
# on the scores of the paper (results package above):
DATA=/path/to/AnyHorizonTS_results/1_per_seed_scores \
EVERY_HORIZON_RUN=/path/to/AnyHorizonTS_results/2_every_horizon_scores bash reproduce/statistics/run_all.sh
```

Every comparison is a family of cells (dataset, head, horizon) with five seeds per arm. In each cell the relative MSE
difference is d = (a - b) / b (positive: the tested arm a is worse than the reference b), and the five per-seed MSEs are
compared with a two-sided Welch's t-test (p < 0.05). The Holm correction is applied within each family. Equivalence is
tested with two one-sided Welch tests at a margin of ±2% of the reference MSE, that is, the 90% Welch interval of the
difference must lie within ±2%; the smallest margin at which each cell would be equivalent and the difference detectable
with 80% power are reported as well. The margin was chosen after the main results were known. `stats_lib.py` implements
these tests; each script below takes `--data DIR` (the folder of the per-seed CSVs, required) and `--out DIR` (default
`reproduce/out`).

| Script | Comparison in the paper |
|---|---|
| `statistics/matched_comparison.py` | relative MSE difference of AnyHorizonTS from matched SparseTSF, official SparseTSF and a truncated single-horizon model (one model trained at 720 alone), 48 cells per column pair |
| `statistics/deficit_tests.py` | table of the deficit tests on ETTh1 (H = 96, 192) and ETTh2 (H = 96), Holm over the three cells of each row; also every cell of the optimizer-settings, window and checkpoint-selection tests of both heads |
| `statistics/window_recency.py` | recency against number of the omitted training windows |
| `statistics/families.py` | Holm and equivalence for the comparison with the official SparseTSF in the main results table, the horizon-set ablation, the look-back and period sweeps and the day-ahead horizons |
| `every_horizon/summarize_every_horizon.py` | every-horizon evaluation |

### Matched SparseTSF (the main comparison)

The matched baseline is the official SparseTSF code at commit b8c2740, trained with its own scripts and hyperparameters,
five seeds (2023-2027) per horizon, with two options added by the patches:

- `--train_window_horizon 720` trains on the window origins of the loader built at H_max = 720 only, the windows that
  AnyHorizonTS trains on (each window keeps its own H-step target; validation and test sets are unchanged);
- `--ahts_optimizer_state` restores the full learning rate in epoch 1 and Adam's betas (0.9, 0.999) after the official,
  never-stepped `OneCycleLR` is constructed, which are the optimizer settings of AnyHorizonTS.

The matched SparseTSF then differs from AnyHorizonTS only in the training procedure: four models, each trained and stopped on
its own horizon, against one model trained on the summed losses and stopped on the validation loss at H_max. Every option of
the patches is off by default, so without options the patched code trains exactly as the official code.

```bash
bash reproduce/matched_sparsetsf/setup_sparsetsf.sh ../SparseTSF   # git clone, checkout b8c2740, apply patches 0001-0004 in order
# the same by hand (apply the patches one at a time, in order):
#   git clone https://github.com/lss-1138/SparseTSF.git ../SparseTSF && cd ../SparseTSF && git checkout b8c2740
#   for p in /path/to/AnyHorizonTS/reproduce/matched_sparsetsf/patches/000*.diff; do git apply "$p"; done
# then put the six dataset files in ../SparseTSF/dataset/
STSF_DIR=../SparseTSF bash reproduce/matched_sparsetsf/run_sparsetsf.sh                                  # G1: H in {96, 192, 336, 720}
STSF_DIR=../SparseTSF TAG=G2 HORIZONS="24 48 144 480" bash reproduce/matched_sparsetsf/run_sparsetsf.sh  # G2: dedicated models at h
STSF_DIR=../SparseTSF TAG=official FLAGS="" bash reproduce/matched_sparsetsf/run_sparsetsf.sh            # official SparseTSF
```

| Patch | Changes |
|---|---|
| `0001-official-runs.diff` | `np.Inf` to `np.inf` (NumPy 2); `SKIP_EPOCH_TEST=1` skips the per-epoch test-loss pass of `train()` (printed only); `SKIP_TEST=1` skips `test()` after training |
| `0002-windows-and-optimizer-options.diff` | `--train_window_horizon` and `--ahts_optimizer_state` |
| `0003-drop-oldest-windows-option.diff` | `--train_window_drop {recent,oldest}`: which origins `--train_window_horizon` drops (the recency test) |
| `0004-scripts.diff` | `scripts/SparseTSF/{linear,mlp}/<dataset>.sh`: horizons from `HORIZONS`, `--itr ${ITR:-5}`, interpreter `${PYTHON:-python}`, extra arguments passed on, `--model_type linear` for the Linear head |

`run_sparsetsf.sh` trains one model per dataset, head, horizon and seed, scores every checkpoint on the per-horizon test windows
(`eval_ckpt.py`: N - L - H + 1 origins, float64 sums) and joins the restored epoch, the window counts and the epoch-1 learning
rate read from the training logs (`make_scores.py`); the G1 output `scores_G1.csv` is the input
`stsf_matched_G1_per_seed.csv` of the statistics. It keeps the conventions of the runs of the paper, which matter for
reproducing a seed: the per-epoch test-loss pass runs on ETTh1 and is skipped on the other datasets (it draws from the global
random number generator, so skipping it changes the batch order of the later epochs); `test()` is skipped on Electricity,
Solar-Energy and Traffic, where it keeps every prediction in host memory; Traffic uses 4 DataLoader workers. On one RTX 3090
the G1 runs took about 20 h and the G2 runs about 18 h. GPU training of the official code is not bit-reproducible on Weather
and the larger datasets, so reruns there differ slightly; one matched run of the paper (Weather/MLP, H = 192, seed 2025)
became unstable and is kept in every number.

### Deficit tests

AnyHorizonTS/MLP trails the official SparseTSF/MLP significantly on ETTh1 at H = 96 and 192 and on ETTh2 at H = 96. Each
deficit test changes the optimizer settings, the training windows or the checkpoint selection of one of the two models:

```bash
STSF_DIR=../SparseTSF bash reproduce/matched_sparsetsf/run_deficit_tests.sh        # all arms; or name some arms
```

| Row of the deficit-test table | Arm (`run_deficit_tests.sh`) | Option |
|---|---|---|
| Official comparison | main runs (`scripts/run_main.sh`) against the official SparseTSF | none |
| AHTS, official settings | `ahts_official_settings` | AnyHorizonTS `--official_first_epoch_lr` |
| STSF, AHTS settings | `stsf_ahts_settings` | SparseTSF `--ahts_optimizer_state` |
| STSF, H_max windows | `stsf_hmax_windows` | SparseTSF `--train_window_horizon 720` |
| STSF, both of the above | `stsf_both` | SparseTSF `--train_window_horizon 720 --ahts_optimizer_state` |
| AHTS, H = 96 selection | `ahts_h96_selection` | AnyHorizonTS `--early_stop_horizons 96` |
| STSF, newest windows | `stsf_newest_windows` | SparseTSF `--train_window_horizon 720 --train_window_drop oldest` |

The optimizer-settings arms cover both heads on ETTh1 and ETTh2 at all four horizons; the window arms cover the three MLP
cells (and the Linear head at H = 96 and 192 for `stsf_hmax_windows`). `statistics/deficit_tests.py` and
`statistics/window_recency.py` summarize them from `deficit_tests_per_seed.csv`.

### Every-horizon evaluation

Every checkpoint is scored without retraining at every horizon h = 1, ..., 720, each on the test windows of a loader built
for h. A model is run once per test origin at its native output length, and the forecast at h is the first h steps: for
AnyHorizonTS this is the prefix identity, and for h that is not a multiple of P it is the serving rule (truncate the next
multiple of P; `--check 24 50 250` runs the model at that multiple directly and compares). Single-horizon SparseTSF models
serve h by truncating the shortest trained horizon of at least h. The references are the four matched and the four official
SparseTSF models, one model trained at H_max alone and truncated, and matched SparseTSF trained at h in {24, 48, 144, 480}
(G2).

```bash
STSF_DIR=../SparseTSF bash reproduce/every_horizon/run_every_horizon.sh   # hours on a CPU; DEVICE=cuda takes minutes
```

It needs the checkpoints of `scripts/run_main.sh` and of the {720} runs in `./checkpoints`, and those of the official, G1 and
G2 runs of `run_sparsetsf.sh`; the summary (`curves.csv`, `ranges.csv`, `dedicated.csv`) goes to `reproduce/out/every_horizon/`.

### Provenance

After patches 0001 and 0002 the SparseTSF tree is byte-identical to the code of the matched runs; patch 0003 differs from the
code of the recency runs only in one comment and one help string, and the scripts of patch 0004 equal the run scripts up to
comments and the interpreter variable. On the per-seed scores of the runs of the paper, the statistics scripts reproduce its logged
results exactly. SparseTSF is licensed under Apache-2.0; the patches modify its files, and the patched files remain
under that license.

## Citation

The citation will be added upon publication.

## Acknowledgements

This code builds on [SparseTSF](https://github.com/lss-1138/SparseTSF) (Apache-2.0) and, through it, on
[LTSF-Linear](https://github.com/cure-lab/LTSF-Linear), [Autoformer](https://github.com/thuml/Autoformer),
[Informer](https://github.com/zhouhaoyi/Informer2020), [PatchTST](https://github.com/yuqinie98/patchtst),
[FITS](https://github.com/VEWOXIC/FITS) and [RevIN](https://github.com/ts-kim/RevIN). The sliceable layers in
`models/modules/` are adapted from [AttentiveNAS](https://github.com/facebookresearch/AttentiveNAS), which in turn adapts
[Once-for-All](https://github.com/mit-han-lab/once-for-all). The authors thank their developers.

## License

Apache License 2.0 (see [LICENSE](LICENSE)), as in SparseTSF. The files in `models/modules/` keep the copyright
headers and the license of AttentiveNAS (CC BY-NC 4.0).

## Contact

Please open an issue, or contact the corresponding author at chakkritt@nu.ac.th.
