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
| Training horizon-set ablation | `bash scripts/run_ablation.sh` (the full set comes from `run_main.sh`) |
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
- **Horizon set.** `--horizons` is a subset of `720 336 192 96` (the data loader provides targets for these four
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
