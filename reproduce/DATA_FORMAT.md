# Per-seed score files read by the statistics

The scripts in `reproduce/statistics/` read the test MSE and MAE of every run, five seeds (2023-2027) per cell, from CSV files
in one folder (`--data DIR`, or `DATA=DIR` for `run_all.sh`); the ten-seed analysis adds the seeds 2028-2032 from separate files
(see *Ten-seed analysis* below). Each horizon is scored on the test windows of a loader built for
that horizon (N - L - H + 1 origins). No scores are included in this repository; the files are written by the runs below.
The files of the runs of the paper are in the folder `1_per_seed_scores` of the results package (link in the README).

## AnyHorizonTS

Rows of `result_windows.csv`, which every AnyHorizonTS run appends to; only the rows with `windows` = `per_horizon` are read.
The setting is the name of the checkpoint folder (`..._<des>_<run>_horizons_<hset>_seed<seed>`), from which the scripts read
the dataset, head, look-back, period, horizon set and seed.

Columns: `setting,H,windows,n_windows,mse,mae`

| File | Content | Produced by |
|---|---|---|
| `ahts_main_per_seed.csv` | full horizon set {96, 192, 336, 720}, both heads, six datasets | `scripts/run_main.sh` |
| `ahts_h720_per_seed.csv` | one model trained at H = 720 alone, read at every horizon (truncated) | `HORIZONS=720` runs (`scripts/run_ablation.sh`) |
| `ahts_reduced_sets_per_seed.csv` | reduced horizon sets {96, 336, 720}, {96, 192, 720}, {96, 720} | `scripts/run_ablation.sh` |
| `ahts_lookback_per_seed.csv` | look-back L = 96 and 336 | `scripts/run_lookback_period.sh` |
| `ahts_period_per_seed.csv` | period P = 4, 6, 12, 24, 48, Linear head, ETTh1 and Weather | `scripts/run_lookback_period.sh` |

## SparseTSF

| File | Columns | Content | Produced by |
|---|---|---|---|
| `stsf_official_per_seed.csv` | `dataset,head,H,seed,n_windows,params,mse,mae,resulttxt_mse,resulttxt_mae` | official SparseTSF, one model per horizon | `reproduce/matched_sparsetsf/run_sparsetsf.sh` with `TAG=official FLAGS=""` (or `eval_ckpt.py` on the checkpoints of the official baseline) |
| `stsf_matched_G1_per_seed.csv` | `dataset,head,H,seed,mse,mae,n_windows,params,restored_epoch,epochs_run,val_loss_restored,steps_per_epoch,train_windows_all,train_windows_kept,epoch1_lr,resulttxt_mse,resulttxt_mae` | matched SparseTSF (training windows of the H_max loader, AnyHorizonTS optimizer settings), H in {96, 192, 336, 720}, with the facts of the training logs | `run_sparsetsf.sh` (`TAG=G1`): its `scores_G1.csv` |

## Other comparisons

| File | Columns | Content | Produced by |
|---|---|---|---|
| `day_ahead_per_seed.csv` | `dataset,head,H,seed,mse_ahts,mae_ahts,mse_stsf,mae_stsf,d_mse_ahts_minus_stsf` | AnyHorizonTS (`--score_horizons`) and official SparseTSF trained at H = 24 (ETTh1, ETTh2, Electricity) and 144 (Solar-Energy) | `--score_horizons 24` / `144` with `--is_training 0`; official scripts with `HORIZONS=24` / `144` |
| `deficit_tests_per_seed.csv` | `arm,head,dataset,H,seed,mse,mae` | the arms of the deficit tests (column `arm`, see `reproduce/statistics/deficit_tests.py`) | `reproduce/matched_sparsetsf/run_deficit_tests.sh` |

## Ten-seed analysis

`reproduce/statistics/ten_seed.py` reads the five-seed files above (`ahts_main_per_seed.csv`, `ahts_h720_per_seed.csv`,
`stsf_matched_G1_per_seed.csv`, `stsf_official_per_seed.csv`; seeds 2023-2027 only, as the other scripts expect) and the files below
with the seeds 2028-2032 of the same cells. Every arm needs all 48 cells (6 datasets x 2 heads x 4 horizons).

| File | Columns | Content | Produced by |
|---|---|---|---|
| `ahts_main_s10_per_seed.csv` | as `ahts_main_per_seed.csv` | AnyHorizonTS, full horizon set, seeds 2028-2032 | `SEED_START=2028 bash scripts/run_main.sh` |
| `ahts_h720_s10_per_seed.csv` | as `ahts_h720_per_seed.csv` | {720} alone, read at every horizon, seeds 2028-2032 | the `HORIZONS=720` runs with `SEED_START=2028` |
| `stsf_matched_G1_s10_per_seed.csv` | as `stsf_matched_G1_per_seed.csv` | matched SparseTSF (G1), seeds 2028-2032 | `SEED_START=2028 STSF_DIR=... bash reproduce/matched_sparsetsf/run_sparsetsf.sh` (its `scores_G1.csv`) |
| `flags_ahts_main_per_seed.csv` | `arm,dataset,head,seed,restored_epoch,epochs_run,val_loss_restored,steps_per_epoch` | facts of the training logs of the AnyHorizonTS full-set runs (`arm` = `ahts_main`), seeds 2023-2032 | `reproduce/statistics/make_flags_ahts.py ahts_main <out> <logs>` |
| `flags_ahts_h720_per_seed.csv` | as above (`arm` = `ahts_h720`) | the same for the {720} runs, seeds 2023-2032 | `make_flags_ahts.py ahts_h720 <out> <logs>` |

`SEED_START` (default 2023) sets the first seed of `run_longExp.py`, both in this code and in SparseTSF with patch 0005; with the
`--itr 5` of the scripts, `SEED_START=2028` trains the seeds 2028-2032 (the checkpoint index `test_<ii>` still counts from 0). The
runs append to the same result files and logs as the runs of seeds 2023-2027, so give them their own `--window_result_file` (or move
the files of the first five seeds away first). `val_loss_restored` is the validation loss of the restored checkpoint, read by the
sensitivity rule of the analysis (a run more than 20% above the median of its cell over the ten seeds is flagged and the families are
repeated without it); for matched SparseTSF it is the column of the same name of its score files.

## Optimizer study

`reproduce/statistics/optimizer_study.py` reads the Adam reference from `ahts_main_per_seed.csv` and `ahts_h720_per_seed.csv` and the
runs of `scripts/run_optimizer_study.sh` from the folder `optimizer_study/` inside `--data` (or `--runs DIR`), with the file names of
that script:

| File | Content |
|---|---|
| `result_windows.csv` | AdamW (DES `Aadamw_<full\|h720>_<head>_<ds>`), Adam + clipping at 1.0 (`Bclip1_...`), SGD with momentum 0.9 at the learning rate chosen on the full set (`Csgd_...`); seeds 2023-2027, `result_windows` format |
| `lr_<head>_<ds>.txt` | the learning rate chosen on the full set (one number; `tools/pick_lr.py`) |
| `own_result_windows.csv` | {720} under SGD at its own learning rate (`Csgdown_h720_<head>_<ds>`), seeds 2023-2027 |
| `lrown_<head>_<ds>.txt` | the learning rate chosen for {720} |
| `gridown_result_windows.csv` | the learning-rate grid of {720}, seed 2023 (`Cgridown<lr>_<head>_<ds>`; read by the bit-identity controls) |
| `update_norms_{adam,sgd}_<head>_etth1.csv` | `--update_norm_log` of the update-norm runs (columns: `utils/update_norms.py`) |

The script also writes `grid_result*.csv`, `grid_vali.csv`, `gridown_vali.csv` (best-epoch validation loss per grid point),
`diag_result*.csv` and `result.csv` / `own_result.csv` / `gridown_result.csv` (shared-window rows), which the statistics do not read.
`<ds>` is `etth1`, `etth2` or `weather`, and `<head>` is `linear` or `mlp`.

## Every-horizon scores

`reproduce/every_horizon/summarize_every_horizon.py` reads the output folder of `run_every_horizon.sh` (`--run DIR`, default
`reproduce/out/every_horizon_scores`): `ahts_main.csv`, `ahts_h720.csv`, `stsf_official.csv`, `stsf_matched_g1.csv` and
`stsf_matched_g2.csv`, each with the columns `kind,set_or_trained,dataset,head,seed,h,n_windows,mse,mae` (one row per checkpoint
and horizon h = 1..720; a file may be gzipped). AnyHorizonTS at h in {24, 48, 144, 480} and the G2 models at their trained h
suffice for the comparison with the dedicated models.
