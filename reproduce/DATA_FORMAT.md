# Per-seed score files read by the statistics

The scripts in `reproduce/statistics/` read the test MSE and MAE of every run, five seeds (2023-2027) per cell, from CSV files
in one folder (`--data DIR`, or `DATA=DIR` for `run_all.sh`). Each horizon is scored on the test windows of a loader built for
that horizon (N - L - H + 1 origins). No scores are included in this repository; the files are written by the runs below.

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

## Every-horizon scores

`reproduce/every_horizon/summarize_every_horizon.py` reads the output folder of `run_every_horizon.sh` (`--run DIR`, default
`reproduce/out/every_horizon_scores`): `ahts_main.csv`, `ahts_h720.csv`, `stsf_official.csv`, `stsf_matched_g1.csv` and
`stsf_matched_g2.csv`, each with the columns `kind,set_or_trained,dataset,head,seed,h,n_windows,mse,mae` (one row per checkpoint
and horizon h = 1..720; a file may be gzipped). AnyHorizonTS at h in {24, 48, 144, 480} and the G2 models at their trained h
suffice for the comparison with the dedicated models.
