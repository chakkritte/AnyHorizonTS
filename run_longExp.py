import argparse
import os
import random

import numpy as np
import torch

from exp.exp_main import Exp_Main

parser = argparse.ArgumentParser(description='AnyHorizonTS: one SparseTSF model for every forecast horizon')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1,
                    help='1 = train each seed, then evaluate it; 0 = evaluate the saved checkpoints of each seed')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='SparseTSF', help='model name (SparseTSF)')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--embed', type=str, default='learned',
                    help='time features encoding of the data loader, options:[timeF, fixed, learned]')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length (H_max for AnyHorizonTS)')

# SparseTSF
parser.add_argument('--enc_in', type=int, default=7, help='number of channels')
parser.add_argument('--period_len', type=int, default=24, help='period length')
parser.add_argument('--model_type', default='linear', help='model type: linear/mlp')
parser.add_argument('--d_model', type=int, default=128, help='hidden width of the MLP head')
parser.add_argument('--linear_bias', type=int, default=0,
                    help='output bias of the Linear head: 0 = none, as in SparseTSF/Linear (default); 1 = with bias, '
                         'as in the June 2026 runs (needed to load their checkpoints)')

# AnyHorizonTS
parser.add_argument('--horizons', '--sandynamic', '--AnyTimeTS', dest='horizons', nargs='+', type=int,
                    default=[720, 336, 192, 96],
                    help='AnyHorizonTS training horizon set, a subset of 720 336 192 96; the model is built at '
                         '--pred_len = H_max (the old names --sandynamic and --AnyTimeTS are accepted)')

# optimization
parser.add_argument('--num_workers', type=int, default=6, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='number of seeds (2023, 2024, ...)')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function: mse, mae, smooth or logcosh')
parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'adamw'],
                    help="optimizer: 'adam', as in SparseTSF (default), or 'adamw' with the PyTorch default weight decay "
                         "of 0.01, as in the June 2026 runs")
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3,
                    help='pct_start (--lradj TST, and the OneCycleLR of --official_first_epoch_lr)')
parser.add_argument('--official_first_epoch_lr', action='store_true', default=False,
                    help='diagnostic, default off (= the paper runs): construct the scheduler of the official SparseTSF code, '
                         'OneCycleLR(max_lr=--learning_rate, pct_start, epochs, steps_per_epoch), instead of CosineAnnealingLR. '
                         'With --lradj type3 it is never stepped, but its constructor sets the optimizer lr to max_lr / 25 for '
                         'epoch 1 (and Adam beta1 to 0.95 for the whole run); adjust_learning_rate restores the full rate after '
                         'epoch 1, so epochs 2 onward follow the same lr sequence as without the flag')
parser.add_argument('--early_stop_on_all_horizons', action='store_true', default=False,
                    help='diagnostic, default off (= the paper runs): early stopping and best-checkpoint selection use the mean '
                         'over --horizons of the per-horizon validation MSE (each on the first H steps of the windows of the '
                         'H_max validation loader) instead of the H_max validation loss alone; training is unchanged')
parser.add_argument('--single_pass', action='store_true', default=False,
                    help='diagnostic, default off (= the paper runs, one forward and backward pass per horizon): train with ONE forward '
                         'and backward pass at H_max per batch and the loss sum over --horizons of MSE(output[:, :H], target[:, :H]); '
                         'the gradient equals that of the accumulated per-horizon passes (prefix identity) up to floating-point '
                         'rounding, validation and early stopping are unchanged. Not combinable with --diag_dir')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=int, default=1, help='1 = use the GPU when one is available, 0 = CPU')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', type=int, help='use multiple gpus', default=0)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multiple gpus')
parser.add_argument('--test_flop', action='store_true', default=False,
                    help='print the MACs and parameters of the model with ptflops after the first evaluation, then exit')

# evaluation and logging
parser.add_argument('--window_result_file', type=str, default='result_windows.csv',
                    help='CSV file to append the results of the paper to (default: result_windows.csv; none = off). '
                         'After training each seed, every horizon is scored on two sets of test windows, streaming in '
                         'float64: per_horizon = every origin whose L + H span fits (N - L - H + 1, the windows of the '
                         'single-horizon baselines, used by every table of the paper) and shared = every origin whose '
                         'L + H_max span fits (N - L - H_max + 1, the windows of test())')
parser.add_argument('--result_file', type=str, default='result_shared.csv',
                    help='CSV file to append the rows of test() to (default: result_shared.csv). test() scores every '
                         'horizon on the shared windows only, so these rows are secondary to --window_result_file')
parser.add_argument('--test_batch_size', type=int, default=64,
                    help='batch size of the evaluation passes after training (the metrics do not depend on it)')
parser.add_argument('--stream_metrics', action='store_true', default=False,
                    help='accumulate test MSE/MAE/RSE per batch in float64 instead of storing all '
                         'predictions (same metrics, a fraction of the host memory)')
parser.add_argument('--score_horizons', type=int, nargs='+', default=None,
                    help='evaluation only (use with --is_training 0), default None = off: score these horizons, each a multiple '
                         'of --period_len and at most --pred_len, on the per_horizon (and shared) windows of --window_result_file '
                         'instead of the benchmark horizons 96, 192, 336, 720, for example 24 on hourly data (day-ahead) or '
                         '144 on Solar-Energy. The largest horizon given sets the shared windows. test() is not run')
parser.add_argument('--diag_dir', type=str, default=None,
                    help='directory for per-epoch, per-horizon loss and '
                         'gradient-norm CSVs (logging only; default None = off)')

args = parser.parse_args()
if args.window_result_file.lower() in ('', 'none'):
    args.window_result_file = None  # no per-horizon scoring
if args.score_horizons:
    assert not args.is_training, '--score_horizons scores saved checkpoints: use it with --is_training 0'
    assert args.window_result_file, '--score_horizons needs --window_result_file'
if args.single_pass:
    assert not args.diag_dir, '--single_pass cannot be combined with --diag_dir (the diagnostics hook the per-horizon backward passes)'

# seeds: --itr k trains and evaluates seeds 2023, ..., 2022 + k
fix_seed_list = range(2023, 2033)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    args.device_ids = [int(id_) for id_ in args.devices.split(',')]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

horizon_tag = f"horizons_{'_'.join(map(str, args.horizons))}"


def train_setting(ii):
    """Name of the checkpoint folder of seed ii."""
    return '{}_{}_{}_ft{}_sl{}_pl{}_{}_{}_{}_{}_seed{}'.format(
        args.model_id, args.model, args.data, args.features, args.seq_len, args.pred_len, args.model_type, args.des,
        ii, horizon_tag, fix_seed_list[ii])


def test_setting(ii, H):
    """Name of the result row of seed ii at horizon H (its fields are ordered differently from train_setting)."""
    return '{}_{}_{}_ft{}_sl{}_pl{}_{}_{}_{}_{}_seed{}'.format(
        args.model_id, args.model, args.data, args.features, args.seq_len, H, args.model_type, args.des,
        horizon_tag, ii, fix_seed_list[ii])


train_batch_size = args.batch_size
for ii in range(args.itr):
    args.batch_size = train_batch_size  # every seed trains with --batch_size
    random.seed(fix_seed_list[ii])
    torch.manual_seed(fix_seed_list[ii])
    np.random.seed(fix_seed_list[ii])
    setting = train_setting(ii)

    exp = Exp_Main(args)  # set experiments
    if args.is_training:
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)  # ends with the best checkpoint loaded
    else:
        exp.model.load_state_dict(torch.load(os.path.join(args.checkpoints, setting, 'checkpoint.pth'),
                                             map_location=exp.device))

    if args.score_horizons:  # any horizons the model can serve (--score_horizons): test_windows only, test() is not run
        exp.test_windows(setting, args.window_result_file, horizons=args.score_horizons)
        torch.cuda.empty_cache()
        continue

    # one model, evaluated at every benchmark horizon by slicing its output projection
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    args.batch_size = args.test_batch_size  # evaluation batch size; restored before the next seed trains
    torch.cuda.empty_cache()
    for H in (96, 192, 336, 720):
        exp.test(test_setting(ii, H), pred_len=H)  # shared windows (loader built at H_max) -> --result_file
        torch.cuda.empty_cache()

    if args.window_result_file:  # per-horizon windows (the paper's tables) and shared windows -> --window_result_file
        exp.test_windows(setting, args.window_result_file)
        torch.cuda.empty_cache()
