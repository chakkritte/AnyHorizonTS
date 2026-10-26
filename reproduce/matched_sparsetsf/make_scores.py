"""Join the scores of eval_ckpt.py (per-horizon test windows, streaming float64) with the facts read from the training logs of the same runs.

Usage: python reproduce/matched_sparsetsf/make_scores.py <eval_csv> <out_csv> <training log> [<training log> ...]
Output columns: dataset, head, H, seed, mse, mae, n_windows (test), params, restored_epoch (epoch of the last 'Saving model' line = the
checkpoint that is scored), epochs_run, val_loss_restored, steps_per_epoch, train_windows_all (official loader), train_windows_kept (after
--train_window_horizon), epoch1_lr (logged by --ahts_optimizer_state), resulttxt_mse, resulttxt_mae (the official test() output where test()
ran). Every scored row must have a log; rows without a log are reported, written with empty fields, and the exit code is 1.
The output of the G1 runs is the input stsf_matched_G1_per_seed.csv of reproduce/statistics/matched_comparison.py.
"""
import csv, re, sys
EVAL, OUT, LOGS = sys.argv[1], sys.argv[2], sys.argv[3:]
DS = {'ETTh1': 'ETTh1', 'ETTh2': 'ETTh2', 'weather': 'Weather', 'Electricity': 'Electricity', 'Solar': 'Solar-Energy', 'traffic': 'Traffic'}
SET = re.compile(r'^(?P<mid>[A-Za-z0-9]+)_720_(?P<H>\d+)_SparseTSF_\w+_ftM_sl720_pl\d+_(?P<head>linear|mlp)_test_\d+_seed(?P<seed>\d+)$')
runs = {}
for fn in LOGS:
    cur = None
    for line in open(fn, errors='replace'):
        line = line.rstrip('\n')
        m = re.match(r'>>>>>>>start training : (\S+?)>+$', line)
        if m:
            s = SET.match(m[1]); assert s, m[1]
            cur = dict(epochs=[], restored=None, tw_all=None, tw_kept=None, lr1=None, key=(DS[s['mid']], s['head'], int(s['H']), int(s['seed'])))
            runs[cur['key']] = cur   # a later occurrence (a re-run) replaces an earlier one
            continue
        if cur is None:
            continue
        m = re.match(r'train_window_horizon \d+: training windows (\d+) -> (\d+)', line)
        if m: cur['tw_all'], cur['tw_kept'] = int(m[1]), int(m[2])
        m = re.match(r'ahts_optimizer_state: epoch-1 lr = ([\d.e-]+) ', line)
        if m: cur['lr1'] = float(m[1])
        m = re.match(r'Epoch: (\d+), Steps: (\d+) \| Train Loss: ([\d.]+) Vali Loss: ([\d.]+)', line)
        if m: cur['epochs'].append((int(m[1]), int(m[2]), float(m[4])))
        elif 'Saving model' in line and cur['epochs']: cur['restored'] = len(cur['epochs'])
rows, nolog = [], []
for r in csv.DictReader(open(EVAL)):
    k = (r['dataset'], r['head'], int(r['H']), int(r['seed']))
    L = runs.get(k)
    f = dict(dataset=r['dataset'], head=r['head'], H=r['H'], seed=r['seed'], mse=r['mse'], mae=r['mae'], n_windows=r['n_windows'], params=r['params'])
    if L and L['epochs'] and L['restored']:
        f.update(restored_epoch=L['restored'], epochs_run=len(L['epochs']), val_loss_restored=L['epochs'][L['restored'] - 1][2], steps_per_epoch=L['epochs'][0][1],
                 train_windows_all=L['tw_all'], train_windows_kept=L['tw_kept'], epoch1_lr=L['lr1'])
    else:
        nolog.append(k)
    f.update(resulttxt_mse=r['resulttxt_mse'], resulttxt_mae=r['resulttxt_mae'])
    rows.append(f)
cols = ['dataset', 'head', 'H', 'seed', 'mse', 'mae', 'n_windows', 'params', 'restored_epoch', 'epochs_run', 'val_loss_restored', 'steps_per_epoch',
        'train_windows_all', 'train_windows_kept', 'epoch1_lr', 'resulttxt_mse', 'resulttxt_mae']
with open(OUT, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
print(f'{len(rows)} scored rows, {len(runs)} runs in {len(LOGS)} logs; rows without a complete log: {len(nolog)} {nolog[:5]}')
sys.exit(1 if nolog else 0)
