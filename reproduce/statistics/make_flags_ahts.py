"""Facts of the AnyHorizonTS runs read from their training logs (the per-seed score files are written by run_longExp.py itself, result_windows format).
Usage: python reproduce/statistics/make_flags_ahts.py <arm> <out_csv> <training log> [...]   (arm: ahts_main or ahts_h720)
Output columns: arm, dataset, head, seed, restored_epoch (epoch of the last 'Saving model' line = the checkpoint that is scored), epochs_run, val_loss_restored
(the 'Vali Loss' of the restored epoch, the quantity of early stopping: by default the validation MSE at H_max), steps_per_epoch. Same definition as
make_scores.py of the matched SparseTSF arm. The ten-seed analysis (ten_seed.py) reads one file per arm with the seeds 2023-2032:
flags_ahts_main_per_seed.csv (logs of scripts/run_main.sh) and flags_ahts_h720_per_seed.csv (logs of the HORIZONS=720 runs).
Exit 1 if a run has no epochs or no saved model."""
import csv, re, sys
ARM, OUT, LOGS = sys.argv[1], sys.argv[2], sys.argv[3:]
DS = {'ETTh1': 'ETTh1', 'ETTh2': 'ETTh2', 'weather': 'Weather', 'Electricity': 'Electricity', 'Solar': 'Solar-Energy', 'traffic': 'Traffic'}
SET = re.compile(r'^(?P<mid>[A-Za-z0-9]+)_720_720_SparseTSF_\w+_ftM_sl720_pl720_(?P<head>linear|mlp)_test_\d+_horizons_[0-9_]+?_seed(?P<seed>\d+)$')
runs = {}
for fn in LOGS:
    cur = None
    for line in open(fn, errors='replace'):
        line = line.rstrip('\n')
        m = re.match(r'>>>>>>>start training : (\S+?)>+$', line)
        if m:
            s = SET.match(m[1]); assert s, m[1]
            cur = dict(epochs=[], restored=None, key=(DS[s['mid']], s['head'], int(s['seed'])))
            runs[cur['key']] = cur
            continue
        if cur is None:
            continue
        m = re.match(r'Epoch: (\d+), Steps: (\d+) \| Train Loss: ([\d.]+) Vali Loss: ([\d.]+)', line)
        if m: cur['epochs'].append((int(m[1]), int(m[2]), float(m[4])))
        elif 'Saving model' in line and cur['epochs']: cur['restored'] = len(cur['epochs'])
rows, bad = [], []
for (ds, head, seed), r in sorted(runs.items()):
    if not (r['epochs'] and r['restored']): bad.append((ds, head, seed)); continue
    rows.append(dict(arm=ARM, dataset=ds, head=head, seed=seed, restored_epoch=r['restored'], epochs_run=len(r['epochs']),
                     val_loss_restored=r['epochs'][r['restored'] - 1][2], steps_per_epoch=r['epochs'][0][1]))
with open(OUT, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['arm', 'dataset', 'head', 'seed', 'restored_epoch', 'epochs_run', 'val_loss_restored', 'steps_per_epoch']); w.writeheader(); w.writerows(rows)
print(f'{len(rows)} runs; incomplete: {len(bad)} {bad[:5]}')
sys.exit(1 if bad else 0)
