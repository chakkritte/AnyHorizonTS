#!/usr/bin/env python3
"""Summarize the result CSVs written by run_longExp.py (mean and sample standard deviation over seeds, per dataset,
head and horizon).

The default input is result_windows.csv (--window_result_file): one row per seed, horizon and window set,

    setting,H,windows,n_windows,mse,mae

``per_horizon`` rows score every horizon H on the windows of a data loader built for that horizon (N - L - H + 1
origins, the protocol of the single-horizon baselines and of every table of the paper); they are summarized by default.
``shared`` rows score every horizon on the windows of a loader built once at H_max (N - L - H_max + 1 origins); select
them with --windows shared. The rows ``Setting, MSE, MAE, RSE`` of test() (--result_file, default result_shared.csv)
are on the shared windows as well.

The setting encodes the model id, model, dataset, features, look-back L (sl), head, description, training horizon set,
run index and seed (in a test() file also the evaluation horizon H, as pl), for example

    ETTh1_720_720_SparseTSF_ETTh1_ftM_sl720_pl720_linear_test_0_horizons_720_336_192_96_seed2023

Rows are grouped by everything except the run index and the seed. If the same row appears more than once (a rerun
appended to the same file), the first occurrence is used. Legacy tags (``sandynamic_``, ``AnyTimeTS_``) are read as
well.

Usage:
    python tools/summarize_results.py
    python tools/summarize_results.py result_windows.csv --hset 720_336_192_96 --decimals 3
    python tools/summarize_results.py results/lookback_windows.csv --csv summary.csv
    python tools/summarize_results.py result_shared.csv --windows shared
"""
import argparse
import csv
import re
import statistics
import sys
from collections import OrderedDict

TAG = r'(?:horizons|sandynamic|AnyTimeTS)'
HEAD = (r'^(?P<model_id>.+?)_(?P<model>SparseTSF)_(?P<data>[A-Za-z0-9]+)_ft(?P<features>[MSU]+)'
        r'_sl(?P<seq_len>\d+)_pl(?P<pl>\d+)_(?P<head>linear|mlp)_(?P<des>[A-Za-z0-9]+)_')
# result_windows.csv: the setting is the name of the checkpoint folder, ..._<des>_<run>_horizons_<hset>_seed<seed>
WINDOWS_SETTING = re.compile(HEAD + r'(?P<run>\d+)_' + TAG + r'_(?P<hset>\d+(?:_\d+)*)_seed(?P<seed>\d+)$')
# result_shared.csv (test()): ..._<des>_horizons_<hset>_<run>_seed<seed>, with the evaluation horizon as pl
TEST_SETTING = re.compile(HEAD + TAG + r'_(?P<hset>\d+(?:_\d+)*)_(?P<run>\d+)_seed(?P<seed>\d+)$')

# model id prefix of the scripts -> dataset name, in the order of the paper
DATASETS = {'etth1': 'ETTh1', 'etth2': 'ETTh2', 'electricity': 'Electricity', 'solar': 'Solar-Energy',
            'traffic': 'Traffic', 'weather': 'Weather'}


def load(paths, windows):
    """Rows of one window set, the window sets that the files hold, and the number of unrecognised rows."""
    rows, seen, present, skipped = [], set(), set(), 0
    for path in paths:
        with open(path, newline='') as f:
            for rec in csv.reader(f):
                if not rec or rec[0].lower() == 'setting':      # blank line or header
                    continue
                m = None
                try:
                    if len(rec) == 6:                           # result_windows.csv
                        setting, H, kind, n_windows = rec[0], int(rec[1]), rec[2], int(rec[3])
                        mse, mae = float(rec[4]), float(rec[5])
                        m = WINDOWS_SETTING.match(setting)
                    else:                                       # test(): Setting, MSE, MAE, RSE
                        setting, kind, n_windows = rec[0], 'shared', None
                        mse, mae = float(rec[1]), float(rec[2])
                        m = TEST_SETTING.match(setting)
                        H = int(m.group('pl')) if m else None
                except (ValueError, IndexError):
                    m = None
                if not m:
                    skipped += 1
                    continue
                present.add(kind)
                if kind != windows or (setting, H, kind) in seen:   # keep the first occurrence of a row
                    continue
                seen.add((setting, H, kind))
                d = m.groupdict()
                d.update(H=H, windows=kind, n_windows=n_windows, mse=mse, mae=mae)
                rows.append(d)
    return rows, present, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('files', nargs='*', default=['result_windows.csv'],
                    help='result CSV file(s) (default: result_windows.csv)')
    ap.add_argument('--windows', choices=['per_horizon', 'shared'], default='per_horizon',
                    help='test windows to summarize: per_horizon = the windows of a loader built for each horizon, the '
                         'protocol of the paper (default); shared = the windows of a loader built once at H_max')
    ap.add_argument('--hset', default=None, help='keep one training horizon set, e.g. 720_336_192_96')
    ap.add_argument('--decimals', type=int, default=4)
    ap.add_argument('--csv', default=None, help='also write the summary to this CSV file')
    a = ap.parse_args()

    rows, present, skipped = load(a.files, a.windows)
    if skipped:
        print(f'({skipped} rows with an unrecognised setting were skipped)', file=sys.stderr)
    if a.hset:
        rows = [r for r in rows if r['hset'] == a.hset]
    if not rows:
        msg = f'no {a.windows} rows' + (f' of horizon set {a.hset}' if a.hset else '') + f' in {", ".join(a.files)}'
        if a.windows == 'per_horizon' and 'per_horizon' not in present and 'shared' in present:
            msg += ('; the files hold shared-window rows only (every horizon on the windows of a loader built once at '
                    'H_max, as test() scores it), which are not the numbers of the paper: summarize result_windows.csv, '
                    'or pass --windows shared to summarize these rows')
        sys.exit(msg)

    groups = OrderedDict()
    for r in rows:
        prefix = r['model_id'].split('_')[0]
        key = (DATASETS.get(prefix.lower(), prefix), r['features'], r['data'], int(r['seq_len']), r['head'], r['des'],
               r['hset'], r['windows'], r['H'])
        groups.setdefault(key, []).append(r)

    fmt = f'{{:.{a.decimals}f}}'
    sd = lambda v: statistics.stdev(v) if len(v) > 1 else float('nan')
    rank = {name: i for i, name in enumerate(DATASETS.values())}
    out = []
    for key in sorted(groups, key=lambda k: (rank.get(k[0], len(rank)), k[0], k[1], k[2], k[4], k[6], k[3], k[5], k[8])):
        name, features, data, L, head, des, hset, windows, H = key
        g = groups[key]
        mse, mae = [r['mse'] for r in g], [r['mae'] for r in g]
        n = {r['n_windows'] for r in g}
        n_windows = n.pop() if len(n) == 1 else 'varies'
        out.append([name if features == 'M' else f'{name} ({features})', head, L, des, hset, windows, H,
                    '-' if n_windows is None else n_windows, len(g),
                    statistics.mean(mse), sd(mse), statistics.mean(mae), sd(mae)])

    header = ['dataset', 'head', 'L', 'des', 'horizon_set', 'windows', 'H', 'n_windows', 'n_seeds',
              'MSE_mean', 'MSE_sd', 'MAE_mean', 'MAE_sd']
    widths = [12, 6, 4, 5, 15, 11, 4, 9, 7, 8, 8, 8, 8]
    print('  '.join(h.ljust(w) for h, w in zip(header, widths)))
    for row in out:
        cells = [str(x) for x in row[:9]] + [fmt.format(x) for x in row[9:]]
        print('  '.join(c.ljust(w) for c, w in zip(cells, widths)))
    if a.csv:
        with open(a.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(out)


if __name__ == '__main__':
    main()
