"""Recency against number of the omitted training windows, official SparseTSF/MLP (ETTh1 H = 96, 192; ETTh2 H = 96).

Arms per cell, seeds 2023-2027, per-horizon test windows:
  all          official SparseTSF, all training windows                         (stsf_official_per_seed.csv)
  drop_recent  the 7,201 OLDEST origins (the count of the H_max = 720 loader); the most recent 624 (H = 96) or 528 (H = 192)
               are dropped (--train_window_horizon 720)                         (arm stsf_hmax_windows of deficit_tests_per_seed.csv)
  drop_oldest  the 7,201 MOST RECENT origins; the oldest are dropped
               (--train_window_horizon 720 --train_window_drop oldest)          (arm stsf_newest_windows)
  ahts         AnyHorizonTS/MLP, main runs                                      (ahts_main_per_seed.csv)
drop_recent and drop_oldest keep the same number of windows, so drop_oldest - drop_recent isolates the recency, and drop_oldest - all
the number. Welch per cell, Holm over the three cells of each comparison, TOST +-2% (stats_lib.py).
Output: <out>/window_recency.csv and the printed report.
Usage: python reproduce/statistics/window_recency.py [--data DIR] [--out DIR]
"""
import argparse, csv, os, re, sys
import numpy as np
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
from stats_lib import cell, holm  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('--data', required=True, help='folder with the per-seed CSVs of the runs (file names and formats: reproduce/DATA_FORMAT.md)')
ap.add_argument('--out', default=os.path.join(D, '..', 'out'), help='output folder (default: reproduce/out)')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
SEEDS = range(2023, 2028)
PAT = re.compile(r'^(?P<mid>ETTh[12])_720_720_SparseTSF_.*_mlp_test_\d+_horizons_720_336_192_96_seed(?P<seed>\d+)$')
src = {'all': {}, 'drop_recent': {}, 'drop_oldest': {}, 'ahts': {}}
for r in csv.DictReader(open(os.path.join(args.data, 'stsf_official_per_seed.csv'))):
    if r['head'] == 'mlp':
        src['all'][(r['dataset'], int(r['H']), int(r['seed']))] = float(r['mse'])
for r in csv.DictReader(open(os.path.join(args.data, 'deficit_tests_per_seed.csv'))):
    arm = {'stsf_hmax_windows': 'drop_recent', 'stsf_newest_windows': 'drop_oldest'}.get(r['arm'])
    if arm and r['head'] == 'mlp':
        src[arm][(r['dataset'], int(r['H']), int(r['seed']))] = float(r['mse'])
for r in csv.DictReader(open(os.path.join(args.data, 'ahts_main_per_seed.csv'))):
    m = PAT.match(r['setting'])
    if r['windows'] == 'per_horizon' and m:
        src['ahts'][(m['mid'], int(r['H']), int(m['seed']))] = float(r['mse'])
rows = []
for ds, H in (('ETTh1', 96), ('ETTh1', 192), ('ETTh2', 96)):
    arm = {k: np.array([v[(ds, H, s)] for s in SEEDS]) for k, v in src.items()}
    gap = arm['drop_recent'].mean() - arm['all'].mean()
    rec = (arm['drop_recent'].mean() - arm['drop_oldest'].mean()) / gap
    print(f"{ds} H={H}: all {arm['all'].mean():.4f}  drop_recent {arm['drop_recent'].mean():.4f}  drop_oldest {arm['drop_oldest'].mean():.4f}  "
          f"AHTS {arm['ahts'].mean():.4f}; share of the drop_recent - all gap removed by keeping the recent windows: {100 * rec:.0f}%")
    for a, b in (('drop_oldest', 'drop_recent'), ('drop_oldest', 'all'), ('drop_recent', 'all'), ('ahts', 'drop_oldest'), ('ahts', 'drop_recent'), ('ahts', 'all')):
        rows.append(dict(dataset=ds, H=H, comparison=f'{a} vs {b}', mean_a=arm[a].mean(), mean_b=arm[b].mean(), **cell(arm[a], arm[b])))
for comp in dict.fromkeys(r['comparison'] for r in rows):
    idx = [i for i, r in enumerate(rows) if r['comparison'] == comp]
    for i, q in zip(idx, holm([rows[i]['welch_p'] for i in idx])):
        rows[i]['holm_p'] = q
for r in rows:
    print(f"  {r['dataset']} H={r['H']:<3d} {r['comparison']:26s} d={r['d_pct']:+6.2f}% p={r['welch_p']:.4f} holm(3)={r['holm_p']:.4f} "
          f"tost2={r['tost_p']:.3f} margin={r['min_margin_pct']:.2f}%")
with open(os.path.join(args.out, 'window_recency.csv'), 'w', newline='') as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader()
    for r in rows:
        w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})
