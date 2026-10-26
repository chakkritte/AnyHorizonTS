"""Deficit tests of the paper: why AnyHorizonTS/MLP trails the official SparseTSF/MLP on ETTh1 (H = 96, 192) and ETTh2 (H = 96).

Each test changes one or two of the optimizer settings, the training windows and the checkpoint selection of ONE model and compares
again, five seeds per arm (per-horizon test windows). Arms (column `arm` of deficit_tests_per_seed.csv; commands in the README):
  ahts                    AnyHorizonTS, main runs (ahts_main_per_seed.csv)
  stsf_official           official SparseTSF (stsf_official_per_seed.csv)
  ahts_official_settings  AnyHorizonTS with the optimizer state of the official SparseTSF code (--official_first_epoch_lr):
                          epoch-1 learning rate / 25 and Adam beta1 = 0.95, both heads, ETTh1 and ETTh2, H = 96..720
  stsf_ahts_settings      SparseTSF with the optimizer state of AnyHorizonTS (--ahts_optimizer_state), both heads, ETTh1 and ETTh2
  stsf_hmax_windows       SparseTSF on the training windows of the H_max = 720 loader only (--train_window_horizon 720; 7,201 windows:
                          its 624 most recent windows at H = 96 and 528 at H = 192 are dropped); MLP three cells, Linear ETTh1/ETTh2 96, 192
  stsf_both               both of the above (MLP, three cells)
  ahts_h96_selection      AnyHorizonTS with early stopping and checkpoint selection on the H = 96 validation loss (--early_stop_horizons 96), MLP
  stsf_newest_windows     SparseTSF on as many windows as the H_max loader, the OLDEST dropped instead of the most recent
                          (--train_window_horizon 720 --train_window_drop oldest), MLP, three cells
Part 1, the table: per row, d = (AHTS arm - STSF arm) / STSF arm and Welch's p in the three cells, Holm over the three cells of the row.
Part 2, every cell of every arm: AHTS against the arm (STSF arms) or the arm against STSF (AHTS arms), and the arm against its own
baseline (STSF arm vs official SparseTSF, AHTS arm vs the main AnyHorizonTS runs); Welch, Holm within each (comparison, head) family,
TOST +-2%.
Outputs: <out>/deficit_tests.csv (part 1), <out>/deficit_tests_all.csv (part 2), printed report.
Usage: python reproduce/statistics/deficit_tests.py [--data DIR] [--out DIR]
"""
import argparse, csv, os, re, sys
import numpy as np
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
from stats_lib import cell, holm, ALPHA  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('--data', required=True, help='folder with the per-seed CSVs of the runs (file names and formats: reproduce/DATA_FORMAT.md)')
ap.add_argument('--out', default=os.path.join(D, '..', 'out'), help='output folder (default: reproduce/out)')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
SEEDS = (2023, 2024, 2025, 2026, 2027)
NAME = {'etth1': 'ETTh1', 'etth2': 'ETTh2', 'electricity': 'Electricity', 'solar': 'Solar-Energy', 'traffic': 'Traffic', 'weather': 'Weather'}
PAT = re.compile(r'^(?P<mid>[A-Za-z0-9]+)_720_720_SparseTSF_.*_(?P<head>linear|mlp)_test_\d+_horizons_720_336_192_96_seed(?P<seed>\d+)$')

arms = {}   # arm -> (head, dataset, H) -> seed -> mse
for r in csv.DictReader(open(os.path.join(args.data, 'ahts_main_per_seed.csv'))):
    m = PAT.match(r['setting'])
    if r['windows'] == 'per_horizon' and m:
        arms.setdefault('ahts', {}).setdefault((m['head'], NAME[m['mid'].lower()], int(r['H'])), {})[int(m['seed'])] = float(r['mse'])
for r in csv.DictReader(open(os.path.join(args.data, 'stsf_official_per_seed.csv'))):
    arms.setdefault('stsf_official', {}).setdefault((r['head'], r['dataset'], int(r['H'])), {})[int(r['seed'])] = float(r['mse'])
for r in csv.DictReader(open(os.path.join(args.data, 'deficit_tests_per_seed.csv'))):
    arms.setdefault(r['arm'], {}).setdefault((r['head'], r['dataset'], int(r['H'])), {})[int(r['seed'])] = float(r['mse'])


def vec(arm, k):
    v = arms[arm][k]
    assert tuple(sorted(v)) == SEEDS, (arm, k, sorted(v))
    return np.array([v[s] for s in SEEDS])


# ---- Part 1: the table (MLP head, the three deficit cells), Holm over the three cells of each row
CELLS = [('ETTh1', 96), ('ETTh1', 192), ('ETTh2', 96)]
TABLE = [  # row label, tested arm a, reference arm b
    ('Official comparison', 'ahts', 'stsf_official'),
    ('AHTS, official settings', 'ahts_official_settings', 'stsf_official'),
    ('STSF, AHTS settings', 'ahts', 'stsf_ahts_settings'),
    ('STSF, H_max windows', 'ahts', 'stsf_hmax_windows'),
    ('STSF, both of the above', 'ahts', 'stsf_both'),
    ('AHTS, H = 96 selection', 'ahts_h96_selection', 'stsf_official'),
    ('STSF, newest windows', 'ahts', 'stsf_newest_windows'),
]
lines, out = [], []
print('Part 1: d = (AHTS - STSF) / STSF (%), Welch p, Holm over the three cells of each row (MLP head)')
for label, ta, tb in TABLE:
    cs = [cell(vec(ta, ('mlp',) + c), vec(tb, ('mlp',) + c)) for c in CELLS]
    d, p = [c['d_pct'] for c in cs], [c['welch_p'] for c in cs]
    h = holm(p)
    sig = [f'{c[0]} {c[1]}' for c, x in zip(CELLS, h) if x < 0.05]
    lines.append(f'{label:<26} ' + '  '.join(f'{c[0]} {c[1]}: d={x:+.2f}% p={q:.4f} holm={y:.4f}' for c, x, q, y in zip(CELLS, d, p, h))
                 + f'  -> Holm-significant: {", ".join(sig) or "none"}')
    for c, x, q, y in zip(CELLS, d, p, h):
        out.append(dict(row=label, dataset=c[0], H=c[1], d_pct=x, welch_p=q, holm_p=y))
with open(os.path.join(args.out, 'deficit_tests.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print('\n'.join(lines))

# ---- Part 2: every cell of every arm
rows = []
for arm in ('ahts_official_settings', 'stsf_ahts_settings', 'stsf_hmax_windows', 'stsf_both', 'ahts_h96_selection', 'stsf_newest_windows'):
    pairs = ((arm, 'stsf_official'), (arm, 'ahts')) if arm.startswith('ahts') else (('ahts', arm), (arm, 'stsf_official'))
    for ta, tb in pairs:
        for k in sorted(arms[arm], key=lambda k: (k[0], k[1], k[2])):
            rows.append(dict(comparison=f'{ta} vs {tb}', head=k[0], dataset=k[1], H=k[2], **cell(vec(ta, k), vec(tb, k))))
fam = lambda r: (r['comparison'], r['head'])
for f in dict.fromkeys(fam(r) for r in rows):
    idx = [i for i, r in enumerate(rows) if fam(r) == f]
    for i, q in zip(idx, holm([rows[i]['welch_p'] for i in idx])):
        rows[i]['holm_p'] = q
        rows[i]['family_size'] = len(idx)
with open(os.path.join(args.out, 'deficit_tests_all.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
    for r in rows:
        w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})
print('\nPart 2: every cell of every arm (d = (a - b) / b; Holm within each comparison and head)')
for r in rows:
    print(f"  {r['comparison']:42s} {r['head']:6s} {r['dataset']} H={r['H']:<3d} d={r['d_pct']:+6.2f}% p={r['welch_p']:.4f} "
          f"holm({r['family_size']})={r['holm_p']:.4f} tost2={r['tost_p']:.3f} margin={r['min_margin_pct']:.2f}%"
          + ('  *' if r['welch_p'] < ALPHA else ''))
