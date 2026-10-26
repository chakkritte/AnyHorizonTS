"""Main comparison of the paper: AnyHorizonTS against matched SparseTSF, official SparseTSF and a truncated single-horizon model.

Matched SparseTSF = the official SparseTSF code trained on the SAME training windows as AnyHorizonTS (the origins of the H = 720
loader, --train_window_horizon 720) and with the SAME optimizer settings (epoch-1 learning rate = --learning_rate, Adam betas
(0.9, 0.999), --ahts_optimizer_state); see reproduce/matched_sparsetsf (group G1, H in {96, 192, 336, 720}). The two still differ
in the training procedure itself: one model trained on the summed losses of the four horizons and stopped on the H = 720
validation loss, against four models, each trained and stopped on its own horizon.

Inputs (in --data): ahts_main_per_seed.csv (AnyHorizonTS, full horizon set), ahts_h720_per_seed.csv (one model trained at 720
alone, read at every horizon = truncated), stsf_official_per_seed.csv, stsf_matched_G1_per_seed.csv (per-horizon test windows,
seeds 2023-2027). Families of 48 cells (6 datasets x 2 heads x 4 horizons):
  main      AnyHorizonTS vs matched SparseTSF (the main comparison)
  official  AnyHorizonTS vs official SparseTSF
  cost      matched vs official SparseTSF (what the matching costs SparseTSF)
  trunc     {720} truncated vs matched SparseTSF
Welch per cell, Holm within each family, equivalence by TOST at +-2% (stats_lib.py); d = (a - b) / b, positive = a worse.
Output: <out>/matched_comparison.csv (one row per family and cell) and the printed summary.
Usage: python reproduce/statistics/matched_comparison.py [--data DIR] [--out DIR]
"""
import argparse, csv, os, re, sys
import numpy as np
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
from stats_lib import cell, holm, binom_tail, ALPHA  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('--data', required=True, help='folder with the per-seed CSVs of the runs (file names and formats: reproduce/DATA_FORMAT.md)')
ap.add_argument('--out', default=os.path.join(D, '..', 'out'), help='output folder (default: reproduce/out)')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
SEEDS = (2023, 2024, 2025, 2026, 2027)
NAME = {'etth1': 'ETTh1', 'etth2': 'ETTh2', 'electricity': 'Electricity', 'solar': 'Solar-Energy', 'traffic': 'Traffic', 'weather': 'Weather'}
ORDER = list(NAME.values())
PAT = re.compile(r'^(?P<mid>[A-Za-z0-9]+)_720_720_SparseTSF_.*_(?P<head>linear|mlp)_test_\d+_horizons_(?P<set>[0-9_]+?)_seed(?P<seed>\d+)$')


def ahts(fn, hset):
    out = {}
    for r in csv.DictReader(open(os.path.join(a.data, fn))):
        m = PAT.match(r['setting'])
        if r['windows'] == 'per_horizon' and m and m['set'] == hset:
            out.setdefault((NAME[m['mid'].lower()], m['head'], int(r['H'])), {})[int(m['seed'])] = float(r['mse'])
    return out


A, T = ahts('ahts_main_per_seed.csv', '720_336_192_96'), ahts('ahts_h720_per_seed.csv', '720')
O, M, info = {}, {}, {}
for r in csv.DictReader(open(os.path.join(a.data, 'stsf_official_per_seed.csv'))):
    O.setdefault((r['dataset'], r['head'], int(r['H'])), {})[int(r['seed'])] = float(r['mse'])
for r in csv.DictReader(open(os.path.join(a.data, 'stsf_matched_G1_per_seed.csv'))):
    k = (r['dataset'], r['head'], int(r['H']))
    M.setdefault(k, {})[int(r['seed'])] = float(r['mse'])
    info.setdefault(k, []).append((int(r['train_windows_all']), int(r['train_windows_kept']), float(r['epoch1_lr']), int(r['restored_epoch']), int(r['epochs_run'])))
vec = lambda d, k: np.array([d[k][s] for s in SEEDS])
assert len(M) == 48 and all(tuple(sorted(v)) == SEEDS for v in M.values())
rows = []
for fam, label, a_src, b_src in (('main', 'AHTS vs matched STSF', A, M), ('cost', 'matched STSF vs official STSF', M, O), ('trunc', '{720} vs matched STSF', T, M),
                                  ('official', 'AHTS vs official STSF', A, O)):
    for ds in ORDER:
        for hd in ('linear', 'mlp'):
            for H in (96, 192, 336, 720):
                k = (ds, hd, H)
                rows.append(dict(family=fam, comparison=label, dataset=ds, head=hd, H=H, **cell(vec(a_src, k), vec(b_src, k))))
for fam in ('main', 'cost', 'trunc', 'official'):
    idx = [i for i, r in enumerate(rows) if r['family'] == fam]
    for i, q in zip(idx, holm([rows[i]['welch_p'] for i in idx])):
        rows[i]['holm_p'] = q
with open(os.path.join(a.out, 'matched_comparison.csv'), 'w', newline='') as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader()
    for r in rows:
        w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})
# window and optimizer facts of the matched runs (read from their training logs by matched_sparsetsf/make_scores.py)
kept = sorted({(k[0], x[0], x[1]) for k, v in info.items() for x in v})
print('training windows (all -> kept) per dataset:', sorted({(d, b) for d, a_, b in kept}))
print('epoch-1 lr equals the script learning rate in all runs:', all(len({x[2] for x in v}) == 1 for v in info.values()))
for fam in ('main', 'official', 'cost', 'trunc'):
    R = [r for r in rows if r['family'] == fam]
    sig = [r for r in R if r['welch_p'] < ALPHA]
    print(f"== {fam} ({R[0]['comparison']}): Welch p<0.05 {len(sig)} of 48 (a lower {sum(r['d_pct'] < 0 for r in sig)}, a higher {sum(r['d_pct'] > 0 for r in sig)}), "
          f"binomial tail {binom_tail(len(sig), 48):.2g}; Holm {sum(r['holm_p'] < ALPHA for r in R)}; TOST+-2% {sum(r['tost_p'] < ALPHA for r in R)} "
          f"(Linear {sum(r['tost_p'] < ALPHA for r in R if r['head'] == 'linear')}, MLP {sum(r['tost_p'] < ALPHA for r in R if r['head'] == 'mlp')}); "
          f"|d|<=1% in {sum(abs(r['d_pct']) <= 1 for r in R)}; mean d Linear {np.mean([r['d_pct'] for r in R if r['head'] == 'linear']):+.3f}%, MLP {np.mean([r['d_pct'] for r in R if r['head'] == 'mlp']):+.3f}%")
    for r in sig:
        print(f"     {r['dataset']:12s} {r['head']:6s} H={r['H']:<3d} d={r['d_pct']:+6.2f}% p={r['welch_p']:.3g} holm={r['holm_p']:.3g} tost={r['tost_p']:.3g}")
    ne = [f"{r['dataset']}/{r['head']}/{r['H']}({r['min_margin_pct']:.1f}%)" for r in R if r['tost_p'] >= ALPHA]
    print('     not equivalent within 2%:', ', '.join(ne))
# Sensitivity: one matched run of the paper became unstable (Weather/MLP H = 192, seed 2025).
# It is kept in every number above; this is the same cell without that run (AnyHorizonTS keeps its five seeds).
k = ('Weather', 'mlp', 192)
mv = np.array([M[k][s] for s in SEEDS if s != 2025])
c = cell(vec(A, k), mv)
print(f"sensitivity Weather/MLP H=192 without matched seed 2025: d={c['d_pct']:+.2f}% p={c['welch_p']:.3f} tost={c['tost_p']:.3g} "
      f"(matched seed MSEs {', '.join(f'{M[k][s]:.4f}' for s in SEEDS)})")
# The four energy datasets: the same families restricted to ETTh1, ETTh2, Electricity, Solar-Energy (32 cells; Holm as above, within 48)
EN = ('ETTh1', 'ETTh2', 'Electricity', 'Solar-Energy')
for fam in ('main', 'official'):
    R = [r for r in rows if r['family'] == fam and r['dataset'] in EN]
    sig = [r for r in R if r['welch_p'] < ALPHA]
    print(f"energy datasets, {fam}: {len(R)} cells; Welch p<0.05 {len(sig)} "
          f"({', '.join(f'{r['dataset']}/{r['head']}/{r['H']} {r['d_pct']:+.2f}%' for r in sig)}); Holm {sum(r['holm_p'] < ALPHA for r in R)}; "
          f"TOST+-2% {sum(r['tost_p'] < ALPHA for r in R)}")
