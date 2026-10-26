"""Holm, equivalence (TOST +-2%), intervals and detectable differences for the other families of tests of the paper.

Inputs (in --data; per-horizon test windows, seeds 2023-2027): ahts_main_per_seed.csv, ahts_reduced_sets_per_seed.csv,
ahts_h720_per_seed.csv, ahts_lookback_per_seed.csv, ahts_period_per_seed.csv, stsf_official_per_seed.csv, day_ahead_per_seed.csv.
Families (a = tested arm, b = reference; d = (a - b) / b, positive = a worse):
  official      AnyHorizonTS (full set) vs official SparseTSF                                  48 cells (main results table)
  trunc720      AnyHorizonTS (full set) vs {720} = one model trained at 720 alone, truncated   48 cells
  h720_vs_full  the same pair in the direction of the ablation ({720} tested, full set as reference)  48 cells
  reduced       each reduced set {96,336,720}, {96,192,720}, {96,720} vs the full set          144 cells (horizon-set ablation)
  lookback96    L = 96 vs L = 720;  lookback336  L = 336 vs L = 720                           48 cells each (look-back table)
  period        P in {4, 6, 12, 48} vs P = 24 (ETTh1) and P in {6, 12, 24, 48} vs P = 4 (Weather), Linear head   32 cells (period table)
  dayahead      AnyHorizonTS vs official SparseTSF trained at H = 24 (ETTh1, ETTh2, Electricity) and 144 (Solar-Energy)   8 cells
Output: <out>/families.csv (one row per cell and family) and a printed summary.
Usage: python reproduce/statistics/families.py [--data DIR] [--out DIR]
"""
import argparse, csv, os, re, sys
import numpy as np
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
from stats_lib import cell, holm, binom_tail, ALPHA  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('--data', required=True, help='folder with the per-seed CSVs of the runs (file names and formats: reproduce/DATA_FORMAT.md)')
ap.add_argument('--out', default=os.path.join(D, '..', 'out'), help='output folder (default: reproduce/out)')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
FS = args.data
SEEDS = (2023, 2024, 2025, 2026, 2027)
NAME = {'etth1': 'ETTh1', 'etth2': 'ETTh2', 'electricity': 'Electricity', 'solar': 'Solar-Energy', 'traffic': 'Traffic', 'weather': 'Weather'}
ORDER = list(NAME.values())
PAT = re.compile(r'^(?P<mid>[A-Za-z0-9]+)_(?P<L>\d+)_720_SparseTSF_.*_sl\d+_pl720_(?P<head>linear|mlp)_(?:test|P(?P<P>\d+))_\d+_horizons_(?P<set>[0-9_]+?)_seed(?P<seed>\d+)$')


def load_ahts(fn):
    out = {}
    for r in csv.DictReader(open(os.path.join(FS, fn))):
        if r['windows'] != 'per_horizon':
            continue
        m = PAT.match(r['setting'])
        assert m, r['setting']
        P = int(m['P']) if m['P'] else None
        k = (NAME[m['mid'].lower()], m['head'], int(r['H']), int(m['L']), m['set'], P)
        out.setdefault(k, {})[int(m['seed'])] = float(r['mse'])
    return out


def vec(d, k):
    v = d[k]
    assert tuple(sorted(v)) == SEEDS, k
    return np.array([v[s] for s in SEEDS])


FULL = '720_336_192_96'
main = load_ahts('ahts_main_per_seed.csv')
red = load_ahts('ahts_reduced_sets_per_seed.csv')
h720 = load_ahts('ahts_h720_per_seed.csv')
lb = load_ahts('ahts_lookback_per_seed.csv')
per = load_ahts('ahts_period_per_seed.csv')
stsf = {}
for r in csv.DictReader(open(os.path.join(FS, 'stsf_official_per_seed.csv'))):
    stsf.setdefault((r['dataset'], r['head'], int(r['H'])), {})[int(r['seed'])] = float(r['mse'])

rows = []


def add(family, label, a, b, **key):
    c = cell(a, b)
    rows.append(dict(family=family, comparison=label, **key, **c))


for ds in ORDER:
    for head in ('linear', 'mlp'):
        for H in (96, 192, 336, 720):
            a = vec(main, (ds, head, H, 720, FULL, None))
            add('official', 'AHTS vs STSF(official)', a, vec(stsf, (ds, head, H)), dataset=ds, head=head, H=H)
            add('trunc720', 'AHTS vs {720} truncated', a, vec(h720, (ds, head, H, 720, '720', None)), dataset=ds, head=head, H=H)
            add('h720_vs_full', '{720} vs full', vec(h720, (ds, head, H, 720, '720', None)), a, dataset=ds, head=head, H=H)
            for s in ('720_336_96', '720_192_96', '720_96'):
                add('reduced', '{' + ','.join(sorted(s.split('_'), key=int)) + '} vs full', vec(red, (ds, head, H, 720, s, None)), a,
                    dataset=ds, head=head, H=H)
            for L in (96, 336):
                add(f'lookback{L}', f'L={L} vs L=720', vec(lb, (ds, head, H, L, FULL, None)), a, dataset=ds, head=head, H=H)
for ds, ref, Ps in (('ETTh1', 24, (4, 6, 12, 48)), ('Weather', 4, (6, 12, 24, 48))):
    for P in Ps:
        for H in (96, 192, 336, 720):
            add('period', f'P={P} vs P={ref}', vec(per, (ds, 'linear', H, 720, FULL, P)), vec(per, (ds, 'linear', H, 720, FULL, ref)),
                dataset=ds, head='linear', H=H)
da = {}
for r in csv.DictReader(open(os.path.join(FS, 'day_ahead_per_seed.csv'))):
    da.setdefault((r['dataset'], r['head'], int(r['H'])), {})[int(r['seed'])] = (float(r['mse_ahts']), float(r['mse_stsf']))
for (ds, head, H), v in sorted(da.items()):
    assert tuple(sorted(v)) == SEEDS
    add('dayahead', 'AHTS vs STSF(official) at day-ahead H', np.array([v[s][0] for s in SEEDS]), np.array([v[s][1] for s in SEEDS]),
        dataset=ds, head=head, H=H)

fams = []
for r in rows:
    if r['family'] not in fams:
        fams.append(r['family'])
for f in fams:
    idx = [i for i, r in enumerate(rows) if r['family'] == f]
    adj = holm([rows[i]['welch_p'] for i in idx])
    for i, q in zip(idx, adj):
        rows[i]['holm_p'] = q
        rows[i]['family_size'] = len(idx)

with open(os.path.join(args.out, 'families.csv'), 'w', newline='') as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0]))
    w.writeheader()
    for r in rows:
        w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})

for f in fams:
    R = [r for r in rows if r['family'] == f]
    n = len(R)
    sig = [r for r in R if r['welch_p'] < ALPHA]
    hs = [r for r in R if r['holm_p'] < ALPHA]
    eq = [r for r in R if r['tost_p'] < ALPHA]
    print(f"== {f}: {n} cells; Welch p<0.05: {len(sig)} (a better {sum(r['d_pct'] < 0 for r in sig)}, a worse {sum(r['d_pct'] > 0 for r in sig)}); "
          f"P(>= {len(sig)} of {n} | null, independent) = {binom_tail(len(sig), n):.2g}; Holm p<0.05: {len(hs)}; "
          f"TOST +-2% equivalent: {len(eq)}; median MDE80 {np.median([r['mde80_pct'] for r in R]):.2f}%; "
          f"median smallest margin {np.median([r['min_margin_pct'] for r in R]):.2f}%")
    for r in hs:
        print(f"   Holm-sig: {r['comparison']:28s} {r['dataset']:12s} {r['head']:6s} H={r['H']:<3d} d={r['d_pct']:+6.2f}% p={r['welch_p']:.2g} holm={r['holm_p']:.2g}")
    if f in ('official', 'trunc720', 'h720_vs_full', 'dayahead'):
        for head in ('linear', 'mlp'):
            Rh = [r for r in R if r['head'] == head]
            ne = [f"{r['dataset']}/{r['H']}({r['min_margin_pct']:.1f}%)" for r in Rh if r['tost_p'] >= ALPHA]
            print(f"   {head}: equivalent within 2%: {sum(r['tost_p'] < ALPHA for r in Rh)}/{len(Rh)}; not shown: {', '.join(ne)}")
