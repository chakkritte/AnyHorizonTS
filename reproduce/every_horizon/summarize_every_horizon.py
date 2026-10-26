"""Every-horizon comparison of the paper, from the scores of score_every_horizon.py.

Arms, per dataset, head, seed and horizon h = 1..720 (per-horizon test windows, N - h + 1 origins):
  ahts       AnyHorizonTS, one model trained on {96, 192, 336, 720}; h served by the prefix of one pass at 720   (ahts_main.csv, required)
  trunc720   one model trained at 720 alone (same code, windows and optimizer), truncated to h                (ahts_h720.csv)
  four_off   four official SparseTSF models (96, 192, 336, 720); h served by the smallest trained horizon >= h, truncated   (stsf_official.csv)
  four_mat   the same with the matched SparseTSF models (G1: windows of the 720 loader and the AnyHorizonTS optimizer settings)
             (stsf_matched_g1.csv)
  ded_mat    matched SparseTSF trained at h itself, for h in {24, 48, 144, 480} (G2)                           (stsf_matched_g2.csv)
Every input except ahts_main.csv is optional (a missing arm is skipped); a file may also be gzipped (<name>.gz).
The default --run folder (reproduce/out/every_horizon_scores) is the output of run_every_horizon.sh. AnyHorizonTS at h in {24, 48, 144, 480}
and the G2 models at their trained h suffice for the comparison with the dedicated models (dedicated.csv); the full curves and range
averages need the scores of every checkpoint at every h (score_every_horizon.py; about 290,000 rows).
Outputs (in --out):
  curves.csv        per dataset, head, h and arm: five-seed mean and s.d. of the MSE, and d = (ahts - arm) / arm with Welch p and 95% interval
  ranges.csv        per dataset, head and arm, the per-seed MSE averaged over a range of h (all 1..720; h < 96; 96 < h < 720 between the
                    training horizons; h not a multiple of P), then Welch, Holm within each (arm, range) family of 12 cells, TOST +-2%
  dedicated.csv     AnyHorizonTS at h against ded_mat, one family of 48 cells (Welch, Holm, TOST +-2%)
  printed summary   counts of h with |d| <= 1% and <= 2%, of h with Welch p < 0.05 for and against AnyHorizonTS, per arm and head
Usage: python reproduce/every_horizon/summarize_every_horizon.py [--run DIR] [--out DIR]
"""
import argparse, csv, gzip, os, sys
from collections import defaultdict
import numpy as np
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(D, '..', 'statistics'))
from stats_lib import cell, holm, welch, ci, ALPHA  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('--run', default=os.path.join(D, '..', 'out', 'every_horizon_scores'), help='folder with the score CSVs of score_every_horizon.py')
ap.add_argument('--out', default=os.path.join(D, '..', 'out', 'every_horizon'), help='output folder (default: reproduce/out/every_horizon)')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
SEEDS = (2023, 2024, 2025, 2026, 2027)
DS = ['ETTh1', 'ETTh2', 'Electricity', 'Solar-Energy', 'Traffic', 'Weather']
PER = {'ETTh1': 24, 'ETTh2': 24, 'Electricity': 24, 'Traffic': 24, 'Weather': 4, 'Solar-Energy': 4}
HMAX = 720


def load(fn):
    out = defaultdict(dict)   # (dataset, head, tag, h) -> seed -> mse
    p = os.path.join(a.run, fn)
    if os.path.exists(p):
        fh = open(p)
    elif os.path.exists(p + '.gz'):
        fh = gzip.open(p + '.gz', 'rt')
    else:
        return None
    for r in csv.DictReader(fh):
        out[(r['dataset'], r['head'], r['set_or_trained'], int(r['h']))][int(r['seed'])] = float(r['mse'])
    return out


ahts, h720, off = load('ahts_main.csv'), load('ahts_h720.csv'), load('stsf_official.csv')
mat, ded = load('stsf_matched_g1.csv'), load('stsf_matched_g2.csv')
assert ahts is not None, f'ahts_main.csv not found in {a.run}'
print('arms found:', ', '.join(n for n, s in (('trunc720', h720), ('four_off', off), ('four_mat', mat), ('ded_mat', ded)) if s) or 'none')


def four(src, ds, head, h, trained=(96, 192, 336, 720)):
    H = min(t for t in trained if t >= h)
    v = src.get((ds, head, str(H), h))
    return v


arms = {}
if h720:
    arms['trunc720'] = lambda ds, hd, h: h720.get((ds, hd, '720', h))
if off:
    arms['four_off'] = lambda ds, hd, h: four(off, ds, hd, h)
if mat:
    arms['four_mat'] = lambda ds, hd, h: four(mat, ds, hd, h)
if ded:
    arms['ded_mat'] = lambda ds, hd, h: ded.get((ds, hd, str(h), h))

vec = lambda v: None if v is None or tuple(sorted(v)) != SEEDS else np.array([v[s] for s in SEEDS])
curves = []
for ds in DS:
    for hd in ('linear', 'mlp'):
        for h in range(1, HMAX + 1):
            x = vec(ahts.get((ds, hd, '720_336_192_96', h)))
            if x is None:
                continue
            for arm, f in arms.items():
                y = vec(f(ds, hd, h))
                if y is None:
                    continue
                t, pv = welch(x, y)
                lo, hi = ci(x, y, 0.95)
                curves.append(dict(dataset=ds, head=hd, h=h, arm=arm, ahts_mse=x.mean(), ahts_sd=x.std(ddof=1), arm_mse=y.mean(),
                                   arm_sd=y.std(ddof=1), d_pct=100 * (x.mean() - y.mean()) / y.mean(), welch_p=pv, ci95_lo=lo, ci95_hi=hi))
if curves:
    with open(os.path.join(a.out, 'curves.csv'), 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=list(curves[0])); w.writeheader()
        for r in curves:
            w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})

RANGES = {'all': lambda h, P: True, 'below96': lambda h, P: h < 96, 'between': lambda h, P: 96 < h < 720 and h not in (192, 336),
          'nonmultiple': lambda h, P: h % P != 0}
ranges = []
for arm, f in arms.items():
    if arm == 'ded_mat':
        continue
    for rg, keep in RANGES.items():
        fam = []
        for ds in DS:
            for hd in ('linear', 'mlp'):
                hs = [h for h in range(1, HMAX + 1) if keep(h, PER[ds])]
                X = [vec(ahts.get((ds, hd, '720_336_192_96', h))) for h in hs]
                Y = [vec(f(ds, hd, h)) for h in hs]
                if any(v is None for v in X + Y):
                    continue
                x, y = np.mean(X, 0), np.mean(Y, 0)          # per-seed MSE averaged over the horizons of the range
                c = cell(x, y)
                fam.append(dict(arm=arm, range=rg, n_h=len(hs), dataset=ds, head=hd, **c))
        for r, q in zip(fam, holm([r['welch_p'] for r in fam])):
            r['holm_p'] = q
        ranges += fam
if ranges:
    with open(os.path.join(a.out, 'ranges.csv'), 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=list(ranges[0])); w.writeheader()
        for r in ranges:
            w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})

dedicated = []                                      # ded_mat: one family of 48 cells (6 datasets x 2 heads x h in {24, 48, 144, 480})
if ded:
    for ds in DS:
        for hd in ('linear', 'mlp'):
            for h in (24, 48, 144, 480):
                x, y = vec(ahts.get((ds, hd, '720_336_192_96', h))), vec(arms['ded_mat'](ds, hd, h))
                dedicated.append(dict(dataset=ds, head=hd, h=h, **cell(x, y)))
    for r, q in zip(dedicated, holm([r['welch_p'] for r in dedicated])):
        r['holm_p'] = q
    with open(os.path.join(a.out, 'dedicated.csv'), 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=list(dedicated[0])); w.writeheader()
        for r in dedicated:
            w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})

for arm in arms:
    for hd in ('linear', 'mlp'):
        R = [r for r in curves if r['arm'] == arm and r['head'] == hd]
        if not R:
            continue
        n = len(R)
        print(f"== {arm:9s} {hd:6s}: {n} (dataset, h) cells; |d|<=1%: {sum(abs(r['d_pct']) <= 1 for r in R)}, |d|<=2%: {sum(abs(r['d_pct']) <= 2 for r in R)}; "
              f"Welch p<0.05 for AHTS: {sum(r['welch_p'] < ALPHA and r['d_pct'] < 0 for r in R)}, against: {sum(r['welch_p'] < ALPHA and r['d_pct'] > 0 for r in R)}")
        for ds in DS:
            Rd = [r for r in R if r['dataset'] == ds]
            if not Rd:
                continue
            worst = max(Rd, key=lambda r: r['d_pct']); best = min(Rd, key=lambda r: r['d_pct'])
            print(f"     {ds:12s} d range {best['d_pct']:+6.2f}% (h={best['h']}) .. {worst['d_pct']:+6.2f}% (h={worst['h']}); "
                  f"p<0.05 for/against: {sum(r['welch_p'] < ALPHA and r['d_pct'] < 0 for r in Rd)}/{sum(r['welch_p'] < ALPHA and r['d_pct'] > 0 for r in Rd)}")
if dedicated:
    sig = [r for r in dedicated if r['welch_p'] < ALPHA]
    print(f"== dedicated (family of {len(dedicated)}): Welch p<0.05 {len(sig)} (AHTS lower {sum(r['d_pct'] < 0 for r in sig)}, higher "
          f"{sum(r['d_pct'] > 0 for r in sig)}); Holm {sum(r['holm_p'] < ALPHA for r in dedicated)}; TOST +-2% {sum(r['tost_p'] < ALPHA for r in dedicated)} "
          f"(per h: {', '.join(f'{h}: ' + str(sum(r['tost_p'] < ALPHA for r in dedicated if r['h'] == h)) + '/12' for h in (24, 48, 144, 480))})")
    for r in dedicated:
        print(f"DED {r['dataset']:12s} {r['head']:6s} h={r['h']:<3d} d={r['d_pct']:+6.2f}% p={r['welch_p']:.3g} holm={r['holm_p']:.3g} "
              f"tost2={r['tost_p']:.3g} margin={r['min_margin_pct']:.2f}%")
if mat:                                             # sensitivity: the failed matched run (Weather/MLP H = 192, seed 2025) serves h = 97..192
    S4 = (2023, 2024, 2026, 2027)
    for rg in ('all', 'between'):
        hs = [h for h in range(1, HMAX + 1) if RANGES[rg](h, PER['Weather'])]
        x = np.mean([vec(ahts.get(('Weather', 'mlp', '720_336_192_96', h))) for h in hs], 0)
        y = np.mean([[four(mat, 'Weather', 'mlp', h)[s] for s in S4] for h in hs], 0)
        c = cell(x, y)
        print(f"SENS four_mat {rg:8s} Weather/MLP without matched seed 2025: d={c['d_pct']:+.2f}% p={c['welch_p']:.3g} tost2={c['tost_p']:.3g} "
              f"margin={c['min_margin_pct']:.2f}%")
for r in ranges:
    print(f"RANGE {r['arm']:9s} {r['range']:11s} {r['dataset']:12s} {r['head']:6s} n_h={r['n_h']:3d} d={r['d_pct']:+6.2f}% p={r['welch_p']:.3g} "
          f"holm={r['holm_p']:.3g} tost2={r['tost_p']:.3g} margin={r['min_margin_pct']:.2f}%")
