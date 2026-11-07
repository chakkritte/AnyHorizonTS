"""Ten-seed analysis: AnyHorizonTS against matched SparseTSF and the truncated {720} model with ten seeds (2023-2032) per arm.

Arms (per-horizon test windows; file names and formats: reproduce/DATA_FORMAT.md):
  AHTS       AnyHorizonTS, full set {96,192,336,720}:  ahts_main_per_seed.csv (2023-2027) + ahts_main_s10_per_seed.csv (2028-2032)
  Matched    matched SparseTSF (G1):                   stsf_matched_G1_per_seed.csv + stsf_matched_G1_s10_per_seed.csv
  Truncated  {720} alone, read by truncation:          ahts_h720_per_seed.csv + ahts_h720_s10_per_seed.csv
  Official   official SparseTSF, five seeds only:      stsf_official_per_seed.csv (2023-2027)
Families of 48 cells (6 datasets x 2 heads x 4 horizons), d = (AHTS - reference) / reference (negative = AnyHorizonTS lower):
  Matched (10 vs 10), Truncated (10 vs 10), Official (AHTS 2023-2027 vs official SparseTSF, 5 vs 5).
Per cell a two-sided Welch test; Holm within each family; TOST at +-2%; the smallest equivalence margin; counts of cells equivalent
at 1/2/3/5% (the margin ladder); the difference detectable with 80% power (stats_lib.py).
Looks: ten seeds (the main analysis); five seeds 2023-2027; the new seeds 2028-2032 alone (replication).
Sensitivity rule (fixed before the runs of seeds 2028-2032): a run whose restored validation loss exceeds the median of its cell over
the ten seeds by more than 20% is flagged, and the families are repeated without the flagged runs. Cell of the rule: AnyHorizonTS arms
(one run serves four horizons) = (arm, dataset, head), from flags_ahts_main_per_seed.csv and flags_ahts_h720_per_seed.csv (seeds
2023-2032, written by make_flags_ahts.py from the training logs); matched SparseTSF = (dataset, head, H), from the column
val_loss_restored of its score files.
Output: <out>/ten_seed.csv (one row per look, family and cell), <out>/ten_seed_flags.csv (flagged runs), and the printed summary:
the looks, then the matched-comparison table (d, Welch p, Holm, smallest margin), the margin ladder per head, the smallest margins
rounded up to two decimals, and the summary of the looks.
Usage: python reproduce/statistics/ten_seed.py --data DIR [--out DIR]
"""
import argparse, csv, math, os, re, sys
import numpy as np
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
from stats_lib import cell, holm, ALPHA  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('--data', required=True, help='folder with the per-seed CSVs of the runs (file names and formats: reproduce/DATA_FORMAT.md)')
ap.add_argument('--out', default=os.path.join(D, '..', 'out'), help='output folder (default: reproduce/out)')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
F = lambda fn: os.path.join(a.data, fn)
NAME = {'etth1': 'ETTh1', 'etth2': 'ETTh2', 'electricity': 'Electricity', 'solar': 'Solar-Energy', 'traffic': 'Traffic', 'weather': 'Weather'}
ORDER = ['ETTh1', 'ETTh2', 'Weather', 'Electricity', 'Traffic', 'Solar-Energy']
HEADS, HS = ('linear', 'mlp'), (96, 192, 336, 720)
OLD, NEW = tuple(range(2023, 2028)), tuple(range(2028, 2033)); ALL = OLD + NEW
PAT = re.compile(r'^(?P<mid>[A-Za-z0-9]+)_720_720_SparseTSF_.*_(?P<head>linear|mlp)_test_\d+_horizons_(?P<set>[0-9_]+?)_seed(?P<seed>\d+)$')


def ahts(paths, hset):
    out = {}
    for p in paths:
        for r in csv.DictReader(open(p)):
            m = PAT.match(r['setting'])
            if r['windows'] == 'per_horizon' and m and m['set'] == hset:
                out.setdefault((NAME[m['mid'].lower()], m['head'], int(r['H'])), {})[int(m['seed'])] = float(r['mse'])  # last row wins
    return out


def stsf(paths):
    out, val = {}, {}
    for p in paths:
        for r in csv.DictReader(open(p)):
            k = (r['dataset'], r['head'], int(r['H']))
            out.setdefault(k, {})[int(r['seed'])] = float(r['mse'])
            if 'val_loss_restored' in r: val.setdefault(k, {})[int(r['seed'])] = float(r['val_loss_restored'])
    return out, val


A = ahts([F('ahts_main_per_seed.csv'), F('ahts_main_s10_per_seed.csv')], '720_336_192_96')
T = ahts([F('ahts_h720_per_seed.csv'), F('ahts_h720_s10_per_seed.csv')], '720')
M, MV = stsf([F('stsf_matched_G1_per_seed.csv'), F('stsf_matched_G1_s10_per_seed.csv')])
O, _ = stsf([F('stsf_official_per_seed.csv')])
for nm, src, need in (('AHTS', A, ALL), ('Truncated', T, ALL), ('Matched', M, ALL), ('Official', O, OLD)):
    assert len(src) == 48 and all(tuple(sorted(v)) == need for v in src.values()), f'{nm}: 48 cells with seeds {need[0]}-{need[-1]} expected'

# flags (20% rule)
FV = {}
for fn in ('flags_ahts_main_per_seed.csv', 'flags_ahts_h720_per_seed.csv'):
    for r in csv.DictReader(open(F(fn))):
        FV.setdefault((r['arm'], r['dataset'], r['head']), {})[int(r['seed'])] = float(r['val_loss_restored'])
assert len(FV) == 24 and all(tuple(sorted(v)) == ALL for v in FV.values()), 'flag files: 2 arms x 12 (dataset, head) x seeds 2023-2032 expected'
assert len(MV) == 48 and all(tuple(sorted(v)) == ALL for v in MV.values()), 'matched SparseTSF: val_loss_restored of seeds 2023-2032 expected'
flagged = {'AHTS': set(), 'Truncated': set(), 'Matched': set()}  # (dataset, head, H, seed)
frows = []
for (arm, ds, hd), v in sorted(FV.items()):
    med = np.median(list(v.values()))
    for s, x in v.items():
        if x > 1.2 * med:
            for H in HS: flagged['AHTS' if arm == 'ahts_main' else 'Truncated'].add((ds, hd, H, s))
            frows.append(dict(arm=arm, dataset=ds, head=hd, H='all', seed=s, val_loss_restored=x, cell_median=med, ratio=x / med))
for k, v in sorted(MV.items()):
    med = np.median(list(v.values()))
    for s, x in v.items():
        if x > 1.2 * med:
            flagged['Matched'].add(k + (s,))
            frows.append(dict(arm='stsf_matched', dataset=k[0], head=k[1], H=k[2], seed=s, val_loss_restored=x, cell_median=med, ratio=x / med))
with open(os.path.join(a.out, 'ten_seed_flags.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['arm', 'dataset', 'head', 'H', 'seed', 'val_loss_restored', 'cell_median', 'ratio']); w.writeheader(); w.writerows(frows)


def fam(name, a_src, b_src, sa, sb, a_arm='AHTS', b_arm=None, drop=False):
    rows = []
    for ds in ORDER:
        for hd in HEADS:
            for H in HS:
                k = (ds, hd, H)
                xa = [a_src[k][s] for s in sa if not (drop and (ds, hd, H, s) in flagged[a_arm])]
                xb = [b_src[k][s] for s in sb if not (drop and b_arm and (ds, hd, H, s) in flagged[b_arm])]
                rows.append(dict(look=name[0], family=name[1], dataset=ds, head=hd, H=H, n_a=len(xa), n_b=len(xb), **cell(np.array(xa), np.array(xb))))
    for r, q in zip(rows, holm([r['welch_p'] for r in rows])): r['holm_p'] = q
    return rows


looks = []
looks += fam(('ten', 'Matched'), A, M, ALL, ALL, b_arm='Matched')
looks += fam(('ten', 'Truncated'), A, T, ALL, ALL, b_arm='Truncated')
looks += fam(('five', 'Official'), A, O, OLD, OLD)
looks += fam(('five', 'Matched'), A, M, OLD, OLD)
looks += fam(('five', 'Truncated'), A, T, OLD, OLD)
looks += fam(('new', 'Matched'), A, M, NEW, NEW)
looks += fam(('new', 'Truncated'), A, T, NEW, NEW)
looks += fam(('ten-sens', 'Matched'), A, M, ALL, ALL, b_arm='Matched', drop=True)
looks += fam(('ten-sens', 'Truncated'), A, T, ALL, ALL, b_arm='Truncated', drop=True)
with open(os.path.join(a.out, 'ten_seed.csv'), 'w', newline='') as fo:
    w = csv.DictWriter(fo, fieldnames=list(looks[0])); w.writeheader()
    for r in looks: w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})

print(f'flagged runs (20% rule): AHTS {len(flagged["AHTS"]) // 4}, Truncated {len(flagged["Truncated"]) // 4}, matched SparseTSF {len(flagged["Matched"])}')
for r in frows: print(f'   {r["arm"]:12s} {r["dataset"]:12s} {r["head"]:6s} H={r["H"]!s:4s} seed {r["seed"]} val {r["val_loss_restored"]:.4f} median {r["cell_median"]:.4f} ratio {r["ratio"]:.2f}')
for lk, fm in [(r['look'], r['family']) for r in looks if r['dataset'] == 'ETTh1' and r['head'] == 'linear' and r['H'] == 96]:
    L = [r for r in looks if r['look'] == lk and r['family'] == fm]
    mm = np.array([r['min_margin_pct'] for r in L])
    sig = [r for r in L if r['welch_p'] < ALPHA]
    print(f'\n== {lk:8s} {fm:9s} 48 cells (n {L[0]["n_a"]} vs {L[0]["n_b"]}): Welch p<0.05 {len(sig)} (AHTS lower {sum(r["d_pct"] < 0 for r in sig)}, higher {sum(r["d_pct"] > 0 for r in sig)}); '
          f'Holm {sum(r["holm_p"] < ALPHA for r in L)}; equivalent at 1/2/3/5%: {[int((mm <= m).sum()) for m in (1, 2, 3, 5)]} '
          f'(Linear 2%: {sum(r["min_margin_pct"] <= 2 for r in L if r["head"] == "linear")}, MLP 2%: {sum(r["min_margin_pct"] <= 2 for r in L if r["head"] == "mlp")}); '
          f'median MDE80 {np.median([r["mde80_pct"] for r in L]):.2f}%; median smallest margin {np.median(mm):.2f}%')
    for r in L:
        if r['holm_p'] < ALPHA: print(f'   Holm-sig {r["dataset"]:12s} {r["head"]:6s} H={r["H"]:<3d} d={r["d_pct"]:+6.2f}% p={r["welch_p"]:.2g} holm={r["holm_p"]:.2g}')
    print('   not equivalent at 2%: ' + ', '.join(f'{r["dataset"]}/{r["head"]}/{r["H"]} (d {r["d_pct"]:+.2f}%, margin {r["min_margin_pct"]:.2f}%)' for r in L if r['min_margin_pct'] > 2))

# ---- the tables of the paper, as text ----
get = lambda lk, fm: {(r['dataset'], r['head'], r['H']): r for r in looks if r['look'] == lk and r['family'] == fm}
COLS = [('ten', 'Matched'), ('five', 'Official'), ('ten', 'Truncated')]
TB = {c: get(*c) for c in COLS}
TORDER = ['ETTh1', 'ETTh2', 'Electricity', 'Solar-Energy', 'Traffic', 'Weather']  # row order of the tables
ceil2 = lambda x: math.ceil(x * 100 - 1e-9) / 100


def tcell(r):
    p = '<0.001' if r['welch_p'] < 0.0005 else f'{r["welch_p"]:.3f}'
    return f'{r["d_pct"]:+.2f} ({p}){"*" if r["holm_p"] < ALPHA else ""}{"+" if r["min_margin_pct"] > 2 else ""}'


def ladder(lk, fm, hd):
    m = np.array([r['min_margin_pct'] for r in looks if r['look'] == lk and r['family'] == fm and (hd == 'Both' or r['head'] == hd.lower())])
    return [int((m <= x).sum()) for x in (1, 2, 3, 5)]


print('\n== Matched comparison: d % (Welch p); * Holm p < 0.05; + smallest equivalence margin > 2%. Columns: Linear Matched (10 v 10), '
      'Official (5 v 5), Truncated (10 v 10), then MLP')
for ds in TORDER:
    for H in HS:
        print(f'   {ds:12s} {H:3d}  ' + '  '.join(f'{tcell(TB[c][(ds, hd, H)]):>18s}' for hd in HEADS for c in COLS))
print('\n== Margin ladder: cells (of 48; 24 per head) equivalent at a margin of 1/2/3/5%')
for lk, fm in COLS:
    for hd in ('Both', 'Linear', 'MLP'):
        print(f'   {fm:9s} {hd:6s} {ladder(lk, fm, hd)}')
print('\n== Smallest equivalence margin, %, rounded up to two decimals (columns as in the matched comparison)')
for ds in TORDER:
    for H in HS:
        print(f'   {ds:12s} {H:3d}  ' + '  '.join(f'{ceil2(TB[c][(ds, hd, H)]["min_margin_pct"]):6.2f}' for hd in HEADS for c in COLS))
print('\n== Summary of the looks: Welch p<0.05 (AHTS lower/higher), Holm, ladder 1/2/3/5%, median MDE80')
LOOKS = [('ten', 'Matched', '2023-2032'), ('ten', 'Truncated', '2023-2032'), ('five', 'Matched', '2023-2027'), ('five', 'Official', '2023-2027'),
         ('five', 'Truncated', '2023-2027'), ('new', 'Matched', '2028-2032'), ('new', 'Truncated', '2028-2032'),
         ('ten-sens', 'Matched', '2023-2032, flagged runs removed'), ('ten-sens', 'Truncated', '2023-2032, flagged runs removed')]
for lk, fm, _ in LOOKS:
    L = [r for r in looks if r['look'] == lk and r['family'] == fm]
    sig = [r for r in L if r['welch_p'] < ALPHA]
    lo, hi = sum(r['d_pct'] < 0 for r in sig), sum(r['d_pct'] > 0 for r in sig)
    mde = np.median([r['mde80_pct'] for r in L])
    print(f'{lk:8s} {fm:9s}: Welch {len(sig)} (AHTS lower {lo}, higher {hi}); Holm {sum(r["holm_p"] < ALPHA for r in L)}; ladder {ladder(lk, fm, "Both")}; '
          f'Linear 2% {ladder(lk, fm, "Linear")[1]}, MLP 2% {ladder(lk, fm, "MLP")[1]}; |d|<=1% in {sum(abs(r["d_pct"]) <= 1 for r in L)}; '
          f'median MDE80 {mde:.2f}%')
    for r in sig: print(f'      sig {r["dataset"]}/{r["head"]}/{int(r["H"])} d={r["d_pct"]:+.2f}% p={r["welch_p"]:.3g} holm={r["holm_p"]:.3g}')
MT, S = TB[('ten', 'Matched')], get('ten-sens', 'Matched')
k = ('Weather', 'mlp', 192)
print(f'Weather/MLP 192 Matched ten seeds: d={MT[k]["d_pct"]:+.2f}% p={MT[k]["welch_p"]:.3f} margin {MT[k]["min_margin_pct"]:.2f}%; '
      f'without flagged run: d={S[k]["d_pct"]:+.2f}% p={S[k]["welch_p"]:.3f} margin {S[k]["min_margin_pct"]:.2f}%')
mlp96 = [(ds, MT[(ds, 'mlp', 96)]['d_pct']) for ds in TORDER]
print('Matched MLP at H=96 (d%): ' + ', '.join(f'{ds} {d:+.2f}' for ds, d in mlp96) + f'; AHTS higher in {sum(d > 0 for _, d in mlp96)} of 6, max {max(d for _, d in mlp96):+.2f}%')
print('median MDE80: ' + ', '.join(f'{fm} {np.median([r["mde80_pct"] for r in looks if r["look"] == lk and r["family"] == fm]):.2f}%' for lk, fm in COLS))
trl = [r for r in looks if r['look'] == 'ten' and r['family'] == 'Truncated' and r['head'] == 'linear']
print(f'Truncated Linear ten seeds: equivalent at 2% {sum(r["min_margin_pct"] <= 2 for r in trl)}/24; '
      f'Holm-sig {[(r["dataset"], int(r["H"]), float(round(r["d_pct"], 2))) for r in trl if r["holm_p"] < ALPHA]}')
EN = ('ETTh1', 'ETTh2', 'Electricity', 'Solar-Energy')
en = [r for r in looks if r['look'] == 'ten' and r['family'] == 'Matched' and r['dataset'] in EN]
print(f'energy datasets ({", ".join(EN)}), Matched ten seeds: equivalent at 2% {sum(r["min_margin_pct"] <= 2 for r in en)}/{len(en)}; Welch-sig: '
      + ', '.join(f'{r["dataset"]}/{r["head"]}/{r["H"]} {r["d_pct"]:+.2f}% p={r["welch_p"]:.3f}' for r in en if r['welch_p'] < ALPHA)
      + f'; Holm {sum(r["holm_p"] < ALPHA for r in en)}')
ep = {}
for p in (F('stsf_matched_G1_per_seed.csv'), F('stsf_matched_G1_s10_per_seed.csv')):
    for r in csv.DictReader(open(p)):
        if (r['dataset'], r['head'], int(r['H'])) == k: ep[int(r['seed'])] = (round(float(r['mse']), 4), r['restored_epoch'], r['epochs_run'])
print(f'matched Weather/MLP 192 per seed (mse, restored, run): {dict(sorted(ep.items()))}')
