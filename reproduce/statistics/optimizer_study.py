"""Optimizer study: the full horizon set {96,192,336,720} against {720} alone (read at every horizon by truncation) per optimizer.

Runs: scripts/run_optimizer_study.sh (ETTh1, ETTh2, Weather x Linear, MLP; seeds 2023-2027; per-horizon test windows).
Inputs (file names and formats: reproduce/DATA_FORMAT.md):
  --data DIR   ahts_main_per_seed.csv and ahts_h720_per_seed.csv: the Adam reference (the main runs and the {720} runs)
  --runs DIR   the output folder of run_optimizer_study.sh (default: DIR/optimizer_study): result_windows.csv (AdamW = DES Aadamw,
               Adam + clipping at 1.0 = Bclip1, SGD momentum 0.9 at the learning rate chosen on the full set = Csgd), lr_<head>_<ds>.txt,
               own_result_windows.csv ({720} under SGD at its own learning rate, Csgdown), gridown_result_windows.csv (its learning-rate
               grid, Cgridown<lr>), lrown_<head>_<ds>.txt, update_norms_{adam,sgd}_<head>_etth1.csv (--update_norm_log)
Part 1, shared learning rate: per optimizer (Adam, AdamW, Adam + clip, SGD), 24 cells; per cell a two-sided Welch test
  (a = full set, b = {720}; d = (full - {720}) / {720}), Holm within each optimizer's 24 cells, TOST at +-2%, the smallest margin and
  the ladder at 1/2/3/5%; mean |d| per horizon; the SGD family again without non-finite (diverged) runs; the runs in which clipping
  never changed the result.
Part 2, own learning rate: the full set under SGD at the learning rate chosen for it against {720} under SGD at its own learning
  rate; a separate Holm family over 24 cells, non-finite runs dropped and counted. Bit-identity controls on the existing runs
  (C1: own-LR seed 2023 = grid run at the chosen LR; C2: grid run at the shared LR = shared-LR {720} seed 2023; C3: where the two
  learning rates are equal, own-LR {720} = shared-LR {720}). P7-P9 are the checks of the predictions recorded before these runs.
Part 3, update norms: ratio of the per-row RMS norms of the update (and of the gradient) of W_max, rows of the forecast steps 1-96
  against rows of the steps 337-720, at the first optimizer step and as the mean over epoch 1 (ETTh1, seed 2023, full set).
Output: <out>/optimizer_study_cells.csv (part 1), <out>/optimizer_study_own_lr_cells.csv (part 2) and the printed summary.
Usage: python reproduce/statistics/optimizer_study.py --data DIR [--runs DIR] [--out DIR]
"""
import argparse, csv, glob, math, os, re, sys
import numpy as np
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
from stats_lib import cell, holm  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('--data', required=True, help='folder with ahts_main_per_seed.csv and ahts_h720_per_seed.csv (reproduce/DATA_FORMAT.md)')
ap.add_argument('--runs', default=None, help='output folder of scripts/run_optimizer_study.sh (default: <data>/optimizer_study)')
ap.add_argument('--out', default=os.path.join(D, '..', 'out'), help='output folder (default: reproduce/out)')
args = ap.parse_args()
RUNS = args.runs or os.path.join(args.data, 'optimizer_study')
os.makedirs(args.out, exist_ok=True)
SEEDS = (2023, 2024, 2025, 2026, 2027)
NAME = {'etth1': 'ETTh1', 'etth2': 'ETTh2', 'weather': 'Weather'}
HS = (96, 192, 336, 720)


def part1():
    v = {}
    # Adam reference: the main runs (full set) and the {720} runs, L = 720, P of the scripts
    PAT = re.compile(r'^(?P<mid>[A-Za-z0-9]+)_(?P<L>\d+)_720_SparseTSF_.*_sl\d+_pl720_(?P<head>linear|mlp)_(?:test|P(?P<P>\d+))_\d+_horizons_(?P<set>[0-9_]+?)_seed(?P<seed>\d+)$')
    for fn, arm in (('ahts_main_per_seed.csv', 'full'), ('ahts_h720_per_seed.csv', 'h720')):
        for r in csv.DictReader(open(os.path.join(args.data, fn))):
            if r['windows'] != 'per_horizon': continue
            m = PAT.match(r['setting']); assert m, r['setting']
            ds = m['mid'].lower()
            if ds not in NAME or int(m['L']) != 720 or m['P']: continue
            if m['set'] != ('720_336_192_96' if arm == 'full' else '720'): continue
            v.setdefault(('Adam', arm, ds, m['head'], int(r['H'])), {})[int(m['seed'])] = float(r['mse'])
    OP = {'Aadamw': 'AdamW', 'Bclip1': 'Adam+clip', 'Csgd': 'SGD'}
    P2 = re.compile(r'_pl720_(?P<head>linear|mlp)_(?P<des>Aadamw|Bclip1|Csgd)_(?P<arm>full|h720)_(?P=head)_(?P<ds>etth1|etth2|weather)_\d+_horizons_[0-9_]+_seed(?P<seed>\d+)$')
    for r in csv.DictReader(open(os.path.join(RUNS, 'result_windows.csv'))):
        if r['windows'] != 'per_horizon': continue
        m = P2.search(r['setting']); assert m, r['setting']
        k = (OP[m['des']], m['arm'], m['ds'], m['head'], int(r['H']))
        s = int(m['seed']); assert s not in v.get(k, {}), (k, s)
        v.setdefault(k, {})[s] = float(r['mse'])

    def vec(k):
        x = v[k]; assert tuple(sorted(x)) == SEEDS, (k, sorted(x)); return np.array([x[s] for s in SEEDS])
    rows = []; summ = {}
    for op in ('Adam', 'AdamW', 'Adam+clip', 'SGD'):
        C = []
        for ds in NAME:
            for head in ('linear', 'mlp'):
                for h in HS:
                    c = cell(vec((op, 'full', ds, head, h)), vec((op, 'h720', ds, head, h)))
                    c.update(optimizer=op, dataset=NAME[ds], head=head, H=h); C.append(c)
        for c, p in zip(C, holm([c['welch_p'] for c in C])): c['holm_p'] = p
        rows += C
        mm = np.array([c['min_margin_pct'] for c in C])
        print(f'== {op}: 24 cells, full set vs {{720}} truncated (d = (full - {{720}}) / {{720}})')
        print(f'   Welch p<0.05: {sum(c["welch_p"] < 0.05 for c in C)}  Holm-significant: {sum(c["holm_p"] < 0.05 for c in C)}  '
              f'equivalent at 1/2/3/5%: {[int((mm <= m).sum()) for m in (1, 2, 3, 5)]}  median MDE {np.median([c["mde80_pct"] for c in C]):.2f}%')
        for c in C:
            print(f'   {c["dataset"]:8s} {c["head"]:6s} H={c["H"]:3d}  full {c["mse_a"]:.4f}  {{720}} {c["mse_b"]:.4f}  d={c["d_pct"]:+6.2f}%  '
                  f'p={c["welch_p"]:.3f} holm={c["holm_p"]:.3f}  min margin {c["min_margin_pct"]:.2f}%')
        for head in ('linear', 'mlp', 'all'):
            for h in HS:
                summ[(op, head, h)] = np.mean([abs(c['d_pct']) for c in C if c['H'] == h and (head == 'all' or c['head'] == head)])
    print('\nMean |d| (%) by H (all 6 dataset x head cells; then Linear, MLP)')
    for op in ('Adam', 'AdamW', 'Adam+clip', 'SGD'):
        print(f'   {op:9s} ' + '  '.join(f'H={h}: {summ[(op,"all",h)]:.2f} (L {summ[(op,"linear",h)]:.2f}, M {summ[(op,"mlp",h)]:.2f})' for h in HS))
    print('\nAbsolute level: mean MSE of the full set per optimizer relative to Adam (same cell), %')
    for op in ('AdamW', 'Adam+clip', 'SGD'):
        d = [100 * (vec((op, 'full', ds, hd, h)).mean() / vec(('Adam', 'full', ds, hd, h)).mean() - 1) for ds in NAME for hd in ('linear', 'mlp') for h in HS]
        print(f'   {op:9s} median {np.median(d):+.2f}%  range {min(d):+.2f}..{max(d):+.2f}%')
    with open(os.path.join(args.out, 'optimizer_study_cells.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['optimizer', 'dataset', 'head', 'H'] + [k for k in rows[0] if k not in ('optimizer', 'dataset', 'head', 'H')])
        w.writeheader(); w.writerows(rows)
    # Two SGD full-set runs of the paper diverged (NaN loss; the early stopper saves a NaN model, since NaN compares False), so their
    # cells are undefined above. Here each SGD cell drops non-finite runs; Holm over the 24 SGD cells again.
    print('\n== SGD sensitivity: diverged (non-finite) runs dropped; Holm over the 24 SGD cells')
    C = []
    for ds in NAME:
        for head in ('linear', 'mlp'):
            for h in HS:
                a = vec(('SGD', 'full', ds, head, h)); b = vec(('SGD', 'h720', ds, head, h))
                na = int((~np.isfinite(a)).sum()); a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
                c = cell(a, b); c.update(dataset=NAME[ds], head=head, H=h, dropped=na); C.append(c)
    for c, p in zip(C, holm([c['welch_p'] for c in C])): c['holm_p'] = p
    mm = np.array([c['min_margin_pct'] for c in C])
    print(f'   Welch p<0.05: {sum(c["welch_p"] < 0.05 for c in C)}  Holm-significant: {sum(c["holm_p"] < 0.05 for c in C)}  '
          f'full lower: {sum(c["d_pct"] < 0 for c in C)}/24  equivalent at 1/2/3/5%: {[int((mm <= m).sum()) for m in (1, 2, 3, 5)]}')
    for c in C:
        print(f'   {c["dataset"]:8s} {c["head"]:6s} H={c["H"]:3d}  n_full={5 - c["dropped"]}  d={c["d_pct"]:+6.2f}%  p={c["welch_p"]:.4f} holm={c["holm_p"]:.3f}  min margin {c["min_margin_pct"]:.2f}%')
    for head in ('linear', 'mlp', 'all'):
        print(f'   mean |d| {head:6s} ' + '  '.join(f'H={h}: {np.mean([abs(c["d_pct"]) for c in C if c["H"] == h and (head == "all" or c["head"] == head)]):.2f}' for h in HS))
    print('   SGD level vs Adam, full set and {720}, median over 24 cells (finite runs):')
    for arm in ('full', 'h720'):
        d = []
        for ds in NAME:
            for hd in ('linear', 'mlp'):
                for h in HS:
                    a = vec(('SGD', arm, ds, hd, h)); d.append(100 * (a[np.isfinite(a)].mean() / vec(('Adam', arm, ds, hd, h)).mean() - 1))
        print(f'     {arm:5s} median {np.median(d):+.2f}%  range {min(d):+.2f}..{max(d):+.2f}%')
    # Where clipping acts: per-seed test MSE under Adam + clip 1.0 equal (|diff| < 1e-9) to the Adam run.
    print('\n== Adam + clip 1.0: runs whose per-seed MSE equals the Adam run (clipping never active in effect), per arm, dataset and head (5 seeds x 4 H)')
    for arm in ('full', 'h720'):
        for ds in NAME:
            for hd in ('linear', 'mlp'):
                same = sum(abs(v[('Adam+clip', arm, ds, hd, h)][s] - v[('Adam', arm, ds, hd, h)][s]) < 1e-9 for h in HS for s in SEEDS)
                print(f'   {arm:5s} {NAME[ds]:8s} {hd:6s} {same}/20 equal to Adam')


def part2():
    GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    CELLS = [(ds, hd) for ds in NAME for hd in ('linear', 'mlp')]
    lr = {c: float(open(os.path.join(RUNS, f'lr_{c[1]}_{c[0]}.txt')).read()) for c in CELLS}
    lro = {c: float(open(os.path.join(RUNS, f'lrown_{c[1]}_{c[0]}.txt')).read()) for c in CELLS}

    def load(fn, pat):
        out = []
        for r in csv.DictReader(open(os.path.join(RUNS, fn))):
            if r['windows'] != 'per_horizon': continue
            m = re.search(pat, r['setting'])
            if m: out.append((m.groupdict(), int(r['H']), float(r['mse']), r['mse']))
        return out
    shared = {}  # (arm, ds, hd, H, seed) -> (mse, raw string)
    for g, H, x, raw in load('result_windows.csv', r'_pl720_(?P<head>linear|mlp)_Csgd_(?P<arm>full|h720)_(?P=head)_(?P<ds>etth1|etth2|weather)_\d+_horizons_[0-9_]+_seed(?P<seed>\d+)$'):
        shared[(g['arm'], g['ds'], g['head'], H, int(g['seed']))] = (x, raw)
    own = {}
    for g, H, x, raw in load('own_result_windows.csv', r'_pl720_(?P<head>linear|mlp)_Csgdown_h720_(?P=head)_(?P<ds>etth1|etth2|weather)_\d+_horizons_720_seed(?P<seed>\d+)$'):
        own[(g['ds'], g['head'], H, int(g['seed']))] = (x, raw)
    grid = {}
    for g, H, x, raw in load('gridown_result_windows.csv', r'_pl720_(?P<head>linear|mlp)_Cgridown(?P<lr>[0-9.]+)_(?P=head)_(?P<ds>etth1|etth2|weather)_\d+_horizons_720_seed(?P<seed>\d+)$'):
        grid[(g['ds'], g['head'], float(g['lr']), H)] = (x, raw)
    assert len(own) == 6 * 4 * 5, len(own)
    print('LR per cell (full set, shared by the E-OPT {720} arm -> {720} own):')
    up = 0
    for c in CELLS:
        st = GRID.index(lro[c]) - GRID.index(lr[c]); up += st >= 1
        print(f'   {NAME[c[0]]:8s} {c[1]:6s} {lr[c]:g} -> {lro[c]:g}  ({st:+d} grid steps)')
    ge = sum(lro[c] >= lr[c] for c in CELLS)
    print(f'P7 ({{720}} LR >= full-set LR in 6/6 and larger by >= 1 step in >= 4/6): {"HELD" if ge == 6 and up >= 4 else "NOT HELD"} ({ge}/6 >=, {up}/6 larger)')
    c1 = [(c, H) for c in CELLS for H in HS if (c + (lro[c], H)) in grid and own[c + (H, 2023)][1] != grid[c + (lro[c], H)][1]]
    c2 = [(c, H) for c in CELLS for H in HS if (c + (lr[c], H)) in grid and shared[('h720',) + c + (H, 2023)][1] != grid[c + (lr[c], H)][1]]
    n2 = sum((c + (lr[c], H)) in grid for c in CELLS for H in HS)
    eqc = [c for c in CELLS if lro[c] == lr[c]]
    c3 = [(c, H, s) for c in eqc for H in HS for s in SEEDS if own[c + (H, s)][1] != shared[('h720',) + c + (H, s)][1]]
    print(f'C1 own seed 2023 = grid at chosen LR: {24 - len(c1)}/24 rows identical {c1[:3]}')
    print(f'C2 grid at shared LR = shared-LR {{720}} seed 2023: {n2 - len(c2)}/{n2} rows identical {c2[:3]}')
    print(f'C3 own = shared {{720}} where the LR is the same ({", ".join(NAME[c[0]] + "/" + c[1] for c in eqc)}): {len(eqc) * 20 - len(c3)}/{len(eqc) * 20} rows identical {c3[:3]}')
    C = []; dropped = []
    for c in CELLS:
        for H in HS:
            a = [shared[('full',) + c + (H, s)][0] for s in SEEDS]; b = [own[c + (H, s)][0] for s in SEEDS]
            fa = [x for x in a if math.isfinite(x)]; fb = [x for x in b if math.isfinite(x)]
            if len(fa) < 5 or len(fb) < 5: dropped.append((c, H, 5 - len(fa), 5 - len(fb)))
            r = cell(np.array(fa), np.array(fb)); r.update(dataset=NAME[c[0]], head=c[1], H=H, lr_full=lr[c], lr_h720=lro[c], n_a=len(fa), n_b=len(fb)); C.append(r)
    for r, p in zip(C, holm([r['welch_p'] for r in C])): r['holm_p'] = p
    mm = np.array([r['min_margin_pct'] for r in C])
    print(f'non-finite runs dropped: {dropped if dropped else "none"}')
    print(f'== SGD, own LR per arm: 24 cells; Welch p<0.05: {sum(r["welch_p"] < 0.05 for r in C)}; Holm-significant: {sum(r["holm_p"] < 0.05 for r in C)} '
          f'(full lower {sum(r["holm_p"] < 0.05 and r["d_pct"] < 0 for r in C)}); equivalent at 1/2/3/5%: {[int((mm <= m).sum()) for m in (1, 2, 3, 5)]}; '
          f'median MDE {np.median([r["mde80_pct"] for r in C]):.2f}%')
    for r in C:
        print(f'   {r["dataset"]:8s} {r["head"]:6s} H={r["H"]:3d}  full {r["mse_a"]:.4f}  {{720}} {r["mse_b"]:.4f}  d={r["d_pct"]:+7.2f}%  p={r["welch_p"]:.3g} holm={r["holm_p"]:.3g}  min margin {r["min_margin_pct"]:.2f}%')
    m96 = lambda hd: np.mean([abs(r['d_pct']) for r in C if r['H'] == 96 and (hd == 'all' or r['head'] == hd)])
    print('mean |d| by H (all; Linear; MLP): ' + '  '.join(f'H={h}: {np.mean([abs(r["d_pct"]) for r in C if r["H"] == h]):.2f} '
          f'({np.mean([abs(r["d_pct"]) for r in C if r["H"] == h and r["head"] == "linear"]):.2f}; {np.mean([abs(r["d_pct"]) for r in C if r["H"] == h and r["head"] == "mlp"]):.2f})' for h in HS))
    nh = sum(r['holm_p'] < 0.05 for r in C)
    print(f'P8 (< 12 Holm-significant and mean |d| at H=96 < 5.27%): {"HELD" if nh < 12 and m96("all") < 5.27 else "NOT HELD"} ({nh} Holm, mean |d| at 96 {m96("all"):.2f}%)')
    lin_full_lower = [r for r in C if r['head'] == 'linear' and r['holm_p'] < 0.05 and r['d_pct'] < 0]
    print(f'P9a (Linear: no cell with the full set Holm-significantly lower): {"HELD" if not lin_full_lower else "NOT HELD"} ({len(lin_full_lower)} cells)')
    print(f'P9b (MLP mean |d| at H=96 > Linear): {"HELD" if m96("mlp") > m96("linear") else "NOT HELD"} (MLP {m96("mlp"):.2f}%, Linear {m96("linear"):.2f}%)')
    keys = ['dataset', 'head', 'H', 'lr_full', 'lr_h720', 'n_a', 'n_b'] + [k for k in C[0] if k not in ('dataset', 'head', 'H', 'lr_full', 'lr_h720', 'n_a', 'n_b')]
    with open(os.path.join(args.out, 'optimizer_study_own_lr_cells.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in C: w.writerow({k: (f'{x:.6g}' if isinstance(x, float) else x) for k, x in r.items()})


def part3():
    files = sorted(glob.glob(os.path.join(RUNS, 'update_norms_*_etth1.csv')))
    if not files:
        print('no update_norms_*_etth1.csv in --runs: part 3 skipped'); return
    print('per-row RMS norm ratio, rows of steps 1-96 / rows of steps 337-720 (update; gradient):')
    for fn in files:
        R = list(csv.DictReader(open(fn)))
        s1 = next(r for r in R if r['step_kind'] == 'step' and r['step'] == '1')
        e1 = next(r for r in R if r['step_kind'] == 'epoch_mean' and r['epoch'] == '1')
        print(f'   {os.path.basename(fn)[13:-4]:18s} step 1: {float(s1["ratio_upd_perrow_first_last"]):.2f}; {float(s1["ratio_grad_perrow_first_last"]):.2f}   '
              f'epoch-1 mean: {float(e1["ratio_upd_perrow_first_last"]):.2f}; {float(e1["ratio_grad_perrow_first_last"]):.2f}')


print('######## Part 1: shared learning rate')
part1()
print('\n######## Part 2: SGD, own learning rate for {720}')
part2()
print('\n######## Part 3: update norms of W_max (ETTh1, seed 2023, full set)')
part3()
