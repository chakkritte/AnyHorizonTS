"""Score saved checkpoints at EVERY horizon h = 1..H_model on the per-horizon test windows.

A model with output length H_model (720 for AnyHorizonTS and for the {720} model; the trained horizon for a SparseTSF model) is run
once per test origin. Its forecast at any h <= H_model is the first h steps of that output: for AnyHorizonTS this is the prefix identity
of the paper and, for h not a multiple of P, the definition of serving h (truncate the next servable horizon); for a single-horizon
SparseTSF model it is the truncation by which such a model serves a shorter horizon. For every h the score uses the windows of a test
loader built for h: the N - h + 1 origins whose h-step target lies in the test partition (the look-back is borrowed from the preceding
partition), as in the tables of the paper (Exp_Main.test_windows). Squared and absolute errors are summed over channels per origin and
step in float64; MSE(h) = sum over the first N - h + 1 origins and the first h steps / (n * h * C).

Usage (one process per code base, because both define models.SparseTSF.Model; run from the repository root):
  # AnyHorizonTS (this repository), full horizon set; --check runs the model directly at the next multiple of P of each h
  python reproduce/every_horizon/score_every_horizon.py --kind ahts --checkpoints ./checkpoints --set 720_336_192_96 \
         --dataset ETTh1 --heads linear mlp --out every_horizon/ahts_main.csv --check 24 50 250
  # one model trained at 720 alone (HORIZONS=720), truncated
  python reproduce/every_horizon/score_every_horizon.py --kind ahts --checkpoints ./checkpoints --set 720 --dataset ETTh1 \
         --out every_horizon/ahts_h720.csv
  # SparseTSF models (official or matched, see reproduce/matched_sparsetsf): --code is the SparseTSF checkout
  python reproduce/every_horizon/score_every_horizon.py --kind stsf --code ../SparseTSF --checkpoints ../SparseTSF/checkpoints_matched_G1 \
         --trained 96 192 336 720 --dataset ETTh1 --out every_horizon/stsf_matched_g1.csv
Options: --code (default: this repository for --kind ahts), --root_path (default: <code>/dataset/), --check H1 H2 ... (control of the
prefix identity: an AnyHorizonTS checkpoint is also run at pred_len = the next multiple of P of each h and the MSEs are compared),
--device cpu|cuda, --seeds, --threads.
Output rows (appended): kind,set_or_trained,dataset,head,seed,h,n_windows,mse,mae
"""
import argparse, os, sys
from types import SimpleNamespace
import numpy as np
import torch

REPO = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
p.add_argument('--kind', choices=['ahts', 'stsf'], required=True)
p.add_argument('--code', default=None, help='code base that defines the model and the data loader (default: this repository for ahts)')
p.add_argument('--checkpoints', required=True)
p.add_argument('--root_path', default=None, help='folder with the dataset files (default: <code>/dataset/)')
p.add_argument('--dataset', required=True, choices=['ETTh1', 'ETTh2', 'Electricity', 'Traffic', 'Weather', 'Solar-Energy'])
p.add_argument('--heads', nargs='+', default=['linear', 'mlp'])
p.add_argument('--seeds', nargs='+', type=int, default=[2023, 2024, 2025, 2026, 2027])
p.add_argument('--set', default='720_336_192_96', help='ahts: horizon set of the run name')
p.add_argument('--trained', nargs='+', type=int, default=[96, 192, 336, 720], help='stsf: trained horizons')
p.add_argument('--out', required=True)
p.add_argument('--check', nargs='*', type=int, default=[])
p.add_argument('--device', default='cpu')
p.add_argument('--bs', type=int, default=32)
p.add_argument('--threads', type=int, default=4)
p.add_argument('--setting_fmt', default=None, help='stsf: format of the checkpoint folder name (default: the official one)')
a = p.parse_args()
if a.code is None:
    if a.kind == 'stsf':
        p.error('--kind stsf needs --code (the SparseTSF checkout whose models/ and data_provider/ the checkpoints belong to)')
    a.code = REPO
torch.set_num_threads(a.threads)
CODE = os.path.realpath(a.code)
ROOT = a.root_path or os.path.join(CODE, 'dataset')
sys.path.insert(0, CODE)
from models.SparseTSF import Model  # noqa: E402
from data_provider.data_loader import Dataset_ETT_hour, Dataset_Custom, Dataset_Solar  # noqa: E402

L = 720
CFG = {'ETTh1': ('ETTh1', 'ETTh1', Dataset_ETT_hour, 'ETTh1.csv', 7, 24, 128),
       'ETTh2': ('ETTh2', 'ETTh2', Dataset_ETT_hour, 'ETTh2.csv', 7, 24, 128),
       'Electricity': ('Electricity', 'custom', Dataset_Custom, 'electricity.csv', 321, 24, 128),
       'Traffic': ('traffic', 'custom', Dataset_Custom, 'traffic.csv', 862, 24, 512),
       'Weather': ('weather', 'custom', Dataset_Custom, 'weather.csv', 21, 4, 128),
       'Solar-Energy': ('Solar', 'Solar', Dataset_Solar, 'solar_AL.txt', 137, 4, 512)}
mid, data, Data, path, enc_in, period, d_mlp = CFG[a.dataset]
dev = torch.device(a.device)
test = Data(root_path=ROOT.rstrip('/') + '/', flag='test', size=[L, 48, 96], features='M', data_path=path, target='OT', timeenc=0, freq='h')
X = np.ascontiguousarray(test.data_x, dtype=np.float32)
T, C = X.shape
n1 = T - L                                     # origins of the h = 1 loader; the h loader uses the first n1 - h + 1 of them
Xp = torch.from_numpy(np.concatenate([X, np.full((L, C), np.nan, np.float32)]))  # NaN beyond the partition: steps that no h uses
Xd = Xp.to(dev)
span = torch.arange(L, device=dev)


def errors(model, Hm):
    """S[o, i], A[o, i]: squared and absolute error summed over channels, origin o, step i (float64; 0 where the target is outside)."""
    S = np.zeros((n1, Hm)); A = np.zeros((n1, Hm))
    ahead = torch.arange(Hm, device=dev)
    with torch.no_grad():
        for s0 in range(0, n1, a.bs):
            idx = torch.arange(s0, min(n1, s0 + a.bs), device=dev)
            out = model(Xd[idx[:, None] + span])[:, :Hm, :]
            e = (out - Xd[idx[:, None] + L + ahead]).double()
            S[s0:s0 + len(idx)] = torch.nan_to_num((e ** 2).sum(-1), nan=0.0).cpu().numpy()
            A[s0:s0 + len(idx)] = torch.nan_to_num(e.abs().sum(-1), nan=0.0).cpu().numpy()
    return S, A


def per_h(S, A, Hm):
    cs, ca = np.cumsum(np.cumsum(S, 1), 0), np.cumsum(np.cumsum(A, 1), 0)
    h = np.arange(1, Hm + 1)
    n = n1 - h + 1
    return h, n, cs[n - 1, h - 1] / (n * h * C), ca[n - 1, h - 1] / (n * h * C)


def build(head, Hm):
    cfg = SimpleNamespace(seq_len=L, pred_len=Hm, enc_in=enc_in, period_len=period, d_model=d_mlp if head == 'mlp' else 128,
                          model_type=head, linear_bias=0)
    return Model(cfg).to(dev).eval()


new = not os.path.exists(a.out)
if os.path.dirname(a.out):
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
fo = open(a.out, 'a')
if new:
    fo.write('kind,set_or_trained,dataset,head,seed,h,n_windows,mse,mae\n')
for head in a.heads:
    for ii, seed in enumerate(a.seeds):
        jobs = []
        if a.kind == 'ahts':
            setting = f'{mid}_720_720_SparseTSF_{data}_ftM_sl720_pl720_{head}_test_{ii}_horizons_{a.set}_seed{seed}'
            jobs.append((a.set, setting, 720))
        else:
            fmt = a.setting_fmt or '{mid}_720_{H}_SparseTSF_{data}_ftM_sl720_pl{H}_{head}_test_{ii}_seed{seed}'
            for H in a.trained:
                jobs.append((str(H), fmt.format(mid=mid, H=H, data=data, head=head, ii=ii, seed=seed), H))
        for tag, setting, Hm in jobs:
            ck = os.path.join(a.checkpoints, setting, 'checkpoint.pth')
            assert os.path.exists(ck), ck
            m = build(head, Hm)
            m.load_state_dict(torch.load(ck, map_location=dev))
            S, A = errors(m, Hm)
            h, n, mse, mae = per_h(S, A, Hm)
            for k in range(Hm):
                fo.write(f'{a.kind},{tag},{a.dataset},{head},{seed},{h[k]},{n[k]},{mse[k]:.10g},{mae[k]:.10g}\n')
            fo.flush()
            for hc in a.check:                       # direct run at the next multiple of P, truncated to hc: the prefix identity
                Hs = -(-hc // period) * period
                m.pred_len, m.seg_num_y = Hs, Hs // period
                Sd, Ad = errors(m, Hs)
                _, _, msed, _ = per_h(Sd, Ad, Hs)
                print(f'CHECK {setting} h={hc} (pass at {Hs}): prefix {mse[hc - 1]:.10f} direct {msed[hc - 1]:.10f} '
                      f'rel {abs(mse[hc - 1] - msed[hc - 1]) / msed[hc - 1]:.2e}', flush=True)
                m.pred_len, m.seg_num_y = Hm, Hm // period
            print(f'{a.kind} {tag} {a.dataset} {head} {seed}: h=1..{Hm}, n(h=1)={n[0]}, mse(h=96)={mse[95] if Hm >= 96 else float("nan"):.6f}', flush=True)
fo.close()
