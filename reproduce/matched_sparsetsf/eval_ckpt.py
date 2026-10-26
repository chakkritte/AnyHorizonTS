"""Score SparseTSF checkpoints (official code, see this folder) on the per-horizon test windows: N - L - H + 1 origins, the windows of
the official test() and of the tables of the paper, streaming on the device with float64 sums.

Usage (from anywhere):
  python reproduce/matched_sparsetsf/eval_ckpt.py --code ../SparseTSF --checkpoints ../SparseTSF/checkpoints_matched_G1 \
         --cells "ETTh1:linear,mlp:96,192,336,720;Traffic:mlp:96" --out scores_eval.csv
<cells> = dataset:heads:horizons separated by ';' (datasets ETTh1, ETTh2, Electricity, Traffic, Weather, Solar-Energy).
Checkpoint names are those of the official run_longExp.py (model_id ..._seed{2023+i}). Where the official test() ran, its output
(--result_txt, default <code>/result.txt) is used as a cross-check of the score. Output columns:
dataset, head, H, seed, n_windows, params, mse, mae, resulttxt_mse, resulttxt_mae.
"""
import argparse, csv, os, re, sys
import torch
from types import SimpleNamespace

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('--code', required=True, help='the SparseTSF checkout (its models/ and data_provider/ are used)')
ap.add_argument('--checkpoints', default=None, help='checkpoint folder (default: <code>/checkpoints)')
ap.add_argument('--root_path', default=None, help='folder with the dataset files (default: <code>/dataset)')
ap.add_argument('--cells', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--result_txt', default=None, help='output of the official test() for the cross-check (default: <code>/result.txt)')
ap.add_argument('--seeds', nargs='+', type=int, default=[2023, 2024, 2025, 2026, 2027])
ap.add_argument('--device', default=None, help='cuda or cpu (default: cuda if available)')
a = ap.parse_args()
C = os.path.realpath(a.code)
CK = os.path.realpath(a.checkpoints) if a.checkpoints else os.path.join(C, 'checkpoints')
ROOT = os.path.realpath(a.root_path) if a.root_path else os.path.join(C, 'dataset')
OUT = os.path.realpath(a.out)
sys.path.insert(0, C)
from models.SparseTSF import Model  # noqa: E402
from data_provider.data_loader import Dataset_ETT_hour, Dataset_Custom, Dataset_Solar  # noqa: E402

dev = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
L, SEEDS = 720, tuple(a.seeds)
# dataset: (model_id prefix, --data, data class, file, enc_in, period_len, MLP d_model) of the official scripts/SparseTSF/*/*.sh
CFG = {'ETTh1': ('ETTh1', 'ETTh1', Dataset_ETT_hour, 'ETTh1.csv', 7, 24, 128),
       'ETTh2': ('ETTh2', 'ETTh2', Dataset_ETT_hour, 'ETTh2.csv', 7, 24, 128),
       'Electricity': ('Electricity', 'custom', Dataset_Custom, 'electricity.csv', 321, 24, 128),
       'Traffic': ('traffic', 'custom', Dataset_Custom, 'traffic.csv', 862, 24, 512),
       'Weather': ('weather', 'custom', Dataset_Custom, 'weather.csv', 21, 4, 128),
       'Solar-Energy': ('Solar', 'Solar', Dataset_Solar, 'solar_AL.txt', 137, 4, 512)}
own = {}
rt = a.result_txt or os.path.join(C, 'result.txt')
if os.path.exists(rt):
    lines = open(rt).read().split('\n')
    for k, line in enumerate(lines):
        m = re.search(r'mse:([0-9.]+), mae:([0-9.]+)', lines[k + 1]) if k + 1 < len(lines) else None
        if line.strip() and m and '_seed' in line:
            own[line.strip()] = (float(m.group(1)), float(m.group(2)))
rows, worst, missing = [], 0.0, []
for cell in a.cells.split(';'):
    ds, heads, hs = cell.split(':')
    mid, data, Data, path, enc_in, period, d_mlp = CFG[ds]
    test = Data(root_path=ROOT + '/', flag='test', size=[L, 48, 96], features='M', data_path=path, target='OT', timeenc=0, freq='h')
    X = torch.tensor(test.data_x, dtype=torch.float32, device=dev)
    T, span = X.shape[0], torch.arange(L, device=dev)
    for head in heads.split(','):
        for H in [int(h) for h in hs.split(',')]:
            ahead = torch.arange(H, device=dev)
            for ii, seed in enumerate(SEEDS):
                setting = f'{mid}_720_{H}_SparseTSF_{data}_ftM_sl720_pl{H}_{head}_test_{ii}_seed{seed}'
                ck = f'{CK}/{setting}/checkpoint.pth'
                if not os.path.exists(ck):
                    missing.append(setting); continue
                m = Model(SimpleNamespace(seq_len=L, pred_len=H, enc_in=enc_in, period_len=period,
                                          d_model=d_mlp if head == 'mlp' else 128, model_type=head)).to(dev).eval()
                m.load_state_dict(torch.load(ck, map_location=dev))
                n = T - L - H + 1
                se = ae = 0.0; cnt = 0
                with torch.no_grad():
                    for s0 in range(0, n, 64):
                        idx = torch.arange(s0, min(n, s0 + 64), device=dev)
                        e = (m(X[idx[:, None] + span]) - X[idx[:, None] + L + ahead]).double()
                        se += float((e ** 2).sum()); ae += float(e.abs().sum()); cnt += e.numel()
                mse, mae = se / cnt, ae / cnt
                ref = own.get(setting)
                if ref: worst = max(worst, abs(mse - ref[0]), abs(mae - ref[1]))
                rows.append(dict(dataset=ds, head=head, H=H, seed=seed, n_windows=n, params=sum(p.numel() for p in m.parameters()),
                                 mse=mse, mae=mae, resulttxt_mse=ref[0] if ref else '', resulttxt_mae=ref[1] if ref else ''))
            print(ds, head, H, 'done', flush=True)
with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['dataset', 'head', 'H', 'seed', 'n_windows', 'params', 'mse', 'mae', 'resulttxt_mse', 'resulttxt_mae'])
    w.writeheader(); w.writerows(rows)
print(f'{len(rows)} checkpoints scored on {dev}; missing {len(missing)}: {missing[:6]}')
print(f'cross-check against the official test() where it ran ({sum(1 for r in rows if r["resulttxt_mse"] != "")} rows): max |difference| = {worst:.1e}')
