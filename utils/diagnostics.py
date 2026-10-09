"""Per-horizon training diagnostics: per-epoch loss and gradient-norm logging for every horizon.

LOGGING ONLY. Nothing in this module touches the forward pass, the loss, the
backward pass, the optimizer, the learning-rate schedule, early stopping or
the random-number stream, so a run with ``--diag_dir`` set follows exactly the
same optimization path as one without it (verified: the seed-2023 diagnostic runs reproduce the test MSE of the
same runs without --diag_dir).

The training loop accumulates the multi-horizon gradient by calling
``backward()`` once per horizon before a single optimizer step (Algorithm 1).
The gradient of a single horizon is therefore recovered exactly as the
difference of the accumulated ``.grad`` buffer before and after that
horizon's ``backward()``.

Per epoch and per horizon H the CSV records
  loss                      mean over batches of L(theta, H)
  grad_norm_total           mean over batches of || dL(theta,H)/dW_max ||_F
  grad_norm_shared          same for the parameters shared by all horizons
                            (conv kernel, and for the MLP head mlp1 weight+bias)
  grad_norm_bias            same for b_max (NaN when the output projection has no bias, e.g. the Linear head
                            with --linear_bias 0)
  cos_shared_vs_hmax        cosine between dL_H/dtheta_shared and dL_Hmax/dtheta_shared
                            on the same batch (1 = aligned, <0 = conflicting)
  cos_wmax_prefix_vs_hmax   cosine between dL_H/dW_max and dL_Hmax/dW_max restricted
                            to the S_y(H) rows both horizons read
  n_batches
  grad_norm_row_r           mean over batches of || dL(theta,H)/dW_max[r,:] ||_2,
                            r = 1..S_y^max (one-based, as in the manuscript)
"""
import csv
import os
import re

import torch


class DiagLogger:
    def __init__(self, exp, setting):
        args, model = exp.args, exp.model
        self.dir = args.diag_dir
        os.makedirs(self.dir, exist_ok=True)
        self.dataset = args.model_id.split('_')[0]
        self.head = args.model_type
        m = re.search(r'seed(\d+)', setting)
        self.seed = int(m.group(1)) if m else -1
        self.period = args.period_len
        self.h_max = max(args.horizons)

        # W_max / b_max: the dynamically sliced output projection of SparseTSF
        dyn = model.linear.linear if self.head == 'linear' else model.mlp2.linear
        self.wmax, self.bmax = dyn.weight, dyn.bias
        dyn_ids = {id(self.wmax), id(self.bmax)}
        self.shared = [p for p in model.parameters()
                       if p.requires_grad and id(p) not in dyn_ids]
        self.n_rows = self.wmax.shape[0]

        self.path = os.path.join(
            self.dir, f'diag_{self.dataset}_{self.head}_seed{self.seed}.csv')
        self.cols = (['dataset', 'head', 'seed', 'epoch', 'horizon', 'loss',
                      'grad_norm_total', 'grad_norm_shared', 'grad_norm_bias',
                      'cos_shared_vs_hmax', 'cos_wmax_prefix_vs_hmax', 'n_batches']
                     + [f'grad_norm_row_{r}' for r in range(1, self.n_rows + 1)])
        with open(self.path, 'w', newline='') as f:
            csv.writer(f).writerow(self.cols)
        self.acc = {}
        self._ref_w = self._ref_s = None
        print(f'[diag] logging per-horizon loss / gradient norms to {self.path} '
              f'(W_max shape {tuple(self.wmax.shape)}, '
              f'{sum(p.numel() for p in self.shared)} shared parameters)')

    @staticmethod
    def _g(p):
        return p.grad.detach() if p.grad is not None else torch.zeros_like(p)

    def before_backward(self, H):
        self._w0 = self._g(self.wmax).clone()
        self._b0 = self._g(self.bmax).clone() if self.bmax is not None else None
        self._s0 = [self._g(p).clone() for p in self.shared]

    def after_backward(self, H, loss_value):
        gw = self._g(self.wmax) - self._w0
        gb = self._g(self.bmax) - self._b0 if self.bmax is not None else None
        gs = torch.cat([(self._g(p) - s0).reshape(-1)
                        for p, s0 in zip(self.shared, self._s0)])
        rows = gw.norm(dim=1)
        if H == self.h_max:                       # H_max is processed first
            self._ref_w, self._ref_s = gw.clone(), gs.clone()
        eps = 1e-30
        cos_s = torch.dot(gs, self._ref_s) / (gs.norm() * self._ref_s.norm() + eps)
        sy = H // self.period
        pw, pr = gw[:sy].reshape(-1), self._ref_w[:sy].reshape(-1)
        cos_w = torch.dot(pw, pr) / (pw.norm() * pr.norm() + eps)

        a = self.acc.setdefault(H, {'loss': 0., 'gw': 0., 'gs': 0., 'gb': 0.,
                                    'cos_s': 0., 'cos_w': 0.,
                                    'rows': torch.zeros_like(rows), 'n': 0})
        a['loss'] = a['loss'] + loss_value
        a['gw'] = a['gw'] + gw.norm()
        a['gs'] = a['gs'] + gs.norm()
        a['gb'] = a['gb'] + gb.norm() if gb is not None else float('nan')
        a['cos_s'] = a['cos_s'] + cos_s
        a['cos_w'] = a['cos_w'] + cos_w
        a['rows'] += rows
        a['n'] += 1

    def end_epoch(self, epoch):
        with open(self.path, 'a', newline='') as f:
            w = csv.writer(f)
            for H in sorted(self.acc, reverse=True):
                a = self.acc[H]
                n = a['n']
                w.writerow([self.dataset, self.head, self.seed, epoch, H,
                            float(a['loss']) / n, float(a['gw']) / n,
                            float(a['gs']) / n, float(a['gb']) / n,
                            float(a['cos_s']) / n, float(a['cos_w']) / n, n]
                           + [v / n for v in a['rows'].tolist()])
        self.acc = {}
