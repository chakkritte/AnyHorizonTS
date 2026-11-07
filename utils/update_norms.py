"""--update_norm_log (optimizer study): per-block L2 norms of the parameter update and of the gradient of the output projection.

LOGGING ONLY: it reads the parameters and the gradients around optimizer.step() and changes nothing, so a run with the
option follows the same optimization path as one without it.

W_max is the output projection of the model built at H_max: the weight of the Linear head (model.linear.linear, shape
S_y x S_x) or of the last layer of the MLP head (model.mlp2.linear, shape S_y x d_model), with S_y = H_max / P rows. Row r
(zero-based) produces the output segment r, i.e. the output steps r*P+1 ... (r+1)*P, because the model reshapes the
(batch*channel, P, S_y) output to (batch, channel, S_y*P) after a permute (segment-major). Hence the horizon blocks in
output-step units map to rows by dividing by the period length P:
    block 1-96    -> rows 0 .. 96/P - 1
    block 97-192  -> rows 96/P .. 192/P - 1
    block 193-336 -> rows 192/P .. 336/P - 1
    block 337-720 -> rows 336/P .. 720/P - 1
(P = 24 on ETTh1/ETTh2, 4 on Weather). The MLP head has an output bias b_max, whose entries of the block's rows are included
in the block norm (the Linear head has none). Norms are taken over the whole block (Frobenius norm, weight and bias together).

CSV columns: step_kind (step = one of the first 50 optimizer steps of the run; epoch_mean = mean over the steps of the epoch;
epoch_last = last step of the epoch), epoch, step (global optimizer step; for epoch_mean the number of steps averaged),
then for each block b in 1-96, 97-192, 193-336, 337-720: upd_<b> and grad_<b> (block Frobenius norms), then
ratio_upd_first_last = upd_1-96 / upd_337-720 and ratio_grad_first_last (ratios of the block norms, which grow with the number of
rows of a block), and ratio_upd_perrow_first_last / ratio_grad_perrow_first_last: the same ratios of the per-row RMS norms
(block norm / sqrt(number of rows), the quantity comparable between blocks of different size).
The gradient is the one the optimizer receives (after clipping, if any).
"""
import csv
import os

import torch

BLOCKS = [(1, 96), (97, 192), (193, 336), (337, 720)]


class UpdateNormLogger:
    def __init__(self, exp, setting):
        self.exp = exp
        a = exp.args
        m = exp.model
        self.wmax = m.linear.linear if a.model_type == 'linear' else m.mlp2.linear
        P = a.period_len
        S_y = self.wmax.weight.shape[0]
        self.slices = [(lo - 1) // P for lo, _ in BLOCKS], [hi // P for _, hi in BLOCKS]
        self.slices = list(zip(*self.slices))
        assert self.slices[-1][1] == S_y, 'W_max has {} rows, the last block ends at row {}'.format(S_y, self.slices[-1][1])
        base = a.update_norm_log
        self.path = base
        os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)
        self.setting = setting
        self.fh = open(base, 'a', newline='')
        self.w = csv.writer(self.fh)
        if os.path.getsize(base) == 0:
            cols = ['setting', 'step_kind', 'epoch', 'step'] + ['upd_%d-%d' % b for b in BLOCKS] + ['grad_%d-%d' % b for b in BLOCKS] \
                + ['ratio_upd_first_last', 'ratio_grad_first_last',
                                                                                 'ratio_upd_perrow_first_last', 'ratio_grad_perrow_first_last']
            self.w.writerow(cols)
        self.gstep = 0
        self.n_ep = 0
        self.epoch = 0

    def _blocks(self, tensors):
        """Frobenius norm per block of (weight, bias or None) tensors laid out by rows."""
        out = []
        for lo, hi in self.slices:
            sq = (tensors[0][lo:hi] ** 2).sum()
            if tensors[1] is not None:
                sq = sq + (tensors[1][lo:hi] ** 2).sum()
            out.append(sq.sqrt())
        return torch.stack(out)

    def _pair(self, w, b):
        return (w.detach(), None if b is None else b.detach())

    def start_epoch(self, epoch):
        self.epoch = epoch
        self.n_ep = 0
        self.sum_u = torch.zeros(len(BLOCKS), device=self.wmax.weight.device)
        self.sum_g = torch.zeros_like(self.sum_u)

    def before_step(self):
        w, b = self.wmax.weight, self.wmax.bias
        self.g = self._blocks((w.grad, None if b is None else b.grad))
        self.w0 = w.detach().clone()
        self.b0 = None if b is None else b.detach().clone()

    def after_step(self):
        w, b = self.wmax.weight, self.wmax.bias
        u = self._blocks((w.detach() - self.w0, None if b is None else b.detach() - self.b0))
        self.last_u, self.last_g = u, self.g
        self.gstep += 1
        self.n_ep += 1
        self.sum_u += u
        self.sum_g += self.g
        if self.gstep <= 50:
            self._row('step', self.epoch, self.gstep, u, self.g)

    def _row(self, kind, epoch, step, u, g):
        u = [float(x) for x in u]
        g = [float(x) for x in g]
        k = ((self.slices[-1][1] - self.slices[-1][0]) / (self.slices[0][1] - self.slices[0][0])) ** 0.5  # per-row RMS ratio = block-norm ratio * sqrt(rows_last / rows_first)
        self.w.writerow([self.setting, kind, epoch, step] + ['%.8g' % x for x in u] + ['%.8g' % x for x in g]
                        + ['%.6g' % (u[0] / u[-1] if u[-1] > 0 else float('nan')), '%.6g' % (g[0] / g[-1] if g[-1] > 0 else float('nan')),
                        '%.6g' % (k * u[0] / u[-1] if u[-1] > 0 else float('nan')), '%.6g' % (k * g[0] / g[-1] if g[-1] > 0 else float('nan'))])
        self.fh.flush()

    def end_epoch(self, epoch):
        if self.n_ep:
            self._row('epoch_mean', epoch, self.n_ep, self.sum_u / self.n_ep, self.sum_g / self.n_ep)
            self._row('epoch_last', epoch, self.gstep, self.last_u, self.last_g)
