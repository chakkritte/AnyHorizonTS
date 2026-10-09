import csv
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import SparseTSF
from utils.diagnostics import DiagLogger
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop

warnings.filterwarnings('ignore')

class LogCoshLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        loss = torch.log(torch.cosh(y_pred - y_true + 1e-12))
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {'SparseTSF': SparseTSF}
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, pred_len):
        data_set, data_loader = data_provider(self.args, flag, pred_len)
        return data_set, data_loader

    def _select_optimizer(self):
        if getattr(self.args, 'optim', 'adam') == 'adamw':  # the June 2026 runs
            return optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)  # as in SparseTSF

    def _select_criterion(self):
        if self.args.loss == "mae":
            criterion = nn.L1Loss()
        elif self.args.loss == "mse":
            criterion = nn.MSELoss()
        elif self.args.loss == "smooth":
            criterion = nn.SmoothL1Loss(beta=1.0)
        elif self.args.loss == "logcosh":
            criterion = LogCoshLoss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _, _, _, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def vali_all_horizons(self, vali_data, vali_loader, criterion):
        """--early_stop_on_all_horizons: mean over --horizons of the per-horizon validation MSE.

        One pass over the windows of the H_max validation loader (the same windows and batches as vali()). For every
        horizon H the model is run at pred_len = H and scored on the first H steps after the look-back, i.e. on
        batch_y_H of the loader (the loader returns the 720, 336, 192 and 96 step targets). Each horizon's loss is
        averaged over batches exactly as in vali(), so the H = 720 term equals vali()'s value. Returns the mean over
        the horizons and stores the per-horizon values in self.vali_per_horizon (logging).
        """
        hs = sorted(self.args.horizons)
        total_loss = {h: [] for h in hs}
        pred_len0 = self.model.pred_len
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y_720, batch_y_336, batch_y_192, batch_y_96, _, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                targets = {720: batch_y_720, 336: batch_y_336, 192: batch_y_192, 96: batch_y_96}
                f_dim = -1 if self.args.features == 'MS' else 0
                for h in hs:
                    self.args.pred_len = h
                    self.model.pred_len = h
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x)
                    else:
                        outputs = self.model(batch_x)
                    outputs = outputs[:, -h:, f_dim:]
                    batch_y = targets[h].float()[:, -h:, f_dim:].to(self.device)
                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()
                    total_loss[h].append(criterion(pred, true))
        self.args.pred_len = pred_len0
        self.model.pred_len = pred_len0
        self.vali_per_horizon = {h: float(np.average(total_loss[h])) for h in hs}
        self.model.train()
        return float(np.mean([self.vali_per_horizon[h] for h in hs]))

    def train(self, setting):

        print("horizons ", self.args.horizons)

        # one training loader at H_max = 720; each batch carries the targets of 720, 336, 192 and 96
        _, train_loader_720 = self._get_data(flag='train', pred_len=720)
        vali_data, vali_loader = self._get_data(flag='val', pred_len=720)
        test_data, test_loader = self._get_data(flag='test', pred_len=720)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader_720)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)

        # stepped only for --lradj TST; the default type3 schedule is set in adjust_learning_rate()
        if getattr(self.args, 'official_first_epoch_lr', False):
            # --official_first_epoch_lr (diagnostic): the scheduler of the official SparseTSF code, exactly as there. It is
            # never stepped with --lradj type3, but the constructor alone sets the optimizer lr to max_lr / 25 (div_factor)
            # for epoch 1 (and Adam beta1 to 0.95, cycle_momentum); adjust_learning_rate restores the full rate after epoch 1.
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                            steps_per_epoch=train_steps,
                                                            pct_start=self.args.pct_start,
                                                            epochs=self.args.train_epochs,
                                                            max_lr=self.args.learning_rate)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.train_epochs)

        f_dim = -1 if self.args.features == 'MS' else 0  # Calculate once outside loop

        # per-horizon diagnostics (logging only, see utils/diagnostics.py); inactive unless --diag_dir is given
        diag = DiagLogger(self, setting) if getattr(self.args, 'diag_dir', None) else None

        for epoch in range(self.args.train_epochs):
            if epoch == 0 and getattr(self.args, 'official_first_epoch_lr', False):
                _g = model_optim.param_groups[0]  # log only: the lr and betas that epoch 1 trains with
                print('official_first_epoch_lr: epoch-1 lr = {:.10g} (learning_rate {:.10g}, ratio {:.6g}), Adam betas = {}'.format(
                    _g['lr'], self.args.learning_rate, _g['lr'] / self.args.learning_rate, _g.get('betas')))
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x_720, batch_y_720, batch_y_336, batch_y_192, batch_y_96, batch_y_48, batch_y_24) in enumerate(train_loader_720):
                iter_count += 1
                model_optim.zero_grad()
                batch_x_720 = batch_x_720.float().to(self.device)
                batch_y_720 = batch_y_720.float().to(self.device)
                batch_y_336 = batch_y_336.float().to(self.device)
                batch_y_192 = batch_y_192.float().to(self.device)
                batch_y_96 = batch_y_96.float().to(self.device)

                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    cumulative_loss = 0

                    # multi-horizon gradient accumulation over the training horizon set (--horizons)
                    all_targets = {
                        720: batch_y_720,
                        336: batch_y_336,
                        192: batch_y_192,
                        96: batch_y_96,
                    }
                    pred_lengths = [pl for pl in [720, 336, 192, 96] if pl in self.args.horizons]
                    batch_y_targets = [all_targets[pl] for pl in pred_lengths]
                    if not pred_lengths:
                        pred_lengths = [self.args.pred_len]
                        batch_y_targets = [batch_y_720]

                    single_pass = getattr(self.args, 'single_pass', False)
                    if single_pass:
                        # --single_pass (diagnostic, default off = the paper runs): ONE forward and backward pass at H_max. By the prefix
                        # identity the forecast at a shorter horizon H is the first H steps of the pass at H_max, so the multi-horizon loss,
                        # sum over H in the horizon set of MSE(output[:, :H], target[:, :H]) with the per-horizon normalization 1 / (B C H),
                        # has the same gradient as the accumulated per-horizon passes below; the pass at H_max is the first of the
                        # descending horizon list.
                        h_max = pred_lengths[0]
                        assert h_max == max(pred_lengths) and diag is None, \
                            '--single_pass needs the horizons in descending order (H_max first) and cannot be combined with --diag_dir'
                        self.args.pred_len = h_max
                        self.model.pred_len = self.args.pred_len
                        outputs_max = self.model(batch_x_720)
                        total_loss = 0
                        for pred_len, batch_y in zip(pred_lengths, batch_y_targets):
                            loss = criterion(outputs_max[:, :pred_len, f_dim:], batch_y[:, -pred_len:, f_dim:].to(self.device))
                            total_loss = total_loss + loss
                        cumulative_loss = total_loss.item()
                        scaler.scale(total_loss).backward()

                    for pred_len, batch_y in ([] if single_pass else zip(pred_lengths, batch_y_targets)):
                        # slice the output projection to the first pred_len / period_len rows (dynamic layer)
                        self.args.pred_len = pred_len
                        self.model.pred_len = self.args.pred_len

                        outputs = self.model(batch_x_720)
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        loss = criterion(outputs, batch_y)
                        cumulative_loss += loss.item()

                        # gradients of all horizons accumulate; one optimizer step per batch below
                        if diag is not None:
                            diag.before_backward(pred_len)
                        scaler.scale(loss).backward()
                        if diag is not None:
                            diag.after_backward(pred_len, loss.detach())

                    train_loss.append(cumulative_loss)  # sum of the per-horizon losses

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                scaler.step(model_optim)
                scaler.update()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            if diag is not None:
                diag.end_epoch(epoch + 1)
            train_loss = np.average(train_loss)

            # validation and logging at H_max; early stopping monitors the H_max validation loss
            self.args.pred_len = 720
            self.model.pred_len = self.args.pred_len

            if getattr(self.args, 'early_stop_on_all_horizons', False):
                # early stopping and checkpoint selection on the mean of the per-horizon validation MSEs
                vali_loss = self.vali_all_horizons(vali_data, vali_loader, criterion)
                print('early_stop_on_all_horizons: epoch {} per-horizon vali MSE {} -> mean {:.7f}'.format(
                    epoch + 1, ', '.join('H={}: {:.7f}'.format(h, v) for h, v in sorted(self.vali_per_horizon.items())), vali_loss))
            else:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        return self.model

    def test_windows(self, setting, out_file, horizons=None):
        """Score the evaluation horizons of test() (96, 192, 336 and 720, whatever the training horizon set) on two
        sets of test windows and append the metrics to `out_file` (CSV).

        `horizons` (--score_horizons, default None = the four horizons above) replaces them by any horizons the model can
        serve: each a multiple of the period length and at most the model's H_max; the shared windows are then those of
        the largest horizon given.

        per_horizon: every origin whose L + H span lies in the test partition (N - L - H + 1 windows), i.e. the
                     windows of a data loader built for H, as used by the single-horizon baselines. These are the
                     numbers of the paper's tables.
        shared:      every origin whose L + H_max span lies in the test partition (N - L - H_max + 1 windows), i.e.
                     the windows of test(), whose loader is built once at H_max.
        Windows are cut from the test partition in chunks of --test_batch_size and the errors are accumulated in
        float64, so host and device memory stay bounded for any dataset and horizon.
        """
        test_data, _ = self._get_data(flag='test', pred_len=720)
        X = torch.from_numpy(np.ascontiguousarray(test_data.data_x)).float().to(self.device)
        Y = torch.from_numpy(np.ascontiguousarray(test_data.data_y)).float().to(self.device)
        f_dim = -1 if self.args.features == 'MS' else 0
        L, T, bs = self.args.seq_len, X.shape[0], self.args.test_batch_size
        pred_len0 = self.model.pred_len
        eval_h = [H for H in (96, 192, 336, 720) if H <= pred_len0]  # the horizons test() scores after training
        if horizons:
            eval_h = sorted(set(horizons))
            assert all(0 < H <= pred_len0 and H % self.args.period_len == 0 for H in eval_h), \
                'every horizon must be a multiple of period_len ({}) and at most H_max ({}): {}'.format(self.args.period_len, pred_len0, eval_h)
        h_max = max(eval_h)  # test() builds its loader at H_max = 720
        span = torch.arange(L, device=self.device)
        rows = []
        self.model.eval()
        with torch.no_grad():
            for H in eval_h:
                self.args.pred_len = H
                self.model.pred_len = H
                ahead = torch.arange(H, device=self.device)
                for windows, n in (('per_horizon', T - L - H + 1), ('shared', T - L - h_max + 1)):
                    se, ae, cnt = 0.0, 0.0, 0
                    for s0 in range(0, n, bs):
                        idx = torch.arange(s0, min(n, s0 + bs), device=self.device)
                        out = self.model(X[idx[:, None] + span])[:, -H:, f_dim:]
                        err = (out - Y[idx[:, None] + L + ahead][:, :, f_dim:]).double()
                        se += float((err ** 2).sum())
                        ae += float(err.abs().sum())
                        cnt += err.numel()
                    rows.append((setting, H, windows, n, se / cnt, ae / cnt))
                    print('{} H={} {} windows={} mse:{}, mae:{}'.format(setting, H, windows, n, se / cnt, ae / cnt))
        self.args.pred_len = pred_len0
        self.model.pred_len = pred_len0
        new_file = not os.path.exists(out_file)
        with open(out_file, 'a') as f:
            if new_file:
                f.write('setting,H,windows,n_windows,mse,mae\n')
            for r in rows:
                f.write('{},{},{},{},{:.10f},{:.10f}\n'.format(*r))

    def test(self, setting, test=0, pred_len=720):
        # Scores `pred_len` on the windows of the loader built once at H_max = 720 (N - L - 720 + 1 origins for every
        # horizon, the 'shared' windows). The per-horizon windows of the paper's tables are scored by test_windows().
        test_data, test_loader = self._get_data(flag='test', pred_len=720)

        self.args.pred_len = pred_len
        self.model.pred_len = self.args.pred_len

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth'), map_location=self.device))

        preds = []
        trues = []
        # --stream_metrics (evaluation memory only): accumulate MSE/MAE/RSE per batch in float64
        # instead of holding every prediction in RAM; training and the reported metrics are unchanged.
        stream = getattr(self.args, 'stream_metrics', False)
        acc = {'se': 0.0, 'ae': 0.0, 't': 0.0, 'tt': 0.0, 'n': 0}
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y_720, batch_y_336, batch_y_192, batch_y_96, _, _) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y_720 = batch_y_720.float().to(self.device)
                batch_y_336 = batch_y_336.float().to(self.device)
                batch_y_192 = batch_y_192.float().to(self.device)
                batch_y_96 = batch_y_96.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]

                if pred_len == 720:
                    batch_y = batch_y_720[:, -self.args.pred_len:, f_dim:].to(self.device)
                elif pred_len == 336:
                    batch_y = batch_y_336[:, -self.args.pred_len:, f_dim:].to(self.device)
                elif pred_len == 192:
                    batch_y = batch_y_192[:, -self.args.pred_len:, f_dim:].to(self.device)
                elif pred_len == 96:
                    batch_y = batch_y_96[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                if stream:
                    d = pred.astype(np.float64) - true.astype(np.float64)
                    acc['se'] += float(np.sum(d * d)); acc['ae'] += float(np.sum(np.abs(d)))
                    t64 = true.astype(np.float64)
                    acc['t'] += float(np.sum(t64)); acc['tt'] += float(np.sum(t64 * t64)); acc['n'] += d.size
                else:
                    preds.append(pred)
                    trues.append(true)
                if i % 20 == 0:  # forecast plots of every 20th batch in ./test_results/<setting>/
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
            exit()

        if stream:
            n = acc['n']
            mse = acc['se'] / n
            mae = acc['ae'] / n
            rse = np.sqrt(acc['se']) / np.sqrt(acc['tt'] - acc['t'] ** 2 / n)
        else:
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))

        # append one row per evaluation (shared windows) to --result_file (header written once)
        csv_file = getattr(self.args, 'result_file', 'result_shared.csv')
        file_exists = os.path.isfile(csv_file) and os.path.getsize(csv_file) > 0
        with open(csv_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["Setting", "MSE", "MAE", "RSE"])
            writer.writerow([setting, mse, mae, rse])
        return mse, mae
