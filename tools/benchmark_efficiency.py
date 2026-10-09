#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_efficiency.py
=======================
Measure the efficiency metrics of the efficiency table of the AnyHorizonTS
paper -- Parameters, MACs, Max Memory, and per-iteration / epoch time -- for the
SparseTSF backbone and the AnyHorizonTS variants, on whatever accelerator is present.

The same script is meant to be run twice and the two outputs combined:

    # on the RTX 3090 box
    python tools/benchmark_efficiency.py --csv efficiency_rtx3090.csv

    # on the Apple M4 (Metal / MPS)
    python tools/benchmark_efficiency.py --csv efficiency_m4.csv

It is fully self-contained: Parameters and MACs are hardware-independent and are
computed with a built-in analytic counter (no ptflops/thop needed). At L=H=720
and enc_in=862 the counter gives 34.14 M MACs for the Linear head (925 parameters,
no output bias, as in SparseTSF/Linear) and 174.4 M for the MLP head (7.86 K
parameters). The counter counts the multiply-accumulates of the weight layers
only; ptflops, with which the efficiency table of the paper is counted, also
counts bias additions and activations and gives 183.0 M for the MLP head (the
Linear head is 34.14 M either way). Max Memory and per-iteration
time are measured on the active device with synthetic GPU-resident batches, so
they exclude data loading; the epoch times of the paper are the
"Epoch: k cost time" lines of actual training runs (training batches only).

Defaults follow that table: Traffic dataset, L = H = 720, enc_in = 862,
period_len = 24, hidden size d_model = 128, batch size 256.

NOTE on devices:
  * CUDA  -> peak memory via torch.cuda.max_memory_allocated() (true peak).
  * MPS   -> torch.mps.current_allocated_memory()/driver_allocated_memory()
             (no reset/peak API exists on MPS in current PyTorch, so the MPS
             figure is the live allocation measured right after the op and is an
             approximation of the peak -- it is flagged as such in the output).
  * CPU   -> process RSS delta via psutil if available, else reported as n/a.

The script ONLY benchmarks the models the authors legitimately re-measure
(SparseTSF/AnyHorizonTS, Linear and MLP heads). To add a baseline (DLinear,
PatchTST, ...) register it in build_model(); the timing/memory/MAC machinery is
model-agnostic.
"""

import argparse
import csv as csvlib
import os
import platform
import statistics
import sys
from time import perf_counter
from types import SimpleNamespace

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Make the repo importable regardless of where the script is launched from.
# (This file lives in <repo>/tools/; models/, exp/, etc. are one level up.)
# --------------------------------------------------------------------------- #
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def make_config(args, model_type):
    """Minimal config object exposing the attributes SparseTSF.Model reads."""
    return SimpleNamespace(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        enc_in=args.enc_in,
        period_len=args.period_len,
        d_model=args.d_model,
        model_type=model_type,
    )


def build_model(name, args):
    """Return an nn.Module for the requested row name.

    Recognised names: 'stsf_linear', 'stsf_mlp', 'ahts_linear', 'ahts_mlp'.
    AnyHorizonTS uses the *same* network as SparseTSF (one shared W_max sliced at
    runtime); the only difference is the training loop, handled later. So the
    module is identical -- 'ahts_*' rows reuse the SparseTSF module of the
    matching head.
    """
    from models.SparseTSF import Model as SparseTSF

    head = "mlp" if name.endswith("mlp") else "linear"
    cfg = make_config(args, head)
    return SparseTSF(cfg).float()


PRETTY = {
    "stsf_linear": "SparseTSF/Linear",
    "stsf_mlp": "SparseTSF/MLP",
    "ahts_linear": "AnyHorizonTS/Linear",
    "ahts_mlp": "AnyHorizonTS/MLP",
}
IS_ATS = {"ahts_linear": True, "ahts_mlp": True, "stsf_linear": False, "stsf_mlp": False}


# --------------------------------------------------------------------------- #
# Parameters & MACs (hardware-independent; computed once on CPU)
# --------------------------------------------------------------------------- #
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def count_macs(model, x_shape, pred_len, period_len):
    """Analytic multiply-accumulate count for one forward pass (batch dim of
    x_shape should be 1 to match the per-sample convention of ptflops).

    Counts every linear and 1-D convolution layer, which is the entire compute of
    these models. The sliceable layers (DynamicLinear, DynamicPointConv1d) apply
    their weights functionally, so they are hooked themselves; their inner
    nn.Linear / nn.Conv1d never run. Uses the *actual* input/output tensor shapes
    seen at run time, so dynamic row-slicing is handled correctly.
    """
    from models.modules.dynamic_ops import DynamicLinear, DynamicPointConv1d

    total = {"macs": 0}
    handles = []

    def hook(module, inputs, output):
        if not isinstance(output, torch.Tensor):
            return
        x = inputs[0]
        if isinstance(module, (nn.Linear, DynamicLinear)):   # macs = out_elems * in_features
            total["macs"] += output.numel() * x.shape[-1]
        else:                                                # Conv1d: macs = out_elems * (in/groups * k)
            conv = module.conv if isinstance(module, DynamicPointConv1d) else module
            total["macs"] += output.numel() * (x.shape[1] // conv.groups) * conv.weight.shape[2]

    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, DynamicLinear, DynamicPointConv1d)):
            handles.append(m.register_forward_hook(hook))

    was_training = model.training
    model.eval()
    model.pred_len = pred_len                   # ensure the head slices to this horizon
    x = torch.zeros(*x_shape)
    model(x)
    for h in handles:
        h.remove()
    model.train(was_training)
    return total["macs"]


# --------------------------------------------------------------------------- #
# Device helpers
# --------------------------------------------------------------------------- #
def pick_device(requested):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_label(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Silicon GPU (MPS)"
    return f"CPU ({platform.processor() or platform.machine()})"


def dev_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def dev_mem_reset(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def dev_peak_mem_mb(device):
    """Peak (CUDA) or live (MPS, approximate) device memory in MB. CPU -> nan."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    if device.type == "mps":
        if hasattr(torch.mps, "current_allocated_memory"):
            return torch.mps.current_allocated_memory() / 1024 ** 2
        return float("nan")
    return float("nan")


# --------------------------------------------------------------------------- #
# Timing & memory measurement
# --------------------------------------------------------------------------- #
def time_fn(fn, device, warmup, iters):
    """Return (mean_ms, std_ms, p50_ms) per call of fn(), with sync + warmup."""
    for _ in range(warmup):
        fn()
    dev_sync(device)
    samples = []
    for _ in range(iters):
        dev_sync(device)
        t0 = perf_counter()
        fn()
        dev_sync(device)
        samples.append((perf_counter() - t0) * 1e3)
    mean = statistics.mean(samples)
    std = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    p50 = statistics.median(samples)
    return mean, std, p50


def measure_peak_mem_mb(fn, device):
    """Peak device memory around a single call of fn() (MB)."""
    dev_mem_reset(device)
    dev_sync(device)
    fn()
    dev_sync(device)
    return dev_peak_mem_mb(device)


# --------------------------------------------------------------------------- #
# Forward / training-step closures
# --------------------------------------------------------------------------- #
def make_inference_fn(model, x, pred_len):
    model.eval()
    model.pred_len = pred_len

    @torch.no_grad()
    def _fn():
        return model(x)

    return _fn


def make_train_step_fn(model, x, y_full, horizons, optimizer, use_amp, device):
    """One AnyHorizonTS / SparseTSF training iteration.

    horizons is a list; for SparseTSF it is a single horizon, for AnyHorizonTS it is
    the full set. The loop slices the head to each horizon, accumulates gradients
    across horizons (no zeroing inside the loop), then takes one optimiser step --
    matching Algorithm 1 / exp_main.py.
    """
    model.train()
    loss_fn = nn.MSELoss()
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    def _fn():
        optimizer.zero_grad(set_to_none=True)
        for h in horizons:
            model.pred_len = h
            target = y_full[:, :h, :]
            if amp_enabled:
                with torch.cuda.amp.autocast():
                    out = model(x)
                    loss = loss_fn(out, target)
                scaler.scale(loss).backward()
            else:
                out = model(x)
                loss = loss_fn(out, target)
                loss.backward()
        if amp_enabled:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

    return _fn


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="AnyHorizonTS/SparseTSF efficiency benchmark.")
    # problem size (defaults: Traffic, L=H=720)
    ap.add_argument("--seq-len", type=int, default=720)
    ap.add_argument("--pred-len", type=int, default=720, help="evaluation horizon for params/MACs/inference (L=H=720 in the paper)")
    ap.add_argument("--enc-in", type=int, default=862, help="number of channels (Traffic=862)")
    ap.add_argument("--period-len", type=int, default=24)
    ap.add_argument("--d-model", type=int, default=128, help="MLP hidden size")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--horizons", type=int, nargs="+", default=[96, 192, 336, 720],
                    help="AnyHorizonTS training horizon set (multiples of period_len)")
    # benchmark control
    ap.add_argument("--models", nargs="+", default=["stsf_linear", "stsf_mlp", "ahts_linear", "ahts_mlp"],
                    choices=["stsf_linear", "stsf_mlp", "ahts_linear", "ahts_mlp"])
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto-detect)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--amp", action="store_true", help="enable CUDA AMP for the training step (matches training)")
    ap.add_argument("--batches-per-epoch", type=int, default=None,
                    help="if given, report Epoch Time = train-step time x this many batches")
    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument("--csv", default=None, help="optional path to write a CSV row per model")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ---- environment banner -------------------------------------------------
    print("=" * 78)
    print("AnyHorizonTS / SparseTSF efficiency benchmark")
    print("-" * 78)
    print(f"  device         : {device.type}  ->  {device_label(device)}")
    print(f"  torch          : {torch.__version__}")
    print(f"  python         : {platform.python_version()}")
    print(f"  platform       : {platform.platform()}")
    print(f"  L=seq_len      : {args.seq_len}")
    print(f"  H=pred_len     : {args.pred_len}  (params/MACs/inference horizon)")
    print(f"  enc_in (chan.) : {args.enc_in}")
    print(f"  period_len     : {args.period_len}")
    print(f"  d_model (MLP)  : {args.d_model}")
    print(f"  batch_size     : {args.batch_size}")
    print(f"  ATS horizons   : {args.horizons}")
    print(f"  amp (cuda)     : {bool(args.amp)}")
    print(f"  warmup/iters   : {args.warmup}/{args.iters}")
    if device.type == "mps":
        print("  [note] MPS has no peak-memory API; 'Max Mem' is the live allocation")
        print("         measured right after the op and approximates the peak.")
    print("=" * 78)

    rows = []
    for name in args.models:
        pretty = PRETTY[name]
        model = build_model(name, args).to(device)

        # --- params & MACs (device-independent; counted on a CPU copy) --------
        cpu_model = build_model(name, args)  # fresh module on CPU for clean counting
        params = count_parameters(cpu_model)
        macs = count_macs(cpu_model, (1, args.seq_len, args.enc_in), args.pred_len, args.period_len)
        del cpu_model

        # --- synthetic inputs on device --------------------------------------
        x = torch.randn(args.batch_size, args.seq_len, args.enc_in, device=device)
        y_full = torch.randn(args.batch_size, max(args.horizons + [args.pred_len]), args.enc_in, device=device)

        # --- inference: latency + memory at H = pred_len ----------------------
        infer_fn = make_inference_fn(model, x, args.pred_len)
        infer_mem = measure_peak_mem_mb(infer_fn, device)
        infer_mean, infer_std, infer_p50 = time_fn(infer_fn, device, args.warmup, args.iters)

        # --- training step: latency + memory ---------------------------------
        # SparseTSF rows train a single horizon (H=pred_len); AnyHorizonTS rows
        # sweep the full horizon set per step.
        train_horizons = list(args.horizons) if IS_ATS[name] else [args.pred_len]
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
        train_fn = make_train_step_fn(model, x, y_full, train_horizons, optimizer, args.amp, device)
        train_mem = measure_peak_mem_mb(train_fn, device)
        train_mean, train_std, train_p50 = time_fn(train_fn, device, args.warmup, args.iters)

        epoch_s = (train_mean * args.batches_per_epoch / 1e3) if args.batches_per_epoch else None

        rows.append(dict(
            model=pretty, device=device.type, device_name=device_label(device),
            params=params, macs=macs,
            infer_mem_mb=infer_mem, infer_ms=infer_mean, infer_ms_std=infer_std,
            train_mem_mb=train_mem, train_ms=train_mean, train_ms_std=train_std,
            train_horizons=train_horizons, epoch_s=epoch_s,
        ))

        del model, optimizer, x, y_full
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- pretty print -------------------------------------------------------
    def fmt_params(p):
        return f"{p/1000:.2f} K" if p < 1e6 else f"{p/1e6:.2f} M"

    def fmt_macs(m):
        return f"{m/1e6:.2f} M" if m < 1e9 else f"{m/1e9:.2f} G"

    print()
    hdr = f"{'Model':18} {'Params':>9} {'MACs':>10} {'InferMem':>10} {'Infer ms':>12} {'TrainMem':>10} {'Train ms':>14}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['model']:18} {fmt_params(r['params']):>9} {fmt_macs(r['macs']):>10} "
              f"{r['infer_mem_mb']:>9.1f}M {r['infer_ms']:>9.3f}±{r['infer_ms_std']:<2.2f} "
              f"{r['train_mem_mb']:>9.1f}M {r['train_ms']:>9.3f}±{r['train_ms_std']:<3.2f}")
    if rows and rows[0]["epoch_s"] is not None:
        print()
        print(f"{'Model':18} {'Epoch Time (s)':>16}   (= train-step x {args.batches_per_epoch} batches)")
        print("-" * 40)
        for r in rows:
            print(f"{r['model']:18} {r['epoch_s']:>16.1f}")

    print()
    print("Legend / mapping to the columns of the efficiency table:")
    print("  Parameters -> 'Params'      MACs -> 'MACs'")
    print("  Max Memory -> 'TrainMem' (training peak; use 'InferMem' if you switch")
    print("                the table to an inference benchmark)")
    print("  Epoch Time -> 'Epoch Time (s)' (only if --batches-per-epoch was given;")
    print("                for Traffic: batches/epoch = floor(0.7*17544 - seq_len - pred_len + 1) / batch_size)")
    if any(r["device"] == "mps" for r in rows):
        print("  [MPS] TrainMem/InferMem are live-allocation approximations of the peak.")

    # ---- optional CSV -------------------------------------------------------
    if args.csv:
        fields = ["model", "device", "device_name", "params", "macs",
                  "infer_mem_mb", "infer_ms", "infer_ms_std",
                  "train_mem_mb", "train_ms", "train_ms_std", "train_horizons", "epoch_s"]
        with open(args.csv, "w", newline="") as f:
            w = csvlib.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                row = dict(r)
                row["train_horizons"] = " ".join(map(str, row["train_horizons"]))
                w.writerow({k: row[k] for k in fields})
        print(f"\nWrote {len(rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
