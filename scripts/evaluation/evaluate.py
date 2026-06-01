#!/usr/bin/env python3
"""
evaluate.py — Test-set evaluation for DownsrUNet ablation study.

Loads best.pt for every completed experiment, runs inference on the
held-out test split (2017), computes MAE / RMSE / SSIM / Bias in °C,
and saves per-run and summary CSVs plus per-sample prediction arrays.

Usage:
    python scripts/evaluation/evaluate.py [--experiments_dir PATH]

Outputs (under experiments/evaluation/):
    test_results.csv     — one row per run (arch, seed, seq_len, all metrics)
    test_summary.csv     — mean ± std per (arch, seq_len) group
    predictions/         — {arch}_s{seed}_T{seq_len}_pred.npy  (N, 251, 251) float32, °C
                           {arch}_s{seed}_T{seq_len}_target.npy (N, 251, 251) float32, °C
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    EXPERIMENTS_DIR, BATCH_SIZE,
    DYN_CHANNELS, STATIC_CHANNELS,
    LOSS_MAX_VAL,
)
from src.data.dataset import UrbanDownscalingDataset
from src.models.downsr_unet import DownsrUNet
from src.losses import _ssim


# ---------------------------------------------------------------------------
# Metrics (all in °C after denormalization)
# ---------------------------------------------------------------------------

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """
    preds, targets : (N, H, W) float32 in °C.
    Returns dict with MAE, RMSE, Bias, SSIM (mean over samples).
    """
    diff = preds - targets
    mae  = np.abs(diff).mean()
    rmse = np.sqrt((diff ** 2).mean())
    bias = diff.mean()

    # SSIM per sample, then average — keep max_val consistent with training
    p_t = torch.from_numpy(preds[:, None])    # (N, 1, H, W)
    t_t = torch.from_numpy(targets[:, None])  # (N, 1, H, W)
    ssim_vals = []
    BS = 32
    for i in range(0, len(p_t), BS):
        s = _ssim(p_t[i:i+BS], t_t[i:i+BS], max_val=LOSS_MAX_VAL).item()
        ssim_vals.append(s)
    ssim = float(np.mean(ssim_vals))

    return {
        "mae_c":  float(mae),
        "rmse_c": float(rmse),
        "bias_c": float(bias),
        "ssim":   ssim,
    }


def compute_seasonal_mae(preds: np.ndarray, targets: np.ndarray,
                          times) -> dict:
    """MAE per season (DJF, MAM, JJA, SON)."""
    import pandas as pd
    months = pd.to_datetime(times).month
    seasons = {
        "DJF": [12, 1, 2],
        "MAM": [3, 4, 5],
        "JJA": [6, 7, 8],
        "SON": [9, 10, 11],
    }
    result = {}
    for name, ms in seasons.items():
        mask = np.isin(months, ms)
        if mask.sum() > 0:
            result[f"mae_{name}"] = float(np.abs(preds[mask] - targets[mask]).mean())
    return result


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, loader, device, urb_mean, urb_std):
    """
    Returns preds_c, targets_c : (N, 251, 251) float32 in °C.
    """
    model.eval()
    all_preds   = []
    all_targets = []

    for era5_seq, static, target in loader:
        era5_seq = era5_seq.to(device, non_blocking=True)
        static   = static.to(device,   non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            pred = model(era5_seq, static)   # (B, T, H, W)

        pred_last = pred[:, -1].float().cpu().numpy()   # (B, H, W) z-scored
        tgt       = target.numpy()                       # (B, H, W) z-scored

        # denormalize to °C
        pred_c = pred_last * urb_std + urb_mean
        tgt_c  = tgt       * urb_std + urb_mean

        all_preds.append(pred_c)
        all_targets.append(tgt_c)

    return np.concatenate(all_preds), np.concatenate(all_targets)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _parse_run_dir(name: str):
    """
    Parse 'arch_seedSS_TLL' → (arch, seed, seq_len).
    Handles names like: unet_seed42_T6, mamba_seed43_T12, convlstm_seed44_T6
    """
    try:
        parts = name.split("_")
        arch    = parts[0]
        seed    = int(parts[1].replace("seed", ""))
        seq_len = int(parts[2].replace("T", ""))
        return arch, seed, seq_len
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_dir", default=str(EXPERIMENTS_DIR))
    args = parser.parse_args()

    exp_dir  = Path(args.experiments_dir)
    eval_dir = exp_dir / "evaluation"
    pred_dir = eval_dir / "predictions"
    eval_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    device = _get_device()
    print(f"Device: {device}")

    # ---- test dataset (loaded once, shared across runs) ----
    print("Loading test dataset (2017)...")
    test_ds = UrbanDownscalingDataset(split="test", seq_len=6, normalize=True)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    urb_mean = test_ds._norm_urb_mean
    urb_std  = test_ds._norm_urb_std
    times    = test_ds._times[test_ds._sample_pos]
    print(f"  Test samples: {len(test_ds)}")

    # ---- discover runs ----
    run_dirs = sorted([
        d for d in exp_dir.iterdir()
        if d.is_dir() and (d / "best.pt").exists()
        and _parse_run_dir(d.name) is not None
    ])
    print(f"Found {len(run_dirs)} completed runs.")

    results = []
    csv_fields = [
        "arch", "seed", "seq_len",
        "mae_c", "rmse_c", "bias_c", "ssim",
        "mae_DJF", "mae_MAM", "mae_JJA", "mae_SON",
        "epochs_trained",
    ]

    for run_dir in run_dirs:
        parsed = _parse_run_dir(run_dir.name)
        if parsed is None:
            continue
        arch, seed, seq_len = parsed
        print(f"\n--- {run_dir.name} ---")

        # load config to get actual seq_len used
        cfg_path = run_dir / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            seq_len = cfg.get("seq_len", seq_len)

        # reload dataset with correct seq_len if different from default
        if seq_len != test_ds.seq_len:
            ds = UrbanDownscalingDataset(split="test", seq_len=seq_len, normalize=True)
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=2, pin_memory=True)
            run_times = ds._times[ds._sample_pos]
        else:
            loader    = test_loader
            run_times = times

        # build model and load best weights
        model = DownsrUNet(bottleneck_type=arch).to(device)
        ckpt  = torch.load(run_dir / "best.pt", map_location=device)
        # best.pt saves state_dict directly (not wrapped)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
        print(f"  Loaded best.pt")

        # epochs trained
        hist = run_dir / "history.csv"
        epochs_trained = 0
        if hist.exists():
            with open(hist) as f:
                rows = list(csv.DictReader(f))
            epochs_trained = int(rows[-1]["epoch"]) if rows else 0

        # inference
        preds_c, targets_c = run_inference(model, loader, device, urb_mean, urb_std)
        print(f"  Inference done: {preds_c.shape}")

        # save predictions
        tag = f"{arch}_s{seed}_T{seq_len}"
        np.save(pred_dir / f"{tag}_pred.npy",   preds_c.astype(np.float32))
        np.save(pred_dir / f"{tag}_target.npy", targets_c.astype(np.float32))

        # compute metrics
        metrics  = compute_metrics(preds_c, targets_c)
        seasonal = compute_seasonal_mae(preds_c, targets_c, run_times)
        row = {
            "arch": arch, "seed": seed, "seq_len": seq_len,
            "epochs_trained": epochs_trained,
            **metrics, **seasonal,
        }
        results.append(row)
        print(f"  MAE={metrics['mae_c']:.4f}°C  RMSE={metrics['rmse_c']:.4f}°C  "
              f"SSIM={metrics['ssim']:.4f}  Bias={metrics['bias_c']:+.4f}°C")

        # free GPU memory between runs
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- save per-run CSV ----
    results_path = eval_dir / "test_results.csv"
    with open(results_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {results_path}")

    # ---- summary: mean ± std per (arch, seq_len) ----
    import pandas as pd
    df = pd.read_csv(results_path)
    summary = (
        df.groupby(["arch", "seq_len"])[["mae_c", "rmse_c", "ssim", "mae_JJA", "mae_DJF"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary_path = eval_dir / "test_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
