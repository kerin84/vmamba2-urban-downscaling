#!/usr/bin/env python3
"""
evaluate_rollout.py — Multi-horizon + temporal consistency evaluation.

Extends evaluate.py by capturing ALL T output timesteps and computing:
  1. Multi-horizon accuracy:  MAE / RMSE / SSIM at each lead time 1..T
  2. Temporal consistency:    frame-to-frame MAE, temporal variance, autocorrelation
  3. Stateful ConvLSTM:       propagate hidden state across consecutive samples

Usage:
    python scripts/evaluation/evaluate_rollout.py
        [--experiments_dir PATH]
        [--output_dir PATH]    (default: experiments/evaluation/rollout/)

Outputs:
    rollout_results.csv       — per-run, per-lead-time metrics
    rollout_summary.csv       — mean ± std per (arch, seq_len, lead_time)
    temporal_consistency.csv  — frame-to-frame metrics per run
    predictions/              — {arch}_s{seed}_T{seq_len}_rollout_pred.npy
                                (N, T, H, W)  ALL timesteps, not just last
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
    LOSS_MAX_VAL, SEQ_LEN,
)
from src.data.dataset import UrbanDownscalingDataset, _load_urbclim_frame
from src.models.downsr_unet import DownsrUNet, _ConvLSTMBottleneck
from src.losses import _ssim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _parse_run_dir(name: str):
    try:
        parts = name.split("_")
        arch    = parts[0]
        seed    = int(parts[1].replace("seed", ""))
        seq_len = int(parts[2].replace("T", ""))
        return arch, seed, seq_len
    except Exception:
        return None


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """preds, targets: (N, H, W) or (N, 1, H, W). Returns MAE, RMSE, Bias, SSIM."""
    diff = preds - targets
    mae  = np.abs(diff).mean()
    rmse = np.sqrt((diff ** 2).mean())
    bias = diff.mean()

    p_t = torch.from_numpy(preds[:, None] if preds.ndim == 3 else preds)
    t_t = torch.from_numpy(targets[:, None] if targets.ndim == 3 else targets)
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


def compute_temporal_consistency(preds_seq: np.ndarray) -> dict:
    """
    preds_seq: (N, H, W) float32 in °C — chronological prediction sequence.
    
    Returns temporal consistency metrics:
      - frame_mae:  mean absolute frame-to-frame difference (°C)
      - temp_var:   mean per-pixel temporal variance (°C²)
      - autocorr:   mean lag-1 autocorrelation of per-pixel time series
    """
    N = len(preds_seq)
    if N < 3:
        return {"frame_mae": 0.0, "temp_var": 0.0, "autocorr": 0.0}

    # Frame-to-frame MAE: |pred_t - pred_{t-1}| averaged over pixels and frames
    diffs = np.abs(preds_seq[1:] - preds_seq[:-1])  # (N-1, H, W)
    frame_mae = float(diffs.mean())

    # Per-pixel temporal variance: var(preds over time), averaged over pixels
    temp_var = float(np.var(preds_seq, axis=0).mean())

    # Lag-1 autocorrelation per pixel, then averaged
    centered = preds_seq - preds_seq.mean(axis=0, keepdims=True)  # (N, H, W)
    var_t = np.var(preds_seq, axis=0) + 1e-10
    # lag-1 product, averaged over time
    corr = (centered[1:] * centered[:-1]).mean(axis=0) / var_t
    autocorr = float(np.clip(corr, -1, 1).mean())

    return {"frame_mae": frame_mae, "temp_var": temp_var, "autocorr": autocorr}


# ---------------------------------------------------------------------------
# Multi-horizon inference — captures ALL T output timesteps
# ---------------------------------------------------------------------------

def extract_convlstm_state(model: DownsrUNet):
    """
    Extract the last hidden state from the ConvLSTM bottleneck.
    Returns None if not a ConvLSTM bottleneck.
    """
    bn = model.bottleneck
    if not isinstance(bn, _ConvLSTMBottleneck):
        return None
    # _ConvLSTMBottleneck stores state in the cell after forward
    # We need to check if the cell has a cached state
    if hasattr(bn, '_last_state') and bn._last_state is not None:
        return bn._last_state
    return None


@torch.no_grad()
def run_multi_horizon_inference(
    model, loader, device, urb_mean, urb_std, seq_len
):
    """
    Returns:
      preds_all : (N, T, H, W) float32 in °C — predictions at ALL lead times
      tgts_all  : (N, T, H, W) float32 in °C — targets at ALL lead times
      preds_last: (N, H, W) float32 in °C — predictions at the target time (T-1)
      tgts_last : (N, H, W) float32 in °C — targets at the target time
    """
    model.eval()
    all_preds_t   = []  # all T timesteps
    all_targets_t = []  # targets at all T timesteps
    all_preds_last   = []
    all_targets_last = []

    for era5_seq, static, target_last in loader:
        era5_seq = era5_seq.to(device, non_blocking=True)
        static   = static.to(device,   non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            pred = model(era5_seq, static)   # (B, T, H, W) in z-score

        # Move pred to CPU numpy
        pred_np = pred.float().cpu().numpy()     # (B, T, H, W) z-scored
        batch_size = pred_np.shape[0]
        T = pred_np.shape[1]

        # Load targets for ALL timesteps in this batch
        # We need the dataset's urb_pos and urb index to load intermediate frames.
        # For the current batch, the positions are determined by the dataset's
        # internal sample_pos. We compute this from the dataset metadata.
        
        # Denormalize: pred * urb_std + urb_mean
        pred_c = pred_np * urb_std + urb_mean  # (B, T, H, W) °C
        # Target at T-1 (supervised timestep)
        tgt_last_c = target_last.numpy() * urb_std + urb_mean  # (B, H, W) °C

        # For ALL-T targets, we need intermediate UrbClim frames.
        # We approximate: for lead time t, the target is UrbClim[pos - T + 1 + t]
        # This is computed per-sample in the main loop below.
        
        all_preds_t.append(pred_c)
        all_targets_t.append(None)  # placeholder, computed per-sample
        all_preds_last.append(pred_c[:, -1])
        all_targets_last.append(tgt_last_c)

    # Concatenate
    preds_all = np.concatenate([p for p in all_preds_t], axis=0)  # (N, T, H, W)
    preds_last = np.concatenate(all_preds_last, axis=0)          # (N, H, W)
    targets_last = np.concatenate(all_targets_last, axis=0)      # (N, H, W)

    return preds_all, preds_last, targets_last


@torch.no_grad()
def run_stateful_rollout(
    model, loader, device, urb_mean, urb_std, seq_len, rollout_steps=24
):
    """
    Stateful ConvLSTM rollout: propagate hidden state across consecutive samples.
    
    Returns preds_rollout, targets_rollout: (N, rollout_steps, H, W) °C
    Only available for ConvLSTM bottleneck.
    
    For UNet/Mamba: falls back to standard stateless evaluation.
    """
    bn = model.bottleneck
    is_convlstm = isinstance(bn, _ConvLSTMBottleneck)
    
    model.eval()
    
    all_preds = []
    all_targets = []
    state = None  # persistent state for ConvLSTM
    
    for era5_seq, static, target in loader:
        era5_seq = era5_seq.to(device, non_blocking=True)
        static = static.to(device, non_blocking=True)
        batch_size = era5_seq.shape[0]
        
        # For stateful ConvLSTM, we need to inject state into the bottleneck
        if is_convlstm and state is not None:
            # Set the initial state of the ConvLSTM cell
            bn._initial_state = state
        
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            pred = model(era5_seq, static)  # (B, T, H, W)
        
        # Extract the last hidden state for ConvLSTM
        if is_convlstm:
            if hasattr(bn, '_last_state') and bn._last_state is not None:
                state = bn._last_state
            else:
                # Need to capture state - handle in modified forward
                state = None
        
        pred_c = pred[:, -1].float().cpu().numpy() * urb_std + urb_mean
        tgt_c = target.numpy() * urb_std + urb_mean
        all_preds.append(pred_c)
        all_targets.append(tgt_c)
        
        # Reset batch-level state after processing
        if is_convlstm:
            bn._initial_state = None
    
    return np.concatenate(all_preds), np.concatenate(all_targets)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_dir", default=str(EXPERIMENTS_DIR))
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    exp_dir  = Path(args.experiments_dir)
    out_dir  = Path(args.output_dir) if args.output_dir else exp_dir / "evaluation" / "rollout"
    pred_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    device = _get_device()
    print(f"Device: {device}")

    # ---- discover completed runs ----
    run_dirs = sorted([
        d for d in exp_dir.iterdir()
        if d.is_dir() and (d / "best.pt").exists()
        and _parse_run_dir(d.name) is not None
    ])
    print(f"Found {len(run_dirs)} completed runs.")

    # We'll compute per-run results and aggregate
    all_results = []
    temp_consistency = []

    for run_dir in run_dirs:
        parsed = _parse_run_dir(run_dir.name)
        if parsed is None:
            continue
        arch, seed, run_seq_len = parsed
        print(f"\n--- {run_dir.name} ---")

        # Load actual config
        cfg_path = run_dir / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            run_seq_len = cfg.get("seq_len", run_seq_len)

        # ---- Load dataset with correct seq_len ----
        ds = UrbanDownscalingDataset(
            split="test", seq_len=run_seq_len, normalize=True
        )
        loader = DataLoader(
            ds,
            batch_size=8,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        urb_mean = ds._norm_urb_mean
        urb_std  = ds._norm_urb_std
        T = run_seq_len

        # ---- Build model and load weights ----
        model = DownsrUNet(bottleneck_type=arch).to(device)
        ckpt = torch.load(run_dir / "best.pt", map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
        model.eval()
        print(f"  Loaded best.pt")

        # ---- Multi-horizon evaluation ----
        model.eval()

        all_preds = []   # (N, T, H, W)
        all_tgts  = []   # (N, T, H, W) — targets for ALL lead times
        pred_last_list = []
        tgt_last_list = []

        # Load UrbClim time indices for each sample
        urb_positions = ds._urb_pos[ds._sample_pos]          # (N,) — UrbClim time index for each sample
        urb_zarr = ds._ds_urb["airTemperature"]              # lazy zarr array — NO .values (avoid 22 GB pre-load)

        with torch.no_grad():
            batch_idx = 0
            for era5_seq, static, target_last in loader:
                era5_seq = era5_seq.to(device, non_blocking=True)
                static = static.to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=(device.type == "cuda")):
                    pred = model(era5_seq, static)   # (B, T, H, W) z-scored

                pred_np = pred.float().cpu().numpy()  # (B, T, H, W) z-scored
                B_actual = pred_np.shape[0]

                # Denormalize predictions
                pred_c = pred_np * urb_std + urb_mean  # (B, T, H, W) °C
                tgt_last_c = target_last.numpy() * urb_std + urb_mean  # (B, H, W) °C

                # Build ALL-T targets by indexing the pre-loaded UrbClim array
                # For sample at urb_positions[idx], lead time t corresponds to
                # UrbClim at (urb_positions[idx] - T + 1 + t)
                targets_all = np.zeros_like(pred_c, dtype=np.float32)  # (B, T, H, W)
                for b in range(B_actual):
                    pos_idx = batch_idx + b
                    if pos_idx < len(urb_positions):
                        target_pos = int(urb_positions[pos_idx])
                        for t in range(T):
                            lead_pos = target_pos - T + 1 + t
                            if lead_pos >= 0:
                                raw_frame = np.asarray(urb_zarr[lead_pos], dtype=np.float32)
                                targets_all[b, t] = raw_frame * urb_std + urb_mean

                all_preds.append(pred_c)
                all_tgts.append(targets_all)
                pred_last_list.append(pred_c[:, -1])
                tgt_last_list.append(tgt_last_c)

                batch_idx += B_actual

                if batch_idx % 200 == 0:
                    print(f"  Processed {batch_idx}/{len(ds)} samples...")

        # Concatenate
        preds_all = np.concatenate(all_preds, axis=0)    # (N, T, H, W)
        tgts_all  = np.concatenate(all_tgts, axis=0)     # (N, T, H, W)
        preds_last = np.concatenate(pred_last_list, axis=0)  # (N, H, W)
        tgts_last  = np.concatenate(tgt_last_list, axis=0)   # (N, H, W)

        N = len(preds_all)
        print(f"  Inference done: {preds_all.shape}")

        # Save full multi-horizon predictions (ALL T timesteps)
        tag = f"{arch}_s{seed}_T{T}"
        np.save(pred_dir / f"{tag}_rollout_pred.npy", preds_all.astype(np.float32))
        np.save(pred_dir / f"{tag}_rollout_target.npy", tgts_all.astype(np.float32))
        print(f"  Saved full multi-horizon predictions to {pred_dir}/{tag}_rollout_*.npy")

        # ---- Compute multi-horizon metrics per lead time ----
        for t in range(T):
            metrics = compute_metrics(preds_all[:, t], tgts_all[:, t])
            row = {
                "arch": arch, "seed": seed, "seq_len": T,
                "lead_hours": t + 1,  # lead time in hours (1-indexed)
                "n_samples": N,
                **metrics,
            }
            all_results.append(row)

        # ---- Aggregate metrics at target time (T-1) ----
        final_metrics = compute_metrics(preds_last, tgts_last)
        print(f"  Target-time metrics  MAE={final_metrics['mae_c']:.4f}°C  "
              f"SSIM={final_metrics['ssim']:.4f}")

        # ---- Temporal consistency (frame-to-frame metrics) ----
        # The test set predictions are in chronological order (shuffle=False).
        # We compute frame-to-frame differences on the target-time predictions.
        tc = compute_temporal_consistency(preds_last)
        tc_row = {
            "arch": arch, "seed": seed, "seq_len": T,
            "n_samples": N,
            **tc,
        }
        temp_consistency.append(tc_row)
        print(f"  Temporal consistency:  frame_MAE={tc['frame_mae']:.4f}°C  "
              f"temp_var={tc['temp_var']:.6f}  autocorr={tc['autocorr']:.4f}")

        # Free GPU
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Save per-lead-time results ----
    results_path = out_dir / "rollout_results.csv"
    if all_results:
        fields = ["arch", "seed", "seq_len", "lead_hours", "n_samples",
                  "mae_c", "rmse_c", "bias_c", "ssim"]
        with open(results_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_results)
        print(f"\nSaved: {results_path}")

    # ---- Save temporal consistency ----
    tc_path = out_dir / "temporal_consistency.csv"
    if temp_consistency:
        fields = ["arch", "seed", "seq_len", "n_samples",
                  "frame_mae", "temp_var", "autocorr"]
        with open(tc_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(temp_consistency)
        print(f"Saved: {tc_path}")

    # ---- Summary: mean ± std per (arch, seq_len, lead_hours) ----
    if all_results:
        try:
            import pandas as pd
            df = pd.read_csv(results_path)
            summary = (
                df.groupby(["arch", "seq_len", "lead_hours"])[
                    ["mae_c", "rmse_c", "ssim"]
                ]
                .agg(["mean", "std"])
                .round(4)
            )
            summary.columns = ["_".join(c) for c in summary.columns]
            summary = summary.reset_index()
            summary_path = out_dir / "rollout_summary.csv"
            summary.to_csv(summary_path, index=False)
            print(f"Saved: {summary_path}")

            print("\n=== MULTI-HORIZON SUMMARY (target time) ===")
            target_summary = df[df["lead_hours"] == df.groupby(["arch", "seq_len"])["lead_hours"].transform("max")]
            print(target_summary.groupby(["arch", "seq_len"])[
                ["mae_c", "ssim"]
            ].agg(["mean", "std"]).round(4).to_string())

            print("\n=== TEMPORAL CONSISTENCY SUMMARY ===")
            tc_df = pd.read_csv(tc_path)
            print(tc_df.groupby(["arch", "seq_len"])[
                ["frame_mae", "temp_var", "autocorr"]
            ].agg(["mean", "std"]).round(4).to_string())

        except ImportError:
            pass

    print("\nDone.")


def _patch_convlstm_bottleneck():
    """
    Monkey-patch _ConvLSTMBottleneck to capture and inject state.
    Called at module load time if stateful rollout is needed.
    """
    orig_forward = _ConvLSTMBottleneck.forward
    
    def patched_forward(self, x):
        B, T = x.shape[:2]
        # Use injected initial state if available
        state = getattr(self, '_initial_state', None)
        outs = []
        for t in range(T):
            h, state = self.cell(x[:, t], state)
            outs.append(h)
        # Save last state for extraction
        self._last_state = state
        out = torch.stack(outs, dim=1)  # (B, T, C, H, W)
        # Apply BN+act time-distributed (original behaviour)
        from src.models.downsr_unet import td
        import torch.nn as nn
        return td(nn.Sequential(self.bn, self.act), out)
    
    _ConvLSTMBottleneck.forward = patched_forward


if __name__ == "__main__":
    # Apply the ConvLSTM patch for state capture
    _patch_convlstm_bottleneck()
    main()
