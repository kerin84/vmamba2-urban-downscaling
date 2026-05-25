#!/usr/bin/env python3
"""
Training script for DownsrUNet ablation study.

Usage (set via docker-compose.yml or shell):
    ARCH=mamba SEQ_LEN=6 SEED=42 python scripts/training/train.py

ARCH     : 'unet' | 'convlstm' | 'mamba'   (default: mamba)
SEQ_LEN  : integer sequence length          (default: 6)
SEED     : integer random seed              (default: 42)

Outputs to experiments/{arch}_seed{seed}_T{seq_len}/:
  config.json   — run hyperparameters
  best.pt       — weights with lowest val_loss
  last.pt       — weights after final epoch
  history.csv   — per-epoch train/val metrics
"""

import os
import sys
import json
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

# --- path setup so imports resolve from project root ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    EXPERIMENTS_DIR, BATCH_SIZE, EPOCHS, LR,
    MAX_STEPS_PER_EPOCH, EARLY_STOPPING_PATIENCE, LOSS_ALPHA, SEQ_LEN,
)
from src.data.dataset import UrbanDownscalingDataset
from src.models.downsr_unet import DownsrUNet
from src.losses import HybridLoss


# ---------------------------------------------------------------------------
# Env / reproducibility
# ---------------------------------------------------------------------------

def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def make_loaders(arch: str, seq_len: int, batch_size: int):
    train_ds = UrbanDownscalingDataset(split="train", seq_len=seq_len)
    val_ds   = UrbanDownscalingDataset(split="val",   seq_len=seq_len)

    # WeightedRandomSampler for summer oversampling (with replacement)
    weights = train_ds.sample_weights
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Epoch loop helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    max_steps: int,
) -> dict:
    model.train()
    total_loss = total_mae = 0.0
    steps = 0

    for era5_seq, static, target in loader:
        era5_seq = era5_seq.to(device, non_blocking=True)   # (B, T, C, 4, 5)
        static   = static.to(device,   non_blocking=True)   # (B, S, H, W)
        target   = target.to(device,   non_blocking=True)   # (B, H, W)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            pred = model(era5_seq, static)             # (B, T, H, W)
            # supervise on last timestep only — matches v1
            loss = criterion(
                pred[:, -1].unsqueeze(1),              # (B, 1, H, W)
                target.unsqueeze(1),                   # (B, 1, H, W)
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            mae = (pred[:, -1] - target).abs().mean().item()

        total_loss += loss.item()
        total_mae  += mae
        steps += 1
        if steps >= max_steps:
            break

    return {"loss": total_loss / steps, "mae": total_mae / steps}


@torch.no_grad()
def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = total_mae = 0.0
    steps = 0

    for era5_seq, static, target in loader:
        era5_seq = era5_seq.to(device, non_blocking=True)
        static   = static.to(device,   non_blocking=True)
        target   = target.to(device,   non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            pred = model(era5_seq, static)
            loss = criterion(
                pred[:, -1].unsqueeze(1),
                target.unsqueeze(1),
            )

        mae = (pred[:, -1] - target).abs().mean().item()
        total_loss += loss.item()
        total_mae  += mae
        steps += 1

    return {"loss": total_loss / steps, "mae": total_mae / steps}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    arch    = os.environ.get("ARCH",    "mamba")
    seq_len = int(os.environ.get("SEQ_LEN", str(SEQ_LEN)))
    seed    = int(os.environ.get("SEED",    "42"))

    _seed_everything(seed)
    device = _get_device()
    print(f"arch={arch}  seq_len={seq_len}  seed={seed}  device={device}")

    # ---- experiment directory ----
    run_dir = Path(EXPERIMENTS_DIR) / f"{arch}_seed{seed}_T{seq_len}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    print("Loading datasets...")
    train_loader, val_loader = make_loaders(arch, seq_len, BATCH_SIZE)

    # ---- model ----
    model = DownsrUNet(bottleneck_type=arch).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params / 1e6:.2f}M")

    # ---- optimizer / loss / AMP ----
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7
    )
    criterion = HybridLoss(alpha=LOSS_ALPHA)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ---- config snapshot ----
    config_snap = {
        "arch": arch, "seq_len": seq_len, "seed": seed,
        "batch_size": BATCH_SIZE, "lr": LR, "epochs": EPOCHS,
        "max_steps_per_epoch": MAX_STEPS_PER_EPOCH,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "loss_alpha": LOSS_ALPHA,
        "n_params": n_params,
        "device": str(device),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_snap, f, indent=2)

    # ---- history CSV ----
    history_path = run_dir / "history.csv"
    csv_fields   = ["epoch", "train_loss", "train_mae", "val_loss", "val_mae", "lr", "elapsed_s"]
    if not history_path.exists():
        with open(history_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    # ---- resume support ----
    start_epoch   = 0
    best_val_loss = float("inf")
    patience_count = 0

    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    if last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch    = ckpt["epoch"] + 1
        best_val_loss  = ckpt.get("best_val_loss", float("inf"))
        patience_count = ckpt.get("patience_count", 0)
        print(f"Resumed from epoch {start_epoch}  best_val={best_val_loss:.6f}")

    # ---- training loop ----
    for epoch in range(start_epoch, EPOCHS):
        t0 = time.time()

        tr = train_epoch(model, train_loader, optimizer, criterion,
                         scaler, device, MAX_STEPS_PER_EPOCH)
        vl = val_epoch(model, val_loader, criterion, device)

        scheduler.step(vl["loss"])
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:03d}/{EPOCHS} | "
            f"train_loss={tr['loss']:.5f}  train_mae={tr['mae']:.4f} | "
            f"val_loss={vl['loss']:.5f}  val_mae={vl['mae']:.4f} | "
            f"lr={current_lr:.2e}  t={elapsed:.0f}s"
        )

        # ---- log history ----
        with open(history_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow({
                "epoch":      epoch + 1,
                "train_loss": round(tr["loss"], 6),
                "train_mae":  round(tr["mae"],  6),
                "val_loss":   round(vl["loss"], 6),
                "val_mae":    round(vl["mae"],  6),
                "lr":         current_lr,
                "elapsed_s":  round(elapsed, 1),
            })

        # ---- checkpoint (last) ----
        torch.save({
            "epoch":          epoch,
            "model":          model.state_dict(),
            "optimizer":      optimizer.state_dict(),
            "scheduler":      scheduler.state_dict(),
            "best_val_loss":  best_val_loss,
            "patience_count": patience_count,
        }, last_path)

        # ---- checkpoint (best) ----
        if vl["loss"] < best_val_loss:
            best_val_loss  = vl["loss"]
            patience_count = 0
            torch.save(model.state_dict(), best_path)
            print(f"  -> new best ({best_val_loss:.6f})")
        else:
            patience_count += 1

        # ---- early stopping ----
        if patience_count >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1} (patience={EARLY_STOPPING_PATIENCE})")
            break

    print(f"Done. Best val_loss={best_val_loss:.6f}  saved to {run_dir}")


if __name__ == "__main__":
    main()
