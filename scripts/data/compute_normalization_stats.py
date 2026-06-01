"""
compute_normalization_stats.py — Computes ERA5 and UrbClim normalization stats
from the training split (2008-2015) and saves to data/normalization_stats.npz.

Must be run once before training. Runtime: ~10 min (streaming over 70k time steps).

Output layout:
  era5_mean    : (DYN_CHANNELS,)  float32 — per-variable mean over train split
  era5_std     : (DYN_CHANNELS,)  float32 — per-variable std
  urbclim_mean : scalar float32   — mean temperature over train split (z-score)
  urbclim_std  : scalar float32   — std temperature over train split (z-score)
  urbclim_min  : scalar float32   — min temperature (kept for evaluation / denorm)
  urbclim_max  : scalar float32   — max temperature (kept for evaluation / denorm)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import xarray as xr
from tqdm import tqdm

from config.config import (
    PATH_ERA5, PATH_URBCLIM, PATH_STATS,
    ERA5_INPUT_VARS, URBCLIM_TARGET_VAR,
    TRAIN_YEARS,
)


def _year_mask(times, years):
    import pandas as pd
    t = pd.to_datetime(times).year
    return np.isin(t, years)


def main():
    print("Computing normalization stats from training years:", TRAIN_YEARS)

    ds_era5  = xr.open_zarr(str(PATH_ERA5),    consolidated=False)
    ds_urb   = xr.open_zarr(str(PATH_URBCLIM), consolidated=False)

    era5_mask = _year_mask(ds_era5.time.values, TRAIN_YEARS)
    urb_mask  = _year_mask(ds_urb.time.values,  TRAIN_YEARS)
    print(f"  ERA5 train steps: {era5_mask.sum():,}  UrbClim: {urb_mask.sum():,}")

    # ERA5: stream variable-by-variable to avoid loading full dataset
    print("ERA5 stats (per variable)...")
    era5_mean = np.zeros(len(ERA5_INPUT_VARS), dtype=np.float64)
    era5_std  = np.zeros(len(ERA5_INPUT_VARS), dtype=np.float64)
    era5_tidx = np.where(era5_mask)[0].tolist()

    for i, var in enumerate(tqdm(ERA5_INPUT_VARS)):
        data = ds_era5[var].isel(time=era5_tidx).values.astype(np.float64)
        era5_mean[i] = data.mean()
        era5_std[i]  = data.std()
        del data

    print("UrbClim stats (z-score + min/max)...")
    urb_tidx = np.where(urb_mask)[0].tolist()
    # Stream in chunks to avoid loading 70k × 251 × 251 × 4 bytes at once
    CHUNK = 500
    urb_sum  = 0.0
    urb_sum2 = 0.0
    urb_n    = 0
    urb_min  = np.inf
    urb_max  = -np.inf
    for start in tqdm(range(0, len(urb_tidx), CHUNK)):
        chunk_idx = urb_tidx[start : start + CHUNK]
        data = ds_urb[URBCLIM_TARGET_VAR].isel(time=chunk_idx).values.astype(np.float64)
        urb_sum  += data.sum()
        urb_sum2 += (data ** 2).sum()
        urb_n    += data.size
        urb_min   = min(urb_min, data.min())
        urb_max   = max(urb_max, data.max())
        del data

    urb_mean = urb_sum / urb_n
    urb_std  = np.sqrt(urb_sum2 / urb_n - urb_mean ** 2)

    PATH_STATS.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(PATH_STATS),
        era5_mean=era5_mean.astype(np.float32),
        era5_std=era5_std.astype(np.float32),
        urbclim_mean=np.float32(urb_mean),
        urbclim_std=np.float32(urb_std),
        urbclim_min=np.float32(urb_min),
        urbclim_max=np.float32(urb_max),
    )
    print(f"Saved to {PATH_STATS}")
    print(f"  ERA5 means: {era5_mean.round(3)}")
    print(f"  ERA5 stds:  {era5_std.round(3)}")
    print(f"  UrbClim mean={urb_mean:.4f}  std={urb_std:.4f}  min={urb_min:.4f}  max={urb_max:.4f}")


if __name__ == "__main__":
    main()
