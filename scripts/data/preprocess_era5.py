#!/usr/bin/env python3
"""
preprocess_era5.py — Crop ERA5-Land zarr to Barcelona domain.

Input:  raw ERA5-Land zarr (558 stations, 25 lats × 32 lons, 18 variables)
Output: era5land_2008-2017.zarr  (≤20 stations, 9 variables, 2008-2017)

Crop: keeps exactly the 4 lats × 5 lons used by the model
      (41.2–41.5 × 1.9–2.3), plus a 1-cell buffer on each side for safety.
      Sea cells (NaN) are kept — dataset.py fills them with nearest-land at load.

Variables: only the 9 in ERA5_INPUT_VARS (config.py).

Usage:
  ERA5_RAW=/path/to/raw_era5.zarr python scripts/data/preprocess_era5.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import ERA5_INPUT_VARS, DATA_DIR

RAW_PATH = Path(os.environ.get(
    "ERA5_RAW",
    str(PROJECT_ROOT / "data" / "era5land_raw.zarr"),
))
OUT_PATH = DATA_DIR / "era5land_2008-2017.zarr"

# Model crop + 1-cell buffer on each side
LAT_MIN, LAT_MAX = 41.1, 41.6
LON_MIN, LON_MAX = 1.8, 2.4


def main():
    print("=" * 60)
    print("preprocess_era5.py")
    print("=" * 60)

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"ERA5 raw zarr not found: {RAW_PATH}")

    print(f"\nOpening: {RAW_PATH}")
    ds = xr.open_zarr(str(RAW_PATH), consolidated=False)
    ids = ds.weatherStation.values
    lats = np.array([float(s.split("_")[0]) for s in ids])
    lons = np.array([float(s.split("_")[1]) for s in ids])

    print(f"  Raw: {ds.sizes['time']:,} timesteps × {len(ids):,} stations × "
          f"{len(list(ds.data_vars))} vars")
    print(f"  Lat range: [{lats.min()}, {lats.max()}]")
    print(f"  Lon range: [{lons.min()}, {lons.max()}]")

    # ── Crop to Barcelona bbox ────────────────────────────────────
    mask = (lats >= LAT_MIN) & (lats <= LAT_MAX) & \
           (lons >= LON_MIN) & (lons <= LON_MAX)
    station_idxs = np.where(mask)[0]
    print(f"\nCrop [{LAT_MIN}-{LAT_MAX}] × [{LON_MIN}-{LON_MAX}]: "
          f"{mask.sum()} stations")
    print(f"  Station IDs: {ids[mask].tolist()}")

    # ── Select variables ──────────────────────────────────────────
    missing_vars = [v for v in ERA5_INPUT_VARS if v not in ds.data_vars]
    if missing_vars:
        raise ValueError(f"Missing vars in raw zarr: {missing_vars}")
    print(f"\nKeeping {len(ERA5_INPUT_VARS)} variables: {ERA5_INPUT_VARS}")

    # ── Time range ────────────────────────────────────────────────
    times = ds.time.values
    t_mask = (times >= np.datetime64("2008-01-01")) & \
             (times <= np.datetime64("2017-12-31T23:59:59"))
    print(f"\nTime: {t_mask.sum():,} timesteps in 2008-2017 "
          f"(of {len(times):,} total)")

    # ── Build output dataset (lazy) ───────────────────────────────
    ds_crop = ds[ERA5_INPUT_VARS].isel(
        weatherStation=station_idxs.tolist(),
        time=np.where(t_mask)[0].tolist(),
    )

    # ── Rechunk: one year per time chunk ─────────────────────────
    n_stations = len(station_idxs)
    ds_crop = ds_crop.chunk({"time": 8760, "weatherStation": n_stations})

    # ── Write ─────────────────────────────────────────────────────
    import shutil
    if OUT_PATH.is_symlink():
        OUT_PATH.unlink()
    elif OUT_PATH.exists():
        shutil.rmtree(OUT_PATH)
    print(f"\nWriting → {OUT_PATH}")
    # Clear v2 encoding inherited from source (codec format incompatible with zarr v3)
    for v in list(ds_crop.data_vars) + list(ds_crop.coords):
        ds_crop[v].encoding = {}
    ds_crop.to_zarr(str(OUT_PATH), mode="w")

    # ── Verify ────────────────────────────────────────────────────
    ds_out = xr.open_zarr(str(OUT_PATH), consolidated=False)
    print(f"\nOutput:")
    print(f"  shape : {dict(ds_out.sizes)}")
    print(f"  vars  : {list(ds_out.data_vars)}")
    print(f"  time  : {ds_out.time.values[0]}  →  {ds_out.time.values[-1]}")
    print(f"  stations: {ds_out.weatherStation.values.tolist()}")

    sample = ds_out["airTemperature"].isel(time=0).values
    nan_ct = np.isnan(sample).sum()
    print(f"  NaN stations at t=0: {nan_ct} (sea cells — filled by dataset.py)")
    print(f"\nDone. {OUT_PATH}")


if __name__ == "__main__":
    main()
