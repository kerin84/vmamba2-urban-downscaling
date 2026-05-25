#!/usr/bin/env python3
"""
inspect_urbclim_raw.py — Inspect raw UrbClim zarr before preprocessing.

Determines:
  1. Dimension names, variable names, dtypes, sizes
  2. Whether the station coordinates form a regular grid in EPSG:3035
     (which enables fast rasterio warp instead of IDW)
  3. Temporal coverage and gaps
  4. NaN pattern and fill rate
  5. Value range sanity check for airTemperature

Output:
  Prints a structured report.
  Writes data/urbclim_raw_inspection.json for use by preprocess_urbclim.py.
  Override path: URBCLIM_INSPECT_JSON=/custom/path.json

Usage:
  URBCLIM_RAW=~/data3/raw_urbclim.zarr python scripts/data/inspect_urbclim_raw.py
  # or set PATH_URBCLIM_RAW in config and run without env var
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_PATH = os.environ.get(
    "URBCLIM_RAW",
    str(PROJECT_ROOT / "data" / "raw_urbclim.zarr"),
)
OUT_JSON = Path(os.environ.get(
    "URBCLIM_INSPECT_JSON",
    str(PROJECT_ROOT / "data" / "urbclim_raw_inspection.json"),
))

REGULARITY_TOL = 0.5   # metres — max spread in step size to call the grid "regular"
SAMPLE_TIMES   = 10    # number of timesteps used for NaN/value checks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(title=""):
    w = 60
    if title:
        pad = (w - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*(w-pad-len(title)-2)}")
    else:
        print("─" * w)


def _check_regularity(coords_1d: np.ndarray, label: str, tol: float = REGULARITY_TOL):
    """
    Returns (is_regular, step_mean, step_std, n_unique).
    A coordinate array is 'regular' if all gaps are within tol of the mean gap.
    """
    unique = np.sort(np.unique(coords_1d))
    if len(unique) < 2:
        return False, 0.0, 0.0, len(unique)
    steps = np.diff(unique)
    step_mean = float(steps.mean())
    step_std  = float(steps.std())
    is_reg = step_std < tol
    status = "REGULAR" if is_reg else "IRREGULAR"
    print(f"  {label}: {len(unique)} unique values  |  "
          f"step mean={step_mean:.2f} m  std={step_std:.4f} m  → {status}")
    return is_reg, step_mean, step_std, len(unique)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def inspect(raw_path: str) -> dict:
    _sep("OPENING ZARR")
    print(f"  Path: {raw_path}")
    ds = xr.open_zarr(raw_path, consolidated=False)
    print(ds)

    report = {"raw_path": raw_path}

    # ── 1. Dimensions & variables ──────────────────────────────────────────
    _sep("DIMENSIONS & VARIABLES")
    report["dims"]      = dict(ds.sizes)
    report["variables"] = list(ds.data_vars)
    report["coords"]    = list(ds.coords)
    for k, v in ds.sizes.items():
        print(f"  dim '{k}': {v}")
    print(f"  data_vars: {list(ds.data_vars)}")
    print(f"  coords:    {list(ds.coords)}")

    # ── 2. Identify station / spatial / time dims ──────────────────────────
    _sep("DIM ROLES")

    # Station dim: any dim that is not 'time' and is used for spatial coverage
    time_dim = next((d for d in ds.sizes if d in ("time", "valid_time")), None)
    if time_dim is None:
        raise RuntimeError("No time dimension found.")
    report["time_dim"] = time_dim
    print(f"  time dim  : '{time_dim}'  ({ds.sizes[time_dim]} steps)")

    # Spatial dim: could be 'weatherStation', 'station', 'points', or x/y
    spatial_candidates = [d for d in ds.sizes if d != time_dim]
    report["spatial_dims"] = spatial_candidates
    print(f"  spatial dims: {spatial_candidates}")

    # ── 3. Check for lat_lon string station IDs (Path C fast-path) ────────
    _sep("STATION ID FORMAT")

    station_dim = spatial_candidates[0] if spatial_candidates else None
    has_latlon_ids = False
    if station_dim and station_dim in ds.coords:
        sample_ids = ds[station_dim].values[:5]
        if all(isinstance(s, str) and "_" in s for s in sample_ids):
            try:
                float(sample_ids[0].split("_")[0])
                float(sample_ids[0].split("_")[1])
                has_latlon_ids = True
            except ValueError:
                pass

    report["has_latlon_ids"]  = has_latlon_ids
    report["station_dim"]     = station_dim
    if has_latlon_ids:
        print(f"  ✓ Station dim '{station_dim}' has lat_lon string IDs  e.g. '{sample_ids[0]}'")
        print(f"  → METHOD: latlon_reshape (fastest — no reprojection needed)")
    else:
        print(f"  ✗ Station dim does not carry lat_lon strings → will use x/y coords")

    # ── 4. Coordinates / x,y variables ────────────────────────────────────
    _sep("COORDINATES")

    # Try to find x, y coords — could be dim-coords or data variables
    x_vals = y_vals = None
    x_name = y_name = None

    for xn in ("x", "X", "easting", "lon", "longitude"):
        if xn in ds.coords or xn in ds.data_vars:
            candidate = ds[xn]
            # If (time, station) pick time=0; if (station,) use directly
            if time_dim in candidate.dims:
                candidate = candidate.isel({time_dim: 0})
            x_vals = candidate.values.ravel().astype(np.float64)
            x_name = xn
            break

    for yn in ("y", "Y", "northing", "lat", "latitude"):
        if yn in ds.coords or yn in ds.data_vars:
            candidate = ds[yn]
            if time_dim in candidate.dims:
                candidate = candidate.isel({time_dim: 0})
            y_vals = candidate.values.ravel().astype(np.float64)
            y_name = yn
            break

    if x_vals is None or y_vals is None:
        print("  WARNING: could not identify x/y coordinate variables.")
        print(f"  Available: {list(ds.coords) + list(ds.data_vars)}")
        report["coord_x_name"] = None
        report["coord_y_name"] = None
    else:
        report["coord_x_name"] = x_name
        report["coord_y_name"] = y_name
        print(f"  x var = '{x_name}'  range [{x_vals.min():.1f}, {x_vals.max():.1f}]")
        print(f"  y var = '{y_name}'  range [{y_vals.min():.1f}, {y_vals.max():.1f}]")

        # Guess CRS from value ranges
        if x_vals.max() > 1e6:
            crs_guess = "EPSG:3035 (Lambert EA) — values are metres"
        elif -180 <= x_vals.min() and x_vals.max() <= 180:
            crs_guess = "WGS84 — values are degrees"
        else:
            crs_guess = "unknown"
        report["crs_guess"] = crs_guess
        print(f"  CRS guess : {crs_guess}")

    # ── 4. Grid regularity ─────────────────────────────────────────────────
    _sep("GRID REGULARITY (key for method selection)")

    report["is_regular_grid"] = False
    report["grid_rows"]  = None
    report["grid_cols"]  = None
    report["step_x_m"]   = None
    report["step_y_m"]   = None

    if x_vals is not None and y_vals is not None:
        reg_x, step_x, std_x, nx = _check_regularity(x_vals, "x (easting) ")
        reg_y, step_y, std_y, ny = _check_regularity(y_vals, "y (northing)")

        n_stations = len(x_vals)
        sqrt_n     = int(round(n_stations ** 0.5))
        is_square  = abs(sqrt_n ** 2 - n_stations) == 0

        if reg_x and reg_y:
            print(f"\n  ✓ Both axes are regular (std < {REGULARITY_TOL} m)")
            # Try to figure out grid shape
            if is_square:
                print(f"  ✓ n_stations={n_stations} is a perfect square → {sqrt_n}×{sqrt_n} grid")
                report["grid_rows"] = sqrt_n
                report["grid_cols"] = sqrt_n
            else:
                # Check if nx * ny == n_stations
                if nx * ny == n_stations:
                    print(f"  ✓ {nx}×{ny} = {n_stations} stations")
                    report["grid_rows"] = ny
                    report["grid_cols"] = nx
                else:
                    print(f"  ? nx={nx}, ny={ny}, nx*ny={nx*ny}, n_stations={n_stations} — mismatch")
            report["is_regular_grid"] = True
            report["step_x_m"]        = round(step_x, 3)
            report["step_y_m"]        = round(step_y, 3)
            print(f"\n  → METHOD: rasterio warp (bilinear reprojection) recommended")
        else:
            print(f"\n  ✗ Grid is NOT regular (step std ≥ {REGULARITY_TOL} m)")
            print(f"  → METHOD: IDW with precomputed weights")

        report["n_stations"] = n_stations
        report["step_x_std"] = round(float(std_x), 4)
        report["step_y_std"] = round(float(std_y), 4)

    # ── 5. Temporal coverage ───────────────────────────────────────────────
    _sep("TEMPORAL COVERAGE")

    times = pd.to_datetime(ds[time_dim].values)
    report["n_timesteps"]  = int(len(times))
    report["time_start"]   = str(times[0])
    report["time_end"]     = str(times[-1])
    report["time_dtype"]   = str(ds[time_dim].dtype)

    print(f"  n_timesteps : {len(times)}")
    print(f"  range       : {times[0]}  →  {times[-1]}")
    print(f"  dtype       : {ds[time_dim].dtype}")

    # Check for gaps (expected: hourly = 1h between consecutive steps)
    if len(times) > 1:
        gaps = np.diff(times.asi8) / 1e9 / 3600  # seconds → hours
        step_hours_mode = float(pd.Series(gaps).mode().iloc[0])
        max_gap = float(gaps.max())
        n_gaps  = int((gaps > step_hours_mode * 1.5).sum())
        report["time_step_hours"] = step_hours_mode
        report["max_gap_hours"]   = max_gap
        report["n_gaps"]          = n_gaps
        print(f"  step mode   : {step_hours_mode:.1f} h")
        print(f"  max gap     : {max_gap:.1f} h")
        if n_gaps:
            print(f"  WARNING: {n_gaps} gaps > 1.5× nominal step")
        else:
            print(f"  No temporal gaps detected")

    # ── 6. Target variable: airTemperature ────────────────────────────────
    _sep("airTemperature — NaN & VALUE RANGE")

    temp_var = None
    for vn in ("airTemperature", "tas", "t2m", "temp", "temperature"):
        if vn in ds.data_vars:
            temp_var = vn
            break

    if temp_var is None:
        print(f"  WARNING: no temperature variable found. vars={list(ds.data_vars)}")
        report["temp_var"] = None
    else:
        report["temp_var"] = temp_var
        print(f"  Variable: '{temp_var}'  shape: {ds[temp_var].shape}  dtype: {ds[temp_var].dtype}")

        # Sample SAMPLE_TIMES timesteps spread across the dataset
        idx_sample = np.linspace(0, len(times) - 1, SAMPLE_TIMES, dtype=int)
        sample = ds[temp_var].isel({time_dim: idx_sample.tolist()}).values.astype(np.float32)

        nan_frac  = float(np.isnan(sample).mean())
        val_min   = float(np.nanmin(sample))
        val_max   = float(np.nanmax(sample))
        val_mean  = float(np.nanmean(sample))
        val_units = "K" if val_mean > 200 else "°C"

        report["temp_nan_frac"] = round(nan_frac, 4)
        report["temp_min"]      = round(val_min, 3)
        report["temp_max"]      = round(val_max, 3)
        report["temp_mean"]     = round(val_mean, 3)
        report["temp_units_guess"] = val_units

        print(f"  NaN fraction (sample): {nan_frac:.2%}")
        print(f"  Value range  (sample): [{val_min:.2f}, {val_max:.2f}]  mean={val_mean:.2f}")
        print(f"  Units guess          : {val_units}")

        if val_units == "K":
            print(f"  NOTE: values appear to be in Kelvin — will subtract 273.15 during preprocessing")
        if nan_frac > 0.05:
            print(f"  WARNING: {nan_frac:.1%} NaN rate — check for sea/out-of-domain stations")

    # ── 7. Memory estimate ────────────────────────────────────────────────
    _sep("MEMORY ESTIMATE (full dataset)")

    n_t = report.get("n_timesteps", 0)
    n_s = report.get("n_stations", 0)
    if n_t and n_s:
        bytes_full = n_t * n_s * 4
        print(f"  Raw (time×stations×float32): {bytes_full/1e9:.2f} GB")
        out_pixels = 251 * 251
        bytes_out  = n_t * out_pixels * 4
        print(f"  Output (time×251×251×float32): {bytes_out/1e9:.2f} GB")
        batch_1000 = 1000 * n_s * 4
        print(f"  Per-batch I/O (1000 timesteps): {batch_1000/1e6:.0f} MB")
        report["mem_raw_gb"]    = round(bytes_full / 1e9, 2)
        report["mem_out_gb"]    = round(bytes_out  / 1e9, 2)
        report["mem_batch_mb"]  = round(batch_1000 / 1e6, 1)

    # ── 8. Recommended method summary ────────────────────────────────────
    _sep("RECOMMENDATION")

    if report.get("has_latlon_ids"):
        method = "latlon_reshape"
        reason = ("weatherStation IDs are already WGS84 lat_lon strings. "
                  "Parse directly and reshape — zero reprojection cost.")
    elif report.get("is_regular_grid"):
        method = "rasterio_warp"
        reason = (f"Grid is regular ({report.get('step_x_m')} m × {report.get('step_y_m')} m). "
                  "Bilinear reprojection is faster and more accurate than IDW.")
    else:
        method = "idw_precomputed"
        reason = "Grid is irregular. Use IDW with precomputed weights (K=12, power=2)."

    report["recommended_method"] = method
    print(f"  Method : {method}")
    print(f"  Reason : {reason}")

    # ── Save JSON ──────────────────────────────────────────────────────────
    _sep()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nInspection saved → {OUT_JSON}")
    print("Pass this file to preprocess_urbclim.py (it reads it automatically).\n")

    return report


if __name__ == "__main__":
    if not Path(RAW_PATH).exists():
        print(f"ERROR: zarr not found at {RAW_PATH}")
        print("Set URBCLIM_RAW=/path/to/raw_urbclim.zarr and re-run.")
        sys.exit(1)
    inspect(RAW_PATH)
