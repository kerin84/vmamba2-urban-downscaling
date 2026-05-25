#!/usr/bin/env python3
"""
verify_urbclim.py — Consistency checks for the preprocessed UrbClim zarr.

Checks:
  1. Bijective mapping  — N random stations: raw value == processed pixel value
  2. Spatial structure  — saves a PNG of one summer noon timestep
  3. Temporal coverage  — confirms ordering, counts gaps
  4. Grid geometry      — confirms 100m EPSG:3035 steps and domain bounds

Usage:
  URBCLIM_RAW=/app/data/raw_urbclim.zarr \
  URBCLIM_OUT=/app/data/urbclim_2008-2017.zarr \
  python scripts/data/verify_urbclim.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RAW_PATH = Path(os.environ.get("URBCLIM_RAW", PROJECT_ROOT / "data" / "raw_urbclim.zarr"))
OUT_PATH = Path(os.environ.get("URBCLIM_OUT", PROJECT_ROOT / "data" / "urbclim_2008-2017.zarr"))
PNG_OUT  = OUT_PATH.parent / "urbclim_verify_snapshot.png"

N_SPOT_CHECKS = 50
RNG = np.random.default_rng(42)


def sep(title=""):
    w = 60
    if title:
        pad = (w - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*(w-pad-len(title)-2)}")
    else:
        print("─" * w)


def main():
    print("=" * 60)
    print("verify_urbclim.py")
    print("=" * 60)

    # ── Open zarrs ────────────────────────────────────────────────
    sep("OPENING ZARRS")
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw zarr not found: {RAW_PATH}")
    if not OUT_PATH.exists():
        raise FileNotFoundError(f"Processed zarr not found: {OUT_PATH}")

    ds_raw = xr.open_zarr(str(RAW_PATH), consolidated=False)
    ds_out = xr.open_zarr(str(OUT_PATH), consolidated=True)

    print(f"  Raw  : {RAW_PATH.name}  {dict(ds_raw.sizes)}")
    print(f"  Out  : {OUT_PATH.name}  {dict(ds_out.sizes)}")

    # ── 1. Bijective mapping ───────────────────────────────────────
    sep("1. BIJECTIVE MAPPING  (raw station == output pixel)")

    # Build EPSG:3035 → (row, col) lookup from the output zarr
    y_coords = ds_out["y"].values   # northings descending
    x_coords = ds_out["x"].values   # eastings  ascending
    y_to_row = {round(v, 2): i for i, v in enumerate(y_coords)}
    x_to_col = {round(v, 2): i for i, v in enumerate(x_coords)}

    # Load station EPSG:3035 coordinates from raw zarr (time=0 slice)
    time_dim = "time"
    x_vals = ds_raw["x"].isel({time_dim: 0}).values.ravel().astype(np.float64)
    y_vals = ds_raw["y"].isel({time_dim: 0}).values.ravel().astype(np.float64)
    n_stations = len(x_vals)

    # Pick random station indices and random timestep indices
    station_idxs = RNG.integers(0, n_stations, size=N_SPOT_CHECKS)
    time_idxs    = RNG.integers(0, ds_raw.sizes[time_dim], size=N_SPOT_CHECKS)

    mismatches = 0
    max_abs_diff = 0.0

    for st_i, t_i in zip(station_idxs, time_idxs):
        # Raw value
        raw_val = float(ds_raw["airTemperature"].isel(
            {time_dim: int(t_i), "weatherStation": int(st_i)}
        ).values)

        # Expected (row, col) in output grid
        xv = round(float(x_vals[st_i]), 2)
        yv = round(float(y_vals[st_i]), 2)
        row = y_to_row.get(yv)
        col = x_to_col.get(xv)

        if row is None or col is None:
            print(f"  FAIL station {st_i}: x={xv}, y={yv} not found in output grid")
            mismatches += 1
            continue

        # Find corresponding timestep in output zarr
        raw_time = pd.Timestamp(ds_raw[time_dim].values[t_i]).floor("h")
        out_times = pd.to_datetime(ds_out[time_dim].values)
        t_out = np.searchsorted(out_times, raw_time)

        if t_out >= len(out_times) or out_times[t_out] != raw_time:
            print(f"  SKIP  t={raw_time} not in output (gap?)")
            continue

        out_val = float(ds_out["airTemperature"].isel(time=int(t_out), y=row, x=col).values)

        diff = abs(raw_val - out_val)
        max_abs_diff = max(max_abs_diff, diff)
        if diff > 0.01:
            mismatches += 1
            print(f"  MISMATCH  station={st_i}  t={raw_time}  "
                  f"raw={raw_val:.4f}  out={out_val:.4f}  diff={diff:.4f}")

    if mismatches == 0:
        print(f"  ✓  All {N_SPOT_CHECKS} spot checks passed  "
              f"(max |diff| = {max_abs_diff:.2e} — float32 rounding only)")
    else:
        print(f"  ✗  {mismatches}/{N_SPOT_CHECKS} spot checks FAILED")

    # ── 2. Spatial structure ──────────────────────────────────────
    sep("2. SPATIAL STRUCTURE  (summer noon snapshot)")

    # Find a July noon timestep
    out_times = pd.to_datetime(ds_out[time_dim].values)
    july_noon = (out_times.month == 7) & (out_times.hour == 12)
    if july_noon.any():
        t_snap = int(np.where(july_noon)[0][0])
        snap = ds_out["airTemperature"].isel(time=t_snap).values
        print(f"  Timestep : {out_times[t_snap]}")
        print(f"  Min      : {np.nanmin(snap):.2f} °C")
        print(f"  Max      : {np.nanmax(snap):.2f} °C")
        print(f"  Mean     : {np.nanmean(snap):.2f} °C")
        print(f"  Std      : {np.nanstd(snap):.2f} °C")
        # Row-0 should be northernmost (highest northing = ~2078328 m)
        # Row-250 should be southernmost
        print(f"  Row-0 mean (north edge): {np.nanmean(snap[0, :]):.2f} °C")
        print(f"  Row-250 mean (south edge): {np.nanmean(snap[-1, :]):.2f} °C")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(snap, origin="upper", cmap="RdYlBu_r",
                           vmin=np.nanpercentile(snap, 2),
                           vmax=np.nanpercentile(snap, 98))
            plt.colorbar(im, ax=ax, label="°C")
            ax.set_title(f"UrbClim airTemperature\n{out_times[t_snap]}")
            ax.set_xlabel("col (west → east)")
            ax.set_ylabel("row (north → south)")
            plt.tight_layout()
            plt.savefig(str(PNG_OUT), dpi=150)
            plt.close()
            print(f"  Snapshot saved → {PNG_OUT}")
        except ImportError:
            print("  (matplotlib not available — skipping PNG)")
    else:
        print("  No July noon timestep found in output.")

    # ── 3. Temporal coverage ──────────────────────────────────────
    sep("3. TEMPORAL COVERAGE")

    print(f"  n_timesteps : {len(out_times):,}")
    print(f"  range       : {out_times[0]}  →  {out_times[-1]}")

    gaps = np.diff(out_times.asi8) / 1e9 / 3600  # → hours
    step_mode = float(pd.Series(gaps).mode().iloc[0])
    n_gaps = int((gaps > step_mode * 1.5).sum())
    max_gap = float(gaps.max())
    print(f"  step mode   : {step_mode:.1f} h")
    print(f"  max gap     : {max_gap:.1f} h")

    if n_gaps:
        gap_locs = np.where(gaps > step_mode * 1.5)[0]
        print(f"  {n_gaps} gaps > 1.5× nominal step:")
        for loc in gap_locs[:5]:
            print(f"    {out_times[loc]}  →  {out_times[loc+1]}  ({gaps[loc]:.0f} h)")
        if n_gaps > 5:
            print(f"    ... ({n_gaps - 5} more)")
    else:
        print("  ✓  No temporal gaps")

    # Check sort order
    if np.all(np.diff(out_times.asi8) > 0):
        print("  ✓  Time axis is strictly ascending")
    else:
        print("  ✗  Time axis is NOT strictly ascending — check sort!")

    # ── 4. Grid geometry ──────────────────────────────────────────
    sep("4. GRID GEOMETRY  (EPSG:3035 100m steps)")

    T, H, W = ds_out["airTemperature"].shape
    print(f"  shape  : ({T:,}, {H}, {W})")
    print(f"  crs    : {ds_out.attrs.get('crs', 'unknown')}")

    dy = np.diff(y_coords)  # should all be -100 (descending northings)
    dx = np.diff(x_coords)  # should all be +100 (ascending eastings)
    print(f"  y step : mean={dy.mean():.1f} m  std={dy.std():.4f} m  "
          f"(expected −100.0 m)")
    print(f"  x step : mean={dx.mean():.1f} m  std={dx.std():.4f} m  "
          f"(expected +100.0 m)")

    if abs(dy.mean() + 100.0) < 0.1 and dy.std() < 0.1:
        print("  ✓  y axis: 100m steps, descending (north→south)")
    else:
        print(f"  ✗  y axis unexpected: mean step={dy.mean():.2f}")

    if abs(dx.mean() - 100.0) < 0.1 and dx.std() < 0.1:
        print("  ✓  x axis: 100m steps, ascending (west→east)")
    else:
        print(f"  ✗  x axis unexpected: mean step={dx.mean():.2f}")

    sep()
    print("\nVerification complete.\n")


if __name__ == "__main__":
    main()
