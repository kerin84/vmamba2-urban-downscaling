#!/usr/bin/env python3
"""
preprocess_urbclim.py — Convert raw UrbClim station zarr → gridded (time, y, x) zarr.

Reads the JSON produced by inspect_urbclim_raw.py and selects the method:

  Path A — rasterio bilinear warp
    Triggered when source stations form a regular EPSG:3035 grid.
    Reprojects (T, rows, cols) batches to the WGS84 target grid.
    Recommended: ~10× faster and mathematically cleaner than IDW for regular grids.

  Path B — IDW with precomputed weights
    Triggered for irregular station networks.
    K=12 neighbours, power=2 — matches legacy hourly_generator.py.
    Weights are computed once and cached in data/.idw_weights.npz.
    Second run (or interrupted+resumed run) skips weight computation entirely.

  Path C — latlon_reshape (no reprojection)
    Triggered when weatherStation dimension IDs are already WGS84 lat_lon strings
    like '41.26025_1.998952'.  Parse lat/lon directly from the IDs, build a
    station→pixel index, and reshape (T, N_stations) → (T, H, W).
    Fastest path — zero reprojection cost.

Output:
  data/urbclim_2008-2017.zarr
    airTemperature : (time, y, x)  float32  °C
    time           : datetime64[ns]
    lat            : (y,)  float32  WGS84 latitude,  descending (north→south)
    lon            : (x,)  float32  WGS84 longitude, ascending  (west→east)
  Chunks: (24, 251, 251) — 1 day per chunk, full spatial slice.

Resume:
  Completed years are tracked in data/.urbclim_progress.json.
  Re-running skips already-done years and continues from where it stopped.
  Interrupt at any point; no work is lost.

Usage:
  URBCLIM_RAW=/path/to/raw_urbclim.zarr python scripts/data/preprocess_urbclim.py

  # Override detected method:
  URBCLIM_METHOD=rasterio_warp     python scripts/data/preprocess_urbclim.py
  URBCLIM_METHOD=idw_precomputed   python scripts/data/preprocess_urbclim.py
  URBCLIM_METHOD=latlon_reshape    python scripts/data/preprocess_urbclim.py

  # Override output path:
  URBCLIM_OUT=/custom/path.zarr    python scripts/data/preprocess_urbclim.py
"""

from __future__ import annotations

import json
import os
import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

RAW_PATH        = Path(os.environ.get("URBCLIM_RAW",   PROJECT_ROOT / "data" / "raw_urbclim.zarr"))
OUT_PATH        = Path(os.environ.get("URBCLIM_OUT",   PROJECT_ROOT / "data" / "urbclim_2008-2017.zarr"))
GRID_COORDS_TXT = PROJECT_ROOT / "data" / "urbclim_grid_coords.txt"
INSPECTION_JSON = Path(os.environ.get(
    "URBCLIM_INSPECT_JSON",
    str(PROJECT_ROOT / "data" / "urbclim_raw_inspection.json"),
))
WEIGHTS_NPZ     = PROJECT_ROOT / "data" / ".idw_weights.npz"
PROGRESS_JSON   = PROJECT_ROOT / "data" / ".urbclim_progress.json"
TMP_DIR         = PROJECT_ROOT / "data" / ".urbclim_tmp"

METHOD_OVERRIDE = os.environ.get("URBCLIM_METHOD", None)   # "rasterio_warp" | "idw_precomputed"

K_NEIGHBOURS = 12
POWER        = 2.0
BATCH_IDW    = 200    # timesteps per batch — memory: ~(200 × 63001 × 2) × float32 ≈ 100 MB
BATCH_WARP   = 500    # timesteps per batch — rasterio is fast, larger batches OK
TARGET_H = TARGET_W = 251


# ---------------------------------------------------------------------------
# Target grid helpers
# ---------------------------------------------------------------------------

def _load_wgs84_target_grid(x_vals: np.ndarray, y_vals: np.ndarray,
                              n: int = TARGET_H) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a regular WGS84 n×n target grid covering the bounding box of the
    EPSG:3035 station points (reprojects corners via pyproj).

    Returns (lat_1d, lon_1d):
      lat_1d : (n,) float64, descending  (row 0 = northernmost)
      lon_1d : (n,) float64, ascending   (col 0 = westernmost)
    """
    from pyproj import Transformer
    tf = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    lons, lats = tf.transform(x_vals, y_vals)

    lat_1d = np.linspace(lats.max(), lats.min(), n, dtype=np.float64)  # descending
    lon_1d = np.linspace(lons.min(), lons.max(), n, dtype=np.float64)  # ascending
    print(f"  Target WGS84 grid: {n}×{n}")
    print(f"    lat [{lat_1d[-1]:.5f}, {lat_1d[0]:.5f}]  "
          f"lon [{lon_1d[0]:.5f}, {lon_1d[-1]:.5f}]")
    return lat_1d, lon_1d


def _build_epsg3035_grid(x_vals: np.ndarray, y_vals: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the EPSG:3035 1D coordinate arrays from flat station vectors.

    Returns (grid_to_zarr, y_3035, x_3035):
      grid_to_zarr : (TARGET_H * TARGET_W,) int32  — pixel → zarr station index
      y_3035       : (TARGET_H,) float64 northings, descending
      x_3035       : (TARGET_W,) float64 eastings, ascending
    """
    unique_x = np.sort(np.unique(x_vals))        # ascending
    unique_y = np.sort(np.unique(y_vals))[::-1]  # descending (north→south)

    if len(unique_y) != TARGET_H or len(unique_x) != TARGET_W:
        raise ValueError(
            f"Expected {TARGET_H}×{TARGET_W} regular EPSG:3035 grid, "
            f"got {len(unique_y)} unique northings × {len(unique_x)} unique eastings."
        )

    x_to_col = {round(v, 2): i for i, v in enumerate(unique_x)}
    y_to_row = {round(v, 2): i for i, v in enumerate(unique_y)}

    zarr_to_pixel = np.array([
        y_to_row[round(yv, 2)] * TARGET_W + x_to_col[round(xv, 2)]
        for xv, yv in zip(x_vals, y_vals)
    ], dtype=np.int32)

    # Validate bijective mapping
    counts = np.bincount(zarr_to_pixel, minlength=TARGET_H * TARGET_W)
    if not np.all(counts == 1):
        bad = int((counts != 1).sum())
        raise ValueError(f"Grid mapping not bijective: {bad} pixel positions have count ≠ 1.")

    grid_to_zarr = np.empty(len(x_vals), dtype=np.int32)
    grid_to_zarr[zarr_to_pixel] = np.arange(len(x_vals), dtype=np.int32)

    print(f"  EPSG:3035 grid: {len(unique_y)}×{len(unique_x)}")
    print(f"    northing [{unique_y[-1]:.0f}, {unique_y[0]:.0f}]  "
          f"easting [{unique_x[0]:.0f}, {unique_x[-1]:.0f}]")
    return grid_to_zarr, unique_y.astype(np.float64), unique_x.astype(np.float64)


# ---------------------------------------------------------------------------
# Progress / resume helpers
# ---------------------------------------------------------------------------

def _load_progress() -> dict:
    if PROGRESS_JSON.exists():
        return json.loads(PROGRESS_JSON.read_text())
    return {"done": []}


def _save_progress(p: dict):
    PROGRESS_JSON.write_text(json.dumps(p, indent=2))


# ---------------------------------------------------------------------------
# Zarr I/O helpers
# ---------------------------------------------------------------------------

def _write_year_zarr(data: np.ndarray, times: pd.DatetimeIndex,
                     y_coords: np.ndarray, x_coords: np.ndarray,
                     crs: str, path: Path):
    """Write one year's data to a temporary zarr."""
    ds = xr.Dataset(
        {"airTemperature": (("time", "y", "x"), data)},
        coords={
            "time": times,
            "y":    ("y", y_coords.astype(np.float64)),
            "x":    ("x", x_coords.astype(np.float64)),
        },
    )
    ds.attrs["units"] = "degC"
    ds.attrs["crs"]   = crs
    if crs == "EPSG:3035":
        ds["y"].attrs["long_name"] = "northing_m"
        ds["x"].attrs["long_name"] = "easting_m"
    else:
        ds["y"].attrs["long_name"] = "latitude"
        ds["x"].attrs["long_name"] = "longitude"
    if path.exists():
        shutil.rmtree(path)
    ds.to_zarr(str(path), mode="w")


def _merge_and_finalise(year_paths: list[Path], y_coords: np.ndarray,
                         x_coords: np.ndarray, crs: str):
    """
    Write per-year zarrs → final output with (24, 251, 251) chunks.
    Memory-safe: fills the final zarr year-by-year via direct zarr writes,
    never materialising the full dataset in RAM.
    """
    import zarr as zarr_module

    print("\nMerging year zarrs into final output...")

    # Collect sorted time arrays across all years
    year_datasets = [xr.open_zarr(str(p), consolidated=False) for p in year_paths]
    all_times = np.concatenate([ds.time.values for ds in year_datasets])
    T_total = len(all_times)
    print(f"  Total timesteps: {T_total:,}  ({len(year_paths)} years)")

    if OUT_PATH.exists():
        shutil.rmtree(OUT_PATH)
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    # Create zarr store and arrays directly — no full in-memory allocation
    z = zarr_module.open(str(OUT_PATH), mode="w")

    time_ns = all_times.astype("datetime64[ns]").astype(np.int64)
    z_time = z.create_dataset("time", data=time_ns, dtype="int64",
                               chunks=(min(T_total, 8760),))
    z_time.attrs["_ARRAY_DIMENSIONS"] = ["time"]
    z_time.attrs["units"] = "nanoseconds since 1970-01-01"
    z_time.attrs["calendar"] = "proleptic_gregorian"

    z_y = z.create_dataset("y", data=y_coords.astype(np.float64), dtype="float64")
    z_y.attrs["_ARRAY_DIMENSIONS"] = ["y"]
    z_y.attrs["long_name"] = "northing_m" if crs == "EPSG:3035" else "latitude"

    z_x = z.create_dataset("x", data=x_coords.astype(np.float64), dtype="float64")
    z_x.attrs["_ARRAY_DIMENSIONS"] = ["x"]
    z_x.attrs["long_name"] = "easting_m" if crs == "EPSG:3035" else "longitude"

    z_temp = z.create_dataset(
        "airTemperature",
        shape=(T_total, TARGET_H, TARGET_W),
        chunks=(24, TARGET_H, TARGET_W),
        dtype="float32",
        fill_value=np.nan,
    )
    z_temp.attrs["_ARRAY_DIMENSIONS"] = ["time", "y", "x"]
    z_temp.attrs["units"] = "degC"

    z.attrs["crs"] = crs
    z.attrs["description"] = (
        f"UrbClim airTemperature preprocessed to {TARGET_H}×{TARGET_W} {crs} grid. "
        "Generated by scripts/data/preprocess_urbclim.py."
    )
    zarr_module.consolidate_metadata(str(OUT_PATH))

    # Fill data year by year — peak RAM ≈ 1 year × 251 × 251 × 4 bytes (~1.3 GB)
    print(f"Writing → {OUT_PATH}")
    t_offset = 0
    for p, ds in zip(year_paths, year_datasets):
        T_year = len(ds.time)
        data = ds["airTemperature"].values.astype(np.float32)
        z_temp[t_offset : t_offset + T_year] = data
        t_offset += T_year
        print(f"  {p.stem}: wrote {T_year} frames  (offset {t_offset - T_year}–{t_offset})")
        del data

    print(f"  Done. {t_offset:,} total frames written.")


# ---------------------------------------------------------------------------
# Path A — rasterio bilinear warp
# ---------------------------------------------------------------------------

def _build_rasterio_transforms(insp: dict, x_vals: np.ndarray, y_vals: np.ndarray,
                                lat_1d: np.ndarray, lon_1d: np.ndarray):
    """
    Returns (src_transform, src_crs, dst_transform, dst_crs, zarr_to_grid_idx, rows, cols).
    zarr_to_grid_idx[k] = zarr station index that maps to flat grid position k (row-major).
    """
    try:
        import rasterio
        from rasterio.transform import from_origin
        from rasterio.crs import CRS
    except ImportError:
        raise ImportError(
            "rasterio is required for Path A. Install with: pip install rasterio>=1.3"
        )

    rows = insp["grid_rows"]
    cols = insp["grid_cols"]
    step_x = insp["step_x_m"]
    step_y = insp["step_y_m"]

    # Sort stations into row-major order (y descending = north first, x ascending)
    unique_x = np.sort(np.unique(x_vals))          # ascending  (west→east)
    unique_y = np.sort(np.unique(y_vals))[::-1]    # descending (north→south)

    if len(unique_y) != rows or len(unique_x) != cols:
        raise ValueError(
            f"Expected {rows}×{cols} grid, got {len(unique_y)} unique Y, {len(unique_x)} unique X."
        )

    # Map zarr station index → flat grid position
    x_tol = step_x * 0.1
    y_tol = step_y * 0.1
    x_to_col = {x: i for i, x in enumerate(unique_x)}
    y_to_row = {y: i for i, y in enumerate(unique_y)}

    zarr_to_grid_idx = np.empty(len(x_vals), dtype=np.int32)
    for i, (xi, yi) in enumerate(zip(x_vals, y_vals)):
        # Nearest unique x/y (tolerant of float rounding)
        ci = int(np.argmin(np.abs(unique_x - xi)))
        ri = int(np.argmin(np.abs(unique_y - yi)))
        zarr_to_grid_idx[i] = ri * cols + ci

    # Validate: each grid position appears exactly once
    counts = np.bincount(zarr_to_grid_idx, minlength=rows * cols)
    if not np.all(counts == 1):
        bad = int((counts != 1).sum())
        raise ValueError(
            f"Grid mapping is not bijective: {bad} positions have count ≠ 1. "
            "Check that station coordinates are a perfect regular grid."
        )

    # Source affine transform (EPSG:3035)
    x_origin = unique_x[0] - step_x / 2   # western edge of col-0
    y_origin = unique_y[0] + step_y / 2   # northern edge of row-0
    src_transform = from_origin(x_origin, y_origin, step_x, step_y)
    src_crs = CRS.from_epsg(3035)

    # Destination affine transform (WGS84 251×251)
    dlat = abs(lat_1d[0] - lat_1d[-1]) / (len(lat_1d) - 1)
    dlon = abs(lon_1d[-1] - lon_1d[0]) / (len(lon_1d) - 1)
    dst_transform = from_origin(
        west=lon_1d[0]  - dlon / 2,
        north=lat_1d[0] + dlat / 2,
        xsize=dlon,
        ysize=dlat,
    )
    dst_crs = CRS.from_epsg(4326)

    return src_transform, src_crs, dst_transform, dst_crs, zarr_to_grid_idx, rows, cols


def _process_year_rasterio(ds_raw, year: int, insp: dict,
                            zarr_to_grid_idx, rows, cols,
                            src_transform, src_crs,
                            dst_transform, dst_crs,
                            lat_1d, lon_1d) -> tuple[np.ndarray, pd.DatetimeIndex]:
    from rasterio.warp import reproject, Resampling

    time_dim  = insp["time_dim"]
    temp_var  = insp["temp_var"]
    kelvin    = insp.get("temp_units_guess") == "K"

    all_times = pd.to_datetime(ds_raw[time_dim].values).floor("h")
    mask      = all_times.year == year
    pos_arr   = np.where(mask)[0]
    times_year = all_times[mask]
    T_year    = len(pos_arr)

    result    = np.empty((T_year, TARGET_H, TARGET_W), dtype=np.float32)
    dst_buf   = np.empty((BATCH_WARP, TARGET_H, TARGET_W), dtype=np.float32)

    pbar = tqdm(range(0, T_year, BATCH_WARP),
                desc=f"  {year} [warp]", unit="batch", leave=False)
    for b_start in pbar:
        b_end  = min(b_start + BATCH_WARP, T_year)
        T      = b_end - b_start
        pos    = pos_arr[b_start:b_end].tolist()

        raw = ds_raw[temp_var].isel({time_dim: pos}).values.astype(np.float32)
        if kelvin:
            raw -= 273.15

        # Reorder stations → row-major (T, rows, cols)
        src = raw[:, zarr_to_grid_idx].reshape(T, rows, cols)

        dst_t = dst_buf[:T]
        reproject(
            source=src,
            destination=dst_t,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        result[b_start:b_end] = dst_t

    return result, times_year


# ---------------------------------------------------------------------------
# Path B — IDW with precomputed weights
# ---------------------------------------------------------------------------

def _build_idw_weights(insp: dict, x_vals: np.ndarray, y_vals: np.ndarray,
                        lat_1d: np.ndarray, lon_1d: np.ndarray) -> tuple:
    """
    Returns (idxs, weights):
      idxs    : (TARGET_H*TARGET_W, K) int32  — station indices per target pixel
      weights : (TARGET_H*TARGET_W, K) float32 — normalised IDW weights

    Computation is expensive (cKDTree query over ~63 K pixels × N stations).
    Result is cached to WEIGHTS_NPZ and reloaded on subsequent runs.
    """
    if WEIGHTS_NPZ.exists():
        print(f"  Loading cached IDW weights from {WEIGHTS_NPZ}")
        npz = np.load(str(WEIGHTS_NPZ))
        return npz["idxs"], npz["weights"]

    print("  Computing IDW weights (one-time cost, ~1–5 min)...")

    # Transform station coords from EPSG:3035 → WGS84
    from pyproj import Transformer
    tf = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    st_lon, st_lat = tf.transform(x_vals, y_vals)
    station_pts = np.column_stack([st_lat, st_lon])   # (N, 2)

    # Build target pixel coordinates
    lat_2d, lon_2d = np.meshgrid(lat_1d, lon_1d, indexing="ij")   # (H, W)
    target_pts = np.column_stack([lat_2d.ravel(), lon_2d.ravel()])  # (H*W, 2)

    # KDTree query
    from scipy.spatial import cKDTree
    tree = cKDTree(station_pts)
    dists, idxs = tree.query(target_pts, k=K_NEIGHBOURS)           # (H*W, K) each

    # IDW weights: w = 1/d^p, normalised per pixel
    dists  = np.maximum(dists, 1e-9).astype(np.float32)
    w_raw  = 1.0 / (dists ** POWER)
    weights = (w_raw / w_raw.sum(axis=1, keepdims=True)).astype(np.float32)
    idxs    = idxs.astype(np.int32)

    np.savez_compressed(str(WEIGHTS_NPZ), idxs=idxs, weights=weights)
    print(f"  Weights saved → {WEIGHTS_NPZ}  ({WEIGHTS_NPZ.stat().st_size/1e6:.1f} MB)")
    return idxs, weights


def _process_year_idw(ds_raw, year: int, insp: dict,
                       idxs: np.ndarray, weights: np.ndarray,
                       lat_1d: np.ndarray, lon_1d: np.ndarray) -> tuple[np.ndarray, pd.DatetimeIndex]:
    time_dim = insp["time_dim"]
    temp_var = insp["temp_var"]
    kelvin   = insp.get("temp_units_guess") == "K"
    n_pixels = TARGET_H * TARGET_W

    all_times  = pd.to_datetime(ds_raw[time_dim].values).floor("h")
    mask       = all_times.year == year
    pos_arr    = np.where(mask)[0]
    times_year = all_times[mask]
    T_year     = len(pos_arr)

    result = np.empty((T_year, TARGET_H, TARGET_W), dtype=np.float32)

    pbar = tqdm(range(0, T_year, BATCH_IDW),
                desc=f"  {year} [IDW]", unit="batch", leave=False)
    for b_start in pbar:
        b_end = min(b_start + BATCH_IDW, T_year)
        T     = b_end - b_start
        pos   = pos_arr[b_start:b_end].tolist()

        # Load raw temps: (T, N_stations)
        raw = ds_raw[temp_var].isel({time_dim: pos}).values.astype(np.float32)
        if kelvin:
            raw -= 273.15

        # IDW: loop over K to avoid (T, n_pixels, K) intermediate tensor (~3 GB)
        # Instead: K passes of (T, n_pixels) each → peak RAM ≈ 2 × T × n_pixels × 4 bytes
        interp = np.zeros((T, n_pixels), dtype=np.float32)
        for k in range(K_NEIGHBOURS):
            # raw[:, idxs[:, k]] : (T, n_pixels) — no extra dim
            interp += raw[:, idxs[:, k]] * weights[:, k]  # (n_pixels,) broadcasts over T

        result[b_start:b_end] = interp.reshape(T, TARGET_H, TARGET_W)

    return result, times_year


# ---------------------------------------------------------------------------
# Path C — EPSG:3035 reshape (no reprojection)
# ---------------------------------------------------------------------------

def _process_year_latlon(ds_raw, year: int, insp: dict,
                          grid_to_zarr: np.ndarray) -> tuple[np.ndarray, pd.DatetimeIndex]:
    time_dim = insp["time_dim"]
    temp_var = insp["temp_var"]
    kelvin   = insp.get("temp_units_guess") == "K"

    all_times  = pd.to_datetime(ds_raw[time_dim].values).floor("h")
    mask       = all_times.year == year
    pos_arr    = np.where(mask)[0]
    times_year = all_times[mask]
    T_year     = len(pos_arr)

    result = np.empty((T_year, TARGET_H, TARGET_W), dtype=np.float32)

    pbar = tqdm(range(0, T_year, BATCH_IDW),
                desc=f"  {year} [reshape]", unit="batch", leave=False)
    for b_start in pbar:
        b_end = min(b_start + BATCH_IDW, T_year)
        T     = b_end - b_start
        pos   = pos_arr[b_start:b_end].tolist()

        raw = ds_raw[temp_var].isel({time_dim: pos}).values.astype(np.float32)
        if kelvin:
            raw -= 273.15

        # Reorder stations to row-major pixel order and reshape
        result[b_start:b_end] = raw[:, grid_to_zarr].reshape(T, TARGET_H, TARGET_W)

    return result, times_year


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("preprocess_urbclim.py")
    print("=" * 60)

    # ── 1. Load inspection report ─────────────────────────────────────────
    if not INSPECTION_JSON.exists():
        raise FileNotFoundError(
            f"Inspection JSON not found: {INSPECTION_JSON}\n"
            "Run scripts/data/inspect_urbclim_raw.py first."
        )
    insp = json.loads(INSPECTION_JSON.read_text())
    print(f"\nInspection: {INSPECTION_JSON.name}")
    print(f"  raw_path : {insp['raw_path']}")
    print(f"  method   : {insp['recommended_method']}")
    print(f"  n_times  : {insp['n_timesteps']:,}")
    print(f"  n_stations: {insp.get('n_stations', '?'):,}")

    # ── 2. Resolve method ─────────────────────────────────────────────────
    # Upgrade older inspection JSONs: if has_latlon_ids but method wasn't updated,
    # prefer latlon_reshape (it's strictly cheaper).
    rec = insp["recommended_method"]
    if insp.get("has_latlon_ids") and rec != "latlon_reshape":
        rec = "latlon_reshape"
        print("  NOTE: upgrading recommended_method → latlon_reshape (has_latlon_ids=True)")

    method = METHOD_OVERRIDE or rec
    valid_methods = ("rasterio_warp", "idw_precomputed", "latlon_reshape")
    if method not in valid_methods:
        raise ValueError(f"Unknown method '{method}'. Use one of {valid_methods}.")
    if method == "rasterio_warp" and not insp.get("is_regular_grid"):
        print("  WARNING: rasterio_warp requested but grid is not regular. Falling back to IDW.")
        method = "idw_precomputed"
    print(f"\nUsing method: {method}")

    # ── 3. Open raw zarr ─────────────────────────────────────────────────
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw zarr not found: {RAW_PATH}")
    print(f"\nOpening raw zarr: {RAW_PATH}")
    ds_raw = xr.open_zarr(str(RAW_PATH), consolidated=False)

    time_dim = insp["time_dim"]

    # Always load x/y EPSG:3035 station coordinates
    x_name = insp.get("coord_x_name", "x")
    y_name = insp.get("coord_y_name", "y")
    x_vals = ds_raw[x_name].isel({time_dim: 0}).values.ravel().astype(np.float64)
    y_vals = ds_raw[y_name].isel({time_dim: 0}).values.ravel().astype(np.float64)
    print(f"  Stations: {len(x_vals):,}  "
          f"x=[{x_vals.min():.0f}, {x_vals.max():.0f}]  "
          f"y=[{y_vals.min():.0f}, {y_vals.max():.0f}]  (EPSG:3035 m)")

    # ── 4. Build spatial index ────────────────────────────────────────────
    print("\nBuilding spatial index...")
    if method == "latlon_reshape":
        # Path C: keep native EPSG:3035 grid — no reprojection
        grid_to_zarr, y_coords, x_coords = _build_epsg3035_grid(x_vals, y_vals)
        output_crs = "EPSG:3035"
        print(f"  Built EPSG:3035 reshape index: {len(grid_to_zarr):,} stations")
    elif method == "rasterio_warp":
        # Path A: reproject EPSG:3035 → WGS84
        print("  Building WGS84 target grid from bounding box...")
        lat_1d, lon_1d = _load_wgs84_target_grid(x_vals, y_vals)
        src_tr, src_crs, dst_tr, dst_crs, zarr_to_grid, rows, cols = \
            _build_rasterio_transforms(insp, x_vals, y_vals, lat_1d, lon_1d)
        y_coords, x_coords = lat_1d, lon_1d
        output_crs = "EPSG:4326"
        print(f"  Source grid: {rows}×{cols}  EPSG:3035")
        print(f"  Dest  grid: {TARGET_H}×{TARGET_W}  WGS84")
    else:
        # Path B: IDW — still targets WGS84 grid
        print("  Building WGS84 target grid from bounding box...")
        lat_1d, lon_1d = _load_wgs84_target_grid(x_vals, y_vals)
        idxs, weights = _build_idw_weights(insp, x_vals, y_vals, lat_1d, lon_1d)
        y_coords, x_coords = lat_1d, lon_1d
        output_crs = "EPSG:4326"

    # ── 6. Determine years to process ─────────────────────────────────────
    all_times = pd.to_datetime(ds_raw[time_dim].values).floor("h")
    all_years = sorted(set(all_times.year.tolist()))
    progress  = _load_progress()
    done_set  = set(progress["done"])
    todo      = [y for y in all_years if y not in done_set]

    print(f"\nYears total  : {all_years}")
    print(f"Already done : {sorted(done_set)}")
    print(f"To process   : {todo}")

    if not todo:
        print("\nAll years already processed. Running final merge only.")
    else:
        TMP_DIR.mkdir(parents=True, exist_ok=True)

        for year in todo:
            print(f"\n── Year {year} ──")
            tmp_path = TMP_DIR / f"urbclim_{year}.zarr"

            if method == "latlon_reshape":
                data, times = _process_year_latlon(
                    ds_raw, year, insp, grid_to_zarr
                )
            elif method == "rasterio_warp":
                data, times = _process_year_rasterio(
                    ds_raw, year, insp,
                    zarr_to_grid, rows, cols,
                    src_tr, src_crs, dst_tr, dst_crs,
                    lat_1d, lon_1d,
                )
            else:
                data, times = _process_year_idw(
                    ds_raw, year, insp, idxs, weights, lat_1d, lon_1d
                )

            print(f"  Frames: {len(times):,}  shape: {data.shape}  "
                  f"range: [{np.nanmin(data):.1f}, {np.nanmax(data):.1f}] °C")

            _write_year_zarr(data, times, y_coords, x_coords, output_crs, tmp_path)
            progress["done"].append(year)
            _save_progress(progress)
            print(f"  Saved → {tmp_path}")

    # ── 7. Final merge ─────────────────────────────────────────────────────
    year_paths = [TMP_DIR / f"urbclim_{y}.zarr" for y in all_years]
    missing = [p for p in year_paths if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Missing year zarrs: {missing}\n"
            "Some years were not processed. Re-run to continue."
        )

    _merge_and_finalise(year_paths, y_coords, x_coords, output_crs)

    # ── 8. Verify output ───────────────────────────────────────────────────
    print("\nVerifying output zarr...")
    ds_out = xr.open_zarr(str(OUT_PATH), consolidated=True)
    T, H, W = ds_out.airTemperature.shape
    print(f"  shape  : ({T:,}, {H}, {W})")
    print(f"  crs    : {ds_out.attrs.get('crs', 'unknown')}")
    print(f"  time   : {pd.to_datetime(ds_out.time.values[0])}  →  "
          f"{pd.to_datetime(ds_out.time.values[-1])}")
    print(f"  y      : [{float(ds_out.y[0]):.1f}, {float(ds_out.y[-1]):.1f}]")
    print(f"  x      : [{float(ds_out.x[0]):.1f}, {float(ds_out.x[-1]):.1f}]")
    sample = ds_out.airTemperature.isel(time=slice(0, 50)).values
    nan_pct = np.isnan(sample).mean()
    print(f"  NaN (first 50 frames): {nan_pct:.2%}")
    if nan_pct > 0.01:
        print("  WARNING: high NaN rate — inspect the interpolation result")
    t_out_gb = T * H * W * 4 / 1e9
    print(f"  Size estimate: {t_out_gb:.2f} GB uncompressed")

    # ── 9. Clean up temp zarrs ─────────────────────────────────────────────
    print(f"\nCleaning up temp zarrs in {TMP_DIR}...")
    shutil.rmtree(TMP_DIR)
    PROGRESS_JSON.unlink(missing_ok=True)
    WEIGHTS_NPZ.unlink(missing_ok=True)   # safe to remove; cheap to recompute if needed
    print("Done.\n")
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
