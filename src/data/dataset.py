"""
UrbanDownscalingDataset — PyTorch Dataset for ERA5→UrbClim downscaling.

Each sample:
  era5_seq  : (SEQ_LEN, DYN_CHANNELS, 4, 5)  float32  — ERA5-Land LR sequence
  static    : (STATIC_CHANNELS, 251, 251)     float32  — static features (constant)
  target    : (251, 251)                       float32  — UrbClim T2m at t

ERA5 grid notes:
  The source zarr covers a 25-lat × 32-lon region at 0.1° resolution (ERA5-Land).
  Over Barcelona, we crop to a 4×5 grid (lats 41.2-41.5, lons 1.9-2.3).
  Only 11 of 20 cells have data (land-only zarr); sea cells are filled by
  nearest-neighbor interpolation from land neighbors at load time.

UrbClim grid:
  Stored as (time, y=251, x=251) in EPSG:3035. y is northings descending,
  x is eastings ascending. Loaded directly as 2D slices at sample time.

Static features:
  Stored as flat (weatherStation=63001,) arrays with WGS84 lat_lon station IDs.
  Mapped to the EPSG:3035 (y, x) grid at init time via pyproj projection.

Normalization:
  Stats are pre-computed by scripts/data/compute_normalization_stats.py and
  stored in data/normalization_stats.npz. This file must exist before training.
  Layout: era5_mean (DYN_CHANNELS,), era5_std (DYN_CHANNELS,),
          urbclim_mean scalar, urbclim_std scalar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset
from scipy.spatial import cKDTree

from config.config import (
    PATH_ERA5, PATH_URBCLIM, PATH_STATIC, PATH_STATS,
    ERA5_INPUT_VARS, URBCLIM_TARGET_VAR, STATIC_FEATURES,
    HR_SHAPE, LR_GRID_SHAPE, DYN_CHANNELS, STATIC_CHANNELS,
    TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    TEMPORAL_MONTH_WEIGHTS, SEQ_LEN,
)

# Barcelona crop: 4 lats × 5 lons (matches LR_GRID_SHAPE)
_BCN_LATS = [41.2, 41.3, 41.4, 41.5]   # south→north (row 0 = south)
_BCN_LONS = [1.9, 2.0, 2.1, 2.2, 2.3]  # west→east


class UrbanDownscalingDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        seq_len: int = SEQ_LEN,
        normalize: bool = True,
    ):
        assert split in ("train", "val", "test")
        self.split = split
        self.seq_len = seq_len
        self.normalize = normalize

        # ---- open zarrs lazily (no data loaded yet) ----
        self._ds_era5   = xr.open_zarr(str(PATH_ERA5),    consolidated=False)
        self._ds_urb    = xr.open_zarr(str(PATH_URBCLIM), consolidated=False)
        self._ds_static = xr.open_zarr(str(PATH_STATIC),  consolidated=False)

        # ---- build ERA5 grid crop ----
        self._era5_grid = _build_era5_grid(self._ds_era5)

        # ---- build flat→(row,col) index for static features ----
        # Projects WGS84 station IDs → EPSG:3035, matches to UrbClim processed grid
        self._static_flat_to_2d = _build_static_index(self._ds_static, self._ds_urb)

        # ---- align timestamps ----
        era5_times = pd.to_datetime(self._ds_era5.time.values).floor("h")
        urb_times  = pd.to_datetime(self._ds_urb.time.values).floor("h")
        common = era5_times.intersection(urb_times)

        years = {"train": TRAIN_YEARS, "val": VAL_YEARS, "test": TEST_YEARS}[split]
        mask = common.year.isin(years)
        self._times = common[mask]

        # map back to positional indices in each zarr
        era5_idx = {t: i for i, t in enumerate(era5_times)}
        urb_idx  = {t: i for i, t in enumerate(urb_times)}
        self._era5_pos = np.array([era5_idx[t] for t in self._times], dtype=np.int32)
        self._urb_pos  = np.array([urb_idx[t]  for t in self._times], dtype=np.int32)

        # ---- sample index with summer oversampling ----
        valid = np.arange(seq_len - 1, len(self._times))
        months = self._times[valid].month
        weights = np.array([TEMPORAL_MONTH_WEIGHTS[m] for m in months], dtype=np.float32)
        weights /= weights.sum()
        self._sample_pos = valid
        self._weights = weights

        # ---- normalization stats ----
        self._norm_era5_mean = None
        self._norm_era5_std  = None
        self._norm_urb_mean  = None
        self._norm_urb_std   = None
        if normalize:
            stats = np.load(str(PATH_STATS))
            self._norm_era5_mean = stats["era5_mean"].astype(np.float32)  # (DYN_CHANNELS,)
            self._norm_era5_std  = stats["era5_std"].astype(np.float32)   # (DYN_CHANNELS,)
            self._norm_urb_mean  = float(stats["urbclim_mean"])
            self._norm_urb_std   = float(stats["urbclim_std"])

        # ---- static grid (preloaded into RAM — only ~3 MB) ----
        self._static_grid = _load_static_grid(
            self._ds_static, self._static_flat_to_2d, STATIC_FEATURES
        )  # (STATIC_CHANNELS, 251, 251)

    def __len__(self) -> int:
        return len(self._sample_pos)

    def __getitem__(self, idx: int):
        pos = self._sample_pos[idx]
        era5_seq = _load_era5_seq(
            self._ds_era5,
            self._era5_pos[pos - self.seq_len + 1 : pos + 1],
            self._era5_grid,
            ERA5_INPUT_VARS,
        )  # (seq_len, DYN_CHANNELS, 4, 5)

        target = _load_urbclim_frame(
            self._ds_urb,
            self._urb_pos[pos],
        )  # (251, 251)

        if self.normalize:
            m = self._norm_era5_mean[None, :, None, None]
            s = self._norm_era5_std[None, :, None, None]
            era5_seq = (era5_seq - m) / (s + 1e-6)
            target   = (target - self._norm_urb_mean) / (self._norm_urb_std + 1e-6)

        return (
            torch.from_numpy(era5_seq),          # (T, C, 4, 5)
            torch.from_numpy(self._static_grid),  # (S, 251, 251) — shared view
            torch.from_numpy(target),             # (251, 251)
        )

    @property
    def sample_weights(self) -> np.ndarray:
        """For use with torch.utils.data.WeightedRandomSampler."""
        return self._weights


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

class _Era5Grid:
    """Precomputed lookup for fast ERA5 indexing."""
    def __init__(self, flat_indices, sea_mask, land_nn_for_sea, n_rows, n_cols):
        self.flat_indices = flat_indices       # (4, 5) int32, -1 for sea
        self.sea_mask     = sea_mask           # (4, 5) bool
        self.land_nn      = land_nn_for_sea    # (4, 5) int32: nearest-land flat idx
        self.n_rows = n_rows
        self.n_cols = n_cols


def _build_era5_grid(ds_era5: xr.Dataset) -> _Era5Grid:
    """
    Maps the available Barcelona ERA5 land points onto a full 4×5 grid.
    Sea cells (missing) get the value of their nearest land neighbor.
    """
    ids = ds_era5.weatherStation.values
    id_to_flat = {s: i for i, s in enumerate(ids)}

    n_rows, n_cols = len(_BCN_LATS), len(_BCN_LONS)
    flat_indices = np.full((n_rows, n_cols), -1, dtype=np.int32)
    for r, lat in enumerate(_BCN_LATS):
        for c, lon in enumerate(_BCN_LONS):
            sid = f"{lat}_{lon}"
            if sid in id_to_flat:
                flat_indices[r, c] = id_to_flat[sid]

    sea_mask = flat_indices == -1

    land_nn = flat_indices.copy()
    if sea_mask.any():
        lats_grid = np.array(_BCN_LATS)
        lons_grid = np.array(_BCN_LONS)
        rr, cc = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
        grid_lat = lats_grid[rr]
        grid_lon = lons_grid[cc]

        land_rc = np.argwhere(~sea_mask)
        land_coords = np.column_stack([grid_lat[~sea_mask], grid_lon[~sea_mask]])
        tree = cKDTree(land_coords)

        sea_rc = np.argwhere(sea_mask)
        sea_coords = np.column_stack([grid_lat[sea_mask], grid_lon[sea_mask]])
        _, nn_idx = tree.query(sea_coords)

        for k, (r, c) in enumerate(sea_rc):
            lr, lc = land_rc[nn_idx[k]]
            land_nn[r, c] = flat_indices[lr, lc]

    return _Era5Grid(flat_indices, sea_mask, land_nn, n_rows, n_cols)


def _build_static_index(
    ds_static: xr.Dataset,
    ds_urb: xr.Dataset,
) -> np.ndarray:
    """
    Returns flat_to_2d: (N,2) int32 where flat_to_2d[i] = (row, col).

    Static zarr has flat weatherStation dim with WGS84 "lat_lon" string IDs.
    Processed UrbClim zarr has EPSG:3035 y/x coords (northings/eastings).
    Projects WGS84 → EPSG:3035, then finds nearest pixel in the 251×251 grid.
    """
    from pyproj import Transformer
    tf = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)

    ids = ds_static.weatherStation.values
    lats = np.array([float(s.split("_")[0]) for s in ids])
    lons = np.array([float(s.split("_")[1]) for s in ids])

    # Project WGS84 → EPSG:3035 (always_xy: input is lon, lat)
    xs_3035, ys_3035 = tf.transform(lons, lats)

    # Build KD-tree over the 251×251 EPSG:3035 pixel centres
    y_coords = ds_urb.y.values  # (251,) northings descending
    x_coords = ds_urb.x.values  # (251,) eastings ascending
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
    grid_pts = np.column_stack([yy.ravel(), xx.ravel()])  # (63001, 2)
    tree = cKDTree(grid_pts)

    station_pts = np.column_stack([ys_3035, xs_3035])
    _, pixel_flat = tree.query(station_pts)  # (N,)

    rows = (pixel_flat // len(x_coords)).astype(np.int32)
    cols = (pixel_flat  % len(x_coords)).astype(np.int32)
    return np.column_stack([rows, cols]).astype(np.int32)


# ---------------------------------------------------------------------------
# Per-sample loaders
# ---------------------------------------------------------------------------

def _load_era5_seq(
    ds_era5: xr.Dataset,
    time_positions: np.ndarray,
    era5_grid: _Era5Grid,
    var_names: list[str],
) -> np.ndarray:
    """Load ERA5 sequence → (seq_len, DYN_CHANNELS, 4, 5) float32."""
    T = len(time_positions)
    C = len(var_names)
    H, W = era5_grid.n_rows, era5_grid.n_cols

    needed_flat = np.unique(era5_grid.land_nn)
    flat_to_local = {fi: k for k, fi in enumerate(needed_flat)}
    local_idx = np.vectorize(flat_to_local.get)(era5_grid.land_nn)  # (H, W)

    out = np.empty((T, C, H, W), dtype=np.float32)
    for c, var in enumerate(var_names):
        data = ds_era5[var].isel(
            time=time_positions.tolist(),
            weatherStation=needed_flat.tolist(),
        ).values.astype(np.float32)
        out[:, c, :, :] = data[:, local_idx]

    return out


def _load_urbclim_frame(
    ds_urb: xr.Dataset,
    time_pos: int,
) -> np.ndarray:
    """Load single UrbClim T2m frame → (251, 251) float32."""
    return ds_urb[URBCLIM_TARGET_VAR].isel(time=time_pos).values.astype(np.float32)


def _load_static_grid(
    ds_static: xr.Dataset,
    flat_to_2d: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    """Load static features → (STATIC_CHANNELS, 251, 251) float32."""
    h, w = HR_SHAPE
    n = len(feature_names)
    grid = np.zeros((n, h, w), dtype=np.float32)
    rows = flat_to_2d[:, 0]
    cols = flat_to_2d[:, 1]
    for k, feat in enumerate(feature_names):
        flat = ds_static[feat].values.astype(np.float32)
        grid[k, rows, cols] = flat
    return grid
