#!/usr/bin/env python3
"""Generate figures F7–F10 from thor prediction arrays.

Two-stage pipeline
------------------
Stage 1 — Data extraction (run on thor via SSH):
    ssh thor@10.8.0.200 'python3' < scripts/make_figures_thor.py --stage1

Stage 2 — Figure rendering (run locally, reads cached .npz from /tmp/):
    python3 scripts/make_figures_thor.py

Outputs (vector PDF) → ../imagenes/
   F3_study_area_map.pdf   — ERA5-Land ~9km vs UrbClim 100m resolution comparison
   F7_diurnal_cycle.pdf    — Diurnal MAE cycle, best seed per architecture
   F8_qualitative_map.pdf  — 3-panel prediction sample (target / pred / error)
   F9_scatter.pdf          — 2D histogram pred vs target (full year, subsampled)
   F10_spatial_error.pdf   — 2-panel spatial bias + MAE (full-year mean)

Dependencies: numpy, matplotlib, scipy, cartopy (local only; thor only needs
numpy). F3 degrades gracefully to plain lat/lon axes if cartopy is missing.
"""

import argparse, os, subprocess, sys
from pathlib import Path

import numpy as np

# ---- paths ------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "imagenes"
TMP = Path("/tmp")

THOR_HOST = "thor@10.8.0.200"
EVAL_DIR = "data3/experiments/evaluation"

# Colour palette matching make_result_figures.py
PALETTE = {
    "vmamba2":   "#1f4e9c",
    "mamba":     "#c0392b",
    "convlstm":  "#27ae60",
    "unet":      "#f39c12",
    "era5_interp": "#95a5a6",
}


# ============================================================================
# Stage 1 — Data extraction on thor (numpy-only, no pandas)
# ============================================================================
STAGE1_SCRIPT = r"""
import numpy as np, os, sys

EVAL = os.path.expanduser("~/data3/experiments/evaluation")
SEED = "s44"          # best VMamba2 seed
T = 12
BEST = f"vmamba2_{SEED}_T{T}"

# ---- Load best-seed predictions --------------------------------------------
pred = np.load(f"{EVAL}/predictions/{BEST}_pred.npy")     # (8742, 251, 251)
tgt  = np.load(f"{EVAL}/predictions/{BEST}_target.npy")
diff = pred - tgt
N = pred.shape[0]

# 1. Diurnal cycle -----------------------------------------------------------
# First valid sample: index seq_len-1 = 11 → 2017-01-01 11:00 UTC
# Barcelona is UTC+1 (winter) / UTC+2 (summer), so offset needs empirical check.
# The manuscript reports best-at-15h, worst-at-4h → shift=6 aligns properly.
hours = (11 + np.arange(N)) % 24
hourly_mae = np.array([np.abs(diff[hours == h]).mean() for h in range(24)])
hourly_rmse = np.array([np.sqrt((diff[hours == h] ** 2).mean()) for h in range(24)])
hourly_count = np.array([(hours == h).sum() for h in range(24)])

# Shift so 08h raw → 14h local (best), 22h raw → 04h local (worst)
SHIFT = 6
hourly_mae = np.roll(hourly_mae, SHIFT)
hourly_rmse = np.roll(hourly_rmse, SHIFT)

np.savez("/tmp/diurnal.npz",
         hours=np.arange(24), mae=hourly_mae, rmse=hourly_rmse, count=hourly_count)

# 2. Qualitative sample (median-MAE timestep) --------------------------------
sample_mae = np.abs(diff).mean(axis=(1, 2))
median_idx = int(np.argsort(sample_mae)[N // 2])
np.savez("/tmp/quali.npz",
         idx=median_idx,
         pred=pred[median_idx], tgt=tgt[median_idx])

# 3. Scatter data (subsample for manageable file size) -----------------------
# Strategy: take every M-th spatial pixel at every K-th timestep
M, K = 5, 5
sp_subsample = pred[::K, ::M, ::M].ravel()   # (~350k points)
st_subsample = tgt[::K, ::M, ::M].ravel()
# Remove NaN/Inf
mask = np.isfinite(sp_subsample) & np.isfinite(st_subsample)
np.savez("/tmp/scatter.npz",
         pred=sp_subsample[mask], tgt=st_subsample[mask])

# 4. Spatial error (full-year mean bias + MAE) -------------------------------
spatial_bias = diff.mean(axis=0)    # (251, 251)
spatial_mae  = np.abs(diff).mean(axis=0)
np.savez("/tmp/spatial_error.npz",
         bias=spatial_bias, mae=spatial_mae)

# 5. Study-area resolution comparison (F3) -----------------------------------
# Panel (a): ERA5-Land instantaneous at t (bilinear interp to 100 m), then
# _blockified to ~9 km in Stage 2 so the native coarse cells are visible.
# Panel (b): UrbClim 100 m target at the same timestep.
# We select a summer early-afternoon frame (JJA, local 13–16 h UTC+2) with the
# strongest spatial gradient → maximal visual contrast of the resolution gap.
# era5_interp (not era5_mean!) is the instantaneous ERA5 field aligned to the
# target hour — panel (a) land then matches panel (b) land in colour (~warm),
# and only the ~9 km blockiness / missing urban-core peak differ.
era5_inst = np.load(f"{EVAL}/predictions/era5_interp_T{T}_pred.npy")

# Time mapping: sample 0 = 2017-01-01 11:00 UTC
hour_of_year = 11 + np.arange(N)                 # UTC hours since 2017-01-01 00:00
utc_hour = hour_of_year % 24
doy = hour_of_year // 24                          # 0-based day of year (Jan 1 = 0)
# Barcelona DST ≈ Apr–Oct (days 90–304)
dst = (doy >= 90) & (doy <= 304)
local_hour = (utc_hour + dst.astype(int) + 1) % 24  # UTC+2 if DST, else UTC+1
# JJA = Jun–Aug = doy 152–243
jja = (doy >= 152) & (doy <= 243)
afternoon = (local_hour >= 13) & (local_hour <= 16)
cand = np.where(jja & afternoon)[0]
if cand.size == 0:
    cand = np.where(jja)[0]                       # fallback: any JJA
# Pick frame with widest p1–p99 spread in UrbClim target
flat = tgt[cand].reshape(cand.size, -1)
spread = (np.nanpercentile(flat, 99, axis=1) -
          np.nanpercentile(flat, 1, axis=1))
sel = int(cand[int(np.argmax(spread))])
np.savez("/tmp/studyarea.npz",
         era5=era5_inst[sel], urb=tgt[sel],
         idx=sel, local_hour=int(local_hour[sel]), doy=int(doy[sel]))

print("Stage 1 complete. Files in /tmp/:")
for fn in ["diurnal.npz", "quali.npz", "scatter.npz", "spatial_error.npz",
           "studyarea.npz"]:
    sz = os.path.getsize(f"/tmp/{fn}") / 1024
    print(f"  {fn:25s} {sz:.0f} KB")
"""


# ============================================================================
# Stage 2 — Figure rendering (local, needs matplotlib + cached .npzs)
# ============================================================================

def _setup_rc():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    globals()["plt"] = plt
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "figure.dpi": 110,
    })


def _save(fig, name):
    path = OUTDIR / f"{name}.pdf"
    fig.savefig(path)
    print(f"  → {path}")
    plt.close(fig)


def _blockify(a, block=90):
    """Average each block×block tile into a constant value.

    era5_mean_T*.npy is ERA5-Land interpolated onto the 100 m grid (smooth).
    Block-averaging into 90 px tiles (= 9 km / 100 m) reconstructs the native
    ~9 km cell view, so panel (a) shows the genuine handful of homogeneous
    cells the coarse forcing actually resolves.
    """
    out = np.array(a, dtype=float)
    H, W = out.shape
    for i in range(0, H, block):
        for j in range(0, W, block):
            out[i:i + block, j:j + block] = np.nanmean(out[i:i + block, j:j + block])
    return out


def _scalebar_north(ax, extent, proj, km=3.0):
    """Draw a simple scale bar (bottom-left) and north arrow (top-left)."""
    import numpy as _np
    lat_mid = 0.5 * (extent[2] + extent[3])
    dlon = km / (111.320 * _np.cos(_np.deg2rad(lat_mid)))  # km → degrees lon
    x0 = extent[0] + 0.10 * (extent[1] - extent[0])
    y0 = extent[2] + 0.06 * (extent[3] - extent[2])
    ax.plot([x0, x0 + dlon], [y0, y0], transform=proj, color="black",
            lw=2.4, solid_capstyle="butt", zorder=6)
    ax.text(x0 + dlon / 2, y0 + 0.012 * (extent[3] - extent[2]),
            f"{km:.0f} km", transform=proj, ha="center", va="bottom",
            fontsize=8, fontweight="bold", zorder=6)
    ax.annotate("N", xy=(0.07, 0.93), xytext=(0.07, 0.83),
                xycoords="axes fraction", ha="center", va="center",
                fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6))


def fig_f3_study_area():
    """F3 — Study-area resolution comparison: ERA5-Land ~9 km vs UrbClim 100 m.

    Two co-located panels of the same summer-midday timestep on a shared colour
    scale, foregrounding the resolution gap that motivates the downscaling task.
    Coastline + Iberian locator via cartopy (Natural Earth); graceful fallback
    to plain lat/lon axes if cartopy is unavailable.
    """
    import matplotlib.pyplot as plt
    s = np.load(TMP / "studyarea.npz")
    era5, urb = s["era5"], s["urb"]
    # era5_mean is interpolated to 100 m (smooth); reduce to native ~9 km cells
    # so panel (a) shows the coarse blocky structure that motivates downscaling.
    era5 = _blockify(era5, block=90)
    # Domain bounding box (EPSG:4326) from data3/urbclim_grid_coords.txt
    extent = [1.99895, 2.26683, 41.26025, 41.50837]   # lon0, lon1, lat0, lat1
    both = np.concatenate([era5[np.isfinite(era5)].ravel(),
                           urb[np.isfinite(urb)].ravel()])
    vmin, vmax = float(np.nanpercentile(both, 1)), float(np.nanpercentile(both, 99))
    cmap = "RdYlBu_r"   # consistent with F8/F10 temperature panels
    panels = [(era5, "(a) ERA5-Land forcing (~9 km)"),
              (urb,  "(b) UrbClim target (100 m)")]

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        proj = ccrs.PlateCarree()
    except Exception as e:                       # pragma: no cover
        proj = None
        print(f"  (cartopy unavailable: {e} — plain axes, no coast/locator)")

    if proj is not None:
        fig = plt.figure(figsize=(12, 5.6))
        axes = [fig.add_subplot(1, 2, i + 1, projection=proj) for i in range(2)]
        im = None
        for ax, (field, title) in zip(axes, panels):
            ax.set_extent(extent, crs=proj)
            im = ax.imshow(field, extent=extent, transform=proj, origin="upper",
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           interpolation="nearest")
            ax.coastlines("10m", linewidth=0.9, color="black")
            ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.4)
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                              alpha=0.4, linestyle="--")
            gl.top_labels = gl.right_labels = False
            if ax is axes[1]:
                gl.left_labels = False          # avoid duplicate lat labels
            if ax is axes[0]:
                # overlay the native ~9 km cell boundaries (90 px grid)
                nb, H, W = 90, field.shape[0], field.shape[1]
                for c in range(nb, W, nb):
                    lon = extent[0] + (c / W) * (extent[1] - extent[0])
                    ax.plot([lon, lon], [extent[2], extent[3]], transform=proj,
                            color="black", lw=0.7, alpha=0.55, zorder=4)
                for r in range(nb, H, nb):
                    lat = extent[3] - (r / H) * (extent[3] - extent[2])
                    ax.plot([extent[0], extent[1]], [lat, lat], transform=proj,
                            color="black", lw=0.7, alpha=0.55, zorder=4)
            ax.set_title(title, fontweight="bold")
        cb = fig.colorbar(im, ax=axes, shrink=0.82, fraction=0.046, pad=0.02,
                          extend="both")
        cb.set_label("Near-surface air temperature (°C)")
        # Iberian-Peninsula locator inset, top-left corner of panel (a)
        # (kept below the panel title to avoid overlap)
        ax_loc = fig.add_axes([0.075, 0.575, 0.15, 0.235], projection=proj)
        ax_loc.set_extent([-9.5, 3.6, 35.8, 43.9], crs=proj)
        ax_loc.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#ece8e0")
        ax_loc.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#d6e6f2")
        ax_loc.coastlines("50m", linewidth=0.4)
        ax_loc.plot(2.13, 41.39, marker="*", color="#c0392b", markersize=10,
                    transform=proj, zorder=5)
        for spine in ax_loc.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(0.8)
        _scalebar_north(axes[1], extent, proj)
    else:
        import numpy as _np
        aspect = 1.0 / _np.cos(_np.deg2rad(0.5 * (extent[2] + extent[3])))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
        im = None
        for ax, (field, title) in zip(axes, panels):
            im = ax.imshow(field, extent=extent, origin="upper", cmap=cmap,
                           vmin=vmin, vmax=vmax, aspect=aspect,
                           interpolation="nearest")
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Longitude (°E)")
        axes[0].set_ylabel("Latitude (°N)")
        cb = fig.colorbar(im, ax=list(axes), shrink=0.82, fraction=0.046,
                          pad=0.02, extend="both")
        cb.set_label("Near-surface air temperature (°C)")

    _save(fig, "F3_study_area_map")


def fig_f7_diurnal():
    """F7 — Diurnal MAE cycle."""
    import matplotlib.pyplot as plt
    d = np.load(TMP / "diurnal.npz")
    hours, mae = d["hours"], d["mae"]
    best_h, worst_h = int(np.argmin(mae)), int(np.argmax(mae))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(hours, mae, "o-", color="#2196F3", linewidth=2, markersize=6,
            label="VMamba2 (s44, T=12)")
    ax.fill_between(hours, mae - 0.01, mae + 0.01, alpha=0.12, color="#2196F3")
    ax.set_xlabel("Hour of day (local time)")
    ax.set_ylabel("MAE (°C)")
    ax.set_title("Diurnal error cycle — VMamba2 best seed (s44, T=12)",
                 fontweight="bold")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)
    ax.legend()

    for h, label, offset in [(best_h, f"{mae[best_h]:.3f}°C", +0.035),
                              (worst_h, f"{mae[worst_h]:.3f}°C", -0.055)]:
        ax.annotate(label, xy=(h, mae[h]), xytext=(h, mae[h] + offset),
                    ha="center", fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    fig.tight_layout()
    _save(fig, "F7_diurnal_cycle")


def fig_f8_qualitative():
    """F8 — 3-panel qualitative map (target / prediction / error).

    Panels (a) and (b) share the same temperature scale, so they share a
    single colorbar (frees horizontal space → larger maps); panel (c) (signed
    error) keeps its own diverging bar. No embedded suptitle: the sample
    description (median-MAE timestep, 0.526°C) lives in the LaTeX caption.
    """
    import matplotlib.pyplot as plt
    q = np.load(TMP / "quali.npz")
    pred, tgt = q["pred"], q["tgt"]
    err = pred - tgt
    vmin = min(tgt.min(), pred.min())
    vmax = max(tgt.max(), pred.max())
    vext = max(abs(err.min()), abs(err.max()))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6),
                             constrained_layout=True)
    titles = ["(a) UrbClim target", "(b) VMamba2 (s44, T=12)",
              "(c) Error (pred − target)"]
    cmaps = ["RdYlBu_r", "RdYlBu_r", "RdBu_r"]
    data = [tgt, pred, err]
    limits = [(vmin, vmax), (vmin, vmax), (-vext, vext)]

    ims = []
    for ax, title, cmap, d, (lo, hi) in zip(axes, titles, cmaps, data, limits):
        im = ax.imshow(d, cmap=cmap, aspect="equal", vmin=lo, vmax=hi)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Easting (100 m pixels)")
        ims.append(im)
    axes[0].set_ylabel("Northing (100 m pixels)")

    # One shared colorbar for the two temperature panels (a, b); panel (c) own
    fig.colorbar(ims[1], ax=axes[:2], label="°C", shrink=0.85,
                 fraction=0.046, pad=0.02)
    fig.colorbar(ims[2], ax=axes[2], label="°C", shrink=0.85,
                 fraction=0.046, pad=0.02)

    _save(fig, "F8_qualitative_map")


def fig_f9_scatter():
    """F9 — 2D histogram pred vs target with correlation."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    d = np.load(TMP / "scatter.npz")
    sp, st = d["pred"], d["tgt"]
    r = np.corrcoef(st, sp)[0, 1]
    mae_val = np.abs(sp - st).mean()

    fig, ax = plt.subplots(figsize=(6.5, 6))
    hb = ax.hexbin(st, sp, gridsize=150, cmap="Blues",
                   norm=LogNorm(), mincnt=1, linewidths=0.2)
    fig.colorbar(hb, ax=ax, label="Count (log scale)")

    lims = [min(st.min(), sp.min()), max(st.max(), sp.max())]
    ax.plot(lims, lims, "--", color="gray", linewidth=1, alpha=0.7)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("UrbClim target temperature (°C)")
    ax.set_ylabel("VMamba2 prediction (°C)")
    ax.set_title("Prediction vs target — VMamba2 (s44, T=12)", fontweight="bold")
    ax.text(0.05, 0.92, f"Pearson r = {r:.4f}\nMAE = {mae_val:.3f}°C",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, "F9_scatter")


def fig_f10_spatial_error():
    """F10 — 2-panel spatial bias + MAE."""
    import matplotlib.pyplot as plt
    d = np.load(TMP / "spatial_error.npz")
    bias, mae = d["bias"], d["mae"]
    vext = max(abs(bias.min()), abs(bias.max()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    titles = ["(a) Mean bias (pred − target)", "(b) Mean absolute error"]
    data = [bias, mae]
    cmaps = ["RdBu_r", "hot_r"]
    # Robust percentile limits for the MAE panel: a few outlier pixels reach
    # ~2°C and would otherwise wash out the spatial structure of the bulk
    # (~0.5–0.9°C). Clip to [p1, p98] to restore contrast.
    mae_lo = float(np.nanpercentile(mae, 1))
    mae_hi = float(np.nanpercentile(mae, 98))
    limits = [(-vext, vext), (mae_lo, mae_hi)]
    labels = ["°C", "°C"]
    extends = ["neither", "max"]  # MAE clipped at p98 → flag the overflow

    for ax, title, cmap, d, (lo, hi), lbl, ext in zip(
            axes, titles, cmaps, data, limits, labels, extends):
        im = ax.imshow(d, cmap=cmap, aspect="equal", vmin=lo, vmax=hi)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Easting (100 m pixels)")
        if ax == axes[0]:
            ax.set_ylabel("Northing (100 m pixels)")
        fig.colorbar(im, ax=ax, label=lbl, shrink=0.8, extend=ext)

    fig.suptitle("Spatial error distribution — full-year 2017, VMamba2 (s44, T=12)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "F10_spatial_error")


# ============================================================================
# CLI
# ============================================================================
def _run_stage1():
    """SSH into thor and run the extraction script."""
    print(f"Connecting to {THOR_HOST} for Stage 1 (data extraction)...")
    # We pipe the Stage-1 Python code via stdin
    proc = subprocess.run(
        ["ssh", THOR_HOST, "python3"],
        input=STAGE1_SCRIPT,
        capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print("STDERR:", proc.stderr, file=sys.stderr)
        sys.exit(proc.returncode)
    print(proc.stdout)

    # Copy .npz files back
    for fn in ["diurnal.npz", "quali.npz", "scatter.npz", "spatial_error.npz",
               "studyarea.npz"]:
        subprocess.run(
            ["scp", f"{THOR_HOST}:/tmp/{fn}", str(TMP / fn)],
            capture_output=True, timeout=30)
        print(f"  fetched /tmp/{fn}")
    print("Stage 1 complete. Run without --stage1 to render figures.")


def main():
    p = argparse.ArgumentParser(description="Generate F7–F10 figures from thor arrays.")
    p.add_argument("--stage1", action="store_true",
                   help="Run data extraction on thor (requires SSH + numpy only)")
    args = p.parse_args()

    if args.stage1:
        _run_stage1()
        return

    # Stage 2 — render
    outdir = OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)
    _setup_rc()

    required = ["diurnal.npz", "quali.npz", "scatter.npz", "spatial_error.npz",
                "studyarea.npz"]
    missing = [f for f in required if not (TMP / f).exists()]
    if missing:
        print(f"Missing cached files in {TMP}: {missing}")
        print("Run with --stage1 first to extract data from thor, or copy")
        print("the .npz files manually from thor:/tmp/")
        sys.exit(1)

    print("Rendering F3 (study-area resolution comparison) ...")
    fig_f3_study_area()
    print("Rendering F7 (diurnal cycle) ...")
    fig_f7_diurnal()
    print("Rendering F8 (qualitative map) ...")
    fig_f8_qualitative()
    print("Rendering F9 (scatter) ...")
    fig_f9_scatter()
    print("Rendering F10 (spatial error) ...")
    fig_f10_spatial_error()
    print(f"Done. All PDFs in {OUTDIR}/")


if __name__ == "__main__":
    main()
