"""
Central configuration for weather_urban_downscaling_v2.

Design principles:
  - Single source of truth for all paths and hyperparameters.
  - All paths are relative to PROJECT_ROOT so the repo is portable.
  - No framework detection or fallback logic — PyTorch is the only backend.
  - Data split is temporal and strictly non-overlapping.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data paths  (source zarrs live outside the repo — set via env or override)
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"

# Inputs — point to extracted_zarr_data or copies therein
PATH_ERA5   = DATA_DIR / "era5land_2008-2017.zarr"
PATH_URBCLIM = DATA_DIR / "urbclim_2008-2017.zarr"
PATH_STATIC  = DATA_DIR / "static_features.zarr"   # rebuilt by build_static_features.py
PATH_BUILDINGS = DATA_DIR / "barcelona_buildings.geojson"

# Derived / cached
PATH_STATIC_GRID = DATA_DIR / "static_grid_251x251.npy"   # (251,251,N_STATIC), float32
PATH_STATS       = DATA_DIR / "normalization_stats.npz"

# Outputs
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# ---------------------------------------------------------------------------
# Spatial dimensions
# ---------------------------------------------------------------------------
HR_SHAPE      = (251, 251)    # UrbClim output grid
LR_GRID_SHAPE = (4, 5)        # ERA5-Land crop over Barcelona bbox
DYN_CHANNELS  = 9             # ERA5 input variables (remapped from source)
STATIC_CHANNELS = 11          # after dropping ndvi_min; see build_static_features.py

# ERA5 variable mapping from extracted zarr → model input channels (ordered)
# Source zarr has 18 vars; we select 9 physically meaningful for downscaling.
ERA5_INPUT_VARS = [
    "airTemperature",       # t2m coarse — primary predictor
    "dewAirTemperature",    # humidity
    "windSpeedEast",        # u10
    "windSpeedNorth",       # v10
    "windSpeed",            # magnitude
    "GHI",                  # global horizontal irradiance (replaces ssrd)
    "totalPrecipitation",
    "highVegetationRatio",  # lai_hv proxy
    "lowVegetationRatio",   # lai_lv proxy
]

# UrbClim target variable
URBCLIM_TARGET_VAR = "airTemperature"

# Static features kept (ndvi_min excluded — consistency bug; SVF recomputed)
STATIC_FEATURES = [
    "avg_height",
    "building_density",
    "elevation",
    "height_index",
    "industrial_index",
    "leisure_index",
    "ndvi_mean",
    "residential_index",
    "services_index",
    "svf",          # recomputed with corrected canyon geometry
    "h_over_w",     # recomputed with corrected canyon geometry
]

# ---------------------------------------------------------------------------
# Temporal split  (strictly non-overlapping; 2017 is the held-out test year)
# ---------------------------------------------------------------------------
TRAIN_YEARS = list(range(2008, 2016))   # 2008–2015
VAL_YEARS   = [2016]
TEST_YEARS  = [2017]

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
SEED          = 42
BATCH_SIZE    = 8
EPOCHS        = 100
LR            = 1e-4
SEQ_LEN       = 6            # default; overridden per run for Mamba ablation
MAX_STEPS_PER_EPOCH = 1000   # caps epoch length; full dataset sampled via shuffle
EARLY_STOPPING_PATIENCE = 15

# Temporal sampler: oversample summer to bias toward heatwave conditions
# Weight per month (1=uniform); Jun–Sep get 3×
TEMPORAL_MONTH_WEIGHTS = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1,
    6: 3, 7: 3, 8: 3, 9: 3,
    10: 1, 11: 1, 12: 1,
}

# Loss: hybrid MSE + SSIM
LOSS_ALPHA = 0.8   # weight on SSIM term

# ---------------------------------------------------------------------------
# Model dimensions
# ---------------------------------------------------------------------------
UNET_BASE_FILTERS    = 64
CONVLSTM_HIDDEN_DIM  = 64
MAMBA_D_MODEL        = 128
MAMBA_D_STATE        = 16
MAMBA_D_CONV         = 4
MAMBA_EXPAND         = 2
