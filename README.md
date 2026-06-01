# VMamba2 — 2D Selective State-Space Scanning for Urban Temperature Downscaling

[![DOI](https://img.shields.io/badge/DOI-pending-blue)](https://doi.org/)
[![arXiv](https://img.shields.io/badge/arXiv-pending-b31b1b)](https://arxiv.org/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: EUPL v1.2](https://img.shields.io/badge/License-EUPL%20v1.2-blue.svg)](LICENSE)

This repository contains the official PyTorch implementation and evaluation code for the manuscript *"2D Selective State-Space Scanning for Parameter-Efficient and Reproducible Urban Temperature Downscaling"* submitted to the **Journal of Computational Science** (Elsevier).

## Overview

VMamba2 replaces the 1D Mamba bottleneck in a U-Net architecture with a 2D cross-directional selective scan that preserves spatial topology. With **3.4× fewer parameters** than ConvLSTM (1.35M vs 4.61M), VMamba2 matches ConvLSTM structural fidelity (SSIM 0.825 vs 0.822; neither difference significant at n=5) while remaining competitive in mean absolute error. The 2D scan substantially outperforms its 1D counterpart (+0.023 SSIM, d=1.76, p=0.02), while the 1D Mamba baseline is near-deterministic (MAE CV 0.6% vs VMamba2 5.6%).

- **Target:** downscaling ERA5-Land (~9 km) → UrbClim (100 m) hourly air temperature over Barcelona
- **Training:** 2008–2015, validation 2016, **test full-year 2017** (8742 hourly samples)
- **Protocol:** 5 seeds × 100 epochs × 2 temporal windows (T=6, T=12)
- **Baselines:** ConvLSTM, non-recurrent U-Net, 1D Mamba

## Repository structure

```
├── config/                    # Experiment configuration
│   └── config.py              # Hyperparameters, paths, loss settings
├── src/
│   ├── data/
│   │   └── dataset.py         # PyTorch Dataset (ERA5-Land + static features)
│   ├── models/
│   │   └── downsr_unet.py     # U-Net + VMamba2 / 1D Mamba / ConvLSTM bottlenecks
│   └── losses.py              # Hybrid MSE+SSIM loss, SSIM implementation
├── scripts/
│   ├── data/                  # Data preprocessing (ERA5, UrbClim, normalization)
│   ├── training/
│   │   └── train.py           # Training loop with optional WandB logging
│   ├── evaluation/
│   │   ├── evaluate.py        # Test-set evaluation (MAE / RMSE / SSIM / bias)
│   │   └── evaluate_rollout.py # Multi-horizon + temporal consistency metrics
│   └── figures/
│       ├── make_result_figures.py  # Generate F4–F6, S1–S2 (table-derived, no data)
│       └── make_figures_thor.py    # Generate F3, F7–F10 (from thor prediction arrays)
├── experiments/
│   └── evaluation/            # Result CSVs from the manuscript test set
│       ├── test_results.csv   # Per-run metrics (22 configurations)
│       ├── test_summary.csv   # Mean ± std per architecture × seq_len
│       └── baseline_results.csv # ERA5-Land baselines
├── requirements.txt           # Python dependencies
├── docker/
│   ├── Dockerfile             # PyTorch 2.2.0 + CUDA 12.1 container
│   └── compose.yml            # Docker Compose for training + evaluation
├── data/                      # Placeholder for datasets (see below)
│   └── urbclim_grid_coords.txt
└── README.md
```

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Data

The manuscript uses two external datasets:

| Dataset | Source | Access |
|---|---|---|
| ERA5-Land reanalysis | Copernicus CDS | [Public](https://cds.climate.copernicus.eu/) |
| UrbClim urban climate | VITO | On request to corresponding author |

Place preprocessed `.zarr` files in `data/` following the expected layout:

```
data/
├── era5land_2008-2017.zarr     # 9 ERA5-Land variables, Barcelona crop
├── urbclim_2008-2017.zarr      # 251×251 grid (63001 pixels)
├── static_features.zarr        # 11 GIS-derived morphology channels
├── normalization_stats.npz     # Precomputed mean/std for normalization
└── urbclim_grid_coords.txt     # Lat/lon for each of the 63001 grid pixels
```

### 3. Training

```bash
python scripts/training/train.py
```

Configuration via `config/config.py`:
- `ARCH`: `"vmamba2"` (proposed), `"mamba"` (1D Mamba ablation), `"convlstm"`, `"unet"`
- `SEQ_LEN`: 6 or 12
- `SEED`: 42–46 (manuscript uses n=5 seeds)
- `BATCH_SIZE`, `EPOCHS`, `LR`, `LOSS_TYPE`, `LOSS_ALPHA`

### 4. Evaluation

```bash
# Full test-set evaluation (generates predictions/ + CSV)
python scripts/evaluation/evaluate.py

# Multi-horizon rollout + temporal consistency
python scripts/evaluation/evaluate_rollout.py
```

Outputs go to `experiments/evaluation/`:
- `test_results.csv` — per-run MAE / RMSE / SSIM / bias
- `test_summary.csv` — mean ± std per (arch, seq_len) group
- `predictions/` — per-sample prediction and target arrays (.npy)

### 5. Figures

```bash
# F4–F6, S1–S2: derived from table values, no external data needed
python scripts/figures/make_result_figures.py

# F3, F7–F10: requires thor prediction arrays (two-stage pipeline)
python scripts/figures/make_figures_thor.py --stage1   # extract from thor
python scripts/figures/make_figures_thor.py              # render PDFs (needs cartopy)
```

## Results

Key results from the full-year 2017 test set (8742 hourly samples, best seed per architecture):

| Model | Params | MAE (°C) | SSIM |
|---|---|---|---|
| VMamba2 (T=12, s44) | 1.35M | 0.583 | 0.843 |
| ConvLSTM (T=6, s42) | 4.61M | 0.591 | 0.839 |
| 1D Mamba (T=6, s43) | 1.20M | 0.757 | 0.818 |
| U-Net (T=6) | 1.95M | 0.777 | 0.805 |

## Citation

```bibtex
@article{cardona2026vmamba2,
  title   = {2D Selective State-Space Scanning for Parameter-Efficient
             and Reproducible Urban Temperature Downscaling},
  author  = {Cardona, Kerin and Mor, Gerard and Cipriano, Jordi
             and Solsona, Francesc},
  journal = {Journal of Computational Science},
  year    = {2026},
  note    = {Under review},
}
```

## License

EUPL v1.2. See [LICENSE](LICENSE).
