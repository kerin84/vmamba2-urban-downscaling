# VMamba2 — Revisiting Spatiotemporal Downscaling with 2D State-Space Models

[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.XXXXXXX-blue)](https://doi.org/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: EUPL v1.2](https://img.shields.io/badge/License-EUPL%20v1.2-blue.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-compose-blue.svg)](docker/)

This repository contains the official PyTorch implementation and evaluation code for the manuscript:

> **Revisiting Spatiotemporal Downscaling with 2D State-Space Models: Preserving Spatial Topology in Urban Temperature Fields**

Submitted to **Computational Urban Science** (Springer Nature), Special Collection: *Revisiting Spatiotemporal Modeling in the Era of GeoAI*.

## Overview

Spatiotemporal GeoAI faces a fundamental tension: recurrent architectures (ConvLSTM) preserve spatial topology but impose sequential, non-parallelisable computation, while attention scales quadratically with sequence length. State-space models (SSMs) offer linear complexity, but standard 1D token flattening destroys the 2D spatial organization of geographic fields.

**VMamba2** is a 2D selective state-space bottleneck for U-Net architectures that reconciles this tension via cross-directional scanning along four cardinal paths. Key findings:

| Metric | VMamba2 (T=12) | ConvLSTM | 1D Mamba | U-Net |
|--------|:---:|:---:|:---:|:---:|
| **SSIM** (mean ± std, n=5) | 0.825 ± 0.013 | 0.822 ± 0.014 | 0.802 ± 0.014 | 0.805 ± 0.004 |
| **MAE** (°C, mean ± std, n=5) | 0.640 ± 0.036 | 0.621 ± 0.033 | 0.765 ± 0.005 | 0.777 ± 0.031 |
| **Parameters** | 1.35M | 4.61M | 1.20M | 1.95M |

- VMamba2 matches ConvLSTM accuracy with **3.4× fewer parameters** (neither difference significant at n=5, Welch's t-test)
- 2D scanning improves SSIM by **+0.023** over 1D Mamba (p=0.02, d=1.76)
- 1D Mamba is near-deterministic (MAE CV 0.6%); VMamba2 CV 5.6% remains within operational bounds
- **Best single seed** (s44, T=12): MAE 0.583 °C, SSIM 0.843

**Experimental setup:**
- **Target:** downscaling ERA5-Land (~9 km) → UrbClim (100 m) hourly air temperature over Barcelona
- **Training:** 2008–2015, validation 2016, test full-year 2017 (8742 hourly samples)
- **Protocol:** 5 independent seeds (42–46) × 2 temporal windows (T=6, T=12)
- **Hardware:** NVIDIA RTX PRO 6000 Blackwell (48 GB), PyTorch 2.7 + CUDA 12.8

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
│   ├── Dockerfile             # PyTorch 2.7 + CUDA 12.8 container
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

### 6. Docker

```bash
docker compose -f docker/compose.yml run --rm train
docker compose -f docker/compose.yml run --rm evaluate
```

The Dockerfile uses `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel`. Volume-mount your data directory when running.

## Reproducibility

All results in the manuscript are derived from 5 independent seeds (42–46) per architecture × sequence length combination. The evaluation scripts in `experiments/evaluation/` contain the raw per-run metrics.

To reproduce the exact figures and tables from the paper:
1. Train all configurations (5 seeds × 4 architectures × 2 sequence lengths = 22 configurations for learned models)
2. Run `evaluate.py` on the 2017 test set
3. Run `make_result_figures.py` to generate F4–F6 and S1–S2
4. Run `make_figures_thor.py` on the prediction arrays to generate F3 and F7–F10

## Citation

```bibtex
@article{cardona2026vmamba2,
  title   = {Revisiting Spatiotemporal Downscaling with 2D State-Space
             Models: Preserving Spatial Topology in Urban Temperature Fields},
  author  = {Cardona, Kerin and Mor, Gerard and Cipriano, Jordi
             and Solsona, Francesc},
  journal = {Computational Urban Science},
  year    = {2026},
  note    = {Under review. Special Collection: Revisiting Spatiotemporal
             Modeling in the Era of GeoAI},
}
```

## License

EUPL v1.2. See [LICENSE](LICENSE).

© 2026 Kerin Cardona and Gerard Mor
