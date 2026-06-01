#!/usr/bin/env python3
"""Generate result figures for the VMamba2 manuscript.

All numbers are transcribed directly from the result tables in main.tex
(full-year 2017 test set, n=5 seeds). No external data is required, so the
figures are fully reproducible from this script alone.

Outputs (vector PDF) go to ../imagenes/ ; PNG previews go to /tmp/figprev/ .
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
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

OUT_PDF = Path(__file__).resolve().parent.parent / "imagenes"
OUT_PNG = Path("/tmp/figprev")
OUT_PDF.mkdir(exist_ok=True)
OUT_PNG.mkdir(exist_ok=True)

# Consistent colour / marker per model across all figures
STYLE = {
    "VMamba2 (T=12)": dict(color="#1f4e9c", marker="o"),
    "VMamba2 (T=6)":  dict(color="#5b9bd5", marker="s"),
    "ConvLSTM (T=6)": dict(color="#e07b39", marker="^"),
    "U-Net (T=6)":    dict(color="#2ca02c", marker="D"),
    "1D Mamba (T=6)": dict(color="#c0392b", marker="v"),
    "ERA5-Land":      dict(color="#7f7f7f", marker="P"),
}


def save(fig, name):
    fig.savefig(OUT_PDF / f"{name}.pdf")
    fig.savefig(OUT_PNG / f"{name}.png")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# --------------------------------------------------------------------------
# Fig. F4 — accuracy vs parameter efficiency (Table: full-frame + efficiency)
# --------------------------------------------------------------------------
def fig_efficiency():
    # model: (params M, SSIM, SSIM_std, MAE, MAE_std)
    data = {
        "VMamba2 (T=12)": (1.35, 0.825, 0.013, 0.640, 0.036),
        "VMamba2 (T=6)":  (1.35, 0.815, 0.004, 0.658, 0.019),
        "ConvLSTM (T=6)": (4.61, 0.822, 0.014, 0.621, 0.033),
        "U-Net (T=6)":    (1.95, 0.805, 0.004, 0.777, 0.031),
        "1D Mamba (T=6)": (1.35, 0.713, 0.103, 0.990, 0.312),
    }
    era5 = dict(ssim=0.578, mae=1.687)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.6, 3.7))

    for ax, idx, std_idx, ylab, era_v, better in [
        (axL, 1, 2, "SSIM (higher is better)", era5["ssim"], "up"),
        (axR, 3, 4, r"MAE ($^\circ$C, lower is better)", era5["mae"], "down"),
    ]:
        for name, v in data.items():
            st = STYLE[name]
            ax.errorbar(v[0], v[idx], yerr=v[std_idx], fmt=st["marker"],
                        color=st["color"], ms=9, capsize=3, mec="white",
                        mew=0.8, elinewidth=1.2, zorder=3, label=name)
        ax.axhline(era_v, ls="--", lw=1, color=STYLE["ERA5-Land"]["color"],
                   zorder=1)
        ax.text(0.98, era_v, " ERA5-Land bilinear", color=STYLE["ERA5-Land"]["color"],
                va="bottom" if better == "up" else "top", ha="right",
                fontsize=7.5, transform=ax.get_yaxis_transform())
        ax.set_xlabel("Trainable parameters (millions)")
        ax.set_ylabel(ylab)
        ax.set_xlim(0.5, 5.2)

    # annotate the headline comparison on the SSIM panel
    axL.annotate("3.4$\\times$ fewer params,\nhigher SSIM",
                 xy=(1.35, 0.825), xytext=(2.4, 0.79),
                 fontsize=8, ha="left",
                 arrowprops=dict(arrowstyle="->", color="#1f4e9c", lw=1.2))
    axL.set_title("(a) Structural fidelity vs. model size")
    axR.set_title("(b) Mean error vs. model size")

    handles, labels = axL.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.04), frameon=False, handletextpad=0.3,
               columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "F4_efficiency_tradeoff")


# --------------------------------------------------------------------------
# Fig. F5 — seasonal MAE (Table: seasonal stratification)
# --------------------------------------------------------------------------
def fig_seasonal():
    seasons = ["JJA", "SON", "DJF", "MAM"]
    data = {  # model: (means, stds) ordered as seasons
        "ConvLSTM (T=6)": ([0.581, 0.606, 0.687, 0.613], [0.042, 0.033, 0.041, 0.026]),
        "VMamba2 (T=12)": ([0.591, 0.613, 0.730, 0.631], [0.044, 0.036, 0.072, 0.031]),
        "VMamba2 (T=6)":  ([0.605, 0.633, 0.747, 0.648], [0.054, 0.034, 0.054, 0.024]),
    }
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(seasons))
    w = 0.26
    for i, (name, (m, s)) in enumerate(data.items()):
        st = STYLE[name]
        ax.bar(x + (i - 1) * w, m, w, yerr=s, capsize=3, label=name,
               color=st["color"], edgecolor="white", linewidth=0.6,
               error_kw=dict(elinewidth=1, alpha=0.7))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}" for s in seasons])
    ax.set_xlabel("Boreal season")
    ax.set_ylabel(r"MAE ($^\circ$C)")
    ax.set_ylim(0.4, 0.85)
    ax.set_title("Seasonal mean absolute error (full-year 2017, $n{=}5$)")
    ax.legend(frameon=False, ncol=1, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "F5_seasonal_mae")


# --------------------------------------------------------------------------
# Fig. F6 — per-seed robustness (Table: VMamba2 per-seed + ConvLSTM best)
# --------------------------------------------------------------------------
def fig_per_seed():
    # per-seed (MAE, SSIM)
    vt12 = {"s42": (0.673, 0.817), "s43": (0.641, 0.814), "s44": (0.583, 0.843),
            "s45": (0.668, 0.833), "s46": (0.635, 0.817)}
    vt6 = {"s42": (0.678, 0.818), "s43": (0.640, 0.817), "s44": (0.637, 0.817),
           "s45": (0.671, 0.808), "s46": (0.662, 0.813)}
    groups = [("VMamba2 (T=6)", vt6), ("VMamba2 (T=12)", vt12)]

    fig, (axM, axS) = plt.subplots(1, 2, figsize=(8.6, 3.7))
    rng = np.random.default_rng(0)
    for gi, (name, d) in enumerate(groups):
        st = STYLE[name]
        maes = np.array([v[0] for v in d.values()])
        ssims = np.array([v[1] for v in d.values()])
        jitter = (rng.random(len(maes)) - 0.5) * 0.18
        for ax, vals in [(axM, maes), (axS, ssims)]:
            ax.scatter(np.full_like(vals, gi) + jitter, vals, s=55,
                       color=st["color"], marker=st["marker"], ec="white",
                       lw=0.7, zorder=3)
            # mean +/- std bar
            ax.errorbar(gi, vals.mean(), yerr=vals.std(ddof=1), fmt="_",
                        color="black", ms=26, mew=1.6, capsize=6,
                        elinewidth=1.4, zorder=2)
        # label the overall best seed (s44, T=12) on the MAE panel
        if name == "VMamba2 (T=12)":
            axM.annotate("best seed (s44)", (gi + jitter[2], maes[2]),
                         textcoords="offset points", xytext=(9, -1),
                         fontsize=7.5, color=st["color"])

    for ax, ylab, ttl in [(axM, r"MAE ($^\circ$C)", "(a) Per-seed MAE"),
                          (axS, "SSIM", "(b) Per-seed SSIM")]:
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["VMamba2\n($T{=}6$)", "VMamba2\n($T{=}12$)"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel(ylab)
        ax.set_title(ttl)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Cross-seed dispersion ($n{=}5$): black bar = mean $\\pm$ s.d.",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    save(fig, "F6_per_seed_robustness")


# --------------------------------------------------------------------------
# SUPPLEMENT
# --------------------------------------------------------------------------
# Full-frame aggregate stats (mean, std) transcribed from Table 2.
FULLFRAME = {  # model: dict(mae,mae_s,rmse,rmse_s,ssim,ssim_s,params)
    "VMamba2 (T=12)": (0.640, 0.036, 0.849, 0.046, 0.825, 0.013, 1.35),
    "ConvLSTM (T=6)": (0.621, 0.033, 0.825, 0.041, 0.822, 0.014, 4.61),
    "VMamba2 (T=6)":  (0.658, 0.019, 0.871, 0.022, 0.815, 0.004, 1.35),
    "U-Net (T=6)":    (0.777, 0.031, 1.028, 0.038, 0.805, 0.004, 1.95),
    "1D Mamba (T=6)": (0.990, 0.312, 1.306, 0.402, 0.713, 0.103, 1.35),
}


def fig_s1_reproducibility():
    """Coefficient of variation (std/mean, %) of MAE and SSIM across seeds."""
    names = list(FULLFRAME)
    cv_mae = [100 * v[1] / v[0] for v in FULLFRAME.values()]
    cv_ssim = [100 * v[5] / v[4] for v in FULLFRAME.values()]

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w / 2, cv_mae, w, label="MAE", color="#c0392b",
           edgecolor="white", linewidth=0.6)
    ax.bar(x + w / 2, cv_ssim, w, label="SSIM", color="#1f4e9c",
           edgecolor="white", linewidth=0.6)
    for xi, (m, s) in enumerate(zip(cv_mae, cv_ssim)):
        ax.text(xi - w / 2, m + 0.4, f"{m:.1f}", ha="center", fontsize=7)
        ax.text(xi + w / 2, s + 0.4, f"{s:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
    ax.set_ylabel("Cross-seed coefficient of variation (%)")
    ax.set_title("Reproducibility: cross-seed CV of test-set metrics ($n{=}5$)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "FigS1_reproducibility_cv")


def fig_s2_param_efficiency():
    """SSIM per million parameters (Table: efficiency)."""
    eff = {  # model: SSIM/Mparam
        "VMamba2 (T=12)": 0.611, "VMamba2 (T=6)": 0.604,
        "1D Mamba (T=6)": 0.528, "U-Net (T=6)": 0.413, "ConvLSTM (T=6)": 0.178,
    }
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    names = list(eff)
    vals = list(eff.values())
    colors = [STYLE[n]["color"] for n in names]
    bars = ax.barh(range(len(names))[::-1], vals, color=colors,
                   edgecolor="white", linewidth=0.6)
    ax.set_yticks(range(len(names))[::-1])
    ax.set_yticklabels(names)
    for b, v in zip(bars, vals):
        ax.text(v + 0.008, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=8)
    ax.set_xlabel("SSIM per million parameters")
    ax.set_xlim(0, 0.7)
    ax.set_title("Structural fidelity per million parameters")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "FigS2_param_efficiency")


if __name__ == "__main__":
    print("Generating result figures...")
    fig_efficiency()
    fig_seasonal()
    fig_per_seed()
    print("Generating supplementary figures...")
    fig_s1_reproducibility()
    fig_s2_param_efficiency()
    print("Done.")
