"""
coarse_grain_cooling.py
-----------------------
Coarse-grains high-resolution simulation data (512x1024) by block averaging
to resolutions: 256x512, 128x256, 64x128, 32x64, 16x32, and 8x16 across
all frames from t=5.0 Myr to simulation end (t=10.0 Myr).

Key Analysis:
- Active cooling window: T in [1.05e4, 0.95e6] K (Lambda(T) = 0 outside).
- Computes coarse-grained resolved cooling: n_bar^2 * Lambda(T_bar).
- Computes subgrid PDF cooling: n_bar^2 * sum_i Lambda(T_i) * PDF(T_i) using
  log-spaced bins from 10^3 to 10^7 K.
- Treats each pixel from each frame as a separate entry in the 2D histogram / scatter.
- Produces individual zero-cooling fraction and cooling distribution plots (matching
  pdf_plot.py style), individual 2D hexbin scatter plots, and combined cross-resolution
  summary figures across the full t in [5.0, 10.0] Myr time series.
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configuration Variables (Easily accessible near the top)
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIM_DIR = PROJECT_ROOT / "simulation_outputs" / "hr_gpu_512x1024"
ATHINPUT_PATH = SIM_DIR / "kh_radiative_512x1024.athinput"
BIN_DIR = SIM_DIR / "bin"
OUTPUT_DIR = PROJECT_ROOT / "explore_data" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Time range for analysis (t = 5.0 Myr to simulation end t = 10.0 Myr)
T_START_MYR = 0.0
T_END_MYR = 10.0

# Active cooling window: Lambda(T) != 0 only inside [T_ACTIVE_MIN, T_ACTIVE_MAX]
T_ACTIVE_MIN = 1.05e4   # 1.05 x 10^4 K
T_ACTIVE_MAX = 0.95e6   # 0.95 x 10^6 K

# Target coarse-grained resolutions: (nx, ny)
TARGET_RESOLUTIONS = [
    (256, 512),   # ds = 2
    (128, 256),   # ds = 4
    (64, 128),    # ds = 8
    (32, 64),     # ds = 16
    (16, 32),     # ds = 32
    (8, 16),      # ds = 64
]

# Temperature binning configuration (10^3 to 10^7 Kelvin)
T_MIN = 1e3
T_MAX = 1e7
NUM_BINS = 100

# Physical constants / parameters
MU = 0.62
M_H = 1.6726e-24  # g

# Output filenames
PLOT_COMBINED_PATH = OUTPUT_DIR / "coarse_grain_cooling_comparison_t5_to_10Myr.png"
PLOT_ALL_ZERO_HIST_PATH = OUTPUT_DIR / "all_resolutions_zero_and_histograms_t5_to_10Myr.png"

# -----------------------------------------------------------------------------
# Environment & Imports
# -----------------------------------------------------------------------------
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "conv_nn"))

import ergane
from pdf_cnn import lambda_cool

# Active window cooling function
def active_lambda_cool(temp):
    """Cooling function masked to active window [T_ACTIVE_MIN, T_ACTIVE_MAX]."""
    lam = lambda_cool(temp, mask=False)
    inactive_mask = (temp < T_ACTIVE_MIN) | (temp > T_ACTIVE_MAX)
    if isinstance(lam, np.ndarray):
        lam = np.where(inactive_mask, 0.0, lam)
    else:
        if inactive_mask:
            lam = 0.0
    return lam

# -----------------------------------------------------------------------------
# Load Simulation Metadata and Filter Frames
# -----------------------------------------------------------------------------
print(f"Loading simulation data from: {SIM_DIR}")
sim = ergane.SimulationData(athinp=str(ATHINPUT_PATH), datafolder=str(BIN_DIR))

# Filter frame numbers for t in [T_START_MYR, T_END_MYR]
target_frame_nums = [
    num for num in sim.frame_numbers
    if (sim.times[num] >= T_START_MYR - 1e-5) and (sim.times[num] <= T_END_MYR + 1e-5)
]

print(f"Total simulation frames: {sim.n_frames}")
print(f"Selected {len(target_frame_nums)} frames from t = {sim.times[target_frame_nums[0]]:.2f} Myr to {sim.times[target_frame_nums[-1]]:.2f} Myr (frames {target_frame_nums[0]} to {target_frame_nums[-1]})")

# -----------------------------------------------------------------------------
# Set up Temperature Bins and Lambda Evaluation
# -----------------------------------------------------------------------------
temp_bins = np.logspace(np.log10(T_MIN), np.log10(T_MAX), NUM_BINS + 1)
log_T_centers = 0.5 * (np.log10(temp_bins[:-1]) + np.log10(temp_bins[1:]))
T_centers = 10.0 ** log_T_centers
lam_centers = active_lambda_cool(T_centers)

# -----------------------------------------------------------------------------
# Process All Frames and Accumulate Pixels Across Resolutions
# -----------------------------------------------------------------------------
res_data = {
    (nx, ny): {
        "cool_cg_list": [],
        "cool_sg_list": [],
        "ds": None,
    }
    for nx, ny in TARGET_RESOLUTIONS
}

print(f"\nProcessing {len(target_frame_nums)} frames across all 6 resolutions...")
for f_num in tqdm(target_frame_nums, desc="Streaming frames"):
    frame = sim.get_frame(f_num)
    rho_cgs = frame.density
    n_cgs = rho_cgs / (MU * M_H)
    T_fine = frame.temperature
    H, W = T_fine.shape
    
    # Precompute bin indices for fine grid temperature cells
    bin_idx_fine = np.clip(np.digitize(T_fine, temp_bins) - 1, 0, NUM_BINS - 1)
    lam_fine = lam_centers[bin_idx_fine]
    
    for nx_cg, ny_cg in TARGET_RESOLUTIONS:
        ds = W // nx_cg
        H_cg, W_cg = ny_cg, nx_cg
        res_data[(nx_cg, ny_cg)]["ds"] = ds
        
        # Reshape into 2D blocks of size (ds, ds)
        T_blocks = T_fine.reshape(H_cg, ds, W_cg, ds).transpose(0, 2, 1, 3).reshape(H_cg, W_cg, ds * ds)
        n_blocks = n_cgs.reshape(H_cg, ds, W_cg, ds).transpose(0, 2, 1, 3).reshape(H_cg, W_cg, ds * ds)
        lam_blocks = lam_fine.reshape(H_cg, ds, W_cg, ds).transpose(0, 2, 1, 3).reshape(H_cg, W_cg, ds * ds)
        
        # 1. Coarse-grained resolved cooling: \bar{n}^2 * \Lambda(\bar{T})
        T_bar = T_blocks.mean(axis=-1)
        n_bar = n_blocks.mean(axis=-1)
        cool_cg = (n_bar ** 2) * active_lambda_cool(T_bar)
        
        # 2. Subgrid PDF-integrated cooling: \bar{n}^2 * \sum_i \Lambda(T_i) * PDF(T_i)
        cool_sg = (n_bar ** 2) * lam_blocks.mean(axis=-1)
        
        res_data[(nx_cg, ny_cg)]["cool_cg_list"].append(cool_cg.ravel())
        res_data[(nx_cg, ny_cg)]["cool_sg_list"].append(cool_sg.ravel())

# -----------------------------------------------------------------------------
# Concatenate and Compute Full-Time-Series Statistics
# -----------------------------------------------------------------------------
results = []
print("\nAggregating multi-frame statistics for each resolution...")

for nx_cg, ny_cg in TARGET_RESOLUTIONS:
    ds = res_data[(nx_cg, ny_cg)]["ds"]
    cg_flat = np.concatenate(res_data[(nx_cg, ny_cg)]["cool_cg_list"])
    sg_flat = np.concatenate(res_data[(nx_cg, ny_cg)]["cool_sg_list"])
    total_entries = cg_flat.size
    
    # Zero cooling pixel statistics (active window: [1.05e4, 0.95e6] K)
    zero_cg_count = int((cg_flat == 0.0).sum())
    zero_sg_count = int((sg_flat == 0.0).sum())
    zero_cg_frac = zero_cg_count / total_entries
    zero_sg_frac = zero_sg_count / total_entries
    
    # Active positive cooling pixels in both
    valid_mask = (cg_flat > 0) & (sg_flat > 0)
    if valid_mask.sum() > 1:
        log_cg = np.log10(cg_flat[valid_mask])
        log_sg = np.log10(sg_flat[valid_mask])
        log_bias = float(np.mean(log_cg - log_sg))
        log_rmse = float(np.sqrt(np.mean((log_cg - log_sg) ** 2)))
        corr, _ = pearsonr(log_sg, log_cg)
    else:
        log_bias, log_rmse, corr = 0.0, 0.0, 0.0
        
    print(f"\nResolution {nx_cg}x{ny_cg} (ds = {ds}x{ds}):")
    print(f"  Total pixel entries across all {len(target_frame_nums)} frames: {total_entries:,}")
    print(f"  Zero cooling pixels CG: {zero_cg_count:,} ({zero_cg_frac*100:.1f}%) | SG: {zero_sg_count:,} ({zero_sg_frac*100:.1f}%)")
    print(f"  Valid active pixels   : {valid_mask.sum():,} / {total_entries:,}")
    print(f"  Log Bias (dex)        : {log_bias:+.3f}")
    print(f"  Log RMSE (dex)        : {log_rmse:.3f}")
    print(f"  Pearson correlation   : {corr:.4f}")
    print(f"  Mean resolved cooling : {cg_flat.mean():.3e} erg/(cm^3 s)")
    print(f"  Mean subgrid cooling  : {sg_flat.mean():.3e} erg/(cm^3 s)")
    print(f"  Ratio (resolved/SG)   : {cg_flat.mean() / (sg_flat.mean() + 1e-40):.2f}x")
    
    res_dict = {
        "nx": nx_cg,
        "ny": ny_cg,
        "ds": ds,
        "cg_flat": cg_flat,
        "sg_flat": sg_flat,
        "total_entries": total_entries,
        "zero_cg_count": zero_cg_count,
        "zero_sg_count": zero_sg_count,
        "zero_cg_frac": zero_cg_frac,
        "zero_sg_frac": zero_sg_frac,
        "valid_mask": valid_mask,
        "log_bias": log_bias,
        "log_rmse": log_rmse,
        "corr": corr,
    }
    results.append(res_dict)

    # -------------------------------------------------------------------------
    # Individual Zero-Cooling and Histogram Plot (Exact style of pdf_plot.py)
    # -------------------------------------------------------------------------
    _fields = {
        "Subgrid PDF": sg_flat,
        "Coarse-Grain": cg_flat,
    }
    _colors = ["darkorange", "royalblue"]
    
    fig_hist, (ax_zero, ax_pos) = plt.subplots(1, 2, figsize=(14, 5.2))
    
    # Panel 1: Zero-fraction bar chart
    zero_fracs = [np.mean(v == 0.0) * 100 for v in _fields.values()]
    bars = ax_zero.bar(_fields.keys(), zero_fracs, color=_colors, width=0.45)
    for bar, frac, count in zip(bars, zero_fracs, [zero_sg_count, zero_cg_count]):
        ax_zero.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{frac:.1f}%\n({count:,} px)",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
        )
    ax_zero.set_ylabel("Fraction of pixels with cooling = 0 (%)", fontsize=11)
    ax_zero.set_title(f"Zero-Cooling Fraction ({nx_cg}x{ny_cg}, ds={ds}x{ds})\nActive Window: [1.05e4, 0.95e6] K", fontsize=11.5, fontweight="bold")
    ax_zero.set_ylim(0, max(zero_fracs) * 1.35 + 2)
    ax_zero.grid(True, linestyle=":", alpha=0.6, axis="y")
    
    # Panel 2: log10(cooling) distribution for positive pixels
    for (label, vals), col in zip(_fields.items(), _colors):
        pos = vals[vals > 0]
        if len(pos) == 0:
            continue
        log_pos = np.log10(pos)
        ax_pos.hist(
            log_pos,
            bins=80,
            density=True,
            histtype="step",
            linewidth=2.2,
            label=f"{label} (N={len(pos):,})",
            color=col,
        )
        
    ax_pos.set_xlabel(r"$\log_{10}(\mathrm{Cooling\;Rate})\;[\mathrm{erg\,cm^{-3}\,s^{-1}}]$", fontsize=11)
    ax_pos.set_ylabel("Probability Density", fontsize=11)
    ax_pos.set_title(r"Distribution of Positive $\log_{10}$(Cooling)", fontsize=11.5, fontweight="bold")
    ax_pos.legend(loc="upper left", fontsize=10)
    ax_pos.set_yscale("log")
    ax_pos.grid(True, linestyle=":", alpha=0.6)
    
    fig_hist.suptitle(
        f"Cooling Rate Histograms & Zero Pixels — Resolution {nx_cg}x{ny_cg} (t = {T_START_MYR:.1f} to {T_END_MYR:.1f} Myr, {len(target_frame_nums)} frames)",
        fontsize=13, fontweight="bold", y=0.98,
    )
    fig_hist.tight_layout()
    hist_out_path = OUTPUT_DIR / f"pdf_cooling_histogram_{nx_cg}x{ny_cg}_t5_to_10Myr.png"
    fig_hist.savefig(hist_out_path, dpi=200, bbox_inches="tight")
    plt.close(fig_hist)
    print(f"  Saved histogram plot: {hist_out_path.name}")

    # -------------------------------------------------------------------------
    # Individual 2D Hexbin Scatter Plot (All pixels across all frames)
    # -------------------------------------------------------------------------
    fig_indiv_sc, ax_sc = plt.subplots(1, 1, figsize=(7.8, 6.6))
    pos_sg = sg_flat[sg_flat > 0]
    pos_cg = cg_flat[cg_flat > 0]
    vmin_i = min(np.percentile(pos_sg, 0.1) if len(pos_sg) else 1e-30, np.percentile(pos_cg, 0.1) if len(pos_cg) else 1e-30)
    vmax_i = max(np.percentile(pos_sg, 99.9) if len(pos_sg) else 1e-20, np.percentile(pos_cg, 99.9) if len(pos_cg) else 1e-20)
    vmin_i_log = np.floor(np.log10(vmin_i))
    vmax_i_log = np.ceil(np.log10(vmax_i))
    lims_i = [10.0 ** vmin_i_log, 10.0 ** vmax_i_log]
    
    if valid_mask.sum() > 0:
        hb_i = ax_sc.hexbin(
            sg_flat[valid_mask],
            cg_flat[valid_mask],
            gridsize=75,
            bins="log",
            xscale="log",
            yscale="log",
            cmap="viridis",
            mincnt=1,
            extent=(vmin_i_log, vmax_i_log, vmin_i_log, vmax_i_log),
        )
        cb_i = fig_indiv_sc.colorbar(hb_i, ax=ax_sc, shrink=0.85, pad=0.03)
        cb_i.set_label("Pixel Entry Count (log)", fontsize=10)
    
    ax_sc.plot(lims_i, lims_i, "r--", linewidth=1.8, label="1:1 Reference")
    ax_sc.set_xlim(lims_i)
    ax_sc.set_ylim(lims_i)
    ax_sc.set_aspect("equal", "box")
    ax_sc.set_xlabel(r"Subgrid $\bar{n}^2 \sum_i \Lambda(T_i) \mathrm{PDF}(T_i)\;[\mathrm{erg\,cm^{-3}\,s^{-1}}]$", fontsize=11)
    ax_sc.set_ylabel(r"Coarse-Grain $\bar{n}^2 \Lambda(\bar{T})\;[\mathrm{erg\,cm^{-3}\,s^{-1}}]$", fontsize=11)
    ax_sc.set_title(f"Coarse vs Subgrid Cooling: {nx_cg} $\\times$ {ny_cg} ($ds={ds}\\times{ds}$)\n$t = {T_START_MYR:.1f}$ to ${T_END_MYR:.1f}$ Myr ({valid_mask.sum():,} active pixels)", fontsize=12, fontweight="bold")
    ax_sc.legend(loc="lower right", fontsize=10)
    
    stats_text = (
        f"Grid: {nx_cg} $\\times$ {ny_cg}\n"
        f"Total Entries: {total_entries:,}\n"
        f"Log Bias: {log_bias:+.2f} dex\n"
        f"Log RMSE: {log_rmse:.2f} dex\n"
        f"Pearson $r$: {corr:.3f}\n"
        f"Ratio $\\langle\\mathrm{{CG}}\\rangle/\\langle\\mathrm{{SG}}\\rangle$: {cg_flat.mean()/(sg_flat.mean()+1e-40):.2f}$\\times$\n"
        f"Zero Pixels: CG {zero_cg_frac*100:.1f}% | SG {zero_sg_frac*100:.1f}%"
    )
    ax_sc.text(0.05, 0.95, stats_text, transform=ax_sc.transAxes, fontsize=9.5, verticalalignment="top",
               bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.92, edgecolor="gray"))
    
    fig_indiv_sc.tight_layout()
    sc_out_path = OUTPUT_DIR / f"coarse_grain_scatter_{nx_cg}x{ny_cg}_t5_to_10Myr.png"
    fig_indiv_sc.savefig(sc_out_path, dpi=200, bbox_inches="tight")
    plt.close(fig_indiv_sc)
    print(f"  Saved scatter plot: {sc_out_path.name}")

# -----------------------------------------------------------------------------
# Combined 6-Panel Comparison Scatter Plot (Full Time-Series)
# -----------------------------------------------------------------------------
print("\nGenerating combined 6-panel comparison figure...")
fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=False, sharey=False)
axes_flat = axes.flatten()

all_positive_sg = [r["sg_flat"][r["valid_mask"]] for r in results if r["valid_mask"].sum() > 0]
all_positive_cg = [r["cg_flat"][r["valid_mask"]] for r in results if r["valid_mask"].sum() > 0]
vmin_data = min(np.percentile(np.concatenate(all_positive_sg), 0.1), np.percentile(np.concatenate(all_positive_cg), 0.1))
vmax_data = max(np.percentile(np.concatenate(all_positive_sg), 99.9), np.percentile(np.concatenate(all_positive_cg), 99.9))

vmin_log = np.floor(np.log10(vmin_data))
vmax_log = np.ceil(np.log10(vmax_data))
lims = [10.0 ** vmin_log, 10.0 ** vmax_log]

for idx, (ax, res) in enumerate(zip(axes_flat, results)):
    sg = res["sg_flat"]
    cg = res["cg_flat"]
    mask = res["valid_mask"]
    
    x_vals = sg[mask]
    y_vals = cg[mask]
    
    hb = ax.hexbin(
        x_vals,
        y_vals,
        gridsize=70,
        bins="log",
        xscale="log",
        yscale="log",
        cmap="viridis",
        mincnt=1,
        extent=(vmin_log, vmax_log, vmin_log, vmax_log),
    )
    
    ax.plot(lims, lims, color="crimson", linestyle="--", linewidth=1.8, label="1:1 Reference")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", "box")
    ax.set_xlabel(r"Subgrid $\bar{n}^2 \sum_i \Lambda(T_i) \mathrm{PDF}(T_i)\;[\mathrm{erg\,cm^{-3}\,s^{-1}}]$", fontsize=11)
    ax.set_ylabel(r"Coarse-Grain $\bar{n}^2 \Lambda(\bar{T})\;[\mathrm{erg\,cm^{-3}\,s^{-1}}]$", fontsize=11)
    ax.set_title(
        f"Resolution: ${res['nx']} \\times {res['ny']}$  ($\\mathrm{{ds}} = {res['ds']} \\times {res['ds']}$)",
        fontsize=13,
        fontweight="bold",
    )
    
    stats_text = (
        f"Grid: {res['nx']} $\\times$ {res['ny']}\n"
        f"Entries: {res['total_entries']:,}\n"
        f"Log Bias: {res['log_bias']:+.2f} dex\n"
        f"Log RMSE: {res['log_rmse']:.2f} dex\n"
        f"Pearson $r$: {res['corr']:.3f}\n"
        f"Ratio $\\langle\\mathrm{{CG}}\\rangle/\\langle\\mathrm{{SG}}\\rangle$: {cg.mean()/(sg.mean()+1e-40):.2f}$\\times$\n"
        f"Zero Pixels: CG {res['zero_cg_frac']*100:.1f}% | SG {res['zero_sg_frac']*100:.1f}%"
    )
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=9.2,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.90, edgecolor="gray"),
    )
    
    cb = fig.colorbar(hb, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Pixel Count (log)", fontsize=9)

plt.suptitle(
    f"Coarse-Grained Resolved vs. Subgrid PDF Cooling (All Frames $t = {T_START_MYR:.1f}$ to ${T_END_MYR:.1f}$ Myr, {len(target_frame_nums)} snapshots)\n"
    f"Simulation: hr_gpu_512x1024 | Active Window: [{T_ACTIVE_MIN:.2e}, {T_ACTIVE_MAX:.2e}] K | Bins: {NUM_BINS} (10³ – 10⁷ K)",
    fontsize=15,
    fontweight="bold",
    y=0.99,
)

plt.tight_layout()
plt.savefig(PLOT_COMBINED_PATH, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved combined comparison plot to: {PLOT_COMBINED_PATH.name}")

# -----------------------------------------------------------------------------
# Cross-Resolution Zero Pixels Summary (Full Time-Series)
# -----------------------------------------------------------------------------
print("\nGenerating cross-resolution zero-fraction summary...")
fig_zh, (ax_z1, ax_z2) = plt.subplots(1, 2, figsize=(15, 5.2))

res_labels = [f"{r['nx']}x{r['ny']}\n(ds={r['ds']})" for r in results]
x_indices = np.arange(len(results))
width = 0.35

zero_cg_counts = [r["zero_cg_count"] for r in results]
zero_sg_counts = [r["zero_sg_count"] for r in results]
b1 = ax_z1.bar(x_indices - width/2, zero_cg_counts, width, label="Coarse-Grain $\\bar{n}^2\\Lambda(\\bar{T})$", color="royalblue", alpha=0.85)
b2 = ax_z1.bar(x_indices + width/2, zero_sg_counts, width, label="Subgrid PDF $\\bar{n}^2\\sum\\Lambda\\mathrm{PDF}$", color="darkorange", alpha=0.85)
ax_z1.set_xticks(x_indices)
ax_z1.set_xticklabels(res_labels, fontsize=10)
ax_z1.set_ylabel("Total Number of Zero Cooling Pixels", fontsize=11)
ax_z1.set_title(f"Zero-Cooling Pixel Counts ($t = {T_START_MYR:.1f} - {T_END_MYR:.1f}$ Myr)", fontsize=12, fontweight="bold")
ax_z1.legend(fontsize=10)
ax_z1.grid(True, linestyle=":", alpha=0.6, axis="y")

zero_cg_pcts = [r["zero_cg_frac"] * 100 for r in results]
zero_sg_pcts = [r["zero_sg_frac"] * 100 for r in results]
b3 = ax_z2.bar(x_indices - width/2, zero_cg_pcts, width, label="Coarse-Grain $\\bar{n}^2\\Lambda(\\bar{T})$", color="royalblue", alpha=0.85)
b4 = ax_z2.bar(x_indices + width/2, zero_sg_pcts, width, label="Subgrid PDF $\\bar{n}^2\\sum\\Lambda\\mathrm{PDF}$", color="darkorange", alpha=0.85)

for rect, pct in zip(b3, zero_cg_pcts):
    ax_z2.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.6, f"{pct:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
for rect, pct in zip(b4, zero_sg_pcts):
    ax_z2.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.6, f"{pct:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax_z2.set_xticks(x_indices)
ax_z2.set_xticklabels(res_labels, fontsize=10)
ax_z2.set_ylabel("Fraction of pixels with cooling = 0 (%)", fontsize=11)
ax_z2.set_title(f"Zero-Cooling Pixel Percentage ($t = {T_START_MYR:.1f} - {T_END_MYR:.1f}$ Myr)\nActive Window: [{T_ACTIVE_MIN:.2e}, {T_ACTIVE_MAX:.2e}] K", fontsize=12, fontweight="bold")
ax_z2.set_ylim(0, max(zero_cg_pcts + zero_sg_pcts) * 1.25 + 3)
ax_z2.legend(fontsize=10)
ax_z2.grid(True, linestyle=":", alpha=0.6, axis="y")

fig_zh.suptitle(f"Zero-Cooling Fraction Analysis Across All Frames $t = {T_START_MYR:.1f}$ to ${T_END_MYR:.1f}$ Myr", fontsize=14, fontweight="bold", y=0.98)
fig_zh.tight_layout()
fig_zh.savefig(PLOT_ALL_ZERO_HIST_PATH, dpi=200, bbox_inches="tight")
plt.close(fig_zh)
print(f"Saved cross-resolution zero-pixels summary to: {PLOT_ALL_ZERO_HIST_PATH.name}")

print("\n--- All coarse grain calculations and plots completed successfully! ---")
