#!/usr/bin/env python3
"""
Explore the time-evolving histogram of the bin with the peak predicted PDF
across the full temperature range (10^3 K to 10^7 K) for the Subgrid model.
Marks the active cooling zone on the visualization.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = (_HERE / "../..").resolve()

sys.path.insert(0, str(PROJECT_ROOT))                            # ergane
sys.path.insert(0, str(PROJECT_ROOT / "models" / "conv_nn"))    # pdf_cnn

from ergane import SimulationData, Frame
from pdf_cnn import (
    ConvNN,
    kernel_size,
    in_channels,
    layer_size1,
    layer_size2,
    layer_size3,
    layer_size4,
    out_channels,
    device,
)

# ---------------------------------------------------------------------------
# Constants & Model Parameters
# ---------------------------------------------------------------------------
TILE_ROWS    = 16
TILE_COLS    = 8
CNN_FINE_RES = (1024, 512)
CNN_DS       = 64

T_edges = np.logspace(3.0, 7.0, out_channels + 1)
T_centers = np.sqrt(T_edges[:-1] * T_edges[1:])
logT_centers = np.log10(T_centers)
logT_edges = np.log10(T_edges)

DEFAULT_MODEL_SAVES_DIR = str(PROJECT_ROOT / "outputs" / "model_saves" / "pdf_model_saves")
MODEL_SAVES_DIR = os.environ.get("MODEL_SAVES_DIR", DEFAULT_MODEL_SAVES_DIR)

SG_BIN_DIR = Path(os.environ.get(
    "SG_BIN_PATH",
    str(PROJECT_ROOT / "simulation_outputs"
        / "subgrid_32x16_vshear31_cf033" / "sg_tiled"),
))
MOCKS_DIR = Path(os.environ.get(
    "SG_MOCKS_DIR",
    str(PROJECT_ROOT / "simulation_outputs"
        / "subgrid_32x16_vshear31_cf033" / "mocks"),
))
MOCKS_DIR.mkdir(parents=True, exist_ok=True)
save_str = str(MOCKS_DIR) + "/"

RESTART_TIME_MYR = 5.0
BIN_DT_MYR       = 0.01

LOGT_ACTIVE_START = float(os.environ.get("LOGT_ACTIVE_START", np.log10(1.05e4)))
LOGT_ACTIVE_END   = float(os.environ.get("LOGT_ACTIVE_END",   np.log10(0.95e6)))

print(f"[explore_peak_pdf] Temperature range: [{T_edges[0]:.1e} K, {T_edges[-1]:.1e} K] (40 bins)")
print(f"[explore_peak_pdf] Active Cooling: [{10**LOGT_ACTIVE_START:.2e} K, {10**LOGT_ACTIVE_END:.2e} K] (logT: [{LOGT_ACTIVE_START:.2f}, {LOGT_ACTIVE_END:.2f}])")


def _frame_fields(fr: Frame) -> dict[str, np.ndarray]:
    d  = np.asarray(fr.density,     dtype=np.float32)
    p  = np.asarray(fr.pressure,    dtype=np.float32)
    t  = np.asarray(fr.temperature, dtype=np.float32)
    ux = np.asarray(fr.velx,        dtype=np.float32)
    uy = np.asarray(fr.vely,        dtype=np.float32)
    ps = np.asarray(fr.passive_scalar, dtype=np.float32) if hasattr(fr, "passive_scalar") else np.zeros_like(d)
    return {"rho": d, "pres": p, "temp": t, "ux": ux, "uy": uy, "ps": ps}


def load_small_sim(sim: SimulationData, frame_nums: list[int], desc: str = "Frames"):
    n = len(frame_nums)
    ny, nx = sim.ny, sim.nx
    arrays = {
        k: np.zeros((n, ny, nx), dtype=np.float32)
        for k in ("rho", "pres", "temp", "ux", "uy", "ps")
    }
    for i, num in enumerate(tqdm(frame_nums, desc=desc)):
        fr = sim.get_frame(num)
        flds = _frame_fields(fr)
        for k, arr in flds.items():
            arrays[k][i] = arr
    return arrays


# ---------------------------------------------------------------------------
# Load SG Simulation Data
# ---------------------------------------------------------------------------
print(f"[explore_peak_pdf] Loading SG simulation from {SG_BIN_DIR} ...")
sg_sim = SimulationData(datafolder=str(SG_BIN_DIR))
sg_frames = sg_sim.frame_numbers
nt = len(sg_frames)
resolution = (sg_sim.ny, sg_sim.nx)
t_myr = RESTART_TIME_MYR + np.arange(nt) * BIN_DT_MYR

print(f"[explore_peak_pdf] Resolution: {resolution}, Timesteps: {nt}, Time range: [{t_myr[0]:.2f}, {t_myr[-1]:.2f}] Myr")

sg = load_small_sim(sg_sim, sg_frames, desc="Loading SG binary frames")
rho  = sg["rho"]
temp = sg["temp"]
ux   = sg["ux"]
uy   = sg["uy"]
ps   = sg["ps"]

# ---------------------------------------------------------------------------
# Fast Batched CNN Model Inference
# ---------------------------------------------------------------------------
print("[explore_peak_pdf] Loading CNN model for fast batched inference ...")
save_dir = MODEL_SAVES_DIR
norm_prefix = f"cnn_{CNN_FINE_RES}_{CNN_DS}"
input_mean = torch.tensor(np.load(os.path.join(save_dir, f"{norm_prefix}_input_mean.npy")), dtype=torch.float32).to(device)
input_std = torch.tensor(np.load(os.path.join(save_dir, f"{norm_prefix}_input_std.npy")), dtype=torch.float32).to(device)
if input_mean.dim() == 1:
    input_mean = input_mean.view(1, -1, 1, 1)
    input_std = input_std.view(1, -1, 1, 1)

model_path = os.path.join(save_dir, f"{norm_prefix}.pth")
state_dict = torch.load(model_path, map_location=device)
ckpt_ksize = state_dict["encoder.0.weight"].shape[-1] if "encoder.0.weight" in state_dict else kernel_size

cnn_model = ConvNN(in_channels, layer_size1, layer_size2, layer_size3, layer_size4, out_channels, ckpt_ksize).to(device)
cnn_model.load_state_dict(state_dict)
cnn_model.eval()

# Extract all 4 tiles for all nt frames: (nt, 4, 5, 16, 8)
n_tile_r = resolution[0] // TILE_ROWS  # 2
n_tile_c = resolution[1] // TILE_COLS  # 2

tile_inputs = []
for ti in range(n_tile_r):
    for tj in range(n_tile_c):
        r0, r1 = ti * TILE_ROWS, (ti + 1) * TILE_ROWS
        c0, c1 = tj * TILE_COLS, (tj + 1) * TILE_COLS
        stack = np.stack([rho[:, r0:r1, c0:c1], temp[:, r0:r1, c0:c1], ux[:, r0:r1, c0:c1], uy[:, r0:r1, c0:c1], ps[:, r0:r1, c0:c1]], axis=1)
        tile_inputs.append((ti, tj, r0, r1, c0, c1, stack))

pred_pdf_all = np.zeros((nt, out_channels, *resolution), dtype=np.float32)

print("[explore_peak_pdf] Running batched GPU forward pass ...")
batch_size = 256
with torch.no_grad():
    for ti, tj, r0, r1, c0, c1, stack in tile_inputs:
        n_samples = stack.shape[0]
        out_list = []
        for b_start in range(0, n_samples, batch_size):
            b_end = min(b_start + batch_size, n_samples)
            inp_batch = torch.from_numpy(stack[b_start:b_end]).to(device)
            inp_batch = (inp_batch - input_mean) / input_std
            pdf_batch = cnn_model.predict_pdf(inp_batch)
            out_list.append(pdf_batch.cpu().numpy())
        tile_pdf = np.concatenate(out_list, axis=0)  # (nt, 40, 16, 8)
        pred_pdf_all[:, :, r0:r1, c0:c1] = tile_pdf

# ---------------------------------------------------------------------------
# Extract Peak Predicted PDF Bins across Full Temperature Range (10^3 - 10^7 K)
# For histograms (Panels 1-3): use argmax (mode) of the predicted PDF.
# For spatial maps (Panel 4): use expectation value <log10 T> = Σ P(Tk)·log10(Tk),
#   which matches the video cell-background coloring and avoids tile-boundary
#   bimodal argmax artifacts where the tiny hot-lobe spuriously wins over the
#   dominant cold-phase peak for deep-cold cells near tile edges.
# ---------------------------------------------------------------------------
print("[explore_peak_pdf] Computing peak predicted PDF bins and expectation values ...")

peak_bins_all = np.zeros((nt, resolution[0], resolution[1]), dtype=int)
peak_logT_all = np.zeros((nt, resolution[0], resolution[1]), dtype=np.float32)
mean_logT_all = np.zeros((nt, resolution[0], resolution[1]), dtype=np.float32)

for t in range(nt):
    pdf_t = pred_pdf_all[t]  # (40, 32, 16)
    peak_bins_all[t] = np.argmax(pdf_t, axis=0)  # (32, 16) - mode
    peak_logT_all[t] = logT_centers[peak_bins_all[t]]  # argmax temperature
    mean_logT_all[t] = np.sum(pdf_t * logT_centers[:, None, None], axis=0)  # expectation value

# ---------------------------------------------------------------------------
# Compute Time-Evolving 2D Histograms across full range (using mode bins)
# ---------------------------------------------------------------------------
H_vol_all  = np.zeros((nt, out_channels), dtype=np.float32)
H_mass_all = np.zeros((nt, out_channels), dtype=np.float32)

for t in range(nt):
    b_all = peak_bins_all[t].ravel()
    w_mass = rho[t].ravel()

    # Volume weighted
    cnts_v = np.bincount(b_all, minlength=out_channels).astype(np.float32)
    H_vol_all[t] = cnts_v / cnts_v.sum()

    # Mass weighted
    cnts_m = np.bincount(b_all, weights=w_mass, minlength=out_channels).astype(np.float32)
    H_mass_all[t] = cnts_m / cnts_m.sum()

# ---------------------------------------------------------------------------
# Generate Diagnostic Visualization
# ---------------------------------------------------------------------------
print("[explore_peak_pdf] Plotting time-evolving peak PDF histogram ...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.32, wspace=0.25)

# Extents for imshow: extent defines pixel EDGES, not centers.
# t_myr values are frame CENTERS, so pad by ±BIN_DT/2 to align pixels correctly.
# logT_edges are genuine bin edges, so no padding needed on Y.
extent_time_logT = [
    t_myr[0]  - BIN_DT_MYR / 2,
    t_myr[-1] + BIN_DT_MYR / 2,
    logT_edges[0],
    logT_edges[-1],
]

# --- Panel 1: Time-Evolving Peak Histogram (Volume-Weighted) ---
ax1 = fig.add_subplot(gs[0, 0])

ax1.axhspan(LOGT_ACTIVE_START, LOGT_ACTIVE_END, color="green", alpha=0.18,
            label="Active Cooling Zone", zorder=1)
ax1.axhline(LOGT_ACTIVE_START, color="cyan", ls="--", lw=1.6,
            label=rf"$T_\mathrm{{active, start}} = {10**LOGT_ACTIVE_START:.2e}\ \mathrm{{K}}$ ({LOGT_ACTIVE_START:.2f})", zorder=4)
ax1.axhline(LOGT_ACTIVE_END, color="lime", ls="--", lw=1.6,
            label=rf"$T_\mathrm{{active, end}} = {10**LOGT_ACTIVE_END:.2e}\ \mathrm{{K}}$ ({LOGT_ACTIVE_END:.2f})", zorder=4)

im1 = ax1.imshow(
    H_vol_all.T,
    origin="lower",
    extent=extent_time_logT,
    aspect="auto",
    cmap="magma",
    norm=mcolors.Normalize(vmin=0, vmax=float(H_vol_all.max())),
    zorder=2,
)

ax1.set_title(r"Time-Evolving Histogram of Peak Predicted PDF ($\text{Vol-Weighted}$)", fontsize=12, weight="bold")
ax1.set_xlabel("Physical Time [Myr]", fontsize=11)
ax1.set_ylabel(r"Peak Predicted Temperature $\log_{10}(T_\mathrm{peak}\ [\mathrm{K}])$", fontsize=11)
ax1.set_ylim(3.0, 7.0)
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label("Fraction of Pixels", fontsize=10)
ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.85)
ax1.grid(True, ls=":", alpha=0.4, color="gray")

# --- Panel 2: Mass-Weighted Time-Evolving Peak Histogram ---
ax2 = fig.add_subplot(gs[0, 1])

ax2.axhspan(LOGT_ACTIVE_START, LOGT_ACTIVE_END, color="green", alpha=0.18, label="Active Cooling Zone", zorder=1)
ax2.axhline(LOGT_ACTIVE_START, color="cyan", ls="--", lw=1.6, label=rf"$T_\mathrm{{active, start}}$ ({LOGT_ACTIVE_START:.2f})", zorder=4)
ax2.axhline(LOGT_ACTIVE_END, color="lime", ls="--", lw=1.6, label=rf"$T_\mathrm{{active, end}}$ ({LOGT_ACTIVE_END:.2f})", zorder=4)

im2 = ax2.imshow(
    H_mass_all.T,
    origin="lower",
    extent=extent_time_logT,
    aspect="auto",
    cmap="magma",
    norm=mcolors.Normalize(vmin=0, vmax=float(H_mass_all.max())),
    zorder=2,
)

ax2.set_title(r"Time-Evolving Histogram of Peak Predicted PDF ($\text{Mass-Weighted}$)", fontsize=12, weight="bold")
ax2.set_xlabel("Physical Time [Myr]", fontsize=11)
ax2.set_ylabel(r"Peak Predicted Temperature $\log_{10}(T_\mathrm{peak}\ [\mathrm{K}])$", fontsize=11)
ax2.set_ylim(3.0, 7.0)
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label("Mass Fraction of Pixels", fontsize=10)
ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.85)
ax2.grid(True, ls=":", alpha=0.4, color="gray")

# --- Panel 3: Snapshot 1D Histograms at Key Physical Times ---
ax3 = fig.add_subplot(gs[1, 0])

sample_times = [5.0, 6.0, 7.5, 9.0, 10.0]
sample_indices = [np.argmin(np.abs(t_myr - st)) for st in sample_times]
colors_list = plt.cm.plasma(np.linspace(0.1, 0.9, len(sample_times)))

ax3.axvspan(LOGT_ACTIVE_START, LOGT_ACTIVE_END, color="green", alpha=0.15, label="Active Cooling Zone")
ax3.axvline(LOGT_ACTIVE_START, color="cyan", ls="--", lw=1.4)
ax3.axvline(LOGT_ACTIVE_END, color="lime", ls="--", lw=1.4)

for idx, col in zip(sample_indices, colors_list):
    actual_t = t_myr[idx]
    ax3.plot(
        logT_centers,
        H_vol_all[idx],
        color=col,
        lw=2.2,
        marker="o",
        markersize=4,
        label=rf"$t = {actual_t:.2f}\ \mathrm{{Myr}}$",
    )

ax3.set_title("Peak Temperature Bin Distribution at Selected Time Snapshots", fontsize=11, weight="bold")
ax3.set_xlabel(r"Peak Predicted Temperature $\log_{10}(T_\mathrm{peak}\ [\mathrm{K}])$", fontsize=11)
ax3.set_ylabel("Pixel Fraction", fontsize=11)
ax3.set_xlim(3.0, 7.0)
ax3.set_ylim(bottom=0)
ax3.grid(True, ls="--", alpha=0.5)
ax3.legend(fontsize=9, loc="upper right")

# --- Panel 4: Spatial Maps of Argmax Predicted Temperature (Early vs Late Time)
# Uses argmax over the full predicted PDF at each cell.
# ---
ax4_sub = gs[1, 1].subgridspec(1, 2, wspace=0.1)
ax4a = fig.add_subplot(ax4_sub[0, 0])
ax4b = fig.add_subplot(ax4_sub[0, 1])

t_early_idx = np.argmin(np.abs(t_myr - 5.5))
t_late_idx  = np.argmin(np.abs(t_myr - 9.5))

norm_peak = mcolors.Normalize(vmin=3.0, vmax=7.0)
cmap_peak = plt.get_cmap("inferno")

im4a = ax4a.imshow(peak_logT_all[t_early_idx], origin="lower", cmap=cmap_peak, norm=norm_peak)
ax4a.set_title(rf"$T_\mathrm{{peak}}$ ($t={t_myr[t_early_idx]:.2f}$ Myr)", fontsize=10, weight="bold")
ax4a.set_xlabel("X (cells)", fontsize=9)
ax4a.set_ylabel("Y (cells)", fontsize=9)

im4b = ax4b.imshow(peak_logT_all[t_late_idx], origin="lower", cmap=cmap_peak, norm=norm_peak)
ax4b.set_title(rf"$T_\mathrm{{peak}}$ ($t={t_myr[t_late_idx]:.2f}$ Myr)", fontsize=10, weight="bold")
ax4b.set_xlabel("X (cells)", fontsize=9)
ax4b.set_yticks([])

cbar4 = fig.colorbar(im4b, ax=[ax4a, ax4b], fraction=0.046, pad=0.04)
cbar4.set_label(r"$\log_{10}(T_\mathrm{peak}\ [\mathrm{K}])$", fontsize=10)

plt.suptitle(
    rf"Subgrid Model: Time-Evolving Peak Predicted Temperature PDF Histogram ($10^3\ \mathrm{{K}} \leq T \leq 10^7\ \mathrm{{K}}$)"
    f"\nActive Cooling Zone: [{10**LOGT_ACTIVE_START:.2e} K, {10**LOGT_ACTIVE_END:.2e} K]",
    fontsize=14,
    weight="bold",
    y=0.99,
)

out_fig_path = save_str + "subgrid_peak_temperature_pdf_evolution.png"
plt.savefig(out_fig_path, dpi=200, bbox_inches="tight")
# Also save as cold_phase_peak_pdf_evolution.png for backward compatibility
plt.savefig(save_str + "cold_phase_peak_pdf_evolution.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"[explore_peak_pdf] Figures saved to {out_fig_path} and cold_phase_peak_pdf_evolution.png")

print("[explore_peak_pdf] Data exploration complete successfully!")
