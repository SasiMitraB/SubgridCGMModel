import os
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Environment, Constants & Setup
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "conv_nn"))

import ergane
from ergane.histograms import fast_log_histogram
from pdf_cnn import ConvNN, in_channels, layer_size1, layer_size2, layer_size3, layer_size4, out_channels, device, lambda_cool, T_centers, T_edges

CM_PER_PC = 3.08568e18          # cm per parsec
M_H       = 1.6726219e-24       # proton mass [g]
MU        = 0.62                # mean molecular weight
# -----------------------------------------------------------------------------
# Configuration: Grid Cell Size Option
# -----------------------------------------------------------------------------
# Option 1: CELL_SIZE = 0.625  (0.625 pc -> DS = 16, grid: 64x32)
# Option 2: CELL_SIZE = 1.25   (1.25 pc  -> DS = 32, grid: 32x16)
CELL_SIZE = 0.625

if np.isclose(CELL_SIZE, 0.625):
    DS = 16
    MODEL_SAVES_DIR = PROJECT_ROOT / "runs" / "run_random_crop_20260909_164548" / "model_saves"
    norm_prefix = "cnn_(512, 256)_32"
elif np.isclose(CELL_SIZE, 1.25) or np.isclose(CELL_SIZE, 1.26):
    DS = 32
    MODEL_SAVES_DIR = PROJECT_ROOT / "runs" / "run_random_crop_20260904_191402" / "model_saves"
    norm_prefix = "cnn_(512, 256)_32"
else:
    raise ValueError(f"Unsupported CELL_SIZE: {CELL_SIZE}. Expected 0.625 or 1.25 pc.")

LOGT_ACTIVE_START = 4.1
LOGT_ACTIVE_END   = 5.9
LAMBDA_CENTERS = lambda_cool(T_centers, mask=True, LOGT_ACTIVE_START=LOGT_ACTIVE_START, LOGT_ACTIVE_END=LOGT_ACTIVE_END)

log_temp_centers = 0.5 * (np.log10(T_edges[:-1]) + np.log10(T_edges[1:]))
nb_bins = len(T_centers)
active_bin_start = int(np.searchsorted(T_centers, 10**LOGT_ACTIVE_START))
active_bin_end = int(np.searchsorted(T_centers, 10**LOGT_ACTIVE_END))
norm_temp_exp = matplotlib.colors.Normalize(vmin=3.0, vmax=7.0)
cmap_temp_exp = plt.get_cmap("inferno")

# Load Simulation Data
hr_sim = ergane.SimulationData(
    athinp=str(PROJECT_ROOT / "simulation_outputs/hr_gpu_512x1024/kh_radiative_512x1024.athinput"),
    datafolder=str(PROJECT_ROOT / "simulation_outputs/hr_gpu_512x1024/bin/")
)

input_mean = torch.tensor(np.load(os.path.join(MODEL_SAVES_DIR, f"{norm_prefix}_input_mean.npy")), dtype=torch.float32, device=device).view(1, -1, 1, 1)
input_std = torch.tensor(np.load(os.path.join(MODEL_SAVES_DIR, f"{norm_prefix}_input_std.npy")), dtype=torch.float32, device=device).view(1, -1, 1, 1)
state_dict = torch.load(os.path.join(MODEL_SAVES_DIR, f"{norm_prefix}.pth"), map_location=device)
model = ConvNN(in_channels, layer_size1, layer_size2, layer_size3, layer_size4, out_channels, state_dict["encoder.0.weight"].shape[-1]).to(device)
model.load_state_dict(state_dict)
model.eval()


def coarse_grain_2d(arr: np.ndarray, ds: int = DS) -> np.ndarray:
    """Coarse-grain a 2D array by factor ds."""
    ny, nx = arr.shape
    if ds <= 1 or ny < ds or nx < ds:
        return arr.copy()
    ny_cg, nx_cg = ny // ds, nx // ds
    return arr[:ny_cg * ds, :nx_cg * ds].reshape(ny_cg, ds, nx_cg, ds).mean(axis=(1, 3))


def compute_frame_cooling(frame) -> np.ndarray:
    """Compute volumetric cooling rate in CGS [erg cm^-3 s^-1]."""
    n = frame.density / (MU * M_H)
    lam = lambda_cool(frame.temperature, mask=True, LOGT_ACTIVE_START=LOGT_ACTIVE_START, LOGT_ACTIVE_END=LOGT_ACTIVE_END)
    return (n ** 2) * lam


def compute_frame_true_pdf(frame, ds: int = DS) -> np.ndarray:
    """Compute ground-truth subgrid temperature PDF (shape: 40, ny_cg, nx_cg)."""
    T = frame.temperature
    ny_cg, nx_cg = T.shape[0] // ds, T.shape[1] // ds
    blocks = T[:ny_cg * ds, :nx_cg * ds].reshape(ny_cg, ds, nx_cg, ds).transpose(0, 2, 1, 3).reshape(ny_cg, nx_cg, -1)
    b_idx = np.clip(np.digitize(blocks, T_edges) - 1, 0, len(T_edges) - 2)
    hists = np.apply_along_axis(lambda m: np.bincount(m, minlength=len(T_edges) - 1), -1, b_idx).astype(np.float32)
    pdf = hists / (hists.sum(axis=-1, keepdims=True) + 1e-12)
    return np.transpose(pdf, (2, 0, 1))


def compute_frame_cooling_cg(frame, ds: int = DS, return_pdf: bool = False):
    """Compute coarse-grained cooling rate using subgrid temperature PDF method."""
    pdf = compute_frame_true_pdf(frame, ds=ds)
    rho_c = coarse_grain_2d(frame.density, ds=ds)
    n_cgs = rho_c / (MU * M_H)
    cooling = (n_cgs ** 2) * np.tensordot(LAMBDA_CENTERS, pdf, axes=(0, 0))
    if return_pdf:
        return cooling, pdf
    return cooling


def compute_frame_cooling_cnn(frame, return_pdf: bool = False, return_gate: bool = False):
    """Predict cooling rate (and optionally PDF and gate) using cached model & batched GPU tiles."""
    if model is None:
        raise ValueError("CNN model is not loaded. Please run load_model() first.")

    if not hasattr(model, "pdf_activation"):
        raise ValueError("Model does not have pdf_activation method.")
    # Resetting the Seed each time to ensure consistency in predictions
    np.random.seed(10)
    torch.manual_seed(10)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(10)

    rho_c = coarse_grain_2d(frame.density / frame.units._density, DS)
    temp_c = coarse_grain_2d(frame.temperature, DS)
    ux_c = coarse_grain_2d(frame.velx / frame.units._velocity, DS)
    uy_c = coarse_grain_2d(frame.vely / frame.units._velocity, DS)
    ps_key = list(frame.scalars.keys())[0] if frame.scalars else None
    ps_c = coarse_grain_2d(getattr(frame, ps_key) if ps_key else np.zeros_like(temp_c), DS)

    fields = [rho_c, temp_c, ux_c, uy_c, ps_c]
    ny_cg, nx_cg = rho_c.shape
    mid_r = ny_cg // 2
    mid_c = nx_cg // 2

    tiles = []
    for i in range(4):
        r = slice(0, mid_r) if i < 2 else slice(mid_r, ny_cg)
        c = slice(0, mid_c) if i % 2 == 0 else slice(mid_c, nx_cg)
        tiles.append(np.stack([f[r, c] for f in fields], axis=0))

    batch = torch.from_numpy(np.stack(tiles, axis=0)).float().to(device)
    with torch.no_grad():
        logits, gate = model((batch - input_mean) / input_std)
        pdf_tiles = model.pdf_activation(logits, gate).cpu().numpy()
        gate_tiles = gate.squeeze(1).cpu().numpy()

    top_p = np.concatenate([pdf_tiles[0], pdf_tiles[1]], axis=2)
    bot_p = np.concatenate([pdf_tiles[2], pdf_tiles[3]], axis=2)
    pdf = np.concatenate([top_p, bot_p], axis=1)

    top_g = np.concatenate([gate_tiles[0], gate_tiles[1]], axis=1)
    bot_g = np.concatenate([gate_tiles[2], gate_tiles[3]], axis=1)
    gate_map = np.concatenate([top_g, bot_g], axis=0)

    n_cgs = (rho_c * frame.units._density) / (MU * M_H)
    cooling = (n_cgs ** 2) * np.tensordot(LAMBDA_CENTERS, pdf, axes=(0, 0))

    if return_pdf and return_gate:
        return cooling, pdf, gate_map
    if return_pdf:
        return cooling, pdf
    if return_gate:
        return cooling, gate_map
    return cooling


# Subsample frames and prepare output directory
FRAME_STEP = 2
frame_indices = hr_sim.frame_numbers[::FRAME_STEP]
out_dir = PROJECT_ROOT / "explore_data" / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)

# # -----------------------------------------------------------------------------
# # 1. Cooling Rate Animation (.gif)
# # -----------------------------------------------------------------------------
frame_0 = hr_sim.get_frame(frame_indices[0])
cool_hr_0 = compute_frame_cooling(frame_0)
cool_cg_0 = compute_frame_cooling_cg(frame_0, DS)
cool_cnn_0 = compute_frame_cooling_cnn(frame_0)

# Compute dynamic percentile limits
sample_step = max(1, len(frame_indices) // 25)
sample_cool_pos = []
for f_idx in frame_indices[::sample_step]:
    f = hr_sim.get_frame(f_idx)
    for c in (compute_frame_cooling(f), compute_frame_cooling_cg(f, DS), compute_frame_cooling_cnn(f)):
        pos_c = c[c > 0]
        if pos_c.size > 0:
            sample_cool_pos.append(pos_c)

if sample_cool_pos:
    all_c = np.concatenate(sample_cool_pos)
    cool_vmin = max(np.percentile(all_c, 1), 1e-30)
    cool_vmax = np.percentile(all_c, 99)
else:
    cool_vmin, cool_vmax = 1e-28, 1e-23

cmap_cool = plt.get_cmap("viridis")
norm_cool = LogNorm(vmin=cool_vmin, vmax=cool_vmax)
extent = [frame_0.x.min() / CM_PER_PC, frame_0.x.max() / CM_PER_PC, frame_0.y.min() / CM_PER_PC, frame_0.y.max() / CM_PER_PC]

fig, (ax_cool_hr, ax_cool_cg, ax_cool_cnn) = plt.subplots(1, 3, figsize=(16, 5.5), sharex=True, sharey=True, constrained_layout=True)

im_cool_hr = ax_cool_hr.imshow(np.clip(cool_hr_0, cool_vmin, None), origin="lower", extent=extent, norm=norm_cool, cmap=cmap_cool, aspect="equal")
ax_cool_hr.set_title(r"HR Cooling: $n^2 \Lambda(T)$", fontsize=11, pad=6)
ax_cool_hr.set_xlabel("x (pc)", fontsize=11)
ax_cool_hr.set_ylabel("y (pc)", fontsize=11)

im_cool_cg = ax_cool_cg.imshow(np.clip(cool_cg_0, cool_vmin, None), origin="lower", extent=extent, norm=norm_cool, cmap=cmap_cool, aspect="equal")
ax_cool_cg.set_title(r"Coarse Grained Subgrid PDF Cooling: $n_{\mathrm{cg}}^2 \sum P_{\mathrm{true}}(T) \Lambda(T)$", fontsize=11, pad=6)
ax_cool_cg.set_xlabel("x (pc)", fontsize=11)

im_cool_cnn = ax_cool_cnn.imshow(np.clip(cool_cnn_0, cool_vmin, None), origin="lower", extent=extent, norm=norm_cool, cmap=cmap_cool, aspect="equal")
ax_cool_cnn.set_title(r"CNN Predicted Subgrid Cooling", fontsize=11, pad=6)
ax_cool_cnn.set_xlabel("x (pc)", fontsize=11)

cbar_cool = fig.colorbar(im_cool_cnn, ax=[ax_cool_hr, ax_cool_cg, ax_cool_cnn], orientation="vertical", fraction=0.03, pad=0.02)
cbar_cool.set_label(r"Cooling Rate ($\mathrm{erg\ cm^{-3}\ s^{-1}}$)", fontsize=10)
title_text = fig.suptitle(f"Time: {hr_sim.times[frame_indices[0]]:.2f} Myr", fontsize=14, fontweight="bold")

cool_gif_path = out_dir / "cooling_hr_vs_cg_vs_cnn.gif"
writer_cool = animation.FFMpegWriter(fps=15)

print(f"Rendering cooling GIF to {cool_gif_path}...")
with writer_cool.saving(fig, str(cool_gif_path), dpi=120):
    for f_idx in tqdm(frame_indices, desc="Cooling frames"):
        frame = hr_sim.get_frame(f_idx)
        im_cool_hr.set_data(np.clip(compute_frame_cooling(frame), cool_vmin, None))
        im_cool_cg.set_data(np.clip(compute_frame_cooling_cg(frame, DS), cool_vmin, None))
        im_cool_cnn.set_data(np.clip(compute_frame_cooling_cnn(frame), cool_vmin, None))
        title_text.set_text(f"Time: {hr_sim.times[f_idx]:.2f} Myr")
        writer_cool.grab_frame()

plt.close(fig)
print(f"Saved: {cool_gif_path}")

from matplotlib.collections import LineCollection, PolyCollection
# -----------------------------------------------------------------------------
# 2. True vs Predicted Subgrid Temperature PDF Comparison (.gif)
# -----------------------------------------------------------------------------
ny_cg, nx_cg = cool_cnn_0.shape  # (64, 32)

LO_LOG, HI_LOG = np.log10(1e-5), np.log10(1.1)
x_bins = np.arange(nb_bins)
x_frac = x_bins / (nb_bins - 1)                                   # bin position within a cell
CELL_X = np.arange(nx_cg)[None, :, None] + x_frac[None, None, :]  # static (ny, nx, nb)
CELL_Y0 = np.arange(ny_cg)[:, None, None].astype(float)

fig_pdf = plt.figure(figsize=(19, 13))
gs_pdf = fig_pdf.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], top=0.90, wspace=0.15)
fig_pdf.text(0.18, 0.92, f"HR TEMPERATURE & {ny_cg}x{nx_cg} GRID", fontsize=13, ha="center", weight="bold")
fig_pdf.text(0.46, 0.92, "TRUE SUBGRID PDFs (Simulation)", fontsize=13, ha="center", weight="bold")
fig_pdf.text(0.74, 0.92, "PREDICTED SUBGRID PDFs (CNN Model)", fontsize=13, ha="center", weight="bold")

# HR Temperature Panel with 32x16 grid overlay
ax_hr_t = fig_pdf.add_subplot(gs_pdf[0])
ax_hr_t.set_xlim(0, nx_cg)
ax_hr_t.set_ylim(0, ny_cg)
ax_hr_t.set_xticks(np.arange(0, nx_cg + 1, 1))
ax_hr_t.set_yticks(np.arange(0, ny_cg + 1, 1))
ax_hr_t.grid(color="white", linestyle="--", linewidth=0.5, alpha=0.6)
ax_hr_t.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
for spine in ax_hr_t.spines.values():
    spine.set_color("grey")
    spine.set_linewidth(0.3)

im_hr_t = ax_hr_t.imshow(
    np.log10(frame_0.temperature),
    origin="lower",
    extent=(0, nx_cg, 0, ny_cg),
    cmap=cmap_temp_exp,
    norm=norm_temp_exp,
    aspect="auto",
    interpolation="nearest",
)

# Static active-temperature band vertices (one quad per cell)
bx0, bx1 = active_bin_start / (nb_bins - 1), active_bin_end / (nb_bins - 1)
band_verts = [[(j + bx0, i), (j + bx1, i), (j + bx1, i + 1), (j + bx0, i + 1)]
              for i in range(ny_cg) for j in range(nx_cg)]

def make_panel(ax):
    ax.set_xlim(0, nx_cg); ax.set_ylim(0, ny_cg)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("grey"); spine.set_linewidth(0.3)
    ax.add_collection(PolyCollection(band_verts, closed=True, facecolors="green",
                                     alpha=0.12, edgecolors="none"))
    im = ax.imshow(np.zeros((ny_cg, nx_cg, 4)), origin="lower",
                   extent=(0, nx_cg, 0, ny_cg), interpolation="nearest", aspect="auto")
    lc = LineCollection(np.zeros((ny_cg * nx_cg, nb_bins, 2)), linewidths=1.0)
    ax.add_collection(lc)
    return im, lc

true_im, true_lc = make_panel(fig_pdf.add_subplot(gs_pdf[1]))
pred_im, pred_lc = make_panel(fig_pdf.add_subplot(gs_pdf[2]))

cbar_ax_pdf = fig_pdf.add_subplot(gs_pdf[3])
sm_pdf = plt.cm.ScalarMappable(cmap=cmap_temp_exp, norm=norm_temp_exp)
sm_pdf.set_array([])
cbar_pdf = fig_pdf.colorbar(sm_pdf, cax=cbar_ax_pdf)
cbar_pdf.set_label(r"$\log_{10}(T\ [\mathrm{K}])$ / Expectation Value of $\log_{10}(T)$", fontsize=12)


def _panel_arrays(pdf):
    """pdf: (nb_bins, ny, nx) -> bg RGBA, curve segments, line colors."""
    p = np.moveaxis(pdf, 0, -1)                                  # (ny, nx, nb)
    rgba = cmap_temp_exp(norm_temp_exp(p @ log_temp_centers))    # (ny, nx, 4)
    # log-scaled y within each cell (replaces the axes' log scale)
    yn = np.clip((np.log10(np.clip(p, 1e-8, None)) - LO_LOG) / (HI_LOG - LO_LOG), 0.0, 1.0)
    segs = np.stack([np.broadcast_to(CELL_X, yn.shape), CELL_Y0 + yn], axis=-1)
    segs = segs.reshape(ny_cg * nx_cg, nb_bins, 2)
    lum = 0.299 * rgba[..., 0] + 0.587 * rgba[..., 1] + 0.114 * rgba[..., 2]
    col = np.repeat(np.where(lum[..., None] < 0.5, 1.0, 0.0), 3, axis=-1)
    colors = np.concatenate([col, np.ones((*lum.shape, 1))], axis=-1).reshape(-1, 4)
    return rgba, segs, colors


def update_pdf_plot(true_pdf, pred_pdf, temp_field=None):
    if temp_field is not None:
        im_hr_t.set_data(np.log10(temp_field))
    for im, lc, pdf in ((true_im, true_lc, true_pdf), (pred_im, pred_lc, pred_pdf)):
        rgba, segs, colors = _panel_arrays(pdf)
        im.set_data(rgba)
        lc.set_segments(segs)
        lc.set_colors(colors)

# Save initial snapshot
cool_0, pred_pdf_0 = compute_frame_cooling_cnn(frame_0, return_pdf=True)
true_pdf_0 = compute_frame_true_pdf(frame_0, DS)
update_pdf_plot(true_pdf_0, pred_pdf_0, frame_0.temperature)
pdf_title = fig_pdf.suptitle(f"Subgrid Temperature PDFs & HR Temperature Field | Time: {hr_sim.times[frame_indices[0]]:.2f} Myr", fontsize=16, y=0.96, fontweight="bold")

pdf_snapshot_path = out_dir / "subgrid_temperature_pdf_compare_t0.png"
fig_pdf.savefig(str(pdf_snapshot_path), dpi=150)
print(f"Saved: {pdf_snapshot_path}")

# Render PDF comparison animation (.gif)
pdf_gif_path = out_dir / "subgrid_temperature_pdf_compare.gif"
writer_pdf = animation.FFMpegWriter(fps=15)

print(f"Rendering PDF comparison GIF to {pdf_gif_path}...")
with writer_pdf.saving(fig_pdf, str(pdf_gif_path), dpi=80):
    for f_idx in tqdm(frame_indices, desc="PDF frames"):
        frame = hr_sim.get_frame(f_idx)
        _, p_pdf = compute_frame_cooling_cnn(frame, return_pdf=True)
        t_pdf = compute_frame_true_pdf(frame, DS)
        update_pdf_plot(t_pdf, p_pdf, frame.temperature)
        pdf_title.set_text(f"Subgrid Temperature PDFs & HR Temperature Field | Time: {hr_sim.times[f_idx]:.2f} Myr")
        writer_pdf.grab_frame()

plt.close(fig_pdf)
print(f"Saved: {pdf_gif_path}")


# -----------------------------------------------------------------------------
# 3. CNN Gate vs Active-Zone PDF Sum Comparison (.gif)
# -----------------------------------------------------------------------------
LOGT_LOW = LOGT_ACTIVE_START
LOGT_HIGH = LOGT_ACTIVE_END

# Find bins corresponding to active cooling window [LOGT_LOW, LOGT_HIGH]
active_bin_mask = (log_temp_centers >= LOGT_LOW) & (log_temp_centers <= LOGT_HIGH)

def compute_active_pdf_sum(pdf: np.ndarray) -> np.ndarray:
    """Sum of PDF values in the active temperature range: (ny_cg, nx_cg)."""
    return np.sum(pdf[active_bin_mask], axis=0)

extent = [
    frame_0.x.min() / CM_PER_PC,
    frame_0.x.max() / CM_PER_PC,
    frame_0.y.min() / CM_PER_PC,
    frame_0.y.max() / CM_PER_PC,
]

fig_gate, (ax_gate, ax_true_sum, ax_pred_sum) = plt.subplots(
    1, 3, figsize=(16, 6), sharex=True, sharey=True, constrained_layout=True
)

_, _, gate_0 = compute_frame_cooling_cnn(frame_0, return_pdf=True, return_gate=True)
true_active_sum_0 = compute_active_pdf_sum(true_pdf_0)
pred_active_sum_0 = compute_active_pdf_sum(pred_pdf_0)

cmap_gate = plt.get_cmap("viridis")
im_gate = ax_gate.imshow(
    gate_0, origin="lower", extent=extent, cmap=cmap_gate, vmin=0.0, vmax=1.0, aspect="equal", interpolation="nearest"
)
ax_gate.set_title(r"CNN Predicted Gate: $g(x,y) \in [0, 1]$", fontsize=11, pad=6)
ax_gate.set_xlabel("x (pc)", fontsize=11)
ax_gate.set_ylabel("y (pc)", fontsize=11)

im_true_sum = ax_true_sum.imshow(
    true_active_sum_0, origin="lower", extent=extent, cmap=cmap_gate, vmin=0.0, vmax=1.0, aspect="equal", interpolation="nearest"
)
ax_true_sum.set_title(rf"True Active PDF Sum: $\sum_{{T \in \mathrm{{active}}}} \mathrm{{PDF}}_{{\mathrm{{true}}}}$", fontsize=11, pad=6)
ax_true_sum.set_xlabel("x (pc)", fontsize=11)

im_pred_sum = ax_pred_sum.imshow(
    pred_active_sum_0, origin="lower", extent=extent, cmap=cmap_gate, vmin=0.0, vmax=1.0, aspect="equal", interpolation="nearest"
)
ax_pred_sum.set_title(rf"Pred Active PDF Sum: $\sum_{{T \in \mathrm{{active}}}} \mathrm{{PDF}}_{{\mathrm{{pred}}}}$", fontsize=11, pad=6)
ax_pred_sum.set_xlabel("x (pc)", fontsize=11)

cbar_gate = fig_gate.colorbar(im_gate, ax=[ax_gate, ax_true_sum, ax_pred_sum], orientation="vertical", fraction=0.03, pad=0.02)
cbar_gate.set_label(r"Value / Probability $\in [0, 1]$", fontsize=10)

gate_title = fig_gate.suptitle(
    f"CNN Gate vs Active Zone PDF Sum | Time: {hr_sim.times[frame_indices[0]]:.2f} Myr",
    fontsize=14,
    fontweight="bold",
)

gate_snapshot_path = out_dir / "gate_vs_active_pdf_compare_t0.png"
fig_gate.savefig(str(gate_snapshot_path), dpi=150)
print(f"Saved: {gate_snapshot_path}")

gate_gif_path = out_dir / "gate_vs_active_pdf_compare.gif"
writer_gate = animation.FFMpegWriter(fps=15)

print(f"Rendering Gate vs Active Zone PDF comparison GIF to {gate_gif_path}...")
with writer_gate.saving(fig_gate, str(gate_gif_path), dpi=120):
    for f_idx in tqdm(frame_indices, desc="Gate comparison frames"):
        frame = hr_sim.get_frame(f_idx)
        _, p_pdf, g_map = compute_frame_cooling_cnn(frame, return_pdf=True, return_gate=True)
        t_pdf = compute_frame_true_pdf(frame, DS)
        
        im_gate.set_data(g_map)
        im_true_sum.set_data(compute_active_pdf_sum(t_pdf))
        im_pred_sum.set_data(compute_active_pdf_sum(p_pdf))
        gate_title.set_text(f"CNN Gate vs Active Zone PDF Sum | Time: {hr_sim.times[f_idx]:.2f} Myr")
        writer_gate.grab_frame()

plt.close(fig_gate)
print(f"Saved: {gate_gif_path}")


