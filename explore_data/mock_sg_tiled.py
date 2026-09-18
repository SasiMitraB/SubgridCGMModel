#!/usr/bin/env python3
"""
mock_sg_tiled.py
===============================================================================
Diagnostic, comparison, and animation suite for the 32x16 tiled CNN setup:
  - HR (512x1024)
  - CG HR (coarse-grained 32x to 32x16)
  - SG (Subgrid CNN model, 32x16, with batched 4-tile inference)
  - LR (Standard ISM cooling, 32x16)

Reproduces the full suite of diagnostic plots and animations from mock_sg.py:
  1. 1D X-averaged vertical profiles (mean ± 1σ across time):
     - Density, Temperature, Pressure, Ux, Uy (primitive profiles)
     - Conserved quantities: Cons Density, Cons MomX, Cons MomY, Cons Energy, Passive Scalar, fmcl
     - Derived quantities: rho*ux, rho*ux*uy, p + rho*uy^2
     - Energy and momentum fluxes: Mass Flux X/Y, T_xx, T_xy, T_yy, E_flux_x, E_flux_y
     - Divergence of fluxes: Div Mass Flux, Div MomX Flux, Div MomY Flux
  2. Volume / Mass / Emissivity weighted temperature PDFs (mean ± 1σ)
  3. Mean Volumetric Cooling Rate profile vs y (<n_H^2 Lambda(T)>) & integrated Sigma_c
  4. Cold gas mass evolution (T < 1e5 K) & linear accretion fits
  5. Static 2D Snapshots:
     - All 6 physical fields side-by-side snapshot
     - 2D Cooling rate comparison snapshot
     - 5-panel subgrid predicted PDF + T + Cooling + Gate + Active Mass snapshot
  6. MP4 Animations (using ffmpeg):
     - temperature_field_evolution.mp4 (HR, CG HR, SG, LR with velocity streamlines)
     - cons_fields_evolution.mp4
     - density_evolution.mp4
     - temperature_pdf_evolution.mp4
     - cooling_rate_evolution.mp4
     - subgrid_predicted_pdf_evolution.mp4 (tiled PDF mini-plots grid, T, Cooling, Gate, Active Mass)
===============================================================================
"""

import os
import sys
import gc
import subprocess
import argparse
import csv
from pathlib import Path
from functools import partial

from scipy.stats import pearsonr

# Add project root and conv_nn to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "conv_nn"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm
import torch

import ergane
from pdf_cnn import (
    ConvNN,
    in_channels,
    layer_size1,
    layer_size2,
    layer_size3,
    layer_size4,
    out_channels,
    device,
    lambda_cool,
    T_centers,
    T_edges,
)

# ── Physical constants & Code Units ──────────────────────────────────────────
CM_PER_PC       = 3.08568e18          # cm per parsec
CM_PER_KM       = 1.0e5               # cm per km
SECONDS_PER_MYR = 3.15576e13          # seconds per Myr
M_H             = 1.6726219e-24       # proton mass [g]
K_B             = 1.380649e-16        # Boltzmann constant in erg/K
M_SUN           = 1.98847e33          # Solar mass in g
YR              = 3.15576e7           # Year in seconds
PC              = 3.08568e18          # Parsec in cm
KPC             = 3.08568e21          # Kiloparsec in cm
MU              = 0.62                # mean molecular weight
GAMMA           = 5.0 / 3.0           # adiabatic index

# Code units
L_cgs = 3.08568e18                    # 1 pc
M_cgs = 4.91417e31
T_cgs = 3.15576e13                    # 1 Myr
V_cgs = L_cgs / T_cgs                 # cm/s (~9.778 km/s)
RHO_cgs = M_cgs / (L_cgs ** 3)        # g/cm^3
P_cgs = RHO_cgs * (V_cgs ** 2)        # dyn/cm^2 or erg/cm^3

len_to_pc = 1.0
n_to_cm3 = RHO_cgs / (MU * M_H)
T_to_K = (V_cgs ** 2) * MU * M_H / K_B
P_over_kB_to_K_cm3 = P_cgs / K_B
vel_to_km_s = V_cgs / 1e5
mflux_to_Msun_yr_kpc2 = (RHO_cgs * V_cgs) / (M_SUN / (YR * (KPC ** 2)))
unit_fix = 1.975e27

LOGT_ACTIVE_START = float(os.environ.get("LOGT_ACTIVE_START", "4.1"))
LOGT_ACTIVE_END   = float(os.environ.get("LOGT_ACTIVE_END", "5.9"))
LAMBDA_CENTERS = lambda_cool(T_centers, mask=True, LOGT_ACTIVE_START=LOGT_ACTIVE_START, LOGT_ACTIVE_END=LOGT_ACTIVE_END)
# Isobaric per-bin weight: Lambda(T_i) / T_i^2, used with n_i = P/(kB T_i)
ISOBARIC_WEIGHT = LAMBDA_CENTERS / T_centers**2

RESTART_TIME_MYR = float(os.environ.get("RESTART_TIME_MYR", "5.0"))
BIN_DT_MYR       = 0.01

CELL_SIZE = float(os.environ.get("CELL_SIZE", "1.25"))
if np.isclose(CELL_SIZE, 0.625) or os.environ.get("PDF_CNN_DOWNSAMPLE") == "16" or os.environ.get("DS") == "16":
    DS = 16
    CELL_SIZE_PC = 0.625
else:
    DS = 32
    CELL_SIZE_PC = 1.25

# Optional environment overrides for resolution, otherwise determined dynamically from simulation data
_ENV_NX2 = os.environ.get("NX2", "")
_ENV_NX1 = os.environ.get("NX1", "")
if _ENV_NX2 and _ENV_NX1:
    RESOLUTION = (int(_ENV_NX2), int(_ENV_NX1))
else:
    RESOLUTION = (64, 32) if DS == 16 else (32, 16)

CELL_LABEL = f"{CELL_SIZE_PC:g} pc"
CG_LABEL = f"CG HR ({CELL_LABEL})"
SG_LABEL = f"SG ({CELL_LABEL})"
LR_LABEL = f"LR ({CELL_LABEL})"

FRAME_STEP = 2  # Subsample every Nth frame for fast animation generation

# Default paths
DEFAULT_HR_ATHINPUT = PROJECT_ROOT / "simulation_outputs/hr_gpu_512x1024/kh_radiative_512x1024.athinput"
DEFAULT_HR_BIN_DIR  = PROJECT_ROOT / "simulation_outputs/hr_gpu_512x1024/bin"
DEFAULT_SG_ATHINPUT = PROJECT_ROOT / f"simulation_outputs/athinputs_{RESOLUTION[0]}x{RESOLUTION[1]}/subgrid_{RESOLUTION[0]}x{RESOLUTION[1]}.athinput"
DEFAULT_SG_BIN_DIR  = PROJECT_ROOT / f"simulation_outputs/subgrid_{RESOLUTION[0]}x{RESOLUTION[1]}_from_snap500/bin"
DEFAULT_LR_ATHINPUT = PROJECT_ROOT / f"simulation_outputs/athinputs_{RESOLUTION[0]}x{RESOLUTION[1]}/hr_build_{RESOLUTION[0]}x{RESOLUTION[1]}.athinput"
DEFAULT_LR_BIN_DIR  = PROJECT_ROOT / f"simulation_outputs/hr_build_{RESOLUTION[0]}x{RESOLUTION[1]}_from_snap500/bin"
DEFAULT_OUT_DIR     = PROJECT_ROOT / "explore_data/outputs/mock_sg_tiled"


def coarse_grain_2d(arr: np.ndarray, ds: int = DS) -> np.ndarray:
    """Coarse-grain a 2D array of shape (Ny, Nx) by factor ds."""
    ny, nx = arr.shape
    if ds <= 1 or ny < ds or nx < ds:
        return arr.copy()
    ny_cg, nx_cg = ny // ds, nx // ds
    return arr[:ny_cg * ds, :nx_cg * ds].reshape(ny_cg, ds, nx_cg, ds).mean(axis=(1, 3))


def compute_volumetric_cooling(rho_cgs: np.ndarray, temp_K: np.ndarray) -> np.ndarray:
    """Compute volumetric cooling rate in CGS [erg cm^-3 s^-1]."""
    n_H = rho_cgs / (MU * M_H)
    lam = lambda_cool(temp_K, mask=True, LOGT_ACTIVE_START=LOGT_ACTIVE_START, LOGT_ACTIVE_END=LOGT_ACTIVE_END)
    return (n_H ** 2) * lam


def compute_color_limits(arr, use_log=False):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0, None
    if use_log:
        positive = finite[finite > 0]
        if positive.size:
            vmin = positive.min()
            vmax = positive.max()
            if vmax <= vmin:
                vmax = vmin * 1.01
            return vmin, vmax, LogNorm(vmin=vmin, vmax=vmax)
    vmin = finite.min()
    vmax = finite.max()
    if vmax <= vmin:
        delta = abs(vmin) * 0.01 if vmin != 0 else 1.0
        vmin -= delta / 2
        vmax += delta / 2
    return vmin, vmax, None


_VIDEO_CODEC = None


def get_video_codec():
    """Detect once whether ffmpeg has NVENC hardware encoding; fall back to mpeg4 otherwise."""
    global _VIDEO_CODEC
    if _VIDEO_CODEC is None:
        try:
            res = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=10)
            if "h264_nvenc" in res.stdout:
                _VIDEO_CODEC = ("h264_nvenc", ["-preset", "p4", "-pix_fmt", "yuv420p"])
            else:
                _VIDEO_CODEC = ("mpeg4", ["-q:v", "2", "-pix_fmt", "yuv420p"])
        except Exception:
            _VIDEO_CODEC = ("mpeg4", ["-q:v", "2", "-pix_fmt", "yuv420p"])
    return _VIDEO_CODEC


def save_animation_funcanim(fig, update_func, frames, output_path, fps=10, blit=False):
    """Render an animation with FuncAnimation, piping frames straight into ffmpeg (no PNG round-trip)."""
    vf_filter = "scale='min(4096,iw)':'min(4096,ih)':force_original_aspect_ratio=decrease,pad=ceil(iw/2)*2:ceil(ih/2)*2"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    codec, codec_args = get_video_codec()
    anim = FuncAnimation(fig, update_func, frames=tqdm(list(frames), desc=f"Video {Path(output_path).name}"), blit=blit)
    try:
        writer = FFMpegWriter(fps=fps, codec=codec, extra_args=codec_args + ["-vf", vf_filter])
        anim.save(str(output_path), writer=writer, dpi=100)
    except Exception as e:
        if codec != "mpeg4":
            print(f"  Warning: {codec} failed ({e}); retrying with mpeg4")
            writer = FFMpegWriter(fps=fps, codec="mpeg4", extra_args=["-q:v", "2", "-pix_fmt", "yuv420p", "-vf", vf_filter])
            anim.save(str(output_path), writer=writer, dpi=100)
        else:
            raise
    plt.close(fig)
    print(f"  Saved video: {output_path}")


def pdf_mass_in_active_range(pdf, T_edges, logt_start=LOGT_ACTIVE_START, logt_end=LOGT_ACTIVE_END):
    """Return the PDF mass between logt_start and logt_end."""
    logt_edges = np.log10(T_edges)
    logt_centers = 0.5 * (logt_edges[:-1] + logt_edges[1:])
    mask = (logt_centers >= logt_start) & (logt_centers <= logt_end)
    return np.sum(pdf[mask], axis=0)


def compute_mean_std(arr, logspace=False):
    if logspace:
        arr = np.log10(np.maximum(arr, 1e-30))
    if arr.ndim == 3:
        arr_1d = np.mean(arr, axis=2)  # average over X
    else:
        arr_1d = arr
    mean = np.mean(arr_1d, axis=0)
    std = np.std(arr_1d, axis=0) if arr_1d.shape[0] > 1 else np.zeros(arr_1d.shape[1])
    return mean, std


def divergence(f, dx, dy):
    dFx_dx = np.gradient(f[0], dy, dx)[1]
    dFy_dy = np.gradient(f[1], dy, dx)[0]
    return dFx_dx + dFy_dy


def trapz_integral(y_vals: np.ndarray, x_coords: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y_vals, x_coords))
    return float(np.trapz(y_vals, x_coords))


def compute_profile_metrics(ref_profile, target_profile):
    """Compute Pearson correlation and RMSD between two 1D profiles."""
    ref = np.asarray(ref_profile).ravel()
    tar = np.asarray(target_profile).ravel()
    rmsd = float(np.sqrt(np.mean((ref - tar) ** 2)))
    if np.all(ref == ref[0]) or np.all(tar == tar[0]) or np.isnan(ref).any() or np.isnan(tar).any():
        r = float("nan")
    else:
        r_val, _ = pearsonr(ref, tar)
        r = float(r_val)
    return r, rmsd


def setup_tiled_pdf_panel(ax, ny_cg=32, nx_cg=16, nb_bins=40,
                          logt_start=LOGT_ACTIVE_START, logt_end=LOGT_ACTIVE_END,
                          t_edges=T_edges):
    """Setup a single-Axes fast PDF grid using LineCollection and PolyCollection."""
    ax.set_xlim(0, nx_cg)
    ax.set_ylim(0, ny_cg)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("grey")
        spine.set_linewidth(0.4)

    log_centers = 0.5 * (np.log10(t_edges[:-1]) + np.log10(t_edges[1:]))
    b_start = int(np.searchsorted(log_centers, logt_start))
    b_end = int(np.searchsorted(log_centers, logt_end))
    bx0 = b_start / (nb_bins - 1)
    bx1 = b_end / (nb_bins - 1)

    band_verts = [
        [(j + bx0, i), (j + bx1, i), (j + bx1, i + 1), (j + bx0, i + 1)]
        for i in range(ny_cg) for j in range(nx_cg)
    ]
    ax.add_collection(PolyCollection(band_verts, closed=True, facecolors="green",
                                     alpha=0.18, edgecolors="none"))
    bg_im = ax.imshow(np.zeros((ny_cg, nx_cg, 4)), origin="lower",
                      extent=(0, nx_cg, 0, ny_cg), interpolation="nearest", aspect="auto")
    lc = LineCollection(np.zeros((ny_cg * nx_cg, nb_bins, 2)), linewidths=1.0)
    ax.add_collection(lc)
    return bg_im, lc


def compute_pdf_panel_arrays(pdf, cmap_temp, norm_temp, log_temp_centers,
                             ny_cg=32, nx_cg=16, nb_bins=40):
    """
    Vectorized conversion of pdf (nb_bins, ny_cg, nx_cg) into:
      rgba: (ny_cg, nx_cg, 4) background cell colors based on expected log10(T)
      segs: (ny_cg * nx_cg, nb_bins, 2) normalized curve coordinates per cell
      colors: (ny_cg * nx_cg, 4) line colors (white/black based on luminance)
    """
    # p: (ny_cg, nx_cg, nb_bins)
    p = np.moveaxis(pdf, 0, -1)
    exp_logt = p @ log_temp_centers
    rgba = cmap_temp(norm_temp(exp_logt))

    # Normalized height within each cell [0, 1] matching single ax.plot(x, y/max(y))
    p_max = np.maximum(p.max(axis=-1, keepdims=True), 1e-12)
    yn = 0.95 * (p / p_max)

    x_frac = np.arange(nb_bins) / (nb_bins - 1)
    cell_x = np.arange(nx_cg)[None, :, None] + x_frac[None, None, :]
    cell_y0 = np.arange(ny_cg)[:, None, None].astype(float)

    segs = np.stack([np.broadcast_to(cell_x, yn.shape), cell_y0 + yn], axis=-1)
    segs = segs.reshape(ny_cg * nx_cg, nb_bins, 2)

    lum = 0.299 * rgba[..., 0] + 0.587 * rgba[..., 1] + 0.114 * rgba[..., 2]
    col = np.repeat(np.where(lum[..., None] < 0.5, 1.0, 0.0), 3, axis=-1)
    colors = np.concatenate([col, np.ones((*lum.shape, 1))], axis=-1).reshape(-1, 4)
    return rgba, segs, colors


def load_history_file(history_path):
    """Load timestep data from Athena++ history file.

    History files are opened in append mode by Athena, so re-running the same
    simulation multiple times leaves earlier runs' rows in the file, each
    restarting from t=0. Keep only the rows from the last (most recent) run.

    Newer history files also carry dt_cfl (columns [2]) and dt_cool ([3]) --
    the per-component candidate timesteps that Mesh::NewTimeStep() combines
    into dt (see codebase_notes/dynamic_dt_calculation.md). Older history
    files without these columns still load fine; dt_cfl/dt_cool come back as
    None in that case.
    """
    # Check the header comment for the real column names -- every hydro .hst
    # file has >=4 columns (time, dt, mass, 1-mom, ...), so a bare column-count
    # check would silently misread "mass"/"1-mom" as dt_cfl/dt_cool for older
    # history files that predate those columns.
    # History files opened in append mode can contain several header blocks
    # (one per re-run), each possibly with a different column layout -- e.g.
    # older runs predating the dt_cfl/dt_cool columns, followed by a newer
    # run that has them. Scan the whole file and keep the verdict from the
    # *last* header block, since that's the one describing the data that
    # survives the reset-trimming below.
    has_components = False
    in_header_block = False
    block_has_components = False
    try:
        with open(history_path) as f:
            for line in f:
                if line.startswith('#'):
                    if not in_header_block:
                        in_header_block = True
                        block_has_components = False
                    if 'dt_cfl' in line and 'dt_cool' in line:
                        block_has_components = True
                elif in_header_block:
                    has_components = block_has_components
                    in_header_block = False
            if in_header_block:
                has_components = block_has_components
    except Exception as e:
        print(f"  Warning: Could not read header of {history_path}: {e}")
        return None, None, None, None

    try:
        cols = (0, 1, 2, 3) if has_components else (0, 1)
        data = np.loadtxt(history_path, comments='#', usecols=cols)
    except Exception as e:
        print(f"  Warning: Could not load history file {history_path}: {e}")
        return None, None, None, None

    if data.ndim == 1:
        data = data.reshape(1, -1)
    times = data[:, 0]
    dts = data[:, 1]
    dt_cfl = data[:, 2] if has_components else None
    dt_cool = data[:, 3] if has_components else None

    # Detect resets (time jumping backward) and keep only the last segment
    resets = np.where(np.diff(times) < 0)[0]
    if len(resets) > 0:
        last_reset = resets[-1] + 1
        times = times[last_reset:]
        dts = dts[last_reset:]
        if has_components:
            dt_cfl = dt_cfl[last_reset:]
            dt_cool = dt_cool[last_reset:]

    return times, dts, dt_cfl, dt_cool


def load_clip_log(path):
    """Load the cooling-rate clip-event CSV written by source_module.py.

    source_module.py's live source_func() logs one row per cell where the
    temperature-floor cap actually engaged (time, grid position, physical
    position, and local rho/temp/cool_rate/cool_max). Returns a dict of
    numpy arrays keyed by column name, or None if no log file exists yet
    (e.g. this run never invoked the subgrid CNN source term).
    """
    if not os.path.isfile(path):
        return None
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = list(reader)
    if not header or not rows:
        return None
    data = np.array(rows, dtype=np.float64)
    return {name: data[:, i] for i, name in enumerate(header)}


def plot_clip_diagnostics(clip_data, out_dir):
    """Scatter of clipped-cell positions (colored by time) + a time histogram.

    Lets a clip event be cross-referenced directly against the logged
    rho/temp/cool_rate/cool_max at that cell without re-opening snapshots.
    """
    n_events = clip_data["time"].size
    cells = set(zip(clip_data["row"].astype(int), clip_data["col"].astype(int)))
    t_min, t_max = clip_data["time"].min(), clip_data["time"].max()
    print(f"  {n_events} clip events across {len(cells)} unique cells, "
          f"t = [{t_min:.3f}, {t_max:.3f}] Myr")

    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(16, 6))

    sc = ax_scatter.scatter(clip_data["x1_pc"], clip_data["x2_pc"], c=clip_data["time"],
                             cmap="viridis", s=12, alpha=0.7)
    ax_scatter.set_xlabel(r"$x_1 \ [\mathrm{pc}]$", fontsize=12)
    ax_scatter.set_ylabel(r"$x_2 \ [\mathrm{pc}]$", fontsize=12)
    ax_scatter.set_title(f"Clip Event Positions ({n_events} events)", fontsize=13, weight="bold")
    ax_scatter.grid(True, ls="--", alpha=0.4)
    plt.colorbar(sc, ax=ax_scatter, label="time [Myr]")

    ax_hist.hist(clip_data["time"], bins=min(50, max(1, n_events)), color="tab:red", alpha=0.75)
    ax_hist.set_xlabel("time [Myr]", fontsize=12)
    ax_hist.set_ylabel("clip events", fontsize=12)
    ax_hist.set_title("Clip Events vs Time", fontsize=13, weight="bold")
    ax_hist.grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_dir / "clip_diagnostics.png", dpi=200)
    plt.close(fig)
    print("  Saved clip_diagnostics.png")


def load_tiled_cnn_model(save_dir=None):
    """Load cached CNN model with batched 16x8 tiling weights."""
    if save_dir is None:
        save_dir = os.environ.get(
            "MODEL_SAVES_DIR",
            str(PROJECT_ROOT / "runs/run_random_crop_20260904_191402/model_saves")
        )
    norm_prefix = os.environ.get("NORM_PREFIX", "")
    if not norm_prefix:
        for cand in ["cnn_(512, 256)_32", "cnn_(1024, 512)_32", "cnn_(1024, 512)_64", "cnn_(2048, 1024)_32"]:
            if os.path.isfile(os.path.join(save_dir, f"{cand}.pth")):
                norm_prefix = cand
                break
    mean_file = os.path.join(save_dir, f"{norm_prefix}_input_mean.npy")
    std_file  = os.path.join(save_dir, f"{norm_prefix}_input_std.npy")
    pth_file  = os.path.join(save_dir, f"{norm_prefix}.pth")

    if not (os.path.isfile(pth_file) and os.path.isfile(mean_file) and os.path.isfile(std_file)):
        raise FileNotFoundError(f"Missing CNN weights or normalization with prefix '{norm_prefix}' at {save_dir}")

    input_mean = torch.tensor(np.load(mean_file), dtype=torch.float32, device=device).view(1, -1, 1, 1)
    input_std  = torch.tensor(np.load(std_file), dtype=torch.float32, device=device).view(1, -1, 1, 1)

    state_dict = torch.load(pth_file, map_location=device)
    ksize = state_dict["encoder.0.weight"].shape[-1]
    model = ConvNN(in_channels, layer_size1, layer_size2, layer_size3, layer_size4, out_channels, ksize).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, input_mean, input_std


def predict_tiled_subgrid_pdf_and_cooling(rho_code, temp_K, ux_code, uy_code, ps_val, pres_code, model, input_mean, input_std):
    """
    Run tiled batched inference on (H, W) fields (default 4x4 grid).

    Cooling uses the isobaric per-bin density n_i = P/(kB T_i) instead of a
    single constant n across the PDF:
        Emissivity = (P/kB)^2 x sum_i PDF(T_i) x Lambda(T_i) / T_i^2

    Returns:
      cool_cgs:  (H, W) [erg cm^-3 s^-1]
      pdf:       (40, H, W)
      gate_map:  (H, W)
    """
    fields = [rho_code, temp_K, ux_code, uy_code, ps_val]
    H, W = rho_code.shape

    # Determine tile layout: default to 16x8 cells per physical 20x10 pc tile (or 32x16 if DS=16)
    grid_env = os.environ.get("TILE_GRID", "").strip()
    if grid_env:
        parts = grid_env.split(",")
        n_tile_rows = int(parts[0].strip())
        n_tile_cols = int(parts[1].strip()) if len(parts) > 1 else n_tile_rows
    else:
        base_h = 32 if DS == 16 else 16
        base_w = 16 if DS == 16 else 8
        n_tile_rows = max(1, H // base_h)
        n_tile_cols = max(1, W // base_w)

    tile_h = H // n_tile_rows
    tile_w = W // n_tile_cols

    tiles = []
    coords = []
    for ti in range(n_tile_rows):
        for tj in range(n_tile_cols):
            r0, r1 = ti * tile_h, (ti + 1) * tile_h
            c0, c1 = tj * tile_w, (tj + 1) * tile_w
            tiles.append(np.stack([f[r0:r1, c0:c1] for f in fields], axis=0))
            coords.append((r0, r1, c0, c1))

    batch = torch.from_numpy(np.stack(tiles, axis=0)).float().to(device)
    with torch.no_grad():
        logits, gate = model((batch - input_mean) / input_std)
        pdf_tiles = model.pdf_activation(logits, gate).cpu().numpy()
        gate_tiles = gate.squeeze(1).cpu().numpy()

    pdf = np.zeros((out_channels, H, W), dtype=np.float32)
    gate_map = np.zeros((H, W), dtype=np.float32)
    for idx, (r0, r1, c0, c1) in enumerate(coords):
        pdf[:, r0:r1, c0:c1] = pdf_tiles[idx]
        gate_map[r0:r1, c0:c1] = gate_tiles[idx]

    p_over_kB = pres_code * P_over_kB_to_K_cm3  # physical P/kB [K cm^-3]
    cooling = (p_over_kB ** 2) * np.tensordot(ISOBARIC_WEIGHT, pdf, axes=(0, 0))

    return cooling, pdf, gate_map


def main():
    parser = argparse.ArgumentParser(description="Full mock_sg comparison & animation suite for 32x16 tiled setup")
    parser.add_argument("--output-dir", default=os.environ.get("MOCK_OUTPUT_DIR", str(DEFAULT_OUT_DIR)), help="Directory to save plots and videos")
    parser.add_argument("--hr-athinput", default=os.environ.get("HR_ATHINPUT", str(DEFAULT_HR_ATHINPUT)))
    parser.add_argument("--hr-bin", default=os.environ.get("HR_BIN_DIR", str(DEFAULT_HR_BIN_DIR)))
    parser.add_argument("--sg-athinput", default=os.environ.get("SG_ATHINPUT", str(DEFAULT_SG_ATHINPUT)))
    parser.add_argument("--sg-bin", default=os.environ.get("SG_BIN_DIR", str(DEFAULT_SG_BIN_DIR)))
    parser.add_argument("--lr-athinput", default=os.environ.get("LR_ATHINPUT", str(DEFAULT_LR_ATHINPUT)))
    parser.add_argument("--lr-bin", default=os.environ.get("LR_BIN_DIR", str(DEFAULT_LR_BIN_DIR)))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(f" MOCK SG TILED: Full Diagnostics & Animation Suite ({CELL_LABEL})")
    print(f" Output directory: {out_dir}")
    print("=" * 75)

    # 1. Load SimulationData via ergane
    print("\n[1] Initializing SimulationData objects via ergane...")
    sim_hr = ergane.SimulationData(athinp=str(args.hr_athinput), datafolder=str(args.hr_bin))
    print(f"  HR Simulation: {len(sim_hr.frame_numbers)} frames found.")

    sim_sg = ergane.SimulationData(athinp=str(args.sg_athinput), datafolder=str(args.sg_bin))
    print(f"  Subgrid Simulation: {len(sim_sg.frame_numbers)} frames found.")

    sim_lr = ergane.SimulationData(athinp=str(args.lr_athinput), datafolder=str(args.lr_bin))
    print(f"  LR Simulation: {len(sim_lr.frame_numbers)} frames found.")

    HR_START_SNAP = 500

    # Number of frames to process
    # If snapshots < 500 were pruned, frame_numbers will start at 500.
    # We find all HR frames >= HR_START_SNAP
    hr_frames_available = [f for f in sim_hr.frame_numbers if f >= HR_START_SNAP]
    available_hr = len(hr_frames_available)
    nt = min(len(sim_sg.frame_numbers), len(sim_lr.frame_numbers), available_hr)
    print(f"  Processing {nt} common frames between HR (from snap {HR_START_SNAP}), SG, and LR...")

    # 2. Extract arrays into memory
    print("\n[2] Loading simulation frames into memory...")
    # Determine shapes for preallocation
    f0_hr = sim_hr.get_frame(hr_frames_available[0])
    ny_hr, nx_hr = f0_hr.density.shape
    f0_sg = sim_sg.get_frame(sim_sg.frame_numbers[0])
    ny_lr, nx_lr = f0_sg.density.shape

    # Preallocate float32 arrays directly to prevent massive peak RAM during np.array(list_of_arrays)
    hr_rho  = np.empty((nt, ny_hr, nx_hr), dtype=np.float32)
    hr_temp = np.empty((nt, ny_hr, nx_hr), dtype=np.float32)
    hr_pres = np.empty((nt, ny_hr, nx_hr), dtype=np.float32)
    hr_ux   = np.empty((nt, ny_hr, nx_hr), dtype=np.float32)
    hr_uy   = np.empty((nt, ny_hr, nx_hr), dtype=np.float32)
    hr_ien  = np.empty((nt, ny_hr, nx_hr), dtype=np.float32)
    hr_ps   = np.empty((nt, ny_hr, nx_hr), dtype=np.float32)

    cg_hr_rho  = np.empty((nt, ny_lr, nx_lr), dtype=np.float32)
    cg_hr_temp = np.empty((nt, ny_lr, nx_lr), dtype=np.float32)
    cg_hr_pres = np.empty((nt, ny_lr, nx_lr), dtype=np.float32)
    cg_hr_ux   = np.empty((nt, ny_lr, nx_lr), dtype=np.float32)
    cg_hr_uy   = np.empty((nt, ny_lr, nx_lr), dtype=np.float32)
    cg_hr_ien  = np.empty((nt, ny_lr, nx_lr), dtype=np.float32)
    cg_hr_ps   = np.empty((nt, ny_lr, nx_lr), dtype=np.float32)

    sg_rho  = np.empty((nt, ny_lr, nx_lr), dtype=np.float32)
    sg_temp = np.empty((nt, ny_lr, nx_lr), dtype=np.float32)
    sg_pres = np.empty((nt, ny_lr, nx_lr), dtype=np.float32)
    sg_ux   = np.empty((nt, ny_lr, nx_lr), dtype=np.float32)
    sg_uy   = np.empty((nt, ny_lr, nx_lr), dtype=np.float32)
    sg_ien  = np.empty((nt, ny_lr, nx_lr), dtype=np.float32)
    sg_ps   = np.empty((nt, ny_lr, nx_lr), dtype=np.float32)
    sg_fmcl = np.empty((nt, ny_lr, nx_lr), dtype=np.float32)

    lr_rho  = np.empty((nt, ny_lr, nx_lr), dtype=np.float32)
    lr_temp = np.empty((nt, ny_lr, nx_lr), dtype=np.float32)
    lr_pres = np.empty((nt, ny_lr, nx_lr), dtype=np.float32)
    lr_ux   = np.empty((nt, ny_lr, nx_lr), dtype=np.float32)
    lr_uy   = np.empty((nt, ny_lr, nx_lr), dtype=np.float32)
    lr_ien  = np.empty((nt, ny_lr, nx_lr), dtype=np.float32)
    lr_ps   = np.empty((nt, ny_lr, nx_lr), dtype=np.float32)
    lr_fmcl = np.empty((nt, ny_lr, nx_lr), dtype=np.float32)

    for i in tqdm(range(nt), desc="Extracting frames"):
        # HR frame
        f_hr = sim_hr.get_frame(hr_frames_available[i])
        rho_h = (f_hr.density / f_hr.units.density).astype(np.float32)
        temp_h = f_hr.temperature.astype(np.float32)
        pres_h = (f_hr.pressure / f_hr.units.pressure).astype(np.float32)
        vx_h = (f_hr.velx / f_hr.units.velocity).astype(np.float32)
        vy_h = (f_hr.vely / f_hr.units.velocity).astype(np.float32)
        ien_h = (f_hr.eint / f_hr.units.pressure).astype(np.float32)
        ps_h = f_hr.scalars['scalar_00'].astype(np.float32) if ('scalar_00' in f_hr.scalars) else np.zeros_like(temp_h)

        hr_rho[i]  = rho_h
        hr_temp[i] = temp_h
        hr_pres[i] = pres_h
        hr_ux[i]   = vx_h
        hr_uy[i]   = vy_h
        hr_ien[i]  = ien_h
        hr_ps[i]   = ps_h

        # CG HR
        cg_r = coarse_grain_2d(rho_h, DS)
        cg_hr_rho[i]  = cg_r
        cg_hr_temp[i] = coarse_grain_2d(temp_h * rho_h, DS) / np.maximum(cg_r, 1e-30)
        cg_hr_pres[i] = coarse_grain_2d(pres_h, DS)
        cg_hr_ux[i]   = coarse_grain_2d(vx_h * rho_h, DS) / np.maximum(cg_r, 1e-30)
        cg_hr_uy[i]   = coarse_grain_2d(vy_h * rho_h, DS) / np.maximum(cg_r, 1e-30)
        cg_hr_ien[i]  = coarse_grain_2d(ien_h, DS)
        cg_hr_ps[i]   = coarse_grain_2d(ps_h * rho_h, DS) / np.maximum(cg_r, 1e-30)

        # SG frame
        f_sg = sim_sg.get_frame(sim_sg.frame_numbers[i])
        sg_rho[i]  = (f_sg.density / f_sg.units.density).astype(np.float32)
        sg_temp[i] = f_sg.temperature.astype(np.float32)
        sg_pres[i] = (f_sg.pressure / f_sg.units.pressure).astype(np.float32)
        sg_ux[i]   = (f_sg.velx / f_sg.units.velocity).astype(np.float32)
        sg_uy[i]   = (f_sg.vely / f_sg.units.velocity).astype(np.float32)
        sg_ien[i]  = (f_sg.eint / f_sg.units.pressure).astype(np.float32)
        sg_ps[i]   = f_sg.scalars['scalar_00'].astype(np.float32) if 'scalar_00' in f_sg.scalars else np.zeros_like(f_sg.temperature, dtype=np.float32)
        sg_fmcl[i] = f_sg.scalars['scalar_01'].astype(np.float32) if 'scalar_01' in f_sg.scalars else (f_sg.temperature < 1e5).astype(np.float32)

        # LR frame
        f_lr = sim_lr.get_frame(sim_lr.frame_numbers[i])
        lr_rho[i]  = (f_lr.density / f_lr.units.density).astype(np.float32)
        lr_temp[i] = f_lr.temperature.astype(np.float32)
        lr_pres[i] = (f_lr.pressure / f_lr.units.pressure).astype(np.float32)
        lr_ux[i]   = (f_lr.velx / f_lr.units.velocity).astype(np.float32)
        lr_uy[i]   = (f_lr.vely / f_lr.units.velocity).astype(np.float32)
        lr_ien[i]  = (f_lr.eint / f_lr.units.pressure).astype(np.float32)
        lr_ps[i]   = f_lr.scalars['scalar_00'].astype(np.float32) if 'scalar_00' in f_lr.scalars else np.zeros_like(f_lr.temperature, dtype=np.float32)
        lr_fmcl[i] = f_lr.scalars['scalar_01'].astype(np.float32) if 'scalar_01' in f_lr.scalars else (f_lr.temperature < 1e5).astype(np.float32)

    # 3. Conserved fields
    hr_cons_rho  = hr_rho
    hr_cons_momx = hr_rho * hr_ux
    hr_cons_momy = hr_rho * hr_uy
    hr_cons_ener = hr_ien + 0.5 * hr_rho * (hr_ux**2 + hr_uy**2)
    hr_cons_ps   = hr_rho * hr_ps
    hr_fmcl      = (hr_temp < 1e5).astype(np.float32)

    cg_hr_cons_rho  = cg_hr_rho
    cg_hr_cons_momx = cg_hr_rho * cg_hr_ux
    cg_hr_cons_momy = cg_hr_rho * cg_hr_uy
    cg_hr_cons_ener = cg_hr_ien + 0.5 * cg_hr_rho * (cg_hr_ux**2 + cg_hr_uy**2)
    cg_hr_cons_ps   = cg_hr_rho * cg_hr_ps
    cg_hr_fmcl      = (cg_hr_temp < 1e5).astype(float)

    sg_cons_rho  = sg_rho
    sg_cons_momx = sg_rho * sg_ux
    sg_cons_momy = sg_rho * sg_uy
    sg_cons_ener = sg_ien + 0.5 * sg_rho * (sg_ux**2 + sg_uy**2)
    sg_cons_ps   = sg_rho * sg_ps

    lr_cons_rho  = lr_rho
    lr_cons_momx = lr_rho * lr_ux
    lr_cons_momy = lr_rho * lr_uy
    lr_cons_ener = lr_ien + 0.5 * lr_rho * (lr_ux**2 + lr_uy**2)
    lr_cons_ps   = lr_rho * lr_ps

    # 4. Compute Emissivity & Tiled CNN Inference
    print("\n[3] Computing Subgrid CNN inferences & cooling fields...")
    model, input_mean, input_std = load_tiled_cnn_model()

    emis_hr = (hr_rho * n_to_cm3)**2 * lambda_cool(hr_temp, mask=True, LOGT_ACTIVE_START=LOGT_ACTIVE_START, LOGT_ACTIVE_END=LOGT_ACTIVE_END)
    emis_cg_hr = np.array([coarse_grain_2d(e, DS) for e in emis_hr])
    emis_lr = (lr_rho * n_to_cm3)**2 * lambda_cool(lr_temp, mask=True, LOGT_ACTIVE_START=LOGT_ACTIVE_START, LOGT_ACTIVE_END=LOGT_ACTIVE_END)

    ny_cg, nx_cg = sg_rho.shape[1], sg_rho.shape[2]
    RESOLUTION = (ny_cg, nx_cg)

    emis_sg = np.zeros_like(sg_rho)
    pred_pdf_all = np.zeros((nt, out_channels, ny_cg, nx_cg), dtype=np.float32)
    pred_gate_all = np.zeros((nt, ny_cg, nx_cg), dtype=np.float32)

    for t in tqdm(range(nt), desc="Tiled CNN Inference"):
        c_sg, p_sg, g_sg = predict_tiled_subgrid_pdf_and_cooling(
            sg_rho[t], sg_temp[t], sg_ux[t], sg_uy[t], sg_ps[t], sg_pres[t],
            model, input_mean, input_std
        )
        emis_sg[t] = c_sg
        pred_pdf_all[t] = p_sg
        pred_gate_all[t] = g_sg

    all_pos_cool = np.concatenate([
        emis_hr[emis_hr > 0],
        emis_cg_hr[emis_cg_hr > 0],
        emis_sg[emis_sg > 0],
        emis_lr[emis_lr > 0],
    ])
    if len(all_pos_cool) > 0:
        cool_vmin = max(np.percentile(all_pos_cool, 1), 1e-28)
        cool_vmax = np.percentile(all_pos_cool, 99)
    else:
        cool_vmin, cool_vmax = 1e-28, 1e-21

    # Coordinates and Domain Extents
    x1min = getattr(sim_sg, 'x1min', getattr(sim_hr, 'x1min', -5.0))
    x1max = getattr(sim_sg, 'x1max', getattr(sim_hr, 'x1max', 5.0))
    x2min = getattr(sim_sg, 'x2min', getattr(sim_hr, 'x2min', -20.0))
    x2max = getattr(sim_sg, 'x2max', getattr(sim_hr, 'x2max', 20.0))
    Lx = float(x1max - x1min)
    Ly = float(x2max - x2min)

    y_cg_hr = np.linspace(x2min, x2max, ny_cg)
    y_sg = np.linspace(x2min, x2max, ny_cg)
    y_lr = np.linspace(x2min, x2max, ny_cg)

    t_restart_myr = RESTART_TIME_MYR + np.arange(nt) * BIN_DT_MYR

    # Table of Pearson Correlation and RMSD: (Profile, CGHR_Subgrid Pearson, CGHR_Subgrid RMSD, CGHR_LR Pearson, CGHR_LR RMSD)
    profile_metrics = []

    # =========================================================================
    # PLOT 1: Primitive Profiles (Mean ± 1σ across time)
    # =========================================================================
    print("\n[4] Generating profiles_mean_with_std_all.png...")
    quantities = [
        ("Density", cg_hr_rho * n_to_cm3, sg_rho * n_to_cm3, lr_rho * n_to_cm3, r"$\log_{10}(n \ [\mathrm{cm}^{-3}])$", True),
        ("Temperature", cg_hr_temp, sg_temp, lr_temp, r"$\log_{10}(T \ [\mathrm{K}])$", True),
        ("Pressure", cg_hr_pres * P_over_kB_to_K_cm3, sg_pres * P_over_kB_to_K_cm3, lr_pres * P_over_kB_to_K_cm3, r"$P/k_B \ [\mathrm{K} \ \mathrm{cm}^{-3}]$", False),
        ("Ux Velocity", cg_hr_ux * vel_to_km_s, sg_ux * vel_to_km_s, lr_ux * vel_to_km_s, r"$u_x \ [\mathrm{km} \ \mathrm{s}^{-1}]$", False),
        ("Uy Velocity", cg_hr_uy * vel_to_km_s, sg_uy * vel_to_km_s, lr_uy * vel_to_km_s, r"$u_y \ [\mathrm{km} \ \mathrm{s}^{-1}]$", False),
    ]

    fig, axs = plt.subplots(5, 1, figsize=(9, 20))
    plt.subplots_adjust(hspace=0.35)

    for idx, (title, hr_arr, sg_arr, lr_arr, ylabel, is_log) in enumerate(quantities):
        hr_m, hr_s = compute_mean_std(hr_arr, logspace=is_log)
        sg_m, sg_s = compute_mean_std(sg_arr, logspace=is_log)
        lr_m, lr_s = compute_mean_std(lr_arr, logspace=is_log)

        sg_r, sg_rmsd = compute_profile_metrics(hr_m, sg_m)
        lr_r, lr_rmsd = compute_profile_metrics(hr_m, lr_m)
        profile_metrics.append((title, sg_r, sg_rmsd, lr_r, lr_rmsd))

        ax = axs[idx]
        ax.plot(y_cg_hr, hr_m, lw=2, ls="-",  marker="^", markersize=4, label=CG_LABEL)
        ax.fill_between(y_cg_hr, hr_m - hr_s, hr_m + hr_s, alpha=0.25)

        ax.plot(y_sg, sg_m, lw=2, ls="-.", marker="o", markersize=5, label=SG_LABEL)
        ax.fill_between(y_sg, sg_m - sg_s, sg_m + sg_s, alpha=0.25)

        ax.plot(y_lr, lr_m, lw=2, ls="--", marker="s", markersize=5, label=LR_LABEL)
        ax.fill_between(y_lr, lr_m - lr_s, lr_m + lr_s, alpha=0.25)

        ax.set_title(f"{title} (Avg over X) — Mean ± 1σ", fontsize=13, weight="bold")
        ax.set_xlabel(r"$y \ [\mathrm{pc}]$", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(out_dir / "profiles_mean_with_std_all.png", dpi=200)
    plt.close(fig)
    print("  Saved profiles_mean_with_std_all.png")

    # =========================================================================
    # PLOT 2: Conserved Quantities Profiles
    # =========================================================================
    print("\n[5] Generating conserved_quantities_mean_with_std.png...")
    quantities_cons = [
        ("Conserved Density", cg_hr_cons_rho * n_to_cm3, sg_cons_rho * n_to_cm3, lr_cons_rho * n_to_cm3, r"$n \ [\mathrm{cm}^{-3}]$"),
        ("Conserved MomX", cg_hr_cons_momx * mflux_to_Msun_yr_kpc2, sg_cons_momx * mflux_to_Msun_yr_kpc2, lr_cons_momx * mflux_to_Msun_yr_kpc2, r"$\rho u_x \ [M_\odot \ \mathrm{yr}^{-1} \ \mathrm{kpc}^{-2}]$"),
        ("Conserved MomY", cg_hr_cons_momy * mflux_to_Msun_yr_kpc2, sg_cons_momy * mflux_to_Msun_yr_kpc2, lr_cons_momy * mflux_to_Msun_yr_kpc2, r"$\rho u_y \ [M_\odot \ \mathrm{yr}^{-1} \ \mathrm{kpc}^{-2}]$"),
        ("Conserved Energy", cg_hr_cons_ener * P_cgs, sg_cons_ener * P_cgs, lr_cons_ener * P_cgs, r"$E \ [\mathrm{erg} \ \mathrm{cm}^{-3}]$"),
        ("Passive Scalar", cg_hr_cons_ps, sg_cons_ps, lr_cons_ps, "Passive Scalar"),
        ("fmcl (T < 1e5)", cg_hr_fmcl, sg_fmcl, lr_fmcl, r"$f_{\mathrm{mcl}} \ (T < 10^5 \ \mathrm{K})$"),
    ]

    fig, axs = plt.subplots(6, 1, figsize=(9, 24))
    plt.subplots_adjust(hspace=0.4)

    for idx, (title, hr_arr, sg_arr, lr_arr, ylabel) in enumerate(quantities_cons):
        hr_m, hr_s = compute_mean_std(hr_arr)
        sg_m, sg_s = compute_mean_std(sg_arr)
        lr_m, lr_s = compute_mean_std(lr_arr)

        sg_r, sg_rmsd = compute_profile_metrics(hr_m, sg_m)
        lr_r, lr_rmsd = compute_profile_metrics(hr_m, lr_m)
        profile_metrics.append((title, sg_r, sg_rmsd, lr_r, lr_rmsd))

        ax = axs[idx]
        ax.plot(y_cg_hr, hr_m, lw=2, ls="-",  marker="^", markersize=4, label=CG_LABEL)
        ax.fill_between(y_cg_hr, hr_m - hr_s, hr_m + hr_s, alpha=0.25)

        ax.plot(y_sg, sg_m, lw=2, ls="-.", marker="o", markersize=5, label=SG_LABEL)
        ax.fill_between(y_sg, sg_m - sg_s, sg_m + sg_s, alpha=0.25)

        ax.plot(y_lr, lr_m, lw=2, ls="--", marker="s", markersize=5, label=LR_LABEL)
        ax.fill_between(y_lr, lr_m - lr_s, lr_m + lr_s, alpha=0.25)

        ax.set_title(f"{title} (Avg over X) — Mean ± 1σ", fontsize=13, weight="bold")
        ax.set_xlabel(r"$y \ [\mathrm{pc}]$", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(out_dir / "conserved_quantities_mean_with_std.png", dpi=200)
    plt.close(fig)
    print("  Saved conserved_quantities_mean_with_std.png")

    # =========================================================================
    # PLOT 3: Derived Fluxes & Tensors
    # =========================================================================
    print("\n[6] Generating derived_quantities_mean_with_std.png & fluxes_mean_std.png...")
    cg_hr_mass_x = cg_hr_rho * cg_hr_ux
    cg_hr_mass_y = cg_hr_rho * cg_hr_uy
    cg_hr_T_xx = cg_hr_rho * cg_hr_ux**2 + cg_hr_pres
    cg_hr_T_xy = cg_hr_rho * cg_hr_ux * cg_hr_uy
    cg_hr_T_yy = cg_hr_rho * cg_hr_uy**2 + cg_hr_pres

    sg_mass_x = sg_rho * sg_ux
    sg_mass_y = sg_rho * sg_uy
    sg_T_xx = sg_rho * sg_ux**2 + sg_pres
    sg_T_xy = sg_rho * sg_ux * sg_uy
    sg_T_yy = sg_rho * sg_uy**2 + sg_pres

    lr_mass_x = lr_rho * lr_ux
    lr_mass_y = lr_rho * lr_uy
    lr_T_xx = lr_rho * lr_ux**2 + lr_pres
    lr_T_xy = lr_rho * lr_ux * lr_uy
    lr_T_yy = lr_rho * lr_uy**2 + lr_pres

    fig, axs = plt.subplots(3, 1, figsize=(9, 15))
    plt.subplots_adjust(hspace=0.35)

    def plot_helper(ax, hr_f, sg_f, lr_f, title, ylabel, conv):
        h_m, h_s = compute_mean_std(hr_f * conv)
        s_m, s_s = compute_mean_std(sg_f * conv)
        l_m, l_s = compute_mean_std(lr_f * conv)

        sg_r, sg_rmsd = compute_profile_metrics(h_m, s_m)
        lr_r, lr_rmsd = compute_profile_metrics(h_m, l_m)
        profile_metrics.append((title, sg_r, sg_rmsd, lr_r, lr_rmsd))

        ax.plot(y_cg_hr, h_m, lw=2, ls="-",  marker="^", markersize=4, label=CG_LABEL)
        ax.fill_between(y_cg_hr, h_m - h_s, h_m + h_s, alpha=0.25)
        ax.plot(y_sg, s_m, lw=2, ls="-.", marker="o", markersize=5, label=SG_LABEL)
        ax.fill_between(y_sg, s_m - s_s, s_m + s_s, alpha=0.25)
        ax.plot(y_lr, l_m, lw=2, ls="--", marker="s", markersize=5, label=LR_LABEL)
        ax.fill_between(y_lr, l_m - l_s, l_m + l_s, alpha=0.25)
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_xlabel(r"$y \ [\mathrm{pc}]$", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(fontsize=10)

    plot_helper(axs[0], cg_hr_mass_x, sg_mass_x, lr_mass_x, r"$\rho u_x$ (Avg over X)", r"$\rho u_x \ [M_\odot \ \mathrm{yr}^{-1} \ \mathrm{kpc}^{-2}]$", mflux_to_Msun_yr_kpc2)
    plot_helper(axs[1], cg_hr_T_xy, sg_T_xy, lr_T_xy, r"$\rho u_x u_y$ (Avg over X)", r"$\rho u_x u_y \ [\mathrm{dyn} \ \mathrm{cm}^{-2}]$", P_cgs)
    plot_helper(axs[2], cg_hr_T_yy, sg_T_yy, lr_T_yy, r"$p + \rho u_y^2$ (Avg over X)", r"$p + \rho u_y^2 \ [\mathrm{dyn} \ \mathrm{cm}^{-2}]$", P_cgs)

    plt.tight_layout()
    plt.savefig(out_dir / "derived_quantities_mean_with_std.png", dpi=200)
    plt.close(fig)
    print("  Saved derived_quantities_mean_with_std.png")

    # Energy flux plot
    cg_hr_E = cg_hr_pres / (GAMMA - 1) + 0.5 * cg_hr_rho * (cg_hr_ux**2 + cg_hr_uy**2)
    sg_E = sg_pres / (GAMMA - 1) + 0.5 * sg_rho * (sg_ux**2 + sg_uy**2)
    lr_E = lr_pres / (GAMMA - 1) + 0.5 * lr_rho * (lr_ux**2 + lr_uy**2)

    fig, axs = plt.subplots(4, 2, figsize=(12, 16))
    plt.subplots_adjust(hspace=0.35)
    plot_helper(axs[0, 0], cg_hr_mass_x, sg_mass_x, lr_mass_x, r"Mass Flux ($\rho u_x$)", r"$\rho u_x \ [M_\odot \ \mathrm{yr}^{-1} \ \mathrm{kpc}^{-2}]$", mflux_to_Msun_yr_kpc2)
    plot_helper(axs[0, 1], cg_hr_mass_y, sg_mass_y, lr_mass_y, r"Mass Flux ($\rho u_y$)", r"$\rho u_y \ [M_\odot \ \mathrm{yr}^{-1} \ \mathrm{kpc}^{-2}]$", mflux_to_Msun_yr_kpc2)
    plot_helper(axs[1, 0], cg_hr_T_xx, sg_T_xx, lr_T_xx, r"Momentum Flux $T_{xx} = \rho u_x^2 + p$", r"$T_{xx} \ [\mathrm{dyn} \ \mathrm{cm}^{-2}]$", P_cgs)
    plot_helper(axs[1, 1], cg_hr_T_xy, sg_T_xy, lr_T_xy, r"Momentum Flux $T_{xy} = \rho u_x u_y$", r"$T_{xy} \ [\mathrm{dyn} \ \mathrm{cm}^{-2}]$", P_cgs)
    plot_helper(axs[2, 0], cg_hr_T_xy, sg_T_xy, lr_T_xy, r"Momentum Flux $T_{yx} = \rho u_x u_y$", r"$T_{yx} \ [\mathrm{dyn} \ \mathrm{cm}^{-2}]$", P_cgs)
    plot_helper(axs[2, 1], cg_hr_T_yy, sg_T_yy, lr_T_yy, r"Momentum Flux $T_{yy} = \rho u_y^2 + p$", r"$T_{yy} \ [\mathrm{dyn} \ \mathrm{cm}^{-2}]$", P_cgs)
    plot_helper(axs[3, 0], (cg_hr_E + cg_hr_pres)*cg_hr_ux, (sg_E + sg_pres)*sg_ux, (lr_E + lr_pres)*lr_ux, r"Energy Flux $(E+p)u_x$", r"$(E+p)u_x \ [\mathrm{erg} \ \mathrm{cm}^{-2} \ \mathrm{s}^{-1}]$", P_cgs * V_cgs)
    plot_helper(axs[3, 1], (cg_hr_E + cg_hr_pres)*cg_hr_uy, (sg_E + sg_pres)*sg_uy, (lr_E + lr_pres)*lr_uy, r"Energy Flux $(E+p)u_y$", r"$(E+p)u_y \ [\mathrm{erg} \ \mathrm{cm}^{-2} \ \mathrm{s}^{-1}]$", P_cgs * V_cgs)

    plt.tight_layout()
    plt.savefig(out_dir / "fluxes_mean_std.png", dpi=200)
    plt.close(fig)
    print("  Saved fluxes_mean_std.png")

    # =========================================================================
    # PLOT 4: Divergence Fluxes
    # =========================================================================
    print("\n[7] Generating divergence_fluxes_mean_std.png...")
    dy = Ly / RESOLUTION[0]
    dx = Lx / RESOLUTION[1]

    cg_hr_div_m = np.zeros_like(cg_hr_mass_x)
    cg_hr_div_x = np.zeros_like(cg_hr_mass_x)
    cg_hr_div_y = np.zeros_like(cg_hr_mass_x)

    sg_div_m = np.zeros_like(sg_mass_x)
    sg_div_x = np.zeros_like(sg_mass_x)
    sg_div_y = np.zeros_like(sg_mass_x)

    lr_div_m = np.zeros_like(lr_mass_x)
    lr_div_x = np.zeros_like(lr_mass_x)
    lr_div_y = np.zeros_like(lr_mass_x)

    for i in range(nt):
        cg_hr_div_m[i] = divergence([cg_hr_mass_x[i], cg_hr_mass_y[i]], dx, dy)
        cg_hr_div_x[i] = divergence([cg_hr_T_xx[i], cg_hr_T_xy[i]], dx, dy)
        cg_hr_div_y[i] = divergence([cg_hr_T_xy[i], cg_hr_T_yy[i]], dx, dy)

        sg_div_m[i] = divergence([sg_mass_x[i], sg_mass_y[i]], dx, dy)
        sg_div_x[i] = divergence([sg_T_xx[i], sg_T_xy[i]], dx, dy)
        sg_div_y[i] = divergence([sg_T_xy[i], sg_T_yy[i]], dx, dy)

        lr_div_m[i] = divergence([lr_mass_x[i], lr_mass_y[i]], dx, dy)
        lr_div_x[i] = divergence([lr_T_xx[i], lr_T_xy[i]], dx, dy)
        lr_div_y[i] = divergence([lr_T_xy[i], lr_T_yy[i]], dx, dy)

    fig, axs = plt.subplots(3, 1, figsize=(10, 13))
    plt.subplots_adjust(hspace=0.35)
    plot_helper(axs[0], cg_hr_div_m, sg_div_m, lr_div_m, r"Div Mass Flux ($\nabla \cdot \mathbf{j}$)", r"$\nabla \cdot (\rho \mathbf{u}) \ [M_\odot \ \mathrm{yr}^{-1} \ \mathrm{kpc}^{-2} \ \mathrm{pc}^{-1}]$", mflux_to_Msun_yr_kpc2)
    plot_helper(axs[1], cg_hr_div_x, sg_div_x, lr_div_x, r"Div MomX Flux ($\nabla \cdot \mathbf{T}_x$)", r"$\nabla \cdot \mathbf{T}_x \ [\mathrm{dyn} \ \mathrm{cm}^{-2} \ \mathrm{pc}^{-1}]$", P_cgs)
    plot_helper(axs[2], cg_hr_div_y, sg_div_y, lr_div_y, r"Div MomY Flux ($\nabla \cdot \mathbf{T}_y$)", r"$\nabla \cdot \mathbf{T}_y \ [\mathrm{dyn} \ \mathrm{cm}^{-2} \ \mathrm{pc}^{-1}]$", P_cgs)

    plt.tight_layout()
    plt.savefig(out_dir / "divergence_fluxes_mean_std.png", dpi=200)
    plt.close(fig)
    print("  Saved divergence_fluxes_mean_std.png")

    # =========================================================================
    # PLOT 5: Cold Gas Mass Evolution (T < 1e5 K)
    # =========================================================================
    print("\n[8] Generating cold_mass_evolution.png...")
    dx_pc = Lx / RESOLUTION[1]
    dy_pc = Ly / RESOLUTION[0]
    cell_area = dx_pc * dy_pc

    mass_hr = np.sum((hr_temp < 1e5) * hr_rho, axis=(1, 2)) * (Lx / hr_rho.shape[2]) * (Ly / hr_rho.shape[1])
    mass_cg_hr = np.sum((cg_hr_temp < 1e5) * cg_hr_rho, axis=(1, 2)) * cell_area
    mass_sg = np.sum((sg_temp < 1e5) * sg_rho, axis=(1, 2)) * cell_area
    mass_lr = np.sum((lr_temp < 1e5) * lr_rho, axis=(1, 2)) * cell_area

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvline(RESTART_TIME_MYR, color="gray", ls="--", lw=1.2, label=f"Restart @ {RESTART_TIME_MYR} Myr")

    ax.plot(t_restart_myr, mass_hr, label="HR", lw=2, ls="-", marker="^", markersize=5)
    ax.plot(t_restart_myr, mass_sg, label="SG (subgrid_model)", lw=2, ls="-.", marker="o", markersize=5)
    ax.plot(t_restart_myr, mass_lr, label="LR (hr_build)", lw=2, ls="--", marker="s", markersize=5)

    if nt > 1:
        s_hr, i_hr = np.polyfit(t_restart_myr, mass_hr, 1)
        s_sg, i_sg = np.polyfit(t_restart_myr, mass_sg, 1)
        s_lr, i_lr = np.polyfit(t_restart_myr, mass_lr, 1)
        ax.plot(t_restart_myr, s_hr * t_restart_myr + i_hr, lw=1.8, ls="--", label=f"HR fit (dM/dt = {s_hr:.3e})")
        ax.plot(t_restart_myr, s_sg * t_restart_myr + i_sg, lw=1.8, ls="--", label=f"SG fit (dM/dt = {s_sg:.3e})")
        ax.plot(t_restart_myr, s_lr * t_restart_myr + i_lr, lw=1.8, ls="--", label=f"LR fit (dM/dt = {s_lr:.3e})")

    ax.set_xlabel("Physical Time [Myr]", fontsize=13)
    ax.set_ylabel(r"Cold Mass ($\rho \cdot \mathrm{pc}^2$)", fontsize=13)
    ax.set_title("Cold Gas Mass ($T < 10^5$ K) Evolution", fontsize=14, weight="bold")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "cold_mass_evolution.png", dpi=200)
    plt.close(fig)
    print("  Saved cold_mass_evolution.png")

    # =========================================================================
    # PLOT 5b: Timestep (dt) vs Time
    # =========================================================================
    print("\n[8b] Generating delta_t_vs_time.png...")

    # Construct history file paths (history files are in parent dir of bin/)
    hr_hist_path = Path(args.hr_bin).parent / "KH.hydro.hst"
    sg_hist_path = Path(args.sg_bin).parent / "KH.hydro.hst"
    lr_hist_path = Path(args.lr_bin).parent / "KH.hydro.hst"

    # Load history data (dt_cfl/dt_cool are None for older history files that
    # predate these columns)
    hr_times_hist, hr_dts, hr_dt_cfl, hr_dt_cool = load_history_file(hr_hist_path)
    sg_times_hist, sg_dts, sg_dt_cfl, sg_dt_cool = load_history_file(sg_hist_path)
    lr_times_hist, lr_dts, lr_dt_cfl, lr_dt_cool = load_history_file(lr_hist_path)

    # For restarted simulations (SG and LR), shift times to physical time by adding restart time
    if sg_times_hist is not None:
        sg_times_hist = sg_times_hist + RESTART_TIME_MYR
    if lr_times_hist is not None:
        lr_times_hist = lr_times_hist + RESTART_TIME_MYR

    def _mask_run(times, *component_arrays):
        """Restrict `times` and any number of same-length arrays to
        [RESTART_TIME_MYR, end], passing None arrays through unchanged."""
        if times is None:
            return (times, *component_arrays)
        mask = times >= RESTART_TIME_MYR
        masked = tuple(arr[mask] if arr is not None else None for arr in component_arrays)
        return (times[mask], *masked)

    hr_times_hist, hr_dts, hr_dt_cfl, hr_dt_cool = _mask_run(hr_times_hist, hr_dts, hr_dt_cfl, hr_dt_cool)
    sg_times_hist, sg_dts, sg_dt_cfl, sg_dt_cool = _mask_run(sg_times_hist, sg_dts, sg_dt_cfl, sg_dt_cool)
    lr_times_hist, lr_dts, lr_dt_cfl, lr_dt_cool = _mask_run(lr_times_hist, lr_dts, lr_dt_cfl, lr_dt_cool)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvline(RESTART_TIME_MYR, color="gray", ls="--", lw=1.2, label=f"Restart @ {RESTART_TIME_MYR} Myr", alpha=0.7)

    if hr_times_hist is not None and hr_dts is not None:
        ax.plot(hr_times_hist, hr_dts * 1e3, label="HR (512x1024)", lw=2, marker="^", markersize=4, alpha=0.8)

    if sg_times_hist is not None and sg_dts is not None:
        ax.plot(sg_times_hist, sg_dts * 1e3, label=SG_LABEL, lw=2, marker="o", markersize=5, alpha=0.8)

    if lr_times_hist is not None and lr_dts is not None:
        ax.plot(lr_times_hist, lr_dts * 1e3, label=LR_LABEL, lw=2, marker="s", markersize=5, alpha=0.8)

    ax.set_xlabel("Physical Time [Myr]", fontsize=13)
    ax.set_ylabel(r"Timestep $\Delta t$ [ms (code units)]", fontsize=13)
    ax.set_title(r"Timestep ($\Delta t$) vs Simulation Time", fontsize=14, weight="bold")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "delta_t_vs_time.png", dpi=200)
    plt.close(fig)
    print("  Saved delta_t_vs_time.png")

    # =========================================================================
    # PLOT 5c: Timestep components (CFL vs cooling-limited) vs Time
    # =========================================================================
    print("\n[8c] Generating delta_t_components_vs_time.png...")

    # dt_cool reports as float_max*cfl_no ("no constraint") when a run has no
    # active cooling source term; drop those points rather than let them blow
    # out the y-axis.
    _DT_SENTINEL = 1.0e30

    def _clip_sentinel(arr):
        if arr is None:
            return None
        return np.where(arr > _DT_SENTINEL, np.nan, arr)

    hr_dt_cool_c = _clip_sentinel(hr_dt_cool)
    sg_dt_cool_c = _clip_sentinel(sg_dt_cool)
    lr_dt_cool_c = _clip_sentinel(lr_dt_cool)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    runs = [
        (axes[0], "HR (512x1024)", hr_times_hist, hr_dts, hr_dt_cfl, hr_dt_cool_c),
        (axes[1], SG_LABEL, sg_times_hist, sg_dts, sg_dt_cfl, sg_dt_cool_c),
        (axes[2], LR_LABEL, lr_times_hist, lr_dts, lr_dt_cfl, lr_dt_cool_c),
    ]
    for ax_i, label, times, dts, dt_cfl, dt_cool in runs:
        ax_i.axvline(RESTART_TIME_MYR, color="gray", ls="--", lw=1.0, alpha=0.7)
        if times is None or dts is None:
            ax_i.set_title(f"{label}\n(no history data)", fontsize=12)
            continue
        ax_i.plot(times, dts * 1e3, label=r"$\Delta t$ (used)", lw=2, color="black", alpha=0.85)
        if dt_cfl is not None:
            ax_i.plot(times, dt_cfl * 1e3, label=r"$\Delta t_\mathrm{CFL}$", lw=1.5,
                      ls="--", marker="^", markersize=3, alpha=0.8)
        if dt_cool is not None:
            ax_i.plot(times, dt_cool * 1e3, label=r"$\Delta t_\mathrm{cool}$", lw=1.5,
                      ls="--", marker="o", markersize=3, alpha=0.8)
        ax_i.set_yscale("log")
        ax_i.set_xlabel("Physical Time [Myr]", fontsize=12)
        ax_i.set_title(label, fontsize=13, weight="bold")
        ax_i.grid(True, ls="--", alpha=0.4)
        ax_i.legend(fontsize=9)

    axes[0].set_ylabel(r"Timestep [ms (code units)]", fontsize=13)
    fig.suptitle(r"CFL-limited vs Cooling-limited Timestep Components", fontsize=15, weight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "delta_t_components_vs_time.png", dpi=200)
    plt.close(fig)
    print("  Saved delta_t_components_vs_time.png")

    # =========================================================================
    # PLOT 6: Emissivity Profile vs Y & Integrated Sigma_c
    # =========================================================================
    print("\n[9] Generating emissivity_profile_vs_y.png...")
    emis_cg_hr_xavg = np.mean(emis_cg_hr, axis=2)
    emis_sg_xavg = np.mean(emis_sg, axis=2)
    emis_lr_xavg = np.mean(emis_lr, axis=2)

    e_cg_m = np.mean(emis_cg_hr_xavg, axis=0)
    e_cg_s = np.std(emis_cg_hr_xavg, axis=0) if nt > 1 else np.zeros_like(e_cg_m)
    e_sg_m = np.mean(emis_sg_xavg, axis=0)
    e_sg_s = np.std(emis_sg_xavg, axis=0) if nt > 1 else np.zeros_like(e_sg_m)
    e_lr_m = np.mean(emis_lr_xavg, axis=0)
    e_lr_s = np.std(emis_lr_xavg, axis=0) if nt > 1 else np.zeros_like(e_lr_m)

    int_cg = trapz_integral(e_cg_m, y_cg_hr)
    int_sg = trapz_integral(e_sg_m, y_sg)
    int_lr = trapz_integral(e_lr_m, y_lr)

    sg_r, sg_rmsd = compute_profile_metrics(e_cg_m, e_sg_m)
    lr_r, lr_rmsd = compute_profile_metrics(e_cg_m, e_lr_m)
    profile_metrics.append((r"Mean Cooling Rate Profile vs $y$", sg_r, sg_rmsd, lr_r, lr_rmsd))

    # Print summary table of profile metrics comparing CGHR vs Subgrid and LR
    print("\n" + "=" * 92)
    print(" PROFILE COMPARISON METRICS: CG HR vs SG (Subgrid) and CG HR vs LR")
    print("=" * 92)
    hdr = f"{'Profile':<36} | {'CGHR_Subgrid Pearson':<21} | {'CGHR_Subgrid RMSD':<18} | {'CGHR LR Pearson':<16} | {'CGHR LR RMSD':<14}"
    print(hdr)
    print("-" * len(hdr))
    csv_rows = []
    for name, s_p, s_rmsd, l_p, l_rmsd in profile_metrics:
        # Clean title for clean terminal display and CSV
        clean_name = name.replace("$", "").replace("\\rho", "rho").replace("\\mathbf{j}", "j").replace("\\mathbf{T}_x", "T_x").replace("\\mathbf{T}_y", "T_y").replace("\\nabla \\cdot", "div")
        s_p_str = f"{s_p:+.4f}" if np.isfinite(s_p) else "N/A"
        l_p_str = f"{l_p:+.4f}" if np.isfinite(l_p) else "N/A"
        print(f"{clean_name:<36} | {s_p_str:<21} | {s_rmsd:<18.4e} | {l_p_str:<16} | {l_rmsd:<14.4e}")
        csv_rows.append({
            "Profile": clean_name,
            "CGHR_Subgrid Pearson": s_p if np.isfinite(s_p) else "",
            "CGHR_Subgrid RMSD": s_rmsd,
            "CGHR LR Pearson": l_p if np.isfinite(l_p) else "",
            "CGHR LR RMSD": l_rmsd,
        })
    print("=" * 92 + "\n")

    # Save to CSV
    csv_path = out_dir / "profile_comparison_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["Profile", "CGHR_Subgrid Pearson", "CGHR_Subgrid RMSD", "CGHR LR Pearson", "CGHR LR RMSD"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  Saved profile comparison metrics to: {csv_path}\n")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_yscale("log")
    ax.plot(y_cg_hr, e_cg_m, lw=2, ls="-", marker="^", markersize=4, label=rf"CG HR ($\Sigma_c = {int_cg:.2e}$)")
    ax.fill_between(y_cg_hr, np.clip(e_cg_m - e_cg_s, 1e-35, None), e_cg_m + e_cg_s, alpha=0.25)

    ax.plot(y_sg, e_sg_m, lw=2, ls="-.", marker="o", markersize=5, label=rf"SG CNN ($\Sigma_c = {int_sg:.2e}$)")
    ax.fill_between(y_sg, np.clip(e_sg_m - e_sg_s, 1e-35, None), e_sg_m + e_sg_s, alpha=0.25)

    ax.plot(y_lr, e_lr_m, lw=2, ls="--", marker="s", markersize=5, label=rf"LR ISM ($\Sigma_c = {int_lr:.2e}$)")
    ax.fill_between(y_lr, np.clip(e_lr_m - e_lr_s, 1e-35, None), e_lr_m + e_lr_s, alpha=0.25)

    ax.set_xlabel(r"$y \ [\mathrm{pc}]$", fontsize=13)
    ax.set_ylabel(r"$\langle n^2 \Lambda(T) \rangle \ [\mathrm{erg} \ \mathrm{cm}^{-3} \ \mathrm{s}^{-1}]$", fontsize=13)
    ax.set_title(r"Mean Cooling Rate Profile vs $y$", fontsize=14, weight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "emissivity_profile_vs_y.png", dpi=200)
    plt.close(fig)
    print("  Saved emissivity_profile_vs_y.png")

    # =========================================================================
    # PLOT 7: Static Snapshots (all_fields_snapshot.png & 5-panel Subgrid PDF)
    # =========================================================================
    print("\n[10] Generating static snapshots...")
    fields_hr = [hr_rho, hr_temp, hr_pres, hr_ux, hr_uy, hr_ien]
    fields_cg_hr = [cg_hr_rho, cg_hr_temp, cg_hr_pres, cg_hr_ux, cg_hr_uy, cg_hr_ien]
    fields_sg = [sg_rho, sg_temp, sg_pres, sg_ux, sg_uy, sg_ien]
    fields_lr = [lr_rho, lr_temp, lr_pres, lr_ux, lr_uy, lr_ien]
    titles = ["Density", "Temperature", "Pressure", "Ux", "Uy", "Internal Energy"]

    fig, axs = plt.subplots(6, 4, figsize=(12, 20))
    for i in range(6):
        f0_hr = fields_hr[i][0]
        f0_cg = fields_cg_hr[i][0]
        f0_sg = fields_sg[i][0]
        f0_lr = fields_lr[i][0]

        arr0 = np.concatenate([f0_hr.flatten(), f0_cg.flatten(), f0_sg.flatten(), f0_lr.flatten()])
        use_log = (i == 0 or i == 1)
        vmin0, vmax0, norm0 = compute_color_limits(arr0, use_log=use_log)

        im0 = axs[i, 0].imshow(f0_hr, origin="lower", cmap="plasma", norm=norm0)
        axs[i, 0].set_title(f"HR {titles[i]}")
        plt.colorbar(im0, ax=axs[i, 0], fraction=0.035, pad=0.02)

        im1 = axs[i, 1].imshow(f0_cg, origin="lower", cmap="plasma", norm=norm0)
        axs[i, 1].set_title(f"CG HR {titles[i]}")
        plt.colorbar(im1, ax=axs[i, 1], fraction=0.035, pad=0.02)

        im2 = axs[i, 2].imshow(f0_sg, origin="lower", cmap="plasma", norm=norm0)
        axs[i, 2].set_title(f"SG {titles[i]}")
        plt.colorbar(im2, ax=axs[i, 2], fraction=0.035, pad=0.02)

        im3 = axs[i, 3].imshow(f0_lr, origin="lower", cmap="plasma", norm=norm0)
        axs[i, 3].set_title(f"LR {titles[i]}")
        plt.colorbar(im3, ax=axs[i, 3], fraction=0.035, pad=0.02)

    plt.tight_layout()
    plt.savefig(out_dir / "all_fields_snapshot.png", dpi=200)
    plt.close(fig)
    print("  Saved all_fields_snapshot.png")

    # 5-Panel subgrid predicted PDF snapshot t=0
    print("  Generating subgrid_predicted_pdf_snapshot_t0.png...")
    fig = plt.figure(figsize=(24, 10))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.1, 0.9, 0.9, 0.9, 0.9], wspace=0.22,
                          left=0.03, right=0.97, top=0.90, bottom=0.08)

    ny_cg, nx_cg = RESOLUTION[0], RESOLUTION[1]
    nb = out_channels
    log_temp_centers = 0.5 * (np.log10(T_edges[:-1]) + np.log10(T_edges[1:]))

    cmap_temp = plt.get_cmap("inferno")
    norm_temp = colors.Normalize(vmin=3.0, vmax=7.0)
    cmap_cool = plt.get_cmap("viridis")
    norm_cool = colors.LogNorm(vmin=cool_vmin, vmax=cool_vmax)
    cmap_gate = plt.get_cmap("plasma")
    norm_gate = colors.Normalize(vmin=0.0, vmax=1.0)
    cmap_active = plt.get_cmap("viridis")
    norm_active = colors.Normalize(vmin=0.0, vmax=1.0)

    # Mini PDF grid via vectorized LineCollection (1 axes instead of 512)
    ax_pdf_grid = fig.add_subplot(gs[0])
    ax_pdf_grid.set_title("Predicted Subgrid PDFs", fontsize=14, weight="bold")
    bg_im_0, lc_0 = setup_tiled_pdf_panel(ax_pdf_grid, ny_cg=ny_cg, nx_cg=nx_cg, nb_bins=nb,
                                          logt_start=LOGT_ACTIVE_START, logt_end=LOGT_ACTIVE_END,
                                          t_edges=T_edges)
    rgba_0, segs_0, colors_0 = compute_pdf_panel_arrays(pred_pdf_all[0], cmap_temp, norm_temp,
                                                        log_temp_centers, ny_cg=ny_cg, nx_cg=nx_cg, nb_bins=nb)
    bg_im_0.set_data(rgba_0)
    lc_0.set_segments(segs_0)
    lc_0.set_colors(colors_0)

    # Panel 1: Temperature map
    ax_temp = fig.add_subplot(gs[1])
    im_temp = ax_temp.imshow(np.log10(sg_temp[0]), origin="lower", cmap=cmap_temp, norm=norm_temp, aspect="auto")
    ax_temp.set_title(r"Subgrid $\log_{10} T$", fontsize=14, weight="bold")
    plt.colorbar(im_temp, ax=ax_temp, fraction=0.046, pad=0.04)

    # Panel 2: Cooling map
    ax_cool = fig.add_subplot(gs[2])
    im_cool = ax_cool.imshow(np.clip(emis_sg[0], cool_vmin, None), origin="lower", cmap=cmap_cool, norm=norm_cool, aspect="auto")
    ax_cool.set_title("Subgrid Cooling Rate", fontsize=15, weight="bold")
    plt.colorbar(im_cool, ax=ax_cool, fraction=0.046, pad=0.04)

    # Panel 3: Gate map
    ax_gate = fig.add_subplot(gs[3])
    im_gate = ax_gate.imshow(pred_gate_all[0], origin="lower", cmap=cmap_gate, norm=norm_gate, aspect="auto")
    ax_gate.set_title("Subgrid Gate Map", fontsize=15, weight="bold")
    plt.colorbar(im_gate, ax=ax_gate, fraction=0.046, pad=0.04)

    # Panel 4: Active PDF mass
    ax_active = fig.add_subplot(gs[4])
    act_mass = pdf_mass_in_active_range(pred_pdf_all[0], T_edges)
    im_active = ax_active.imshow(act_mass, origin="lower", cmap=cmap_active, norm=norm_active, aspect="auto")
    ax_active.set_title("Active PDF Mass", fontsize=15, weight="bold")
    plt.colorbar(im_active, ax=ax_active, fraction=0.046, pad=0.04)

    fig.suptitle(f"Subgrid Predicted Temperature PDF Grid ({CELL_LABEL}), T, Cooling, Gate, & Active Mass | t = {t_restart_myr[0]:.2f} Myr", fontsize=18, weight="bold")
    plt.savefig(out_dir / "subgrid_predicted_pdf_snapshot_t0.png", dpi=200)
    plt.close(fig)
    print("  Saved subgrid_predicted_pdf_snapshot_t0.png")

    # =========================================================================
    # 8. ANIMATIONS (MP4 video rendering)
    # =========================================================================
    print("\n[11] Rendering full MP4 animation suite...")

    anim_frames = range(0, nt, FRAME_STEP)

    # (A) temperature_field_evolution.mp4 (Temperature map with velocity streamlines)
    x_cg = np.linspace(x1min, x1max, nx_cg)
    y_cg = np.linspace(x2min, x2max, ny_cg)
    ny_hr_sim, nx_hr_sim = hr_temp.shape[1], hr_temp.shape[2]
    x_hr_full = np.linspace(x1min, x1max, nx_hr_sim)
    y_hr_full = np.linspace(x2min, x2max, ny_hr_sim)
    step_hr_y = max(1, ny_hr_sim // 64)
    step_hr_x = max(1, nx_hr_sim // 32)
    sy_hr, sx_hr = slice(0, ny_hr_sim, step_hr_y), slice(0, nx_hr_sim, step_hr_x)
    x_hr_sub = x_hr_full[sx_hr]
    y_hr_sub = y_hr_full[sy_hr]

    fig_a, axs_a = plt.subplots(1, 4, figsize=(14, 4.5))
    lbls_a = [
        f"HR ({nx_hr_sim}x{ny_hr_sim}) Temperature",
        f"CG HR ({CELL_LABEL}) Temperature",
        f"SG ({CELL_LABEL}) Temperature",
        f"LR ({CELL_LABEL}) Temperature",
    ]
    ims_a = []
    stream_artists_a = [None, None, None, None]
    for ax, lbl in zip(axs_a, lbls_a):
        im = ax.imshow(np.zeros((2, 2)), origin="lower", extent=[x1min, x1max, x2min, x2max],
                       cmap="inferno", vmin=3.0, vmax=7.0, aspect="auto")
        ax.set_title(lbl)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ims_a.append(im)
    axs_a[0].set_ylabel(r"$y \ [\mathrm{pc}]$")
    plt.tight_layout()

    def update_temperature_field(frame):
        t_sims = [hr_temp[frame], cg_hr_temp[frame], sg_temp[frame], lr_temp[frame]]
        ux_sims = [hr_ux[frame], cg_hr_ux[frame], sg_ux[frame], lr_ux[frame]]
        uy_sims = [hr_uy[frame], cg_hr_uy[frame], sg_uy[frame], lr_uy[frame]]
        for i, (ax, im, t_arr, ux_arr, uy_arr) in enumerate(zip(axs_a, ims_a, t_sims, ux_sims, uy_sims)):
            im.set_data(np.log10(t_arr))
            # streamplot has no incremental update API; the old artists must be removed and redrawn.
            # StreamplotSet.arrows is a non-attached PatchCollection facade in this matplotlib
            # version (NotImplementedError on .remove()) -- the actual arrow patches live in
            # ax.patches, so they must be cleared from there instead.
            if stream_artists_a[i] is not None:
                stream_artists_a[i].lines.remove()
                for patch in list(ax.patches):
                    patch.remove()
            if i == 0:
                sp = ax.streamplot(x_hr_sub, y_hr_sub, ux_arr[sy_hr, sx_hr], uy_arr[sy_hr, sx_hr],
                                   color="white", density=0.7, linewidth=0.8, arrowsize=0.8)
            else:
                sp = ax.streamplot(x_cg, y_cg, ux_arr, uy_arr,
                                   color="white", density=0.7, linewidth=0.8, arrowsize=0.8)
            stream_artists_a[i] = sp
            # Lock limits strictly to avoid streamplot autoscaling jitter across frames
            ax.set_xlim(x1min, x1max)
            ax.set_ylim(x2min, x2max)
            ax.set_xlabel(f"Timestep: {frame}")
        return []

    save_animation_funcanim(fig_a, update_temperature_field, anim_frames,
                            str(out_dir / "temperature_field_evolution.mp4"), fps=10, blit=False)

    # (B) density_evolution.mp4
    vmin_h, vmax_h = hr_rho[0][hr_rho[0] > 0].min(), hr_rho[0].max()
    arrs_b = [hr_rho, cg_hr_rho, sg_rho, lr_rho]
    lbls_b = [
        f"HR ({hr_rho.shape[1]}x{hr_rho.shape[2]}) Density",
        f"CG HR ({CELL_LABEL}) Density",
        f"SG ({CELL_LABEL}) Density",
        f"LR ({CELL_LABEL}) Density",
    ]
    fig_b, axs_b = plt.subplots(1, 4, figsize=(14, 4.5))
    ims_b, xlabels_b = [], []
    for ax, arr, lbl in zip(axs_b, arrs_b, lbls_b):
        im = ax.imshow(arr[0], origin="lower", cmap="plasma", norm=LogNorm(vmin=vmin_h, vmax=vmax_h), animated=True)
        ax.set_title(lbl)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        xlabel = ax.set_xlabel("Timestep: 0")
        xlabel.set_animated(True)
        ims_b.append(im)
        xlabels_b.append(xlabel)
    plt.tight_layout()

    def update_density(frame):
        artists = []
        for im, xlabel, arr in zip(ims_b, xlabels_b, arrs_b):
            im.set_data(arr[frame])
            xlabel.set_text(f"Timestep: {frame}")
            artists.append(im)
            artists.append(xlabel)
        return artists

    save_animation_funcanim(fig_b, update_density, anim_frames,
                            str(out_dir / "density_evolution.mp4"), fps=10, blit=True)

    # (C) cooling_rate_evolution.mp4
    fig_c = plt.figure(figsize=(19, 5))
    gs_c = fig_c.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.04], wspace=0.25, top=0.86, bottom=0.15, left=0.05, right=0.94)
    norm_c = LogNorm(vmin=cool_vmin, vmax=cool_vmax)
    cmap_c = plt.get_cmap("viridis")
    suptitle_c = fig_c.suptitle("", fontsize=16, weight="bold", y=0.96)
    suptitle_c.set_animated(True)

    axes_c = [fig_c.add_subplot(gs_c[i]) for i in range(4)]
    fields_c = [emis_hr, emis_cg_hr, emis_sg, emis_lr]
    lbls_c = [f"HR ({hr_rho.shape[1]}x{hr_rho.shape[2]})", f"CG HR ({CELL_LABEL})", f"SG ({CELL_LABEL})", f"LR ({CELL_LABEL})"]
    ims_c = []
    for ax, fld, lbl in zip(axes_c, fields_c, lbls_c):
        im = ax.imshow(np.clip(fld[0], cool_vmin, None), origin="lower", cmap=cmap_c, norm=norm_c, animated=True)
        ax.set_title(lbl, fontsize=13)
        ax.set_xlabel("Y (pixels)", fontsize=11)
        ax.set_ylabel("X (pixels)", fontsize=11)
        ims_c.append(im)

    cbar_ax_c = fig_c.add_subplot(gs_c[4])
    sm_c = plt.cm.ScalarMappable(cmap=cmap_c, norm=norm_c)
    sm_c.set_array([])
    cbar_c = fig_c.colorbar(sm_c, cax=cbar_ax_c)
    cbar_c.set_label(r"Cooling Rate $n^2\Lambda(T)$ (erg / cm$^3$ / s)", fontsize=12)

    def update_cooling(frame):
        suptitle_c.set_text(rf"Cooling Rate Comparison | $t = {t_restart_myr[frame]:.2f}$ Myr")
        for im, fld in zip(ims_c, fields_c):
            im.set_data(np.clip(fld[frame], cool_vmin, None))
        return ims_c + [suptitle_c]

    save_animation_funcanim(fig_c, update_cooling, anim_frames,
                            str(out_dir / "cooling_rate_evolution.mp4"), fps=10, blit=True)

    # (D) temperature_pdf_evolution.mp4
    bins_p = np.logspace(4, 6, 150)
    x_centers_d = bins_p[:-1]
    fig_d, ax_d = plt.subplots(figsize=(7, 5))
    ax_d.set_xscale("log")
    ax_d.set_yscale("log")
    ax_d.set_xlabel("Temperature [K]", fontsize=12)
    ax_d.set_ylabel("PDF (volume-weighted)", fontsize=12)
    ax_d.set_ylim(1e-7, 1e-3)
    ax_d.set_xlim(bins_p[0], bins_p[-1])
    ax_d.grid(True, which="both", ls="--", alpha=0.5)
    title_d = ax_d.set_title("", fontsize=13, weight="bold")
    title_d.set_animated(True)

    (line_hr_d,) = ax_d.plot([], [], lw=2.0, ls="-", marker="^", markersize=4, label="HR", animated=True)
    (line_cg_d,) = ax_d.plot([], [], lw=2.0, ls=":", marker="d", markersize=4, label=CG_LABEL, animated=True)
    (line_sg_d,) = ax_d.plot([], [], lw=2.0, ls="-.", marker="o", markersize=4, label=SG_LABEL, animated=True)
    (line_lr_d,) = ax_d.plot([], [], lw=2.0, ls="--", marker="s", markersize=4, label=LR_LABEL, animated=True)
    ax_d.legend(fontsize=10)
    plt.tight_layout()

    def update_temp_pdf(frame):
        h_hr, _ = np.histogram(hr_temp[frame].ravel(), bins=bins_p, density=True)
        h_cg, _ = np.histogram(cg_hr_temp[frame].ravel(), bins=bins_p, density=True)
        h_sg, _ = np.histogram(sg_temp[frame].ravel(), bins=bins_p, density=True)
        h_lr, _ = np.histogram(lr_temp[frame].ravel(), bins=bins_p, density=True)
        line_hr_d.set_data(x_centers_d, h_hr)
        line_cg_d.set_data(x_centers_d, h_cg)
        line_sg_d.set_data(x_centers_d, h_sg)
        line_lr_d.set_data(x_centers_d, h_lr)
        title_d.set_text(f"Temperature PDF | Time step {frame + 1} (t = {t_restart_myr[frame]:.2f} Myr)")
        return [line_hr_d, line_cg_d, line_sg_d, line_lr_d, title_d]

    save_animation_funcanim(fig_d, update_temp_pdf, anim_frames,
                            str(out_dir / "temperature_pdf_evolution.mp4"), fps=10, blit=True)

    # (E) subgrid_predicted_pdf_evolution.mp4
    fig_e = plt.figure(figsize=(24, 10))
    gs_e = fig_e.add_gridspec(1, 5, width_ratios=[1.1, 0.9, 0.9, 0.9, 0.9], wspace=0.22,
                              left=0.03, right=0.97, top=0.90, bottom=0.08)

    ax_pdf_grid_e = fig_e.add_subplot(gs_e[0])
    ax_pdf_grid_e.set_title("Predicted Subgrid PDFs", fontsize=14, weight="bold")
    bg_im_e, lc_e = setup_tiled_pdf_panel(ax_pdf_grid_e, ny_cg=ny_cg, nx_cg=nx_cg, nb_bins=nb,
                                          logt_start=LOGT_ACTIVE_START, logt_end=LOGT_ACTIVE_END,
                                          t_edges=T_edges)
    bg_im_e.set_animated(True)
    lc_e.set_animated(True)

    ax_temp_e = fig_e.add_subplot(gs_e[1])
    im_temp_e = ax_temp_e.imshow(np.log10(sg_temp[0]), origin="lower", cmap=cmap_temp, norm=norm_temp, aspect="auto", animated=True)
    ax_temp_e.set_title(r"Subgrid $\log_{10} T$", fontsize=14, weight="bold")
    plt.colorbar(im_temp_e, ax=ax_temp_e, fraction=0.046, pad=0.04)

    ax_cool_e = fig_e.add_subplot(gs_e[2])
    im_cool_e = ax_cool_e.imshow(np.clip(emis_sg[0], cool_vmin, None), origin="lower", cmap=cmap_cool, norm=norm_cool, aspect="auto", animated=True)
    ax_cool_e.set_title("Subgrid Cooling Rate", fontsize=14, weight="bold")
    plt.colorbar(im_cool_e, ax=ax_cool_e, fraction=0.046, pad=0.04)

    ax_gate_e = fig_e.add_subplot(gs_e[3])
    im_gate_e = ax_gate_e.imshow(pred_gate_all[0], origin="lower", cmap=cmap_gate, norm=norm_gate, aspect="auto", animated=True)
    ax_gate_e.set_title("Subgrid Gate Map", fontsize=14, weight="bold")
    plt.colorbar(im_gate_e, ax=ax_gate_e, fraction=0.046, pad=0.04)

    ax_active_e = fig_e.add_subplot(gs_e[4])
    im_active_e = ax_active_e.imshow(pdf_mass_in_active_range(pred_pdf_all[0], T_edges), origin="lower", cmap=cmap_active, norm=norm_active, aspect="auto", animated=True)
    ax_active_e.set_title("Active PDF Mass", fontsize=14, weight="bold")
    plt.colorbar(im_active_e, ax=ax_active_e, fraction=0.046, pad=0.04)

    suptitle_e = fig_e.suptitle("", fontsize=16, weight="bold")
    suptitle_e.set_animated(True)

    def update_subgrid_pdf(frame):
        rgba, segs, seg_colors = compute_pdf_panel_arrays(pred_pdf_all[frame], cmap_temp, norm_temp,
                                                           log_temp_centers, ny_cg=ny_cg, nx_cg=nx_cg, nb_bins=nb)
        bg_im_e.set_data(rgba)
        lc_e.set_segments(segs)
        lc_e.set_colors(seg_colors)
        im_temp_e.set_data(np.log10(sg_temp[frame]))
        im_cool_e.set_data(np.clip(emis_sg[frame], cool_vmin, None))
        im_gate_e.set_data(pred_gate_all[frame])
        im_active_e.set_data(pdf_mass_in_active_range(pred_pdf_all[frame], T_edges))
        suptitle_e.set_text(f"Subgrid Predicted Temperature PDF Grid ({CELL_LABEL}), T, Cooling, Gate, & Active Mass | t = {t_restart_myr[frame]:.2f} Myr")
        return [bg_im_e, lc_e, im_temp_e, im_cool_e, im_gate_e, im_active_e, suptitle_e]

    save_animation_funcanim(fig_e, update_subgrid_pdf, anim_frames,
                            str(out_dir / "subgrid_predicted_pdf_evolution.mp4"), fps=10, blit=True)

    # =========================================================================
    # CLIPPING DIAGNOSTICS: where/when the cooling-rate floor cap engaged
    # =========================================================================
    print("\n[7] Checking for cooling-rate clip log...")
    clip_log_path = os.environ.get("CLIP_LOG_PATH", str(PROJECT_ROOT / "outputs" / "clip_events.csv"))
    clip_data = load_clip_log(clip_log_path)
    if clip_data is None:
        print(f"  No clip log found at {clip_log_path} -- skipping "
              f"(only the live source_module.py sim path writes this).")
    else:
        plot_clip_diagnostics(clip_data, out_dir)

    print("\n" + "=" * 75)
    print(f" ALL DIAGNOSTICS & ANIMATIONS COMPLETED! Saved to: {out_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
