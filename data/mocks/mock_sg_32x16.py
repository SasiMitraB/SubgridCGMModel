"""
mock_sg_32x16.py
----------------
Benchmark script comparing:
  - HR reference  (1024×2048, vshear_31_coldfrac_0.33) — ergane, streaming, CG on-the-fly
  - SG-tiled      (32×16, CNN subgrid restart)          — ergane, all frames loaded
  - LR-ISM        (32×16, ISM cooling restart)          — ergane, all frames loaded

All three simulations are loaded using ergane (ergane.SimulationData).
HR frames are streamed one at a time to avoid memory pressure (~50 MB/frame).
SG/LR frames are loaded all at once since the 32×16 arrays are tiny (~8 MB total).

No caches are built or required.
"""

from __future__ import annotations

import gc
import os
import sys
import subprocess
import tempfile
import shutil
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from tqdm import tqdm
import multiprocessing

# ---------------------------------------------------------------------------
# Path setup — add project root so ergane and model imports work
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = (_HERE / "../..").resolve()

sys.path.insert(0, str(PROJECT_ROOT))                            # ergane
sys.path.insert(0, str(PROJECT_ROOT / "models" / "conv_nn"))    # pdf_cnn

from ergane import SimulationData, Frame
from pdf_cnn import (
    compute_cooling_rate,
    lambda_cool,
    out_channels,
    snapshot_pred_16x8,
)

# ---------------------------------------------------------------------------
# Environment / configuration
# ---------------------------------------------------------------------------
HR_DS            = int(os.environ.get("PDF_CNN_DOWNSAMPLE", "64"))
RESTART_TIME_MYR = 5.0
BIN_DT_MYR       = 0.01          # matches bin_w_dt in config

LOGT_ACTIVE_START = float(os.environ.get("LOGT_ACTIVE_START", np.log10(1.05e4)))
LOGT_ACTIVE_END   = float(os.environ.get("LOGT_ACTIVE_END",   np.log10(0.95e6)))

HR_SIM_DIR = Path(os.environ.get(
    "HR_SIM_OUTPUT",
    str(PROJECT_ROOT / "simulation_outputs"
        / "hr_gpu_sweep_1024x2048_2xlength" / "vshear_31_coldfrac_0.33"),
))
SG_BIN_DIR = Path(os.environ.get(
    "SG_BIN_PATH",
    str(PROJECT_ROOT / "simulation_outputs"
        / "subgrid_32x16_vshear31_cf033" / "sg_tiled"),
))
LR_BIN_DIR = Path(os.environ.get(
    "LR_BIN_PATH",
    str(PROJECT_ROOT / "simulation_outputs"
        / "subgrid_32x16_vshear31_cf033" / "lr_build_ism"),
))
save_path = Path(os.environ.get(
    "SG_MOCKS_DIR",
    str(PROJECT_ROOT / "simulation_outputs"
        / "subgrid_32x16_vshear31_cf033" / "mocks"),
))
save_path.mkdir(parents=True, exist_ok=True)
save_str = str(save_path) + "/"

DEFAULT_MODEL_SAVES_DIR = str(PROJECT_ROOT / "outputs" / "model_saves" / "pdf_model_saves")
MODEL_SAVES_DIR = os.environ.get("MODEL_SAVES_DIR", DEFAULT_MODEL_SAVES_DIR)


def _find_available_model(save_dir: str):
    """
    Find the first available (resolution, downsample) model configuration.
    Checks user-specified env vars first if present, then candidate defaults:
      1. resolution=(1024, 512), downsample=64
      2. resolution=(512, 256), downsample=32
    """
    candidates = []

    if "PDF_CNN_RESOLUTION" in os.environ and "PDF_CNN_DOWNSAMPLE" in os.environ:
        _res_parts = os.environ["PDF_CNN_RESOLUTION"].split(",")
        cand_env = ((int(_res_parts[0]), int(_res_parts[1])), int(os.environ["PDF_CNN_DOWNSAMPLE"]))
        candidates.append(cand_env)

    default_candidates = [
        ((1024, 512), 64),
        ((512, 256), 32),
    ]
    for cand in default_candidates:
        if cand not in candidates:
            candidates.append(cand)

    for res, ds in candidates:
        norm_prefix = f"cnn_{res}_{ds}"
        model_path = os.path.join(save_dir, f"{norm_prefix}.pth")
        mean_path = os.path.join(save_dir, f"{norm_prefix}_input_mean.npy")
        std_path = os.path.join(save_dir, f"{norm_prefix}_input_std.npy")

        if os.path.isfile(model_path) and os.path.isfile(mean_path) and os.path.isfile(std_path):
            return res, ds, norm_prefix, model_path, mean_path, std_path

    checked_str = ", ".join([f"cnn_{res}_{ds}" for res, ds in candidates])
    raise FileNotFoundError(
        f"No suitable CNN model found in {save_dir}. Checked configurations: {checked_str}"
    )


try:
    CNN_FINE_RES, CNN_DS, _, _, _, _ = _find_available_model(MODEL_SAVES_DIR)
except FileNotFoundError:
    if MODEL_SAVES_DIR != DEFAULT_MODEL_SAVES_DIR:
        CNN_FINE_RES, CNN_DS, _, _, _, _ = _find_available_model(DEFAULT_MODEL_SAVES_DIR)
        MODEL_SAVES_DIR = DEFAULT_MODEL_SAVES_DIR
    else:
        raise

# CNN tile parameters (model trained on 16x8 coarse crops)
TILE_ROWS = CNN_FINE_RES[0] // CNN_DS  # 16
TILE_COLS = CNN_FINE_RES[1] // CNN_DS  # 8

print(f"[mock_sg_32x16] Using CNN model: fine_resolution={CNN_FINE_RES}, downsample={CNN_DS}, "
      f"tile_size={TILE_ROWS}×{TILE_COLS} from {MODEL_SAVES_DIR}")

T_edges   = np.logspace(3.0, 7.0, out_channels + 1)
T_centers = np.sqrt(T_edges[:-1] * T_edges[1:])

# ---------------------------------------------------------------------------
# Physical unit constants (code → CGS)
# ---------------------------------------------------------------------------
m_H    = 1.6726219e-24
k_B    = 1.380649e-16
M_sun  = 1.98847e33
yr     = 3.15576e7
pc     = 3.08568e18
kpc    = 3.08568e21

L_cgs  = 3.08568e18
M_cgs  = 4.91417e31
T_cgs  = 3.15576e13
mu     = 0.62

V_cgs                  = L_cgs / T_cgs
RHO_cgs                = M_cgs / L_cgs**3
P_cgs                  = RHO_cgs * V_cgs**2
len_to_pc              = L_cgs / pc
n_to_cm3               = RHO_cgs / (mu * m_H)
T_to_K                 = V_cgs**2 * mu * m_H / k_B
P_over_kB_to_K_cm3    = P_cgs / k_B
vel_to_km_s            = V_cgs / 1e5
mflux_to_Msun_yr_kpc2  = (RHO_cgs * V_cgs) / (M_sun / (yr * kpc**2))
unit_fix               = 1.975e27

# Domain extents (pc) — 2× length domain matching vshear_31_coldfrac_0.33
Lx_pc = 20.0    # |x1max - x1min| = 10 - (-10)
Ly_pc = 40.0    # |x2max - x2min| = 20 - (-20)

# ---------------------------------------------------------------------------
# Coarse-grain helper (block-mean, no skimage)
# ---------------------------------------------------------------------------

def cg2d(arr: np.ndarray, ds: int) -> np.ndarray:
    """Block-mean coarse-grain a 2D array (ny, nx) by factor ds."""
    ny, nx = arr.shape
    return arr.reshape(ny // ds, ds, nx // ds, ds).mean(axis=(1, 3))


# ---------------------------------------------------------------------------
# Helper: extract named fields from an ergane Frame
# ---------------------------------------------------------------------------

def _frame_fields(frame: Frame) -> dict[str, np.ndarray]:
    """
    Pull all needed physical fields from an ergane Frame.
    Conserved fields are derived from primitives (equivalent for block-mean CG).
    """
    rho  = np.asarray(frame.density,     dtype=np.float64)
    pres = np.asarray(frame.pressure,    dtype=np.float64)
    eint = np.asarray(frame.eint,        dtype=np.float64)
    ux   = np.asarray(frame.velx,        dtype=np.float64)
    uy   = np.asarray(frame.vely,        dtype=np.float64)
    temp = np.asarray(frame.temperature, dtype=np.float64)
    ps   = np.asarray(frame.scalars.get("scalar_00", np.zeros_like(rho)),
                      dtype=np.float64)
    # scalar_01 is fmcl (passive mixing fraction) for SG; absent for HR/LR
    fmcl_scalar = frame.scalars.get("scalar_01", None)
    fmcl = (np.asarray(fmcl_scalar, dtype=np.float64)
            if fmcl_scalar is not None else (temp < 1e5).astype(np.float64))

    return {
        "rho":       rho,
        "pres":      pres,
        "eint":      eint,
        "ux":        ux,
        "uy":        uy,
        "temp":      temp,
        "ps":        ps,
        "fmcl":      fmcl,
        # conserved — derived from primitives
        "cons_rho":  rho,
        "cons_mx":   rho * ux,
        "cons_my":   rho * uy,
        "cons_ener": eint + 0.5 * rho * (ux**2 + uy**2),
        "cons_ps":   rho * ps,
    }


# ---------------------------------------------------------------------------
# Helper: load ALL frames from a small (LR/SG) ergane sim into numpy arrays
# ---------------------------------------------------------------------------

def load_small_sim(
    sim: SimulationData,
    frame_nums: list[int],
    desc: str = "",
) -> dict[str, np.ndarray]:
    """
    Load every frame in frame_nums from sim and stack into (n, ny, nx) arrays.
    Suitable for small grids (32×16) where all frames fit easily in memory.
    """
    n = len(frame_nums)
    first_f = _frame_fields(sim.get_frame(frame_nums[0]))
    shape   = (n, *next(iter(first_f.values())).shape)

    arrays: dict[str, np.ndarray] = {
        k: np.zeros(shape, dtype=np.float32) for k in first_f
    }
    for k, arr in first_f.items():
        arrays[k][0] = arr

    for i, num in enumerate(tqdm(frame_nums[1:], desc=desc or "Loading frames"), 1):
        flds = _frame_fields(sim.get_frame(num))
        for k, arr in flds.items():
            arrays[k][i] = arr

    return arrays


# ---------------------------------------------------------------------------
# Open all three sims with ergane
# ---------------------------------------------------------------------------
print(f"[mock_sg_32x16] Opening ergane sims ...")
print(f"  HR : {HR_SIM_DIR}")
print(f"  SG : {SG_BIN_DIR}")
print(f"  LR : {LR_BIN_DIR}")

hr_sim = SimulationData(datafolder=str(HR_SIM_DIR))
sg_sim = SimulationData(datafolder=str(SG_BIN_DIR))
lr_sim = SimulationData(datafolder=str(LR_BIN_DIR))

# Dynamic resolutions from simulation data
hr_resolution = (hr_sim.ny, hr_sim.nx)  # (2048, 1024)
resolution    = (sg_sim.ny, sg_sim.nx)  # (32, 16)
lr_resolution = (lr_sim.ny, lr_sim.nx)  # (32, 16)
HR_DS         = hr_sim.ny // sg_sim.ny  # 2048 // 32 = 64

print(f"[mock_sg_32x16] HR resolution: {hr_resolution}")
print(f"[mock_sg_32x16] SG resolution: {resolution}")
print(f"[mock_sg_32x16] LR resolution: {lr_resolution}")
print(f"[mock_sg_32x16] HR downsample: {HR_DS}x")

# Restrict HR to t >= 5 Myr (post-restart window)
hr_times = hr_sim.times   # reads ASCII headers only — fast
hr_frames_5_10 = [
    n for n, t in zip(hr_sim.frame_numbers, hr_times)
    if t >= RESTART_TIME_MYR - 0.001
]
print(f"[mock_sg_32x16] HR frames in [5,10] Myr: {len(hr_frames_5_10)} "
      f"(#{hr_frames_5_10[0]} → #{hr_frames_5_10[-1]})")

# SG / LR: all frames (the restarted sims start from their own frame 0)
sg_frames = sg_sim.frame_numbers
lr_frames = lr_sim.frame_numbers

# Align to common count
n_common = min(len(hr_frames_5_10), len(sg_frames), len(lr_frames))
hr_frames = hr_frames_5_10[:n_common]
sg_frames = sg_frames[:n_common]
lr_frames = lr_frames[:n_common]
nt        = n_common
print(f"[mock_sg_32x16] Common frames: {n_common}")

t_myr = RESTART_TIME_MYR + np.arange(n_common) * BIN_DT_MYR

# ---------------------------------------------------------------------------
# Load SG and LR (tiny arrays — load all at once)
# ---------------------------------------------------------------------------
print("[mock_sg_32x16] Loading SG (CNN-tiled) ...")
sg = load_small_sim(sg_sim, sg_frames, desc="SG frames")

print("[mock_sg_32x16] Loading LR (ISM cooling) ...")
lr = load_small_sim(lr_sim, lr_frames, desc="LR frames")

# Convenience aliases matching original mock_sg.py variable names
rho   = sg["rho"];   pres  = sg["pres"];  temp  = sg["temp"]
ien   = sg["eint"];  ux    = sg["ux"];    uy    = sg["uy"]
ps    = sg["ps"];    fmcl  = sg["fmcl"]
cons_rho  = sg["cons_rho"];  cons_momx = sg["cons_mx"]
cons_momy = sg["cons_my"];   cons_ener = sg["cons_ener"];  cons_ps = sg["cons_ps"]

lr_rho  = lr["rho"];   lr_pres = lr["pres"];  lr_temp = lr["temp"]
lr_ien  = lr["eint"];  lr_ux   = lr["ux"];    lr_uy   = lr["uy"]
lr_ps   = lr["ps"];    lr_fmcl = lr["fmcl"]
lr_cons_rho  = lr["cons_rho"];  lr_cons_momx = lr["cons_mx"]
lr_cons_momy = lr["cons_my"];   lr_cons_ener = lr["cons_ener"]; lr_cons_ps = lr["cons_ps"]

del sg, lr
gc.collect()

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def compute_color_limits(arr, use_log=False):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0, None
    if use_log:
        pos = finite[finite > 0]
        if pos.size:
            vmin, vmax = pos.min(), pos.max()
            if vmax <= vmin:
                vmax = vmin * 1.01
            return vmin, vmax, mcolors.LogNorm(vmin=vmin, vmax=vmax)
    vmin, vmax = finite.min(), finite.max()
    if vmax <= vmin:
        delta = abs(vmin) * 0.01 if vmin != 0 else 1.0
        vmin -= delta / 2; vmax += delta / 2
    return vmin, vmax, None


def parallel_save_animation(render_func, frames_list, output_path,
                             fps=10, num_workers=16):
    temp_dir = tempfile.mkdtemp()
    try:
        ctx    = multiprocessing.get_context("fork")
        worker = partial(render_func, temp_dir=temp_dir)
        with ctx.Pool(processes=num_workers) as pool:
            list(tqdm(pool.imap(worker, frames_list), total=len(frames_list),
                      desc=os.path.basename(output_path)))
        cmd = ["ffmpeg", "-y", "-r", str(fps),
               "-i", os.path.join(temp_dir, "frame_%04d.png"),
               "-c:v", "h264_nvenc", "-preset", "p4", "-pix_fmt", "yuv420p",
               output_path]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.decode())
    finally:
        shutil.rmtree(temp_dir)


def compute_mean_std(arr, logspace=False):
    a = np.log10(arr) if logspace else arr
    if a.ndim == 3:
        a = np.mean(a, axis=2)
    return a.mean(axis=0), a.std(axis=0)


def divergence(f, dx, dy):
    dFx_dx = np.gradient(f[0], dy, dx)[1]
    dFy_dy = np.gradient(f[1], dy, dx)[0]
    return dFx_dx + dFy_dy


def compute_E(rho, ux, uy, pres, gamma=5.0 / 3.0):
    return pres / (gamma - 1.0) + 0.5 * rho * (ux**2 + uy**2)


def make_derived_plot(hr_field, sg_field, lr_field, title, ylabel, ax, conv_factor=1.0):
    hr_mean, hr_std = compute_mean_std(hr_field * conv_factor)
    sg_mean, sg_std = compute_mean_std(sg_field * conv_factor)
    lr_mean, lr_std = compute_mean_std(lr_field * conv_factor)

    y_hr = np.linspace(-20.0, 20.0, len(hr_mean))
    y_sg = np.linspace(-20.0, 20.0, len(sg_mean))
    y_lr = np.linspace(-20.0, 20.0, len(lr_mean))

    ax.plot(y_hr, hr_mean, lw=2, ls="-",  marker="^", markersize=4,
            label=f"CG HR ({resolution[0]}×{resolution[1]})")
    ax.fill_between(y_hr, hr_mean - hr_std, hr_mean + hr_std, alpha=0.25)

    ax.plot(y_sg, sg_mean, lw=2, ls="-.", marker="o", markersize=5,
            label=f"SG ({resolution[0]}×{resolution[1]})")
    ax.fill_between(y_sg, sg_mean - sg_std, sg_mean + sg_std, alpha=0.25)

    ax.plot(y_lr, lr_mean, lw=2, ls="--", marker="s", markersize=5,
            label=f"LR ({lr_resolution[0]}×{lr_resolution[1]})")
    ax.fill_between(y_lr, lr_mean - lr_std, lr_mean + lr_std, alpha=0.25)

    ax.set_title(title)
    ax.set_xlabel(r"$y\ [\mathrm{pc}]$")
    ax.set_ylabel(ylabel)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()


# ---------------------------------------------------------------------------
# Step 1: Stream HR for profile plots & fluxes (accumulate tiny 32×16 CG arrays)
# ---------------------------------------------------------------------------
print("[mock_sg_32x16] Streaming HR → CG profile & flux arrays ...")

cg_hr: dict[str, np.ndarray] = {
    k: np.zeros((n_common, *resolution), dtype=np.float32)
    for k in ("rho", "temp", "pres", "ux", "uy", "eint",
              "fmcl", "cons_rho", "cons_mx", "cons_my", "cons_ener",
              "flux_mass_x", "flux_mass_y",
              "flux_T_xx", "flux_T_xy", "flux_T_yy",
              "flux_E_x", "flux_E_y")
}

cg_hr_div_mass = np.zeros((n_common, resolution[0]), dtype=np.float32)
cg_hr_div_momx = np.zeros((n_common, resolution[0]), dtype=np.float32)
cg_hr_div_momy = np.zeros((n_common, resolution[0]), dtype=np.float32)

dx_hr = Lx_pc / hr_resolution[1]
dy_hr = Ly_pc / hr_resolution[0]

for i, num in enumerate(tqdm(hr_frames, desc="HR → CG fields")):
    fr   = hr_sim.get_frame(num)
    flds = _frame_fields(fr)
    for k in ("rho", "temp", "pres", "ux", "uy", "eint",
              "fmcl", "cons_rho", "cons_mx", "cons_my", "cons_ener"):
        cg_hr[k][i] = cg2d(flds[k], HR_DS)

    # Nonlinear flux terms on fine grid
    rho_f  = flds["rho"]
    ux_f   = flds["ux"]
    uy_f   = flds["uy"]
    pres_f = flds["pres"]

    hr_mx  = rho_f * ux_f
    hr_my  = rho_f * uy_f
    hr_txx = hr_mx * ux_f + pres_f
    hr_txy = hr_mx * uy_f
    hr_tyy = hr_my * uy_f + pres_f
    hr_E   = compute_E(rho_f, ux_f, uy_f, pres_f)
    hr_Ex  = (hr_E + pres_f) * ux_f
    hr_Ey  = (hr_E + pres_f) * uy_f

    cg_hr["flux_mass_x"][i] = cg2d(hr_mx, HR_DS)
    cg_hr["flux_mass_y"][i] = cg2d(hr_my, HR_DS)
    cg_hr["flux_T_xx"][i]   = cg2d(hr_txx, HR_DS)
    cg_hr["flux_T_xy"][i]   = cg2d(hr_txy, HR_DS)
    cg_hr["flux_T_yy"][i]   = cg2d(hr_tyy, HR_DS)
    cg_hr["flux_E_x"][i]    = cg2d(hr_Ex, HR_DS)
    cg_hr["flux_E_y"][i]    = cg2d(hr_Ey, HR_DS)

    div_m  = np.mean(divergence([hr_mx, hr_my], dx_hr, dy_hr), axis=1)
    div_tx = np.mean(divergence([hr_txx, hr_txy], dx_hr, dy_hr), axis=1)
    div_ty = np.mean(divergence([hr_txy, hr_tyy], dx_hr, dy_hr), axis=1)

    cg_hr_div_mass[i] = div_m.reshape(resolution[0], HR_DS).mean(axis=1)
    cg_hr_div_momx[i] = div_tx.reshape(resolution[0], HR_DS).mean(axis=1)
    cg_hr_div_momy[i] = div_ty.reshape(resolution[0], HR_DS).mean(axis=1)

    del fr, flds, rho_f, ux_f, uy_f, pres_f, hr_mx, hr_my, hr_txx, hr_txy, hr_tyy, hr_E, hr_Ex, hr_Ey, div_m, div_tx, div_ty

# ---------------------------------------------------------------------------
# Step 2: Profile plots
# ---------------------------------------------------------------------------

quantities = [
    ("Density",     cg_hr["rho"]  * n_to_cm3, rho  * n_to_cm3, lr_rho  * n_to_cm3,
     r"$\log_{10}(n\ [\mathrm{cm}^{-3}])$", True),
    ("Temperature", cg_hr["temp"], temp, lr_temp,
     r"$\log_{10}(T\ [\mathrm{K}])$", True),
    ("Pressure",    cg_hr["pres"] * P_over_kB_to_K_cm3, pres * P_over_kB_to_K_cm3,
     lr_pres * P_over_kB_to_K_cm3, r"$P/k_B\ [\mathrm{K\ cm}^{-3}]$", False),
    ("Ux Velocity", cg_hr["ux"] * vel_to_km_s, ux * vel_to_km_s, lr_ux * vel_to_km_s,
     r"$u_x\ [\mathrm{km\ s}^{-1}]$", False),
    ("Uy Velocity", cg_hr["uy"] * vel_to_km_s, uy * vel_to_km_s, lr_uy * vel_to_km_s,
     r"$u_y\ [\mathrm{km\ s}^{-1}]$", False),
]

fig, axs = plt.subplots(5, 1, figsize=(9, 20))
plt.subplots_adjust(hspace=0.35)
for idx, (title, hr_arr, sg_arr, lr_arr, ylabel, is_log) in enumerate(quantities):
    hrm, hrs = compute_mean_std(hr_arr, logspace=is_log)
    sgm, sgs = compute_mean_std(sg_arr, logspace=is_log)
    lrm, lrs = compute_mean_std(lr_arr, logspace=is_log)

    y_cg = np.linspace(-20.0, 20.0, len(hrm))
    y_sg = np.linspace(-20.0, 20.0, len(sgm))
    y_lr = np.linspace(-20.0, 20.0, len(lrm))

    ax = axs[idx]
    ax.plot(y_cg, hrm, lw=2, ls="-",  marker="^", markersize=4,
            label=f"CG HR ({resolution[0]}×{resolution[1]})")
    ax.fill_between(y_cg, hrm - hrs, hrm + hrs, alpha=0.25)
    ax.plot(y_sg, sgm, lw=2, ls="-.", marker="o", markersize=5,
            label=f"SG ({resolution[0]}×{resolution[1]})")
    ax.fill_between(y_sg, sgm - sgs, sgm + sgs, alpha=0.25)
    ax.plot(y_lr, lrm, lw=2, ls="--", marker="s", markersize=5,
            label=f"LR ({lr_resolution[0]}×{lr_resolution[1]})")
    ax.fill_between(y_lr, lrm - lrs, lrm + lrs, alpha=0.25)
    ax.set_title(f"{title} (Avg over X) — Mean ± 1σ")
    ax.set_xlabel(r"$y\ [\mathrm{pc}]$"); ax.set_ylabel(ylabel)
    ax.grid(True, ls="--", alpha=0.5); ax.legend()

plt.tight_layout()
plt.savefig(save_str + "profiles_mean_with_std_all.png", dpi=200)
plt.close(fig)
print("profiles_mean_with_std_all.png saved")

# ---------------------------------------------------------------------------
# Step 3: Conserved profiles
# ---------------------------------------------------------------------------
quantities_cons = [
    ("Conserved Density",
     cg_hr["cons_rho"] * n_to_cm3, cons_rho * n_to_cm3, lr_cons_rho * n_to_cm3,
     r"$n\ [\mathrm{cm}^{-3}]$"),
    ("Conserved MomX",
     cg_hr["cons_mx"] * mflux_to_Msun_yr_kpc2, cons_momx * mflux_to_Msun_yr_kpc2,
     lr_cons_momx * mflux_to_Msun_yr_kpc2,
     r"$\rho u_x\ [M_\odot\ \mathrm{yr}^{-1}\ \mathrm{kpc}^{-2}]$"),
    ("Conserved MomY",
     cg_hr["cons_my"] * mflux_to_Msun_yr_kpc2, cons_momy * mflux_to_Msun_yr_kpc2,
     lr_cons_momy * mflux_to_Msun_yr_kpc2,
     r"$\rho u_y\ [M_\odot\ \mathrm{yr}^{-1}\ \mathrm{kpc}^{-2}]$"),
    ("Conserved Energy",
     cg_hr["cons_ener"] * P_cgs, cons_ener * P_cgs, lr_cons_ener * P_cgs,
     r"$E\ [\mathrm{erg\ cm}^{-3}]$"),
    ("fmcl (T < 1e5 K)", cg_hr["fmcl"], fmcl, lr_fmcl, r"$f_\mathrm{mcl}$"),
]

fig, axs = plt.subplots(5, 1, figsize=(9, 20))
plt.subplots_adjust(hspace=0.4)
for idx, (title, hr_arr, sg_arr, lr_arr, ylabel) in enumerate(quantities_cons):
    hrm, hrs = compute_mean_std(hr_arr)
    sgm, sgs = compute_mean_std(sg_arr)
    lrm, lrs = compute_mean_std(lr_arr)
    y_cg = np.linspace(-20.0, 20.0, len(hrm))
    y_sg = np.linspace(-20.0, 20.0, len(sgm))
    y_lr = np.linspace(-20.0, 20.0, len(lrm))
    ax = axs[idx]
    ax.plot(y_cg, hrm, lw=2, ls="-",  marker="^", markersize=4,
            label=f"CG HR ({resolution[0]}×{resolution[1]})")
    ax.fill_between(y_cg, hrm - hrs, hrm + hrs, alpha=0.25)
    ax.plot(y_sg, sgm, lw=2, ls="-.", marker="o", markersize=5,
            label=f"SG ({resolution[0]}×{resolution[1]})")
    ax.fill_between(y_sg, sgm - sgs, sgm + sgs, alpha=0.25)
    ax.plot(y_lr, lrm, lw=2, ls="--", marker="s", markersize=5,
            label=f"LR ({lr_resolution[0]}×{lr_resolution[1]})")
    ax.fill_between(y_lr, lrm - lrs, lrm + lrs, alpha=0.25)
    ax.set_title(f"{title} (Avg over X) — Mean ± 1σ")
    ax.set_xlabel(r"$y\ [\mathrm{pc}]$"); ax.set_ylabel(ylabel)
    ax.grid(True, ls="--", alpha=0.5); ax.legend()

plt.tight_layout()
plt.savefig(save_str + "conserved_quantities_mean_with_std.png", dpi=200)
plt.close(fig)
print("conserved_quantities_mean_with_std.png saved")

del cg_hr["cons_rho"], cg_hr["cons_mx"], cg_hr["cons_my"], cg_hr["cons_ener"]
gc.collect()

# ---------------------------------------------------------------------------
# Step 3b: Derived quantities (rho*ux, rho*ux*uy, p + rho*uy^2)
# ---------------------------------------------------------------------------
print("[mock_sg_32x16] Plotting derived quantities ...")
sg_rho_ux = rho * ux
lr_rho_ux = lr_rho * lr_ux

sg_rho_ux_uy = rho * ux * uy
lr_rho_ux_uy = lr_rho * lr_ux * lr_uy

sg_mom_flux_y = pres + rho * uy**2
lr_mom_flux_y = lr_pres + lr_rho * lr_uy**2

fig, axs = plt.subplots(3, 1, figsize=(9, 15))
plt.subplots_adjust(hspace=0.35)

make_derived_plot(
    cg_hr["flux_mass_x"],
    sg_rho_ux,
    lr_rho_ux,
    r"$\rho u_x$ (Avg over X) — Mean ± 1σ",
    r"$\rho u_x \ [M_\odot \ \mathrm{yr}^{-1} \ \mathrm{kpc}^{-2}]$",
    axs[0],
    conv_factor=mflux_to_Msun_yr_kpc2,
)

make_derived_plot(
    cg_hr["flux_T_xy"],
    sg_rho_ux_uy,
    lr_rho_ux_uy,
    r"$\rho u_x u_y$ (Avg over X) — Mean ± 1σ",
    r"$\rho u_x u_y \ [\mathrm{dyn} \ \mathrm{cm}^{-2}]$",
    axs[1],
    conv_factor=P_cgs,
)

make_derived_plot(
    cg_hr["flux_T_yy"],
    sg_mom_flux_y,
    lr_mom_flux_y,
    r"$p + \rho u_y^2$ (Avg over X) — Mean ± 1σ",
    r"$p + \rho u_y^2 \ [\mathrm{dyn} \ \mathrm{cm}^{-2}]$",
    axs[2],
    conv_factor=P_cgs,
)

plt.tight_layout()
plt.savefig(save_str + "derived_quantities_mean_with_std.png", dpi=200)
plt.close(fig)
print("derived_quantities_mean_with_std.png saved")

# ---------------------------------------------------------------------------
# Step 3c: Fluxes (Mass, Momentum, Energy Fluxes)
# ---------------------------------------------------------------------------
print("[mock_sg_32x16] Plotting full flux tensor and energy fluxes ...")
sg_mass_x = rho * ux
sg_mass_y = rho * uy
lr_mass_x = lr_rho * lr_ux
lr_mass_y = lr_rho * lr_uy

sg_T_xx = rho * ux**2 + pres
sg_T_xy = rho * ux * uy
sg_T_yy = rho * uy**2 + pres

lr_T_xx = lr_rho * lr_ux**2 + lr_pres
lr_T_xy = lr_rho * lr_ux * lr_uy
lr_T_yy = lr_rho * lr_uy**2 + lr_pres

sg_E = compute_E(rho, ux, uy, pres)
lr_E = compute_E(lr_rho, lr_ux, lr_uy, lr_pres)

sg_E_flux_x = (sg_E + pres) * ux
sg_E_flux_y = (sg_E + pres) * uy
lr_E_flux_x = (lr_E + lr_pres) * lr_ux
lr_E_flux_y = (lr_E + lr_pres) * lr_uy

fig, axs = plt.subplots(4, 2, figsize=(12, 16))
plt.subplots_adjust(hspace=0.35)

make_derived_plot(
    cg_hr["flux_mass_x"], sg_mass_x, lr_mass_x,
    r"Mass Flux ($\rho u_x$)",
    r"$\rho u_x \ [M_\odot \ \mathrm{yr}^{-1} \ \mathrm{kpc}^{-2}]$",
    axs[0, 0], conv_factor=mflux_to_Msun_yr_kpc2,
)
make_derived_plot(
    cg_hr["flux_mass_y"], sg_mass_y, lr_mass_y,
    r"Mass Flux ($\rho u_y$)",
    r"$\rho u_y \ [M_\odot \ \mathrm{yr}^{-1} \ \mathrm{kpc}^{-2}]$",
    axs[0, 1], conv_factor=mflux_to_Msun_yr_kpc2,
)

make_derived_plot(
    cg_hr["flux_T_xx"], sg_T_xx, lr_T_xx,
    r"Momentum Flux $T_{xx} = \rho u_x^2 + p$",
    r"$T_{xx} \ [\mathrm{dyn} \ \mathrm{cm}^{-2}]$",
    axs[1, 0], conv_factor=P_cgs,
)
make_derived_plot(
    cg_hr["flux_T_xy"], sg_T_xy, lr_T_xy,
    r"Momentum Flux $T_{xy} = \rho u_x u_y$",
    r"$T_{xy} \ [\mathrm{dyn} \ \mathrm{cm}^{-2}]$",
    axs[1, 1], conv_factor=P_cgs,
)

make_derived_plot(
    cg_hr["flux_T_xy"], sg_T_xy, lr_T_xy,
    r"Momentum Flux $T_{yx} = \rho u_x u_y$",
    r"$T_{yx} \ [\mathrm{dyn} \ \mathrm{cm}^{-2}]$",
    axs[2, 0], conv_factor=P_cgs,
)
make_derived_plot(
    cg_hr["flux_T_yy"], sg_T_yy, lr_T_yy,
    r"Momentum Flux $T_{yy} = \rho u_y^2 + p$",
    r"$T_{yy} \ [\mathrm{dyn} \ \mathrm{cm}^{-2}]$",
    axs[2, 1], conv_factor=P_cgs,
)

make_derived_plot(
    cg_hr["flux_E_x"], sg_E_flux_x, lr_E_flux_x,
    r"Energy Flux $(E+p)u_x$",
    r"$(E+p)u_x \ [\mathrm{erg} \ \mathrm{cm}^{-2} \ \mathrm{s}^{-1}]$",
    axs[3, 0], conv_factor=P_cgs * V_cgs,
)
make_derived_plot(
    cg_hr["flux_E_y"], sg_E_flux_y, lr_E_flux_y,
    r"Energy Flux $(E+p)u_y$",
    r"$(E+p)u_y \ [\mathrm{erg} \ \mathrm{cm}^{-2} \ \mathrm{s}^{-1}]$",
    axs[3, 1], conv_factor=P_cgs * V_cgs,
)

plt.tight_layout()
plt.savefig(save_str + "fluxes_mean_std.png", dpi=200)
plt.close(fig)
print("fluxes_mean_std.png saved")

# ---------------------------------------------------------------------------
# Step 3d: Divergence of Fluxes
# ---------------------------------------------------------------------------
print("[mock_sg_32x16] Computing divergence fluxes ...")
dx_sg = Lx_pc / resolution[1]
dy_sg = Ly_pc / resolution[0]

sg_div_mass = np.zeros_like(sg_mass_x)
sg_div_momx = np.zeros_like(sg_mass_x)
sg_div_momy = np.zeros_like(sg_mass_x)

lr_div_mass = np.zeros_like(lr_mass_x)
lr_div_momx = np.zeros_like(lr_mass_x)
lr_div_momy = np.zeros_like(lr_mass_x)

for t in range(nt):
    sg_div_mass[t] = divergence([sg_mass_x[t], sg_mass_y[t]], dx_sg, dy_sg)
    sg_div_momx[t] = divergence([sg_T_xx[t], sg_T_xy[t]], dx_sg, dy_sg)
    sg_div_momy[t] = divergence([sg_T_xy[t], sg_T_yy[t]], dx_sg, dy_sg)

    lr_div_mass[t] = divergence([lr_mass_x[t], lr_mass_y[t]], dx_sg, dy_sg)
    lr_div_momx[t] = divergence([lr_T_xx[t], lr_T_xy[t]], dx_sg, dy_sg)
    lr_div_momy[t] = divergence([lr_T_xy[t], lr_T_yy[t]], dx_sg, dy_sg)

fig, axs = plt.subplots(3, 1, figsize=(10, 13))
plt.subplots_adjust(hspace=0.35)

make_derived_plot(
    cg_hr_div_mass, sg_div_mass, lr_div_mass,
    r"Div Mass Flux ($\nabla \cdot \mathbf{j}$)",
    r"$\nabla \cdot (\rho \mathbf{u}) \ [M_\odot \ \mathrm{yr}^{-1} \ \mathrm{kpc}^{-2} \ \mathrm{pc}^{-1}]$",
    axs[0], conv_factor=mflux_to_Msun_yr_kpc2,
)
make_derived_plot(
    cg_hr_div_momx, sg_div_momx, lr_div_momx,
    r"Div MomX Flux ($\nabla \cdot \mathbf{T}_x$)",
    r"$\nabla \cdot \mathbf{T}_x \ [\mathrm{dyn} \ \mathrm{cm}^{-2} \ \mathrm{pc}^{-1}]$",
    axs[1], conv_factor=P_cgs,
)
make_derived_plot(
    cg_hr_div_momy, sg_div_momy, lr_div_momy,
    r"Div MomY Flux ($\nabla \cdot \mathbf{T}_y$)",
    r"$\nabla \cdot \mathbf{T}_y \ [\mathrm{dyn} \ \mathrm{cm}^{-2} \ \mathrm{pc}^{-1}]$",
    axs[2], conv_factor=P_cgs,
)

plt.tight_layout()
plt.savefig(save_str + "divergence_fluxes_mean_std.png", dpi=200)
plt.close(fig)
print("divergence_fluxes_mean_std.png saved")

del sg_mass_x, sg_mass_y, lr_mass_x, lr_mass_y
del sg_T_xx, sg_T_xy, sg_T_yy, lr_T_xx, lr_T_xy, lr_T_yy
del sg_E, lr_E, sg_E_flux_x, sg_E_flux_y, lr_E_flux_x, lr_E_flux_y
del sg_div_mass, sg_div_momx, sg_div_momy, lr_div_mass, lr_div_momx, lr_div_momy
del cg_hr_div_mass, cg_hr_div_momx, cg_hr_div_momy
gc.collect()

# ---------------------------------------------------------------------------
# Step 4: Cold-mass evolution — ergane for all three (HR streams, SG/LR use
#          already-loaded arrays)
# ---------------------------------------------------------------------------
print("[mock_sg_32x16] Computing cold mass evolution ...")

def cold_mass(rho_arr, temp_arr):
    dx = Lx_pc / rho_arr.shape[2]
    dy = Ly_pc / rho_arr.shape[1]
    return np.sum(rho_arr * (temp_arr < 1e5), axis=(1, 2)) * dx * dy

mass_sg = cold_mass(rho, temp)
mass_lr = cold_mass(lr_rho, lr_temp)

# HR: stream via ergane
mass_hr    = np.zeros(nt, dtype=np.float64)
dx_hr, dy_hr = Lx_pc / hr_resolution[1], Ly_pc / hr_resolution[0]
for i, num in enumerate(tqdm(hr_frames, desc="HR cold mass (streaming)")):
    fr       = hr_sim.get_frame(num)
    rho_hr_f = np.asarray(fr.density,     dtype=np.float64)
    temp_hr_f = np.asarray(fr.temperature, dtype=np.float64)
    mass_hr[i] = np.sum(rho_hr_f * (temp_hr_f < 1e5)) * dx_hr * dy_hr
    del fr, rho_hr_f, temp_hr_f

slope_hr, int_hr = np.polyfit(t_myr, mass_hr, 1)
slope_sg, int_sg = np.polyfit(t_myr, mass_sg, 1)
slope_lr, int_lr = np.polyfit(t_myr, mass_lr, 1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t_myr, mass_hr, lw=2, ls="-",  marker="^", markersize=5, label="HR")
ax.plot(t_myr, mass_sg, lw=2, ls="-.", marker="o", markersize=5, label="SG (tiled)")
ax.plot(t_myr, mass_lr, lw=2, ls="--", marker="s", markersize=5, label="LR (ISM)")
ax.plot(t_myr, slope_hr * t_myr + int_hr, lw=1.5, ls=":",
        label=f"HR fit  (ṁ = {slope_hr:.3e})")
ax.plot(t_myr, slope_sg * t_myr + int_sg, lw=1.5, ls=":",
        label=f"SG fit  (ṁ = {slope_sg:.3e})")
ax.plot(t_myr, slope_lr * t_myr + int_lr, lw=1.5, ls=":",
        label=f"LR fit  (ṁ = {slope_lr:.3e})")
ax.set_xlabel("Physical Time [Myr]")
ax.set_ylabel(r"Cold Gas Mass ($T < 10^5$ K) [ρ·pc²]")
ax.set_title("Cold Gas Mass Evolution (post-restart 5→10 Myr)")
ax.grid(True, ls="--", alpha=0.5); ax.legend()
plt.tight_layout()
plt.savefig(save_str + "cold_mass_evolution.png", dpi=200)
plt.close(fig)
print("cold_mass_evolution.png saved")

# ---------------------------------------------------------------------------
# Step 5: Emissivity (SG via CNN 4-tile evaluation; LR and HR via n²Λ; HR streams)
# ---------------------------------------------------------------------------
print("[mock_sg_32x16] Computing SG subgrid emissivity & predicted PDFs (4-tile CNN) ...")
emis_sg = np.zeros((nt, *resolution), dtype=np.float32)
pred_pdf_all = np.zeros((nt, out_channels, *resolution), dtype=np.float32)

n_tile_r = resolution[0] // TILE_ROWS  # 32 // 16 = 2
n_tile_c = resolution[1] // TILE_COLS  # 16 // 8 = 2

for t in tqdm(range(nt), desc="SG emissivity (CNN 4-tile)"):
    pdf_t = np.zeros((out_channels, *resolution), dtype=np.float32)
    for ti in range(n_tile_r):
        for tj in range(n_tile_c):
            r0, r1 = ti * TILE_ROWS, (ti + 1) * TILE_ROWS
            c0, c1 = tj * TILE_COLS, (tj + 1) * TILE_COLS
            pdf_t[:, r0:r1, c0:c1] = snapshot_pred_16x8(
                rho[t, r0:r1, c0:c1],
                temp[t, r0:r1, c0:c1],
                ux[t, r0:r1, c0:c1],
                uy[t, r0:r1, c0:c1],
                ps[t, r0:r1, c0:c1],
                fine_resolution=CNN_FINE_RES,
                downsample=CNN_DS,
                model_save_dir=MODEL_SAVES_DIR,
            )
    pred_pdf_all[t] = pdf_t
    emis_sg[t] = compute_cooling_rate(pdf_t, T_centers,
                                      is_pdf=True, rho_cg=rho[t]) / unit_fix

n_lr_cgs = lr_rho * n_to_cm3
emis_lr  = (n_lr_cgs**2 * lambda_cool(lr_temp, mask=True)).astype(np.float32)
del n_lr_cgs

# ---------------------------------------------------------------------------
# Step 5b: Peak Predicted Temperature PDF Time-Evolving Histogram (10^3 - 10^7 K)
# ---------------------------------------------------------------------------
print("[mock_sg_32x16] Computing and plotting peak predicted PDF evolution across 10^3 to 10^7 K ...")

log_temp_centers = 0.5 * (np.log10(T_edges[:-1]) + np.log10(T_edges[1:]))
log_temp_edges   = np.log10(T_edges)
active_bin_start = int(np.searchsorted(log_temp_centers, LOGT_ACTIVE_START))
active_bin_end   = int(np.searchsorted(log_temp_centers, LOGT_ACTIVE_END))

peak_bins_all = np.zeros((nt, resolution[0], resolution[1]), dtype=int)
peak_logT_all = np.zeros((nt, resolution[0], resolution[1]), dtype=np.float32)

for t in range(nt):
    pdf_t = pred_pdf_all[t]
    p_bins = np.argmax(pdf_t, axis=0)
    peak_bins_all[t] = p_bins
    peak_logT_all[t] = log_temp_centers[p_bins]

H_vol_all  = np.zeros((nt, out_channels), dtype=np.float32)
H_mass_all = np.zeros((nt, out_channels), dtype=np.float32)

for t in range(nt):
    b_all = peak_bins_all[t].ravel()
    w_mass = rho[t].ravel()
    cnts_v = np.bincount(b_all, minlength=out_channels).astype(np.float32)
    H_vol_all[t] = cnts_v / cnts_v.sum()
    cnts_m = np.bincount(b_all, weights=w_mass, minlength=out_channels).astype(np.float32)
    H_mass_all[t] = cnts_m / cnts_m.sum()

fig_cp = plt.figure(figsize=(16, 12))
gs_cp = fig_cp.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.32, wspace=0.25)
# imshow extent defines pixel EDGES; t_myr values are frame CENTERS, so pad by ±BIN_DT/2.
# log_temp_edges are genuine bin edges, so no padding needed on Y.
extent_time_logT = [
    t_myr[0]  - BIN_DT_MYR / 2,
    t_myr[-1] + BIN_DT_MYR / 2,
    log_temp_edges[0],
    log_temp_edges[-1],
]

# Panel 1: Vol-weighted
ax_cp1 = fig_cp.add_subplot(gs_cp[0, 0])
ax_cp1.axhspan(LOGT_ACTIVE_START, LOGT_ACTIVE_END, color="green", alpha=0.18, label="Active Cooling Zone", zorder=1)
ax_cp1.axhline(LOGT_ACTIVE_START, color="cyan", ls="--", lw=1.6, label=rf"$T_\mathrm{{active, start}} = {10**LOGT_ACTIVE_START:.2e}\ \mathrm{{K}}$ ({LOGT_ACTIVE_START:.2f})", zorder=4)
ax_cp1.axhline(LOGT_ACTIVE_END, color="lime", ls="--", lw=1.6, label=rf"$T_\mathrm{{active, end}} = {10**LOGT_ACTIVE_END:.2e}\ \mathrm{{K}}$ ({LOGT_ACTIVE_END:.2f})", zorder=4)

im_cp1 = ax_cp1.imshow(H_vol_all.T, origin="lower", extent=extent_time_logT, aspect="auto", cmap="magma",
                       norm=mcolors.Normalize(vmin=0, vmax=float(H_vol_all.max())), zorder=2)
ax_cp1.set_title(r"Time-Evolving Histogram of Peak Predicted PDF ($\text{Vol-Weighted}$)", fontsize=12, weight="bold")
ax_cp1.set_xlabel("Physical Time [Myr]", fontsize=11)
ax_cp1.set_ylabel(r"Peak Predicted Temperature $\log_{10}(T_\mathrm{peak}\ [\mathrm{K}])$", fontsize=11)
ax_cp1.set_ylim(3.0, 7.0)
cbar_cp1 = plt.colorbar(im_cp1, ax=ax_cp1, fraction=0.046, pad=0.04)
cbar_cp1.set_label("Fraction of Pixels", fontsize=10)
ax_cp1.legend(loc="upper right", fontsize=8.5, framealpha=0.85)
ax_cp1.grid(True, ls=":", alpha=0.4, color="gray")

# Panel 2: Mass-weighted
ax_cp2 = fig_cp.add_subplot(gs_cp[0, 1])
ax_cp2.axhspan(LOGT_ACTIVE_START, LOGT_ACTIVE_END, color="green", alpha=0.18, label="Active Cooling Zone", zorder=1)
ax_cp2.axhline(LOGT_ACTIVE_START, color="cyan", ls="--", lw=1.6, label=rf"$T_\mathrm{{active, start}}$ ({LOGT_ACTIVE_START:.2f})", zorder=4)
ax_cp2.axhline(LOGT_ACTIVE_END, color="lime", ls="--", lw=1.6, label=rf"$T_\mathrm{{active, end}}$ ({LOGT_ACTIVE_END:.2f})", zorder=4)

im_cp2 = ax_cp2.imshow(H_mass_all.T, origin="lower", extent=extent_time_logT, aspect="auto", cmap="magma",
                       norm=mcolors.Normalize(vmin=0, vmax=float(H_mass_all.max())), zorder=2)
ax_cp2.set_title(r"Time-Evolving Histogram of Peak Predicted PDF ($\text{Mass-Weighted}$)", fontsize=12, weight="bold")
ax_cp2.set_xlabel("Physical Time [Myr]", fontsize=11)
ax_cp2.set_ylabel(r"Peak Predicted Temperature $\log_{10}(T_\mathrm{peak}\ [\mathrm{K}])$", fontsize=11)
ax_cp2.set_ylim(3.0, 7.0)
cbar_cp2 = plt.colorbar(im_cp2, ax=ax_cp2, fraction=0.046, pad=0.04)
cbar_cp2.set_label("Mass Fraction of Pixels", fontsize=10)
ax_cp2.legend(loc="upper right", fontsize=8.5, framealpha=0.85)
ax_cp2.grid(True, ls=":", alpha=0.4, color="gray")

# Panel 3: Snapshot 1D Histograms
ax_cp3 = fig_cp.add_subplot(gs_cp[1, 0])
sample_times = [5.0, 6.0, 7.5, 9.0, 10.0]
sample_indices = [np.argmin(np.abs(t_myr - st)) for st in sample_times]
colors_list = plt.cm.plasma(np.linspace(0.1, 0.9, len(sample_times)))
ax_cp3.axvspan(LOGT_ACTIVE_START, LOGT_ACTIVE_END, color="green", alpha=0.15, label="Active Cooling Zone")
ax_cp3.axvline(LOGT_ACTIVE_START, color="cyan", ls="--", lw=1.4)
ax_cp3.axvline(LOGT_ACTIVE_END, color="lime", ls="--", lw=1.4)

for idx, col in zip(sample_indices, colors_list):
    actual_t = t_myr[idx]
    ax_cp3.plot(log_temp_centers, H_vol_all[idx],
                color=col, lw=2.2, marker="o", markersize=4, label=rf"$t = {actual_t:.2f}\ \mathrm{{Myr}}$")

ax_cp3.set_title("Peak Temperature Bin Distribution at Selected Time Snapshots", fontsize=11, weight="bold")
ax_cp3.set_xlabel(r"Peak Predicted Temperature $\log_{10}(T_\mathrm{peak}\ [\mathrm{K}])$", fontsize=11)
ax_cp3.set_ylabel("Pixel Fraction", fontsize=11)
ax_cp3.set_xlim(3.0, 7.0)
ax_cp3.set_ylim(bottom=0)
ax_cp3.grid(True, ls="--", alpha=0.5)
ax_cp3.legend(fontsize=9, loc="upper right")

# Panel 4: Spatial Maps of Peak Temperature
ax_cp4_sub = gs_cp[1, 1].subgridspec(1, 2, wspace=0.1)
ax_cp4a = fig_cp.add_subplot(ax_cp4_sub[0, 0])
ax_cp4b = fig_cp.add_subplot(ax_cp4_sub[0, 1])
t_early_idx = np.argmin(np.abs(t_myr - 5.5))
t_late_idx  = np.argmin(np.abs(t_myr - 9.5))
norm_peak = mcolors.Normalize(vmin=3.0, vmax=7.0)
im_cp4a = ax_cp4a.imshow(peak_logT_all[t_early_idx], origin="lower", cmap="inferno", norm=norm_peak)
ax_cp4a.set_title(rf"$T_\mathrm{{peak}}$ ($t={t_myr[t_early_idx]:.2f}$ Myr)", fontsize=10, weight="bold")
ax_cp4a.set_xlabel("X (cells)", fontsize=9); ax_cp4a.set_ylabel("Y (cells)", fontsize=9)
im_cp4b = ax_cp4b.imshow(peak_logT_all[t_late_idx], origin="lower", cmap="inferno", norm=norm_peak)
ax_cp4b.set_title(rf"$T_\mathrm{{peak}}$ ($t={t_myr[t_late_idx]:.2f}$ Myr)", fontsize=10, weight="bold")
ax_cp4b.set_xlabel("X (cells)", fontsize=9); ax_cp4b.set_yticks([])
cbar_cp4 = fig_cp.colorbar(im_cp4b, ax=[ax_cp4a, ax_cp4b], fraction=0.046, pad=0.04)
cbar_cp4.set_label(r"$\log_{10}(T_\mathrm{peak}\ [\mathrm{K}])$", fontsize=10)

plt.suptitle(
    rf"Subgrid Model: Time-Evolving Peak Predicted Temperature PDF Histogram ($10^3\ \mathrm{{K}} \leq T \leq 10^7\ \mathrm{{K}}$)"
    f"\nActive Cooling Zone: [{10**LOGT_ACTIVE_START:.2e} K, {10**LOGT_ACTIVE_END:.2e} K]",
    fontsize=14, weight="bold", y=0.99,
)
plt.savefig(save_str + "subgrid_peak_temperature_pdf_evolution.png", dpi=200, bbox_inches="tight")
plt.savefig(save_str + "cold_phase_peak_pdf_evolution.png", dpi=200, bbox_inches="tight")
plt.close(fig_cp)
print("subgrid_peak_temperature_pdf_evolution.png saved")

del peak_bins_all, peak_logT_all, H_vol_all, H_mass_all
gc.collect()

# ---------------------------------------------------------------------------
# Step 6: Emissivity profile vs y (stream HR; use accumulated CG arrays for SG/LR)
# ---------------------------------------------------------------------------
print("[mock_sg_32x16] Computing emissivity profiles ...")

emis_cg_hr_xavg = np.zeros((nt, resolution[0]), dtype=np.float32)
for i, num in enumerate(tqdm(hr_frames, desc="HR emissivity profile")):
    fr    = hr_sim.get_frame(num)
    rho_  = np.asarray(fr.density,     dtype=np.float64)
    temp_ = np.asarray(fr.temperature, dtype=np.float64)
    n_    = rho_ * n_to_cm3
    e_    = n_**2 * lambda_cool(temp_, mask=True)
    e_cg  = cg2d(e_, HR_DS)                         # (32, 16)
    emis_cg_hr_xavg[i] = np.mean(e_cg, axis=1)      # x-avg → (32,)
    del fr, rho_, temp_, n_, e_, e_cg

emis_sg_xavg = np.mean(emis_sg, axis=2)             # (nt, 32)
emis_lr_xavg = np.mean(emis_lr, axis=2)             # (nt, 32)

y_pc = np.linspace(-20.0, 20.0, resolution[0])
int_cg_hr = np.trapezoid(emis_cg_hr_xavg.mean(axis=0), y_pc)
int_sg    = np.trapezoid(emis_sg_xavg.mean(axis=0),    y_pc)
int_lr    = np.trapezoid(emis_lr_xavg.mean(axis=0),    y_pc)

fig, ax = plt.subplots(figsize=(7, 5))
ax.set_yscale("log")

for xavg, label, ls, mk, integral in [
    (emis_cg_hr_xavg, "CG HR", "-",  "^", int_cg_hr),
    (emis_sg_xavg,    "SG",    "-.", "o", int_sg),
    (emis_lr_xavg,    "LR",    "--", "s", int_lr),
]:
    m = xavg.mean(axis=0); s = xavg.std(axis=0)
    ax.plot(y_pc, m, lw=2, ls=ls, marker=mk, markersize=4,
            label=rf"{label} (Σ = {integral:.2e})")
    ax.fill_between(y_pc, np.clip(m - s, 1e-30, None), m + s, alpha=0.25)

ax.set_xlabel(r"$y\ [\mathrm{pc}]$")
ax.set_ylabel(r"$\langle n^2\Lambda(T)\rangle\ [\mathrm{erg\ cm}^{-3}\ \mathrm{s}^{-1}]$")
ax.set_title("Mean Cooling Rate Profile vs y")
ax.grid(True, ls="--", alpha=0.5); ax.legend()
plt.tight_layout()
plt.savefig(save_str + "emissivity_profile_vs_y.png", dpi=200)
plt.close(fig)
print("emissivity_profile_vs_y.png saved")

del emis_cg_hr_xavg, emis_sg_xavg, emis_lr_xavg
gc.collect()

# ---------------------------------------------------------------------------
# Step 7: Temperature PDFs — ergane streaming for HR; direct for SG/LR
# ---------------------------------------------------------------------------
print("[mock_sg_32x16] Computing temperature PDFs ...")
LOGT_START_PDF = float(os.environ.get("LOGT_ACTIVE_START", np.log10(1.1e4)))
LOGT_END_PDF = float(os.environ.get("LOGT_ACTIVE_END", np.log10(0.9e6)))
bins_pdf = np.logspace(LOGT_START_PDF, LOGT_END_PDF, 200)
bin_ctrs = np.sqrt(bins_pdf[:-1] * bins_pdf[1:])
Tmin, Tmax = bins_pdf[0], bins_pdf[-1]

def _pdf_stats_arrays(temp_arr, weight_arr):
    pdfs = []
    for t in range(temp_arr.shape[0]):
        v, w = temp_arr[t].ravel(), weight_arr[t].ravel()
        mask = (v >= Tmin) & (v <= Tmax) & np.isfinite(v) & (w > 0)
        h, _ = np.histogram(v[mask], bins=bins_pdf, weights=w[mask], density=True)
        pdfs.append(h)
    p = np.array(pdfs)
    return p.mean(axis=0), p.std(axis=0)

# SG / LR — use in-memory arrays
sg_vol_m,  sg_vol_s  = _pdf_stats_arrays(temp,    np.ones_like(temp))
sg_mass_m, sg_mass_s = _pdf_stats_arrays(temp,    rho)
lr_vol_m,  lr_vol_s  = _pdf_stats_arrays(lr_temp, np.ones_like(lr_temp))
lr_mass_m, lr_mass_s = _pdf_stats_arrays(lr_temp, lr_rho)

# HR — stream via ergane
hr_vol_pdfs, hr_mass_pdfs = [], []
for num in tqdm(hr_frames, desc="HR PDF (streaming)"):
    fr    = hr_sim.get_frame(num)
    t_    = np.asarray(fr.temperature, dtype=np.float64).ravel()
    rho_  = np.asarray(fr.density,     dtype=np.float64).ravel()
    mask  = (t_ >= Tmin) & (t_ <= Tmax) & np.isfinite(t_)
    h_v, _ = np.histogram(t_[mask], bins=bins_pdf, density=True)
    h_m, _ = np.histogram(t_[mask], bins=bins_pdf, weights=rho_[mask], density=True)
    hr_vol_pdfs.append(h_v); hr_mass_pdfs.append(h_m)
    del fr, t_, rho_

hr_vol_m  = np.mean(hr_vol_pdfs,  axis=0); hr_vol_s  = np.std(hr_vol_pdfs,  axis=0)
hr_mass_m = np.mean(hr_mass_pdfs, axis=0); hr_mass_s = np.std(hr_mass_pdfs, axis=0)
del hr_vol_pdfs, hr_mass_pdfs

pdf_sets = [
    ("Volume Weighted",
     hr_vol_m,  hr_vol_s,  sg_vol_m,  sg_vol_s,  lr_vol_m,  lr_vol_s),
    ("Mass Weighted",
     hr_mass_m, hr_mass_s, sg_mass_m, sg_mass_s, lr_mass_m, lr_mass_s),
]

fig, axs = plt.subplots(2, 1, figsize=(7, 10))
for ax, (title, hrm, hrs, sgm, sgs, lrm, lrs) in zip(axs, pdf_sets):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(bin_ctrs, hrm, lw=2, ls="-",  marker="^", markersize=4, label="CG HR")
    ax.fill_between(bin_ctrs, np.clip(hrm - hrs, 1e-30, None), hrm + hrs, alpha=0.25)
    ax.plot(bin_ctrs, sgm, lw=2, ls="-.", marker="o", markersize=5, label="SG tiled")
    ax.fill_between(bin_ctrs, np.clip(sgm - sgs, 1e-30, None), sgm + sgs, alpha=0.25)
    ax.plot(bin_ctrs, lrm, lw=2, ls="--", marker="s", markersize=5, label="LR ISM")
    ax.fill_between(bin_ctrs, np.clip(lrm - lrs, 1e-30, None), lrm + lrs, alpha=0.25)
    ax.set_xlim(Tmin, Tmax); ax.set_title(f"{title} Temperature PDF (Mean ± 1σ)")
    ax.set_xlabel("Temperature [K]"); ax.set_ylabel("PDF")
    ax.grid(True, which="both", ls="--", alpha=0.5); ax.legend()

plt.tight_layout()
plt.savefig(save_str + "temperature_pdfs_all_weightings.png", dpi=200)
plt.close(fig)
print("temperature_pdfs_all_weightings.png saved")
gc.collect()

# ---------------------------------------------------------------------------
# Step 8: Animations (Density, Temperature, Cooling-rate, and Subgrid PDF Grid)
# ---------------------------------------------------------------------------

# Color limits — sample 20 HR frames via ergane
print("[mock_sg_32x16] Sampling HR fields for animation color limits ...")
sample_idx = np.linspace(0, nt - 1, min(20, nt), dtype=int)
rho_samples, temp_samples, emis_samples = [], [], []

for i in sample_idx:
    fr    = hr_sim.get_frame(hr_frames[i])
    rho_  = np.asarray(fr.density,     dtype=np.float64)
    temp_ = np.asarray(fr.temperature, dtype=np.float64)
    n_    = rho_ * n_to_cm3
    e_    = n_**2 * lambda_cool(temp_, mask=True)
    rho_samples.append(rho_[rho_ > 0].astype(np.float32))
    temp_samples.append(temp_[temp_ > 0].astype(np.float32))
    emis_samples.append(e_[e_ > 0].astype(np.float32))
    del fr, rho_, temp_, n_, e_

all_pos_rho  = np.concatenate(rho_samples  + [rho[rho > 0],   lr_rho[lr_rho > 0]])
all_pos_temp = np.concatenate(temp_samples + [temp[temp > 0], lr_temp[lr_temp > 0]])
all_pos_emis = np.concatenate(emis_samples + [emis_sg[emis_sg > 0], emis_lr[emis_lr > 0]])

rho_vmin  = max(float(np.percentile(all_pos_rho,  1)), 1e-4)
rho_vmax  = float(np.percentile(all_pos_rho,  99))
temp_vmin = max(float(np.percentile(all_pos_temp, 1)), 1e3)
temp_vmax = float(np.percentile(all_pos_temp, 99))
cool_vmin = max(float(np.percentile(all_pos_emis, 1)), 1e-30)
cool_vmax = float(np.percentile(all_pos_emis, 99))

del rho_samples, temp_samples, emis_samples, all_pos_rho, all_pos_temp, all_pos_emis
gc.collect()

_sg_rho      = rho;          _lr_rho  = lr_rho
_sg_temp     = temp;         _lr_temp = lr_temp
_sg_emis     = emis_sg;      _lr_emis = emis_lr
_pred_pdf_all = pred_pdf_all


def parallel_chunk_animation(worker_chunk_func, total_frames, output_path, fps=10, num_workers=8):
    temp_dir = tempfile.mkdtemp()
    try:
        ctx = multiprocessing.get_context("fork")
        chunks = [list(range(i, total_frames, num_workers)) for i in range(num_workers)]
        chunks = [c for c in chunks if len(c) > 0]
        worker = partial(worker_chunk_func, temp_dir=temp_dir)
        with ctx.Pool(processes=len(chunks)) as pool:
            list(tqdm(pool.imap_unordered(worker, chunks), total=len(chunks),
                      desc=os.path.basename(output_path)))

        cmd = ["ffmpeg", "-y", "-r", str(fps),
               "-i", os.path.join(temp_dir, "frame_%04d.png"),
               "-c:v", "h264_nvenc", "-preset", "p4", "-pix_fmt", "yuv420p",
               output_path]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res.returncode != 0:
            cmd = ["ffmpeg", "-y", "-r", str(fps),
                   "-i", os.path.join(temp_dir, "frame_%04d.png"),
                   "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                   output_path]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if res.returncode != 0:
                raise RuntimeError(res.stderr.decode())
    finally:
        shutil.rmtree(temp_dir)


def render_frame_density(frame_idx, temp_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import numpy as np, os

    hr_num = hr_frames[frame_idx]
    fr     = hr_sim.get_frame(hr_num)
    cg_hr_rho_f = cg2d(np.asarray(fr.density, dtype=np.float64), HR_DS)
    del fr

    norm_rho = LogNorm(vmin=rho_vmin, vmax=rho_vmax)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(rf"Density | t = {t_myr[frame_idx]:.2f} Myr", fontsize=14, weight="bold")

    for ax, arr, label in zip(
        axs,
        [cg_hr_rho_f, _sg_rho[frame_idx], _lr_rho[frame_idx]],
        [f"CG HR ({resolution[0]}×{resolution[1]})",
         f"SG tiled ({resolution[0]}×{resolution[1]})",
         f"LR ISM ({lr_resolution[0]}×{lr_resolution[1]})"],
    ):
        im = ax.imshow(np.clip(arr, rho_vmin, None), origin="lower", cmap="plasma",
                       norm=norm_rho)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(temp_dir, f"frame_{frame_idx:04d}.png"), dpi=150)
    plt.close(fig)


def render_frame_temperature(frame_idx, temp_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import numpy as np, os

    hr_num = hr_frames[frame_idx]
    fr     = hr_sim.get_frame(hr_num)
    cg_hr_temp_f = cg2d(np.asarray(fr.temperature, dtype=np.float64), HR_DS)
    del fr

    norm_temp = LogNorm(vmin=temp_vmin, vmax=temp_vmax)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(rf"Temperature [K] | t = {t_myr[frame_idx]:.2f} Myr",
                 fontsize=14, weight="bold")

    for ax, arr, label in zip(
        axs,
        [cg_hr_temp_f, _sg_temp[frame_idx], _lr_temp[frame_idx]],
        [f"CG HR ({resolution[0]}×{resolution[1]})",
         f"SG tiled ({resolution[0]}×{resolution[1]})",
         f"LR ISM ({lr_resolution[0]}×{lr_resolution[1]})"],
    ):
        im = ax.imshow(np.clip(arr, temp_vmin, None), origin="lower", cmap="inferno",
                       norm=norm_temp)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(temp_dir, f"frame_{frame_idx:04d}.png"), dpi=150)
    plt.close(fig)


def render_frame_cooling(frame_idx, temp_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, matplotlib.colors as colors
    import numpy as np, os

    hr_num = hr_frames[frame_idx]
    fr     = hr_sim.get_frame(hr_num)
    rho_   = np.asarray(fr.density,     dtype=np.float64)
    temp_  = np.asarray(fr.temperature, dtype=np.float64)
    n_     = rho_ * n_to_cm3
    emis_hr_f = cg2d(n_**2 * lambda_cool(temp_, mask=True), HR_DS)
    del fr, rho_, temp_, n_

    norm_cool = colors.LogNorm(vmin=cool_vmin, vmax=cool_vmax)
    cmap_cool = plt.get_cmap("viridis")

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(rf"Cooling Rate | t = {t_myr[frame_idx]:.2f} Myr",
                 fontsize=14, weight="bold")

    for ax, field, label in zip(
        axs,
        [emis_hr_f, _sg_emis[frame_idx], _lr_emis[frame_idx]],
        [f"CG HR ({resolution[0]}×{resolution[1]})",
         f"SG tiled ({resolution[0]}×{resolution[1]})",
         f"LR ISM ({lr_resolution[0]}×{lr_resolution[1]})"],
    ):
        im = ax.imshow(np.clip(field, cool_vmin, None), origin="lower",
                       cmap=cmap_cool, norm=norm_cool)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(temp_dir, f"frame_{frame_idx:04d}.png"), dpi=150)
    plt.close(fig)


def worker_render_subgrid_pdf(frames_list, temp_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np, os

    nx, ny = resolution[0], resolution[1]  # 32, 16
    nb = out_channels                      # 40

    cmap_temp = plt.get_cmap("inferno")
    norm_temp = colors.Normalize(vmin=3.0, vmax=7.0)

    cmap_cool = plt.get_cmap("viridis")
    norm_cool = colors.LogNorm(vmin=cool_vmin, vmax=cool_vmax)

    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.18,
                          left=0.03, right=0.97, top=0.92, bottom=0.06)

    # 32x16 grid of PDF mini-plots
    sub_gs = gs[0].subgridspec(nx, ny, hspace=0.06, wspace=0.06)

    lines, axes = [], []
    x = np.arange(nb)

    for i in range(nx):
        row_l, row_a = [], []
        for j in range(ny):
            ax = fig.add_subplot(sub_gs[i, j])
            ax.axvspan(active_bin_start, active_bin_end, color="green", alpha=0.18, lw=0)
            (line,) = ax.plot([], [], lw=0.9)
            ax.set_xlim(0, nb - 1)
            ax.set_ylim(0, 1.05)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("grey")
                spine.set_linewidth(0.3)
            row_l.append(line)
            row_a.append(ax)
        lines.append(row_l)
        axes.append(row_a)

    # Side panel 1: Subgrid Resolved Temperature Map
    ax_temp = fig.add_subplot(gs[1])
    temp_0 = np.log10(_sg_temp[0] + 1e-8)
    im_temp = ax_temp.imshow(temp_0, origin="lower", cmap=cmap_temp, norm=norm_temp, aspect="auto")
    ax_temp.set_title("Subgrid $T$ Map", fontsize=15, weight="bold")
    ax_temp.set_xlabel("X (pixels)", fontsize=13)
    ax_temp.set_ylabel("Y (pixels)", fontsize=13)
    cbar_temp = plt.colorbar(im_temp, ax=ax_temp, fraction=0.046, pad=0.04)
    cbar_temp.set_label(r"$\log_{10}(T\ [\mathrm{K}])$ / Expectation Value", fontsize=12)

    # Side panel 2: Subgrid Cooling Rate Map
    ax_cool = fig.add_subplot(gs[2])
    cool_0 = np.clip(_sg_emis[0], cool_vmin, None)
    im_cool = ax_cool.imshow(cool_0, origin="lower", cmap=cmap_cool, norm=norm_cool, aspect="auto")
    ax_cool.set_title("Subgrid Cooling Rate Map", fontsize=15, weight="bold")
    ax_cool.set_xlabel("X (pixels)", fontsize=13)
    ax_cool.set_ylabel("Y (pixels)", fontsize=13)
    cbar_cool = plt.colorbar(im_cool, ax=ax_cool, fraction=0.046, pad=0.04)
    cbar_cool.set_label(r"Cooling Rate $[\mathrm{erg}\ \mathrm{cm}^{-3}\ \mathrm{s}^{-1}]$", fontsize=12)

    title_text = fig.suptitle("", fontsize=18, weight="bold")

    for frame_idx in frames_list:
        pdf_frame = _pred_pdf_all[frame_idx]
        temp_frame = np.log10(_sg_temp[frame_idx] + 1e-8)
        cool_frame = np.clip(_sg_emis[frame_idx], cool_vmin, None)

        for i in range(nx):
            ii = nx - 1 - i  # flip vertically so row 0 is top (y = +20 pc)
            for j in range(ny):
                y = pdf_frame[:, ii, j]
                exp_val = np.sum(y * log_temp_centers)
                y_norm = y / (y.max() + 1e-12)

                bg_color = cmap_temp(norm_temp(exp_val))
                lum = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
                line_color = "white" if lum < 0.5 else "black"

                lines[i][j].set_data(x, y_norm)
                lines[i][j].set_color(line_color)
                axes[i][j].set_facecolor(bg_color)

        im_temp.set_data(temp_frame)
        im_cool.set_data(cool_frame)
        title_text.set_text(rf"Subgrid Predicted Temperature PDF Grid ($32 \times 16$), $T$, & Cooling | $t = {t_myr[frame_idx]:.2f}$ Myr")
        
        frame_out = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
        fig.savefig(frame_out, dpi=120)

        if frame_idx == 0:
            fig.savefig(save_str + "subgrid_predicted_pdf_snapshot_t0.png", dpi=200)

    plt.close(fig)


print("[mock_sg_32x16] Rendering density animation ...")
parallel_save_animation(render_frame_density, range(nt),
                        save_str + "density_evolution.mp4", fps=10, num_workers=8)
print("density_evolution.mp4 saved")

print("[mock_sg_32x16] Rendering temperature animation ...")
parallel_save_animation(render_frame_temperature, range(nt),
                        save_str + "temperature_evolution.mp4", fps=10, num_workers=8)
print("temperature_evolution.mp4 saved")

print("[mock_sg_32x16] Rendering cooling-rate animation ...")
parallel_save_animation(render_frame_cooling, range(nt),
                        save_str + "cooling_rate_evolution.mp4", fps=10, num_workers=8)
print("cooling_rate_evolution.mp4 saved")

print("[mock_sg_32x16] Rendering Subgrid Predicted PDF grid animation ...")
parallel_chunk_animation(worker_render_subgrid_pdf, nt,
                         save_str + "subgrid_predicted_pdf_evolution.mp4", fps=10, num_workers=8)
print("subgrid_predicted_pdf_evolution.mp4 saved")

del emis_sg, emis_lr
gc.collect()

print(f"\n[mock_sg_32x16] All outputs written to: {save_str}")

