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


# ---------------------------------------------------------------------------
# Step 1: Stream HR for profile plots (accumulate tiny 32×16 CG arrays)
# ---------------------------------------------------------------------------
print("[mock_sg_32x16] Streaming HR → CG profile arrays ...")

cg_hr: dict[str, np.ndarray] = {
    k: np.zeros((n_common, *resolution), dtype=np.float32)
    for k in ("rho", "temp", "pres", "ux", "uy", "eint",
              "fmcl", "cons_rho", "cons_mx", "cons_my", "cons_ener")
}

for i, num in enumerate(tqdm(hr_frames, desc="HR → CG fields")):
    fr   = hr_sim.get_frame(num)
    flds = _frame_fields(fr)
    for k in cg_hr:
        cg_hr[k][i] = cg2d(flds[k], HR_DS)   # (2048,1024) → (32,16)
    del fr, flds                               # free full-res immediately

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
print("[mock_sg_32x16] Computing SG subgrid emissivity (4-tile CNN) ...")
emis_sg = np.zeros((nt, *resolution), dtype=np.float32)

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
    emis_sg[t] = compute_cooling_rate(pdf_t, T_centers,
                                      is_pdf=True, rho_cg=rho[t]) / unit_fix

n_lr_cgs = lr_rho * n_to_cm3
emis_lr  = (n_lr_cgs**2 * lambda_cool(lr_temp, mask=True)).astype(np.float32)
del n_lr_cgs

# Color limits — sample 20 HR frames via ergane
print("[mock_sg_32x16] Sampling HR emissivity for color limits ...")
sample_idx = np.linspace(0, nt - 1, min(20, nt), dtype=int)
emis_samples = []
for i in sample_idx:
    fr    = hr_sim.get_frame(hr_frames[i])
    rho_  = np.asarray(fr.density,     dtype=np.float64)
    temp_ = np.asarray(fr.temperature, dtype=np.float64)
    n_    = rho_ * n_to_cm3
    e_    = n_**2 * lambda_cool(temp_, mask=True)
    emis_samples.append(e_[e_ > 0].astype(np.float32))
    del fr, rho_, temp_, n_, e_

all_pos   = np.concatenate(emis_samples + [emis_sg[emis_sg > 0], emis_lr[emis_lr > 0]])
cool_vmin = max(np.percentile(all_pos, 1), 1e-30)
cool_vmax = np.percentile(all_pos, 99)
del emis_samples, all_pos
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
LOGT_START, LOGT_END = 4.1, 5.9
bins_pdf = np.logspace(LOGT_START, LOGT_END, 200)
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
# Step 8: Density and cooling-rate animations
#   All three sims are accessed via ergane in fork workers.
#   HR: stream one frame per worker call.
#   SG/LR: index into in-memory arrays (fork inherits parent memory).
# ---------------------------------------------------------------------------

# Globals shared into forked workers via fork()
_sg_rho  = rho;       _lr_rho  = lr_rho
_sg_emis = emis_sg;   _lr_emis = emis_lr
_sg_temp = temp;      _lr_temp = lr_temp
_cg_hr_rho  = cg_hr["rho"]     # already coarse-grained, in memory
_cg_hr_temp = cg_hr["temp"]


def render_frame_density(frame_idx, temp_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import numpy as np, os

    # HR: stream one frame via the inherited ergane handle
    hr_num = hr_frames[frame_idx]
    fr     = hr_sim.get_frame(hr_num)
    cg_hr_rho_f = cg2d(np.asarray(fr.density, dtype=np.float64), HR_DS)
    del fr

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(rf"Density | t = {t_myr[frame_idx]:.2f} Myr", fontsize=14)

    for ax, arr, label in zip(
        axs,
        [cg_hr_rho_f, _sg_rho[frame_idx], _lr_rho[frame_idx]],
        [f"CG HR ({resolution[0]}×{resolution[1]})",
         f"SG ({resolution[0]}×{resolution[1]})",
         f"LR ({lr_resolution[0]}×{lr_resolution[1]})"],
    ):
        pos = arr[arr > 0]
        vmin, vmax = (pos.min(), pos.max()) if pos.size else (1e-4, 1)
        ax.imshow(arr, origin="lower", cmap="plasma",
                  norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Y (pixels)")

    plt.tight_layout()
    plt.savefig(os.path.join(temp_dir, f"frame_{frame_idx:04d}.png"), dpi=150)
    plt.close(fig)


def render_frame_cooling(frame_idx, temp_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, matplotlib.colors as colors
    import numpy as np, os

    # HR: stream one frame
    hr_num = hr_frames[frame_idx]
    fr     = hr_sim.get_frame(hr_num)
    rho_   = np.asarray(fr.density,     dtype=np.float64)
    temp_  = np.asarray(fr.temperature, dtype=np.float64)
    n_     = rho_ * n_to_cm3
    emis_hr_f = cg2d(n_**2 * lambda_cool(temp_, mask=True), HR_DS)
    del fr, rho_, temp_, n_

    norm_cool = colors.LogNorm(vmin=cool_vmin, vmax=cool_vmax)
    cmap_cool = plt.get_cmap("viridis")

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(rf"Cooling Rate | t = {t_myr[frame_idx]:.2f} Myr",
                 fontsize=14, weight="bold")

    for ax, field, label in zip(
        axs,
        [emis_hr_f, _sg_emis[frame_idx], _lr_emis[frame_idx]],
        [f"CG HR ({resolution[0]}×{resolution[1]})",
         f"SG tiled ({resolution[0]}×{resolution[1]})",
         f"LR ISM ({lr_resolution[0]}×{lr_resolution[1]})"],
    ):
        ax.imshow(np.clip(field, cool_vmin, None), origin="lower",
                  cmap=cmap_cool, norm=norm_cool)
        ax.set_title(label, fontsize=12); ax.set_xlabel("Y (pixels)")

    plt.tight_layout()
    plt.savefig(os.path.join(temp_dir, f"frame_{frame_idx:04d}.png"), dpi=150)
    plt.close(fig)


print("[mock_sg_32x16] Rendering density animation ...")
parallel_save_animation(render_frame_density, range(nt),
                        save_str + "density_evolution.mp4", fps=10, num_workers=8)
print("density_evolution.mp4 saved")

print("[mock_sg_32x16] Rendering cooling-rate animation ...")
parallel_save_animation(render_frame_cooling, range(nt),
                        save_str + "cooling_rate_evolution.mp4", fps=10, num_workers=8)
print("cooling_rate_evolution.mp4 saved")

del emis_sg, emis_lr
gc.collect()

print(f"\n[mock_sg_32x16] All outputs written to: {save_str}")
