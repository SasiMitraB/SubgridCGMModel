# Python script for comparing the results of HR, SG, and LR simulations

import gc
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib.animation as animation
from data_preprocess import simulation_data
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from tqdm import tqdm
import multiprocessing
import tempfile
import shutil
import subprocess
from functools import partial


bins_pdf = np.logspace(4, 6, 200)
window = 10


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


def parallel_save_animation(render_func, frames_list, output_path, fps=10, num_workers=16):
    """
    Renders frames in parallel using render_func(frame, temp_dir)
    and stitches them together using ffmpeg.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        # Use 'fork' start method to inherit parent process's memory space
        # (avoiding serialization of large numpy arrays).
        ctx = multiprocessing.get_context('fork')
        worker = partial(render_func, temp_dir=temp_dir)
        
        with ctx.Pool(processes=num_workers) as pool:
            list(tqdm(pool.imap(worker, frames_list), total=len(frames_list), desc=os.path.basename(output_path)))
            
        # Stitch frames with ffmpeg
        cmd = [
            'ffmpeg', '-y',
            '-r', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%04d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            output_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res.returncode != 0:
            print(f"Error compiling video {output_path} with ffmpeg: {res.stderr.decode()}")
            raise RuntimeError(res.stderr.decode())
    finally:
        shutil.rmtree(temp_dir)


def render_frame_all_fields(frame, temp_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import numpy as np
    import os

    fig, axs = plt.subplots(6, 3, figsize=(8, 20))
    for i in range(6):
        f_hr = fields_hr[i][frame]
        f_sg = fields_sg[i][frame]
        f_lr = fields_lr[i][frame]

        arr = np.concatenate([f_hr.flatten(), f_sg.flatten(), f_lr.flatten()])
        use_log = (i == 0 or i == 1)
        vmin, vmax, norm = compute_color_limits(arr, use_log=use_log)

        im0 = axs[i, 0].imshow(f_hr, origin='lower', cmap='plasma', norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax)
        axs[i, 0].set_title(f"HR {titles[i]}")
        plt.colorbar(im0, ax=axs[i, 0], fraction=0.035, pad=0.02)

        im1 = axs[i, 1].imshow(f_sg, origin='lower', cmap='plasma', norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax)
        axs[i, 1].set_title(f"SG {titles[i]}")
        plt.colorbar(im1, ax=axs[i, 1], fraction=0.035, pad=0.02)

        im2 = axs[i, 2].imshow(f_lr, origin='lower', cmap='plasma', norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax)
        axs[i, 2].set_title(f"LR {titles[i]}")
        plt.colorbar(im2, ax=axs[i, 2], fraction=0.035, pad=0.02)

    for ax in axs.flat:
        ax.set_xlabel(f"Timestep: {frame}")

    plt.tight_layout()
    plt.savefig(os.path.join(temp_dir, f"frame_{frame:04d}.png"), dpi=200)
    plt.close(fig)


def render_frame_cons_fields(frame, temp_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import numpy as np
    import os

    fig, axs = plt.subplots(6, 3, figsize=(8, 20))
    for i in range(6):
        f_hr = cons_fields_hr[i][frame]
        f_sg = cons_fields_sg[i][frame]
        f_lr = cons_fields_lr[i][frame]

        arr = np.concatenate([f_hr.flatten(), f_sg.flatten(), f_lr.flatten()])
        use_log = (i == 0 or i == 3)
        vmin, vmax, norm = compute_color_limits(arr, use_log=use_log)

        im0 = axs[i, 0].imshow(f_hr, origin="lower", cmap="plasma", norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax)
        plt.colorbar(im0, ax=axs[i, 0], fraction=0.035, pad=0.02)
        axs[i, 0].set_title(f"HR {cons_titles[i]}")

        im1 = axs[i, 1].imshow(f_sg, origin="lower", cmap="plasma", norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax)
        plt.colorbar(im1, ax=axs[i, 1], fraction=0.035, pad=0.02)
        axs[i, 1].set_title(f"SG {cons_titles[i]}")

        im2 = axs[i, 2].imshow(f_lr, origin="lower", cmap="plasma", norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax)
        plt.colorbar(im2, ax=axs[i, 2], fraction=0.035, pad=0.02)
        axs[i, 2].set_title(f"LR {cons_titles[i]}")

    for ax in axs.flat:
        ax.set_xlabel(f"Timestep: {frame}")

    plt.tight_layout()
    plt.savefig(os.path.join(temp_dir, f"frame_{frame:04d}.png"), dpi=200)
    plt.close(fig)


def render_frame_rho(frame, temp_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import numpy as np
    import os

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    # To match original static colorbar, we compute limits from frame 0
    vmin_hr = cg_hr_rho[0][cg_hr_rho[0] > 0].min()
    vmax_hr = cg_hr_rho[0].max()
    im_hr_rho = axs[0].imshow(cg_hr_rho[frame], origin="lower", cmap="plasma", norm=LogNorm(vmin=vmin_hr, vmax=vmax_hr))
    axs[0].set_title(rf"HR (${hr_resolution[0]} \times {hr_resolution[1]}$) Density")
    plt.colorbar(im_hr_rho, ax=axs[0], fraction=0.046, pad=0.04)

    vmin_sg = rho[0][rho[0] > 0].min()
    vmax_sg = rho[0].max()
    im_rho = axs[1].imshow(rho[frame], origin="lower", cmap="plasma", norm=LogNorm(vmin=vmin_sg, vmax=vmax_sg))
    axs[1].set_title(rf"SG (${resolution[0]} \times {resolution[1]}$) Density (sigma=3)")
    plt.colorbar(im_rho, ax=axs[1], fraction=0.046, pad=0.04)

    vmin_lr = lr_rho[0][lr_rho[0] > 0].min()
    vmax_lr = lr_rho[0].max()
    im_lr_rho = axs[2].imshow(lr_rho[frame], origin="lower", cmap="plasma", norm=LogNorm(vmin=vmin_lr, vmax=vmax_lr))
    axs[2].set_title(rf"LR (${lr_resolution[0]} \times {lr_resolution[1]}$) Density")
    plt.colorbar(im_lr_rho, ax=axs[2], fraction=0.046, pad=0.04)

    for ax in axs.flat:
        ax.set_xlabel(f"Timestep: {frame}")

    plt.tight_layout()
    plt.savefig(os.path.join(temp_dir, f"frame_{frame:04d}.png"), dpi=200)
    plt.close(fig)


def render_frame_temp_pdf(frame, temp_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("PDF (volume-weighted, time-avg 10 steps)")
    ax.set_ylim(1e-7, 1e-3)
    ax.set_xlim(bins_pdf[0], bins_pdf[-1])
    ax.grid(True, which="both", ls="--", alpha=0.5)

    ax.set_title(f"Time step {frame + 1}")
    end = min(frame + window, temp.shape[0])
    h_hr, _ = np.histogram(cg_hr_temp[frame:end].ravel(), bins=bins_pdf, density=True)
    h_lr, _ = np.histogram(lr_temp[frame:end].ravel(), bins=bins_pdf, density=True)
    h_sg, _ = np.histogram(temp[frame:end].ravel(), bins=bins_pdf, density=True)

    ax.plot(bins_pdf[:-1], h_hr, lw=2.0, label="HR")
    ax.plot(bins_pdf[:-1], h_lr, lw=2.0, label="LR")
    ax.plot(bins_pdf[:-1], h_sg, lw=2.0, label="SG")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(temp_dir, f"frame_{frame:04d}.png"), dpi=200)
    plt.close(fig)


def divergence(f, dx, dy):
    dFx_dx = np.gradient(f[0], dy, dx)[1]
    dFy_dy = np.gradient(f[1], dy, dx)[0]
    return dFx_dx + dFy_dy


def lambda_cool(temp):
    """
    Cooling function ISMCoolFn translated from AthenaK C++.
    Works on scalars or numpy arrays (any shape).
    Returns Λ(T) in erg cm^3 / s.
    """
    logt = np.log10(temp)

    lhd = np.array(
        [
            -22.5977,
            -21.9689,
            -21.5972,
            -21.4615,
            -21.4789,
            -21.5497,
            -21.6211,
            -21.6595,
            -21.6426,
            -21.5688,
            -21.4771,
            -21.3755,
            -21.2693,
            -21.1644,
            -21.0658,
            -20.9778,
            -20.8986,
            -20.8281,
            -20.7700,
            -20.7223,
            -20.6888,
            -20.6739,
            -20.6815,
            -20.7051,
            -20.7229,
            -20.7208,
            -20.7058,
            -20.6896,
            -20.6797,
            -20.6749,
            -20.6709,
            -20.6748,
            -20.7089,
            -20.8031,
            -20.9647,
            -21.1482,
            -21.2932,
            -21.3767,
            -21.4129,
            -21.4291,
            -21.4538,
            -21.5055,
            -21.5740,
            -21.6300,
            -21.6615,
            -21.6766,
            -21.6886,
            -21.7073,
            -21.7304,
            -21.7491,
            -21.7607,
            -21.7701,
            -21.7877,
            -21.8243,
            -21.8875,
            -21.9738,
            -22.0671,
            -22.1537,
            -22.2265,
            -22.2821,
            -22.3213,
            -22.3462,
            -22.3587,
            -22.3622,
            -22.3590,
            -22.3512,
            -22.3420,
            -22.3342,
            -22.3312,
            -22.3346,
            -22.3445,
            -22.3595,
            -22.3780,
            -22.4007,
            -22.4289,
            -22.4625,
            -22.4995,
            -22.5353,
            -22.5659,
            -22.5895,
            -22.6059,
            -22.6161,
            -22.6208,
            -22.6213,
            -22.6184,
            -22.6126,
            -22.6045,
            -22.5945,
            -22.5831,
            -22.5707,
            -22.5573,
            -22.5434,
            -22.5287,
            -22.5140,
            -22.4992,
            -22.4844,
            -22.4695,
            -22.4543,
            -22.4392,
            -22.4237,
            -22.4087,
            -22.3928,
        ]
    )

    lam = np.zeros_like(temp, dtype=float)

    # turn off cooling below 1e4 K
    mask_off = logt <= 4.0
    lam[mask_off] = 0.0

    # KI02 regime (4.0 < logT <= 4.2)
    mask_ki = (logt > 4.0) & (logt <= 4.2)
    if np.any(mask_ki):
        lam[mask_ki] = 2.0e-19 * np.exp(
            -1.184e5 / (temp[mask_ki] + 1.0e3)
        ) + 2.8e-28 * np.sqrt(temp[mask_ki]) * np.exp(-92.0 / temp[mask_ki])

    # CGOLS fit (logT > 8.15)
    mask_hi = logt > 8.15
    lam[mask_hi] = 10.0 ** (0.45 * logt[mask_hi] - 26.065)

    # SPEX interpolation (4.2 < logT <= 8.15)
    mask_mid = (logt > 4.2) & (logt <= 8.15)
    if np.any(mask_mid):
        ipps = (25.0 * logt[mask_mid] - 103).astype(int)
        # Clamp to [0,100] like C++
        ipps = np.clip(ipps, 0, 100)
        x0 = 4.12 + 0.04 * ipps
        dx = logt[mask_mid] - x0
        logcool = (lhd[ipps + 1] * dx - lhd[ipps] * (dx - 0.04)) * 25.0
        lam[mask_mid] = 10.0**logcool

    return lam


# =============================================================================
# RESTART TIME CONVENTION
# -----------------------------------------------------------------------------
# The pipeline runs an initial LR simulation from 0 → 5 Myr and then
# branches two restart simulations from the 5 Myr checkpoint:
#
#   lr_build_ism  : hr_build/src/athena  with ISM cooling  (5 → 10 Myr)
#   subgrid_model : subgrid_model/src/athena with CNN       (5 → 10 Myr)
#
# Both restarted simulations write bin files starting at frame 0, which
# physically corresponds to t = RESTART_TIME_MYR = 5 Myr.
# Therefore ALL frames in lr_build_ism and subgrid_model are in the
# "steady-state phase" and no start= skip is needed.
#
# For time-axis plots in Myr:
#   t_sg[i] = RESTART_TIME_MYR + i * BIN_DT_MYR
#   t_lr[i] = RESTART_TIME_MYR + i * BIN_DT_MYR
# ========================================================================
RESTART_TIME_MYR = 5.0   # physical time of the restart file (Myr)
BIN_DT_MYR       = 0.01  # bin output cadence (matches bin_w_dt / bin_u_dt in config)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

resolution = (32, 16)
file_path = os.path.join(PROJECT_ROOT, "simulation_outputs/subgrid_model/bin")
save_path = (
    os.path.join(os.environ.get("SG_MOCKS_DIR", "mocks/sg"), f"sc{resolution}") + "/"
)
os.makedirs(save_path, exist_ok=True)


# --- subgrid_model (CNN restart, 5 → 10 Myr) ---
# AthenaK continues file numbering from the restart checkpoint.
# The LR run wrote frames 00000-00500 (0→5 Myr at bin_dt=0.01).
# Both restarted simulations therefore produce frames 00501-01000.
# start=501 tells input_data to open KH.hydro_w.00501.bin as frame 0.
sim_data = simulation_data()
sim_data.resolution = resolution
sim_data.input_data(file_path, start=501)
sim_data.input_cons_data(file_path, start=501)

rho = sim_data.rho
pres = sim_data.pressure
temp = sim_data.temp
ien = sim_data.eint
ux = sim_data.ux
uy = sim_data.uy
fmcl = sim_data.frho

cons_rho = sim_data.cons_rho
cons_momx = sim_data.cons_momx
cons_momy = sim_data.cons_momy
cons_ener = sim_data.cons_ener
cons_ps = sim_data.cons_ps

lr_frac = np.zeros_like(temp)
lr_frac[temp < sim_data.T_cutoff] = 1.0
frac = sim_data.frho

# Physical time axis for restarted simulations (both share the same axis)
n_sg = rho.shape[0]
t_sg_myr = RESTART_TIME_MYR + np.arange(n_sg) * BIN_DT_MYR

lr_resolution = resolution
# lr_build_ism: ISM-cooling restart (5 → 10 Myr).  Frame 0 = t=5 Myr.
lr_file_path = os.path.join(PROJECT_ROOT, "simulation_outputs/lr_build_ism/bin")
lr_sim_data = simulation_data()
lr_sim_data.resolution = lr_resolution
lr_sim_data.input_data(lr_file_path, start=501)
lr_rho = lr_sim_data.rho[: rho.shape[0]]
lr_temp = lr_sim_data.temp[: rho.shape[0]]
lr_pres = lr_sim_data.pressure[: rho.shape[0]]
lr_ux = lr_sim_data.ux[: rho.shape[0]]
lr_uy = lr_sim_data.uy[: rho.shape[0]]
lr_ien = lr_sim_data.eint[: rho.shape[0]]

lr_sim_data.input_cons_data(lr_file_path, start=501)
lr_cons_rho = lr_sim_data.cons_rho[: rho.shape[0]]
lr_cons_momx = lr_sim_data.cons_momx[: rho.shape[0]]
lr_cons_momy = lr_sim_data.cons_momy[: rho.shape[0]]
lr_cons_ener = lr_sim_data.cons_ener[: rho.shape[0]]
lr_cons_ps = lr_sim_data.cons_ps[: rho.shape[0]]

lr_fmcl = (lr_temp < 1e5).astype(float)

hr_resolution = (1024, 512)
hr_downsample = 32
hr_file_path = os.path.join(PROJECT_ROOT, "simulation_outputs/hr_build_1024")
hr_sim_data = simulation_data()
hr_sim_data.resolution = hr_resolution
hr_sim_data.down_sample = hr_downsample
# hr_sim_data.input_data(hr_file_path)
# hr_rho = hr_sim_data.rho
# hr_temp = hr_sim_data.temp
hr_folder_path = os.path.join(PROJECT_ROOT, f"simulation_outputs/hr_build_1024/cache/sc{hr_resolution}_{hr_downsample}")
# Memory-map the large HR arrays so the OS pages them in on demand
# instead of loading every frame into RAM at once.
_n = rho.shape[0]
hr_rho  = np.load(f"{hr_folder_path}/rho.npy",      mmap_mode="r")[:_n]
hr_temp = np.load(f"{hr_folder_path}/temp.npy",     mmap_mode="r")[:_n]
hr_pres = np.load(f"{hr_folder_path}/pressure.npy", mmap_mode="r")[:_n]
hr_ux   = np.load(f"{hr_folder_path}/ux.npy",       mmap_mode="r")[:_n]
hr_uy   = np.load(f"{hr_folder_path}/uy.npy",       mmap_mode="r")[:_n]
hr_ien  = np.load(f"{hr_folder_path}/eint.npy",     mmap_mode="r")[:_n]

hr_cons_rho  = np.load(f"{hr_folder_path}/cons_rho.npy",  mmap_mode="r")[:_n]
hr_cons_momx = np.load(f"{hr_folder_path}/cons_mx.npy",   mmap_mode="r")[:_n]
hr_cons_momy = np.load(f"{hr_folder_path}/cons_my.npy",   mmap_mode="r")[:_n]
hr_cons_ener = np.load(f"{hr_folder_path}/cons_ener.npy", mmap_mode="r")[:_n]
hr_cons_ps   = np.load(f"{hr_folder_path}/cons_ps.npy",   mmap_mode="r")[:_n]

cg_hr_rho = np.zeros(
    (
        hr_rho.shape[0],
        hr_rho.shape[1] // hr_downsample,
        hr_rho.shape[2] // hr_downsample,
    )
)
cg_hr_temp = np.zeros(
    (
        hr_rho.shape[0],
        hr_rho.shape[1] // hr_downsample,
        hr_rho.shape[2] // hr_downsample,
    )
)
cg_hr_pres = np.zeros(
    (
        hr_rho.shape[0],
        hr_rho.shape[1] // hr_downsample,
        hr_rho.shape[2] // hr_downsample,
    )
)
cg_hr_ux = np.zeros(
    (
        hr_rho.shape[0],
        hr_rho.shape[1] // hr_downsample,
        hr_rho.shape[2] // hr_downsample,
    )
)
cg_hr_uy = np.zeros(
    (
        hr_rho.shape[0],
        hr_rho.shape[1] // hr_downsample,
        hr_rho.shape[2] // hr_downsample,
    )
)
cg_hr_ien = np.zeros(
    (
        hr_rho.shape[0],
        hr_rho.shape[1] // hr_downsample,
        hr_rho.shape[2] // hr_downsample,
    )
)

cg_hr_cons_rho = np.zeros(
    (
        hr_rho.shape[0],
        hr_rho.shape[1] // hr_downsample,
        hr_rho.shape[2] // hr_downsample,
    )
)
cg_hr_cons_momx = np.zeros(
    (
        hr_rho.shape[0],
        hr_rho.shape[1] // hr_downsample,
        hr_rho.shape[2] // hr_downsample,
    )
)
cg_hr_cons_momy = np.zeros(
    (
        hr_rho.shape[0],
        hr_rho.shape[1] // hr_downsample,
        hr_rho.shape[2] // hr_downsample,
    )
)
cg_hr_cons_ener = np.zeros(
    (
        hr_rho.shape[0],
        hr_rho.shape[1] // hr_downsample,
        hr_rho.shape[2] // hr_downsample,
    )
)
cg_hr_cons_ps = np.zeros(
    (
        hr_rho.shape[0],
        hr_rho.shape[1] // hr_downsample,
        hr_rho.shape[2] // hr_downsample,
    )
)

cg_hr_fmcl = np.zeros(
    (
        hr_rho.shape[0],
        hr_rho.shape[1] // hr_downsample,
        hr_rho.shape[2] // hr_downsample,
    )
)

for i in tqdm(range(hr_rho.shape[0]), desc="Calculating CG HR"):
    cg_hr_rho[i] = hr_sim_data.coarse_grain(hr_rho[i])
    cg_hr_temp[i] = hr_sim_data.coarse_grain(hr_temp[i])
    cg_hr_pres[i] = hr_sim_data.coarse_grain(hr_pres[i])
    cg_hr_ux[i] = hr_sim_data.coarse_grain(hr_ux[i])
    cg_hr_uy[i] = hr_sim_data.coarse_grain(hr_uy[i])
    cg_hr_ien[i] = hr_sim_data.coarse_grain(hr_ien[i])

    cg_hr_cons_rho[i] = hr_sim_data.coarse_grain(hr_cons_rho[i])
    cg_hr_cons_momx[i] = hr_sim_data.coarse_grain(hr_cons_momx[i])
    cg_hr_cons_momy[i] = hr_sim_data.coarse_grain(hr_cons_momy[i])
    cg_hr_cons_ener[i] = hr_sim_data.coarse_grain(hr_cons_ener[i])
    cg_hr_cons_ps[i] = hr_sim_data.coarse_grain(hr_cons_ps[i])

    cg_hr_fmcl[i] = hr_sim_data.calc_fmcl(hr_rho[i], hr_temp[i])

cg_hr_rho  = cg_hr_rho[:  rho.shape[0]]
cg_hr_temp = cg_hr_temp[:temp.shape[0]]
cg_hr_pres = cg_hr_pres[:temp.shape[0]]


def compute_mean_std(arr, logspace=False):
    if logspace:
        arr = np.log10(arr)

    arr_1d = np.mean(arr, axis=2)  # avg over X
    mean = arr_1d.mean(axis=0)  # mean over time
    std = arr_1d.std(axis=0)  # std over time

    return mean, std


quantities = [
    ("Density", cg_hr_rho, cg_hr_temp, rho, temp, lr_rho),
    ("Temperature", cg_hr_temp, cg_hr_temp, temp, temp, lr_temp),
    ("Pressure", cg_hr_pres, cg_hr_temp, pres, temp, lr_pres),
    ("Ux Velocity", cg_hr_ux, cg_hr_temp, ux, temp, lr_ux),
    ("Uy Velocity", cg_hr_uy, cg_hr_temp, uy, temp, lr_uy),
]

fig, axs = plt.subplots(5, 1, figsize=(9, 20))
plt.subplots_adjust(hspace=0.35)

for idx, (title, hr_arr, _, sg_arr, _, lr_arr) in enumerate(quantities):
    is_log = title in ("Density", "Temperature")

    hr_mean, hr_std = compute_mean_std(hr_arr, logspace=is_log)
    sg_mean, sg_std = compute_mean_std(sg_arr, logspace=is_log)
    lr_mean, lr_std = compute_mean_std(lr_arr, logspace=is_log)

    ax = axs[idx]

    ax.plot(hr_mean, lw=2, label=f"HR ({hr_resolution[0]}×{hr_resolution[1]})")
    ax.fill_between(
        np.arange(len(hr_mean)), hr_mean - hr_std, hr_mean + hr_std, alpha=0.25
    )

    ax.plot(sg_mean, lw=2, label=f"SG ({resolution[0]}×{resolution[1]})")
    ax.fill_between(
        np.arange(len(sg_mean)), sg_mean - sg_std, sg_mean + sg_std, alpha=0.25
    )

    ax.plot(lr_mean, lw=2, label=f"LR ({lr_resolution[0]}×{lr_resolution[1]})")
    ax.fill_between(
        np.arange(len(lr_mean)), lr_mean - lr_std, lr_mean + lr_std, alpha=0.25
    )

    ax.set_title(f"{title} (Avg over X) — Mean ± 1σ")
    ax.set_xlabel("Y")
    ax.set_ylabel(("log10 " if is_log else "") + title)

    if is_log:
        ax.set_yscale("linear")  # already plotting log(mean), so keep linear scale
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()

plt.tight_layout()
plt.savefig(save_path + "profiles_mean_with_std_all.png", dpi=200)
plt.close(fig)

print("profiles_mean_with_std_all.png saved")

quantities_cons = [
    ("Conserved Density", cg_hr_cons_rho, cons_rho, lr_cons_rho),
    ("Conserved MomX", cg_hr_cons_momx, cons_momx, lr_cons_momx),
    ("Conserved MomY", cg_hr_cons_momy, cons_momy, lr_cons_momy),
    ("Conserved Energy", cg_hr_cons_ener, cons_ener, lr_cons_ener),
    ("Passive Scalar", cg_hr_cons_ps, cons_ps, lr_cons_ps),
    ("fmcl (T < 1e5)", cg_hr_fmcl, fmcl, lr_fmcl),
]

fig, axs = plt.subplots(6, 1, figsize=(9, 24))
plt.subplots_adjust(hspace=0.4)

for idx, (title, hr_arr, sg_arr, lr_arr) in enumerate(quantities_cons):
    hr_mean, hr_std = compute_mean_std(hr_arr)
    sg_mean, sg_std = compute_mean_std(sg_arr)
    lr_mean, lr_std = compute_mean_std(lr_arr)

    ax = axs[idx]

    ax.plot(hr_mean, lw=2, label=f"HR ({hr_resolution[0]}×{hr_resolution[1]})")
    ax.fill_between(
        np.arange(len(hr_mean)), hr_mean - hr_std, hr_mean + hr_std, alpha=0.25
    )

    ax.plot(sg_mean, lw=2, label=f"SG ({resolution[0]}×{resolution[1]})")
    ax.fill_between(
        np.arange(len(sg_mean)), sg_mean - sg_std, sg_mean + sg_std, alpha=0.25
    )

    ax.plot(lr_mean, lw=2, label=f"LR ({lr_resolution[0]}×{lr_resolution[1]})")
    ax.fill_between(
        np.arange(len(lr_mean)), lr_mean - lr_std, lr_mean + lr_std, alpha=0.25
    )

    ax.set_title(f"{title} (Avg over X) — Mean ± 1σ")
    ax.set_xlabel("Y")
    ax.set_ylabel(title)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()

plt.tight_layout()
plt.savefig(save_path + "conserved_quantities_mean_with_std.png", dpi=200)
plt.close(fig)

print("conserved_quantities_mean_with_std.png saved")


def make_derived_plot(hr_field, sg_field, lr_field, title, ylabel, ax):
    hr_mean, hr_std = compute_mean_std(hr_field)
    sg_mean, sg_std = compute_mean_std(sg_field)
    lr_mean, lr_std = compute_mean_std(lr_field)

    ax.plot(hr_mean, lw=2, label=f"HR ({hr_resolution[0]}×{hr_resolution[1]})")
    ax.fill_between(
        np.arange(len(hr_mean)), hr_mean - hr_std, hr_mean + hr_std, alpha=0.25
    )

    ax.plot(sg_mean, lw=2, label=f"SG ({resolution[0]}×{resolution[1]})")
    ax.fill_between(
        np.arange(len(sg_mean)), sg_mean - sg_std, sg_mean + sg_std, alpha=0.25
    )

    ax.plot(lr_mean, lw=2, label=f"LR ({lr_resolution[0]}×{lr_resolution[1]})")
    ax.fill_between(
        np.arange(len(lr_mean)), lr_mean - lr_std, lr_mean + lr_std, alpha=0.25
    )

    ax.set_title(title)
    ax.set_xlabel("Y")
    ax.set_ylabel(ylabel)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()


# === Derived quantities ===
# Compute only what's needed for the next plot, then free immediately.

# 1. rho * ux
hr_rho_ux = cg_hr_rho * cg_hr_ux
sg_rho_ux = rho * ux
lr_rho_ux = lr_rho * lr_ux

# 2. rho * ux * uy
hr_rho_ux_uy = cg_hr_rho * cg_hr_ux * cg_hr_uy
sg_rho_ux_uy = rho * ux * uy
lr_rho_ux_uy = lr_rho * lr_ux * lr_uy

# 3. p + rho * uy^2
hr_mom_flux_y = cg_hr_pres + cg_hr_rho * cg_hr_uy**2
sg_mom_flux_y = pres + rho * uy**2
lr_mom_flux_y = lr_pres + lr_rho * lr_uy**2


# === Plot ===
fig, axs = plt.subplots(3, 1, figsize=(9, 15))
plt.subplots_adjust(hspace=0.35)

make_derived_plot(
    hr_rho_ux, sg_rho_ux, lr_rho_ux, "ρ uₓ (Avg over X) — Mean ± 1σ", "ρ uₓ", axs[0]
)

make_derived_plot(
    hr_rho_ux_uy,
    sg_rho_ux_uy,
    lr_rho_ux_uy,
    "ρ uₓ uᵧ (Avg over X) — Mean ± 1σ",
    "ρ uₓ uᵧ",
    axs[1],
)

make_derived_plot(
    hr_mom_flux_y,
    sg_mom_flux_y,
    lr_mom_flux_y,
    "p + ρ uᵧ² (Avg over X) — Mean ± 1σ",
    "Momentum Flux (y)",
    axs[2],
)

plt.tight_layout()
plt.savefig(save_path + "derived_quantities_mean_with_std.png", dpi=200)
plt.close(fig)

print("derived_quantities_mean_with_std.png saved")

# Free derived arrays now that the plot is done
del hr_rho_ux, sg_rho_ux, lr_rho_ux
del hr_rho_ux_uy, sg_rho_ux_uy, lr_rho_ux_uy
del hr_mom_flux_y, sg_mom_flux_y, lr_mom_flux_y
gc.collect()

nt = hr_rho.shape[0]
cg_hr_mass_x = np.zeros((nt, resolution[0], resolution[1]))
cg_hr_mass_y = np.zeros_like(cg_hr_mass_x)
cg_hr_T_xx = np.zeros_like(cg_hr_mass_x)
cg_hr_T_xy = np.zeros_like(cg_hr_mass_x)
cg_hr_T_yy = np.zeros_like(cg_hr_mass_x)
cg_hr_E_flux_x = np.zeros_like(cg_hr_mass_x)
cg_hr_E_flux_y = np.zeros_like(cg_hr_mass_x)

gamma = 1.6667

for i in tqdm(range(nt), desc="CG HR Fluxes"):
    hr_rho_i = hr_rho[i]
    hr_ux_i = hr_ux[i]
    hr_uy_i = hr_uy[i]
    hr_pres_i = hr_pres[i]

    hr_E_i = hr_pres_i / (gamma - 1) + 0.5 * hr_rho_i * (hr_ux_i**2 + hr_uy_i**2)

    hr_mass_x_i = hr_rho_i * hr_ux_i
    hr_mass_y_i = hr_rho_i * hr_uy_i
    hr_T_xx_i = hr_rho_i * hr_ux_i**2 + hr_pres_i
    hr_T_xy_i = hr_rho_i * hr_ux_i * hr_uy_i
    hr_T_yy_i = hr_rho_i * hr_uy_i**2 + hr_pres_i
    hr_E_flux_x_i = (hr_E_i + hr_pres_i) * hr_ux_i
    hr_E_flux_y_i = (hr_E_i + hr_pres_i) * hr_uy_i

    cg_hr_mass_x[i] = hr_sim_data.coarse_grain(hr_mass_x_i)
    cg_hr_mass_y[i] = hr_sim_data.coarse_grain(hr_mass_y_i)
    cg_hr_T_xx[i] = hr_sim_data.coarse_grain(hr_T_xx_i)
    cg_hr_T_xy[i] = hr_sim_data.coarse_grain(hr_T_xy_i)
    cg_hr_T_yy[i] = hr_sim_data.coarse_grain(hr_T_yy_i)
    cg_hr_E_flux_x[i] = hr_sim_data.coarse_grain(hr_E_flux_x_i)
    cg_hr_E_flux_y[i] = hr_sim_data.coarse_grain(hr_E_flux_y_i)

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


def compute_E(rho, ux, uy, pres, gamma=1.6667):
    return pres / (gamma - 1) + 0.5 * rho * (ux**2 + uy**2)


sg_E = compute_E(rho, ux, uy, pres)
lr_E = compute_E(lr_rho, lr_ux, lr_uy, lr_pres)

sg_E_flux_x = (sg_E + pres) * ux
sg_E_flux_y = (sg_E + pres) * uy
lr_E_flux_x = (lr_E + lr_pres) * lr_ux
lr_E_flux_y = (lr_E + lr_pres) * lr_uy

fig, axs = plt.subplots(4, 2, figsize=(12, 16))
plt.subplots_adjust(hspace=0.35)

print(cg_hr_mass_x.shape, sg_mass_x.shape, lr_mass_x.shape)
make_derived_plot(
    cg_hr_mass_x, sg_mass_x, lr_mass_x, "Mass Flux (ρ uₓ)", "ρ uₓ", axs[0, 0]
)
print(cg_hr_mass_y.shape, sg_mass_y.shape, lr_mass_y.shape)
make_derived_plot(
    cg_hr_mass_y, sg_mass_y, lr_mass_y, "Mass Flux (ρ uᵧ)", "ρ uᵧ", axs[0, 1]
)

make_derived_plot(
    cg_hr_T_xx, sg_T_xx, lr_T_xx, "Momentum Flux Tₓₓ = ρuₓ² + p", "Tₓₓ", axs[1, 0]
)
make_derived_plot(
    cg_hr_T_xy, sg_T_xy, lr_T_xy, "Momentum Flux Tₓᵧ = ρuₓuᵧ", "Tₓᵧ", axs[1, 1]
)

make_derived_plot(
    cg_hr_T_xy, sg_T_xy, lr_T_xy, "Momentum Flux Tᵧₓ = ρuₓuᵧ", "Tᵧₓ", axs[2, 0]
)
make_derived_plot(
    cg_hr_T_yy, sg_T_yy, lr_T_yy, "Momentum Flux Tᵧᵧ = ρuᵧ² + p", "Tᵧᵧ", axs[2, 1]
)

make_derived_plot(
    cg_hr_E_flux_x,
    sg_E_flux_x,
    lr_E_flux_x,
    "Energy Flux (E+p)uₓ",
    "(E+p)uₓ",
    axs[3, 0],
)
make_derived_plot(
    cg_hr_E_flux_y,
    sg_E_flux_y,
    lr_E_flux_y,
    "Energy Flux (E+p)uᵧ",
    "(E+p)uᵧ",
    axs[3, 1],
)

plt.tight_layout()
plt.savefig(save_path + "fluxes_mean_std.png", dpi=200)
plt.close(fig)

print("fluxes_mean_std.png saved")

cg_hr_div_mass = np.zeros_like(cg_hr_mass_x)
cg_hr_div_momx = np.zeros_like(cg_hr_mass_x)
cg_hr_div_momy = np.zeros_like(cg_hr_mass_x)

sg_div_mass = np.zeros_like(sg_mass_x)
sg_div_momx = np.zeros_like(sg_mass_x)
sg_div_momy = np.zeros_like(sg_mass_x)

lr_div_mass = np.zeros_like(lr_mass_x)
lr_div_momx = np.zeros_like(lr_mass_x)
lr_div_momy = np.zeros_like(lr_mass_x)

dy = sim_data.total_length / resolution[0]
dx = sim_data.total_width / resolution[1]

for i in tqdm(range(nt), desc="Divergence Fluxes"):
    cg_hr_div_mass[i] = divergence([cg_hr_mass_x[i], cg_hr_mass_y[i]], dx, dy)
    cg_hr_div_momx[i] = divergence([cg_hr_T_xx[i], cg_hr_T_xy[i]], dx, dy)
    cg_hr_div_momy[i] = divergence([cg_hr_T_xy[i], cg_hr_T_yy[i]], dx, dy)

    sg_div_mass[i] = divergence([sg_mass_x[i], sg_mass_y[i]], dx, dy)
    sg_div_momx[i] = divergence([sg_T_xx[i], sg_T_xy[i]], dx, dy)
    sg_div_momy[i] = divergence([sg_T_xy[i], sg_T_yy[i]], dx, dy)

    lr_div_mass[i] = divergence([lr_mass_x[i], lr_mass_y[i]], dx, dy)
    lr_div_momx[i] = divergence([lr_T_xx[i], lr_T_xy[i]], dx, dy)
    lr_div_momy[i] = divergence([lr_T_xy[i], lr_T_yy[i]], dx, dy)

fig, axs = plt.subplots(3, 1, figsize=(10, 13))
plt.subplots_adjust(hspace=0.35)

make_derived_plot(
    cg_hr_div_mass, sg_div_mass, lr_div_mass, "Div Mass Flux", "∇·(ρu)", axs[0]
)
make_derived_plot(
    cg_hr_div_momx, sg_div_momx, lr_div_momx, "Div MomX Flux", "∇·Tₓ", axs[1]
)
make_derived_plot(
    cg_hr_div_momy, sg_div_momy, lr_div_momy, "Div MomY Flux", "∇·Tᵧ", axs[2]
)

plt.tight_layout()
plt.savefig(save_path + "divergence_fluxes_mean_std.png", dpi=200)
plt.close(fig)

print("divergence_fluxes_mean_std.png saved")

# Free all flux/divergence arrays now that both plots are done
del cg_hr_mass_x, cg_hr_mass_y
del cg_hr_T_xx, cg_hr_T_xy, cg_hr_T_yy
del cg_hr_E_flux_x, cg_hr_E_flux_y
del cg_hr_div_mass, cg_hr_div_momx, cg_hr_div_momy
del sg_mass_x, sg_mass_y, sg_T_xx, sg_T_xy, sg_T_yy
del sg_E, sg_E_flux_x, sg_E_flux_y
del sg_div_mass, sg_div_momx, sg_div_momy
del lr_mass_x, lr_mass_y, lr_T_xx, lr_T_xy, lr_T_yy
del lr_E, lr_E_flux_x, lr_E_flux_y
del lr_div_mass, lr_div_momx, lr_div_momy
gc.collect()


def compute_cold_mass(rho_arr, temp_arr, nx, ny):
    dx_pc = sim_data.total_width / nx
    dy_pc = sim_data.total_length / ny
    area = dx_pc * dy_pc
    thr = np.power(10, 5.0)
    res = []
    for t in range(rho_arr.shape[0]):
        mask = temp_arr[t] < thr
        res.append(np.sum(rho_arr[t] * mask) * area)
    return np.array(res)


def compute_fmcl_mass_sg(rho_arr, fmcl_arr, nx, ny):
    dx_pc = sim_data.total_width / nx
    dy_pc = sim_data.total_length / ny
    area = dx_pc * dy_pc
    res = []
    for t in range(rho_arr.shape[0]):
        res.append(np.sum(rho_arr[t] * fmcl_arr[t]) * area)
    return np.array(res)


mass_hr = compute_cold_mass(cg_hr_rho, cg_hr_temp, resolution[0], resolution[1])
mass_sg = compute_cold_mass(rho, temp, resolution[0], resolution[1])
mass_lr = compute_cold_mass(lr_rho, lr_temp, lr_resolution[0], lr_resolution[1])

fmcl_sg = compute_fmcl_mass_sg(rho, fmcl, resolution[0], resolution[1])

# Truncate SG/LR arrays to the same number of common timesteps.
# Both lr_build_ism and subgrid_model restart at t=5 Myr, so their frames
# are already in the steady-state phase — no further trimming needed.
n_common = min(len(mass_hr), len(mass_sg), len(mass_lr))
mass_hr  = mass_hr[:n_common]
mass_sg  = mass_sg[:n_common]
mass_lr  = mass_lr[:n_common]
fmcl_sg  = fmcl_sg[:n_common]

# Physical time axes (Myr)
# HR: loaded with start=501, so frame 0 corresponds to some early snapshot;
#     use its own frame index (we don't know its exact start time here).
t_hr_frames = np.arange(n_common)  # HR frame indices (relative)
# SG / LR: frame 0 = t=RESTART_TIME_MYR (5 Myr)
t_restart_myr = RESTART_TIME_MYR + np.arange(n_common) * BIN_DT_MYR

# Linear fits in physical Myr for restarted runs, frame-index for HR
slope_hr,  intercept_hr  = np.polyfit(t_hr_frames,     mass_hr,  1)
slope_sg,  intercept_sg  = np.polyfit(t_restart_myr,   mass_sg,  1)
slope_lr,  intercept_lr  = np.polyfit(t_restart_myr,   mass_lr,  1)
slope_fmc, intercept_fmc = np.polyfit(t_restart_myr,   fmcl_sg,  1)

fit_hr  = slope_hr  * t_hr_frames   + intercept_hr
fit_sg  = slope_sg  * t_restart_myr + intercept_sg
fit_lr  = slope_lr  * t_restart_myr + intercept_lr
fit_fmc = slope_fmc * t_restart_myr + intercept_fmc

fig, ax = plt.subplots(figsize=(10, 6))

# HR is plotted on its own frame-index axis (secondary x); main axis in Myr
ax.axvline(RESTART_TIME_MYR, color="gray", ls="--", lw=1.2, label=f"Restart @ {RESTART_TIME_MYR} Myr")

ax.plot(t_restart_myr, mass_sg,  label="SG (subgrid_model)", lw=2)
ax.plot(t_restart_myr, mass_lr,  label="LR (lr_build_ism)",  lw=2)
ax.plot(t_restart_myr, fmcl_sg, label="SG fmcl", lw=2, ls="--")

ax.plot(t_restart_myr, fit_sg,  lw=1.8, ls=":",
        label=f"SG fit (d/dt = {slope_sg:.3e} Myr⁻¹)")
ax.plot(t_restart_myr, fit_lr,  lw=1.8, ls=":",
        label=f"LR fit (d/dt = {slope_lr:.3e} Myr⁻¹)")
ax.plot(t_restart_myr, fit_fmc, lw=1.8, ls=":",
        label=f"SG fmcl fit (d/dt = {slope_fmc:.3e} Myr⁻¹)")

ax.set_xlabel("Physical Time [Myr]")
ax.set_ylabel("Mass (g pc²/cm³)")
ax.set_title("Cold Gas Mass (T < 1e5 K) Evolution — Steady-State Phase (post-restart)")
ax.grid(True, ls="--", alpha=0.5)
ax.legend()
plt.tight_layout()
plt.savefig(save_path + "cold_mass_evolution.png", dpi=200)
plt.close()

print("Cold mass evolution plot saved (with fit slopes)")

fields_hr = [cg_hr_rho, cg_hr_temp, cg_hr_pres, cg_hr_ux, cg_hr_uy, cg_hr_ien]
fields_sg = [rho, temp, pres, ux, uy, ien]
fields_lr = [lr_rho, lr_temp, lr_pres, lr_ux, lr_uy, lr_ien]

titles = ["Density", "Temperature", "Pressure", "Ux", "Uy", "Internal Energy"]

fig, axs = plt.subplots(6, 3, figsize=(8, 20))

for i in range(6):
    f0_hr = fields_hr[i][0]
    f0_sg = fields_sg[i][0]
    f0_lr = fields_lr[i][0]

    arr0 = np.concatenate([f0_hr.flatten(), f0_sg.flatten(), f0_lr.flatten()])
    use_log = (i == 0 or i == 1)
    vmin0, vmax0, norm0 = compute_color_limits(arr0, use_log=use_log)

    axs[i, 0].imshow(f0_hr, origin="lower", cmap="plasma", norm=norm0)
    axs[i, 0].set_title(f"HR {titles[i]}")
    plt.colorbar(axs[i, 0].images[0], ax=axs[i, 0], fraction=0.035, pad=0.02)

    axs[i, 1].imshow(f0_sg, origin="lower", cmap="plasma", norm=norm0)
    axs[i, 1].set_title(f"SG {titles[i]}")
    plt.colorbar(axs[i, 1].images[0], ax=axs[i, 1], fraction=0.035, pad=0.02)

    axs[i, 2].imshow(f0_lr, origin="lower", cmap="plasma", norm=norm0)
    axs[i, 2].set_title(f"LR {titles[i]}")
    plt.colorbar(axs[i, 2].images[0], ax=axs[i, 2], fraction=0.035, pad=0.02)

plt.tight_layout()
plt.savefig(save_path + "all_fields_snapshot.png", dpi=200)
plt.close(fig)
print("Saved snapshot of all fields")

print("Saving all fields evolution animation...")
parallel_save_animation(render_frame_all_fields, range(nt), save_path + "all_fields_evolution.mp4", fps=10, num_workers=16)
print("Saved updated animation with correct dynamic colorbars")

cons_fields_hr = [
    cg_hr_cons_rho,
    cg_hr_cons_momx,
    cg_hr_cons_momy,
    cg_hr_cons_ener,
    cg_hr_cons_ps,
    cg_hr_fmcl,
]

cons_fields_sg = [cons_rho, cons_momx, cons_momy, cons_ener, cons_ps, fmcl]

cons_fields_lr = [
    lr_cons_rho,
    lr_cons_momx,
    lr_cons_momy,
    lr_cons_ener,
    lr_cons_ps,
    lr_fmcl,
]

cons_titles = [
    "Cons Density",
    "Cons MomX",
    "Cons MomY",
    "Cons Energy",
    "Cons Passive Scalar",
    "fmcl",
]




print("Saving conserved-field animation...")
parallel_save_animation(render_frame_cons_fields, range(nt), save_path + "cons_fields_evolution.mp4", fps=10, num_workers=16)
print("Saved conserved-field animation with dynamic colorbars")

print("Saving density evolution animation...")
parallel_save_animation(render_frame_rho, range(nt), save_path + "density_evolution.mp4", fps=10, num_workers=16)
print("Density evolution animation saved")

bins_pdf = np.logspace(4, 6, 200)
window = 10

print("Saving temperature PDF evolution animation...")
parallel_save_animation(render_frame_temp_pdf, range(nt), save_path + "temperature_pdf_evolution.mp4", fps=6.66667, num_workers=16)
print("Temperature PDF evolution animation saved")

# ============================================================
# Mean temperature PDFs ±1σ across all timesteps
# Volume / Mass / Emissivity weighted
# HR uses FULL-resolution fields
# ============================================================

Tmin = 1.1e4
Tmax = 0.9e6

bins = np.logspace(np.log10(Tmin), np.log10(Tmax), 50)
bin_centers = np.sqrt(bins[:-1] * bins[1:])


# ------------------------------------------------------------
# Generic weighted PDF function
# ------------------------------------------------------------


def compute_weighted_pdf_stats(temp_arr, weight_arr, weight_fn=None):
    """Compute mean/std of a weighted temperature PDF across all timesteps.

    Parameters
    ----------
    temp_arr   : array (nt, ny, nx)
    weight_arr : array (nt, ny, nx) or None when weight_fn is provided
    weight_fn  : optional callable(rho_frame, temp_frame) -> weight_frame
                 Used when weight_arr is None to compute weights on-the-fly
                 without keeping a full (nt, ny, nx) array in memory.
    """
    pdfs = []

    for t in range(temp_arr.shape[0]):
        vals = temp_arr[t].ravel()

        if weight_arr is not None:
            weights = weight_arr[t].ravel()
        else:
            raise ValueError("Provide either weight_arr or weight_fn")

        mask = (
            (vals >= Tmin)
            & (vals <= Tmax)
            & np.isfinite(vals)
            & np.isfinite(weights)
            & (weights > 0)
        )

        hist, _ = np.histogram(vals[mask], bins=bins, weights=weights[mask], density=True)
        pdfs.append(hist)

    pdfs = np.array(pdfs)
    return np.mean(pdfs, axis=0), np.std(pdfs, axis=0)


# ------------------------------------------------------------
# Weight definitions
# ------------------------------------------------------------

# =========================
# HR uses FULL resolution
# =========================
# Avoid allocating full (nt, 1024, 512) weight arrays.
# Pass lightweight sentinels; compute_weighted_pdf_stats handles them.

w_hr_vol  = np.ones_like(hr_temp)  # ones array, same shape but cheap in practice
w_hr_mass = hr_rho                 # mmap-backed, no copy
w_hr_emis = None                   # computed inline per-frame (see below)

# =========================
# SG
# =========================

w_sg_vol  = np.ones_like(temp)
w_sg_mass = rho
w_sg_emis = None

# =========================
# LR
# =========================

w_lr_vol  = np.ones_like(lr_temp)
w_lr_mass = lr_rho
w_lr_emis = None


# ------------------------------------------------------------
# Compute PDFs
# ------------------------------------------------------------

# Compute emissivity weights on-the-fly to avoid storing a full HR array
def _emis_weight(rho_t, temp_t):
    return rho_t**2 * lambda_cool(temp_t)


def compute_weighted_pdf_stats_fn(temp_arr, rho_arr, weight_fn):
    """Like compute_weighted_pdf_stats but derives weights per-frame."""
    pdfs = []
    for t in range(temp_arr.shape[0]):
        vals    = temp_arr[t].ravel()
        weights = weight_fn(rho_arr[t], temp_arr[t]).ravel()
        mask = (
            (vals >= Tmin)
            & (vals <= Tmax)
            & np.isfinite(vals)
            & np.isfinite(weights)
            & (weights > 0)
        )
        hist, _ = np.histogram(vals[mask], bins=bins, weights=weights[mask], density=True)
        pdfs.append(hist)
    pdfs = np.array(pdfs)
    return np.mean(pdfs, axis=0), np.std(pdfs, axis=0)


pdf_sets = {
    "Volume Weighted": (
        compute_weighted_pdf_stats(hr_temp, w_hr_vol),
        compute_weighted_pdf_stats(temp, w_sg_vol),
        compute_weighted_pdf_stats(lr_temp, w_lr_vol),
    ),
    "Mass Weighted": (
        compute_weighted_pdf_stats(hr_temp, w_hr_mass),
        compute_weighted_pdf_stats(temp, w_sg_mass),
        compute_weighted_pdf_stats(lr_temp, w_lr_mass),
    ),
    "Emissivity Weighted": (
        # compute per-frame to avoid storing full (nt,1024,512) emissivity array
        compute_weighted_pdf_stats_fn(hr_temp, hr_rho, _emis_weight),
        compute_weighted_pdf_stats_fn(temp,    rho,    _emis_weight),
        compute_weighted_pdf_stats_fn(lr_temp, lr_rho, _emis_weight),
    ),
}

del w_hr_vol, w_hr_mass, w_sg_vol, w_sg_mass, w_lr_vol, w_lr_mass
gc.collect()


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, axs = plt.subplots(3, 1, figsize=(7, 14))

for ax, (title, pdf_data) in zip(axs, pdf_sets.items()):
    (hr_mean, hr_std), (sg_mean, sg_std), (lr_mean, lr_std) = pdf_data

    ax.set_xscale("log")
    ax.set_yscale("log")

    # HR
    ax.plot(bin_centers, hr_mean, lw=2, label="HR")
    ax.fill_between(
        bin_centers,
        np.clip(hr_mean - hr_std, 1e-30, None),
        hr_mean + hr_std,
        alpha=0.25,
    )

    # SG
    ax.plot(bin_centers, sg_mean, lw=2, label="SG")
    ax.fill_between(
        bin_centers,
        np.clip(sg_mean - sg_std, 1e-30, None),
        sg_mean + sg_std,
        alpha=0.25,
    )

    # LR
    ax.plot(bin_centers, lr_mean, lw=2, label="LR")
    ax.fill_between(
        bin_centers,
        np.clip(lr_mean - lr_std, 1e-30, None),
        lr_mean + lr_std,
        alpha=0.25,
    )

    ax.set_xlim(Tmin, Tmax)
    ax.set_ylim(1e-8, 1e-2)

    ax.set_title(f"{title} Temperature PDF")
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("PDF")

    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()

plt.tight_layout()
plt.savefig(save_path + "temperature_pdfs_all_weightings.png", dpi=200)
plt.close(fig)

print("temperature_pdfs_all_weightings.png saved")

# ============================================================
# <n^2 Lambda(T)> profile vs y
# Averaged over x and time
# HR uses FULL-resolution fields
# ============================================================

# ------------------------------------------------------------
# Compute emissivity fields
# epsilon ~ rho^2 * Lambda(T)
# ------------------------------------------------------------

# Compute emissivity fields for the profile plot
# HR is mmap-backed so this triggers a full read, but only once
emis_hr = hr_rho**2 * lambda_cool(hr_temp)
emis_sg = rho**2    * lambda_cool(temp)
emis_lr = lr_rho**2 * lambda_cool(lr_temp)


# ------------------------------------------------------------
# Average along x and time
# ------------------------------------------------------------

# shape: (time, y, x)

# Average over x
emis_hr_xavg = np.mean(emis_hr, axis=2)
emis_sg_xavg = np.mean(emis_sg, axis=2)
emis_lr_xavg = np.mean(emis_lr, axis=2)

# Average over time
emis_hr_mean = np.mean(emis_hr_xavg, axis=0)
emis_hr_std = np.std(emis_hr_xavg, axis=0)

emis_sg_mean = np.mean(emis_sg_xavg, axis=0)
emis_sg_std = np.std(emis_sg_xavg, axis=0)

emis_lr_mean = np.mean(emis_lr_xavg, axis=0)
emis_lr_std = np.std(emis_lr_xavg, axis=0)


# ------------------------------------------------------------
# y coordinates
# ------------------------------------------------------------

y_hr = np.linspace(0, sim_data.total_length, hr_rho.shape[1])
y_sg = np.linspace(0, sim_data.total_length, rho.shape[1])
y_lr = np.linspace(0, sim_data.total_length, lr_rho.shape[1])


# ------------------------------------------------------------
# Integrated emissivity profiles
# Integral over y
# ------------------------------------------------------------

int_hr = np.trapezoid(emis_hr_mean, y_hr)
int_sg = np.trapezoid(emis_sg_mean, y_sg)
int_lr = np.trapezoid(emis_lr_mean, y_lr)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))

ax.set_yscale("log")

# HR
ax.plot(y_hr, emis_hr_mean, lw=2, label=rf"HR (Sig_c = {int_hr:.2e})")

ax.fill_between(
    y_hr,
    np.clip(emis_hr_mean - emis_hr_std, 1e-30, None),
    emis_hr_mean + emis_hr_std,
    alpha=0.25,
)

# SG
ax.plot(y_sg, emis_sg_mean, lw=2, label=rf"SG (Sig_c = {int_sg:.2e})")

ax.fill_between(
    y_sg,
    np.clip(emis_sg_mean - emis_sg_std, 1e-30, None),
    emis_sg_mean + emis_sg_std,
    alpha=0.25,
)

# LR
ax.plot(y_lr, emis_lr_mean, lw=2, label=rf"LR (Sig_c = {int_lr:.2e})")

ax.fill_between(
    y_lr,
    np.clip(emis_lr_mean - emis_lr_std, 1e-30, None),
    emis_lr_mean + emis_lr_std,
    alpha=0.25,
)

ax.set_xlabel("y")
ax.set_ylabel(r"$\langle n^2 \Lambda(T) \rangle$")

ax.set_ylim(2e-28, 1e-24)

ax.set_title(r"Mean Emissivity Profile vs $y$")

ax.grid(True, ls="--", alpha=0.5)
ax.legend()

plt.tight_layout()
plt.savefig(save_path + "emissivity_profile_vs_y.png", dpi=200)
plt.close(fig)

print("emissivity_profile_vs_y.png saved")

del emis_hr, emis_sg, emis_lr
del emis_hr_xavg, emis_sg_xavg, emis_lr_xavg
gc.collect()

# # --- data arrays (nt, ny, nx) ---
# nt, ny_hr, nx_hr = cg_hr_rho.shape
# ny_sg, nx_sg = rho.shape[1], rho.shape[2]
# ny_lr, nx_lr = lr_rho.shape[1], lr_rho.shape[2]

# # --- domain size in x [pc] ---
# Lx = 10.0

# # --- wavenumbers (1/pc) ---
# kx_hr = 2*np.pi*np.fft.rfftfreq(nx_hr, d=Lx/nx_hr)
# kx_sg = 2*np.pi*np.fft.rfftfreq(nx_sg, d=Lx/nx_sg)
# kx_lr = 2*np.pi*np.fft.rfftfreq(nx_lr, d=Lx/nx_lr)

# # --- storage ---
# spectra_hr, spectra_sg, spectra_lr = [], [], []

# # --- compute spectra ---
# for t in range(nt):
#     # HR
#     fhat_hr = np.fft.rfft(cg_hr_rho[t], axis=-1)        # FFT along x (last axis)
#     power_hr = np.mean(np.abs(fhat_hr)**2, axis=0)      # average over y
#     spectra_hr.append(power_hr)

#     # SG
#     fhat_sg = np.fft.rfft(rho[t], axis=-1)
#     power_sg = np.mean(np.abs(fhat_sg)**2, axis=0)
#     spectra_sg.append(power_sg)

#     # LR
#     fhat_lr = np.fft.rfft(lr_rho[t], axis=-1)
#     power_lr = np.mean(np.abs(fhat_lr)**2, axis=0)
#     spectra_lr.append(power_lr)

# spectra_hr = np.array(spectra_hr)
# spectra_sg = np.array(spectra_sg)
# spectra_lr = np.array(spectra_lr)

# # --- animate ---
# fig, ax = plt.subplots(figsize=(7,5))

# line_hr, = ax.loglog(kx_hr, spectra_hr[0] + 1e-30, label="HR", color="red")  # add epsilon to avoid log(0)
# line_sg, = ax.loglog(kx_sg, spectra_sg[0] + 1e-30, label="SG", color="blue")
# line_lr, = ax.loglog(kx_lr, spectra_lr[0] + 1e-30, label="LR", color="green")

# ax.set_xlabel(r"$k_x$ [1/pc]")
# ax.set_ylabel("Power Spectrum")
# ax.set_ylim(1e-12, 1e1)
# ax.set_title("Fourier Spectrum Evolution")
# ax.legend(loc = "lower right")

# def update(frame):
#     line_hr.set_ydata(spectra_hr[frame] + 1e-30)
#     line_sg.set_ydata(spectra_sg[frame] + 1e-30)
#     line_lr.set_ydata(spectra_lr[frame] + 1e-30)
#     ax.set_title(f"Fourier Spectrum (timestep {frame})")
#     return [line_hr, line_sg, line_lr]

# ani = animation.FuncAnimation(fig, update, frames=nt, interval=100, blit=False)

# ani.save(save_path + "fourier_spectrum_hr_sg_lr.mp4", writer="ffmpeg")
# plt.close(fig)
# print("Fourier spectrum evolution (HR, SG, LR) saved")

# --- domain sizes in pc (adjust if different)
Lx, Ly = 10.0, 20.0  # box size in x, y

# --- pixel sizes
dx_hr, dy_hr = Lx / cg_hr_rho.shape[2], Ly / cg_hr_rho.shape[1]
dx_sg, dy_sg = Lx / rho.shape[2], Ly / rho.shape[1]
dx_lr, dy_lr = Lx / lr_rho.shape[2], Ly / lr_rho.shape[1]

cell_area_hr = dx_hr * dy_hr
cell_area_sg = dx_sg * dy_sg
cell_area_lr = dx_lr * dy_lr

# --- cut indices (0 .. 512/7 pixels in y)
ycut_hr = cg_hr_rho.shape[1] // 7
ycut_sg = rho.shape[1] // 7
ycut_lr = lr_rho.shape[1] // 7

# --- integrate mass over region
mass_hr = np.sum(cg_hr_rho[:, :ycut_hr, :], axis=(1, 2)) * cell_area_hr
mass_sg = np.sum(rho[: len(mass_hr), :ycut_sg, :], axis=(1, 2)) * cell_area_sg
mass_lr = np.sum(lr_rho[: len(mass_hr), :ycut_lr, :], axis=(1, 2)) * cell_area_lr

# --- plot vs physical time (Myr) ---
# mass_sg / mass_lr / mass_hr are already truncated to n_common above.
n_region = min(mass_hr.shape[0], n_sg)
t_region_myr = RESTART_TIME_MYR + np.arange(n_region) * BIN_DT_MYR

fig, ax = plt.subplots(figsize=(6, 5))

ax.axvline(RESTART_TIME_MYR, color="gray", ls="--", lw=1.2,
           label=f"Restart @ {RESTART_TIME_MYR} Myr")
ax.plot(t_region_myr, mass_sg[:n_region], label="SG (subgrid_model)", color="blue")
ax.plot(t_region_myr, mass_lr[:n_region], label="LR (lr_build_ism)",  color="green")

ax.set_xlabel("Physical Time [Myr]")
ax.set_ylabel("Gas Mass [ρ·pc²]")
ax.set_title("Mass in initial cold region — Steady-State Phase (post-restart)")
ax.legend()

plt.tight_layout()
plt.savefig(save_path + "gas_mass_evolution.png", dpi=200)
plt.close(fig)
print("Gas mass evolution plot saved")
