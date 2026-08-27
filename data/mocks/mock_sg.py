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

from models.conv_nn.pdf_cnn import (
    compute_cooling_rate,
    lambda_cool,
    out_channels,
    snapshot_pred_16x8,
)


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


LOGT_ACTIVE_START = float(os.environ.get("LOGT_ACTIVE_START", "4.1"))
LOGT_ACTIVE_END = float(os.environ.get("LOGT_ACTIVE_END", "5.9"))
bins_pdf = np.logspace(LOGT_ACTIVE_START, LOGT_ACTIVE_END, 200)
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

        # Stitch frames with ffmpeg.
        # Use h264_nvenc (NVIDIA hardware encoder) since libx264 is not available
        # in this ffmpeg build. Falls back to mpeg4 if nvenc is unavailable.
        cmd = [
            'ffmpeg', '-y',
            '-r', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%04d.png'),
            '-c:v', 'h264_nvenc',
            '-preset', 'p4',
            '-pix_fmt', 'yuv420p',
            output_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res.returncode != 0:
            print(f"Error compiling video {output_path} with ffmpeg: {res.stderr.decode()}")
            raise RuntimeError(res.stderr.decode())
    finally:
        shutil.rmtree(temp_dir)


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


def render_frame_all_fields(frame, temp_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import numpy as np
    import os

    fig, axs = plt.subplots(6, 4, figsize=(11, 20))
    for i in range(6):
        f_hr = fields_hr[i][frame]
        f_cg_hr = fields_cg_hr[i][frame]
        f_sg = fields_sg[i][frame]
        f_lr = fields_lr[i][frame]

        arr = np.concatenate([f_hr.flatten(), f_cg_hr.flatten(), f_sg.flatten(), f_lr.flatten()])
        use_log = (i == 0 or i == 1)
        vmin, vmax, norm = compute_color_limits(arr, use_log=use_log)

        im0 = axs[i, 0].imshow(f_hr, origin='lower', cmap='plasma', norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax)
        axs[i, 0].set_title(f"HR {titles[i]}")
        plt.colorbar(im0, ax=axs[i, 0], fraction=0.035, pad=0.02)

        im1 = axs[i, 1].imshow(f_cg_hr, origin='lower', cmap='plasma', norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax)
        axs[i, 1].set_title(f"CG HR {titles[i]}")
        plt.colorbar(im1, ax=axs[i, 1], fraction=0.035, pad=0.02)

        im2 = axs[i, 2].imshow(f_sg, origin='lower', cmap='plasma', norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax)
        axs[i, 2].set_title(f"SG {titles[i]}")
        plt.colorbar(im2, ax=axs[i, 2], fraction=0.035, pad=0.02)

        im3 = axs[i, 3].imshow(f_lr, origin='lower', cmap='plasma', norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax)
        axs[i, 3].set_title(f"LR {titles[i]}")
        plt.colorbar(im3, ax=axs[i, 3], fraction=0.035, pad=0.02)

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

    fig, axs = plt.subplots(6, 4, figsize=(11, 20))
    for i in range(6):
        f_hr = cons_fields_hr[i][frame]
        f_cg_hr = cons_fields_cg_hr[i][frame]
        f_sg = cons_fields_sg[i][frame]
        f_lr = cons_fields_lr[i][frame]

        arr = np.concatenate([f_hr.flatten(), f_cg_hr.flatten(), f_sg.flatten(), f_lr.flatten()])
        use_log = (i == 0 or i == 3)
        vmin, vmax, norm = compute_color_limits(arr, use_log=use_log)

        im0 = axs[i, 0].imshow(f_hr, origin="lower", cmap="plasma", norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax)
        plt.colorbar(im0, ax=axs[i, 0], fraction=0.035, pad=0.02)
        axs[i, 0].set_title(f"HR {cons_titles[i]}")

        im1 = axs[i, 1].imshow(f_cg_hr, origin="lower", cmap="plasma", norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax)
        plt.colorbar(im1, ax=axs[i, 1], fraction=0.035, pad=0.02)
        axs[i, 1].set_title(f"CG HR {cons_titles[i]}")

        im2 = axs[i, 2].imshow(f_sg, origin="lower", cmap="plasma", norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax)
        plt.colorbar(im2, ax=axs[i, 2], fraction=0.035, pad=0.02)
        axs[i, 2].set_title(f"SG {cons_titles[i]}")

        im3 = axs[i, 3].imshow(f_lr, origin="lower", cmap="plasma", norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax)
        plt.colorbar(im3, ax=axs[i, 3], fraction=0.035, pad=0.02)
        axs[i, 3].set_title(f"LR {cons_titles[i]}")

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

    fig, axs = plt.subplots(1, 4, figsize=(14, 4.5))

    # To match original static colorbar, we compute limits from frame 0
    vmin_hr = hr_rho[0][hr_rho[0] > 0].min()
    vmax_hr = hr_rho[0].max()
    im_hr_rho = axs[0].imshow(hr_rho[frame], origin="lower", cmap="plasma", norm=LogNorm(vmin=vmin_hr, vmax=vmax_hr))
    axs[0].set_title(rf"HR (${hr_resolution[0]} \times {hr_resolution[1]}$) Density")
    plt.colorbar(im_hr_rho, ax=axs[0], fraction=0.046, pad=0.04)

    vmin_cg_hr = cg_hr_rho[0][cg_hr_rho[0] > 0].min()
    vmax_cg_hr = cg_hr_rho[0].max()
    im_cg_hr_rho = axs[1].imshow(cg_hr_rho[frame], origin="lower", cmap="plasma", norm=LogNorm(vmin=vmin_cg_hr, vmax=vmax_cg_hr))
    axs[1].set_title(rf"CG HR (${resolution[0]} \times {resolution[1]}$) Density")
    plt.colorbar(im_cg_hr_rho, ax=axs[1], fraction=0.046, pad=0.04)

    vmin_sg = rho[0][rho[0] > 0].min()
    vmax_sg = rho[0].max()
    im_rho = axs[2].imshow(rho[frame], origin="lower", cmap="plasma", norm=LogNorm(vmin=vmin_sg, vmax=vmax_sg))
    axs[2].set_title(rf"SG (${resolution[0]} \times {resolution[1]}$) Density")
    plt.colorbar(im_rho, ax=axs[2], fraction=0.046, pad=0.04)

    vmin_lr = lr_rho[0][lr_rho[0] > 0].min()
    vmax_lr = lr_rho[0].max()
    im_lr_rho = axs[3].imshow(lr_rho[frame], origin="lower", cmap="plasma", norm=LogNorm(vmin=vmin_lr, vmax=vmax_lr))
    axs[3].set_title(rf"LR (${lr_resolution[0]} \times {lr_resolution[1]}$) Density")
    plt.colorbar(im_lr_rho, ax=axs[3], fraction=0.046, pad=0.04)

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
    h_hr, _ = np.histogram(hr_temp[frame:end].ravel(), bins=bins_pdf, density=True)
    h_cg_hr, _ = np.histogram(cg_hr_temp[frame:end].ravel(), bins=bins_pdf, density=True)
    h_lr, _ = np.histogram(lr_temp[frame:end].ravel(), bins=bins_pdf, density=True)
    h_sg, _ = np.histogram(temp[frame:end].ravel(), bins=bins_pdf, density=True)

    ax.plot(bins_pdf[:-1], h_hr, lw=2.0, ls="-",  marker="^", markersize=4, label="HR")
    ax.plot(bins_pdf[:-1], h_cg_hr, lw=2.0, ls=":", marker="d", markersize=4, label="CG HR")
    ax.plot(bins_pdf[:-1], h_sg, lw=2.0, ls="-.", marker="o", markersize=4, label="SG")
    ax.plot(bins_pdf[:-1], h_lr, lw=2.0, ls="--", marker="s", markersize=4, label="LR")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(temp_dir, f"frame_{frame:04d}.png"), dpi=200)
    plt.close(fig)


def render_frame_cooling_rate(frame, temp_dir):
    """
    Render one frame of the cooling-rate comparison animation.
    Panels: HR Full-Res | CG HR | SG (snapshot_pred_16x8) | LR
    Matches the viridis colormap, percentile LogNorm scaling, and layout of pdf_plot.py.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np
    import os

    f_hr = emis_hr[frame]       # full-resolution HR emissivity
    f_cg_hr = emis_cg_hr[frame] # coarse-grained HR emissivity
    f_sg = emis_sg[frame]
    f_lr = emis_lr[frame]

    norm_cool = colors.LogNorm(vmin=cool_vmin, vmax=cool_vmax)
    cmap_cool = plt.get_cmap("viridis")

    fig = plt.figure(figsize=(19, 5))
    gs = fig.add_gridspec(
        1, 5,
        width_ratios=[1, 1, 1, 1, 0.04],
        wspace=0.25,
        top=0.86, bottom=0.15, left=0.05, right=0.94
    )

    t_myr = RESTART_TIME_MYR + frame * BIN_DT_MYR
    fig.suptitle(
        rf"Cooling Rate Comparison | t = {t_myr:.2f} Myr",
        fontsize=16, weight="bold", y=0.96
    )

    ax_hr = fig.add_subplot(gs[0])
    ax_cg_hr = fig.add_subplot(gs[1])
    ax_sg = fig.add_subplot(gs[2])
    ax_lr = fig.add_subplot(gs[3])

    cool_axes = [ax_hr, ax_cg_hr, ax_sg, ax_lr]
    cool_fields = [f_hr, f_cg_hr, f_sg, f_lr]
    cool_labels = [
        f"HR ({hr_resolution[0]}×{hr_resolution[1]})",
        f"CG HR ({resolution[0]}×{resolution[1]})",
        f"SG ({resolution[0]}×{resolution[1]})",
        f"LR ({lr_resolution[0]}×{lr_resolution[1]})",
    ]

    for ax, field, title in zip(cool_axes, cool_fields, cool_labels):
        ax.imshow(
            np.clip(field, cool_vmin, None),
            origin="lower",
            cmap=cmap_cool,
            norm=norm_cool,
        )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Y (pixels)", fontsize=11)
        ax.set_ylabel("X (pixels)", fontsize=11)

    # Shared colorbar for cooling rate
    cbar_ax_cool = fig.add_subplot(gs[4])
    sm_cool = plt.cm.ScalarMappable(cmap=cmap_cool, norm=norm_cool)
    sm_cool.set_array([])
    cbar_cool = fig.colorbar(sm_cool, cax=cbar_ax_cool)
    cbar_cool.set_label(r"Cooling Rate $n^2\Lambda(T)$ (erg / cm$^3$ / s)", fontsize=12)
    cbar_cool.ax.tick_params(labelsize=10)

    plt.savefig(os.path.join(temp_dir, f"frame_{frame:04d}.png"), dpi=150)
    plt.close(fig)


def worker_render_subgrid_pdf(frames_list, temp_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np, os

    nx, ny = resolution[0], resolution[1]
    nb = out_channels

    log_temp_centers = 0.5 * (np.log10(T_edges[:-1]) + np.log10(T_edges[1:]))
    active_bin_start = int(np.searchsorted(log_temp_centers, LOGT_ACTIVE_START))
    active_bin_end   = int(np.searchsorted(log_temp_centers, LOGT_ACTIVE_END))

    cmap_temp = plt.get_cmap("inferno")
    norm_temp = colors.Normalize(vmin=3.0, vmax=7.0)

    cmap_cool = plt.get_cmap("viridis")
    norm_cool = colors.LogNorm(vmin=cool_vmin, vmax=cool_vmax)

    cmap_gate = plt.get_cmap("plasma")
    norm_gate = colors.Normalize(vmin=0.0, vmax=1.0)

    fig = plt.figure(figsize=(32, 16))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 1.0], wspace=0.22,
                          left=0.03, right=0.97, top=0.92, bottom=0.06)

    # nx x ny grid of PDF mini-plots (taking up entire panel 0)
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

    # Side panel 1: Subgrid Resolved Temperature Map (aspect='auto' to match grid size)
    ax_temp = fig.add_subplot(gs[1])
    temp_0 = np.log10(temp[0] + 1e-8)
    im_temp = ax_temp.imshow(temp_0, origin="lower", cmap=cmap_temp, norm=norm_temp, aspect="auto")
    ax_temp.set_title("Subgrid $T$ Map", fontsize=15, weight="bold")
    ax_temp.set_xlabel("X (pixels)", fontsize=13)
    ax_temp.set_ylabel("Y (pixels)", fontsize=13)
    cbar_temp = plt.colorbar(im_temp, ax=ax_temp, fraction=0.046, pad=0.04)
    cbar_temp.set_label(r"$\log_{10}(T\ [\mathrm{K}])$ / Expectation Value", fontsize=12)

    # Side panel 2: Subgrid Cooling Rate Map (aspect='auto' to match grid size)
    ax_cool = fig.add_subplot(gs[2])
    cool_0 = np.clip(emis_sg[0], cool_vmin, None)
    im_cool = ax_cool.imshow(cool_0, origin="lower", cmap=cmap_cool, norm=norm_cool, aspect="auto")
    ax_cool.set_title("Subgrid Cooling Rate Map", fontsize=15, weight="bold")
    ax_cool.set_xlabel("X (pixels)", fontsize=13)
    ax_cool.set_ylabel("Y (pixels)", fontsize=13)
    cbar_cool = plt.colorbar(im_cool, ax=ax_cool, fraction=0.046, pad=0.04)
    cbar_cool.set_label(r"Cooling Rate $[\mathrm{erg}\ \mathrm{cm}^{-3}\ \mathrm{s}^{-1}]$", fontsize=12)

    # Side panel 3: Subgrid Mixing Gate Value Map (aspect='auto' to match grid size)
    ax_gate = fig.add_subplot(gs[3])
    gate_0 = pred_gate_all[0]
    im_gate = ax_gate.imshow(gate_0, origin="lower", cmap=cmap_gate, norm=norm_gate, aspect="auto")
    ax_gate.set_title("Subgrid Mixing Gate Map", fontsize=15, weight="bold")
    ax_gate.set_xlabel("X (pixels)", fontsize=13)
    ax_gate.set_ylabel("Y (pixels)", fontsize=13)
    cbar_gate = plt.colorbar(im_gate, ax=ax_gate, fraction=0.046, pad=0.04)
    cbar_gate.set_label(r"Gate Value $g \in [0, 1]$", fontsize=12)

    title_text = fig.suptitle("", fontsize=18, weight="bold")

    for frame_idx in frames_list:
        pdf_frame = pred_pdf_all[frame_idx]
        temp_frame = np.log10(temp[frame_idx] + 1e-8)
        cool_frame = np.clip(emis_sg[frame_idx], cool_vmin, None)
        gate_frame = pred_gate_all[frame_idx]
        t_myr_cur = RESTART_TIME_MYR + frame_idx * BIN_DT_MYR

        for i in range(nx):
            ii = nx - 1 - i  # flip vertically so row 0 is top
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
        im_gate.set_data(gate_frame)
        title_text.set_text(rf"Subgrid Predicted Temperature PDF Grid (${nx} \times {ny}$), $T$, Cooling, & Gate | $t = {t_myr_cur:.2f}$ Myr")
        
        frame_out = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
        fig.savefig(frame_out, dpi=120)

        if frame_idx == 0:
            fig.savefig(save_path + "subgrid_predicted_pdf_snapshot_t0.png", dpi=200)

    plt.close(fig)


def divergence(f, dx, dy):
    dFx_dx = np.gradient(f[0], dy, dx)[1]
    dFy_dy = np.gradient(f[1], dy, dx)[0]
    return dFx_dx + dFy_dy


# NOTE: lambda_cool and compute_cooling_rate are imported from models.conv_nn.pdf_cnn



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

# --- Physical Constants & Unit Conversions ---
m_H = 1.6726219e-24   # Hydrogen mass in g
k_B = 1.380649e-16    # Boltzmann constant in erg/K
M_sun = 1.98847e33    # Solar mass in g
yr = 3.15576e7        # Year in seconds
pc = 3.08568e18       # Parsec in cm
kpc = 3.08568e21      # Kiloparsec in cm

# --- Code Units ---
L_cgs = 3.08568e18    # length unit (1 pc)
M_cgs = 4.91417e31    # mass unit
T_cgs = 3.15576e13    # time unit (1 Myr)
mu = 0.62

# --- Derived Unit Conversions ---
V_cgs = L_cgs / T_cgs                                                # Velocity unit in cm/s (~9.778 km/s)
RHO_cgs = M_cgs / (L_cgs**3)                                         # Density unit in g/cm^3 (~1.67e-24 g/cm^3)
P_cgs = RHO_cgs * V_cgs**2                                           # Pressure unit in dyn/cm^2 or erg/cm^3

len_to_pc = L_cgs / pc                                               # 1.0 pc per code length
n_to_cm3 = RHO_cgs / (mu * m_H)                                      # ~1.61 cm^-3 per code density
T_to_K = V_cgs**2 * mu * m_H / k_B                                   # ~71.8 K per code temperature
P_over_kB_to_K_cm3 = P_cgs / k_B                                     # P/k_B in K cm^-3 per code pressure
vel_to_km_s = V_cgs / 1e5                                            # Velocity in km/s per code velocity (~ 9.778 km/s)
mflux_to_Msun_yr_kpc2 = (RHO_cgs * V_cgs) / (M_sun / (yr * kpc**2))   # ~0.0247 M_sun/yr/kpc^2 per code mass flux
unit_fix = 1.975e27                                                  # Code units energy rate conversion factor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Derive LR/SG resolution from the same env vars used by the rest of the pipeline.
# PDF_CNN_RESOLUTION = "hr_nx1,hr_nx2"; arrays are stored as (nx2, nx1) in bin files.
_cnn_res = os.environ.get("PDF_CNN_RESOLUTION", "1024,512").split(",")
_cnn_ds  = int(os.environ.get("PDF_CNN_DOWNSAMPLE", "64"))
resolution = (int(_cnn_res[0]) // _cnn_ds, int(_cnn_res[1]) // _cnn_ds)
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
ps = sim_data.ps
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
lr_ps = lr_sim_data.ps[: rho.shape[0]]

lr_sim_data.input_cons_data(lr_file_path, start=501)
lr_cons_rho = lr_sim_data.cons_rho[: rho.shape[0]]
lr_cons_momx = lr_sim_data.cons_momx[: rho.shape[0]]
lr_cons_momy = lr_sim_data.cons_momy[: rho.shape[0]]
lr_cons_ener = lr_sim_data.cons_ener[: rho.shape[0]]
lr_cons_ps = lr_sim_data.cons_ps[: rho.shape[0]]

lr_fmcl = (lr_temp < 1e5).astype(float)

hr_resolution = (int(_cnn_res[0]), int(_cnn_res[1]))
hr_downsample = _cnn_ds
hr_file_path = os.environ.get("HR_SIM_OUTPUT", os.path.join(PROJECT_ROOT, "simulation_outputs/hr_build_512"))
hr_sim_data = simulation_data()
hr_sim_data.resolution = hr_resolution
hr_sim_data.down_sample = hr_downsample
# hr_sim_data.input_data(hr_file_path)
# hr_rho = hr_sim_data.rho
# hr_temp = hr_sim_data.temp
hr_cache_base = os.environ.get("SUBGRID_CACHE_PATH", os.path.join(hr_file_path, "cache"))
hr_folder_path = os.path.join(hr_cache_base, f"sc{hr_resolution}_{hr_downsample}")
# Memory-map the large HR arrays so the OS pages them in on demand
# instead of loading every frame into RAM at once.
_n = rho.shape[0]
hr_rho  = np.load(f"{hr_folder_path}/rho.npy",      mmap_mode="r")[-_n:]
hr_temp = np.load(f"{hr_folder_path}/temp.npy",     mmap_mode="r")[-_n:]
hr_pres = np.load(f"{hr_folder_path}/pressure.npy", mmap_mode="r")[-_n:]
hr_ux   = np.load(f"{hr_folder_path}/ux.npy",       mmap_mode="r")[-_n:]
hr_uy   = np.load(f"{hr_folder_path}/uy.npy",       mmap_mode="r")[-_n:]
hr_ien  = np.load(f"{hr_folder_path}/eint.npy",     mmap_mode="r")[-_n:]
hr_ps   = np.load(f"{hr_folder_path}/ps.npy",       mmap_mode="r")[-_n:]

hr_cons_rho  = np.load(f"{hr_folder_path}/cons_rho.npy",  mmap_mode="r")[-_n:]
hr_cons_momx = np.load(f"{hr_folder_path}/cons_mx.npy",   mmap_mode="r")[-_n:]
hr_cons_momy = np.load(f"{hr_folder_path}/cons_my.npy",   mmap_mode="r")[-_n:]
hr_cons_ener = np.load(f"{hr_folder_path}/cons_ener.npy", mmap_mode="r")[-_n:]
hr_cons_ps   = np.load(f"{hr_folder_path}/cons_ps.npy",   mmap_mode="r")[-_n:]

hr_fmcl = (hr_temp < 1e5).astype(float)


def coarse_grain_array(arr, ds=64):
    """Coarse grains a 2D or 3D array by factor ds."""
    if arr.ndim == 2:
        return arr.reshape(arr.shape[0] // ds, ds, arr.shape[1] // ds, ds).mean(axis=(1, 3))
    elif arr.ndim == 3:
        return arr.reshape(arr.shape[0], arr.shape[1] // ds, ds, arr.shape[2] // ds, ds).mean(axis=(2, 4))
    else:
        raise ValueError(f"Unsupported array dimension: {arr.ndim}")


# Coarse-grained HR primitive fields
cg_hr_rho  = coarse_grain_array(hr_rho,  hr_downsample)
cg_hr_temp = coarse_grain_array(hr_temp, hr_downsample)
cg_hr_pres = coarse_grain_array(hr_pres, hr_downsample)
cg_hr_ux   = coarse_grain_array(hr_ux,   hr_downsample)
cg_hr_uy   = coarse_grain_array(hr_uy,   hr_downsample)
cg_hr_ien  = coarse_grain_array(hr_ien,  hr_downsample)

# Coarse-grained HR conserved fields
cg_hr_cons_rho  = coarse_grain_array(hr_cons_rho,  hr_downsample)
cg_hr_cons_momx = coarse_grain_array(hr_cons_momx, hr_downsample)
cg_hr_cons_momy = coarse_grain_array(hr_cons_momy, hr_downsample)
cg_hr_cons_ener = coarse_grain_array(hr_cons_ener, hr_downsample)
cg_hr_cons_ps   = coarse_grain_array(hr_cons_ps,   hr_downsample)
cg_hr_fmcl      = coarse_grain_array(hr_fmcl,      hr_downsample)


def compute_mean_std(arr, logspace=False):
    if logspace:
        arr = np.log10(arr)

    if arr.ndim == 3:
        arr_1d = np.mean(arr, axis=2)  # avg over X
    else:
        arr_1d = arr
    mean = arr_1d.mean(axis=0)  # mean over time
    std = arr_1d.std(axis=0)  # std over time

    return mean, std


quantities = [
    ("Density", cg_hr_rho * n_to_cm3, rho * n_to_cm3, lr_rho * n_to_cm3, r"$\log_{10}(n \ [\mathrm{cm}^{-3}])$", True),
    ("Temperature", cg_hr_temp, temp, lr_temp, r"$\log_{10}(T \ [\mathrm{K}])$", True),
    ("Pressure", cg_hr_pres * P_over_kB_to_K_cm3, pres * P_over_kB_to_K_cm3, lr_pres * P_over_kB_to_K_cm3, r"$P/k_B \ [\mathrm{K} \ \mathrm{cm}^{-3}]$", False),
    ("Ux Velocity", cg_hr_ux * vel_to_km_s, ux * vel_to_km_s, lr_ux * vel_to_km_s, r"$u_x \ [\mathrm{km} \ \mathrm{s}^{-1}]$", False),
    ("Uy Velocity", cg_hr_uy * vel_to_km_s, uy * vel_to_km_s, lr_uy * vel_to_km_s, r"$u_y \ [\mathrm{km} \ \mathrm{s}^{-1}]$", False),
]

fig, axs = plt.subplots(5, 1, figsize=(9, 20))
plt.subplots_adjust(hspace=0.35)

for idx, (title, hr_arr, sg_arr, lr_arr, ylabel, is_log) in enumerate(quantities):
    hr_mean, hr_std = compute_mean_std(hr_arr, logspace=is_log)
    sg_mean, sg_std = compute_mean_std(sg_arr, logspace=is_log)
    lr_mean, lr_std = compute_mean_std(lr_arr, logspace=is_log)

    y_hr = np.linspace(-10.0, 10.0, len(hr_mean))
    y_sg = np.linspace(-10.0, 10.0, len(sg_mean))
    y_lr = np.linspace(-10.0, 10.0, len(lr_mean))

    ax = axs[idx]

    ax.plot(y_hr, hr_mean, lw=2, ls="-",  marker="^", markersize=4, label=f"CG HR ({resolution[0]}×{resolution[1]})")
    ax.fill_between(y_hr, hr_mean - hr_std, hr_mean + hr_std, alpha=0.25)

    ax.plot(y_sg, sg_mean, lw=2, ls="-.", marker="o", markersize=5, label=f"SG ({resolution[0]}×{resolution[1]})")
    ax.fill_between(y_sg, sg_mean - sg_std, sg_mean + sg_std, alpha=0.25)

    ax.plot(y_lr, lr_mean, lw=2, ls="--", marker="s", markersize=5, label=f"LR ({lr_resolution[0]}×{lr_resolution[1]})")
    ax.fill_between(y_lr, lr_mean - lr_std, lr_mean + lr_std, alpha=0.25)

    ax.set_title(f"{title} (Avg over X) — Mean ± 1σ")
    ax.set_xlabel(r"$y \ [\mathrm{pc}]$")
    ax.set_ylabel(ylabel)

    if is_log:
        ax.set_yscale("linear")  # already plotting log(mean), so keep linear scale
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()

plt.tight_layout()
plt.savefig(save_path + "profiles_mean_with_std_all.png", dpi=200)
plt.close(fig)

print("profiles_mean_with_std_all.png saved")

quantities_cons = [
    ("Conserved Density", cg_hr_cons_rho * n_to_cm3, cons_rho * n_to_cm3, lr_cons_rho * n_to_cm3, r"$n \ [\mathrm{cm}^{-3}]$"),
    ("Conserved MomX", cg_hr_cons_momx * mflux_to_Msun_yr_kpc2, cons_momx * mflux_to_Msun_yr_kpc2, lr_cons_momx * mflux_to_Msun_yr_kpc2, r"$\rho u_x \ [M_\odot \ \mathrm{yr}^{-1} \ \mathrm{kpc}^{-2}]$"),
    ("Conserved MomY", cg_hr_cons_momy * mflux_to_Msun_yr_kpc2, cons_momy * mflux_to_Msun_yr_kpc2, lr_cons_momy * mflux_to_Msun_yr_kpc2, r"$\rho u_y \ [M_\odot \ \mathrm{yr}^{-1} \ \mathrm{kpc}^{-2}]$"),
    ("Conserved Energy", cg_hr_cons_ener * P_cgs, cons_ener * P_cgs, lr_cons_ener * P_cgs, r"$E \ [\mathrm{erg} \ \mathrm{cm}^{-3}]$"),
    ("Passive Scalar", cg_hr_cons_ps, cons_ps, lr_cons_ps, "Passive Scalar"),
    ("fmcl (T < 1e5)", cg_hr_fmcl, fmcl, lr_fmcl, r"$f_{\mathrm{mcl}} \ (T < 10^5 \ \mathrm{K})$"),
]

fig, axs = plt.subplots(6, 1, figsize=(9, 24))
plt.subplots_adjust(hspace=0.4)

for idx, (title, hr_arr, sg_arr, lr_arr, ylabel) in enumerate(quantities_cons):
    hr_mean, hr_std = compute_mean_std(hr_arr)
    sg_mean, sg_std = compute_mean_std(sg_arr)
    lr_mean, lr_std = compute_mean_std(lr_arr)

    y_hr = np.linspace(-10.0, 10.0, len(hr_mean))
    y_sg = np.linspace(-10.0, 10.0, len(sg_mean))
    y_lr = np.linspace(-10.0, 10.0, len(lr_mean))

    ax = axs[idx]

    ax.plot(y_hr, hr_mean, lw=2, ls="-",  marker="^", markersize=4, label=f"CG HR ({resolution[0]}×{resolution[1]})")
    ax.fill_between(y_hr, hr_mean - hr_std, hr_mean + hr_std, alpha=0.25)

    ax.plot(y_sg, sg_mean, lw=2, ls="-.", marker="o", markersize=5, label=f"SG ({resolution[0]}×{resolution[1]})")
    ax.fill_between(y_sg, sg_mean - sg_std, sg_mean + sg_std, alpha=0.25)

    ax.plot(y_lr, lr_mean, lw=2, ls="--", marker="s", markersize=5, label=f"LR ({lr_resolution[0]}×{lr_resolution[1]})")
    ax.fill_between(y_lr, lr_mean - lr_std, lr_mean + lr_std, alpha=0.25)

    ax.set_title(f"{title} (Avg over X) — Mean ± 1σ")
    ax.set_xlabel(r"$y \ [\mathrm{pc}]$")
    ax.set_ylabel(ylabel)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()

plt.tight_layout()
plt.savefig(save_path + "conserved_quantities_mean_with_std.png", dpi=200)
plt.close(fig)

print("conserved_quantities_mean_with_std.png saved")


def make_derived_plot(hr_field, sg_field, lr_field, title, ylabel, ax, conv_factor=1.0):
    hr_mean, hr_std = compute_mean_std(hr_field * conv_factor)
    sg_mean, sg_std = compute_mean_std(sg_field * conv_factor)
    lr_mean, lr_std = compute_mean_std(lr_field * conv_factor)

    y_hr = np.linspace(-10.0, 10.0, len(hr_mean))
    y_sg = np.linspace(-10.0, 10.0, len(sg_mean))
    y_lr = np.linspace(-10.0, 10.0, len(lr_mean))

    ax.plot(y_hr, hr_mean, lw=2, ls="-",  marker="^", markersize=4, label=f"CG HR ({resolution[0]}×{resolution[1]})")
    ax.fill_between(y_hr, hr_mean - hr_std, hr_mean + hr_std, alpha=0.25)

    ax.plot(y_sg, sg_mean, lw=2, ls="-.", marker="o", markersize=5, label=f"SG ({resolution[0]}×{resolution[1]})")
    ax.fill_between(y_sg, sg_mean - sg_std, sg_mean + sg_std, alpha=0.25)

    ax.plot(y_lr, lr_mean, lw=2, ls="--", marker="s", markersize=5, label=f"LR ({lr_resolution[0]}×{lr_resolution[1]})")
    ax.fill_between(y_lr, lr_mean - lr_std, lr_mean + lr_std, alpha=0.25)

    ax.set_title(title)
    ax.set_xlabel(r"$y \ [\mathrm{pc}]$")
    ax.set_ylabel(ylabel)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()


# === Derived quantities ===
# Compute only what's needed for the next plot, then free immediately.

# 1. rho * ux
cg_hr_rho_ux = coarse_grain_array(hr_rho * hr_ux, hr_downsample)
sg_rho_ux = rho * ux
lr_rho_ux = lr_rho * lr_ux

# 2. rho * ux * uy
cg_hr_rho_ux_uy = coarse_grain_array(hr_rho * hr_ux * hr_uy, hr_downsample)
sg_rho_ux_uy = rho * ux * uy
lr_rho_ux_uy = lr_rho * lr_ux * lr_uy

# 3. p + rho * uy^2
cg_hr_mom_flux_y = coarse_grain_array(hr_pres + hr_rho * hr_uy**2, hr_downsample)
sg_mom_flux_y = pres + rho * uy**2
lr_mom_flux_y = lr_pres + lr_rho * lr_uy**2


# === Plot ===
fig, axs = plt.subplots(3, 1, figsize=(9, 15))
plt.subplots_adjust(hspace=0.35)

make_derived_plot(
    cg_hr_rho_ux,
    sg_rho_ux,
    lr_rho_ux,
    r"$\rho u_x$ (Avg over X) — Mean ± 1σ",
    r"$\rho u_x \ [M_\odot \ \mathrm{yr}^{-1} \ \mathrm{kpc}^{-2}]$",
    axs[0],
    conv_factor=mflux_to_Msun_yr_kpc2,
)

make_derived_plot(
    cg_hr_rho_ux_uy,
    sg_rho_ux_uy,
    lr_rho_ux_uy,
    r"$\rho u_x u_y$ (Avg over X) — Mean ± 1σ",
    r"$\rho u_x u_y \ [\mathrm{dyn} \ \mathrm{cm}^{-2}]$",
    axs[1],
    conv_factor=P_cgs,
)

make_derived_plot(
    cg_hr_mom_flux_y,
    sg_mom_flux_y,
    lr_mom_flux_y,
    r"$p + \rho u_y^2$ (Avg over X) — Mean ± 1σ",
    r"$p + \rho u_y^2 \ [\mathrm{dyn} \ \mathrm{cm}^{-2}]$",
    axs[2],
    conv_factor=P_cgs,
)

plt.tight_layout()
plt.savefig(save_path + "derived_quantities_mean_with_std.png", dpi=200)
plt.close(fig)

print("derived_quantities_mean_with_std.png saved")

# Free derived arrays now that the plot is done
del cg_hr_rho_ux, sg_rho_ux, lr_rho_ux
del cg_hr_rho_ux_uy, sg_rho_ux_uy, lr_rho_ux_uy
del cg_hr_mom_flux_y, sg_mom_flux_y, lr_mom_flux_y
gc.collect()

nt = hr_rho.shape[0]


def compute_E(rho, ux, uy, pres, gamma=1.6667):
    return pres / (gamma - 1) + 0.5 * rho * (ux**2 + uy**2)


cg_hr_mass_x = coarse_grain_array(hr_rho * hr_ux, hr_downsample)
cg_hr_mass_y = coarse_grain_array(hr_rho * hr_uy, hr_downsample)
cg_hr_T_xx = coarse_grain_array(hr_rho * hr_ux**2 + hr_pres, hr_downsample)
cg_hr_T_xy = coarse_grain_array(hr_rho * hr_ux * hr_uy, hr_downsample)
cg_hr_T_yy = coarse_grain_array(hr_rho * hr_uy**2 + hr_pres, hr_downsample)

hr_E = compute_E(hr_rho, hr_ux, hr_uy, hr_pres)
cg_hr_E_flux_x = coarse_grain_array((hr_E + hr_pres) * hr_ux, hr_downsample)
cg_hr_E_flux_y = coarse_grain_array((hr_E + hr_pres) * hr_uy, hr_downsample)

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
    cg_hr_mass_x, sg_mass_x, lr_mass_x, r"Mass Flux ($\rho u_x$)", r"$\rho u_x \ [M_\odot \ \mathrm{yr}^{-1} \ \mathrm{kpc}^{-2}]$", axs[0, 0], conv_factor=mflux_to_Msun_yr_kpc2
)
make_derived_plot(
    cg_hr_mass_y, sg_mass_y, lr_mass_y, r"Mass Flux ($\rho u_y$)", r"$\rho u_y \ [M_\odot \ \mathrm{yr}^{-1} \ \mathrm{kpc}^{-2}]$", axs[0, 1], conv_factor=mflux_to_Msun_yr_kpc2
)

make_derived_plot(
    cg_hr_T_xx, sg_T_xx, lr_T_xx, r"Momentum Flux $T_{xx} = \rho u_x^2 + p$", r"$T_{xx} \ [\mathrm{dyn} \ \mathrm{cm}^{-2}]$", axs[1, 0], conv_factor=P_cgs
)
make_derived_plot(
    cg_hr_T_xy, sg_T_xy, lr_T_xy, r"Momentum Flux $T_{xy} = \rho u_x u_y$", r"$T_{xy} \ [\mathrm{dyn} \ \mathrm{cm}^{-2}]$", axs[1, 1], conv_factor=P_cgs
)

make_derived_plot(
    cg_hr_T_xy, sg_T_xy, lr_T_xy, r"Momentum Flux $T_{yx} = \rho u_x u_y$", r"$T_{yx} \ [\mathrm{dyn} \ \mathrm{cm}^{-2}]$", axs[2, 0], conv_factor=P_cgs
)
make_derived_plot(
    cg_hr_T_yy, sg_T_yy, lr_T_yy, r"Momentum Flux $T_{yy} = \rho u_y^2 + p$", r"$T_{yy} \ [\mathrm{dyn} \ \mathrm{cm}^{-2}]$", axs[2, 1], conv_factor=P_cgs
)

make_derived_plot(
    cg_hr_E_flux_x,
    sg_E_flux_x,
    lr_E_flux_x,
    r"Energy Flux $(E+p)u_x$",
    r"$(E+p)u_x \ [\mathrm{erg} \ \mathrm{cm}^{-2} \ \mathrm{s}^{-1}]$",
    axs[3, 0],
    conv_factor=P_cgs * V_cgs,
)
make_derived_plot(
    cg_hr_E_flux_y,
    sg_E_flux_y,
    lr_E_flux_y,
    r"Energy Flux $(E+p)u_y$",
    r"$(E+p)u_y \ [\mathrm{erg} \ \mathrm{cm}^{-2} \ \mathrm{s}^{-1}]$",
    axs[3, 1],
    conv_factor=P_cgs * V_cgs,
)

plt.tight_layout()
plt.savefig(save_path + "fluxes_mean_std.png", dpi=200)
plt.close(fig)

print("fluxes_mean_std.png saved")

hr_div_mass = np.zeros((nt, hr_resolution[0]))
hr_div_momx = np.zeros((nt, hr_resolution[0]))
hr_div_momy = np.zeros((nt, hr_resolution[0]))

sg_div_mass = np.zeros_like(sg_mass_x)
sg_div_momx = np.zeros_like(sg_mass_x)
sg_div_momy = np.zeros_like(sg_mass_x)

lr_div_mass = np.zeros_like(lr_mass_x)
lr_div_momx = np.zeros_like(lr_mass_x)
lr_div_momy = np.zeros_like(lr_mass_x)

dy = (sim_data.total_length * len_to_pc) / resolution[0]
dx = (sim_data.total_width * len_to_pc) / resolution[1]

dy_hr = (sim_data.total_length * len_to_pc) / hr_resolution[0]
dx_hr = (sim_data.total_width * len_to_pc) / hr_resolution[1]

for i in tqdm(range(nt), desc="Divergence Fluxes"):
    hr_mx_i = hr_rho[i] * hr_ux[i]
    hr_my_i = hr_rho[i] * hr_uy[i]
    hr_txx_i = hr_rho[i] * hr_ux[i]**2 + hr_pres[i]
    hr_txy_i = hr_rho[i] * hr_ux[i] * hr_uy[i]
    hr_tyy_i = hr_rho[i] * hr_uy[i]**2 + hr_pres[i]

    hr_div_mass[i] = np.mean(divergence([hr_mx_i, hr_my_i], dx_hr, dy_hr), axis=1)
    hr_div_momx[i] = np.mean(divergence([hr_txx_i, hr_txy_i], dx_hr, dy_hr), axis=1)
    hr_div_momy[i] = np.mean(divergence([hr_txy_i, hr_tyy_i], dx_hr, dy_hr), axis=1)

    sg_div_mass[i] = divergence([sg_mass_x[i], sg_mass_y[i]], dx, dy)
    sg_div_momx[i] = divergence([sg_T_xx[i], sg_T_xy[i]], dx, dy)
    sg_div_momy[i] = divergence([sg_T_xy[i], sg_T_yy[i]], dx, dy)

    lr_div_mass[i] = divergence([lr_mass_x[i], lr_mass_y[i]], dx, dy)
    lr_div_momx[i] = divergence([lr_T_xx[i], lr_T_xy[i]], dx, dy)
    lr_div_momy[i] = divergence([lr_T_xy[i], lr_T_yy[i]], dx, dy)

cg_hr_div_mass = hr_div_mass.reshape(nt, resolution[0], hr_downsample).mean(axis=2)
cg_hr_div_momx = hr_div_momx.reshape(nt, resolution[0], hr_downsample).mean(axis=2)
cg_hr_div_momy = hr_div_momy.reshape(nt, resolution[0], hr_downsample).mean(axis=2)

fig, axs = plt.subplots(3, 1, figsize=(10, 13))
plt.subplots_adjust(hspace=0.35)

make_derived_plot(
    cg_hr_div_mass, sg_div_mass, lr_div_mass, r"Div Mass Flux ($\nabla \cdot \mathbf{j}$)", r"$\nabla \cdot (\rho \mathbf{u}) \ [M_\odot \ \mathrm{yr}^{-1} \ \mathrm{kpc}^{-2} \ \mathrm{pc}^{-1}]$", axs[0], conv_factor=mflux_to_Msun_yr_kpc2
)
make_derived_plot(
    cg_hr_div_momx, sg_div_momx, lr_div_momx, r"Div MomX Flux ($\nabla \cdot \mathbf{T}_x$)", r"$\nabla \cdot \mathbf{T}_x \ [\mathrm{dyn} \ \mathrm{cm}^{-2} \ \mathrm{pc}^{-1}]$", axs[1], conv_factor=P_cgs
)
make_derived_plot(
    cg_hr_div_momy, sg_div_momy, lr_div_momy, r"Div MomY Flux ($\nabla \cdot \mathbf{T}_y$)", r"$\nabla \cdot \mathbf{T}_y \ [\mathrm{dyn} \ \mathrm{cm}^{-2} \ \mathrm{pc}^{-1}]$", axs[2], conv_factor=P_cgs
)

plt.tight_layout()
plt.savefig(save_path + "divergence_fluxes_mean_std.png", dpi=200)
plt.close(fig)

print("divergence_fluxes_mean_std.png saved")

# Free all flux/divergence arrays now that both plots are done
del cg_hr_mass_x, cg_hr_mass_y, cg_hr_T_xx, cg_hr_T_xy, cg_hr_T_yy
del hr_E, cg_hr_E_flux_x, cg_hr_E_flux_y
del hr_div_mass, hr_div_momx, hr_div_momy
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


mass_hr = compute_cold_mass(hr_rho, hr_temp, hr_rho.shape[1], hr_rho.shape[2])
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

# Physical time axis (Myr) — HR uses the last 500 snapshots corresponding to 5 -> 10 Myr
t_restart_myr = RESTART_TIME_MYR + np.arange(n_common) * BIN_DT_MYR

# Linear fits in physical Myr for all simulations
slope_hr,  intercept_hr  = np.polyfit(t_restart_myr,   mass_hr,  1)
slope_sg,  intercept_sg  = np.polyfit(t_restart_myr,   mass_sg,  1)
slope_lr,  intercept_lr  = np.polyfit(t_restart_myr,   mass_lr,  1)
slope_fmc, intercept_fmc = np.polyfit(t_restart_myr,   fmcl_sg,  1)

fit_hr  = slope_hr  * t_restart_myr + intercept_hr
fit_sg  = slope_sg  * t_restart_myr + intercept_sg
fit_lr  = slope_lr  * t_restart_myr + intercept_lr
fit_fmc = slope_fmc * t_restart_myr + intercept_fmc

fig, ax = plt.subplots(figsize=(10, 6))

# HR is plotted on its own frame-index axis (secondary x); main axis in Myr
ax.axvline(RESTART_TIME_MYR, color="gray", ls="--", lw=1.2, label=f"Restart @ {RESTART_TIME_MYR} Myr")

ax.plot(t_restart_myr, mass_hr, label="HR",                lw=2, ls="-",  marker="^", markersize=5)
ax.plot(t_restart_myr, mass_sg, label="SG (subgrid_model)", lw=2, ls="-.", marker="o", markersize=5)
ax.plot(t_restart_myr, mass_lr, label="LR (lr_build_ism)",  lw=2, ls="--", marker="s", markersize=5)
#ax.plot(t_restart_myr, fmcl_sg, label="SG fmcl", lw=2, ls="--")

ax.plot(t_restart_myr, fit_sg,  lw=1.8, ls="--",
        label=f"SG fit (d/dt = {slope_sg:.3e} Myr⁻¹)")
ax.plot(t_restart_myr, fit_lr,  lw=1.8, ls="--",
        label=f"LR fit (d/dt = {slope_lr:.3e} Myr⁻¹)")
ax.plot(t_restart_myr, fit_hr,  lw=1.8, ls="--",
        label=f"HR fit (d/dt = {slope_hr:.3e} Myr⁻¹)")
#ax.plot(t_restart_myr, fit_fmc, lw=1.8, ls=":",
#        label=f"SG fmcl fit (d/dt = {slope_fmc:.3e} Myr⁻¹)")

ax.set_xlabel("Physical Time [Myr]")
ax.set_ylabel("Mass (g pc²/cm³)")
ax.set_title("Cold Gas Mass (T < 1e5 K) Evolution — Steady-State Phase (post-restart)")
ax.grid(True, ls="--", alpha=0.5)
ax.legend()
plt.tight_layout()
plt.savefig(save_path + "cold_mass_evolution.png", dpi=200)
plt.close()

print("Cold mass evolution plot saved (with fit slopes)")

fields_hr = [hr_rho, hr_temp, hr_pres, hr_ux, hr_uy, hr_ien]
fields_cg_hr = [cg_hr_rho, cg_hr_temp, cg_hr_pres, cg_hr_ux, cg_hr_uy, cg_hr_ien]
fields_sg = [rho, temp, pres, ux, uy, ien]
fields_lr = [lr_rho, lr_temp, lr_pres, lr_ux, lr_uy, lr_ien]

titles = ["Density", "Temperature", "Pressure", "Ux", "Uy", "Internal Energy"]

fig, axs = plt.subplots(6, 4, figsize=(11, 20))

for i in range(6):
    f0_hr = fields_hr[i][0]
    f0_cg_hr = fields_cg_hr[i][0]
    f0_sg = fields_sg[i][0]
    f0_lr = fields_lr[i][0]

    arr0 = np.concatenate([f0_hr.flatten(), f0_cg_hr.flatten(), f0_sg.flatten(), f0_lr.flatten()])
    use_log = (i == 0 or i == 1)
    vmin0, vmax0, norm0 = compute_color_limits(arr0, use_log=use_log)

    axs[i, 0].imshow(f0_hr, origin="lower", cmap="plasma", norm=norm0)
    axs[i, 0].set_title(f"HR {titles[i]}")
    plt.colorbar(axs[i, 0].images[0], ax=axs[i, 0], fraction=0.035, pad=0.02)

    axs[i, 1].imshow(f0_cg_hr, origin="lower", cmap="plasma", norm=norm0)
    axs[i, 1].set_title(f"CG HR {titles[i]}")
    plt.colorbar(axs[i, 1].images[0], ax=axs[i, 1], fraction=0.035, pad=0.02)

    axs[i, 2].imshow(f0_sg, origin="lower", cmap="plasma", norm=norm0)
    axs[i, 2].set_title(f"SG {titles[i]}")
    plt.colorbar(axs[i, 2].images[0], ax=axs[i, 2], fraction=0.035, pad=0.02)

    axs[i, 3].imshow(f0_lr, origin="lower", cmap="plasma", norm=norm0)
    axs[i, 3].set_title(f"LR {titles[i]}")
    plt.colorbar(axs[i, 3].images[0], ax=axs[i, 3], fraction=0.035, pad=0.02)

plt.tight_layout()
plt.savefig(save_path + "all_fields_snapshot.png", dpi=200)
plt.close(fig)
print("Saved snapshot of all fields")

print("Saving all fields evolution animation...")
parallel_save_animation(render_frame_all_fields, range(nt), save_path + "all_fields_evolution.mp4", fps=10, num_workers=16)
print("Saved updated animation with correct dynamic colorbars")

cons_fields_hr = [
    hr_cons_rho,
    hr_cons_momx,
    hr_cons_momy,
    hr_cons_ener,
    hr_cons_ps,
    hr_fmcl,
]

cons_fields_cg_hr = [
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

Tmin = 10**LOGT_ACTIVE_START
Tmax = 10**LOGT_ACTIVE_END

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
# Compute emissivity fields in physical CGS units (erg cm^-3 s^-1)
# HR and LR use n^2 * Lambda(T) with n = rho * n_to_cm3
# SG uses CNN subgrid model PDF integration (converted from code units)
# ------------------------------------------------------------

n_hr = hr_rho * n_to_cm3
n_lr = lr_rho * n_to_cm3

emis_hr = n_hr**2 * lambda_cool(hr_temp, mask=True)
emis_cg_hr = coarse_grain_array(emis_hr, hr_downsample)
emis_lr = n_lr**2 * lambda_cool(lr_temp, mask=True)  # no narrow-T mask: LR uses full ISM cooling curve

_cnn_res = os.environ.get("PDF_CNN_RESOLUTION", "1024,512").split(",")
_fine_res = (int(_cnn_res[0]), int(_cnn_res[1]))
_cnn_ds = int(os.environ.get("PDF_CNN_DOWNSAMPLE", "64"))
T_edges = np.logspace(3.0, 7.0, out_channels + 1)
T_centers = np.sqrt(T_edges[:-1] * T_edges[1:])

emis_sg = np.zeros_like(rho)
pred_pdf_all = np.zeros((rho.shape[0], out_channels, *resolution), dtype=np.float32)
pred_gate_all = np.zeros((rho.shape[0], *resolution), dtype=np.float32)
print("Computing SG subgrid cooling rate using snapshot_pred_16x8...")
for t in tqdm(range(rho.shape[0])):
    pdf_t, gate_t = snapshot_pred_16x8(
        rho[t], temp[t], ux[t], uy[t], ps[t],
        fine_resolution=_fine_res, downsample=_cnn_ds,
        return_gate=True,
    )
    pred_pdf_all[t] = pdf_t
    pred_gate_all[t] = gate_t
    cool_code = compute_cooling_rate(
        pdf_t, T_centers,
        is_pdf=True, rho_cg=rho[t]
    )
    # Convert code units cooling rate back to CGS emissivity (erg cm^-3 s^-1)
    # compute_cooling_rate already computes n_code^2 * sum(PDF*Lambda) * unit_fix
    emis_sg[t] = cool_code / unit_fix

all_pos_cool = np.concatenate([
    emis_hr[emis_hr > 0],
    emis_cg_hr[emis_cg_hr > 0],
    emis_sg[emis_sg > 0],
    emis_lr[emis_lr > 0],
])
if len(all_pos_cool) > 0:
    cool_vmin = max(np.percentile(all_pos_cool, 1), 1e-30)
    cool_vmax = np.percentile(all_pos_cool, 99)
else:
    cool_vmin, cool_vmax = 1e-28, 1e-18


# ------------------------------------------------------------
# Compute PDFs
# ------------------------------------------------------------

# Compute emissivity weights on-the-fly to avoid storing a full HR array
def _emis_weight(rho_t, temp_t):
    n_t = rho_t * n_to_cm3
    return n_t**2 * lambda_cool(temp_t, mask=True)



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
        # compute per-frame for HR/LR using lambda_cool; pass precomputed emis_sg for SG
        compute_weighted_pdf_stats_fn(hr_temp, hr_rho, _emis_weight),
        compute_weighted_pdf_stats(temp, emis_sg),
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
    ax.plot(bin_centers, hr_mean, lw=2, ls="-", marker="^", markersize=4, label="HR")
    ax.fill_between(
        bin_centers,
        np.clip(hr_mean - hr_std, 1e-30, None),
        hr_mean + hr_std,
        alpha=0.25,
    )

    # SG
    ax.plot(bin_centers, sg_mean, lw=2, ls="-.", marker="o", markersize=5, label="SG")
    ax.fill_between(
        bin_centers,
        np.clip(sg_mean - sg_std, 1e-30, None),
        sg_mean + sg_std,
        alpha=0.25,
    )

    # LR
    ax.plot(bin_centers, lr_mean, lw=2, ls="--", marker="s", markersize=5, label="LR")
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
# CG HR uses Coarse-Grained HR fields
# ============================================================



# ------------------------------------------------------------
# Average along x and time
# ------------------------------------------------------------

# shape: (time, y, x)

# Average over x
emis_cg_hr_xavg = np.mean(emis_cg_hr, axis=2)
emis_sg_xavg = np.mean(emis_sg, axis=2)
emis_lr_xavg = np.mean(emis_lr, axis=2)

# Average over time
emis_cg_hr_mean = np.mean(emis_cg_hr_xavg, axis=0)
emis_cg_hr_std = np.std(emis_cg_hr_xavg, axis=0)

emis_sg_mean = np.mean(emis_sg_xavg, axis=0)
emis_sg_std = np.std(emis_sg_xavg, axis=0)

emis_lr_mean = np.mean(emis_lr_xavg, axis=0)
emis_lr_std = np.std(emis_lr_xavg, axis=0)


# ------------------------------------------------------------
# y coordinates
# ------------------------------------------------------------

y_cg_hr = np.linspace(-10.0, 10.0, emis_cg_hr.shape[1])
y_sg = np.linspace(-10.0, 10.0, rho.shape[1])
y_lr = np.linspace(-10.0, 10.0, lr_rho.shape[1])


# ------------------------------------------------------------
# Integrated emissivity profiles
# Integral over y
# ------------------------------------------------------------

int_cg_hr = np.trapezoid(emis_cg_hr_mean, y_cg_hr)
int_sg = np.trapezoid(emis_sg_mean, y_sg)
int_lr = np.trapezoid(emis_lr_mean, y_lr)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))

ax.set_yscale("log")

# CG HR
ax.plot(y_cg_hr, emis_cg_hr_mean, lw=2, ls="-", marker="^", markersize=4, label=rf"CG HR (Sig_c = {int_cg_hr:.2e})")

ax.fill_between(
    y_cg_hr,
    np.clip(emis_cg_hr_mean - emis_cg_hr_std, 1e-30, None),
    emis_cg_hr_mean + emis_cg_hr_std,
    alpha=0.25,
)

# SG
ax.plot(y_sg, emis_sg_mean, lw=2, ls="-.", marker="o", markersize=5, label=rf"SG (Sig_c = {int_sg:.2e})")

ax.fill_between(
    y_sg,
    np.clip(emis_sg_mean - emis_sg_std, 1e-30, None),
    emis_sg_mean + emis_sg_std,
    alpha=0.25,
)

# LR
ax.plot(y_lr, emis_lr_mean, lw=2, ls="--", marker="s", markersize=5, label=rf"LR (Sig_c = {int_lr:.2e})")

ax.fill_between(
    y_lr,
    np.clip(emis_lr_mean - emis_lr_std, 1e-30, None),
    emis_lr_mean + emis_lr_std,
    alpha=0.25,
)

ax.set_xlabel(r"$y \ [\mathrm{pc}]$")
ax.set_ylabel(r"$\langle n^2 \Lambda(T) \rangle \ [\mathrm{erg} \ \mathrm{cm}^{-3} \ \mathrm{s}^{-1}]$")

all_emis_vals = np.concatenate([emis_cg_hr_mean, emis_sg_mean, emis_lr_mean])
pos_vals = all_emis_vals[all_emis_vals > 0]
if len(pos_vals) > 0:
    ymin = max(pos_vals.min() * 0.5, 1e-10)
    ymax = pos_vals.max() * 2.0
    # ax.set_ylim(ymin, ymax)

ax.set_title(r"Mean Cooling Rate Profile vs $y$")

ax.grid(True, ls="--", alpha=0.5)
ax.legend()

plt.tight_layout()
plt.savefig(save_path + "emissivity_profile_vs_y.png", dpi=200)
plt.close(fig)

print("emissivity_profile_vs_y.png saved")

print("Saving cooling rate evolution animation...")
parallel_save_animation(
    render_frame_cooling_rate,
    range(nt),
    save_path + "cooling_rate_evolution.mp4",
    fps=10,
    num_workers=16,
)
print("Cooling rate evolution animation saved")

print("Saving subgrid predicted PDF, temperature, and cooling evolution animation...")
parallel_chunk_animation(
    worker_render_subgrid_pdf,
    nt,
    save_path + "subgrid_predicted_pdf_evolution.mp4",
    fps=10,
    num_workers=8,
)
print("Subgrid predicted PDF evolution animation saved")

del emis_hr, emis_cg_hr, emis_sg, emis_lr
del pred_pdf_all, pred_gate_all
del emis_cg_hr_xavg, emis_sg_xavg, emis_lr_xavg
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
dx_hr, dy_hr = Lx / hr_rho.shape[2], Ly / hr_rho.shape[1]
dx_sg, dy_sg = Lx / rho.shape[2], Ly / rho.shape[1]
dx_lr, dy_lr = Lx / lr_rho.shape[2], Ly / lr_rho.shape[1]

cell_area_hr = dx_hr * dy_hr
cell_area_sg = dx_sg * dy_sg
cell_area_lr = dx_lr * dy_lr

# --- cut indices (0 .. 512/7 pixels in y)
ycut_hr = hr_rho.shape[1] // 7
ycut_sg = rho.shape[1] // 7
ycut_lr = lr_rho.shape[1] // 7

# --- integrate mass over region
mass_hr = np.sum(hr_rho[:, :ycut_hr, :], axis=(1, 2)) * cell_area_hr
mass_sg = np.sum(rho[: len(mass_hr), :ycut_sg, :], axis=(1, 2)) * cell_area_sg
mass_lr = np.sum(lr_rho[: len(mass_hr), :ycut_lr, :], axis=(1, 2)) * cell_area_lr

# --- plot vs physical time (Myr) ---
# mass_sg / mass_lr / mass_hr are already truncated to n_common above.
n_region = min(mass_hr.shape[0], n_sg)
t_region_myr = RESTART_TIME_MYR + np.arange(n_region) * BIN_DT_MYR

fig, ax = plt.subplots(figsize=(6, 5))

ax.axvline(RESTART_TIME_MYR, color="gray", ls="--", lw=1.2,
           label=f"Restart @ {RESTART_TIME_MYR} Myr")
ax.plot(t_region_myr, mass_hr[:n_region], label="HR",                color="red",   ls="-",  marker="^", markersize=5)
ax.plot(t_region_myr, mass_sg[:n_region], label="SG (subgrid_model)", color="blue",  ls="-.", marker="o", markersize=5)
ax.plot(t_region_myr, mass_lr[:n_region], label="LR (lr_build_ism)",  color="green", ls="--", marker="s", markersize=5)

ax.set_xlabel("Physical Time [Myr]")
ax.set_ylabel("Gas Mass [ρ·pc²]")
ax.set_title("Mass in initial cold region — Steady-State Phase (post-restart)")
ax.legend()

plt.tight_layout()
plt.savefig(save_path + "gas_mass_evolution.png", dpi=200)
plt.close(fig)
print("Gas mass evolution plot saved")
