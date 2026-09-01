#!/usr/bin/env python3
"""
visualize_sweeps.py

Generates visualization outputs for parameter sweep simulation folders:
1. Parallel MP4 & GIF animations for the density field comparing HR, LR, and Subgrid Model simulations (using multiprocessing).
2. 3-panel profile plot comparing steady-state (second half of simulation) mean ± std profiles of
   density <log10(n)>, temperature <log10(T)>, and perpendicular mass flux <rho * v_y>.

Uses `ergane` for lazy loading, physical unit parsing, and AthenaK binary output reading.
"""

import os
import sys
import argparse
import tempfile
import shutil
import subprocess
import multiprocessing
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

# Ensure project root and ergane are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ergane import SimulationData, Units

# Conversion constants
PC_IN_CM = 3.08568e18




def _render_frame_worker(task_info):
    """
    Worker function executed in parallel across CPU cores to render a single frame.
    """
    i, (n_hr, n_lr, n_sg), datafolder_hr, datafolder_lr, datafolder_sg, athinp_hr, athinp_lr, athinp_sg, extent_hr, extent_lr, extent_sg, vmin, vmax, temp_dir = task_info
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from ergane import SimulationData
    import os

    sim_h = SimulationData(datafolder=datafolder_hr, athinp=athinp_hr)
    sim_l = SimulationData(datafolder=datafolder_lr, athinp=athinp_lr)
    sim_s = SimulationData(datafolder=datafolder_sg, athinp=athinp_sg)

    f_h = sim_h.get_frame(n_hr)
    f_l = sim_l.get_frame(n_lr)
    f_s = sim_s.get_frame(n_sg)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True, constrained_layout=True)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    if f_h.density is not None:
        axes[0].imshow(f_h.density, origin='lower', extent=extent_hr, norm=norm, cmap='viridis', aspect='auto')
    axes[0].set_title("HR Build (512x256)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("x [pc]")
    axes[0].set_ylabel("y (perpendicular) [pc]")

    if f_l.density is not None:
        axes[1].imshow(f_l.density, origin='lower', extent=extent_lr, norm=norm, cmap='viridis', aspect='auto')
    axes[1].set_title("LR Build (32x16, ISM Cooling)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("x [pc]")

    if f_s.density is not None:
        im2 = axes[2].imshow(f_s.density, origin='lower', extent=extent_sg, norm=norm, cmap='viridis', aspect='auto')
    axes[2].set_title("Subgrid Model (32x16, CNN Closure)", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("x [pc]")

    cbar = fig.colorbar(im2, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label(f"Density [{sim_h.units.label('density')}]", fontsize=11)

    t_myr = f_h.time / sim_h.units.time
    fig.suptitle(f"Density Field Evolution — Time: {t_myr:.2f} Myr", fontsize=14, fontweight='bold')

    out_file = os.path.join(temp_dir, f"frame_{i:04d}.png")
    fig.savefig(out_file, dpi=120)
    plt.close(fig)


def create_density_animation(sim_hr, sim_lr, sim_sg, output_mp4, output_gif=None, fps=30, max_frames=None, num_workers=16):
    """
    Creates a side-by-side 3-panel animation of the 2D density field using multiprocessing parallel rendering across CPU cores.
    """
    print(f"[Animation] Rendering density animation in parallel using {num_workers} workers...")
    
    hr_frames = sim_hr.frame_numbers
    lr_frames = sim_lr.frame_numbers
    sg_frames = sim_sg.frame_numbers
    
    total_frames = len(hr_frames)
    if max_frames and total_frames > max_frames:
        step = max(1, total_frames // max_frames)
        selected_indices = list(range(0, total_frames, step))
    else:
        selected_indices = list(range(total_frames))
        
    sample_indices = [0, len(selected_indices) // 2, -1]
    dens_vals = []
    for idx in sample_indices:
        f_idx = selected_indices[idx]
        for sim in (sim_hr, sim_lr, sim_sg):
            try:
                frame_num = sim.frame_numbers[min(f_idx, len(sim.frame_numbers) - 1)]
                f = sim.get_frame(frame_num)
                if f.density is not None:
                    dens_vals.append(f.density)
            except Exception:
                pass
                
    if dens_vals:
        all_dens = np.concatenate([d.flatten() for d in dens_vals if d is not None])
        pos_dens = all_dens[all_dens > 0]
        vmin = np.percentile(pos_dens, 1) if pos_dens.size else 1e-4
        vmax = np.percentile(pos_dens, 99) if pos_dens.size else 0.1
    else:
        vmin, vmax = 1e-4, 0.1

    f_hr0 = sim_hr.get_frame(hr_frames[0])
    f_lr0 = sim_lr.get_frame(lr_frames[0])
    f_sg0 = sim_sg.get_frame(sg_frames[0])

    extent_hr = [f_hr0.x[0]/PC_IN_CM, f_hr0.x[-1]/PC_IN_CM, f_hr0.y[0]/PC_IN_CM, f_hr0.y[-1]/PC_IN_CM]
    extent_lr = [f_lr0.x[0]/PC_IN_CM, f_lr0.x[-1]/PC_IN_CM, f_lr0.y[0]/PC_IN_CM, f_lr0.y[-1]/PC_IN_CM]
    extent_sg = [f_sg0.x[0]/PC_IN_CM, f_sg0.x[-1]/PC_IN_CM, f_sg0.y[0]/PC_IN_CM, f_sg0.y[-1]/PC_IN_CM]

    temp_dir = tempfile.mkdtemp()
    try:
        tasks = []
        for i, idx in enumerate(selected_indices):
            n_hr = hr_frames[min(idx, len(hr_frames) - 1)]
            n_lr = lr_frames[min(idx, len(lr_frames) - 1)]
            n_sg = sg_frames[min(idx, len(sg_frames) - 1)]
            
            task_info = (
                i, (n_hr, n_lr, n_sg),
                str(sim_hr._datafolder), str(sim_lr._datafolder), str(sim_sg._datafolder),
                str(sim_hr._athinp_path) if sim_hr._athinp_path else None,
                str(sim_lr._athinp_path) if sim_lr._athinp_path else None,
                str(sim_sg._athinp_path) if sim_sg._athinp_path else None,
                extent_hr, extent_lr, extent_sg, vmin, vmax, temp_dir
            )
            tasks.append(task_info)
            
        ctx = multiprocessing.get_context('fork')
        with ctx.Pool(processes=num_workers) as pool:
            list(tqdm(pool.imap(_render_frame_worker, tasks), total=len(tasks), desc="Rendering frames"))

        # Stitch with ffmpeg into mp4
        vf_filter = "scale='min(4096,iw)':'min(4096,ih)':force_original_aspect_ratio=decrease,pad=ceil(iw/2)*2:ceil(ih/2)*2"
        cmd_mp4 = [
            'ffmpeg', '-y',
            '-r', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%04d.png'),
            '-vf', vf_filter,
            '-c:v', 'h264_nvenc',
            '-preset', 'p4',
            '-pix_fmt', 'yuv420p',
            str(output_mp4)
        ]
        res = subprocess.run(cmd_mp4, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res.returncode != 0:
            cmd_fb = [
                'ffmpeg', '-y',
                '-r', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%04d.png'),
                '-vf', vf_filter,
                '-c:v', 'mpeg4',
                '-q:v', '2',
                '-pix_fmt', 'yuv420p',
                str(output_mp4)
            ]
            res = subprocess.run(cmd_fb, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        if res.returncode == 0:
            print(f"Saved MP4 animation: {output_mp4}")
        else:
            print(f"Error compiling MP4: {res.stderr.decode()}")

        if output_gif:
            cmd_gif = [
                'ffmpeg', '-y',
                '-r', str(min(fps, 10)),
                '-i', os.path.join(temp_dir, 'frame_%04d.png'),
                str(output_gif)
            ]
            res_gif = subprocess.run(cmd_gif, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if res_gif.returncode == 0:
                print(f"Saved GIF animation: {output_gif}")

    finally:
        shutil.rmtree(temp_dir)


def plot_perpendicular_profiles(sim_hr, sim_lr, sim_sg, output_png):
    """
    Plots steady-state profiles along the y-axis (perpendicular to the shear layer)
    averaged over the second half of each simulation (steady state), showing mean and std (shaded region):
    1. Density profile <log10(n)> [cm^-3]
    2. Temperature profile <log10(T)> [K]
    3. Perpendicular Mass Flux <rho * v_y> [M_sun / yr / kpc^2]
    """
    print(f"[Profiles] Computing steady-state perpendicular profiles (mean ± std) for {output_png.name}...")
    
    # Physical conversion constants
    m_H = 1.6726219e-24   # Hydrogen mass in g
    k_B = 1.380649e-16    # Boltzmann constant in erg/K
    M_sun = 1.98847e33    # Solar mass in g
    yr = 3.15576e7        # Year in seconds
    pc = 3.08568e18       # Parsec in cm
    kpc = 3.08568e21      # Kiloparsec in cm

    L_cgs = 3.08568e18    # 1 pc
    M_cgs = 4.91417e31    # mass unit
    T_cgs = 3.15576e13    # 1 Myr
    mu = 0.62

    V_cgs = L_cgs / T_cgs
    RHO_cgs = M_cgs / (L_cgs**3)

    len_to_pc = L_cgs / pc
    n_to_cm3 = RHO_cgs / (mu * m_H)
    T_to_K = V_cgs**2 * mu * m_H / k_B
    mflux_to_Msun_yr_kpc2 = (RHO_cgs * V_cgs) / (M_sun / (yr * kpc**2))

    sims_dict = {
        'HR (512x256)': (sim_hr, 'b', '-', 'hr'),
        'LR (32x16, ISM)': (sim_lr, 'r', '--', 'lr'),
        'Subgrid Model (32x16, CNN)': (sim_sg, 'g', '-.', 'sg')
    }

    results = {}

    for label, (sim, color, linestyle, mode) in sims_dict.items():
        sim.set_units(Units.code())
        start_frame = sim.n_frames // 2
        
        density_log_profiles = []
        temp_log_profiles = []
        mflux_profiles = []

        for num in sim.frame_numbers[start_frame:]:
            f = sim.get_frame(num)
            if f.density is None or f.pressure is None or f.vely is None:
                continue
            
            dens = f.density
            pres = f.pressure
            vely = f.vely

            temp_K = (pres / (dens + np.finfo(np.float64).tiny)) * T_to_K
            n_cm3 = dens * n_to_cm3

            # Spatial average across x-axis (axis=1) for each y row
            density_log_profiles.append(np.mean(np.log10(n_cm3 + np.finfo(np.float64).tiny), axis=1))
            temp_log_profiles.append(np.mean(np.log10(temp_K + np.finfo(np.float64).tiny), axis=1))
            mflux_profiles.append(np.mean(dens * vely * mflux_to_Msun_yr_kpc2, axis=1))
            

        dens_arr = np.asarray(density_log_profiles)
        temp_arr = np.asarray(temp_log_profiles)
        mflux_arr = np.asarray(mflux_profiles)

        # Coordinate y in pc
        f_last = sim.get_frame(sim.frame_numbers[-1])
        y_pc = f_last.yc * len_to_pc

        results[label] = {
            'y': y_pc,
            'color': color,
            'linestyle': linestyle,
            'mean_dens': np.mean(dens_arr, axis=0),
            'std_dens': np.std(dens_arr, axis=0),
            'mean_temp': np.mean(temp_arr, axis=0),
            'std_temp': np.std(temp_arr, axis=0),
            'mean_mflux': np.mean(mflux_arr, axis=0),
            'std_mflux': np.std(mflux_arr, axis=0),
        }

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(9, 12), constrained_layout=True)

    for label, res in results.items():
        y = res['y']
        c = res['color']
        ls = res['linestyle']

        # 1. Density Profile
        ax1.plot(y, res['mean_dens'], color=c, linestyle=ls, label=label, linewidth=2.0)
        ax1.fill_between(y, res['mean_dens'] - res['std_dens'], res['mean_dens'] + res['std_dens'], color=c, alpha=0.2)

        # 2. Temperature Profile
        ax2.plot(y, res['mean_temp'], color=c, linestyle=ls, label=label, linewidth=2.0)
        ax2.fill_between(y, res['mean_temp'] - res['std_temp'], res['mean_temp'] + res['std_temp'], color=c, alpha=0.2)

        # 3. Perpendicular Mass Flux Profile
        ax3.plot(y, res['mean_mflux'], color=c, linestyle=ls, label=label, linewidth=2.0)
        ax3.fill_between(y, res['mean_mflux'] - res['std_mflux'], res['mean_mflux'] + res['std_mflux'], color=c, alpha=0.2)

    # Formatting ax1 (Density)
    ax1.set_ylabel(r"$\log_{10}(n) \ [\mathrm{cm}^{-3}]$", fontsize=11)
    ax1.set_title("Steady-State Perpendicular Profiles (Mean ± Std over 2nd Half of Sim)", fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, which="both", ls=":", alpha=0.6)

    # Formatting ax2 (Temperature)
    ax2.set_ylabel(r"$\log_{10}(T \ [\mathrm{K}])$", fontsize=11)
    ax2.grid(True, which="both", ls=":", alpha=0.6)

    # Formatting ax3 (Perpendicular Mass Flux)
    ax3.set_xlabel(r"$y \ [\mathrm{pc}]$ (perpendicular to shear layer)", fontsize=11)
    ax3.set_ylabel(r"$\rho v_y \ [M_\odot \ \mathrm{yr}^{-1} \ \mathrm{kpc}^{-2}]$", fontsize=11)
    ax3.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.7)
    ax3.grid(True, which="both", ls=":", alpha=0.6)

    fig.savefig(str(output_png), dpi=200)
    print(f"Saved steady-state profiles plot: {output_png}")

    output_pdf = output_png.with_suffix('.pdf')
    fig.savefig(str(output_pdf))
    print(f"Saved steady-state profiles PDF: {output_pdf}")

    plt.close(fig)


def process_pair_directory(pair_dir: Path):
    """
    Loads simulation outputs in pair_dir and generates animations and profile plots.
    """
    pair_dir = Path(pair_dir)
    print("=" * 72)
    print(f"Processing visualizations for parameter pair: {pair_dir.name}")
    print("=" * 72)
    
    hr_dir = pair_dir / "hr_build_512x256"
    lr_dir = pair_dir / "lr_build_32x16"
    sg_dir = pair_dir / "subgrid_model_32x16"
    
    hr_athinput = pair_dir / "hr_512x256.athinput"
    lr_athinput = pair_dir / "lr_32x16.athinput"
    sg_athinput = pair_dir / "subgrid_32x16.athinput"
    
    if not (hr_dir.exists() and lr_dir.exists() and sg_dir.exists()):
        print(f"Error: Missing one or more simulation output subdirectories in {pair_dir}")
        return
        
    try:
        sim_hr = SimulationData(datafolder=hr_dir, athinp=hr_athinput if hr_athinput.exists() else None)
        sim_lr = SimulationData(datafolder=lr_dir, athinp=lr_athinput if lr_athinput.exists() else None)
        sim_sg = SimulationData(datafolder=sg_dir, athinp=sg_athinput if sg_athinput.exists() else None)
    except Exception as e:
        print(f"Error loading simulation data with ergane: {e}")
        return

    # 1. Density field animation (parallel using 16 workers)
    output_mp4 = pair_dir / "density_animation.mp4"
    output_gif = pair_dir / "density_animation.gif"
    create_density_animation(sim_hr, sim_lr, sim_sg, output_mp4=output_mp4, output_gif=output_gif, num_workers=16)
    
    # 2. Perpendicular profiles plot
    output_png = pair_dir / "profiles_perpendicular.png"
    plot_perpendicular_profiles(sim_hr, sim_lr, sim_sg, output_png=output_png)


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations (density animation & profiles) for parameter sweep runs.")
    parser.add_argument("sweep_dir", help="Path to a parameter pair folder or the sweeps root folder")
    args = parser.parse_args()
    
    sweep_path = Path(args.sweep_dir).resolve()
    
    if not sweep_path.exists():
        print(f"Error: Directory does not exist: {sweep_path}")
        sys.exit(1)
        
    if (sweep_path / "hr_build_512x256").exists():
        process_pair_directory(sweep_path)
    else:
        pair_dirs = sorted([d for d in sweep_path.iterdir() if d.is_dir() and (d / "hr_build_512x256").exists()])
        if not pair_dirs:
            print(f"No valid parameter pair folders found under {sweep_path}")
            sys.exit(1)
        for d in pair_dirs:
            process_pair_directory(d)


if __name__ == "__main__":
    main()
