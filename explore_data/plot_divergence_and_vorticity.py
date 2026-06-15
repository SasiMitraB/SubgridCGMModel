import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LogNorm
from tqdm import tqdm
import skimage.measure

# Add project root and data directory to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))

import bin_convert

def load_snapshot_data(file_path):
    """
    Load simulation snapshot variables: density, velx, vely, and grid coordinates.
    Works for both primitive (velx, vely) and conserved (mom1, mom2) variables.
    """
    fd = bin_convert.read_binary(file_path)
    
    # Grid info
    Nx1, Nx2 = fd["Nx1"], fd["Nx2"]
    x1min, x1max = fd["x1min"], fd["x1max"]
    x2min, x2max = fd["x2min"], fd["x2max"]
    
    # Calculate cell spacing
    dx = (x1max - x1min) / Nx1
    dy = (x2max - x2min) / Nx2
    
    # Load density
    rho = bin_convert.make_2D_array(fd, "dens")
    
    # Load or compute velocity components (u_x, u_y)
    var_names = fd["var_names"]
    if "velx" in var_names and "vely" in var_names:
        ux = bin_convert.make_2D_array(fd, "velx")
        uy = bin_convert.make_2D_array(fd, "vely")
    elif "mom1" in var_names and "mom2" in var_names:
        mom1 = bin_convert.make_2D_array(fd, "mom1")
        mom2 = bin_convert.make_2D_array(fd, "mom2")
        ux = mom1 / rho
        uy = mom2 / rho
    else:
        raise ValueError(f"Could not find velocity or momentum variables in file: {file_path}")
        
    return rho, ux, uy, dx, dy, x1min, x1max, x2min, x2max

def compute_divergence_and_vorticity(ux, uy, dx, dy):
    """
    Compute divergence and vorticity of a 2D velocity field.
    Assumes standard row-major numpy arrays where axis 0 is y and axis 1 is x.
    """
    # np.gradient returns (d/dy, d/dx) for a 2D array
    dux_dy, dux_dx = np.gradient(ux, dy, dx)
    duy_dy, duy_dx = np.gradient(uy, dy, dx)
    
    # Divergence = du_x/dx + du_y/dy
    div = dux_dx + duy_dy
    
    # Vorticity (z-component) = du_y/dx - du_x/dy
    vort = duy_dx - dux_dy
    
    return div, vort

def save_animation_with_progress(anim, path, total_frames, fps=10):
    """
    Save animation with a progress bar. Tries MP4 first, then falls back to GIF.
    """
    pbar = tqdm(total=total_frames, desc=f"Saving {os.path.basename(path)}")
    
    def progress_callback(current_frame, total_frames_unused):
        pbar.update(current_frame - pbar.n)
        
    try:
        # Try saving as MP4 using ffmpeg writer
        anim.save(path, writer="ffmpeg", fps=fps, progress_callback=progress_callback)
        print(f"\nSuccessfully saved animation to: {path}")
    except Exception as e:
        print(f"\nWarning: Failed to save as MP4 using ffmpeg: {e}")
        gif_path = path.replace(".mp4", ".gif")
        print(f"Falling back to saving as GIF using pillow writer at: {gif_path}")
        anim.save(gif_path, writer="pillow", fps=fps, progress_callback=progress_callback)
        print(f"Successfully saved fallback animation to: {gif_path}")
    finally:
        pbar.close()

def main():
    sim_bin_dir = os.path.join(PROJECT_ROOT, "simulation_outputs", "hr_build", "bin")
    output_dir = os.path.join(PROJECT_ROOT, "outputs", "explore_data")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Scanning simulation files in: {sim_bin_dir}")
    prim_files = sorted(glob.glob(os.path.join(sim_bin_dir, "KH.hydro_w.*.bin")))
    cons_files = sorted(glob.glob(os.path.join(sim_bin_dir, "KH.hydro_u.*.bin")))
    
    use_files = prim_files if len(prim_files) > 0 else cons_files
    if len(use_files) == 0:
        print("Error: No binary files found.")
        sys.exit(1)
        
    n_files = len(use_files)
    print(f"Found {n_files} snapshot files.")
    
    # --- Figure 1: Spatial Evolution Map (3 Snapshots x 3 Fields) ---
    print("\nGenerating Figure 1: Spatial map of Density, Divergence, and Vorticity...")
    snap_indices = [int(n_files * 0.1), int(n_files * 0.5), int(n_files * 0.9)]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 18), sharex=True, sharey=True)
    
    for row_idx, file_idx in enumerate(snap_indices):
        file_path = use_files[file_idx]
        file_name = os.path.basename(file_path)
        print(f"  Processing row {row_idx+1}/3: snapshot {file_idx} ({file_name})")
        
        rho, ux, uy, dx, dy, x1min, x1max, x2min, x2max = load_snapshot_data(file_path)
        div, vort = compute_divergence_and_vorticity(ux, uy, dx, dy)
        
        extent = [x1min, x1max, x2min, x2max]
        
        # 1. Density Map
        norm_rho = LogNorm(vmin=max(1e-5, rho.min()), vmax=rho.max()) if rho.max() / (rho.min() + 1e-10) > 10 else None
        im0 = axes[row_idx, 0].imshow(rho, origin="lower", extent=extent, cmap="plasma", norm=norm_rho)
        axes[row_idx, 0].set_title(f"Density | Step {file_idx}\n({file_name})")
        fig.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)
        
        # 2. Divergence Map
        div_limit = max(abs(div.min()), abs(div.max()))
        if div_limit == 0:
            div_limit = 1.0
        norm_div = TwoSlopeNorm(vmin=-div_limit, vcenter=0.0, vmax=div_limit)
        im1 = axes[row_idx, 1].imshow(div, origin="lower", extent=extent, cmap="coolwarm", norm=norm_div)
        axes[row_idx, 1].set_title(f"Divergence ($\\nabla \\cdot \\vec{{u}}$) | Step {file_idx}")
        fig.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)
        
        # 3. Vorticity Map
        vort_limit = max(abs(vort.min()), abs(vort.max()))
        if vort_limit == 0:
            vort_limit = 1.0
        norm_vort = TwoSlopeNorm(vmin=-vort_limit, vcenter=0.0, vmax=vort_limit)
        im2 = axes[row_idx, 2].imshow(vort, origin="lower", extent=extent, cmap="PRGn", norm=norm_vort)
        axes[row_idx, 2].set_title(f"Vorticity ($\\omega_z$) | Step {file_idx}")
        fig.colorbar(im2, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)
        
        # Axis labels
        for col_idx in range(3):
            if row_idx == 2:
                axes[row_idx, col_idx].set_xlabel("x")
            axes[row_idx, col_idx].set_ylabel("y")
            
    plt.tight_layout()
    spatial_plot_path = os.path.join(output_dir, "divergence_vorticity_spatial.png")
    plt.savefig(spatial_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 1 to: {spatial_plot_path}")
    
    # --- Figure 2: Time Evolution Plot of RMS Values ---
    print("\nGenerating Figure 2: Time evolution of RMS Divergence and Vorticity...")
    
    times = []
    rms_divs = []
    rms_vorts = []
    
    # Track global limits for high-resolution animations
    max_div_fine = 0.0
    max_vort_fine = 0.0
    
    # Track global limits for coarse-grained 16x8 comparison animations
    max_div_coarse = 0.0
    max_vort_coarse = 0.0
    max_div_diff = 0.0
    max_vort_diff = 0.0
    
    stride = 10
    indices = range(0, n_files, stride)
    
    for idx in tqdm(indices, desc="Processing history"):
        file_path = use_files[idx]
        fd = bin_convert.read_binary(file_path)
        
        # Check time
        times.append(fd["time"])
        
        # Load snapshot and compute
        rho, ux, uy, dx, dy, _, _, _, _ = load_snapshot_data(file_path)
        div, vort = compute_divergence_and_vorticity(ux, uy, dx, dy)
        
        # Compute RMS
        rms_divs.append(np.sqrt(np.mean(div**2)))
        rms_vorts.append(np.sqrt(np.mean(vort**2)))
        
        # Track limits for fine animation (99.5th percentile to ignore noise)
        max_div_fine = max(max_div_fine, np.percentile(np.abs(div), 99.5))
        max_vort_fine = max(max_vort_fine, np.percentile(np.abs(vort), 99.5))
        
        # Coarse-grain (downscale to 16x8 by factor of 64)
        div_A = skimage.measure.block_reduce(div, (64, 64), np.mean)
        vort_A = skimage.measure.block_reduce(vort, (64, 64), np.mean)
        
        ux_cg = skimage.measure.block_reduce(ux, (64, 64), np.mean)
        uy_cg = skimage.measure.block_reduce(uy, (64, 64), np.mean)
        dx_coarse = dx * 64
        dy_coarse = dy * 64
        
        div_B, vort_B = compute_divergence_and_vorticity(ux_cg, uy_cg, dx_coarse, dy_coarse)
        
        # Track limits for coarse comparison animation
        max_div_coarse = max(max_div_coarse, np.percentile(np.abs(div_A), 99.5), np.percentile(np.abs(div_B), 99.5))
        max_vort_coarse = max(max_vort_coarse, np.percentile(np.abs(vort_A), 99.5), np.percentile(np.abs(vort_B), 99.5))
        max_div_diff = max(max_div_diff, np.percentile(np.abs(div_A - div_B), 99.5))
        max_vort_diff = max(max_vort_diff, np.percentile(np.abs(vort_A - vort_B), 99.5))
        
    if max_div_fine == 0: max_div_fine = 1.0
    if max_vort_fine == 0: max_vort_fine = 1.0
    if max_div_coarse == 0: max_div_coarse = 1.0
    if max_vort_coarse == 0: max_vort_coarse = 1.0
    if max_div_diff == 0: max_div_diff = 1.0
    if max_vort_diff == 0: max_vort_diff = 1.0
        
    # Plotting RMS
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = "tab:blue"
    ax1.set_xlabel("Time (Myr)")
    ax1.set_ylabel("RMS Divergence", color=color)
    line1, = ax1.plot(times, rms_divs, color=color, lw=2, label="RMS Divergence")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color = "tab:green"
    ax2.set_ylabel("RMS Vorticity", color=color)
    line2, = ax2.plot(times, rms_vorts, color=color, lw=2, linestyle="--", label="RMS Vorticity")
    ax2.tick_params(axis="y", labelcolor=color)
    
    plt.title("Time Evolution of Compressibility and Turbulence (hr_build)")
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")
    
    plt.tight_layout()
    evolution_plot_path = os.path.join(output_dir, "divergence_vorticity_evolution.png")
    plt.savefig(evolution_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 2 to: {evolution_plot_path}")
    
    # --- Figure 3: Fine Animation of Density, Divergence, and Vorticity ---
    print("\nGenerating Figure 3 (Fine Animation): Density, Divergence, and Vorticity over time...")
    import matplotlib.animation as animation
    
    fig_anim, axes_anim = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    
    first_file = use_files[0]
    rho_0, ux_0, uy_0, dx_0, dy_0, x1min, x1max, x2min, x2max = load_snapshot_data(first_file)
    div_0, vort_0 = compute_divergence_and_vorticity(ux_0, uy_0, dx_0, dy_0)
    extent = [x1min, x1max, x2min, x2max]
    
    norm_rho = LogNorm(vmin=max(1e-5, rho_0.min()), vmax=rho_0.max()) if rho_0.max() / (rho_0.min() + 1e-10) > 10 else None
    im0 = axes_anim[0].imshow(rho_0, origin="lower", extent=extent, cmap="plasma", norm=norm_rho)
    axes_anim[0].set_title("Density")
    fig_anim.colorbar(im0, ax=axes_anim[0], fraction=0.046, pad=0.04)
    
    norm_div = TwoSlopeNorm(vmin=-max_div_fine, vcenter=0.0, vmax=max_div_fine)
    im1 = axes_anim[1].imshow(div_0, origin="lower", extent=extent, cmap="coolwarm", norm=norm_div)
    axes_anim[1].set_title("Divergence ($\\nabla \\cdot \\vec{{u}}$)")
    fig_anim.colorbar(im1, ax=axes_anim[1], fraction=0.046, pad=0.04)
    
    norm_vort = TwoSlopeNorm(vmin=-max_vort_fine, vcenter=0.0, vmax=max_vort_fine)
    im2 = axes_anim[2].imshow(vort_0, origin="lower", extent=extent, cmap="PRGn", norm=norm_vort)
    axes_anim[2].set_title("Vorticity ($\\omega_z$)")
    fig_anim.colorbar(im2, ax=axes_anim[2], fraction=0.046, pad=0.04)
    
    for ax in axes_anim:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
    title_anim = fig_anim.suptitle("Kelvin-Helmholtz Instability | Step 0 | Time = 0.00 Myr", fontsize=14)
    plt.tight_layout()
    
    def update_fine(frame_idx):
        file_idx = indices[frame_idx]
        file_path = use_files[file_idx]
        fd = bin_convert.read_binary(file_path)
        time = fd["time"]
        
        rho_f, ux_f, uy_f, dx_f, dy_f, _, _, _, _ = load_snapshot_data(file_path)
        div_f, vort_f = compute_divergence_and_vorticity(ux_f, uy_f, dx_f, dy_f)
        
        im0.set_data(rho_f)
        im1.set_data(div_f)
        im2.set_data(vort_f)
        
        title_anim.set_text(f"Kelvin-Helmholtz Instability | Step {file_idx} | Time = {time:.2f} Myr")
        return im0, im1, im2, title_anim
        
    anim = animation.FuncAnimation(fig_anim, update_fine, frames=len(indices), blit=False)
    anim_plot_path = os.path.join(output_dir, "divergence_vorticity_animation.mp4")
    save_animation_with_progress(anim, anim_plot_path, len(indices), fps=10)
    plt.close(fig_anim)
    
    # --- Figure 4: Coarse-grained (16x8) Comparison Animation ---
    print("\nGenerating Figure 4 (Coarse Comparison Animation): Case A vs Case B at 16x8...")
    
    fig_comp, axes_comp = plt.subplots(2, 3, figsize=(18, 11), sharex=True, sharey=True)
    
    # Setup initial fields for frame 0
    rho_cg_0 = skimage.measure.block_reduce(rho_0, (64, 64), np.mean)
    div_A_0 = skimage.measure.block_reduce(div_0, (64, 64), np.mean)
    vort_A_0 = skimage.measure.block_reduce(vort_0, (64, 64), np.mean)
    
    ux_cg_0 = skimage.measure.block_reduce(ux_0, (64, 64), np.mean)
    uy_cg_0 = skimage.measure.block_reduce(uy_0, (64, 64), np.mean)
    dx_coarse_0 = dx_0 * 64
    dy_coarse_0 = dy_0 * 64
    div_B_0, vort_B_0 = compute_divergence_and_vorticity(ux_cg_0, uy_cg_0, dx_coarse_0, dy_coarse_0)
    
    # Set norms
    norm_rho_cg = LogNorm(vmin=max(1e-5, rho_cg_0.min()), vmax=rho_cg_0.max()) if rho_cg_0.max() / (rho_cg_0.min() + 1e-10) > 10 else None
    norm_div_cg = TwoSlopeNorm(vmin=-max_div_coarse, vcenter=0.0, vmax=max_div_coarse)
    norm_vort_cg = TwoSlopeNorm(vmin=-max_vort_coarse, vcenter=0.0, vmax=max_vort_coarse)
    norm_div_diff = TwoSlopeNorm(vmin=-max_div_diff, vcenter=0.0, vmax=max_div_diff)
    norm_vort_diff = TwoSlopeNorm(vmin=-max_vort_diff, vcenter=0.0, vmax=max_vort_diff)
    
    # Plot static structure and colorbars
    # Top-Left: Coarse Density
    im_rho = axes_comp[0, 0].imshow(rho_cg_0, origin="lower", extent=extent, cmap="plasma", norm=norm_rho_cg)
    axes_comp[0, 0].set_title("Downscaled Density (16x8)")
    fig_comp.colorbar(im_rho, ax=axes_comp[0, 0], fraction=0.046, pad=0.04)
    
    # Top-Middle: Case A Divergence (Calculate then Downscale)
    im_div_A = axes_comp[0, 1].imshow(div_A_0, origin="lower", extent=extent, cmap="coolwarm", norm=norm_div_cg)
    axes_comp[0, 1].set_title("Div Case A\n(Calculate Fine -> Downscale)")
    fig_comp.colorbar(im_div_A, ax=axes_comp[0, 1], fraction=0.046, pad=0.04)
    
    # Top-Right: Case B Divergence (Downscale then Calculate)
    im_div_B = axes_comp[0, 2].imshow(div_B_0, origin="lower", extent=extent, cmap="coolwarm", norm=norm_div_cg)
    axes_comp[0, 2].set_title("Div Case B\n(Downscale Vel -> Calculate Coarse)")
    fig_comp.colorbar(im_div_B, ax=axes_comp[0, 2], fraction=0.046, pad=0.04)
    
    # Bottom-Left: Subgrid Vorticity (Case A - Case B)
    im_vort_diff = axes_comp[1, 0].imshow(vort_A_0 - vort_B_0, origin="lower", extent=extent, cmap="bwr", norm=norm_vort_diff)
    axes_comp[1, 0].set_title("Subgrid Vorticity\n(Case A - Case B)")
    fig_comp.colorbar(im_vort_diff, ax=axes_comp[1, 0], fraction=0.046, pad=0.04)
    
    # Bottom-Middle: Case A Vorticity
    im_vort_A = axes_comp[1, 1].imshow(vort_A_0, origin="lower", extent=extent, cmap="PRGn", norm=norm_vort_cg)
    axes_comp[1, 1].set_title("Vort Case A\n(Calculate Fine -> Downscale)")
    fig_comp.colorbar(im_vort_A, ax=axes_comp[1, 1], fraction=0.046, pad=0.04)
    
    # Bottom-Right: Case B Vorticity
    im_vort_B = axes_comp[1, 2].imshow(vort_B_0, origin="lower", extent=extent, cmap="PRGn", norm=norm_vort_cg)
    axes_comp[1, 2].set_title("Vort Case B\n(Downscale Vel -> Calculate Coarse)")
    fig_comp.colorbar(im_vort_B, ax=axes_comp[1, 2], fraction=0.046, pad=0.04)
    
    for ax in axes_comp.ravel():
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
    title_comp = fig_comp.suptitle("KHI Downscaled Grid (16x8) Comparison | Step 0 | Time = 0.00 Myr", fontsize=14)
    plt.tight_layout()
    
    def update_coarse(frame_idx):
        file_idx = indices[frame_idx]
        file_path = use_files[file_idx]
        fd = bin_convert.read_binary(file_path)
        time = fd["time"]
        
        rho_f, ux_f, uy_f, dx_f, dy_f, _, _, _, _ = load_snapshot_data(file_path)
        div_f, vort_f = compute_divergence_and_vorticity(ux_f, uy_f, dx_f, dy_f)
        
        # Case A:
        rho_cg_f = skimage.measure.block_reduce(rho_f, (64, 64), np.mean)
        div_A_f = skimage.measure.block_reduce(div_f, (64, 64), np.mean)
        vort_A_f = skimage.measure.block_reduce(vort_f, (64, 64), np.mean)
        
        # Case B:
        ux_cg_f = skimage.measure.block_reduce(ux_f, (64, 64), np.mean)
        uy_cg_f = skimage.measure.block_reduce(uy_f, (64, 64), np.mean)
        dx_coarse_f = dx_f * 64
        dy_coarse_f = dy_f * 64
        div_B_f, vort_B_f = compute_divergence_and_vorticity(ux_cg_f, uy_cg_f, dx_coarse_f, dy_coarse_f)
        
        # Set data
        im_rho.set_data(rho_cg_f)
        im_div_A.set_data(div_A_f)
        im_div_B.set_data(div_B_f)
        im_vort_diff.set_data(vort_A_f - vort_B_f)
        im_vort_A.set_data(vort_A_f)
        im_vort_B.set_data(vort_B_f)
        
        title_comp.set_text(f"KHI Downscaled Grid (16x8) Comparison | Step {file_idx} | Time = {time:.2f} Myr")
        return im_rho, im_div_A, im_div_B, im_vort_diff, im_vort_A, im_vort_B, title_comp
        
    anim_coarse = animation.FuncAnimation(fig_comp, update_coarse, frames=len(indices), blit=False)
    anim_coarse_path = os.path.join(output_dir, "divergence_vorticity_comparison_animation.mp4")
    save_animation_with_progress(anim_coarse, anim_coarse_path, len(indices), fps=10)
    plt.close(fig_comp)
    
    print("\nExplore data script completed successfully!")

if __name__ == "__main__":
    main()
