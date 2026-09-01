#!/usr/bin/env python3
"""
generate_data_animations.py
---------------------------
Generates a single, cohesive 4-panel animation video with a white background:
1. Train: High-Res Temperature (vshear_31_coldfrac_0.33, 1024x2048)
2. Train: Coarse-Grained (32x16) with dynamic 16x8 crop window
3. Test: High-Res Temperature (hr_build_512, 256x512)
4. Test: Coarse-Grained (16x8) for unseen domain testing

Axes show physical coordinates [pc] instead of pixel indices.
Cached in plotting_for_paper/sep_1_2026/assets/data_overview.mp4
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
from tqdm import tqdm

# Configure ffmpeg path
if Path("/usr/bin/ffmpeg").exists():
    plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"

# Path configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import ergane

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DIR = PROJECT_ROOT / "simulation_outputs" / "hr_gpu_sweep_1024x2048_2xlength" / "vshear_31_coldfrac_0.33"
TRAIN_ATHINP = TRAIN_DIR / "kh_radiative_vshear_31_coldfrac_0.33.athinput"

TEST_DIR = PROJECT_ROOT / "simulation_outputs" / "hr_build_512"
TEST_ATHINP = TEST_DIR / "kh_radiative_256x512.athinput"

def coarse_grain_2d(arr: np.ndarray, ds: int) -> np.ndarray:
    ny, nx = arr.shape
    if ds <= 1 or ny < ds or nx < ds:
        return arr.copy()
    ny_cg = ny // ds
    nx_cg = nx // ds
    return arr[:ny_cg * ds, :nx_cg * ds].reshape(ny_cg, ds, nx_cg, ds).mean(axis=(1, 3))


def generate_single_overview_animation(n_samples=100, fps=15, overwrite=False):
    out_mp4 = ASSETS_DIR / "data_overview.mp4"

    if out_mp4.exists() and not overwrite:
        print(f"[CACHE] Animation already exists at {out_mp4}", flush=True)
        return

    print("Loading Training Simulation via Ergane...", flush=True)
    sim_train = ergane.SimulationData(athinp=str(TRAIN_ATHINP), datafolder=str(TRAIN_DIR))

    print("Loading Testing Simulation via Ergane...", flush=True)
    sim_test = ergane.SimulationData(athinp=str(TEST_ATHINP), datafolder=str(TEST_DIR))

    total_frames = min(sim_train.n_frames, sim_test.n_frames)
    sample_indices = np.linspace(0, total_frames - 1, n_samples, dtype=int)

    train_frame_nums = [sim_train.frame_numbers[idx] for idx in sample_indices]
    test_frame_nums = [sim_test.frame_numbers[idx] for idx in sample_indices]

    print(f"Preloading {len(sample_indices)} frames...", flush=True)
    train_hr_list = []
    train_cg_list = []
    test_hr_list = []
    test_cg_list = []

    for fn_tr, fn_te in tqdm(zip(train_frame_nums, test_frame_nums), total=len(sample_indices), desc="Loading frames"):
        f_tr = sim_train.get_frame(fn_tr)
        T_tr = f_tr.temperature
        train_hr_list.append(T_tr)
        train_cg_list.append(coarse_grain_2d(T_tr, ds=64))

        f_te = sim_test.get_frame(fn_te)
        T_te = f_te.temperature
        test_hr_list.append(T_te)
        test_cg_list.append(coarse_grain_2d(T_te, ds=32))

    # Physical extents in pc
    extent_train = [sim_train.x1min, sim_train.x1max, sim_train.x2min, sim_train.x2max]  # [-10, 10, -20, 20]
    extent_test = [sim_test.x1min, sim_test.x1max, sim_test.x2min, sim_test.x2max]      # [-5, 5, -10, 10]

    # Pre-generate random 16x8 box coordinates in physical units for train_cg
    # Train CG grid is 32x16 over [-10, 10] x [-20, 20] (dx = 1.25 pc, dy = 1.25 pc)
    # Box size is 8x16 cells = 10 pc x 20 pc
    dx_tr = (sim_train.x1max - sim_train.x1min) / 16.0
    dy_tr = (sim_train.x2max - sim_train.x2min) / 32.0
    box_w = 8 * dx_tr   # 10 pc
    box_h = 16 * dy_tr  # 20 pc

    rng = np.random.RandomState(42)
    max_ix = 16 - 8
    max_iy = 32 - 16
    box_positions = []
    for _ in range(len(train_cg_list)):
        ix = rng.randint(0, max_ix + 1)
        iy = rng.randint(0, max_iy + 1)
        bx = sim_train.x1min + ix * dx_tr
        by = sim_train.x2min + iy * dy_tr
        box_positions.append((bx, by))

    vmin, vmax = 1e3, 1e7
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = "inferno"

    print("Building 4-panel matplotlib figure with white background...", flush=True)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "text.color": "#222222",
        "font.family": "sans-serif"
    })

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), dpi=140, gridspec_kw={"width_ratios": [1, 1, 1, 1]})

    # Panel 1: Train HR
    im1 = axes[0].imshow(train_hr_list[0], origin="lower", cmap=cmap, norm=norm, extent=extent_train, aspect="auto")
    axes[0].set_title("Train: High-Res", fontsize=11, fontweight="bold", pad=8)
    axes[0].set_xlabel("x [pc]", fontsize=9)
    axes[0].set_ylabel("y [pc]", fontsize=9)
    axes[0].tick_params(labelsize=8)

    # Panel 2: Train Coarse
    im2 = axes[1].imshow(train_cg_list[0], origin="lower", cmap=cmap, norm=norm, extent=extent_train, aspect="auto", interpolation="nearest")
    axes[1].set_title("Train: Coarse (32×16)", fontsize=11, fontweight="bold", pad=8)
    axes[1].set_xlabel("x [pc]", fontsize=9)
    axes[1].set_ylabel("y [pc]", fontsize=9)
    axes[1].tick_params(labelsize=8)

    # Initial crop box
    bx0, by0 = box_positions[0]
    rect = patches.Rectangle((bx0, by0), box_w, box_h, linewidth=2.2, edgecolor="#00ffff", facecolor="none", linestyle="--", label="16×8 Patch")
    axes[1].add_patch(rect)
    axes[1].legend(loc="upper right", fontsize=7.5, framealpha=0.85)

    # Panel 3: Test HR
    im3 = axes[2].imshow(test_hr_list[0], origin="lower", cmap=cmap, norm=norm, extent=extent_test, aspect="auto")
    axes[2].set_title("Test: High-Res", fontsize=11, fontweight="bold", pad=8)
    axes[2].set_xlabel("x [pc]", fontsize=9)
    axes[2].set_ylabel("y [pc]", fontsize=9)
    axes[2].tick_params(labelsize=8)

    # Panel 4: Test Coarse
    im4 = axes[3].imshow(test_cg_list[0], origin="lower", cmap=cmap, norm=norm, extent=extent_test, aspect="auto", interpolation="nearest")
    axes[3].set_title("Test: Coarse (16×8)", fontsize=11, fontweight="bold", pad=8)
    axes[3].set_xlabel("x [pc]", fontsize=9)
    axes[3].set_ylabel("y [pc]", fontsize=9)
    axes[3].tick_params(labelsize=8)

    # Shared colorbar on right
    fig.subplots_adjust(right=0.88, wspace=0.35, bottom=0.15, top=0.88, left=0.06)
    cbar_ax = fig.add_axes([0.90, 0.18, 0.015, 0.68])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label("Temperature T [K]", fontsize=12, labelpad=8)
    cbar.ax.tick_params(labelsize=8)

    print("Rendering animation frames...", flush=True)

    def update(frame_idx):
        im1.set_data(train_hr_list[frame_idx])
        im2.set_data(train_cg_list[frame_idx])
        bx, by = box_positions[frame_idx]
        rect.set_xy((bx, by))
        im3.set_data(test_hr_list[frame_idx])
        im4.set_data(test_cg_list[frame_idx])
        return [im1, im2, rect, im3, im4]

    ani = animation.FuncAnimation(fig, update, frames=len(train_hr_list), interval=1000/fps, blit=False)
    ani.save(str(out_mp4), writer="ffmpeg", fps=fps, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    plt.close(fig)
    print(f"=== Successfully generated and cached: {out_mp4} ===", flush=True)


if __name__ == "__main__":
    generate_single_overview_animation(n_samples=100, fps=15, overwrite=True)
