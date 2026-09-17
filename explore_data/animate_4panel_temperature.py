#!/usr/bin/env python3
"""
animate_4panel_temperature.py
─────────────────────────────────────────────────────────────────────────────
Generates a 4-panel animated video (MP4) comparing the two simulations in:
  /home/sasi/Projects/SubgridCGMModel/simulation_outputs/hr_gpu_sweep_1024x2048_2xlength

Panels:
  Top Row:
    - Top-left:  Sim 1 (vshear_31_coldfrac_0.33) full-res Temperature field with velocity streamlines
    - Top-right: Sim 2 (vshear_31_coldfrac_0.67) full-res Temperature field with velocity streamlines
  Bottom Row:
    - Bottom-left:  Sim 1 coarse-grained Temperature to (32, 16) with highlighted 16x8 patch
    - Bottom-right: Sim 2 coarse-grained Temperature to (32, 16) with highlighted 16x8 patch
"""

import os
import sys
import shutil
import tempfile
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

PROJECT_ROOT = Path("/home/sasi/Projects/SubgridCGMModel")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "ergane"))
sys.path.insert(0, str(PROJECT_ROOT / "ergane"))

import ergane
SWEEP_DIR    = PROJECT_ROOT / "simulation_outputs" / "hr_gpu_sweep_1024x2048_2xlength"
OUT_DIR      = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Grid domain settings
EXTENT = [-10.0, 10.0, -20.0, 20.0]  # [x_min, x_max, y_min, y_max] in pc
VMIN, VMAX = 3.0, 7.0                # log10(T [K]) color scale

def coarse_grain_2d(arr: np.ndarray, target_shape: tuple[int, int] = (32, 16)) -> np.ndarray:
    """Coarse-grain 2D array of shape (Ny, Nx) down to target_shape (ty, tx)."""
    ny, nx = arr.shape
    ty, tx = target_shape
    sy, sx = ny // ty, nx // tx
    return arr[:ty * sy, :tx * sx].reshape(ty, sy, tx, sx).mean(axis=(1, 3))


def stitch_frames_with_ffmpeg(temp_dir: str, output_path: str, fps: int = 15):
    """Stitch PNG frames into MP4 or GIF with ffmpeg."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    is_gif = output_path.lower().endswith(".gif")

    if is_gif:
        # High quality GIF conversion using two-pass palettegen/paletteuse
        filter_complex = f"fps={fps},scale=1280:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer"
        cmd_gif = [
            "ffmpeg", "-y",
            "-pattern_type", "glob",
            "-i", os.path.join(temp_dir, "frame_*.png"),
            "-vf", filter_complex,
            output_path,
        ]
        res = subprocess.run(cmd_gif, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg gif creation failed: {res.stderr.decode()[:300]}")
        return

    vf_filter = "scale='min(4096,iw)':'min(4096,ih)':force_original_aspect_ratio=decrease,pad=ceil(iw/2)*2:ceil(ih/2)*2"

    cmd_nvenc = [
        "ffmpeg", "-y",
        "-r", str(fps),
        "-pattern_type", "glob",
        "-i", os.path.join(temp_dir, "frame_*.png"),
        "-vf", vf_filter,
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    res = subprocess.run(cmd_nvenc, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if res.returncode == 0:
        return

    cmd_x264 = [
        "ffmpeg", "-y",
        "-r", str(fps),
        "-pattern_type", "glob",
        "-i", os.path.join(temp_dir, "frame_*.png"),
        "-vf", vf_filter,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    res2 = subprocess.run(cmd_x264, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if res2.returncode == 0:
        return

    cmd_mpeg4 = [
        "ffmpeg", "-y",
        "-r", str(fps),
        "-pattern_type", "glob",
        "-i", os.path.join(temp_dir, "frame_*.png"),
        "-vf", vf_filter,
        "-c:v", "mpeg4",
        "-q:v", "2",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    res3 = subprocess.run(cmd_mpeg4, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if res3.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {res3.stderr.decode()[:300]}")


def render_single_frame(
    frame_idx: int,
    temp_dir: str,
    sim1_dir: str,
    sim2_dir: str,
    athinp1: str,
    athinp2: str,
    patch_bounds: tuple[float, float, float, float],
    patch_indices: tuple[int, int],
):
    """Worker task to render a single multi-panel frame."""
    s1 = ergane.SimulationData(athinp=athinp1, datafolder=sim1_dir)
    s2 = ergane.SimulationData(athinp=athinp2, datafolder=sim2_dir)

    f1 = s1.get_frame(frame_idx)
    f2 = s2.get_frame(frame_idx)

    # Simulation code time unit is exactly 1 Myr (tlim=10.0 Myr, dt=0.01 Myr)
    time_myr = float(s1.times[frame_idx]) if hasattr(s1, "times") else (frame_idx * 0.01)

    T1 = f1.temperature
    T2 = f2.temperature
    vx1 = f1.velx
    vy1 = f1.vely
    vx2 = f2.velx
    vy2 = f2.vely

    T1_cg = coarse_grain_2d(T1, (32, 16))
    T2_cg = coarse_grain_2d(T2, (32, 16))

    # Streamline grid subsampling
    x_s = np.linspace(EXTENT[0], EXTENT[1], 40)
    y_s = np.linspace(EXTENT[2], EXTENT[3], 80)
    step_x = 1024 // 40
    step_y = 2048 // 80
    vx1_s = vx1[::step_y, ::step_x][:80, :40]
    vy1_s = vy1[::step_y, ::step_x][:80, :40]
    vx2_s = vx2[::step_y, ::step_x][:80, :40]
    vy2_s = vy2[::step_y, ::step_x][:80, :40]

    x_min_p, y_min_p, w_p, h_p = patch_bounds
    r_start, c_start = patch_indices

    fig, axes = plt.subplots(1, 4, figsize=(11, 7), sharex=True, sharey=True)

    # 1. Sim 1 Full Res
    ax0 = axes[0]
    im = ax0.imshow(np.log10(T1), origin="lower", extent=EXTENT, vmin=VMIN, vmax=VMAX, cmap="inferno", aspect="auto")
    ax0.streamplot(x_s, y_s, vx1_s, vy1_s, color="white", density=0.7, linewidth=0.7, arrowsize=0.7)
    ax0.set_title(r"$\mathbf{v_{shear}=31,\ f_{cold}=0.33}$" + "\nFull Res (1024 × 2048)", fontsize=11)
    ax0.set_xlabel(r"$x \ [\mathrm{pc}]$", fontsize=11)
    ax0.set_ylabel(r"$y \ [\mathrm{pc}]$", fontsize=11)

    # 2. Sim 1 Coarse Grained (32x16)
    ax1 = axes[1]
    ax1.imshow(np.log10(T1_cg), origin="lower", extent=EXTENT, vmin=VMIN, vmax=VMAX, cmap="inferno", aspect="auto")
    rect1 = Rectangle((x_min_p, y_min_p), w_p, h_p, edgecolor="cyan", facecolor="none", lw=2.5, ls="--")
    ax1.add_patch(rect1)
    ax1.set_title(r"$\mathbf{v_{shear}=31,\ f_{cold}=0.33}$" + f"\nCoarse (32 × 16) | Patch [{r_start}:{r_start+16}, {c_start}:{c_start+8}]", fontsize=11)
    ax1.set_xlabel(r"$x \ [\mathrm{pc}]$", fontsize=11)

    # 3. Sim 2 Full Res
    ax2 = axes[2]
    ax2.imshow(np.log10(T2), origin="lower", extent=EXTENT, vmin=VMIN, vmax=VMAX, cmap="inferno", aspect="auto")
    ax2.streamplot(x_s, y_s, vx2_s, vy2_s, color="white", density=0.7, linewidth=0.7, arrowsize=0.7)
    ax2.set_title(r"$\mathbf{v_{shear}=31,\ f_{cold}=0.67}$" + "\nFull Res (1024 × 2048)", fontsize=11)
    ax2.set_xlabel(r"$x \ [\mathrm{pc}]$", fontsize=11)

    # 4. Sim 2 Coarse Grained (32x16)
    ax3 = axes[3]
    ax3.imshow(np.log10(T2_cg), origin="lower", extent=EXTENT, vmin=VMIN, vmax=VMAX, cmap="inferno", aspect="auto")
    rect2 = Rectangle((x_min_p, y_min_p), w_p, h_p, edgecolor="cyan", facecolor="none", lw=2.5, ls="--")
    ax3.add_patch(rect2)
    ax3.set_title(r"$\mathbf{v_{shear}=31,\ f_{cold}=0.67}$" + f"\nCoarse (32 × 16) | Patch [{r_start}:{r_start+16}, {c_start}:{c_start+8}]", fontsize=11)
    ax3.set_xlabel(r"$x \ [\mathrm{pc}]$", fontsize=11)

    for a in axes:
        a.set_xlim(EXTENT[0], EXTENT[1])
        a.set_ylim(EXTENT[2], EXTENT[3])

    fig.subplots_adjust(left=0.05, right=0.92, bottom=0.12, top=0.87, wspace=0.1)
    cbar_ax = fig.add_axes([0.935, 0.15, 0.015, 0.69])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r"$\log_{10}(T \ [\mathrm{K}])$", fontsize=12)

    fig.suptitle(f"Turbulent Radiative Mixing Layer — Snapshot #{frame_idx:04d} (t = {time_myr:.3f} Myr)", fontsize=14, weight="bold")

    out_file = os.path.join(temp_dir, f"frame_{frame_idx:05d}.png")
    plt.savefig(out_file, dpi=120)
    plt.close(fig)


def _worker_wrapper(args):
    return render_single_frame(*args)


def main():
    parser = argparse.ArgumentParser(description="Generate 4-panel animated comparison plot.")
    parser.add_argument("--step", type=int, default=2, help="Snapshot sampling step (default: 2)")
    parser.add_argument("--start", type=int, default=0, help="Starting frame number (default: 0)")
    parser.add_argument("--end", type=int, default=1000, help="Ending frame number (default: 1000)")
    parser.add_argument("--fps", type=int, default=15, help="Animation FPS (default: 15)")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers (default: 16)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for patch selection (default: 42)")
    parser.add_argument("--output", type=str, default=str(OUT_DIR / "temperature_4panel_comparison.mp4"), help="Output MP4 path")
    args = parser.parse_args()

    sim1_dir = SWEEP_DIR / "vshear_31_coldfrac_0.33"
    sim2_dir = SWEEP_DIR / "vshear_31_coldfrac_0.67"
    athinp1 = list(sim1_dir.glob("*.athinput"))[0]
    athinp2 = list(sim2_dir.glob("*.athinput"))[0]

    dx_cg = (EXTENT[1] - EXTENT[0]) / 16.0  # 20 / 16 = 1.25 pc
    dy_cg = (EXTENT[3] - EXTENT[2]) / 32.0  # 40 / 32 = 1.25 pc
    w_p = 8 * dx_cg
    h_p = 16 * dy_cg

    # Seed RNG to generate a unique random 16x8 patch for each timestep
    rng = np.random.default_rng(args.seed)
    frame_indices = list(range(args.start, args.end + 1, args.step))

    # Pre-generate per-frame patch coordinates
    frame_patches = []
    for _ in frame_indices:
        r_start = int(rng.integers(0, 32 - 16 + 1))  # 0 to 16 inclusive (ny_cg=32)
        c_start = int(rng.integers(0, 16 - 8 + 1))   # 0 to 8 inclusive (nx_cg=16)
        x_min_p = EXTENT[0] + c_start * dx_cg
        y_min_p = EXTENT[2] + r_start * dy_cg
        frame_patches.append(((x_min_p, y_min_p, w_p, h_p), (r_start, c_start)))

    print(f"Random 16x8 patches will be drawn per timestep ({len(frame_indices)} frames, seed={args.seed}).")
    print(f"Rendering {len(frame_indices)} frames (range {args.start} to {args.end}, step={args.step}) using {args.workers} workers...")

    temp_dir = tempfile.mkdtemp(prefix="anim_4panel_")
    try:
        tasks = [
            (
                f_idx,
                temp_dir,
                str(sim1_dir),
                str(sim2_dir),
                str(athinp1),
                str(athinp2),
                patch_bounds,
                patch_idx,
            )
            for f_idx, (patch_bounds, patch_idx) in zip(frame_indices, frame_patches)
        ]

        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                list(tqdm(executor.map(_worker_wrapper, tasks), total=len(tasks), desc="Rendering frames"))
        else:
            for task in tqdm(tasks, desc="Rendering frames"):
                _worker_wrapper(task)

        print(f"Stitching frames into {args.output} at {args.fps} FPS...")
        stitch_frames_with_ffmpeg(temp_dir, args.output, fps=args.fps)
        print(f"Successfully generated animation: {args.output}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
