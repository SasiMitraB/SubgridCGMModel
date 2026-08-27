#!/usr/bin/env python3
"""
interface_histogram.py

"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Ensure project root and ergane are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ergane import SimulationData


def compute_hist(
    frame,
    ymin: float = 0.0,
    ymax: float = 5.0,
    bins: np.ndarray = None,
    mode: str = "pdf_log",
):
    """
    Extract temperature in code y in [ymin, ymax] and compute normalized distribution.

    mode:
      - 'pdf_log' : dP / d(log10 T), normalized such that integral over log10(T) is 1.
      - 'fraction': Fraction of cells / volume in each bin.
      - 'density' : Standard dP / dT.
    """
    if frame is None or frame.temperature is None:
        return np.zeros(len(bins) - 1)

    # Frame y-coordinates in code units
    scale_l = getattr(frame.units, "length", 1.0) if frame.units else 1.0
    yc_code = frame.yc / scale_l if scale_l != 0 else frame.yc

    mask = (yc_code >= ymin) & (yc_code <= ymax)

    if not np.any(mask):
        return np.zeros(len(bins) - 1)

    temp_subset = frame.temperature[mask, :].ravel()
    temp_valid = temp_subset[np.isfinite(temp_subset) & (temp_subset > 0)]

    if len(temp_valid) == 0:
        return np.zeros(len(bins) - 1)

    counts, _ = np.histogram(temp_valid, bins=bins)
    total = np.sum(counts)
    if total == 0:
        return np.zeros(len(bins) - 1)

    if mode == "pdf_log":
        dlogT = np.diff(np.log10(bins))
        return (counts / total) / dlogT
    elif mode == "fraction":
        return counts / total
    elif mode == "density":
        dT = np.diff(bins)
        return (counts / total) / dT
    else:
        return counts


def create_histogram_animation(
    datafolder_high: str | Path,
    datafolder_low: str | Path,
    athinp: str | Path | None = None,
    ymin: float = 0.0,
    ymax: float = 5.0,
    n_bins: int = 50,
    t_min_temp: float = 1e4,
    t_max_temp: float = 1.6e6,
    mode: str = "pdf_log",
    fps: int = 15,
    save_path: str | Path | None = None,
    dpi: int = 150,
):
    datafolder_high = Path(datafolder_high)
    datafolder_low = Path(datafolder_low)
    athinp = Path(athinp) if athinp else None

    print(f"[INFO] Loading high-resolution simulation (512x256) from: {datafolder_high}")
    sim_high = SimulationData(datafolder=datafolder_high, athinp=athinp)

    print(f"[INFO] Loading low-resolution simulation (16x8) from: {datafolder_low}")
    sim_low = SimulationData(datafolder=datafolder_low, athinp=athinp)

    # Determine common frames
    common_frames = sorted(set(sim_high.frame_numbers).intersection(set(sim_low.frame_numbers)))

    if not common_frames:
        common_frames = sim_high.frame_numbers

    if not common_frames:
        raise ValueError("No simulation output frames found in specified directories.")

    print(f"[INFO] Found {len(common_frames)} frames for animation.")

    # Temperature bins in log-space covering 10^4 K to 1.6 x 10^6 K
    bins = np.logspace(np.log10(t_min_temp), np.log10(t_max_temp), n_bins + 1)

    # Set up figure
    fig, ax = plt.subplots(figsize=(9, 6), dpi=dpi)

    frame0_high = sim_high.get_frame(common_frames[0])
    frame0_low = (
        sim_low.frame_at(frame0_high.time)
        if common_frames[0] not in sim_low.frame_numbers
        else sim_low.get_frame(common_frames[0])
    )

    hist0_high = compute_hist(frame0_high, ymin=ymin, ymax=ymax, bins=bins, mode=mode)
    hist0_low = compute_hist(frame0_low, ymin=ymin, ymax=ymax, bins=bins, mode=mode)

    (line_high,) = ax.step(
        bins[:-1],
        hist0_high,
        where="post",
        color="#2563EB",
        lw=2.2,
        label=f"512×256 ({sim_high.nx}×{sim_high.ny})",
    )
    (line_low,) = ax.step(
        bins[:-1],
        hist0_low,
        where="post",
        color="#DC2626",
        lw=2.0,
        linestyle="--",
        label=f"16×8 ({sim_low.nx}×{sim_low.ny})",
    )

    ax.set_xscale("log")
    ax.set_xlim(t_min_temp, t_max_temp)
    ax.set_yscale('log')

    # Vertical reference lines for hot & cold phase temperatures
    ax.axvline(1.5e4, color="#3B82F6", linestyle=":", alpha=0.6, label="Cold Phase (10⁴ K)")
    ax.axvline(1.5e6, color="#EF4444", linestyle=":", alpha=0.6, label="Hot Phase (10⁶ K)")

    ylabel_map = {
        "pdf_log": r"$\mathrm{d}P / \mathrm{d}(\log_{10} T)$",
        "fraction": "Fraction of Cells",
        "density": r"$\mathrm{d}P / \mathrm{d}T\quad [\mathrm{K}^{-1}]$",
    }
    ax.set_xlabel("Temperature [K]", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel_map.get(mode, "Probability Density"), fontsize=12, fontweight="bold")

    scale_t = getattr(frame0_high.units, "time", 1.0) if frame0_high.units else 1.0
    t_code0 = frame0_high.time / scale_t if scale_t != 0 else frame0_high.time

    title_text = ax.set_title(
        f"Temperature Distribution in $y \\in [{ymin:.1f}, {ymax:.1f}]$\n"
        f"Frame #{common_frames[0]:05d} | Code Time $t = {t_code0:.3f}$",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )

    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(loc="upper center", frameon=True, fontsize=10, ncol=2)

    # Initial y-limit calculation
    max_val = max(np.max(hist0_high), np.max(hist0_low), 1.0)
    ax.set_ylim(0, max_val * 1.25)
    plt.tight_layout()

    def update(frame_num):
        nonlocal max_val
        fh = sim_high.get_frame(frame_num)
        fl = (
            sim_low.frame_at(fh.time)
            if frame_num not in sim_low.frame_numbers
            else sim_low.get_frame(frame_num)
        )

        hh = compute_hist(fh, ymin=ymin, ymax=ymax, bins=bins, mode=mode)
        hl = compute_hist(fl, ymin=ymin, ymax=ymax, bins=bins, mode=mode)

        line_high.set_ydata(hh)
        line_low.set_ydata(hl)

        cur_max = max(np.max(hh), np.max(hl), 1.0)
        if cur_max > max_val or cur_max < max_val * 0.4:
            max_val = cur_max
            ax.set_ylim(0, max_val * 1.25)

        t_code = fh.time / scale_t if scale_t != 0 else fh.time
        title_text.set_text(
            f"Temperature Distribution in $y \\in [{ymin:.1f}, {ymax:.1f}]$\n"
            f"Frame #{frame_num:05d} | Code Time $t = {t_code:.3f}$"
        )
        return line_high, line_low, title_text

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=common_frames,
        interval=1000 // fps,
        blit=False,
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Saving animation to: {save_path} ...")

        if save_path.suffix.lower() == ".gif":
            anim.save(save_path, writer="pillow", fps=fps, dpi=dpi)
        else:
            anim.save(save_path, writer="ffmpeg", fps=fps, dpi=dpi)

        print(f"[OK] Animation saved to: {save_path}")
        plt.close(fig)
    else:
        print("[INFO] Displaying interactive animation window...")
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Animate temperature histogram comparison (y in [0, 5]) between 512x256 and 16x8 simulations."
    )
    parser.add_argument(
        "--datafolder-high",
        type=str,
        default="kh_radiative_cooling/outputs_512x256",
        help="Path to 512x256 simulation outputs",
    )
    parser.add_argument(
        "--datafolder-low",
        type=str,
        default="kh_radiative_cooling/outputs_16x8",
        help="Path to 16x8 simulation outputs",
    )
    parser.add_argument(
        "--athinp",
        type=str,
        default="kh_radiative_cooling/kh_cooling.athinput",
        help="Path to athinput file",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=0.0,
        help="Lower y-boundary for sampling temperature in code units (default: 0.0)",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=5.0,
        help="Upper y-boundary for sampling temperature in code units (default: 5.0)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of log-spaced temperature bins",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pdf_log", "fraction", "density"],
        default="pdf_log",
        help="Distribution mode: 'pdf_log' (dP/dlog10 T), 'fraction' (cell fraction), 'density' (dP/dT)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Frames per second for animation",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save animation (.gif or .mp4)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for animation rendering",
    )

    args = parser.parse_args()

    create_histogram_animation(
        datafolder_high=args.datafolder_high,
        datafolder_low=args.datafolder_low,
        athinp=args.athinp,
        ymin=args.ymin,
        ymax=args.ymax,
        n_bins=args.bins,
        mode=args.mode,
        fps=args.fps,
        save_path=args.save,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
