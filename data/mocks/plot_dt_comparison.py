#!/usr/bin/env python3
"""
Plot dt vs time for LR and Subgrid simulations.

Reads log files from both simulations and creates a comparison plot.
Optionally saves as PNG and/or displays interactively.
"""

import os
import re
import sys
from pathlib import Path

import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    print("WARNING: matplotlib not available. Skipping plot generation.", file=sys.stderr)
    sys.exit(0)


def parse_log_file(log_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse Athena log file and extract time and dt arrays.

    Expected format:
        elapsed=... cycle=... time=... dt=...

    Returns:
        (times, dts) - numpy arrays of time and dt values
    """
    times = []
    dts = []

    if not os.path.exists(log_path):
        print(f"WARNING: Log file not found: {log_path}", file=sys.stderr)
        return np.array([]), np.array([])

    with open(log_path, "r") as f:
        for line in f:
            # Parse lines with time and dt
            time_match = re.search(r"time=(\d+\.?\d*[eE][+-]?\d+)", line)
            dt_match = re.search(r"dt=(\d+\.?\d*[eE][+-]?\d+)", line)

            if time_match and dt_match:
                times.append(float(time_match.group(1)))
                dts.append(float(dt_match.group(1)))

    return np.array(times), np.array(dts)


def plot_dt_comparison(
    lr_log: str,
    sg_log: str,
    output_dir: str,
    title: str = "dt vs Time Comparison: LR (ISM) vs Subgrid (CNN)",
) -> str:
    """
    Create and save a comparison plot of dt vs time.

    Args:
        lr_log: Path to LR simulation log file
        sg_log: Path to Subgrid simulation log file
        output_dir: Directory to save the plot
        title: Title for the plot

    Returns:
        Path to saved plot file
    """
    # Parse log files
    lr_times, lr_dts = parse_log_file(lr_log)
    sg_times, sg_dts = parse_log_file(sg_log)

    if len(lr_times) == 0 and len(sg_times) == 0:
        print("ERROR: No data found in log files", file=sys.stderr)
        return ""

    if len(lr_times) == 0:
        print("WARNING: No LR data found", file=sys.stderr)
    if len(sg_times) == 0:
        print("WARNING: No Subgrid data found", file=sys.stderr)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Convert times from simulation time units to Myr for readability
    if len(lr_times) > 0:
        lr_times_myr = lr_times * 1e3  # Assuming time is in units where 1e-3 = 1 Myr
        ax = axes[0]
        ax.plot(lr_times_myr, lr_dts * 1e6, "b.-", linewidth=2, markersize=4, label="LR (ISM)")
        ax.set_xlabel("Time (Myr)", fontsize=12)
        ax.set_ylabel("dt (μs)", fontsize=12)
        ax.set_title("LR Simulation: dt vs Time", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

    if len(sg_times) > 0:
        sg_times_myr = sg_times * 1e3
        ax = axes[1]
        ax.plot(sg_times_myr, sg_dts * 1e6, "r.-", linewidth=2, markersize=4, label="Subgrid (CNN)")
        ax.set_xlabel("Time (Myr)", fontsize=12)
        ax.set_ylabel("dt (μs)", fontsize=12)
        ax.set_title("Subgrid Simulation: dt vs Time", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dt_vs_time_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {output_path}")

    # Also create an overlay comparison if both datasets exist
    if len(lr_times) > 0 and len(sg_times) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(lr_times_myr, lr_dts * 1e6, "b.-", linewidth=2, markersize=4, label="LR (ISM)", alpha=0.7)
        ax.plot(sg_times_myr, sg_dts * 1e6, "r.-", linewidth=2, markersize=4, label="Subgrid (CNN)", alpha=0.7)

        ax.set_xlabel("Time (Myr)", fontsize=12)
        ax.set_ylabel("dt (simulation time units)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)

        plt.tight_layout()
        overlay_path = os.path.join(output_dir, "dt_vs_time_overlay.png")
        plt.savefig(overlay_path, dpi=150, bbox_inches="tight")
        print(f"Saved overlay plot: {overlay_path}")
        plt.close()

    plt.close()
    return output_path


def main():
    """Main entry point."""
    # Get paths from environment variables or arguments
    project_root = os.getenv("PROJECT_ROOT", "/home/sasi/Projects/SubgridCGMModel")
    lr_output_dir = os.getenv("LR_OUTPUT_DIR", f"{project_root}/simulation_outputs/lr_build_ism")
    sg_output_dir = os.getenv("SG_OUTPUT_DIR", f"{project_root}/simulation_outputs/subgrid_model")
    output_dir = os.getenv("SG_MOCKS_DIR", f"{project_root}/outputs/sg_mocks")

    # Find log files in output directories
    lr_log = os.path.join(lr_output_dir, "KH.log")
    sg_log = os.path.join(sg_output_dir, "KH.log")

    # If standard log names don't exist, try to find any .log file
    if not os.path.exists(lr_log):
        for fname in os.listdir(lr_output_dir) if os.path.isdir(lr_output_dir) else []:
            if fname.endswith(".log"):
                lr_log = os.path.join(lr_output_dir, fname)
                break

    if not os.path.exists(sg_log):
        for fname in os.listdir(sg_output_dir) if os.path.isdir(sg_output_dir) else []:
            if fname.endswith(".log"):
                sg_log = os.path.join(sg_output_dir, fname)
                break

    print(f"LR log file:       {lr_log}")
    print(f"Subgrid log file:  {sg_log}")
    print(f"Output directory:  {output_dir}")
    print()

    plot_path = plot_dt_comparison(lr_log, sg_log, output_dir)

    if plot_path:
        print(f"\nPlot comparison complete: {plot_path}")
    else:
        print("WARNING: Plot generation failed or skipped")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
