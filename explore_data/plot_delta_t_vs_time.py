#!/usr/bin/env python3
"""
Standalone script to regenerate the delta_t_vs_time.png plot.
Shows timestep vs physical time from restart onwards.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESTART_TIME_MYR = 5.0
OUTPUT_DIR = Path(PROJECT_ROOT / "tiled_outputs/run_tiled64x32_20260917_171804")

# Paths
HR_HIST = PROJECT_ROOT / "simulation_outputs/hr_gpu_512x1024/KH.hydro.hst"
SG_HIST = PROJECT_ROOT / "simulation_outputs/subgrid_64x32_from_snap500/KH.hydro.hst"
LR_HIST = PROJECT_ROOT / "simulation_outputs/hr_build_64x32_from_snap500/KH.hydro.hst"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_history_file(history_path):
    """Load timestep data from Athena++ history file."""
    try:
        data = np.loadtxt(history_path, comments='#', usecols=(0, 1))
        if data.ndim == 1:
            data = data.reshape(1, -1)
        times = data[:, 0]
        dts = data[:, 1]
        return times, dts
    except Exception as e:
        print(f"  Warning: Could not load history file {history_path}: {e}")
        return None, None


# Load history data
print("Loading history files...")
hr_times_hist, hr_dts = load_history_file(HR_HIST)
sg_times_hist, sg_dts = load_history_file(SG_HIST)
lr_times_hist, lr_dts = load_history_file(LR_HIST)

# For restarted simulations, shift times to physical time
if sg_times_hist is not None:
    sg_times_hist = sg_times_hist + RESTART_TIME_MYR
if lr_times_hist is not None:
    lr_times_hist = lr_times_hist + RESTART_TIME_MYR

# Filter data to show only from RESTART_TIME_MYR onwards
print(f"Filtering data to show times >= {RESTART_TIME_MYR} Myr...")
if hr_times_hist is not None:
    mask_hr = hr_times_hist >= RESTART_TIME_MYR
    hr_times_hist = hr_times_hist[mask_hr]
    hr_dts = hr_dts[mask_hr]
    print(f"  HR: {len(hr_times_hist)} points")

if sg_times_hist is not None:
    mask_sg = sg_times_hist >= RESTART_TIME_MYR
    sg_times_hist = sg_times_hist[mask_sg]
    sg_dts = sg_dts[mask_sg]
    print(f"  SG: {len(sg_times_hist)} points")

if lr_times_hist is not None:
    mask_lr = lr_times_hist >= RESTART_TIME_MYR
    lr_times_hist = lr_times_hist[mask_lr]
    lr_dts = lr_dts[mask_lr]
    print(f"  LR: {len(lr_times_hist)} points")

# Create plot
print("Creating plot...")
fig, ax = plt.subplots(figsize=(11, 6))
ax.axvline(RESTART_TIME_MYR, color="gray", ls="--", lw=1.2, label=f"Restart @ {RESTART_TIME_MYR} Myr", alpha=0.7)

if hr_times_hist is not None and hr_dts is not None:
    ax.plot(hr_times_hist, hr_dts * 1e3, label="HR (512×1024)", lw=2, marker="^", markersize=4, alpha=0.8)

if sg_times_hist is not None and sg_dts is not None:
    ax.plot(sg_times_hist, sg_dts * 1e3, label="SG (0.625 pc)", lw=2, marker="o", markersize=5, alpha=0.8)

if lr_times_hist is not None and lr_dts is not None:
    ax.plot(lr_times_hist, lr_dts * 1e3, label="LR (0.625 pc)", lw=2, marker="s", markersize=5, alpha=0.8)

ax.set_xlabel("Physical Time [Myr]", fontsize=13)
ax.set_ylabel(r"Timestep $\Delta t$ [ms (code units)]", fontsize=13)
ax.set_title(r"Timestep ($\Delta t$) vs Simulation Time (from restart)", fontsize=14, weight="bold")
ax.grid(True, ls="--", alpha=0.5)
ax.legend(fontsize=11, loc="best")
plt.tight_layout()

output_file = OUTPUT_DIR / "delta_t_vs_time.png"
plt.savefig(output_file, dpi=200)
plt.close(fig)
print(f"✓ Saved plot to: {output_file}")
