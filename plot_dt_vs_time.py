#!/usr/bin/env python3
"""
Plot dt vs time for multiple simulation runs.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Simulation output directories
runs = {
    "hr_build (32x16)": "/home/sasi/Projects/SubgridCGMModel/simulation_outputs/hr_build_32x16_from_snap500",
    "hr_build (64x32)": "/home/sasi/Projects/SubgridCGMModel/simulation_outputs/hr_build_64x32_from_snap500",
    "subgrid (32x16)": "/home/sasi/Projects/SubgridCGMModel/simulation_outputs/subgrid_32x16_from_snap500",
    "subgrid (64x32)": "/home/sasi/Projects/SubgridCGMModel/simulation_outputs/subgrid_64x32_from_snap500",
}

fig, ax = plt.subplots(figsize=(10, 6))

for label, dirpath in runs.items():
    hst_file = Path(dirpath) / "KH.hydro.hst"
    if not hst_file.exists():
        print(f"Warning: {hst_file} not found")
        continue

    # Read history file, skipping comments
    data = np.loadtxt(hst_file, comments='#')

    # Extract time (col 1) and dt (col 2)
    time = data[:, 0]
    dt = data[:, 1]

    ax.plot(time, dt, marker='o', markersize=3, label=label, alpha=0.7)

ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("Timestep (dt)", fontsize=12)
ax.set_title("Timestep Evolution Across Runs", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/home/sasi/Projects/SubgridCGMModel/dt_vs_time.png", dpi=150)
print("Plot saved to dt_vs_time.png")
plt.show()
