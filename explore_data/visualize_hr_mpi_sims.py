import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ergane

PROJECT_ROOT = Path("/home/sasi/Projects/SubgridCGMModel")
SIM_ROOT     = PROJECT_ROOT / "simulation_outputs"
OUT_ROOT     = PROJECT_ROOT / "outputs"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ── Resolution list ─────────────────────────────────────────────────────────
RESOLUTIONS = [
    "8x16", "16x8",
    "16x32", "32x16",
    "32x64", "64x32",
    "64x128", "128x64",
    "128x256", "256x128",
    "256x512", "512x256",
    "512x1024", "1024x512",
]

# ── Discover simulation directories (GPU first, fallback to MPI) ─────────────
simulations = []
for res in RESOLUTIONS:
    gpu_dir = SIM_ROOT / f"hr_gpu_{res}"
    mpi_dir = SIM_ROOT / f"hr_mpi_{res}"
    build_dir = SIM_ROOT / f"hr_build_{res}"

    datafolder = None
    name = None
    if gpu_dir.is_dir():
        datafolder = gpu_dir
        name = f"hr_gpu_{res}"
    elif mpi_dir.is_dir():
        datafolder = mpi_dir
        name = f"hr_mpi_{res}"
    elif build_dir.is_dir():
        datafolder = build_dir
        name = f"hr_build_{res}"

    if datafolder is not None:
        athinp = datafolder / f"kh_radiative_{res}.athinput"
        if not athinp.is_file():
            athinp_candidates = list(datafolder.glob("*.athinput"))
            if athinp_candidates:
                athinp = athinp_candidates[0]

        if athinp.is_file():
            # Save the visualization to the same folder as the run
            save_path = datafolder / f"{name}.mp4"
            simulations.append({
                "name":       name,
                "athinp":     str(athinp),
                "datafolder": str(datafolder),
                "save_path":  str(save_path),
            })

# ── Render and save each simulation ─────────────────────────────────────────
if not simulations:
    print(f"No simulation output directories found under {SIM_ROOT}.")
else:
    for sim in simulations:
        print(f"\n[{sim['name']}] Loading simulation data from {sim['datafolder']} ...")
        try:
            sim_data = ergane.SimulationData(
                athinp=sim['athinp'],
                datafolder=sim['datafolder'],
            )

            if sim_data.n_frames == 0:
                print(f"  [{sim['name']}] No frames found in {sim['datafolder']}, skipping.")
                continue

            print(f"  [{sim['name']}] Rendering animation ({sim_data.n_frames} frames available) ...")
            mpl_viz = sim_data.visualize(
                backend="matplotlib",
                interval=80,
            )

            print(f"  [{sim['name']}] Saving animation to {sim['save_path']} ...")
            mpl_viz.ani.save(
                sim['save_path'],
                writer="ffmpeg",
                fps=60,
                dpi=150,
            )
            print(f"  [{sim['name']}] Saved animation to: {sim['save_path']}")
        except Exception as e:
            print(f"  ERROR generating visualization for {sim['name']}: {e}")

print("\nAll visualizations processed.")
