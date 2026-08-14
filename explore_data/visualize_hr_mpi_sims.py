import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ergane

PROJECT_ROOT = '/home/sasi/Projects/SubgridCGMModel'
SIM_ROOT     = os.path.join(PROJECT_ROOT, 'simulation_outputs')
OUT_ROOT     = os.path.join(PROJECT_ROOT, 'outputs')
os.makedirs(OUT_ROOT, exist_ok=True)

# ── Simulation configurations ────────────────────────────────────────────────
simulations = [
    {
        'name':       'hr_mpi_128x256',
        'athinp':     os.path.join(SIM_ROOT, 'hr_mpi_128x256', 'kh_radiative_128x256.athinput'),
        'datafolder': os.path.join(SIM_ROOT, 'hr_mpi_128x256'),
        'save_path':  os.path.join(OUT_ROOT, 'hr_mpi_128x256.mp4'),
    },
    {
        'name':       'hr_mpi_256x512',
        'athinp':     os.path.join(SIM_ROOT, 'hr_mpi_256x512', 'kh_radiative_256x512.athinput'),
        'datafolder': os.path.join(SIM_ROOT, 'hr_mpi_256x512'),
        'save_path':  os.path.join(OUT_ROOT, 'hr_mpi_256x512.mp4'),
    },
    {
        'name':       'hr_mpi_512x1024',
        'athinp':     os.path.join(SIM_ROOT, 'hr_mpi_512x1024', 'kh_radiative_512x1024.athinput'),
        'datafolder': os.path.join(SIM_ROOT, 'hr_mpi_512x1024'),
        'save_path':  os.path.join(OUT_ROOT, 'hr_mpi_512x1024.mp4'),
    },
    {
        'name':       'hr_mpi_1024x2048',
        'athinp':     os.path.join(SIM_ROOT, 'hr_mpi_1024x2048', 'kh_radiative_1024x2048.athinput'),
        'datafolder': os.path.join(SIM_ROOT, 'hr_mpi_1024x2048'),
        'save_path':  os.path.join(OUT_ROOT, 'hr_mpi_1024x2048.mp4'),
    },
]

# ── Render and save each simulation ─────────────────────────────────────────
for sim in simulations:
    print(f"\n[{sim['name']}] Loading simulation data ...")
    sim_data = ergane.SimulationData(
        athinp=sim['athinp'],
        datafolder=sim['datafolder'],
    )

    print(f"[{sim['name']}] Rendering animation ...")
    mpl_viz = sim_data.visualize(
        backend="matplotlib",
        interval=80,
    )

    print(f"[{sim['name']}] Saving to {sim['save_path']} ...")
    mpl_viz.ani.save(
        sim['save_path'],
        writer="ffmpeg",
        fps=60,
        dpi=150,
    )
    print(f"[{sim['name']}] Saved animation to: {sim['save_path']}")

print("\nAll done.")
