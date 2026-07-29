import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ergane

hr_sim_data = ergane.SimulationData(
    athinp='/home/sasi/Projects/SubgridCGMModel/builds/subgrid_vertical/sg_vertical.athinput',
    datafolder='/home/sasi/Projects/SubgridCGMModel/simulation_outputs/subgrid_vertical_run_ismcooling_hr'
)

# ── Save a matplotlib animation to disk ──────────────────────────────────────
# Uncomment the block below to run.
PROJECT_ROOT = '/home/sasi/Projects/SubgridCGMModel'
SAVE_PATH = PROJECT_ROOT + "/outputs/subgrid_vertical_ismcooling_hr.mp4"


mpl_viz = hr_sim_data.visualize(
    backend="matplotlib",
    interval=80,
)
mpl_viz.ani.save(
    str(SAVE_PATH),
    writer="ffmpeg",
    fps=60,
    dpi=150,
)
print(f"Saved animation to: {SAVE_PATH}")
