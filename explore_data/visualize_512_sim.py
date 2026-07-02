import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ergane

hr_sim_data = ergane.SimulationData(
    athinp='/home/sasi/Projects/SubgridCGMModel/builds/hr_build/src/kh_radiative_128.athinput',
    datafolder='/home/sasi/Projects/SubgridCGMModel/simulation_outputs/hr_build_512'
)

# ── Save a matplotlib animation to disk ──────────────────────────────────────
# Uncomment the block below to run.
PROJECT_ROOT = '/home/sasi/Projects/SubgridCGMModel'
SAVE_PATH = PROJECT_ROOT + "/outputs/hr_build_video_1024.mp4"


mpl_viz = hr_sim_data.visualize(
    fields=["density", "pressure"],
    backend="matplotlib",
    interval=80,
)
mpl_viz.ani.save(
    str(SAVE_PATH),
    writer="ffmpeg",
    fps=15,
    dpi=150,
)
print(f"Saved animation to: {SAVE_PATH}")
