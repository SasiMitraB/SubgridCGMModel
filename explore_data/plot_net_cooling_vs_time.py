#!/usr/bin/env python3
"""
Backfill net_cooling_vs_time.png / .csv for runs in tiled_outputs/ without
re-running the full mock_sg_tiled.py suite (animations, profiles, ...).

For each run, the manifest.txt written by run_variable_box_subgrid.sh gives the
HR / SG / LR data locations and CNN configuration. Net cooling per unit
mixing-layer area,
    Sigma_c(t) = (1/Lx) * integral n^2 Lambda(T) dx dy   [erg cm^-2 s^-1],
is computed at every snapshot for HR (full resolution), SG (tiled CNN PDF,
isobaric) and LR (n^2 Lambda at the coarse cell), exactly as in
mock_sg_tiled.py. A summary figure overlaying all runs is written to
tiled_outputs/net_cooling_vs_time_all_runs.png.

mock_sg_tiled.py reads its configuration (DS, tiling, model) from environment
variables at import time, so each run is processed in its own subprocess.

Usage:
    python explore_data/plot_net_cooling_vs_time.py                  # all runs + summary
    python explore_data/plot_net_cooling_vs_time.py --run-dir tiled_outputs/run_tiled_box_10x20_16x8_...
    python explore_data/plot_net_cooling_vs_time.py --summary-only
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TILED_DIR = PROJECT_ROOT / "tiled_outputs"
HR_ARCHIVE_ROOT = Path("/data/sasi/variable_box_size_runs")


def parse_manifest(path: Path) -> dict:
    entries = {}
    for line in path.read_text().splitlines():
        if " : " in line:
            key, val = line.split(" : ", 1)
            entries[key.strip()] = val.strip()
    return entries


def resolve_hr_dir(m: dict) -> Path:
    """HR runs are moved to the HDD archive after the pipeline finishes, so the
    recorded output directory may no longer exist."""
    candidates = [m.get("HR archive location"), m.get("HR output directory")]
    candidates = [Path(c) for c in candidates if c]
    candidates.append(HR_ARCHIVE_ROOT / Path(m["HR output directory"]).name)
    for c in candidates:
        if (c / "bin").is_dir():
            return c
    raise FileNotFoundError(f"No HR bin directory found among {candidates}")


def run_config(run_dir: Path) -> dict:
    m = parse_manifest(run_dir / "manifest.txt")
    nx1, nx2 = re.search(r"nx1=(\d+).*nx2=(\d+)", m["Coarse grid"]).groups()
    logt_lo, logt_hi = re.search(r"\[([\d.]+),\s*([\d.]+)\]", m["Active log10(T) range"]).groups()
    restart_t = re.search(r"t=([\d.]+)", m["Restart snapshot"]).group(1)
    return {
        "env": {
            "CELL_SIZE": m["Coarse cell size (CNN)"].split()[0],
            "DS": m["Downsample factor (DS)"],
            "NX1": nx1,
            "NX2": nx2,
            "TILE_GRID": m["Tile grid (rows,cols)"],
            "MODEL_SAVES_DIR": m["Model saves dir"],
            "NORM_PREFIX": m["Norm prefix"],
            "LOGT_ACTIVE_START": logt_lo,
            "LOGT_ACTIVE_END": logt_hi,
            "RESTART_TIME_MYR": restart_t,
        },
        "hr_athinput": m["HR athinput"],
        "hr_bin": str(resolve_hr_dir(m) / "bin"),
        "sg_athinput": m["Subgrid athinput"],
        "sg_bin": str(Path(m["Subgrid output directory"]) / "bin"),
        "lr_athinput": m["LR athinput"],
        "lr_bin": str(Path(m["LR output directory"]) / "bin"),
        "box": (m["Box length X1 (width)"].split()[0], m["Box length X2 (height)"].split()[0]),
        "cell": m["Coarse cell size (CNN)"].split()[0],
    }


def process_run(run_dir: Path):
    """Compute and plot Sigma_c(t) for one run. Must be called in a process
    whose environment was set from the run's manifest (see main)."""
    cfg = run_config(run_dir)
    sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "explore_data")]
    import ergane
    from tqdm import tqdm
    import mock_sg_tiled as msg

    sim_hr = ergane.SimulationData(athinp=cfg["hr_athinput"], datafolder=cfg["hr_bin"])
    sim_sg = ergane.SimulationData(athinp=cfg["sg_athinput"], datafolder=cfg["sg_bin"])
    sim_lr = ergane.SimulationData(athinp=cfg["lr_athinput"], datafolder=cfg["lr_bin"])

    # Same frame alignment as mock_sg_tiled.main()
    hr_frames = [f for f in sim_hr.frame_numbers if f >= 500]
    nt = min(len(sim_sg.frame_numbers), len(sim_lr.frame_numbers), len(hr_frames))
    Ly = float(sim_sg.x2max - sim_sg.x2min)

    model, input_mean, input_std = msg.load_tiled_cnn_model()
    lam_kw = dict(mask=True, LOGT_ACTIVE_START=msg.LOGT_ACTIVE_START, LOGT_ACTIVE_END=msg.LOGT_ACTIVE_END)

    def lr_emis(f):
        rho = (f.density / f.units.density).astype(np.float64)
        return (rho * msg.n_to_cm3) ** 2 * msg.lambda_cool(f.temperature.astype(np.float32), **lam_kw)

    sigma_hr = np.empty(nt)
    sigma_sg = np.empty(nt)
    sigma_lr = np.empty(nt)
    for i in tqdm(range(nt), desc=run_dir.name):
        sigma_hr[i] = msg.compute_net_cooling(lr_emis(sim_hr.get_frame(hr_frames[i])), Ly)
        sigma_lr[i] = msg.compute_net_cooling(lr_emis(sim_lr.get_frame(sim_lr.frame_numbers[i])), Ly)

        f = sim_sg.get_frame(sim_sg.frame_numbers[i])
        temp = f.temperature.astype(np.float32)
        ps = f.scalars["scalar_00"].astype(np.float32) if "scalar_00" in f.scalars else np.zeros_like(temp)
        cool_sg, _, _ = msg.predict_tiled_subgrid_pdf_and_cooling(
            (f.density / f.units.density).astype(np.float32), temp,
            (f.velx / f.units.velocity).astype(np.float32),
            (f.vely / f.units.velocity).astype(np.float32),
            ps, (f.pressure / f.units.pressure).astype(np.float32),
            model, input_mean, input_std,
        )
        sigma_sg[i] = msg.compute_net_cooling(cool_sg, Ly)

    t_myr = msg.RESTART_TIME_MYR + np.arange(nt) * msg.BIN_DT_MYR
    lx, ly = cfg["box"]
    msg.plot_net_cooling_vs_time(t_myr, sigma_hr, sigma_sg, sigma_lr, run_dir,
                                 title_suffix=f" — box {lx}x{ly} pc, {msg.CELL_LABEL} cells")
    print(f"Saved {run_dir / 'net_cooling_vs_time.png'}")


def load_csv(path: Path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def plot_summary(run_dirs, out_path: Path):
    """Grid of Sigma_c(t): rows = CNN cell size, columns = box size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(PROJECT_ROOT / "explore_data"))
    from mock_sg_tiled import NET_COOLING_COLORS

    runs = {}
    for d in run_dirs:
        csv_path = d / "net_cooling_vs_time.csv"
        if not csv_path.is_file():
            continue
        cfg = run_config(d)
        # Latest run wins if a configuration was repeated (dirs are timestamp-sorted)
        runs[(float(cfg["cell"]), float(cfg["box"][0]), cfg["box"])] = load_csv(csv_path)
    if not runs:
        print("No net_cooling_vs_time.csv files found; skipping summary.")
        return

    cells = sorted({k[0] for k in runs}, reverse=True)
    boxes = sorted({(k[1], k[2]) for k in runs})
    fig, axs = plt.subplots(len(cells), len(boxes), figsize=(4.2 * len(boxes), 3.6 * len(cells)),
                            sharex=True, sharey=True, squeeze=False)
    for i, cell in enumerate(cells):
        for j, (bx, box) in enumerate(boxes):
            ax = axs[i, j]
            data = runs.get((cell, bx, box))
            ax.set_title(f"box {box[0]}x{box[1]} pc, {cell:g} pc cells", fontsize=11)
            ax.grid(True, which="both", ls="--", alpha=0.4)
            if data is None:
                ax.text(0.5, 0.5, "no run", transform=ax.transAxes, ha="center", va="center", color="gray")
                continue
            ax.plot(data["t_myr"], data["sigma_c_hr"], lw=2, ls="-", color=NET_COOLING_COLORS["HR"], label="HR")
            ax.plot(data["t_myr"], data["sigma_c_sg"], lw=2, ls="-.", color=NET_COOLING_COLORS["SG"], label="SG (CNN)")
            ax.plot(data["t_myr"], data["sigma_c_lr"], lw=2, ls="--", color=NET_COOLING_COLORS["LR"], label="LR (ISM)")
            ax.set_yscale("log")
    for ax in axs[-1]:
        ax.set_xlabel("Physical Time [Myr]", fontsize=11)
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$\Sigma_c$ [erg cm$^{-2}$ s$^{-1}$]", fontsize=11)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Net Cooling vs Time — all tiled runs", fontsize=14, weight="bold", y=1.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiled-dir", default=str(DEFAULT_TILED_DIR))
    parser.add_argument("--run-dir", help="Process only this run (in the current process)")
    parser.add_argument("--summary-only", action="store_true", help="Only rebuild the all-runs summary from existing CSVs")
    args = parser.parse_args()

    if args.run_dir:
        process_run(Path(args.run_dir).resolve())
        return

    tiled_dir = Path(args.tiled_dir).resolve()
    run_dirs = sorted(d for d in tiled_dir.glob("run_tiled_*") if (d / "manifest.txt").is_file())

    if not args.summary_only:
        failed = []
        for d in run_dirs:
            print(f"\n=== {d.name} ===", flush=True)
            env = {**os.environ, **run_config(d)["env"]}
            ret = subprocess.run([sys.executable, __file__, "--run-dir", str(d)], env=env)
            if ret.returncode != 0:
                failed.append(d.name)
        if failed:
            print(f"\nFailed runs: {failed}")

    plot_summary(run_dirs, tiled_dir / "net_cooling_vs_time_all_runs.png")


if __name__ == "__main__":
    main()
