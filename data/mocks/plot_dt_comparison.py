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


# dt_cool is reported as float_max*cfl ("no constraint") when a run has no active
# cooling timestep; such points are dropped rather than blowing out the y-axis.
_DT_SENTINEL = 1.0e30


def load_history_file(history_path: str):
    """Load (time, dt, dt_cfl, dt_cool) from an athena KH.hydro.hst file.

    History files are opened in append mode, so re-runs leave earlier rows behind,
    each restarting from its initial time; only the last run is kept. dt_cfl/dt_cool
    are None for history files that predate those columns (checked against the
    last header block, since column layouts can differ between re-runs).
    """
    if not os.path.exists(history_path):
        print(f"WARNING: History file not found: {history_path}", file=sys.stderr)
        return None, None, None, None

    has_components = False
    in_header_block = False
    block_has_components = False
    with open(history_path) as f:
        for line in f:
            if line.startswith("#"):
                if not in_header_block:
                    in_header_block = True
                    block_has_components = False
                if "dt_cfl" in line and "dt_cool" in line:
                    block_has_components = True
            elif in_header_block:
                has_components = block_has_components
                in_header_block = False
        if in_header_block:
            has_components = block_has_components

    try:
        cols = (0, 1, 2, 3) if has_components else (0, 1)
        data = np.loadtxt(history_path, comments="#", usecols=cols, ndmin=2)
    except Exception as e:
        print(f"WARNING: Could not load history file {history_path}: {e}", file=sys.stderr)
        return None, None, None, None
    if len(data) == 0:
        return None, None, None, None

    times = data[:, 0]
    resets = np.where(np.diff(times) < 0)[0]
    start = resets[-1] + 1 if len(resets) > 0 else 0
    data = data[start:]

    times, dts = data[:, 0], data[:, 1]
    dt_cfl = data[:, 2] if has_components else None
    dt_cool = None
    if has_components:
        dt_cool = np.where(data[:, 3] > _DT_SENTINEL, np.nan, data[:, 3])
        if np.all(np.isnan(dt_cool)):
            print(f"WARNING: dt_cool in {history_path} is the no-constraint sentinel everywhere "
                  "(run had no cooling timestep, e.g. subgrid run without user_cooling = true)",
                  file=sys.stderr)
    return times, dts, dt_cfl, dt_cool


def load_dt_cool_log(log_path: str):
    """Load the per-call dt_cool log written by source_module.py.

    Returns (time, dt_cool_clipped, dt_cool_noclip, n_clipped) with one entry per
    cycle, taken from the cycle's LAST source call -- that is the dtnew AthenaK
    uses for the next timestep (and what the history file's dt_cool reports).
    Values are already scaled by cfl_no. n_clipped is summed over the cycle's stages.
    The last element says whether the clip was applied (False for COOL_CLIP=0 runs).
    """
    if not log_path or not os.path.exists(log_path):
        if log_path:
            print(f"WARNING: dt_cool log not found: {log_path}", file=sys.stderr)
        return None
    try:
        import pandas as pd
        d = pd.read_csv(log_path)
    except Exception as e:
        print(f"WARNING: Could not load dt_cool log {log_path}: {e}", file=sys.stderr)
        return None
    if len(d) == 0:
        return None
    # Appended re-runs restart the clock; keep only the last run
    resets = np.where(np.diff(d["time"].values) < 0)[0]
    if len(resets) > 0:
        d = d.iloc[resets[-1] + 1:]
    g = d.groupby("time", sort=True)
    last = g.tail(1).set_index("time")
    n_clip = g["n_clipped"].sum()
    applied = bool(d["clip_applied"].iloc[-1]) if "clip_applied" in d.columns else True
    return (last.index.values, last["dt_cool_clipped"].values,
            last["dt_cool_noclip"].values, n_clip.reindex(last.index).values, applied)


def plot_dt_components(runs, output_dir: str, restart_time: float) -> str:
    """Plot dt (used), dt_CFL and dt_cool vs physical time, one panel per run.

    runs: list of (label, history_path, shift, dt_cool_log_path) where shift=True
    means the run's history time starts at 0 and must be offset by restart_time.
    Runs whose history already starts at >= restart_time (restarts from an rst
    file) are never shifted. dt_cool_log_path (or None) is source_module.py's
    per-call log; when given, the panel shows dt_cool with and without the
    cooling-rate clip, and marks the cycles where clipping happened.
    """
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 5.5), sharey=True, squeeze=False)
    axes = axes[0]
    any_data = False

    for ax, (label, hst_path, shift, dt_log_path) in zip(axes, runs):
        ax.axvline(restart_time, color="gray", ls="--", lw=1.0, alpha=0.7)
        times, dts, dt_cfl, dt_cool = load_history_file(hst_path) if hst_path else (None,) * 4
        if times is None:
            ax.set_title(f"{label}\n(no history data)", fontsize=12)
            continue
        offset = restart_time if (shift and times[0] < restart_time - 1e-6) else 0.0
        times = times + offset
        mask = times >= restart_time - 1e-6
        times, dts = times[mask], dts[mask]
        dt_cfl = dt_cfl[mask] if dt_cfl is not None else None
        dt_cool = dt_cool[mask] if dt_cool is not None else None
        if len(times) == 0:
            ax.set_title(f"{label}\n(no data after {restart_time} Myr)", fontsize=12)
            continue
        any_data = True

        ax.plot(times, dts * 1e3, label=r"$\Delta t$ (used)", lw=2, color="black", alpha=0.85)
        if dt_cfl is not None:
            ax.plot(times, dt_cfl * 1e3, label=r"$\Delta t_\mathrm{CFL}$", lw=1.5,
                    ls="--", marker="^", markersize=3, alpha=0.8)
        dt_log = load_dt_cool_log(dt_log_path)
        if dt_log is not None:
            lt, dtc_clip, dtc_raw, nclip = dt_log[:4]
            lt = lt + offset
            keep = lt >= restart_time - 1e-6
            lt, dtc_clip, dtc_raw, nclip = lt[keep], dtc_clip[keep], dtc_raw[keep], nclip[keep]
            applied = dt_log[4]
            clip_lbl = "used" if applied else "NOT used: COOL_CLIP=0"
            ax.plot(lt, dtc_clip * 1e3, label=rf"$\Delta t_\mathrm{{cool}}$ (with clipping, {clip_lbl})",
                    lw=1.2, color="tab:orange", alpha=0.85)
            ax.plot(lt, dtc_raw * 1e3, label=r"$\Delta t_\mathrm{cool}$ (without clipping" + (")" if applied else ", used)"),
                    lw=1.2, ls="--", color="tab:red", alpha=0.8)
            clipped = nclip > 0
            if np.any(clipped):
                ax.plot(lt[clipped], np.full(clipped.sum(), 0.02), "|", color="tab:red",
                        transform=ax.get_xaxis_transform(), markersize=8, alpha=0.6,
                        label=f"cycle with {'clipping' if applied else 'would-be clipping'} ({clipped.sum()}/{len(lt)})")
        elif dt_cool is not None and not np.all(np.isnan(dt_cool)):
            ax.plot(times, dt_cool * 1e3, label=r"$\Delta t_\mathrm{cool}$", lw=1.5,
                    ls="--", marker="o", markersize=3, alpha=0.8)
        title = label
        if dt_cfl is None:
            title += "\n(history has no dt_cfl/dt_cool columns)"
        elif dt_log is None and (dt_cool is None or np.all(np.isnan(dt_cool))):
            title += "\n(no cooling timestep reported)"
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_yscale("log")
        ax.set_xlabel("Physical Time [Myr]", fontsize=12)
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(fontsize=9)

    if not any_data:
        plt.close(fig)
        print("ERROR: No history data found for timestep components plot", file=sys.stderr)
        return ""

    axes[0].set_ylabel("Timestep [ms (code units)]", fontsize=13)
    fig.suptitle("CFL-limited vs Cooling-limited Timestep Components", fontsize=15, weight="bold")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "delta_t_components_vs_time.png")
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved timestep components plot: {output_path}")
    return output_path


def main():
    """Main entry point."""
    # Get paths from environment variables or arguments
    project_root = os.getenv("PROJECT_ROOT", "/home/sasi/Projects/SubgridCGMModel")
    lr_output_dir = os.getenv("LR_OUTPUT_DIR", f"{project_root}/simulation_outputs/lr_build_ism")
    sg_output_dir = os.getenv("SG_OUTPUT_DIR", f"{project_root}/simulation_outputs/subgrid_model")
    output_dir = os.getenv("SG_MOCKS_DIR", f"{project_root}/outputs/sg_mocks")

    # Explicit log paths (e.g. the pipeline's per-step logs, which hold athena's stdout)
    # take precedence; otherwise look for KH.log in the output directories.
    lr_log = os.getenv("LR_LOG", os.path.join(lr_output_dir, "KH.log"))
    sg_log = os.getenv("SG_LOG", os.path.join(sg_output_dir, "KH.log"))

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

    # Timestep components (dt, dt_CFL, dt_cool) from the history files
    restart_time = float(os.getenv("RESTART_TIME_MYR", "5.0"))
    hr_output_dir = os.getenv("HR_OUTPUT_DIR", "")
    nx1, nx2 = os.getenv("SIM_NX1"), os.getenv("SIM_NX2")
    coarse = f" ({nx1}x{nx2})" if nx1 and nx2 else ""
    dt_cool_log = os.getenv("DT_COOL_LOG_PATH", "")
    runs = []
    if hr_output_dir:
        runs.append(("HR", os.path.join(hr_output_dir, "KH.hydro.hst"), False, None))
    runs += [
        (f"SG{coarse}", os.path.join(sg_output_dir, "KH.hydro.hst"), True, dt_cool_log or None),
        (f"LR{coarse}", os.path.join(lr_output_dir, "KH.hydro.hst"), True, None),
    ]
    components_path = plot_dt_components(runs, output_dir, restart_time)

    if plot_path:
        print(f"\nPlot comparison complete: {plot_path}")
    if not plot_path and not components_path:
        print("WARNING: Plot generation failed or skipped")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
