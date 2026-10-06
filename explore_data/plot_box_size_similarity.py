#!/usr/bin/env python3
"""
Plot profile similarity (Pearson r and RMSD vs coarse-grained HR) as a
function of box size, for every run in tiled_outputs/.

Each run directory written by run_variable_box_subgrid.sh contains
  - manifest.txt                   (box size, CNN cell size, tile shape)
  - profile_comparison_metrics.csv (per-profile Pearson/RMSD for SG and LR)

Runs are grouped by CNN cell size (one figure pair per cell size), plus a
combined figure pair overlaying all cell sizes. If the same
box/cell/tile configuration was run more than once, the latest run is used.
Profiles whose metrics are identical to an earlier profile in every run
(e.g. "Mass Flux (rho u_x)" == "Conserved MomX") are shown only once.

Usage:
    python explore_data/plot_box_size_similarity.py
    python explore_data/plot_box_size_similarity.py --tiled-dir tiled_outputs --output-dir tiled_outputs/box_size_similarity
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SG_COLOR = "#2a78d6"
LR_COLOR = "#eb6834"
TEXT_SECONDARY = "#5f5e57"
GRID_COLOR = "#e4e3dc"

# CSV profile name -> (mathtext title, units of the profile, and hence of its RMSD).
# Every profile is <q>_{x,t}(y): q averaged over x, then over time, as computed
# in explore_data/mock_sg_tiled.py. Density and temperature are compared in
# log10 space there, so their RMSD is in dex.
AVG = r"\langle {} \rangle_{{x,t}}"
PROFILE_LABELS = {
    "Density": (rf"${AVG.format(r'\log_{10} n')}$", "dex"),
    "Temperature": (rf"${AVG.format(r'\log_{10} T')}$", "dex"),
    "Pressure": (rf"${AVG.format(r'P / k_B')}$", r"K cm$^{-3}$"),
    "Ux Velocity": (rf"${AVG.format('u_x')}$", r"km s$^{-1}$"),
    "Uy Velocity": (rf"${AVG.format('u_y')}$", r"km s$^{-1}$"),
    "Conserved Density": (rf"${AVG.format('n')}$", r"cm$^{-3}$"),
    "Conserved MomX": (rf"${AVG.format(r'\rho u_x')}$  (momentum / mass flux)",
                       r"M$_\odot$ yr$^{-1}$ kpc$^{-2}$"),
    "Conserved MomY": (rf"${AVG.format(r'\rho u_y')}$  (momentum / mass flux)",
                       r"M$_\odot$ yr$^{-1}$ kpc$^{-2}$"),
    "Conserved Energy": (rf"${AVG.format('E')}$,  $E = e + \frac{{1}}{{2}}\rho |\mathbf{{u}}|^2$",
                         r"erg cm$^{-3}$"),
    "Passive Scalar": (rf"${AVG.format(r'\rho s')}$  (passive scalar)", "code density units"),
    "fmcl (T < 1e5)": (rf"${AVG.format(r'f_\mathrm{cold}')}$  ($T < 10^5$ K)", "dimensionless"),
    "rho u_x u_y (Avg over X)": (rf"${AVG.format(r'T_{xy}')} = {AVG.format(r'\rho u_x u_y')}$",
                                 r"dyn cm$^{-2}$"),
    "p + rho u_y^2 (Avg over X)": (rf"${AVG.format(r'T_{yy}')} = {AVG.format(r'p + \rho u_y^2')}$",
                                   r"dyn cm$^{-2}$"),
    "Momentum Flux T_{xx} = rho u_x^2 + p": (rf"${AVG.format(r'T_{xx}')} = {AVG.format(r'p + \rho u_x^2')}$",
                                             r"dyn cm$^{-2}$"),
    "Energy Flux (E+p)u_x": (rf"${AVG.format(r'(E + p)\,u_x')}$", r"erg cm$^{-2}$ s$^{-1}$"),
    "Energy Flux (E+p)u_y": (rf"${AVG.format(r'(E + p)\,u_y')}$", r"erg cm$^{-2}$ s$^{-1}$"),
    "Div Mass Flux (div j)": (rf"${AVG.format(r'\nabla \cdot (\rho \mathbf{u})')}$",
                              r"M$_\odot$ yr$^{-1}$ kpc$^{-2}$ pc$^{-1}$"),
    "Div MomX Flux (div T_x)": (rf"${AVG.format(r'\nabla \cdot \mathbf{T}_x')}$,  $\mathbf{{T}}_x = (T_{{xx}}, T_{{xy}})$",
                                r"dyn cm$^{-2}$ pc$^{-1}$"),
    "Div MomY Flux (div T_y)": (rf"${AVG.format(r'\nabla \cdot \mathbf{T}_y')}$,  $\mathbf{{T}}_y = (T_{{xy}}, T_{{yy}})$",
                                r"dyn cm$^{-2}$ pc$^{-1}$"),
    "Mean Cooling Rate Profile vs y": (rf"${AVG.format(r'n^2 \Lambda(T)')}$  (cooling rate)",
                                       r"erg cm$^{-3}$ s$^{-1}$"),
}

SERIES = [
    # (label, csv column prefix, color, marker, linestyle)
    ("Subgrid (CNN) vs CG HR", "CGHR_Subgrid", SG_COLOR, "o", "-"),
    ("LR (ISM cooling) vs CG HR", "CGHR LR", LR_COLOR, "s", "--"),
]


def parse_manifest(path):
    text = path.read_text()

    def grab(pattern, cast=str):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None

    return {
        "lx": grab(r"Box length X1 \(width\)\s*:\s*([0-9.]+)", float),
        "ly": grab(r"Box length X2 \(height\)\s*:\s*([0-9.]+)", float),
        "cell_size": grab(r"Coarse cell size \(CNN\)\s*:\s*([0-9.]+)", float),
        "tile_shape": grab(r"Tile shape \(cells\)\s*:\s*(\d+ rows x \d+ cols)"),
        "timestamp": grab(r"Timestamp\s*:\s*(\d{8}_\d{6})"),
        "model_dir": grab(r"Model saves dir\s*:\s*(\S+)"),
    }


def parse_metrics(path):
    metrics = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            vals = {}
            for _, prefix, *_ in SERIES:
                for kind in ("Pearson", "RMSD"):
                    raw = row.get(f"{prefix} {kind}", "")
                    vals[(prefix, kind)] = float(raw) if raw not in ("", None) else np.nan
            metrics[row["Profile"]] = vals
    return metrics


def collect_runs(tiled_dir):
    runs = {}
    for run_dir in sorted(tiled_dir.iterdir()):
        manifest = run_dir / "manifest.txt"
        metrics = run_dir / "profile_comparison_metrics.csv"
        if not (manifest.is_file() and metrics.is_file()):
            continue
        info = parse_manifest(manifest)
        if info["lx"] is None or info["ly"] is None or info["cell_size"] is None:
            print(f"  Skipping {run_dir.name}: manifest missing box/cell size")
            continue
        info["name"] = run_dir.name
        info["metrics"] = parse_metrics(metrics)
        key = (info["lx"], info["ly"], info["cell_size"], info["tile_shape"])
        if key not in runs or (info["timestamp"] or "") > (runs[key]["timestamp"] or ""):
            runs[key] = info
    return list(runs.values())


def unique_profiles(runs):
    """Profile names in CSV order, dropping exact duplicates of earlier profiles."""
    order = []
    for run in runs:
        for name in run["metrics"]:
            if name not in order:
                order.append(name)

    def signature(name):
        return tuple(
            tuple(run["metrics"].get(name, {}).get(k, np.nan) for k in sorted(run["metrics"][order[0]]))
            for run in runs
        )

    kept, seen = [], {}
    for name in order:
        sig = signature(name)
        if sig in seen:
            print(f"  '{name}' duplicates '{seen[sig]}' -- plotted once")
            continue
        seen[sig] = name
        kept.append(name)
    return kept


def tick_label(run):
    label = f"{run['lx']:g}×{run['ly']:g}"
    if run["tile_shape"]:
        rows, cols = re.findall(r"\d+", run["tile_shape"])
        label += f"\n({rows}×{cols} tiles)"
    return label


# Line style per CNN cell size in the combined (all-cell-size) figures.
CELL_STYLES = [("-", "o", None), ("--", "s", "white"), (":", "^", None), ("-.", "D", "white")]


def per_cell_lines(runs):
    """One line per SERIES entry, for a single cell-size group."""
    return [dict(label=label, runs=runs, prefix=prefix, color=color, marker=marker, ls=ls, mfc=None)
            for label, prefix, color, marker, ls in SERIES]


def combined_lines(groups):
    """One line per (cell size, SERIES entry): color = method, line/marker style = cell size."""
    lines = []
    for (cell_size, runs), (ls, marker, mfc) in zip(groups, CELL_STYLES):
        for label, prefix, color, _, _ in SERIES:
            lines.append(dict(label=f"{label}  ({cell_size:g} pc cells)", runs=runs, prefix=prefix,
                              color=color, marker=marker, ls=ls, mfc=mfc))
    return lines


def plot_metric(lines, profiles, kind, subtitle, out_path, show_tiles=True):
    all_runs = [r for line in lines for r in line["runs"]]
    x_runs = {}
    for r in sorted(all_runs, key=lambda r: (r["ly"], r["lx"])):
        x_runs.setdefault(r["ly"], r)
    x_ticks = np.array(list(x_runs))

    ncols = 4
    nrows = int(np.ceil(len(profiles) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)

    for ax, profile in zip(axes.flat, profiles):
        all_vals = []
        for line in lines:
            runs = sorted(line["runs"], key=lambda r: (r["ly"], r["lx"]))
            x = np.array([r["ly"] for r in runs])
            y = np.array([r["metrics"].get(profile, {}).get((line["prefix"], kind), np.nan) for r in runs])
            all_vals.append(y)
            mfc = line["mfc"] or line["color"]
            ax.plot(x, y, color=line["color"], marker=line["marker"], ls=line["ls"], lw=2, ms=7,
                    markerfacecolor=mfc,
                    markeredgecolor="white" if line["mfc"] is None else line["color"],
                    markeredgewidth=1.2, label=line["label"])

        vals = np.concatenate(all_vals)
        finite = vals[np.isfinite(vals)]
        if kind == "Pearson":
            # Fit the axis to the data (with padding) so near-perfect profiles
            # (r ~ 0.99) still show their variation with box size.
            lo, hi = (finite.min(), finite.max()) if finite.size else (0.0, 1.0)
            pad = max(0.08 * (hi - lo), 0.002)
            ax.set_ylim(lo - pad, min(hi + pad, 1.0 + pad))
            if hi + pad >= 1.0:
                ax.axhline(1.0, color=TEXT_SECONDARY, lw=0.8, ls=":")
        elif finite.size and np.all(finite > 0) and finite.max() / finite.min() > 20:
            ax.set_yscale("log")

        title, units = PROFILE_LABELS.get(profile, (profile, "profile units"))
        ax.set_title(title, fontsize=10, pad=14)
        if kind == "RMSD":
            ax.set_ylabel(f"RMSD [{units}]", fontsize=8)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([tick_label(r) if show_tiles else f"{r['lx']:g}×{r['ly']:g}"
                            for r in x_runs.values()], fontsize=7, color=TEXT_SECONDARY)
        ax.tick_params(axis="y", labelsize=8, colors=TEXT_SECONDARY)
        ax.grid(True, color=GRID_COLOR, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    for ax in list(axes.flat)[len(profiles):]:
        ax.set_visible(False)

    if kind == "Pearson":
        for row in axes:
            row[0].set_ylabel("Pearson $r$ (dimensionless)", fontsize=9)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend_ncol = len(SERIES)
    fig.legend(handles, labels, loc="upper center", ncol=legend_ncol, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 1.0))
    legend_rows = int(np.ceil(len(lines) / legend_ncol))
    if kind == "Pearson":
        metric = r"Pearson $r$ between $q_\mathrm{run}(y)$ and $q_\mathrm{CG\,HR}(y)$"
    else:
        metric = (r"RMSD $= \sqrt{\langle (q_\mathrm{run}(y) - q_\mathrm{CG\,HR}(y))^2 \rangle_y}$")
    fig.suptitle(f"{metric} vs box size  ({subtitle})\n"
                 r"$\langle q \rangle_{x,t}(y)$: $x$-average, then time average over the restart window;"
                 r"  x-axis: box $L_x \times L_y$ [pc]",
                 fontsize=13, y=1.02 + 0.02 * legend_rows)
    fig.tight_layout(rect=(0, 0, 1, 1.0 - 0.015 * legend_rows))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def write_summary_csv(runs, profiles, out_path):
    runs = sorted(runs, key=lambda r: (r["ly"], r["lx"]))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "LX_pc", "LY_pc", "cell_size_pc", "tile_shape", "profile",
                    "SG_pearson", "SG_rmsd", "LR_pearson", "LR_rmsd"])
        for r in runs:
            for p in profiles:
                m = r["metrics"].get(p, {})
                w.writerow([r["name"], r["lx"], r["ly"], r["cell_size"], r["tile_shape"], p,
                            m.get(("CGHR_Subgrid", "Pearson")), m.get(("CGHR_Subgrid", "RMSD")),
                            m.get(("CGHR LR", "Pearson")), m.get(("CGHR LR", "RMSD"))])
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiled-dir", type=Path, default=PROJECT_ROOT / "tiled_outputs")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: <tiled-dir>/box_size_similarity")
    args = parser.parse_args()

    out_dir = args.output_dir or (args.tiled_dir / "box_size_similarity")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(args.tiled_dir)
    if not runs:
        raise SystemExit(f"No runs with manifest.txt + profile_comparison_metrics.csv in {args.tiled_dir}")

    groups = []
    for cell_size in sorted({r["cell_size"] for r in runs}, reverse=True):
        group = [r for r in runs if r["cell_size"] == cell_size]
        groups.append((cell_size, group))
        print(f"\nCell size {cell_size:g} pc: {len(group)} runs")
        for r in sorted(group, key=lambda r: r["ly"]):
            print(f"  {r['lx']:g}x{r['ly']:g} pc, tiles {r['tile_shape']}  <- {r['name']}")
        profiles = unique_profiles(group)
        tag = f"cell{cell_size:g}pc"
        subtitle = f"CNN cell size {cell_size:g} pc"
        lines = per_cell_lines(group)
        plot_metric(lines, profiles, "Pearson", subtitle, out_dir / f"pearson_vs_box_size_{tag}.png")
        plot_metric(lines, profiles, "RMSD", subtitle, out_dir / f"rmsd_vs_box_size_{tag}.png")
        write_summary_csv(group, profiles, out_dir / f"similarity_vs_box_size_{tag}.csv")

    if len(groups) > 1:
        if len(groups) > len(CELL_STYLES):
            raise SystemExit(f"Only {len(CELL_STYLES)} cell sizes can be overlaid; got {len(groups)}")
        print(f"\nAll cell sizes combined: {len(runs)} runs")
        profiles = unique_profiles(runs)
        subtitle = "CNN cell sizes " + ", ".join(f"{c:g}" for c, _ in groups) + " pc"
        lines = combined_lines(groups)
        plot_metric(lines, profiles, "Pearson", subtitle, out_dir / "pearson_vs_box_size_all_cells.png",
                    show_tiles=False)
        plot_metric(lines, profiles, "RMSD", subtitle, out_dir / "rmsd_vs_box_size_all_cells.png",
                    show_tiles=False)
        write_summary_csv(runs, profiles, out_dir / "similarity_vs_box_size_all_cells.csv")


if __name__ == "__main__":
    main()
