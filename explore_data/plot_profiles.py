"""
plot_profiles.py (formerly cooling_rate_animation.py)
─────────────────────────────────────────────────────────────────────────────
Computes and plots y-profiles for all four HR MPI simulations:
  - Log Number Density: log10(n_H)    [log10(cm⁻³)]
  - Log Temperature:  log10(T)      [log10(K)]
  - Cooling rate:     q_cool        [erg s⁻¹ cm⁻³]
  - Pressure:         P             [dyn cm⁻²]
  - Velocity X:       v_x           [km s⁻¹]
  - Velocity Y:       v_y           [km s⁻¹]
  - Horizontal flux:  n_H v_x       [cm⁻² s⁻¹]
  - Vertical flux:    n_H v_y       [cm⁻² s⁻¹]

All profiles are:
  - Computed as a function of y-position [pc] on the x-axis (matching dynamics_test.py)
  - Averaged across x for each snapshot
  - Time-averaged over the last 500 snapshots (showing mean ± 1σ across time)
  - Compared across all four resolutions (128x256, 256x512, 512x1024, 1024x2048)
"""

import os
import sys
import gc
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import ergane

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/sasi/Projects/SubgridCGMModel")
SIM_ROOT     = PROJECT_ROOT / "simulation_outputs"
OUT_ROOT     = PROJECT_ROOT / "outputs"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ── Simulation configurations ────────────────────────────────────────────────
simulations = [
    {
        "name":       "hr_512",
        "label":      r"$512 \times 256$",
        "athinp":     "/home/sasi/Projects/SubgridCGMModel/simulation_outputs/hr_build_512/kh_radiative_256x512.athinput",
        "datafolder": "/home/sasi/Projects/SubgridCGMModel/simulation_outputs/hr_build_512",
        "downsample": 32,
    },
]

# ── Physical constants ────────────────────────────────────────────────────────
CM_PER_PC       = 3.08568e18          # cm per parsec
CM_PER_KM       = 1.0e5               # cm per km
SECONDS_PER_MYR = 3.15576e13          # seconds per Myr
M_H             = 1.6726219e-24       # proton mass [g]
MU              = 0.62                # mean molecular weight

# ── Active-temperature mask bounds for cooling ────────────────────────────────
LOGT_ACTIVE_START = np.log10(1.1e4)   # ~ 4.041
LOGT_ACTIVE_END   = np.log10(0.9e6)   # ~ 5.954


# ── Cooling function (from pdf_cnn.py lambda_cool) ───────────────────────────

def lambda_cool(temp: np.ndarray, mask: bool = True) -> np.ndarray:
    """
    ISMCoolFn cooling curve translated from AthenaK C++.
    Returns Λ(T) in erg cm³ s⁻¹.
    """
    temp = np.asarray(temp, dtype=float)
    scalar_input = temp.ndim == 0
    temp = np.atleast_1d(temp)

    logt = np.log10(temp)

    lhd = np.array([
        -22.5977, -21.9689, -21.5972, -21.4615, -21.4789, -21.5497,
        -21.6211, -21.6595, -21.6426, -21.5688, -21.4771, -21.3755,
        -21.2693, -21.1644, -21.0658, -20.9778, -20.8986, -20.8281,
        -20.7700, -20.7223, -20.6888, -20.6739, -20.6815, -20.7051,
        -20.7229, -20.7208, -20.7058, -20.6896, -20.6797, -20.6749,
        -20.6709, -20.6748, -20.7089, -20.8031, -20.9647, -21.1482,
        -21.2932, -21.3767, -21.4129, -21.4291, -21.4538, -21.5055,
        -21.5740, -21.6300, -21.6615, -21.6766, -21.6886, -21.7073,
        -21.7304, -21.7491, -21.7607, -21.7701, -21.7877, -21.8243,
        -21.8875, -21.9738, -22.0671, -22.1537, -22.2265, -22.2821,
        -22.3213, -22.3462, -22.3587, -22.3622, -22.3590, -22.3512,
        -22.3420, -22.3342, -22.3312, -22.3346, -22.3445, -22.3595,
        -22.3780, -22.4007, -22.4289, -22.4625, -22.4995, -22.5353,
        -22.5659, -22.5895, -22.6059, -22.6161, -22.6208, -22.6213,
        -22.6184, -22.6126, -22.6045, -22.5945, -22.5831, -22.5707,
        -22.5573, -22.5434, -22.5287, -22.5140, -22.4992, -22.4844,
        -22.4695, -22.4543, -22.4392, -22.4237, -22.4087, -22.3928,
    ])

    lam = np.zeros_like(temp, dtype=float)

    # T <= 1e4 K -> no cooling
    lam[logt <= 4.0] = 0.0

    # KI02 regime (4.0 < logT <= 4.2)
    mk = (logt > 4.0) & (logt <= 4.2)
    if np.any(mk):
        lam[mk] = (
            2.0e-19 * np.exp(-1.184e5 / (temp[mk] + 1.0e3))
            + 2.8e-28 * np.sqrt(temp[mk]) * np.exp(-92.0 / temp[mk])
        )

    # CGOLS fit (logT > 8.15)
    mhi = logt > 8.15
    lam[mhi] = 10.0 ** (0.45 * logt[mhi] - 26.065)

    # SPEX interpolation (4.2 < logT <= 8.15)
    mm = (logt > 4.2) & (logt <= 8.15)
    if np.any(mm):
        ipps = np.clip((25.0 * logt[mm] - 103).astype(int), 0, 100)
        x0   = 4.12 + 0.04 * ipps
        dx   = logt[mm] - x0
        logcool = (lhd[ipps + 1] * dx - lhd[ipps] * (dx - 0.04)) * 25.0
        lam[mm] = 10.0 ** logcool

    if mask:
        mask_off = (logt < LOGT_ACTIVE_START) | (logt > LOGT_ACTIVE_END)
        lam[mask_off] = 0.0

    return lam[0] if scalar_input else lam


# ── Coarse-graining helper ────────────────────────────────────────────────────

def coarse_grain_2d(arr: np.ndarray, ds: int = 32) -> np.ndarray:
    """Coarse-grain a 2D array of shape (ny, nx) by factor ds."""
    ny, nx = arr.shape
    return arr.reshape(ny // ds, ds, nx // ds, ds).mean(axis=(1, 3))


# ── Field extractors ─────────────────────────────────────────────────────────

def compute_physical_fields(frame: ergane.simulation_data.Frame) -> dict[str, np.ndarray]:
    """
    Extract all 8 physical fields for a single snapshot:
      - log10_number_density: log10(n_H) [log10(cm⁻³)]
      - log10_temperature:    log10(T)   [log10(K)]
      - cooling:              q_cool     [erg s⁻¹ cm⁻³]
      - pressure:             P          [dyn cm⁻²]
      - velx:                 v_x        [km s⁻¹]
      - vely:                 v_y        [km s⁻¹]
      - mass_flux_x:          ρ v_x      [g cm⁻² s⁻¹]
      - mass_flux_y:          ρ v_y      [g cm⁻² s⁻¹]
    """
    rho_cgs = frame.density              # [g cm⁻³]
    P_cgs   = frame.pressure             # [dyn cm⁻²]
    temp_K  = frame.temperature          # [K]
    vx_kms  = frame.velx                 # [km s⁻¹]
    vy_kms  = frame.vely                 # [km s⁻¹]

    # Number density: n_H = ρ / (μ m_H) [cm⁻³]
    n_H = rho_cgs / (MU * M_H)

    # Log10 fields
    log10_nH = np.log10(np.maximum(n_H, 1e-30))
    log10_T  = np.log10(np.maximum(temp_K, 1.0))

    # Physical volumetric cooling rate: n_H² Λ(T)
    lam    = lambda_cool(temp_K, mask=True) # [erg cm³ s⁻¹]
    q_cool = (n_H ** 2) * lam               # [erg s⁻¹ cm⁻³]

    # Fluxes using number density in CGS: n_H * v (with v in cm/s) -> [cm⁻² s⁻¹]
    flux_x = n_H * (vx_kms * CM_PER_KM) # [cm⁻² s⁻¹]
    flux_y = n_H * (vy_kms * CM_PER_KM) # [cm⁻² s⁻¹]

    return {
        "log10_number_density": log10_nH,
        "log10_temperature":    log10_T,
        "cooling":              q_cool,
        "pressure":             P_cgs,
        "velx":                 vx_kms,
        "vely":                 vy_kms,
        "flux_x":               flux_x,
        "flux_y":               flux_y,
    }


def compute_coarse_grained_fields(frame: ergane.simulation_data.Frame, ds: int = 32) -> dict[str, np.ndarray]:
    """
    Compute coarse-grained versions of physical fields matching mock_sg.py:
      - Primitive fields (rho, P, T, vx, vy) are coarse-grained by factor ds.
      - Cooling rate emis_cg is the coarse-grained fine cooling rate n_H^2 Lambda(T).
      - Fluxes are coarse-grained fine fluxes.
    """
    rho_cgs = frame.density
    P_cgs   = frame.pressure
    temp_K  = frame.temperature
    vx_kms  = frame.velx
    vy_kms  = frame.vely

    n_H = rho_cgs / (MU * M_H)
    lam = lambda_cool(temp_K, mask=True)
    q_cool = (n_H ** 2) * lam
    flux_x = n_H * (vx_kms * CM_PER_KM)
    flux_y = n_H * (vy_kms * CM_PER_KM)

    # Coarse grain primitives and quantities
    rho_cg  = coarse_grain_2d(rho_cgs, ds)
    n_H_cg  = rho_cg / (MU * M_H)
    temp_cg = coarse_grain_2d(temp_K, ds)
    P_cg    = coarse_grain_2d(P_cgs, ds)
    vx_cg   = coarse_grain_2d(vx_kms, ds)
    vy_cg   = coarse_grain_2d(vy_kms, ds)

    # Cooling rate: coarse-grained fine emissivity (matching mock_sg.py emis_cg_hr)
    q_cool_cg = coarse_grain_2d(q_cool, ds)
    flux_x_cg = coarse_grain_2d(flux_x, ds)
    flux_y_cg = coarse_grain_2d(flux_y, ds)

    log10_nH_cg = np.log10(np.maximum(n_H_cg, 1e-30))
    log10_T_cg  = np.log10(np.maximum(temp_cg, 1.0))

    return {
        "log10_number_density": log10_nH_cg,
        "log10_temperature":    log10_T_cg,
        "cooling":              q_cool_cg,
        "pressure":             P_cg,
        "velx":                 vx_cg,
        "vely":                 vy_cg,
        "flux_x":               flux_x_cg,
        "flux_y":               flux_y_cg,
    }


def x_average_profile(frame: ergane.simulation_data.Frame, values: np.ndarray) -> np.ndarray:
    """Compute the x-averaged profile of a 2-D field as a function of y."""
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D field, got shape {values.shape!r}.")
    dx = np.abs(np.diff(frame.x))
    if values.shape[1] != dx.size:
        raise ValueError(
            f"Field shape {values.shape!r} is incompatible with x grid of size {dx.size!r}."
        )
    weighted_sum = np.sum(values * dx[None, :], axis=1)
    return weighted_sum / np.sum(dx)


def get_y_coords_pc(frame: ergane.simulation_data.Frame) -> np.ndarray:
    """Return cell-centre y-coordinates in parsecs."""
    return frame.yc / CM_PER_PC


# ── Metadata for plotting ────────────────────────────────────────────────────

FIELD_CONFIGS = [
    {
        "key":       "log10_number_density",
        "title":     r"Log-Number-Density Profile $\langle \log_{10} n_{\rm H} \rangle_x$",
        "ylabel":    r"$\langle \log_{10} (n_{\rm H} / \mathrm{cm^{-3}}) \rangle_x$",
        "filename":  "profile_log10_number_density",
        "yscale":    "linear",
    },
    {
        "key":       "log10_temperature",
        "title":     r"Log-Temperature Profile $\langle \log_{10} T \rangle_x$",
        "ylabel":    r"$\langle \log_{10} (T / \mathrm{K}) \rangle_x$",
        "filename":  "profile_log10_temperature",
        "yscale":    "linear",
    },
    {
        "key":       "cooling",
        "title":     r"Mean Cooling Rate Profile vs $y$",
        "ylabel":    r"$\langle n^2 \Lambda(T) \rangle \ [\mathrm{erg} \ \mathrm{cm}^{-3} \ \mathrm{s}^{-1}]$",
        "filename":  "profile_cooling_rate",
        "yscale":    "log",
    },
    {
        "key":       "pressure",
        "title":     r"Pressure Profile $\langle P \rangle_x$",
        "ylabel":    r"$\langle P \rangle_x \ [\mathrm{dyn\ cm^{-2}}]$",
        "filename":  "profile_pressure",
        "yscale":    "linear",
    },
    {
        "key":       "velx",
        "title":     r"Horizontal Velocity Profile $\langle v_x \rangle_x$",
        "ylabel":    r"$\langle v_x \rangle_x \ [\mathrm{km\ s^{-1}}]$",
        "filename":  "profile_velx",
        "yscale":    "linear",
    },
    {
        "key":       "vely",
        "title":     r"Vertical Velocity Profile $\langle v_y \rangle_x$",
        "ylabel":    r"$\langle v_y \rangle_x \ [\mathrm{km\ s^{-1}}]$",
        "filename":  "profile_vely",
        "yscale":    "linear",
    },
    {
        "key":       "flux_x",
        "title":     r"Horizontal Flux Profile $\langle n_{\rm H} v_x \rangle_x$",
        "ylabel":    r"$\langle n_{\rm H} v_x \rangle_x \ [\mathrm{cm^{-2}\ s^{-1}}]$",
        "filename":  "profile_flux_x",
        "yscale":    "linear",
    },
    {
        "key":       "flux_y",
        "title":     r"Vertical Flux Profile $\langle n_{\rm H} v_y \rangle_x$",
        "ylabel":    r"$\langle n_{\rm H} v_y \rangle_x \ [\mathrm{cm^{-2}\ s^{-1}}]$",
        "filename":  "profile_flux_y",
        "yscale":    "linear",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def main():
    profile_results = {}

    for sim in simulations:
        name  = sim["name"]
        label = sim["label"]
        ds    = sim.get("downsample", 32)
        print(f"\n{'='*60}")
        print(f"  Computing profiles for {name} ({label}, downsample={ds})")
        print(f"{'='*60}")

        sim_data = ergane.SimulationData(
            athinp=str(sim["athinp"]),
            datafolder=str(sim["datafolder"]),
        )

        n_frames   = sim_data.n_frames
        frame_nums = sim_data.frame_numbers
        print(f"  {n_frames} frames available (#{frame_nums[0]}–#{frame_nums[-1]})")

        # Use last 500 snapshots
        n_avg       = min(500, n_frames)
        avg_indices = frame_nums[-n_avg:]
        print(f"  Time-averaging over last {n_avg} snapshots …")

        # Pre-read grid
        frame0 = sim_data.get_frame(frame_nums[0])
        y_pc_raw = get_y_coords_pc(frame0)
        ny_raw   = y_pc_raw.size
        nx_raw   = frame0.xc.size

        ny_cg = ny_raw // ds
        nx_cg = nx_raw // ds
        y_pc_cg = y_pc_raw.reshape(ny_cg, ds).mean(axis=1)

        # Storage for all fields: field_name -> array of shape (n_avg, ny)
        field_stacks_raw = {cfg["key"]: np.zeros((n_avg, ny_raw), dtype=np.float64) for cfg in FIELD_CONFIGS}
        field_stacks_cg  = {cfg["key"]: np.zeros((n_avg, ny_cg), dtype=np.float64) for cfg in FIELD_CONFIGS}

        for idx, fn in enumerate(tqdm(avg_indices, desc=f"  [{name}] Snapshots", unit="frame")):
            f = sim_data.get_frame(fn)
            fields_raw = compute_physical_fields(f)
            fields_cg  = compute_coarse_grained_fields(f, ds=ds)

            for key in FIELD_CONFIGS:
                k = key["key"]
                field_stacks_raw[k][idx] = x_average_profile(f, fields_raw[k])
                field_stacks_cg[k][idx]  = np.mean(fields_cg[k], axis=1)

            del f
            if idx % 100 == 0:
                gc.collect()

        # Compute mean and standard deviation over time
        sim_summary = {
            "y_pc_raw":  y_pc_raw,
            "y_pc_cg":   y_pc_cg,
            "label_raw": rf"HR (${ny_raw} \times {nx_raw}$)",
            "label_cg":  rf"CG HR (${ny_cg} \times {nx_cg}$)",
            "ny_raw":    ny_raw,
            "nx_raw":    nx_raw,
            "ny_cg":     ny_cg,
            "nx_cg":     nx_cg,
        }
        for key in FIELD_CONFIGS:
            k = key["key"]
            with np.errstate(all="ignore"):
                sim_summary[f"{k}_raw_mean"] = np.nanmean(field_stacks_raw[k], axis=0)
                sim_summary[f"{k}_raw_std"]  = np.nanstd(field_stacks_raw[k],  axis=0)
                sim_summary[f"{k}_cg_mean"]  = np.nanmean(field_stacks_cg[k],  axis=0)
                sim_summary[f"{k}_cg_std"]   = np.nanstd(field_stacks_cg[k],   axis=0)

        profile_results[name] = sim_summary
        print(f"  Done {name} (ny_raw={ny_raw}, ny_cg={ny_cg}).")

    # ══════════════════════════════════════════════════════════════════════════
    # PLOTTING: CONSOLIDATED 8-PANEL COMPARISON FIGURE
    # ══════════════════════════════════════════════════════════════════════════

    print("\nPlotting consolidated 8-panel profile comparison figure …")

    fig, axes = plt.subplots(4, 2, figsize=(14, 18), sharex=True)
    axes_flat = axes.flatten()

    for ax, cfg in zip(axes_flat, FIELD_CONFIGS):
        key    = cfg["key"]
        ylabel = cfg["ylabel"]
        title  = cfg["title"]
        is_log = (cfg["yscale"] == "log")

        for idx, sim in enumerate(simulations):
            name      = sim["name"]
            res       = profile_results[name]
            y_pc_raw  = res["y_pc_raw"]
            y_pc_cg   = res["y_pc_cg"]
            m_raw     = res[f"{key}_raw_mean"]
            s_raw     = res[f"{key}_raw_std"]
            m_cg      = res[f"{key}_cg_mean"]
            s_cg      = res[f"{key}_cg_std"]

            label_raw = res["label_raw"]
            label_cg  = res["label_cg"]

            if key == "cooling":
                int_raw = np.trapezoid(m_raw, y_pc_raw)
                int_cg  = np.trapezoid(m_cg, y_pc_cg)
                label_raw = rf"{label_raw} ($\Sigma_c = {int_raw:.2e}$)"
                label_cg  = rf"{label_cg} ($\Sigma_c = {int_cg:.2e}$)"

            # Raw HR line
            ax.plot(y_pc_raw, m_raw, lw=2, ls="-", color="tab:blue", label=label_raw)
            if is_log:
                ax.fill_between(
                    y_pc_raw,
                    np.clip(m_raw - s_raw, 1e-30, None),
                    m_raw + s_raw,
                    color="tab:blue",
                    alpha=0.2,
                    linewidth=0,
                )
            else:
                ax.fill_between(
                    y_pc_raw,
                    m_raw - s_raw,
                    m_raw + s_raw,
                    color="tab:blue",
                    alpha=0.2,
                    linewidth=0,
                )

            # Coarse-grained HR line
            ax.plot(y_pc_cg, m_cg, lw=2, ls="-.", marker="^", markersize=4, color="tab:orange", label=label_cg)
            if is_log:
                ax.fill_between(
                    y_pc_cg,
                    np.clip(m_cg - s_cg, 1e-30, None),
                    m_cg + s_cg,
                    color="tab:orange",
                    alpha=0.25,
                    linewidth=0,
                )
            else:
                ax.fill_between(
                    y_pc_cg,
                    m_cg - s_cg,
                    m_cg + s_cg,
                    color="tab:orange",
                    alpha=0.25,
                    linewidth=0,
                )

        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, which="both" if is_log else "major", ls="--", alpha=0.4)
        if is_log:
            ax.set_yscale("log")
            # Set a reasonable bottom limit to avoid blanking on non-positive values
            all_vals = np.concatenate([m_raw[m_raw > 0], m_cg[m_cg > 0]])
            if len(all_vals) > 0:
                ax.set_ylim(bottom=max(all_vals.min() * 0.5, 1e-30))
        ax.legend(title="Dataset", fontsize=8, loc="best")

    axes[3, 0].set_xlabel(r"$y \ [\mathrm{pc}]$", fontsize=12)
    axes[3, 1].set_xlabel(r"$y \ [\mathrm{pc}]$", fontsize=12)

    fig.suptitle(
        "Mean Vertical Profiles vs y for HR and CG HR\n"
        r"(Time-averaged over last 500 snapshots, showing mean $\pm\ 1\sigma$ over time)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()

    for ext in ("png", "pdf"):
        out_file = OUT_ROOT / f"hr_mpi_all_profiles_comparison.{ext}"
        fig.savefig(out_file, dpi=200, bbox_inches="tight")
        print(f"  Saved 8-panel figure -> {out_file}")

    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PLOTTING: INDIVIDUAL FIELD COMPARISON FIGURES
    # ══════════════════════════════════════════════════════════════════════════

    print("\nPlotting individual field comparison figures …")
    for cfg in FIELD_CONFIGS:
        key      = cfg["key"]
        ylabel   = cfg["ylabel"]
        title    = cfg["title"]
        filename = cfg["filename"]
        is_log   = (cfg["yscale"] == "log")

        fig_single, ax_single = plt.subplots(figsize=(8, 5.5))

        for idx, sim in enumerate(simulations):
            name      = sim["name"]
            res       = profile_results[name]
            y_pc_raw  = res["y_pc_raw"]
            y_pc_cg   = res["y_pc_cg"]
            m_raw     = res[f"{key}_raw_mean"]
            s_raw     = res[f"{key}_raw_std"]
            m_cg      = res[f"{key}_cg_mean"]
            s_cg      = res[f"{key}_cg_std"]

            label_raw = res["label_raw"]
            label_cg  = res["label_cg"]

            if key == "cooling":
                int_raw = np.trapezoid(m_raw, y_pc_raw)
                int_cg  = np.trapezoid(m_cg, y_pc_cg)
                label_raw = rf"{label_raw} ($\Sigma_c = {int_raw:.2e}$)"
                label_cg  = rf"{label_cg} ($\Sigma_c = {int_cg:.2e}$)"

            ax_single.plot(y_pc_raw, m_raw, lw=2, ls="-", color="tab:blue", label=label_raw)
            if is_log:
                ax_single.fill_between(
                    y_pc_raw,
                    np.clip(m_raw - s_raw, 1e-30, None),
                    m_raw + s_raw,
                    color="tab:blue",
                    alpha=0.2,
                    linewidth=0,
                )
            else:
                ax_single.fill_between(
                    y_pc_raw,
                    m_raw - s_raw,
                    m_raw + s_raw,
                    color="tab:blue",
                    alpha=0.2,
                    linewidth=0,
                )

            ax_single.plot(y_pc_cg, m_cg, lw=2, ls="-.", marker="^", markersize=5, color="tab:orange", label=label_cg)
            if is_log:
                ax_single.fill_between(
                    y_pc_cg,
                    np.clip(m_cg - s_cg, 1e-30, None),
                    m_cg + s_cg,
                    color="tab:orange",
                    alpha=0.25,
                    linewidth=0,
                )
            else:
                ax_single.fill_between(
                    y_pc_cg,
                    m_cg - s_cg,
                    m_cg + s_cg,
                    color="tab:orange",
                    alpha=0.25,
                    linewidth=0,
                )

        ax_single.set_xlabel(r"$y \ [\mathrm{pc}]$", fontsize=12)
        ax_single.set_ylabel(ylabel, fontsize=12)
        ax_single.set_title(
            f"{title}\n"
            r"(Time-averaged over last 500 snapshots, showing mean $\pm\ 1\sigma$ across time)",
            fontsize=13,
        )
        ax_single.grid(True, which="both" if is_log else "major", ls="--", alpha=0.4)
        if is_log:
            ax_single.set_yscale("log")
            all_vals = np.concatenate([m_raw[m_raw > 0], m_cg[m_cg > 0]])
            if len(all_vals) > 0:
                ax_single.set_ylim(bottom=max(all_vals.min() * 0.5, 1e-30))
        ax_single.legend(title="Dataset", fontsize=10, loc="best")
        fig_single.tight_layout()

        for ext in ("png", "pdf"):
            out_file = OUT_ROOT / f"{filename}_comparison.{ext}"
            fig_single.savefig(out_file, dpi=200, bbox_inches="tight")
            print(f"  Saved {out_file}")

        plt.close(fig_single)

    print("\nAll done! All profile plots successfully generated.")


if __name__ == "__main__":
    main()

