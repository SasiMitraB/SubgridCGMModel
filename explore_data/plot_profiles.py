"""
plot_profiles.py (formerly cooling_rate_animation.py)
─────────────────────────────────────────────────────────────────────────────
Computes and plots y-profiles for all four HR MPI simulations:
  - Log Number Density: log10(n_H)    [log10(cm⁻³)]
  - Log Temperature:  log10(T)      [log10(K)]
  - Cooling rate:     normalized <n_H^2 Lambda(T)> / (p_0 / t_0) [dimensionless]
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

# ── Resolution list ─────────────────────────────────────────────────────────
RESOLUTIONS = [
    (["8x16", "16x8"],          r"$16 \times 8$"),
    (["16x32", "32x16"],        r"$32 \times 16$"),
    (["32x64", "64x32"],        r"$64 \times 32$"),
    (["64x128", "128x64"],      r"$128 \times 64$"),
    (["128x256", "256x128"],    r"$256 \times 128$"),
    (["256x512", "512x256"],    r"$512 \times 256$"),
    (["512x1024", "1024x512"],  r"$1024 \times 512$"),
]


# ── Simulation configurations (Auto-discover GPU runs, fallback to MPI/build) ─
simulations = []
seen_folders = set()
for res_aliases, res_label in RESOLUTIONS:
    found = False
    for res_tag in res_aliases:
        if found:
            break
        for prefix in ["hr_gpu", "hr_mpi", "hr_build", "subgrid", "lr"]:
            datafolder = SIM_ROOT / f"{prefix}_{res_tag}"
            if not datafolder.is_dir() and prefix == "hr_build":
                alt = SIM_ROOT / f"{prefix}_{res_tag.split('x')[0]}"
                if alt.is_dir():
                    datafolder = alt
            if datafolder.is_dir() and str(datafolder) not in seen_folders:
                athinp = datafolder / f"kh_radiative_{res_tag}.athinput"
                if not athinp.is_file():
                    matches = list(datafolder.glob("*.athinput"))
                    if matches:
                        athinp = matches[0]
                if athinp.is_file():
                    simulations.append({
                        "name":       f"{prefix}_{res_tag}",
                        "label":      res_label,
                        "athinp":     str(athinp),
                        "datafolder": str(datafolder),
                        "downsample": 32,
                    })
                    seen_folders.add(str(datafolder))
                    found = True
                    break

# ── Physical constants ────────────────────────────────────────────────────────
CM_PER_PC       = 3.08568e18          # cm per parsec
CM_PER_KM       = 1.0e5               # cm per km
SECONDS_PER_MYR = 3.15576e13          # seconds per Myr
M_H             = 1.6726219e-24       # proton mass [g]
MU              = 0.62                # mean molecular weight
GAMMA           = 5.0 / 3.0           # adiabatic index

# ── Active-temperature mask bounds for cooling ────────────────────────────────
LOGT_ACTIVE_START = 4.1 # 10^4.1 ~ 1.26e4 K
LOGT_ACTIVE_END   = 5.9 # 10^5.9 ~ 7.94e5 K
# LOGT_ACTIVE_START = np.log10(1.05e4)
# LOGT_ACTIVE_END = np.log10(0.95e6)

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


# Characteristic intermediate temperature T_mid = sqrt(10^4 * 10^6) = 10^5 K
T_MID = float(np.sqrt(1.0e4 * 1.0e6))
LAMBDA_T_MID = float(lambda_cool(T_MID, mask=False))


# ── Coarse-graining helper ────────────────────────────────────────────────────

def coarse_grain_2d(arr: np.ndarray, ds: int = 32) -> np.ndarray:
    """Coarse-grain a 2D array of shape (ny, nx) by factor ds."""
    ny, nx = arr.shape
    if ds <= 1 or ny < ds or nx < ds:
        return arr.copy()
    ny_cg = ny // ds
    nx_cg = nx // ds
    return arr[:ny_cg * ds, :nx_cg * ds].reshape(ny_cg, ds, nx_cg, ds).mean(axis=(1, 3))


def compute_cooling_normalization(athinp_path: str | Path, units: ergane.Units | None = None) -> tuple[float, float, float]:
    """
    Compute equilibrium pressure p_0, cooling time t_0 at geometric mean temperature T_0,
    and normalization scale p_0 / t_0 in CGS.

    p_0: Initial equilibrium pressure in CGS [dyn cm⁻²] from athinput 'press'.
    t_0: Cooling time at geometric mean temperature T_0 = sqrt(T_cold * T_hot) [s],
         t_0 = p_0 / ((gamma - 1) * n_0^2 * Lambda(T_0)).
    norm: p_0 / t_0 = (gamma - 1) * n_0^2 * Lambda(T_0) [erg s⁻¹ cm⁻³].
    """
    params = ergane.parse_athinput(athinp_path)

    # AthenaK code unit conversions to CGS
    if units is not None and getattr(units, "pressure", None) is not None:
        p_unit = float(units.pressure)
        rho_unit = float(units.density)
    else:
        length_cgs = float(params.get("units", {}).get("length_cgs", CM_PER_PC))
        time_cgs   = float(params.get("units", {}).get("time_cgs",   SECONDS_PER_MYR))
        mass_cgs   = float(params.get("units", {}).get("mass_cgs",   4.91417e31))
        rho_unit   = mass_cgs / (length_cgs ** 3)
        v_unit     = length_cgs / time_cgs
        p_unit     = rho_unit * (v_unit ** 2)

    press_code    = float(params.get("problem", {}).get("press", 14.02645))
    rho_cold_code = float(params.get("problem", {}).get("rho_cold", 0.1))
    rho_hot_code  = float(params.get("problem", {}).get("rho_hot", 0.001))
    gamma         = float(params.get("hydro", {}).get("gamma", GAMMA))

    # CGS initial equilibrium pressure
    p0_cgs = press_code * p_unit

    # Geometric mean density at pressure equilibrium: rho_0 = sqrt(rho_cold * rho_hot)
    rho0_cgs = np.sqrt(rho_cold_code * rho_hot_code) * rho_unit
    n0_cgs   = rho0_cgs / (MU * M_H)

    # Cooling rate at geometric mean temperature T_0:
    # Lambda(T_0) evaluated at T_MID = sqrt(10^4 * 10^6) = 10^5 K
    lam_t0 = LAMBDA_T_MID

    # Cooling timescale: t_0 = p_0 / ((gamma - 1) * n_0^2 * Lambda(T_0))
    t0_cgs = p0_cgs / ((gamma - 1.0) * (n0_cgs ** 2) * lam_t0)
    p0_over_t0 = p0_cgs / t0_cgs

    return p0_cgs, t0_cgs, p0_over_t0


# ── Field extractors ─────────────────────────────────────────────────────────

def compute_physical_fields(frame: ergane.Frame, p0_over_t0: float = 1.0) -> dict[str, np.ndarray]:
    """
    Extract physical fields for a single snapshot:
      - log10_number_density: log10(n_H) [log10(cm⁻³)]
      - log10_temperature:    log10(T)   [log10(K)]
      - cooling:              n_H² Λ(T) / (p_0 / t_0) [dimensionless]
      - pressure:             P          [dyn cm⁻²]
      - velx:                 v_x        [km s⁻¹]
      - vely:                 v_y        [km s⁻¹]
      - flux_x:               n_H v_x    [cm⁻² s⁻¹]
      - flux_y:               n_H v_y    [cm⁻² s⁻¹]
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

    # Normalized cooling rate: n_H² Λ(T) / (p_0 / t_0)
    lam    = lambda_cool(temp_K, mask=True) # [erg cm³ s⁻¹]
    q_cool_norm = ((n_H ** 2) * lam) / p0_over_t0

    # Dimensionless cooling time ratio:
    # t_cool / t_0 = lambda(T_0) / lambda(T), where T_0 = sqrt(10^4 * 10^6)
    with np.errstate(divide="ignore", invalid="ignore"):
        tcool_ratio = np.where(lam > 0, LAMBDA_T_MID / lam, np.nan)

    # Fluxes using number density in CGS: n_H * v (with v in cm/s) -> [cm⁻² s⁻¹]
    flux_x = n_H * (vx_kms * CM_PER_KM) # [cm⁻² s⁻¹]
    flux_y = n_H * (vy_kms * CM_PER_KM) # [cm⁻² s⁻¹]

    return {
        "log10_number_density": log10_nH,
        "log10_temperature":    log10_T,
        "cooling":              q_cool_norm,
        "tcool_ratio":          tcool_ratio,
        "pressure":             P_cgs,
        "velx":                 vx_kms,
        "vely":                 vy_kms,
        "flux_x":               flux_x,
        "flux_y":               flux_y,
    }


def compute_coarse_grained_fields(frame: ergane.Frame, ds: int = 32, p0_over_t0: float = 1.0) -> dict[str, np.ndarray]:
    """
    Compute coarse-grained versions of physical fields matching mock_sg.py:
      - Primitive fields (rho, P, T, vx, vy) are coarse-grained by factor ds.
      - Cooling rate emis_cg is the coarse-grained fine normalized cooling rate.
      - Fluxes are coarse-grained fine fluxes.
    """
    rho_cgs = frame.density
    P_cgs   = frame.pressure
    temp_K  = frame.temperature
    vx_kms  = frame.velx
    vy_kms  = frame.vely

    n_H = rho_cgs / (MU * M_H)
    lam = lambda_cool(temp_K, mask=True)
    q_cool_norm = ((n_H ** 2) * lam) / p0_over_t0
    with np.errstate(divide="ignore", invalid="ignore"):
        tcool_ratio = np.where(lam > 0, LAMBDA_T_MID / lam, np.nan)
    flux_x = n_H * (vx_kms * CM_PER_KM)
    flux_y = n_H * (vy_kms * CM_PER_KM)

    ny, nx = rho_cgs.shape
    if ds <= 1 or ny < ds or nx < ds:
        log10_nH = np.log10(np.maximum(n_H, 1e-30))
        log10_T  = np.log10(np.maximum(temp_K, 1.0))
        return {
            "log10_number_density": log10_nH,
            "log10_temperature":    log10_T,
            "cooling":              q_cool_norm,
            "tcool_ratio":          tcool_ratio,
            "pressure":             P_cgs,
            "velx":                 vx_kms,
            "vely":                 vy_kms,
            "flux_x":               flux_x,
            "flux_y":               flux_y,
        }

    # Coarse grain primitives and quantities
    rho_cg  = coarse_grain_2d(rho_cgs, ds)
    n_H_cg  = rho_cg / (MU * M_H)
    temp_cg = coarse_grain_2d(temp_K, ds)
    P_cg    = coarse_grain_2d(P_cgs, ds)
    vx_cg   = coarse_grain_2d(vx_kms, ds)
    vy_cg   = coarse_grain_2d(vy_kms, ds)

    # Cooling rate: coarse-grained fine normalized emissivity
    q_cool_cg      = coarse_grain_2d(q_cool_norm, ds)
    tcool_ratio_cg = coarse_grain_2d(tcool_ratio, ds)
    flux_x_cg      = coarse_grain_2d(flux_x, ds)
    flux_y_cg      = coarse_grain_2d(flux_y, ds)

    log10_nH_cg = np.log10(np.maximum(n_H_cg, 1e-30))
    log10_T_cg  = np.log10(np.maximum(temp_cg, 1.0))

    return {
        "log10_number_density": log10_nH_cg,
        "log10_temperature":    log10_T_cg,
        "cooling":              q_cool_cg,
        "tcool_ratio":          tcool_ratio_cg,
        "pressure":             P_cg,
        "velx":                 vx_cg,
        "vely":                 vy_cg,
        "flux_x":               flux_x_cg,
        "flux_y":               flux_y_cg,
    }


def x_average_profile(frame: ergane.Frame, values: np.ndarray) -> np.ndarray:
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


def get_y_coords_pc(frame: ergane.Frame) -> np.ndarray:
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
        "title":     r"Normalized Cooling Rate $\langle n^2 \Lambda(T) \rangle / (p_0 / t_0)$ vs $y$",
        "ylabel":    r"$\langle n^2 \Lambda(T) \rangle / (p_0 / t_0)$",
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
    if not simulations:
        print(f"No simulation output directories found under {SIM_ROOT}.")
        return

    profile_results = {}
    valid_simulations = []

    for sim in simulations:
        name  = sim["name"]
        label = sim["label"]
        ds    = sim.get("downsample", 32)
        print(f"\n{'='*60}")
        print(f"  Computing profiles for {name} ({label}, downsample={ds})")
        print(f"{'='*60}")

        try:
            sim_data = ergane.SimulationData(
                athinp=str(sim["athinp"]),
                datafolder=str(sim["datafolder"]),
            )

            p0_cgs, t0_cgs, p0_over_t0 = compute_cooling_normalization(sim["athinp"], sim_data.units)
            print(f"  Cooling normalization: p0={p0_cgs:.3e} dyn/cm^2, t0={t0_cgs:.3e} s ({t0_cgs / SECONDS_PER_MYR:.4f} Myr), p0/t0={p0_over_t0:.3e} erg s^-1 cm^-3")

            n_frames   = sim_data.n_frames
            frame_nums = sim_data.frame_numbers
            if n_frames == 0:
                print(f"  No frames found for {name}, skipping.")
                continue

            print(f"  {n_frames} frames available (#{frame_nums[0]}–#{frame_nums[-1]})")

            # Use last 500 snapshots
            n_avg       = min(250, n_frames)
            avg_indices = frame_nums[-n_avg:]
            print(f"  Time-averaging over last {n_avg} snapshots …")

            # Pre-read grid
            frame0 = sim_data.get_frame(frame_nums[0])
            y_pc_raw = get_y_coords_pc(frame0)
            ny_raw   = y_pc_raw.size
            nx_raw   = frame0.xc.size

            # Storage for all fields: field_name -> array of shape (n_avg, ny)
            field_stacks_raw = {cfg["key"]: np.zeros((n_avg, ny_raw), dtype=np.float64) for cfg in FIELD_CONFIGS}

            for idx, fn in enumerate(tqdm(avg_indices, desc=f"  [{name}] Snapshots", unit="frame")):
                f = sim_data.get_frame(fn)
                fields_raw = compute_physical_fields(f, p0_over_t0=p0_over_t0)

                for key in FIELD_CONFIGS:
                    k = key["key"]
                    field_stacks_raw[k][idx] = x_average_profile(f, fields_raw[k])

                del f
                if idx % 100 == 0:
                    gc.collect()

            # Compute mean and standard deviation over time
            sim_summary = {
                "y_pc":  y_pc_raw,
                "label": label,
                "name":  name,
                "ny":    ny_raw,
                "nx":    nx_raw,
            }
            for key in FIELD_CONFIGS:
                k = key["key"]
                with np.errstate(all="ignore"):
                    sim_summary[f"{k}_mean"] = np.nanmean(field_stacks_raw[k], axis=0)
                    sim_summary[f"{k}_std"]  = np.nanstd(field_stacks_raw[k],  axis=0)

            profile_results[name] = sim_summary
            valid_simulations.append(sim)
            print(f"  Done {name} (ny={ny_raw}, nx={nx_raw}).")
        except Exception as e:
            print(f"  ERROR processing {name}: {e}")

    if not valid_simulations:
        print("No valid simulation data could be processed.")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # PLOTTING: CONSOLIDATED 8-PANEL COMPARISON FIGURE
    # ══════════════════════════════════════════════════════════════════════════

    print("\nPlotting consolidated 8-panel profile comparison figure …")

    palette = plt.cm.tab10.colors
    nrows, ncols = 4, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 16), sharex=True)
    axes_flat = axes.flatten()

    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))

    for idx_field, cfg in enumerate(FIELD_CONFIGS):
        ax     = axes_flat[idx_field]
        key    = cfg["key"]
        ylabel = cfg["ylabel"]
        title  = cfg["title"]
        is_log = (cfg["yscale"] == "log")

        for idx, sim in enumerate(valid_simulations):
            name  = sim["name"]
            res   = profile_results[name]
            y_pc  = res["y_pc"]
            m     = res[f"{key}_mean"]
            s     = res[f"{key}_std"]
            label = res["label"]
            color = palette[idx % len(palette)]

            if key == "cooling" and trapz_fn is not None:
                int_val = trapz_fn(m, y_pc)
                label = rf"{label} ($\Sigma_c = {int_val:.2e}$)"

            ax.plot(y_pc, m, lw=2, ls="-", color=color, label=label)
            if is_log:
                ax.fill_between(
                    y_pc,
                    np.clip(m - s, 1e-30, None),
                    m + s,
                    color=color,
                    alpha=0.2,
                    linewidth=0,
                )
            else:
                ax.fill_between(
                    y_pc,
                    m - s,
                    m + s,
                    color=color,
                    alpha=0.2,
                    linewidth=0,
                )

        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, which="both" if is_log else "major", ls="--", alpha=0.4)
        if is_log:
            ax.set_yscale("log")
            all_m = [profile_results[s["name"]][f"{key}_mean"] for s in valid_simulations]
            all_vals = np.concatenate([v[np.isfinite(v) & (v > 0)] for v in all_m if len(v[np.isfinite(v) & (v > 0)]) > 0], axis=0) if all_m else []
            if len(all_vals) > 0:
                ax.set_ylim(bottom=max(all_vals.min() * 0.5, 1e-30))
        ax.legend(title="Resolution", fontsize=9, loc="best")

    # Hide any unused axes if number of fields < total subplot panels
    for ax in axes_flat[len(FIELD_CONFIGS):]:
        ax.set_visible(False)

    for col in range(ncols):
        axes[nrows - 1, col].set_xlabel(r"$y \ [\mathrm{pc}]$", fontsize=12)

    fig.suptitle(
        "Mean Vertical Profiles vs y across Resolutions\n"
        r"(Time-averaged over last 500 snapshots, showing mean $\pm\ 1\sigma$ over time)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()

    for ext in ("png", "pdf"):
        out_file = OUT_ROOT / f"hr_resolution_sweep_profiles.{ext}"
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

        for idx, sim in enumerate(valid_simulations):
            name  = sim["name"]
            res   = profile_results[name]
            y_pc  = res["y_pc"]
            m     = res[f"{key}_mean"]
            s     = res[f"{key}_std"]
            label = res["label"]
            color = palette[idx % len(palette)]

            if key == "cooling" and trapz_fn is not None:
                int_val = trapz_fn(m, y_pc)
                label = rf"{label} ($\Sigma_c = {int_val:.2e}$)"

            ax_single.plot(y_pc, m, lw=2, ls="-", color=color, label=label)
            if is_log:
                ax_single.fill_between(
                    y_pc,
                    np.clip(m - s, 1e-30, None),
                    m + s,
                    color=color,
                    alpha=0.2,
                    linewidth=0,
                )
            else:
                ax_single.fill_between(
                    y_pc,
                    m - s,
                    m + s,
                    color=color,
                    alpha=0.2,
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
            all_m = [profile_results[s["name"]][f"{key}_mean"] for s in valid_simulations]
            all_vals = np.concatenate([v[np.isfinite(v) & (v > 0)] for v in all_m if len(v[np.isfinite(v) & (v > 0)]) > 0], axis=0) if all_m else []
            if len(all_vals) > 0:
                ax_single.set_ylim(bottom=max(all_vals.min() * 0.5, 1e-30))
            # if key == "cooling":
            #     ax_single.set_ylim(bottom=1e-26)
        ax_single.legend(title="Resolution", fontsize=10, loc="best")
        fig_single.tight_layout()

        for ext in ("png", "pdf"):
            out_file = OUT_ROOT / f"{filename}_comparison.{ext}"
            fig_single.savefig(out_file, dpi=200, bbox_inches="tight")
            print(f"  Saved {out_file}")

        plt.close(fig_single)

    print("\nAll done! All profile plots successfully generated.")


if __name__ == "__main__":
    main()
