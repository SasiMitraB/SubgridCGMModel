"""
check_quasi_steady_state.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compute quasi-steady-state (QSS) diagnostics for every simulation in
`simulation_outputs/` and save results to `outputs/qss_diagnostics/`.

Diagnostics (from Fielding+ / Lecoanet+ paper):
  1. Time criterion      — has the simulation run past t0 = t_cool(T0)?
  2. Mass flux           — is <rho * vy>(y) flat (constant in y)?
  3. Density profile     — is <rho>(y, t) stationary in the TRML frame?
  4. Momentum flux       — is <rho*vy^2> + <p> constant in y?
  5. Temperature profile — does <T>(y) fit a tanh with stable z0?
  6. Temperature PDF     — is Pv(T) stable over time?
  7. Energy balance      — does d/dy<rho*vy*B> balance -<n^2 * Lambda(T)>?

RAM strategy
------------
* Frames are loaded ONE AT A TIME and immediately discarded.
* Only 1D horizontal-averages (shape = ny) are kept in memory, never
  the full 2D arrays.
* A configurable stride (FRAME_STRIDE) selects every Nth frame so we
  don't process all 2000 frames for 7 simulations.

Usage
-----
    python explore_data/check_quasi_steady_state.py

Output files (per simulation) in `outputs/qss_diagnostics/<sim_name>/`:
    summary.txt          — human-readable QSS verdict
    diagnostics.npz      — numpy archive of all computed profiles/arrays
    plots/               — one PNG per diagnostic
"""

from __future__ import annotations

import os
import sys
import gc
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for scripts
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import wasserstein_distance

# ── Project root on sys.path so we can `import ergane` ──────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ergane.bin_reader import read_binary, make_2D_array
from ergane.athinput_parser import parse_athinput

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SIM_OUTPUT_ROOT = ROOT / "simulation_outputs"
ATHINPUT_ROOT   = ROOT / "builds" / "hr_build" / "src"
RESULTS_ROOT    = ROOT / "outputs" / "qss_diagnostics"

# Process every Nth frame to save RAM and time.
# 10 → ~200 frames from a 2000-frame run, plenty for diagnostics.
FRAME_STRIDE = 5

# Number of PDF snapshots to compare for stability (evenly spaced after t0)
N_PDF_SNAPS = 5

# Temperature PDF bins
N_TEMP_BINS = 100

# Adiabatic index (same for all sims from athinput)
GAMMA = 5.0 / 3.0   # ≈ 1.6667

# Mean molecular weight (mu) for ISM cooling — from athinput units section
MU_DEFAULT = 0.62   # proton masses per particle

# ISM Cooling function Lambda(T) — translated from AthenaK C++ ISMCoolFn.
# T is in physical Kelvin. Returns Λ(T) in erg cm^3 / s.
# The `mask` flag zeros Λ outside log10(T) ∈ [4.5, 5.5] (the TRML regime).
def lambda_cool(temp, mask=False):
    """
    Cooling function ISMCoolFn translated from AthenaK C++.
    Works on scalars or numpy arrays (any shape).
    Returns Λ(T) in erg cm^3 / s.
    """
    temp = np.asarray(temp, dtype=float)
    logt = np.log10(np.maximum(temp, 1.0))  # guard against log(0)

    lhd = np.array(
        [
            -22.5977, -21.9689, -21.5972, -21.4615, -21.4789,
            -21.5497, -21.6211, -21.6595, -21.6426, -21.5688,
            -21.4771, -21.3755, -21.2693, -21.1644, -21.0658,
            -20.9778, -20.8986, -20.8281, -20.7700, -20.7223,
            -20.6888, -20.6739, -20.6815, -20.7051, -20.7229,
            -20.7208, -20.7058, -20.6896, -20.6797, -20.6749,
            -20.6709, -20.6748, -20.7089, -20.8031, -20.9647,
            -21.1482, -21.2932, -21.3767, -21.4129, -21.4291,
            -21.4538, -21.5055, -21.5740, -21.6300, -21.6615,
            -21.6766, -21.6886, -21.7073, -21.7304, -21.7491,
            -21.7607, -21.7701, -21.7877, -21.8243, -21.8875,
            -21.9738, -22.0671, -22.1537, -22.2265, -22.2821,
            -22.3213, -22.3462, -22.3587, -22.3622, -22.3590,
            -22.3512, -22.3420, -22.3342, -22.3312, -22.3346,
            -22.3445, -22.3595, -22.3780, -22.4007, -22.4289,
            -22.4625, -22.4995, -22.5353, -22.5659, -22.5895,
            -22.6059, -22.6161, -22.6208, -22.6213, -22.6184,
            -22.6126, -22.6045, -22.5945, -22.5831, -22.5707,
            -22.5573, -22.5434, -22.5287, -22.5140, -22.4992,
            -22.4844, -22.4695, -22.4543, -22.4392, -22.4237,
            -22.4087, -22.3928,
        ]
    )

    lam = np.zeros_like(temp, dtype=float)

    # No cooling below 10^4 K
    mask_off = logt <= 4.0
    lam[mask_off] = 0.0

    # KI02 regime: 4.0 < log10(T) <= 4.2
    mask_ki = (logt > 4.0) & (logt <= 4.2)
    if np.any(mask_ki):
        lam[mask_ki] = (
            2.0e-19 * np.exp(-1.184e5 / (temp[mask_ki] + 1.0e3))
            + 2.8e-28 * np.sqrt(temp[mask_ki]) * np.exp(-92.0 / temp[mask_ki])
        )

    # CGOLS fit: log10(T) > 8.15
    mask_hi = logt > 8.15
    lam[mask_hi] = 10.0 ** (0.45 * logt[mask_hi] - 26.065)

    # SPEX table interpolation: 4.2 < log10(T) <= 8.15
    mask_mid = (logt > 4.2) & (logt <= 8.15)
    if np.any(mask_mid):
        ipps = (25.0 * logt[mask_mid] - 103).astype(int)
        ipps = np.clip(ipps, 0, 100)
        x0 = 4.12 + 0.04 * ipps
        dx = logt[mask_mid] - x0
        logcool = (lhd[ipps + 1] * dx - lhd[ipps] * (dx - 0.04)) * 25.0
        lam[mask_mid] = 10.0 ** logcool

    if mask:
        # Restrict to TRML window: 10^4.5 < T < 10^5.5
        mask_trml = (logt < 4.5) | (logt > 5.5)
        lam[mask_trml] = 0.0

    return lam


def t_cool_code(p0, rho0, T0_kelvin, gamma, mu, length_cgs, time_cgs, mass_cgs):
    """
    Cooling time at the geometric-mean temperature T0 (in Kelvin) in code units.

    Converts code-unit pressure and density to CGS, computes the cooling timescale
    t0_cgs = p_cgs / ((gamma - 1) * n_cgs^2 * Lambda(T0)), and converts back to
    code time units by dividing by time_cgs.
    """
    m_H_cgs = 1.6726e-24
    rho_unit_cgs = mass_cgs / length_cgs**3
    v_unit_cgs = length_cgs / time_cgs
    pressure_unit_cgs = rho_unit_cgs * v_unit_cgs**2

    rho0_cgs = rho0 * rho_unit_cgs
    p0_cgs   = p0   * pressure_unit_cgs
    n0_cgs   = rho0_cgs / (mu * m_H_cgs)

    lam_T0 = lambda_cool(np.array([T0_kelvin]))[0]  # erg cm^3 / s
    if lam_T0 == 0:
        return np.inf
    t0_cgs = p0_cgs / ((gamma - 1.0) * n0_cgs**2 * lam_T0)
    return t0_cgs / time_cgs


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_2d(filedata, var: str) -> np.ndarray:
    """Assemble a sorted 2-D array (ny, nx) from mesh-block data."""
    return make_2D_array(filedata, var)


def horiz_mean(arr2d: np.ndarray) -> np.ndarray:
    """Horizontal (x-direction, axis=1) average → shape (ny,)."""
    return arr2d.mean(axis=1)


def get_prim_from_cons(filedata) -> dict[str, np.ndarray]:
    """
    Convert conserved hydro_u variables to primitives.

    AthenaK conserved hydro_u stores:
        dens  — mass density  ρ
        mom1  — x-momentum   ρ*vx
        mom2  — y-momentum   ρ*vy
        mom3  — z-momentum   (for 2-D sims, always 0)
        eint  — total energy  E = ½ρv² + p/(γ-1)

    Returns dict with keys: rho, vx, vy, press, T
    All arrays are 2-D (ny, nx).
    """
    rho  = make_2d(filedata, "dens")
    mom1 = make_2d(filedata, "mom1")
    mom2 = make_2d(filedata, "mom2")
    etot = make_2d(filedata, "ener")  # AthenaK names total energy 'ener' in hydro_u

    vx = mom1 / (rho + 1e-30)
    vy = mom2 / (rho + 1e-30)
    KE = 0.5 * rho * (vx**2 + vy**2)
    p  = (GAMMA - 1.0) * (etot - KE)
    p  = np.maximum(p, 1e-30)   # floor for numerical safety
    T  = p / (rho + 1e-30)      # code-unit "temperature" ∝ p/rho

    return {"rho": rho, "vx": vx, "vy": vy, "press": p, "T": T}


def read_time_from_header(path: Path) -> float:
    """Read only the time from a bin file header — very fast, no data loaded."""
    with open(path, "rb") as fp:
        fp.readline()  # "Athena version=1.1"
        n = int(fp.readline().split(b"=")[-1])
        pheader = {}
        for _ in range(n - 1):
            k, v = fp.readline().decode().split("=")
            pheader[k.strip()] = v.strip()
    return float(pheader.get("time", 0.0))


def tanh_profile(y, A, B, y0, z0):
    """Tanh temperature model: T(y) = A * tanh((y - y0) / z0) + B."""
    return A * np.tanh((y - y0) / z0) + B


def discover_athinput(sim_name: str) -> Path | None:
    """Try to find a matching athinput for a simulation folder name."""
    # Direct match: kh_radiative_512 → kh_radiative_512.athinput
    candidate = ATHINPUT_ROOT / f"{sim_name}.athinput"
    if candidate.exists():
        return candidate
    # Strip trailing suffix for variants like kh_radiative_512_mu1
    for p in sorted(ATHINPUT_ROOT.glob("*.athinput")):
        stem = p.stem  # e.g. kh_radiative_512
        if sim_name.startswith(stem):
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main diagnostic engine
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnostics(sim_dir: Path, out_dir: Path) -> dict:
    """
    Run all QSS diagnostics for a single simulation directory.

    Returns a results dict (also saved to `out_dir`).
    """
    sim_name = sim_dir.name
    print(f"\n{'='*60}")
    print(f"  Simulation: {sim_name}")
    print(f"{'='*60}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    # ── 1. Discover bin files ─────────────────────────────────────────────────
    bin_dir = sim_dir / "bin"
    if not bin_dir.is_dir():
        print(f"  [SKIP] No bin/ directory found in {sim_dir}")
        return {}

    bin_files = sorted(bin_dir.glob("*.hydro_u.*.bin"))
    if not bin_files:
        # Fallback: hydro_w
        bin_files = sorted(bin_dir.glob("*.hydro_w.*.bin"))
    if not bin_files:
        print(f"  [SKIP] No recognisable bin files in {bin_dir}")
        return {}

    print(f"  Found {len(bin_files)} bin frames")

    # ── 2. Load athinput parameters ───────────────────────────────────────────
    athinp_path = discover_athinput(sim_name)
    params = parse_athinput(athinp_path) if (athinp_path and athinp_path.exists()) else {}

    gamma = float(params.get("hydro", {}).get("gamma", GAMMA))
    rho_cold = float(params.get("problem", {}).get("rho_cold", 0.1))
    rho_hot  = float(params.get("problem", {}).get("rho_hot",  0.001))
    press    = float(params.get("problem", {}).get("press",    13.925))
    mu       = float(params.get("units",   {}).get("mu", MU_DEFAULT))

    # Code-unit temperatures ∝ p / rho
    T_cold = press / rho_cold
    T_hot  = press / rho_hot
    T0     = np.sqrt(T_cold * T_hot)   # geometric mean (code units)

    # Code-unit → Kelvin conversion factor
    # T [K] = T_code * (mu * m_H / k_B) * v_unit^2
    # where v_unit = length_cgs / time_cgs
    length_cgs = float(params.get("units", {}).get("length_cgs", 3.08568e18))
    time_cgs   = float(params.get("units", {}).get("time_cgs",   3.15576e13))
    mass_cgs   = float(params.get("units", {}).get("mass_cgs",   4.91417e31))
    m_H_cgs    = 1.6726e-24   # g
    k_B_cgs    = 1.3807e-16   # erg / K
    v_unit     = length_cgs / time_cgs          # cm / s
    T_to_K     = mu * m_H_cgs / k_B_cgs * v_unit**2   # K per code-unit temperature
    T0_kelvin  = T0 * T_to_K

    rho_unit_cgs      = mass_cgs / length_cgs**3
    v_unit_cgs        = length_cgs / time_cgs
    pressure_unit_cgs = rho_unit_cgs * v_unit_cgs**2

    # ── 3. Build time index (fast — headers only) ─────────────────────────────
    print("  Building time index…", end=" ", flush=True)
    times_all = np.array([read_time_from_header(f) for f in bin_files])
    print(f"done  ({times_all[0]:.3f} → {times_all[-1]:.3f})")

    # Effective cooling time in code units
    # t0 ~ p / [(gamma-1) * rho_mix^2 * Lambda(T0)] in code units
    # Use geometric-mean density as the reference
    rho0 = np.sqrt(rho_cold * rho_hot)
    t0   = t_cool_code(press, rho0, T0_kelvin, gamma, mu, length_cgs, time_cgs, mass_cgs)

    # Clamp t0 to the simulation time range as a sanity check
    # If t0 >> t_final, the sim may still be approaching QSS
    # For a rough estimate: t0 ~ tlim / 3 is typical in these KH runs
    # Fall back to a fraction of t_final if formula gives nonsense
    if not np.isfinite(t0) or t0 > 10 * times_all[-1]:
        t0 = times_all[-1] / 3.0  # heuristic: expect QSS after ~1/3 of run
        print(f"  [WARN] Cooling time formula gave unphysical t0; using heuristic t0 = t_final/3 = {t0:.4f}")

    chi = rho_cold / rho_hot   # density contrast

    print(f"  T_cold={T_cold:.2f}  T_hot={T_hot:.2f}  T0={T0:.2f}")
    print(f"  t0 (cooling time) = {t0:.4f} [code units]")
    print(f"  chi (rho_cold/rho_hot) = {chi:.1f}")

    t_final = times_all[-1]
    t0_reached = t_final >= t0
    print(f"  Time criterion: t_final={t_final:.3f}, t0={t0:.4f} → {'✓ PASSED' if t0_reached else '✗ NOT REACHED'}")


    # Select strided subset of frames
    stride_idx   = np.arange(0, len(bin_files), FRAME_STRIDE)
    bin_subset   = [bin_files[i] for i in stride_idx]
    times_subset = times_all[stride_idx]

    # Frames after t0
    post_t0_mask = times_subset >= t0
    post_t0_idx  = np.where(post_t0_mask)[0]

    print(f"  Processing {len(bin_subset)} frames (stride={FRAME_STRIDE}), "
          f"{post_t0_mask.sum()} are post-t0")

    # ── 4. Read first frame to get grid geometry ───────────────────────────────
    fd0   = read_binary(str(bin_subset[0]))
    Nx1   = fd0["Nx1"]
    Nx2   = fd0["Nx2"]
    x2f   = np.linspace(fd0["x2min"], fd0["x2max"], Nx2 + 1)
    y_cen = 0.5 * (x2f[:-1] + x2f[1:])     # cell-centred y coords, shape (Nx2,)
    del fd0
    gc.collect()

    # ── 5. Accumulate horizontal averages  ────────────────────────────────────
    # Storage: (n_frames, Nx2) — only 1D profiles, RAM-cheap
    n_frames = len(bin_subset)

    prof_rho      = np.full((n_frames, Nx2), np.nan, dtype=np.float32)
    prof_mass_flux= np.full((n_frames, Nx2), np.nan, dtype=np.float32)
    prof_mom_flux = np.full((n_frames, Nx2), np.nan, dtype=np.float32)
    prof_T        = np.full((n_frames, Nx2), np.nan, dtype=np.float32)
    prof_enrg_flux= np.full((n_frames, Nx2), np.nan, dtype=np.float32)
    prof_cooling  = np.full((n_frames, Nx2), np.nan, dtype=np.float32)

    # PDF storage: only for N_PDF_SNAPS frames after t0
    # Note: in code units T = p/rho, so T_hot >> T_cold (hot gas has lower rho)
    T_min_global = 1.1 * min(T_cold, T_hot)
    T_max_global = 0.9 * max(T_cold, T_hot)
    T_bins = np.linspace(T_min_global, T_max_global, N_TEMP_BINS + 1)
    T_centres = 0.5 * (T_bins[:-1] + T_bins[1:])
    pdf_times = []
    pdf_hists = []   # list of (N_TEMP_BINS,) arrays

    # Pick which frames to use for PDFs: N_PDF_SNAPS evenly-spaced post-t0
    if post_t0_mask.sum() >= N_PDF_SNAPS:
        pdf_frame_idxs = set(
            post_t0_idx[np.round(
                np.linspace(0, len(post_t0_idx) - 1, N_PDF_SNAPS)
            ).astype(int)]
        )
    else:
        pdf_frame_idxs = set(post_t0_idx.tolist())

    print(f"  Loading frames…")
    for fi, (bfile, t) in enumerate(zip(bin_subset, times_subset)):
        if fi % 20 == 0:
            print(f"    frame {fi:4d}/{n_frames}  t={t:.3f}", flush=True)

        try:
            fd = read_binary(str(bfile))
        except Exception as e:
            print(f"    [WARN] Could not read {bfile.name}: {e}")
            continue

        # Primitive fields (2D)
        rho  = make_2d(fd, "dens")
        mom2 = make_2d(fd, "mom2")   # ρ*vy
        mom1 = make_2d(fd, "mom1")   # ρ*vx
        etot = make_2d(fd, "ener")   # total energy (conserved, AthenaK hydro_u key)

        vy   = mom2 / (rho + 1e-30)
        vx   = mom1 / (rho + 1e-30)
        KE   = 0.5 * rho * (vx**2 + vy**2)
        p    = np.maximum((gamma - 1.0) * (etot - KE), 1e-30)
        T    = p / (rho + 1e-30)

        # Bernoulli number (dominated by enthalpy for subsonic):
        # B = v^2/2 + gamma * p / ((gamma-1) * rho)
        bernoulli = (vx**2 + vy**2) / 2.0 + gamma * p / ((gamma - 1.0) * rho)

        # Cooling ~ n^2 * Lambda(T)  [n = rho/mu in code units]
        # Convert code-unit T to Kelvin for lambda_cool, then calculate
        # physical cooling rate in CGS, and convert back to code units.
        T_kelvin = T * T_to_K
        lam_val  = lambda_cool(T_kelvin)  # erg cm^3 / s
        n_cgs    = (rho * rho_unit_cgs) / (mu * m_H_cgs)
        cooling_cgs = n_cgs**2 * lam_val  # erg / (cm^3 s)
        cooling  = cooling_cgs * (time_cgs / pressure_unit_cgs)

        # 1D horizontal averages
        prof_rho[fi]       = horiz_mean(rho).astype(np.float32)
        prof_mass_flux[fi] = horiz_mean(mom2).astype(np.float32)          # <rho*vy>
        prof_mom_flux[fi]  = horiz_mean(rho * vy**2 + p).astype(np.float32)  # <rho*vy^2 + p>
        prof_T[fi]         = horiz_mean(T).astype(np.float32)
        prof_enrg_flux[fi] = horiz_mean(rho * vy * bernoulli).astype(np.float32)  # <rho*vy*B>
        prof_cooling[fi]   = horiz_mean(cooling).astype(np.float32)       # <n^2*Lambda>

        # PDF (only selected frames)
        if fi in pdf_frame_idxs:
            mask_trml = (T > 1.1 * min(T_cold, T_hot)) & (T < 0.9 * max(T_cold, T_hot))
            T_flat = T[mask_trml].ravel()
            if len(T_flat) > 0:
                hist, _ = np.histogram(T_flat, bins=T_bins, density=True)
            else:
                hist = np.zeros(N_TEMP_BINS)
            pdf_hists.append(hist.astype(np.float32))
            pdf_times.append(float(t))

        # Free memory immediately
        del fd, rho, mom1, mom2, etot, vy, vx, KE, p, T, bernoulli, cooling
        gc.collect()

    print("  Frame loading complete.")

    # ─────────────────────────────────────────────────────────────────────────
    # Analysis: compute per-diagnostic metrics
    # ─────────────────────────────────────────────────────────────────────────

    results = {
        "sim_name":      sim_name,
        "t0":            t0,
        "t_final":       t_final,
        "t0_reached":    t0_reached,
        "y_cen":         y_cen,
        "times":         times_subset,
        "gamma":         gamma,
        "rho_cold":      rho_cold,
        "rho_hot":       rho_hot,
        "T_cold":        T_cold,
        "T_hot":         T_hot,
        "T0":            T0,
        "chi":           chi,
        "T_bins":        T_bins,
        "T_centres":     T_centres,
        "pdf_times":     np.array(pdf_times),
        "pdf_hists":     np.array(pdf_hists) if pdf_hists else np.zeros((1, N_TEMP_BINS)),
    }

    # ── Diagnostic 2: Mass flux flatness ─────────────────────────────────────
    # Use the time-averaged mass flux profile over post-t0 frames
    if post_t0_mask.sum() > 0:
        mf_post = prof_mass_flux[post_t0_mask]
        mean_mf  = np.nanmean(mf_post, axis=0)
        # Flatness metric: std / |mean| across y (ignoring z-boundaries)
        core = slice(Nx2 // 6, 5 * Nx2 // 6)  # central 2/3 of domain
        mf_flatness = (np.nanstd(mean_mf[core]) /
                       (np.abs(np.nanmean(mean_mf[core])) + 1e-10))
        mf_flat = mf_flatness < 0.15   # threshold: 15% variation = "flat"
    else:
        mean_mf    = np.zeros(Nx2)
        mf_flatness = np.nan
        mf_flat    = False

    results.update({
        "prof_mass_flux":  prof_mass_flux,
        "mean_mass_flux":  mean_mf,
        "mf_flatness":     float(mf_flatness),
        "mf_flat":         mf_flat,
    })
    print(f"  [2] Mass flux flatness = {mf_flatness:.3f}  → {'✓ FLAT' if mf_flat else '✗ NOT FLAT'}")

    # ── Diagnostic 3: Density stationarity in TRML frame ─────────────────────
    # TRML drift velocity (simulation frame, cold phase stationary):
    # v_TRML = -u_{y,h} / (chi - 1)
    # Estimate u_{y,h} as the mass-flux at the hot boundary (top of domain)
    hot_boundary_slice = slice(4 * Nx2 // 5, Nx2)   # top 20%
    cold_boundary_slice = slice(0, Nx2 // 5)         # bottom 20%

    if post_t0_mask.sum() > 1:
        # Boosted y-coordinates for each frame
        dy = y_cen[1] - y_cen[0]
        boosted_profiles = []
        base_times = times_subset[post_t0_mask]
        base_profs  = prof_rho[post_t0_mask]

        # Estimate mean hot-phase vy from mass flux / hot-phase density
        rho_h_est = rho_hot
        mf_h_arr  = prof_mass_flux[post_t0_mask][:, hot_boundary_slice].mean(axis=1)
        uy_h_arr  = mf_h_arr / (rho_h_est + 1e-30)
        v_trml_arr = -uy_h_arr / (chi - 1.0 + 1e-10)

        # Compute std of TRML-frame profiles as a stationarity measure
        cumulative_shift = 0.0
        boosted = [base_profs[0]]
        for i in range(1, len(base_profs)):
            dt = base_times[i] - base_times[i - 1]
            v1 = v_trml_arr[i] if np.isfinite(v_trml_arr[i]) else 0.0
            v0 = v_trml_arr[i - 1] if np.isfinite(v_trml_arr[i - 1]) else 0.0
            v_avg = 0.5 * (v1 + v0)
            cumulative_shift += v_avg * dt / dy
            boosted.append(np.roll(base_profs[i], int(round(cumulative_shift))))
        boosted = np.array(boosted)
        rho_std_trml = np.nanstd(boosted, axis=0).mean()
        rho_mean_val = np.nanmean(np.abs(boosted))
        rho_stat_score = rho_std_trml / (rho_mean_val + 1e-10)
        rho_stationary = rho_stat_score < 0.10  # 10% time-variation threshold
    else:
        boosted         = prof_rho[post_t0_mask] if post_t0_mask.sum() > 0 else prof_rho[:1]
        rho_std_trml    = np.nan
        rho_stat_score  = np.nan
        rho_stationary  = False

    results.update({
        "prof_rho":         prof_rho,
        "prof_rho_boosted": boosted,
        "rho_stat_score":   float(rho_stat_score) if not np.isnan(rho_stat_score) else None,
        "rho_stationary":   rho_stationary,
    })
    print(f"  [3] Density stationarity score = {rho_stat_score:.4f}  → {'✓ STATIONARY' if rho_stationary else '✗ DRIFTING'}")

    # ── Diagnostic 4: Momentum flux conservation ─────────────────────────────
    if post_t0_mask.sum() > 0:
        mflux_post = prof_mom_flux[post_t0_mask]
        mean_mflux = np.nanmean(mflux_post, axis=0)
        core = slice(Nx2 // 6, 5 * Nx2 // 6)
        mom_flatness = (np.nanstd(mean_mflux[core]) /
                        (np.abs(np.nanmean(mean_mflux[core])) + 1e-10))
        mom_flat = mom_flatness < 0.10
    else:
        mean_mflux   = np.zeros(Nx2)
        mom_flatness = np.nan
        mom_flat     = False

    results.update({
        "prof_mom_flux":   prof_mom_flux,
        "mean_mom_flux":   mean_mflux,
        "mom_flatness":    float(mom_flatness),
        "mom_flat":        mom_flat,
    })
    print(f"  [4] Momentum flux flatness = {mom_flatness:.3f}  → {'✓ FLAT' if mom_flat else '✗ NOT FLAT'}")

    # ── Diagnostic 5: Tanh temperature fit ───────────────────────────────────
    if post_t0_mask.sum() > 0:
        mean_T = np.nanmean(prof_T[post_t0_mask], axis=0)
        T_amp  = (T_hot - T_cold) / 2.0
        T_off  = (T_hot + T_cold) / 2.0
        try:
            p0_fit = [T_amp, T_off, 0.0, 1.0]
            bounds  = ([-np.inf, -np.inf, -5.0, 0.01],
                       [ np.inf,  np.inf,  5.0, 10.0])
            popt, _ = curve_fit(tanh_profile, y_cen, mean_T,
                                p0=p0_fit, bounds=bounds, maxfev=5000)
            T_fit_arr = tanh_profile(y_cen, *popt)
            tanh_residual = np.sqrt(np.mean((mean_T - T_fit_arr)**2)) / (T_amp + 1e-10)
            tanh_good    = tanh_residual < 0.10  # <10% rms residual
            tanh_z0      = float(abs(popt[3]))
        except Exception as e:
            print(f"    [WARN] Tanh fit failed: {e}")
            tanh_residual = np.nan
            tanh_good     = False
            tanh_z0       = np.nan
            popt          = [T_amp, T_off, 0.0, 1.0]
            T_fit_arr     = np.zeros(Nx2)
    else:
        mean_T       = np.zeros(Nx2)
        tanh_residual = np.nan
        tanh_good     = False
        tanh_z0       = np.nan
        popt         = [0, 0, 0, 1]
        T_fit_arr    = np.zeros(Nx2)

    results.update({
        "prof_T":          prof_T,
        "mean_T":          mean_T,
        "tanh_residual":   float(tanh_residual) if not np.isnan(tanh_residual) else None,
        "tanh_good":       tanh_good,
        "tanh_z0":         float(tanh_z0) if not np.isnan(tanh_z0) else None,
        "tanh_fit":        T_fit_arr,
        "tanh_popt":       list(popt),
    })
    print(f"  [5] Tanh fit residual = {tanh_residual:.4f}, z0 = {tanh_z0:.4f}  → {'✓ GOOD FIT' if tanh_good else '✗ POOR FIT'}")

    # ── Diagnostic 6: PDF stability ───────────────────────────────────────────
    if len(pdf_hists) >= 2:
        pdf_arr = np.array(pdf_hists)
        # Compare first vs. last Wasserstein distance using correct weights and centres
        sum0 = np.sum(pdf_arr[0])
        sum1 = np.sum(pdf_arr[-1])
        if sum0 > 0 and sum1 > 0:
            pdf_wd = wasserstein_distance(
                T_centres, T_centres,
                u_weights=pdf_arr[0],
                v_weights=pdf_arr[-1]
            )
        else:
            pdf_wd = np.nan
        bin_span = T_max_global - T_min_global
        pdf_stable = pdf_wd < 0.05 * bin_span   # 5% of the temperature range
    else:
        pdf_wd     = np.nan
        pdf_stable = False

    results.update({
        "pdf_wasserstein":  float(pdf_wd) if not np.isnan(pdf_wd) else None,
        "pdf_stable":       pdf_stable,
    })
    print(f"  [6] PDF Wasserstein distance (first vs last) = {pdf_wd:.4f}  → {'✓ STABLE' if pdf_stable else '✗ EVOLVING'}")

    # ── Diagnostic 7: Energy balance ─────────────────────────────────────────
    if post_t0_mask.sum() > 0:
        mean_enrg_flux = np.nanmean(prof_enrg_flux[post_t0_mask], axis=0)
        mean_cooling   = np.nanmean(prof_cooling[post_t0_mask], axis=0)

        # d/dy (<rho*vy*B>) via finite difference
        dy = y_cen[1] - y_cen[0]
        d_enrg_flux_dy = np.gradient(mean_enrg_flux, dy)

        # Energy balance residual: |d/dy(E_flux) + cooling| / |cooling|
        # In the TRML the two should balance (opposite signs)
        core  = slice(Nx2 // 4, 3 * Nx2 // 4)
        numer = np.abs(d_enrg_flux_dy[core] + mean_cooling[core])
        denom = np.abs(mean_cooling[core]) + 1e-30
        energy_balance_err = float(np.nanmean(numer / denom))
        energy_balanced    = energy_balance_err < 0.5
    else:
        mean_enrg_flux    = np.zeros(Nx2)
        mean_cooling      = np.zeros(Nx2)
        d_enrg_flux_dy    = np.zeros(Nx2)
        energy_balance_err = np.nan
        energy_balanced    = False

    results.update({
        "prof_enrg_flux":      prof_enrg_flux,
        "prof_cooling":        prof_cooling,
        "mean_enrg_flux":      mean_enrg_flux,
        "mean_cooling":        mean_cooling,
        "d_enrg_flux_dy":      d_enrg_flux_dy,
        "energy_balance_err":  float(energy_balance_err) if not np.isnan(energy_balance_err) else None,
        "energy_balanced":     energy_balanced,
    })
    print(f"  [7] Energy balance error = {energy_balance_err:.3f}  → {'✓ BALANCED' if energy_balanced else '✗ IMBALANCED'}")

    # ─────────────────────────────────────────────────────────────────────────
    # Save diagnostics
    # ─────────────────────────────────────────────────────────────────────────

    _save_npz(results, out_dir)
    _save_summary(results, out_dir)
    _make_plots(results, out_dir / "plots", y_cen, times_subset)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Saving utilities
# ─────────────────────────────────────────────────────────────────────────────

def _save_npz(results: dict, out_dir: Path):
    """Save all array/scalar results to a compressed .npz file."""
    save_dict = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            save_dict[k] = v
        elif isinstance(v, (int, float, bool)) or v is None:
            save_dict[k] = np.array(v if v is not None else np.nan)
        elif isinstance(v, list):
            try:
                save_dict[k] = np.array(v)
            except Exception:
                pass  # skip non-numeric lists
    np.savez_compressed(out_dir / "diagnostics.npz", **save_dict)
    print(f"  → Saved diagnostics.npz")


def _fmt(v, fmt='.4f'):
    """Format a value that may be None or NaN safely."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return format(v, fmt)


def _save_summary(results: dict, out_dir: Path):
    """Write a human-readable summary.txt."""
    n = results["sim_name"]
    t0 = results["t0"]
    tf = results["t_final"]

    checks = {
        "1. Time criterion (t >= t0)":    results["t0_reached"],
        "2. Mass flux flat in z":          results["mf_flat"],
        "3. Density stationary (TRML fr)": results["rho_stationary"],
        "4. Momentum flux flat in z":      results["mom_flat"],
        "5. Temperature fits tanh":        results["tanh_good"],
        "6. PDF stable over time":         results["pdf_stable"],
        "7. Energy balance satisfied":     results["energy_balanced"],
    }

    n_passed = sum(1 for v in checks.values() if v)
    verdict  = ("QUASI-STEADY STATE REACHED" if n_passed >= 5
                else "QSS NOT CONFIRMED" if n_passed >= 3
                else "NOT IN QUASI-STEADY STATE")

    lines = [
        f"Quasi-Steady State Diagnostics — {n}",
        "=" * 60,
        f"  t_final = {tf:.4f}  |  t0 (cooling time) = {t0:.4f}",
        f"  T_cold  = {results['T_cold']:.4f}  |  T_hot = {results['T_hot']:.4f}",
        f"  chi     = {results['chi']:.1f}",
        "",
        "Diagnostic results:",
    ]
    for name, passed in checks.items():
        mark = "✓" if passed else "✗"
        lines.append(f"  {mark}  {name}")

    lines += [
        "",
        f"Metrics:",
        f"  Mass flux flatness index:      {_fmt(results.get('mf_flatness'))}  (< 0.15 = flat)",
        f"  Density stationarity score:    {_fmt(results.get('rho_stat_score'))}  (< 0.10 = stat.)",
        f"  Momentum flux flatness index:  {_fmt(results.get('mom_flatness'))}  (< 0.10 = flat)",
        f"  Tanh fit residual:             {_fmt(results.get('tanh_residual'))}  (< 0.10 = good)",
        f"  Tanh width z0:                 {_fmt(results.get('tanh_z0'))}",
        f"  PDF Wasserstein distance:      {_fmt(results.get('pdf_wasserstein'))}  (< 0.05 = stable)",
        f"  Energy balance error:          {_fmt(results.get('energy_balance_err'))}  (< 0.50 = balanced)",
        "",
        f"  Passed {n_passed}/7 diagnostics",
        f"  VERDICT: {verdict}",
    ]
    summary_text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
    print(f"  → Saved summary.txt")
    print(f"\n  {'─'*50}")
    print(f"  VERDICT: {verdict}  ({n_passed}/7 passed)")
    print(f"  {'─'*50}")


def _shade(ax, y, mu, sigma, color, alpha_fill=0.25, **line_kw):
    """Plot mean line + ±1σ shaded band."""
    ax.plot(y, mu, color=color, **line_kw)
    ax.fill_between(y, mu - sigma, mu + sigma, color=color, alpha=alpha_fill, linewidth=0)


def _make_plots(results: dict, plot_dir: Path, y_cen: np.ndarray, times: np.ndarray):
    """Generate one PNG per diagnostic."""
    sim = results["sim_name"]
    t0  = results["t0"]
    post = times >= t0

    # Pre-compute temporal mean and std for all post-t0 profiles
    def _stats(prof2d):
        """Return (mean, std) along time axis for post-t0 rows."""
        sub = prof2d[post]
        if sub.shape[0] == 0:
            z = np.zeros(prof2d.shape[1])
            return z, z
        return np.nanmean(sub, axis=0), np.nanstd(sub, axis=0)

    mf_mu,   mf_sd   = _stats(results["prof_mass_flux"])
    mfx_mu,  mfx_sd  = _stats(results["prof_mom_flux"])
    T_mu,    T_sd    = _stats(results["prof_T"])
    ef_mu,   ef_sd   = _stats(results["prof_enrg_flux"])
    cool_mu, cool_sd = _stats(results["prof_cooling"])
    rho_mu,  rho_sd  = _stats(results["prof_rho"])

    # Energy flux gradient ± propagated std
    dy = y_cen[1] - y_cen[0]
    def _grad_std(prof2d):
        sub = prof2d[post]
        if sub.shape[0] == 0:
            return np.zeros(prof2d.shape[1]), np.zeros(prof2d.shape[1])
        grads = np.array([np.gradient(row, dy) for row in sub])
        return np.nanmean(grads, axis=0), np.nanstd(grads, axis=0)

    def_mu, def_sd = _grad_std(results["prof_enrg_flux"])

    verdict_kw = dict(transform=None, ha="right", va="bottom", fontsize=10)

    # ── Plot 1: mass flux ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    mf = results["prof_mass_flux"]
    # individual pre-t0 traces (faint grey)
    for i in np.where(~post)[0]:
        ax.plot(y_cen, mf[i], color="grey", alpha=0.10, lw=0.5)
    # ±1σ band + mean for post-t0
    _shade(ax, y_cen, mf_mu, mf_sd, color="steelblue", alpha_fill=0.25,
           lw=2, label=r"mean $\pm1\sigma$ (post-$t_0$)")
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("y [code units]")
    ax.set_ylabel(r"$\langle\rho v_y\rangle$")
    ax.set_title(f"{sim} — Mass flux profile")
    flat_label = "FLAT ✓" if results["mf_flat"] else "NOT FLAT ✗"
    ax.text(0.98, 0.02, flat_label, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10,
            color="green" if results["mf_flat"] else "red")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(plot_dir / "mass_flux.png", dpi=120)
    plt.close(fig)

    # ── Plot 2: momentum flux ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    _shade(ax, y_cen, mfx_mu, mfx_sd, color="firebrick", alpha_fill=0.20,
           lw=2, label=r"mean $\pm1\sigma$ (post-$t_0$)")
    ax.set_xlabel("y [code units]")
    ax.set_ylabel(r"$\langle\rho v_y^2 + p\rangle$")
    ax.set_title(f"{sim} — Momentum flux (should be flat)")
    flat_label = "FLAT ✓" if results["mom_flat"] else "NOT FLAT ✗"
    ax.text(0.98, 0.02, flat_label, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10,
            color="green" if results["mom_flat"] else "red")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(plot_dir / "momentum_flux.png", dpi=120)
    plt.close(fig)

    # ── Plot 3: temperature profile + tanh fit ────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    _shade(ax, y_cen, T_mu, T_sd, color="darkorange", alpha_fill=0.22,
           lw=2, label=r"mean $\pm1\sigma$ (post-$t_0$)")
    z0_str = f"{results['tanh_z0']:.3f}" if results["tanh_z0"] is not None else "N/A"
    ax.plot(y_cen, results["tanh_fit"], color="navy", ls="--", lw=1.6,
            label=f"tanh fit ($z_0$={z0_str})")
    ax.set_xlabel("y [code units]")
    ax.set_ylabel("T [code units]")
    ax.set_title(f"{sim} — Temperature profile")
    fit_label = "GOOD FIT ✓" if results["tanh_good"] else "POOR FIT ✗"
    ax.text(0.98, 0.02, fit_label, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10,
            color="green" if results["tanh_good"] else "red")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(plot_dir / "temperature_tanh_fit.png", dpi=120)
    plt.close(fig)

    # ── Plot 4: Temperature PDF evolution ─────────────────────────────────────
    pdf_hists = results["pdf_hists"]
    pdf_times = results["pdf_times"]
    if pdf_hists.shape[0] > 0 and pdf_hists.shape[1] > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        cmap = plt.cm.viridis
        # shaded ±1σ envelope across all PDF snapshots
        if pdf_hists.shape[0] > 1:
            pdf_mu_all = np.nanmean(pdf_hists, axis=0)
            pdf_sd_all = np.nanstd(pdf_hists, axis=0)
            ax.fill_between(results["T_centres"],
                            pdf_mu_all - pdf_sd_all,
                            pdf_mu_all + pdf_sd_all,
                            color="grey", alpha=0.25, linewidth=0,
                            label=r"$\pm1\sigma$ envelope")
        for i in range(len(pdf_hists)):
            frac = i / max(len(pdf_hists) - 1, 1)
            ax.plot(results["T_centres"], pdf_hists[i],
                    color=cmap(frac), alpha=0.85, lw=1.3,
                    label=f"t={pdf_times[i]:.2f}")
        ax.set_xlabel("T [code units]")
        ax.set_ylabel(r"$\mathcal{P}_V(T)$  [volume-weighted PDF]")
        ax.set_title(f"{sim} — Temperature PDF stability")
        stab_label = "STABLE ✓" if results["pdf_stable"] else "EVOLVING ✗"
        ax.text(0.98, 0.96, stab_label, transform=ax.transAxes,
                ha="right", va="top", fontsize=10,
                color="green" if results["pdf_stable"] else "red")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plot_dir / "temperature_pdf.png", dpi=120)
        plt.close(fig)

    # ── Plot 5: energy balance ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    _shade(ax, y_cen, def_mu, def_sd, color="steelblue", alpha_fill=0.20,
           lw=1.8, label=r"mean $d_y\langle\rho v_y\mathcal{B}\rangle\pm1\sigma$")
    _shade(ax, y_cen, -cool_mu, cool_sd, color="firebrick", alpha_fill=0.20,
           lw=1.8, ls="--", label=r"$-\langle n^2\Lambda\rangle\pm1\sigma$")
    ax.set_xlabel("y [code units]")
    ax.set_ylabel("Energy flux div. / Cooling [code units]")
    ax.set_title(f"{sim} — Energy balance (should overlap)")
    bal_label = "BALANCED ✓" if results["energy_balanced"] else "IMBALANCED ✗"
    ax.text(0.98, 0.02, bal_label, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10,
            color="green" if results["energy_balanced"] else "red")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "energy_balance.png", dpi=120)
    plt.close(fig)

    # ── Plot 6: density evolution (TRML frame) ────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    rho_p = results["prof_rho_boosted"]
    rho_mu_b, rho_sd_b = np.nanmean(rho_p, axis=0), np.nanstd(rho_p, axis=0)
    cmap  = plt.cm.plasma
    post_idxs = np.where(post)[0]
    stride_p  = max(1, len(post_idxs) // 10)
    # individual coloured traces (faint, behind the band)
    for ii, idx in enumerate(post_idxs[::stride_p]):
        frac = ii / max(len(post_idxs[::stride_p]) - 1, 1)
        idx_boosted = np.where(post_idxs == idx)[0][0]
        ax.plot(y_cen, rho_p[idx_boosted], color=cmap(frac), alpha=0.35, lw=0.7)
    # mean ±1σ overlay
    _shade(ax, y_cen, rho_mu_b, rho_sd_b, color="black", alpha_fill=0.25,
           lw=2.0, label=r"mean $\pm1\sigma$")
    ax.set_xlabel("y [code units]")
    ax.set_ylabel(r"$\langle\rho\rangle$")
    ax.set_title(f"{sim} — Density profiles over time (TRML frame)\n"
                 "(traces: purple→yellow = early→late; black = mean±σ)")
    stat_label = "STATIONARY ✓" if results["rho_stationary"] else "DRIFTING ✗"
    ax.text(0.98, 0.02, stat_label, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10,
            color="green" if results["rho_stationary"] else "red")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(plot_dir / "density_evolution.png", dpi=120)
    plt.close(fig)

    # ── Plot 7: time series of total mass flux ────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    total_mf = results["prof_mass_flux"].mean(axis=1)
    ax.plot(times, total_mf, color="teal", lw=1.2)
    ax.axvline(t0, color="red", ls="--", lw=1.5, label=f"$t_0$={t0:.3f}")
    if post.sum() > 0:
        post_mean = np.nanmean(total_mf[post])
        post_std  = np.nanstd(total_mf[post])
        ax.axhspan(post_mean - post_std, post_mean + post_std,
                   color="teal", alpha=0.18, label=r"post-$t_0$ mean±σ")
        ax.axhline(post_mean, color="teal", lw=1.2, ls="--", alpha=0.7)
    ax.set_xlabel("t [code units]")
    ax.set_ylabel(r"Domain-avg $\langle\rho v_y\rangle$")
    ax.set_title(f"{sim} — Total mass flux vs time")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(plot_dir / "mass_flux_timeseries.png", dpi=120)
    plt.close(fig)

    print(f"  → Saved 7 diagnostic plots to {plot_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    sim_dirs = sorted(p for p in SIM_OUTPUT_ROOT.iterdir() if p.is_dir())

    if not sim_dirs:
        print(f"No simulation directories found in {SIM_OUTPUT_ROOT}")
        return

    print(f"Found {len(sim_dirs)} simulation(s):")
    for d in sim_dirs:
        print(f"  {d.name}")

    all_results = {}
    for sim_dir in sim_dirs:
        out_dir = RESULTS_ROOT / sim_dir.name
        try:
            res = run_diagnostics(sim_dir, out_dir)
            all_results[sim_dir.name] = res
        except Exception as e:
            print(f"\n[ERROR] {sim_dir.name}: {e}")
            import traceback
            traceback.print_exc()

    # ── Global summary table ──────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  GLOBAL QSS SUMMARY")
    print("=" * 70)
    header = (f"{'Simulation':<25} {'t0_ok':>6} {'MF':>5} {'Dens':>6} "
              f"{'MomF':>6} {'Tanh':>6} {'PDF':>6} {'Enrg':>6} {'Score':>7}")
    print(header)
    print("-" * 70)
    for name, res in all_results.items():
        if not res:
            continue
        marks = [
            "✓" if res["t0_reached"]    else "✗",
            "✓" if res["mf_flat"]       else "✗",
            "✓" if res["rho_stationary"] else "✗",
            "✓" if res["mom_flat"]      else "✗",
            "✓" if res["tanh_good"]     else "✗",
            "✓" if res["pdf_stable"]    else "✗",
            "✓" if res["energy_balanced"] else "✗",
        ]
        score = sum(1 for m in marks if m == "✓")
        row = (f"{name:<25} " +
               "  ".join(f"{m:>4}" for m in marks) +
               f"  {score}/7")
        print(row)
    print("=" * 70)
    print(f"\nResults saved to: {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
