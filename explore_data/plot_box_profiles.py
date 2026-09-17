"""
explore_data/plot_box_profiles.py
─────────────────────────────────────────────────────────────────────────────
Computes and plots profiles vs normalized y (y / box_height) comparing:
  - Box 10pc x 20pc
  - Box 20pc x 40pc
for each resolution in the sweep:
  8x16, 16x32, 32x64, 64x128, 128x256, 256x512, 512x1024

Outputs a separate 8-panel comparison plot (and individual field plots)
for EACH resolution in outputs/box_sweep_profiles/.
"""

import os
import sys
import gc
from pathlib import Path

PROJECT_ROOT = Path("/home/sasi/Projects/SubgridCGMModel")
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import ergane

# ── Paths ─────────────────────────────────────────────────────────────────────
SIM_ROOT = PROJECT_ROOT / "simulation_outputs"
BOX_SWEEP_ROOT = SIM_ROOT / "box_sweep"
OUT_ROOT = PROJECT_ROOT / "outputs" / "box_sweep_profiles"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ── Resolutions ───────────────────────────────────────────────────────────────
RESOLUTIONS = [
    ("8x16",     r"$16 \times 8$"),
    ("16x32",    r"$32 \times 16$"),
    ("32x64",    r"$64 \times 32$"),
    ("64x128",   r"$128 \times 64$"),
    ("128x256",  r"$256 \times 128$"),
    ("256x512",  r"$512 \times 256$"),
    ("512x1024", r"$1024 \times 512$"),
]

BOX_CASES = [
    ("box_10x20", r"10 pc $\times$ 20 pc", 20.0),
    ("box_20x40", r"20 pc $\times$ 40 pc", 40.0),
    ("box_30x60", r"30 pc $\times$ 60 pc", 60.0),
]

# ── Physical constants ────────────────────────────────────────────────────────
CM_PER_PC       = 3.08568e18          # cm per parsec
CM_PER_KM       = 1.0e5               # cm per km
SECONDS_PER_MYR = 3.15576e13          # seconds per Myr
M_H             = 1.6726219e-24       # proton mass [g]
MU              = 0.62                # mean molecular weight
GAMMA           = 5.0 / 3.0           # adiabatic index

LOGT_ACTIVE_START = 4.1
LOGT_ACTIVE_END   = 5.9


def lambda_cool(temp: np.ndarray, mask: bool = True) -> np.ndarray:
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
    lam[logt <= 4.0] = 0.0

    mk = (logt > 4.0) & (logt <= 4.2)
    if np.any(mk):
        lam[mk] = (
            2.0e-19 * np.exp(-1.184e5 / (temp[mk] + 1.0e3))
            + 2.8e-28 * np.sqrt(temp[mk]) * np.exp(-92.0 / temp[mk])
        )

    mhi = logt > 8.15
    lam[mhi] = 10.0 ** (0.45 * logt[mhi] - 26.065)

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


T_MID = float(np.sqrt(1.0e4 * 1.0e6))
LAMBDA_T_MID = float(lambda_cool(T_MID, mask=False))


def compute_cooling_normalization(athinp_path: str | Path, units: ergane.Units | None = None) -> tuple[float, float, float]:
    params = ergane.parse_athinput(athinp_path)
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

    p0_cgs   = press_code * p_unit
    rho0_cgs = np.sqrt(rho_cold_code * rho_hot_code) * rho_unit
    n0_cgs   = rho0_cgs / (MU * M_H)

    lam_t0   = LAMBDA_T_MID
    t0_cgs   = p0_cgs / ((gamma - 1.0) * (n0_cgs ** 2) * lam_t0)
    p0_over_t0 = p0_cgs / t0_cgs

    return p0_cgs, t0_cgs, p0_over_t0


def compute_physical_fields(frame: ergane.Frame, p0_over_t0: float = 1.0) -> dict[str, np.ndarray]:
    rho_cgs = frame.density              # [g cm⁻³]
    P_cgs   = frame.pressure             # [dyn cm⁻²]
    temp_K  = frame.temperature          # [K]
    vx_kms  = frame.velx                 # [km s⁻¹]
    vy_kms  = frame.vely                 # [km s⁻¹]

    n_H = rho_cgs / (MU * M_H)
    log10_nH = np.log10(np.maximum(n_H, 1e-30))
    log10_T  = np.log10(np.maximum(temp_K, 1.0))

    lam = lambda_cool(temp_K, mask=True)
    q_cool_norm = ((n_H ** 2) * lam) / p0_over_t0

    flux_x = n_H * (vx_kms * CM_PER_KM)
    flux_y = n_H * (vy_kms * CM_PER_KM)

    return {
        "log10_number_density": log10_nH,
        "log10_temperature":    log10_T,
        "cooling":              q_cool_norm,
        "pressure":             P_cgs,
        "velx":                 vx_kms,
        "vely":                 vy_kms,
        "flux_x":               flux_x,
        "flux_y":               flux_y,
    }


def x_average_profile(frame: ergane.Frame, values: np.ndarray) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D field, got shape {values.shape!r}.")
    dx = np.abs(np.diff(frame.x))
    weighted_sum = np.sum(values * dx[None, :], axis=1)
    return weighted_sum / np.sum(dx)


FIELD_CONFIGS = [
    {
        "key":       "log10_number_density",
        "title":     r"Log-Number-Density $\langle \log_{10} n_{\rm H} \rangle_x$",
        "ylabel":    r"$\langle \log_{10} (n_{\rm H} / \mathrm{cm^{-3}}) \rangle_x$",
        "filename":  "profile_log10_number_density",
        "yscale":    "linear",
    },
    {
        "key":       "log10_temperature",
        "title":     r"Log-Temperature $\langle \log_{10} T \rangle_x$",
        "ylabel":    r"$\langle \log_{10} (T / \mathrm{K}) \rangle_x$",
        "filename":  "profile_log10_temperature",
        "yscale":    "linear",
    },
    {
        "key":       "cooling",
        "title":     r"Normalized Cooling Rate $\langle n^2 \Lambda(T) \rangle / (p_0 / t_0)$",
        "ylabel":    r"$\langle n^2 \Lambda(T) \rangle / (p_0 / t_0)$",
        "filename":  "profile_cooling_rate",
        "yscale":    "log",
    },
    {
        "key":       "pressure",
        "title":     r"Pressure $\langle P \rangle_x$",
        "ylabel":    r"$\langle P \rangle_x \ [\mathrm{dyn\ cm^{-2}}]$",
        "filename":  "profile_pressure",
        "yscale":    "linear",
    },
    {
        "key":       "velx",
        "title":     r"Horizontal Velocity $\langle v_x \rangle_x$",
        "ylabel":    r"$\langle v_x \rangle_x \ [\mathrm{km\ s^{-1}}]$",
        "filename":  "profile_velx",
        "yscale":    "linear",
    },
    {
        "key":       "vely",
        "title":     r"Vertical Velocity $\langle v_y \rangle_x$",
        "ylabel":    r"$\langle v_y \rangle_x \ [\mathrm{km\ s^{-1}}]$",
        "filename":  "profile_vely",
        "yscale":    "linear",
    },
    {
        "key":       "flux_x",
        "title":     r"Horizontal Flux $\langle n_{\rm H} v_x \rangle_x$",
        "ylabel":    r"$\langle n_{\rm H} v_x \rangle_x \ [\mathrm{cm^{-2}\ s^{-1}}]$",
        "filename":  "profile_flux_x",
        "yscale":    "linear",
    },
    {
        "key":       "flux_y",
        "title":     r"Vertical Flux $\langle n_{\rm H} v_y \rangle_x$",
        "ylabel":    r"$\langle n_{\rm H} v_y \rangle_x \ [\mathrm{cm^{-2}\ s^{-1}}]$",
        "filename":  "profile_flux_y",
        "yscale":    "linear",
    },
]


def process_sim(datafolder: Path, athinp: Path):
    sim_data = ergane.SimulationData(athinp=str(athinp), datafolder=str(datafolder))
    p0_cgs, t0_cgs, p0_over_t0 = compute_cooling_normalization(str(athinp), sim_data.units)

    n_frames = sim_data.n_frames
    if n_frames == 0:
        return None

    frame_nums = sim_data.frame_numbers
    n_avg = min(250, n_frames)
    avg_indices = frame_nums[-n_avg:]

    frame0 = sim_data.get_frame(frame_nums[0])
    y_pc_raw = frame0.yc / CM_PER_PC
    ny_raw = y_pc_raw.size

    field_stacks = {cfg["key"]: np.zeros((n_avg, ny_raw), dtype=np.float64) for cfg in FIELD_CONFIGS}

    for idx, fn in enumerate(tqdm(avg_indices, desc=f"  [{datafolder.name}]", unit="frame", leave=False)):
        f = sim_data.get_frame(fn)
        fields_raw = compute_physical_fields(f, p0_over_t0=p0_over_t0)
        for cfg in FIELD_CONFIGS:
            k = cfg["key"]
            field_stacks[k][idx] = x_average_profile(f, fields_raw[k])
        del f
        if idx % 100 == 0:
            gc.collect()

    summary = {
        "y_pc": y_pc_raw,
        "ny": ny_raw,
        "sim_data": sim_data,
    }
    for cfg in FIELD_CONFIGS:
        k = cfg["key"]
        with np.errstate(all="ignore"):
            summary[f"{k}_mean"] = np.nanmean(field_stacks[k], axis=0)
            summary[f"{k}_std"]  = np.nanstd(field_stacks[k], axis=0)

    return summary


def plot_for_resolution(res_tag: str, res_label: str):
    print(f"\n{'='*70}\nProcessing resolution: {res_tag} ({res_label})\n{'='*70}")

    case_results = []
    for box_suffix, box_label, box_height in BOX_CASES:
        folder_candidates = [
            BOX_SWEEP_ROOT / f"hr_gpu_{res_tag}_{box_suffix}",
            SIM_ROOT / f"hr_gpu_{res_tag}_{box_suffix}",
        ]
        # fallback for 10x20 to baseline hr_gpu_<res_tag> if run there
        if box_suffix == "box_10x20":
            folder_candidates.append(SIM_ROOT / f"hr_gpu_{res_tag}")

        datafolder = None
        for fc in folder_candidates:
            if fc.is_dir():
                datafolder = fc
                break

        if datafolder is None:
            print(f"  [Skip] Folder not found for {res_tag} {box_suffix}")
            continue

        athinp = datafolder / f"kh_radiative_{res_tag}_{box_suffix}.athinput"
        if not athinp.is_file():
            matches = list(datafolder.glob("*.athinput"))
            if matches:
                athinp = matches[0]

        if not athinp or not athinp.is_file():
            print(f"  [Skip] Athinput not found in {datafolder}")
            continue

        print(f"  Loading {box_label} from: {datafolder}")
        res_data = process_sim(datafolder, athinp)
        if res_data is not None:
            # Normalized vertical coordinate: y / box_height
            y_norm = res_data["y_pc"] / box_height
            res_data["y_norm"] = y_norm
            res_data["box_label"] = box_label
            res_data["box_height"] = box_height
            res_data["box_suffix"] = box_suffix
            case_results.append(res_data)

    if len(case_results) == 0:
        print(f"No completed runs found for resolution {res_tag}.")
        return

    # ── 8-Panel Comparison Plot for this resolution ──────────────────────────
    nrows, ncols = 4, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 16), sharex=True)
    axes_flat = axes.flatten()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx_field, cfg in enumerate(FIELD_CONFIGS):
        ax     = axes_flat[idx_field]
        key    = cfg["key"]
        ylabel = cfg["ylabel"]
        title  = cfg["title"]
        is_log = (cfg["yscale"] == "log")

        for c_idx, case in enumerate(case_results):
            y_n = case["y_norm"]
            m   = case[f"{key}_mean"]
            s   = case[f"{key}_std"]
            lbl = case["box_label"]
            col = colors[c_idx % len(colors)]

            ax.plot(y_n, m, lw=2, label=lbl, color=col)
            if is_log:
                ax.fill_between(y_n, np.clip(m - s, 1e-30, None), m + s, color=col, alpha=0.25, linewidth=0)
            else:
                ax.fill_between(y_n, m - s, m + s, color=col, alpha=0.25, linewidth=0)

        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, which="both" if is_log else "major", ls="--", alpha=0.4)
        if is_log:
            ax.set_yscale("log")
        ax.legend(title="Domain Box Size", fontsize=9, loc="best")

    for col in range(ncols):
        axes[nrows - 1, col].set_xlabel(r"$y \ /\ L_y$ (Normalized Box Height)", fontsize=12)

    fig.suptitle(
        f"Vertical Profiles vs $y / L_y$ — Resolution {res_label} ({res_tag})\n"
        r"(Comparing 10pc $\times$ 20pc vs 20pc $\times$ 40pc vs 30pc $\times$ 60pc; time-averaged over late snapshots)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()

    out_file = OUT_ROOT / f"box_comparison_profiles_{res_tag}.png"
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] Resolution {res_tag} 8-panel plot -> {out_file}")

    # ── Individual Field Plots for this resolution ───────────────────────────
    for cfg in FIELD_CONFIGS:
        key      = cfg["key"]
        ylabel   = cfg["ylabel"]
        title    = cfg["title"]
        filename = cfg["filename"]
        is_log   = (cfg["yscale"] == "log")

        fig_s, ax_s = plt.subplots(figsize=(8, 5.5))
        for c_idx, case in enumerate(case_results):
            y_n = case["y_norm"]
            m   = case[f"{key}_mean"]
            s   = case[f"{key}_std"]
            lbl = case["box_label"]
            col = colors[c_idx % len(colors)]

            ax_s.plot(y_n, m, lw=2, label=lbl, color=col)
            if is_log:
                ax_s.fill_between(y_n, np.clip(m - s, 1e-30, None), m + s, color=col, alpha=0.25, linewidth=0)
            else:
                ax_s.fill_between(y_n, m - s, m + s, color=col, alpha=0.25, linewidth=0)

        ax_s.set_xlabel(r"$y \ /\ L_y$ (Normalized Box Height)", fontsize=12)
        ax_s.set_ylabel(ylabel, fontsize=12)
        ax_s.set_title(
            f"{title} vs $y / L_y$ — Res {res_label} ({res_tag})\n"
            r"(Comparing 10pc $\times$ 20pc vs 20pc $\times$ 40pc vs 30pc $\times$ 60pc)",
            fontsize=12,
        )
        ax_s.grid(True, which="both" if is_log else "major", ls="--", alpha=0.4)
        if is_log:
            ax_s.set_yscale("log")
        ax_s.legend(title="Domain Box Size", fontsize=10, loc="best")
        fig_s.tight_layout()

        single_out = OUT_ROOT / f"{filename}_{res_tag}.png"
        fig_s.savefig(single_out, dpi=200, bbox_inches="tight")
        plt.close(fig_s)

    # ── Animated Video: 3 simulations side-by-side + dynamic profile vs y/Ly ─
    render_temperature_streamline_animation(res_tag, res_label, case_results)


def render_temperature_streamline_animation(res_tag: str, res_label: str, case_results: list[dict]):
    """
    Renders an MP4 animation for this resolution showing:
      - 3 simulation maps side-by-side: log10(Temperature) overlaid with velocity streamlines (vx, vy)
      - A 4th panel showing the instantaneous profile of log10(T) vs y / Ly for all 3 simulations
    """
    print(f"\n  [Video] Rendering temperature + streamline animation for {res_tag} ...")
    valid_cases = [c for c in case_results if "sim_data" in c and c["sim_data"] is not None and c["sim_data"].n_frames > 0]
    if len(valid_cases) == 0:
        print(f"  [Video Skip] No simulation data available for {res_tag}")
        return

    n_sims = len(valid_cases)
    min_frames = min(c["sim_data"].n_frames for c in valid_cases)
    # Stride to create smooth ~15-20s video (e.g. 100-200 frames at 15 fps)
    step = max(1, min_frames // 150)
    frame_indices = list(range(0, min_frames, step))
    if len(frame_indices) > 0 and frame_indices[-1] != min_frames - 1:
        frame_indices.append(min_frames - 1)

    import matplotlib.gridspec as gridspec
    from matplotlib.animation import FFMpegWriter

    # Figure layout: 1 row with (n_sims + 1) columns: n_sims maps + 1 profile plot
    fig = plt.figure(figsize=(5 * n_sims + 5.5, 7.0))
    width_ratios = [1.0] * n_sims + [1.35]
    gs = gridspec.GridSpec(1, n_sims + 1, width_ratios=width_ratios, wspace=0.32)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    video_out = OUT_ROOT / f"box_temperature_streamlines_profiles_{res_tag}.mp4"
    writer = FFMpegWriter(fps=15, bitrate=2500)

    try:
        with writer.saving(fig, str(video_out), dpi=130):
            for f_idx in tqdm(frame_indices, desc=f"  [Video {res_tag}]", unit="frame", leave=False):
                fig.clf()
                sim_axes = [fig.add_subplot(gs[0, i]) for i in range(n_sims)]
                prof_ax  = fig.add_subplot(gs[0, n_sims])

                t_current_myr = 0.0

                for i, case in enumerate(valid_cases):
                    ax = sim_axes[i]
                    sd = case["sim_data"]
                    Ly = case["box_height"]
                    fn = sd.frame_numbers[f_idx]
                    f = sd.get_frame(fn)

                    if i == 0 and len(sd.times) > f_idx:
                        t_current_myr = sd.times[f_idx]

                    temp = f.temperature
                    logT = np.log10(np.maximum(temp, 1.0))
                    vx = f.velx
                    vy = f.vely

                    x_norm = (f.xc / CM_PER_PC) / Ly
                    y_norm = (f.yc / CM_PER_PC) / Ly

                    dx = (x_norm[1] - x_norm[0]) if len(x_norm) > 1 else 0.05
                    dy = (y_norm[1] - y_norm[0]) if len(y_norm) > 1 else 0.05
                    extent = [x_norm[0] - 0.5 * dx, x_norm[-1] + 0.5 * dx,
                              y_norm[0] - 0.5 * dy, y_norm[-1] + 0.5 * dy]

                    im = ax.imshow(
                        logT,
                        origin="lower",
                        extent=extent,
                        cmap="inferno",
                        vmin=4.0,
                        vmax=6.0,
                        aspect="auto",
                    )

                    # Streamlines
                    # Downsample grid if resolution is high to keep streamplot fast and clear
                    step_x = max(1, len(x_norm) // 64)
                    step_y = max(1, len(y_norm) // 128)
                    x_s = x_norm[::step_x]
                    y_s = y_norm[::step_y]
                    vx_s = vx[::step_y, ::step_x]
                    vy_s = vy[::step_y, ::step_x]

                    try:
                        ax.streamplot(
                            x_s, y_s, vx_s, vy_s,
                            color="white",
                            density=0.75,
                            linewidth=0.75,
                            arrowsize=0.75,
                        )
                    except Exception:
                        pass

                    ax.set_title(f"{case['box_label']}\n({case['box_suffix']})", fontsize=11, fontweight="bold")
                    ax.set_xlabel(r"$x \ /\ L_y$", fontsize=10)
                    ax.set_ylabel(r"$y \ /\ L_y$", fontsize=10)
                    ax.set_ylim(-0.5, 0.5)

                    # Instantaneous profile: average across x
                    profile_T = x_average_profile(f, logT)
                    col = colors[i % len(colors)]
                    prof_ax.plot(
                        y_norm,
                        profile_T,
                        lw=2.2,
                        color=col,
                        label=case["box_label"],
                    )
                    del f

                # Colorbar for temperature
                cbar = fig.colorbar(im, ax=sim_axes, orientation="horizontal", fraction=0.045, pad=0.12, aspect=30)
                cbar.set_label(r"$\log_{10} (T \ [\mathrm{K}])$", fontsize=10)

                # Format rightmost profile plot
                prof_ax.set_title(r"Instantaneous Profile $\langle \log_{10} T \rangle_x$", fontsize=11, fontweight="bold")
                prof_ax.set_xlabel(r"$y \ /\ L_y$ (Normalized Box Height)", fontsize=10)
                prof_ax.set_ylabel(r"$\langle \log_{10} (T / \mathrm{K}) \rangle_x$", fontsize=10)
                prof_ax.set_xlim(-0.5, 0.5)
                prof_ax.set_ylim(3.8, 6.2)
                prof_ax.grid(True, ls="--", alpha=0.4)
                prof_ax.legend(title="Domain Box Size", fontsize=9, loc="best")

                fig.suptitle(
                    f"Resolution {res_label} ({res_tag}) — Temperature & Velocity Streamlines vs Domain Size\n"
                    f"Time: {t_current_myr:.2f} Myr",
                    fontsize=13,
                    fontweight="bold",
                    y=0.98,
                )

                writer.grab_frame()

        plt.close(fig)
        print(f"  [Saved Video] -> {video_out}")
    except Exception as e:
        plt.close(fig)
        print(f"  [Video Warning] Failed rendering animation for {res_tag}: {e}")


def main():
    target_res = sys.argv[1] if len(sys.argv) > 1 else None
    for res_tag, res_label in RESOLUTIONS:
        if target_res and target_res != res_tag:
            continue
        plot_for_resolution(res_tag, res_label)
    print("\nAll requested resolution plots completed successfully.")


if __name__ == "__main__":
    main()
