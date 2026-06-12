"""
benchmark_analysis.py
---------------------
Benchmarking analysis for the log_cnn (GMM-CNN) subgrid model.

Reads simulation binary output, runs model predictions, and writes all
benchmark plots to SUBGRID_OUTPUT_DIR.

Environment variables (set by benchmark.sh, or set manually):
    SUBGRID_SIM_BIN     - path to the simulation bin/ directory
    SUBGRID_OUTPUT_DIR  - where to write benchmark plots
    SUBGRID_HST_FILE    - path to KH.hydro.hst (optional)
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
from scipy.signal import correlate2d
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Resolve paths from environment (set by benchmark.sh)
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

SIM_BIN_DIR  = os.environ.get("SUBGRID_SIM_BIN",
                               os.path.join(PROJECT_ROOT, "simulation_outputs",
                                            "subgrid_model", "bin"))
OUTPUT_DIR   = os.environ.get("SUBGRID_OUTPUT_DIR",
                               os.path.join(PROJECT_ROOT, "outputs", "benchmark"))
HST_FILE     = os.environ.get("SUBGRID_HST_FILE",
                               os.path.join(PROJECT_ROOT, "simulation_outputs",
                                            "subgrid_model", "KH.hydro.hst"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Make sure data/ is importable
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "conv_nn"))

import bin_convert
from data_preprocess import simulation_data

# ---------------------------------------------------------------------------
# Model / simulation parameters (must match log_cnn.py training config)
# ---------------------------------------------------------------------------
RESOLUTION  = (512, 256)
DOWNSAMPLE  = 32
IN_CHANNELS = 5
OUT_BINS    = 40   # number of PDF bins

T_EDGES   = np.logspace(3.0, 7.0, OUT_BINS + 1)
T_CENTERS = np.sqrt(T_EDGES[:-1] * T_EDGES[1:])

# --- cooling function (duplicated here for self-containedness) ---
def lambda_cool(temp):
    """ISMCoolFn cooling rate in erg cm^3 s^-1, masked to 4.5 < logT < 5.5."""
    lhd = np.array([
        -22.5977, -21.9689, -21.5972, -21.4615, -21.4789, -21.5497, -21.6211, -21.6595,
        -21.6426, -21.5688, -21.4771, -21.3755, -21.2693, -21.1644, -21.0658, -20.9778,
        -20.8986, -20.8281, -20.7700, -20.7223, -20.6888, -20.6739, -20.6815, -20.7051,
        -20.7229, -20.7208, -20.7058, -20.6896, -20.6797, -20.6749, -20.6709, -20.6748,
        -20.7089, -20.8031, -20.9647, -21.1482, -21.2932, -21.3767, -21.4129, -21.4291,
        -21.4538, -21.5055, -21.5740, -21.6300, -21.6615, -21.6766, -21.6886, -21.7073,
        -21.7304, -21.7491, -21.7607, -21.7701, -21.7877, -21.8243, -21.8875, -21.9738,
        -22.0671, -22.1537, -22.2265, -22.2821, -22.3213, -22.3462, -22.3587, -22.3622,
        -22.3590, -22.3512, -22.3420, -22.3342, -22.3312, -22.3346, -22.3445, -22.3595,
        -22.3780, -22.4007, -22.4289, -22.4625, -22.4995, -22.5353, -22.5659, -22.5895,
        -22.6059, -22.6161, -22.6208, -22.6213, -22.6184, -22.6126, -22.6045, -22.5945,
        -22.5831, -22.5707, -22.5573, -22.5434, -22.5287, -22.5140, -22.4992, -22.4844,
        -22.4695, -22.4543, -22.4392, -22.4237, -22.4087, -22.3928
    ])
    logt = np.log10(temp)
    lam  = np.zeros_like(temp, dtype=float)
    mask_off = logt <= 4.0
    lam[mask_off] = 0.0
    mask_ki = (logt > 4.0) & (logt <= 4.2)
    if np.any(mask_ki):
        lam[mask_ki] = (2.0e-19 * np.exp(-1.184e5 / (temp[mask_ki] + 1.0e3)) +
                        2.8e-28 * np.sqrt(temp[mask_ki]) * np.exp(-92.0 / temp[mask_ki]))
    mask_hi = logt > 8.15
    lam[mask_hi] = 10.0 ** (0.45 * logt[mask_hi] - 26.065)
    mask_mid = (logt > 4.2) & (logt <= 8.15)
    if np.any(mask_mid):
        ipps = np.clip((25.0 * logt[mask_mid] - 103).astype(int), 0, 100)
        x0   = 4.12 + 0.04 * ipps
        dx   = logt[mask_mid] - x0
        logcool = (lhd[ipps + 1] * dx - lhd[ipps] * (dx - 0.04)) * 25.0
        lam[mask_mid] = 10.0 ** logcool
    
    # 4.5 < logT < 5.5 mask from log_plot.py
    mask_cool_range = (logt < 4.5) | (logt > 5.5)
    lam[mask_cool_range] = 0.0
    return lam


# ---------------------------------------------------------------------------
# Load simulation data
# ---------------------------------------------------------------------------
def load_sim_data():
    """Load all primitive and conserved binary snapshots."""
    print(f"\n[1/6] Loading simulation data from: {SIM_BIN_DIR}")

    prim_files  = sorted(glob.glob(os.path.join(SIM_BIN_DIR, "KH.hydro_w.*.bin")))
    cons_files  = sorted(glob.glob(os.path.join(SIM_BIN_DIR, "KH.hydro_u.*.bin")))

    if not prim_files and not cons_files:
        # Try loading just conserved files if primitives are absent
        cons_files = sorted(glob.glob(os.path.join(SIM_BIN_DIR, "KH.hydro_u.*.bin")))
        if not cons_files:
            raise FileNotFoundError(
                f"No AthenaK .bin files found in {SIM_BIN_DIR}.\n"
                "Expected files like KH.hydro_w.NNNNN.bin or KH.hydro_u.NNNNN.bin"
            )
        prim_files = cons_files  # fall back — will load same files for both

    use_files = prim_files if prim_files else cons_files
    n = len(use_files)
    print(f"  Found {n} snapshots.")

    sim = simulation_data()

    # Probe first file for dimensions
    probe = bin_convert.read_binary(use_files[0])
    Nx1, Nx2 = probe["Nx1"], probe["Nx2"]
    sim.resolution = (Nx1, Nx2)
    print(f"  Grid size: {Nx1} x {Nx2}")

    # Dynamically set downsample factor based on dimensions
    if Nx1 < DOWNSAMPLE or Nx2 < DOWNSAMPLE:
        print(f"  [warn] Grid size {Nx1}x{Nx2} is smaller than downsample factor {DOWNSAMPLE}.")
        print("  Setting down_sample to 1 for this low-resolution run.")
        sim.down_sample = 1
    else:
        sim.down_sample = DOWNSAMPLE

    sim.rho      = np.zeros((n, Nx1, Nx2))
    sim.temp     = np.zeros((n, Nx1, Nx2))
    sim.pressure = np.zeros((n, Nx1, Nx2))
    sim.ux       = np.zeros((n, Nx1, Nx2))
    sim.uy       = np.zeros((n, Nx1, Nx2))
    sim.eint     = np.zeros((n, Nx1, Nx2))
    sim.ps       = np.zeros((n, Nx1, Nx2))
    sim.frho     = np.zeros((n, Nx1, Nx2))

    def _load_prim(i, fpath):
        fd = bin_convert.read_binary(fpath)
        sim.rho[i]  = bin_convert.make_2D_array(fd, "dens").T
        if "velx" in fd["var_names"]:
            sim.ux[i]   = bin_convert.make_2D_array(fd, "velx").T
        if "vely" in fd["var_names"]:
            sim.uy[i]   = bin_convert.make_2D_array(fd, "vely").T
        if "eint" in fd["var_names"]:
            sim.eint[i] = bin_convert.make_2D_array(fd, "eint").T
        elif "Eint" in fd["var_names"]:
            sim.eint[i] = bin_convert.make_2D_array(fd, "Eint").T
        if "s_00" in fd["var_names"]:
            sim.ps[i]   = bin_convert.make_2D_array(fd, "s_00").T
        if "s_01" in fd["var_names"]:
            sim.frho[i] = bin_convert.make_2D_array(fd, "s_01").T
        sim.pressure[i] = 2.0 / 3.0 * sim.eint[i]
        sim.temp[i]     = (sim.pressure[i] * 1.59916e-14 / sim.rho[i]) / 1.381e-16

    for i, fp in enumerate(tqdm(use_files, desc="Loading primitives")):
        _load_prim(i, fp)

    # Conserved quantities (optional — needed for source term calc)
    sim.cons_rho  = np.zeros_like(sim.rho)
    sim.cons_momx = np.zeros_like(sim.rho)
    sim.cons_momy = np.zeros_like(sim.rho)
    sim.cons_ener = np.zeros_like(sim.rho)
    sim.cons_ps   = np.zeros_like(sim.rho)

    for i, fp in enumerate(tqdm(cons_files[:n], desc="Loading conserved")):
        try:
            fd = bin_convert.read_binary(fp)
            vn = fd["var_names"]
            if "dens" in vn:
                sim.cons_rho[i]  = bin_convert.make_2D_array(fd, "dens").T
            if "mom1" in vn:
                sim.cons_momx[i] = bin_convert.make_2D_array(fd, "mom1").T
            if "mom2" in vn:
                sim.cons_momy[i] = bin_convert.make_2D_array(fd, "mom2").T
            if "ener" in vn:
                sim.cons_ener[i] = bin_convert.make_2D_array(fd, "ener").T
            if "r_00" in vn:
                sim.cons_ps[i]   = bin_convert.make_2D_array(fd, "r_00").T
        except Exception as e:
            print(f"  [warn] Could not load conserved file {fp}: {e}")

    print("  Data loaded successfully.")
    return sim


# ---------------------------------------------------------------------------
# CNN prediction
# ---------------------------------------------------------------------------
def predict_pdf(sim):
    """
    Run log_cnn model on every timestep and return predicted PDFs.
    Returns: pred_pdf shape (n_t, OUT_BINS, nx, ny)
    """
    print("\n[2/6] Running log_cnn model predictions …")
    try:
        from conv_nn.log_cnn import snapshot_pred
    except ImportError:
        # Try direct import if models/ is on sys.path
        try:
            from log_cnn import snapshot_pred
        except ImportError:
            print("  [WARN] Could not import log_cnn.snapshot_pred — skipping CNN predictions.")
            return None

    nt = sim.rho.shape[0]
    nx = sim.resolution[0] // sim.down_sample
    ny = sim.resolution[1] // sim.down_sample
    pred_pdf = np.zeros((nt, OUT_BINS, nx, ny))
    success_count = 0

    for i in tqdm(range(nt), desc="CNN inference"):
        try:
            pred_pdf[i] = snapshot_pred(
                sim.rho[i], sim.temp[i], sim.pressure[i],
                sim.ux[i], sim.uy[i], sim.eint[i], sim.ps[i],
                sim.down_sample, sim.resolution
            )
            success_count += 1
        except Exception as e:
            # Silence per-step errors to avoid log clutter, since they are expected on low-res mock snapshots
            pass

    if success_count == 0:
        print("  [WARN] All CNN predictions failed (likely due to missing weight/normalization files for this resolution). Skipping CNN predictions.")
        return None

    # Normalize
    pred_pdf /= (pred_pdf.sum(axis=1, keepdims=True) + 1e-12)
    print(f"  Predictions complete. Successfully predicted {success_count}/{nt} snapshots.")
    return pred_pdf


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def savefig(name, dpi=150, tight=True):
    path = os.path.join(OUTPUT_DIR, name)
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {name}")
    return path


# ===========================================================================
# PLOT 1 — History file diagnostics
# ===========================================================================
def plot_history():
    print("\n[3/6] Plotting history-file diagnostics …")
    if not os.path.isfile(HST_FILE):
        print(f"  [skip] HST file not found: {HST_FILE}")
        return

    data = np.loadtxt(HST_FILE, comments='#')
    if data.ndim == 1:
        data = data[np.newaxis, :]

    t      = data[:, 0]
    dt     = data[:, 1]
    mass   = data[:, 2]
    mom1   = data[:, 3]
    mom2   = data[:, 4]
    tot_E  = data[:, 6]
    KE_x   = data[:, 7]
    KE_y   = data[:, 8]

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Simulation History — KH Instability (log_cnn run)", fontsize=14)

    axs[0, 0].plot(t, mass);   axs[0, 0].set_title("Total Mass"); axs[0, 0].set_xlabel("time")
    axs[0, 1].plot(t, tot_E);  axs[0, 1].set_title("Total Energy"); axs[0, 1].set_xlabel("time")
    axs[0, 2].plot(t, dt);     axs[0, 2].set_title("Timestep dt"); axs[0, 2].set_xlabel("time")
    axs[1, 0].plot(t, KE_x, label="KE_x"); axs[1, 0].plot(t, KE_y, label="KE_y")
    axs[1, 0].legend(); axs[1, 0].set_title("Kinetic Energy"); axs[1, 0].set_xlabel("time")
    axs[1, 1].plot(t, mom1, label="mom1"); axs[1, 1].plot(t, mom2, label="mom2")
    axs[1, 1].legend(); axs[1, 1].set_title("Momentum"); axs[1, 1].set_xlabel("time")
    # Conservation violation
    axs[1, 2].plot(t, 100 * (mass - mass[0]) / mass[0], label="mass %")
    axs[1, 2].plot(t, 100 * (tot_E - tot_E[0]) / tot_E[0], label="energy %")
    axs[1, 2].legend(); axs[1, 2].set_title("Conservation Error (%)"); axs[1, 2].set_xlabel("time")

    savefig("history_diagnostics.png")


# ===========================================================================
# PLOT 2 — Snapshot of density and temperature fields
# ===========================================================================
def plot_field_snapshots(sim):
    print("\n[4/6] Plotting field snapshots …")
    nt = sim.rho.shape[0]
    sample_idxs = [0, nt // 4, nt // 2, 3 * nt // 4, nt - 1]

    fig, axs = plt.subplots(len(sample_idxs), 3, figsize=(15, 4 * len(sample_idxs)))
    fig.suptitle("Field Snapshots — Density, Temperature, f_mcl", fontsize=13)

    for row, idx in enumerate(sample_idxs):
        t_label = f"Step {idx}"

        # Density
        im0 = axs[row, 0].imshow(sim.rho[idx], origin="lower", cmap="plasma",
                                  norm=LogNorm(vmin=np.percentile(sim.rho[idx][sim.rho[idx]>0], 1),
                                               vmax=np.percentile(sim.rho[idx], 99)))
        axs[row, 0].set_title(f"Density  [{t_label}]"); plt.colorbar(im0, ax=axs[row, 0])

        # Temperature
        im1 = axs[row, 1].imshow(sim.temp[idx], origin="lower", cmap="inferno",
                                  norm=LogNorm(vmin=max(1e3, np.percentile(sim.temp[idx], 1)),
                                               vmax=np.percentile(sim.temp[idx], 99)))
        axs[row, 1].set_title(f"Temperature  [{t_label}]"); plt.colorbar(im1, ax=axs[row, 1])

        # f_mcl
        fmcl = sim.calc_fmcl(sim.rho[idx], sim.temp[idx])
        im2 = axs[row, 2].imshow(fmcl, origin="lower", cmap="viridis", vmin=0, vmax=1)
        axs[row, 2].set_title(f"f_mcl (CG)  [{t_label}]"); plt.colorbar(im2, ax=axs[row, 2])

    savefig("field_snapshots.png")


# ===========================================================================
# PLOT 3 — Coarse-grained temperature vs. prediction PDF comparison
# ===========================================================================
def plot_pdf_comparison(sim, pred_pdf):
    print("\n  Plotting PDF comparison …")
    
    nt = sim.rho.shape[0]
    nx = sim.resolution[0] // sim.down_sample
    ny = sim.resolution[1] // sim.down_sample
    
    # Calculate True PDF
    true_pdf = sim.calc_pixel_pdf(bins=OUT_BINS)
    true_pdf /= (true_pdf.sum(axis=1, keepdims=True) + 1e-12)

    # Compute coarse-grained temperature
    cg_temp = np.zeros((nt, nx, ny))
    for t in range(nt):
        cg_temp[t] = sim.coarse_grain(sim.temp[t])

    # 1. True mean vs predicted mean PDF comparison (if pred_pdf is available)
    if pred_pdf is not None:
        true_mean = true_pdf.mean(axis=(0, 2, 3))   # (bins,)
        pred_mean = pred_pdf.mean(axis=(0, 2, 3))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(T_CENTERS, true_mean, label="True (mean)", lw=2)
        ax.plot(T_CENTERS, pred_mean, label="Predicted (mean)", lw=2, linestyle="--")
        ax.set_xscale("log"); ax.set_xlabel("Temperature [K]"); ax.set_ylabel("PDF")
        ax.set_title("Mean Temperature PDF: True vs Predicted"); ax.legend()
        savefig("pdf_mean_comparison.png")

        def kl(p, q):
            p = np.clip(p, 1e-12, None); q = np.clip(q, 1e-12, None)
            return np.sum(p * np.log(p / q), axis=0)  # sum over bins (axis=0)

        kl_vals = np.array([kl(true_pdf[i], pred_pdf[i]).mean() for i in range(nt)])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(kl_vals, lw=2)
        ax.set_xlabel("Timestep"); ax.set_ylabel("Mean KL Divergence (true ‖ pred)")
        ax.set_title("KL Divergence Over Time"); ax.grid(True, alpha=0.3)
        savefig("pdf_kl_divergence.png")

        # 3. Spatial Map of time-averaged KL Divergence (New diagnostic)
        kl_grid = np.zeros((nx, ny))
        for t in range(nt):
            kl_grid += kl(true_pdf[t], pred_pdf[t])
        kl_grid /= nt

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(kl_grid, origin="lower", cmap="coolwarm", norm=LogNorm(vmin=1e-3, vmax=10.0))
        ax.set_title("Time-Averaged KL Divergence (True ‖ Pred)")
        ax.set_xlabel("Y cell index")
        ax.set_ylabel("X cell index")
        plt.colorbar(im, ax=ax, label="KL Divergence")
        savefig("kl_divergence_spatial_map.png")

    # 4. True PDF animation (log_pdf_animation.gif) and snapshots (log_pdf_snapshot_t0.png, log_temp_snapshot_t0.png)
    print("  Creating True PDF grid animation & snapshots...")
    fig = plt.figure(figsize=(ny * 2.2, nx * 1.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])

    # ---- PDF GRID ----
    pdf_axes = np.empty((nx, ny), dtype=object)
    sub_gs = gs[0].subgridspec(nx, ny)
    for i in range(nx):
        for j in range(ny):
            ax = fig.add_subplot(sub_gs[i, j])
            pdf_axes[i, j] = ax
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.3)
            ax.set_xticks([])
            ax.set_yticks([])

    # ---- TEMP PANEL ----
    temp_ax = fig.add_subplot(gs[1])
    log_temp0 = np.log10(cg_temp[0] + 1e-8)
    temp_im = temp_ax.imshow(log_temp0, origin="lower", cmap="inferno")
    temp_ax.set_title(r"$\log_{10}$ Temp (CG)")
    cbar = plt.colorbar(temp_im, ax=temp_ax, fraction=0.046)
    cbar.set_label(r"$\log_{10}$ Temperature")

    # ---- INIT LINES ----
    x_bins = np.arange(OUT_BINS)
    lines = []
    for i in range(nx):
        row = []
        for j in range(ny):
            line, = pdf_axes[i, j].plot([], [], lw=1)
            pdf_axes[i, j].set_xlim(0, OUT_BINS - 1)
            pdf_axes[i, j].set_ylim(0, 1)
            row.append(line)
        lines.append(row)

    def init_anim():
        for i in range(nx):
            for j in range(ny):
                lines[i][j].set_data([], [])
        return sum(lines, [])

    def update_anim(frame):
        pdf = true_pdf[frame]
        for i in range(nx):
            for j in range(ny):
                ii = nx - 1 - i  # Flip vertically to match imshow
                y = pdf[:, ii, j]
                y = y / (y.max() + 1e-8)
                lines[i][j].set_data(x_bins, y)

        log_temp = np.log10(cg_temp[frame] + 1e-8)
        temp_im.set_data(log_temp)
        fig.suptitle(f"True PDF Grid & CG Temperature | t = {frame}", fontsize=14)

        if frame == 0:
            # Save first frame snapshots
            fig.savefig(os.path.join(OUTPUT_DIR, "log_pdf_snapshot_t0.png"), dpi=300, bbox_inches="tight")
            plt.imsave(os.path.join(OUTPUT_DIR, "log_temp_snapshot_t0.png"), log_temp, cmap="inferno", origin="lower")
            print("  Saved log_pdf_snapshot_t0.png and log_temp_snapshot_t0.png")

        return sum(lines, []) + [temp_im]

    try:
        anim = animation.FuncAnimation(fig, update_anim, frames=nt, init_func=init_anim, blit=False)
        anim.save(os.path.join(OUTPUT_DIR, "log_pdf_animation.gif"), writer="pillow", fps=10)
        print("  Saved log_pdf_animation.gif")
    except Exception as e:
        print(f"  [warn] Could not save True PDF animation: {e}")
    finally:
        plt.close(fig)

    # 5. True vs Predicted PDF Comparison Animation (log_pdf_compare_animation.gif) and snapshot (log_pdf_compare_t0.png)
    if pred_pdf is not None:
        print("  Creating True vs Predicted PDF comparison animation...")
        mu = 0.62
        kb = 1.380649e-16
        T = T_CENTERS[:, None, None]  # (bins, 1, 1)

        true_cool = np.zeros((nt, nx, ny))
        pred_cool = np.zeros((nt, nx, ny))
        cg_pressure = np.zeros((nt, nx, ny))

        for t in range(nt):
            rho = sim.rho[t]
            temp = sim.temp[t]
            n = rho / mu
            lam = lambda_cool(temp)
            fine_cool = lam * n**2 * 1.975e27
            true_cool[t] = sim.coarse_grain(fine_cool)

            cg_pressure[t] = sim.coarse_grain(sim.pressure[t])
            P = cg_pressure[t][None, :, :]
            n_cg = P / (kb * T)
            lam_cg = lambda_cool(T)
            pred_cool[t] = np.sum(pred_pdf[t] * lam_cg * n_cg**2, axis=0)

        fig2 = plt.figure(figsize=(ny * 4.0, nx * 1.8))
        gs2 = fig2.add_gridspec(1, 2, width_ratios=[1, 1])

        # ---- LEFT (TRUE) ----
        true_axes = np.empty((nx, ny), dtype=object)
        sub_gs_left = gs2[0].subgridspec(nx, ny)
        for i in range(nx):
            for j in range(ny):
                ax = fig2.add_subplot(sub_gs_left[i, j])
                true_axes[i, j] = ax
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.3)
                ax.set_xticks([])
                ax.set_yticks([])

        # ---- RIGHT (PRED) ----
        pred_axes = np.empty((nx, ny), dtype=object)
        sub_gs_right = gs2[1].subgridspec(nx, ny)
        for i in range(nx):
            for j in range(ny):
                ax = fig2.add_subplot(sub_gs_right[i, j])
                pred_axes[i, j] = ax
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.3)
                ax.set_xticks([])
                ax.set_yticks([])

        # ---- LINES + TEXT ----
        true_lines, pred_lines = [], []
        true_texts, pred_texts = [], []
        for i in range(nx):
            row_tl, row_pl = [], []
            row_tt, row_pt = [], []
            for j in range(ny):
                lt, = true_axes[i, j].plot([], [], lw=1)
                lp, = pred_axes[i, j].plot([], [], lw=1, color='r')
                true_axes[i, j].set_xlim(0, OUT_BINS - 1)
                true_axes[i, j].set_ylim(0, 1)
                pred_axes[i, j].set_xlim(0, OUT_BINS - 1)
                pred_axes[i, j].set_ylim(0, 1)

                ttxt = true_axes[i, j].text(0.95, 0.95, "", transform=true_axes[i, j].transAxes,
                                            fontsize=6, color="black", ha="right", va="top")
                ptxt = pred_axes[i, j].text(0.95, 0.95, "", transform=pred_axes[i, j].transAxes,
                                            fontsize=6, color="black", ha="right", va="top")
                row_tl.append(lt)
                row_pl.append(lp)
                row_tt.append(ttxt)
                row_pt.append(ptxt)
            true_lines.append(row_tl)
            pred_lines.append(row_pl)
            true_texts.append(row_tt)
            pred_texts.append(row_pt)

        def init_compare():
            for i in range(nx):
                for j in range(ny):
                    true_lines[i][j].set_data([], [])
                    pred_lines[i][j].set_data([], [])
                    true_texts[i][j].set_text("")
                    pred_texts[i][j].set_text("")
            return sum(true_lines, []) + sum(pred_lines, [])

        def update_compare(frame):
            t_pdf = true_pdf[frame]
            p_pdf = pred_pdf[frame]
            for i in range(nx):
                for j in range(ny):
                    ii = nx - 1 - i
                    y_t = t_pdf[:, ii, j]
                    y_p = p_pdf[:, ii, j]
                    y_t /= (y_t.max() + 1e-8)
                    y_p /= (y_p.max() + 1e-8)
                    true_lines[i][j].set_data(x_bins, y_t)
                    pred_lines[i][j].set_data(x_bins, y_p)

                    tc = true_cool[frame, ii, j]
                    pc = pred_cool[frame, ii, j]
                    true_texts[i][j].set_text(f"{tc:.1e}")
                    pred_texts[i][j].set_text(f"{pc:.1e}")

            fig2.suptitle(f"True vs Predicted PDFs (annotated cooling) | t = {frame}", fontsize=14)

            if frame == 0:
                fig2.savefig(os.path.join(OUTPUT_DIR, "log_pdf_compare_t0.png"), dpi=300, bbox_inches="tight")
                print("  Saved log_pdf_compare_t0.png")

            return sum(true_lines, []) + sum(pred_lines, []) + sum(true_texts, []) + sum(pred_texts, [])

        try:
            anim2 = animation.FuncAnimation(fig2, update_compare, frames=nt, init_func=init_compare, blit=False)
            anim2.save(os.path.join(OUTPUT_DIR, "log_pdf_compare_animation.gif"), writer="pillow", fps=10)
            print("  Saved log_pdf_compare_animation.gif")
        except Exception as e:
            print(f"  [warn] Could not save PDF comparison animation: {e}")
        finally:
            plt.close(fig2)


# ===========================================================================
# PLOT 4 — Cooling rate: true vs. predicted
# ===========================================================================
def plot_cooling(sim, pred_pdf):
    print("\n  Plotting cooling diagnostics …")
    if pred_pdf is None:
        print("  [skip] No predictions available.")
        return

    nt   = sim.rho.shape[0]
    nx   = sim.resolution[0] // sim.down_sample
    ny   = sim.resolution[1] // sim.down_sample
    mu   = 0.62
    kb   = 1.380649e-16
    T    = T_CENTERS[:, None, None]   # (bins, 1, 1)

    true_cool  = np.zeros((nt, nx, ny))
    pred_cool  = np.zeros((nt, nx, ny))
    cg_pressure = np.zeros((nt, nx, ny))
    cg_temp = np.zeros((nt, nx, ny))

    for i in tqdm(range(nt), desc="Cooling calc"):
        rho_i  = sim.rho[i];   temp_i  = sim.temp[i]
        pres_i = sim.pressure[i]
        n_i    = rho_i / mu
        lam_i  = lambda_cool(temp_i)
        true_cool[i] = sim.coarse_grain(lam_i * n_i**2 * 1.975e27)
        cg_pressure[i] = sim.coarse_grain(pres_i)
        cg_temp[i] = sim.coarse_grain(temp_i)

        P = cg_pressure[i][None, :, :]
        n_cg = P / (kb * T)
        lam_cg = lambda_cool(T)
        pred_cool[i] = np.sum(pred_pdf[i] * lam_cg * n_cg**2, axis=0)

    # 1. Scatter plot (log scale)
    eps = 1e-30
    tv = np.clip(true_cool.ravel(), eps, None)
    pv = np.clip(pred_cool.ravel(), eps, None)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(tv, pv, s=1, alpha=0.2, c="steelblue")
    lim = [min(tv.min(), pv.min()), max(tv.max(), pv.max())]
    ax.plot(lim, lim, "r--", lw=1, label="y = x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("True Cooling Rate"); ax.set_ylabel("Predicted Cooling Rate")
    ax.set_title("Cooling Rate: True vs Predicted"); ax.legend()
    savefig("cooling_scatter.png")

    # 2. Mean cooling rate over time
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(true_cool.mean(axis=(1, 2)), label="True", lw=2)
    ax.plot(pred_cool.mean(axis=(1, 2)), label="Predicted", lw=2, linestyle="--")
    ax.set_xlabel("Timestep"); ax.set_ylabel("Mean Cooling Rate")
    ax.set_title("Mean Cooling Rate Over Time"); ax.legend(); ax.grid(True, alpha=0.3)
    savefig("cooling_mean_evolution.png")

    # 3. R² score
    ss_res = np.sum((tv - pv)**2)
    ss_tot = np.sum((tv - tv.mean())**2)
    r2 = 1 - ss_res / ss_tot
    print(f"  Cooling R² = {r2:.4f}")
    with open(os.path.join(OUTPUT_DIR, "cooling_r2.txt"), "w") as f:
        f.write(f"Cooling R² (log scale): {r2:.6f}\n")

    # 4. Global cooling scatter plot (log_cooling_scatter_global.png) from log_plot.py
    print("  Creating global cooling scatter plot...")
    true_vals = true_cool.flatten()
    pred_vals = pred_cool.flatten()
    temp_vals = cg_temp.flatten()

    true_vals = np.clip(true_vals, eps, None)
    pred_vals = np.clip(pred_vals, eps, None)
    temp_vals = np.clip(temp_vals, eps, None)

    plt.figure(figsize=(6, 6))
    sc = plt.scatter(true_vals, pred_vals, c=temp_vals, s=2, alpha=0.3, cmap='plasma', norm=LogNorm(vmin=1e3, vmax=1e8))
    
    # y = x reference
    plt.plot([1e-1, 1e3], [1e-1, 1e3], 'r--', lw=1)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1e-1, 1e3)
    plt.ylim(1e-1, 1e3)
    plt.xlabel("True Cooling")
    plt.ylabel("Predicted Cooling")
    plt.title("Cooling: True vs Predicted (All Pixels, All Timesteps)")

    # Annotate points near True ≈ 1
    mask_ann = (true_vals > 0.8) & (true_vals < 1.25)
    indices_ann = np.where(mask_ann)[0]
    if len(indices_ann) > 0:
        n_annotate = min(10, len(indices_ann))
        chosen_ann = np.random.choice(indices_ann, size=n_annotate, replace=False)
        for idx_a in chosen_ann:
            plt.text(true_vals[idx_a], pred_vals[idx_a], f"{temp_vals[idx_a]:.1e}", fontsize=6, alpha=0.7)

    cbar = plt.colorbar(sc)
    cbar.set_label("Temperature (K)")
    savefig("log_cooling_scatter_global.png")

    # 5. Global cooling PDF distribution (log_cooling_pdf_global.png) from log_plot.py
    print("  Creating cooling PDF distribution plot...")
    plt.figure(figsize=(7, 5))
    bins_hist = np.logspace(np.log10(min(true_vals.min(), pred_vals.min()) + 1e-12),
                            np.log10(max(true_vals.max(), pred_vals.max())), 100)
    plt.hist(true_vals, bins=bins_hist, density=True, histtype='step', linewidth=2, label='True Cooling')
    plt.hist(pred_vals, bins=bins_hist, density=True, histtype='step', linewidth=2, label='Predicted Cooling')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1e-2, 1e3)
    plt.xlabel("Cooling")
    plt.ylabel("PDF")
    plt.title("Cooling PDF Distribution")
    plt.legend()
    savefig("log_cooling_pdf_global.png")

    # 6. Time-averaged 2D cooling error map (New diagnostic)
    print("  Creating cooling error map...")
    cooling_ratio = np.zeros((nx, ny))
    count = np.zeros((nx, ny))
    for t in range(nt):
        t_cool = true_cool[t]
        p_cool = pred_cool[t]
        mask_sig = t_cool > 1e-2
        if np.any(mask_sig):
            cooling_ratio[mask_sig] += np.log10(p_cool[mask_sig] / t_cool[mask_sig])
            count[mask_sig] += 1
            
    cooling_ratio = np.divide(cooling_ratio, count, out=np.zeros_like(cooling_ratio), where=count > 0)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cooling_ratio, origin="lower", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Time-Averaged Cooling Log-Error\n" + r"$\langle \log_{10}(Cool_{pred} / Cool_{true}) \rangle$")
    ax.set_xlabel("Y cell index")
    ax.set_ylabel("X cell index")
    plt.colorbar(im, ax=ax, label="Log Fractional Error")
    savefig("cooling_error_map.png")

    # 7. Gas temperature phase evolution plot (New diagnostic)
    print("  Creating temperature phase evolution plot...")
    true_pdf = sim.calc_pixel_pdf(bins=OUT_BINS)
    true_pdf /= (true_pdf.sum(axis=1, keepdims=True) + 1e-12)

    cold_mask = T_CENTERS < 10**4.5
    warm_mask = (T_CENTERS >= 10**4.5) & (T_CENTERS <= 10**5.5)
    hot_mask = T_CENTERS > 10**5.5

    true_phases = {"cold": [], "warm": [], "hot": []}
    pred_phases = {"cold": [], "warm": [], "hot": []}

    for t in range(nt):
        cg_rho_t = sim.coarse_grain(sim.rho[t])
        t_pdf = true_pdf[t]
        p_pdf = pred_pdf[t]

        total_mass = np.sum(cg_rho_t)
        
        true_cold = np.sum(np.sum(t_pdf[cold_mask, :, :], axis=0) * cg_rho_t) / total_mass
        true_warm = np.sum(np.sum(t_pdf[warm_mask, :, :], axis=0) * cg_rho_t) / total_mass
        true_hot = np.sum(np.sum(t_pdf[hot_mask, :, :], axis=0) * cg_rho_t) / total_mass

        pred_cold = np.sum(np.sum(p_pdf[cold_mask, :, :], axis=0) * cg_rho_t) / total_mass
        pred_warm = np.sum(np.sum(p_pdf[warm_mask, :, :], axis=0) * cg_rho_t) / total_mass
        pred_hot = np.sum(np.sum(p_pdf[hot_mask, :, :], axis=0) * cg_rho_t) / total_mass

        true_phases["cold"].append(true_cold)
        true_phases["warm"].append(true_warm)
        true_phases["hot"].append(true_hot)

        pred_phases["cold"].append(pred_cold)
        pred_phases["warm"].append(pred_warm)
        pred_phases["hot"].append(pred_hot)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(true_phases["cold"], 'b-', label="Cold (True)", lw=1.5)
    ax.plot(pred_phases["cold"], 'b--', label="Cold (Pred)", lw=1.5)
    ax.plot(true_phases["warm"], 'g-', label="Warm (True)", lw=1.5)
    ax.plot(pred_phases["warm"], 'g--', label="Warm (Pred)", lw=1.5)
    ax.plot(true_phases["hot"], 'r-', label="Hot (True)", lw=1.5)
    ax.plot(pred_phases["hot"], 'r--', label="Hot (Pred)", lw=1.5)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Density-Weighted Mass Fraction")
    ax.set_title("Temperature Phase Evolution")
    ax.legend(ncol=3)
    ax.grid(True, alpha=0.3)
    savefig("phase_evolution.png")


# ===========================================================================
# PLOT 5 — f_mcl evolution (cold gas mass fraction)
# ===========================================================================
def plot_fmcl_evolution(sim):
    print("\n  Plotting f_mcl evolution …")
    nt = sim.rho.shape[0]
    fmcl_mean = np.zeros(nt)
    for i in range(nt):
        fmcl_mean[i] = sim.calc_fmcl(sim.rho[i], sim.temp[i]).mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(fmcl_mean, lw=2, color="teal")
    ax.set_xlabel("Timestep"); ax.set_ylabel("Mean f_mcl")
    ax.set_title("Cold Gas Mass Fraction (f_mcl) Evolution"); ax.grid(True, alpha=0.3)
    savefig("fmcl_evolution.png")

    # Distribution at first and last timestep
    fmcl_0 = sim.calc_fmcl(sim.rho[0],    sim.temp[0]).ravel()
    fmcl_f = sim.calc_fmcl(sim.rho[-1],   sim.temp[-1]).ravel()
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 50)
    ax.hist(fmcl_0, bins=bins, density=True, alpha=0.6, label="t=0",  histtype="stepfilled")
    ax.hist(fmcl_f, bins=bins, density=True, alpha=0.6, label="t=end",histtype="stepfilled")
    ax.set_xlabel("f_mcl"); ax.set_ylabel("Normalized count")
    ax.set_title("f_mcl Distribution: Initial vs. Final"); ax.legend()
    savefig("fmcl_histogram.png")


# ===========================================================================
# PLOT 6 — Source term statistics
# ===========================================================================
def plot_source_terms(sim):
    print("\n  Computing source terms …")
    try:
        source_term = sim.calc_all_source_terms()  # (nt, 5, nx, ny)
    except Exception as e:
        print(f"  [skip] Source term calculation failed: {e}")
        return

    channel_labels = ["Density", "Momentum X", "Momentum Y", "Energy", "Cold Density"]
    nt = source_term.shape[0]

    # --- magnitude over time ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for ch, label in enumerate(channel_labels):
        mag = np.abs(source_term[:, ch]).mean(axis=(1, 2))
        ax.plot(mag, label=label)
    ax.set_xlabel("Timestep"); ax.set_ylabel("|Source Term| (mean)")
    ax.set_title("Source Term Magnitudes Over Time")
    ax.legend(); ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    savefig("source_term_magnitudes.png")

    # --- snapshot at t=0, mid, end ---
    for t_idx, t_label in [(0, "t0"), (nt // 2, "tmid"), (nt - 1, "tend")]:
        fig, axs = plt.subplots(1, 5, figsize=(20, 3))
        for ch in range(5):
            data = source_term[t_idx, ch]
            vmax = np.percentile(np.abs(data), 98)
            im = axs[ch].imshow(data, origin="lower", cmap="coolwarm",
                                 vmin=-vmax, vmax=vmax)
            axs[ch].set_title(channel_labels[ch])
            plt.colorbar(im, ax=axs[ch], fraction=0.046)
        fig.suptitle(f"Source Terms — {t_label}")
        savefig(f"source_term_snapshot_{t_label}.png")

    # --- per-channel std of source term ---
    fig, axs = plt.subplots(1, 5, figsize=(18, 4))
    cg_temp = np.zeros((nt, sim.resolution[0] // sim.down_sample, sim.resolution[1] // sim.down_sample))
    for i in range(nt):
        cg_temp[i] = sim.coarse_grain(sim.temp[i])

    temp_flat = cg_temp.ravel()
    temp_bins = np.logspace(4, 6.5, 100)
    bin_centers = np.sqrt(temp_bins[:-1] * temp_bins[1:])

    for ch, (ax, label) in enumerate(zip(axs, channel_labels)):
        S = source_term[:, ch].ravel()
        inds = np.digitize(temp_flat, temp_bins)
        means, stds = [], []
        for b in range(1, len(temp_bins)):
            mask = inds == b
            if mask.sum() > 0:
                means.append(S[mask].mean()); stds.append(S[mask].std())
            else:
                means.append(np.nan); stds.append(np.nan)
        means, stds = np.array(means), np.array(stds)
        valid = ~np.isnan(stds)
        ax.plot(bin_centers[valid], stds[valid])
        ax.set_xscale("log"); ax.set_xlabel("T [K]"); ax.set_ylabel("σ(Source)")
        ax.set_title(label); ax.grid(True, alpha=0.3)

    fig.suptitle("Source Term Std vs. Temperature")
    savefig("source_term_std_vs_temp.png")


# ===========================================================================
# PLOT 7 — FFT spectral analysis of source terms
# ===========================================================================
def plot_fft(sim):
    print("\n  Computing FFT spectral analysis …")
    try:
        source_term = sim.calc_source_term()   # (nt, nx, ny) — FMCL source term
    except Exception as e:
        print(f"  [skip] FFT calc failed: {e}")
        return

    nt = source_term.shape[0]
    st_x = np.mean(source_term, axis=2)   # avg over Y  → (nt, nx_cg)
    fst  = np.fft.fftshift(np.fft.fft(st_x, axis=1), axes=1)
    k    = np.fft.fftshift(np.fft.fftfreq(st_x.shape[1]))

    t_idxs = min(7, nt)
    colors = plt.cm.plasma(np.linspace(1, 0, t_idxs))
    fig, ax = plt.subplots(figsize=(9, 5))
    for j, t_idx in enumerate(np.linspace(0, nt - 1, t_idxs, dtype=int)):
        ax.plot(np.abs(k), np.abs(fst[t_idx]), color=colors[j],
                label=f"t={5*t_idx/nt:.1f} Myr")
    ax.set_xlim(0, np.max(np.abs(k)))
    ax.set_xlabel(r"$k_x$"); ax.set_ylabel("FFT Amplitude")
    ax.set_title("Source Term FFT Along X")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    savefig("source_term_fft.png")


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Subgrid CGM Model — Benchmark Analysis (log_cnn)")
    print("=" * 60)
    print(f"  Sim bin dir  : {SIM_BIN_DIR}")
    print(f"  Output dir   : {OUTPUT_DIR}")
    print(f"  Hst file     : {HST_FILE}")

    # 1. Load data
    sim = load_sim_data()

    # 2. Run predictions
    pred_pdf = predict_pdf(sim)

    # 3. History plots
    plot_history()

    # 4. Field snapshots
    plot_field_snapshots(sim)

    # 5. PDF comparison
    plot_pdf_comparison(sim, pred_pdf)
    plot_cooling(sim, pred_pdf)

    # 6. f_mcl
    plot_fmcl_evolution(sim)

    # 7. Source terms
    plot_source_terms(sim)

    # 8. FFT
    plot_fft(sim)

    print("\n[6/6] All plots saved to:", OUTPUT_DIR)
    print("Done.")
