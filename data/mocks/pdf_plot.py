# Python script to plot the actual and predicted PDFs using a discrete form for the PDFs (n bins in log temp space)

import os
import sys

import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from models.conv_nn.pdf_cnn import snapshot_pred, snapshot_pred_with_gate, lambda_cool, compute_cooling_rate

# Set PyTorch device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# =========================
# RUN TOGGLES
# =========================
RUN_PDF_ANIMATION = False
RUN_COOLING_SCATTER = True
RUN_COOLING_HISTOGRAM = True
RUN_PDF_COMPARE_ANIMATION = True
RUN_COOLING_COMPARE_ANIMATION = True
RUN_FOURWAY_COMPARE_ANIMATION = True

# =========================
# SETTINGS
# =========================
resolution = (512, 256)
downsample = 32
bins = 40


# =========================
# IMPORT YOUR CLASS
# =========================
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from data_preprocess import simulation_data

# =========================
# QUANTITATIVE METRICS  (Change #4)
# =========================
def print_metrics(true, pred, label):
    """
    Log-space bias, RMSE, and Pearson correlation for cooling rate arrays.
    Only pixels where both true and pred are positive are included.
    """
    mask = (true > 0) & (pred > 0)
    num_pixels = mask.sum()

    print(f"\n--- {label} ---")
    print(f"  Pixels used : {num_pixels} / {true.size}")

    if num_pixels < 2:
        print("  Log-Bias    : N/A (insufficient positive pixels)")
        print("  Log-RMSE    : N/A (insufficient positive pixels)")
        print("  Correlation : N/A (insufficient positive pixels)")
        return

    log_true = np.log10(true[mask])
    log_pred = np.log10(pred[mask])

    bias = np.mean(log_pred - log_true)
    rmse = np.sqrt(np.mean((log_pred - log_true) ** 2))
    corr, _ = pearsonr(log_true, log_pred)

    print(f"  Log-Bias    : {bias:+.3f} dex")
    print(f"  Log-RMSE    :  {rmse:.3f} dex")
    print(f"  Correlation :  {corr:.4f}")


# Define PDF bins and log temperature centers for background color calculations
temp_bins = np.logspace(3.0, 7.0, bins + 1)
log_temp_centers = 0.5 * (np.log10(temp_bins[:-1]) + np.log10(temp_bins[1:]))
cmap = plt.get_cmap("inferno")
norm = colors.Normalize(vmin=3.0, vmax=7.0)

folder_path = f"/Volumes/PortableSSD/Projects/SubgridCGMModel/simulation_outputs/hr_build/cache/sc(512, 256)_32"

PDF_MOCKS_DIR = os.environ.get("PDF_MOCKS_DIR", "mocks/pdf")
os.makedirs(PDF_MOCKS_DIR, exist_ok=True)

mp4_path = os.path.join(PDF_MOCKS_DIR, "pdf_animation.mp4")
first_frame_path = os.path.join(PDF_MOCKS_DIR, "pdf_snapshot_t0.png")
temp_frame_path = os.path.join(PDF_MOCKS_DIR, "temp_snapshot_t0.png")


# =========================
# LOAD DATA
# =========================
print("Loading data...")

sim_data = simulation_data()
sim_data.down_sample = downsample
sim_data.resolution = resolution

sim_data.rho = np.load(f"{folder_path}/rho.npy")
sim_data.temp = np.load(f"{folder_path}/temp.npy")
sim_data.pressure = np.load(f"{folder_path}/pressure.npy")
sim_data.ux = np.load(f"{folder_path}/ux.npy")
sim_data.uy = np.load(f"{folder_path}/uy.npy")
sim_data.eint = np.load(f"{folder_path}/eint.npy")
sim_data.ps = np.load(f"{folder_path}/ps.npy")

print("Data loaded.")

# Derive the Code's internal Temperature Unit factor ONCE globally
# T_phys = T_unit * (P_code / rho_code), so T_unit = T_phys * (rho_code / P_code)
# (If your code uses v0 = 1 km/s and mu = 0.62, this will perfectly equal ~75.0 K)
mu = 0.62
T_unit = float(np.median(sim_data.temp[0] * sim_data.rho[0] / sim_data.pressure[0]))
print(f"Derived internal T_unit = {T_unit:.2f} K")


# =========================
# COMPUTE PDF
# =========================
print("Computing pixel PDFs...")

temp_pdf = sim_data.calc_pixel_pdf(bins=bins)
temp_pdf /= temp_pdf.sum(axis=1, keepdims=True) + 1e-12

nt, nb, nx, ny = temp_pdf.shape
print(f"Shape: nt={nt}, bins={nb}, nx={nx}, ny={ny}")

# ─── Predict CNN temperature PDFs, gate, and vorticity ───
conv_temp_pdf = np.zeros_like(temp_pdf)
cnn_gate_maps = np.zeros((nt, nx, ny))  # gate ∈ (0, 1) per cell
cnn_vort_maps = np.zeros((nt, nx, ny))  # |ω| per cell
for i in tqdm(range(temp_pdf.shape[0]), desc="Predicting CNN temperature PDFs"):
    pdf_i, gate_i, vort_i = snapshot_pred_with_gate(
        sim_data.rho[i],
        sim_data.temp[i],
        sim_data.pressure[i],
        sim_data.ux[i],
        sim_data.uy[i],
        sim_data.eint[i],
        sim_data.ps[i],
        downsample,
        resolution,
    )
    conv_temp_pdf[i] = pdf_i
    cnn_gate_maps[i] = gate_i
    cnn_vort_maps[i] = vort_i
conv_temp_pdf /= conv_temp_pdf.sum(axis=1, keepdims=True) + 1e-12


# =========================
# COARSE-GRAIN TEMP
# =========================
print("Computing coarse-grained temperature...")

cg_rho = np.zeros((nt, nx, ny))
cg_temp = np.zeros((nt, nx, ny))

for t in tqdm(range(nt), desc="Coarse-graining temperature & density"):
    cg_rho[t] = sim_data.coarse_grain(sim_data.rho[t])
    cg_temp[t] = sim_data.coarse_grain(sim_data.temp[t])


# =========================
# FIGURE SETUP
# =========================
if RUN_PDF_ANIMATION:
    fig = plt.figure(figsize=(ny * 2.2, nx * 1.8))

    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])

    # ---- PDF GRID ----
    pdf_axes = np.empty((nx, ny), dtype=object)
    sub_gs = gs[0].subgridspec(nx, ny)

    for i in range(nx):
        for j in range(ny):
            ax = fig.add_subplot(sub_gs[i, j])
            pdf_axes[i, j] = ax

            # square borders
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("grey")
                spine.set_linewidth(0.3)

            ax.set_xticks([])
            ax.set_yticks([])

    # ---- TEMP PANEL ----
    temp_ax = fig.add_subplot(gs[1])

    log_temp0 = np.log10(cg_temp[0] + 1e-8)
    temp_im = temp_ax.imshow(log_temp0, origin="lower", cmap="inferno", norm=norm)

    temp_ax.set_title(r"$\log_{10}$ Temp (CG)")
    cbar = plt.colorbar(temp_im, ax=temp_ax, fraction=0.046)
    cbar.set_label(r"$\log_{10}$ Temperature / Expectation Value", fontsize=24)
    cbar.ax.tick_params(labelsize=18)

    # =========================
    # INIT LINES
    # =========================
    x = np.arange(nb)

    lines = []
    for i in range(nx):
        row = []
        for j in range(ny):
            (line,) = pdf_axes[i, j].plot([], [], lw=1)
            pdf_axes[i, j].set_xlim(0, nb - 1)
            pdf_axes[i, j].set_ylim(0, 1)
            row.append(line)
        lines.append(row)

    # =========================
    # INIT FUNCTION
    # =========================
    def init():
        for i in range(nx):
            for j in range(ny):
                lines[i][j].set_data([], [])
                pdf_axes[i, j].set_facecolor("black")
        return sum(lines, [])

    # =========================
    # UPDATE FUNCTION
    # =========================
    def update(frame):

        pdf = temp_pdf[frame]

        for i in range(nx):
            for j in range(ny):
                # FIX: flip vertically to match imshow
                ii = nx - 1 - i

                y = pdf[:, ii, j]

                # Compute expectation value of log10 temperature
                exp_val = np.sum(y * log_temp_centers)

                y = y / (y.max() + 1e-8)

                lines[i][j].set_data(x, y)

                # Set background color of subplot based on expectation value
                bg_color = cmap(norm(exp_val))
                pdf_axes[i, j].set_facecolor(bg_color)

                # Determine dynamic contrasting line color (luminance-based)
                lum = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
                lines[i][j].set_color("white" if lum < 0.5 else "black")

        # update temp (log scale CG)
        log_temp = np.log10(cg_temp[frame] + 1e-8)
        temp_im.set_data(log_temp)

        fig.suptitle(f"t = {frame}", fontsize=48)

        # Save first frame
        if frame == 0:
            fig.savefig(first_frame_path, dpi=300)
            plt.imsave(temp_frame_path, log_temp, cmap="inferno")
            print(f"Saved PDF snapshot → {first_frame_path}")
            print(f"Saved temp snapshot → {temp_frame_path}")

        return sum(lines, []) + [temp_im]

    # =========================
    # ANIMATION
    # =========================
    print("Creating animation...")

    anim = animation.FuncAnimation(fig, update, frames=nt, init_func=init, blit=False)

    print("Saving MP4...")
    with tqdm(total=nt, desc="Saving MP4") as pbar:
        anim.save(
            mp4_path,
            writer="ffmpeg",
            fps=10,
            progress_callback=lambda i, n: pbar.update(1),
        )

    print(f"Saved animation → {mp4_path}")

    plt.close()


# =========================
# TRUE vs PRED PDF COMPARISON
# =========================
# =====================================================================
# COOLING COMPUTATION BLOCK  (Changes #1, #2)
# Three separate cooling fields to isolate model error vs closure error
# =====================================================================
print("Computing cooling rates (True Fine, True Isobaric, CNN Isobaric)...")

# Shared constants — identical across all three computations
mu = 0.62
kb = 1.380649e-16
unit_fix = 1.975e27

# ---- PDF bin centres (geometric mean of edges) ----
temp_bins = np.logspace(3, 7, nb + 1)
temp_centers = np.sqrt(temp_bins[:-1] * temp_bins[1:])  # (nb,)

# ---- Active cooling window for shading (Change #3) ----
active_bin_start = np.searchsorted(temp_centers, 10**4.5)
active_bin_end = np.searchsorted(temp_centers, 10**5.5)

# ------------------------------------------------------------------
# (A) Cool_True_Fine : fine-grid truth averaged to coarse blocks
#     Uses compute_cooling_rate() in scalar mode — no approximation.
# ------------------------------------------------------------------
print("  (A) True Fine cooling...")
true_cool = np.zeros((nt, nx, ny))
for t in tqdm(range(nt), desc="True fine cooling"):
    rho = sim_data.rho[t]
    temp = sim_data.temp[t]
    n = rho / mu
    lam = lambda_cool(temp, mask=True)
    fine_cool = lam * n**2 * unit_fix
    true_cool[t] = sim_data.coarse_grain(fine_cool)


# ------------------------------------------------------------------
# (B) Coarse-grain pressure (shared by both isobaric variants)
# ------------------------------------------------------------------
cg_pressure = np.zeros((nt, nx, ny))
for t in tqdm(range(nt), desc="Coarse-graining pressure"):
    cg_pressure[t] = sim_data.coarse_grain(sim_data.pressure[t])

# ------------------------------------------------------------------
# (C) Cool_True_Isobaric : use TRUE PDF + isobaric n(T)=P/(kb*T)
#     If this disagrees badly with (A) → isobaric assumption is wrong.
# ------------------------------------------------------------------
print("  (C) True Isobaric cooling (using simulation PDF)...")
true_iso_cool = np.zeros((nt, nx, ny))
for t in tqdm(range(nt), desc="True isobaric cooling"):
    true_iso_cool[t] = compute_cooling_rate(
        temp_pdf[t],  # (nb, nx, ny)  – true PDF
        temp_centers,  # (nb,)
        pressure=cg_pressure[t],
        is_pdf=True,
        is_isobaric=True,
        T_unit=T_unit,
    )

# ------------------------------------------------------------------
# (D) Cool_CNN_Isobaric : use CNN PDF + isobaric n(T)=P/(kb*T)
#     Compared to (C) this isolates the CNN's contribution only.
# ------------------------------------------------------------------
print("  (D) CNN Isobaric cooling (using CNN PDF)...")
cnn_cool = np.zeros((nt, nx, ny))
for t in tqdm(range(nt), desc="CNN isobaric cooling"):
    cnn_cool[t] = compute_cooling_rate(
        conv_temp_pdf[t],  # (nb, nx, ny)  – CNN PDF
        temp_centers,  # (nb,)
        pressure=cg_pressure[t],
        is_pdf=True,
        is_isobaric=True,
        T_unit=T_unit,
    )

print("Cooling computation done.")

# =========================
# METRICS  (Change #4)
# =========================
print("\n=== Quantitative Benchmarking Metrics ===")
# Flatten across all pixels and timesteps for a global assessment
print_metrics(
    true_cool.flatten(),
    true_iso_cool.flatten(),
    "Physics Closure Error  (True Fine vs True Isobaric)",
)
print_metrics(
    true_iso_cool.flatten(),
    cnn_cool.flatten(),
    "CNN Prediction Error   (True Isobaric vs CNN Isobaric)",
)
print_metrics(
    true_cool.flatten(),
    cnn_cool.flatten(),
    "Total Error            (True Fine vs CNN Isobaric)",
)

# =========================
# COARSE-GRAIN TEMPERATURE
# =========================
if RUN_COOLING_SCATTER:
    # For a single timestep, scatter true_iso_cool vs cnn_cool
    # coloured by: (a) CNN active-window mass, (b) true active-window mass
    t = 0
    active_mass_true = temp_pdf[t, active_bin_start:active_bin_end].sum(
        axis=0
    )  # (nx,ny)
    active_mass_cnn = conv_temp_pdf[t, active_bin_start:active_bin_end].sum(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc = axes[0].scatter(active_mass_true.flat, active_mass_cnn.flat, s=2, alpha=0.3)
    axes[0].set_xlabel("True active-window mass")
    axes[0].set_ylabel("CNN active-window mass")
    axes[0].set_title("PDF mass in cooling window")

    sc2 = axes[1].scatter(
        true_iso_cool[t].flat,
        cnn_cool[t].flat,
        c=active_mass_cnn.flat,
        s=2,
        alpha=0.3,
        cmap="hot",
    )
    plt.colorbar(sc2, ax=axes[1], label="CNN active-window mass")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("Cooling scatter coloured by CNN window mass")
    plt.tight_layout()
    plt.show()

    cg_temp = np.zeros((nt, nx, ny))

    for t in tqdm(range(nt), desc="Coarse-graining temperature (global scatter)"):
        cg_temp[t] = sim_data.coarse_grain(sim_data.temp[t])

    # =========================
    # GLOBAL THREE-WAY SCATTER  (Change #2)
    # =========================
    print("Creating three-way cooling scatter plots...")

    from matplotlib.colors import LogNorm

    SCATTER_MIN = 1e0  # Points with x OR y below this threshold are excluded entirely

    # Flatten the three cooling fields (raw values; no eps clipping)
    temp_flat = cg_temp.flatten()
    flat_true = true_cool.flatten()
    flat_true_iso = true_iso_cool.flatten()
    flat_cnn = cnn_cool.flatten()

    fig_sc, axes_sc = plt.subplots(1, 3, figsize=(18, 6))

    _scatter_pairs = [
        (
            flat_true,
            flat_true_iso,
            "True Fine",
            "True Isobaric",
            "Physics Closure Error",
        ),
        (
            flat_true_iso,
            flat_cnn,
            "True Isobaric",
            "CNN Isobaric",
            "CNN Prediction Error",
        ),
        (flat_true, flat_cnn, "True Fine", "CNN Isobaric", "Total Error"),
    ]

    for ax, (xv, yv, xl, yl, title) in zip(axes_sc, _scatter_pairs):
        # Only plot points where BOTH axes are above the threshold
        mask = (xv >= SCATTER_MIN) & (yv >= SCATTER_MIN)
        xm, ym, tm = xv[mask], yv[mask], np.clip(temp_flat[mask], 1e3, None)

        sc = ax.scatter(
            xm,
            ym,
            c=tm,
            s=1,
            alpha=0.2,
            cmap="plasma",
            norm=LogNorm(vmin=1e3, vmax=1e8),
        )
        # Reference line across the plotted range
        if len(xm):
            _lim = [min(xm.min(), ym.min()), max(xm.max(), ym.max())]
            ax.plot(_lim, _lim, "r--", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"{title}\n({mask.sum():,} / {len(xv):,} points)")
        plt.colorbar(sc, ax=ax, label="Temperature (K)")

    fig_sc.suptitle("Cooling Rate Comparisons (All Pixels, All Timesteps)", fontsize=16)
    fig_sc.tight_layout()
    fig_sc.savefig(
        os.path.join(PDF_MOCKS_DIR, "pdf_cooling_scatter_threeway.png"), dpi=200
    )
    plt.show()
    print("Saved three-way scatter plot.")


# ============================================================
# HISTOGRAM: Zero-Fraction + Positive-Only log10 (Change #5)
# ============================================================
if RUN_COOLING_HISTOGRAM:
    print("Creating improved histogram plots...")

    _fields = {
        "True Fine": true_cool.flatten(),
        "True Isobaric": true_iso_cool.flatten(),
        "CNN Isobaric": cnn_cool.flatten(),
    }
    _colors = ["steelblue", "darkorange", "mediumseagreen"]

    fig_hist, (ax_zero, ax_pos) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Panel 1: Zero-fraction bar chart ----
    zero_fracs = [np.mean(v == 0.0) * 100 for v in _fields.values()]
    bars = ax_zero.bar(_fields.keys(), zero_fracs, color=_colors, width=0.5)
    for bar, frac in zip(bars, zero_fracs):
        ax_zero.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{frac:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax_zero.set_ylabel("Fraction of pixels with cooling = 0 (%)")
    ax_zero.set_title("Zero-Cooling Fraction")
    ax_zero.set_ylim(0, max(zero_fracs) * 1.25 + 1)

    # ---- Panel 2: log10(cooling) distribution for positive pixels ----
    for (label, vals), col in zip(_fields.items(), _colors):
        pos = vals[vals > 0]
        if len(pos) == 0:
            continue
        log_pos = np.log10(pos)
        ax_pos.hist(
            log_pos,
            bins=80,
            density=True,
            histtype="step",
            linewidth=2,
            label=label,
            color=col,
        )

    ax_pos.set_xlabel(r"$\log_{10}$(Cooling Rate)")
    ax_pos.set_ylabel("Probability Density")
    ax_pos.set_title(r"Distribution of Positive $\log_{10}$(Cooling)")
    ax_pos.legend()
    ax_pos.set_yscale("log")

    fig_hist.suptitle("Cooling Rate Histograms", fontsize=14)
    fig_hist.tight_layout()
    fig_hist.savefig(os.path.join(PDF_MOCKS_DIR, "pdf_cooling_histogram.png"), dpi=200)
    plt.show()
    print("Saved histogram plot.")

# =========================
# TRUE vs PRED PDF ANIMATION
# =========================
if RUN_PDF_COMPARE_ANIMATION:
    print("Creating TRUE vs PRED PDF comparison animation...")

    mp4_path_compare = os.path.join(PDF_MOCKS_DIR, "pdf_compare_animation.mp4")
    snapshot_compare_path = os.path.join(PDF_MOCKS_DIR, "pdf_compare_t0.png")

    fig2 = plt.figure(figsize=(ny * 4.0, nx * 1.8))
    gs2 = fig2.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], top=0.90, wspace=0.15)

    # Add section titles for True vs Predicted groups
    fig2.text(
        0.24,
        0.92,
        "TRUE PDFs (Simulation)",
        fontsize=36,
        ha="center",
        va="center",
        weight="bold",
    )
    fig2.text(
        0.72,
        0.92,
        "PREDICTED PDFs (CNN Model)",
        fontsize=36,
        ha="center",
        va="center",
        weight="bold",
    )

    # ---- LEFT (TRUE) ----
    true_axes = np.empty((nx, ny), dtype=object)
    sub_gs_left = gs2[0].subgridspec(nx, ny)

    for i in range(nx):
        for j in range(ny):
            ax = fig2.add_subplot(sub_gs_left[i, j])
            true_axes[i, j] = ax

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("grey")
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
                spine.set_color("grey")
                spine.set_linewidth(0.3)

            ax.set_xticks([])
            ax.set_yticks([])

    # ---- COLORBAR AXIS ----
    cbar_ax2 = fig2.add_subplot(gs2[2])
    sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm2.set_array([])
    cbar2 = fig2.colorbar(sm2, cax=cbar_ax2)
    cbar2.set_label(r"Expectation Value of $\log_{10}$ Temperature", fontsize=36)
    cbar2.ax.tick_params(labelsize=28)

    # =========================
    # LINES + TEXT
    # =========================
    true_lines, pred_lines = [], []
    true_texts, pred_texts = [], []

    for i in range(nx):
        row_tl, row_pl = [], []
        row_tt, row_pt = [], []

        for j in range(ny):
            (lt,) = true_axes[i, j].plot([], [], lw=1)
            (lp,) = pred_axes[i, j].plot([], [], lw=1, color="r")

            # Change #3: log-scale y-axis so cooling-window tails are visible
            true_axes[i, j].set_xlim(0, nb - 1)
            true_axes[i, j].set_yscale("log")
            true_axes[i, j].set_ylim(1e-5, 1.1)

            pred_axes[i, j].set_xlim(0, nb - 1)
            pred_axes[i, j].set_yscale("log")
            pred_axes[i, j].set_ylim(1e-5, 1.1)

            # TEXT
            ttxt = true_axes[i, j].text(
                0.95,
                0.95,
                "",
                transform=true_axes[i, j].transAxes,
                fontsize=6,
                color="black",
                ha="right",
                va="top",
            )

            ptxt = pred_axes[i, j].text(
                0.95,
                0.95,
                "",
                transform=pred_axes[i, j].transAxes,
                fontsize=6,
                color="black",
                ha="right",
                va="top",
            )

            row_tl.append(lt)
            row_pl.append(lp)
            row_tt.append(ttxt)
            row_pt.append(ptxt)

        true_lines.append(row_tl)
        pred_lines.append(row_pl)
        true_texts.append(row_tt)
        pred_texts.append(row_pt)

    x = np.arange(nb)

    # =========================
    # INIT
    # =========================
    def init_compare():
        for i in range(nx):
            for j in range(ny):
                true_lines[i][j].set_data([], [])
                pred_lines[i][j].set_data([], [])
                true_texts[i][j].set_text("")
                pred_texts[i][j].set_text("")
                true_axes[i, j].set_facecolor("black")
                pred_axes[i, j].set_facecolor("black")
        return sum(true_lines, []) + sum(pred_lines, [])

    # =========================
    # UPDATE
    # =========================
    def update_compare(frame):

        true_pdf = temp_pdf[frame]
        pred_pdf = conv_temp_pdf[frame]

        for i in range(nx):
            for j in range(ny):
                ii = nx - 1 - i

                y_true = true_pdf[:, ii, j]
                y_pred = pred_pdf[:, ii, j]

                # Expectation values before scaling for plotting
                exp_val_true = np.sum(y_true * log_temp_centers)
                exp_val_pred = np.sum(y_pred * log_temp_centers)

                # Change #3: add floor before log-scale plotting; DO NOT renorm by max()
                y_true_plot = y_true + 1e-8
                y_pred_plot = y_pred + 1e-8

                true_lines[i][j].set_data(x, y_true_plot)
                pred_lines[i][j].set_data(x, y_pred_plot)

                # Set background color of subplot based on expectation value
                bg_true = cmap(norm(exp_val_true))
                bg_pred = cmap(norm(exp_val_pred))
                true_axes[i, j].set_facecolor(bg_true)
                pred_axes[i, j].set_facecolor(bg_pred)

                # Determine dynamic contrasting line and text color
                lum_true = 0.299 * bg_true[0] + 0.587 * bg_true[1] + 0.114 * bg_true[2]
                true_color = "white" if lum_true < 0.5 else "black"
                true_lines[i][j].set_color(true_color)
                true_texts[i][j].set_color(true_color)

                lum_pred = 0.299 * bg_pred[0] + 0.587 * bg_pred[1] + 0.114 * bg_pred[2]
                pred_color = "white" if lum_pred < 0.5 else "black"
                pred_lines[i][j].set_color(pred_color)
                pred_texts[i][j].set_color(pred_color)

                # Change #3: shade the active cooling window on both panels
                for ax in [true_axes[i, j], pred_axes[i, j]]:
                    # Remove old span patches (avoid stacking)
                    for patch in list(ax.patches):
                        patch.remove()
                    ax.axvspan(
                        active_bin_start,
                        active_bin_end,
                        color="green",
                        alpha=0.12,
                        lw=0,
                    )

                # ---- cooling values: show all three ----
                tc = true_cool[frame, ii, j]
                tic = true_iso_cool[frame, ii, j]
                pc = cnn_cool[frame, ii, j]

                true_texts[i][j].set_text(f"F:{tc:.1e}\nI:{tic:.1e}")
                pred_texts[i][j].set_text(f"{pc:.1e}")

        fig2.suptitle(f"True vs Predicted PDFs | t = {frame}", fontsize=48, y=0.96)

        if frame == 0:
            fig2.savefig(snapshot_compare_path, dpi=300)
            print(f"Saved comparison snapshot → {snapshot_compare_path}")

        return (
            sum(true_lines, [])
            + sum(pred_lines, [])
            + sum(true_texts, [])
            + sum(pred_texts, [])
        )

    # =========================
    # ANIMATION
    # =========================
    anim2 = animation.FuncAnimation(
        fig2, update_compare, frames=nt, init_func=init_compare, blit=False
    )

    print("Saving comparison MP4...")
    with tqdm(total=nt, desc="Saving comparison MP4") as pbar:
        anim2.save(
            mp4_path_compare,
            writer="ffmpeg",
            fps=10,
            progress_callback=lambda i, n: pbar.update(1),
        )

    print(f"Saved comparison animation → {mp4_path_compare}")

    plt.close(fig2)


# ============================================================
# COOLING COMPARISON ANIMATION (NEW)
# ============================================================
if RUN_COOLING_COMPARE_ANIMATION:
    print("Creating cooling comparison animation...")

    mp4_path_cooling_compare = os.path.join(
        PDF_MOCKS_DIR, "pdf_cooling_compare_animation.mp4"
    )
    snapshot_cooling_compare_path = os.path.join(
        PDF_MOCKS_DIR, "pdf_cooling_compare_t0.png"
    )

    # Setup figure 3 (aspect ratio matched to nx/ny)
    fig3 = plt.figure(figsize=(12, 6))
    gs3 = fig3.add_gridspec(
        1,
        3,
        width_ratios=[1, 1, 0.05],
        top=0.85,
        bottom=0.15,
        left=0.08,
        right=0.90,
        wspace=0.15,
    )

    # Add section titles
    fig3.text(
        0.28,
        0.90,
        "TRUE ISOBARIC COOLING (Simulation)",
        fontsize=16,
        ha="center",
        va="center",
        weight="bold",
    )
    fig3.text(
        0.70,
        0.90,
        "PREDICTED COOLING (CNN Model)",
        fontsize=16,
        ha="center",
        va="center",
        weight="bold",
    )

    # Compute global vmin/vmax for cooling rates
    all_pos_cool = np.concatenate(
        [true_iso_cool[true_iso_cool > 0], cnn_cool[cnn_cool > 0]]
    )
    if len(all_pos_cool) > 0:
        cool_vmin = max(np.percentile(all_pos_cool, 1), 1e-10)
        cool_vmax = np.percentile(all_pos_cool, 99)
    else:
        cool_vmin = 1e-28
        cool_vmax = 1e-18

    norm_cool = colors.LogNorm(vmin=cool_vmin, vmax=cool_vmax)
    cmap_cool = plt.get_cmap("viridis")

    # ---- LEFT (TRUE) ----
    ax_t = fig3.add_subplot(gs3[0])
    ax_t.set_title("True Isobaric", fontsize=14)
    # ---- RIGHT (PRED) ----
    ax_p = fig3.add_subplot(gs3[1])
    ax_p.set_title("CNN Prediction", fontsize=14)

    # Plot initial frames
    im_t = ax_t.imshow(
        np.clip(true_iso_cool[0], cool_vmin, None),
        origin="lower",
        cmap=cmap_cool,
        norm=norm_cool,
    )
    im_p = ax_p.imshow(
        np.clip(cnn_cool[0], cool_vmin, None),
        origin="lower",
        cmap=cmap_cool,
        norm=norm_cool,
    )

    for ax in [ax_t, ax_p]:
        ax.set_xlabel("Y (pixels)", fontsize=12)
        ax.set_ylabel("X (pixels)", fontsize=12)

    # ---- COLORBAR AXIS ----
    cbar_ax3 = fig3.add_subplot(gs3[2])
    sm3 = plt.cm.ScalarMappable(cmap=cmap_cool, norm=norm_cool)
    sm3.set_array([])
    cbar3 = fig3.colorbar(sm3, cax=cbar_ax3)
    cbar3.set_label("Cooling Rate (erg / cm$^3$ / s)", fontsize=14)
    cbar3.ax.tick_params(labelsize=10)

    def init_cooling_compare():
        im_t.set_data(np.clip(true_iso_cool[0], cool_vmin, None))
        im_p.set_data(np.clip(cnn_cool[0], cool_vmin, None))
        return [im_t, im_p]

    def update_cooling_compare(frame):
        # Update imshow data
        im_t.set_data(np.clip(true_iso_cool[frame], cool_vmin, None))
        im_p.set_data(np.clip(cnn_cool[frame], cool_vmin, None))

        fig3.suptitle(f"Cooling Rate Comparison | t = {frame}", fontsize=18, y=0.96)

        if frame == 0:
            fig3.savefig(snapshot_cooling_compare_path, dpi=300)
            print(
                f"Saved cooling comparison snapshot → {snapshot_cooling_compare_path}"
            )

        return [im_t, im_p]

    anim3 = animation.FuncAnimation(
        fig3,
        update_cooling_compare,
        frames=nt,
        init_func=init_cooling_compare,
        blit=False,
    )

    print("Saving cooling comparison MP4...")
    with tqdm(total=nt, desc="Saving cooling comparison MP4") as pbar:
        anim3.save(
            mp4_path_cooling_compare,
            writer="ffmpeg",
            fps=10,
            progress_callback=lambda i, n: pbar.update(1),
        )

    print(f"Saved cooling comparison animation → {mp4_path_cooling_compare}")
    plt.close(fig3)


# ============================================================
# FOUR-WAY COMPARISON ANIMATION
# Panels: True Cooling | Predicted Cooling | Vorticity | Gate
# ============================================================
if RUN_FOURWAY_COMPARE_ANIMATION:
    print("Creating four-way comparison animation (cooling / vorticity / gate)...")

    mp4_path_fourway = os.path.join(PDF_MOCKS_DIR, "pdf_fourway_compare_animation.mp4")
    snapshot_fourway_path = os.path.join(PDF_MOCKS_DIR, "pdf_fourway_compare_t0.png")

    # ---- Shared colormap + norm for cooling panels ----
    all_pos_cool4 = np.concatenate([true_cool[true_cool > 0], cnn_cool[cnn_cool > 0]])
    if len(all_pos_cool4) > 0:
        cool4_vmin = max(np.percentile(all_pos_cool4, 1), 1e-10)
        cool4_vmax = np.percentile(all_pos_cool4, 99)
    else:
        cool4_vmin, cool4_vmax = 1e-28, 1e-18
    norm_cool4 = colors.LogNorm(vmin=cool4_vmin, vmax=cool4_vmax)
    cmap_cool4 = plt.get_cmap("magma")

    # ---- Vorticity: symmetric linear norm across all timesteps ----
    vort_abs_max = np.percentile(np.abs(cnn_vort_maps), 99)
    vort_abs_max = max(vort_abs_max, 1e-30)
    norm_vort = colors.Normalize(vmin=-vort_abs_max, vmax=vort_abs_max)
    cmap_vort = plt.get_cmap("RdBu_r")

    # ---- Gate: always in [0, 1] ----
    norm_gate = colors.Normalize(vmin=0.0, vmax=1.0)
    cmap_gate = plt.get_cmap("viridis")

    # ---- Build figure: 4 image panels + 4 colorbars ----
    fig4 = plt.figure(figsize=(22, 6))
    fig4.patch.set_facecolor("#0d0d0d")

    # gridspec: 4 image cols + 4 narrow cbar cols
    gs4 = fig4.add_gridspec(
        1,
        8,
        width_ratios=[1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05],
        top=0.82,
        bottom=0.12,
        left=0.05,
        right=0.97,
        wspace=0.05,
    )

    panel_specs = [
        (
            gs4[0],
            gs4[1],
            "True Cooling",
            cmap_cool4,
            norm_cool4,
            "Cooling Rate\n(erg / cm³ / s)",
        ),
        (
            gs4[2],
            gs4[3],
            "Predicted Cooling",
            cmap_cool4,
            norm_cool4,
            "Cooling Rate\n(erg / cm³ / s)",
        ),
        (gs4[4], gs4[5], "Vorticity ω", cmap_vort, norm_vort, "Vorticity (code units)"),
        (gs4[6], gs4[7], "Gate g(x,y)", cmap_gate, norm_gate, "Gate value ∈ (0, 1)"),
    ]

    axes4, ims4, cbars4 = [], [], []
    for img_spec, cbar_spec, title, cmap_p, norm_p, cbar_lbl in panel_specs:
        ax = fig4.add_subplot(img_spec)
        ax.set_facecolor("#0d0d0d")
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.set_title(title, fontsize=13, color="white", pad=6, fontweight="bold")
        axes4.append(ax)

        cax = fig4.add_subplot(cbar_spec)
        axes4.append(cax)  # store for reference

        sm = plt.cm.ScalarMappable(cmap=cmap_p, norm=norm_p)
        sm.set_array([])
        cb = fig4.colorbar(sm, cax=cax)
        cb.set_label(cbar_lbl, fontsize=9, color="white")
        cb.ax.tick_params(colors="white", labelsize=7)
        cbars4.append(cb)

    # Unpack only the image axes (indices 0, 2, 4, 6)
    ax_tc, _, ax_pc, _, ax_vt, _, ax_gt, _ = [fig4.axes[k] for k in range(8)]

    def _clip_cool(arr, frame):
        return np.clip(arr[frame], cool4_vmin, None)

    # Initial imshow frames
    im_tc4 = ax_tc.imshow(
        _clip_cool(true_cool, 0), origin="lower", cmap=cmap_cool4, norm=norm_cool4
    )
    im_pc4 = ax_pc.imshow(
        _clip_cool(cnn_cool, 0), origin="lower", cmap=cmap_cool4, norm=norm_cool4
    )
    im_vt4 = ax_vt.imshow(
        cnn_vort_maps[0], origin="lower", cmap=cmap_vort, norm=norm_vort
    )
    im_gt4 = ax_gt.imshow(
        cnn_gate_maps[0], origin="lower", cmap=cmap_gate, norm=norm_gate
    )

    for ax in [ax_tc, ax_pc, ax_vt, ax_gt]:
        ax.set_xlabel("Y (cells)", fontsize=9, color="white")
        ax.set_ylabel("X (cells)", fontsize=9, color="white")

    title4 = fig4.suptitle(
        "True Cooling | Predicted Cooling | Vorticity | Gate   —   t = 0",
        fontsize=15,
        color="white",
        y=0.97,
        fontweight="bold",
    )

    def init_fourway():
        im_tc4.set_data(_clip_cool(true_cool, 0))
        im_pc4.set_data(_clip_cool(cnn_cool, 0))
        im_vt4.set_data(cnn_vort_maps[0])
        im_gt4.set_data(cnn_gate_maps[0])
        return [im_tc4, im_pc4, im_vt4, im_gt4]

    def update_fourway(frame):
        im_tc4.set_data(_clip_cool(true_cool, frame))
        im_pc4.set_data(_clip_cool(cnn_cool, frame))
        im_vt4.set_data(cnn_vort_maps[frame])
        im_gt4.set_data(cnn_gate_maps[frame])
        title4.set_text(
            f"True Cooling | Predicted Cooling | Vorticity | Gate   —   t = {frame}"
        )
        if frame == 0:
            fig4.savefig(snapshot_fourway_path, dpi=300, facecolor=fig4.get_facecolor())
            print(f"Saved four-way snapshot → {snapshot_fourway_path}")
        return [im_tc4, im_pc4, im_vt4, im_gt4]

    anim4 = animation.FuncAnimation(
        fig4, update_fourway, frames=nt, init_func=init_fourway, blit=False
    )

    print("Saving four-way comparison MP4...")
    with tqdm(total=nt, desc="Saving four-way MP4") as pbar:
        anim4.save(
            mp4_path_fourway,
            writer="ffmpeg",
            fps=10,
            progress_callback=lambda i, n: pbar.update(1),
        )

    print(f"Saved four-way comparison animation → {mp4_path_fourway}")
    plt.close(fig4)
