# Python script to plot the actual and predicted PDFs using a discrete form for the PDFs (n bins in log temp space)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
from scipy.stats import pearsonr
from tqdm import tqdm
import sys
import os
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from models.conv_nn.pdf_cnn import snapshot_pred



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
RUN_PDF_ANIMATION = True
RUN_COOLING_SCATTER = True
RUN_COOLING_HISTOGRAM = True
RUN_PDF_COMPARE_ANIMATION = True
RUN_COOLING_COMPARE_ANIMATION = True

# =========================
# SETTINGS
# =========================
resolution = (1024, 512)
downsample = 64
bins = 40


# =========================
# IMPORT YOUR CLASS
# =========================
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from data_preprocess import simulation_data

def lambda_cool(temp):
    """
    Cooling function ISMCoolFn translated from AthenaK C++.
    Works on scalars or numpy arrays (any shape).
    Returns Λ(T) in erg cm^3 / s.
    """
    logt = np.log10(temp)

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

    lam = np.zeros_like(temp, dtype=float)

    # turn off cooling below 1e4 K
    mask_off = logt <= 4.0
    lam[mask_off] = 0.0

    # KI02 regime (4.0 < logT <= 4.2)
    mask_ki = (logt > 4.0) & (logt <= 4.2)
    if np.any(mask_ki):
        lam[mask_ki] = (2.0e-19*np.exp(-1.184e5/(temp[mask_ki] + 1.0e3)) +
                        2.8e-28*np.sqrt(temp[mask_ki])*np.exp(-92.0/temp[mask_ki]))

    # CGOLS fit (logT > 8.15)
    mask_hi = logt > 8.15
    lam[mask_hi] = 10.0**(0.45*logt[mask_hi] - 26.065)

    # SPEX interpolation (4.2 < logT <= 8.15)
    mask_mid = (logt > 4.2) & (logt <= 8.15)
    if np.any(mask_mid):
        ipps = (25.0*logt[mask_mid] - 103).astype(int)
        # Clamp to [0,100] like C++
        ipps = np.clip(ipps, 0, 100)
        x0 = 4.12 + 0.04*ipps
        dx = logt[mask_mid] - x0
        logcool = (lhd[ipps+1]*dx - lhd[ipps]*(dx - 0.04)) * 25.0
        lam[mask_mid] = 10.0**logcool

    mask_off = (logt < 4.5) | (logt > 5.5)
    lam[mask_off] = 0.0

    return lam


def lambda_cool_torch(temp, device):
    """
    Cooling function ISMCoolFn translated from AthenaK C++ in PyTorch.
    Works on PyTorch tensors on the specified device.
    Returns Λ(T) in erg cm^3 / s.
    """
    logt = torch.log10(temp)

    lhd_data = [
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
    ]
    lhd = torch.tensor(lhd_data, dtype=torch.float32, device=device)

    lam = torch.zeros_like(temp, dtype=torch.float32, device=device)

    # turn off cooling below 1e4 K
    mask_off = logt <= 4.0
    lam[mask_off] = 0.0

    # KI02 regime (4.0 < logT <= 4.2)
    mask_ki = (logt > 4.0) & (logt <= 4.2)
    if torch.any(mask_ki):
        temp_ki = temp[mask_ki]
        lam[mask_ki] = (2.0e-19 * torch.exp(-1.184e5 / (temp_ki + 1.0e3)) +
                        2.8e-28 * torch.sqrt(temp_ki) * torch.exp(-92.0 / temp_ki))

    # CGOLS fit (logT > 8.15)
    mask_hi = logt > 8.15
    if torch.any(mask_hi):
        lam[mask_hi] = 10.0**(0.45 * logt[mask_hi] - 26.065)

    # SPEX interpolation (4.2 < logT <= 8.15)
    mask_mid = (logt > 4.2) & (logt <= 8.15)
    if torch.any(mask_mid):
        logt_mid = logt[mask_mid]
        ipps = (25.0 * logt_mid - 103).to(torch.long)
        ipps = torch.clamp(ipps, 0, 100)
        x0 = 4.12 + 0.04 * ipps
        dx = logt_mid - x0
        logcool = (lhd[ipps + 1] * dx - lhd[ipps] * (dx - 0.04)) * 25.0
        lam[mask_mid] = 10.0**logcool

    # Final mask
    mask_off_final = (logt < 4.5) | (logt > 5.5)
    lam[mask_off_final] = 0.0

    return lam


# =========================
# CENTRALIZED COOLING FUNCTION  (Change #1)
# =========================
def compute_cooling_rate(rho_or_pdf, temp, pressure=None, is_pdf=False, is_isobaric=False, T_unit=None):
    """
    Standardized cooling calculation using internal Code Units.
    Both modes calculate an effective `rho_code` and pass it through the exact same physics.
    """
    mu       = 0.62
    unit_fix = 1.975e27  # The grouped conversion (rho_0 * L_0) / (m_H^2 * v_0^3)

    if not is_pdf:
        # --- Mode 1: Fine-grid scalar path ---
        # We ALREADY have the code density.
        rho_eff = rho_or_pdf
        lam = lambda_cool(temp)
        
        n_code = rho_eff / mu
        return lam * (n_code**2) * unit_fix

    else:
        # --- Mode 2: PDF-integrated path ---
        pdf       = rho_or_pdf           # (nb, nx, ny)
        T_centers = temp                 # (nb,)
        lam       = lambda_cool(T_centers)   # (nb,)

        if is_isobaric:
            if T_unit is None:
                raise ValueError("T_unit must be provided for isobaric calculation.")
            
            # Reconstruct the code density that WOULD exist at this temperature under isobaric assumption
            # Formula: rho_code = P_code * (T_unit / T_phys)
            # Shapes : (nx, ny)  * ( scalar / (nb,) ) -> (nb, nx, ny)
            rho_eff = pressure[None, :, :] * (T_unit / T_centers[:, None, None])
        else:
            raise ValueError("Non-isobaric PDF cooling not supported here.")

        n_code = rho_eff / mu
        
        # Now it is mathematically identical to the fine-grid path!
        cooling_per_bin = lam[:, None, None] * (n_code**2) * unit_fix
        
        return np.sum(pdf * cooling_per_bin, axis=0)              # (nx,ny)


# =========================
# QUANTITATIVE METRICS  (Change #4)
# =========================
def print_metrics(true, pred, label):
    """
    Log-space bias, RMSE, and Pearson correlation for cooling rate arrays.
    Only pixels where both true and pred are positive are included.
    """
    mask     = (true > 0) & (pred > 0)
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
    rmse = np.sqrt(np.mean((log_pred - log_true)**2))
    corr, _ = pearsonr(log_true, log_pred)

    print(f"  Log-Bias    : {bias:+.3f} dex")
    print(f"  Log-RMSE    :  {rmse:.3f} dex")
    print(f"  Correlation :  {corr:.4f}")



# Define PDF bins and log temperature centers for background color calculations
temp_bins = np.logspace(3.0, 7.0, bins + 1)
log_temp_centers = 0.5 * (np.log10(temp_bins[:-1]) + np.log10(temp_bins[1:]))
cmap = plt.get_cmap("inferno")
norm = colors.Normalize(vmin=3.0, vmax=7.0)

folder_path = f"/Volumes/PortableSSD/Projects/SubgridCGMModel/simulation_outputs/hr_build/cache/sc(1024, 512)_64"

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
temp_pdf /= (temp_pdf.sum(axis=1, keepdims=True) + 1e-12)

nt, nb, nx, ny = temp_pdf.shape
print(f"Shape: nt={nt}, bins={nb}, nx={nx}, ny={ny}")

# ─── Predict CNN temperature PDFs ───
conv_temp_pdf = np.zeros_like(temp_pdf)
for i in tqdm(range(temp_pdf.shape[0]), desc="Predicting CNN temperature PDFs"):
    conv_temp_pdf[i] = snapshot_pred(
        sim_data.rho[i], sim_data.temp[i], sim_data.pressure[i],
        sim_data.ux[i], sim_data.uy[i], sim_data.eint[i], sim_data.ps[i],
        downsample, resolution
    )
conv_temp_pdf /= (conv_temp_pdf.sum(axis=1, keepdims=True) + 1e-12)



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
    fig = plt.figure(figsize=(ny*2.2, nx*1.8))

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
            line, = pdf_axes[i, j].plot([], [], lw=1)
            pdf_axes[i, j].set_xlim(0, nb-1)
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

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=nt,
        init_func=init,
        blit=False
    )

    print("Saving MP4...")
    with tqdm(total=nt, desc="Saving MP4") as pbar:
        anim.save(mp4_path, writer="ffmpeg", fps=10, progress_callback=lambda i, n: pbar.update(1))

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
mu       = 0.62
kb       = 1.380649e-16
unit_fix = 1.975e27

# ---- PDF bin centres (geometric mean of edges) ----
temp_bins    = np.logspace(3, 7, nb + 1)
temp_centers = np.sqrt(temp_bins[:-1] * temp_bins[1:])  # (nb,)

# ---- Active cooling window for shading (Change #3) ----
active_bin_start = np.searchsorted(temp_centers, 10**4.5)
active_bin_end   = np.searchsorted(temp_centers, 10**5.5)

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
    lam = lambda_cool(temp)
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
        temp_pdf[t],          # (nb, nx, ny)  – true PDF
        temp_centers,         # (nb,)
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
        conv_temp_pdf[t],     # (nb, nx, ny)  – CNN PDF
        temp_centers,         # (nb,)
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
print_metrics(true_cool.flatten(), true_iso_cool.flatten(),
              "Physics Closure Error  (True Fine vs True Isobaric)")
print_metrics(true_iso_cool.flatten(), cnn_cool.flatten(),
              "CNN Prediction Error   (True Isobaric vs CNN Isobaric)")
print_metrics(true_cool.flatten(), cnn_cool.flatten(),
              "Total Error            (True Fine vs CNN Isobaric)")

# =========================
# COARSE-GRAIN TEMPERATURE
# =========================
if RUN_COOLING_SCATTER:

    # For a single timestep, scatter true_iso_cool vs cnn_cool
    # coloured by: (a) CNN active-window mass, (b) true active-window mass
    t = 0
    active_mass_true = temp_pdf[t, active_bin_start:active_bin_end].sum(axis=0)  # (nx,ny)
    active_mass_cnn  = conv_temp_pdf[t, active_bin_start:active_bin_end].sum(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc = axes[0].scatter(active_mass_true.flat, active_mass_cnn.flat, s=2, alpha=0.3)
    axes[0].set_xlabel("True active-window mass")
    axes[0].set_ylabel("CNN active-window mass")
    axes[0].set_title("PDF mass in cooling window")

    sc2 = axes[1].scatter(true_iso_cool[t].flat, cnn_cool[t].flat, 
                        c=active_mass_cnn.flat, s=2, alpha=0.3, cmap='hot')
    plt.colorbar(sc2, ax=axes[1], label="CNN active-window mass")
    axes[1].set_xscale('log'); axes[1].set_yscale('log')
    axes[1].set_title("Cooling scatter coloured by CNN window mass")
    plt.tight_layout(); plt.show()

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
    temp_flat     = cg_temp.flatten()
    flat_true     = true_cool    .flatten()
    flat_true_iso = true_iso_cool.flatten()
    flat_cnn      = cnn_cool     .flatten()

    fig_sc, axes_sc = plt.subplots(1, 3, figsize=(18, 6))

    _scatter_pairs = [
        (flat_true,     flat_true_iso,  "True Fine",     "True Isobaric",   "Physics Closure Error"),
        (flat_true_iso, flat_cnn,       "True Isobaric", "CNN Isobaric",    "CNN Prediction Error"),
        (flat_true,     flat_cnn,       "True Fine",     "CNN Isobaric",    "Total Error"),
    ]

    for ax, (xv, yv, xl, yl, title) in zip(axes_sc, _scatter_pairs):
        # Only plot points where BOTH axes are above the threshold
        mask = (xv >= SCATTER_MIN) & (yv >= SCATTER_MIN)
        xm, ym, tm = xv[mask], yv[mask], np.clip(temp_flat[mask], 1e3, None)

        sc = ax.scatter(xm, ym, c=tm, s=1, alpha=0.2,
                        cmap='plasma', norm=LogNorm(vmin=1e3, vmax=1e8))
        # Reference line across the plotted range
        if len(xm):
            _lim = [min(xm.min(), ym.min()), max(xm.max(), ym.max())]
            ax.plot(_lim, _lim, 'r--', lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"{title}\n({mask.sum():,} / {len(xv):,} points)")
        plt.colorbar(sc, ax=ax, label="Temperature (K)")

    fig_sc.suptitle("Cooling Rate Comparisons (All Pixels, All Timesteps)", fontsize=16)
    fig_sc.tight_layout()
    fig_sc.savefig(os.path.join(PDF_MOCKS_DIR, "pdf_cooling_scatter_threeway.png"), dpi=200)
    plt.show()
    print("Saved three-way scatter plot.")


# ============================================================
# HISTOGRAM: Zero-Fraction + Positive-Only log10 (Change #5)
# ============================================================
if RUN_COOLING_HISTOGRAM:
    print("Creating improved histogram plots...")

    _fields   = {
        "True Fine"    : true_cool    .flatten(),
        "True Isobaric": true_iso_cool.flatten(),
        "CNN Isobaric" : cnn_cool     .flatten(),
    }
    _colors = ['steelblue', 'darkorange', 'mediumseagreen']

    fig_hist, (ax_zero, ax_pos) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Panel 1: Zero-fraction bar chart ----
    zero_fracs = [
        np.mean(v == 0.0) * 100
        for v in _fields.values()
    ]
    bars = ax_zero.bar(_fields.keys(), zero_fracs, color=_colors, width=0.5)
    for bar, frac in zip(bars, zero_fracs):
        ax_zero.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{frac:.1f}%", ha='center', va='bottom', fontsize=11)
    ax_zero.set_ylabel("Fraction of pixels with cooling = 0 (%)")
    ax_zero.set_title("Zero-Cooling Fraction")
    ax_zero.set_ylim(0, max(zero_fracs) * 1.25 + 1)

    # ---- Panel 2: log10(cooling) distribution for positive pixels ----
    for (label, vals), col in zip(_fields.items(), _colors):
        pos = vals[vals > 0]
        if len(pos) == 0:
            continue
        log_pos = np.log10(pos)
        ax_pos.hist(log_pos, bins=80, density=True, histtype='step',
                    linewidth=2, label=label, color=col)

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

    fig2 = plt.figure(figsize=(ny*4.0, nx*1.8))
    gs2 = fig2.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], top=0.90, wspace=0.15)

    # Add section titles for True vs Predicted groups
    fig2.text(0.24, 0.92, "TRUE PDFs (Simulation)", fontsize=36, ha="center", va="center", weight="bold")
    fig2.text(0.72, 0.92, "PREDICTED PDFs (CNN Model)", fontsize=36, ha="center", va="center", weight="bold")

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

            lt, = true_axes[i, j].plot([], [], lw=1)
            lp, = pred_axes[i, j].plot([], [], lw=1, color='r')

            # Change #3: log-scale y-axis so cooling-window tails are visible
            true_axes[i, j].set_xlim(0, nb-1)
            true_axes[i, j].set_yscale("log")
            true_axes[i, j].set_ylim(1e-5, 1.1)

            pred_axes[i, j].set_xlim(0, nb-1)
            pred_axes[i, j].set_yscale("log")
            pred_axes[i, j].set_ylim(1e-5, 1.1)

            # TEXT
            ttxt = true_axes[i, j].text(
                0.95, 0.95, "",
                transform=true_axes[i, j].transAxes,
                fontsize=6,
                color="black",
                ha="right",  
                va="top"     
            )

            ptxt = pred_axes[i, j].text(
                0.95, 0.95, "",
                transform=pred_axes[i, j].transAxes,
                fontsize=6,
                color="black",
                ha="right",
                va="top"
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
                    ax.axvspan(active_bin_start, active_bin_end,
                               color='green', alpha=0.12, lw=0)

                # ---- cooling values: show all three ----
                tc  = true_cool    [frame, ii, j]
                tic = true_iso_cool[frame, ii, j]
                pc  = cnn_cool     [frame, ii, j]

                true_texts[i][j].set_text(f"F:{tc:.1e}\nI:{tic:.1e}")
                pred_texts[i][j].set_text(f"{pc:.1e}")

        fig2.suptitle(f"True vs Predicted PDFs | t = {frame}", fontsize=48, y=0.96)

        if frame == 0:
            fig2.savefig(snapshot_compare_path, dpi=300)
            print(f"Saved comparison snapshot → {snapshot_compare_path}")

        return (
            sum(true_lines, []) +
            sum(pred_lines, []) +
            sum(true_texts, []) +
            sum(pred_texts, [])
        )


    # =========================
    # ANIMATION
    # =========================
    anim2 = animation.FuncAnimation(
        fig2,
        update_compare,
        frames=nt,
        init_func=init_compare,
        blit=False
    )

    print("Saving comparison MP4...")
    with tqdm(total=nt, desc="Saving comparison MP4") as pbar:
        anim2.save(mp4_path_compare, writer="ffmpeg", fps=10, progress_callback=lambda i, n: pbar.update(1))

    print(f"Saved comparison animation → {mp4_path_compare}")

    plt.close(fig2)


# ============================================================
# COOLING COMPARISON ANIMATION (NEW)
# ============================================================
if RUN_COOLING_COMPARE_ANIMATION:
    print("Creating cooling comparison animation...")

    mp4_path_cooling_compare = os.path.join(PDF_MOCKS_DIR, "pdf_cooling_compare_animation.mp4")
    snapshot_cooling_compare_path = os.path.join(PDF_MOCKS_DIR, "pdf_cooling_compare_t0.png")

    # Setup figure 3 (aspect ratio matched to nx/ny)
    fig3 = plt.figure(figsize=(12, 6))
    gs3 = fig3.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], top=0.85, bottom=0.15, left=0.08, right=0.90, wspace=0.15)

    # Add section titles
    fig3.text(0.28, 0.90, "TRUE ISOBARIC COOLING (Simulation)", fontsize=16, ha="center", va="center", weight="bold")
    fig3.text(0.70, 0.90, "PREDICTED COOLING (CNN Model)", fontsize=16, ha="center", va="center", weight="bold")

    # Compute global vmin/vmax for cooling rates
    all_pos_cool = np.concatenate([true_iso_cool[true_iso_cool > 0], cnn_cool[cnn_cool > 0]])
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
    im_t = ax_t.imshow(np.clip(true_iso_cool[0], cool_vmin, None), origin="lower", cmap=cmap_cool, norm=norm_cool)
    im_p = ax_p.imshow(np.clip(cnn_cool[0], cool_vmin, None), origin="lower", cmap=cmap_cool, norm=norm_cool)

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
            print(f"Saved cooling comparison snapshot → {snapshot_cooling_compare_path}")
            
        return [im_t, im_p]

    anim3 = animation.FuncAnimation(
        fig3,
        update_cooling_compare,
        frames=nt,
        init_func=init_cooling_compare,
        blit=False
    )

    print("Saving cooling comparison MP4...")
    with tqdm(total=nt, desc="Saving cooling comparison MP4") as pbar:
        anim3.save(mp4_path_cooling_compare, writer="ffmpeg", fps=10, progress_callback=lambda i, n: pbar.update(1))

    print(f"Saved cooling comparison animation → {mp4_path_cooling_compare}")
    plt.close(fig3)