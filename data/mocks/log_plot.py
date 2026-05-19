import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from conv_nn.log_cnn import snapshot_pred

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


# =========================
# SETTINGS
# =========================
resolution = (512, 256)
downsample = 32
bins = 40

folder_path = f"/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/datafiles/c{resolution}_128"

os.makedirs("mocks/pdf", exist_ok=True)

gif_path = "mocks/pdf/log_pdf_animation.gif"
first_frame_path = "mocks/pdf/log_pdf_snapshot_t0.png"
temp_frame_path = "mocks/pdf/log_temp_snapshot_t0.png"


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


# =========================
# COMPUTE PDF
# =========================
print("Computing pixel PDFs...")

temp_pdf = sim_data.calc_pixel_pdf(bins=bins)
temp_pdf /= (temp_pdf.sum(axis=1, keepdims=True) + 1e-12)

conv_temp_pdf = np.zeros_like(temp_pdf)
for i in range(temp_pdf.shape[0]):
    conv_temp_pdf[i] = snapshot_pred(sim_data.rho[i], sim_data.temp[i], sim_data.pressure[i], \
                                        sim_data.ux[i], sim_data.uy[i], sim_data.eint[i], sim_data.ps[i],
                                        downsample, (sim_data.resolution[0], sim_data.resolution[1]))
conv_temp_pdf /= (conv_temp_pdf.sum(axis=1, keepdims=True) + 1e-12)

nt, nb, nx, ny = temp_pdf.shape

print(f"Shape: nt={nt}, bins={nb}, nx={nx}, ny={ny}")


# =========================
# COARSE-GRAIN TEMP
# =========================
print("Computing coarse-grained temperature...")

cg_rho = np.zeros((nt, nx, ny))
cg_temp = np.zeros((nt, nx, ny))

for t in range(nt):
    cg_rho[t] = sim_data.coarse_grain(sim_data.rho[t])
    cg_temp[t] = sim_data.coarse_grain(sim_data.temp[t])


# =========================
# FIGURE SETUP
# =========================
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
            spine.set_linewidth(0.3)

        ax.set_xticks([])
        ax.set_yticks([])

# ---- TEMP PANEL ----
temp_ax = fig.add_subplot(gs[1])

log_temp0 = np.log10(cg_temp[0])
temp_im = temp_ax.imshow(log_temp0, origin="lower", cmap="inferno")

temp_ax.set_title(r"$\log_{10}$ Temp (CG)")
cbar = plt.colorbar(temp_im, ax=temp_ax, fraction=0.046)
cbar.set_label(r"$\log_{10}$ Temperature")


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
            y = y / (y.max() + 1e-8)

            lines[i][j].set_data(x, y)

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

print("Saving GIF...")

anim.save(gif_path, writer="pillow", fps=10)

print(f"Saved animation → {gif_path}")

plt.close()

# =========================
# TRUE vs PRED PDF COMPARISON
# =========================
# =========================
# COOLING COMPUTATION BLOCK
# =========================
print("Computing cooling rates (true vs CNN)...")

mu = 0.62
kb = 1.380649e-16

# ---- IMPORTANT: match your PDF bins ----
temp_bins = np.logspace(3, 7, nb + 1)
temp_centers = 0.5 * (temp_bins[:-1] + temp_bins[1:])  # (bins,)

# =========================
# TRUE COOLING (fine → coarse)
# =========================
true_cool = np.zeros((nt, nx, ny))

for t in range(nt):
    rho = sim_data.rho[t]
    temp = sim_data.temp[t]

    n = rho / mu
    lam = lambda_cool(temp)

    fine_cool = lam * n**2 * 1.975e27
    true_cool[t] = sim_data.coarse_grain(fine_cool)


# =========================
# COARSE-GRAIN PRESSURE
# =========================
cg_pressure = np.zeros((nt, nx, ny))

for t in range(nt):
    cg_pressure[t] = sim_data.coarse_grain(sim_data.pressure[t])


# =========================
# CNN COOLING (PDF + isobaric)
# =========================
cnn_cool = np.zeros((nt, nx, ny))

T = temp_centers[:, None, None]   # (bins,1,1)

for t in range(nt):

    pdf = conv_temp_pdf[t]  # (bins,nx,ny)
    P = cg_pressure[t][None, :, :]  # (1,nx,ny)

    n = P / (kb * T)
    lam = lambda_cool(T)

    cool = lam * n**2 

    cnn_cool[t] = np.sum(pdf * cool, axis=0)

print("Cooling computation done.")

# =========================
# COARSE-GRAIN TEMPERATURE
# =========================
cg_temp = np.zeros((nt, nx, ny))

for t in range(nt):
    cg_temp[t] = sim_data.coarse_grain(sim_data.temp[t])

# =========================
# GLOBAL SCATTER (ALL DATA)
# =========================
print("Creating global cooling scatter plot...")

true_vals = true_cool.flatten()
pred_vals = cnn_cool.flatten()
temp_vals = cg_temp.flatten()   # temperature for coloring

# Avoid log issues
eps = 1e-30
true_vals = np.clip(true_vals, eps, None)
pred_vals = np.clip(pred_vals, eps, None)
temp_vals = np.clip(temp_vals, eps, None)

plt.figure(figsize=(6, 6))

from matplotlib.colors import LogNorm

sc = plt.scatter(true_vals, pred_vals,
                 c=temp_vals,
                 s=2,
                 alpha=0.3,
                 cmap='plasma',
                 norm=LogNorm(vmin=1e3, vmax=1e8))

# =========================
# Annotate points near True ≈ 1
# =========================

# Define window around 1 (in log space this is better)
mask = (true_vals > 0.8) & (true_vals < 1.25)

indices = np.where(mask)[0]

# Randomly pick a few points to avoid clutter
n_annotate = min(10, len(indices))
chosen = np.random.choice(indices, size=n_annotate, replace=False)

for i in chosen:
    plt.text(true_vals[i], pred_vals[i],
             f"{temp_vals[i]:.1e}",
             fontsize=6,
             alpha=0.7)

# y = x reference
plt.plot([1e-1, 1e3], [1e-1, 1e3], 'r--', lw=1)

plt.xscale("log")
plt.yscale("log")

plt.xlim(1e-1, 1e3)
plt.ylim(1e-1, 1e3)

plt.xlabel("True Cooling")
plt.ylabel("Predicted Cooling")
plt.title("Cooling: True vs Predicted (All Pixels, All Timesteps)")

# Colorbar
cbar = plt.colorbar(sc)
cbar.set_label("Temperature (K)")

plt.tight_layout()
plt.savefig("mocks/pdf/log_cooling_scatter_global.png", dpi=300)
plt.show()

print("Saved global scatter plot.")

# ============================================================
# PDF DISTRIBUTION PLOT
# ============================================================

print("Creating PDF distribution plot...")

plt.figure(figsize=(7, 5))

# Log-spaced bins
bins = np.logspace(
    np.log10(min(true_vals.min(), pred_vals.min())),
    np.log10(max(true_vals.max(), pred_vals.max())),
    100
)

# True distribution
plt.hist(
    true_vals,
    bins=bins,
    density=True,
    histtype='step',
    linewidth=2,
    label='True Cooling'
)

# Predicted distribution
plt.hist(
    pred_vals,
    bins=bins,
    density=True,
    histtype='step',
    linewidth=2,
    label='Predicted Cooling'
)

plt.xscale("log")
plt.yscale("log")

plt.xlim(1e0, 1e3)

plt.xlabel("Cooling")
plt.ylabel("PDF")
plt.title("Cooling PDF Distribution")

plt.legend()

plt.tight_layout()
plt.savefig("mocks/pdf/log_cooling_pdf_global.png", dpi=300)
plt.show()

print("Saved PDF distribution plot.")

# =========================
# TRUE vs PRED PDF ANIMATION
# =========================
print("Creating TRUE vs PRED PDF comparison animation...")

gif_path_compare = "mocks/pdf/log_pdf_compare_animation.gif"
snapshot_compare_path = "mocks/pdf/log_pdf_compare_t0.png"

fig2 = plt.figure(figsize=(ny*4.0, nx*1.8))
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

        true_axes[i, j].set_xlim(0, nb-1)
        true_axes[i, j].set_ylim(0, 1)

        pred_axes[i, j].set_xlim(0, nb-1)
        pred_axes[i, j].set_ylim(0, 1)

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

            y_true /= (y_true.max() + 1e-8)
            y_pred /= (y_pred.max() + 1e-8)

            true_lines[i][j].set_data(x, y_true)
            pred_lines[i][j].set_data(x, y_pred)

            # ---- cooling values ----
            tc = true_cool[frame, ii, j]
            pc = cnn_cool[frame, ii, j]

            true_texts[i][j].set_text(f"{tc:.1e}")
            pred_texts[i][j].set_text(f"{pc:.1e}")

    fig2.suptitle(f"True vs Predicted PDFs | t = {frame}", fontsize=48)

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

print("Saving comparison GIF...")

anim2.save(gif_path_compare, writer="pillow", fps=10)

print(f"Saved comparison animation → {gif_path_compare}")

plt.close(fig2)