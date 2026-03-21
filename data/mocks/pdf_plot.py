import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from conv_nn.pdf_cnn import snapshot_pred

# =========================
# IMPORT YOUR CLASS
# =========================
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from data_preprocess import simulation_data


# =========================
# SETTINGS
# =========================
resolution = (512, 256)
downsample = 32
bins = 20

folder_path = f"/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/datafiles/c{resolution}_128"

os.makedirs("mocks/pdf", exist_ok=True)

gif_path = "mocks/pdf/pdf_animation.gif"
first_frame_path = "mocks/pdf/pdf_snapshot_t0.png"
temp_frame_path = "mocks/pdf/temp_snapshot_t0.png"


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
print("Creating TRUE vs PRED PDF comparison animation...")

gif_path_compare = "mocks/pdf/pdf_compare_animation.gif"
snapshot_compare_path = "mocks/pdf/pdf_compare_t0.png"

# ---- FIGURE ----
fig2 = plt.figure(figsize=(ny*4.0, nx*1.8))

gs2 = fig2.add_gridspec(1, 2, width_ratios=[1, 1])

# LEFT = TRUE
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

# RIGHT = PRED
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

# ---- INIT LINES ----
true_lines = []
pred_lines = []

for i in range(nx):
    row_true = []
    row_pred = []
    for j in range(ny):

        lt, = true_axes[i, j].plot([], [], lw=1)
        lp, = pred_axes[i, j].plot([], [], lw=1, color='r')

        true_axes[i, j].set_xlim(0, nb-1)
        true_axes[i, j].set_ylim(0, 1)

        pred_axes[i, j].set_xlim(0, nb-1)
        pred_axes[i, j].set_ylim(0, 1)

        row_true.append(lt)
        row_pred.append(lp)

    true_lines.append(row_true)
    pred_lines.append(row_pred)


# =========================
# INIT
# =========================
def init_compare():
    for i in range(nx):
        for j in range(ny):
            true_lines[i][j].set_data([], [])
            pred_lines[i][j].set_data([], [])
    return sum(true_lines, []) + sum(pred_lines, [])


# =========================
# UPDATE
# =========================
def update_compare(frame):

    true_pdf = temp_pdf[frame]
    pred_pdf = conv_temp_pdf[frame]

    for i in range(nx):
        for j in range(ny):

            ii = nx - 1 - i  # match orientation

            y_true = true_pdf[:, ii, j]
            y_pred = pred_pdf[:, ii, j]

            # normalize visually
            y_true = y_true / (y_true.max() + 1e-8)
            y_pred = y_pred / (y_pred.max() + 1e-8)

            true_lines[i][j].set_data(x, y_true)
            pred_lines[i][j].set_data(x, y_pred)

    fig2.suptitle(f"True vs Predicted PDFs | t = {frame}", fontsize=48)

    # Save t0 snapshot
    if frame == 0:
        fig2.savefig(snapshot_compare_path, dpi=300)
        print(f"Saved comparison snapshot → {snapshot_compare_path}")

    return sum(true_lines, []) + sum(pred_lines, [])


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
