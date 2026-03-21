import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys

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
bins = 200

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

print("Data loaded.")


# =========================
# COMPUTE PDF
# =========================
print("Computing pixel PDFs...")

temp_pdf = sim_data.calc_pixel_pdf(bins=bins)
temp_pdf /= (temp_pdf.sum(axis=1, keepdims=True) + 1e-12)

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