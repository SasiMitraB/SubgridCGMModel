import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys

# =========================
# IMPORT YOUR CLASS
# =========================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))
from data_preprocess import simulation_data


# =========================
# SETTINGS
# =========================
resolution = (512, 256)
downsample = 32
bins = 64

folder_path = f"/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/datafiles/c{resolution}_128"

gif_path = "mocks/pdf/pdf_animation.gif"
first_frame_path = "mocks/pdf/pdf_snapshot_t0.png"


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

# Normalize safely
temp_pdf /= (temp_pdf.sum(axis=1, keepdims=True) + 1e-12)

nt, nb, nx, ny = temp_pdf.shape

print(f"Shape: nt={nt}, bins={nb}, nx={nx}, ny={ny}")


# =========================
# SETUP FIGURE
# =========================
fig, axes = plt.subplots(nx, ny, figsize=(ny*1.8, nx*1.8))

x = np.arange(nb)

# store line objects for animation
lines = []
for i in range(nx):
    row = []
    for j in range(ny):
        line, = axes[i, j].plot([], [], lw=1)
        axes[i, j].set_xlim(0, nb-1)
        axes[i, j].set_ylim(0, 1)
        axes[i, j].axis("off")
        row.append(line)
    lines.append(row)


# =========================
# INITIAL FRAME
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

            y = pdf[:, i, j]
            y = y / (y.max() + 1e-8)  # normalize for visibility

            lines[i][j].set_data(x, y)

    fig.suptitle(f"Pixel PDFs (t = {frame})", fontsize=12)

    # Save first frame as PNG
    if frame == 0:
        plt.savefig(first_frame_path, dpi=300)
        print(f"Saved first snapshot → {first_frame_path}")

    return sum(lines, [])


# =========================
# CREATE ANIMATION
# =========================
print("Creating animation...")

anim = animation.FuncAnimation(
    fig,
    update,
    frames=nt,
    init_func=init,
    blit=True
)

# =========================
# SAVE GIF
# =========================
print("Saving GIF...")

anim.save(gif_path, writer="pillow", fps=10)

print(f"Saved animation → {gif_path}")

plt.close()