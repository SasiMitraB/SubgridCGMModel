import numpy as np
import matplotlib.pyplot as plt
import imageio
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from conv_nn.pdf_cnn import snapshot_pred
from data_preprocess import simulation_data


# =========================
# SETTINGS
# =========================
resolution = (512, 256)
downsample = 32
bins = 40

folder_path = f"/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/datafiles/c{resolution}_128"

os.makedirs("mocks/pdf", exist_ok=True)

video_path = "mocks/pdf/pdf_animation.mp4"
video_compare_path = "mocks/pdf/pdf_compare_animation.mp4"

first_frame_path = "mocks/pdf/pdf_snapshot_t0.png"
temp_frame_path = "mocks/pdf/temp_snapshot_t0.png"
snapshot_compare_path = "mocks/pdf/pdf_compare_t0.png"


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
    conv_temp_pdf[i] = snapshot_pred(
        sim_data.rho[i], sim_data.temp[i], sim_data.pressure[i],
        sim_data.ux[i], sim_data.uy[i], sim_data.eint[i], sim_data.ps[i],
        downsample, (sim_data.resolution[0], sim_data.resolution[1])
    )

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
# CREATE FRAMES (MAIN)
# =========================
print("Generating frames for main animation...")

frames = []

for frame in range(nt):

    fig = plt.figure(figsize=(ny*2.2, nx*1.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])

    # PDF grid
    sub_gs = gs[0].subgridspec(nx, ny)

    pdf = temp_pdf[frame]
    x = np.arange(nb)

    for i in range(nx):
        for j in range(ny):

            ax = fig.add_subplot(sub_gs[i, j])

            ii = nx - 1 - i
            y = pdf[:, ii, j]
            y = y / (y.max() + 1e-8)

            ax.plot(x, y, lw=1)
            ax.set_xlim(0, nb-1)
            ax.set_ylim(0, 1)

            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.3)

    # TEMP PANEL
    temp_ax = fig.add_subplot(gs[1])
    log_temp = np.log10(cg_temp[frame] + 1e-8)
    im = temp_ax.imshow(log_temp, origin="lower", cmap="inferno")

    temp_ax.set_title(r"$\log_{10}$ Temp (CG)")
    plt.colorbar(im, ax=temp_ax, fraction=0.046)

    fig.suptitle(f"t = {frame}", fontsize=32)

    # SAVE SNAPSHOT
    if frame == 0:
        fig.savefig(first_frame_path, dpi=300)
        plt.imsave(temp_frame_path, log_temp, cmap="inferno")
        print(f"Saved PDF snapshot → {first_frame_path}")

    # CONVERT FIG → IMAGE
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    frames.append(image)
    plt.close(fig)


# =========================
# SAVE VIDEO
# =========================
print("Saving MP4...")

imageio.mimsave(video_path, frames, fps=10)

print(f"Saved → {video_path}")


# =========================
# TRUE vs PRED COMPARISON
# =========================
print("Generating comparison frames...")

frames_compare = []

for frame in range(nt):

    fig = plt.figure(figsize=(ny*4.0, nx*1.8))
    gs = fig.add_gridspec(1, 2)

    sub_left = gs[0].subgridspec(nx, ny)
    sub_right = gs[1].subgridspec(nx, ny)

    x = np.arange(nb)

    true_pdf = temp_pdf[frame]
    pred_pdf = conv_temp_pdf[frame]

    for i in range(nx):
        for j in range(ny):

            ii = nx - 1 - i

            y_true = true_pdf[:, ii, j]
            y_pred = pred_pdf[:, ii, j]

            y_true /= (y_true.max() + 1e-8)
            y_pred /= (y_pred.max() + 1e-8)

            # TRUE
            ax1 = fig.add_subplot(sub_left[i, j])
            ax1.plot(x, y_true, lw=1)
            ax1.set_xlim(0, nb-1)
            ax1.set_ylim(0, 1)
            ax1.set_xticks([])
            ax1.set_yticks([])

            for spine in ax1.spines.values():
                spine.set_linewidth(0.3)

            # PRED
            ax2 = fig.add_subplot(sub_right[i, j])
            ax2.plot(x, y_pred, lw=1, color='r')
            ax2.set_xlim(0, nb-1)
            ax2.set_ylim(0, 1)
            ax2.set_xticks([])
            ax2.set_yticks([])

            for spine in ax2.spines.values():
                spine.set_linewidth(0.3)

    fig.suptitle(f"True vs Predicted PDFs | t = {frame}", fontsize=32)

    if frame == 0:
        fig.savefig(snapshot_compare_path, dpi=300)
        print(f"Saved comparison snapshot → {snapshot_compare_path}")

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    frames_compare.append(image)
    plt.close(fig)


# =========================
# SAVE COMPARISON VIDEO
# =========================
print("Saving comparison MP4...")

imageio.mimsave(video_compare_path, frames_compare, fps=10)

print(f"Saved → {video_compare_path}")