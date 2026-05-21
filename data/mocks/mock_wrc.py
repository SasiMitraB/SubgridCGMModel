# Python script for plotting the results of CNN training without radiative cooling

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import data_preprocess
from data_preprocess import simulation_data
from feedforward_nn.fnn import snapshot_pred as fnn_snapshot_pred
from conv_nn.cnn import snapshot_pred as conv_snapshot_pred
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
from tqdm import tqdm

resolution = (512, 512)  
downsample = 8
# file_path = f"/data3/home/dipayandatta/Subgrid_CGM_Models/data/files/Subgrid CGM Models/Without_cooling/rk2, plm/{resolution[0]}_{resolution[1]}_prateek/bin"
file_path = f"/data3/home/dipayandatta/Subgrid_CGM_Models/athenak/kh_build/src/{resolution[0]}_{resolution[1]}_2/bin"
save_path = f"mocks/wrc/{resolution}_{downsample}_2/"
kernel_save_path = f"mocks/wrc/kernel/"
os.makedirs(save_path, exist_ok=True)
os.makedirs(kernel_save_path, exist_ok=True)
save_kernel = 5

sim_data = simulation_data()
sim_data.resolution = resolution
sim_data.down_sample = downsample

folder_path = f"data_saves/{resolution}_{downsample}_2"
if os.path.exists(f"{folder_path}"):
    sim_data.rho = np.load(f"{folder_path}/rho.npy")
    sim_data.temp = np.load(f"{folder_path}/temp.npy")
    sim_data.pressure = np.load(f"{folder_path}/pressure.npy")
    sim_data.ux = np.load(f"{folder_path}/ux.npy")
    sim_data.uy = np.load(f"{folder_path}/uy.npy")
    sim_data.eint = np.load(f"{folder_path}/eint.npy")
    sim_data.ps = np.load(f"{folder_path}/ps.npy")
else:
    sim_data.input_data(file_path)
    os.makedirs(folder_path, exist_ok=True)
    np.save(f"{folder_path}/rho.npy", sim_data.rho)
    np.save(f"{folder_path}/temp.npy", sim_data.temp)
    np.save(f"{folder_path}/pressure.npy", sim_data.pressure)
    np.save(f"{folder_path}/ux.npy", sim_data.ux)
    np.save(f"{folder_path}/uy.npy", sim_data.uy)
    np.save(f"{folder_path}/eint.npy", sim_data.eint)
    np.save(f"{folder_path}/ps.npy", sim_data.ps)

print("Input data loaded")

high_res_rho = sim_data.rho
cg_rho = np.zeros((high_res_rho.shape[0], high_res_rho.shape[1] // sim_data.down_sample, high_res_rho.shape[2] // sim_data.down_sample))
fmcl_data = np.zeros((sim_data.rho.shape[0], sim_data.rho.shape[1] // sim_data.down_sample, sim_data.rho.shape[2] // sim_data.down_sample))
for i in tqdm(range(high_res_rho.shape[0]), desc = "Coarse Graining"):
    cg_rho[i] = sim_data.coarse_grain(high_res_rho[i])
    fmcl_data[i] = sim_data.calc_fmcl(sim_data.rho[i], sim_data.temp[i])

# Plot the final high res rho and cg rho 
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(wspace=0.5)
extent_hr = [-5, 5, -5, 5]
extent_cg = [-5, 5, -5, 5]
axs[0].set_xlabel('x (pc)')
axs[0].set_ylabel('y (pc)')
axs[1].set_xlabel('x (pc)')
axs[1].set_ylabel('y (pc)')
im1 = axs[0].imshow(high_res_rho[-1], origin='lower', cmap='plasma', norm=LogNorm(), extent=extent_hr)
axs[0].set_title(rf'HR (${resolution[0]} \times {resolution[1]}$) Density')
plt.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04, label=r'Density ($cm^{-3}$)')
im2 = axs[1].imshow(cg_rho[-1], origin='lower', cmap='plasma', norm=LogNorm(), extent=extent_cg)
axs[1].set_title(rf'CG (${resolution[0]//downsample} \times {resolution[1]//downsample}$) Density')
plt.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04, label=r'Density ($cm^{-3}$)')
plt.savefig(save_path + "high_res_vs_cg_rho.jpg", dpi=500)
plt.close() 
print("High res vs CG rho plot saved")

# Plot the final fmcl data 
fig, ax = plt.subplots(figsize=(6, 6))
extent = [-5, 5, -5, 5]
ax.set_xlabel('x (pc)')
ax.set_ylabel('y (pc)')
im3 = ax.imshow(fmcl_data[-1], origin='lower', cmap='plasma', extent=extent)
ax.set_title(r'$f_{m}^{cl}$')
plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04, label=r'$f_{m}^{cl}$')
plt.savefig(save_path + "fmcl_data.jpg", dpi=500)
plt.close()
print("FMCL data plot saved")

# Plot the fmcl histogram
fmcl_hist_data = sim_data.fmcl_hist()
n_bins = 10
bins = np.linspace(0, 1, n_bins + 1) ** (1 / 2)
plt.hist(fmcl_hist_data, bins=bins, color='black', histtype='step')
min_count = np.histogram(fmcl_hist_data, bins=bins)[0].min()
plt.axhline(min_count, color='red', linestyle='--', label=f'Min count: {min_count}')
plt.legend()
plt.xlabel(r'$f_{m}^{cl}$')
plt.ylabel('Counts')
plt.yscale('log')
plt.title(r"Histogram of Cold gas Mass fraction")
plt.savefig(save_path + "fmcl_hist__bins.jpg", dpi=500)
plt.close()
print("Cold gas mass fraction histogram saved")

source_term = sim_data.calc_source_term()   
source_term_pred_fnn = np.zeros_like(source_term)
source_term_pred_cnn = np.zeros_like(source_term)

for i in tqdm(range(source_term.shape[0]), desc = "Predicting source term"):
    source_term_pred_fnn[i] = fnn_snapshot_pred(sim_data.rho[i], sim_data.temp[i], sim_data.pressure[i], \
                                        sim_data.ux[i], sim_data.uy[i], sim_data.eint[i], sim_data.ps[i], \
                                        downsample, (sim_data.resolution[0], sim_data.resolution[1]))
    source_term_pred_cnn[i] = conv_snapshot_pred(sim_data.rho[i], sim_data.temp[i], sim_data.pressure[i], \
                                        sim_data.ux[i], sim_data.uy[i], sim_data.eint[i], sim_data.ps[i], \
                                        downsample, (sim_data.resolution[0], sim_data.resolution[1]))
    
np.save(kernel_save_path + f"{resolution}_0.npy", source_term)
np.save(kernel_save_path + f"{resolution}_{save_kernel}.npy", source_term_pred_cnn)

# Plot the source term, predicted source term and residuals for the 50th, 100th and 150th timestep
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
plt.subplots_adjust(wspace=0.5)
extent = [-5, 5, -5, 5]
axs[0].set_xlabel('x (pc)')
axs[0].set_ylabel('y (pc)')
axs[1].set_xlabel('x (pc)')
axs[1].set_ylabel('y (pc)')
axs[2].set_xlabel('x (pc)')
axs[2].set_ylabel('y (pc)')
im_src = axs[0].imshow(source_term[50], origin='lower', cmap='viridis', vmin=-1, vmax=1, extent=extent)
axs[0].set_title('Source Term')
plt.colorbar(im_src, ax=axs[0], fraction=0.046, pad=0.04)
im_pred_fnn = axs[1].imshow(source_term_pred_fnn[50], origin='lower', cmap='viridis', vmin=-1, vmax=1, extent=extent)
fnn_res = np.sqrt(np.mean((source_term[50] - source_term_pred_fnn[50]) ** 2))
axs[1].set_title(f'Predicted Source Term (FNN)\nResidual: {fnn_res:.2f}')
plt.colorbar(im_pred_fnn, ax=axs[1], fraction=0.046, pad=0.04)
im_pred_cnn = axs[2].imshow(source_term_pred_cnn[50], origin='lower', cmap='viridis', vmin=-1, vmax=1, extent=extent)
cnn_res = np.sqrt(np.mean((source_term[50] - source_term_pred_cnn[50]) ** 2))
axs[2].set_title(f'Predicted Source Term (CNN)\nResidual: {cnn_res:.2f}')
plt.colorbar(im_pred_cnn, ax=axs[2], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(save_path + f"{resolution}_source_term_50.pdf")
plt.close()

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
plt.subplots_adjust(wspace=0.5)
axs[0].set_xlabel('x (pc)')
axs[0].set_ylabel('y (pc)')
axs[1].set_xlabel('x (pc)')
axs[1].set_ylabel('y (pc)')
axs[2].set_xlabel('x (pc)')
axs[2].set_ylabel('y (pc)')
im_src = axs[0].imshow(source_term[100], origin='lower', cmap='viridis', vmin=-1, vmax=1, extent=extent)
axs[0].set_title('Source Term')
plt.colorbar(im_src, ax=axs[0], fraction=0.046, pad=0.04)
im_pred_fnn = axs[1].imshow(source_term_pred_fnn[100], origin='lower', cmap='viridis', vmin=-1, vmax=1, extent=extent)
fnn_res = np.sqrt(np.mean((source_term[100] - source_term_pred_fnn[100]) ** 2))
axs[1].set_title(f'Predicted Source Term (FNN)\nResidual: {fnn_res:.2f}')
plt.colorbar(im_pred_fnn, ax=axs[1], fraction=0.046, pad=0.04)
im_pred_cnn = axs[2].imshow(source_term_pred_cnn[100], origin='lower', cmap='viridis', vmin=-1, vmax=1, extent=extent)
cnn_res = np.sqrt(np.mean((source_term[100] - source_term_pred_cnn[100]) ** 2))
axs[2].set_title(f'Predicted Source Term (CNN)\nResidual: {cnn_res:.2f}')
plt.colorbar(im_pred_cnn, ax=axs[2], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(save_path + f"{resolution}_source_term_100.pdf")
plt.close()

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
plt.subplots_adjust(wspace=0.5)
axs[0].set_xlabel('x (pc)')
axs[0].set_ylabel('y (pc)')
axs[1].set_xlabel('x (pc)')
axs[1].set_ylabel('y (pc)')
axs[2].set_xlabel('x (pc)')
axs[2].set_ylabel('y (pc)')
im_src = axs[0].imshow(source_term[150], origin='lower', cmap='viridis', vmin=-1, vmax=1, extent=extent)
axs[0].set_title('Source Term')
plt.colorbar(im_src, ax=axs[0], fraction=0.046, pad=0.04)
im_pred_fnn = axs[1].imshow(source_term_pred_fnn[150], origin='lower', cmap='viridis', vmin=-1, vmax=1, extent=extent)
fnn_res = np.sqrt(np.mean((source_term[150] - source_term_pred_fnn[150]) ** 2))
axs[1].set_title(f'Predicted Source Term (FNN)\nResidual: {fnn_res:.2f}')
plt.colorbar(im_pred_fnn, ax=axs[1], fraction=0.046, pad=0.04)
im_pred_cnn = axs[2].imshow(source_term_pred_cnn[150], origin='lower', cmap='viridis', vmin=-1, vmax=1, extent=extent)
cnn_res = np.sqrt(np.mean((source_term[150] - source_term_pred_cnn[150]) ** 2))
axs[2].set_title(f'Predicted Source Term (CNN)\nResidual: {cnn_res:.2f}')
plt.colorbar(im_pred_cnn, ax=axs[2], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(save_path + f"{resolution}_source_term_150.pdf")
plt.close()

print("Source term and predicted source term plots saved")

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

im_src = axs[0].imshow(source_term[0], origin='lower', cmap='coolwarm', vmin=-0.2, vmax=0.2)
axs[0].set_title('Source Term')
plt.colorbar(im_src, ax=axs[0], fraction=0.046, pad=0.04)

im_pred_fnn = axs[1].imshow(source_term_pred_fnn[0], origin='lower', cmap='coolwarm', vmin=-0.2, vmax=0.2)
axs[1].set_title('Predicted Source Term (FNN)')
plt.colorbar(im_pred_fnn, ax=axs[1], fraction=0.046, pad=0.04)

im_pred_cnn = axs[2].imshow(source_term_pred_cnn[0], origin='lower', cmap='coolwarm', vmin=-0.2, vmax=0.2)
axs[2].set_title('Predicted Source Term (CNN)')
plt.colorbar(im_pred_cnn, ax=axs[2], fraction=0.046, pad=0.04)

def update_source(frame):
    im_src.set_data(source_term[frame])
    im_pred_fnn.set_data(source_term_pred_fnn[frame])
    fnn_res = np.sqrt(np.mean((source_term[frame] - source_term_pred_fnn[frame]) ** 2))
    axs[1].set_title(f'FNN Residual: {fnn_res:.2f}')
    im_pred_cnn.set_data(source_term_pred_cnn[frame])
    cnn_res = np.sqrt(np.mean((source_term[frame] - source_term_pred_cnn[frame]) ** 2))
    axs[2].set_title(f'CNN Residual: {cnn_res:.2f}')
    for ax in axs.flat:
        ax.set_xlabel(f'Timestep: {frame}')
    return [im_src, im_pred_fnn, im_pred_cnn]

ani_source = animation.FuncAnimation(fig, update_source, frames=source_term.shape[0], interval=100, blit=True)
ani_source.save(save_path + "source_term_evolution.mp4", writer='ffmpeg')
plt.close()
print("Source term animation saved")

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

im1 = axs[0, 0].imshow(high_res_rho[0], origin='lower', cmap='plasma', norm=LogNorm())
axs[0, 0].set_title(rf'HR (${resolution[0]} \times {resolution[0]}$) Density')
plt.colorbar(im1, ax=axs[0, 0], fraction=0.046, pad=0.04)

im2 = axs[0, 1].imshow(cg_rho[0], origin='lower', cmap='plasma', norm=LogNorm())
axs[0, 1].set_title(rf'CG (${resolution[0]//downsample} \times {resolution[1]//downsample}$) Density')
plt.colorbar(im2, ax=axs[0, 1], fraction=0.046, pad=0.04)

im3 = axs[1, 0].imshow(source_term[0], origin='lower', cmap='viridis', vmin = -1, vmax = 1)
axs[1, 0].set_title('Source Term')
plt.colorbar(im3, ax=axs[1, 0], fraction=0.046, pad=0.04)

im4 = axs[1, 1].imshow(fmcl_data[0], origin='lower', cmap='viridis')
axs[1, 1].set_title(r'$f_{m}^{cl}$')
plt.colorbar(im4, ax=axs[1, 1], fraction=0.046, pad=0.04)

def update_all(frame):
    im1.set_data(high_res_rho[frame])
    im2.set_data(cg_rho[frame])
    im3.set_data(source_term[frame])
    im4.set_data(fmcl_data[frame])
    for ax in axs.flat:
        ax.set_xlabel(f'Timestep: {frame}')
    return [im1, im2, im3, im4]

ani = animation.FuncAnimation(fig, update_all, frames=high_res_rho.shape[0], interval=100, blit=True)
ani.save(save_path + "all_evolution.mp4", writer='ffmpeg')
plt.close()
print("All evolution animation saved")

hist_data = sim_data.fmcl_hist()
plt.hist(hist_data, bins=10, histtype='step', color='black')
plt.xlabel(r'$f_{m}^{cl}$')
plt.ylabel('Counts')
plt.yscale('log')
plt.title(r"Histogram of Cold gas Mass fraction")
plt.savefig(save_path + "fmcl_hist.jpg", dpi=500)
plt.close()
print("Histogram saved")
