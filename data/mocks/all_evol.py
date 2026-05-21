# Python script to plot the different source terms and subgrid fluxes and their correlations and Fourier transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import data_preprocess
from data_preprocess import simulation_data
from conv_nn.indiv_cnn import snapshot_pred as conv_snapshot_pred
from conv_nn.flux_cnn import snapshot_pred as conv_flux_pred
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import corner
from scipy.ndimage import gaussian_filter

resolution = (512, 256)  
downsample = 32
# file_path = f"/data3/home/dipayandatta/Subgrid_CGM_Models/data/files/Subgrid CGM Models/Without_cooling/rk2, plm/{resolution[0]}_{resolution[1]}_prateek/bin"
save_path = f"mocks/all/ncc{resolution}_{downsample}/"
kernel_save_path = f"mocks/wrc/kernel/"
std_save_path = f"std_saves/cc{resolution}_{downsample}/"
os.makedirs(save_path, exist_ok=True)
os.makedirs(kernel_save_path, exist_ok=True)
os.makedirs(std_save_path, exist_ok=True)
save_kernel = 5

sim_data = simulation_data()
sim_data.resolution = resolution
sim_data.down_sample = downsample

folder_path = f"/tmp/dipayandatta/datafiles/cc{resolution}_{downsample}"
file_path = f"/tmp/dipayandatta/athenak/kh_build/src/cc{resolution[0]}_{resolution[1]}/bin"
if os.path.exists(f"{folder_path}"):
    sim_data.rho = np.load(f"{folder_path}/rho.npy")
    sim_data.temp = np.load(f"{folder_path}/temp.npy")
    sim_data.pressure = np.load(f"{folder_path}/pressure.npy")
    sim_data.ux = np.load(f"{folder_path}/ux.npy")
    sim_data.uy = np.load(f"{folder_path}/uy.npy")
    sim_data.eint = np.load(f"{folder_path}/eint.npy")
    sim_data.ps = np.load(f"{folder_path}/ps.npy")
    sim_data.cons_rho = np.load(f"{folder_path}/cons_rho.npy")
    sim_data.cons_momx = np.load(f"{folder_path}/cons_mx.npy")
    sim_data.cons_momy = np.load(f"{folder_path}/cons_my.npy")
    sim_data.cons_ener = np.load(f"{folder_path}/cons_ener.npy")
    sim_data.cons_ps = np.load(f"{folder_path}/cons_ps.npy")
else:
    sim_data.input_data(file_path, start=501)
    sim_data.input_cons_data(file_path, start=501)
    os.makedirs(folder_path, exist_ok=True)
    np.save(f"{folder_path}/rho.npy", sim_data.rho)
    np.save(f"{folder_path}/temp.npy", sim_data.temp)
    np.save(f"{folder_path}/pressure.npy", sim_data.pressure)
    np.save(f"{folder_path}/ux.npy", sim_data.ux)
    np.save(f"{folder_path}/uy.npy", sim_data.uy)
    np.save(f"{folder_path}/eint.npy", sim_data.eint)
    np.save(f"{folder_path}/ps.npy", sim_data.ps)
    np.save(f"{folder_path}/cons_rho.npy", sim_data.cons_rho)
    np.save(f"{folder_path}/cons_mx.npy", sim_data.cons_momx)
    np.save(f"{folder_path}/cons_my.npy", sim_data.cons_momy)
    np.save(f"{folder_path}/cons_ener.npy", sim_data.cons_ener)
    np.save(f"{folder_path}/cons_ps.npy", sim_data.cons_ps)

print("Input data loaded")

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

    return lam

def cool_flux(density, temp):
    mu = 0.62   
    n = density / mu
    cool_rate_cgs = lambda_cool(temp) * n**2    # erg/cm^3/s
    return cool_rate_cgs * 1.975e27

high_res_rho = sim_data.rho
high_res_temp = sim_data.temp  
high_res_flux = np.zeros_like(high_res_rho)
cg_rho = np.zeros((high_res_rho.shape[0], high_res_rho.shape[1] // sim_data.down_sample, high_res_rho.shape[2] // sim_data.down_sample))
cg_pressure = np.zeros_like(cg_rho)
cg_temp = np.zeros_like(cg_rho)

for i in tqdm(range(high_res_rho.shape[0]), desc = "Coarse Graining"):
    cg_rho[i] = sim_data.coarse_grain(high_res_rho[i])
    cg_pressure[i] = sim_data.coarse_grain(sim_data.pressure[i])
    cg_temp[i] = sim_data.coarse_grain(high_res_temp[i])

shape = (sim_data.rho.shape[0], sim_data.rho.shape[1] // sim_data.down_sample, sim_data.rho.shape[2] // sim_data.down_sample)
fields = ['rho', 'temp', 'ux', 'uy', 'ps', 'fmcl', 'eint']
cg = {f'cg_{field}': np.zeros(shape) for field in fields}

for i in range(sim_data.rho.shape[0]):
    for field in fields:
        if field in ['rho', 'temp', 'ux', 'uy', 'ps', 'eint']:
            cg[f'cg_{field}'][i] = sim_data.coarse_grain(getattr(sim_data, field)[i])
        elif field in ['fmcl']:
            cg[f'cg_{field}'][i] = sim_data.calc_fmcl(sim_data.rho[i], sim_data.temp[i])

source_term = sim_data.calc_all_source_terms()
source_term_pred_cnn = np.zeros_like(source_term)

subgrid_flux = sim_data.calc_subgrid_flux()
subgrid_flux_pred_cnn = np.zeros_like(subgrid_flux)

for i in tqdm(range(source_term.shape[0]), desc = "Predicting source term"):
    source_term_pred_cnn[i] = conv_snapshot_pred(sim_data.rho[i], sim_data.temp[i], sim_data.pressure[i], \
                                        sim_data.ux[i], sim_data.uy[i], sim_data.eint[i], sim_data.ps[i], \
                                        downsample, (sim_data.resolution[0], sim_data.resolution[1]))

for i in tqdm(range(subgrid_flux.shape[0]), desc = "Predicting subgrid flux"):
    subgrid_flux_pred_cnn[i] = conv_flux_pred(sim_data.rho[i], sim_data.temp[i], sim_data.pressure[i], \
                                        sim_data.ux[i], sim_data.uy[i], sim_data.eint[i], sim_data.ps[i],
                                        downsample, (sim_data.resolution[0], sim_data.resolution[1]))

fig, axs = plt.subplots(1, 1, figsize=(5, 5))

im_src = axs.imshow(subgrid_flux[0, 7], origin='lower', cmap='viridis')
cbar_src = plt.colorbar(im_src, ax=axs, fraction=0.046, pad=0.04)

def update_flux(frame):
    src = subgrid_flux[frame, 7]
    im_src.set_data(src)

    all_vals = src.ravel()
    mean = np.mean(all_vals)
    std = np.std(all_vals)
    vmin = mean - 5 * std
    vmax = mean + 5 * std

    im_src.set_clim(vmin, vmax)
    cbar_src.update_normal(im_src)

    global_title.set_text(f'Timestep: {frame}')
    return im_src

plt.tight_layout(rect=[0, 0, 1, 0.97])
global_title = fig.suptitle(f'Timestep: 0', fontsize=16, y=0.985)

ani_flux = animation.FuncAnimation(
    fig, update_flux,
    frames=subgrid_flux.shape[0],
    interval=100,
    blit=False
)

ani_flux.save(save_path + "cooling_flux_evolution.mp4", writer='ffmpeg')
plt.close()
print("Cooling flux animation saved")

def compute_corr(gt, pred, n):
    gt_flat   = gt.reshape(10, -1)
    pred_flat = pred.reshape(10, -1)
    idx = np.random.choice(gt_flat.shape[1], size=n, replace=False)
    gt_sample   = gt_flat[:, idx]
    pred_sample = pred_flat[:, idx]
    corr_gt   = np.corrcoef(gt_sample)
    corr_pred = np.corrcoef(pred_sample)
    return corr_gt, corr_pred

gt = subgrid_flux
pred = subgrid_flux_pred_cnn

corr_500 = compute_corr(gt, pred, 5000)
corr_50  = compute_corr(gt, pred, 500)
corr_5   = compute_corr(gt, pred, 50)

samples = [5000, 500, 50]
corrs   = [corr_500, corr_50, corr_5]

fig, axs = plt.subplots(3, 2, figsize=(10, 12), sharex=True, sharey=True)

for i, (n, (corr_gt, corr_pred)) in enumerate(zip(samples, corrs)):
    im0 = axs[i, 0].imshow(corr_gt, vmin=-1, vmax=1, cmap='coolwarm')
    im1 = axs[i, 1].imshow(corr_pred, vmin=-1, vmax=1, cmap='coolwarm')

    axs[i, 0].set_title(f"GT Corr ({n} samples)")
    axs[i, 1].set_title(f"Pred Corr ({n} samples)")

    for j in range(2):
        axs[i, j].set_xticks(range(10))
        axs[i, j].set_yticks(range(10))
        axs[i, j].set_xlabel("Channel")
        axs[i, j].set_ylabel("Channel")

fig.colorbar(im0, ax=axs.ravel().tolist(), fraction=0.03, pad=0.02)

plt.savefig(save_path + "flux_correlation_all.png", dpi=300)
plt.close(fig)

print("All flux correlations saved.")

source_term_plot = np.transpose(source_term, axes=(1, 0, 2, 3))
pred_source_term_plot = np.transpose(source_term_pred_cnn, axes=(1, 0, 2, 3))

gt_flat   = source_term_plot.reshape(5, -1)       
pred_flat = pred_source_term_plot.reshape(5, -1)  

idx = np.random.choice(gt_flat.shape[1], size=200, replace=False)

gt_sample   = gt_flat[:, idx]
pred_sample = pred_flat[:, idx]

corr_gt   = np.corrcoef(gt_sample)
corr_pred = np.corrcoef(pred_sample)

fig, axs = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
im0 = axs[0].imshow(corr_gt, vmin=-1, vmax=1, cmap='coolwarm')
im1 = axs[1].imshow(corr_pred, vmin=-1, vmax=1, cmap='coolwarm')
axs[0].set_title("Actual Correlations")
axs[1].set_title("CNN Predicted Correlations")
for ax in axs:
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xlabel("Channel"); ax.set_ylabel("Channel")
fig.colorbar(im0, ax=axs, fraction=0.046, pad=0.01)
# plt.tight_layout()
plt.savefig(save_path + "random_correlation_matrices.png", dpi=300)
plt.close(fig)

corr_diff = np.mean((corr_gt - corr_pred)**2)
r_corr = 1 - np.linalg.norm(corr_pred - corr_gt) / np.linalg.norm(corr_gt)
print(f"Mean squared correlation difference: {corr_diff:.3e}")
print(f"Correlation consistency score: {r_corr:.3f}")

st = source_term_plot[3]
st_pred = pred_source_term_plot[3]
smooth_st_pred = np.zeros_like(st_pred)
adapt_st_pred = np.zeros_like(st_pred)

for i in range(st.shape[0]):
    smooth_st_pred[i] = gaussian_filter(st_pred[i], sigma=3)

    v = st_pred[i]
    w = np.clip((np.abs(v)-np.percentile(np.abs(v),75)) / (np.percentile(np.abs(v),90)-np.percentile(np.abs(v),75)+1e-12), 0, 1)
    A, B = gaussian_filter(v, 0.0), gaussian_filter(v, 3.0)
    adapt_st_pred[i] = (1-w)*A + w*B

st_x = np.mean(st, axis=1)
st_x_pred = np.mean(st_pred, axis=1)
smooth_st_x_pred = np.mean(smooth_st_pred, axis=1)
adapt_st_x_pred = np.mean(adapt_st_pred, axis=1)

fst_x = np.fft.fft(st_x, axis=1)
fst_x_pred = np.fft.fft(st_x_pred, axis=1)
fst_x_smooth = np.fft.fft(smooth_st_x_pred, axis=1)
fst_x_adapt = np.fft.fft(adapt_st_x_pred, axis=1)
k_val = np.fft.fftfreq(st_x.shape[1], d=10 / st_x.shape[1])

fst_x = np.fft.fftshift(fst_x, axes=1)
fst_x_pred = np.fft.fftshift(fst_x_pred, axes=1)
fst_x_smooth = np.fft.fftshift(fst_x_smooth, axes=1)
fst_x_adapt = np.fft.fftshift(fst_x_adapt, axes=1)
k_val = np.fft.fftshift(k_val)

t_idxs = 7
colors = plt.cm.plasma(np.linspace(1, 0, t_idxs))

plt.figure(figsize=(10, 6))
for t_idx in tqdm(np.linspace(0, fst_x.shape[0] - 1, t_idxs, dtype=int), desc="Plotting FFT along X"):
    plt.plot(np.abs(k_val), np.abs(fst_x[t_idx]), color=colors[int((t_idx / fst_x.shape[0]) * t_idxs)], label=f'{5*t_idx/fst_x.shape[0]:.2} Myr')
plt.xlabel(r'$k_x$')
plt.xlim(0.0, np.max(np.abs(k_val)))
plt.ylabel(r'FFT Coefficients')
plt.legend()
plt.title(rf'Energy Source Term FFT')
plt.savefig(f"{save_path}/fft_x_energy.png", dpi=300)
plt.clf()

plt.figure(figsize=(10, 6))
for t_idx in tqdm(np.linspace(0, fst_x_pred.shape[0] - 1, t_idxs, dtype=int), desc="Plotting Predicted FFT along X"):
    plt.plot(np.abs(k_val), np.abs(fst_x_pred[t_idx]), color=colors[int((t_idx / fst_x_pred.shape[0]) * t_idxs)], label=f'{5*t_idx/fst_x_pred.shape[0]:.2} Myr')
plt.xlabel(r'$k_x$')
plt.xlim(0.0, np.max(np.abs(k_val)))
plt.ylabel(r'FFT Coefficients')
plt.legend()
plt.title(rf'Predicted Energy Source Term FFT')
plt.savefig(f"{save_path}/predicted_fft_x_energy.png", dpi=300)
plt.clf()

plt.figure(figsize=(10, 6))
for t_idx in tqdm(np.linspace(0, fst_x_smooth.shape[0] - 1, t_idxs, dtype=int), desc="Plotting Smoothed Predicted FFT along X"):
    plt.plot(np.abs(k_val), np.abs(fst_x_smooth[t_idx]), color=colors[int((t_idx / fst_x_smooth.shape[0]) * t_idxs)], label=f'{5*t_idx/fst_x_smooth.shape[0]:.2} Myr')
plt.xlabel(r'$k_x$')
plt.xlim(0.0, np.max(np.abs(k_val)))
plt.ylabel(r'FFT Coefficients')
plt.legend()
plt.title(rf'Smoothed Predicted Energy Source Term FFT')
plt.savefig(f"{save_path}/smoothed_predicted_fft_x_energy.png", dpi=300)
plt.clf()

plt.figure(figsize=(10, 6))
for t_idx in tqdm(np.linspace(0, fst_x_adapt.shape[0] - 1, t_idxs, dtype=int), desc="Plotting Adaptively Smoothed Predicted FFT along X"):
    plt.plot(np.abs(k_val), np.abs(fst_x_adapt[t_idx]), color=colors[int((t_idx / fst_x_adapt.shape[0]) * t_idxs)], label=f'{5*t_idx/fst_x_adapt.shape[0]:.2} Myr')
plt.xlabel(r'$k_x$')
plt.xlim(0.0, np.max(np.abs(k_val)))
plt.ylabel(r'FFT Coefficients')
plt.legend()
plt.title(rf'Adaptively Smoothed Predicted Energy Source Term FFT')
plt.savefig(f"{save_path}/adaptively_smoothed_predicted_fft_x_energy.png", dpi=300)
plt.clf()

ener_bins = np.linspace(np.min(cg['cg_eint']), np.max(cg['cg_eint']), 1000)
bin_centers = 0.5 * (ener_bins[1:] + ener_bins[:-1])
for i in range(3, 4, 1):
    # flatten temp and source_term
    E = cg["cg_eint"].ravel()
    S = source_term_plot[i].ravel()
    pS = pred_source_term_plot[i].ravel()

    # digitize temps into bins
    inds = np.digitize(E, ener_bins)

    means = []
    pred_means = []
    stds = []
    pred_stds = []
    for b in range(1, len(ener_bins)):
        mask = inds == b
        if mask.sum() > 0:
            means.append(S[mask].mean())
            pred_means.append(pS[mask].mean())
            stds.append(S[mask].std())
            pred_stds.append(pS[mask].std())
        else:
            means.append(np.nan)
            pred_means.append(np.nan)
            stds.append(np.nan)
            pred_stds.append(np.nan)
    
    means = np.array(means)
    pred_means = np.array(pred_means)
    stds = np.array(stds)
    pred_stds = np.array(pred_stds)

    # scatter plot
    plt.figure(figsize=(6,5))
    plt.scatter(E, S, s=1, alpha=1.0, c="tab:blue")
    plt.scatter(E, pS, s=1, alpha=0.3, c="tab:orange")
    plt.xscale("log")   
    plt.xlabel("Internal Energy")
    plt.ylabel("Source Term")
    plt.title(f"Predicted Source Term vs IEN (channel {i})")
    plt.tight_layout()
    plt.savefig(save_path + "scatter_plot/" + f"epred_scatter_{i}.png", dpi=300)
    plt.clf()

temp_bins = np.logspace(5, 6, 1000)  
bin_centers = 0.5 * (temp_bins[1:] + temp_bins[:-1])
for i in range(source_term_plot.shape[0]):
    # flatten temp and source_term
    T = cg["cg_temp"].ravel()
    S = source_term_plot[i].ravel()
    pS = pred_source_term_plot[i].ravel()

    # digitize temps into bins
    inds = np.digitize(T, temp_bins)

    means = []
    pred_means = []
    stds = []
    pred_stds = []
    for b in range(1, len(temp_bins)):
        mask = inds == b
        if mask.sum() > 0:
            means.append(S[mask].mean())
            pred_means.append(pS[mask].mean())
            stds.append(S[mask].std())
            pred_stds.append(pS[mask].std())
        else:
            means.append(np.nan)
            pred_means.append(np.nan)
            stds.append(np.nan)
            pred_stds.append(np.nan)
    
    means = np.array(means)
    pred_means = np.array(pred_means)
    stds = np.array(stds)
    pred_stds = np.array(pred_stds)

    std_val = np.vstack([bin_centers, means, stds]).T
    np.save(std_save_path + f"std_{i}.npy", std_val)

    # plot std vs temp
    plt.figure(figsize=(6,5))
    plt.plot(bin_centers, stds, linestyle='-', label='True', color='tab:blue')
    plt.plot(bin_centers, pred_stds, linestyle='--', label='Predicted', color='tab:orange')
    plt.xscale("log")
    plt.xlabel("Temperature [K]")
    plt.ylabel("Std(Source Term)")
    plt.title(f"Std of Source Term vs Temperature (channel {i})")
    plt.tight_layout()
    plt.legend()
    plt.savefig(save_path + "scatter_plot/" + f"std_{i}.png", dpi=300)
    plt.clf()

    # scatter plot
    plt.figure(figsize=(6,5))
    plt.scatter(T, S, s=1, alpha=1.0, c="tab:blue")
    plt.scatter(T, pS, s=1, alpha=0.3, c="tab:orange")
    plt.xscale("log")   
    plt.xlabel("Temperature [K]")
    plt.ylabel("Source Term")
    plt.title(f"Predicted Source Term vs Temperature (channel {i})")
    plt.tight_layout()
    plt.savefig(save_path + "scatter_plot/" + f"pred_scatter_{i}.png", dpi=300)
    plt.clf()

    # Plot histograms (normalized as PDFs)
    plt.figure(figsize=(6, 4))
    plt.hist(S, bins=100, density=True, alpha=0.5,
             color="tab:blue", label="True")
    plt.hist(pS, bins=100, density=True, alpha=0.5,
             color="tab:orange", label="Pred")

    plt.title(f"PDF Channel {i}")
    plt.xlabel("Source Term Value")
    plt.ylabel("Normalized PDF")
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path + f"pdf_channel_{i}.png", dpi=300)
    plt.clf()

n_channels = source_term_plot.shape[0]
flattened = [source_term_plot[i].ravel() for i in range(n_channels)]
data = np.vstack(flattened).T  
fig = corner.corner(
    data,
    labels=[f"Channel {i}" for i in range(n_channels)],
    show_titles=True,
    title_fmt=".2e",
    plot_datapoints=True,
    plot_density=False,
    color="tab:blue",
    hist_bin_factor=1.2
)

pred_flattened = [pred_source_term_plot[i].ravel() for i in range(n_channels)]
pred_data = np.vstack(pred_flattened).T  
corner.corner(
    pred_data,
    fig=fig,                    
    plot_datapoints=True,
    plot_density=False,
    color="tab:orange",
    hist_bin_factor=1.2
)
plt.savefig(save_path + "corner_overlay.png", dpi=300)
plt.clf()

dy = sim_data.total_length / cg_rho.shape[1]
dx = sim_data.total_width / cg_rho.shape[2]

rho_lim = np.zeros(high_res_rho.shape[0])
ux_lim = np.zeros(high_res_rho.shape[0])
uy_lim = np.zeros(high_res_rho.shape[0])
en_lim = np.zeros(high_res_rho.shape[0])
fmcl_lim = np.zeros(high_res_rho.shape[0])
for i in range(high_res_rho.shape[0]):
    rho_lim[i] = (np.abs(np.max(source_term[i][0])) + np.abs(np.min(source_term[i][0])))/2
    ux_lim[i] = (np.abs(np.max(source_term[i][1])) + np.abs(np.min(source_term[i][1])))/2
    uy_lim[i] = (np.abs(np.max(source_term[i][2])) + np.abs(np.min(source_term[i][2])))/2
    en_lim[i] = (np.abs(np.max(source_term[i][3])) + np.abs(np.min(source_term[i][3])))/2
    fmcl_lim[i] = (np.abs(np.max(source_term[i][4])) + np.abs(np.min(source_term[i][4])))/2
time = sim_data.delta_time * np.arange(high_res_rho.shape[0])
plt.plot(time, rho_lim, label='Density Source Term')
plt.plot(time, ux_lim, label='X-Momentum Source Term')
plt.plot(time, uy_lim, label='Y-Momentum Source Term')
plt.plot(time, en_lim, label='Energy Source Term')
plt.plot(time, fmcl_lim, label='FMCL Source Term')
plt.xlabel('Time (Myr)')
plt.xscale('log')
plt.ylabel('Source Term Magnitude')
plt.yscale('log')
plt.legend()
plt.savefig(save_path + "source_term_magnitude.png", dpi=300)
plt.close()
print("Source term magnitude plot saved")

# fig, axs = plt.subplots(5, 2, figsize=(8, 15))
fig, axs = plt.subplots(5, 2, figsize=(5, 15))

for i in range(5):
    src = source_term[0, i]
    pred = source_term_pred_cnn[0, i]

    # Compute mean and std for color scaling
    all_vals = np.concatenate([src.ravel(), pred.ravel()])
    mean = np.mean(all_vals)
    std = np.std(all_vals)
    vmin = mean - 5 * std
    vmax = mean + 5 * std

    # Plot source term
    im_src = axs[i, 0].imshow(src, origin='lower', cmap='coolwarm', vmin=vmin, vmax=vmax)
    axs[i, 0].set_title(f'True Channel {i}')
    axs[i, 0].set_xlabel(f'Source Int: {np.trapezoid(np.trapezoid(src, dx=dx, axis=1), dx=dy, axis=0):.2f}')
    plt.colorbar(im_src, ax=axs[i, 0], fraction=0.046, pad=0.04)

    # Plot predicted source term
    im_pred = axs[i, 1].imshow(pred, origin='lower', cmap='coolwarm', vmin=vmin, vmax=vmax)
    axs[i, 1].set_title(f'Pred Channel {i}: R2 = {1 - np.sum((src - pred) ** 2) / np.sum((src - np.mean(src)) ** 2):.2f}')
    axs[i, 1].set_xlabel(f'Pred Int: {np.trapezoid(np.trapezoid(pred, dx=dx, axis=1), dx=dy, axis=0):.2f}')
    plt.colorbar(im_pred, ax=axs[i, 1], fraction=0.046, pad=0.04)

fig.suptitle("Timestep: 0", fontsize=16, y=0.99)
plt.tight_layout()
plt.savefig(save_path + "source_term_snapshot_t0.png", dpi=300)
plt.close(fig)
print("Source term snapshot saved")

# fig, axs = plt.subplots(5, 2, figsize=(8, 15))
fig, axs = plt.subplots(5, 2, figsize=(5, 15))

im_src_list = []
im_pred_list = []
cbar_src_list = []
cbar_pred_list = []

for i in range(5):
    im_src = axs[i, 0].imshow(source_term[0, i], origin='lower', cmap='coolwarm')
    im_pred = axs[i, 1].imshow(source_term_pred_cnn[0, i], origin='lower', cmap='coolwarm')
    
    axs[i, 0].set_title(f'True Channel {i}')
    axs[i, 0].set_xlabel(f'Source Int: {np.trapezoid(np.trapezoid(source_term[0, i], dx=dx, axis=1), dx=dy, axis=0):.2f}')
    axs[i, 1].set_title(f'Pred Channel {i}')
    axs[i, 1].set_xlabel(f'Source Int: {np.trapezoid(np.trapezoid(source_term_pred_cnn[0, i], dx=dx, axis=1), dx=dy, axis=0):.2f}')

    cbar_src = plt.colorbar(im_src, ax=axs[i, 0], fraction=0.046, pad=0.04)
    cbar_pred = plt.colorbar(im_pred, ax=axs[i, 1], fraction=0.046, pad=0.04)
    
    im_src_list.append(im_src)
    im_pred_list.append(im_pred)
    cbar_src_list.append(cbar_src)
    cbar_pred_list.append(cbar_pred)

def update_source(frame):
    for i in range(5):
        src = source_term[frame, i]
        pred = source_term_pred_cnn[frame, i]

        im_src_list[i].set_data(src)
        im_pred_list[i].set_data(pred)

        all_vals = np.concatenate([src.ravel(), pred.ravel()])
        mean = np.mean(all_vals)
        std = np.std(all_vals)
        vmin = mean - 5 * std
        vmax = mean + 5 * std

        im_src_list[i].set_clim(vmin, vmax)
        im_pred_list[i].set_clim(vmin, vmax)

        cbar_src_list[i].update_normal(im_src_list[i])
        cbar_pred_list[i].update_normal(im_pred_list[i])

        axs[i, 1].set_title(f'Pred Channel {i}: R2 = {1 - np.sum((src - pred) ** 2) / np.sum((src - np.mean(src)) ** 2):.2f}')
        axs[i, 0].set_xlabel(f'Source Int: {np.trapezoid(np.trapezoid(src, dx=dx, axis=1), dx=dy, axis=0):.2f}')
        axs[i, 1].set_xlabel(f'Pred Int: {np.trapezoid(np.trapezoid(pred, dx=dx, axis=1), dx=dy, axis=0):.2f}')

    global_title.set_text(f'Timestep: {frame}')
    return im_src_list + im_pred_list

plt.tight_layout(rect=[0, 0, 1, 0.97]) 
global_title = fig.suptitle(f'Timestep: 0', fontsize=16, y=0.985) 

ani_source = animation.FuncAnimation(
    fig, update_source,
    frames=source_term.shape[0],
    interval=100,
    blit=True
)

ani_source.save(save_path + "source_term_evolution.mp4", writer='ffmpeg')
plt.close()
print("Source term animation saved")

fig, axs = plt.subplots(10, 2, figsize=(5, 30))
im_src_list = []
im_pred_list = []
cbar_src_list = []
cbar_pred_list = []

for i in range(10):
    im_src = axs[i, 0].imshow(subgrid_flux[0, i], origin='lower', cmap='viridis')
    im_pred = axs[i, 1].imshow(subgrid_flux_pred_cnn[0, i], origin='lower', cmap='viridis')

    cbar_src = plt.colorbar(im_src, ax=axs[i, 0], fraction=0.046, pad=0.04)
    cbar_pred = plt.colorbar(im_pred, ax=axs[i, 1], fraction=0.046, pad=0.04)
    
    im_src_list.append(im_src)
    im_pred_list.append(im_pred)
    cbar_src_list.append(cbar_src)
    cbar_pred_list.append(cbar_pred)

def update_flux(frame):
    for i in range(10):
        src = subgrid_flux[frame, i]
        pred = subgrid_flux_pred_cnn[frame, i]

        im_src_list[i].set_data(src)
        im_pred_list[i].set_data(pred)

        all_vals = np.concatenate([src.ravel(), pred.ravel()])
        mean = np.mean(all_vals)
        std = np.std(all_vals)
        vmin = mean - 5 * std
        vmax = mean + 5 * std

        im_src_list[i].set_clim(vmin, vmax)
        im_pred_list[i].set_clim(vmin, vmax)

        cbar_src_list[i].update_normal(im_src_list[i])
        cbar_pred_list[i].update_normal(im_pred_list[i])

    global_title.set_text(f'Timestep: {frame}')
    return im_src_list + im_pred_list

plt.tight_layout(rect=[0, 0, 1, 0.97])
global_title = fig.suptitle(f'Timestep: 0', fontsize=16, y=0.985)
ani_flux = animation.FuncAnimation(
    fig, update_flux,
    frames=subgrid_flux.shape[0],
    interval=100,
    blit=True
)
ani_flux.save(save_path + "subgrid_flux_evolution.mp4", writer='ffmpeg')
plt.close()
print("Subgrid flux animation saved")

flux_1d = subgrid_flux.mean(axis=3)              
flux_pred_1d = subgrid_flux_pred_cnn.mean(axis=3)

flux_mean = flux_1d.mean(axis=0)                 
flux_std = flux_1d.std(axis=0)

flux_pred_mean = flux_pred_1d.mean(axis=0)
flux_pred_std = flux_pred_1d.std(axis=0)

Y = np.arange(flux_mean.shape[1])                

fig, axs = plt.subplots(10, 1, figsize=(6, 30), sharex=True)

for i in range(10):
    ax = axs[i]

    ax.plot(Y, flux_mean[i], label="Source", lw=2)
    ax.fill_between(Y,
                    flux_mean[i] - flux_std[i],
                    flux_mean[i] + flux_std[i],
                    alpha=0.2)

    ax.plot(Y, flux_pred_mean[i], label="Predicted", lw=2)
    ax.fill_between(Y,
                    flux_pred_mean[i] - flux_pred_std[i],
                    flux_pred_mean[i] + flux_pred_std[i],
                    alpha=0.2)

    ax.set_title(f"Channel {i}")
    ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig(save_path + "subgrid_flux_1d_mean_std.png", dpi=300)
plt.close()
print("Subgrid flux 1D mean/std plot saved")