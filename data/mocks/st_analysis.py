# Plot different properties of the source terms (like autocorrelation, structure function, Fourier transform) 

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
from conv_nn.cnn import snapshot_pred as cnn_snapshot_pred
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
from tqdm import tqdm
from scipy.signal import correlate2d

resolution = (1024, 1024)  
nn_model = "c" # f (Feedforward NN) or c (Convolutional NN)
downsample = 8
file_path = f"/data3/home/dipayandatta/Subgrid_CGM_Models/data/files/Subgrid CGM Models/Without_cooling/rk2, plm/{resolution[0]}_{resolution[1]}_prateek/bin"
save_path = f"mocks/src/{resolution}_{downsample}/"
kernel_save_path = f"mocks/src/kernels/"
os.makedirs(save_path, exist_ok=True)
os.makedirs(kernel_save_path, exist_ok=True)
save_kernel = 11

sim_data = simulation_data()
sim_data.resolution = resolution
sim_data.down_sample = downsample
if nn_model == "f":
    sim_data.rho = np.load(f"../feedforward_nn/data_saves/{resolution}_{downsample}/rho.npy")
    sim_data.temp = np.load(f"../feedforward_nn/data_saves/{resolution}_{downsample}/temp.npy")
    sim_data.pressure = np.load(f"../feedforward_nn/data_saves/{resolution}_{downsample}/pressure.npy")
    sim_data.ux = np.load(f"../feedforward_nn/data_saves/{resolution}_{downsample}/ux.npy")
    sim_data.uy = np.load(f"../feedforward_nn/data_saves/{resolution}_{downsample}/uy.npy")
    sim_data.eint = np.load(f"../feedforward_nn/data_saves/{resolution}_{downsample}/eint.npy")
    sim_data.ps = np.load(f"../feedforward_nn/data_saves/{resolution}_{downsample}/ps.npy")
elif nn_model == "c":
    sim_data.rho = np.load(f"../conv_nn/data_saves/{resolution}_{downsample}/rho.npy")
    sim_data.temp = np.load(f"../conv_nn/data_saves/{resolution}_{downsample}/temp.npy")
    sim_data.pressure = np.load(f"../conv_nn/data_saves/{resolution}_{downsample}/pressure.npy")
    sim_data.ux = np.load(f"../conv_nn/data_saves/{resolution}_{downsample}/ux.npy")
    sim_data.uy = np.load(f"../conv_nn/data_saves/{resolution}_{downsample}/uy.npy")
    sim_data.eint = np.load(f"../conv_nn/data_saves/{resolution}_{downsample}/eint.npy")
    sim_data.ps = np.load(f"../conv_nn/data_saves/{resolution}_{downsample}/ps.npy")
print("Input data loaded")

high_res_rho = sim_data.rho
cg_rho = np.zeros((high_res_rho.shape[0], high_res_rho.shape[1] // sim_data.down_sample, high_res_rho.shape[2] // sim_data.down_sample))
fmcl_data = np.zeros((sim_data.rho.shape[0], sim_data.rho.shape[1] // sim_data.down_sample, sim_data.rho.shape[2] // sim_data.down_sample))
for i in tqdm(range(high_res_rho.shape[0]), desc = "Calculating cold gas mass fraction"):
    fmcl_data[i] = sim_data.calc_fmcl(sim_data.rho[i], sim_data.temp[i])

source_term = sim_data.calc_source_term()   
source_term_pred = np.zeros_like(source_term)

if nn_model == "f":
    snapshot_pred = fnn_snapshot_pred
elif nn_model == "c":
    snapshot_pred = cnn_snapshot_pred

for i in tqdm(range(source_term.shape[0]), desc = "Predicting source term"):
    source_term_pred[i] = snapshot_pred(sim_data.rho[i], sim_data.temp[i], sim_data.pressure[i], \
                                        sim_data.ux[i], sim_data.uy[i], sim_data.eint[i], sim_data.ps[i], \
                                        downsample, (sim_data.resolution[0], sim_data.resolution[1]))
residuals = source_term - source_term_pred

# Perform the FFT of source terms along x
st_x = np.mean(source_term, axis=1)
st_x_pred = np.mean(source_term_pred, axis=1)

fst_x = np.fft.fft(st_x, axis=1)
fst_x_pred = np.fft.fft(st_x_pred, axis=1)
k_val = np.fft.fftfreq(st_x.shape[1], d=10/cg_rho.shape[1])

fst_x = np.fft.fftshift(fst_x, axes=1)
fst_x_pred = np.fft.fftshift(fst_x_pred, axes=1)
k_val = np.fft.fftshift(k_val)

t_idxs = 7
colors = plt.cm.plasma(np.linspace(1, 0, t_idxs))
plt.figure(figsize=(10, 6))
for t_idx in tqdm(np.linspace(0, fst_x.shape[0] - 1, t_idxs, dtype=int), desc="Plotting FFT along X"):
    plt.plot(k_val, np.abs(fst_x[t_idx]), color=colors[int((t_idx/fst_x.shape[0])*t_idxs)], label=f'{t_idx/fst_x.shape[0]:.2} Myr')
plt.xlabel(r'$k\,(pc^{-1})$')
plt.xlim(0.0, np.max(k_val))
plt.ylabel(r'$C_k$')
plt.legend()
plt.title(rf'Source Term FFT along X for ${resolution[0]} \times {resolution[1]}$ resolution')
plt.savefig(f"{save_path}/fft_x.png", dpi=300)
plt.clf()

# FFT of predicted source terms along x
plt.figure(figsize=(10, 6))
for t_idx in tqdm(np.linspace(0, fst_x_pred.shape[0] - 1, t_idxs, dtype=int), desc="Plotting Predicted FFT along X"):
    plt.plot(k_val, np.abs(fst_x_pred[t_idx]), color=colors[int((t_idx/fst_x_pred.shape[0])*t_idxs)], label=f'{t_idx/fst_x_pred.shape[0]:.2} Myr')
plt.xlabel(r'$k\,(pc^{-1})$')
plt.xlim(0.0, np.max(k_val))
plt.ylabel(r'$C_k$')
plt.legend()
if nn_model == "f":
    plt.title(rf'Predicted Source Term FFT along X for ${resolution[0]} \times {resolution[1]}$ resolution (Feedforward NN)')
elif nn_model == "c":
    plt.title(rf'Predicted Source Term FFT along X for ${resolution[0]} \times {resolution[1]}$ resolution (Convolutional NN)')
plt.savefig(f"{save_path}/predicted_fft_x.png", dpi=300)
plt.clf()

def compute_structure_function_2d(field: np.ndarray) -> np.ndarray:
    field = field - np.mean(field)
    f2 = field**2
    autocorr = correlate2d(field, field, mode='full')
    norm = correlate2d(np.ones_like(field), np.ones_like(field), mode='full')
    sq_diff = correlate2d(f2, np.ones_like(field), mode='full') \
            + correlate2d(np.ones_like(field), f2, mode='full') \
            - 2 * autocorr
    sq_diff /= norm
    sq_diff /= np.max(sq_diff)  
    return sq_diff

def compute_autocorrelation_2d(field: np.ndarray) -> np.ndarray:
    field = field - np.mean(field)  
    autocorr = correlate2d(field, field, mode='full')
    norm = correlate2d(np.ones_like(field), np.ones_like(field), mode='full')
    autocorr /= norm
    autocorr /= np.max(autocorr)  
    return autocorr

def radial_average(structure_2d: np.ndarray) -> tuple:
    center = np.array(structure_2d.shape) // 2
    y, x = np.indices(structure_2d.shape)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(np.int32)

    tbin = np.bincount(r.ravel(), structure_2d.ravel())
    nr = np.bincount(r.ravel())
    radial_prof = tbin / nr
    return np.arange(len(radial_prof)), radial_prof

def time_autocorrelation(field: np.ndarray) -> tuple:
    T, Nx, Ny = field.shape
    field_demeaned = field - np.mean(field, axis=0)
    flat_field = field_demeaned.reshape(T, -1)

    autocorr = np.zeros(T)
    for tau in tqdm(range(T), desc="Calculating Time Autocorrelation"):
        prod = flat_field[:T - tau] * flat_field[tau:]
        autocorr[tau] = np.mean(prod)

    autocorr /= autocorr[0]
    lags = np.arange(T)
    return lags, autocorr

lags, autocorr = time_autocorrelation(source_term)
lags, autocorr_pred = time_autocorrelation(source_term_pred)
plt.figure(figsize=(10, 6))
plt.plot(lags, autocorr, color='blue', label='Source Term')
plt.plot(lags, autocorr_pred, color='red', label='Predicted Source Term')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5, label='Zero Line')
plt.xlabel(r'Time Steps')
plt.ylabel(r'$C(\tau)$')
plt.legend()
if nn_model == "f":
    plt.title(rf'Time Autocorrelation for ${resolution[0]} \times {resolution[1]}$ resolution (Feedforward NN)')
elif nn_model == "c":
    plt.title(rf'Time Autocorrelation for ${resolution[0]} \times {resolution[1]}$ resolution (Convolutional NN)')
plt.savefig(f"{save_path}/time_autocorrelation.png", dpi=300)
plt.clf()

np.save(kernel_save_path + f"tacf_{resolution}_{save_kernel}.npy", np.array([lags, autocorr, autocorr_pred]))

plt.figure(figsize=(10, 6))
for t_idx in tqdm(np.linspace(0, source_term.shape[0] - 1, t_idxs, dtype=int), desc="Plotting Autocorrelation"):
    autocorr_2d = compute_autocorrelation_2d(source_term[t_idx])
    r, radial_prof = radial_average(autocorr_2d)
    plt.plot(r, radial_prof, color=colors[int((t_idx/source_term.shape[0])*t_idxs)], label=f'{t_idx/source_term.shape[0]:.2} Myr')
# plt.xlabel(r'$r\,(pc)$')
# plt.xlim(0, 1.5)
plt.xlabel(r'$r$ (pixels)')
plt.xlim(0, 10)
plt.ylabel(r'$C(r)$')
plt.title(rf'Autocorrelation for ${resolution[0]} \times {resolution[1]}$ resolution')
plt.legend()
plt.savefig(f"{save_path}/autocorrelation.png", dpi=300)
plt.clf()

plt.figure(figsize=(10, 6))
for t_idx in tqdm(np.linspace(0, source_term.shape[0] - 1, t_idxs, dtype=int), desc="Plotting Predicted Autocorrelation"):
    autocorr_2d_pred = compute_autocorrelation_2d(source_term_pred[t_idx])
    r, radial_prof_pred = radial_average(autocorr_2d_pred)
    plt.plot(r, radial_prof_pred, color=colors[int((t_idx/source_term.shape[0])*t_idxs)], label=f'{t_idx/source_term.shape[0]:.2} Myr')
# plt.xlabel(r'$r\,(pc)$')
# plt.xlim(0, 1.5)
plt.xlabel(r'$r$ (pixels)')
plt.xlim(0, 10)
plt.ylabel(r'$C(r)$')
if nn_model == "f":
    plt.title(rf'Predicted Autocorrelation for ${resolution[0]} \times {resolution[1]}$ resolution (Feedforward NN)')
elif nn_model == "c":
    plt.title(rf'Predicted Autocorrelation for ${resolution[0]} \times {resolution[1]}$ resolution (Convolutional NN)')
plt.legend()
plt.savefig(f"{save_path}/predicted_autocorrelation.png", dpi=300)
plt.clf()

plt.figure(figsize=(10, 6))
for t_idx in tqdm(np.linspace(0, source_term.shape[0] - 1, t_idxs, dtype=int), desc="Plotting Structure Function"):
    structure_2d = compute_structure_function_2d(source_term[t_idx])
    r, radial_prof = radial_average(structure_2d)
    plt.plot(r, radial_prof, color=colors[int((t_idx/source_term.shape[0])*t_idxs)], label=f'{t_idx/source_term.shape[0]:.2} Myr')
# plt.xlabel(r'$r\,(pc)$')
# plt.xlim(0, 1.5)
plt.xlabel(r'$r$ (pixels)')
# plt.xlim(0, 10)
plt.ylabel(r'$S(r)$')
plt.title(rf'Structure Function for ${resolution[0]} \times {resolution[1]}$ resolution')
plt.legend()
plt.savefig(f"{save_path}/structure_function.png", dpi=300)
plt.clf()

plt.figure(figsize=(10, 6))
for t_idx in tqdm(np.linspace(0, source_term.shape[0] - 1, t_idxs, dtype=int), desc="Plotting Predicted Structure Function"):
    structure_2d_pred = compute_structure_function_2d(source_term_pred[t_idx])
    r, radial_prof = radial_average(structure_2d_pred)
    plt.plot(r, radial_prof, color=colors[int((t_idx/source_term.shape[0])*t_idxs)], label=f'{t_idx/source_term.shape[0]:.2} Myr')
# plt.xlabel(r'$r\,(pc)$')
# plt.xlim(0, 1.5)
plt.xlabel(r'$r$ (pixels)')
# plt.xlim(0, 10)
plt.ylabel(r'$S(r)$')
if nn_model == "f":
    plt.title(rf'Predicted Structure Function for ${resolution[0]} \times {resolution[1]}$ resolution (Feedforward NN)')
elif nn_model == "c":
    plt.title(rf'Predicted Structure Function for ${resolution[0]} \times {resolution[1]}$ resolution (Convolutional NN)')
plt.legend()
plt.savefig(f"{save_path}/predicted_structure_function.png", dpi=300)
plt.clf()