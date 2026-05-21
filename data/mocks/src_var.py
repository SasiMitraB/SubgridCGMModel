# Plots standard deviations of the source terms

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import data_preprocess
from data_preprocess import simulation_data
from conv_nn.indiv_cnn import snapshot_pred as conv_snapshot_pred
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
from tqdm import tqdm
import corner

resolution = (512, 256)  
downsample = 8
# file_path = f"/data3/home/dipayandatta/Subgrid_CGM_Models/data/files/Subgrid CGM Models/Without_cooling/rk2, plm/{resolution[0]}_{resolution[1]}_prateek/bin"
save_path = f"mocks/src/{resolution}_{downsample}/"
os.makedirs(save_path, exist_ok=True)

sim_data = simulation_data()
sim_data.resolution = resolution
sim_data.down_sample = downsample

folder_path = f"/tmp/dipayandatta/datafiles/l{resolution}_8"
file_path = f"/tmp/dipayandatta/athenak/kh_build/src/l{resolution[0]}_{resolution[1]}/bin"
if os.path.exists(f"{folder_path}"):
    saved_rho = np.load(f"{folder_path}/rho.npy")
    saved_temp = np.load(f"{folder_path}/temp.npy")
    saved_pressure = np.load(f"{folder_path}/pressure.npy")
    saved_ux = np.load(f"{folder_path}/ux.npy")
    saved_uy = np.load(f"{folder_path}/uy.npy")
    saved_eint = np.load(f"{folder_path}/eint.npy")
    saved_ps = np.load(f"{folder_path}/ps.npy")
else:
    sim_data.input_data(file_path, start=501)
    os.makedirs(folder_path, exist_ok=True)
    np.save(f"{folder_path}/rho.npy", sim_data.rho)
    np.save(f"{folder_path}/temp.npy", sim_data.temp)
    np.save(f"{folder_path}/pressure.npy", sim_data.pressure)
    np.save(f"{folder_path}/ux.npy", sim_data.ux)
    np.save(f"{folder_path}/uy.npy", sim_data.uy)
    np.save(f"{folder_path}/eint.npy", sim_data.eint)
    np.save(f"{folder_path}/ps.npy", sim_data.ps)

print("Input data loaded")

filters = range(1, 101)
time_steps = sim_data.delta_time * np.array(filters)
stds8 = np.zeros((5, len(filters)))
stds1 = np.zeros((5, len(filters)))

for i, filter in enumerate(filters):
    del sim_data

    sim_data = simulation_data()
    sim_data.resolution = resolution
    sim_data.down_sample = 8
    sim_data.total_time = 1.0

    sim_data.rho = saved_rho
    sim_data.temp = saved_temp
    sim_data.pressure = saved_pressure
    sim_data.ux = saved_ux
    sim_data.uy = saved_uy
    sim_data.eint = saved_eint
    sim_data.ps = saved_ps

    sim_data.filter_timesteps(filter)
    print(f"Time step = {sim_data.delta_time:.4e} Myr")

    source_term = sim_data.calc_all_source_terms()
    source_term_plot = np.transpose(source_term, axes=(1, 0, 2, 3))

    for channel in range(source_term_plot.shape[0]):
        S = source_term_plot[channel].ravel()
        stds8[channel][i] = np.std(S)

    del sim_data

    sim_data = simulation_data()
    sim_data.resolution = resolution
    sim_data.down_sample = 1
    sim_data.total_time = 1.0

    sim_data.rho = saved_rho
    sim_data.temp = saved_temp
    sim_data.pressure = saved_pressure
    sim_data.ux = saved_ux
    sim_data.uy = saved_uy
    sim_data.eint = saved_eint
    sim_data.ps = saved_ps

    sim_data.filter_timesteps(filter)
    print(f"Time step = {sim_data.delta_time:.4e} Myr")

    source_term = sim_data.calc_all_source_terms()
    source_term_plot = np.transpose(source_term, axes=(1, 0, 2, 3))

    for channel in range(source_term_plot.shape[0]):
        S = source_term_plot[channel].ravel()
        stds1[channel][i] = np.std(S)

# print(f"Minimum Energy Std = {np.min(stds[3])}")

channel_labels = ['Density', 'Momentum X', 'Momentum Y', 'Energy', 'Cold Density']
for channel in range(stds8.shape[0]):
    plt.plot(time_steps, stds8[channel], label=f'{channel_labels[channel]} (DS=8)', color='red')
    plt.plot(time_steps, stds1[channel], label=f'{channel_labels[channel]} (DS=1)', color='blue')
    plt.xlabel('Time Step (Myr)')
    plt.xscale('log')
    plt.ylabel('Std of Source Terms')
    plt.title('Stds of Source Terms vs Time Step')
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_path}/std_channel_{channel}.png", dpi=300)
    plt.close()
