
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ergane
from ergane import athinput_parser
from ergane.units import Units
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np



hr_sim = ergane.SimulationData(
    athinp='/Volumes/PortableSSD/Projects/SubgridCGMModel/builds/hr_build/src/kh_radiative_512.athinput',
    datafolder='/Volumes/PortableSSD/Projects/SubgridCGMModel/simulation_outputs/hr_build'
)
lr_sim = ergane.SimulationData(
    athinp='/Volumes/PortableSSD/Projects/SubgridCGMModel/builds/hr_build/src/kh_radiative_256.athinput',
    datafolder='/Volumes/PortableSSD/Projects/SubgridCGMModel/simulation_outputs/lr_build'
)
sg_sim = ergane.SimulationData(
    athinp='/Volumes/PortableSSD/Projects/SubgridCGMModel/builds/subgrid_model/src/neural_network.athinput',
    datafolder='/Volumes/PortableSSD/Projects/SubgridCGMModel/simulation_outputs/subgrid_model'
)

# Physical unit systems are automatically loaded from simulation parameters (athinput)




# Profiles along y axis averaged along x axis 
steady_state_start = 500
fields = ['density', 'temp', 'pressure', 'velx', 'vely']

profiles = {
    'hr': {f: [] for f in fields},
    'hr_coarse': {f: [] for f in fields},
    'lr': {f: [] for f in fields},
    'subgrid': {f: [] for f in fields}
}

for i in tqdm(range(steady_state_start, hr_sim.n_frames), desc='Calculating Profiles HR'):
    rho = hr_sim.density[i]
    press = hr_sim.pressure[i]
    vx = hr_sim.velx[i]
    vy = hr_sim.vely[i]
    temp = hr_sim.temperature[i]
    
    data = {
        'density': np.log10(np.maximum(rho, 1e-35)),
        'pressure': press,
        'velx': vx,
        'vely': vy,
        'temp': np.log10(np.maximum(temp, 1.0))
    }
    
    for f in fields:
        profiles['hr'][f].append(np.mean(data[f], axis=1))
        
    # Coarse grain (reshape and average) linear fields first, then take log10
    rho_coarse = rho.reshape(16, 32, 8, 32).mean(axis=(1, 3))
    press_coarse = press.reshape(16, 32, 8, 32).mean(axis=(1, 3))
    vx_coarse = vx.reshape(16, 32, 8, 32).mean(axis=(1, 3))
    vy_coarse = vy.reshape(16, 32, 8, 32).mean(axis=(1, 3))
    temp_coarse = temp.reshape(16, 32, 8, 32).mean(axis=(1, 3))
    
    cg_data = {
        'density': np.log10(np.maximum(rho_coarse, 1e-35)),
        'pressure': press_coarse,
        'velx': vx_coarse,
        'vely': vy_coarse,
        'temp': np.log10(np.maximum(temp_coarse, 1.0))
    }
    
    for f in fields:
        profiles['hr_coarse'][f].append(np.mean(cg_data[f], axis=1))


for i in tqdm(range(lr_sim.n_frames), desc='Calculating Profiles LR'):
    rho = lr_sim.density[i]
    press = lr_sim.pressure[i]
    vx = lr_sim.velx[i]
    vy = lr_sim.vely[i]
    temp = lr_sim.temperature[i]
    
    data = {
        'density': np.log10(np.maximum(rho, 1e-35)),
        'pressure': press,
        'velx': vx,
        'vely': vy,
        'temp': np.log10(np.maximum(temp, 1.0))
    }
    
    for f in fields:
        profiles['lr'][f].append(np.mean(data[f], axis=1))

for i in tqdm(range(sg_sim.n_frames), desc='Calculating Profiles SG'):
    rho = sg_sim.density[i + 501]
    press = sg_sim.pressure[i + 501]
    vx = sg_sim.velx[i + 501]
    vy = sg_sim.vely[i + 501]
    temp = sg_sim.temperature[i + 501]
    
    data = {
        'density': np.log10(np.maximum(rho, 1e-35)),
        'pressure': press,
        'velx': vx,
        'vely': vy,
        'temp': np.log10(np.maximum(temp, 1.0))
    }
    
    for f in fields:
        profiles['subgrid'][f].append(np.mean(data[f], axis=1))

# Convert to numpy arrays
for key in profiles:
    for f in fields:
        profiles[key][f] = np.asarray(profiles[key][f], dtype=np.float64)


# Extract y coordinate arrays (cell centers) and scale to physical length scales (parsecs)
first_hr_frame = hr_sim.get_frame(hr_sim.frame_numbers[0])
y_hr = first_hr_frame.yc / first_hr_frame.units.length

first_lr_frame = lr_sim.get_frame(lr_sim.frame_numbers[0])
y_lr = first_lr_frame.yc / first_lr_frame.units.length

fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)

def plot_profile_with_sd(ax, y_hr, y_lr, field, ylabel, is_legend=False):
    hr_data = profiles['hr'][field]
    coarse_data = profiles['hr_coarse'][field]
    lr_data = profiles['lr'][field]
    sg_data = profiles['subgrid'][field]

    # High Resolution
    line_hr, = ax.plot(y_hr, np.mean(hr_data, axis=0), label='High Resolution')
    ax.fill_between(y_hr, 
                    np.mean(hr_data, axis=0) - np.std(hr_data, axis=0),
                    np.mean(hr_data, axis=0) + np.std(hr_data, axis=0),
                    color=line_hr.get_color(), alpha=0.15)

    # Coarse Grained High Res
    line_coarse, = ax.plot(y_lr, np.mean(coarse_data, axis=0), label='Coarse Grained High Res (16x8)', linestyle='--')
    ax.fill_between(y_lr,
                    np.mean(coarse_data, axis=0) - np.std(coarse_data, axis=0),
                    np.mean(coarse_data, axis=0) + np.std(coarse_data, axis=0),
                    color=line_coarse.get_color(), alpha=0.15)

    # Low Resolution
    line_lr, = ax.plot(y_lr, np.mean(lr_data, axis=0), label='Low Resolution')
    ax.fill_between(y_lr,
                    np.mean(lr_data, axis=0) - np.std(lr_data, axis=0),
                    np.mean(lr_data, axis=0) + np.std(lr_data, axis=0),
                    color=line_lr.get_color(), alpha=0.15)

    # Subgrid
    line_sg, = ax.plot(y_lr, np.mean(sg_data, axis=0), label='Subgrid')
    ax.fill_between(y_lr,
                    np.mean(sg_data, axis=0) - np.std(sg_data, axis=0),
                    np.mean(sg_data, axis=0) + np.std(sg_data, axis=0),
                    color=line_sg.get_color(), alpha=0.15)

    ax.set_ylabel(ylabel)
    if is_legend:
        ax.legend()

# Subplots mapping: (ax, field_name, label_name, is_legend)
plot_mapping = [
    (axes[0, 0], 'density', r"Log Density $\langle \log_{10}(\rho) \rangle_x$ ($g\ cm^{-3}$)", True),
    (axes[0, 1], 'temp', r"Log Temperature $\langle \log_{10}(T) \rangle_x$ ($K$)", False),
    (axes[1, 0], 'pressure', r"Pressure $\langle P \rangle_x$ (dyn cm⁻²)", False),
    (axes[1, 1], 'velx', r"Velocity X $\langle u_x \rangle_x$ (cm s⁻¹)", False),
    (axes[2, 0], 'vely', r"Velocity Y $\langle u_y \rangle_x$ (cm s⁻¹)", False)
]

for ax, field, label, is_leg in plot_mapping:
    plot_profile_with_sd(ax, y_hr, y_lr, field, label, is_leg)

# Hide the unused subplot in bottom-right
fig.delaxes(axes[2, 1])

# Label x-axes of bottom subplots in both columns
axes[2, 0].set_xlabel("Y Position (pc)")
axes[1, 1].set_xlabel("Y Position (pc)")
axes[1, 1].tick_params(labelbottom=True)

plt.tight_layout()
plt.show()
