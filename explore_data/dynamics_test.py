
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ergane
from ergane import athinput_parser
from ergane.units import Units
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np


# New Version that truncates to 10^4.5 to 10^5.5
def lambda_cool(temp, mask=False):
    """
    Cooling function ISMCoolFn translated from AthenaK C++.
    Works on scalars or numpy arrays (any shape).
    Returns Λ(T) in erg cm^3 / s.
    """
    logt = np.log10(temp)

    lhd = np.array(
        [
            -22.5977,
            -21.9689,
            -21.5972,
            -21.4615,
            -21.4789,
            -21.5497,
            -21.6211,
            -21.6595,
            -21.6426,
            -21.5688,
            -21.4771,
            -21.3755,
            -21.2693,
            -21.1644,
            -21.0658,
            -20.9778,
            -20.8986,
            -20.8281,
            -20.7700,
            -20.7223,
            -20.6888,
            -20.6739,
            -20.6815,
            -20.7051,
            -20.7229,
            -20.7208,
            -20.7058,
            -20.6896,
            -20.6797,
            -20.6749,
            -20.6709,
            -20.6748,
            -20.7089,
            -20.8031,
            -20.9647,
            -21.1482,
            -21.2932,
            -21.3767,
            -21.4129,
            -21.4291,
            -21.4538,
            -21.5055,
            -21.5740,
            -21.6300,
            -21.6615,
            -21.6766,
            -21.6886,
            -21.7073,
            -21.7304,
            -21.7491,
            -21.7607,
            -21.7701,
            -21.7877,
            -21.8243,
            -21.8875,
            -21.9738,
            -22.0671,
            -22.1537,
            -22.2265,
            -22.2821,
            -22.3213,
            -22.3462,
            -22.3587,
            -22.3622,
            -22.3590,
            -22.3512,
            -22.3420,
            -22.3342,
            -22.3312,
            -22.3346,
            -22.3445,
            -22.3595,
            -22.3780,
            -22.4007,
            -22.4289,
            -22.4625,
            -22.4995,
            -22.5353,
            -22.5659,
            -22.5895,
            -22.6059,
            -22.6161,
            -22.6208,
            -22.6213,
            -22.6184,
            -22.6126,
            -22.6045,
            -22.5945,
            -22.5831,
            -22.5707,
            -22.5573,
            -22.5434,
            -22.5287,
            -22.5140,
            -22.4992,
            -22.4844,
            -22.4695,
            -22.4543,
            -22.4392,
            -22.4237,
            -22.4087,
            -22.3928,
        ]
    )

    lam = np.zeros_like(temp, dtype=float)

    # turn off cooling below 1e4 K
    mask_off = logt <= 4.0
    lam[mask_off] = 0.0

    # KI02 regime (4.0 < logT <= 4.2)
    mask_ki = (logt > 4.0) & (logt <= 4.2)
    if np.any(mask_ki):
        lam[mask_ki] = 2.0e-19 * np.exp(
            -1.184e5 / (temp[mask_ki] + 1.0e3)
        ) + 2.8e-28 * np.sqrt(temp[mask_ki]) * np.exp(-92.0 / temp[mask_ki])

    # CGOLS fit (logT > 8.15)
    mask_hi = logt > 8.15
    lam[mask_hi] = 10.0 ** (0.45 * logt[mask_hi] - 26.065)

    # SPEX interpolation (4.2 < logT <= 8.15)
    mask_mid = (logt > 4.2) & (logt <= 8.15)
    if np.any(mask_mid):
        ipps = (25.0 * logt[mask_mid] - 103).astype(int)
        # Clamp to [0,100] like C++
        ipps = np.clip(ipps, 0, 100)
        x0 = 4.12 + 0.04 * ipps
        dx = logt[mask_mid] - x0
        logcool = (lhd[ipps + 1] * dx - lhd[ipps] * (dx - 0.04)) * 25.0
        lam[mask_mid] = 10.0**logcool
    if mask:
        mask_off = (logt < 4.5) | (logt > 5.5)
        lam[mask_off] = 0.0

    return lam




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
fields = ['density', 'temp', 'pressure', 'velx', 'vely', 'emissivity']

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
    
    emis = (rho**2) * lambda_cool(temp, mask=True)
    
    data = {
        'density': np.log10(np.maximum(rho, 1e-35)),
        'pressure': press,
        'velx': vx,
        'vely': vy,
        'temp': np.log10(np.maximum(temp, 1.0)),
        'emissivity': emis
    }
    
    for f in fields:
        profiles['hr'][f].append(np.mean(data[f], axis=1))
        
    # Coarse grain (reshape and average) linear fields first, then take log10
    rho_coarse = rho.reshape(16, 32, 8, 32).mean(axis=(1, 3))
    press_coarse = press.reshape(16, 32, 8, 32).mean(axis=(1, 3))
    vx_coarse = vx.reshape(16, 32, 8, 32).mean(axis=(1, 3))
    vy_coarse = vy.reshape(16, 32, 8, 32).mean(axis=(1, 3))
    temp_coarse = temp.reshape(16, 32, 8, 32).mean(axis=(1, 3))
    emis_coarse = emis.reshape(16, 32, 8, 32).mean(axis=(1, 3))
    
    cg_data = {
        'density': np.log10(np.maximum(rho_coarse, 1e-35)),
        'pressure': press_coarse,
        'velx': vx_coarse,
        'vely': vy_coarse,
        'temp': np.log10(np.maximum(temp_coarse, 1.0)),
        'emissivity': emis_coarse
    }
    
    for f in fields:
        profiles['hr_coarse'][f].append(np.mean(cg_data[f], axis=1))


for i in tqdm(range(lr_sim.n_frames), desc='Calculating Profiles LR'):
    rho = lr_sim.density[i]
    press = lr_sim.pressure[i]
    vx = lr_sim.velx[i]
    vy = lr_sim.vely[i]
    temp = lr_sim.temperature[i]
    emis = (rho**2) * lambda_cool(temp, mask=True)
    
    data = {
        'density': np.log10(np.maximum(rho, 1e-35)),
        'pressure': press,
        'velx': vx,
        'vely': vy,
        'temp': np.log10(np.maximum(temp, 1.0)),
        'emissivity': emis
    }
    
    for f in fields:
        profiles['lr'][f].append(np.mean(data[f], axis=1))

for i in tqdm(range(sg_sim.n_frames), desc='Calculating Profiles SG'):
    rho = sg_sim.density[i + 501]
    press = sg_sim.pressure[i + 501]
    vx = sg_sim.velx[i + 501]
    vy = sg_sim.vely[i + 501]
    temp = sg_sim.temperature[i + 501]
    emis = (rho**2) * lambda_cool(temp, mask=True)
    
    data = {
        'density': np.log10(np.maximum(rho, 1e-35)),
        'pressure': press,
        'velx': vx,
        'vely': vy,
        'temp': np.log10(np.maximum(temp, 1.0)),
        'emissivity': emis
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

# =======================================================
# Emissivity Profile Plot
# =======================================================
fig_emis, ax_emis = plt.subplots(figsize=(8, 6))

def plot_emissivity(ax, y_coords, data_array, label, color, ls='-'):
    # Average over time
    mean_emis = np.mean(data_array, axis=0)
    std_emis = np.std(data_array, axis=0)
    
    # Integrated Emissivity over the Y-domain
    sig_c = np.trapezoid(mean_emis, y_coords)
    
    line, = ax.plot(y_coords, mean_emis, label=rf"{label} ($\Sigma_c$={sig_c:.2e})", color=color, ls=ls, lw=2)
    ax.fill_between(y_coords,
                    np.clip(mean_emis - std_emis, 1e-30, None), # Prevent log-scale crash on negatives
                    mean_emis + std_emis,
                    color=line.get_color(), alpha=0.15)

plot_emissivity(ax_emis, y_hr, profiles['hr']['emissivity'], 'High Resolution', 'blue')
plot_emissivity(ax_emis, y_lr, profiles['hr_coarse']['emissivity'], 'Coarse Grained HR', 'blue', ls='--')
plot_emissivity(ax_emis, y_lr, profiles['lr']['emissivity'], 'Low Resolution', 'orange')
plot_emissivity(ax_emis, y_lr, profiles['subgrid']['emissivity'], 'Subgrid', 'green')

ax_emis.set_yscale("log")
#ax_emis.set_ylim(2e-28, 1e-24)
ax_emis.set_xlabel("Y Position (pc)")
ax_emis.set_ylabel(r"$\langle \rho^2 \Lambda(T) \rangle_x$ (erg cm$^{-3}$ s$^{-1}$)")
ax_emis.set_title(r"Mean Emissivity Profile vs $y$")
ax_emis.grid(True, ls="--", alpha=0.5)
ax_emis.legend()

plt.tight_layout()
plt.show()