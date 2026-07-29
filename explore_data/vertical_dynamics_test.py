import os
import sys
import gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Append project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from ergane import SimulationData

# --- Physical Constants ---
m_H = 1.6726219e-24   # Hydrogen mass in g
k_B = 1.380649e-16    # Boltzmann constant in erg/K
M_sun = 1.98847e33    # Solar mass in g
yr = 3.15576e7        # Year in seconds
pc = 3.08568e18       # Parsec in cm
kpc = 3.08568e21      # Kiloparsec in cm

# --- Code Units from athinput ---
L_cgs = 3.08568e18    # length unit (1 pc)
M_cgs = 4.91417e31    # mass unit
T_cgs = 3.15576e13    # time unit (1 Myr)
mu = 0.62

# --- Derived Unit Conversions ---
V_cgs = L_cgs / T_cgs                             # Velocity unit in cm/s
RHO_cgs = M_cgs / (L_cgs**3)                      # Density unit in g/cm^3

len_to_pc = L_cgs / pc                            # 1.0 pc per code length
n_to_cm3 = RHO_cgs / (mu * m_H)                   # ~1.61 cm^-3 per code density
T_to_K = V_cgs**2 * mu * m_H / k_B                # ~71.8 K per code temperature
mflux_to_Msun_yr_kpc2 = (RHO_cgs * V_cgs) / (M_sun / (yr * kpc**2)) # ~0.0247 M_sun/yr/kpc^2 per code flux

# Define simulations in a dictionary
sims = {
    'HR': SimulationData(datafolder='/home/sasi/Projects/SubgridCGMModel/simulation_outputs/subgrid_vertical_run_ismcooling_hr/bin'),
    'LR': SimulationData(datafolder='/home/sasi/Projects/SubgridCGMModel/simulation_outputs/subgrid_vertical_run_ismcooling'),
    'Subgrid': SimulationData(datafolder='/home/sasi/Projects/SubgridCGMModel/simulation_outputs/subgrid_vertical_run_usersourceterms')
}

# Create figure with 3 subplots sharing the X-axis
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 12))

for label, sim in sims.items():
    density_log_profiles = []
    temperature_profiles = []
    mass_flux_profiles = []
    rhovy_flux_profiles = []
    
    
    start_frame = 500
    if sim.n_frames <= start_frame:
        start_frame = 0  # Fallback just in case a sim has fewer than 500 frames
        
    for i in tqdm(range(start_frame, sim.n_frames), desc=f"Processing {label}"):
        # Load raw arrays (in code units)
        density = sim.density[i]
        temperature = sim.temperature[i]
        velx = sim.velx[i]
        vely = sim.vely[i]
        
        # 1. Density -> Log10 Number Density [cm^-3]
        density_log_profiles.append(np.mean(np.log10(density * n_to_cm3), axis=0))

        # 2. Temperature -> Kelvin [K]
        # (If sim.temperature already returns Kelvin, remove `* T_to_K`)
        temperature_profiles.append(np.mean(np.log10(temperature), axis=0))

        # 3. Mass Flux along x -> [Msun / yr / kpc^2]
        mass_flux = density * velx
        mass_flux_profiles.append(np.mean(mass_flux * mflux_to_Msun_yr_kpc2, axis=0))

        rhovy_flux_profiles.append(np.mean( velx , axis=0))

    # Convert to arrays and compute statistics
    density_log_profiles = np.asarray(density_log_profiles)
    temp_profiles = np.asarray(temperature_profiles)
    mflux_profiles = np.asarray(mass_flux_profiles)
    rhovy_flux_profiles = np.asarray(rhovy_flux_profiles)
    
    mean_density_log = np.mean(density_log_profiles, axis=0)
    std_density_log = np.std(density_log_profiles, axis=0)
    
    mean_temp = np.mean(temp_profiles, axis=0)
    std_temp = np.std(temp_profiles, axis=0)
    
    mean_mflux = np.mean(mflux_profiles, axis=0)
    std_mflux = np.std(mflux_profiles, axis=0)

    mean_velx = np.mean(rhovy_flux_profiles, axis=0)
    std_velx = np.std(rhovy_flux_profiles, axis=0)

    # Dynamically calculate cell-centered xrange in parsecs
    n_cells = len(mean_density_log)
    edges = np.linspace(-5, 5, n_cells + 1)
    xrange = 0.5 * (edges[:-1] + edges[1:]) * len_to_pc

    # --- Plot Density on top subplot (ax1) ---
    ax1.plot(xrange, mean_density_log, label=label)
    ax1.fill_between(
        xrange, 
        mean_density_log - std_density_log, 
        mean_density_log + std_density_log, 
        alpha=0.2
    )

    # --- Plot Temperature on middle subplot (ax2) ---
    ax2.plot(xrange, mean_temp, label=label)
    ax2.fill_between(
        xrange, 
        mean_temp - std_temp, 
        mean_temp + std_temp, 
        alpha=0.2
    )

    # --- Plot Mass Flux on bottom subplot (ax3) ---
    ax3.plot(xrange, mean_mflux, label=label)
    ax3.fill_between(
        xrange, 
        mean_mflux - std_mflux, 
        mean_mflux + std_mflux, 
        alpha=0.2
    )

    #ax4.plot(xrange, mean_velx, label=label)
    #ax4.fill_between(
    #    xrange, 
    #    mean_velx - std_velx, 
    #    mean_velx + std_velx,
    #    alpha=0.2,
    #)

    # Free up memory before loading the next simulation
    del density_log_profiles, temp_profiles, mflux_profiles
    gc.collect()

# Formatting ax1 (Density)
ax1.set_ylabel(r"$\log_{10}(n) \ [\mathrm{cm}^{-3}]$")
ax1.legend(loc='best')
ax1.set_title("Vertical Profiles Comparison")

# Formatting ax2 (Temperature)
ax2.set_ylabel(r"$\log_{10} \left( T \ [\mathrm{K}] \right)$")

# Formatting ax3 (Mass Flux)
ax3.set_xlabel(r"$x \ [\mathrm{pc}]$")
ax3.set_ylabel(r"$\rho v_x \ [M_\odot \ \mathrm{yr}^{-1} \ \mathrm{kpc}^{-2}]$")
ax3.axhline(0, color='black', linewidth=0.5, linestyle='--') # Zero flux line for reference

# ax4.set_xlabel(r"$x \ [\mathrm{pc}]$")
# ax4.set_ylabel(r'$v_x$')

# Adjust layout and save
plt.tight_layout()
plt.savefig("vertical_profiles_comparison.png", dpi=150)
plt.close()