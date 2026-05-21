# Initial test script for checking the fmcl calculation

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.getcwd())
from data_preprocess import simulation_data
from matplotlib.colors import LogNorm

high_res_rho = np.random.choice([1e-3, 1e-1], size=(4096, 4096), p=[0.9, 0.1]) # cm^{-3}

plt.imshow(high_res_rho, cmap='grey_r', norm=LogNorm(vmin=1e-3, vmax=1e-1))
plt.colorbar(label=r'Density ($cm^{-3}$)')
plt.title(r"High Resolution (4096 $\times$ 4096) Density")
plt.xticks([])
plt.yticks([])
plt.savefig("mocks/hr_unif_density.jpg", dpi = 500)
plt.close()

sim_data = simulation_data()
sim_data.rho = high_res_rho
cg_sim_data = sim_data.coarse_grain(sim_data.rho, 256)

plt.imshow(cg_sim_data, cmap='grey_r')
plt.colorbar(label=r'Density ($cm^{-3}$)')
plt.title(r"Coarse Grained (16 $\times$ 16) Density")
plt.xticks([]) 
plt.yticks([])
plt.savefig("mocks/cg_unif_density.jpg", dpi = 500)
plt.close()

pressure = 1e3 * np.ones_like(sim_data.rho) # K cm^{-3}
sim_data.temp = pressure / sim_data.rho

fmcl_data = sim_data.calc_fmcl(sim_data.rho, sim_data.temp, 256)
plt.imshow(fmcl_data, cmap='viridis')
plt.colorbar(label=r'$f_{m}^{cl}$')
plt.title(r"Cold gas Mass fraction (16 $\times$ 16)")
plt.xticks([])
plt.yticks([])
plt.savefig("mocks/fmcl_unif.jpg", dpi = 500)
plt.close()