# Initial test script for checking different quantities

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.getcwd())
from data_preprocess import simulation_data
from matplotlib.colors import LogNorm

df = pd.read_csv("mocks/1Darray_0.csv")
shape = (256, 512)
high_res_rho = df['dens'].to_numpy().reshape(shape)[::-1, ::-1]
high_res_temp = df['Temperature'].to_numpy().reshape(shape)[::-1, ::-1]

sim_data = simulation_data()
sim_data.rho = high_res_rho
sim_data.temp = high_res_temp
cg_sim_rho = sim_data.coarse_grain(sim_data.rho, 16)
cg_sim_temp = sim_data.coarse_grain(sim_data.temp, 16)

norm = LogNorm(vmin=1e-3, vmax=1e-1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
im0 = axes[0].imshow(high_res_rho, cmap='viridis', norm=norm)
axes[0].set_title(r"High Resolution (256 $\times$ 512) Density")
axes[0].set_xticks([])
axes[0].set_yticks([])
im1 = axes[1].imshow(cg_sim_rho, cmap='viridis', norm=norm)
axes[1].set_title(r"Coarse Grained (16 $\times$ 32) Density")
axes[1].set_xticks([])
axes[1].set_yticks([])
fig.colorbar(im1, ax=axes, location='right', shrink=0.85, label=r'Density ($cm^{-3}$)')
plt.savefig("mocks/sim0_density.jpg", dpi=500)
plt.close()

norm = LogNorm(vmin=1e4, vmax=1e6)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
im0 = axes[0].imshow(high_res_temp, cmap='viridis', norm=norm)
axes[0].set_title(r"High Resolution (256 $\times$ 512) Temperature")
axes[0].set_xticks([])
axes[0].set_yticks([])
im1 = axes[1].imshow(cg_sim_temp, cmap='viridis', norm=norm)
axes[1].set_title(r"Coarse Grained (16 $\times$ 32) Temperature")
axes[1].set_xticks([])
axes[1].set_yticks([])
fig.colorbar(im1, ax=axes, location='right', shrink=0.85, label=r'Temperature (K)')
plt.savefig("mocks/sim0_temp.jpg", dpi=500)
plt.close()

fmcl_data = sim_data.calc_fmcl(sim_data.rho, sim_data.temp, 16)
plt.imshow(fmcl_data, cmap='viridis')
plt.colorbar(label=r'$f_{m}^{cl}$')
plt.title(r"Cold gas Mass fraction (16 $\times$ 32)") 
plt.xticks([])
plt.yticks([])
plt.savefig("mocks/fmcl_sim0.jpg", dpi = 500)
plt.close()

hist_data = fmcl_data.flatten()
bins = np.linspace(0, 1, 101)
plt.hist(hist_data, bins=bins, histtype='step', color='black')
plt.xlabel(r'$f_{m}^{cl}$')
plt.ylabel(r'Number of pixels')
plt.title(r'Histogram of Cold gas Mass fraction')
plt.savefig("mocks/fmcl_hist_sim0.jpg", dpi=300)
plt.close()

