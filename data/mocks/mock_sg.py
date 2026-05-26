# Python script for comparing the results of HR, SG, and LR simulations

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
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

def divergence(f, dx, dy):
    dFx_dx = np.gradient(f[0], dy, dx)[1]
    dFy_dy = np.gradient(f[1], dy, dx)[0]
    return dFx_dx + dFy_dy

resolution = (16, 8)
file_path = f"/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/sd_build/src/spc{resolution[0]}_{resolution[1]}/bin"
save_path = f"mocks/sg/spc{resolution}/"
os.makedirs(save_path, exist_ok=True)

sim_data = simulation_data()
sim_data.resolution = resolution
sim_data.input_data(file_path, start=501)
sim_data.input_cons_data(file_path, start=501)

rho = sim_data.rho
pres = sim_data.pressure
temp = sim_data.temp
ien = sim_data.eint
ux = sim_data.ux
uy = sim_data.uy
fmcl = sim_data.frho

cons_rho = sim_data.cons_rho
cons_momx = sim_data.cons_momx
cons_momy = sim_data.cons_momy
cons_ener = sim_data.cons_ener
cons_ps = sim_data.cons_ps

lr_frac = np.zeros_like(temp)
lr_frac[temp < sim_data.T_cutoff] = 1.0
frac = sim_data.frho

lr_resolution = resolution
lr_file_path = f"/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/kh_build/src/sc{lr_resolution[0]}_{lr_resolution[1]}/bin"
lr_sim_data = simulation_data()
lr_sim_data.resolution = lr_resolution
lr_sim_data.input_data(lr_file_path, start=501)
lr_rho = lr_sim_data.rho
lr_temp = lr_sim_data.temp
lr_pres = lr_sim_data.pressure
lr_ux = lr_sim_data.ux
lr_uy = lr_sim_data.uy
lr_ien = lr_sim_data.eint

lr_sim_data.input_cons_data(lr_file_path, start=501)
lr_cons_rho = lr_sim_data.cons_rho
lr_cons_momx = lr_sim_data.cons_momx
lr_cons_momy = lr_sim_data.cons_momy
lr_cons_ener = lr_sim_data.cons_ener
lr_cons_ps = lr_sim_data.cons_ps

lr_fmcl = (lr_temp < 1e5).astype(float)

hr_resolution = (512, 256)
hr_downsample = 32
hr_file_path = f"/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/kh_build/src/sc{hr_resolution[0]}_{hr_resolution[1]}/bin"
hr_sim_data = simulation_data()
hr_sim_data.resolution = hr_resolution
hr_sim_data.down_sample = hr_downsample
# hr_sim_data.input_data(hr_file_path)
# hr_rho = hr_sim_data.rho
# hr_temp = hr_sim_data.temp
hr_folder_path = f"/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/datafiles/sc{hr_resolution}_{hr_downsample}"
hr_rho = np.load(f"{hr_folder_path}/rho.npy")
hr_temp = np.load(f"{hr_folder_path}/temp.npy")
hr_pres = np.load(f"{hr_folder_path}/pressure.npy")
hr_ux = np.load(f"{hr_folder_path}/ux.npy")
hr_uy = np.load(f"{hr_folder_path}/uy.npy")
hr_ien = np.load(f"{hr_folder_path}/eint.npy")

hr_cons_rho = np.load(f"{hr_folder_path}/cons_rho.npy")
hr_cons_momx = np.load(f"{hr_folder_path}/cons_mx.npy")
hr_cons_momy = np.load(f"{hr_folder_path}/cons_my.npy")
hr_cons_ener = np.load(f"{hr_folder_path}/cons_ener.npy")
hr_cons_ps = np.load(f"{hr_folder_path}/cons_ps.npy")

cg_hr_rho = np.zeros((hr_rho.shape[0], hr_rho.shape[1] // hr_downsample, hr_rho.shape[2] // hr_downsample))
cg_hr_temp = np.zeros((hr_rho.shape[0], hr_rho.shape[1] // hr_downsample, hr_rho.shape[2] // hr_downsample))
cg_hr_pres = np.zeros((hr_rho.shape[0], hr_rho.shape[1] // hr_downsample, hr_rho.shape[2] // hr_downsample))    
cg_hr_ux = np.zeros((hr_rho.shape[0], hr_rho.shape[1] // hr_downsample, hr_rho.shape[2] // hr_downsample))
cg_hr_uy = np.zeros((hr_rho.shape[0], hr_rho.shape[1] // hr_downsample, hr_rho.shape[2] // hr_downsample))
cg_hr_ien = np.zeros((hr_rho.shape[0], hr_rho.shape[1] // hr_downsample, hr_rho.shape[2] // hr_downsample))

cg_hr_cons_rho = np.zeros((hr_rho.shape[0], hr_rho.shape[1] // hr_downsample, hr_rho.shape[2] // hr_downsample))
cg_hr_cons_momx = np.zeros((hr_rho.shape[0], hr_rho.shape[1] // hr_downsample, hr_rho.shape[2] // hr_downsample))
cg_hr_cons_momy = np.zeros((hr_rho.shape[0], hr_rho.shape[1] // hr_downsample, hr_rho.shape[2] // hr_downsample))
cg_hr_cons_ener = np.zeros((hr_rho.shape[0], hr_rho.shape[1] // hr_downsample, hr_rho.shape[2] // hr_downsample))
cg_hr_cons_ps = np.zeros((hr_rho.shape[0], hr_rho.shape[1] // hr_downsample, hr_rho.shape[2] // hr_downsample))

cg_hr_fmcl = np.zeros((hr_rho.shape[0], hr_rho.shape[1] // hr_downsample, hr_rho.shape[2] // hr_downsample))

for i in tqdm(range(hr_rho.shape[0]), desc="Calculating CG HR"):
    cg_hr_rho[i] = hr_sim_data.coarse_grain(hr_rho[i])
    cg_hr_temp[i] = hr_sim_data.coarse_grain(hr_temp[i])
    cg_hr_pres[i] = hr_sim_data.coarse_grain(hr_pres[i])
    cg_hr_ux[i] = hr_sim_data.coarse_grain(hr_ux[i])
    cg_hr_uy[i] = hr_sim_data.coarse_grain(hr_uy[i])
    cg_hr_ien[i] = hr_sim_data.coarse_grain(hr_ien[i])

    cg_hr_cons_rho[i] = hr_sim_data.coarse_grain(hr_cons_rho[i])
    cg_hr_cons_momx[i] = hr_sim_data.coarse_grain(hr_cons_momx[i])
    cg_hr_cons_momy[i] = hr_sim_data.coarse_grain(hr_cons_momy[i])
    cg_hr_cons_ener[i] = hr_sim_data.coarse_grain(hr_cons_ener[i])
    cg_hr_cons_ps[i] = hr_sim_data.coarse_grain(hr_cons_ps[i])

    cg_hr_fmcl[i] = hr_sim_data.calc_fmcl(hr_rho[i], hr_temp[i])

cg_hr_rho = cg_hr_rho[:rho.shape[0]]
cg_hr_temp = cg_hr_temp[:temp.shape[0]]
cg_hr_pres = cg_hr_pres[:temp.shape[0]]

def compute_mean_std(arr, logspace=False):
    if logspace:
        arr = np.log10(arr)

    arr_1d = np.mean(arr, axis=2)   # avg over X
    mean = arr_1d.mean(axis=0)      # mean over time
    std  = arr_1d.std(axis=0)       # std over time

    return mean, std

quantities = [
    ("Density",      cg_hr_rho, cg_hr_temp, rho, temp, lr_rho),
    ("Temperature",  cg_hr_temp, cg_hr_temp, temp, temp, lr_temp),
    ("Pressure",     cg_hr_pres, cg_hr_temp, pres, temp, lr_pres),
    ("Ux Velocity",  cg_hr_ux, cg_hr_temp, ux, temp, lr_ux),
    ("Uy Velocity",  cg_hr_uy, cg_hr_temp, uy, temp, lr_uy)
]

fig, axs = plt.subplots(5, 1, figsize=(9, 20))
plt.subplots_adjust(hspace=0.35)

for idx, (title, hr_arr, _, sg_arr, _, lr_arr) in enumerate(quantities):

    is_log = title in ("Density", "Temperature")

    hr_mean, hr_std = compute_mean_std(hr_arr, logspace=is_log)
    sg_mean, sg_std = compute_mean_std(sg_arr, logspace=is_log)
    lr_mean, lr_std = compute_mean_std(lr_arr, logspace=is_log)

    ax = axs[idx]

    ax.plot(hr_mean, lw=2, label=f"HR ({hr_resolution[0]}×{hr_resolution[1]})")
    ax.fill_between(np.arange(len(hr_mean)), hr_mean-hr_std, hr_mean+hr_std, alpha=0.25)

    ax.plot(sg_mean, lw=2, label=f"SG ({resolution[0]}×{resolution[1]})")
    ax.fill_between(np.arange(len(sg_mean)), sg_mean-sg_std, sg_mean+sg_std, alpha=0.25)

    ax.plot(lr_mean, lw=2, label=f"LR ({lr_resolution[0]}×{lr_resolution[1]})")
    ax.fill_between(np.arange(len(lr_mean)), lr_mean-lr_std, lr_mean+lr_std, alpha=0.25)

    ax.set_title(f"{title} (Avg over X) — Mean ± 1σ")
    ax.set_xlabel("Y")
    ax.set_ylabel(("log10 " if is_log else "") + title)

    if is_log:
        ax.set_yscale("linear")  # already plotting log(mean), so keep linear scale
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()

plt.tight_layout()
plt.savefig(save_path + "profiles_mean_with_std_all.png", dpi=200)
plt.close(fig)

print("profiles_mean_with_std_all.png saved")

quantities_cons = [
    ("Conserved Density",       cg_hr_cons_rho,  cons_rho,    lr_cons_rho),
    ("Conserved MomX",          cg_hr_cons_momx, cons_momx,   lr_cons_momx),
    ("Conserved MomY",          cg_hr_cons_momy, cons_momy,   lr_cons_momy),
    ("Conserved Energy",        cg_hr_cons_ener, cons_ener,   lr_cons_ener),
    ("Passive Scalar",          cg_hr_cons_ps,   cons_ps,     lr_cons_ps),
    ("fmcl (T < 1e5)",          cg_hr_fmcl,      fmcl,     lr_fmcl)
]

fig, axs = plt.subplots(6, 1, figsize=(9, 24))
plt.subplots_adjust(hspace=0.4)

for idx, (title, hr_arr, sg_arr, lr_arr) in enumerate(quantities_cons):

    hr_mean, hr_std = compute_mean_std(hr_arr)
    sg_mean, sg_std = compute_mean_std(sg_arr)
    lr_mean, lr_std = compute_mean_std(lr_arr)

    ax = axs[idx]

    ax.plot(hr_mean, lw=2, label=f"HR ({hr_resolution[0]}×{hr_resolution[1]})")
    ax.fill_between(np.arange(len(hr_mean)), hr_mean-hr_std, hr_mean+hr_std, alpha=0.25)

    ax.plot(sg_mean, lw=2, label=f"SG ({resolution[0]}×{resolution[1]})")
    ax.fill_between(np.arange(len(sg_mean)), sg_mean-sg_std, sg_mean+sg_std, alpha=0.25)

    ax.plot(lr_mean, lw=2, label=f"LR ({lr_resolution[0]}×{lr_resolution[1]})")
    ax.fill_between(np.arange(len(lr_mean)), lr_mean-lr_std, lr_mean+lr_std, alpha=0.25)

    ax.set_title(f"{title} (Avg over X) — Mean ± 1σ")
    ax.set_xlabel("Y")
    ax.set_ylabel(title)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()

plt.tight_layout()
plt.savefig(save_path + "conserved_quantities_mean_with_std.png", dpi=200)
plt.close(fig)

print("conserved_quantities_mean_with_std.png saved")

def make_derived_plot(hr_field, sg_field, lr_field, title, ylabel, ax):
    hr_mean, hr_std = compute_mean_std(hr_field)
    sg_mean, sg_std = compute_mean_std(sg_field)
    lr_mean, lr_std = compute_mean_std(lr_field)

    ax.plot(hr_mean, lw=2, label=f"HR ({hr_resolution[0]}×{hr_resolution[1]})")
    ax.fill_between(np.arange(len(hr_mean)), hr_mean-hr_std, hr_mean+hr_std, alpha=0.25)

    ax.plot(sg_mean, lw=2, label=f"SG ({resolution[0]}×{resolution[1]})")
    ax.fill_between(np.arange(len(sg_mean)), sg_mean-sg_std, sg_mean+sg_std, alpha=0.25)

    ax.plot(lr_mean, lw=2, label=f"LR ({lr_resolution[0]}×{lr_resolution[1]})")
    ax.fill_between(np.arange(len(lr_mean)), lr_mean-lr_std, lr_mean+lr_std, alpha=0.25)

    ax.set_title(title)
    ax.set_xlabel("Y")
    ax.set_ylabel(ylabel)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()


# === Derived quantities ===

# 1. rho * ux
hr_rho_ux = cg_hr_rho * cg_hr_ux
sg_rho_ux = rho        * ux
lr_rho_ux = lr_rho     * lr_ux

# 2. rho * ux * uy
hr_rho_ux_uy = cg_hr_rho * cg_hr_ux * cg_hr_uy
sg_rho_ux_uy = rho        * ux        * uy
lr_rho_ux_uy = lr_rho     * lr_ux     * lr_uy

# 3. p + rho * uy^2
hr_mom_flux_y = cg_hr_pres + cg_hr_rho * cg_hr_uy**2
sg_mom_flux_y = pres       + rho       * uy**2
lr_mom_flux_y = lr_pres    + lr_rho    * lr_uy**2


# === Plot ===
fig, axs = plt.subplots(3, 1, figsize=(9, 15))
plt.subplots_adjust(hspace=0.35)

make_derived_plot(hr_rho_ux,    sg_rho_ux,    lr_rho_ux,
                  "ρ uₓ (Avg over X) — Mean ± 1σ", "ρ uₓ", axs[0])

make_derived_plot(hr_rho_ux_uy, sg_rho_ux_uy, lr_rho_ux_uy,
                  "ρ uₓ uᵧ (Avg over X) — Mean ± 1σ", "ρ uₓ uᵧ", axs[1])

make_derived_plot(hr_mom_flux_y, sg_mom_flux_y, lr_mom_flux_y,
                  "p + ρ uᵧ² (Avg over X) — Mean ± 1σ", "Momentum Flux (y)", axs[2])

plt.tight_layout()
plt.savefig(save_path + "derived_quantities_mean_with_std.png", dpi=200)
plt.close(fig)

print("derived_quantities_mean_with_std.png saved")

nt = hr_rho.shape[0]
cg_hr_mass_x    = np.zeros((nt, resolution[0], resolution[1]))
cg_hr_mass_y    = np.zeros_like(cg_hr_mass_x)
cg_hr_T_xx      = np.zeros_like(cg_hr_mass_x)
cg_hr_T_xy      = np.zeros_like(cg_hr_mass_x)
cg_hr_T_yy      = np.zeros_like(cg_hr_mass_x)
cg_hr_E_flux_x  = np.zeros_like(cg_hr_mass_x)
cg_hr_E_flux_y  = np.zeros_like(cg_hr_mass_x)

gamma = 1.6667

for i in tqdm(range(nt), desc="CG HR Fluxes"):
    hr_rho_i  = hr_rho[i]
    hr_ux_i   = hr_ux[i]
    hr_uy_i   = hr_uy[i]
    hr_pres_i = hr_pres[i]

    hr_E_i = hr_pres_i/(gamma - 1) + 0.5 * hr_rho_i * (hr_ux_i**2 + hr_uy_i**2)

    hr_mass_x_i   = hr_rho_i * hr_ux_i
    hr_mass_y_i   = hr_rho_i * hr_uy_i
    hr_T_xx_i     = hr_rho_i * hr_ux_i**2 + hr_pres_i
    hr_T_xy_i     = hr_rho_i * hr_ux_i * hr_uy_i
    hr_T_yy_i     = hr_rho_i * hr_uy_i**2 + hr_pres_i
    hr_E_flux_x_i = (hr_E_i + hr_pres_i) * hr_ux_i
    hr_E_flux_y_i = (hr_E_i + hr_pres_i) * hr_uy_i

    cg_hr_mass_x[i]    = hr_sim_data.coarse_grain(hr_mass_x_i)
    cg_hr_mass_y[i]    = hr_sim_data.coarse_grain(hr_mass_y_i)
    cg_hr_T_xx[i]      = hr_sim_data.coarse_grain(hr_T_xx_i)
    cg_hr_T_xy[i]      = hr_sim_data.coarse_grain(hr_T_xy_i)
    cg_hr_T_yy[i]      = hr_sim_data.coarse_grain(hr_T_yy_i)
    cg_hr_E_flux_x[i]  = hr_sim_data.coarse_grain(hr_E_flux_x_i)
    cg_hr_E_flux_y[i]  = hr_sim_data.coarse_grain(hr_E_flux_y_i)

sg_mass_x = rho    * ux
sg_mass_y = rho    * uy
lr_mass_x = lr_rho * lr_ux
lr_mass_y = lr_rho * lr_uy

sg_T_xx = rho * ux**2 + pres
sg_T_xy = rho * ux * uy
sg_T_yy = rho * uy**2 + pres

lr_T_xx = lr_rho * lr_ux**2 + lr_pres
lr_T_xy = lr_rho * lr_ux * lr_uy
lr_T_yy = lr_rho * lr_uy**2 + lr_pres

def compute_E(rho, ux, uy, pres, gamma=1.6667):
    return pres/(gamma - 1) + 0.5 * rho * (ux**2 + uy**2)

sg_E = compute_E(rho, ux, uy, pres)
lr_E = compute_E(lr_rho, lr_ux, lr_uy, lr_pres)

sg_E_flux_x = (sg_E + pres)    * ux
sg_E_flux_y = (sg_E + pres)    * uy
lr_E_flux_x = (lr_E + lr_pres) * lr_ux
lr_E_flux_y = (lr_E + lr_pres) * lr_uy

fig, axs = plt.subplots(4, 2, figsize=(12, 16))
plt.subplots_adjust(hspace=0.35)

print(cg_hr_mass_x.shape, sg_mass_x.shape, lr_mass_x.shape)
make_derived_plot(cg_hr_mass_x, sg_mass_x, lr_mass_x, "Mass Flux (ρ uₓ)", "ρ uₓ", axs[0, 0])
print(cg_hr_mass_y.shape, sg_mass_y.shape, lr_mass_y.shape)
make_derived_plot(cg_hr_mass_y, sg_mass_y, lr_mass_y, "Mass Flux (ρ uᵧ)", "ρ uᵧ", axs[0, 1])

make_derived_plot(cg_hr_T_xx, sg_T_xx, lr_T_xx, "Momentum Flux Tₓₓ = ρuₓ² + p", "Tₓₓ", axs[1, 0])
make_derived_plot(cg_hr_T_xy, sg_T_xy, lr_T_xy, "Momentum Flux Tₓᵧ = ρuₓuᵧ", "Tₓᵧ", axs[1, 1])

make_derived_plot(cg_hr_T_xy, sg_T_xy, lr_T_xy, "Momentum Flux Tᵧₓ = ρuₓuᵧ", "Tᵧₓ", axs[2, 0])
make_derived_plot(cg_hr_T_yy, sg_T_yy, lr_T_yy, "Momentum Flux Tᵧᵧ = ρuᵧ² + p", "Tᵧᵧ", axs[2, 1])

make_derived_plot(cg_hr_E_flux_x, sg_E_flux_x, lr_E_flux_x, "Energy Flux (E+p)uₓ", "(E+p)uₓ", axs[3, 0])
make_derived_plot(cg_hr_E_flux_y, sg_E_flux_y, lr_E_flux_y, "Energy Flux (E+p)uᵧ", "(E+p)uᵧ", axs[3, 1])

plt.tight_layout()
plt.savefig(save_path + "fluxes_mean_std.png", dpi=200)
plt.close(fig)

print("fluxes_mean_std.png saved")

cg_hr_div_mass = np.zeros_like(cg_hr_mass_x)
cg_hr_div_momx = np.zeros_like(cg_hr_mass_x)
cg_hr_div_momy = np.zeros_like(cg_hr_mass_x)

sg_div_mass = np.zeros_like(sg_mass_x)
sg_div_momx = np.zeros_like(sg_mass_x)
sg_div_momy = np.zeros_like(sg_mass_x)

lr_div_mass = np.zeros_like(lr_mass_x)
lr_div_momx = np.zeros_like(lr_mass_x)
lr_div_momy = np.zeros_like(lr_mass_x)

dy = sim_data.total_length / resolution[0]
dx = sim_data.total_width / resolution[1]

for i in range(nt):
    cg_hr_div_mass[i] = divergence([cg_hr_mass_x[i], cg_hr_mass_y[i]], dx, dy)
    cg_hr_div_momx[i] = divergence([cg_hr_T_xx[i],   cg_hr_T_xy[i]],   dx, dy)
    cg_hr_div_momy[i] = divergence([cg_hr_T_xy[i],   cg_hr_T_yy[i]],   dx, dy)

    sg_div_mass[i] = divergence([sg_mass_x[i], sg_mass_y[i]], dx, dy)
    sg_div_momx[i] = divergence([sg_T_xx[i],   sg_T_xy[i]],   dx, dy)
    sg_div_momy[i] = divergence([sg_T_xy[i],   sg_T_yy[i]],   dx, dy)

    lr_div_mass[i] = divergence([lr_mass_x[i], lr_mass_y[i]], dx, dy)
    lr_div_momx[i] = divergence([lr_T_xx[i],   lr_T_xy[i]],   dx, dy)
    lr_div_momy[i] = divergence([lr_T_xy[i],   lr_T_yy[i]],   dx, dy)

fig, axs = plt.subplots(3, 1, figsize=(10, 13))
plt.subplots_adjust(hspace=0.35)

make_derived_plot(cg_hr_div_mass, sg_div_mass, lr_div_mass, "Div Mass Flux", "∇·(ρu)", axs[0])
make_derived_plot(cg_hr_div_momx, sg_div_momx, lr_div_momx, "Div MomX Flux", "∇·Tₓ", axs[1])
make_derived_plot(cg_hr_div_momy, sg_div_momy, lr_div_momy, "Div MomY Flux", "∇·Tᵧ", axs[2])

plt.tight_layout()
plt.savefig(save_path + "divergence_fluxes_mean_std.png", dpi=200)
plt.close(fig)

print("divergence_fluxes_mean_std.png saved")

def compute_cold_mass(rho_arr, temp_arr, nx, ny):
    dx_pc = sim_data.total_width / nx
    dy_pc = sim_data.total_length / ny
    area = dx_pc * dy_pc
    thr = 1e5
    res = []
    for t in range(rho_arr.shape[0]):
        mask = temp_arr[t] < thr
        res.append(np.sum(rho_arr[t] * mask) * area)
    return np.array(res)

def compute_fmcl_mass_sg(rho_arr, fmcl_arr, nx, ny):
    dx_pc = sim_data.total_width / nx
    dy_pc = sim_data.total_length / ny
    area = dx_pc * dy_pc
    res = []
    for t in range(rho_arr.shape[0]):
        res.append(np.sum(rho_arr[t] * fmcl_arr[t]) * area)
    return np.array(res)

mass_hr = compute_cold_mass(cg_hr_rho, cg_hr_temp, resolution[0], resolution[1])
mass_sg = compute_cold_mass(rho,        temp,       resolution[0],   resolution[1])
mass_lr = compute_cold_mass(lr_rho,     lr_temp,    lr_resolution[0], lr_resolution[1])

fmcl_sg = compute_fmcl_mass_sg(rho, fmcl, resolution[0], resolution[1])

t = np.arange(len(mass_hr))

slope_hr,  intercept_hr  = np.polyfit(t, mass_hr, 1)
slope_sg,  intercept_sg  = np.polyfit(t, mass_sg, 1)
slope_lr,  intercept_lr  = np.polyfit(t, mass_lr, 1)
slope_fmc, intercept_fmc = np.polyfit(t, fmcl_sg, 1)

fit_hr  = slope_hr  * t + intercept_hr
fit_sg  = slope_sg  * t + intercept_sg
fit_lr  = slope_lr  * t + intercept_lr
fit_fmc = slope_fmc * t + intercept_fmc

plt.figure(figsize=(10, 6))

plt.plot(t, mass_hr, label="HR", lw=2)
plt.plot(t, mass_sg, label="SG", lw=2)
plt.plot(t, mass_lr, label="LR", lw=2)
plt.plot(t, fmcl_sg, label="SG fmcl", lw=2, ls="--")

plt.plot(t, fit_hr,  lw=1.8, ls=":", label=f"HR fit (slope = {slope_hr:.3e})")
plt.plot(t, fit_sg,  lw=1.8, ls=":", label=f"SG fit (slope = {slope_sg:.3e})")
plt.plot(t, fit_lr,  lw=1.8, ls=":", label=f"LR fit (slope = {slope_lr:.3e})")
plt.plot(t, fit_fmc, lw=1.8, ls=":", label=f"SG fmcl fit (slope = {slope_fmc:.3e})")

plt.xlabel("Timestep")
plt.ylabel("Mass (g pc²/cm³)")
plt.title("Cold Gas Mass (T < 1e5) Evolution + Linear Fits")
plt.grid(True, ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(save_path + "cold_mass_evolution.png", dpi=200)
plt.close()

print("Cold mass evolution plot saved (with fit slopes)")

# fields_hr = [cg_hr_rho, cg_hr_temp, cg_hr_pres, cg_hr_ux, cg_hr_uy, cg_hr_ien]
# fields_sg = [rho,         temp,       pres,       ux,       uy,       ien]
# fields_lr = [lr_rho,      lr_temp,    lr_pres,    lr_ux,    lr_uy,    lr_ien]

# titles = ["Density", "Temperature", "Pressure", "Ux", "Uy", "Internal Energy"]

# fig, axs = plt.subplots(6, 3, figsize=(8, 20))

# for i in range(6):
#     f0_hr = fields_hr[i][0]
#     f0_sg = fields_sg[i][0]
#     f0_lr = fields_lr[i][0]

#     arr0 = np.concatenate([f0_hr.flatten(), f0_sg.flatten(), f0_lr.flatten()])
#     vmin0 = arr0[arr0 > 0].min() if np.any(arr0 > 0) else arr0.min()
#     vmax0 = arr0.max()

#     use_log = (i == 0 or i == 1) and vmin0 > 0
#     norm0 = LogNorm(vmin=vmin0, vmax=vmax0) if use_log else None

#     axs[i, 0].imshow(f0_hr, origin='lower', cmap='plasma', norm=norm0)
#     axs[i, 0].set_title(f"HR {titles[i]}")
#     plt.colorbar(axs[i, 0].images[0], ax=axs[i, 0], fraction=0.035, pad=0.02)

#     axs[i, 1].imshow(f0_sg, origin='lower', cmap='plasma', norm=norm0)
#     axs[i, 1].set_title(f"SG {titles[i]}")
#     plt.colorbar(axs[i, 1].images[0], ax=axs[i, 1], fraction=0.035, pad=0.02)

#     axs[i, 2].imshow(f0_lr, origin='lower', cmap='plasma', norm=norm0)
#     axs[i, 2].set_title(f"LR {titles[i]}")
#     plt.colorbar(axs[i, 2].images[0], ax=axs[i, 2], fraction=0.035, pad=0.02)

# plt.tight_layout()
# plt.savefig(save_path + "all_fields_snapshot.png", dpi=200)
# plt.close(fig)
# print("Saved snapshot of all fields")

# fig, axs = plt.subplots(6, 3, figsize=(8, 20))

# ims = []
# colorbars = []

# for i in range(6):
#     im0 = axs[i, 0].imshow(fields_hr[i][0], origin='lower', cmap='plasma')
#     axs[i, 0].set_title(f"HR {titles[i]}")
#     cb0 = plt.colorbar(im0, ax=axs[i, 0], fraction=0.035, pad=0.02)

#     im1 = axs[i, 1].imshow(fields_sg[i][0], origin='lower', cmap='plasma')
#     axs[i, 1].set_title(f"SG {titles[i]}")
#     cb1 = plt.colorbar(im1, ax=axs[i, 1], fraction=0.035, pad=0.02)

#     im2 = axs[i, 2].imshow(fields_lr[i][0], origin='lower', cmap='plasma')
#     axs[i, 2].set_title(f"LR {titles[i]}")
#     cb2 = plt.colorbar(im2, ax=axs[i, 2], fraction=0.035, pad=0.02)

#     ims.append([im0, im1, im2])
#     colorbars.append([cb0, cb1, cb2])

# def update_all(frame):
#     updated = []

#     for i in range(6):
#         f_hr = fields_hr[i][frame]
#         f_sg = fields_sg[i][frame]
#         f_lr = fields_lr[i][frame]

#         arr = np.concatenate([f_hr.flatten(), f_sg.flatten(), f_lr.flatten()])
#         vmin = arr[arr > 0].min() if np.any(arr > 0) else arr.min()
#         vmax = arr.max()

#         use_log = (i == 0 or i == 1) and vmin > 0
#         norm = LogNorm(vmin=vmin, vmax=vmax) if use_log else None

#         if norm:
#             ims[i][0].set_norm(norm)
#             ims[i][1].set_norm(norm)
#             ims[i][2].set_norm(norm)
#         else:
#             ims[i][0].set_clim(vmin, vmax)
#             ims[i][1].set_clim(vmin, vmax)
#             ims[i][2].set_clim(vmin, vmax)

#         ims[i][0].set_data(f_hr)
#         ims[i][1].set_data(f_sg)
#         ims[i][2].set_data(f_lr)

#         for cb in colorbars[i]:
#             cb.update_normal(ims[i][0])

#         updated.extend(ims[i])

#     for ax in axs.flat:
#         ax.set_xlabel(f"Timestep: {frame}")

#     return updated

# ani_all = animation.FuncAnimation(
#     fig, update_all, frames=rho.shape[0], interval=100, blit=False
# )

# plt.tight_layout()
# ani_all.save(save_path + "all_fields_evolution.gif", writer="ffmpeg")
# plt.close(fig)
# print("Saved updated animation with correct dynamic colorbars")

# cons_fields_hr = [
#     cg_hr_cons_rho,
#     cg_hr_cons_momx,
#     cg_hr_cons_momy,
#     cg_hr_cons_ener,
#     cg_hr_cons_ps,
#     cg_hr_fmcl
# ]

# cons_fields_sg = [
#     cons_rho,
#     cons_momx,
#     cons_momy,
#     cons_ener,
#     cons_ps,
#     fmcl
# ]

# cons_fields_lr = [
#     lr_cons_rho,
#     lr_cons_momx,
#     lr_cons_momy,
#     lr_cons_ener,
#     lr_cons_ps,
#     lr_fmcl
# ]

# cons_titles = [
#     "Cons Density",
#     "Cons MomX",
#     "Cons MomY",
#     "Cons Energy",
#     "Cons Passive Scalar",
#     "fmcl"
# ]

# fig, axs = plt.subplots(6, 3, figsize=(8, 20))
# ims = []
# cbs = []

# for i in range(6):
#     im0 = axs[i, 0].imshow(cons_fields_hr[i][0], origin='lower', cmap='plasma')
#     cb0 = plt.colorbar(im0, ax=axs[i, 0], fraction=0.035, pad=0.02)

#     im1 = axs[i, 1].imshow(cons_fields_sg[i][0], origin='lower', cmap='plasma')
#     cb1 = plt.colorbar(im1, ax=axs[i, 1], fraction=0.035, pad=0.02)

#     im2 = axs[i, 2].imshow(cons_fields_lr[i][0], origin='lower', cmap='plasma')
#     cb2 = plt.colorbar(im2, ax=axs[i, 2], fraction=0.035, pad=0.02)

#     axs[i, 0].set_title(f"HR {cons_titles[i]}")
#     axs[i, 1].set_title(f"SG {cons_titles[i]}")
#     axs[i, 2].set_title(f"LR {cons_titles[i]}")

#     ims.append([im0, im1, im2])
#     cbs.append([cb0, cb1, cb2])


# def update_cons(frame):
#     updated = []

#     for i in range(6):
#         f_hr = cons_fields_hr[i][frame]
#         f_sg = cons_fields_sg[i][frame]
#         f_lr = cons_fields_lr[i][frame]

#         arr = np.concatenate([f_hr.flatten(), f_sg.flatten(), f_lr.flatten()])
#         vmin = arr[arr > 0].min() if np.any(arr > 0) else arr.min()
#         vmax = arr.max()

#         use_log = (i == 0 or i == 3) and vmin > 0    
#         norm = LogNorm(vmin=vmin, vmax=vmax) if use_log else None

#         if norm:
#             ims[i][0].set_norm(norm)
#             ims[i][1].set_norm(norm)
#             ims[i][2].set_norm(norm)
#         else:
#             ims[i][0].set_clim(vmin, vmax)
#             ims[i][1].set_clim(vmin, vmax)
#             ims[i][2].set_clim(vmin, vmax)

#         ims[i][0].set_data(f_hr)
#         ims[i][1].set_data(f_sg)
#         ims[i][2].set_data(f_lr)

#         for cb in cbs[i]:
#             cb.update_normal(ims[i][0])

#         updated.extend(ims[i])

#     for ax in axs.flat:
#         ax.set_xlabel(f"Timestep: {frame}")

#     return updated


# ani_cons = animation.FuncAnimation(
#     fig,
#     update_cons,
#     frames=cons_rho.shape[0],
#     interval=100,
#     blit=False
# )

# plt.tight_layout()
# ani_cons.save(save_path + "cons_fields_evolution.gif", writer="ffmpeg")
# plt.close(fig)

# print("Saved conserved-field animation with dynamic colorbars")

# fig, axs = plt.subplots(1, 3, figsize=(10, 5))

# im_hr_rho = axs[0].imshow(cg_hr_rho[0], origin='lower', cmap='plasma', norm=LogNorm())
# axs[0].set_title(rf'HR (${hr_resolution[0]} \times {hr_resolution[1]}$) Density')
# plt.colorbar(im_hr_rho, ax=axs[0], fraction=0.046, pad=0.04)

# im_rho = axs[1].imshow(rho[0], origin='lower', cmap='plasma', norm=LogNorm())
# axs[1].set_title(rf'SG (${resolution[0]} \times {resolution[1]}$) Density (sigma=3)')
# plt.colorbar(im_rho, ax=axs[1], fraction=0.046, pad=0.04)

# im_lr_rho = axs[2].imshow(lr_rho[0], origin='lower', cmap='plasma', norm=LogNorm())
# axs[2].set_title(rf'LR (${lr_resolution[0]} \times {lr_resolution[1]}$) Density')
# plt.colorbar(im_lr_rho, ax=axs[2], fraction=0.046, pad=0.04)

# def update_rho(frame):
#     im_hr_rho.set_data(cg_hr_rho[frame])
#     im_rho.set_data(rho[frame])
#     im_lr_rho.set_data(lr_rho[frame])
#     for ax in axs.flat:
#         ax.set_xlabel(f'Timestep: {frame}')
#     return [im_rho, im_hr_rho, im_lr_rho]

# ani_rho = animation.FuncAnimation(fig, update_rho, frames=rho.shape[0], interval=100, blit=True)
# ani_rho.save(save_path + "density_evolution.gif", writer='ffmpeg')
# plt.close(fig)
# print("Density evolution animation saved")

# bins = np.logspace(4, 6, 200)
# window = 10

# fig, ax = plt.subplots(figsize=(6, 5))
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlabel("Temperature [K]")
# ax.set_ylabel("PDF (volume-weighted, time-avg 10 steps)")
# ax.set_ylim(1e-7, 1e-3)
# ax.set_xlim(bins[0], bins[-1])

# (line_hr,) = ax.plot([], [], lw=2.0, label="HR")
# (line_lr,) = ax.plot([], [], lw=2.0, label="LR")
# (line_sg,) = ax.plot([], [], lw=2.0, label="SG")
# ax.legend()

# def update(frame):
#     ax.set_title(f"Time step {frame+1}")
#     end = min(frame + window, temp.shape[0])
#     h_hr, _ = np.histogram(cg_hr_temp[frame:end].ravel(), bins=bins, density=True)
#     h_lr, _ = np.histogram(lr_temp[frame:end].ravel(), bins=bins, density=True)
#     h_sg, _ = np.histogram(temp[frame:end].ravel(), bins=bins, density=True)
#     line_hr.set_data(bins[:-1], h_hr)
#     line_lr.set_data(bins[:-1], h_lr)
#     line_sg.set_data(bins[:-1], h_sg)
#     return line_hr, line_lr, line_sg
#     # return [line_sg]

# anim = FuncAnimation(fig, update, frames=temp.shape[0], interval=150, blit=True)
# plt.tight_layout()
# anim.save(save_path + "temperature_pdf_evolution.mp4", writer="ffmpeg")
# plt.close(fig)
# print("Temperature PDF evolution animation saved")

# # --- data arrays (nt, ny, nx) ---
# nt, ny_hr, nx_hr = cg_hr_rho.shape
# ny_sg, nx_sg = rho.shape[1], rho.shape[2]
# ny_lr, nx_lr = lr_rho.shape[1], lr_rho.shape[2]

# # --- domain size in x [pc] ---
# Lx = 10.0

# # --- wavenumbers (1/pc) ---
# kx_hr = 2*np.pi*np.fft.rfftfreq(nx_hr, d=Lx/nx_hr)
# kx_sg = 2*np.pi*np.fft.rfftfreq(nx_sg, d=Lx/nx_sg)
# kx_lr = 2*np.pi*np.fft.rfftfreq(nx_lr, d=Lx/nx_lr)

# # --- storage ---
# spectra_hr, spectra_sg, spectra_lr = [], [], []

# # --- compute spectra ---
# for t in range(nt):
#     # HR
#     fhat_hr = np.fft.rfft(cg_hr_rho[t], axis=-1)        # FFT along x (last axis)
#     power_hr = np.mean(np.abs(fhat_hr)**2, axis=0)      # average over y
#     spectra_hr.append(power_hr)

#     # SG
#     fhat_sg = np.fft.rfft(rho[t], axis=-1)
#     power_sg = np.mean(np.abs(fhat_sg)**2, axis=0)
#     spectra_sg.append(power_sg)

#     # LR
#     fhat_lr = np.fft.rfft(lr_rho[t], axis=-1)
#     power_lr = np.mean(np.abs(fhat_lr)**2, axis=0)
#     spectra_lr.append(power_lr)

# spectra_hr = np.array(spectra_hr)
# spectra_sg = np.array(spectra_sg)
# spectra_lr = np.array(spectra_lr)

# # --- animate ---
# fig, ax = plt.subplots(figsize=(7,5))

# line_hr, = ax.loglog(kx_hr, spectra_hr[0] + 1e-30, label="HR", color="red")  # add epsilon to avoid log(0)
# line_sg, = ax.loglog(kx_sg, spectra_sg[0] + 1e-30, label="SG", color="blue")
# line_lr, = ax.loglog(kx_lr, spectra_lr[0] + 1e-30, label="LR", color="green")

# ax.set_xlabel(r"$k_x$ [1/pc]")
# ax.set_ylabel("Power Spectrum")
# ax.set_ylim(1e-12, 1e1)
# ax.set_title("Fourier Spectrum Evolution")
# ax.legend(loc = "lower right")

# def update(frame):
#     line_hr.set_ydata(spectra_hr[frame] + 1e-30)
#     line_sg.set_ydata(spectra_sg[frame] + 1e-30)
#     line_lr.set_ydata(spectra_lr[frame] + 1e-30)
#     ax.set_title(f"Fourier Spectrum (timestep {frame})")
#     return [line_hr, line_sg, line_lr]

# ani = animation.FuncAnimation(fig, update, frames=nt, interval=100, blit=False)

# ani.save(save_path + "fourier_spectrum_hr_sg_lr.mp4", writer="ffmpeg")
# plt.close(fig)
# print("Fourier spectrum evolution (HR, SG, LR) saved")

# # --- domain sizes in pc (adjust if different)
# Lx, Ly = 10.0, 20.0   # box size in x, y

# # --- pixel sizes
# dx_hr, dy_hr = Lx / cg_hr_rho.shape[2], Ly / cg_hr_rho.shape[1]
# dx_sg, dy_sg = Lx / rho.shape[2],    Ly / rho.shape[1]
# dx_lr, dy_lr = Lx / lr_rho.shape[2], Ly / lr_rho.shape[1]

# cell_area_hr = dx_hr * dy_hr
# cell_area_sg = dx_sg * dy_sg
# cell_area_lr = dx_lr * dy_lr

# # --- cut indices (0 .. 512/7 pixels in y)
# ycut_hr = cg_hr_rho.shape[1] // 7
# ycut_sg = rho.shape[1]    // 7
# ycut_lr = lr_rho.shape[1] // 7

# # --- integrate mass over region
# mass_hr = np.sum(cg_hr_rho[:, :ycut_hr, :], axis=(1, 2)) * cell_area_hr
# mass_sg = np.sum(rho[:,    :ycut_sg, :], axis=(1, 2)) * cell_area_sg
# mass_lr = np.sum(lr_rho[:, :ycut_lr, :], axis=(1, 2)) * cell_area_lr

# # --- plot vs time
# fig, ax = plt.subplots(figsize=(6, 5))
# timesteps = np.arange(cg_hr_rho.shape[0])  

# ax.plot(timesteps, mass_hr, label="HR", color="red")
# ax.plot(timesteps, mass_sg, label="SG", color="blue")
# ax.plot(timesteps, mass_lr, label="LR", color="green")

# ax.set_xlabel("Timestep")
# ax.set_ylabel("Gas Mass [ρ·pc²]")   
# ax.set_title("Mass in initial cold part")
# ax.legend()

# plt.tight_layout()
# plt.savefig(save_path + "gas_mass_evolution.png", dpi=200)
# plt.close(fig)
# print("Gas mass evolution plot saved")

