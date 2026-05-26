# Python script for preprocessing the simulation data, calculating source terms, fluxes and PDFs

import numpy as np 
import os
import skimage.measure
import matplotlib.pyplot as plt
import bin_convert
from tqdm import tqdm

def divergence(f, dx, dy):
    dFx_dx = np.gradient(f[0], dy, dx)[1]
    dFy_dy = np.gradient(f[1], dy, dx)[0]
    return dFx_dx + dFy_dy

import numpy as np

# Cooling function from AthenaK code
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

class simulation_data():

    T_cutoff: float = 1e5 # K
    down_sample: int = 32
    total_time: float = 5.0 # Myr
    delta_time: float = 0.01 # Myr
    total_length: float = 40 # pc
    total_width: float = 20 # pc
    resolution: tuple = (512, 256) 
    gamma: float = 5./3.

    def __init__(self: "simulation_data") -> None:

        # time step X length X width arrays
        self.rho: np.ndarray = None
        self.temp: np.ndarray = None
        self.pressure: np.ndarray = None
        self.ux: np.ndarray = None
        self.uy: np.ndarray = None
        self.eint: np.ndarray = None
        self.ps: np.ndarray = None

        self.cons_rho: np.ndarray = None
        self.cons_momx: np.ndarray = None
        self.cons_momy: np.ndarray = None
        self.cons_ener: np.ndarray = None
        self.cons_ps: np.ndarray = None

    def input_data(self: "simulation_data", filepath: str, start:int = 0) -> None:

        cwd = os.getcwd()
        os.chdir(filepath)
        num_snaps = len([f for f in os.listdir(filepath) if f.endswith('.bin')])//2

        self.rho = np.zeros((num_snaps, self.resolution[0], self.resolution[1]))
        self.temp = np.zeros_like(self.rho)
        self.pressure = np.zeros_like(self.rho)
        self.ux = np.zeros_like(self.rho)
        self.uy = np.zeros_like(self.rho)
        self.eint = np.zeros_like(self.rho)
        self.ps = np.zeros_like(self.rho)
        self.frho = np.zeros_like(self.rho)

        for i in tqdm(range(num_snaps), desc="Loading data"):
            if start == 0:
                file_data = bin_convert.read_binary(f"KH.hydro_w.{i:05d}.bin")
            else:
                file_data = bin_convert.read_binary(f"KH.hydro_w.{i+start:05d}.bin")
            self.rho[i] = bin_convert.make_2D_array(file_data, "dens")
            self.ux[i] = bin_convert.make_2D_array(file_data, "velx")
            self.uy[i] = bin_convert.make_2D_array(file_data, "vely")
            self.eint[i] = bin_convert.make_2D_array(file_data, "eint")
            self.ps[i] = bin_convert.make_2D_array(file_data, "s_00")
            try:
                self.frho[i] = bin_convert.make_2D_array(file_data, "s_01")
            except Exception:
                pass

            self.pressure[i] = 2./3. * self.eint[i]
            self.temp[i] = (self.pressure[i] * 1.59916e-14 / self.rho[i]) * (1. / 1.381e-16)

        os.chdir(cwd)
        return
    
    def input_cons_data(self: "simulation_data", filepath: str, start:int = 0) -> None:

        cwd = os.getcwd()
        os.chdir(filepath)
        num_snaps = len([f for f in os.listdir(filepath) if f.endswith('.bin')])//2

        self.cons_rho = np.zeros((num_snaps, self.resolution[0], self.resolution[1]))
        self.cons_momx = np.zeros_like(self.cons_rho)
        self.cons_momy = np.zeros_like(self.cons_rho)
        self.cons_ener = np.zeros_like(self.cons_rho)
        self.cons_ps = np.zeros_like(self.cons_rho)

        for i in tqdm(range(num_snaps), desc="Loading cons data"):
            if start == 0:
                file_data = bin_convert.read_binary(f"KH.hydro_u.{i:05d}.bin")
            else:
                file_data = bin_convert.read_binary(f"KH.hydro_u.{i+start:05d}.bin")
            self.cons_rho[i] = bin_convert.make_2D_array(file_data, "dens")
            self.cons_momx[i] = bin_convert.make_2D_array(file_data, "mom1")
            self.cons_momy[i] = bin_convert.make_2D_array(file_data, "mom2")
            self.cons_ener[i] = bin_convert.make_2D_array(file_data, "ener")
            self.cons_ps[i] = bin_convert.make_2D_array(file_data, "r_00")

        os.chdir(cwd)
        return

    def filter_timesteps(self: "simulation_data", filter: int = 1) -> None:
        self.delta_time = (self.total_time/(self.rho.shape[0] - 1))*filter
        self.rho = self.rho[::filter]
        self.temp = self.temp[::filter]
        self.pressure = self.pressure[::filter]
        self.ux = self.ux[::filter]
        self.uy = self.uy[::filter]
        self.eint = self.eint[::filter]
        self.ps = self.ps[::filter]

    def coarse_grain(self: "simulation_data", quan: np.ndarray) -> np.ndarray:
        return skimage.measure.block_reduce(quan, (self.down_sample, self.down_sample), np.mean)
    
    def calc_fmcl(self: "simulation_data", rho: np.ndarray, temp: np.ndarray) -> np.ndarray:
        rho_block = rho.reshape(rho.shape[0] // self.down_sample, self.down_sample, rho.shape[1] // self.down_sample, self.down_sample)
        temp_block = temp.reshape(temp.shape[0] // self.down_sample, self.down_sample, temp.shape[1] // self.down_sample, self.down_sample)
        fc = np.sum(rho_block * (temp_block < self.T_cutoff), axis=(1, 3))
        fh = np.sum(rho_block * (temp_block > self.T_cutoff), axis=(1, 3))
        fmcl = fc / (fc + fh)
        return fmcl
    
    def fmcl_hist(self: "simulation_data") -> None:
        fmcl_data = np.zeros((self.rho.shape[0], self.rho.shape[1] // self.down_sample, self.rho.shape[2] // self.down_sample))
        for i in range(self.rho.shape[0]):
            fmcl_data[i] = self.calc_fmcl(self.rho[i], self.temp[i])
        hist_data = fmcl_data.flatten()
        return hist_data
    
    # Calculate individual source terms
    def calc_source_term(self: "simulation_data") -> np.ndarray:

        fmcl = np.zeros((self.rho.shape[0], self.rho.shape[1] // self.down_sample, self.rho.shape[2] // self.down_sample))
        cg_rho = np.zeros_like(fmcl)
        cg_ux = np.zeros_like(fmcl)
        cg_uy = np.zeros_like(fmcl)
        source_term = np.zeros_like(fmcl)

        for i in tqdm(range(self.rho.shape[0]), desc="Calculating source term"):

            fmcl[i] = self.calc_fmcl(self.rho[i], self.temp[i])
            cg_rho[i] = self.coarse_grain(self.rho[i])
            cg_ux[i] = self.coarse_grain(self.ux[i])
            cg_uy[i] = self.coarse_grain(self.uy[i])

            term1 = np.zeros((cg_rho[i].shape[0], cg_rho[i].shape[1]))
            term2 = np.zeros_like(term1)

            if i == 0:
                fmcl[i+1] = self.calc_fmcl(self.rho[i+1], self.temp[i+1])
                cg_rho[i+1] = self.coarse_grain(self.rho[i+1])
                term1 = (fmcl[i+1] * cg_rho[i+1] - fmcl[i] * cg_rho[i]) / self.delta_time
            elif i == self.rho.shape[0] - 1:
                term1 = (fmcl[i] * cg_rho[i] - fmcl[i-1] * cg_rho[i-1]) / self.delta_time
            else:
                fmcl[i+1] = self.calc_fmcl(self.rho[i+1], self.temp[i+1])
                cg_rho[i+1] = self.coarse_grain(self.rho[i+1])
                term1 = (fmcl[i+1] * cg_rho[i+1] - fmcl[i-1] * cg_rho[i-1]) / (2 * self.delta_time)

            dx = self.total_length / cg_rho.shape[1]
            dy = self.total_width / cg_rho.shape[2]
            term2 = cg_ux[i] * np.gradient(fmcl[i] * cg_rho[i], dy, dx)[1] + cg_uy[i] * np.gradient(fmcl[i] * cg_rho[i], dy, dx)[0] \
                + cg_rho[i] * fmcl[i] * (np.gradient(cg_ux[i], dy, dx)[1] + np.gradient(cg_uy[i], dy, dx)[0])
            
            source_term[i] = term1 + term2

        return source_term
    
    # Calculate and store all source terms in a single array
    def calc_all_source_terms(self: "simulation_data") -> np.ndarray:

        fmcl = np.zeros((self.rho.shape[0], self.rho.shape[1] // self.down_sample, self.rho.shape[2] // self.down_sample))
        cg_rho = np.zeros_like(fmcl)
        cg_temp = np.zeros_like(fmcl)
        cg_ux = np.zeros_like(fmcl)
        cg_uy = np.zeros_like(fmcl)
        cg_pressure = np.zeros_like(fmcl)

        cg_momx = np.zeros_like(fmcl)
        cg_momy = np.zeros_like(fmcl)
        cg_ener = np.zeros_like(fmcl)

        source_term = np.zeros((self.rho.shape[0], 5, self.rho.shape[1] // self.down_sample, self.rho.shape[2] // self.down_sample))
        div_term = np.zeros_like(source_term)

        dy = self.total_length / cg_rho.shape[1]
        dx = self.total_width / cg_rho.shape[2]

        dY = self.total_length / self.rho.shape[1]
        dX = self.total_width / self.rho.shape[2]

        for i in tqdm(range(self.rho.shape[0]), desc="Calculating source term"):

            fmcl[i] = self.calc_fmcl(self.rho[i], self.temp[i])
            cg_rho[i] = self.coarse_grain(self.rho[i])
            cg_temp[i] = self.coarse_grain(self.temp[i])
            cg_ux[i] = self.coarse_grain(self.ux[i])
            cg_uy[i] = self.coarse_grain(self.uy[i])
            cg_pressure[i] = self.coarse_grain(self.pressure[i])

            cg_momx[i] = self.coarse_grain(self.cons_momx[i])
            cg_momy[i] = self.coarse_grain(self.cons_momy[i])
            cg_ener[i] = self.coarse_grain(self.cons_ener[i])

            term = np.zeros((5, cg_rho[i].shape[0], cg_rho[i].shape[1]))

            if i == 0:

                # rho source term
                cg_rho[i+1] = self.coarse_grain(self.rho[i+1])
                term[0] = (cg_rho[i+1] - cg_rho[i]) / self.delta_time

                # momentum source terms
                # cg_ux[i+1] = self.coarse_grain(self.ux[i+1])
                # cg_uy[i+1] = self.coarse_grain(self.uy[i+1])
                # term[1] = (cg_ux[i+1] * cg_rho[i+1] - cg_ux[i] * cg_rho[i]) / self.delta_time
                # term[2] = (cg_uy[i+1] * cg_rho[i+1] - cg_uy[i] * cg_rho[i]) / self.delta_time

                cg_momx[i+1] = self.coarse_grain(self.cons_momx[i+1])
                cg_momy[i+1] = self.coarse_grain(self.cons_momy[i+1])
                term[1] = (cg_momx[i+1] - cg_momx[i]) / self.delta_time
                term[2] = (cg_momy[i+1] - cg_momy[i]) / self.delta_time

                # energy source term
                # cg_pressure[i+1] = self.coarse_grain(self.pressure[i+1])
                # e1 = cg_pressure[i+1] / (self.gamma - 1) + cg_rho[i+1] * (cg_ux[i+1]**2 + cg_uy[i+1]**2) / 2
                # e0 = cg_pressure[i] / (self.gamma - 1) + cg_rho[i] * (cg_ux[i]**2 + cg_uy[i]**2) / 2
                # term[3] = (e1 - e0) / self.delta_time

                cg_ener[i+1] = self.coarse_grain(self.cons_ener[i+1])
                term[3] = (cg_ener[i+1] - cg_ener[i]) / self.delta_time

                # fmcl source term
                fmcl[i+1] = self.calc_fmcl(self.rho[i+1], self.temp[i+1])
                term[4] = (fmcl[i+1] * cg_rho[i+1] - fmcl[i] * cg_rho[i]) / self.delta_time

            elif i == self.rho.shape[0] - 1:

                # rho source term
                term[0] = (cg_rho[i] - cg_rho[i-1]) / self.delta_time

                # momentum source terms
                # term[1] = (cg_ux[i] * cg_rho[i] - cg_ux[i-1] * cg_rho[i-1]) / self.delta_time
                # term[2] = (cg_uy[i] * cg_rho[i] - cg_uy[i-1] * cg_rho[i-1]) / self.delta_time

                term[1] = (cg_momx[i] - cg_momx[i-1]) / self.delta_time
                term[2] = (cg_momy[i] - cg_momy[i-1]) / self.delta_time

                # energy source term
                # e1 = cg_pressure[i] / (self.gamma - 1) + cg_rho[i] * (cg_ux[i]**2 + cg_uy[i]**2) / 2
                # e0 = cg_pressure[i-1] / (self.gamma - 1) + cg_rho[i-1] * (cg_ux[i-1]**2 + cg_uy[i-1]**2) / 2
                # term[3] = (e1 - e0) / self.delta_time

                term[3] = (cg_ener[i] - cg_ener[i-1]) / self.delta_time

                # fmcl source term
                term[4] = (fmcl[i] * cg_rho[i] - fmcl[i-1] * cg_rho[i-1]) / self.delta_time

            else:

                # rho source term
                cg_rho[i+1] = self.coarse_grain(self.rho[i+1])
                term[0] = (cg_rho[i+1] - cg_rho[i-1]) / (2 * self.delta_time)

                # momentum source terms
                # cg_ux[i+1] = self.coarse_grain(self.ux[i+1])
                # cg_uy[i+1] = self.coarse_grain(self.uy[i+1])
                # term[1] = (cg_ux[i+1] * cg_rho[i+1] - cg_ux[i-1] * cg_rho[i-1]) / (2 * self.delta_time)
                # term[2] = (cg_uy[i+1] * cg_rho[i+1] - cg_uy[i-1] * cg_rho[i-1]) / (2 * self.delta_time)

                cg_momx[i+1] = self.coarse_grain(self.cons_momx[i+1])
                cg_momy[i+1] = self.coarse_grain(self.cons_momy[i+1])
                term[1] = (cg_momx[i+1] - cg_momx[i-1]) / (2 * self.delta_time)
                term[2] = (cg_momy[i+1] - cg_momy[i-1]) / (2 * self.delta_time)

                # energy source term
                # cg_pressure[i+1] = self.coarse_grain(self.pressure[i+1])
                # e2 = cg_pressure[i+1] / (self.gamma - 1) + cg_rho[i+1] * (cg_ux[i+1]**2 + cg_uy[i+1]**2) / 2
                # e0 = cg_pressure[i-1] / (self.gamma - 1) + cg_rho[i-1] * (cg_ux[i-1]**2 + cg_uy[i-1]**2) / 2
                # term[3] = (e2 - e0) / (2 * self.delta_time)

                cg_ener[i+1] = self.coarse_grain(self.cons_ener[i+1])
                term[3] = (cg_ener[i+1] - cg_ener[i-1]) / (2 * self.delta_time)

                # fmcl source term
                fmcl[i+1] = self.calc_fmcl(self.rho[i+1], self.temp[i+1])
                term[4] = (fmcl[i+1] * cg_rho[i+1] - fmcl[i-1] * cg_rho[i-1]) / (2 * self.delta_time)

            source_term[i] = term

            cg_uv = np.array([cg_ux[i], cg_uy[i]])

            # rho flux
            # div_term[i][0] = divergence(cg_rho[i]*cg_uv, dx, dy)
            div_term[i][0] = divergence(np.array([self.coarse_grain(self.rho[i]*self.ux[i]), self.coarse_grain(self.rho[i]*self.uy[i])]), dx, dy)

            # momentum flux
            # div_term[i][1] = np.gradient(cg_rho[i]*cg_ux[i]**2, dy, dx)[1] + np.gradient(cg_rho[i]*cg_ux[i]*cg_uy[i], dy, dx)[1] \
            #             + np.gradient(cg_pressure[i], dy, dx)[1]
            div_term[i][1] = np.gradient(self.coarse_grain(self.rho[i]*self.ux[i]**2), dy, dx)[1] + np.gradient(self.coarse_grain(self.rho[i]*self.ux[i]*self.uy[i]), dy, dx)[1] \
                        + np.gradient(cg_pressure[i], dy, dx)[1]
            # div_term[i][2] = np.gradient(cg_rho[i]*cg_ux[i]*cg_uy[i], dy, dx)[0] + np.gradient(cg_rho[i]*cg_uy[i]**2, dy, dx)[0] \
            #             + np.gradient(cg_pressure[i], dy, dx)[0]
            div_term[i][2] = np.gradient(self.coarse_grain(self.rho[i]*self.ux[i]*self.uy[i]), dy, dx)[0] + np.gradient(self.coarse_grain(self.rho[i]*self.uy[i]**2), dy, dx)[0] \
                        + np.gradient(cg_pressure[i], dy, dx)[0]

            # energy flux
            # div_term[i][3] = divergence((self.gamma*cg_pressure[i]/(self.gamma-1) + cg_rho[i]*(cg_ux[i]**2 + cg_uy[i]**2)/2)*cg_uv, dx, dy)
            div_term[i][3] = divergence(np.array([self.coarse_grain(self.ux[i]*(self.gamma*self.pressure[i]/(self.gamma-1) + self.rho[i]*(self.ux[i]**2 + self.uy[i]**2)/2)), \
                                                self.coarse_grain(self.uy[i]*(self.gamma*self.pressure[i]/(self.gamma-1) + self.rho[i]*(self.ux[i]**2 + self.uy[i]**2)/2))]), dx, dy)

            # cooling flux
            temp = (cg_pressure[i]/cg_rho[i]) * 1.59916e-14 / 1.381e-16 # mass weighted temperature
            # temp = cg_temp[i] # volume weighted temperature
            rho_cgs = cg_rho[i] 
            mu = 0.62   
            n = rho_cgs / mu
            cool_rate_cgs = lambda_cool(temp) * n**2    # erg/cm^3/s
            cool_rate_code = cool_rate_cgs * 1.975e27
            # div_term[i][3] += cool_rate_code

            # fmcl flux
            div_term[i][4] = divergence(fmcl[i]*cg_rho[i]*cg_uv, dx, dy)

        return (source_term + div_term)

    # Calculate the subgrid fluxes instead of the source terms
    def calc_subgrid_flux(self: "simulation_data") -> np.ndarray:

        fmcl = np.zeros((self.rho.shape[0], self.rho.shape[1] // self.down_sample, self.rho.shape[2] // self.down_sample))
        cg_rho = np.zeros_like(fmcl)
        cg_temp = np.zeros_like(fmcl)
        cg_ux = np.zeros_like(fmcl)
        cg_uy = np.zeros_like(fmcl)
        cg_pressure = np.zeros_like(fmcl)

        cg_momx = np.zeros_like(fmcl)
        cg_momy = np.zeros_like(fmcl)
        cg_ener = np.zeros_like(fmcl)

        subgrid_flux = np.zeros((self.rho.shape[0], 12, self.rho.shape[1] // self.down_sample, self.rho.shape[2] // self.down_sample))

        dy = self.total_length / cg_rho.shape[1]
        dx = self.total_width / cg_rho.shape[2]

        dY = self.total_length / self.rho.shape[1]
        dX = self.total_width / self.rho.shape[2]

        for i in tqdm(range(self.rho.shape[0]), desc="Calculating subgrid fluxes"):

            fmcl[i] = self.calc_fmcl(self.rho[i], self.temp[i])
            cg_rho[i] = self.coarse_grain(self.rho[i])
            cg_temp[i] = self.coarse_grain(self.temp[i])
            cg_ux[i] = self.coarse_grain(self.ux[i])
            cg_uy[i] = self.coarse_grain(self.uy[i])
            cg_pressure[i] = self.coarse_grain(self.pressure[i])

            cg_momx[i] = self.coarse_grain(self.cons_momx[i])
            cg_momy[i] = self.coarse_grain(self.cons_momy[i])
            cg_ener[i] = self.coarse_grain(self.cons_ener[i])

            cg_uv = np.array([cg_ux[i], cg_uy[i]])

            # rho fluxes
            subgrid_flux[i][0] = self.coarse_grain(self.rho[i]*self.ux[i]) - cg_rho[i]*cg_ux[i]
            subgrid_flux[i][1] = self.coarse_grain(self.rho[i]*self.uy[i]) - cg_rho[i]*cg_uy[i]

            # momentum fluxes
            subgrid_flux[i][2] = self.coarse_grain(self.rho[i]*self.ux[i]**2 + self.pressure[i]) - (cg_rho[i]*cg_ux[i]**2 + cg_pressure[i])
            subgrid_flux[i][3] = self.coarse_grain(self.rho[i]*self.ux[i]*self.uy[i]) - (cg_rho[i]*cg_ux[i]*cg_uy[i])
            subgrid_flux[i][4] = self.coarse_grain(self.rho[i]*self.uy[i]**2 + self.pressure[i]) - (cg_rho[i]*cg_uy[i]**2 + cg_pressure[i])

            # energy fluxes
            # subgrid_flux[i][5] = self.coarse_grain(self.ux[i]*(self.gamma*self.pressure[i]/(self.gamma-1) + self.rho[i]*(self.ux[i]**2 + self.uy[i]**2)/2)) - cg_ux[i]*(self.gamma*cg_pressure[i]/(self.gamma-1) + cg_rho[i]*(cg_ux[i]**2 + cg_uy[i]**2)/2)
            # subgrid_flux[i][5] = self.coarse_grain(self.ux[i]*(self.cons_ener[i] + self.pressure[i])) - cg_ux[i]*(cg_ener[i] + cg_pressure[i])
            subgrid_flux[i][5] = self.coarse_grain(self.ux[i]*(self.gamma*self.pressure[i]/(self.gamma-1))) - cg_ux[i]*(self.gamma*cg_pressure[i]/(self.gamma-1))
            subgrid_flux[i][6] = self.coarse_grain(self.ux[i]*(self.rho[i]*(self.ux[i]**2 + self.uy[i]**2)/2)) - cg_ux[i]*(cg_rho[i]*(cg_ux[i]**2 + cg_uy[i]**2)/2)

            # subgrid_flux[i][6] = self.coarse_grain(self.uy[i]*(self.gamma*self.pressure[i]/(self.gamma-1) + self.rho[i]*(self.ux[i]**2 + self.uy[i]**2)/2)) - cg_uy[i]*(self.gamma*cg_pressure[i]/(self.gamma-1) + cg_rho[i]*(cg_ux[i]**2 + cg_uy[i]**2)/2)
            # subgrid_flux[i][6] = self.coarse_grain(self.uy[i]*(self.cons_ener[i] + self.pressure[i])) - cg_uy[i]*(cg_ener[i] + cg_pressure[i])
            subgrid_flux[i][7] = self.coarse_grain(self.uy[i]*(self.gamma*self.pressure[i]/(self.gamma-1))) - cg_uy[i]*(self.gamma*cg_pressure[i]/(self.gamma-1))
            subgrid_flux[i][8] = self.coarse_grain(self.uy[i]*(self.rho[i]*(self.ux[i]**2 + self.uy[i]**2)/2)) - cg_uy[i]*(cg_rho[i]*(cg_ux[i]**2 + cg_uy[i]**2)/2)

            # cooling fluxes

            # mu = 0.62
            # n = self.rho[i] / mu
            # cool_rate = lambda_cool(self.temp[i]) * n**2 * 1.975e27    # erg/cm^3/s to code units
            # subgrid_flux[i][7] = - skimage.measure.block_reduce(cool_rate, (self.down_sample, self.down_sample), np.sum)

            if i == 0:
                deriv_term = (self.cons_ener[i + 1] - self.cons_ener[i]) / self.delta_time
            elif i == self.rho.shape[0] - 1:
                deriv_term = (self.cons_ener[i] - self.cons_ener[i-1]) / self.delta_time
            else:
                deriv_term = (self.cons_ener[i+1] - self.cons_ener[i-1]) / (2 * self.delta_time)
            div_term = divergence(np.array([self.ux[i]*(self.gamma*self.pressure[i]/(self.gamma-1) + self.rho[i]*(self.ux[i]**2 + self.uy[i]**2)/2), \
                                            self.uy[i]*(self.gamma*self.pressure[i]/(self.gamma-1) + self.rho[i]*(self.ux[i]**2 + self.uy[i]**2)/2)]), dX, dY)
            subgrid_flux[i][9] = self.coarse_grain(deriv_term + div_term)
            
            # fmcl fluxes
            mask = np.where(self.temp[i] < self.T_cutoff, 1, 0)
            subgrid_flux[i][10] = self.coarse_grain(mask*self.rho[i]*self.ux[i]) - fmcl[i]*cg_rho[i]*cg_ux[i]
            subgrid_flux[i][11] = self.coarse_grain(mask*self.rho[i]*self.uy[i]) - fmcl[i]*cg_rho[i]*cg_uy[i] 

        return subgrid_flux
    
    # Calculate the PDF of each coarse-grained pixel
    def calc_pixel_pdf(self: "simulation_data", bins: int = 200) -> np.ndarray:

        ds = self.down_sample
        nt, nx, ny = self.temp.shape

        pixel_pdf = np.zeros((nt, nx//ds, ny//ds, bins))

        temp_bins = np.logspace(3.0, 7.0, bins+1)

        for i in tqdm(range(nt), desc="Calculating pixel PDFs"):
            for j in range(nx//ds):
                for k in range(ny//ds):

                    block = self.temp[
                        i,
                        j*ds:(j+1)*ds,
                        k*ds:(k+1)*ds
                    ]

                    hist, _ = np.histogram(block.ravel(), bins=temp_bins)

                    if hist.sum() > 0:
                        pixel_pdf[i,j,k] = hist / hist.sum()

        pixel_pdf = np.transpose(pixel_pdf, (0, 3, 1, 2))
        return pixel_pdf
                    
    

