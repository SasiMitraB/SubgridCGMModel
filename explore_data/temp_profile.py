import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ergane
import numpy as np
from matplotlib import pyplot as plt

PROJECT_ROOT = '/home/sasi/Projects/SubgridCGMModel'
SIM_ROOT     = os.path.join(PROJECT_ROOT, 'simulation_outputs')
OUT_ROOT     = os.path.join(PROJECT_ROOT, 'outputs')
os.makedirs(OUT_ROOT, exist_ok=True)

# ── Simulation configurations ────────────────────────────────────────────────
simulations = [
    # {
    #     'name':       'hr_mpi_128x256',
    #     'athinp':     os.path.join(SIM_ROOT, 'hr_mpi_128x256', 'kh_radiative_128x256.athinput'),
    #     'datafolder': os.path.join(SIM_ROOT, 'hr_mpi_128x256'),
    #     'save_path':  os.path.join(OUT_ROOT, 'hr_mpi_128x256.mp4'),
    # },
    # {
    #     'name':       'hr_mpi_256x512',
    #     'athinp':     os.path.join(SIM_ROOT, 'hr_mpi_256x512', 'kh_radiative_256x512.athinput'),
    #     'datafolder': os.path.join(SIM_ROOT, 'hr_mpi_256x512'),
    #     'save_path':  os.path.join(OUT_ROOT, 'hr_mpi_256x512.mp4'),
    # },
    {
        'name':       'hr_mpi_512x1024',
        'athinp':     os.path.join(SIM_ROOT, 'hr_mpi_512x1024', 'kh_radiative_512x1024.athinput'),
        'datafolder': os.path.join(SIM_ROOT, 'hr_mpi_512x1024'),
        'save_path':  os.path.join(OUT_ROOT, 'hr_mpi_512x1024.mp4'),
    },
    # {
    #     'name':       'hr_mpi_1024x2048',
    #     'athinp':     os.path.join(SIM_ROOT, 'hr_mpi_1024x2048', 'kh_radiative_1024x2048.athinput'),
    #     'datafolder': os.path.join(SIM_ROOT, 'hr_mpi_1024x2048'),
    #     'save_path':  os.path.join(OUT_ROOT, 'hr_mpi_1024x2048.mp4'),
    # },
]

# ── Render and save each simulation ─────────────────────────────────────────
for sim in simulations:
    print(f"\n[{sim['name']}] Loading simulation data ...")
    sim_data = ergane.SimulationData(
        athinp=sim['athinp'],
        datafolder=sim['datafolder'],
    )

    rho_snapshot_1 = sim_data.density[0]
    pressure_snapshot_1 = sim_data.pressure[0]
    m_p = 1.67e-24
    kb = 1.38e-16
    mu = 0.62
    temp_snapshot_1 = pressure_snapshot_1*mu*m_p/(rho_snapshot_1*kb)
    temp_snapshot_1 = np.log10(temp_snapshot_1)
    
    plt.figure()
    plt.axhline(y=4,linestyle="--")
    for i in range(len(temp_snapshot_1[0, :])):
        y_profile = temp_snapshot_1[:, i]
        plt.plot(y_profile)
    plt.ylabel("Log(T)")
    plt.xlabel("z")
    plt.savefig("y_profile.png")
    plt.close()

    print(y_profile[0])


    plt.figure()
    rho_snapshot_1 = np.log10(rho_snapshot_1)
    for i in range(len(rho_snapshot_1[0, :])):
        y_profile = rho_snapshot_1[:, i]
        plt.plot(y_profile)
    plt.ylabel("Log(rho)")
    plt.xlabel("z")
    plt.savefig("y_profile_rho.png")
    plt.close() 

    plt.figure()
    pressure_snapshot_1 = np.log10(pressure_snapshot_1)
    for i in range(len(pressure_snapshot_1[0,:])):
        y_profile = pressure_snapshot_1[:, i]
        plt.plot(y_profile)
    plt.ylabel("Log(Pressure)")
    plt.xlabel("z")
    plt.savefig("y_profile_pressure.png")
    plt.close() 