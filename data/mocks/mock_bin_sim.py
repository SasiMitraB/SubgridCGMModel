# Initial Test script for reading binary files

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sys
import os
sys.path.append(os.getcwd())
import bin_convert

# Binary files should be 2D. For now, this code works only for uniform grids. (meshblock levels are assumed to be same and 0).

def make_2D_array(file_data, property):
    """ A function to make 2D arrays for each property. 
    The file_data is a dictionary containing the file data. See bin_convert.py for details.
    The property is the name of the property to be plotted.
    The function returns a 2D array for the property."""

    # Get the number of mesh blocks and their dimensions. This function only works for 2D bin files.
    nmb = file_data['n_mbs']    # Number of mesh blocks
    nx1 = file_data['nx1_mb']   # Number of cells in the x1 direction (horizontal) per meshblock
    nx2 = file_data['nx2_mb']   # Number of cells in the x2 direction (vertical). nx3 is 1 for 2D files.

    # Initialize an empty array to hold the 2D data
    Arr = np.zeros((file_data['Nx2'], file_data['Nx1']))    # Nx2, Nx1 are total number of cells.
    property_arr = file_data['mb_data'][property]   # 4D array containing the property data for each mesh block

    for mb in range(nmb):
        mb_logical_indices = file_data['mb_logical'][mb]    # Logical indices of the mesh block in the 2D grid 
        I = mb_logical_indices[0]*nx1                       # Logical index in the x1 direction (horizontal)
        J = mb_logical_indices[1]*nx2                       # Logical index in the x2 direction (vertical)
        for i in range(nx1):
            for j in range(nx2):
                # Fill the 2D array with the property data
                Arr[J + j][I + i] = property_arr[mb][0][j][i]
    return Arr

def plot_figure(file_data, property):
    """ A function to plot the a property using matplotlib.
     Takes file_data dictionary and property and plots a figure."""

    Arr = make_2D_array(file_data, property)      # Make 2D array for the property

    plt.figure(figsize=(10, 6))
    plt.imshow(Arr, cmap='viridis', aspect='auto')
    plt.colorbar(label=property)
    plt.gca().invert_yaxis()                      # Invert the y-axis so that (0,0) is at the bottom left corner in the figure.
    plt.clim(vmin=np.min(Arr), vmax=np.max(Arr))  # Set the color limits to the min and max of the data
    plt.title(f'2D Plot of {property} at Time = {file_data["time"]}')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.show()

if __name__ == "__main__" :

    # file_path = "mocks/KH.hydro_w.00000.bin"
    file_path = "mocks/KH.hydro_w.00150.bin"
    # 'dens', 'Temperature', 'velx', 'vely', 'eint', 's_00'
    file_data = bin_convert.read_binary(file_path)
    data_dict = file_data['mb_data']

    high_res_rho = make_2D_array(file_data, "dens")
    # high_res_temp = make_2D_array(file_data, "Temperature")

    plt.imshow(high_res_rho, cmap='inferno', norm=LogNorm(), origin='lower')
    plt.colorbar(label=r"Density ($cm^{-3}$)")
    plt.title(r"High Resolution (512 $\times$ 512) Density")
    plt.savefig("mocks/bin_sim_density.jpg", dpi=500)
    plt.clf()
