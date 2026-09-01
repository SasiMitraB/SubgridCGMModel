#!/usr/bin/env python3
"""
downsample_ic.py: Downsamples an AthenaK simulation snapshot to a coarse resolution
and exports a raw binary file for AthenaK pgen initial condition loading.

Output binary format:
A contiguous 1D/3D float64 array of shape (7, Nx2, Nx1):
  Channel 0: Density (dens)
  Channel 1: Internal energy density (eint = P / (gamma - 1))
  Channel 2: Velocity X (velx)
  Channel 3: Velocity Y (vely)
  Channel 4: Velocity Z (velz)
  Channel 5: Passive scalar 1 (s_00 / tracer)
  Channel 6: Passive scalar 2 (s_01 / cold gas mass fraction)
"""

import os
import sys
import argparse
import numpy as np

# Add project root to sys.path so ergane can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ergane.bin_reader import read_binary, make_2D_array


def block_average_2d(arr_2d, target_ny, target_nx):
    """
    Downsamples a 2D array of shape (Ny, Nx) to (target_ny, target_nx)
    using exact cell-averaging (block mean).
    """
    ny, nx = arr_2d.shape
    if ny % target_ny == 0 and nx % target_nx == 0:
        by = ny // target_ny
        bx = nx // target_nx
        return arr_2d.reshape(target_ny, by, target_nx, bx).mean(axis=(1, 3))
    else:
        # Fallback to scipy zoom if not cleanly divisible
        from scipy.ndimage import zoom
        zoom_y = target_ny / ny
        zoom_x = target_nx / nx
        return zoom(arr_2d, (zoom_y, zoom_x), order=1)


def downsample_snapshot(
    input_bin: str,
    target_nx1: int,
    target_nx2: int,
    output_bin: str,
    gamma: float = 1.666667,
    cold_rho_thresh: float = None,
):
    print(f"Loading input binary snapshot: {input_bin}")
    filedata = read_binary(input_bin)

    var_names = filedata["var_names"]
    Nx1_in = filedata["Nx1"]
    Nx2_in = filedata["Nx2"]
    time = filedata.get("time", 0.0)
    cycle = filedata.get("cycle", 0)
    print(f"  Snapshot info: time={time}, cycle={cycle}, grid=({Nx2_in} x {Nx1_in}), vars={var_names}")
    print(f"  Target coarse grid: Nx1={target_nx1}, Nx2={target_nx2}")

    # Check if input is primitive (hydro_w) or conserved (hydro_u)
    is_primitive = "velx" in var_names and "eint" in var_names

    if is_primitive:
        dens_hr = make_2D_array(filedata, "dens")
        eint_hr = make_2D_array(filedata, "eint")
        velx_hr = make_2D_array(filedata, "velx")
        vely_hr = make_2D_array(filedata, "vely")
        velz_hr = make_2D_array(filedata, "velz") if "velz" in var_names else np.zeros_like(dens_hr)
        s0_hr = make_2D_array(filedata, "s_00") if "s_00" in var_names else np.zeros_like(dens_hr)
        has_s1 = "s_01" in var_names
        s1_hr = make_2D_array(filedata, "s_01") if has_s1 else None
    else:
        # Conserved variables (hydro_u)
        dens_hr = make_2D_array(filedata, "dens")
        mom1_hr = make_2D_array(filedata, "mom1")
        mom2_hr = make_2D_array(filedata, "mom2")
        mom3_hr = make_2D_array(filedata, "mom3") if "mom3" in var_names else np.zeros_like(dens_hr)
        ener_hr = make_2D_array(filedata, "ener")

        velx_hr = mom1_hr / np.maximum(dens_hr, 1e-30)
        vely_hr = mom2_hr / np.maximum(dens_hr, 1e-30)
        velz_hr = mom3_hr / np.maximum(dens_hr, 1e-30)
        ke_hr = 0.5 * dens_hr * (velx_hr**2 + vely_hr**2 + velz_hr**2)
        eint_hr = np.maximum(ener_hr - ke_hr, 1e-20)

        # Passive scalars in hydro_u are mass densities (rho * s)
        has_s0 = "r_00" in var_names
        s0_hr = (make_2D_array(filedata, "r_00") / np.maximum(dens_hr, 1e-30)) if has_s0 else np.zeros_like(dens_hr)
        has_s1 = "r_01" in var_names
        s1_hr = (make_2D_array(filedata, "r_01") / np.maximum(dens_hr, 1e-30)) if has_s1 else None

    # Handle cold gas fraction (s1_hr) if not directly in snapshot
    if s1_hr is None:
        if cold_rho_thresh is None:
            # Automatic threshold midway between min and max density (log scale)
            rho_min, rho_max = float(dens_hr.min()), float(dens_hr.max())
            cold_rho_thresh = np.sqrt(rho_min * rho_max)
        print(f"  Synthesizing cold gas mass fraction with threshold rho > {cold_rho_thresh:.5e}")
        # Cold indicator (1 where dense/cold, 0 where hot)
        is_cold = (dens_hr >= cold_rho_thresh).astype(np.float64)
        s1_hr = is_cold

    # Cell-averaging (coarse-graining):
    # 1. Volume average of density
    dens_lr = block_average_2d(dens_hr, target_nx2, target_nx1)

    # 2. Volume average of internal energy density
    eint_lr = block_average_2d(eint_hr, target_nx2, target_nx1)

    # 3. Mass-weighted average of velocities (exact momentum conservation)
    momx_lr = block_average_2d(dens_hr * velx_hr, target_nx2, target_nx1)
    momy_lr = block_average_2d(dens_hr * vely_hr, target_nx2, target_nx1)
    momz_lr = block_average_2d(dens_hr * velz_hr, target_nx2, target_nx1)
    velx_lr = momx_lr / np.maximum(dens_lr, 1e-30)
    vely_lr = momy_lr / np.maximum(dens_lr, 1e-30)
    velz_lr = momz_lr / np.maximum(dens_lr, 1e-30)

    # 4. Mass-weighted average of passive scalars
    s0_mass_lr = block_average_2d(dens_hr * s0_hr, target_nx2, target_nx1)
    s0_lr = s0_mass_lr / np.maximum(dens_lr, 1e-30)

    s1_mass_lr = block_average_2d(dens_hr * s1_hr, target_nx2, target_nx1)
    s1_lr = s1_mass_lr / np.maximum(dens_lr, 1e-30)

    # Stack channels: [dens, eint, velx, vely, velz, s0, s1]
    # Shape: (7, target_nx2, target_nx1)
    payload = np.stack([dens_lr, eint_lr, velx_lr, vely_lr, velz_lr, s0_lr, s1_lr], axis=0).astype(np.float64)

    os.makedirs(os.path.dirname(os.path.abspath(output_bin)), exist_ok=True)
    payload.tofile(output_bin)

    print(f"Successfully saved downsampled initial condition:")
    print(f"  File: {output_bin}")
    print(f"  Shape: {payload.shape} (Nvars=7, Nx2={target_nx2}, Nx1={target_nx1})")
    print(f"  Size: {os.path.getsize(output_bin)} bytes")
    print(f"  Density range: [{dens_lr.min():.4e}, {dens_lr.max():.4e}]")
    print(f"  Internal energy range: [{eint_lr.min():.4e}, {eint_lr.max():.4e}]")
    print(f"  Velx range: [{velx_lr.min():.4e}, {velx_lr.max():.4e}]")
    print(f"  Vely range: [{vely_lr.min():.4e}, {vely_lr.max():.4e}]")
    print(f"  Cold fraction range: [{s1_lr.min():.4f}, {s1_lr.max():.4f}]")


def main():
    parser = argparse.ArgumentParser(description="Downsample AthenaK snapshot to coarse initial condition file")
    parser.add_argument("--input", "-i", required=True, help="Input AthenaK .bin snapshot path")
    parser.add_argument("--nx1", type=int, required=True, help="Target coarse grid Nx1 (horizontal/width)")
    parser.add_argument("--nx2", type=int, required=True, help="Target coarse grid Nx2 (vertical/height)")
    parser.add_argument("--output", "-o", required=True, help="Output binary file path (e.g., ic_16x8.bin)")
    parser.add_argument("--gamma", type=float, default=1.666667, help="Adiabatic index gamma (default: 5/3)")
    parser.add_argument("--cold_rho_thresh", type=float, default=None, help="Density threshold for cold phase")

    args = parser.parse_args()
    downsample_snapshot(
        input_bin=args.input,
        target_nx1=args.nx1,
        target_nx2=args.nx2,
        output_bin=args.output,
        gamma=args.gamma,
        cold_rho_thresh=args.cold_rho_thresh,
    )


if __name__ == "__main__":
    main()
