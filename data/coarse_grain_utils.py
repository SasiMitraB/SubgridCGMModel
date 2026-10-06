"""
Shared coarse-graining utilities for consistent treatment across preprocessing and simulations.

All coarse-graining follows these principles:
- Density, pressure, internal energy: volume averages (simple block mean)
- Velocity, passive scalars: mass-weighted averages (conserve momentum and mass)
- Temperature: derived from coarse fields via equation of state (NOT averaged)
- Subgrid KE: computed as diagnostic for turbulent closure targets
"""

import os

import numpy as np


def block_mean(f, b):
    """
    Spatial block average of a 2D field.

    Args:
        f: 2D array of shape (ny, nx)
        b: Block size (downsampling factor)

    Returns:
        Coarse-grained array of shape (ny//b, nx//b)
    """
    ny, nx = f.shape
    if ny % b != 0 or nx % b != 0:
        raise ValueError(f"Grid ({ny}, {nx}) not divisible by block size {b}")
    return f.reshape(ny // b, b, nx // b, b).mean(axis=(1, 3))


def coarse_grain(rho, ux, uy, P, scalars=None, b=32,
                 P_unit=1.59916e-14, mu=0.62, k_b=1.3807e-16):
    """
    Coarse-grain primitive variables consistently.

    Handles density, momentum conservation via mass-weighted velocity averaging,
    pressure, mass-weighted scalar averaging, and derives temperature from the
    equation of state to avoid bias toward the hot phase.

    Args:
        rho: Density field (ny, nx)
        ux: Velocity X field (ny, nx)
        uy: Velocity Y field (ny, nx)
        P: Pressure field (ny, nx)
        scalars: List of passive scalar fields (mass fractions), each (ny, nx)
        b: Block size (downsampling factor)
        P_unit: Pressure unit conversion factor
        mu: Mean molecular weight
        k_b: Boltzmann constant (erg/K)

    Returns:
        Dictionary with coarse-grained fields:
        - rho_c: Coarse density (volume average)
        - ux_c: Coarse velocity X (mass-weighted)
        - uy_c: Coarse velocity Y (mass-weighted)
        - P_c: Coarse pressure (volume average)
        - s_c: List of coarse passive scalars (mass-weighted)
        - T_c: Coarse temperature (from EOS on coarse fields)
        - k_sgs: Subgrid kinetic energy (diagnostic)
    """
    if scalars is None:
        scalars = []

    # Volume averages: density, pressure
    rho_c = block_mean(rho, b)
    P_c = block_mean(P, b)

    # Mass-weighted velocity averages (conserve momentum)
    momx_c = block_mean(rho * ux, b)
    momy_c = block_mean(rho * uy, b)
    ux_c = momx_c / np.maximum(rho_c, 1e-30)
    uy_c = momy_c / np.maximum(rho_c, 1e-30)

    # Mass-weighted scalar averages (conserve mass of each species)
    s_c = []
    for s in scalars:
        s_mass_c = block_mean(rho * s, b)
        s_c.append(s_mass_c / np.maximum(rho_c, 1e-30))

    # Temperature derived from equation of state on coarse fields
    # This is the Favre (mass-weighted) temperature: T_tilde = rho*T / rho
    # Equivalent to the harmonic mean in near-isobaric flows
    T_c = (P_c * P_unit / rho_c) * (mu / k_b)

    # Subgrid kinetic energy: the KE difference due to subgrid velocity fluctuations
    # k_sgs = 0.5 * rho * (u^2 - u_bar^2) when spatially averaged
    ke_fine_c = block_mean(rho * (ux**2 + uy**2), b)  # rho * u^2 averaged
    ke_coarse_c = rho_c * (ux_c**2 + uy_c**2)         # rho_c * u_c^2
    k_sgs = 0.5 * (ke_fine_c - ke_coarse_c)

    return {
        'rho_c': rho_c,
        'ux_c': ux_c,
        'uy_c': uy_c,
        'P_c': P_c,
        's_c': s_c,
        'T_c': T_c,
        'k_sgs': k_sgs,
    }


def compute_subgrid_cooling(rho, temp, rho_c, T_c, lambda_cool_fn, b=32):
    """
    Compute fine-grid cooling and compare to coarse-grid cooling for subgrid term.

    Args:
        rho: Fine-resolution density
        temp: Fine-resolution temperature
        rho_c: Coarse density
        T_c: Coarse temperature
        lambda_cool_fn: Cooling function Lambda(T) returning erg cm^3 / s
        b: Block size

    Returns:
        Dictionary with:
        - cool_true: True cooling (fine-grid lambda coarse-averaged)
        - cool_closure: Closure approximation (lambda of coarse temperature)
        - tau_lambda: Subgrid cooling term (difference)
    """
    # Fine-grid cooling: number density squared times cooling function
    mu = 0.62
    n = rho / mu  # Number density (not mass density)
    cool_fine = n**2 * lambda_cool_fn(temp)

    # True subgrid average: average the fine-scale cooling
    cool_true = block_mean(cool_fine, b)

    # Closure approximation: compute from coarse temperature
    n_c = rho_c / mu
    cool_closure = n_c**2 * lambda_cool_fn(T_c)

    # Subgrid term: what the closure model needs to represent
    tau_lambda = cool_true - cool_closure

    return {
        'cool_true': cool_true,
        'cool_closure': cool_closure,
        'tau_lambda': tau_lambda,
    }


# Version tag for CNN input caches (cg_inputs.npy). Bump it whenever
# cnn_input_fields changes so stale caches are rebuilt instead of reused.
CNN_INPUT_SCHEME = "coarse_grain_utils_v1"
CNN_INPUT_SCHEME_FILE = "scheme.txt"


def cnn_input_fields(rho, ux, uy, P, s0, b, P_unit=1.59916e-14, mu=0.62, k_b=1.3807e-16):
    """
    Coarse-grain one snapshot into the ConvNN input stack.

    Channel order matches training and source_module inference:
    (rho, T, ux, uy, s0), with the averaging conventions of coarse_grain().

    Returns:
        float32 array of shape (5, ny//b, nx//b)
    """
    cg = coarse_grain(rho, ux, uy, P, scalars=[s0], b=b, P_unit=P_unit, mu=mu, k_b=k_b)
    return np.stack([cg['rho_c'], cg['T_c'], cg['ux_c'], cg['uy_c'], cg['s_c'][0]]).astype(np.float32)


def cache_scheme_matches(cache_dir):
    """True if cache_dir was written with the current CNN_INPUT_SCHEME."""
    path = os.path.join(cache_dir, CNN_INPUT_SCHEME_FILE)
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return f.read().strip() == CNN_INPUT_SCHEME


def write_cache_scheme(cache_dir):
    """Tag cache_dir with the current CNN_INPUT_SCHEME."""
    with open(os.path.join(cache_dir, CNN_INPUT_SCHEME_FILE), "w") as f:
        f.write(CNN_INPUT_SCHEME + "\n")
