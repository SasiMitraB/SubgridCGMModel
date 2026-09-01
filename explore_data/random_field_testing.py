"""
Generate 2D continuous random fields (Perlin noise and Gaussian Random Fields)
at 1024x512 resolution with varying hotspot-to-coldspot transition scales.
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.conv_nn import pdf_cnn

# Default pretrained Optuna model weights directory
DEFAULT_OPTUNA_MODEL_DIR = os.path.join(
    PROJECT_ROOT, "runs", "run_optuna_20260831_205922", "model_saves"
)


# =====================================================================
# 1. Vectorized 2D Perlin Noise Implementation (Pure NumPy)
# =====================================================================

def _fade(t):
    """Smooth 5th-order polynomial fade curve: 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def generate_perlin_noise_2d(shape=(1024, 512), res=(8, 16), seed=None):
    """
    Generate a 2D Perlin noise array.

    Parameters
    ----------
    shape : tuple of int (ny, nx)
        Output field resolution, e.g. (1024, 512).
    res : tuple of int (res_y, res_x)
        Number of grid cells along each axis.
        Higher res -> faster hotspot-to-coldspot variation.
        Lower res  -> slower, broader variation.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        2D array of shape `shape` with values normalized roughly in [-1, 1].
    """
    ny, nx = shape
    res_y, res_x = res

    rng = np.random.default_rng(seed)

    # Grid of random unit gradient vectors at cell vertices
    angles = 2.0 * np.pi * rng.random((res_y + 1, res_x + 1))
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    # Coordinate grid
    y = np.linspace(0, res_y, ny, endpoint=False)
    x = np.linspace(0, res_x, nx, endpoint=False)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    # Integer cell coordinates
    y0 = np.floor(yy).astype(int)
    x0 = np.floor(xx).astype(int)
    y1 = y0 + 1
    x1 = x0 + 1

    # Fractional offsets within cell
    dy = yy - y0
    dx = xx - x0

    # Distance vectors to 4 cell corners
    d00 = np.stack((dx, dy), axis=-1)
    d10 = np.stack((dx - 1.0, dy), axis=-1)
    d01 = np.stack((dx, dy - 1.0), axis=-1)
    d11 = np.stack((dx - 1.0, dy - 1.0), axis=-1)

    # Dot products with vertex gradients
    # gradients[y, x] has shape (res_y+1, res_x+1, 2)
    g00 = gradients[y0, x0]
    g10 = gradients[y0, x1]
    g01 = gradients[y1, x0]
    g11 = gradients[y1, x1]

    n00 = np.sum(d00 * g00, axis=-1)
    n10 = np.sum(d10 * g10, axis=-1)
    n01 = np.sum(d01 * g01, axis=-1)
    n11 = np.sum(d11 * g11, axis=-1)

    # Fade curves for interpolation
    fx = _fade(dx)
    fy = _fade(dy)

    # Bilinear interpolation
    nx0 = n00 * (1.0 - fx) + n10 * fx
    nx1 = n01 * (1.0 - fx) + n11 * fx
    noise = nx0 * (1.0 - fy) + nx1 * fy

    # Normalize to [-1, 1]
    noise_min, noise_max = noise.min(), noise.max()
    if noise_max > noise_min:
        noise = 2.0 * (noise - noise_min) / (noise_max - noise_min) - 1.0

    return noise


def generate_fractal_perlin_noise_2d(shape=(1024, 512), base_res=(4, 8),
                                     octaves=4, persistence=0.5, lacunarity=2.0, seed=None):
    """
    Generate fractal Perlin noise (fractional Brownian motion) by combining octaves.

    Parameters
    ----------
    shape : tuple of int (ny, nx)
        Output resolution.
    base_res : tuple of int (res_y, res_x)
        Base grid cell count for the lowest frequency octave.
    octaves : int
        Number of noise layers combined.
    persistence : float
        Amplitude scaling per octave (default: 0.5).
    lacunarity : float
        Frequency multiplier per octave (default: 2.0).
    seed : int, optional
        Random seed.
    """
    rng = np.random.default_rng(seed)
    total_noise = np.zeros(shape, dtype=np.float64)
    amplitude = 1.0
    frequency_y, frequency_x = base_res
    max_amp = 0.0

    for _ in range(octaves):
        layer_seed = rng.integers(0, 1_000_000)
        layer = generate_perlin_noise_2d(
            shape=shape,
            res=(int(round(frequency_y)), int(round(frequency_x))),
            seed=layer_seed
        )
        total_noise += amplitude * layer
        max_amp += amplitude
        amplitude *= persistence
        frequency_y *= lacunarity
        frequency_x *= lacunarity

    return total_noise / max_amp


# =====================================================================
# 2. Gaussian Random Field (Fourier Power-Law Method)
# =====================================================================

def generate_gaussian_random_field_2d(shape=(1024, 512), power_spectrum_index=2.5, seed=None):
    """
    Generate a 2D Gaussian Random Field (GRF) with power spectrum P(k) ~ k^(-beta).

    Parameters
    ----------
    shape : tuple of int (ny, nx)
        Field resolution.
    power_spectrum_index : float
        Spectral index beta.
        - Low beta (~1.0): Rapid, sharp transitions between hotspots and coldspots.
        - High beta (~3.5): Very smooth, gradual transitions across wide scales.
    seed : int, optional
        Random seed.
    """
    ny, nx = shape
    rng = np.random.default_rng(seed)

    # 2D wavevector grid
    ky = np.fft.fftfreq(ny).reshape(-1, 1)
    kx = np.fft.fftfreq(nx).reshape(1, -1)
    k = np.sqrt(ky**2 + kx**2)
    k[0, 0] = 1e-10  # Avoid zero-division

    # Power spectrum amplitude: A(k) ~ sqrt(P(k)) = k^(-beta/2)
    amplitude = k ** (-power_spectrum_index / 2.0)
    amplitude[0, 0] = 0.0  # Zero mean

    # Complex Gaussian white noise in Fourier space
    phases = rng.uniform(0, 2 * np.pi, shape)
    gaussian_noise = rng.standard_normal(shape)
    fourier_field = amplitude * gaussian_noise * np.exp(1j * phases)

    # Inverse FFT to real space
    field = np.real(np.fft.ifft2(fourier_field))

    # Standardize to zero mean, unit variance, and scaled roughly to [-1, 1]
    field = (field - field.mean()) / (field.std() + 1e-10)
    field = np.clip(field / 3.0, -1.0, 1.0)
    return field


# =====================================================================
# 3. AthenaK Mock Hydrodynamic Snapshot Generator
# =====================================================================

def generate_mock_athenak_snapshot(
    field,
    rho_cold=0.1,
    rho_hot=0.001,
    T_cold=1.0e4,
    T_hot=1.0e6,
    vx_hot=28.1818,
    vx_cold=-2.8182,
    turb_scale=0.25,
    gamma=5.0 / 3.0,
    mu=0.62,
    kb=1.3807e-16,
    P_unit=1.59916e-14,
    seed=None,
):
    """
    Synthesize a physically consistent AthenaK CGM hydrodynamic snapshot from a 2D scalar field.

    Parameters
    ----------
    field : np.ndarray (ny, nx)
        2D base random field (e.g. from Perlin, Fractal Perlin, or GRF).
    rho_cold : float
        Cold cloud mass density in code units (default: 0.1).
    rho_hot : float
        Hot ambient background mass density in code units (default: 0.001).
    T_cold : float
        Cold phase temperature in Kelvin (default: 1e4 K).
    T_hot : float
        Hot phase temperature in Kelvin (default: 1e6 K).
    vx_hot : float
        Hot stream horizontal velocity in km/s (default: +28.18 km/s).
    vx_cold : float
        Cold stream horizontal velocity in km/s (default: -2.82 km/s).
    turb_scale : float
        Turbulent velocity dispersion fraction relative to shear velocity jump.
    gamma : float
        Adiabatic index (default: 5/3).
    mu : float
        Mean molecular weight (default: 0.62).
    kb : float
        Boltzmann constant in CGS (erg/K).
    P_unit : float
        AthenaK pressure conversion unit in CGS (erg/cm^3).
    seed : int, optional
        Seed for divergence-free turbulent velocity streamfunction.

    Returns
    -------
    dict of str -> np.ndarray
        Dictionary containing 2D arrays (ny, nx):
        - 'passive_scalar' : float array in [0, 1] (1 = cold gas, 0 = hot ambient)
        - 'temperature'    : Temperature in Kelvin (10^4 K to 10^6 K)
        - 'density'        : Mass density in code units (isobaric ~ T^-1)
        - 'ux'             : Horizontal velocity in km/s (shear + solenoidal turbulence)
        - 'uy'             : Vertical velocity in km/s (solenoidal turbulence)
        - 'pressure'       : Thermal pressure in AthenaK code units
        - 'eint'           : Specific internal energy P / ((gamma - 1) * rho)
    """
    ny, nx = field.shape

    # 1. Passive Scalar s in [0, 1] (1 = pure cold cloud, 0 = pure hot background)
    f_min, f_max = field.min(), field.max()
    if f_max > f_min:
        # Invert so positive hotspot / ridge in field can represent cold cloud if desired,
        # or direct map: s in [0, 1]
        s = (field - f_min) / (f_max - f_min)
    else:
        s = np.zeros_like(field)

    # 2. Temperature T in Kelvin (logarithmic multiphase mixing between 10^4 K and 10^6 K)
    log_T_cold = np.log10(T_cold)
    log_T_hot = np.log10(T_hot)
    log_T = log_T_hot - s * (log_T_hot - log_T_cold)
    temp = 10.0 ** log_T

    # 3. Density rho in AthenaK code units (quasi-isobaric pressure equilibrium: rho ~ 1/T)
    # At T_cold (1e4 K) -> rho = rho_cold (0.1), at T_hot (1e6 K) -> rho = rho_hot (0.001)
    rho = rho_cold * (T_cold / temp)

    # 4. Thermal pressure and internal energy
    # P_code = (rho * temp * kb / (mu * P_unit))
    # In isobaric balance, pressure is nearly uniform across the domain
    P_base = (rho_cold * T_cold * kb) / (mu * P_unit)
    pressure = (rho * temp * kb) / (mu * P_unit)
    eint = pressure / ((gamma - 1.0) * rho)

    # 5. Velocity field (ux, uy)
    # (a) Mean shear flow correlated with the passive scalar / cloud location
    ux_shear = vx_hot + s * (vx_cold - vx_hot)

    # (b) Solenoidal (divergence-free) turbulent fluctuations via streamfunction psi(x, y)
    # dux = d(psi)/dy, duy = -d(psi)/dx => div(du) = 0
    rng = np.random.default_rng(seed)
    stream_seed = rng.integers(0, 1_000_000)
    psi = generate_fractal_perlin_noise_2d(
        shape=(ny, nx),
        base_res=(8, 4),
        octaves=4,
        persistence=0.5,
        seed=stream_seed
    )

    # Numerical derivatives for streamfunction
    dux = np.gradient(psi, axis=0)  # d(psi)/dy
    duy = -np.gradient(psi, axis=1) # -d(psi)/dx

    # Scale turbulence relative to shear velocity jump |vx_hot - vx_cold|
    shear_jump = abs(vx_hot - vx_cold)
    du_rms = np.sqrt(np.mean(dux**2 + duy**2)) + 1e-10
    turb_amp = turb_scale * shear_jump

    dux = turb_amp * (dux / du_rms)
    duy = turb_amp * (duy / du_rms)

    ux = ux_shear + dux
    uy = duy

    return {
        "density": rho,
        "temperature": temp,
        "ux": ux,
        "uy": uy,
        "passive_scalar": s,
        "pressure": pressure,
        "eint": eint,
    }


def plot_mock_snapshot(snapshot, title="AthenaK Mock Snapshot", save_path=None):
    """
    Plot a 5-variable mock snapshot (Density, Temperature, ux, uy, Passive Scalar).
    """
    fig, axes = plt.subplots(3, 2, figsize=(11, 15), constrained_layout=True)

    ny, nx = snapshot["density"].shape

    # 1. Density
    im0 = axes[0, 0].imshow(np.log10(snapshot["density"]), cmap="viridis", origin="lower", aspect="auto")
    axes[0, 0].set_title(r"Mass Density $\log_{10}(\rho)$ [code units]", fontsize=12, fontweight="bold")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # 2. Temperature
    im1 = axes[0, 1].imshow(np.log10(snapshot["temperature"]), cmap="coolwarm", origin="lower", aspect="auto")
    axes[0, 1].set_title(r"Temperature $\log_{10}(T)$ [K]", fontsize=12, fontweight="bold")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # 3. Passive Scalar
    im2 = axes[1, 0].imshow(snapshot["passive_scalar"], cmap="cividis", origin="lower", aspect="auto", vmin=0, vmax=1)
    axes[1, 0].set_title("Passive Scalar $s \in [0, 1]$ (Cold Gas Fraction)", fontsize=12, fontweight="bold")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 4. Horizontal Velocity ux
    im3 = axes[1, 1].imshow(snapshot["ux"], cmap="RdBu_r", origin="lower", aspect="auto")
    axes[1, 1].set_title(r"Horizontal Velocity $u_x$ [km/s]", fontsize=12, fontweight="bold")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 5. Vertical Velocity uy
    im4 = axes[2, 0].imshow(snapshot["uy"], cmap="bwr", origin="lower", aspect="auto")
    axes[2, 0].set_title(r"Vertical Velocity $u_y$ [km/s] (Solenoidal Eddies)", fontsize=12, fontweight="bold")
    fig.colorbar(im4, ax=axes[2, 0], fraction=0.046, pad=0.04)

    # 6. Velocity Magnitude & Streamlines
    speed = np.sqrt(snapshot["ux"]**2 + snapshot["uy"]**2)
    im5 = axes[2, 1].imshow(speed, cmap="magma", origin="lower", aspect="auto")
    axes[2, 1].set_title(r"Velocity Magnitude $|\mathbf{u}|$ [km/s]", fontsize=12, fontweight="bold")
    fig.colorbar(im5, ax=axes[2, 1], fraction=0.046, pad=0.04)

    # Clean ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(title, fontsize=15, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved snapshot figure to: {save_path}")
    else:
        plt.show()


# =====================================================================
# 4. Demonstration & Visualization (Executed Directly)
# =====================================================================

# Target resolution: 1024 height (vertical) x 512 width (horizontal)
shape = (1024, 512)
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "random_fields")
os.makedirs(output_dir, exist_ok=True)
print(f"Generating 2D continuous random fields at resolution: {shape[1]} (width) x {shape[0]} (height)...")
print(f"All figures will be saved in: {output_dir}")

# -----------------------------------------------------------------
# Case A: Perlin Noise with varying spatial frequency (res_y, res_x)
# -----------------------------------------------------------------
perlin_configs = [
    {"title": "1. Very Slow Transition (Broad scale)",  "res": (4, 2),   "seed": 42},
    {"title": "2. Moderate Transition (Medium scale)",  "res": (12, 6),  "seed": 101},
    {"title": "3. Fast Transition (Fine scale)",        "res": (32, 16), "seed": 202},
    {"title": "4. Very Fast Transition (Rapid scale)",  "res": (64, 32), "seed": 303},
]

perlin_realizations = []
for cfg in perlin_configs:
    field = generate_perlin_noise_2d(shape=shape, res=cfg["res"], seed=cfg["seed"])
    perlin_realizations.append((cfg["title"], field))

# -----------------------------------------------------------------
# Case B: Multi-Octave Fractal Perlin Noise (fBm)
# -----------------------------------------------------------------
fractal_configs = [
    {"title": "Fractal: Smooth (Base 4x2, 3 octaves)",  "base_res": (4, 2),   "octaves": 3, "seed": 10},
    {"title": "Fractal: Medium (Base 8x4, 4 octaves)",  "base_res": (8, 4),   "octaves": 4, "seed": 20},
    {"title": "Fractal: Dynamic (Base 16x8, 5 octaves)","base_res": (16, 8),  "octaves": 5, "seed": 30},
    {"title": "Fractal: Rough (Base 32x16, 5 octaves)", "base_res": (32, 16), "octaves": 5, "seed": 40},
]

fractal_realizations = []
for cfg in fractal_configs:
    field = generate_fractal_perlin_noise_2d(
        shape=shape,
        base_res=cfg["base_res"],
        octaves=cfg["octaves"],
        persistence=0.55,
        lacunarity=2.0,
        seed=cfg["seed"]
    )
    fractal_realizations.append((cfg["title"], field))

# -----------------------------------------------------------------
# Case C: Gaussian Random Fields with power-law index beta
# -----------------------------------------------------------------
grf_configs = [
    {"title": "GRF: beta=3.5 (Very Smooth / Slow)", "beta": 3.5, "seed": 111},
    {"title": "GRF: beta=2.5 (Standard CGM/ISM-like)", "beta": 2.5, "seed": 222},
    {"title": "GRF: beta=1.8 (Fast Fluctuation)", "beta": 1.8, "seed": 333},
    {"title": "GRF: beta=1.2 (Rapid / Turbulent)", "beta": 1.2, "seed": 444},
]

grf_realizations = []
for cfg in grf_configs:
    field = generate_gaussian_random_field_2d(shape=shape, power_spectrum_index=cfg["beta"], seed=cfg["seed"])
    grf_realizations.append((cfg["title"], field))

# -----------------------------------------------------------------
# Plotting & Comparison of Raw Scalar Fields
# -----------------------------------------------------------------
fig_path = os.path.join(output_dir, "perlin_field_realizations.png")
fig, axes = plt.subplots(4, 3, figsize=(12, 18), constrained_layout=True)

col_headers = [
    "Single-Scale Perlin Noise\n(Varying Grid Frequency)",
    "Fractal Perlin Noise (fBm)\n(Multi-Octave Roughness)",
    "Gaussian Random Field\n(Varying Power Spectrum P(k)~k^-β)"
]

for col, title in enumerate(col_headers):
    axes[0, col].set_title(title, fontsize=13, fontweight="bold", pad=10)

for row in range(4):
    # Column 0: Single-scale Perlin
    p_title, p_field = perlin_realizations[row]
    im0 = axes[row, 0].imshow(p_field, cmap="RdBu_r", origin="lower", aspect="auto", vmin=-1, vmax=1)
    axes[row, 0].set_ylabel(p_title.split(". ")[-1], fontsize=10, fontweight="semibold")
    axes[row, 0].set_xticks([])
    axes[row, 0].set_yticks([])

    # Column 1: Fractal Perlin
    f_title, f_field = fractal_realizations[row]
    im1 = axes[row, 1].imshow(f_field, cmap="RdBu_r", origin="lower", aspect="auto", vmin=-1, vmax=1)
    axes[row, 1].set_ylabel(f_title.split(": ")[-1], fontsize=10, fontweight="semibold")
    axes[row, 1].set_xticks([])
    axes[row, 1].set_yticks([])

    # Column 2: Gaussian Random Field
    g_title, g_field = grf_realizations[row]
    im2 = axes[row, 2].imshow(g_field, cmap="RdBu_r", origin="lower", aspect="auto", vmin=-1, vmax=1)
    axes[row, 2].set_ylabel(g_title.split(": ")[-1], fontsize=10, fontweight="semibold")
    axes[row, 2].set_xticks([])
    axes[row, 2].set_yticks([])

cbar = fig.colorbar(im0, ax=axes, orientation="horizontal", fraction=0.03, pad=0.03)
cbar.set_label("Field Value (Coldspot: Blue [-1]  ↔  Hotspot: Red [+1])", fontsize=11, fontweight="bold")

plt.suptitle("2D Continuous Random Fields (512 Width x 1024 Height)\nVarying Hotspot-to-Coldspot Transition Rates",
             fontsize=15, fontweight="bold")

plt.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved visualization plot to: {fig_path}")

# Plot 1D Transects (both Horizontal & Vertical Slices)
transect_path = os.path.join(output_dir, "transition_gradient_transects.png")
fig_tr, (ax_h, ax_v) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

mid_y = shape[0] // 2
mid_x = shape[1] // 2

for (p_title, p_field), cfg in zip(perlin_realizations, perlin_configs):
    label = f"Res {cfg['res']} ({p_title.split('. ')[-1]})"
    ax_h.plot(p_field[mid_y, :], label=label, lw=1.8)
    ax_v.plot(p_field[:, mid_x], label=label, lw=1.8)

ax_h.set_title(f"Horizontal Transect (at Y = {mid_y})", fontsize=11, fontweight="bold")
ax_h.set_xlabel("X Pixel Coordinate (0 to 511)", fontsize=10)
ax_h.set_ylabel("Field Value", fontsize=10)
ax_h.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax_h.legend(loc="upper right", fontsize=8, frameon=True)
ax_h.grid(True, alpha=0.3)

ax_v.set_title(f"Vertical Transect (at X = {mid_x})", fontsize=11, fontweight="bold")
ax_v.set_xlabel("Y Pixel Coordinate (0 to 1023)", fontsize=10)
ax_v.set_ylabel("Field Value", fontsize=10)
ax_v.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax_v.legend(loc="upper right", fontsize=8, frameon=True)
ax_v.grid(True, alpha=0.3)

plt.suptitle("1D Transects Demonstrating Spatial Gradient Scale Across Dimensions", fontsize=13, fontweight="bold")
plt.savefig(transect_path, dpi=200)
plt.close()
print(f"Saved transect comparison plot to: {transect_path}")


def coarse_grain_field(field, target_shape=(16, 8)):
    """
    Coarse-grain a 2D array by 2D block averaging to target_shape (ny_coarse, nx_coarse).
    For (1024, 512) -> (16, 8), this is a 64x64 block average.
    """
    ny, nx = field.shape
    t_ny, t_nx = target_shape
    block_y = ny // t_ny
    block_x = nx // t_nx
    return field.reshape(t_ny, block_y, t_nx, block_x).mean(axis=(1, 3))


def coarse_grain_snapshot(snapshot, target_shape=(16, 8)):
    """
    Coarse-grain all 2D hydrodynamic variables in a snapshot dictionary to target_shape.
    """
    return {k: coarse_grain_field(v, target_shape=target_shape) for k, v in snapshot.items()}


def compute_true_subgrid_pdf(temp_fine, T_edges, target_shape=(16, 8)):
    """
    Compute true pixel temperature PDF within each coarse cell from the fine grid.

    Returns
    -------
    np.ndarray, shape (40, 16, 8)
        Probability distribution of temperature within each 64x64 pixel subgrid block.
    """
    ny, nx = temp_fine.shape
    t_ny, t_nx = target_shape
    block_y = ny // t_ny
    block_x = nx // t_nx
    n_bins = len(T_edges) - 1

    bin_idx = np.digitize(temp_fine, T_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    blocks = bin_idx.reshape(t_ny, block_y, t_nx, block_x).transpose(0, 2, 1, 3).reshape(t_ny, t_nx, -1)

    pdf = np.zeros((n_bins, t_ny, t_nx), dtype=np.float32)
    for i in range(t_ny):
        for j in range(t_nx):
            counts = np.bincount(blocks[i, j], minlength=n_bins)
            pdf[:, i, j] = counts / counts.sum()
    return pdf


def plot_fine_vs_coarse_comparison(fine_snap, coarse_snap, title="Fine (1024x512) vs Coarse (16x8) Comparison", save_path=None):
    """
    Side-by-side comparison of 1024x512 fine resolution vs 16x8 coarse-grained fields
    with synchronized colorbars per variable.
    """
    fig, axes = plt.subplots(5, 2, figsize=(10, 18), constrained_layout=True)
    variables = [
        (r"Density $\log_{10}(\rho)$", lambda s: np.log10(s["density"]), "viridis"),
        (r"Temperature $\log_{10}(T)$ [K]", lambda s: np.log10(s["temperature"]), "coolwarm"),
        (r"Passive Scalar $s \in [0, 1]$", lambda s: s["passive_scalar"], "cividis"),
        (r"Velocity $u_x$ [km/s]", lambda s: s["ux"], "RdBu_r"),
        (r"Velocity $u_y$ [km/s]", lambda s: s["uy"], "bwr"),
    ]

    axes[0, 0].set_title("Original Fine (1024x512)", fontsize=12, fontweight="bold", pad=8)
    axes[0, 1].set_title("Coarse-Grained 64x (16x8)", fontsize=12, fontweight="bold", pad=8)

    for row_idx, (var_name, var_func, cmap) in enumerate(variables):
        fine_data = var_func(fine_snap)
        coarse_data = var_func(coarse_snap)

        vmin = min(fine_data.min(), coarse_data.min())
        vmax = max(fine_data.max(), coarse_data.max())

        axes[row_idx, 0].imshow(fine_data, cmap=cmap, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
        axes[row_idx, 0].set_ylabel(var_name, fontsize=10, fontweight="bold")
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])

        im1 = axes[row_idx, 1].imshow(coarse_data, cmap=cmap, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
        axes[row_idx, 1].set_xticks([])
        axes[row_idx, 1].set_yticks([])

        fig.colorbar(im1, ax=[axes[row_idx, 0], axes[row_idx, 1]], fraction=0.046, pad=0.03)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved comparison plot to: {save_path}")
    else:
        plt.show()


def plot_cooling_comparison(true_cooling, coarse_cooling, cnn_cooling, title="Cooling Comparison", save_path=None):
    """
    4-panel comparison of True Ground Truth Cooling vs Unresolved Coarse Cooling vs CNN Predicted Cooling.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)

    eps = 1e-35
    log_true = np.log10(np.maximum(true_cooling, eps))
    log_coarse = np.log10(np.maximum(coarse_cooling, eps))
    log_cnn = np.log10(np.maximum(cnn_cooling, eps))

    # Shared color limits across all 3 cooling maps
    vmin = min(log_true.min(), log_coarse.min(), log_cnn.min())
    vmax = max(log_true.max(), log_coarse.max(), log_cnn.max())

    # 1. True Cooling
    im0 = axes[0, 0].imshow(log_true, cmap="inferno", origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(r"1. True Ground-Truth Cooling" + "\n" + r"$\log_{10} \langle \rho^2 \Lambda(T) \rangle_{64\times64}$", fontsize=11, fontweight="bold")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # 2. Unresolved Coarse Cooling
    im1 = axes[0, 1].imshow(log_coarse, cmap="inferno", origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(r"2. Unresolved Coarse (Single-T)" + "\n" + r"$\log_{10} [\bar{\rho}^2 \Lambda(\bar{T})]$", fontsize=11, fontweight="bold")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # 3. CNN Subgrid Cooling
    im2 = axes[1, 0].imshow(log_cnn, cmap="inferno", origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title(r"3. CNN Subgrid Model Cooling" + "\n" + r"$\log_{10} [\bar{\rho}^2 \sum P_{\rm CNN}(T_k) \Lambda(T_k)]$", fontsize=11, fontweight="bold")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 4. Cooling Ratio / Discrepancy comparison
    ratio_coarse = (coarse_cooling + eps) / (true_cooling + eps)
    ratio_cnn = (cnn_cooling + eps) / (true_cooling + eps)

    # Log10 error map of CNN vs True
    log_err_cnn = np.log10(ratio_cnn)
    lim = max(abs(log_err_cnn.min()), abs(log_err_cnn.max()), 0.5)
    im3 = axes[1, 1].imshow(log_err_cnn, cmap="coolwarm", origin="lower", aspect="auto", vmin=-lim, vmax=lim)
    axes[1, 1].set_title(r"4. CNN Fidelity: $\log_{10}(\mathcal{C}_{\rm CNN} / \mathcal{C}_{\rm True})$" + "\n(0 = Exact Match, Red=Overcool, Blue=Undercool)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=13, fontweight="bold")
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved cooling comparison plot to: {save_path}")
    else:
        plt.show()


def plot_pdf_comparison(true_pdf, cnn_pdf, coarse_snap, title="Subgrid Temperature PDF Comparison", save_path=None):
    """
    Compare Ground Truth vs CNN-predicted Temperature PDFs globally and for representative coarse cells.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    T_centers = pdf_cnn.T_centers
    logT = np.log10(T_centers)
    lam = pdf_cnn.lambda_cool(T_centers)

    # 1. Global Domain-Averaged PDF
    true_mean_pdf = true_pdf.mean(axis=(1, 2))
    cnn_mean_pdf = cnn_pdf.mean(axis=(1, 2))

    ax0 = axes[0, 0]
    ax0.plot(logT, true_mean_pdf, "k-", lw=2.5, label="True Ground-Truth PDF")
    ax0.plot(logT, cnn_mean_pdf, "r--", lw=2.0, label="CNN Predicted PDF")
    ax0.set_title("Domain-Averaged Subgrid Temperature PDF", fontsize=11, fontweight="bold")
    ax0.set_xlabel(r"$\log_{10}(T)$ [K]", fontsize=10)
    ax0.set_ylabel("Probability Density", fontsize=10)
    ax0.legend(loc="upper right", frameon=True)
    ax0.grid(True, alpha=0.3)

    # Secondary axis for cooling curve Lambda(T)
    ax0_twin = ax0.twinx()
    ax0_twin.plot(logT, np.log10(np.maximum(lam, 1e-35)), color="blue", alpha=0.25, linestyle=":", lw=1.5, label=r"$\Lambda(T)$")
    ax0_twin.set_ylabel(r"$\log_{10}\Lambda(T)$", color="blue", fontsize=9)
    ax0_twin.tick_params(axis='y', labelcolor='blue')

    # Find representative cells based on passive scalar s (Cold, Mixing, Hot)
    s = coarse_snap["passive_scalar"]
    ny_c, nx_c = s.shape

    # Cold core cell (max s)
    cold_idx = np.unravel_index(np.argmax(s), s.shape)
    # Hot ambient cell (min s)
    hot_idx = np.unravel_index(np.argmin(s), s.shape)
    # Mixing cell (closest to s=0.5)
    mix_idx = np.unravel_index(np.argmin(np.abs(s - 0.5)), s.shape)

    cells = [
        (axes[0, 1], mix_idx, f"Mixing Layer Cell (y={mix_idx[0]}, x={mix_idx[1]}, s={s[mix_idx]:.2f})"),
        (axes[1, 0], cold_idx, f"Cold Cloud Core Cell (y={cold_idx[0]}, x={cold_idx[1]}, s={s[cold_idx]:.2f})"),
        (axes[1, 1], hot_idx, f"Hot Background Cell (y={hot_idx[0]}, x={hot_idx[1]}, s={s[hot_idx]:.2f})"),
    ]

    for ax, idx, c_title in cells:
        t_p = true_pdf[:, idx[0], idx[1]]
        c_p = cnn_pdf[:, idx[0], idx[1]]
        cell_T = coarse_snap["temperature"][idx[0], idx[1]]

        ax.plot(logT, t_p, "k-", lw=2.2, label="True Subgrid PDF")
        ax.plot(logT, c_p, "r--", lw=2.0, label="CNN Prediction")
        ax.axvline(np.log10(cell_T), color="green", linestyle="-.", alpha=0.7, label=r"Resolved Mean $\bar{T}$")
        ax.set_title(c_title, fontsize=11, fontweight="bold")
        ax.set_xlabel(r"$\log_{10}(T)$ [K]", fontsize=10)
        ax.set_ylabel("Probability Density", fontsize=10)
        ax.legend(loc="upper right", fontsize=8, frameon=True)
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved PDF comparison plot to: {save_path}")
    else:
        plt.show()


# =====================================================================
# 5. Generate, Evaluate CNN, Compare Cooling & Save Per-Field Folders
# =====================================================================
print("\n" + "="*70)
print(f"Loading pretrained Optuna CNN model from:\n{DEFAULT_OPTUNA_MODEL_DIR}")
print("="*70)

# Collect all 12 field configurations across all 3 generators
all_field_configs = []
for p_title, p_field in perlin_realizations:
    slug = "perlin_" + p_title.split(". ")[-1].lower().split(" (")[0].replace(" ", "_")
    all_field_configs.append((p_title, slug, p_field, 500 + len(all_field_configs)))

for f_title, f_field in fractal_realizations:
    slug = "fractal_" + f_title.split(": ")[-1].lower().split(" (")[0].replace(" ", "_")
    all_field_configs.append((f_title, slug, f_field, 600 + len(all_field_configs)))

for g_title, g_field in grf_realizations:
    slug = "grf_" + g_title.split(": ")[-1].lower().split(" (")[0].replace(" ", "_").replace("=", "_").replace(".", "_")
    all_field_configs.append((g_title, slug, g_field, 700 + len(all_field_configs)))

snapshots_base_dir = os.path.join(output_dir, "mock_snapshots")
os.makedirs(snapshots_base_dir, exist_ok=True)

# Cache cooling function at bin centers for vectorized CNN cooling calculation
lambda_centers = pdf_cnn.lambda_cool(pdf_cnn.T_centers)  # shape: (40,)

summary_results = []

for title, slug, raw_field, s_seed in all_field_configs:
    # 1. Create separate directory for this field
    field_dir = os.path.join(snapshots_base_dir, slug)
    os.makedirs(field_dir, exist_ok=True)
    print(f"\nProcessing field: {title}\n -> Target folder: {field_dir}")

    # 2. Generate fine snapshot (1024, 512)
    fine_snap = generate_mock_athenak_snapshot(raw_field, seed=s_seed)

    # 3. Coarse-grain by 64x block averaging to (16, 8)
    coarse_snap = coarse_grain_snapshot(fine_snap, target_shape=(16, 8))

    # 4. Predict CNN Subgrid Temperature PDF using pretrained Optuna model
    cnn_pdf = pdf_cnn.snapshot_pred_16x8(
        rho=coarse_snap["density"],
        temp=coarse_snap["temperature"],
        ux=coarse_snap["ux"],
        uy=coarse_snap["uy"],
        ps=coarse_snap["passive_scalar"],
        fine_resolution=(1024, 512),
        downsample=64,
        model_save_dir=DEFAULT_OPTUNA_MODEL_DIR,
    )  # shape: (40, 16, 8)

    # 5. Compute Ground-Truth Subgrid PDF from Fine Grid
    true_pdf = compute_true_subgrid_pdf(
        fine_snap["temperature"],
        pdf_cnn.T_edges,
        target_shape=(16, 8)
    )  # shape: (40, 16, 8)

    # 6. Compute Radiative Cooling Rates
    # (a) Original fine pixel cooling: rho^2 * Lambda(T)
    fine_cooling = (fine_snap["density"]**2) * pdf_cnn.lambda_cool(fine_snap["temperature"])
    # True block-averaged ground truth cell cooling:
    true_cooling_16x8 = coarse_grain_field(fine_cooling, target_shape=(16, 8))

    # (b) Unresolved Single-T coarse cooling: rho_bar^2 * Lambda(T_bar)
    coarse_cooling_16x8 = (coarse_snap["density"]**2) * pdf_cnn.lambda_cool(coarse_snap["temperature"])

    # (c) CNN Predicted subgrid cooling: rho_bar^2 * sum_k [ P_CNN(T_k) * Lambda(T_k) ]
    cnn_cooling_16x8 = (coarse_snap["density"]**2) * np.tensordot(lambda_centers, cnn_pdf, axes=(0, 0))

    # Compute bulk cooling error metrics
    tot_true_cool = true_cooling_16x8.sum()
    tot_coarse_cool = coarse_cooling_16x8.sum()
    tot_cnn_cool = cnn_cooling_16x8.sum()

    coarse_err = abs(tot_coarse_cool - tot_true_cool) / (tot_true_cool + 1e-30) * 100.0
    cnn_err = abs(tot_cnn_cool - tot_true_cool) / (tot_true_cool + 1e-30) * 100.0

    print(f"  [Cooling Totals] True: {tot_true_cool:.3e} | Coarse: {tot_coarse_cool:.3e} (err: {coarse_err:.1f}%) | CNN: {tot_cnn_cool:.3e} (err: {cnn_err:.1f}%)")

    summary_results.append({
        "title": title,
        "slug": slug,
        "true_cool": tot_true_cool,
        "coarse_cool": tot_coarse_cool,
        "cnn_cool": tot_cnn_cool,
        "coarse_err_pct": coarse_err,
        "cnn_err_pct": cnn_err,
    })

    # 7. Save Plots inside the field's folder
    # (a) Fine vs Coarse Hydro Fields side-by-side
    comp_plot_path = os.path.join(field_dir, "fine_vs_coarse_comparison.png")
    plot_fine_vs_coarse_comparison(
        fine_snap, coarse_snap,
        title=f"Hydro Fields: Fine (1024x512) vs Coarse (16x8)\n{title}",
        save_path=comp_plot_path
    )

    # (b) Cooling Comparison (True vs Coarse Single-T vs CNN)
    cooling_plot_path = os.path.join(field_dir, "cooling_comparison.png")
    plot_cooling_comparison(
        true_cooling_16x8, coarse_cooling_16x8, cnn_cooling_16x8,
        title=f"Radiative Cooling Comparison\n{title}",
        save_path=cooling_plot_path
    )

    # (c) Temperature PDF Subgrid Comparison (True vs CNN)
    pdf_plot_path = os.path.join(field_dir, "pdf_subgrid_comparison.png")
    plot_pdf_comparison(
        true_pdf, cnn_pdf, coarse_snap,
        title=f"Subgrid Temperature PDF Comparison (40 Bins)\n{title}",
        save_path=pdf_plot_path
    )

    # (d) Diagnostic 6-panel snapshots
    fine_plot_path = os.path.join(field_dir, "original_fine_snapshot.png")
    plot_mock_snapshot(fine_snap, title=f"Original Fine Snapshot (1024x512)\n{title}", save_path=fine_plot_path)

    coarse_plot_path = os.path.join(field_dir, "coarse_grained_snapshot_16x8.png")
    plot_mock_snapshot(coarse_snap, title=f"Coarse-Grained 64x Snapshot (16x8)\n{title}", save_path=coarse_plot_path)

    # 8. Save raw NumPy data (.npz) for all fields, PDFs, and cooling arrays
    npz_data = {
        "true_cooling_16x8": true_cooling_16x8,
        "coarse_cooling_16x8": coarse_cooling_16x8,
        "cnn_cooling_16x8": cnn_cooling_16x8,
        "true_pdf_40x16x8": true_pdf,
        "cnn_pdf_40x16x8": cnn_pdf,
        "T_centers": pdf_cnn.T_centers,
        "T_edges": pdf_cnn.T_edges,
    }
    for k in fine_snap:
        npz_data[f"fine_{k}"] = fine_snap[k]
        npz_data[f"coarse_{k}"] = coarse_snap[k]

    npz_path = os.path.join(field_dir, "snapshot_data.npz")
    np.savez_compressed(npz_path, **npz_data)
    print(f"  -> Saved all arrays (fields, cooling, PDFs) to: {npz_path}")

# =====================================================================
# 6. Overall Multi-Field Cooling Benchmark Summary Plot
# =====================================================================
summary_fig, (ax_bar, ax_err) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

labels = [s["slug"].replace("_", " ").title() for s in summary_results]
x = np.arange(len(labels))
width = 0.28

# Bar chart of Total Cooling
ax_bar.bar(x - width, [s["true_cool"] for s in summary_results], width, label="True Ground Truth", color="black", alpha=0.85)
ax_bar.bar(x, [s["cnn_cool"] for s in summary_results], width, label="CNN Subgrid Model", color="crimson", alpha=0.85)
ax_bar.bar(x + width, [s["coarse_cool"] for s in summary_results], width, label="Unresolved Coarse (Single-T)", color="dodgerblue", alpha=0.85)
ax_bar.set_ylabel("Total Domain Radiative Cooling Rate", fontsize=11, fontweight="bold")
ax_bar.set_title("Total Radiative Cooling Across Random Fields", fontsize=12, fontweight="bold")
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
ax_bar.set_yscale("log")
ax_bar.legend(loc="upper right", frameon=True)
ax_bar.grid(True, which="both", alpha=0.2)

# Bar chart of Percentage Errors
ax_err.bar(x - width/2, [s["cnn_err_pct"] for s in summary_results], width, label="CNN Model Error (%)", color="crimson", alpha=0.85)
ax_err.bar(x + width/2, [s["coarse_err_pct"] for s in summary_results], width, label="Unresolved Coarse Error (%)", color="dodgerblue", alpha=0.85)
ax_err.set_ylabel("Relative Cooling Error (%)", fontsize=11, fontweight="bold")
ax_err.set_title("Cooling Discrepancy Relative to Fine Ground Truth", fontsize=12, fontweight="bold")
ax_err.set_xticks(x)
ax_err.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
ax_err.set_yscale("log")
ax_err.legend(loc="upper right", frameon=True)
ax_err.grid(True, which="both", alpha=0.2)

benchmark_plot_path = os.path.join(output_dir, "all_fields_cooling_benchmark.png")
plt.suptitle("Radiative Cooling & Subgrid CNN Benchmark Across 12 Random Field Configurations", fontsize=14, fontweight="bold")
plt.savefig(benchmark_plot_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"\nSaved overall cooling benchmark plot to: {benchmark_plot_path}")
print(f"\nAll mock AthenaK snapshots processed and saved into {len(all_field_configs)} separate folders under:\n{snapshots_base_dir}")



