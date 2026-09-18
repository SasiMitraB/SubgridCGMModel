# Python script to coarse-grain the hr_build_512 dataset to several resolutions
# and compare two ways of estimating the coarse-cell emissivity from the
# subgrid temperature PDF:
#
#   Version 1 (isochoric, constant-n):
#       eps_1 = n^2 * sum_i pdf(T_i) * Lambda(T_i)
#
#   Version 2 (isobaric, per-bin density from P = n_i kB T_i):
#       eps_2 = (P/kB)^2 * sum_i pdf(T_i) * Lambda(T_i) / T_i^2
#
# For each resolution, t_cool = e_int / eps is computed for both versions and
# plotted against the coarse-grained temperature.

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from models.conv_nn.pdf_cnn import lambda_cool

# =========================
# SETTINGS
# =========================
FINE_RESOLUTION = (512, 256)          # (nx, ny) of the hr_build_512 dataset
TARGET_COARSE_RESOLUTIONS = [(16, 8), (32, 16)]
BINS = 40
TEMP_BIN_EDGES = np.logspace(3.0, 7.0, BINS + 1)
TEMP_BIN_CENTERS = np.sqrt(TEMP_BIN_EDGES[:-1] * TEMP_BIN_EDGES[1:])

NUM_FRAMES = int(os.environ.get("TCOOL_NUM_FRAMES", "50"))

CACHE_PATH = os.environ.get(
    "SUBGRID_CACHE_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../simulation_outputs/hr_build_512/cache")),
)
CACHE_DIR = os.path.join(CACHE_PATH, f"sc{FINE_RESOLUTION}_32")

OUT_DIR = os.environ.get("PDF_MOCKS_DIR", os.path.join(os.path.dirname(__file__), "tcool_compare"))
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# PHYSICAL CONSTANTS (same code-unit conventions as pdf_plot.py)
# =========================
kb = 1.3807e-16               # Boltzmann constant [erg / K]
mu = 0.62                     # mean molecular weight
gamma = 5.0 / 3.0

_L_cgs = 3.08568e18            # 1 pc in cm
_T_cgs = 3.15576e13            # 1 Myr in s
_M_cgs = 4.91417e31            # code mass unit [g]
_RHO_cgs = _M_cgs / _L_cgs ** 3
m_H = 1.67262e-24
n_to_cm3 = _RHO_cgs / (mu * m_H)      # cm^-3 per code density unit
_V_cgs = _L_cgs / _T_cgs
_P_cgs = _RHO_cgs * _V_cgs ** 2       # code pressure/energy-density unit [erg/cm^3]

LAMBDA_CENTERS = lambda_cool(TEMP_BIN_CENTERS, mask=True)   # (BINS,), erg cm^3 / s


def coarse_block_mean(field: np.ndarray, downsample: int) -> np.ndarray:
    """Block-average a (H, W) field by an integer factor along both axes."""
    h, w = field.shape
    nbx, nby = h // downsample, w // downsample
    blocks = field[: nbx * downsample, : nby * downsample].reshape(nbx, downsample, nby, downsample)
    return blocks.mean(axis=(1, 3))


def coarse_temp_pdf(temp: np.ndarray, downsample: int) -> np.ndarray:
    """Vectorized per-block temperature PDF. Returns (BINS, nbx, nby)."""
    h, w = temp.shape
    nbx, nby = h // downsample, w // downsample
    blocks = (
        temp[: nbx * downsample, : nby * downsample]
        .reshape(nbx, downsample, nby, downsample)
        .transpose(0, 2, 1, 3)
        .reshape(nbx * nby, downsample * downsample)
    )

    bin_idx = np.clip(np.digitize(blocks, TEMP_BIN_EDGES) - 1, 0, BINS - 1)
    flat_idx = np.arange(nbx * nby)[:, None] * BINS + bin_idx
    counts = np.bincount(flat_idx.ravel(), minlength=nbx * nby * BINS).reshape(nbx * nby, BINS)
    pdf = counts / counts.sum(axis=1, keepdims=True)
    return pdf.reshape(nbx, nby, BINS).transpose(2, 0, 1)


def emissivity_and_tcool(pdf: np.ndarray, rho_cg: np.ndarray, pressure_cg: np.ndarray):
    """pdf: (BINS, nbx, nby); rho_cg, pressure_cg: (nbx, nby) in code units."""
    n_phys = rho_cg * n_to_cm3 / mu                       # cm^-3
    p_cgs = pressure_cg * _P_cgs                          # erg/cm^3
    e_int_cgs = p_cgs / (gamma - 1.0)                     # erg/cm^3

    # Version 1: isochoric emissivity, constant n across the subgrid PDF
    eps1 = (n_phys ** 2) * np.sum(pdf * LAMBDA_CENTERS[:, None, None], axis=0)

    # Version 2: isobaric emissivity, n_i = P / (kB T_i) per bin
    weight2 = LAMBDA_CENTERS[:, None, None] / (TEMP_BIN_CENTERS[:, None, None] ** 2)
    eps2 = ((p_cgs / kb) ** 2) * np.sum(pdf * weight2, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_cool1 = np.where(eps1 > 0, e_int_cgs / eps1, np.nan) / _T_cgs   # Myr
        t_cool2 = np.where(eps2 > 0, e_int_cgs / eps2, np.nan) / _T_cgs   # Myr

    return t_cool1, t_cool2


if __name__ == "__main__":
    print(f"Loading fine-resolution cache from: {CACHE_DIR}")
    rho_full = np.load(os.path.join(CACHE_DIR, "rho.npy"), mmap_mode="r")
    temp_full = np.load(os.path.join(CACHE_DIR, "temp.npy"), mmap_mode="r")
    pressure_full = np.load(os.path.join(CACHE_DIR, "pressure.npy"), mmap_mode="r")

    nt = rho_full.shape[0]
    frame_idx = np.linspace(0, nt - 1, min(NUM_FRAMES, nt)).astype(int)
    print(f"Using {len(frame_idx)} of {nt} timesteps: {frame_idx[0]}..{frame_idx[-1]}")

    fig, axes = plt.subplots(1, len(TARGET_COARSE_RESOLUTIONS), figsize=(7 * len(TARGET_COARSE_RESOLUTIONS), 6), squeeze=False)
    axes = axes[0]

    for ax, (target_nx, target_ny) in zip(axes, TARGET_COARSE_RESOLUTIONS):
        downsample = FINE_RESOLUTION[0] // target_nx
        assert downsample == FINE_RESOLUTION[1] // target_ny, "Non-uniform downsample factor requested"
        print(f"\n=== Coarse resolution {target_nx}x{target_ny} (downsample={downsample}) ===")

        t_cg_all, t1_all, t2_all = [], [], []
        for t in frame_idx:
            temp_t = np.asarray(temp_full[t])
            rho_t = np.asarray(rho_full[t])
            pressure_t = np.asarray(pressure_full[t])

            temp_cg = coarse_block_mean(temp_t, downsample)
            rho_cg = coarse_block_mean(rho_t, downsample)
            pressure_cg = coarse_block_mean(pressure_t, downsample)
            pdf = coarse_temp_pdf(temp_t, downsample)

            t_cool1, t_cool2 = emissivity_and_tcool(pdf, rho_cg, pressure_cg)

            t_cg_all.append(temp_cg.ravel())
            t1_all.append(t_cool1.ravel())
            t2_all.append(t_cool2.ravel())

        t_cg_all = np.concatenate(t_cg_all)
        t1_all = np.concatenate(t1_all)
        t2_all = np.concatenate(t2_all)

        ax.scatter(t_cg_all, t1_all, s=4, alpha=0.3, label=r"V1: $n^2\sum_i \mathrm{pdf}(T_i)\Lambda(T_i)$", color="tab:blue")
        ax.scatter(t_cg_all, t2_all, s=4, alpha=0.3, label=r"V2: $(P/k_B)^2\sum_i \mathrm{pdf}(T_i)\Lambda(T_i)/T_i^2$", color="tab:red")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Coarse-grained Temperature $\bar{T}$ [K]")
        ax.set_ylabel(r"$t_{\rm cool}$ [Myr]")
        ax.set_title(f"Coarse resolution {target_nx}x{target_ny}")
        ax.legend(fontsize=9, markerscale=3)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "tcool_vs_temp_emissivity_versions.png")
    fig.savefig(out_path, dpi=200)
    print(f"\nSaved figure -> {out_path}")
    plt.show()
