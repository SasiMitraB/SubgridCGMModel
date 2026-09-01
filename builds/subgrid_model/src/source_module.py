# Python script for predicting the source term using pdf_cnn.py with tiled inference.
# Supports any grid that is a multiple of the tile size (16 rows × 8 cols).
# The CNN (trained on 16×8 crops) is applied to each tile independently and
# the resulting PDFs are stitched back into the full-domain cooling rate.
#
# Key design decisions:
#  - Model + normalization stats are loaded ONCE (lazy globals) to avoid
#    per-timestep I/O overhead.
#  - The tile size is the CNN's native coarse-grid size: 16 rows × 8 cols.
#  - Hard-cut tiling (no overlap blending) is used for simplicity.

import os
import sys
import types

# ---------------------------------------------------------------------------
# Import ConvNN and helpers from pdf_cnn.py.
#
# pdf_cnn.py imports data_preprocess which in turn imports bin_convert (a C
# extension not available in the build tree).  We stub out bin_convert before
# the import so Python never tries to load the real extension.
# ---------------------------------------------------------------------------

_PDF_CNN_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../models/conv_nn")
)
_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DEFAULT_MODEL_SAVE_DIR = os.path.join(
    PROJECT_ROOT, "outputs", "model_saves", "pdf_model_saves"
)
MODEL_SAVE_DIR = os.environ.get("MODEL_SAVES_DIR", DEFAULT_MODEL_SAVE_DIR)
for _p in (_PDF_CNN_DIR, _DATA_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Stub bin_convert so data_preprocess imports cleanly in the build context.
if "bin_convert" not in sys.modules:
    sys.modules["bin_convert"] = types.ModuleType("bin_convert")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdf_cnn import (
    ConvNN,
    compute_cooling_rate,
    device,
    dropout_rate,
    in_channels,
    isobaric_emissivity_from_pdf,
    kernel_size,
    lambda_cool,
    layer_size1,
    layer_size2,
    layer_size3,
    layer_size4,
    learning_rate,
    out_channels,
    weight_decay,
)

# ---------------------------------------------------------------------------
# Configuration (driven by env vars, matching the training pipeline)
# ---------------------------------------------------------------------------
import random

seed = int(os.environ.get("GLOBAL_SEED", "10"))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

_res_str = os.environ.get("PDF_CNN_RESOLUTION", "1024,512").split(",")
# resolution is (nx2, nx1) = (rows, cols) of the FINE grid
resolution = (int(_res_str[0]), int(_res_str[1]))
downsample = int(os.environ.get("PDF_CNN_DOWNSAMPLE", "64"))

# Tile size: the coarse-grid size the model was trained on
# Default tile size (e.g., 16 rows × 8 cols or 32 rows × 16 cols)
TILE_ROWS = int(os.environ.get("TILE_ROWS", os.environ.get("CROP_H_CG", str(resolution[0] // downsample))))
TILE_COLS = int(os.environ.get("TILE_COLS", os.environ.get("CROP_W_CG", str(resolution[1] // downsample))))

layer_size4 = 256
layer_size5 = 512

# Domain extents (code units) — must match the athinput
# x2 ∈ [-20, 20] (2× length domain, matching vshear_31_coldfrac_0.33)
total_length: float = 40.0   # |x2max - x2min| = 20 - (-20)
total_width: float = 20.0    # |x1max - x1min| = 10 - (-10)

gamma: float = 5.0 / 3.0
P_unit: float = 1.59916e-14   # pressure unit in dyne/cm^2 per code pressure
mu: float = 0.62
kb: float = 1.3807e-16         # Boltzmann constant in erg/K

T_edges = np.logspace(3.0, 7.0, out_channels + 1)
T_centers = np.sqrt(T_edges[:-1] * T_edges[1:])

# ---------------------------------------------------------------------------
# Global model cache — loaded once, reused every timestep
# ---------------------------------------------------------------------------
_model_cache = None  # ConvNN instance
_input_mean_cache = None  # torch.Tensor, shape (1, 5, 1, 1)
_input_std_cache = None   # torch.Tensor, shape (1, 5, 1, 1)


def _find_available_model(save_dir: str):
    """
    Find the first available (resolution, downsample) model configuration.
    Checks user-specified env vars first if present, then candidate defaults:
      1. resolution=(1024, 512), downsample=64
      2. resolution=(512, 256), downsample=32
    """
    candidates = []

    if "PDF_CNN_RESOLUTION" in os.environ and "PDF_CNN_DOWNSAMPLE" in os.environ:
        _res_parts = os.environ["PDF_CNN_RESOLUTION"].split(",")
        cand_env = ((int(_res_parts[0]), int(_res_parts[1])), int(os.environ["PDF_CNN_DOWNSAMPLE"]))
        candidates.append(cand_env)

    default_candidates = [
        ((2048, 1024), 32),
        ((1024, 512), 64),
        ((512, 256), 32),
    ]
    for cand in default_candidates:
        if cand not in candidates:
            candidates.append(cand)

    for res, ds in candidates:
        norm_prefix = f"cnn_{res}_{ds}"
        model_path = os.path.join(save_dir, f"{norm_prefix}.pth")
        mean_path = os.path.join(save_dir, f"{norm_prefix}_input_mean.npy")
        std_path = os.path.join(save_dir, f"{norm_prefix}_input_std.npy")

        if os.path.isfile(model_path) and os.path.isfile(mean_path) and os.path.isfile(std_path):
            return res, ds, norm_prefix, model_path, mean_path, std_path

    checked_str = ", ".join([f"cnn_{res}_{ds}" for res, ds in candidates])
    raise FileNotFoundError(
        f"No suitable CNN model found in {save_dir}. Checked configurations: {checked_str}"
    )


def _load_model():
    """Lazy-load the CNN model and normalization statistics into global cache."""
    global _model_cache, _input_mean_cache, _input_std_cache
    global resolution, downsample, TILE_ROWS, TILE_COLS

    if _model_cache is not None:
        return  # Already loaded

    save_dir = os.environ.get("MODEL_SAVES_DIR", MODEL_SAVE_DIR)
    res, ds, norm_prefix, model_path, mean_path, std_path = _find_available_model(save_dir)

    resolution = res
    downsample = ds
    TILE_ROWS = int(os.environ.get("TILE_ROWS", os.environ.get("CROP_H_CG", str(resolution[0] // downsample))))
    TILE_COLS = int(os.environ.get("TILE_COLS", os.environ.get("CROP_W_CG", str(resolution[1] // downsample))))

    # --- Load normalization stats ---
    mean_arr = np.load(mean_path)
    std_arr  = np.load(std_path)

    _input_mean_cache = torch.tensor(mean_arr, dtype=torch.float32).to(device)
    _input_std_cache  = torch.tensor(std_arr,  dtype=torch.float32).to(device)

    # Ensure shape (1, C, 1, 1) for broadcasting
    if _input_mean_cache.dim() == 1:
        _input_mean_cache = _input_mean_cache.view(1, -1, 1, 1)
        _input_std_cache  = _input_std_cache.view(1, -1, 1, 1)

    # --- Load model weights ---
    state_dict = torch.load(model_path, map_location=device)

    ckpt_ksize = kernel_size
    if "encoder.0.weight" in state_dict:
        ckpt_ksize = state_dict["encoder.0.weight"].shape[-1]

    _model_cache = ConvNN(
        in_channels, layer_size1, layer_size2, layer_size3, layer_size4,
        out_channels, ckpt_ksize,
    ).to(device)
    _model_cache.load_state_dict(state_dict)
    _model_cache.eval()

    print(
        f"[source_module] Loaded CNN model from {model_path}",
        flush=True,
    )
    print(
        f"[source_module] Tile size: {TILE_ROWS}×{TILE_COLS} "
        f"(resolution={resolution}, downsample={downsample})",
        flush=True,
    )


def _predict_tile(rho_tile, temp_tile, ux_tile, uy_tile, ps_tile):
    """
    Run a single 16×8 tile through the cached CNN.

    Parameters
    ----------
    rho_tile, temp_tile, ux_tile, uy_tile, ps_tile : np.ndarray, shape (16, 8)
        Already coarse-grained field tiles.

    Returns
    -------
    pdf_tile : np.ndarray, shape (40, 16, 8)
    """
    _load_model()

    stack = np.stack(
        [rho_tile, temp_tile, ux_tile, uy_tile, ps_tile], axis=0
    ).astype(np.float32)                              # (5, 16, 8)

    x = torch.from_numpy(stack).unsqueeze(0).to(device)  # (1, 5, 16, 8)
    x = (x - _input_mean_cache) / (_input_std_cache + 1e-8)

    with torch.no_grad():
        pdf = _model_cache.predict_pdf(x)  # (1, 40, 16, 8)
        pdf = pdf[0].cpu().numpy()          # (40, 16, 8)

    return pdf


def source_func(rho, pres, ux, uy, ps, fmcl, bdt=None):
    """
    Compute subgrid source terms for a 32×16 (or any multiple of 16×8) grid.

    Called from C++ UserSourceTerm via pybind11.  Inputs arrive as 2-D numpy
    arrays of shape (nmb*Ni, Nj), where for a single MeshBlock at 32×16:
        rho.shape == pres.shape == ... == (32, 16)

    Tiling strategy
    ---------------
    The CNN was trained on 16×8 coarse tiles.  For a 32×16 grid we split into
    a 2×2 grid of 16×8 tiles, run inference on each, then stitch the PDFs.

    Returns
    -------
    source_terms : np.ndarray, shape (5, nmb*Ni*Nj)
        Row 0: mass source (zero)
        Row 1: x-momentum source (zero)
        Row 2: y-momentum source (zero)
        Row 3: energy source = −cooling_rate
        Row 4: fmcl source (zero)
    """
    # ------------------------------------------------------------------
    # 0. Ensure model is loaded (and tile dimensions set)
    # ------------------------------------------------------------------
    _load_model()

    # ------------------------------------------------------------------
    # 1. Compute temperature from code-unit density and pressure
    # ------------------------------------------------------------------
    rho_arr  = np.asarray(rho,  dtype=np.float64)
    pres_arr = np.asarray(pres, dtype=np.float64)
    ux_arr   = np.asarray(ux,   dtype=np.float64)
    uy_arr   = np.asarray(uy,   dtype=np.float64)
    ps_arr   = np.asarray(ps,   dtype=np.float64)

    # T [K] = (P * P_unit / rho) * (mu / k_B)
    # Note: pres from AthenaK is internal energy density = P/(gamma-1),
    # so we recover P = pres * (gamma - 1).  However, subgrid.cpp already
    # passes w0(IEN)*gm1 (i.e. P directly) as the pres argument.
    temp_arr = (pres_arr * P_unit / np.maximum(rho_arr, 1e-30)) * (mu / kb)

    # ------------------------------------------------------------------
    # 1. Transpose from C++ (Ni, Nj) layout to Python (rows, cols)
    #    C++ passes arrays as (nmb*Ni, Nj) = (nx1_total, nx2_total)
    #    We want shape (nx2_total, nx1_total) = (rows, cols) for tiling.
    # ------------------------------------------------------------------
    cg = {
        "rho":  rho_arr.T.copy(),   # (rows, cols)
        "temp": temp_arr.T.copy(),
        "ux":   ux_arr.T.copy(),
        "uy":   uy_arr.T.copy(),
        "ps":   ps_arr.T.copy(),
    }

    H, W = cg["rho"].shape  # e.g. (32, 16)

    # ------------------------------------------------------------------
    # 2. Validate that the grid is a multiple of the tile size
    # ------------------------------------------------------------------
    if H % TILE_ROWS != 0 or W % TILE_COLS != 0:
        raise ValueError(
            f"Grid shape ({H}, {W}) is not a multiple of tile size "
            f"({TILE_ROWS}, {TILE_COLS}).  Check PDF_CNN_RESOLUTION and "
            f"PDF_CNN_DOWNSAMPLE env vars."
        )

    n_tile_rows = H // TILE_ROWS
    n_tile_cols = W // TILE_COLS

    # ------------------------------------------------------------------
    # 3. Tiled inference: run CNN on each tile, stitch PDFs together
    # ------------------------------------------------------------------
    full_pdf = np.zeros((out_channels, H, W), dtype=np.float32)  # (40, H, W)

    for ti in range(n_tile_rows):
        for tj in range(n_tile_cols):
            r0, r1 = ti * TILE_ROWS, (ti + 1) * TILE_ROWS
            c0, c1 = tj * TILE_COLS, (tj + 1) * TILE_COLS

            tile_pdf = _predict_tile(
                rho_tile  = cg["rho" ][r0:r1, c0:c1],
                temp_tile = cg["temp"][r0:r1, c0:c1],
                ux_tile   = cg["ux"  ][r0:r1, c0:c1],
                uy_tile   = cg["uy"  ][r0:r1, c0:c1],
                ps_tile   = cg["ps"  ][r0:r1, c0:c1],
            )
            full_pdf[:, r0:r1, c0:c1] = tile_pdf  # (40, tile_rows, tile_cols)

    # ------------------------------------------------------------------
    # 4. Compute cooling rate from stitched PDFs
    #    emissivity = n² × Σᵢ PDF(Tᵢ) Λ(Tᵢ) × unit_fix   (code units)
    # ------------------------------------------------------------------
    lambda_tensor = torch.tensor(
        lambda_cool(T_centers, mask=True), dtype=torch.float32
    )
    pdf_tensor  = torch.from_numpy(full_pdf).unsqueeze(0).float()   # (1, 40, H, W)
    rho_tensor  = torch.from_numpy(cg["rho"].astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    temp_tensor = torch.from_numpy(cg["temp"].astype(np.float32)).unsqueeze(0).unsqueeze(0) # (1,1,H,W)
    t_centers_tensor = torch.tensor(T_centers, dtype=torch.float32)

    emiss = isobaric_emissivity_from_pdf(
        pdf=pdf_tensor,
        rho=rho_tensor,
        T_coarse=temp_tensor,
        T_centers_tensor=t_centers_tensor,
        lambda_tensor=lambda_tensor,
        mu=mu,
        unit_fix=1.975e27,
    )
    cool_rate = emiss.squeeze().cpu().numpy()  # (H, W)
    # ------------------------------------------------------------------
    # 4b. Temperature floor: clip the sink so applying
    #     source[3] = -cool_rate for one timestep cannot push the gas
    #     below T_TARGET (default 1e4 K; override via COOL_TFLOOR env var).
    #
    #     `pres` is code-unit pressure (subgrid.cpp passes w0(IEN)*gm1),
    #     so E_int = pres / (gamma - 1) in code units.
    #     The source returned is dE_int/dt, so:
    #         E_new = E_old - cool_rate * bdt
    #     We require E_new >= E_floor, with
    #         p_floor = rho * k_B * T_TARGET / (mu * P_unit)   (code units)
    #         E_floor = p_floor / (gamma - 1)
    #     =>  cool_rate <= (E_old - E_floor) / bdt
    #     Cells already at/below the floor get cool_max = 0 (no further
    #     cooling). Heating (cool_rate < 0) is left untouched.
    # ------------------------------------------------------------------
    T_TARGET = float(os.environ.get("COOL_TFLOOR", "1.0e4"))  # K
    if bdt is not None and float(bdt) > 0.0 and T_TARGET > 0.0:
        # Code-unit pressure corresponding to T_TARGET (same layout as rho_arr/pres_arr)
        p_floor = (rho_arr * kb * T_TARGET) / (mu * P_unit)

        e_curr  = pres_arr  / (gamma - 1.0)
        e_floor = p_floor   / (gamma - 1.0)

        # Max permissible cooling rate (per unit time) for this step.
        cool_max = np.maximum(e_curr - e_floor, 0.0) / float(bdt)  # (Ni, Nj) = (nx1, nx2)

        # cool_rate is in (H, W) = (rows, cols) = (nx2, nx1) layout;
        # transpose cool_max to match before clipping.
        cool_max_py = np.ascontiguousarray(cool_max.T)              # (H, W)

        # Clip the sink. np.minimum preserves negative (heating) values.
        cool_rate = np.minimum(cool_rate, cool_max_py)
    # ------------------------------------------------------------------
    # 5. Build source term array
    # ------------------------------------------------------------------
    
    source_term = np.zeros((5, H, W), dtype=np.float64)
    source_term[3] = -cool_rate  # energy sink

    # ------------------------------------------------------------------
    # 6. Return shape expected by subgrid.cpp:
    #    subgrid.cpp calls: S_arr.unchecked<2>() with shape (5, nmb*Ni*Nj)
    #    Our source_term is (5, H, W) in Python (rows, cols) layout.
    #    Transpose back to C++ (Ni, Nj) = (cols, rows) layout before flatten.
    # ------------------------------------------------------------------
    # source_term[ch] has shape (H, W) = (rows, cols) = (nx2, nx1)
    # C++ expects layout (nx1, nx2) — i.e. transposed.
    final_term = np.transpose(source_term, axes=(0, 2, 1))  # (5, W, H) = (5, nx1, nx2)
    return final_term.reshape(5, -1)
