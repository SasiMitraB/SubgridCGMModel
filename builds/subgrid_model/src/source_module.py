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

import csv
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

# ---------------------------------------------------------------------------
# Cooling-rate clip tracking
#
# The temperature-floor cap in source_func() (see below) is a safety valve,
# not something that should be triggering constantly. We log every clipped
# cell -- sim time, grid position, physical position, and the local field
# values -- to a CSV so a run can be audited after the fact for how often
# and where the cap is actually engaging.
#
# There is no simulation clock available here (UserSourceTerm only passes
# bdt, not pm->time), so elapsed time is reconstructed by accumulating bdt
# across calls; this is exact as long as a multi-stage integrator's per-stage
# bdt values sum to the full timestep, which is the case for AthenaK's
# integrators. CLIP_LOG_START_TIME lets this be offset to match a restart.
#
# This assumes a single-process run (no MPI), matching the subgrid run
# scripts in shell_scripts/ -- concurrent writers would need file locking.
# ---------------------------------------------------------------------------
CLIP_LOG_PATH = os.environ.get(
    "CLIP_LOG_PATH", os.path.join(PROJECT_ROOT, "outputs", "clip_events.csv")
)
_clip_log_header_written = False
_sim_time_accum = float(os.environ.get("CLIP_LOG_START_TIME", "0.0"))

_CLIP_LOG_FIELDS = [
    "time", "row", "col", "x2_pc", "x1_pc",
    "rho", "temp_K", "cool_rate_raw", "cool_max", "bdt",
]


def _log_clip_events(time_now, bdt, rows, cols, x2min, dx2, x1min, dx1,
                      rho_hw, temp_hw, cool_raw_hw, cool_max_hw):
    """Append one CSV row per clipped cell to CLIP_LOG_PATH."""
    global _clip_log_header_written

    os.makedirs(os.path.dirname(CLIP_LOG_PATH), exist_ok=True)
    write_header = not _clip_log_header_written and not os.path.isfile(CLIP_LOG_PATH)

    with open(CLIP_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(_CLIP_LOG_FIELDS)
        for r, c in zip(rows, cols):
            writer.writerow([
                time_now,
                int(r), int(c),
                x2min + (r + 0.5) * dx2,
                x1min + (c + 0.5) * dx1,
                float(rho_hw[r, c]),
                float(temp_hw[r, c]),
                float(cool_raw_hw[r, c]),
                float(cool_max_hw[r, c]),
                bdt,
            ])
    _clip_log_header_written = True


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
        ((1024, 512), 32),
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
        "pres": pres_arr.T.copy(),  # code-unit gas pressure, for isobaric cooling only
    }

    H, W = cg["rho"].shape  # e.g. (32, 16)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 2. CNN Inference: Single tile vs Tiled grid
    #    Controlled via CNN_TILING_MODE ("single" vs "tiled")
    #    and TILE_GRID ("n_rows,n_cols", e.g. "4,4")
    # ------------------------------------------------------------------
    fields = [cg["rho"], cg["temp"], cg["ux"], cg["uy"], cg["ps"]]
    tiling_mode = os.environ.get("CNN_TILING_MODE", "single").strip().lower()

    if tiling_mode == "tiled":
        grid_str = os.environ.get("TILE_GRID", "4,4").split(",")
        n_tile_rows = int(grid_str[0].strip())
        n_tile_cols = int(grid_str[1].strip()) if len(grid_str) > 1 else n_tile_rows

        if H % n_tile_rows != 0 or W % n_tile_cols != 0:
            raise ValueError(
                f"Grid shape ({H}, {W}) is not divisible by TILE_GRID "
                f"({n_tile_rows}, {n_tile_cols})."
            )

        tile_h = H // n_tile_rows
        tile_w = W // n_tile_cols

        tiles = []
        coords = []
        for ti in range(n_tile_rows):
            for tj in range(n_tile_cols):
                r0, r1 = ti * tile_h, (ti + 1) * tile_h
                c0, c1 = tj * tile_w, (tj + 1) * tile_w
                tiles.append(np.stack([f[r0:r1, c0:c1] for f in fields], axis=0))
                coords.append((r0, r1, c0, c1))

        batch = torch.from_numpy(np.stack(tiles, axis=0)).float().to(device)  # (N_tiles, 5, tile_h, tile_w)
        batch_norm = (batch - _input_mean_cache) / (_input_std_cache + 1e-8)

        with torch.no_grad():
            logits, gate = _model_cache(batch_norm)
            pdf_tiles = _model_cache.pdf_activation(logits, gate).cpu().numpy()  # (N_tiles, 40, tile_h, tile_w)

        full_pdf = np.zeros((out_channels, H, W), dtype=np.float32)  # (40, H, W)
        for idx, (r0, r1, c0, c1) in enumerate(coords):
            full_pdf[:, r0:r1, c0:c1] = pdf_tiles[idx]

    else:
        # Default / "single": pass entire coarse snapshot directly through CNN
        single_tile = np.stack(fields, axis=0)  # (5, H, W)
        batch = torch.from_numpy(single_tile).unsqueeze(0).float().to(device)  # (1, 5, H, W)
        batch_norm = (batch - _input_mean_cache) / (_input_std_cache + 1e-8)

        with torch.no_grad():
            logits, gate = _model_cache(batch_norm)
            full_pdf = _model_cache.pdf_activation(logits, gate).cpu().numpy()[0]  # (40, H, W)

    # ------------------------------------------------------------------
    # 4. Compute cooling rate from stitched PDFs
    #    Isobaric per-bin density n_i = P/(kB T_i):
    #    emissivity = (P/kB)^2 x Sum_i PDF(T_i) Lambda(T_i) / T_i^2   (code units)
    # ------------------------------------------------------------------
    lambda_tensor = torch.tensor(
        lambda_cool(T_centers, mask=True), dtype=torch.float32
    )
    pdf_tensor  = torch.from_numpy(full_pdf).unsqueeze(0).float()   # (1, 40, H, W)
    pres_tensor = torch.from_numpy(cg["pres"].astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    t_centers_tensor = torch.tensor(T_centers, dtype=torch.float32)

    emiss = isobaric_emissivity_from_pdf(
        pdf=pdf_tensor,
        pressure=pres_tensor,
        T_centers_tensor=t_centers_tensor,
        lambda_tensor=lambda_tensor,
    )
    cool_rate = emiss.squeeze().cpu().numpy()  # (H, W)
    # cool_rate is a pure sink by construction (see isobaric_emissivity_from_pdf
    # in pdf_cnn.py: emiss = (P/kB)^2 * sum(pdf * lambda / T^2) * unit_fix, and
    # both the PDF weights and Lambda(T) are non-negative) -- there is no heating
    # case to guard against. However, it is NOT bounded in magnitude: a
    # dense, actively-cooling cell in a TRML can have a genuinely very
    # short local cooling time. Applying that raw rate unconditionally for
    # one substep of size bdt could overshoot straight through any
    # physical temperature floor, and (worse) feeding the raw rate into
    # dt_cool below would peg the timestep to that one cell's cooling
    # time forever: dt shrinks, the state barely evolves each step, so
    # the model keeps seeing ~the same input and keeps predicting the
    # same huge rate -- an explicit-integration deadlock, not a crash.
    #
    # Standard fix (same idea as radiative-cooling papers this model is
    # based on): cap the sink so this stage's update can cool a cell to
    # the floor temperature but no further. A capped cell's own dt_cool
    # then comes out to ~bdt (see below), so it stops driving the global
    # timestep down without limit; uncapped cells still constrain dt_cool
    # normally.
    # ------------------------------------------------------------------
    global _sim_time_accum
    e_curr = pres_arr.T / (gamma - 1.0)  # (H, W), code-unit internal energy
    T_TARGET = float(os.environ.get("COOL_TFLOOR", "1.0e4"))  # K
    if bdt is not None and float(bdt) > 0.0 and T_TARGET > 0.0:
        p_floor = (rho_arr.T * kb * T_TARGET) / (mu * P_unit)
        e_floor = p_floor / (gamma - 1.0)
        cool_max = np.maximum(e_curr - e_floor, 0.0) / float(bdt)

        clip_mask = cool_rate > cool_max
        if np.any(clip_mask):
            rows, cols = np.nonzero(clip_mask)
            dx2 = total_length / H
            dx1 = total_width / W
            _log_clip_events(
                _sim_time_accum, float(bdt), rows, cols,
                -total_length / 2.0, dx2, -total_width / 2.0, dx1,
                cg["rho"], cg["temp"], cool_rate, cool_max,
            )

        cool_rate = np.minimum(cool_rate, cool_max)

    if bdt is not None:
        _sim_time_accum += float(bdt)

    # ------------------------------------------------------------------
    # 5. Build source term array
    # ------------------------------------------------------------------
    source_term = np.zeros((5, H, W), dtype=np.float64)
    source_term[3] = -cool_rate  # energy sink

    # ------------------------------------------------------------------
    # 5b. Cooling timestep: dt_cool = min_cell( E_int / cool_rate ), using
    #     the same (capped) cool_rate that was actually applied above, so
    #     a capped cell reports back ~bdt (the step just taken) instead of
    #     an unboundedly small number.
    # ------------------------------------------------------------------
    dt_cool = float(np.min(e_curr / (cool_rate + 1e-30)))

    # ------------------------------------------------------------------
    # 6. Return shape expected by subgrid.cpp:
    #    subgrid.cpp calls: S_arr.unchecked<2>() with shape (5, nmb*Ni*Nj)
    #    Our source_term is (5, H, W) in Python (rows, cols) layout.
    #    Transpose back to C++ (Ni, Nj) = (cols, rows) layout before flatten.
    # ------------------------------------------------------------------
    # source_term[ch] has shape (H, W) = (rows, cols) = (nx2, nx1)
    # C++ expects layout (nx1, nx2) — i.e. transposed.
    final_term = np.transpose(source_term, axes=(0, 2, 1))  # (5, W, H) = (5, nx1, nx2)
    return final_term.reshape(5, -1), dt_cool
