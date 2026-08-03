# Python script for manually predicting the source term using different CNNs

import os
import sys
import types

# ---------------------------------------------------------------------------
# Import ConvNN and lambda_cool directly from pdf_cnn.py.
#
# pdf_cnn.py imports data_preprocess which in turn imports bin_convert (a C
# extension that is not available in the build tree).  We stub out bin_convert
# in sys.modules before the import so Python never tries to load the real
# extension, while still allowing all the model/physics code to be imported.
# ---------------------------------------------------------------------------

_PDF_CNN_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../models/conv_nn")
)
_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data")))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_SAVE_DIR = os.environ.get(
    "MODEL_SAVES_DIR",
    os.path.join(PROJECT_ROOT, "outputs", "model_saves", "pdf_model_saves"),
)
LOSS_PLOT_DIR = os.environ.get(
    "LOSS_PLOTS_DIR",
    os.path.join(PROJECT_ROOT, "outputs", "loss_plots", "pdf_loss_plots"),
)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOSS_PLOT_DIR, exist_ok=True)
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
    batch_size,
    device,
    dropout_rate,
    in_channels,
    kernel_size,
    lambda_cool,
    layer_size1,
    layer_size2,
    layer_size3,
    layer_size4,
    learning_rate,
    out_channels,
    snapshot_pred_16x8,
    weight_decay,
)
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, sobel

np.random.seed(10)
_res_str = os.environ.get("PDF_CNN_RESOLUTION", "1024,512").split(",")
resolution = (int(_res_str[0]), int(_res_str[1]))
downsample  = int(os.environ.get("PDF_CNN_DOWNSAMPLE", "64"))
# resolution is (1024, 512) = (nx2, nx1) = (rows, cols)
shape = (resolution[0] // downsample, resolution[1] // downsample)
layer_size4 = 256
layer_size5 = 512
total_length: float = 20
total_width: float = 10
gamma: float = 5.0 / 3.0
T_edges = np.logspace(3.0, 7.0, out_channels + 1)
T_centers = np.sqrt(T_edges[:-1] * T_edges[1:])
logT_centers = torch.log10(torch.tensor(T_centers, dtype=torch.float32))


def divergence(f, dx, dy):
    dFx_dx = np.gradient(f[0], dy, dx)[1]
    dFy_dy = np.gradient(f[1], dy, dx)[0]
    return dFx_dx + dFy_dy


# NOTE: lambda_cool and ConvNN are imported from models/conv_nn/pdf_cnn.py.


def source_func(rho, pres, ux, uy, ps, fmcl):

    global resolution, downsample
    global cnn_model, shape
    global input_mean, input_std
    global device
    global total_length, total_width

    # ------------------------------------------------------------
    # Temperature
    # ------------------------------------------------------------

    temp = (np.array(pres) * 1.59916e-14 / np.array(rho)) * (1.0 / 1.381e-16)
    # ------------------------------------------------------------
    # Allocate source term
    # ------------------------------------------------------------

    source_term = np.zeros((5, shape[0], shape[1]))

    # ------------------------------------------------------------
    # Build input fields
    # ------------------------------------------------------------

    fields = ["rho", "temp", "ux", "uy", "ps"]

    shape = (resolution[0] // downsample, resolution[1] // downsample)

    cg = {f"cg_{field}": np.zeros(shape) for field in fields}

    for field in fields:
        cg[f"cg_{field}"] = np.transpose(np.array(locals()[field]))

    pdf = snapshot_pred_16x8(
        rho=cg["cg_rho"],
        temp=cg["cg_temp"],
        ux=cg["cg_ux"],
        uy=cg["cg_uy"],
        ps=cg["cg_ps"],
        fine_resolution=resolution,
        downsample=downsample,
    )

    # # Testing the generalizability
    # pdf = snapshot_pred_16x8(
    #     rho=cg["cg_rho"][:, ::-1],
    #     temp=cg["cg_temp"][:, ::-1],
    #     ux=cg["cg_ux"][:, ::-1],
    #     uy=cg["cg_uy"][:, ::-1],
    #     ps=cg["cg_ps"][:, ::-1],
    #     fine_resolution=resolution,
    #     downsample=downsample,
    # )

    # ------------------------------------------------------------
    # Cell sizes
    # ------------------------------------------------------------

    if rho.shape[1] > rho.shape[0]:
        dy = total_length / rho.shape[1]
        dx = total_width / rho.shape[0]

    else:
        dy = total_length / rho.shape[0]
        dx = total_width / rho.shape[1]

    # ------------------------------------------------------------
    # Cooling source term (Corrected for Code Units + Isobaric)
    # ------------------------------------------------------------
    # Physics constants matching your simulation scales
    T_unit = 115.797      # derived from (1.59916e-14 / 1.381e-16)
    mu = 0.62             # mean molecular weight
    unit_fix = 1.975e27   # grouped conversion factor for code units

    nb = out_channels
    temp_bins = np.logspace(3, 7, nb + 1)
    temp_centers = np.sqrt(temp_bins[:-1] * temp_bins[1:])
    T = temp_centers[:, None, None]

    # Pressure is in Code Units
    P_code = np.transpose(pres)[None, :, :]

    # 1. Isobaric Density Reconstruction (IN CODE UNITS)
    rho_per_bin = P_code * (T_unit / T)
    n_code_per_bin = rho_per_bin / mu

    # 2. Emissivity per bin (converted back to Code Units)
    lam = lambda_cool(T, mask=True)
    cool_per_bin = lam * (n_code_per_bin**2) * unit_fix

    # 3. Integrate across the predicted subgrid PDF
    cool_rate = np.sum(pdf * cool_per_bin, axis=0)
    #cool_rate = cool_rate[:, ::-1]

    # energy source term
    source_term[3] = -cool_rate

    # # ------------------------------------------------------------
    # # Adaptive smoothing
    # # ------------------------------------------------------------

    # for channel in range(3, 4):
    #     v = source_term[channel]

    #     w = np.clip(
    #         (np.abs(v) - np.percentile(np.abs(v), 75))
    #         / (np.percentile(np.abs(v), 90) - np.percentile(np.abs(v), 75) + 1e-12),
    #         0,
    #         1,
    #     )

    #     A = gaussian_filter(v, 0.0)

    #     B = gaussian_filter(v, kernel_size / 3)

    #     source_term[channel] = (1 - w) * A + w * B

    # ------------------------------------------------------------
    # Return shape
    # ------------------------------------------------------------

    final_term = np.transpose(source_term, axes=(0, 2, 1))

    return final_term.reshape(5, -1)
