# Python script for manually predicting the source term using different CNNs

import os
import random
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
    snapshot_pred_16x8,
    weight_decay,
)
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, sobel

seed = int(os.environ.get("GLOBAL_SEED", "10"))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
P_unit: float = 1.59916e-14
mu: float = 0.62
kb: float = 1.3807e-16
T_edges = np.logspace(3.0, 7.0, out_channels + 1)
T_centers = np.sqrt(T_edges[:-1] * T_edges[1:])
logT_centers = torch.log10(torch.tensor(T_centers, dtype=torch.float32))


def divergence(f, dx, dy):
    dFx_dx = np.gradient(f[0], dy, dx)[1]
    dFy_dy = np.gradient(f[1], dy, dx)[0]
    return dFx_dx + dFy_dy


# NOTE: lambda_cool and ConvNN are imported from models/conv_nn/pdf_cnn.py.


def source_func(rho, pres, ux, uy, ps, fmcl, bdt=None):

    global resolution, downsample
    global cnn_model, shape
    global input_mean, input_std
    global device
    global total_length, total_width
    global P_unit, mu, kb

    # ------------------------------------------------------------
    # Temperature (matches data_preprocess.py formula)
    # ------------------------------------------------------------

    temp = (np.array(pres) * P_unit / np.array(rho)) * (mu / kb)
    # ------------------------------------------------------------
    # Build input fields & allocate source term
    # ------------------------------------------------------------

    fields = ["rho", "temp", "ux", "uy", "ps"]

    cg = {}
    for field in fields:
        cg[f"cg_{field}"] = np.transpose(np.array(locals()[field]))

    shape = cg["cg_rho"].shape
    source_term = np.zeros((5, shape[0], shape[1]))

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
    # Cooling source term (Code Units)
    # ------------------------------------------------------------
    # Uses isobaric_emissivity_from_pdf from training: n^2 * \sum_i PDF(T_i) * Lambda(T_i) * unit_fix (in code units)
    lambda_tensor = torch.tensor(lambda_cool(T_centers, mask=True), dtype=torch.float32)
    pdf_tensor = torch.from_numpy(pdf).unsqueeze(0).float()
    rho_tensor = torch.from_numpy(cg["cg_rho"]).unsqueeze(0).unsqueeze(0).float()
    temp_tensor = torch.from_numpy(cg["cg_temp"]).unsqueeze(0).unsqueeze(0).float()
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
    cool_rate = emiss.squeeze().cpu().numpy()

    # Safety cap: bound energy cooling rate per cell relative to local thermal energy.
    # e_int = P / (gamma - 1).  The cap prevents removing more than 50% of a
    # cell's internal energy per timestep.  We use the actual Athena half-timestep
    # (bdt) passed from C++; fall back to a conservative default only if not provided.
    e_int = np.transpose(pres) / (gamma - 1.0)
    dt_cap = float(bdt) if (bdt is not None and float(bdt) > 0.0) else 0.001
    max_cool_rate = np.maximum(0.0, 0.5 * e_int / dt_cap)
    cool_rate = np.clip(cool_rate, 0.0, max_cool_rate)

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

    #     A = gaussian_filter(v, 3)

    #     B = gaussian_filter(v, kernel_size / 3)

    #     source_term[channel] = (1 - w) * A + w * B

    # ------------------------------------------------------------
    # Return shape
    # ------------------------------------------------------------

    final_term = np.transpose(source_term, axes=(0, 2, 1))

    return final_term.reshape(5, -1)
