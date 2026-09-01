# CNN to learn the PDF using discrete bins
# Physical Intuition of what's going on here

# Fine simulation                    Coarse simulation
# ┌──────────────┐                  ┌──────────┐
# │ T varies at  │   coarse-grain   │ 1 T per  │
# │ pixel level  │ ──────────────►  │ cell     │
# │ → rich PDF   │                  │ → lost   │
# └──────────────┘                  └──────────┘
#                                          │
#                             CNN predicts │
#                                          ▼
#                                   ┌──────────┐
#                                   │ PDF(T)   │
#                                   │ 40 bins  │
#                                   └────┬─────┘
#                                        │
#                      ρ² × Σ p(Tᵢ)Λ(Tᵢ) │
#                                        ▼
#                                   ┌──────────┐
#                                   │ Emissivity│
#                                   │ (cooling) │
#                                   └──────────┘

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch

torch.cuda.empty_cache()
# pyrefly: ignore [missing-import]
import os
import sys

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

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
# TODO: Set these to your local simulation data directories
DATA_PATH = os.environ.get("SUBGRID_DATA_PATH", "/path/to/simulation/bin")
CACHE_PATH = os.environ.get("SUBGRID_CACHE_PATH", "/path/to/cache")
import data_preprocess
from data_preprocess import simulation_data

# =========================
# TRAINING / MODEL HYPERPARAMETERS
# =========================
DEFAULT_FINE_RESOLUTION = (1024, 512)
DEFAULT_DOWNSAMPLE = 32


def _parse_resolution(value: str, default: tuple[int, int]) -> tuple[int, int]:
    try:
        width_str, height_str = value.split(",")
        return int(width_str.strip()), int(height_str.strip())
    except (ValueError, AttributeError):
        return default


HYPERPARAMS = {
    "seed": 10,
    "device": "mps",
    "resolution": _parse_resolution(
        os.environ.get("PDF_CNN_RESOLUTION", "1024,512"), DEFAULT_FINE_RESOLUTION
    ),
    "downsample": int(os.environ.get("PDF_CNN_DOWNSAMPLE", str(DEFAULT_DOWNSAMPLE))),
    "in_channels": 5,
    "out_channels": 40,
    "layer_size1": 32,
    "layer_size2": 64,
    "layer_size3": 128,
    "layer_size4": 256,
    "kernel_size": 5,
    "num_epochs": 1000,
    "print_every": 50,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "dropout_rate": 0.2,
    "alpha_gate": float(os.environ.get("PDF_CNN_ALPHA_GATE", "0.0")),
    "alpha_mean_temp": 10,
    "alpha_emiss": float(os.environ.get("PDF_CNN_ALPHA_EMISS", "10.0")),
    "alpha_leak": float(os.environ.get("PDF_CNN_ALPHA_LEAK", "10.0")),
    "alpha_active_wasserstein": float(
        os.environ.get(
            "PDF_CNN_ALPHA_ACTIVE_WASSERSTEIN",
            os.environ.get("PDF_CNN_ALPHA_ACTIVE_KL", "100.0"),
        )
    ),
    "alpha_inactive_wasserstein": float(
        os.environ.get(
            "PDF_CNN_ALPHA_INACTIVE_WASSERSTEIN",
            os.environ.get("PDF_CNN_ALPHA_INACTIVE_KL", "100.0"),
        )
    ),
    "alpha_active_kl": float(
        os.environ.get(
            "PDF_CNN_ALPHA_ACTIVE_WASSERSTEIN",
            os.environ.get("PDF_CNN_ALPHA_ACTIVE_KL", "100.0"),
        )
    ),
    "alpha_inactive_kl": float(
        os.environ.get(
            "PDF_CNN_ALPHA_INACTIVE_WASSERSTEIN",
            os.environ.get("PDF_CNN_ALPHA_INACTIVE_KL", "10.0"),
        )
    ),
    "gate_epochs": int(os.environ.get("PDF_CNN_GATE_EPOCHS", "200")),
    "gate_learning_rate": float(os.environ.get("PDF_CNN_GATE_LR", "1e-3")),
    "freeze_gate": True,
    "train_fraction": 0.50,
    "val_fraction": 0.25,
    "grad_clip_max_norm": 1.0,
}

np.random.seed(HYPERPARAMS["seed"])
torch.manual_seed(HYPERPARAMS["seed"])
if torch.cuda.is_available():   
    torch.cuda.manual_seed_all(HYPERPARAMS["seed"])

# Set PyTorch device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

resolution = HYPERPARAMS["resolution"]
downsample = HYPERPARAMS["downsample"]
in_channels = HYPERPARAMS["in_channels"]
out_channels = HYPERPARAMS["out_channels"]
layer_size1 = HYPERPARAMS["layer_size1"]
layer_size2 = HYPERPARAMS["layer_size2"]
layer_size3 = HYPERPARAMS["layer_size3"]
layer_size4 = HYPERPARAMS["layer_size4"]
kernel_size = HYPERPARAMS["kernel_size"]
num_epochs = HYPERPARAMS["num_epochs"]
print_every = HYPERPARAMS["print_every"]
batch_size = HYPERPARAMS["batch_size"]
learning_rate = HYPERPARAMS["learning_rate"]
weight_decay = HYPERPARAMS["weight_decay"]
dropout_rate = HYPERPARAMS["dropout_rate"]

T_edges = np.logspace(3.0, 7.0, out_channels + 1)
T_centers = np.sqrt(T_edges[:-1] * T_edges[1:])

logT_centers = torch.log10(torch.tensor(T_centers, dtype=torch.float32))

LOGT_ACTIVE_START = float(os.environ.get("LOGT_ACTIVE_START", "4.021189299069938"))
LOGT_ACTIVE_END = float(os.environ.get("LOGT_ACTIVE_END", "5.977723605288848"))


# New Version that truncates to 10^4.5 to 10^5.5
def lambda_cool(temp, mask=False, LOGT_ACTIVE_START=LOGT_ACTIVE_START, LOGT_ACTIVE_END=LOGT_ACTIVE_END):
    """
    Cooling function ISMCoolFn translated from AthenaK C++.
    Works on scalars or numpy arrays (any shape).
    Returns Λ(T) in erg cm^3 / s.
    """
    logt = np.log10(temp)

    lhd = np.array(
        [
            -22.5977,
            -21.9689,
            -21.5972,
            -21.4615,
            -21.4789,
            -21.5497,
            -21.6211,
            -21.6595,
            -21.6426,
            -21.5688,
            -21.4771,
            -21.3755,
            -21.2693,
            -21.1644,
            -21.0658,
            -20.9778,
            -20.8986,
            -20.8281,
            -20.7700,
            -20.7223,
            -20.6888,
            -20.6739,
            -20.6815,
            -20.7051,
            -20.7229,
            -20.7208,
            -20.7058,
            -20.6896,
            -20.6797,
            -20.6749,
            -20.6709,
            -20.6748,
            -20.7089,
            -20.8031,
            -20.9647,
            -21.1482,
            -21.2932,
            -21.3767,
            -21.4129,
            -21.4291,
            -21.4538,
            -21.5055,
            -21.5740,
            -21.6300,
            -21.6615,
            -21.6766,
            -21.6886,
            -21.7073,
            -21.7304,
            -21.7491,
            -21.7607,
            -21.7701,
            -21.7877,
            -21.8243,
            -21.8875,
            -21.9738,
            -22.0671,
            -22.1537,
            -22.2265,
            -22.2821,
            -22.3213,
            -22.3462,
            -22.3587,
            -22.3622,
            -22.3590,
            -22.3512,
            -22.3420,
            -22.3342,
            -22.3312,
            -22.3346,
            -22.3445,
            -22.3595,
            -22.3780,
            -22.4007,
            -22.4289,
            -22.4625,
            -22.4995,
            -22.5353,
            -22.5659,
            -22.5895,
            -22.6059,
            -22.6161,
            -22.6208,
            -22.6213,
            -22.6184,
            -22.6126,
            -22.6045,
            -22.5945,
            -22.5831,
            -22.5707,
            -22.5573,
            -22.5434,
            -22.5287,
            -22.5140,
            -22.4992,
            -22.4844,
            -22.4695,
            -22.4543,
            -22.4392,
            -22.4237,
            -22.4087,
            -22.3928,
        ]
    )

    lam = np.zeros_like(temp, dtype=float)

    # turn off cooling below 1e4 K
    mask_off = logt <= 4.0
    lam[mask_off] = 0.0

    # KI02 regime (4.0 < logT <= 4.2)
    mask_ki = (logt > 4.0) & (logt <= 4.2)
    if np.any(mask_ki):
        lam[mask_ki] = 2.0e-19 * np.exp(
            -1.184e5 / (temp[mask_ki] + 1.0e3)
        ) + 2.8e-28 * np.sqrt(temp[mask_ki]) * np.exp(-92.0 / temp[mask_ki])

    # CGOLS fit (logT > 8.15)
    mask_hi = logt > 8.15
    lam[mask_hi] = 10.0 ** (0.45 * logt[mask_hi] - 26.065)

    # SPEX interpolation (4.2 < logT <= 8.15)
    mask_mid = (logt > 4.2) & (logt <= 8.15)
    if np.any(mask_mid):
        ipps = (25.0 * logt[mask_mid] - 103).astype(int)
        # Clamp to [0,100] like C++
        ipps = np.clip(ipps, 0, 100)
        x0 = 4.12 + 0.04 * ipps
        dx = logt[mask_mid] - x0
        logcool = (lhd[ipps + 1] * dx - lhd[ipps] * (dx - 0.04)) * 25.0
        lam[mask_mid] = 10.0**logcool
    if mask:
        mask_off = (logt < LOGT_ACTIVE_START) | (logt > LOGT_ACTIVE_END)
        lam[mask_off] = 0.0

    return lam


# =========================
# CENTRALIZED COOLING FUNCTION  (Change #1)
# =========================
def compute_cooling_rate(
    rho_or_pdf, temp, pressure=None, is_pdf=False, is_isobaric=False, T_unit=None,
    rho_cg=None,
):
    """
    Standardized cooling calculation using internal Code Units.

    Formula:  Cooling = n² × Σᵢ PDF(Tᵢ) × Λ(Tᵢ) × unit_fix

    where n = rho / (mu) is the coarse-cell number density.
    Λ(Tᵢ) is evaluated with mask=True so only the active cooling window
    (LOGT_ACTIVE_START – LOGT_ACTIVE_END) contributes.

    Modes
    -----
    is_pdf=False : Fine-grid scalar path.
        rho_or_pdf  — code density field (any shape)
        temp        — temperature field (same shape)
        Returns     — cooling rate field (same shape)

    is_pdf=True  : PDF-integrated path.
        rho_or_pdf  — PDF array of shape (nb, nx, ny)
        temp        — T_centers array of shape (nb,)
        rho_cg      — coarse-grained code density (nx, ny)  [REQUIRED]
        Returns     — cooling rate field (nx, ny)
    """
    mu = 0.62
    unit_fix = 1.975e27  # grouped conversion: (rho_0 * L_0) / (m_H^2 * v_0^3)

    if not is_pdf:
        # --- Mode 1: Fine-grid scalar path ---
        rho_eff = rho_or_pdf
        lam = lambda_cool(temp, mask=True)
        n_code = rho_eff / mu
        return lam * (n_code**2) * unit_fix

    else:
        # --- Mode 2: PDF-integrated path ---
        # n² × Σᵢ PDF(Tᵢ) Λ(Tᵢ)   where n = rho_cg / mu (constant per cell)
        if rho_cg is None:
            raise ValueError(
                "rho_cg must be provided for PDF-mode cooling "
                "(coarse-grained code density, shape (nx, ny))."
            )

        pdf = rho_or_pdf         # (nb, nx, ny)
        T_centers = temp         # (nb,)
        lam = lambda_cool(T_centers, mask=True)  # (nb,)  — masked to active window

        # Σᵢ PDF(Tᵢ) Λ(Tᵢ)  →  (nx, ny)
        lambda_sum = np.sum(pdf * lam[:, None, None], axis=0)

        # n = rho_cg / mu  →  (nx, ny)
        n_cg = rho_cg / mu

        return (n_cg**2) * lambda_sum * unit_fix  # (nx, ny)


lambda_vals = lambda_cool(T_centers)

# take log safely
log_lambda = np.log10(lambda_vals + 1e-40)

# normalize to [0,1]
log_lambda -= log_lambda.min()
log_lambda /= log_lambda.max() + 1e-30

lambda_weights = torch.tensor(log_lambda, dtype=torch.float32)
lambda_tensor = torch.tensor(lambda_vals, dtype=torch.float32)


def nn_data(resolution: tuple, downsample: int) -> tuple:
    """A function to load the data and return the inputs and outputs for the Conv neural network."""

    sim_data = simulation_data()
    sim_data.down_sample = downsample
    sim_data.resolution = resolution

    folder_path = os.path.join(CACHE_PATH, f"sc{resolution}_{downsample}")
    file_path = DATA_PATH
    if os.path.exists(f"{folder_path}"):
        sim_data.rho = np.load(f"{folder_path}/rho.npy")
        sim_data.temp = np.load(f"{folder_path}/temp.npy")
        sim_data.pressure = np.load(f"{folder_path}/pressure.npy")
        sim_data.ux = np.load(f"{folder_path}/ux.npy")
        sim_data.uy = np.load(f"{folder_path}/uy.npy")
        sim_data.eint = np.load(f"{folder_path}/eint.npy")
        sim_data.ps = np.load(f"{folder_path}/ps.npy")

        if os.path.exists(f"{folder_path}/cons_rho.npy"):
            sim_data.cons_rho = np.load(f"{folder_path}/cons_rho.npy")
            sim_data.cons_momx = np.load(f"{folder_path}/cons_mx.npy")
            sim_data.cons_momy = np.load(f"{folder_path}/cons_my.npy")
            sim_data.cons_ener = np.load(f"{folder_path}/cons_ener.npy")
            sim_data.cons_ps = np.load(f"{folder_path}/cons_ps.npy")
    else:
        sim_data.input_data(file_path)
        sim_data.input_cons_data(file_path)
        os.makedirs(folder_path, exist_ok=True)

        np.save(f"{folder_path}/rho.npy", sim_data.rho)
        np.save(f"{folder_path}/temp.npy", sim_data.temp)
        np.save(f"{folder_path}/pressure.npy", sim_data.pressure)
        np.save(f"{folder_path}/ux.npy", sim_data.ux)
        np.save(f"{folder_path}/uy.npy", sim_data.uy)
        np.save(f"{folder_path}/eint.npy", sim_data.eint)
        np.save(f"{folder_path}/ps.npy", sim_data.ps)

        np.save(f"{folder_path}/cons_rho.npy", sim_data.cons_rho)
        np.save(f"{folder_path}/cons_mx.npy", sim_data.cons_momx)
        np.save(f"{folder_path}/cons_my.npy", sim_data.cons_momy)
        np.save(f"{folder_path}/cons_ener.npy", sim_data.cons_ener)
        np.save(f"{folder_path}/cons_ps.npy", sim_data.cons_ps)

    print("Input data loaded")

    shape = (
        sim_data.rho.shape[0],
        sim_data.rho.shape[1] // sim_data.down_sample,
        sim_data.rho.shape[2] // sim_data.down_sample,
    )
    fields = ["rho", "temp", "ux", "uy", "ps"]
    cg = {f"cg_{field}": np.zeros(shape) for field in fields}

    for i in range(sim_data.rho.shape[0]):
        for field in fields:
            if field in ["rho", "temp", "ux", "uy", "ps"]:
                cg[f"cg_{field}"][i] = sim_data.coarse_grain(
                    getattr(sim_data, field)[i]
                )
    temp_pdf = sim_data.calc_pixel_pdf(bins=out_channels)
    temp_pdf /= temp_pdf.sum(axis=1, keepdims=True)

    input_tensors = [
        torch.from_numpy(cg[f"cg_{f}"]).unsqueeze(1).float() for f in fields
    ]
    # input_tensors = [
    #     torch.from_numpy(cg[f'cg_{f}'][100:]).unsqueeze(1).float()
    #     for f in fields
    # ]
    input_tensor = torch.cat(input_tensors, dim=1)
    output_tensor = torch.from_numpy(temp_pdf).float()
    # output_tensor = torch.from_numpy(source_term[100:]).unsqueeze(1).float()

    return input_tensor, output_tensor


def snapshot_pred(
    rho: np.ndarray,
    temp: np.ndarray,
    pressure: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    eint: np.ndarray,
    ps: np.ndarray,
    downsample: int,
    resolution: np.ndarray,
) -> np.ndarray:
    """
    Predict pixel temperature PDFs for a given snapshot.
    Returns: (bins, nx, ny)
    """

    sim_data = simulation_data()
    sim_data.down_sample = downsample
    sim_data.resolution = resolution

    shape = (resolution[0] // downsample, resolution[1] // downsample)

    fields = ["rho", "temp", "ux", "uy", "ps"]
    cg = {f"cg_{field}": np.zeros(shape) for field in fields}

    # -------------------------
    # Coarse-grain inputs
    # -------------------------
    for field in fields:
        if field in ["rho", "temp", "ux", "uy", "ps"]:
            cg[f"cg_{field}"] = sim_data.coarse_grain(locals()[field])

    # -------------------------
    # Build input tensor
    # -------------------------
    input_tensors = [
        torch.from_numpy(cg[f"cg_{f}"]).unsqueeze(0).float() for f in fields
    ]

    input_tensor = torch.cat(input_tensors, dim=0)  # (C, nx, ny)
    input_tensor = input_tensor.unsqueeze(0).to(device)  # (1, C, nx, ny)

    # -------------------------
    # Normalize input (IMPORTANT)
    # -------------------------
    # Load and convert directly to tensors on the MPS device
    input_mean = torch.tensor(
        np.load(
            os.path.join(
                MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}_input_mean.npy"
            )
        ),
        dtype=torch.float32,
    ).to(device)

    input_std = torch.tensor(
        np.load(
            os.path.join(MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}_input_std.npy")
        ),
        dtype=torch.float32,
    ).to(device)

    # Now both are MPS tensors, this math works seamlessly
    input_tensor = (input_tensor - input_mean) / input_std

    # -------------------------
    # Load model
    # -------------------------
    model_path = os.path.join(MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}.pth")
    state_dict = torch.load(model_path, map_location=device)
    ckpt_ksize = kernel_size
    if "encoder.0.weight" in state_dict:
        ckpt_ksize = state_dict["encoder.0.weight"].shape[-1]

    cnn_model = ConvNN(
        in_channels,
        layer_size1,
        layer_size2,
        layer_size3,
        layer_size4,
        out_channels,
        ckpt_ksize,
    ).to(device)

    cnn_model.load_state_dict(state_dict)
    cnn_model.eval()

    # -------------------------
    # Predict PDF
    # -------------------------
    with torch.no_grad():
        pdf = cnn_model.predict_pdf(input_tensor)  # (1, bins, nx, ny)

        pdf = pdf[0].cpu().numpy()  # (bins, nx, ny)

    return pdf


def snapshot_pred_16x8(
    rho: np.ndarray,
    temp: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    ps: np.ndarray,
    fine_resolution: tuple = (512, 256),
    downsample: int = 32,
    model_save_dir: str = None,
    return_gate: bool = False,
):
    """
    Predict pixel temperature PDFs (and optionally gate values) for a snapshot
    whose fields are ALREADY coarse-grained to 16×8. No `simulation_data.coarse_grain` is performed.

    Parameters
    ----------
    rho, temp, ux, uy, ps : np.ndarray, shape (16, 8)
        Already-coarse input fields.
    fine_resolution : tuple, default (1024, 512)
        Fine-grid resolution the saved model was trained against. Used ONLY
        to build the model/normalization file names.
    downsample : int, default 64
        Downsample factor that produced the 16×8 grid; used only for naming.
    model_save_dir : str, optional
        Directory where model weights and normalization files are saved.
        Defaults to os.environ["MODEL_SAVES_DIR"] or module default.
    return_gate : bool, default False
        If True, returns tuple (pdf, gate) where gate has shape (16, 8).

    Returns
    -------
    pdf : np.ndarray, shape (out_channels, 16, 8)  ≡ (40, 16, 8)
        Predicted temperature PDF at each coarse cell. Axis 0 indexes the
        40 temperature bins defined by `T_edges`.
    gate : np.ndarray, shape (16, 8) [only if return_gate is True]
        Predicted gate value per coarse cell ∈ (0, 1).
    """

    # ---- 1. Shape check -------------------------------------------------
    fields = {"rho": rho, "temp": temp, "ux": ux, "uy": uy, "ps": ps}
    expected_shape = rho.shape
    for name, arr in fields.items():
        if tuple(arr.shape) != expected_shape:
            raise ValueError(
                f"Field '{name}' must have shape {expected_shape}, "
                f"got {tuple(arr.shape)}"
            )

    # ---- 2. Build input tensor (1, C=5, 16, 8) --------------------------
    stack = np.stack(
        [fields[f] for f in ["rho", "temp", "ux", "uy", "ps"]],
        axis=0,
    ).astype(np.float32)  # (5, 16, 8)

    input_tensor = torch.from_numpy(stack).unsqueeze(0).to(device)  # (1, 5, 16, 8)

    # ---- 3. Load normalization statistics -------------------------------
    save_dir = model_save_dir or os.environ.get("MODEL_SAVES_DIR", MODEL_SAVE_DIR)
    norm_prefix = f"cnn_{fine_resolution}_{downsample}"

    input_mean = torch.tensor(
        np.load(os.path.join(save_dir, f"{norm_prefix}_input_mean.npy")),
        dtype=torch.float32,
    ).to(device)
    input_std = torch.tensor(
        np.load(os.path.join(save_dir, f"{norm_prefix}_input_std.npy")),
        dtype=torch.float32,
    ).to(device)

    # The trainer saved them with shape (1, C, 1, 1); be defensive.
    if input_mean.dim() == 1:
        input_mean = input_mean.view(1, -1, 1, 1)
        input_std = input_std.view(1, -1, 1, 1)

    input_tensor = (input_tensor - input_mean) / input_std

    # ---- 4. Load model --------------------------------------------------
    model_path = os.path.join(save_dir, f"{norm_prefix}.pth")
    state_dict = torch.load(model_path, map_location=device)
    ckpt_ksize = kernel_size
    if "encoder.0.weight" in state_dict:
        ckpt_ksize = state_dict["encoder.0.weight"].shape[-1]

    cnn_model = ConvNN(
        in_channels,
        layer_size1,
        layer_size2,
        layer_size3,
        layer_size4,
        out_channels,
        ckpt_ksize,
    ).to(device)
    cnn_model.load_state_dict(state_dict)
    cnn_model.eval()

    # ---- 5. Predict PDF (and Gate) --------------------------------------
    with torch.no_grad():
        logits, gate = cnn_model.forward(input_tensor)
        pdf = cnn_model.pdf_activation(logits, gate)
        pdf = pdf[0].cpu().numpy()  # (40, 16, 8)
        gate = gate[0, 0].cpu().numpy()  # (16, 8)

    if return_gate:
        return pdf, gate
    return pdf


# Aliases for arbitrary coarse-grained shapes (e.g., 32x16, 16x8)
snapshot_pred_cg = snapshot_pred_16x8
snapshot_pred_coarse = snapshot_pred_16x8


def snapshot_pred_with_gate(
    rho: np.ndarray,
    temp: np.ndarray,
    pressure: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    eint: np.ndarray,
    ps: np.ndarray,
    downsample: int,
    resolution: np.ndarray,
) -> tuple:
    """
    Predict pixel temperature PDFs, gate values, and vorticity magnitude for a
    given snapshot.

    Returns
    -------
    pdf          : np.ndarray, shape (bins, nx, ny)  — predicted PDF
    gate         : np.ndarray, shape (nx, ny)         — gate ∈ (0, 1)
    vorticity_mag: np.ndarray, shape (nx, ny)         — |ω| from VorticityLayer
    """

    sim_data = simulation_data()
    sim_data.down_sample = downsample
    sim_data.resolution = resolution

    shape = (resolution[0] // downsample, resolution[1] // downsample)

    fields = ["rho", "temp", "ux", "uy", "ps"]
    cg = {f"cg_{field}": np.zeros(shape) for field in fields}

    for field in fields:
        if field in ["rho", "temp", "ux", "uy", "ps"]:
            cg[f"cg_{field}"] = sim_data.coarse_grain(locals()[field])

    input_tensors = [
        torch.from_numpy(cg[f"cg_{f}"]).unsqueeze(0).float() for f in fields
    ]

    input_tensor = torch.cat(input_tensors, dim=0)  # (C, nx, ny)
    input_tensor = input_tensor.unsqueeze(0).to(device)  # (1, C, nx, ny)

    input_mean = torch.tensor(
        np.load(
            os.path.join(
                MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}_input_mean.npy"
            )
        ),
        dtype=torch.float32,
    ).to(device)

    input_std = torch.tensor(
        np.load(
            os.path.join(MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}_input_std.npy")
        ),
        dtype=torch.float32,
    ).to(device)

    input_tensor = (input_tensor - input_mean) / input_std

    model_path = os.path.join(MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}.pth")
    state_dict = torch.load(model_path, map_location=device)
    ckpt_ksize = kernel_size
    if "encoder.0.weight" in state_dict:
        ckpt_ksize = state_dict["encoder.0.weight"].shape[-1]

    cnn_model = ConvNN(
        in_channels,
        layer_size1,
        layer_size2,
        layer_size3,
        layer_size4,
        out_channels,
        ckpt_ksize,
    ).to(device)

    cnn_model.load_state_dict(state_dict)
    cnn_model.eval()

    with torch.no_grad():
        # Append 8 mixing-layer physics channels
        x_enriched = cnn_model.mixing(input_tensor)  # (1, C+8, H, W)

        # Gate from the 8 mixing features (last 8 channels)
        mixing_feats = x_enriched[:, -cnn_model._N_MIXING :, :, :]  # (1, 8, H, W)
        gate_raw = cnn_model.gate_branch(mixing_feats)  # (1, 1, H, W)

        features = cnn_model.encoder(x_enriched)
        logits = cnn_model.decoder(features)
        pdf_tensor = cnn_model.pdf_activation(logits, gate_raw)  # (1, bins, H, W)

        pdf = pdf_tensor[0].cpu().numpy()  # (bins, nx, ny)
        gate = gate_raw[0, 0].cpu().numpy()  # (nx, ny)
        # Return |ω| from mixing features (ch 0 of mixing_feats = |ω|)
        vorticity_mag = mixing_feats[0, 0].cpu().numpy()  # (nx, ny)

    return pdf, gate, vorticity_mag


class ThresholdedSoftmax(nn.Module):
    """
    Thresholded-softmax Gate for PDF bins.

    Steps:
      1. Apply softmax along the bin axis (dim=1) to get a proper
         probability distribution.
      2. Zero out any bin whose softmax probability is below `threshold`
         (hard sparsity — below 1e-3 by default the bin is treated as
         empty and sent to exactly 0).
      3. Re-normalize the surviving bins so they still sum to 1.

    This preserves the PDF constraint (non-negative, sums to 1) while
    suppressing near-zero bins cleanly, without the gradient issues of
    a pure sparsemax projection.
    """

    def __init__(self, threshold=1e-3, eps=1e-12):
        super().__init__()
        self.threshold = threshold
        self.eps = eps

    def forward(self, logits):
        # Step 1: standard softmax over bin dimension
        p = F.softmax(logits, dim=1)  # (B, bins, nx, ny), sums to 1

        # Step 2: threshold — bins below `threshold` become exactly 0
        p = p * (p >= self.threshold).float()

        # Step 3: re-normalize so survivors still sum to 1
        return p / (p.sum(dim=1, keepdim=True) + self.eps)


class MixingLayerFeatures(nn.Module):
    """
    Builds a feature tensor capturing mixing-layer physics:
      ch 0: |ω|                              (vorticity magnitude)
      ch 1: signed ω                         (KH rolls have a characteristic sign)
      ch 2: |∇T|                             (thermal contrast — Sobel on T)
      ch 3: |∇ρ|                             (density contrast — baroclinic source)
      ch 4: cos θ = (∇T · ∇ρ)/(|∇T||∇ρ|)     (baroclinic alignment)
      ch 5: strain rate magnitude |σ|        (compressive mixing)
      ch 6: T|ω|                             (Temperature Weighted Vorticity)
      ch 7: (T - T̄)²  proxy                  (coarse-cell T variance; high when multi-phase)

    Input : (B, C, H, W)  — normalized simulation fields
    Output: (B, C+8, H, W) — original channels concatenated with the 8 mixing features
    """

    # Number of mixing-layer feature channels appended to the input.
    N_MIXING = 8

    def __init__(self, T_idx=1, rho_idx=0, ux_idx=2, uy_idx=3):
        super().__init__()
        self.T_idx = T_idx
        self.rho_idx = rho_idx
        self.ux_idx = ux_idx
        self.uy_idx = uy_idx

        # Sobel ∂/∂x  (1, 1, 3, 3)
        self.register_buffer(
            "dx_kernel",
            torch.tensor(
                [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32
            ).unsqueeze(0),
        )

        # Sobel ∂/∂y  (1, 1, 3, 3)
        self.register_buffer(
            "dy_kernel",
            torch.tensor(
                [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32
            ).unsqueeze(0),
        )

    def forward(self, x):
        # x: (B, C, H, W)  — normalized inputs
        ux = x[:, self.ux_idx : self.ux_idx + 1]  # (B,1,H,W)
        uy = x[:, self.uy_idx : self.uy_idx + 1]
        T = x[:, self.T_idx : self.T_idx + 1]
        rho = x[:, self.rho_idx : self.rho_idx + 1]

        # --- velocity gradients ---
        duy_dx = F.conv2d(uy, self.dx_kernel, padding=1)
        dux_dy = F.conv2d(ux, self.dy_kernel, padding=1)
        dux_dx = F.conv2d(ux, self.dx_kernel, padding=1)
        duy_dy = F.conv2d(uy, self.dy_kernel, padding=1)

        omega = duy_dx - dux_dy  # signed vorticity
        strain = torch.sqrt(
            (dux_dx - duy_dy) ** 2 + (duy_dx + dux_dy) ** 2 + 1e-12
        )  # |σ|

        # --- temperature gradients ---
        dT_dx = F.conv2d(T, self.dx_kernel, padding=1)
        dT_dy = F.conv2d(T, self.dy_kernel, padding=1)
        gradT = torch.sqrt(dT_dx**2 + dT_dy**2 + 1e-12)  # |∇T|

        # --- density gradients ---
        drho_dx = F.conv2d(rho, self.dx_kernel, padding=1)
        drho_dy = F.conv2d(rho, self.dy_kernel, padding=1)
        gradRho = torch.sqrt(drho_dx**2 + drho_dy**2 + 1e-12)  # |∇ρ|

        # --- baroclinic alignment: cos θ between ∇T and ∇ρ ---
        baroclinic = (dT_dx * drho_dx + dT_dy * drho_dy) / (
            gradT * gradRho + 1e-12
        )  # ∈ [-1, 1]

        # --- coarse-cell T variance proxy: (T - T_mean_local)^2 ---
        # Use a box-blur (3×3 average) to get a local mean, then square the residual.
        box = torch.ones(1, 1, 3, 3, dtype=T.dtype, device=T.device) / 9.0
        T_local_mean = F.conv2d(T, box, padding=1)
        T_var_proxy = (T - T_local_mean) ** 2  # (B,1,H,W)

        mixing_features = torch.cat(
            [
                omega.abs(),  # ch 0
                omega,  # ch 1
                gradT,  # ch 2
                gradRho,  # ch 3
                baroclinic,  # ch 4
                strain,  # ch 5
                T.abs() * omega.abs(),  # ch 6
                T_var_proxy,  # ch 7
            ],
            dim=1,
        )  # (B, 8, H, W)

        return torch.cat([x, mixing_features], dim=1)  # (B, C+8, H, W)




class MixingLayerGate(nn.Module):
    """
    Learns a spatial gate g(x,y) ∈ [0,1] from the full set of mixing-layer
    physics features produced by MixingLayerFeatures.

    g ≈ 0 → single-phase cell  (PDF collapses to a peak-bin delta)
    g ≈ 1 → mixing-layer cell  (full broad PDF is permitted)

    Input : (B, 8, H, W)  — the 8 mixing channels from MixingLayerFeatures
    Output: (B, 1, H, W)  — gate value per spatial cell
    """

    def __init__(self, n_mixing=MixingLayerFeatures.N_MIXING, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.gate_net = nn.Sequential(
            nn.Conv2d(n_mixing, 16, kernel_size, padding=padding),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid(),  # output ∈ (0, 1)
        )
        # Initialize the final Conv2d (gate_net[-2]) so the gate starts "practically"
        # closed (sigmoid(-3) ≈ 0.05); the model then learns to open it during training.
        nn.init.constant_(self.gate_net[-2].bias, -3.0)  # sigmoid(-3) ≈ 0.05
        nn.init.normal_(self.gate_net[-2].weight, std=1e-3)

    def forward(self, mixing_features):
        # mixing_features: (B, 8, H, W)
        return self.gate_net(mixing_features)  # (B, 1, H, W)


class GatedThresholdedSoftmax(nn.Module):
    """
    Differentiable Vorticity-gated PDF activation.

    When gate ≈ 0: PDF collapses to a single-phase peak using a low-temp softmax.
                   (Acts like a Delta function, but preserves gradients!)
    When gate ≈ 1: PDF remains a broad multiphase distribution.
    """
    def __init__(self, threshold=5e-3, sharp_temp=0.02, eps=1e-12):
        super().__init__()
        self.threshold = threshold
        self.sharp_temp = sharp_temp # Controls how "sharp" the delta function is
        self.eps = eps

    def forward(self, logits, gate):
        # --- 1. Multiphase (Broad) Branch ---
        p_broad = F.softmax(logits, dim=1)
        
        # Hard thresholding is bad for gradients. Only do it during evaluation/inference.
        if not self.training:
            p_broad = p_broad * (p_broad >= self.threshold).float()
            
        p_broad = p_broad / (p_broad.sum(dim=1, keepdim=True) + self.eps)

        # --- 2. Single-Phase (Sharp) Branch ---
        # Replaces torch.argmax() with a low-temperature Softmax.
        # This creates a differentiable Delta function!
        p_sharp = F.softmax(logits / self.sharp_temp, dim=1)

        # --- 3. Gated Interpolation ---
        # gate=0 -> relies entirely on the differentiable sharp peak
        # gate=1 -> relies entirely on the broad distribution
        gated = gate * p_broad + (1.0 - gate) * p_sharp

        # Final renormalization for numerical stability
        return gated / (gated.sum(dim=1, keepdim=True) + self.eps)


# VorticityLayer has been superseded by MixingLayerFeatures, which provides
# a richer 8-channel physics descriptor (vorticity, thermal/density gradients,
# baroclinic alignment, strain rate, densimetric vorticity, T-variance proxy).
# Kept as a thin alias for any legacy references.
VorticityLayer = MixingLayerFeatures


class ConvNN(nn.Module):
    """
    CNN Model for PDF prediction (with MixingLayerFeatures + MixingLayerGate)

    Architecture:
    Input: (B, 5, 16, 8)  [rho, temp, ux, uy, ps]
             │
        ┌────▼───────────────────┐
        │  MixingLayerFeatures   │  8 physics channels:
        │    5 → 13              │  |ω|, ω, |∇T|, |∇ρ|, cos θ_baroclinic,
        └────┬───────────────────┘  |σ|, ρ|ω|, T-var proxy
             │              │        mixing_features (B,8,H,W)
             │              └──────────────────►  ┌──────────────────┐
        ┌────▼────────┐                           │ MixingLayerGate  │
        │   Encoder   │  4× Conv2d + BN + ReLU    │  16→8→1 convs    │
        │ 13→32→64    │  (+ Dropout in 1st 2)     │  gate ∈ (0,1)    │
        │  →128→256   │                           └──────┬───────────┘
        └────┬────────┘                                  │
             │                                           │
        ┌────▼─────────┐                                 │
        │   Decoder    │  4× Conv2d + BN + ReLU          │
        │ 256→128→64   │                                 │
        │  →32→40      │                                 │
        └────┬─────────┘                                 │
             │  logits                                   │ gate
             └────────────────┬──────────────────────────┘
                              ▼
                    forward returns (logits, gate)
                    predict_pdf applies GatedThresholdedSoftmax
    """

    # How many extra channels MixingLayerFeatures appends.
    _N_MIXING = MixingLayerFeatures.N_MIXING  # 8

    def __init__(
        self,
        in_channels,
        layer_size1,
        layer_size2,
        layer_size3,
        layer_size4,
        out_channels,
        kernel_size,
    ):

        super().__init__()
        padding = kernel_size // 2

        # MixingLayerFeatures appends 8 physics channels derived from
        # vorticity, temperature/density gradients, strain, and their cross-terms.
        self.mixing = MixingLayerFeatures(T_idx=1, rho_idx=0, ux_idx=2, uy_idx=3)

        # Gate branch: consumes all 8 mixing features → scalar gate ∈ (0,1)
        self.gate_branch = MixingLayerGate(
            n_mixing=self._N_MIXING, kernel_size=kernel_size
        )

        # Encoder: original in_channels + 8 mixing features
        encoder_in = in_channels + self._N_MIXING
        self.encoder = nn.Sequential(
            nn.Conv2d(encoder_in, layer_size1, kernel_size, padding=padding),
            nn.BatchNorm2d(layer_size1),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(layer_size1, layer_size2, kernel_size, padding=padding),
            nn.BatchNorm2d(layer_size2),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(layer_size2, layer_size3, kernel_size, padding=padding),
            nn.BatchNorm2d(layer_size3),
            nn.ReLU(),
            nn.Conv2d(layer_size3, layer_size4, kernel_size, padding=padding),
            nn.BatchNorm2d(layer_size4),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(layer_size4, layer_size3, kernel_size, padding=padding),
            nn.BatchNorm2d(layer_size3),
            nn.ReLU(),
            nn.Conv2d(layer_size3, layer_size2, kernel_size, padding=padding),
            nn.BatchNorm2d(layer_size2),
            nn.ReLU(),
            nn.Conv2d(layer_size2, layer_size1, kernel_size, padding=padding),
            nn.BatchNorm2d(layer_size1),
            nn.ReLU(),
            nn.Conv2d(layer_size1, out_channels, kernel_size=1),
        )

        self.pdf_activation = GatedThresholdedSoftmax()
        self._gate_frozen = False

    def train(self, mode=True):
        """Override train to ensure gate_branch remains in eval mode when frozen."""
        super().train(mode)
        if getattr(self, "_gate_frozen", False):
            self.gate_branch.eval()
        return self

    def freeze_gate_branch(self):
        """Freeze all parameters in the gate branch and set it to eval mode."""
        self._gate_frozen = True
        for param in self.gate_branch.parameters():
            param.requires_grad = False
        self.gate_branch.eval()

    def unfreeze_gate_branch(self):
        """Unfreeze all parameters in the gate branch and set it to train mode."""
        self._gate_frozen = False
        for param in self.gate_branch.parameters():
            param.requires_grad = True
        self.gate_branch.train()

    def forward(self, x):
        # Append 8 mixing-layer physics channels
        x_enriched = self.mixing(x)  # (B, C+8, H, W)

        # Gate from the 8 mixing features (last 8 channels of x_enriched)
        mixing_feats = x_enriched[:, -self._N_MIXING :, :, :]  # (B, 8, H, W)
        gate = self.gate_branch(mixing_feats)  # (B, 1, H, W)

        # Main prediction path uses the full enriched tensor
        features = self.encoder(x_enriched)
        logits = self.decoder(features)

        return logits, gate  # both needed for the gated loss

    def predict_pdf(self, x):
        """Apply GatedThresholdedSoftmax and return the final PDF."""
        logits, gate = self.forward(x)
        return self.pdf_activation(logits, gate)


class MeanTemperatureLoss(nn.Module):
    """
    Enforces <T>_pred = T_coarse (volume-weighted identity).
    Done in log10 space so the loss isn't dominated by the hot tail.
    """

    def __init__(self, logT_centers=None, eps=1e-12):
        super().__init__()
        if logT_centers is None:
            logT_tensor = torch.tensor(np.log10(T_centers), dtype=torch.float32)
        elif isinstance(logT_centers, torch.Tensor):
            logT_tensor = logT_centers.clone().detach().float()
        elif isinstance(logT_centers, np.ndarray):
            if np.all(logT_centers > 100):
                logT_tensor = torch.tensor(np.log10(logT_centers), dtype=torch.float32)
            else:
                logT_tensor = torch.tensor(logT_centers, dtype=torch.float32)
        else:
            logT_tensor = torch.as_tensor(logT_centers, dtype=torch.float32)

        self.register_buffer("logT_centers", logT_tensor)
        self.eps = eps

    def forward(self, pred_pdf, T_coarse):
        """
        pred_pdf  : (B, bins, nx, ny)  — normalized PDF
        T_coarse  : (B, 1, nx, ny)     — coarse-grained temperature field
        """
        logT = self.logT_centers.to(pred_pdf.device).view(1, -1, 1, 1)
        log_mean_pred = torch.logsumexp(
            torch.log(pred_pdf + self.eps) + logT * np.log(10),
            dim=1,
            keepdim=True,
        ) / np.log(10)

        log_T_coarse = torch.log10(T_coarse.clamp(min=1.0))

        return F.mse_loss(log_mean_pred, log_T_coarse)


def emissivity_from_pdf(
    pdf,
    rho=None,
    T_coarse=None,
    T_centers_tensor=None,
    lambda_tensor=None,
    mu=0.62,
    unit_fix=1.975e27,
):
    """
    Computes cell-averaged radiative cooling emissivity:

        Emissivity = n² × Σᵢ PDF(Tᵢ) × Λ(Tᵢ) × unit_fix

    where n = rho / mu is the coarse-cell number density (constant per cell).
    Λ(Tᵢ) should already be masked to the active cooling window before being
    passed as `lambda_tensor`.

    Parameters:
    -----------
    pdf : torch.Tensor of shape (B, bins, H, W)
        Normalized temperature PDF.
    rho : torch.Tensor of shape (B, 1, H, W) or None
        Coarse-grained code density. If None, defaults to mu (n = 1).
    T_coarse : torch.Tensor of shape (B, 1, H, W), optional
        Coarse-grained temperature (in K).  [kept for API compatibility]
    T_centers_tensor : torch.Tensor of shape (bins,), optional
        Temperature bin centers (in K).    [kept for API compatibility]
    lambda_tensor : torch.Tensor of shape (bins,), optional
        Cooling curve Λ(T) in erg cm³/s, pre-masked to the active window.
    mu : float
        Mean molecular weight (default 0.62).
    unit_fix : float
        Unit conversion factor to code units (default 1.975e27).

    Returns:
    --------
    emiss : torch.Tensor of shape (B, 1, H, W)
        Cell emissivity in code units.
    """
    if lambda_tensor is None:
        tc = (
            T_centers_tensor
            if T_centers_tensor is not None
            else torch.tensor(T_centers, dtype=torch.float32)
        )
        lam_np = lambda_cool(
            tc.cpu().numpy() if isinstance(tc, torch.Tensor) else tc, mask=True
        )
        lambda_tensor = torch.tensor(lam_np, dtype=torch.float32, device=pdf.device)

    # Σᵢ PDF(Tᵢ) Λ(Tᵢ)  →  (B, 1, H, W)
    lam = lambda_tensor.to(pdf.device).view(1, -1, 1, 1)
    lambda_sum = torch.sum(pdf * lam, dim=1, keepdim=True)

    # n = rho / mu  →  (B, 1, H, W)
    if rho is not None:
        n_coarse = rho / mu
    else:
        n_coarse = 1.0

    emiss = (n_coarse**2) * lambda_sum * unit_fix
    return emiss


# Alias for backwards compatibility
isobaric_emissivity_from_pdf = emissivity_from_pdf


class EmissivityLoss(nn.Module):
    """
    Emissivity Matching Loss:
    Enforces consistency between predicted and true cell-averaged radiative cooling emissivity:
        Emissivity = n² × Σᵢ PDF(Tᵢ) × Λ(Tᵢ) × unit_fix

    where n = rho / mu.

    In log10 space:
        loss = MSE(log10(emiss_pred + eps), log10(emiss_true + eps))
    """

    def __init__(
        self,
        T_centers_input=None,
        lambda_tensor_input=None,
        mu=0.62,
        unit_fix=1.975e27,
        eps=1e-6,
    ):
        super().__init__()
        if T_centers_input is None:
            tc = torch.tensor(T_centers, dtype=torch.float32)
        elif isinstance(T_centers_input, torch.Tensor):
            tc = T_centers_input.clone().detach().float()
        else:
            tc = torch.tensor(T_centers_input, dtype=torch.float32)

        if lambda_tensor_input is None:
            lam = lambda_cool(tc.cpu().numpy(), mask=True)
            lam_t = torch.tensor(lam, dtype=torch.float32)
        elif isinstance(lambda_tensor_input, torch.Tensor):
            lam_t = lambda_tensor_input.clone().detach().float()
        else:
            lam_t = torch.tensor(lambda_tensor_input, dtype=torch.float32)

        self.register_buffer("lambda_tensor", lam_t)
        self.mu = mu
        self.unit_fix = unit_fix
        self.eps = eps

    def forward(self, pred_pdf, true_pdf, rho=None):
        emiss_pred = emissivity_from_pdf(
            pdf=pred_pdf,
            rho=rho,
            lambda_tensor=self.lambda_tensor,
            mu=self.mu,
            unit_fix=self.unit_fix,
        )
        emiss_true = emissivity_from_pdf(
            pdf=true_pdf,
            rho=rho,
            lambda_tensor=self.lambda_tensor,
            mu=self.mu,
            unit_fix=self.unit_fix,
        )

        log_pred = torch.log10(emiss_pred + self.eps)
        log_true = torch.log10(emiss_true + self.eps)

        return F.mse_loss(log_pred, log_true)


# Alias for backwards compatibility
IsobaricEmissivityLoss = EmissivityLoss


class ZonedWassersteinLoss(nn.Module):
    """
    Splits the 1D Wasserstein-1 (Earth Mover's) distance between predicted and true PDFs
    into two pieces along the temperature-bin axis:

      W1_active   : bins inside [LOGT_ACTIVE_START, LOGT_ACTIVE_END]
      W1_inactive : all remaining bins

    For discrete 1D distributions, W1 is the integral (sum) of the absolute difference
    between cumulative distribution functions (CDFs):
      CDF_pred = cumsum(pred_pdf, dim=1)
      CDF_true = cumsum(true_pdf, dim=1)
      wass_diff = |CDF_pred - CDF_true|

    total = alpha_active * mean(W1_active) + alpha_inactive * mean(W1_inactive)
    """

    def __init__(
        self,
        active_bin_mask=None,
        alpha_active=1.0,
        alpha_inactive=0.1,
    ):
        super().__init__()
        if active_bin_mask is None:
            logt = np.log10(T_centers)
            mask_np = (logt >= LOGT_ACTIVE_START) & (logt <= LOGT_ACTIVE_END)
            active_bin_mask = torch.tensor(mask_np, dtype=torch.float32)
        elif not isinstance(active_bin_mask, torch.Tensor):
            active_bin_mask = torch.tensor(active_bin_mask, dtype=torch.float32)

        # (1, bins, 1, 1) so it broadcasts against (B, bins, nx, ny)
        self.register_buffer("active_mask", active_bin_mask.view(1, -1, 1, 1).float())
        self.alpha_active = alpha_active
        self.alpha_inactive = alpha_inactive

    def forward(self, pred_pdf, true_pdf):
        mask = self.active_mask.to(pred_pdf.device)

        cdf_pred = torch.cumsum(pred_pdf, dim=1)
        cdf_true = torch.cumsum(true_pdf, dim=1)
        wass_diff = torch.abs(cdf_pred - cdf_true)  # (B, bins, nx, ny)

        wass_active = torch.sum(wass_diff * mask, dim=1)  # (B, nx, ny)
        wass_inactive = torch.sum(wass_diff * (1.0 - mask), dim=1)  # (B, nx, ny)

        loss_active, loss_inactive = torch.mean(wass_active), torch.mean(wass_inactive)
        total = self.alpha_active * loss_active + self.alpha_inactive * loss_inactive
        return total, loss_active.detach(), loss_inactive.detach()


# Alias for backwards compatibility
ZonedSymmetricKLLoss = ZonedWassersteinLoss


class LeakageLoss(nn.Module):
    """
    Active window mass leakage loss:
    Penalizes the log10 discrepancy between total predicted mass and total true mass
    in the active cooling window [LOGT_ACTIVE_START, LOGT_ACTIVE_END]:
        L_leak = mean( (log10(pred_active_mass + eps) - log10(true_active_mass + eps))^2 )
    """

    def __init__(
        self,
        active_bin_mask=None,
        eps=1e-12,
    ):
        super().__init__()
        if active_bin_mask is None:
            logt = np.log10(T_centers)
            mask_np = (logt >= LOGT_ACTIVE_START) & (logt <= LOGT_ACTIVE_END)
            active_bin_mask = torch.tensor(mask_np, dtype=torch.float32)
        elif not isinstance(active_bin_mask, torch.Tensor):
            active_bin_mask = torch.tensor(active_bin_mask, dtype=torch.float32)

        # (1, bins, 1, 1) so it broadcasts against (B, bins, nx, ny)
        self.register_buffer("active_mask", active_bin_mask.view(1, -1, 1, 1).float())
        self.eps = eps

    def forward(self, pred_pdf, true_pdf):
        mask = self.active_mask.to(pred_pdf.device)
        pred_mass = torch.sum(pred_pdf * mask, dim=1)
        true_mass = torch.sum(true_pdf * mask, dim=1)

        leak_loss = (
            torch.log10(pred_mass + self.eps) - torch.log10(true_mass + self.eps)
        ).pow(2).mean()

        return leak_loss


class GatedPDFLoss(nn.Module):
    """
    Composite PDF loss with core terms:
      1. Zoned Wasserstein-1 distance (active vs inactive temperature zones)
      2. Gate supervision loss (BCE against active-window PDF mass)
      3. Mean temperature matching loss (<T>_pred vs T_coarse in log-space)
      4. Emissivity matching loss (MSE on log10(emissivity))
      5. Active window mass leakage loss (MSE on log10(active mass))
    """

    def __init__(
        self,
        alpha_gate=50.0,
        alpha_mean_temp=10.0,
        alpha_emiss=10.0,
        alpha_leak=10.0,
        alpha_active_wasserstein=None,
        alpha_inactive_wasserstein=None,
        alpha_active_kl=1.0,
        alpha_inactive_kl=0.1,
        entropy_threshold=0.05,  # kept for backwards compatibility
        logT_centers=None,
        T_centers=None,
        lambda_tensor=None,
        mu=0.62,
        unit_fix=1.975e27,
        active_bin_mask=None,  # NEW: accept an external mask
    ):
        super().__init__()
        self.alpha_gate = alpha_gate
        self.alpha_mean_temp = alpha_mean_temp
        self.alpha_emiss = alpha_emiss
        self.alpha_leak = alpha_leak
        self.entropy_threshold = entropy_threshold

        alpha_act = (
            alpha_active_wasserstein
            if alpha_active_wasserstein is not None
            else alpha_active_kl
        )
        alpha_inact = (
            alpha_inactive_wasserstein
            if alpha_inactive_wasserstein is not None
            else alpha_inactive_kl
        )

        self.activation = GatedThresholdedSoftmax()
        self.zoned_wasserstein = ZonedWassersteinLoss(
            alpha_active=alpha_act,
            alpha_inactive=alpha_inact,
            active_bin_mask=active_bin_mask,
        )
        self.zoned_kl = self.zoned_wasserstein  # backwards compatibility alias
        self.mean_temp_loss = MeanTemperatureLoss(logT_centers)
        self.emiss_loss = EmissivityLoss(
            T_centers_input=T_centers,
            lambda_tensor_input=lambda_tensor,
            mu=mu,
            unit_fix=unit_fix,
        )
        self.leak_loss = LeakageLoss()

        # --- NEW: Initialize the active cooling window mask ---
        if active_bin_mask is None:
            tc = T_centers if T_centers is not None else globals().get("T_centers")
            if isinstance(tc, torch.Tensor):
                tc = tc.cpu().numpy()
            logt = np.log10(tc)
            mask_np = (logt >= LOGT_ACTIVE_START) & (logt <= LOGT_ACTIVE_END)
            active_bin_mask = torch.tensor(mask_np, dtype=torch.float32)
        elif not isinstance(active_bin_mask, torch.Tensor):
            active_bin_mask = torch.tensor(active_bin_mask, dtype=torch.float32)

        # (1, bins, 1, 1) so it broadcasts against (B, bins, nx, ny)
        self.register_buffer("active_mask", active_bin_mask.view(1, -1, 1, 1).float())

    def forward(self, logits, gate, true_pdf, rho=None, T_coarse=None, return_components=False):
        pred_pdf = self.activation(logits, gate)

        # 1. Zoned Wasserstein-1 distance (replaces KL divergence)
        wass_loss, wass_active, wass_inactive = self.zoned_wasserstein(pred_pdf, true_pdf)

        # 2. Gate supervision loss based on active cooling window mass (only if alpha_gate > 0)
        if self.alpha_gate > 0:
            mask = self.active_mask.to(true_pdf.device)
            active_mass = torch.sum(true_pdf * mask, dim=1, keepdim=True)  # (B, 1, nx, ny)
            gate_target = (active_mass > 1e-8).float()
            gate_clamped = torch.clamp(gate, min=1e-7, max=1.0 - 1e-7)
            gate_loss = F.binary_cross_entropy(gate_clamped, gate_target)
        else:
            gate_loss = torch.tensor(0.0, device=true_pdf.device)

        # 3. Mean temperature matching loss
        mean_temp_loss = (
            self.mean_temp_loss(pred_pdf, T_coarse)
            if T_coarse is not None and self.alpha_mean_temp > 0
            else torch.tensor(0.0, device=true_pdf.device)
        )

        # 4. Emissivity matching loss
        emiss_loss = (
            self.emiss_loss(pred_pdf, true_pdf, rho=rho)
            if self.alpha_emiss > 0
            else torch.tensor(0.0, device=true_pdf.device)
        )

        # 5. Active window mass leakage loss
        leak_loss = (
            self.leak_loss(pred_pdf, true_pdf)
            if self.alpha_leak > 0
            else torch.tensor(0.0, device=true_pdf.device)
        )

        total_loss = (
            wass_loss
            + (self.alpha_gate * gate_loss if self.alpha_gate > 0 else 0.0)
            + self.alpha_mean_temp * mean_temp_loss
            + self.alpha_emiss * emiss_loss
            + self.alpha_leak * leak_loss
        )

        if return_components:
            def _to_float(val):
                return val.item() if isinstance(val, torch.Tensor) else float(val)

            components = {
                "total": _to_float(total_loss),
                "wasserstein_total": _to_float(wass_loss),
                "wasserstein_active": _to_float(wass_active),
                "wasserstein_inactive": _to_float(wass_inactive),
                # Aliases for backwards compatibility with legacy readers
                "kl_total": _to_float(wass_loss),
                "kl_active": _to_float(wass_active),
                "kl_inactive": _to_float(wass_inactive),
                "gate": _to_float(gate_loss),
                "mean_temp": _to_float(mean_temp_loss),
                "emiss": _to_float(emiss_loss),
                "leak": _to_float(leak_loss),
            }
            return total_loss, components

        return total_loss


# Alias for backwards compatibility
GatedPDFEmissivityLoss = GatedPDFLoss


def plot_gate_training(gate_history, save_path=None):
    """Plot BCE Loss and Binary Accuracy for the standalone Gate training stage."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = gate_history["epochs"]
    # 1. BCE Loss
    ax1.plot(epochs, gate_history["train_loss"], label="Train BCE Loss", color="tab:blue", lw=1.5)
    ax1.plot(epochs, gate_history["val_loss"], label="Val BCE Loss", color="tab:orange", lw=1.5)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("Gate Pretraining: BCE Loss")
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")
    ax1.legend(frameon=True)

    # 2. Binary Accuracy
    ax2.plot(epochs, [acc * 100 for acc in gate_history["train_acc"]], label="Train Accuracy (%)", color="tab:green", lw=1.5)
    ax2.plot(epochs, [acc * 100 for acc in gate_history["val_acc"]], label="Val Accuracy (%)", color="tab:red", lw=1.5)
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Gate Pretraining: Classification Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=True)

    plt.suptitle("Stage 1: MixingLayerGate Pretraining Progress", fontsize=14, y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved gate training plot to: {save_path}")

    plt.close(fig)


def train_gate_branch(
    cnn_model,
    train_loader,
    val_loader,
    active_mask,
    device,
    epochs=200,
    lr=1e-3,
    weight_decay=1e-5,
    grad_clip_max_norm=1.0,
    save_path=None,
):
    """
    Stage 1: Pre-train the MixingLayerGate branch independently on the binary target:
        gate_target = (active_mass > 1e-8).float()
    """
    print(f"\n{'='*60}")
    print(f"STAGE 1: Pretraining Gate Branch ({epochs} epochs, lr={lr})")
    print(f"{'='*60}")

    cnn_model.unfreeze_gate_branch()
    # Freeze encoder/decoder during gate pretraining
    for p in cnn_model.encoder.parameters():
        p.requires_grad = False
    for p in cnn_model.decoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        cnn_model.gate_branch.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=max(1, len(train_loader)),
        epochs=epochs,
        pct_start=0.15,
    )

    epochs_arr = []
    train_bce_arr = []
    val_bce_arr = []
    train_acc_arr = []
    val_acc_arr = []

    mask = active_mask.to(device)
    best_val_loss = float("inf")
    best_gate_state = None

    for epoch in tqdm(range(epochs), desc="Gate Pretraining"):
        if hasattr(train_loader.dataset, "resample"):
            train_loader.dataset.resample()
        if hasattr(val_loader.dataset, "resample"):
            val_loader.dataset.resample()
            if hasattr(train_loader.dataset, "input_mean") and hasattr(val_loader.dataset, "set_norm_stats"):
                val_loader.dataset.set_norm_stats(train_loader.dataset.input_mean, train_loader.dataset.input_std)

        cnn_model.gate_branch.train()
        for inputs, labels, rho, temp in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Data augmentations
            if torch.rand(1).item() > 0.5:
                inputs = torch.flip(inputs, [3])
                labels = torch.flip(labels, [3])
                inputs = inputs.clone()
                inputs[:, 2] = -inputs[:, 2]  # negate ux
            if torch.rand(1).item() > 0.5:
                inputs = torch.flip(inputs, [2])
                labels = torch.flip(labels, [2])
                inputs = inputs.clone()
                inputs[:, 3] = -inputs[:, 3]  # negate uy

            # Target: active cooling window mass > 1e-8
            active_mass = torch.sum(labels * mask, dim=1, keepdim=True)
            gate_target = (active_mass > 1e-8).float()

            with torch.no_grad():
                x_enriched = cnn_model.mixing(inputs)
                mixing_feats = x_enriched[:, -cnn_model._N_MIXING :, :, :]

            gate = cnn_model.gate_branch(mixing_feats)
            gate_clamped = torch.clamp(gate, min=1e-7, max=1.0 - 1e-7)
            loss = F.binary_cross_entropy(gate_clamped, gate_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                cnn_model.gate_branch.parameters(), max_norm=grad_clip_max_norm
            )
            optimizer.step()
            scheduler.step()

        # Evaluate epoch
        cnn_model.gate_branch.eval()
        with torch.no_grad():
            tr_loss, tr_correct, tr_total = 0.0, 0, 0
            for inputs, labels, _, _ in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                active_mass = torch.sum(labels * mask, dim=1, keepdim=True)
                gate_target = (active_mass > 1e-8).float()

                x_enriched = cnn_model.mixing(inputs)
                mixing_feats = x_enriched[:, -cnn_model._N_MIXING :, :, :]
                gate = cnn_model.gate_branch(mixing_feats)
                gate_clamped = torch.clamp(gate, min=1e-7, max=1.0 - 1e-7)

                tr_loss += F.binary_cross_entropy(gate_clamped, gate_target).item()
                pred_binary = (gate > 0.5).float()
                tr_correct += (pred_binary == gate_target).sum().item()
                tr_total += gate_target.numel()

            val_loss, val_correct, val_total = 0.0, 0, 0
            for inputs, labels, _, _ in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                active_mass = torch.sum(labels * mask, dim=1, keepdim=True)
                gate_target = (active_mass > 1e-8).float()

                x_enriched = cnn_model.mixing(inputs)
                mixing_feats = x_enriched[:, -cnn_model._N_MIXING :, :, :]
                gate = cnn_model.gate_branch(mixing_feats)
                gate_clamped = torch.clamp(gate, min=1e-7, max=1.0 - 1e-7)

                val_loss += F.binary_cross_entropy(gate_clamped, gate_target).item()
                pred_binary = (gate > 0.5).float()
                val_correct += (pred_binary == gate_target).sum().item()
                val_total += gate_target.numel()

            epoch_tr_loss = tr_loss / max(1, len(train_loader))
            epoch_val_loss = val_loss / max(1, len(val_loader))
            epoch_tr_acc = tr_correct / max(1, tr_total)
            epoch_val_acc = val_correct / max(1, val_total)

            epochs_arr.append(epoch + 1)
            train_bce_arr.append(epoch_tr_loss)
            val_bce_arr.append(epoch_val_loss)
            train_acc_arr.append(epoch_tr_acc)
            val_acc_arr.append(epoch_val_acc)

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_val_acc = epoch_val_acc
                best_gate_state = {k: v.cpu().clone() for k, v in cnn_model.gate_branch.state_dict().items()}

    # Restore best gate weights
    if best_gate_state is not None:
        cnn_model.gate_branch.load_state_dict(best_gate_state)
        print(f"Restored best gate branch weights (Val BCE: {best_val_loss:.6f}, Val Acc: {best_val_acc*100:.2f}%)")

    if save_path:
        torch.save(cnn_model.gate_branch.state_dict(), save_path)
        print(f"Saved pretrained gate branch weights to: {save_path}")

    # Unfreeze encoder/decoder for Stage 2
    for p in cnn_model.encoder.parameters():
        p.requires_grad = True
    for p in cnn_model.decoder.parameters():
        p.requires_grad = True

    return {
        "epochs": epochs_arr,
        "train_loss": train_bce_arr,
        "val_loss": val_bce_arr,
        "train_acc": train_acc_arr,
        "val_acc": val_acc_arr,
    }


def plot_loss_breakdown(
    epochs_array,
    train_history,
    val_history,
    test_history=None,
    save_path=None,
    hyperparams=None,
):
    """
    Plot total loss and all individual loss terms across training epochs.
    Creates a 2x3 grid:
      1. Total Loss
      2. Zoned Wasserstein Loss (Total, Active, Inactive)
      3. Gate Supervision Loss (BCE)
      4. Mean Temperature Loss (MSE log10 T)
      5. Emissivity Loss (MSE log10 Emissivity)
      6. Active Window Mass Leakage Loss
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    axes = axes.flatten()

    def _safe_set_yscale_log(axis, *arrays):
        has_positive = any(any(v > 1e-12 for v in a if isinstance(v, (int, float))) for a in arrays if len(a) > 0)
        if has_positive:
            axis.set_yscale("log")

    # 1. Total Loss
    ax = axes[0]
    ax.plot(epochs_array, train_history["total"], label="Train Total", color="tab:blue", lw=1.5)
    ax.plot(epochs_array, val_history["total"], label="Val Total", color="tab:orange", lw=1.5)
    if test_history and "total" in test_history:
        ax.axhline(test_history["total"], color="tab:red", linestyle="--", alpha=0.8, label=f"Test: {test_history['total']:.4f}")
    ax.set_ylabel("Total Loss")
    ax.set_title("Total Loss")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    _safe_set_yscale_log(ax, train_history["total"], val_history["total"])

    # 2. Zoned Wasserstein Loss
    ax = axes[1]
    w_train_tot = train_history.get("wasserstein_total", train_history.get("kl_total", []))
    w_val_tot = val_history.get("wasserstein_total", val_history.get("kl_total", []))
    w_train_act = train_history.get("wasserstein_active", train_history.get("kl_active", []))
    w_val_act = val_history.get("wasserstein_active", val_history.get("kl_active", []))
    w_train_inact = train_history.get("wasserstein_inactive", train_history.get("kl_inactive", []))
    w_val_inact = val_history.get("wasserstein_inactive", val_history.get("kl_inactive", []))

    ax.plot(epochs_array, w_train_tot, label="Train W1 Total", color="tab:blue", lw=1.5)
    ax.plot(epochs_array, w_val_tot, label="Val W1 Total", color="tab:orange", lw=1.5)
    ax.plot(epochs_array, w_train_act, label="Train Active W1", color="tab:green", lw=1.0, linestyle=":")
    ax.plot(epochs_array, w_val_act, label="Val Active W1", color="tab:olive", lw=1.0, linestyle=":")
    ax.plot(epochs_array, w_train_inact, label="Train Inactive W1", color="tab:purple", lw=1.0, linestyle="--")
    ax.plot(epochs_array, w_val_inact, label="Val Inactive W1", color="tab:pink", lw=1.0, linestyle="--")
    ax.set_ylabel("Wasserstein-1 Loss")
    alpha_act = (
        hyperparams.get("alpha_active_wasserstein", hyperparams.get("alpha_active_kl", 1.0))
        if hyperparams
        else 1.0
    )
    alpha_inact = (
        hyperparams.get("alpha_inactive_wasserstein", hyperparams.get("alpha_inactive_kl", 0.1))
        if hyperparams
        else 0.1
    )
    ax.set_title(f"Zoned Wasserstein Loss (α_act={alpha_act}, α_inact={alpha_inact})")
    ax.legend(frameon=True, fontsize=8)
    ax.grid(True, alpha=0.3)
    _safe_set_yscale_log(ax, w_train_tot, w_val_tot)

    # 3. Gate Supervision Loss
    ax = axes[2]
    ax.plot(epochs_array, train_history["gate"], label="Train Gate BCE", color="tab:blue", lw=1.5)
    ax.plot(epochs_array, val_history["gate"], label="Val Gate BCE", color="tab:orange", lw=1.5)
    if test_history and "gate" in test_history:
        ax.axhline(test_history["gate"], color="tab:red", linestyle="--", alpha=0.8, label=f"Test: {test_history['gate']:.4f}")
    ax.set_ylabel("BCE Loss")
    alpha_gate = hyperparams.get("alpha_gate", 0.0) if hyperparams else 0.0
    ax.set_title(f"Gate Supervision Loss (α_gate={alpha_gate})")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    _safe_set_yscale_log(ax, train_history["gate"], val_history["gate"])

    # 4. Mean Temperature Loss
    ax = axes[3]
    ax.plot(epochs_array, train_history["mean_temp"], label="Train Mean Temp", color="tab:blue", lw=1.5)
    ax.plot(epochs_array, val_history["mean_temp"], label="Val Mean Temp", color="tab:orange", lw=1.5)
    if test_history and "mean_temp" in test_history:
        ax.axhline(test_history["mean_temp"], color="tab:red", linestyle="--", alpha=0.8, label=f"Test: {test_history['mean_temp']:.4f}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("MSE (log10 T)")
    alpha_temp = hyperparams.get("alpha_mean_temp", 10.0) if hyperparams else 10.0
    ax.set_title(f"Mean Temperature Matching (α_temp={alpha_temp})")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    _safe_set_yscale_log(ax, train_history["mean_temp"], val_history["mean_temp"])

    # 5. Emissivity Loss
    ax = axes[4]
    ax.plot(epochs_array, train_history["emiss"], label="Train Emissivity", color="tab:blue", lw=1.5)
    ax.plot(epochs_array, val_history["emiss"], label="Val Emissivity", color="tab:orange", lw=1.5)
    if test_history and "emiss" in test_history:
        ax.axhline(test_history["emiss"], color="tab:red", linestyle="--", alpha=0.8, label=f"Test: {test_history['emiss']:.4f}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("MSE (log10 Emissivity)")
    alpha_emiss = hyperparams.get("alpha_emiss", 10.0) if hyperparams else 10.0
    ax.set_title(f"Emissivity Matching Loss (α_emiss={alpha_emiss})")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    _safe_set_yscale_log(ax, train_history["emiss"], val_history["emiss"])

    # 6. Active Window Mass Leakage Loss
    ax = axes[5]
    ax.plot(epochs_array, train_history["leak"], label="Train Leakage", color="tab:blue", lw=1.5)
    ax.plot(epochs_array, val_history["leak"], label="Val Leakage", color="tab:orange", lw=1.5)
    if test_history and "leak" in test_history:
        ax.axhline(test_history["leak"], color="tab:red", linestyle="--", alpha=0.8, label=f"Test: {test_history['leak']:.4f}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("MSE (log10 Active Mass)")
    alpha_leak = hyperparams.get("alpha_leak", 10.0) if hyperparams else 10.0
    ax.set_title(f"Active Window Leakage Loss (α_leak={alpha_leak})")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    _safe_set_yscale_log(ax, train_history["leak"], val_history["leak"])

    plt.suptitle("Training & Validation Loss Terms Breakdown", fontsize=16, y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved loss components plot to: {save_path}")

    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PDF CNN with configurable hyperparameters")
    parser.add_argument("--alpha_active_wasserstein", "--alpha_active_kl", type=float, default=HYPERPARAMS.get("alpha_active_wasserstein", 100.0), help="Active zone Wasserstein loss weight")
    parser.add_argument("--alpha_inactive_wasserstein", "--alpha_inactive_kl", type=float, default=HYPERPARAMS.get("alpha_inactive_wasserstein", 10.0), help="Inactive zone Wasserstein loss weight")
    parser.add_argument("--alpha_gate", type=float, default=HYPERPARAMS["alpha_gate"], help="Gate loss weight")
    parser.add_argument("--alpha_mean_temp", type=float, default=HYPERPARAMS.get("alpha_mean_temp", 10.0), help="Mean temperature loss weight")
    parser.add_argument("--alpha_emiss", type=float, default=HYPERPARAMS.get("alpha_emiss", 10.0), help="Emissivity loss weight")
    parser.add_argument("--alpha_leak", type=float, default=HYPERPARAMS.get("alpha_leak", 10.0), help="Active window mass leakage loss weight")
    # Gate pretraining arguments
    parser.add_argument("--gate_epochs", type=int, default=HYPERPARAMS["gate_epochs"], help="Epochs for gate pretraining (Stage 1)")
    parser.add_argument("--gate_learning_rate", type=float, default=HYPERPARAMS["gate_learning_rate"], help="Learning rate for gate pretraining")
    parser.add_argument("--skip_gate_training", action="store_true", help="Skip Stage 1 gate pretraining")
    parser.add_argument("--gate_weights_path", type=str, default=None, help="Path to pre-trained gate weights")
    parser.add_argument("--freeze_gate", action="store_true", default=True, help="Freeze gate branch during Stage 2 PDF training")
    parser.add_argument("--no_freeze_gate", action="store_false", dest="freeze_gate", help="Do not freeze gate branch during Stage 2")
    # Main training arguments
    parser.add_argument("--alpha_profile", type=float, default=None, help="Profile loss weight (alias)")
    parser.add_argument("--alpha_active_pdf", type=float, default=None, help="Active PDF loss weight (alias)")
    parser.add_argument("--num_epochs", type=int, default=HYPERPARAMS["num_epochs"], help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=HYPERPARAMS["learning_rate"], help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=HYPERPARAMS["batch_size"], help="Batch size")
    parser.add_argument("--model_save_dir", type=str, default=None, help="Directory to save model weights")
    parser.add_argument("--loss_plot_dir", type=str, default=None, help="Directory to save loss plots")

    args, _ = parser.parse_known_args()

    HYPERPARAMS["alpha_active_wasserstein"] = args.alpha_active_wasserstein
    HYPERPARAMS["alpha_inactive_wasserstein"] = args.alpha_inactive_wasserstein
    HYPERPARAMS["alpha_active_kl"] = args.alpha_active_wasserstein
    HYPERPARAMS["alpha_inactive_kl"] = args.alpha_inactive_wasserstein
    HYPERPARAMS["alpha_gate"] = args.alpha_gate
    HYPERPARAMS["alpha_mean_temp"] = args.alpha_mean_temp
    HYPERPARAMS["alpha_emiss"] = args.alpha_emiss
    HYPERPARAMS["alpha_leak"] = args.alpha_leak
    HYPERPARAMS["gate_epochs"] = args.gate_epochs
    HYPERPARAMS["gate_learning_rate"] = args.gate_learning_rate
    HYPERPARAMS["freeze_gate"] = args.freeze_gate
    HYPERPARAMS["num_epochs"] = args.num_epochs
    HYPERPARAMS["learning_rate"] = args.learning_rate
    HYPERPARAMS["batch_size"] = args.batch_size

    num_epochs = HYPERPARAMS["num_epochs"]
    learning_rate = HYPERPARAMS["learning_rate"]
    batch_size = HYPERPARAMS["batch_size"]

    if args.model_save_dir:
        MODEL_SAVE_DIR = args.model_save_dir
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    if args.loss_plot_dir:
        LOSS_PLOT_DIR = args.loss_plot_dir
        os.makedirs(LOSS_PLOT_DIR, exist_ok=True)

    file_path = DATA_PATH

    print("Training all fluxes model")
    print(
        f"Hyperparameters: alpha_active_wasserstein={HYPERPARAMS['alpha_active_wasserstein']}, "
        f"alpha_inactive_wasserstein={HYPERPARAMS['alpha_inactive_wasserstein']}, "
        f"alpha_gate={HYPERPARAMS['alpha_gate']}, "
        f"alpha_mean_temp={HYPERPARAMS['alpha_mean_temp']}, "
        f"alpha_emiss={HYPERPARAMS['alpha_emiss']}, "
        f"alpha_leak={HYPERPARAMS['alpha_leak']}, "
        f"gate_epochs={HYPERPARAMS['gate_epochs']}, "
        f"freeze_gate={HYPERPARAMS['freeze_gate']}, "
        f"epochs={num_epochs}, lr={learning_rate}"
    )
    print(f"Saving model to: {MODEL_SAVE_DIR}")

    # Initialize model
    cnn_model = ConvNN(
        in_channels,
        layer_size1,
        layer_size2,
        layer_size3,
        layer_size4,
        out_channels,
        kernel_size,
    ).to(device)

    alpha_gate_stage2 = 0.0 if HYPERPARAMS["freeze_gate"] else HYPERPARAMS["alpha_gate"]

    criterion = GatedPDFLoss(
        alpha_gate=alpha_gate_stage2,
        alpha_mean_temp=HYPERPARAMS.get("alpha_mean_temp", 10.0),
        alpha_emiss=HYPERPARAMS.get("alpha_emiss", 10.0),
        alpha_leak=HYPERPARAMS.get("alpha_leak", 10.0),
        alpha_active_wasserstein=HYPERPARAMS.get("alpha_active_wasserstein", 100.0),
        alpha_inactive_wasserstein=HYPERPARAMS.get("alpha_inactive_wasserstein", 10.0),
    )

    # Load dataset
    cnn_data = nn_data(resolution, downsample)
    input_tensor, output_tensor = cnn_data

    input_tensor = input_tensor.to(device)
    output_tensor = output_tensor.to(device)

    # Numerical stability for PDFs
    output_tensor = torch.clamp(output_tensor, min=1e-12)
    output_tensor = output_tensor / output_tensor.sum(dim=1, keepdim=True)

    print("Normalizing input tensor")

    input_mean = input_tensor.mean(dim=(0, 2, 3), keepdim=True)
    input_std = input_tensor.std(dim=(0, 2, 3), keepdim=True)
    input_std[input_std == 0] = 1.0

    np.save(
        os.path.join(MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}_input_mean.npy"),
        input_mean.cpu().numpy(),
    )
    np.save(
        os.path.join(MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}_input_std.npy"),
        input_std.cpu().numpy(),
    )

    input_tensor_norm = (input_tensor - input_mean) / input_std

    # Store raw (un-normalized) rho and temp as tensors so the loss functions
    # always receive physical quantities, not z-scored ones.
    rho_tensor = input_tensor[:, 0:1]   # (N, 1, nx, ny), un-normalized density
    temp_tensor = input_tensor[:, 1:2]  # (N, 1, nx, ny), un-normalized coarse-grain temp
    dataset = TensorDataset(input_tensor_norm, output_tensor, rho_tensor, temp_tensor)

    num_samples = len(dataset)
    print("Number of samples:", num_samples)

    indices = np.random.permutation(num_samples)

    train_end = int(HYPERPARAMS["train_fraction"] * num_samples)
    val_end = int((HYPERPARAMS["train_fraction"] + HYPERPARAMS["val_fraction"]) * num_samples)

    train_dataset = Subset(dataset, indices[:train_end])
    val_dataset = Subset(dataset, indices[train_end:val_end])
    test_dataset = Subset(dataset, indices[val_end:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # ── STAGE 1: GATE PRETRAINING ─────────────────────────────────────
    gate_save_path = os.path.join(MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}_gate.pth")
    gate_plot_path = os.path.join(LOSS_PLOT_DIR, f"cnn_{resolution}_{downsample}_gate_loss.jpg")

    if args.gate_weights_path and os.path.exists(args.gate_weights_path):
        print(f"\nLoading pretrained gate weights from: {args.gate_weights_path}")
        cnn_model.gate_branch.load_state_dict(torch.load(args.gate_weights_path, map_location=device))
    elif not args.skip_gate_training and HYPERPARAMS["gate_epochs"] > 0:
        gate_history = train_gate_branch(
            cnn_model=cnn_model,
            train_loader=train_loader,
            val_loader=validation_loader,
            active_mask=criterion.active_mask,
            device=device,
            epochs=HYPERPARAMS["gate_epochs"],
            lr=HYPERPARAMS["gate_learning_rate"],
            weight_decay=weight_decay,
            grad_clip_max_norm=HYPERPARAMS["grad_clip_max_norm"],
            save_path=gate_save_path,
        )
        plot_gate_training(gate_history, save_path=gate_plot_path)
    else:
        print("\nSkipping Stage 1 gate pretraining.")

    # ── STAGE 2: FREEZE GATE AND TRAIN MAIN PDF CNN ──────────────────
    if HYPERPARAMS["freeze_gate"]:
        print("\nFreezing gate branch parameters for Stage 2 training.")
        cnn_model.freeze_gate_branch()
    else:
        print("\nGate branch remains trainable for Stage 2.")
        cnn_model.unfreeze_gate_branch()

    trainable_params = [p for p in cnn_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=learning_rate, weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=HYPERPARAMS["learning_rate"],
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.2 # Spend the first 20% of epochs warming up
    )

    epochs_array = []
    train_history = {
        "total": [],
        "wasserstein_total": [],
        "wasserstein_active": [],
        "wasserstein_inactive": [],
        "kl_total": [],
        "kl_active": [],
        "kl_inactive": [],
        "gate": [],
        "mean_temp": [],
        "emiss": [],
        "leak": [],
    }
    val_history = {
        "total": [],
        "wasserstein_total": [],
        "wasserstein_active": [],
        "wasserstein_inactive": [],
        "kl_total": [],
        "kl_active": [],
        "kl_inactive": [],
        "gate": [],
        "mean_temp": [],
        "emiss": [],
        "leak": [],
    }

    print(f"\n{'='*60}")
    print(f"STAGE 2: Training Main PDF CNN ({num_epochs} epochs, lr={learning_rate})")
    print(f"{'='*60}")

    # Training loop
    for epoch in tqdm(range(num_epochs), desc='Training Loop'):
        # Calculate a ramp factor from 0.0 to 1.0 over the first 100 epochs
        ramp = min(1.0, epoch / 100.0)

        cnn_model.train()

        for inputs, labels, rho, temp in train_loader:

            # 1. Random horizontal flip (50% chance)
            if torch.rand(1).item() > 0.5:
                inputs = torch.flip(inputs, [3])
                labels = torch.flip(labels, [3])
                rho = torch.flip(rho, [3])
                temp = torch.flip(temp, [3])
                # ux (ch 2) points along x; negate after flipping x-axis
                inputs = inputs.clone()
                inputs[:, 2] = -inputs[:, 2]
                
            # 2. Random vertical flip (50% chance)
            if torch.rand(1).item() > 0.5:
                inputs = torch.flip(inputs, [2])
                labels = torch.flip(labels, [2])
                rho = torch.flip(rho, [2])
                temp = torch.flip(temp, [2])
                # uy (ch 3) points along y; negate after flipping y-axis
                inputs = inputs.clone()
                inputs[:, 3] = -inputs[:, 3]
            
            logits, gate = cnn_model(inputs)

            # Pass logits, gate, labels, rho, and temp to the gated loss
            loss = criterion(logits, gate, labels, rho, temp)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_params, max_norm=HYPERPARAMS["grad_clip_max_norm"]
            )
            optimizer.step()
            scheduler.step()

        cnn_model.eval()

        with torch.no_grad():
            train_totals = {k: 0.0 for k in train_history}
            for x_batch, y_batch, rho_batch, temp_batch in train_loader:
                logits_b, gate_b = cnn_model(x_batch)
                _, comp = criterion(
                    logits_b, gate_b, y_batch, rho_batch, temp_batch, return_components=True
                )
                for k, v in comp.items():
                    train_totals[k] += v

            n_train_batches = max(1, len(train_loader))
            for k in train_history:
                train_history[k].append(train_totals[k] / n_train_batches)

            val_totals = {k: 0.0 for k in val_history}
            for x_batch, y_batch, rho_batch, temp_batch in validation_loader:
                logits_b, gate_b = cnn_model(x_batch)
                _, comp = criterion(
                    logits_b, gate_b, y_batch, rho_batch, temp_batch, return_components=True
                )
                for k, v in comp.items():
                    val_totals[k] += v

            n_val_batches = max(1, len(validation_loader))
            for k in val_history:
                val_history[k].append(val_totals[k] / n_val_batches)

            train_loss = train_history["total"][-1]
            val_loss = val_history["total"][-1]

        epochs_array.append(epoch + 1)
        train_loss_arr = train_history["total"]
        val_loss_arr = val_history["total"]

        # Early stopping
        window_size = 200

        if len(val_loss_arr) >= window_size:

            val_loss_ma = np.convolve(
                val_loss_arr,
                np.ones(window_size)/window_size,
                mode='valid'
            )

            if len(val_loss_ma) > 1 and val_loss_ma[-1] > np.min(val_loss_ma[:-1]) and epoch >= 499:

                print(f"Early stopping at epoch {epoch+1}")
                break

    # Testing
    cnn_model.eval()

    with torch.no_grad():
        test_totals = {k: 0.0 for k in train_history}
        for x_batch, y_batch, rho_batch, temp_batch in test_loader:
            logits_b, gate_b = cnn_model(x_batch)
            _, comp = criterion(
                logits_b, gate_b, y_batch, rho_batch, temp_batch, return_components=True
            )
            for k, v in comp.items():
                test_totals[k] += v

        n_test_batches = max(1, len(test_loader))
        test_history = {k: test_totals[k] / n_test_batches for k in train_history}
        test_loss = test_history["total"]

    print(f"Test Total Loss: {test_loss:.6f}")
    for k, v in test_history.items():
        if k != "total":
            print(f"  Test {k}: {v:.6f}")

    # Save model
    torch.save(
        cnn_model.state_dict(),
        os.path.join(MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}.pth"),
    )

    # Save loss history data (.npz)
    history_save_path = os.path.join(
        LOSS_PLOT_DIR, f"cnn_{resolution}_{downsample}_loss_history.npz"
    )
    np.savez(
        history_save_path,
        epochs=np.array(epochs_array),
        train_total=np.array(train_history["total"]),
        val_total=np.array(val_history["total"]),
        train_wasserstein_total=np.array(train_history["wasserstein_total"]),
        val_wasserstein_total=np.array(val_history["wasserstein_total"]),
        train_wasserstein_active=np.array(train_history["wasserstein_active"]),
        val_wasserstein_active=np.array(val_history["wasserstein_active"]),
        train_wasserstein_inactive=np.array(train_history["wasserstein_inactive"]),
        val_wasserstein_inactive=np.array(val_history["wasserstein_inactive"]),
        # Aliases for backward compatibility
        train_kl_total=np.array(train_history["kl_total"]),
        val_kl_total=np.array(val_history["kl_total"]),
        train_kl_active=np.array(train_history["kl_active"]),
        val_kl_active=np.array(val_history["kl_active"]),
        train_kl_inactive=np.array(train_history["kl_inactive"]),
        val_kl_inactive=np.array(val_history["kl_inactive"]),
        train_gate=np.array(train_history["gate"]),
        val_gate=np.array(val_history["gate"]),
        train_mean_temp=np.array(train_history["mean_temp"]),
        val_mean_temp=np.array(val_history["mean_temp"]),
        train_emiss=np.array(train_history["emiss"]),
        val_emiss=np.array(val_history["emiss"]),
        train_leak=np.array(train_history["leak"]),
        val_leak=np.array(val_history["leak"]),
    )
    print(f"Saved loss history arrays to: {history_save_path}")

    # 1. Plot overall total loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_array, train_history["total"], label="Train Loss")
    plt.plot(epochs_array, val_history["total"], label="Validation Loss")
    plt.axhline(train_history["total"][-1], linestyle="--", label=f"Final Train: {train_history['total'][-1]:.4f}")
    plt.axhline(val_history["total"][-1], linestyle="--", label=f"Final Val: {val_history['total'][-1]:.4f}")
    plt.axhline(test_loss, linestyle="--", color="red", label=f"Test: {test_loss:.4f}")
    plt.xlabel("Epochs")
    plt.ylabel("PDF Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    loss_plot_file = os.path.join(LOSS_PLOT_DIR, f"cnn_{resolution}_{downsample}_loss.jpg")
    plt.savefig(loss_plot_file, dpi=500)
    plt.close()
    print(f"Saved total loss plot to: {loss_plot_file}")

    # 2. Plot detailed breakdown of all loss components
    components_plot_file = os.path.join(
        LOSS_PLOT_DIR, f"cnn_{resolution}_{downsample}_loss_components.jpg"
    )
    plot_loss_breakdown(
        epochs_array=epochs_array,
        train_history=train_history,
        val_history=val_history,
        test_history=test_history,
        save_path=components_plot_file,
        hyperparams=HYPERPARAMS,
    )
