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
HYPERPARAMS = {
    "seed": 10,
    "device": "mps",
    "resolution": (512, 256),
    "downsample": 32,
    "in_channels": 5,
    "out_channels": 40,
    "layer_size1": 32,
    "layer_size2": 64,
    "layer_size3": 128,
    "layer_size4": 256,
    "kernel_size": 7,
    "num_epochs": 1000,
    "print_every": 50,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-3,
    "dropout_rate": 0.3,
    "alpha_emiss": 30.0,
    "alpha_profile": 20.0,
    "alpha_gate": 10.0,
    "train_fraction": 0.50,
    "val_fraction": 0.25,
    "grad_clip_max_norm": 1.0,
}

np.random.seed(HYPERPARAMS["seed"])
device = torch.device(HYPERPARAMS["device"])
print("Using Apple MPS GPU")

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


# New Version that truncates to 10^4.5 to 10^5.5
def lambda_cool(temp, mask=False):
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
        mask_off = (logt < 4.5) | (logt > 5.5)
        lam[mask_off] = 0.0

    return lam


# =========================
# CENTRALIZED COOLING FUNCTION  (Change #1)
# =========================
def compute_cooling_rate(
    rho_or_pdf, temp, pressure=None, is_pdf=False, is_isobaric=False, T_unit=None
):
    """
    Standardized cooling calculation using internal Code Units.
    Both modes calculate an effective `rho_code` and pass it through the exact same physics.
    """
    mu = 0.62
    unit_fix = 1.975e27  # The grouped conversion (rho_0 * L_0) / (m_H^2 * v_0^3)

    if not is_pdf:
        # --- Mode 1: Fine-grid scalar path ---
        # We ALREADY have the code density.
        rho_eff = rho_or_pdf
        lam = lambda_cool(temp, mask=True)

        n_code = rho_eff / mu
        return lam * (n_code**2) * unit_fix

    else:
        # --- Mode 2: PDF-integrated path ---
        pdf = rho_or_pdf  # (nb, nx, ny)
        T_centers = temp  # (nb,)
        lam = lambda_cool(T_centers, mask=True)  # (nb,)

        if is_isobaric:
            if T_unit is None:
                raise ValueError("T_unit must be provided for isobaric calculation.")

            # Reconstruct the code density that WOULD exist at this temperature under isobaric assumption
            # Formula: rho_code = P_code * (T_unit / T_phys)
            # Shapes : (nx, ny)  * ( scalar / (nb,) ) -> (nb, nx, ny)
            rho_eff = pressure[None, :, :] * (T_unit / T_centers[:, None, None])
        else:
            raise ValueError("Non-isobaric PDF cooling not supported here.")

        n_code = rho_eff / mu

        # Now it is mathematically identical to the fine-grid path!
        cooling_per_bin = lam[:, None, None] * (n_code**2) * unit_fix

        return np.sum(pdf * cooling_per_bin, axis=0)  # (nx,ny)


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

        sim_data.cons_rho = np.load(f"{folder_path}/cons_rho.npy")
        sim_data.cons_momx = np.load(f"{folder_path}/cons_mx.npy")
        sim_data.cons_momy = np.load(f"{folder_path}/cons_my.npy")
        sim_data.cons_ener = np.load(f"{folder_path}/cons_ener.npy")
        sim_data.cons_ps = np.load(f"{folder_path}/cons_ps.npy")
    else:
        sim_data.input_data(file_path, start=501)
        sim_data.input_cons_data(file_path, start=501)
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

    cnn_model = ConvNN(
        in_channels,
        layer_size1,
        layer_size2,
        layer_size3,
        layer_size4,
        out_channels,
        kernel_size,
    ).to(device)

    cnn_model.load_state_dict(torch.load(model_path, map_location=device))
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
) -> np.ndarray:
    """
    Predict pixel temperature PDFs for a snapshot whose fields are ALREADY
    coarse-grained to 16×8. No `simulation_data.coarse_grain` is performed.

    Parameters
    ----------
    rho, temp, ux, uy, ps : np.ndarray, shape (16, 8)
        Already-coarse input fields.
    fine_resolution : tuple, default (1024, 512)
        Fine-grid resolution the saved model was trained against. Used ONLY
        to build the model/normalization file names. Since 16×8 =
        (1024/64, 512/64), the defaults are (1024, 512) and downsample=64.
    downsample : int, default 64
        Downsample factor that produced the 16×8 grid; used only for naming.

    Returns
    -------
    pdf : np.ndarray, shape (out_channels, 16, 8)  ≡ (40, 16, 8)
        Predicted temperature PDF at each coarse cell. Axis 0 indexes the
        40 temperature bins defined by `T_edges`.
    """

    # ---- 1. Shape check -------------------------------------------------
    expected_shape = (16, 8)
    fields = {"rho": rho, "temp": temp, "ux": ux, "uy": uy, "ps": ps}
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
    # File-name convention matches the trainer: f"cnn_{resolution}_{downsample}_*"
    norm_prefix = f"cnn_{fine_resolution}_{downsample}"

    input_mean = torch.tensor(
        np.load(os.path.join(MODEL_SAVE_DIR, f"{norm_prefix}_input_mean.npy")),
        dtype=torch.float32,
    ).to(device)
    input_std = torch.tensor(
        np.load(os.path.join(MODEL_SAVE_DIR, f"{norm_prefix}_input_std.npy")),
        dtype=torch.float32,
    ).to(device)

    # The trainer saved them with shape (1, C, 1, 1); be defensive.
    if input_mean.dim() == 1:
        input_mean = input_mean.view(1, -1, 1, 1)
        input_std = input_std.view(1, -1, 1, 1)

    input_tensor = (input_tensor - input_mean) / input_std

    # ---- 4. Load model --------------------------------------------------
    model_path = os.path.join(MODEL_SAVE_DIR, f"{norm_prefix}.pth")
    cnn_model = ConvNN(
        in_channels,
        layer_size1,
        layer_size2,
        layer_size3,
        layer_size4,
        out_channels,
        kernel_size,
    ).to(device)
    cnn_model.load_state_dict(torch.load(model_path, map_location=device))
    cnn_model.eval()

    # ---- 5. Predict PDF -------------------------------------------------
    with torch.no_grad():
        pdf = cnn_model.predict_pdf(input_tensor)  # (1, 40, 16, 8)
        pdf = pdf[0].cpu().numpy()  # (40, 16, 8)

    return pdf


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

    cnn_model = ConvNN(
        in_channels,
        layer_size1,
        layer_size2,
        layer_size3,
        layer_size4,
        out_channels,
        kernel_size,
    ).to(device)

    cnn_model.load_state_dict(torch.load(model_path, map_location=device))
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

    def __init__(self, threshold=1e-4, eps=1e-12):
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
      ch 6: ρ|ω|                             (densimetric vorticity, weighs shear by inertia)
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
                rho.abs() * omega.abs(),  # ch 6
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
    Vorticity-gated PDF activation.

    When gate ≈ 0: PDF collapses to a near-delta function at the argmax bin
                   (single-phase cell — no sub-grid mixing).
    When gate ≈ 1: PDF is the full ThresholdedSoftmax output
                   (highly mixed cell).

    Interpolation:
        gated = gate * p_thresh + (1 - gate) * delta_peak
    followed by renormalization so the output always sums to 1.
    """

    def __init__(self, threshold=1e-3, eps=1e-12):
        super().__init__()
        self.threshold = threshold
        self.eps = eps

    def forward(self, logits, gate):
        # gate: (B, 1, H, W), broadcast over bin dim
        p = F.softmax(logits, dim=1)  # (B, bins, H, W)
        p = p * (p >= self.threshold).float()
        p = p / (p.sum(dim=1, keepdim=True) + self.eps)

        # Build delta function at argmax bin
        peak_idx = torch.argmax(p, dim=1, keepdim=True)  # (B, 1, H, W)
        delta = torch.zeros_like(p).scatter_(1, peak_idx, 1.0)

        # Interpolate: gate=0 → delta, gate=1 → full PDF
        gated = gate * p + (1.0 - gate) * delta

        # Renormalize
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
            nn.Dropout(dropout_rate),
            nn.Conv2d(layer_size1, layer_size2, kernel_size, padding=padding),
            nn.BatchNorm2d(layer_size2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
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


class WassersteinLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = ThresholdedSoftmax()

    def forward(self, logits, target):
        """
        logits: (B, bins, nx, ny) — raw logits from model
        target: (B, bins, nx, ny) — must be normalized PDF
        """
        pred_pdf = self.activation(logits)

        # Compute CDF along bin axis
        cdf_pred = torch.cumsum(pred_pdf, dim=1)
        cdf_target = torch.cumsum(target, dim=1)

        # Wasserstein-1 distance
        loss = torch.mean(torch.abs(cdf_pred - cdf_target))

        return loss


class KLWithLeakageLoss(nn.Module):
    def __init__(self, alpha=0, T0=1e6, width=0.1):
        super().__init__()
        self.alpha = alpha
        self.activation = ThresholdedSoftmax()

        # store logT info
        self.logT_centers = logT_centers
        self.logT0 = np.log10(T0)
        self.width = width

    def forward(self, logits, target):
        pred_pdf = self.activation(logits)
        log_probs = torch.log(pred_pdf + 1e-12)

        # expand weights
        weights = lambda_weights.to(target.device)[None, :, None, None]

        # weighted KL
        kl_elementwise = target * (torch.log(target + 1e-12) - log_probs)
        weighted_kl = kl_elementwise * weights

        # Step 1: calculate the KLdivergence loss per pixel
        kl_per_pixel = torch.sum(weighted_kl, dim=1)
        # Step 2: calculate the mean across all the pixels
        kl_loss = torch.mean(kl_per_pixel)

        # Peak bin index from TRUE PDF
        peak_idx = torch.argmax(target, dim=1)  # (B, nx, ny)

        # Get logT of peak bin
        logT_peak = self.logT_centers.to(target.device)[peak_idx]

        # Temperature mask (Gaussian around 1e6 K)
        temp_mask = torch.exp(-((logT_peak - self.logT0) ** 2) / (2 * self.width**2))

        # Predicted mass at peak
        peak_prob = torch.gather(pred_pdf, 1, peak_idx.unsqueeze(1)).squeeze(1)

        # True peak value
        true_peak = torch.max(target, dim=1).values  # (B, nx, ny)

        # CONDITION: sharp + near 1e6 K
        condition = (temp_mask > 0.5) & (true_peak > 0.9)
        final_mask = condition.float()

        # Leakage = true - pred (only where condition holds)
        leakage = torch.clamp(true_peak - peak_prob, min=0.0)

        # Apply mask
        masked_leakage = leakage * final_mask

        leakage_loss = torch.mean(masked_leakage)

        return kl_loss + self.alpha * leakage_loss


# Introducing Emissivity into the loss function
# Function to Define the Emissivity


def emissivity_from_pdf(pdf, rho, lambda_tensor):
    """
    pdf : (B,bins,nx,ny)
    rho : (B,1,nx,ny)
    """

    cooling = lambda_tensor.to(pdf.device)

    cooling = cooling.view(1, -1, 1, 1)

    mean_lambda = torch.sum(pdf * cooling, dim=1, keepdim=True)

    emiss = rho**2 * mean_lambda

    return emiss


class PDFEmissivityLoss(nn.Module):
    def __init__(self, alpha_emiss=1.0, alpha_profile=1.0):
        super().__init__()

        self.alpha_emiss = alpha_emiss
        self.alpha_profile = alpha_profile

        self.activation = ThresholdedSoftmax()

    def forward(self, logits, true_pdf, rho):

        pred_pdf = self.activation(logits)

        # --- PDF loss ---
        # Step 1: calculate the KLdivergence loss per pixel
        kl_elementwise = true_pdf * (
            torch.log(true_pdf + 1e-12) - torch.log(pred_pdf + 1e-12)
        )
        kl_per_pixel = torch.sum(kl_elementwise, dim=1)
        # Step 2: calculate the mean across all the pixels
        pdf_loss = torch.mean(kl_per_pixel)

        # --- Emissivity maps: shape (B, 1, nx, ny) ---
        emiss_pred = emissivity_from_pdf(pred_pdf, rho, lambda_tensor)
        emiss_true = emissivity_from_pdf(true_pdf, rho, lambda_tensor)

        # BEFORE: collapsed to a single scalar per batch via max()
        # max_emiss_pred = torch.amax(emiss_pred, dim=(2,3))  ← discards spatial info

        # AFTER: keep all spatial cells, compute MSE across every (b, i, j)
        emiss_loss_pixelwise = F.mse_loss(
            torch.log10(emiss_pred + 1e-30), torch.log10(emiss_true + 1e-30)
        )
        # F.mse_loss with default reduction='mean' averages over B*1*nx*ny automatically

        # Keep profile loss if you want — it's cheap and constrains large-scale structure
        profile_pred = emiss_pred.mean(dim=3)  # (B, 1, nx)
        profile_true = emiss_true.mean(dim=3)
        profile_loss = F.mse_loss(
            torch.log10(profile_pred + 1e-30), torch.log10(profile_true + 1e-30)
        )

        total_loss = (
            pdf_loss
            + self.alpha_emiss * emiss_loss_pixelwise
            + self.alpha_profile * profile_loss
        )

        return total_loss


class GatedPDFEmissivityLoss(nn.Module):
    """
    Extends PDFEmissivityLoss with an explicit gate supervision term.

    Gate target is derived from the entropy of the true PDF:
        - High entropy (broad, multi-phase)  → gate_target = 1
        - Low entropy  (sharp, single-phase) → gate_target = 0

    Total loss = KL(pred ‖ true) + α_emiss * emiss_MSE
                 + α_profile * profile_MSE + α_gate * BCE(gate, gate_target)
    """

    def __init__(
        self,
        alpha_emiss=1.0,
        alpha_profile=1.0,
        alpha_gate=1.0,
        entropy_threshold=0.1,
    ):
        super().__init__()
        self.alpha_emiss = alpha_emiss
        self.alpha_profile = alpha_profile
        self.alpha_gate = alpha_gate
        self.entropy_threshold = entropy_threshold

        self.activation = GatedThresholdedSoftmax()

    def forward(self, logits, gate, true_pdf, rho):

        pred_pdf = self.activation(logits, gate)

        # --- PDF loss ---
        # Step 1: calculate the KLdivergence loss per pixel
        kl_elementwise = true_pdf * (
            torch.log(true_pdf + 1e-12) - torch.log(pred_pdf + 1e-12)
        )
        kl_per_pixel = torch.sum(kl_elementwise, dim=1)
        # Step 2: calculate the mean across all the pixels
        pdf_loss = torch.mean(kl_per_pixel)

        # --- Emissivity maps: shape (B, 1, nx, ny) ---
        emiss_pred = emissivity_from_pdf(pred_pdf, rho, lambda_tensor)
        emiss_true = emissivity_from_pdf(true_pdf, rho, lambda_tensor)

        emiss_loss = F.mse_loss(
            torch.log10(emiss_pred + 1e-30), torch.log10(emiss_true + 1e-30)
        )

        profile_pred = emiss_pred.mean(dim=3)  # (B, 1, nx)
        profile_true = emiss_true.mean(dim=3)
        profile_loss = F.mse_loss(
            torch.log10(profile_pred + 1e-30), torch.log10(profile_true + 1e-30)
        )

        # --- Gate supervision via true-PDF entropy ---
        # entropy ∈ [0, log(bins)]; normalize to [0, 1]
        entropy = -(true_pdf * torch.log(true_pdf + 1e-12)).sum(dim=1, keepdim=True)
        entropy_norm = entropy / np.log(true_pdf.shape[1])
        gate_target = (entropy_norm > self.entropy_threshold).float()

        gate_loss = F.binary_cross_entropy(gate, gate_target)

        return (
            pdf_loss
            + self.alpha_emiss * emiss_loss
            + self.alpha_profile * profile_loss
            + self.alpha_gate * gate_loss
        )


if __name__ == "__main__":
    file_path = DATA_PATH

    print("Training all fluxes model")

    # torch.cuda.empty_cache()

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

    criterion = GatedPDFEmissivityLoss(
        alpha_emiss=HYPERPARAMS["alpha_emiss"],
        alpha_profile=HYPERPARAMS["alpha_profile"],
        alpha_gate=HYPERPARAMS["alpha_gate"],
    )
    # criterion = nn.KLDivLoss(reduction="batchmean")
    # criterion = KLWithLeakageLoss()
    # criterion = WassersteinLoss()

    optimizer = torch.optim.Adam(
        cnn_model.parameters(), lr=learning_rate, weight_decay=weight_decay
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

    # Store raw (un-normalized) rho as a third tensor so the emissivity
    # loss always receives the physical density, not the z-scored one.
    rho_tensor = input_tensor[:, 0:1]  # (N, 1, nx, ny), un-normalized
    dataset = TensorDataset(input_tensor_norm, output_tensor, rho_tensor)

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

    epochs_array = []
    train_loss_arr = []
    val_loss_arr = []

    # Training loop
    for epoch in range(num_epochs):
        cnn_model.train()

        for inputs, labels, rho in train_loader:
            logits, gate = cnn_model(inputs)

            # rho is the un-normalized physical density from the dataset
            # Pass logits, gate, labels, and rho to the gated loss
            loss = criterion(logits, gate, labels, rho)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                cnn_model.parameters(), max_norm=HYPERPARAMS["grad_clip_max_norm"]
            )
            optimizer.step()

        cnn_model.eval()

        with torch.no_grad():
            train_loss_total = 0
            val_loss_total = 0

            # Train evaluation
            for x_batch, y_batch, rho_batch in train_loader:
                logits_b, gate_b = cnn_model(x_batch)

                train_loss_total += criterion(
                    logits_b, gate_b, y_batch, rho_batch
                ).item()

            train_loss = train_loss_total / len(train_loader)

            # Validation evaluation
            for x_batch, y_batch, rho_batch in validation_loader:
                logits_b, gate_b = cnn_model(x_batch)

                val_loss_total += criterion(logits_b, gate_b, y_batch, rho_batch).item()

            val_loss = val_loss_total / len(validation_loader)

        if (epoch + 1) % print_every == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )

        epochs_array.append(epoch + 1)
        train_loss_arr.append(train_loss)
        val_loss_arr.append(val_loss)

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
        test_loss_total = 0

        for x_batch, y_batch, rho_batch in test_loader:
            logits_b, gate_b = cnn_model(x_batch)

            test_loss_total += criterion(logits_b, gate_b, y_batch, rho_batch).item()

        test_loss = test_loss_total / len(test_loader)

    print(f"Test Loss: {test_loss:.6f}")

    # Save model
    torch.save(
        cnn_model.state_dict(),
        os.path.join(MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}.pth"),
    )

    # Plot loss
    plt.figure(figsize=(10, 5))

    plt.plot(epochs_array, train_loss_arr, label="Train Loss")
    plt.plot(epochs_array, val_loss_arr, label="Validation Loss")

    plt.axhline(train_loss_arr[-1], linestyle="--")
    plt.axhline(val_loss_arr[-1], linestyle="--")
    plt.axhline(test_loss, linestyle="--", color="red")

    plt.xlabel("Epochs")
    # plt.ylabel("KL Divergence")
    # plt.ylabel("Wasserstein Loss")
    # plt.ylabel("KL Divergence")
    plt.ylabel("PDF + Emissivity Loss")
    plt.title("Training Loss")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(LOSS_PLOT_DIR, f"cnn_{resolution}_{downsample}_loss.jpg"), dpi=500
    )

    plt.close()
