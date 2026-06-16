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

import numpy as np
import matplotlib.pyplot as plt
import torch
torch.cuda.empty_cache()
# pyrefly: ignore [missing-import]
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data')))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
MODEL_SAVE_DIR = os.environ.get("MODEL_SAVES_DIR", os.path.join(PROJECT_ROOT, 'outputs', 'model_saves', 'pdf_model_saves'))
LOSS_PLOT_DIR = os.environ.get("LOSS_PLOTS_DIR", os.path.join(PROJECT_ROOT, 'outputs', 'loss_plots', 'pdf_loss_plots'))
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOSS_PLOT_DIR, exist_ok=True)
# TODO: Set these to your local simulation data directories
DATA_PATH = os.environ.get('SUBGRID_DATA_PATH', '/path/to/simulation/bin')
CACHE_PATH = os.environ.get('SUBGRID_CACHE_PATH', '/path/to/cache')
import data_preprocess
from data_preprocess import simulation_data

np.random.seed(10)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
#print("Using CPU")
device = torch.device("mps")
print("Using Apple MPS GPU")


resolution = (1024, 512)  
downsample = 64
in_channels = 5
out_channels = 40
layer_size1 = 32
layer_size2 = 64
layer_size3 = 128
layer_size4 = 256
kernel_size = 5
num_epochs = 1000
print_every = 50
batch_size = 64
learning_rate = 1e-3
weight_decay = 1e-3
dropout_rate = 0.3

T_edges = np.logspace(3.0, 7.0, out_channels + 1)
T_centers = np.sqrt(T_edges[:-1] * T_edges[1:])

logT_centers = torch.log10(torch.tensor(T_centers, dtype=torch.float32))

def lambda_cool(temp):
    """
    Cooling function ISMCoolFn translated from AthenaK C++.
    Works on scalars or numpy arrays (any shape).
    Returns Λ(T) in erg cm^3 / s.
    """
    logt = np.log10(temp)

    lhd = np.array([
        -22.5977, -21.9689, -21.5972, -21.4615, -21.4789, -21.5497, -21.6211, -21.6595,
        -21.6426, -21.5688, -21.4771, -21.3755, -21.2693, -21.1644, -21.0658, -20.9778,
        -20.8986, -20.8281, -20.7700, -20.7223, -20.6888, -20.6739, -20.6815, -20.7051,
        -20.7229, -20.7208, -20.7058, -20.6896, -20.6797, -20.6749, -20.6709, -20.6748,
        -20.7089, -20.8031, -20.9647, -21.1482, -21.2932, -21.3767, -21.4129, -21.4291,
        -21.4538, -21.5055, -21.5740, -21.6300, -21.6615, -21.6766, -21.6886, -21.7073,
        -21.7304, -21.7491, -21.7607, -21.7701, -21.7877, -21.8243, -21.8875, -21.9738,
        -22.0671, -22.1537, -22.2265, -22.2821, -22.3213, -22.3462, -22.3587, -22.3622,
        -22.3590, -22.3512, -22.3420, -22.3342, -22.3312, -22.3346, -22.3445, -22.3595,
        -22.3780, -22.4007, -22.4289, -22.4625, -22.4995, -22.5353, -22.5659, -22.5895,
        -22.6059, -22.6161, -22.6208, -22.6213, -22.6184, -22.6126, -22.6045, -22.5945,
        -22.5831, -22.5707, -22.5573, -22.5434, -22.5287, -22.5140, -22.4992, -22.4844,
        -22.4695, -22.4543, -22.4392, -22.4237, -22.4087, -22.3928
    ])

    lam = np.zeros_like(temp, dtype=float)

    # turn off cooling below 1e4 K
    mask_off = logt <= 4.0
    lam[mask_off] = 0.0

    # KI02 regime (4.0 < logT <= 4.2)
    mask_ki = (logt > 4.0) & (logt <= 4.2)
    if np.any(mask_ki):
        lam[mask_ki] = (2.0e-19*np.exp(-1.184e5/(temp[mask_ki] + 1.0e3)) +
                        2.8e-28*np.sqrt(temp[mask_ki])*np.exp(-92.0/temp[mask_ki]))

    # CGOLS fit (logT > 8.15)
    mask_hi = logt > 8.15
    lam[mask_hi] = 10.0**(0.45*logt[mask_hi] - 26.065)

    # SPEX interpolation (4.2 < logT <= 8.15)
    mask_mid = (logt > 4.2) & (logt <= 8.15)
    if np.any(mask_mid):
        ipps = (25.0*logt[mask_mid] - 103).astype(int)
        # Clamp to [0,100] like C++
        ipps = np.clip(ipps, 0, 100)
        x0 = 4.12 + 0.04*ipps
        dx = logt[mask_mid] - x0
        logcool = (lhd[ipps+1]*dx - lhd[ipps]*(dx - 0.04)) * 25.0
        lam[mask_mid] = 10.0**logcool

    return lam

lambda_vals = lambda_cool(T_centers)

# take log safely
log_lambda = np.log10(lambda_vals + 1e-40)

# normalize to [0,1]
log_lambda -= log_lambda.min()
log_lambda /= (log_lambda.max() + 1e-30)

lambda_weights = torch.tensor(log_lambda, dtype=torch.float32)
lambda_tensor = torch.tensor(lambda_vals, dtype=torch.float32)

def nn_data(resolution: tuple, downsample: int) -> tuple:
    """ A function to load the data and return the inputs and outputs for the Conv neural network."""

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
        sim_data.input_data(file_path, start = 501)
        sim_data.input_cons_data(file_path, start = 501)
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

    shape = (sim_data.rho.shape[0], sim_data.rho.shape[1] // sim_data.down_sample, sim_data.rho.shape[2] // sim_data.down_sample)
    fields = ['rho', 'temp', 'ux', 'uy', 'ps']
    cg = {f'cg_{field}': np.zeros(shape) for field in fields}

    for i in range(sim_data.rho.shape[0]):
        for field in fields:
            if field in ['rho', 'temp', 'ux', 'uy', 'ps']:
                cg[f'cg_{field}'][i] = sim_data.coarse_grain(getattr(sim_data, field)[i])
    temp_pdf = sim_data.calc_pixel_pdf(bins = out_channels)
    temp_pdf /= temp_pdf.sum(axis=1, keepdims=True)

    input_tensors = [torch.from_numpy(cg[f'cg_{f}']).unsqueeze(1).float() for f in fields]
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
    resolution: np.ndarray
) -> np.ndarray:
    """
    Predict pixel temperature PDFs for a given snapshot.
    Returns: (bins, nx, ny)
    """

    sim_data = simulation_data()
    sim_data.down_sample = downsample
    sim_data.resolution = resolution

    shape = (resolution[0] // downsample, resolution[1] // downsample)

    fields = ['rho', 'temp', 'ux', 'uy', 'ps']
    cg = {f'cg_{field}': np.zeros(shape) for field in fields}

    # -------------------------
    # Coarse-grain inputs
    # -------------------------
    for field in fields:
        if field in ['rho', 'temp', 'ux', 'uy', 'ps']:
            cg[f'cg_{field}'] = sim_data.coarse_grain(locals()[field])

    # -------------------------
    # Build input tensor
    # -------------------------
    input_tensors = [
        torch.from_numpy(cg[f'cg_{f}']).unsqueeze(0).float()
        for f in fields
    ]

    input_tensor = torch.cat(input_tensors, dim=0)   # (C, nx, ny)
    input_tensor = input_tensor.unsqueeze(0).to(device)         # (1, C, nx, ny)


    # -------------------------
    # Normalize input (IMPORTANT)
    # -------------------------
    # Load and convert directly to tensors on the MPS device
    input_mean = torch.tensor(np.load(
        os.path.join(MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}_input_mean.npy")
    ), dtype=torch.float32).to(device)
    
    input_std = torch.tensor(np.load(
        os.path.join(MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}_input_std.npy")
    ), dtype=torch.float32).to(device)

    # Now both are MPS tensors, this math works seamlessly
    input_tensor = (input_tensor - input_mean) / input_std

    # -------------------------
    # Load model
    # -------------------------
    model_path = os.path.join(MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}.pth")

    cnn_model = ConvNN(
        in_channels, layer_size1, layer_size2,
        layer_size3, layer_size4, out_channels, kernel_size
    ).to(device)

    cnn_model.load_state_dict(torch.load(model_path, map_location=device))
    cnn_model.eval()

    # -------------------------
    # Predict PDF
    # -------------------------
    with torch.no_grad():

        pdf = cnn_model.predict_pdf(input_tensor)   # (1, bins, nx, ny)

        pdf = pdf[0].cpu().numpy()  # (bins, nx, ny)

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
    resolution: np.ndarray
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

    fields = ['rho', 'temp', 'ux', 'uy', 'ps']
    cg = {f'cg_{field}': np.zeros(shape) for field in fields}

    for field in fields:
        if field in ['rho', 'temp', 'ux', 'uy', 'ps']:
            cg[f'cg_{field}'] = sim_data.coarse_grain(locals()[field])

    input_tensors = [
        torch.from_numpy(cg[f'cg_{f}']).unsqueeze(0).float()
        for f in fields
    ]

    input_tensor = torch.cat(input_tensors, dim=0)   # (C, nx, ny)
    input_tensor = input_tensor.unsqueeze(0).to(device)  # (1, C, nx, ny)

    input_mean = torch.tensor(np.load(
        os.path.join(MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}_input_mean.npy")
    ), dtype=torch.float32).to(device)

    input_std = torch.tensor(np.load(
        os.path.join(MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}_input_std.npy")
    ), dtype=torch.float32).to(device)

    input_tensor = (input_tensor - input_mean) / input_std

    model_path = os.path.join(MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}.pth")

    cnn_model = ConvNN(
        in_channels, layer_size1, layer_size2,
        layer_size3,layer_size4, out_channels, kernel_size
    ).to(device)

    cnn_model.load_state_dict(torch.load(model_path, map_location=device))
    cnn_model.eval()

    with torch.no_grad():
        # Run the full forward pass to get logits + gate
        x_with_vort = cnn_model.vorticity(input_tensor)         # (1, C+1, H, W)
        vort_mag = x_with_vort[:, -1:, :, :].abs()              # (1, 1, H, W)
        gate_raw = cnn_model.gate_branch(vort_mag)               # (1, 1, H, W)
        features = cnn_model.encoder(x_with_vort)
        logits = cnn_model.decoder(features)
        pdf_tensor = cnn_model.pdf_activation(logits, gate_raw)  # (1, bins, H, W)

        pdf          = pdf_tensor[0].cpu().numpy()               # (bins, nx, ny)
        gate         = gate_raw[0, 0].cpu().numpy()              # (nx, ny)
        vorticity_mag = vort_mag[0, 0].cpu().numpy()             # (nx, ny)

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
        p = F.softmax(logits, dim=1)          # (B, bins, nx, ny), sums to 1

        # Step 2: threshold — bins below `threshold` become exactly 0
        p = p * (p >= self.threshold).float()

        # Step 3: re-normalize so survivors still sum to 1
        return p / (p.sum(dim=1, keepdim=True) + self.eps)


class VorticityGate(nn.Module):
    """
    Learns a spatial gate g(x,y) ∈ [0,1] from vorticity magnitude.

    g ≈ 0 → single-phase cell (PDF collapses to peak bin delta)
    g ≈ 1 → mixing cell (full broad PDF is allowed)
    """
    def __init__(self, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.gate_net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid(),   # output ∈ (0, 1)
        )

    def forward(self, vorticity_mag):
        # vorticity_mag: (B, 1, H, W)
        return self.gate_net(vorticity_mag)   # (B, 1, H, W)


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
    def __init__(self, threshold=1e-4, eps=1e-12):
        super().__init__()
        self.threshold = threshold
        self.eps = eps

    def forward(self, logits, gate):
        # gate: (B, 1, H, W), broadcast over bin dim
        p = F.softmax(logits, dim=1)            # (B, bins, H, W)
        p = p * (p >= self.threshold).float()
        p = p / (p.sum(dim=1, keepdim=True) + self.eps)

        # Build delta function at argmax bin
        peak_idx = torch.argmax(p, dim=1, keepdim=True)   # (B, 1, H, W)
        delta = torch.zeros_like(p).scatter_(1, peak_idx, 1.0)

        # Interpolate: gate=0 → delta, gate=1 → full PDF
        gated = gate * p + (1.0 - gate) * delta

        # Renormalize
        return gated / (gated.sum(dim=1, keepdim=True) + self.eps)

class VorticityLayer(nn.Module):
    """
    Computes 2D vorticity from velocity components (ux, uy) using finite differences.
    
    Vorticity ω = ∂uy/∂x - ∂ux/∂y
    
    Input: (B, C, H, W) where C >= 2 and channels [..., ux_idx, uy_idx, ...]
    Output: (B, C+1, H, W) — original channels + vorticity appended as last channel
    """
    def __init__(self, ux_idx=2, uy_idx=3):
        super().__init__()
        # Sobel-like kernels for finite differences
        # ∂/∂x kernel (detect horizontal gradients)
        self.register_buffer('dx_kernel', torch.tensor([
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]
        ], dtype=torch.float32).unsqueeze(0))  # (1, 1, 3, 3)
        
        # ∂/∂y kernel (detect vertical gradients)  
        self.register_buffer('dy_kernel', torch.tensor([
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]]
        ], dtype=torch.float32).unsqueeze(0))  # (1, 1, 3, 3)
        
        self.ux_idx = ux_idx
        self.uy_idx = uy_idx
        self.padding = 1
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        ux = x[:, self.ux_idx:self.ux_idx+1, :, :]   # (B, 1, H, W)
        uy = x[:, self.uy_idx:self.uy_idx+1, :, :]   # (B, 1, H, W)
        
        # Compute gradients using conv2d
        duy_dx = F.conv2d(uy, self.dx_kernel, padding=self.padding, groups=1)
        dux_dy = F.conv2d(ux, self.dy_kernel, padding=self.padding, groups=1)
        
        # Vorticity = ∂uy/∂x - ∂ux/∂y
        vorticity = duy_dx - dux_dy  # (B, 1, H, W)
        
        # Concatenate as new channel
        return torch.cat([x, vorticity], dim=1)  # (B, C+1, H, W)

class ConvNN(nn.Module):
    """
    CNN Model for PDF prediction (with Vorticity layer + Vorticity Gate)

    Architecture:
    Input: (B, 5, 16, 8)  [rho, temp, ux, uy, ps]
             │
        ┌────▼────────────┐
        │  VorticityLayer │  ω = ∂uy/∂x - ∂ux/∂y  →  appended as 6th channel
        │    5 → 6        │
        └────┬────────────┘
             │           │
        ┌────▼────────┐  └──── vort_mag ────►  ┌──────────────┐
        │   Encoder   │  3× Conv2d + BN + ReLU │ VorticityGate│
        │ 6→32→64→128 │  (+ Dropout in 1st 2)  │  gate ∈(0,1) │
        └────┬────────┘                        └──────┬───────┘
             │                                         │
        ┌────▼─────────┐                               │
        │   Decoder    │  3× Conv2d + BN + ReLU        │
        │ 128→64→32→40 │                               │
        └────┬─────────┘                               │
             │  logits                                 │ gate
             └────────────────┬────────────────────────┘
                              ▼
                    forward returns (logits, gate)
                    predict_pdf applies GatedThresholdedSoftmax
    """
    def __init__(self, in_channels, layer_size1, layer_size2, layer_size3, layer_size4, out_channels, kernel_size):

        super().__init__()
        padding = kernel_size // 2

        # Vorticity layer: automatically computes ω from ux, uy and appends it
        self.vorticity = VorticityLayer(ux_idx=2, uy_idx=3)

        # Lightweight gate branch: takes |ω| and outputs g ∈ (0,1) per cell
        self.gate_branch = VorticityGate(kernel_size)

        # Encoder now takes in_channels + 1 because vorticity adds a channel
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels + 1, layer_size1, kernel_size, padding=padding),
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
            nn.ReLU()
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
        x_with_vort = self.vorticity(x)              # (B, C+1, H, W)

        # Gate from vorticity magnitude only
        vort_mag = x_with_vort[:, -1:, :, :]  # (B, 1, H, W)
        gate = self.gate_branch(vort_mag)            # (B, 1, H, W)

        # Main prediction path
        features = self.encoder(x_with_vort)
        logits = self.decoder(features)

        return logits, gate   # both needed for the gated loss

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
        temp_mask = torch.exp(
            -((logT_peak - self.logT0) ** 2) / (2 * self.width ** 2)
        )

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


def emissivity_from_pdf(
    pdf,
    rho,
    lambda_tensor
):
    """
    pdf : (B,bins,nx,ny)
    rho : (B,1,nx,ny)
    """

    cooling = lambda_tensor.to(pdf.device)

    cooling = cooling.view(
        1,
        -1,
        1,
        1
    )

    mean_lambda = torch.sum(
        pdf * cooling,
        dim=1,
        keepdim=True
    )

    emiss = rho**2 * mean_lambda

    return emiss


class PDFEmissivityLoss(nn.Module):

    def __init__(
        self,
        alpha_emiss=1.0,
        alpha_profile=1.0
    ):
        super().__init__()

        self.alpha_emiss = alpha_emiss
        self.alpha_profile = alpha_profile

        self.activation = ThresholdedSoftmax()

    def forward(self, logits, true_pdf, rho):

        pred_pdf = self.activation(logits)

        # --- PDF loss ---
        # Step 1: calculate the KLdivergence loss per pixel
        kl_elementwise = true_pdf * (torch.log(true_pdf + 1e-12) - torch.log(pred_pdf + 1e-12))
        kl_per_pixel = torch.sum(kl_elementwise, dim=1)
        # Step 2: calculate the mean across all the pixels
        pdf_loss = torch.mean(kl_per_pixel)

        # --- Emissivity maps: shape (B, 1, nx, ny) ---
        emiss_pred = emissivity_from_pdf(pred_pdf, rho, lambda_tensor)
        emiss_true  = emissivity_from_pdf(true_pdf,    rho, lambda_tensor)

        # BEFORE: collapsed to a single scalar per batch via max()
        # max_emiss_pred = torch.amax(emiss_pred, dim=(2,3))  ← discards spatial info
        
        # AFTER: keep all spatial cells, compute MSE across every (b, i, j)
        emiss_loss_pixelwise = F.mse_loss(
            torch.log10(emiss_pred + 1e-30),
            torch.log10(emiss_true  + 1e-30)
        )
        # F.mse_loss with default reduction='mean' averages over B*1*nx*ny automatically

        # Keep profile loss if you want — it's cheap and constrains large-scale structure
        profile_pred = emiss_pred.mean(dim=3)   # (B, 1, nx)
        profile_true  = emiss_true.mean(dim=3)
        profile_loss = F.mse_loss(
            torch.log10(profile_pred + 1e-30),
            torch.log10(profile_true  + 1e-30)
        )

        total_loss = (
            pdf_loss
            + self.alpha_emiss   * emiss_loss_pixelwise
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
        kl_elementwise = true_pdf * (torch.log(true_pdf + 1e-12) - torch.log(pred_pdf + 1e-12))
        kl_per_pixel = torch.sum(kl_elementwise, dim=1)
        # Step 2: calculate the mean across all the pixels
        pdf_loss = torch.mean(kl_per_pixel)

        # --- Emissivity maps: shape (B, 1, nx, ny) ---
        emiss_pred = emissivity_from_pdf(pred_pdf, rho, lambda_tensor)
        emiss_true  = emissivity_from_pdf(true_pdf, rho, lambda_tensor)

        emiss_loss = F.mse_loss(
            torch.log10(emiss_pred + 1e-30),
            torch.log10(emiss_true  + 1e-30)
        )

        profile_pred = emiss_pred.mean(dim=3)   # (B, 1, nx)
        profile_true  = emiss_true.mean(dim=3)
        profile_loss = F.mse_loss(
            torch.log10(profile_pred + 1e-30),
            torch.log10(profile_true  + 1e-30)
        )

        # --- Gate supervision via true-PDF entropy ---
        # entropy ∈ [0, log(bins)]; normalize to [0, 1]
        entropy = -(true_pdf * torch.log(true_pdf + 1e-12)).sum(dim=1, keepdim=True)
        entropy_norm = entropy / np.log(true_pdf.shape[1])
        gate_target = (entropy_norm > self.entropy_threshold).float()

        gate_loss = F.binary_cross_entropy(gate, gate_target)

        return (
            pdf_loss
            + self.alpha_emiss   * emiss_loss
            + self.alpha_profile * profile_loss
            + self.alpha_gate    * gate_loss
        )


if __name__ == "__main__":

    file_path = DATA_PATH

    print("Training all fluxes model")

    #torch.cuda.empty_cache()

    # Initialize model
    cnn_model = ConvNN(in_channels, layer_size1, layer_size2, layer_size3, layer_size4,
                       out_channels, kernel_size).to(device)

    criterion = GatedPDFEmissivityLoss(
        alpha_emiss=30.0, alpha_profile=20.0, alpha_gate=1.0
    )
    # criterion = nn.KLDivLoss(reduction="batchmean")
    # criterion = KLWithLeakageLoss()
    # criterion = WassersteinLoss()

    optimizer = torch.optim.Adam(
        cnn_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
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

    input_mean = input_tensor.mean(dim=(0,2,3), keepdim=True)
    input_std = input_tensor.std(dim=(0,2,3), keepdim=True)
    input_std[input_std == 0] = 1.0

    np.save(os.path.join(MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}_input_mean.npy"),
            input_mean.cpu().numpy())
    np.save(os.path.join(MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}_input_std.npy"),
            input_std.cpu().numpy())

    input_tensor_norm = (input_tensor - input_mean) / input_std

    # Store raw (un-normalized) rho as a third tensor so the emissivity
    # loss always receives the physical density, not the z-scored one.
    rho_tensor = input_tensor[:, 0:1]   # (N, 1, nx, ny), un-normalized
    dataset = TensorDataset(input_tensor_norm, output_tensor, rho_tensor)

    num_samples = len(dataset)
    print("Number of samples:", num_samples)

    indices = np.random.permutation(num_samples)

    train_end = int(0.50 * num_samples)
    val_end = int(0.75 * num_samples)

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
            torch.nn.utils.clip_grad_norm_(cnn_model.parameters(), max_norm=1.0)
            optimizer.step()

        cnn_model.eval()

        with torch.no_grad():

            train_loss_total = 0
            val_loss_total = 0

            # Train evaluation
            for x_batch, y_batch, rho_batch in train_loader:

                logits_b, gate_b = cnn_model(x_batch)

                train_loss_total += criterion(logits_b, gate_b, y_batch, rho_batch).item()

            train_loss = train_loss_total / len(train_loader)

            # Validation evaluation
            for x_batch, y_batch, rho_batch in validation_loader:

                logits_b, gate_b = cnn_model(x_batch)

                val_loss_total += criterion(logits_b, gate_b, y_batch, rho_batch).item()

            val_loss = val_loss_total / len(validation_loader)

        if (epoch + 1) % print_every == 0:

            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )

        epochs_array.append(epoch+1)
        train_loss_arr.append(train_loss)
        val_loss_arr.append(val_loss)

        # # Early stopping
        # window_size = 200

        # if len(val_loss_arr) >= window_size:

        #     val_loss_ma = np.convolve(
        #         val_loss_arr,
        #         np.ones(window_size)/window_size,
        #         mode='valid'
        #     )

        #     if len(val_loss_ma) > 1 and val_loss_ma[-1] > np.min(val_loss_ma[:-1]) and epoch >= 499:

        #         print(f"Early stopping at epoch {epoch+1}")
        #         break

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
        os.path.join(MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}.pth")
    )

    # Plot loss
    plt.figure(figsize=(10,5))

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
        os.path.join(LOSS_PLOT_DIR, f"cnn_{resolution}_{downsample}_loss.jpg"),
        dpi=500
    )

    plt.close()
