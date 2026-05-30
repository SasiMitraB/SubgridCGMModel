# Python script for predicting the source term using
# a 3-lognormal GMM PDF model per pixel

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

np.random.seed(10)
device = torch.device('cpu')
resolution = (512, 256)
downsample = 32
in_channels = 5
layer_size1 = 32
layer_size2 = 64
layer_size3 = 128
kernel_size = 5
out_channels = 40
total_length: float = 40
total_width: float = 20
T_edges = np.logspace(3.0, 7.0, out_channels + 1)
T_centers = 0.5 * (T_edges[:-1] + T_edges[1:])
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
        lam[mask_ki] = (2.0e-19 * np.exp(-1.184e5 / (temp[mask_ki] + 1.0e3)) +
                         2.8e-28 * np.sqrt(temp[mask_ki]) * np.exp(-92.0 / temp[mask_ki]))

    # CGOLS fit (logT > 8.15)
    mask_hi = logt > 8.15
    lam[mask_hi] = 10.0 ** (0.45 * logt[mask_hi] - 26.065)

    # SPEX interpolation (4.2 < logT <= 8.15)
    mask_mid = (logt > 4.2) & (logt <= 8.15)
    if np.any(mask_mid):
        ipps = (25.0 * logt[mask_mid] - 103).astype(int)
        ipps = np.clip(ipps, 0, 100)
        x0 = 4.12 + 0.04 * ipps
        dx = logt[mask_mid] - x0
        logcool = (lhd[ipps + 1] * dx - lhd[ipps] * (dx - 0.04)) * 25.0
        lam[mask_mid] = 10.0 ** logcool

    mask_off = (logt < 4.3) | (logt > 5.7)
    lam[mask_off] = 0.0

    return lam


class GMM_CNN(nn.Module):
    """
    CNN that predicts, per pixel, the parameters of a
    3-component Gaussian mixture in log10(T) space
    (i.e. 3 lognormals in T space):
      - mixture weights  (3 channels)
      - ordered means    (3 channels)
      - standard devs    (3 channels)
    """

    def __init__(
        self,
        in_channels,
        layer_size1,
        layer_size2,
        layer_size3,
        kernel_size,
        n_components=3
    ):

        super().__init__()

        self.n_components = n_components
        padding = kernel_size // 2

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, layer_size1, kernel_size, padding=padding),
            nn.BatchNorm2d(layer_size1),
            nn.ReLU(),

            nn.Conv2d(layer_size1, layer_size2, kernel_size, padding=padding),
            nn.BatchNorm2d(layer_size2),
            nn.ReLU(),

            nn.Conv2d(layer_size2, layer_size3, kernel_size, padding=padding),
            nn.BatchNorm2d(layer_size3),
            nn.ReLU(),
        )

        # 3 weights + 3 means + 3 sigmas = 9 channels
        self.head = nn.Conv2d(layer_size3, 9, kernel_size=1)

    def forward(self, x):

        x = self.encoder(x)
        params = self.head(x)

        # --- split raw parameters ---
        raw_weights = params[:, 0:3]

        raw_mu0  = params[:, 3]
        raw_dmu1 = params[:, 4]
        raw_dmu2 = params[:, 5]

        raw_sigma = params[:, 6:9]

        # --- mixture weights (softmax → sum to 1) ---
        weights = torch.softmax(raw_weights, dim=1)

        # --- ordered means  mu0 < mu1 < mu2 ---
        mu0 = 3.0 + 4.0 * torch.sigmoid(raw_mu0)       # ∈ [3, 7]

        dmu1 = F.softplus(raw_dmu1)
        dmu2 = F.softplus(raw_dmu2)

        mu1 = mu0 + dmu1
        mu2 = mu1 + dmu2

        mu = torch.stack([mu0, mu1, mu2], dim=1)

        # --- standard deviations ---
        sigma = F.softplus(raw_sigma) + 1e-3

        return weights, mu, sigma


def build_gmm_pdf(weights, mu, sigma, logT_centers):
    """
    Build the discrete PDF from GMM parameters.

    weights : (B, 3, nx, ny)
    mu      : (B, 3, nx, ny)
    sigma   : (B, 3, nx, ny)

    returns:
        pdf : (B, bins, nx, ny)   normalised discrete PDF
    """

    bins = len(logT_centers)
    logT = logT_centers.to(weights.device).view(1, 1, bins, 1, 1)

    weights = weights.unsqueeze(2)   # (B,3,1,nx,ny)
    mu      = mu.unsqueeze(2)
    sigma   = sigma.unsqueeze(2)

    gauss = torch.exp(-0.5 * ((logT - mu) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)

    pdf = (weights * gauss).sum(dim=1)          # (B, bins, nx, ny)

    # normalise discrete PDF
    pdf = pdf / (pdf.sum(dim=1, keepdim=True) + 1e-12)

    return pdf


# ── load normalisation stats ──────────────────────────────────────
input_mean = np.load( f'/Volumes/PortableSSD/Projects/SubgridCGMModel/outputs/model_saves/log_model_saves/cnn_(512, 256)_32_input_mean.npy'
)
input_std = np.load( f'/Volumes/PortableSSD/Projects/SubgridCGMModel/outputs/model_saves/log_model_saves/cnn_(512, 256)_32_input_std.npy'
)


def source_func(rho, pres, ux, uy, ps, fmcl):

    global resolution, downsample, input_mean, input_std, device
    global total_length, total_width

    # ── temperature ───────────────────────────────────────────────
    temp = (
        np.array(pres) * 1.59916e-14 / np.array(rho)
    ) * (1.0 / 1.381e-16)

    # ── build coarse-grained input fields ─────────────────────────
    fields = ['rho', 'temp', 'ux', 'uy', 'ps']
    shape = (resolution[0] // downsample, resolution[1] // downsample)

    cg = {f'cg_{f}': np.zeros(shape) for f in fields}
    for field in fields:
        cg[f'cg_{field}'] = np.transpose(np.array(locals()[field]))

    # ── build & normalise input tensor ────────────────────────────
    input_tensors = [
        torch.from_numpy(cg[f'cg_{f}']).unsqueeze(0).float()
        for f in fields
    ]
    input_tensor = torch.cat(input_tensors, dim=0).unsqueeze(0)   # (1, C, nx, ny)

    input_tensor = (
        input_tensor
        - torch.tensor(input_mean, dtype=torch.float32)
    ) / torch.tensor(input_std, dtype=torch.float32)
    input_tensor = input_tensor.to(device)

    # ── allocate output ───────────────────────────────────────────
    source_term = np.zeros((5, shape[0], shape[1]))

    # ── load 3-lognormal GMM model ────────────────────────────────
    model_path = ( f'/Volumes/PortableSSD/Projects/SubgridCGMModel/outputs/model_saves/log_model_saves/cnn_(512, 256)_32.pth'
    )

    cnn_model = GMM_CNN(
        in_channels,
        layer_size1,
        layer_size2,
        layer_size3,
        kernel_size
    ).to(device)

    cnn_model.load_state_dict(torch.load(model_path, map_location=device))
    cnn_model.eval()

    # ── predict GMM PDF ───────────────────────────────────────────
    with torch.no_grad():

        weights, mu, sigma = cnn_model(input_tensor)

        pdf = build_gmm_pdf(weights, mu, sigma, logT_centers)

        pdf = pdf[0].cpu().numpy()   # (bins, nx, ny)

    # ── ensure normalisation ──────────────────────────────────────
    pdf /= (pdf.sum(axis=0, keepdims=True) + 1e-12)

    # ── cell sizes ────────────────────────────────────────────────
    if rho.shape[1] > rho.shape[0]:
        dy = total_length / rho.shape[1]
        dx = total_width  / rho.shape[0]
    else:
        dy = total_length / rho.shape[0]
        dx = total_width  / rho.shape[1]

    # ── cooling source term ───────────────────────────────────────
    kb = 1.380649e-16
    nb = out_channels
    temp_bins    = np.logspace(3, 7, nb + 1)
    temp_centers = 0.5 * (temp_bins[:-1] + temp_bins[1:])

    T = temp_centers[:, None, None]          # (bins, 1, 1)
    P = np.transpose(pres)[None, :, :]       # (1, nx, ny)
    n = P / (kb * T)
    cool = lambda_cool(T) * n ** 2
    cool_rate = np.sum(pdf * cool, axis=0)

    source_term[3] = -cool_rate              # energy source term

    # ── adaptive smoothing ────────────────────────────────────────
    for channel in range(3, 4):

        v = source_term[channel]
        w = np.clip(
            (np.abs(v) - np.percentile(np.abs(v), 75))
            /
            (np.percentile(np.abs(v), 90)
             - np.percentile(np.abs(v), 75) + 1e-12),
            0, 1
        )
        A = gaussian_filter(v, 0.0)
        B = gaussian_filter(v, kernel_size / 3)
        source_term[channel] = (1 - w) * A + w * B

    # ── return shape ──────────────────────────────────────────────
    final_term = np.transpose(source_term, axes=(0, 2, 1))
    return final_term.reshape(5, -1)