# CNN to learn the PDF using an analytic form (3 lognormals)

import numpy as np
import matplotlib.pyplot as plt
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))
import data_preprocess
from data_preprocess import simulation_data

np.random.seed(10)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

resolution = (512, 256)  
downsample = 32
in_channels = 5
out_channels = 40
layer_size1 = 32
layer_size2 = 64
layer_size3 = 128
kernel_size = 5
num_epochs = 1000
print_every = 50
batch_size = 64
learning_rate = 1e-3
weight_decay = 1e-3
dropout_rate = 0.3

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

def nn_data(resolution: tuple, downsample: int) -> tuple:
    """ A function to load the data and return the inputs and outputs for the Conv neural network."""

    sim_data = simulation_data()
    sim_data.down_sample = downsample
    sim_data.resolution = resolution

    folder_path = f"/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/datafiles/sc{resolution}_{downsample}"
    file_path = f"/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/kh_build/src/sc{resolution[0]}_{resolution[1]}/bin"
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

    Returns:
        pdf : (bins, nx, ny)
    """

    sim_data = simulation_data()
    sim_data.down_sample = downsample
    sim_data.resolution = resolution

    shape = (
        resolution[0] // downsample,
        resolution[1] // downsample
    )

    fields = ['rho', 'temp', 'ux', 'uy', 'ps']

    cg = {
        f'cg_{field}': np.zeros(shape)
        for field in fields
    }

    # ---------------------------------
    # Coarse-grain inputs
    # ---------------------------------

    for field in fields:

        cg[f'cg_{field}'] = sim_data.coarse_grain(
            locals()[field]
        )

    # ---------------------------------
    # Build input tensor
    # ---------------------------------

    input_tensors = [

        torch.from_numpy(
            cg[f'cg_{f}']
        ).unsqueeze(0).float()

        for f in fields
    ]

    input_tensor = torch.cat(
        input_tensors,
        dim=0
    )

    # (1,C,nx,ny)

    input_tensor = input_tensor.unsqueeze(0)

    # ---------------------------------
    # Normalize
    # ---------------------------------

    input_mean = np.load(
        f"/ptmp/mpa/dipda/subgrid/SubgridCGMModel/conv_nn/log_model_saves/cnn_{resolution}_{downsample}_input_mean.npy"
    )

    input_std = np.load(
        f"/ptmp/mpa/dipda/subgrid/SubgridCGMModel/conv_nn/log_model_saves/cnn_{resolution}_{downsample}_input_std.npy"
    )

    input_mean = torch.from_numpy(input_mean).float()
    input_std = torch.from_numpy(input_std).float()

    input_tensor = (
        input_tensor - input_mean
    ) / input_std

    input_tensor = input_tensor.to(device)

    # ---------------------------------
    # Load model
    # ---------------------------------

    model_path = (
        f"/ptmp/mpa/dipda/subgrid/SubgridCGMModel/conv_nn/log_model_saves/"
        f"cnn_{resolution}_{downsample}.pth"
    )

    cnn_model = GMM_CNN(
        in_channels,
        layer_size1,
        layer_size2,
        layer_size3,
        kernel_size
    ).to(device)

    cnn_model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    cnn_model.eval()

    # ---------------------------------
    # Predict PDF
    # ---------------------------------

    with torch.no_grad():

        weights, mu, sigma = cnn_model(
            input_tensor
        )

        pdf = build_gmm_pdf(
            weights,
            mu,
            sigma,
            logT_centers
        )

        # remove batch dimension

        pdf = pdf[0].cpu().numpy()

    return pdf

class GMM_CNN(nn.Module):

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

            nn.Conv2d(
                in_channels,
                layer_size1,
                kernel_size,
                padding=padding
            ),
            nn.BatchNorm2d(layer_size1),
            nn.ReLU(),

            nn.Conv2d(
                layer_size1,
                layer_size2,
                kernel_size,
                padding=padding
            ),
            nn.BatchNorm2d(layer_size2),
            nn.ReLU(),

            nn.Conv2d(
                layer_size2,
                layer_size3,
                kernel_size,
                padding=padding
            ),
            nn.BatchNorm2d(layer_size3),
            nn.ReLU(),
        )

        # outputs:
        #
        # weights (3)
        # means   (3)
        # sigmas  (3)
        #
        # total = 9 channels

        self.head = nn.Conv2d(
            layer_size3,
            9,
            kernel_size=1
        )

    def forward(self, x):

        x = self.encoder(x)

        params = self.head(x)

        # ----------------------------------------
        # split parameters
        # ----------------------------------------

        raw_weights = params[:, 0:3]

        raw_mu0  = params[:, 3]
        raw_dmu1 = params[:, 4]
        raw_dmu2 = params[:, 5]

        raw_sigma = params[:, 6:9]

        # ----------------------------------------
        # mixture weights
        # ----------------------------------------

        weights = torch.softmax(
            raw_weights,
            dim=1
        )

        # ----------------------------------------
        # ordered means
        #
        # mu0 < mu1 < mu2
        # ----------------------------------------

        mu0 = 3.0 + 4.0 * torch.sigmoid(raw_mu0)

        dmu1 = F.softplus(raw_dmu1)
        dmu2 = F.softplus(raw_dmu2)

        mu1 = mu0 + dmu1
        mu2 = mu1 + dmu2

        mu = torch.stack(
            [mu0, mu1, mu2],
            dim=1
        )

        # ----------------------------------------
        # sigmas
        # ----------------------------------------

        sigma = (
            F.softplus(raw_sigma)
            + 1e-3
        )

        return weights, mu, sigma

class WassersteinLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, target):
        """
        logits: (B, bins, nx, ny)
        target: (B, bins, nx, ny) — must be normalized PDF
        """

        # Convert logits → probabilities
        pred = torch.softmax(logits, dim=1)

        # Compute CDF along bin axis
        cdf_pred = torch.cumsum(pred, dim=1)
        cdf_target = torch.cumsum(target, dim=1)

        # Wasserstein-1 distance
        loss = torch.mean(torch.abs(cdf_pred - cdf_target))

        return loss

def build_gmm_pdf(
    weights,
    mu,
    sigma,
    logT_centers
):

    """
    weights : (B,3,nx,ny)
    mu      : (B,3,nx,ny)
    sigma   : (B,3,nx,ny)

    returns:
        pdf : (B,bins,nx,ny)
    """

    bins = len(logT_centers)

    logT = logT_centers.to(weights.device)

    # (1,1,bins,1,1)

    logT = logT.view(
        1,
        1,
        bins,
        1,
        1
    )

    weights = weights.unsqueeze(2)
    mu = mu.unsqueeze(2)
    sigma = sigma.unsqueeze(2)

    gauss = torch.exp(
        -0.5 *
        ((logT - mu) / sigma)**2
    )

    gauss = gauss / (
        np.sqrt(2*np.pi)
        * sigma
    )

    pdf = weights * gauss

    pdf = pdf.sum(dim=1)

    # normalize discrete PDF

    pdf = pdf / (
        pdf.sum(dim=1, keepdim=True)
        + 1e-12
    )

    return pdf

class KLWithLeakageLoss(nn.Module):
    def __init__(self, alpha=0, T0=1e6, width=0.1):
        super().__init__()
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.alpha = alpha

        # store logT info
        self.logT_centers = logT_centers
        self.logT0 = np.log10(T0)
        self.width = width

    def forward(self, log_probs, target):
        # KL loss
        # kl_loss = self.kl(log_probs, target)

        # expand weights
        weights = lambda_weights.to(target.device)[None, :, None, None]

        # weighted KL
        kl_elementwise = target * (torch.log(target + 1e-12) - log_probs)
        weighted_kl = kl_elementwise * weights

        kl_loss = torch.mean(weighted_kl)

        # Convert to probabilities
        pred = torch.exp(log_probs)

        # Peak bin index from TRUE PDF
        peak_idx = torch.argmax(target, dim=1)  # (B, nx, ny)

        # Get logT of peak bin
        logT_peak = self.logT_centers.to(target.device)[peak_idx]

        # Temperature mask (Gaussian around 1e6 K)
        temp_mask = torch.exp(
            -((logT_peak - self.logT0) ** 2) / (2 * self.width ** 2)
        )

        # Predicted mass at peak
        peak_prob = torch.gather(pred, 1, peak_idx.unsqueeze(1)).squeeze(1)

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
        
if __name__ == "__main__":

    file_path = f"/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/kh_build/src/c{resolution[0]}_{resolution[1]}/bin"

    print("Training all fluxes model")

    torch.cuda.empty_cache()

    # Initialize model
    cnn_model = GMM_CNN(
        in_channels,
        layer_size1,
        layer_size2,
        layer_size3,
        kernel_size
    ).to(device)

    criterion = nn.KLDivLoss(reduction="batchmean")
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

    np.save(f"log_model_saves/cnn_{resolution}_{downsample}_input_mean.npy",
            input_mean.cpu().numpy())
    np.save(f"log_model_saves/cnn_{resolution}_{downsample}_input_std.npy",
            input_std.cpu().numpy())

    input_tensor_norm = (input_tensor - input_mean) / input_std

    dataset = TensorDataset(input_tensor_norm, output_tensor)

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

        for inputs, labels in train_loader:

            weights, mu, sigma = cnn_model(inputs)

            pred_pdf = build_gmm_pdf(
                weights,
                mu,
                sigma,
                logT_centers
            )

            loss = criterion(
                torch.log(pred_pdf + 1e-12),
                labels
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        cnn_model.eval()

        with torch.no_grad():

            train_loss_total = 0
            val_loss_total = 0

            # Train evaluation
            for x_batch, y_batch in train_loader:

                weights, mu, sigma = cnn_model(x_batch)

                pred_pdf = build_gmm_pdf(
                    weights,
                    mu,
                    sigma,
                    logT_centers
                )

                train_loss_total += criterion(
                    torch.log(pred_pdf + 1e-12),
                    y_batch
                ).item()

            train_loss = train_loss_total / len(train_loader)

            # Validation evaluation
            for x_batch, y_batch in validation_loader:

                weights, mu, sigma = cnn_model(x_batch)

                pred_pdf = build_gmm_pdf(
                    weights,
                    mu,
                    sigma,
                    logT_centers
                )

                val_loss_total += criterion(
                    torch.log(pred_pdf + 1e-12),
                    y_batch
                ).item()

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

        for x_batch, y_batch in test_loader:

            weights, mu, sigma = cnn_model(x_batch)

            pred_pdf = build_gmm_pdf(
                weights,
                mu,
                sigma,
                logT_centers
            )

            test_loss_total += criterion(
                torch.log(pred_pdf + 1e-12),
                y_batch
            ).item()

        test_loss = test_loss_total / len(test_loader)

    print(f"Test Loss: {test_loss:.6f}")

    # Save model
    torch.save(
        cnn_model.state_dict(),
        f"log_model_saves/cnn_{resolution}_{downsample}.pth"
    )

    # Plot loss
    plt.figure(figsize=(10,5))

    plt.plot(epochs_array, train_loss_arr, label="Train Loss")
    plt.plot(epochs_array, val_loss_arr, label="Validation Loss")

    plt.axhline(train_loss_arr[-1], linestyle="--")
    plt.axhline(val_loss_arr[-1], linestyle="--")
    plt.axhline(test_loss, linestyle="--", color="red")

    plt.xlabel("Epochs")
    plt.ylabel("KL Divergence")
    # plt.ylabel("Wasserstein Loss")
    plt.title("Training Loss")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"log_loss_plots/cnn_{resolution}_{downsample}_loss.jpg",
        dpi=500
    )

    plt.close()
