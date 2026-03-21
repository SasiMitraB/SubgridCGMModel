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
in_channels = 6
out_channels = 20
layer_size1 = 32
layer_size2 = 64
layer_size3 = 128
kernel_size = 5
num_epochs = 1000
print_every = 50
batch_size = 64
learning_rate = 1e-3
weight_decay = 1e-3
dropout_rate = 0.1

def nn_data(resolution: tuple, downsample: int) -> tuple:
    """ A function to load the data and return the inputs and outputs for the Conv neural network."""

    sim_data = simulation_data()
    sim_data.down_sample = downsample
    sim_data.resolution = resolution

    folder_path = f"/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/datafiles/c{resolution}_128"
    file_path = f"/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/kh_build/src/c{resolution[0]}_{resolution[1]}/bin"
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
    fields = ['rho', 'temp', 'ux', 'uy', 'ps', 'fmcl']
    cg = {f'cg_{field}': np.zeros(shape) for field in fields}

    for i in range(sim_data.rho.shape[0]):
        for field in fields:
            if field in ['rho', 'temp', 'ux', 'uy', 'ps']:
                cg[f'cg_{field}'][i] = sim_data.coarse_grain(getattr(sim_data, field)[i])
            elif field in ['fmcl']:
                cg[f'cg_{field}'][i] = sim_data.calc_fmcl(sim_data.rho[i], sim_data.temp[i])
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

    fields = ['rho', 'temp', 'ux', 'uy', 'ps', 'fmcl']
    cg = {f'cg_{field}': np.zeros(shape) for field in fields}

    # -------------------------
    # Coarse-grain inputs
    # -------------------------
    for field in fields:
        if field in ['rho', 'temp', 'ux', 'uy', 'ps']:
            cg[f'cg_{field}'] = sim_data.coarse_grain(locals()[field])
        elif field == 'fmcl':
            cg[f'cg_{field}'] = sim_data.calc_fmcl(rho, temp)

    # -------------------------
    # Build input tensor
    # -------------------------
    input_tensors = [
        torch.from_numpy(cg[f'cg_{f}']).unsqueeze(0).float()
        for f in fields
    ]

    input_tensor = torch.cat(input_tensors, dim=0)   # (C, nx, ny)
    input_tensor = input_tensor.unsqueeze(0)         # (1, C, nx, ny)

    # -------------------------
    # Normalize input (IMPORTANT)
    # -------------------------
    input_mean = np.load(
        f"/ptmp/mpa/dipda/subgrid/SubgridCGMModel/conv_nn/pdf_model_saves/cnn_{resolution}_{downsample}_input_mean.npy"
    )
    input_std = np.load(
        f"/ptmp/mpa/dipda/subgrid/SubgridCGMModel/conv_nn/pdf_model_saves/cnn_{resolution}_{downsample}_input_std.npy"
    )

    input_tensor = (input_tensor - input_mean) / input_std
    input_tensor = input_tensor.to(device)

    # -------------------------
    # Load model
    # -------------------------
    model_path = f"/ptmp/mpa/dipda/subgrid/SubgridCGMModel/conv_nn/pdf_model_saves/cnn_{resolution}_{downsample}.pth"

    cnn_model = ConvNN(
        in_channels, layer_size1, layer_size2,
        layer_size3, out_channels, kernel_size
    ).to(device)

    cnn_model.load_state_dict(torch.load(model_path, map_location=device))
    cnn_model.eval()

    # -------------------------
    # Predict PDF
    # -------------------------
    with torch.no_grad():

        logits = cnn_model(input_tensor)   # (1, bins, nx, ny)

        pdf = torch.softmax(logits, dim=1)   # convert to PDF

        pdf = pdf[0].cpu().numpy()  # (bins, nx, ny)

    return pdf

class ConvNN(nn.Module):

    def __init__(self, in_channels, layer_size1, layer_size2, layer_size3, out_channels, kernel_size):

        super().__init__()
        padding = kernel_size // 2

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, layer_size1, kernel_size, padding=padding),
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
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(layer_size3, layer_size2, kernel_size, padding=padding),
            nn.BatchNorm2d(layer_size2),
            nn.ReLU(),

            nn.Conv2d(layer_size2, layer_size1, kernel_size, padding=padding),
            nn.BatchNorm2d(layer_size1),
            nn.ReLU(),

            nn.Conv2d(layer_size1, out_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
if __name__ == "__main__":

    file_path = f"/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/kh_build/src/c{resolution[0]}_{resolution[1]}/bin"

    print("Training all fluxes model")

    torch.cuda.empty_cache()

    # Initialize model
    cnn_model = ConvNN(in_channels, layer_size1, layer_size2, layer_size3,
                       out_channels, kernel_size).to(device)

    criterion = nn.KLDivLoss(reduction="batchmean")

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

    np.save(f"pdf_model_saves/cnn_{resolution}_{downsample}_input_mean.npy",
            input_mean.cpu().numpy())
    np.save(f"pdf_model_saves/cnn_{resolution}_{downsample}_input_std.npy",
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

            outputs = cnn_model(inputs)

            log_probs = torch.log_softmax(outputs, dim=1)

            loss = criterion(log_probs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        cnn_model.eval()

        with torch.no_grad():

            train_loss_total = 0
            val_loss_total = 0

            # Train evaluation
            for x_batch, y_batch in train_loader:

                preds = cnn_model(x_batch)
                log_preds = torch.log_softmax(preds, dim=1)

                train_loss_total += criterion(log_preds, y_batch).item()

            train_loss = train_loss_total / len(train_loader)

            # Validation evaluation
            for x_batch, y_batch in validation_loader:

                preds = cnn_model(x_batch)
                log_preds = torch.log_softmax(preds, dim=1)

                val_loss_total += criterion(log_preds, y_batch).item()

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

            preds = cnn_model(x_batch)

            log_preds = torch.log_softmax(preds, dim=1)

            test_loss_total += criterion(log_preds, y_batch).item()

        test_loss = test_loss_total / len(test_loader)

    print(f"Test Loss: {test_loss:.6f}")

    # Save model
    torch.save(
        cnn_model.state_dict(),
        f"pdf_model_saves/cnn_{resolution}_{downsample}.pth"
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
    plt.title("Training Loss")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"pdf_loss_plots/cnn_{resolution}_{downsample}_loss.jpg",
        dpi=500
    )

    plt.close()
