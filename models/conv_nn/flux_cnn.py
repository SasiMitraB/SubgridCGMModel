# CNN to learn individually the fluxes for all channels

import numpy as np
import matplotlib.pyplot as plt
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data')))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'model_saves')
LOSS_PLOT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'loss_plots')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOSS_PLOT_DIR, exist_ok=True)
# TODO: Set these to your local simulation data directories
DATA_PATH = os.environ.get('SUBGRID_DATA_PATH', '/path/to/simulation/bin')
CACHE_PATH = os.environ.get('SUBGRID_CACHE_PATH', '/path/to/cache')
import data_preprocess
from data_preprocess import simulation_data

seed = 10
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

resolution = (512, 256)  
downsample = 32
in_channels = 6
out_channels = 1
layer_size1 = 32
layer_size2 = 64
layer_size3 = 128
kernel_size = 5
num_epochs = 1000
print_every = 50
batch_size = 32
learning_rate = 1e-3
weight_decay = 1e-4
dropout_rate = 0.5

def nn_data(resolution: tuple, downsample: int, channel: int) -> tuple:
    """ A function to load the data and return the inputs and outputs for the Conv neural network."""

    sim_data = simulation_data()
    sim_data.down_sample = downsample
    sim_data.resolution = resolution

    folder_path = CACHE_PATH
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
    fields = ['rho', 'temp', 'ux', 'uy', 'ps', 'fmcl']
    cg = {f'cg_{field}': np.zeros(shape) for field in fields}

    for i in range(sim_data.rho.shape[0]):
        for field in fields:
            if field in ['rho', 'temp', 'ux', 'uy', 'ps']:
                cg[f'cg_{field}'][i] = sim_data.coarse_grain(getattr(sim_data, field)[i])
            elif field in ['fmcl']:
                cg[f'cg_{field}'][i] = sim_data.calc_fmcl(sim_data.rho[i], sim_data.temp[i])
    subgrid_flux = np.transpose(sim_data.calc_subgrid_flux(), axes=(1, 0, 2, 3))[channel]

    input_tensors = [torch.from_numpy(cg[f'cg_{f}']).unsqueeze(1).float() for f in fields]
    # input_tensors = [
    #     torch.from_numpy(cg[f'cg_{f}'][100:]).unsqueeze(1).float() 
    #     for f in fields
    # ]
    input_tensor = torch.cat(input_tensors, dim=1)
    output_tensor = torch.from_numpy(subgrid_flux).unsqueeze(1).float()
    # output_tensor = torch.from_numpy(source_term[100:]).unsqueeze(1).float()

    return input_tensor, output_tensor

def snapshot_pred(rho: np.ndarray, temp: np.ndarray, pressure: np.ndarray, ux: np.ndarray, uy: np.ndarray, eint: np.ndarray, ps: np.ndarray, downsample: int, resolution: np.ndarray) -> np.ndarray:
    """ A function to predict the source terms for a given snapshot using the trained model."""

    sim_data = simulation_data()
    sim_data.down_sample = downsample
    sim_data.resolution = resolution

    shape = (resolution[0] // downsample, resolution[1] // downsample)
    fields = ['rho', 'temp', 'ux', 'uy', 'ps', 'fmcl']
    cg = {f'cg_{field}': np.zeros(shape) for field in fields}

    for field in fields:
        if field in ['rho', 'temp', 'ux', 'uy', 'ps']:
            cg[f'cg_{field}'] = sim_data.coarse_grain(locals()[field])
        elif field in ['fmcl']:
            cg[f'cg_{field}'] = sim_data.calc_fmcl(rho, temp)
    
    channels = np.arange(10)  
    subgrid_flux = np.zeros((10, shape[0], shape[1]))

    input_tensors = [torch.from_numpy(cg[f'cg_{f}']).unsqueeze(0).float() for f in fields]
    input_tensor = torch.cat(input_tensors, dim=0)
    input_tensor = input_tensor.unsqueeze(0)
    input_mean = np.load(f"{MODEL_SAVE_DIR}/flux_model_saves/cnn_{sim_data.resolution}_{downsample}_0_input_mean.npy")
    input_std = np.load(f"{MODEL_SAVE_DIR}/flux_model_saves/cnn_{sim_data.resolution}_{downsample}_0_input_std.npy")
    input_tensor = (input_tensor - input_mean) / input_std
    input_tensor = input_tensor.to(device)

    global in_channels, layer_size1, layer_size2, layer_size3, out_channels, kernel_size
    # if resolution == (256, 256) and downsample == 8:
    #     kernel_size = 3
    # elif resolution == (512, 512) and downsample == 8:
    #     kernel_size = 5
    # elif resolution == (1024, 1024) and downsample == 8:
    #     kernel_size = 7

    for channel in channels:

        model_path = f"{MODEL_SAVE_DIR}/flux_model_saves/cnn_{sim_data.resolution}_{downsample}_{channel}.pth"
        cnn_model = ConvNN(in_channels, layer_size1, layer_size2, layer_size3, out_channels, kernel_size).to(device)
        # cnn_model = ResUNet(in_channels=in_channels, out_channels=out_channels, base=64, depth=4, dropout=dropout_rate).to(device)
        cnn_model.load_state_dict(torch.load(model_path, map_location=device))
        cnn_model.eval()

        with torch.no_grad():
            output_mean = torch.from_numpy(np.load(f"{MODEL_SAVE_DIR}/flux_model_saves/cnn_{sim_data.resolution}_{downsample}_{channel}_output_mean.npy"))
            output_std = torch.from_numpy(np.load(f"{MODEL_SAVE_DIR}/flux_model_saves/cnn_{sim_data.resolution}_{downsample}_{channel}_output_std.npy"))
            output_mean = output_mean.to(device)
            output_std = output_std.to(device)
            pred = cnn_model(input_tensor)  
            pred = pred * output_std + output_mean  
            subgrid_flux[channel] = pred.squeeze().cpu().numpy()  
    
    return subgrid_flux

class ConvNN(nn.Module):

    def __init__(self, in_channels, layer_size1, layer_size2, layer_size3, out_channels, kernel_size):

        super(ConvNN, self).__init__()
        padding = kernel_size // 2
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, layer_size1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(layer_size1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Conv2d(layer_size1, layer_size2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(layer_size2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Conv2d(layer_size2, layer_size3, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(layer_size3),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(layer_size3, layer_size2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(layer_size2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Conv2d(layer_size2, layer_size1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(layer_size1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Conv2d(layer_size1, out_channels, kernel_size=1),  
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
# Residual UNet
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, groups=1, p=0.0):
        super().__init__()
        pad = k // 2
        self.proj = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)
        self.block = nn.Sequential(
            nn.GroupNorm(groups, in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, k, padding=pad, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p) if p > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, k, padding=pad, bias=False),
        )

    def forward(self, x):
        return self.block(x) + self.proj(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, p=0.0):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.res  = ResBlock(out_ch, out_ch, p=p)

    def forward(self, x):
        return self.res(self.down(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, p=0.0):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2, bias=False)
        self.res  = ResBlock(out_ch * 2, out_ch, p=p)

    def forward(self, x, skip):
        x = self.up(x)
        # pad to match (safe with odd sizes)
        dy, dx = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([x, skip], dim=1)
        return self.res(x)

class ResUNet(nn.Module):

    def __init__(self, in_channels, out_channels, base=64, depth=4, dropout=0.1):
        super().__init__()
        chs = [base * (2 ** i) for i in range(depth)]
        self.stem = ResBlock(in_channels, chs[0], p=dropout)
        self.downs = nn.ModuleList([Down(chs[i], chs[i+1], p=dropout) for i in range(depth - 1)])
        self.bottleneck = ResBlock(chs[-1], chs[-1], p=dropout)
        self.ups = nn.ModuleList([Up(chs[i+1], chs[i], p=dropout) for i in reversed(range(depth - 1))])
        self.head = nn.Conv2d(chs[0], out_channels, 1)

    def forward(self, x):
        skips = []
        x = self.stem(x); skips.append(x)
        for d in self.downs:
            x = d(x); skips.append(x)
        x = self.bottleneck(x)
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)
        return self.head(x)
    
class MSEWithIntegral(nn.Module):
    
    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = lam

    def forward(self, pred, target):
        if pred.dim() == 3:   # [B,H,W] -> [B,1,H,W]
            pred   = pred.unsqueeze(1)
            target = target.unsqueeze(1)

        mse = F.mse_loss(pred, target, reduction='mean')
        # per-sample, per-channel mean over spatial dims
        mean_pred = pred.mean(dim=(2, 3))
        mean_targ = target.mean(dim=(2, 3))
        mean_pen  = F.mse_loss(mean_pred, mean_targ, reduction='mean')
        return mse + self.lam * mean_pen

# Scale-invariant / per-snapshot-normalized loss 
def per_sample_normalized_mse(pred, target, eps=1e-6, reduction='mean'):
    """ Normalizes the error by each sample's target std so 'small-range' snapshots are weighted fairly vs 'large-range' ones. """
    # dims to reduce over (C,H,W)
    dims = tuple(range(1, target.ndim))
    std = target.float().std(dim=dims, keepdim=True, unbiased=False).clamp_min(eps)
    diff = (pred - target) / std
    loss = (diff ** 2).mean(dim=dims)
    return loss.mean() if reduction == 'mean' else loss

# Regularized Huber Loss 
def per_sample_normalized_huber(
    pred, target, delta=1.0, eps=1e-6, reduction='mean',
    lambda_reg=1e-3
):
    # dims to normalize over (C,H,W)
    dims = tuple(range(1, target.ndim))
    std = target.float().std(dim=dims, keepdim=True, unbiased=False).clamp_min(eps)

    # normalize
    pred_norm = pred / std
    target_norm = target / std

    # Huber loss (per-pixel)
    diff = pred_norm - target_norm
    abs_diff = diff.abs()
    huber = torch.where(
        abs_diff < delta,
        0.5 * diff**2,
        delta * (abs_diff - 0.5 * delta)
    )

    # reduce over (C,H,W)
    loss = huber.mean(dim=dims)  # (N,)

    if reduction == 'mean':
        loss = loss.mean()

    # Regularization: penalize max absolute prediction in batch
    if lambda_reg > 0.0:
        reg = torch.max(pred.abs())
        loss = loss + lambda_reg * reg

    return loss

# Regularized Huber Loss with Hot-Phase Penalty 
def per_sample_normalized_huber_hot(
    pred, target, input_tensor=None, temp_channel=1,
    delta=1.0, eps=1e-6, reduction='mean',
    lambda_reg=0.0, lambda_hot=1.0, # 1.0
    T_cut=1e5, T_width=1e4
):

    dims = tuple(range(1, target.ndim))
    std = target.float().std(dim=dims, keepdim=True, unbiased=False).clamp_min(eps)

    pred_norm = pred / std
    target_norm = target / std

    diff = pred_norm - target_norm
    abs_diff = diff.abs()
    huber = torch.where(
        abs_diff < delta,
        0.5 * diff**2,
        delta * (abs_diff - 0.5 * delta)
    )

    loss = huber.mean(dim=dims)
    if reduction == 'mean':
        loss = loss.mean()

    if lambda_reg > 0.0:
        reg = torch.max(pred.abs())
        loss = loss + lambda_reg * reg

    # Temperature-dependent penalty
    if lambda_hot > 0.0 and input_tensor is not None:
        T = input_tensor[:, temp_channel]  # (N,H,W)

        # logistic ramp: low at 1e4, ~0.5 at 1e5, high at 1e6
        weight = 1.0 / (1.0 + torch.exp(-(T - T_cut) / T_width))
        weight = weight.unsqueeze(1)  # match pred shape

        masked_pred = pred.abs() * weight
        mean_hot = masked_pred.mean()
        max_hot = masked_pred.max()

        hot_penalty = mean_hot + 0.1 * max_hot
        loss = loss + lambda_hot * hot_penalty

    return loss

if __name__ == "__main__":

    # Load the data
    file_path = DATA_PATH
    # channels = np.arange(10)
    channels = [7] 

    for channel in channels:

        print(f"Training model for channel {channel}")
        torch.cuda.empty_cache()

        # Initialize the model, loss function, and optimizer
        cnn_model = ConvNN(in_channels, layer_size1, layer_size2, layer_size3, out_channels, kernel_size).to(device)
        criterion = nn.MSELoss()
        # cnn_model = ResUNet(in_channels=in_channels, out_channels=out_channels, base=64, depth=4, dropout=dropout_rate).to(device)
        # criterion = per_sample_normalized_mse   
        # criterion = per_sample_normalized_huber_hot
        # criterion = MSEWithIntegral(lam=0.5)
        optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        cnn_data = nn_data(resolution, downsample, channel)
        input_tensor, output_tensor = cnn_data
        input_tensor = input_tensor.to(device)
        output_tensor = output_tensor.to(device)

        print("Normalizing the input and output tensors")
        input_mean = input_tensor.mean(dim=(0, 2, 3), keepdim=True)  
        input_std = input_tensor.std(dim=(0, 2, 3), keepdim=True)
        input_std[input_std == 0] = 1.0  
        np.save(os.path.join(MODEL_SAVE_DIR, "flux_model_saves", f"cnn_{resolution}_{downsample}_{channel}_input_mean.npy"), input_mean.cpu().numpy())
        np.save(os.path.join(MODEL_SAVE_DIR, "flux_model_saves", f"cnn_{resolution}_{downsample}_{channel}_input_std.npy"), input_std.cpu().numpy())
        input_tensor_norm = (input_tensor - input_mean) / input_std

        output_mean = output_tensor.mean()
        output_std = output_tensor.std()
        print("Output mean:", output_mean.item())
        print("Output std:", output_std.item())
        if output_std == 0:
            output_std = 1.0
        np.save(os.path.join(MODEL_SAVE_DIR, "flux_model_saves", f"cnn_{resolution}_{downsample}_{channel}_output_mean.npy"), output_mean.cpu().numpy())
        np.save(os.path.join(MODEL_SAVE_DIR, "flux_model_saves", f"cnn_{resolution}_{downsample}_{channel}_output_std.npy"), output_std.cpu().numpy())
        output_tensor_norm = (output_tensor - output_mean) / output_std

        dataset = TensorDataset(input_tensor_norm, output_tensor_norm)
        num_samples = len(dataset)
        print("Number of samples in the dataset:", num_samples)
        # indices = np.concatenate(((np.arange(num_samples // 5), np.random.permutation(np.arange(num_samples // 5, num_samples)))))
        indices = np.random.permutation(num_samples)
        train_end = int(0.33 * num_samples)
        val_end = int(0.67 * num_samples)
        test_end = int(1.00 * num_samples)

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:test_end]
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Training the model

        epochs_array = []
        train_loss_arr = []
        train_r2_arr = []
        val_loss_arr = []
        val_r2_arr = []

        for epoch in range(num_epochs):
            for i, (inputs, labels) in enumerate(train_loader):

                # Forward pass
                outputs = cnn_model(inputs)
                loss = criterion(outputs, labels)
                # loss = criterion(outputs, labels, input_tensor=inputs)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            cnn_model.eval()
            with torch.no_grad():

                train_preds = []
                train_targets = []
                for x_batch, y_batch in train_loader:
                    preds = cnn_model(x_batch)
                    train_preds.append(preds.cpu())
                    train_targets.append(y_batch.cpu())

                train_preds = torch.cat(train_preds)
                train_targets = torch.cat(train_targets)
                train_loss = criterion(train_preds, train_targets).item()

                train_ss_res = torch.sum((train_targets - train_preds) ** 2)
                train_ss_tot = torch.sum((train_targets - torch.mean(train_targets, dim=0, keepdim=True)) ** 2)
                train_r2 = 1 - train_ss_res / train_ss_tot

                val_preds = []
                val_targets = []
                for x_batch, y_batch in validation_loader:
                    preds = cnn_model(x_batch)
                    val_preds.append(preds.cpu())
                    val_targets.append(y_batch.cpu())

                val_preds = torch.cat(val_preds)
                val_targets = torch.cat(val_targets)
                val_loss = criterion(val_preds, val_targets).item()

                val_ss_res = torch.sum((val_targets - val_preds) ** 2)
                val_ss_tot = torch.sum((val_targets - torch.mean(val_targets, dim=0, keepdim=True)) ** 2)
                val_r2 = 1 - val_ss_res / val_ss_tot

            if (epoch + 1) % print_every == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train R2: {train_r2:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val R2: {val_r2:.4f}")
            
            epochs_array.append(epoch + 1)
            train_loss_arr.append(train_loss)
            train_r2_arr.append(train_r2.item() if hasattr(train_r2, 'item') else float(train_r2))
            val_loss_arr.append(val_loss)
            val_r2_arr.append(val_r2.item() if hasattr(val_r2, 'item') else float(val_r2))

            window_size = 200
            if len(val_loss_arr) >= window_size:
                val_loss_ma = np.convolve(val_loss_arr, np.ones(window_size)/window_size, mode='valid')
                if len(val_loss_ma) > 1 and val_loss_ma[-1] > np.min(val_loss_ma[:-1]) and epoch >= 499:
                    print(f"Early stopping at epoch {epoch+1} due to no improvement in moving average validation loss.")
                    break

            cnn_model.train()

        # Testing the model
        cnn_model.eval()
        with torch.no_grad():
            test_preds = []
            test_targets = []
            for x_batch, y_batch in test_loader:
                preds = cnn_model(x_batch)
                test_preds.append(preds.cpu())
                test_targets.append(y_batch.cpu())

            test_preds = torch.cat(test_preds)
            test_targets = torch.cat(test_targets)
            test_loss = criterion(test_preds, test_targets).item()
            
            test_ss_res = torch.sum((test_targets - test_preds) ** 2)
            test_ss_tot = torch.sum((test_targets - torch.mean(test_targets, dim=0, keepdim=True)) ** 2)
            test_r2 = 1 - test_ss_res / test_ss_tot
            print(f"Test Loss: {test_loss:.4f}, Test R2: {test_r2:.4f}")

        # Save the model
        torch.save(cnn_model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'flux_model_saves', f'cnn_{resolution}_{downsample}_{channel}.pth'))

        # Plotting the training and validation loss
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_array, train_loss_arr, label='Train Loss', color='g')
        plt.plot(epochs_array, val_loss_arr, label='Validation Loss', color='b')
        plt.axhline(y=train_loss_arr[-1], color='g', linestyle='--', label=f'Final Train Loss = {train_loss_arr[-1]:.4f}')
        plt.axhline(y=val_loss_arr[-1], color='b', linestyle='--', label=f'Final Validation Loss = {val_loss_arr[-1]:.4f}')
        plt.axhline(y=test_loss, color='r', linestyle='--', label=f'Final Test Loss = {test_loss:.4f}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss vs Epochs')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs_array, train_r2_arr, label=r'Train $R^2$', color='g')
        plt.plot(epochs_array, val_r2_arr, label=r'Validation $R^2$', color='b')
        plt.axhline(y=train_r2_arr[-1], color='g', linestyle='--', label=rf'Final Train $R^2$ = {train_r2_arr[-1]:.4f}')
        plt.axhline(y=val_r2_arr[-1], color='b', linestyle='--', label=rf'Final Validation $R^2$ = {val_r2_arr[-1]:.4f}')
        plt.axhline(y=test_r2, color='r', linestyle='--', label=rf'Final Test $R^2$ = {test_r2:.4f}')
        plt.xlabel('Epochs')
        plt.ylabel(r'$R^2$ Score')
        plt.title(r'$R^2$ Score vs Epochs')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(LOSS_PLOT_DIR, "flux_loss_plots", f"cnn_{resolution}_{downsample}_{channel}_loss.jpg"), dpi=500)
        plt.close()


