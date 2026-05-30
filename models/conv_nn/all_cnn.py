# CNN to learn jointly the source terms for all channels

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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

np.random.seed(10)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resolution = (512, 512)  
downsample = 8
in_channels = 6
out_channels = 5
layer_size1 = 32
layer_size2 = 64
layer_size3 = 128
layer_size4 = 256
layer_size5 = 512
kernel_size = 5
num_epochs = 1000
print_every = 50
batch_size = 64
learning_rate = 1e-3
weight_decay = 1e-4
dropout_rate = 0.5

def nn_data(resolution: tuple, downsample: int) -> tuple:
    """ A function to load the data and return the inputs and outputs for the Conv neural network."""

    sim_data = simulation_data()
    sim_data.down_sample = downsample
    sim_data.resolution = resolution

    folder_path = f"../data/data_saves/{resolution}_{downsample}_2"
    file_path = folder_path + "/bin/"
    if os.path.exists(f"{folder_path}"):
        sim_data.rho = np.load(f"{folder_path}/rho.npy")
        sim_data.temp = np.load(f"{folder_path}/temp.npy")
        sim_data.pressure = np.load(f"{folder_path}/pressure.npy")
        sim_data.ux = np.load(f"{folder_path}/ux.npy")
        sim_data.uy = np.load(f"{folder_path}/uy.npy")
        sim_data.eint = np.load(f"{folder_path}/eint.npy")
        sim_data.ps = np.load(f"{folder_path}/ps.npy")
    else:
        sim_data.input_data(file_path)
        os.makedirs(folder_path, exist_ok=True)
        np.save(f"{folder_path}/rho.npy", sim_data.rho)
        np.save(f"{folder_path}/temp.npy", sim_data.temp)
        np.save(f"{folder_path}/pressure.npy", sim_data.pressure)
        np.save(f"{folder_path}/ux.npy", sim_data.ux)
        np.save(f"{folder_path}/uy.npy", sim_data.uy)
        np.save(f"{folder_path}/eint.npy", sim_data.eint)
        np.save(f"{folder_path}/ps.npy", sim_data.ps)
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
    source_term = sim_data.calc_all_source_terms()

    input_tensors = [torch.from_numpy(cg[f'cg_{f}']).unsqueeze(1).float() for f in fields]
    input_tensor = torch.cat(input_tensors, dim=1)
    output_tensor = torch.from_numpy(source_term).float()

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

    input_tensors = [torch.from_numpy(cg[f'cg_{f}']).unsqueeze(0).float() for f in fields]
    input_tensor = torch.cat(input_tensors, dim=0)
    input_tensor = input_tensor.unsqueeze(0)
    input_mean = np.load(f"{MODEL_SAVE_DIR}/model_saves/cnn_{sim_data.resolution}_{downsample}_input_mean.npy")
    input_std = np.load(f"{MODEL_SAVE_DIR}/model_saves/cnn_{sim_data.resolution}_{downsample}_input_std.npy")
    input_tensor = (input_tensor - input_mean) / input_std
    input_tensor = input_tensor.to(device)

    model_path = f"{MODEL_SAVE_DIR}/model_saves/cnn_{sim_data.resolution}_{downsample}.pth"
    global in_channels, layer_size1, layer_size2, layer_size3, layer_size4, layer_size5, out_channels
    if resolution == (256, 256) and downsample == 8:
        kernel_size = 3
    elif resolution == (512, 512) and downsample == 8:
        kernel_size = 5
    elif resolution == (1024, 1024) and downsample == 8:
        kernel_size = 7
    cnn_model = ConvNN(in_channels, layer_size1, layer_size2, layer_size3, layer_size4, layer_size5, out_channels, kernel_size).to(device)
    cnn_model.load_state_dict(torch.load(model_path, map_location=device))
    cnn_model.eval()

    with torch.no_grad():
        output_mean = torch.from_numpy(np.load(f"{MODEL_SAVE_DIR}/model_saves/cnn_{sim_data.resolution}_{downsample}_output_mean.npy"))
        output_std = torch.from_numpy(np.load(f"{MODEL_SAVE_DIR}/model_saves/cnn_{sim_data.resolution}_{downsample}_output_std.npy"))
        output_mean = output_mean.to(device)
        output_std = output_std.to(device)
        pred = cnn_model(input_tensor)  
        pred = pred * output_std + output_mean  
        source_term = pred[0].cpu().numpy()  
    
    return source_term

class ConvNN(nn.Module):

    def __init__(self, in_channels, layer_size1, layer_size2, layer_size3, layer_size4, layer_size5, out_channels, kernel_size):

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

            nn.Conv2d(layer_size3, layer_size4, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(layer_size4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Conv2d(layer_size4, layer_size5, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(layer_size5),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(layer_size5, layer_size4, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(layer_size4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),   

            nn.Conv2d(layer_size4, layer_size3, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(layer_size3),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

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
    
if __name__ == "__main__":

    # Initialize the model, loss function, and optimizer
    cnn_model = ConvNN(in_channels, layer_size1, layer_size2, layer_size3, layer_size4, layer_size5, out_channels, kernel_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Load the data
    # file_path = f"/data3/home/dipayandatta/Subgrid_CGM_Models/data/files/Subgrid CGM Models/Without_cooling/rk2, plm/{resolution[0]}_{resolution[1]}_prateek/bin"
    file_path = DATA_PATH
    cnn_data = nn_data(resolution, downsample)
    input_tensor, output_tensor = cnn_data
    input_tensor = input_tensor.to(device)
    output_tensor = output_tensor.to(device)

    print("Normalizing the input and output tensors")
    input_mean = input_tensor.mean(dim=(0, 2, 3), keepdim=True)  
    input_std = input_tensor.std(dim=(0, 2, 3), keepdim=True)
    input_std[input_std == 0] = 1.0  
    np.save(os.path.join(MODEL_SAVE_DIR, "model_saves", f"cnn_{resolution}_{downsample}_input_mean.npy"), input_mean.cpu().numpy())
    np.save(os.path.join(MODEL_SAVE_DIR, "model_saves", f"cnn_{resolution}_{downsample}_input_std.npy"), input_std.cpu().numpy())
    input_tensor_norm = (input_tensor - input_mean) / input_std

    output_mean = output_tensor.mean(dim=(0, 2, 3), keepdim=True)  
    output_std = output_tensor.std(dim=(0, 2, 3), keepdim=True)
    output_std[output_std == 0] = 1.0 
    np.save(os.path.join(MODEL_SAVE_DIR, "model_saves", f"cnn_{resolution}_{downsample}_output_mean.npy"), output_mean.cpu().numpy())
    np.save(os.path.join(MODEL_SAVE_DIR, "model_saves", f"cnn_{resolution}_{downsample}_output_std.npy"), output_std.cpu().numpy())
    output_tensor_norm = (output_tensor - output_mean) / output_std

    dataset = TensorDataset(input_tensor_norm, output_tensor_norm)
    num_samples = len(dataset)
    print("Number of samples in the dataset:", num_samples)
    indices = np.random.permutation(num_samples)
    train_end = int(0.50 * num_samples)
    val_end = int(0.75 * num_samples)
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
            if len(val_loss_ma) > 1 and val_loss_ma[-1] > np.min(val_loss_ma[:-1]):
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
    torch.save(cnn_model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'model_saves', f'cnn_{resolution}_{downsample}.pth'))

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
    plt.savefig(os.path.join(LOSS_PLOT_DIR, "loss_plots", f"cnn_{resolution}_{downsample}_loss.jpg"), dpi=500)
    plt.close()


