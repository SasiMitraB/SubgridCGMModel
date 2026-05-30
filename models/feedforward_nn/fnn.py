# Feedforward Neural Network script for predicting the fmcl source term in the subgrid CGM model

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
input_size = 48
hidden_size1 = 256
hidden_size2 = 128
hidden_size3 = 64
output_size = 1
num_epochs = 1000
print_every = 50
batch_size = (resolution[0] // downsample) * (resolution[1] // downsample) // 2
learning_rate = 5e-4
weight_decay = 1e-4
dropout_rate = 0.4

def nn_data(filepath: str, resolution: tuple, downsample: int) -> tuple:
    """ A function to load the data and return the inputs and outputs for the neural network."""
    
    sim_data = simulation_data()
    sim_data.down_sample = downsample
    sim_data.resolution = resolution

    folder_path = f"../data/data_saves/{resolution}_{downsample}"
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
    fields = ['rho', 'temp', 'pressure', 'ux', 'uy', 'eint', 'ps', 'fmcl']
    cg = {f'cg_{field}': np.zeros(shape) for field in fields}
    grad = {f'grad_{field}_{axis}': np.zeros(shape) for field in fields for axis in ['x', 'y']}
    hessian = {f'hessian_{field}_{axis}': np.zeros(shape) for field in fields for axis in ['xx', 'xy', 'yy']}

    dx = sim_data.total_length / (sim_data.rho.shape[1] // sim_data.down_sample)
    dy = sim_data.total_width / (sim_data.rho.shape[2] // sim_data.down_sample)

    for i in range(sim_data.rho.shape[0]):

        for field in fields:
            if field in ['rho', 'temp', 'pressure', 'ux', 'uy', 'eint', 'ps']:
                cg[f'cg_{field}'][i] = sim_data.coarse_grain(getattr(sim_data, field)[i])
            elif field in ['fmcl']:
                cg[f'cg_{field}'][i] = sim_data.calc_fmcl(sim_data.rho[i], sim_data.temp[i])
            for axis in ['x', 'y']:
                if axis == 'x':
                    grad[f'grad_{field}_{axis}'][i] = np.gradient(cg[f'cg_{field}'][i], dy, dx)[1]
                elif axis == 'y':
                    grad[f'grad_{field}_{axis}'][i] = np.gradient(cg[f'cg_{field}'][i], dy, dx)[0]
            for hess_axis in ['xx', 'xy', 'yy']:
                if hess_axis == 'xx':
                    hessian[f'hessian_{field}_{hess_axis}'][i] = np.gradient(grad[f'grad_{field}_x'][i], dy, dx)[1]
                elif hess_axis == 'xy':
                    hessian[f'hessian_{field}_{hess_axis}'][i] = np.gradient(grad[f'grad_{field}_x'][i], dy, dx)[0]
                elif hess_axis == 'yy':
                    hessian[f'hessian_{field}_{hess_axis}'][i] = np.gradient(grad[f'grad_{field}_y'][i], dy, dx)[1]

    cg_flat = {key: val.reshape(-1) for key, val in cg.items()}
    grad_flat = {key: val.reshape(-1) for key, val in grad.items()}
    hessian_flat = {key: val.reshape(-1) for key, val in hessian.items()}
    del cg, grad, hessian

    source_term = sim_data.calc_source_term()
    source_term_flat = source_term.reshape(-1)
    del source_term

    fmcl_vals = cg_flat['cg_fmcl']
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1) ** (1 / 2)
    bin_indices = np.digitize(fmcl_vals, bins) - 1
    binned_indices = [np.where(bin_indices == i)[0] for i in range(n_bins)]
    n_min = min(len(indices) for indices in binned_indices if len(indices) > 0)
    np.random.seed(10)
    balanced_indices = np.concatenate([np.random.choice(indices, n_min, replace=False) for indices in binned_indices if len(indices) >= n_min])
    np.random.shuffle(balanced_indices)

    cg_balanced = {k: v[balanced_indices] for k, v in cg_flat.items()}
    grad_balanced = {k: v[balanced_indices] for k, v in grad_flat.items()}
    hessian_balanced = {k: v[balanced_indices] for k, v in hessian_flat.items()}
    source_term_balanced = source_term_flat[balanced_indices]

    del cg_flat, grad_flat, hessian_flat, source_term_flat, sim_data

    all_arrays = [torch.tensor(cg_balanced[k]) for k in cg_balanced] + [torch.tensor(grad_balanced[k]) for k in grad_balanced] \
        + [torch.tensor(hessian_balanced[k]) for k in hessian_balanced]
    input_tensor = torch.stack(all_arrays, dim=1)
    input_tensor = input_tensor.type(torch.float32)
    output_tensor = torch.tensor(source_term_balanced, dtype=torch.float32)
    
    return input_tensor, output_tensor

def snapshot_pred(rho: np.ndarray, temp: np.ndarray, pressure: np.ndarray, ux: np.ndarray, uy: np.ndarray, eint: np.ndarray, ps: np.ndarray, downsample: int, resolution: np.ndarray) -> np.ndarray:
    """ A function to predict the source term for a given snapshot using the trained model."""

    sim_data = simulation_data()
    sim_data.down_sample = downsample
    sim_data.resolution = resolution

    shape = (resolution[0] // downsample, resolution[1] // downsample)
    fields = ['rho', 'temp', 'pressure', 'ux', 'uy', 'eint', 'ps', 'fmcl']
    cg = {f'cg_{field}': np.zeros(shape) for field in fields}
    grad = {f'grad_{field}_{axis}': np.zeros(shape) for field in fields for axis in ['x', 'y']}
    hessian = {f'hessian_{field}_{axis}': np.zeros(shape) for field in fields for axis in ['xx', 'xy', 'yy']}

    dx = sim_data.total_length / (resolution[0] // downsample)
    dy = sim_data.total_width / (resolution[1] // downsample)

    for field in fields:
        if field in ['rho', 'temp', 'pressure', 'ux', 'uy', 'eint', 'ps']:
            cg[f'cg_{field}'] = sim_data.coarse_grain(locals()[field])
        elif field in ['fmcl']:
            cg[f'cg_{field}'] = sim_data.calc_fmcl(rho, temp)
        for axis in ['x', 'y']:
            if axis == 'x':
                grad[f'grad_{field}_{axis}'] = np.gradient(cg[f'cg_{field}'], dy, dx)[1]
            elif axis == 'y':
                grad[f'grad_{field}_{axis}'] = np.gradient(cg[f'cg_{field}'], dy, dx)[0]
        for hess_axis in ['xx', 'xy', 'yy']:    
            if hess_axis == 'xx':
                hessian[f'hessian_{field}_{hess_axis}'] = np.gradient(grad[f'grad_{field}_x'], dy, dx)[1]
            elif hess_axis == 'xy':
                hessian[f'hessian_{field}_{hess_axis}'] = np.gradient(grad[f'grad_{field}_x'], dy, dx)[0]
            elif hess_axis == 'yy':
                hessian[f'hessian_{field}_{hess_axis}'] = np.gradient(grad[f'grad_{field}_y'], dy, dx)[1]

    cg_flat = {key: val.reshape(-1) for key, val in cg.items()}
    grad_flat = {key: val.reshape(-1) for key, val in grad.items()}
    hessian_flat = {key: val.reshape(-1) for key, val in hessian.items()}
    all_arrays = [torch.tensor(cg_flat[k]) for k in cg_flat] + [torch.tensor(grad_flat[k]) for k in grad_flat] \
        + [torch.tensor(hessian_flat[k]) for k in hessian_flat]
    input_tensor = torch.stack(all_arrays, dim=1)
    input_tensor = input_tensor.type(torch.float32)
    input_mean = np.load(f"{MODEL_SAVE_DIR}/model_saves/fnn_{sim_data.resolution}_{downsample}_input_mean.npy")
    # input_mean = np.load(f"{MODEL_SAVE_DIR}/model_saves/fnn_(1024, 1024)_8_input_mean.npy")
    input_std = np.load(f"{MODEL_SAVE_DIR}/model_saves/fnn_{sim_data.resolution}_{downsample}_input_std.npy")
    # input_std = np.load(f"{MODEL_SAVE_DIR}/model_saves/fnn_(1024, 1024)_8_input_std.npy")
    input_tensor = (input_tensor - input_mean) / input_std
    input_tensor = input_tensor.to(device)

    model_path = f"{MODEL_SAVE_DIR}/model_saves/fnn_{sim_data.resolution}_{downsample}.pth"
    # model_path = f"{MODEL_SAVE_DIR}/model_saves/fnn_(1024, 1024)_8.pth"
    global input_size, hidden_size1, hidden_size2, hidden_size3, output_size, fnn_model
    fnn_model = feedforwardNN(input_size, output_size, hidden_size1, hidden_size2, hidden_size3).to(device)
    fnn_model.load_state_dict(torch.load(model_path, map_location=device))
    fnn_model.eval()

    with torch.no_grad():
        pred = fnn_model(input_tensor).squeeze()
        pred = pred.cpu().numpy()
        output_mean = np.load(f"{MODEL_SAVE_DIR}/model_saves/fnn_{sim_data.resolution}_{downsample}_output_mean.npy")
        # output_mean = np.load(f"{MODEL_SAVE_DIR}/model_saves/fnn_(1024, 1024)_8_output_mean.npy")
        output_std = np.load(f"{MODEL_SAVE_DIR}/model_saves/fnn_{sim_data.resolution}_{downsample}_output_std.npy")
        # output_std = np.load(f"{MODEL_SAVE_DIR}/model_saves/fnn_(1024, 1024)_8_output_std.npy")
        pred = pred * output_std + output_mean
        pred = pred.reshape(shape)

    return pred

class feedforwardNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim1, hidden_dim2, hidden_dim3):
        super(feedforwardNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.bn3 = nn.BatchNorm1d(hidden_dim3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc4 = nn.Linear(hidden_dim3, output_dim)

    def forward(self, x):

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.dropout3(out)

        out = self.fc4(out)

        return out
    
if __name__ == "__main__":

    # Initialize the model, loss function and optimizer
    fnn_model = feedforwardNN(input_size, output_size, hidden_size1, hidden_size2, hidden_size3).to(device)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(fnn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Load the data
    # file_path = f"/data3/home/dipayandatta/Subgrid_CGM_Models/data/files/Subgrid CGM Models/Without_cooling/rk2, plm/{resolution[0]}_{resolution[1]}_prateek/bin"
    file_path = DATA_PATH
    fnn_data = nn_data(file_path, resolution, downsample)
    input_tensor, output_tensor = fnn_data
    input_tensor = input_tensor.to(device)
    output_tensor = output_tensor.to(device)

    input_mean = input_tensor.mean(dim=0, keepdim=True)
    input_std = input_tensor.std(dim=0, keepdim=True)
    input_std[input_std == 0] = 1.0
    np.save(os.path.join(MODEL_SAVE_DIR, "model_saves", f"fnn_{resolution}_{downsample}_input_mean.npy"), input_mean.cpu().numpy())
    np.save(os.path.join(MODEL_SAVE_DIR, "model_saves", f"fnn_{resolution}_{downsample}_input_std.npy"), input_std.cpu().numpy())
    input_tensor_norm = (input_tensor - input_mean) / input_std

    output_mean = output_tensor.mean()
    output_std = output_tensor.std()
    print("Output mean:", output_mean.item())
    print("Output std:", output_std.item())
    if output_std == 0:
        output_std = 1.0
    np.save(os.path.join(MODEL_SAVE_DIR, "model_saves", f"fnn_{resolution}_{downsample}_output_mean.npy"), output_mean.cpu().numpy())
    np.save(os.path.join(MODEL_SAVE_DIR, "model_saves", f"fnn_{resolution}_{downsample}_output_std.npy"), output_std.cpu().numpy())
    output_tensor_norm = (output_tensor - output_mean) / output_std

    dataset = TensorDataset(input_tensor_norm, output_tensor_norm)
    num_samples = len(dataset)
    print("Number of samples in the dataset:", num_samples)
    indices = np.random.permutation(num_samples)
    train_end = int(0.6 * num_samples)
    val_end = int(0.8 * num_samples)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
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
            outputs = fnn_model(inputs)
            loss = criterion(outputs.squeeze(), labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        fnn_model.eval()
        with torch.no_grad():

            train_preds = []
            train_targets = []
            for x_batch, y_batch in train_loader:
                preds = fnn_model(x_batch).squeeze()
                train_preds.append(preds.cpu())
                train_targets.append(y_batch.cpu())
            train_preds = torch.cat(train_preds)
            train_targets = torch.cat(train_targets)
            train_loss = criterion(train_preds, train_targets).item()
            train_ss_res = torch.sum((train_targets - train_preds) ** 2)
            train_ss_tot = torch.sum((train_targets - torch.mean(train_targets)) ** 2)
            train_r2 = 1 - train_ss_res / train_ss_tot

            val_preds = []
            val_targets = []
            for x_batch, y_batch in validation_loader:
                preds = fnn_model(x_batch).squeeze()
                val_preds.append(preds.cpu())
                val_targets.append(y_batch.cpu())
            val_preds = torch.cat(val_preds)
            val_targets = torch.cat(val_targets)
            val_loss = criterion(val_preds, val_targets).item()
            val_ss_res = torch.sum((val_targets - val_preds) ** 2)
            val_ss_tot = torch.sum((val_targets - torch.mean(val_targets)) ** 2)
            val_r2 = 1 - val_ss_res / val_ss_tot

        if (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train R2: {train_r2:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val R2: {val_r2:.4f}")
        
        epochs_array.append(epoch + 1)
        train_loss_arr.append(train_loss)
        train_r2_arr.append(train_r2)
        val_loss_arr.append(val_loss)
        val_r2_arr.append(val_r2)

        window_size = 200
        if len(val_loss_arr) >= window_size:
            val_loss_ma = np.convolve(val_loss_arr, np.ones(window_size)/window_size, mode='valid')
            if len(val_loss_ma) > 1 and val_loss_ma[-1] > np.min(val_loss_ma[:-1]):
                print(f"Early stopping at epoch {epoch+1} due to no improvement in moving average validation loss.")
                break

        fnn_model.train()

    # Testing the model
    fnn_model.eval()
    with torch.no_grad():
        test_preds = []
        test_targets = []
        for x_batch, y_batch in test_loader:
            preds = fnn_model(x_batch).squeeze()
            test_preds.append(preds.cpu())
            test_targets.append(y_batch.cpu())
        test_preds = torch.cat(test_preds)
        test_targets = torch.cat(test_targets)

        # output_mean = np.load(os.path.join(MODEL_SAVE_DIR, "model_saves", f"fnn_{resolution}_{downsample}_output_mean.npy"))
        # output_std = np.load(os.path.join(MODEL_SAVE_DIR, "model_saves", f"fnn_{resolution}_{downsample}_output_std.npy"))
        # tp = (test_preds.detach().cpu().numpy())* output_std + output_mean
        # tt = (test_targets.detach().cpu().numpy())* output_std + output_mean
        # for i in range(tp.shape[0]):
        #     print("Target:", tt[i], "Predicted:", tp[i])
        # print(len(tp[tp < 0]), len(tp[tp > 0]), len(tt[tt < 0]), len(tt[tt > 0]))
        # plt.plot(range(tp.shape[0]), tp, label='Predicted', color='g', lw=0.1)
        # plt.plot(range(tt.shape[0]), tt, label='Target', color='b', lw=0.1)
        # plt.xlabel('Sample')
        # plt.ylabel('Source Term')
        # plt.title('Predicted vs Target Source Term')
        # plt.legend()
        # plt.savefig('Predicted vs Target Source Term.jpg', dpi=500)
        # plt.close()

        test_loss = criterion(test_preds, test_targets).item()
        test_ss_res = torch.sum((test_targets - test_preds) ** 2)
        test_ss_tot = torch.sum((test_targets - torch.mean(test_targets)) ** 2)
        test_r2 = 1 - test_ss_res / test_ss_tot
        print(f"Test Loss: {test_loss:.4f}, Test R2: {test_r2:.4f}")

    # Save the model
    torch.save(fnn_model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'model_saves', f'fnn_{resolution}_{downsample}.pth'))

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
    plt.savefig(os.path.join(LOSS_PLOT_DIR, "loss_plots", f"fnn_{resolution}_{downsample}_loss.jpg"), dpi=500)
    plt.close()
