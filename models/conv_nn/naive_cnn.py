# =====================================================================
# NAIVE BASELINE: Plain Encoder-Decoder + KL Divergence
# No physics-informed branches, no physics-informed losses.
# Same data pipeline / normalization / split / augmentation as the
# physics-informed model, so the comparison is fair.
# =====================================================================

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data")))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_SAVE_DIR = os.environ.get(
    "MODEL_SAVES_DIR",
    os.path.join(PROJECT_ROOT, "outputs", "model_saves", "pdf_model_saves_naive"),
)
LOSS_PLOT_DIR = os.environ.get(
    "LOSS_PLOTS_DIR",
    os.path.join(PROJECT_ROOT, "outputs", "loss_plots", "pdf_loss_plots_naive"),
)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOSS_PLOT_DIR, exist_ok=True)

DATA_PATH = os.environ.get("SUBGRID_DATA_PATH", "/path/to/simulation/bin")
CACHE_PATH = os.environ.get("SUBGRID_CACHE_PATH", "/path/to/cache")

import data_preprocess
from data_preprocess import simulation_data

# =========================
# HYPERPARAMETERS  (matched to the physics-informed run)
# =========================
DEFAULT_FINE_RESOLUTION = (1024, 512)
DEFAULT_DOWNSAMPLE = 32


def _parse_resolution(value, default):
    try:
        w, h = value.split(",")
        return int(w.strip()), int(h.strip())
    except (ValueError, AttributeError):
        return default


HYPERPARAMS = {
    "seed": 10,
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
    "kernel_size": 9,
    "num_epochs": 1000,
    "print_every": 50,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "dropout_rate": 0.2,
    "train_fraction": 0.50,
    "val_fraction": 0.25,
    "grad_clip_max_norm": 1.0,
}

np.random.seed(HYPERPARAMS["seed"])
torch.manual_seed(HYPERPARAMS["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(HYPERPARAMS["seed"])

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

resolution   = HYPERPARAMS["resolution"]
downsample   = HYPERPARAMS["downsample"]
in_channels  = HYPERPARAMS["in_channels"]
out_channels = HYPERPARAMS["out_channels"]
layer_size1  = HYPERPARAMS["layer_size1"]
layer_size2  = HYPERPARAMS["layer_size2"]
layer_size3  = HYPERPARAMS["layer_size3"]
layer_size4  = HYPERPARAMS["layer_size4"]
kernel_size  = HYPERPARAMS["kernel_size"]
num_epochs   = HYPERPARAMS["num_epochs"]
print_every  = HYPERPARAMS["print_every"]
batch_size   = HYPERPARAMS["batch_size"]
learning_rate = HYPERPARAMS["learning_rate"]
weight_decay  = HYPERPARAMS["weight_decay"]
dropout_rate  = HYPERPARAMS["dropout_rate"]


# =========================
# DATA
# =========================
def nn_data(resolution, downsample):
    """Load coarse-grained inputs and pixel-PDF targets."""
    sim_data = simulation_data()
    sim_data.down_sample = downsample
    sim_data.resolution = resolution

    folder_path = os.path.join(CACHE_PATH, f"sc{resolution}_{downsample}")
    file_path = DATA_PATH
    if os.path.exists(folder_path):
        sim_data.rho       = np.load(f"{folder_path}/rho.npy")
        sim_data.temp      = np.load(f"{folder_path}/temp.npy")
        sim_data.pressure  = np.load(f"{folder_path}/pressure.npy")
        sim_data.ux        = np.load(f"{folder_path}/ux.npy")
        sim_data.uy        = np.load(f"{folder_path}/uy.npy")
        sim_data.eint      = np.load(f"{folder_path}/eint.npy")
        sim_data.ps        = np.load(f"{folder_path}/ps.npy")
    else:
        sim_data.input_data(file_path)
        os.makedirs(folder_path, exist_ok=True)
        np.save(f"{folder_path}/rho.npy",      sim_data.rho)
        np.save(f"{folder_path}/temp.npy",     sim_data.temp)
        np.save(f"{folder_path}/pressure.npy", sim_data.pressure)
        np.save(f"{folder_path}/ux.npy",       sim_data.ux)
        np.save(f"{folder_path}/uy.npy",       sim_data.uy)
        np.save(f"{folder_path}/eint.npy",     sim_data.eint)
        np.save(f"{folder_path}/ps.npy",       sim_data.ps)

    print("Input data loaded")

    shape = (
        sim_data.rho.shape[0],
        sim_data.rho.shape[1] // sim_data.down_sample,
        sim_data.rho.shape[2] // sim_data.down_sample,
    )
    fields = ["rho", "temp", "ux", "uy", "ps"]
    cg = {f"cg_{f}": np.zeros(shape) for f in fields}

    for i in range(sim_data.rho.shape[0]):
        for f in fields:
            cg[f"cg_{f}"][i] = sim_data.coarse_grain(getattr(sim_data, f)[i])

    temp_pdf = sim_data.calc_pixel_pdf(bins=out_channels)
    temp_pdf /= temp_pdf.sum(axis=1, keepdims=True)

    input_tensor = torch.cat(
        [torch.from_numpy(cg[f"cg_{f}"]).unsqueeze(1).float() for f in fields],
        dim=1,
    )
    output_tensor = torch.from_numpy(temp_pdf).float()
    return input_tensor, output_tensor


# =========================
# NAIVE MODEL  (plain encoder-decoder + softmax)
# =========================
class NaiveConvNN(nn.Module):
    """
    Plain convolutional encoder-decoder.
    Input  : (B, 5, H, W)        [rho, temp, ux, uy, ps]
    Output : (B, 40, H, W)        bin probabilities (softmax over dim=1)
    No mixing-layer features, no gate, no thresholding.
    """
    def __init__(self, in_channels, s1, s2, s3, s4, out_channels, kernel_size):
        super().__init__()
        pad = kernel_size // 2

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, s1, kernel_size, padding=pad),
            nn.BatchNorm2d(s1), nn.ReLU(), nn.Dropout2d(dropout_rate),
            nn.Conv2d(s1, s2, kernel_size, padding=pad),
            nn.BatchNorm2d(s2), nn.ReLU(), nn.Dropout2d(dropout_rate),
            nn.Conv2d(s2, s3, kernel_size, padding=pad),
            nn.BatchNorm2d(s3), nn.ReLU(),
            nn.Conv2d(s3, s4, kernel_size, padding=pad),
            nn.BatchNorm2d(s4), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(s4, s3, kernel_size, padding=pad),
            nn.BatchNorm2d(s3), nn.ReLU(),
            nn.Conv2d(s3, s2, kernel_size, padding=pad),
            nn.BatchNorm2d(s2), nn.ReLU(),
            nn.Conv2d(s2, s1, kernel_size, padding=pad),
            nn.BatchNorm2d(s1), nn.ReLU(),
            nn.Conv2d(s1, out_channels, kernel_size=1),
        )

    def forward(self, x):
        # Return log-probs (KL signature expects log-probs).
        logits = self.decoder(self.encoder(x))
        return F.log_softmax(logits, dim=1)

    def predict_pdf(self, x):
        return torch.exp(self.forward(x))


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
    Predict pixel temperature PDFs for a given snapshot using NaiveConvNN.
    Returns: (bins, nx, ny)
    """
    sim_data = simulation_data()
    sim_data.down_sample = downsample
    sim_data.resolution = resolution

    shape = (resolution[0] // downsample, resolution[1] // downsample)

    fields = ["rho", "temp", "ux", "uy", "ps"]
    cg = {f"cg_{field}": np.zeros(shape) for field in fields}

    for field in fields:
        cg[f"cg_{field}"] = sim_data.coarse_grain(locals()[field])

    input_tensors = [
        torch.from_numpy(cg[f"cg_{f}"]).unsqueeze(0).float() for f in fields
    ]
    input_tensor = torch.cat(input_tensors, dim=0).unsqueeze(0).to(device)

    input_mean = torch.tensor(
        np.load(os.path.join(MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}_input_mean.npy")),
        dtype=torch.float32,
    ).to(device)
    input_std = torch.tensor(
        np.load(os.path.join(MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}_input_std.npy")),
        dtype=torch.float32,
    ).to(device)

    input_tensor = (input_tensor - input_mean) / input_std

    model_path = os.path.join(MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}.pth")
    state_dict = torch.load(model_path, map_location=device)
    ckpt_ksize = kernel_size
    if "encoder.0.weight" in state_dict:
        ckpt_ksize = state_dict["encoder.0.weight"].shape[-1]

    cnn_model = NaiveConvNN(
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
        pdf = cnn_model.predict_pdf(input_tensor)

    return pdf.squeeze(0).cpu().numpy()


# =========================
# KL DIVERGENCE LOSS
# =========================
# nn.KLDivLoss expects  (input=log_probs, target=probs), both same shape.
criterion = nn.KLDivLoss(reduction="batchmean")


# =========================
# TRAINING
# =========================
if __name__ == "__main__":
    print("Training NAIVE encoder-decoder baseline (KL-only)")

    model = NaiveConvNN(
        in_channels, layer_size1, layer_size2,
        layer_size3, layer_size4, out_channels, kernel_size,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)

    input_tensor, output_tensor = nn_data(resolution, downsample)
    input_tensor = input_tensor.to(device)
    output_tensor = output_tensor.to(device)

    # Numerical stability for the target PDF.
    output_tensor = torch.clamp(output_tensor, min=1e-12)
    output_tensor = output_tensor / output_tensor.sum(dim=1, keepdim=True)

    print("Normalizing input tensor")
    input_mean = input_tensor.mean(dim=(0, 2, 3), keepdim=True)
    input_std  = input_tensor.std(dim=(0, 2, 3), keepdim=True)
    input_std[input_std == 0] = 1.0
    np.save(os.path.join(MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}_input_mean.npy"),
            input_mean.cpu().numpy())
    np.save(os.path.join(MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}_input_std.npy"),
            input_std.cpu().numpy())

    input_tensor_norm = (input_tensor - input_mean) / input_std
    dataset = TensorDataset(input_tensor_norm, output_tensor)

    num_samples = len(dataset)
    print("Number of samples:", num_samples)

    indices = np.random.permutation(num_samples)
    train_end = int(HYPERPARAMS["train_fraction"] * num_samples)
    val_end   = int((HYPERPARAMS["train_fraction"] + HYPERPARAMS["val_fraction"]) * num_samples)

    train_dataset = Subset(dataset, indices[:train_end])
    val_dataset   = Subset(dataset, indices[train_end:val_end])
    test_dataset  = Subset(dataset, indices[val_end:])

    train_loader      = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(val_dataset,   batch_size=batch_size)
    test_loader       = DataLoader(test_dataset,  batch_size=batch_size)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate,
        steps_per_epoch=len(train_loader), epochs=num_epochs, pct_start=0.2,
    )

    epochs_array, train_loss_arr, val_loss_arr = [], [], []

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            # Same augmentations as the physics-informed run.
            if torch.rand(1).item() > 0.5:
                inputs = torch.flip(inputs, [3])
                labels = torch.flip(labels, [3])
            if torch.rand(1).item() > 0.5:
                inputs = torch.flip(inputs, [2])
                labels = torch.flip(labels, [2])

            log_probs = model(inputs)            # (B, bins, H, W)
            loss = criterion(log_probs, labels)  # KL(labels || pred)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=HYPERPARAMS["grad_clip_max_norm"])
            optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            train_loss_total = 0.0
            for x_b, y_b in train_loader:
                train_loss_total += criterion(model(x_b), y_b).item()
            train_loss = train_loss_total / len(train_loader)

            val_loss_total = 0.0
            for x_b, y_b in validation_loader:
                val_loss_total += criterion(model(x_b), y_b).item()
            val_loss = val_loss_total / len(validation_loader)

        if (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        epochs_array.append(epoch + 1)
        train_loss_arr.append(train_loss)
        val_loss_arr.append(val_loss)

        # Early stopping (same heuristic as the physics-informed run)
        window_size = 200
        if len(val_loss_arr) >= window_size:
            val_loss_ma = np.convolve(val_loss_arr,
                                      np.ones(window_size) / window_size,
                                      mode="valid")
            if (len(val_loss_ma) > 1
                and val_loss_ma[-1] > np.min(val_loss_ma[:-1])
                and epoch >= 499):
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Test
    model.eval()
    with torch.no_grad():
        test_loss_total = 0.0
        for x_b, y_b in test_loader:
            test_loss_total += criterion(model(x_b), y_b).item()
        test_loss = test_loss_total / len(test_loader)
    print(f"Test Loss: {test_loss:.6f}")

    torch.save(model.state_dict(),
               os.path.join(MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}.pth"))

    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_array, train_loss_arr, label="Train Loss")
    plt.plot(epochs_array, val_loss_arr,   label="Validation Loss")
    plt.axhline(train_loss_arr[-1], linestyle="--")
    plt.axhline(val_loss_arr[-1],   linestyle="--")
    plt.axhline(test_loss, linestyle="--", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("KL Divergence")
    plt.title("Naive Encoder-Decoder — Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(LOSS_PLOT_DIR,
                             f"cnn_{resolution}_{downsample}_loss.jpg"), dpi=500)
    plt.close()