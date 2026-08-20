"""
random_snapshot_training.py
---------------------------
Snapshot-Split Random Crop Training Pipeline for Subgrid CGM PDF Model.

Optimized Lightweight Preprocessing:
- Caches 64x coarse-grained fields (shape: 32x16) and per-cell temperature PDFs
  (40 bins) instead of giant fine grids.
- Total disk/RAM usage for all 1001 snapshots is < 100 MB.
- Draws random 16x8 coarse subgrid crops per snapshot each epoch.
- Maintains snapshot-level split (disjoint train/val/test snapshots).
- Validation uses training normalization statistics to avoid normalization drift.
- Exponential Moving Average (EMA) of normalization stats is saved for inference.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ensure project root and data directory are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

# Import model architecture and loss components from pdf_cnn
from models.conv_nn.pdf_cnn import (
    ConvNN,
    GatedPDFLoss,
    MixingLayerFeatures,
    MixingLayerGate,
    GatedThresholdedSoftmax,
    MeanTemperatureLoss,
    EmissivityLoss,
    ZonedSymmetricKLLoss,
    LeakageLoss,
    HYPERPARAMS,
    DEFAULT_FINE_RESOLUTION,
    DEFAULT_DOWNSAMPLE,
    _parse_resolution,
)

try:
    import data_preprocess
    from data_preprocess import simulation_data
except ImportError:
    simulation_data = None


# ===================================================================== #
#  Device Helper                                                        #
# ===================================================================== #
def get_device(device_override: str | None = None) -> torch.device:
    """Select best available device (CUDA, MPS, CPU) or respect override."""
    if device_override:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ===================================================================== #
#  Step 1: Snapshot Splitter                                            #
# ===================================================================== #
def split_snapshots(
    N_snaps: int,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits N_snaps snapshot indices into train / val / test with no overlap.
    The remaining fraction (1 - train_frac - val_frac) goes to test.

    Returns three index arrays: train_idx, val_idx, test_idx.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N_snaps)

    t_end = int(train_frac * N_snaps)
    v_end = int((train_frac + val_frac) * N_snaps)

    return idx[:t_end], idx[t_end:v_end], idx[v_end:]


# ===================================================================== #
#  Coarse-Grained Data Cacher & Loader                                  #
# ===================================================================== #
def load_or_create_coarse_data(
    data_path: str,
    cache_path: str,
    resolution: tuple[int, int] = (2048, 1024),
    downsample: int = 64,
    out_channels: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads or computes coarse-grained (32x16) physical fields and temperature PDFs.

    Returns:
    --------
    cg_inputs : np.ndarray
        Shape (N_snaps, 5, cH, cW), dtype float32 (rho, temp, ux, uy, ps)
    cg_pdfs : np.ndarray
        Shape (N_snaps, out_channels, cH, cW), dtype float32
    """
    cache_dir = Path(cache_path) / f"coarse_{resolution[0]}x{resolution[1]}_ds{downsample}_bins{out_channels}"
    inputs_file = cache_dir / "cg_inputs.npy"
    pdfs_file = cache_dir / "cg_pdfs.npy"

    if inputs_file.exists() and pdfs_file.exists():
        try:
            cg_inputs = np.load(inputs_file)
            cg_pdfs = np.load(pdfs_file)
            if cg_inputs.ndim == 4 and cg_inputs.shape[0] > 0 and cg_pdfs.shape[0] == cg_inputs.shape[0]:
                print(f"Loaded coarse data from cache: {cache_dir} ({cg_inputs.shape[0]} snapshots, shape: {cg_inputs.shape[2:]})")
                return cg_inputs, cg_pdfs
            print(f"Cached data in {cache_dir} was empty or corrupted. Recomputing...")
        except Exception as e:
            print(f"Failed to read cache {cache_dir}: {e}. Recomputing...")

    try:
        import bin_convert
    except ImportError:
        bin_convert = None

    if bin_convert is None:
        raise RuntimeError("`bin_convert` module not found to process simulation binary files.")

    data_dir = Path(data_path)
    if not data_dir.exists():
        raise FileNotFoundError(f"Binary directory does not exist: {data_dir}")

    # Find all hydro_w binary files
    w_files = sorted(
        [f for f in os.listdir(data_dir) if ".hydro_w." in f and f.endswith(".bin")],
        key=lambda f: int(f.split(".")[-2]),
    )

    if not w_files:
        raise ValueError(
            f"No valid .bin snapshots found in {data_path}. "
            f"Expected binary files matching pattern '*.hydro_w.*.bin'."
        )

    num_snaps = len(w_files)
    H, W = resolution[0], resolution[1]
    cH = H // downsample  # 2048 // 64 = 32
    cW = W // downsample  # 1024 // 64 = 16
    T_edges = np.logspace(3.0, 7.0, out_channels + 1)

    print(f"Processing & coarse-graining {num_snaps} snapshots ({H}x{W} -> {cH}x{cW}, ds={downsample})...")

    # Physics constants matching simulation_data
    gamma = 5.0 / 3.0
    mu = 0.62
    kb = 1.3807e-16
    P_unit = 1.59916e-14

    cg_inputs = np.zeros((num_snaps, 5, cH, cW), dtype=np.float32)
    cg_pdfs = np.zeros((num_snaps, out_channels, cH, cW), dtype=np.float32)

    cwd = os.getcwd()
    try:
        os.chdir(data_dir)
        for idx, fname in enumerate(tqdm(w_files, desc="Coarse-graining binary snapshots")):
            file_data = bin_convert.read_binary(fname)
            rho_fine = bin_convert.make_2D_array(file_data, "dens").astype(np.float32)
            ux_fine = bin_convert.make_2D_array(file_data, "velx").astype(np.float32)
            uy_fine = bin_convert.make_2D_array(file_data, "vely").astype(np.float32)
            eint_fine = bin_convert.make_2D_array(file_data, "eint").astype(np.float32)
            ps_fine = bin_convert.make_2D_array(file_data, "s_00").astype(np.float32)

            pressure_fine = (gamma - 1.0) * eint_fine
            temp_fine = (pressure_fine * P_unit / rho_fine) * (mu / kb)

            # Block averaging for inputs
            cg_inputs[idx, 0] = rho_fine.reshape(cH, downsample, cW, downsample).mean(axis=(1, 3))
            cg_inputs[idx, 1] = temp_fine.reshape(cH, downsample, cW, downsample).mean(axis=(1, 3))
            cg_inputs[idx, 2] = ux_fine.reshape(cH, downsample, cW, downsample).mean(axis=(1, 3))
            cg_inputs[idx, 3] = uy_fine.reshape(cH, downsample, cW, downsample).mean(axis=(1, 3))
            cg_inputs[idx, 4] = ps_fine.reshape(cH, downsample, cW, downsample).mean(axis=(1, 3))

            # Temperature PDF calculation per coarse cell
            temp_blocks = temp_fine.reshape(cH, downsample, cW, downsample).swapaxes(1, 2).reshape(cH, cW, -1)
            for j in range(cH):
                for k in range(cW):
                    hist, _ = np.histogram(temp_blocks[j, k], bins=T_edges)
                    total = hist.sum()
                    if total > 0:
                        cg_pdfs[idx, :, j, k] = hist / total
                    else:
                        cg_pdfs[idx, :, j, k] = 1.0 / out_channels
    finally:
        os.chdir(cwd)

    # Save compact cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(inputs_file, cg_inputs)
    np.save(pdfs_file, cg_pdfs)
    print(f"Saved coarse data cache to {cache_dir} (Total size: ~{(cg_inputs.nbytes + cg_pdfs.nbytes) / (1024**2):.1f} MB)")

    return cg_inputs, cg_pdfs


# ===================================================================== #
#  Step 2: SnapshotCropDataset                                          #
# ===================================================================== #
class SnapshotCropDataset(Dataset):
    """
    Random-crop dataset over coarse-grained (32x16) snapshots.
    Extracts (16x8) random subregions each epoch.

    Parameters
    ----------
    cg_inputs : np.ndarray
        (N_snaps_full, 5, cH_full, cW_full) coarse physical fields
    cg_pdfs : np.ndarray
        (N_snaps_full, out_channels, cH_full, cW_full) discrete PDFs
    snap_indices : array-like
        Which snapshot indices this dataset split owns.
    crop_h_cg, crop_w_cg : int
        Spatial size of each coarse crop (default 16x8).
    n_crops_per_snap : int
        Number of random crops to draw per snapshot on each resample().
    ema_alpha : float
        EMA smoothing factor for normalisation statistics across epochs.
    """

    def __init__(
        self,
        cg_inputs: np.ndarray,
        cg_pdfs: np.ndarray,
        snap_indices: np.ndarray,
        crop_h_cg: int = 16,
        crop_w_cg: int = 8,
        n_crops_per_snap: int = 8,
        ema_alpha: float = 0.9,
    ):
        self.cg_inputs = cg_inputs
        self.cg_pdfs = cg_pdfs
        self.snap_indices = np.asarray(snap_indices)
        self.crop_h_cg = crop_h_cg
        self.crop_w_cg = crop_w_cg
        self.n_crops_per_snap = n_crops_per_snap
        self.ema_alpha = ema_alpha

        self.cH_full = cg_inputs.shape[2]  # 32
        self.cW_full = cg_inputs.shape[3]  # 16
        self.n_fields = cg_inputs.shape[1]  # 5
        self.out_channels = cg_pdfs.shape[1]  # 40

        assert self.cH_full >= crop_h_cg, f"Coarse height {self.cH_full} < crop {crop_h_cg}"
        assert self.cW_full >= crop_w_cg, f"Coarse width {self.cW_full} < crop {crop_w_cg}"

        # Public normalisation stats
        self.input_mean: torch.Tensor = None  # (1, C, 1, 1)
        self.input_std: torch.Tensor = None  # (1, C, 1, 1)

        # EMA accumulators
        self._ema_mean: torch.Tensor = None
        self._ema_std: torch.Tensor = None
        self._first_resample = True

        # Data buffers
        self._inputs: torch.Tensor = None
        self._pdfs: torch.Tensor = None
        self._rho_cg: torch.Tensor = None
        self._temp_cg: torch.Tensor = None

        self.resample()

    def resample(self):
        """
        Draw n_crops_per_snap random 16x8 crops per snapshot.
        Recomputes normalization stats and updates the EMA.
        """
        total = len(self.snap_indices) * self.n_crops_per_snap
        inputs_buf = np.zeros((total, self.n_fields, self.crop_h_cg, self.crop_w_cg), dtype=np.float32)
        pdfs_buf = np.zeros((total, self.out_channels, self.crop_h_cg, self.crop_w_cg), dtype=np.float32)

        max_y = self.cH_full - self.crop_h_cg + 1
        max_x = self.cW_full - self.crop_w_cg + 1

        idx = 0
        for snap in self.snap_indices:
            for _ in range(self.n_crops_per_snap):
                y0 = np.random.randint(0, max_y)
                x0 = np.random.randint(0, max_x)

                inputs_buf[idx] = self.cg_inputs[snap, :, y0 : y0 + self.crop_h_cg, x0 : x0 + self.crop_w_cg]
                pdfs_buf[idx] = self.cg_pdfs[snap, :, y0 : y0 + self.crop_h_cg, x0 : x0 + self.crop_w_cg]
                idx += 1

        self._inputs = torch.from_numpy(inputs_buf)
        self._pdfs = torch.from_numpy(pdfs_buf)

        # Clamping PDFs for numerical stability
        self._pdfs = torch.clamp(self._pdfs, min=1e-12)
        self._pdfs = self._pdfs / self._pdfs.sum(dim=1, keepdim=True)

        self._rho_cg = self._inputs[:, 0:1]  # rho channel
        self._temp_cg = self._inputs[:, 1:2]  # temp channel

        # Recompute current-epoch stats
        cur_mean = self._inputs.mean(dim=(0, 2, 3), keepdim=True)
        cur_std = self._inputs.std(dim=(0, 2, 3), keepdim=True)
        cur_std[cur_std == 0] = 1.0

        # EMA update
        if self._first_resample:
            self._ema_mean = cur_mean.clone()
            self._ema_std = cur_std.clone()
            self._first_resample = False
        else:
            alpha = self.ema_alpha
            self._ema_mean = alpha * self._ema_mean + (1.0 - alpha) * cur_mean
            self._ema_std = alpha * self._ema_std + (1.0 - alpha) * cur_std

        self.input_mean = cur_mean
        self.input_std = cur_std

    def set_norm_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """Override internal stats with externally provided ones."""
        self.input_mean = mean
        self.input_std = std

    @property
    def ema_mean(self) -> torch.Tensor:
        return self._ema_mean

    @property
    def ema_std(self) -> torch.Tensor:
        return self._ema_std

    def __len__(self) -> int:
        return self._inputs.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_norm = (self._inputs[i] - self.input_mean.squeeze(0)) / self.input_std.squeeze(0)
        return x_norm, self._pdfs[i], self._rho_cg[i], self._temp_cg[i]


# ===================================================================== #
#  CLI Arguments                                                        #
# ===================================================================== #
def parse_args():
    parser = argparse.ArgumentParser(
        description="Snapshot-Split Random Crop Training Pipeline for Subgrid CGM PDF Model"
    )
    # Data & Resolution
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.environ.get("SUBGRID_DATA_PATH", "/path/to/simulation/bin"),
        help="Path to simulation binary files",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default=os.environ.get("SUBGRID_CACHE_PATH", "/path/to/cache"),
        help="Path to preprocessed cache directory",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=os.environ.get("PDF_CNN_RESOLUTION", "2048,1024"),
        help="Full fine grid resolution (e.g. '2048,1024')",
    )
    parser.add_argument(
        "--crop_h",
        type=int,
        default=None,
        help="Crop height from fine grid (e.g. 1024 -> 16 coarse cells with ds=64)",
    )
    parser.add_argument(
        "--crop_w",
        type=int,
        default=None,
        help="Crop width from fine grid (e.g. 512 -> 8 coarse cells with ds=64)",
    )
    parser.add_argument(
        "--crop_h_cg",
        type=int,
        default=16,
        help="Crop height in coarse cells (default 16)",
    )
    parser.add_argument(
        "--crop_w_cg",
        type=int,
        default=8,
        help="Crop width in coarse cells (default 8)",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=int(os.environ.get("PDF_CNN_DOWNSAMPLE", "64")),
        help="Coarse-graining factor (default 64 -> 32x16 coarse grid for 2048x1024)",
    )
    # Split fractions & crop counts
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.60,
        help="Fraction of snapshots for training",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.20,
        help="Fraction of snapshots for validation",
    )
    parser.add_argument(
        "--n_crops_train",
        type=int,
        default=8,
        help="Number of random crops per snapshot for training",
    )
    parser.add_argument(
        "--n_crops_val",
        type=int,
        default=4,
        help="Number of random crops per snapshot for validation",
    )
    parser.add_argument(
        "--n_crops_test",
        type=int,
        default=4,
        help="Number of random crops per snapshot for test",
    )
    parser.add_argument(
        "--ema_alpha",
        type=float,
        default=0.9,
        help="EMA smoothing factor for normalization statistics",
    )
    # Hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=HYPERPARAMS.get("num_epochs", 1000),
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=HYPERPARAMS.get("batch_size", 64),
        help="Mini-batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=HYPERPARAMS.get("learning_rate", 1e-3),
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=HYPERPARAMS.get("weight_decay", 1e-5),
        help="Weight decay",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=HYPERPARAMS.get("seed", 42),
        help="Random seed for snapshot split",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override ('cuda', 'mps', 'cpu')",
    )
    # Loss Weights
    parser.add_argument(
        "--alpha_active_kl",
        type=float,
        default=float(os.environ.get("PDF_CNN_ALPHA_ACTIVE_KL", "10.0")),
        help="Active zone KL loss weight",
    )
    parser.add_argument(
        "--alpha_inactive_kl",
        type=float,
        default=float(os.environ.get("PDF_CNN_ALPHA_INACTIVE_KL", "10.0")),
        help="Inactive zone KL loss weight",
    )
    parser.add_argument(
        "--alpha_gate",
        type=float,
        default=HYPERPARAMS.get("alpha_gate", 50.0),
        help="Gate loss weight",
    )
    parser.add_argument(
        "--alpha_mean_temp",
        type=float,
        default=HYPERPARAMS.get("alpha_mean_temp", 10.0),
        help="Mean temperature loss weight",
    )
    parser.add_argument(
        "--alpha_emiss",
        type=float,
        default=float(os.environ.get("PDF_CNN_ALPHA_EMISS", "10.0")),
        help="Emissivity loss weight",
    )
    parser.add_argument(
        "--alpha_leak",
        type=float,
        default=float(os.environ.get("PDF_CNN_ALPHA_LEAK", "10.0")),
        help="Active window mass leakage loss weight",
    )
    # Directories
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default=os.environ.get(
            "MODEL_SAVES_DIR",
            str(PROJECT_ROOT / "outputs" / "model_saves" / "pdf_model_saves"),
        ),
        help="Directory to save model weights and stats",
    )
    parser.add_argument(
        "--loss_plot_dir",
        type=str,
        default=os.environ.get(
            "LOSS_PLOTS_DIR",
            str(PROJECT_ROOT / "outputs" / "loss_plots" / "pdf_loss_plots"),
        ),
        help="Directory to save loss plots",
    )
    return parser.parse_args()


# ===================================================================== #
#  Main Training Execution                                              #
# ===================================================================== #
def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Directories
    model_save_dir = Path(args.model_save_dir)
    loss_plot_dir = Path(args.loss_plot_dir)
    model_save_dir.mkdir(parents=True, exist_ok=True)
    loss_plot_dir.mkdir(parents=True, exist_ok=True)

    # Resolution setup
    res = _parse_resolution(args.resolution, (2048, 1024))
    downsample = args.downsample
    out_channels = HYPERPARAMS.get("out_channels", 40)
    if args.crop_h is not None:
        crop_h_cg = args.crop_h // downsample
    else:
        crop_h_cg = args.crop_h_cg

    if args.crop_w is not None:
        crop_w_cg = args.crop_w // downsample
    else:
        crop_w_cg = args.crop_w_cg

    print(
        f"Fine Grid Resolution: {res}, Downsample: {downsample}, "
        f"Coarse Full Grid: ({res[0]//downsample}, {res[1]//downsample}), Crop: ({crop_h_cg}, {crop_w_cg})"
    )

    # Load / create compact coarse dataset
    cg_inputs, cg_pdfs = load_or_create_coarse_data(
        data_path=args.data_path,
        cache_path=args.cache_path,
        resolution=res,
        downsample=downsample,
        out_channels=out_channels,
    )

    N_snaps = cg_inputs.shape[0]
    print(f"Dataset ready: {N_snaps} coarse snapshots available.")

    # Split snapshots
    train_idx, val_idx, test_idx = split_snapshots(
        N_snaps=N_snaps,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    print(
        f"Snapshots — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}"
    )

    # Initialize Datasets
    train_dataset = SnapshotCropDataset(
        cg_inputs=cg_inputs,
        cg_pdfs=cg_pdfs,
        snap_indices=train_idx,
        crop_h_cg=crop_h_cg,
        crop_w_cg=crop_w_cg,
        n_crops_per_snap=args.n_crops_train,
        ema_alpha=args.ema_alpha,
    )

    val_dataset = SnapshotCropDataset(
        cg_inputs=cg_inputs,
        cg_pdfs=cg_pdfs,
        snap_indices=val_idx,
        crop_h_cg=crop_h_cg,
        crop_w_cg=crop_w_cg,
        n_crops_per_snap=args.n_crops_val,
        ema_alpha=args.ema_alpha,
    )

    test_dataset = SnapshotCropDataset(
        cg_inputs=cg_inputs,
        cg_pdfs=cg_pdfs,
        snap_indices=test_idx,
        crop_h_cg=crop_h_cg,
        crop_w_cg=crop_w_cg,
        n_crops_per_snap=args.n_crops_test,
        ema_alpha=args.ema_alpha,
    )

    # Initialize Model
    cnn_model = ConvNN(
        in_channels=HYPERPARAMS.get("in_channels", 5),
        layer_size1=HYPERPARAMS.get("layer_size1", 32),
        layer_size2=HYPERPARAMS.get("layer_size2", 64),
        layer_size3=HYPERPARAMS.get("layer_size3", 128),
        layer_size4=HYPERPARAMS.get("layer_size4", 256),
        out_channels=out_channels,
        kernel_size=HYPERPARAMS.get("kernel_size", 9),
    ).to(device)

    # Performance optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            cnn_model = torch.compile(cnn_model, mode="reduce-overhead")
            print("Compiled cnn_model with torch.compile(mode='reduce-overhead')")
        except Exception as e:
            print(f"torch.compile skipped / failed: {e}")

    # Initialize Criterion
    criterion = GatedPDFLoss(
        alpha_gate=args.alpha_gate,
        alpha_mean_temp=args.alpha_mean_temp,
        alpha_emiss=args.alpha_emiss,
        alpha_leak=args.alpha_leak,
        alpha_active_kl=args.alpha_active_kl,
        alpha_inactive_kl=args.alpha_inactive_kl,
    )

    optimizer = torch.optim.AdamW(
        cnn_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    num_epochs = args.epochs
    batch_size = args.batch_size
    steps_per_epoch = max(1, len(train_dataset) // batch_size)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        pct_start=0.2,
    )

    epochs_array: list[int] = []
    train_loss_arr: list[float] = []
    val_loss_arr: list[float] = []

    grad_clip_max_norm = HYPERPARAMS.get("grad_clip_max_norm", 1.0)

    print(
        f"Starting training for {num_epochs} epochs | Batch size: {batch_size} | Device: {device}"
    )

    # Training Loop
    for epoch in tqdm(range(num_epochs), desc="Training"):
        # ── 1. Fresh crop pools ───────────────────────────────────────
        train_dataset.resample()
        val_dataset.resample()

        # Val uses train's current-epoch stats so the loss is comparable
        val_dataset.set_norm_stats(train_dataset.input_mean, train_dataset.input_std)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
        )

        # ── 2. Train ──────────────────────────────────────────────────
        cnn_model.train()
        for inputs, labels, rho, temp in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            rho = rho.to(device)
            temp = temp.to(device)

            # Augmentation: flip & negate velocity components
            if torch.rand(1).item() > 0.5:
                inputs = torch.flip(inputs, [3])
                labels = torch.flip(labels, [3])
                rho = torch.flip(rho, [3])
                temp = torch.flip(temp, [3])
                inputs = inputs.clone()
                inputs[:, 2] = -inputs[:, 2]  # negate ux

            if torch.rand(1).item() > 0.5:
                inputs = torch.flip(inputs, [2])
                labels = torch.flip(labels, [2])
                rho = torch.flip(rho, [2])
                temp = torch.flip(temp, [2])
                inputs = inputs.clone()
                inputs[:, 3] = -inputs[:, 3]  # negate uy

            logits, gate = cnn_model(inputs)
            loss = criterion(logits, gate, labels, rho, temp)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cnn_model.parameters(), grad_clip_max_norm)
            optimizer.step()
            scheduler.step()

        # ── 3. Validate ───────────────────────────────────────────────
        cnn_model.eval()
        with torch.no_grad():
            train_loss = sum(
                criterion(
                    *(cnn_model(x.to(device))[:2]),
                    y.to(device),
                    r.to(device),
                    t.to(device),
                ).item()
                for x, y, r, t in train_loader
            ) / max(1, len(train_loader))

            val_loss = sum(
                criterion(
                    *(cnn_model(x.to(device))[:2]),
                    y.to(device),
                    r.to(device),
                    t.to(device),
                ).item()
                for x, y, r, t in val_loader
            ) / max(1, len(val_loader))

        epochs_array.append(epoch + 1)
        train_loss_arr.append(train_loss)
        val_loss_arr.append(val_loss)

        if (epoch + 1) % 50 == 0 or epoch == num_epochs - 1:
            tqdm.write(
                f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

        # ── 4. Early stopping (moving-average logic) ───────────────────
        if len(val_loss_arr) >= 200:
            ma = np.convolve(val_loss_arr, np.ones(200) / 200, mode="valid")
            if len(ma) > 1 and ma[-1] > np.min(ma[:-1]) and epoch >= 499:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # ── 5. Save EMA normalisation stats for inference ─────────────────
    raw_model = getattr(cnn_model, "_orig_mod", cnn_model)
    ema_mean_np = train_dataset.ema_mean.cpu().numpy()
    ema_std_np = train_dataset.ema_std.cpu().numpy()
    state_dict = raw_model.state_dict()

    target_configs = [
        (res, downsample),
        ((512, 256), 32),
        ((1024, 512), 64),
    ]

    for (r_tuple, ds_val) in target_configs:
        mean_path = model_save_dir / f"cnn_{r_tuple}_{ds_val}_input_mean.npy"
        std_path = model_save_dir / f"cnn_{r_tuple}_{ds_val}_input_std.npy"
        model_path = model_save_dir / f"cnn_{r_tuple}_{ds_val}.pth"

        np.save(mean_path, ema_mean_np)
        np.save(std_path, ema_std_np)
        torch.save(state_dict, model_path)
        print(f"Saved weights & stats -> cnn_{r_tuple}_{ds_val}")

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_array, train_loss_arr, label="Train Loss")
    plt.plot(epochs_array, val_loss_arr, label="Validation Loss")
    plt.axhline(train_loss_arr[-1], linestyle="--", alpha=0.7)
    plt.axhline(val_loss_arr[-1], linestyle="--", alpha=0.7)
    plt.xlabel("Epochs")
    plt.ylabel("PDF Loss")
    plt.title("Snapshot-Split Random Crop Training Loss")
    plt.legend()
    plt.tight_layout()

    loss_plot_file = loss_plot_dir / f"cnn_{res}_{downsample}_loss.jpg"
    plt.savefig(loss_plot_file, dpi=300)
    plt.close()
    print(f"Saved loss plot to {loss_plot_file}")

    # Step 5: Test Evaluation
    print("\nEvaluating on Test Set with EMA stats...")
    test_dataset.resample()
    test_dataset.set_norm_stats(train_dataset.ema_mean, train_dataset.ema_std)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    cnn_model.eval()
    with torch.no_grad():
        test_loss = sum(
            criterion(
                *(cnn_model(x.to(device))[:2]),
                y.to(device),
                r.to(device),
                t.to(device),
            ).item()
            for x, y, r, t in test_loader
        ) / max(1, len(test_loader))

    print(f"Test Loss: {test_loss:.6f}")


if __name__ == "__main__":
    main()
