#!/usr/bin/env python3
"""
optimize_subgrid.py — Closed-loop Optuna Bayesian Optimization for SubgridCGM

Optimizes PDF CNN loss hyperparameters (alpha_active_wasserstein, alpha_inactive_wasserstein,
alpha_emiss, alpha_gate, alpha_mean_temp, alpha_leak, etc.) using the random snapshot crop
training methods and data from random_subsample_pipeline.sh (random_snapshot_training.py).

Evaluates downstream Athena-K subgrid simulation restarts (5 -> 10 Myr) by scoring:
1. Integrated emissivity (area under the profile)
2. Emissivity profile width (spatial standard deviation along y)
3. Top boundary mass flux
against dynamically calculated high-resolution reference benchmarks.
"""

import argparse
import os
import shutil
import subprocess
import sys
import numpy as np
import optuna

# ---------------------------------------------------------------------------
# Project paths & imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "conv_nn")

for p in (PROJECT_ROOT, DATA_DIR, MODELS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_preprocess import simulation_data
from pdf_cnn import (
    compute_cooling_rate,
    lambda_cool,
    out_channels,
    snapshot_pred_16x8,
)


def coarse_grain_array(arr: np.ndarray, ds: int = 32) -> np.ndarray:
    """Coarse grains a 2D or 3D array by factor ds (matches mock_sg.py)."""
    if arr.ndim == 2:
        return arr.reshape(arr.shape[0] // ds, ds, arr.shape[1] // ds, ds).mean(axis=(1, 3))
    elif arr.ndim == 3:
        return arr.reshape(arr.shape[0], arr.shape[1] // ds, ds, arr.shape[2] // ds, ds).mean(axis=(2, 4))
    else:
        raise ValueError(f"Unsupported array dimension: {arr.ndim}")


def compute_emissivity_stats(emis_profile: np.ndarray, y_coords: np.ndarray) -> tuple[float, float]:
    """
    Computes the integrated emissivity (area under profile) and spatial width (std-dev).

    Parameters:
    -----------
    emis_profile : np.ndarray
        1D emissivity profile along y (averaged over x and time).
    y_coords : np.ndarray
        1D coordinate array corresponding to the profile cells along y.

    Returns:
    --------
    int_emiss : float
        Total integrated emissivity: integral(emis(y) dy).
    emiss_width : float
        Spatial standard deviation (width) of the emissivity distribution:
        sigma_y = sqrt( integral( (y - mu_y)^2 * emis(y) dy ) / int_emiss ).
    """
    int_emiss = float(np.trapezoid(emis_profile, y_coords))
    if int_emiss > 1e-30:
        y_mean = float(np.trapezoid(y_coords * emis_profile, y_coords) / int_emiss)
        y_var = float(np.trapezoid((y_coords - y_mean) ** 2 * emis_profile, y_coords) / int_emiss)
        emiss_width = float(np.sqrt(max(y_var, 0.0)))
    else:
        emiss_width = 0.0
    return int_emiss, emiss_width


def parse_res_tuple(res_str: str) -> tuple[int, int]:
    """Parses '2048,1024' or '(2048, 1024)' into (2048, 1024)."""
    cleaned = res_str.replace("(", "").replace(")", "").strip()
    parts = cleaned.split(",")
    return (int(parts[0].strip()), int(parts[1].strip()))


def resolve_hr_cache_folder(
    custom_folder: str = None,
    hr_eval_output: str = None,
    hr_res: tuple[int, int] = (512, 256),
    hr_downsample: int = 32,
) -> str:
    """Resolves HR cache folder path matching mock_sg.py and pdf_cnn.py."""
    if custom_folder and os.path.exists(custom_folder):
        return custom_folder

    potential_bases = []
    if hr_eval_output:
        potential_bases.append(os.path.join(hr_eval_output, "cache"))
        potential_bases.append(hr_eval_output)
    if os.environ.get("SUBGRID_CACHE_PATH"):
        potential_bases.append(os.environ["SUBGRID_CACHE_PATH"])
    if os.environ.get("HR_SIM_OUTPUT"):
        potential_bases.append(os.path.join(os.environ["HR_SIM_OUTPUT"], "cache"))
    # Fallback to standard hr_build_512 cache
    potential_bases.append(os.path.join(PROJECT_ROOT, "simulation_outputs", "hr_build_512", "cache"))

    for base in potential_bases:
        if not os.path.exists(base):
            continue

        # 1. Format: sc(512, 256)_32
        cand1 = os.path.join(base, f"sc{hr_res}_{hr_downsample}")
        if os.path.exists(cand1) and os.path.isfile(os.path.join(cand1, "rho.npy")):
            return cand1

        # 2. Format: sc512,256_32
        cand2 = os.path.join(base, f"sc{hr_res[0]},{hr_res[1]}_{hr_downsample}")
        if os.path.exists(cand2) and os.path.isfile(os.path.join(cand2, "rho.npy")):
            return cand2

        # 3. Direct files in base
        if os.path.isfile(os.path.join(base, "rho.npy")):
            return base

        # 4. Any sc* subfolder with rho.npy
        for entry in sorted(os.listdir(base)):
            sub = os.path.join(base, entry)
            if os.path.isdir(sub) and entry.startswith("sc") and os.path.isfile(os.path.join(sub, "rho.npy")):
                return sub

    return os.path.join(PROJECT_ROOT, "simulation_outputs", "hr_build_512", "cache", f"sc{hr_res}_{hr_downsample}")



def compute_hr_references(
    hr_cache_folder: str = None,
    hr_eval_output: str = None,
    hr_res: tuple[int, int] = (512, 256),
    hr_downsample: int = 32,
    n_frames: int = 500,
) -> tuple[float, float, float]:
    """
    Computes high-resolution reference observables:
    1. Integrated emissivity (area under profile)
    2. Emissivity profile width / std-dev
    3. Mass flux at top boundary
    directly from cached HR simulation snapshots.
    """
    actual_cache_folder = resolve_hr_cache_folder(
        custom_folder=hr_cache_folder,
        hr_eval_output=hr_eval_output,
        hr_res=hr_res,
        hr_downsample=hr_downsample,
    )

    print("=" * 80)
    print("Computing HR reference values dynamically from cached HR simulation data...")
    print(f"  Cache Folder : {actual_cache_folder}")
    print(f"  Resolution   : {hr_res}")
    print(f"  Downsample   : {hr_downsample}")
    print("=" * 80)

    if not os.path.exists(actual_cache_folder):
        raise FileNotFoundError(f"HR cache folder not found at: {actual_cache_folder}")

    # Memory-map the large HR arrays from cache
    hr_rho = np.load(os.path.join(actual_cache_folder, "rho.npy"), mmap_mode="r")[-n_frames:]
    hr_temp = np.load(os.path.join(actual_cache_folder, "temp.npy"), mmap_mode="r")[-n_frames:]
    hr_uy = np.load(os.path.join(actual_cache_folder, "uy.npy"), mmap_mode="r")[-n_frames:]

    rho0 = 1e-3
    du = 31.0918

    # 1 & 2. HR Emissivity Profile: Integrated Emissivity & Width (Std-Dev)
    emis_hr = hr_rho**2 * lambda_cool(hr_temp, mask=True)
    emis_cg_hr = coarse_grain_array(emis_hr, hr_downsample)
    emis_cg_hr_xavg = np.mean(emis_cg_hr, axis=2)
    emis_cg_hr_mean = np.mean(emis_cg_hr_xavg, axis=0)

    y_cg_hr = np.linspace(-10.0, 10.0, emis_cg_hr.shape[1])
    int_hr_emiss, width_hr_emiss = compute_emissivity_stats(emis_cg_hr_mean, y_cg_hr)

    # 3. HR Mass Flux at top boundary
    mass_flux_hr = float(np.mean((hr_rho * hr_uy)[:, -1, :]) / (rho0 * du))

    print(f"Dynamically Calculated HR Ground Truth:")
    print(f"  HR Integrated Emissivity : {int_hr_emiss:.6e}")
    print(f"  HR Emissivity Width (std): {width_hr_emiss:.6e}")
    print(f"  HR Mass Flux             : {mass_flux_hr:.6e}")
    print("=" * 80)

    return int_hr_emiss, width_hr_emiss, mass_flux_hr


class DummyTrial:
    """Mock trial for evaluating a fixed set of hyperparameters without Optuna."""

    def __init__(self, params: dict, number: int = 0):
        self._trial = optuna.trial.FixedTrial(params)
        self.number = number

    def suggest_float(self, *args, **kwargs):
        return self._trial.suggest_float(*args, **kwargs)

    def suggest_int(self, *args, **kwargs):
        return self._trial.suggest_int(*args, **kwargs)

    def suggest_categorical(self, *args, **kwargs):
        return self._trial.suggest_categorical(*args, **kwargs)


def create_checkpoint_aliases(model_dir: str, res: tuple[int, int], downsample: int) -> None:
    """Ensures model checkpoints and normalization statistics are aliased for downstream simulation/eval."""
    train_model = os.path.join(model_dir, f"cnn_{res}_{downsample}.pth")
    train_mean = os.path.join(model_dir, f"cnn_{res}_{downsample}_input_mean.npy")
    train_std = os.path.join(model_dir, f"cnn_{res}_{downsample}_input_std.npy")

    # If canonical name missing, find first available non-gate file
    if not os.path.isfile(train_model):
        for f in sorted(os.listdir(model_dir)):
            if f.startswith("cnn_") and f.endswith(".pth") and "gate" not in f:
                train_model = os.path.join(model_dir, f)
                break
        for f in sorted(os.listdir(model_dir)):
            if f.startswith("cnn_") and f.endswith("_input_mean.npy") and "gate" not in f:
                train_mean = os.path.join(model_dir, f)
                break
        for f in sorted(os.listdir(model_dir)):
            if f.startswith("cnn_") and f.endswith("_input_std.npy") and "gate" not in f:
                train_std = os.path.join(model_dir, f)
                break

    if os.path.isfile(train_model) and os.path.isfile(train_mean) and os.path.isfile(train_std):
        target_configs = [
            ((512, 256), 32),
            ((1024, 512), 64),
            ((2048, 1024), 64),
        ]
        for (r_tuple, ds_val) in target_configs:
            dst_m = os.path.join(model_dir, f"cnn_{r_tuple}_{ds_val}.pth")
            dst_mean = os.path.join(model_dir, f"cnn_{r_tuple}_{ds_val}_input_mean.npy")
            dst_std = os.path.join(model_dir, f"cnn_{r_tuple}_{ds_val}_input_std.npy")
            if not os.path.exists(dst_m):
                shutil.copyfile(train_model, dst_m)
            if not os.path.exists(dst_mean):
                shutil.copyfile(train_mean, dst_mean)
            if not os.path.exists(dst_std):
                shutil.copyfile(train_std, dst_std)


def train_cnn_trial(trial, model_dir: str, epochs: int, python_exe: str, config: dict) -> dict:
    """
    Suggests hyperparameters and trains PDF CNN using random snapshot crops
    matching random_subsample_pipeline.sh (random_snapshot_training.py).
    """
    print("=" * 80)
    print(f"Trial {trial.number}: Proposing hyperparameters and training CNN with random crop pipeline...")
    print("=" * 80)

    # Optuna loss hyperparameter search space
    alpha_active_wass = trial.suggest_float("alpha_active_wasserstein", 0.1, 100.0, log=True)
    alpha_inact_wass = trial.suggest_float("alpha_inactive_wasserstein", 0.1, 100.0, log=True)
    alpha_emiss = trial.suggest_float("alpha_emiss", 0.1, 500.0, log=True)
    alpha_mean_temp = trial.suggest_float("alpha_mean_temp", 0.1, 50.0, log=True)
    alpha_leak = trial.suggest_float("alpha_leak", 0.1, 50.0, log=True)
    alpha_gate = trial.suggest_float("alpha_gate", 0.0, 10.0)

    print(f"Trial {trial.number} Loss Weights:")
    print(f"  alpha_active_wasserstein   : {alpha_active_wass:.4f}")
    print(f"  alpha_inactive_wasserstein : {alpha_inact_wass:.4f}")
    print(f"  alpha_emiss                : {alpha_emiss:.4f}")
    print(f"  alpha_mean_temp            : {alpha_mean_temp:.4f}")
    print(f"  alpha_leak                 : {alpha_leak:.4f}")
    print(f"  alpha_gate                 : {alpha_gate:.4f}")
    print(f"  epochs                     : {epochs}")

    os.makedirs(model_dir, exist_ok=True)
    loss_plot_dir = os.path.join(model_dir, "loss_plots")
    os.makedirs(loss_plot_dir, exist_ok=True)

    fine_res = config["train_resolution"]
    downsample = config["train_downsample"]
    crop_h_cg = config["crop_h_cg"]
    crop_w_cg = config["crop_w_cg"]
    crop_h = crop_h_cg * downsample
    crop_w = crop_w_cg * downsample

    cmd = [
        python_exe,
        os.path.join(PROJECT_ROOT, "random_snapshot_training.py"),
        "--data_path", config["train_data_path"],
        "--cache_path", config["train_cache_path"],
        "--resolution", f"{fine_res[0]},{fine_res[1]}",
        "--downsample", str(downsample),
        "--crop_h", str(crop_h),
        "--crop_w", str(crop_w),
        "--crop_h_cg", str(crop_h_cg),
        "--crop_w_cg", str(crop_w_cg),
        "--train_frac", str(config["train_frac"]),
        "--val_frac", str(config["val_frac"]),
        "--n_crops_train", str(config["n_crops_train"]),
        "--n_crops_val", str(config["n_crops_val"]),
        "--n_crops_test", str(config["n_crops_test"]),
        "--ema_alpha", str(config["ema_alpha"]),
        "--epochs", str(epochs),
        "--batch_size", str(config["batch_size"]),
        "--learning_rate", str(config["learning_rate"]),
        "--weight_decay", str(config["weight_decay"]),
        "--seed", str(config["seed"]),
        "--alpha_active_wasserstein", str(alpha_active_wass),
        "--alpha_inactive_wasserstein", str(alpha_inact_wass),
        "--alpha_gate", str(alpha_gate),
        "--alpha_mean_temp", str(alpha_mean_temp),
        "--alpha_emiss", str(alpha_emiss),
        "--alpha_leak", str(alpha_leak),
        "--gate_epochs", str(config["gate_epochs"]),
        "--gate_learning_rate", str(config["gate_lr"]),
        "--model_save_dir", model_dir,
        "--loss_plot_dir", loss_plot_dir,
    ]

    res = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if res.returncode != 0:
        print(f"Trial {trial.number}: Training failed with exit code {res.returncode}:")
        print(res.stderr[-1000:] if len(res.stderr) > 1000 else res.stderr)
        raise RuntimeError(f"CNN training failed in trial {trial.number}")

    create_checkpoint_aliases(model_dir, fine_res, downsample)
    print(f"Trial {trial.number}: Training completed successfully.")

    return {
        "alpha_active_wasserstein": alpha_active_wass,
        "alpha_inactive_wasserstein": alpha_inact_wass,
        "alpha_emiss": alpha_emiss,
        "alpha_gate": alpha_gate,
        "alpha_mean_temp": alpha_mean_temp,
        "alpha_leak": alpha_leak,
    }


def run_athena_trial(
    trial,
    output_dir: str,
    model_dir: str,
    athena_dir: str,
    athinput_path: str,
    restart_path: str,
    config: dict,
) -> None:
    """Runs Athena-K subgrid model simulation restarting from checkpoint."""
    print("=" * 80)
    print(f"Trial {trial.number}: Running Athena-K subgrid simulation...")
    print(f"  Output Dir : {output_dir}")
    print(f"  Athinput   : {athinput_path}")
    print(f"  Restart    : {restart_path}")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    athena_bin = os.path.join(athena_dir, "athena")
    if not os.path.isfile(athena_bin):
        raise FileNotFoundError(f"Athena binary not found at {athena_bin}")

    fine_res = config["train_resolution"]
    downsample = config["train_downsample"]
    crop_h_cg = config["crop_h_cg"]
    crop_w_cg = config["crop_w_cg"]

    env = os.environ.copy()
    env["MODEL_SAVES_DIR"] = model_dir
    env["PDF_CNN_RESOLUTION"] = f"{fine_res[0]},{fine_res[1]}"
    env["PDF_CNN_DOWNSAMPLE"] = str(downsample)
    env["TILE_ROWS"] = str(crop_h_cg)
    env["TILE_COLS"] = str(crop_w_cg)
    env["CROP_H_CG"] = str(crop_h_cg)
    env["CROP_W_CG"] = str(crop_w_cg)

    # Locate virtualenv site-packages
    venv_site = os.path.join(PROJECT_ROOT, "venv", "lib", "python3.10", "site-packages")
    if not os.path.exists(venv_site):
        venv_lib = os.path.join(PROJECT_ROOT, "venv", "lib")
        if os.path.exists(venv_lib):
            for entry in os.listdir(venv_lib):
                sp = os.path.join(venv_lib, entry, "site-packages")
                if os.path.exists(sp):
                    venv_site = sp
                    break

    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{athena_dir}:{venv_site}:{PROJECT_ROOT}:{DATA_DIR}:{MODELS_DIR}:{current_pythonpath}"

    cmd = [
        athena_bin,
        "-i", athinput_path,
        "-d", output_dir,
        "-r", restart_path,
    ]

    result = subprocess.run(
        cmd,
        cwd=athena_dir,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print(f"Trial {trial.number}: Athena-K exited with code {result.returncode}")
        print(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
        raise RuntimeError(f"Athena-K failed during trial {trial.number}")

    bin_dir = os.path.join(output_dir, "bin")
    if not os.path.exists(bin_dir) or len(os.listdir(bin_dir)) == 0:
        raise RuntimeError(f"Athena-K failed to produce snapshot outputs in {bin_dir}")

    print(f"Trial {trial.number}: Athena-K simulation finished.")


def compare_simulation(
    trial,
    output_dir: str,
    hr_emissivity: float,
    hr_emiss_width: float,
    hr_mass_flux: float,
    model_dir: str = None,
    config: dict = None,
    w_emiss: float = 1.0,
    w_width: float = 1.0,
    w_mass: float = 1.0,
) -> float:
    """
    Computes physical observables:
    1. Integrated emissivity (area under profile)
    2. Emissivity profile width / std-dev
    3. Top boundary mass flux
    and scores relative errors against HR ground truth targets.
    """
    print("=" * 80)
    print(f"Trial {trial.number}: Evaluating physical observables against HR reference...")
    print("=" * 80)

    rho0 = 1e-3
    du = 31.0918
    unit_fix = 1.975e27

    sim_data = simulation_data()
    crop_h_cg = config.get("crop_h_cg", 16) if config else 16
    crop_w_cg = config.get("crop_w_cg", 8) if config else 8
    sim_data.resolution = (crop_h_cg, crop_w_cg)
    bin_path = os.path.join(output_dir, "bin")

    # Load restart simulation outputs (t = 5 -> 10 Myr)
    sim_data.input_data(bin_path, start=500)

    rho = sim_data.rho
    temp = sim_data.temp
    ux = sim_data.ux
    uy = sim_data.uy
    ps = sim_data.ps

    if rho is None or len(rho) == 0:
        raise RuntimeError(f"No valid simulation snapshots found in {bin_path}")

    fine_res = config["train_resolution"] if config else (2048, 1024)
    cnn_ds = config["train_downsample"] if config else 64
    T_edges = np.logspace(3.0, 7.0, out_channels + 1)
    T_centers = np.sqrt(T_edges[:-1] * T_edges[1:])

    emis_arr = np.zeros_like(rho)
    for t in range(rho.shape[0]):
        pdf_t = snapshot_pred_16x8(
            rho[t], temp[t], ux[t], uy[t], ps[t],
            fine_resolution=fine_res, downsample=cnn_ds,
            model_save_dir=model_dir,
        )
        cool_code = compute_cooling_rate(
            pdf_t, T_centers, is_pdf=True, rho_cg=rho[t]
        )
        emis_arr[t] = cool_code / unit_fix

    # 1 & 2. Subgrid Emissivity Profile: Integrated Emissivity & Width (Std-Dev)
    emis_sg_xavg = np.mean(emis_arr, axis=2)
    emis_sg_mean = np.mean(emis_sg_xavg, axis=0)

    y_sg = np.linspace(-10.0, 10.0, rho.shape[1])
    int_emiss, emiss_width = compute_emissivity_stats(emis_sg_mean, y_sg)

    # 3. Top boundary mass flux
    mass_flux = float(np.mean((rho * uy)[:, -1, :]) / (rho0 * du))

    # Compute relative errors
    emiss_err = abs(int_emiss - hr_emissivity) / (abs(hr_emissivity) + 1e-30)
    width_err = abs(emiss_width - hr_emiss_width) / (abs(hr_emiss_width) + 1e-30)
    mass_err = abs(mass_flux - hr_mass_flux) / (abs(hr_mass_flux) + 1e-30)

    score = w_emiss * emiss_err + w_width * width_err + w_mass * mass_err

    print(f"Trial {trial.number} Evaluation Metrics:")
    print(f"  Integrated Emissivity (Area) : {int_emiss:.6e} (Target: {hr_emissivity:.6e}, Rel Err: {emiss_err:.2%})")
    print(f"  Emissivity Profile Width(std): {emiss_width:.6e} (Target: {hr_emiss_width:.6e}, Rel Err: {width_err:.2%})")
    print(f"  Mass Flux                    : {mass_flux:.6e} (Target: {hr_mass_flux:.6e}, Rel Err: {mass_err:.2%})")
    print(f"  Weighted Objective Score     : {score:.6f} (Weights: emiss={w_emiss}, width={w_width}, mass={w_mass})")
    print("=" * 80)

    return float(score)


def main():
    parser = argparse.ArgumentParser(description="SubgridCGM Optuna Hyperparameter Optimization")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--epochs_per_trial", type=int, default=500, help="Training epochs per trial")
    parser.add_argument("--study_name", type=str, default="cnn_hyperparams", help="Optuna study name")
    parser.add_argument("--storage", type=str, default="sqlite:///cnn_hyperparams.db", help="Optuna SQLite storage URL")
    parser.add_argument("--athinput", type=str, required=True, help="Path to sg_sim.athinput")
    parser.add_argument("--restart_file", type=str, required=True, help="Path to 5 Myr restart file KH.00005.rst")
    parser.add_argument("--athena_dir", type=str, default=os.path.join(PROJECT_ROOT, "builds", "subgrid_model", "src"), help="Athena subgrid build src dir")
    parser.add_argument("--output_best_dir", type=str, default=os.path.join(PROJECT_ROOT, "outputs", "best_optuna_model"), help="Directory to save the best model")

    # Training Data & Crop Configuration (matches random_subsample_pipeline.sh)
    parser.add_argument(
        "--train_data_path",
        type=str,
        default=os.environ.get(
            "HR_TRAIN_BIN_DIR",
            os.path.join(PROJECT_ROOT, "simulation_outputs", "hr_gpu_sweep_1024x2048_2xlength", "vshear_31_coldfrac_0.67", "bin"),
        ),
        help="Path to fine-grid simulation binary training data",
    )
    parser.add_argument(
        "--train_cache_path",
        type=str,
        default=os.environ.get(
            "HR_TRAIN_CACHE_DIR",
            os.path.join(PROJECT_ROOT, "simulation_outputs", "hr_gpu_sweep_1024x2048_2xlength", "vshear_31_coldfrac_0.67", "cache"),
        ),
        help="Path to precomputed coarse-grain cache directory",
    )
    parser.add_argument("--resolution", type=str, default=os.environ.get("PDF_CNN_RESOLUTION", "2048,1024"), help="Full fine-grid training resolution (H,W)")
    parser.add_argument("--downsample", type=int, default=int(os.environ.get("PDF_CNN_DOWNSAMPLE", "64")), help="Downsampling factor for coarse graining")
    parser.add_argument("--crop_h_cg", type=int, default=int(os.environ.get("CROP_H_CG", "16")), help="Coarse crop height")
    parser.add_argument("--crop_w_cg", type=int, default=int(os.environ.get("CROP_W_CG", "8")), help="Coarse crop width")
    parser.add_argument("--train_frac", type=float, default=0.60, help="Train snapshot fraction")
    parser.add_argument("--val_frac", type=float, default=0.20, help="Val snapshot fraction")
    parser.add_argument("--n_crops_train", type=int, default=8, help="Random crops per snapshot for training")
    parser.add_argument("--n_crops_val", type=int, default=4, help="Random crops per snapshot for validation")
    parser.add_argument("--n_crops_test", type=int, default=4, help="Random crops per snapshot for testing")
    parser.add_argument("--ema_alpha", type=float, default=0.9, help="EMA alpha for normalisation stats")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Training learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Optimizer weight decay")
    parser.add_argument("--gate_epochs", type=int, default=200, help="Gate branch pretraining epochs")
    parser.add_argument("--gate_lr", type=float, default=1e-3, help="Gate pretraining learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Evaluation Reference Data
    parser.add_argument(
        "--hr_eval_output",
        type=str,
        default=os.environ.get("HR_EVAL_OUTPUT", os.path.join(PROJECT_ROOT, "simulation_outputs", "hr_build_512")),
        help="Path to evaluation HR simulation output directory",
    )
    parser.add_argument("--hr_cache_folder", type=str, default=None, help="Explicit path to cached HR simulation dataset")
    parser.add_argument("--hr_eval_resolution", type=str, default=os.environ.get("HR_EVAL_RESOLUTION", "512,256"), help="HR benchmark resolution (H,W)")
    parser.add_argument("--hr_eval_downsample", type=int, default=int(os.environ.get("HR_EVAL_DOWNSAMPLE", "32")), help="Downsample factor for HR reference coarse-graining")

    # Target Optimization Weights
    parser.add_argument("--w_emiss", type=float, default=1.0, help="Weight for integrated emissivity error")
    parser.add_argument("--w_width", type=float, default=1.0, help="Weight for emissivity profile width error")
    parser.add_argument("--w_mass", type=float, default=1.0, help="Weight for mass flux error")

    parser.add_argument("--keep_trials", action="store_true", help="Keep all trial simulation outputs")
    parser.add_argument("--single_trial", action="store_true", help="Run a single trial with default parameters")
    args = parser.parse_args()

    python_exe = sys.executable

    train_res = parse_res_tuple(args.resolution)
    hr_eval_res = parse_res_tuple(args.hr_eval_resolution)

    config = {
        "train_data_path": args.train_data_path,
        "train_cache_path": args.train_cache_path,
        "train_resolution": train_res,
        "train_downsample": args.downsample,
        "crop_h_cg": args.crop_h_cg,
        "crop_w_cg": args.crop_w_cg,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "n_crops_train": args.n_crops_train,
        "n_crops_val": args.n_crops_val,
        "n_crops_test": args.n_crops_test,
        "ema_alpha": args.ema_alpha,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gate_epochs": args.gate_epochs,
        "gate_lr": args.gate_lr,
        "seed": args.seed,
    }

    # Calculate HR reference values dynamically once at the beginning
    hr_emissivity, hr_emiss_width, hr_mass_flux = compute_hr_references(
        hr_cache_folder=args.hr_cache_folder,
        hr_eval_output=args.hr_eval_output,
        hr_res=hr_eval_res,
        hr_downsample=args.hr_eval_downsample,
    )

    def objective(trial):
        trial_name = f"optuna_trial_{trial.number}"
        trial_output_dir = os.path.join(PROJECT_ROOT, "simulation_outputs", trial_name)
        trial_model_dir = os.path.join(PROJECT_ROOT, "runs", "trial_models", trial_name)

        try:
            train_cnn_trial(trial, trial_model_dir, args.epochs_per_trial, python_exe, config)
            run_athena_trial(
                trial,
                output_dir=trial_output_dir,
                model_dir=trial_model_dir,
                athena_dir=args.athena_dir,
                athinput_path=args.athinput,
                restart_path=args.restart_file,
                config=config,
            )
            score = compare_simulation(
                trial,
                trial_output_dir,
                hr_emissivity=hr_emissivity,
                hr_emiss_width=hr_emiss_width,
                hr_mass_flux=hr_mass_flux,
                model_dir=trial_model_dir,
                config=config,
                w_emiss=args.w_emiss,
                w_width=args.w_width,
                w_mass=args.w_mass,
            )
        finally:
            if not args.keep_trials:
                if os.path.exists(trial_output_dir):
                    shutil.rmtree(trial_output_dir, ignore_errors=True)
                if os.path.exists(trial_model_dir):
                    shutil.rmtree(trial_model_dir, ignore_errors=True)

        return score

    if args.single_trial:
        print("Running single dummy trial test...")
        dummy = DummyTrial(
            {
                "alpha_active_wasserstein": 10.0,
                "alpha_inactive_wasserstein": 10.0,
                "alpha_emiss": 10.0,
                "alpha_gate": 0.0,
                "alpha_mean_temp": 10.0,
                "alpha_leak": 10.0,
            },
            number=0,
        )
        score = objective(dummy)
        print(f"Dummy trial completed with score: {score:.6f}")
        return

    print("=" * 80)
    print(f"Initializing Optuna study '{args.study_name}' (storage: {args.storage})")
    print("=" * 80)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",
    )

    print(f"Completed trials before this run: {len(study.trials)}")
    study.optimize(objective, n_trials=args.n_trials)

    print("=" * 80)
    print("OPTUNA OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Best Score : {study.best_value:.6f}")
    print(f"Best Trial : #{study.best_trial.number}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Re-train and save the best model weights
    os.makedirs(args.output_best_dir, exist_ok=True)
    print("=" * 80)
    print(f"Retraining final best model and saving to {args.output_best_dir}...")
    print("=" * 80)
    best_trial = DummyTrial(study.best_params, number=study.best_trial.number)
    train_cnn_trial(best_trial, args.output_best_dir, args.epochs_per_trial, python_exe, config)
    print(f"Best model saved successfully in: {args.output_best_dir}")


if __name__ == "__main__":
    main()
