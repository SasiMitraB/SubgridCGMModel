#!/usr/bin/env python3
"""
optimize_subgrid.py — Closed-loop Optuna Bayesian Optimization for SubgridCGM

Optimizes PDF CNN loss hyperparameters (alpha_emiss, alpha_gate, alpha_mean_temp, etc.)
by training a model, executing an Athena-K subgrid simulation restart (5 -> 10 Myr),
and scoring downstream physical observables against dynamically calculated
high-resolution reference benchmarks (matching data/mocks/mock_sg.py).
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


def coarse_grain_array(arr, ds=32):
    """Coarse grains a 2D or 3D array by factor ds (matches mock_sg.py)."""
    if arr.ndim == 2:
        return arr.reshape(arr.shape[0] // ds, ds, arr.shape[1] // ds, ds).mean(axis=(1, 3))
    elif arr.ndim == 3:
        return arr.reshape(arr.shape[0], arr.shape[1] // ds, ds, arr.shape[2] // ds, ds).mean(axis=(2, 4))
    else:
        raise ValueError(f"Unsupported array dimension: {arr.ndim}")


def resolve_hr_cache_folder(custom_folder: str = None) -> str:
    """Resolves HR cache folder path matching mock_sg.py and pdf_cnn.py."""
    if custom_folder and os.path.exists(custom_folder):
        return custom_folder

    hr_file_path = os.environ.get(
        "HR_SIM_OUTPUT",
        os.path.join(PROJECT_ROOT, "simulation_outputs", "hr_build_512"),
    )
    hr_cache_base = os.environ.get(
        "SUBGRID_CACHE_PATH",
        os.path.join(hr_file_path, "cache"),
    )

    _res_str = os.environ.get("PDF_CNN_RESOLUTION", "512,256").split(",")
    hr_resolution = (int(_res_str[0].strip()), int(_res_str[1].strip()))
    hr_downsample = int(os.environ.get("PDF_CNN_DOWNSAMPLE", "32"))

    # Canonical format in pdf_cnn.py: f"sc{hr_resolution}_{hr_downsample}" -> sc(512, 256)_32
    candidate = os.path.join(hr_cache_base, f"sc{hr_resolution}_{hr_downsample}")
    if os.path.exists(candidate):
        return candidate

    # Alternative string format without tuple parens: sc512,256_32
    candidate_alt = os.path.join(hr_cache_base, f"sc{hr_resolution[0]},{hr_resolution[1]}_{hr_downsample}")
    if os.path.exists(candidate_alt):
        return candidate_alt

    # Scan directory for any matching sc* folder
    if os.path.exists(hr_cache_base):
        for entry in sorted(os.listdir(hr_cache_base)):
            sub = os.path.join(hr_cache_base, entry)
            if os.path.isdir(sub) and entry.startswith("sc"):
                return sub

    return candidate


def compute_hr_references(hr_cache_folder: str = None, hr_downsample: int = 32, n_frames: int = 500):
    """
    Computes high-resolution reference observables (integrated emissivity and top mass flux)
    directly from the cached HR simulation snapshots, matching data/mocks/mock_sg.py.
    """
    actual_cache_folder = resolve_hr_cache_folder(hr_cache_folder)

    print("=" * 80)
    print("Computing HR reference values dynamically from cached HR simulation data...")
    print(f"  Cache Folder : {actual_cache_folder}")
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

    # 1. HR Integrated Emissivity
    emis_hr = hr_rho**2 * lambda_cool(hr_temp, mask=True)
    emis_cg_hr = coarse_grain_array(emis_hr, hr_downsample)
    emis_cg_hr_xavg = np.mean(emis_cg_hr, axis=2)
    emis_cg_hr_mean = np.mean(emis_cg_hr_xavg, axis=0)

    y_cg_hr = np.linspace(-10.0, 10.0, emis_cg_hr.shape[1])
    int_hr_emiss = float(np.trapezoid(emis_cg_hr_mean, y_cg_hr))

    # 2. HR Mass Flux at top boundary
    mass_flux_hr = float(np.mean((hr_rho * hr_uy)[:, -1, :]) / (rho0 * du))

    print(f"Dynamically Calculated HR Ground Truth:")
    print(f"  HR Integrated Emissivity : {int_hr_emiss:.6e}")
    print(f"  HR Mass Flux             : {mass_flux_hr:.6e}")
    print("=" * 80)

    return int_hr_emiss, mass_flux_hr


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


def train_cnn_trial(trial, model_dir: str, epochs: int, python_exe: str) -> dict:
    """Suggests hyperparameters and trains PDF CNN."""
    print("=" * 80)
    print(f"Trial {trial.number}: Proposing hyperparameters and training CNN...")
    print("=" * 80)

    alpha_emiss = trial.suggest_float("alpha_emiss", 1.0, 500.0, log=True)
    alpha_gate = trial.suggest_float("alpha_gate", 0.1, 100.0, log=True)
    alpha_mean_temp = trial.suggest_float("alpha_mean_temp", 0.1, 50.0, log=True)

    print(f"Trial {trial.number} Hyperparameters:")
    print(f"  alpha_emiss    : {alpha_emiss:.4f}")
    print(f"  alpha_gate     : {alpha_gate:.4f}")
    print(f"  alpha_mean_temp: {alpha_mean_temp:.4f}")
    print(f"  epochs         : {epochs}")

    os.makedirs(model_dir, exist_ok=True)

    cmd = [
        python_exe,
        os.path.join(MODELS_DIR, "pdf_cnn.py"),
        "--alpha_emiss", str(alpha_emiss),
        "--alpha_gate", str(alpha_gate),
        "--alpha_mean_temp", str(alpha_mean_temp),
        "--num_epochs", str(epochs),
        "--model_save_dir", model_dir,
    ]

    res = subprocess.run(
        cmd,
        cwd=MODELS_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if res.returncode != 0:
        print(f"Trial {trial.number}: Training failed with exit code {res.returncode}:")
        print(res.stderr[-1000:] if len(res.stderr) > 1000 else res.stderr)
        raise RuntimeError(f"CNN training failed in trial {trial.number}")

    print(f"Trial {trial.number}: Training completed successfully.")
    return {
        "alpha_emiss": alpha_emiss,
        "alpha_gate": alpha_gate,
        "alpha_mean_temp": alpha_mean_temp,
    }


def run_athena_trial(
    trial,
    output_dir: str,
    model_dir: str,
    athena_dir: str,
    athinput_path: str,
    restart_path: str,
) -> None:
    """Runs Athena-K subgrid model simulation restarting from 5 Myr checkpoint."""
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

    env = os.environ.copy()
    env["MODEL_SAVES_DIR"] = model_dir
    venv_site = os.path.join(PROJECT_ROOT, "venv", "lib", "python3.10", "site-packages")
    if not os.path.exists(venv_site):
        venv_site = os.path.join(PROJECT_ROOT, "venv", "lib")

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
    hr_mass_flux: float,
    model_dir: str = None,
) -> float:
    """Computes integrated emissivity and mass flux errors against HR reference."""
    print("=" * 80)
    print(f"Trial {trial.number}: Evaluating physical observables against HR reference...")
    print("=" * 80)

    rho0 = 1e-3
    du = 31.0918
    unit_fix = 1.975e27

    sim_data = simulation_data()
    sim_data.resolution = (16, 8)
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

    # Evaluate subgrid cooling rate using CNN prediction
    _res_str = os.environ.get("PDF_CNN_RESOLUTION", "512,256").split(",")
    _fine_res = (int(_res_str[0].strip()), int(_res_str[1].strip()))
    _cnn_ds = int(os.environ.get("PDF_CNN_DOWNSAMPLE", "32"))
    T_edges = np.logspace(3.0, 7.0, out_channels + 1)
    T_centers = np.sqrt(T_edges[:-1] * T_edges[1:])

    emis_arr = np.zeros_like(rho)
    for t in range(rho.shape[0]):
        pdf_t = snapshot_pred_16x8(
            rho[t], temp[t], ux[t], uy[t], ps[t],
            fine_resolution=_fine_res, downsample=_cnn_ds,
            model_save_dir=model_dir,
        )
        cool_code = compute_cooling_rate(
            pdf_t, T_centers, is_pdf=True, rho_cg=rho[t]
        )
        # Convert to CGS emissivity matching mock_sg.py
        emis_arr[t] = cool_code / unit_fix

    emis_sg_xavg = np.mean(emis_arr, axis=2)
    emis_sg_mean = np.mean(emis_sg_xavg, axis=0)

    y_sg = np.linspace(-10.0, 10.0, rho.shape[1])
    int_emiss = float(np.trapezoid(emis_sg_mean, y_sg))

    # Mass flux at top boundary
    mass_flux = float(np.mean((rho * uy)[:, -1, :]) / (rho0 * du))

    emiss_err = abs(int_emiss - hr_emissivity) / (abs(hr_emissivity) + 1e-30)
    mass_err = abs(mass_flux - hr_mass_flux) / (abs(hr_mass_flux) + 1e-30)
    score = emiss_err + mass_err

    print(f"Trial {trial.number} Evaluation Metrics:")
    print(f"  Integrated Emissivity : {int_emiss:.6e} (Target: {hr_emissivity:.6e}, Rel Err: {emiss_err:.2%})")
    print(f"  Mass Flux             : {mass_flux:.6e} (Target: {hr_mass_flux:.6e}, Rel Err: {mass_err:.2%})")
    print(f"  Objective Score (Sum) : {score:.6f}")
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
    parser.add_argument("--hr_cache_folder", type=str, default=None, help="Path to cached HR simulation dataset")
    parser.add_argument("--hr_downsample", type=int, default=32, help="Downsample factor for HR coarse-graining")
    parser.add_argument("--keep_trials", action="store_true", help="Keep all trial simulation outputs")
    parser.add_argument("--single_trial", action="store_true", help="Run a single trial with default parameters")
    args = parser.parse_args()

    python_exe = sys.executable

    # Calculate HR reference values dynamically once at the beginning
    hr_emissivity, hr_mass_flux = compute_hr_references(
        hr_cache_folder=args.hr_cache_folder,
        hr_downsample=args.hr_downsample,
    )

    def objective(trial):
        trial_name = f"optuna_trial_{trial.number}"
        trial_output_dir = os.path.join(PROJECT_ROOT, "simulation_outputs", trial_name)
        trial_model_dir = os.path.join(PROJECT_ROOT, "runs", "trial_models", trial_name)

        try:
            train_cnn_trial(trial, trial_model_dir, args.epochs_per_trial, python_exe)
            run_athena_trial(
                trial,
                output_dir=trial_output_dir,
                model_dir=trial_model_dir,
                athena_dir=args.athena_dir,
                athinput_path=args.athinput,
                restart_path=args.restart_file,
            )
            score = compare_simulation(
                trial,
                trial_output_dir,
                hr_emissivity=hr_emissivity,
                hr_mass_flux=hr_mass_flux,
                model_dir=trial_model_dir,
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
                "alpha_emiss": 10.0,
                "alpha_gate": 50.0,
                "alpha_mean_temp": 10.0,
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
    train_cnn_trial(best_trial, args.output_best_dir, args.epochs_per_trial, python_exe)
    print(f"Best model saved successfully in: {args.output_best_dir}")


if __name__ == "__main__":
    main()
