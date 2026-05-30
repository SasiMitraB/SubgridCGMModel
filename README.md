# Subgrid CGM Modelling Project

This repository contains the CGM (Circumgalactic Medium) subgrid modelling pipeline. The project is split into two primary components:
1. **AthenaK** (C++/Kokkos-based MHD simulator): Generates high-fidelity Kelvin-Helmholtz instability simulations to produce raw training data.
2. **ML Pipeline** (Python/PyTorch): Preprocesses simulation outputs and trains FNN, CNN, and ConvLSTM models to predict subgrid source terms and temperature PDFs.

---

## 📂 Project Structure

The repository is cleanly organized for active development:

```
SubgridCGMModel/
├── athenak/              # AthenaK MHD simulation code (C++/Kokkos)
│   ├── src/              # Source code
│   ├── external/         # External dependencies
│   ├── kokkos/           # Kokkos library
│   ├── inputs/           # Simulation parameter input files
│   ├── scripts/          # Build and utility scripts
│   ├── vis/              # Visualization tools
│   └── ...
├── models/               # Machine Learning Neural Network pipelines
│   ├── feedforward_nn/   # Fully Connected FNN models (fnn.py)
│   ├── conv_nn/          # Convolutional Neural Networks (CNNs for fields/fluxes/PDFs)
│   └── conv_lstm/        # Convolutional LSTM model architectures
├── data/                 # Raw data preprocessing & conversion scripts
│   ├── mocks/            # Baseline mocks and validation datasets
│   └── ...
├── outputs/              # Consolidated training outputs (Git-ignored locally)
│   ├── model_saves/      # Saved .pth weights and normalization .npy arrays
│   └── loss_plots/       # Matplotlib training and validation loss curves
├── docs/                 # Documentation (build instructions, architecture logs)
│   └── build_instructions.md
├── shell_scripts/        # Utility and automation scripts
├── archived_builds/      # **Predecessor's builds, configs, and input catalogs (see archived_builds/INDEX.md)**
│   ├── build_configs_backup/     # Backed-up CMake configurations
│   ├── build_full/, test_build_full/, test_dir_full/  # Full archived builds
│   ├── laptop_build_hr/, laptop_build_sg/, kh_build/, sd_build/  # Specialized builds
│   ├── predecessor_catalogs/     # Input files and build documentation
│   └── INDEX.md          # Guide to archived materials
├── requirements.txt      # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

### Key Directories

- **Active Development**: `athenak/`, `models/`, `data/`, `outputs/`, `docs/`
- **Reference Only**: `archived_builds/` - Contains Dipayan's work; safely ignore unless referencing configurations

---

## 🚀 Getting Started

### 1. Prerequisites
Install all the Python pipeline dependencies:
```bash
pip install -r requirements.txt
```

### 2. Building AthenaK
Detailed C++/Kokkos build recipes and environment specifications (HPC & local) are documented in [docs/build_instructions.md](file:///Volumes/PortableSSD/Projects/SubgridCGMModel/docs/build_instructions.md).

### 3. Training & Inference
ML pipeline scripts are located under `models/`. Each training script automatically saves its output models and loss plots into the unified `outputs/` directory structure. 

All absolute path bottlenecks have been replaced with relative paths or environment variables:
- **`SUBGRID_DATA_PATH`**: Environment variable override for raw simulation binary outputs.
- **`SUBGRID_CACHE_PATH`**: Environment variable override for local npy data cache.
