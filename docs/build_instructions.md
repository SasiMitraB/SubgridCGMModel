# Build & Run Instructions

## Python Dependencies (ML Pipeline)

```bash
pip install numpy scipy matplotlib h5py tqdm torch scikit-image pybind11
```

Or install from the project requirements file:

```bash
pip install -r requirements.txt
```

## Setting Data Paths

Before running any training script, set the environment variables pointing to your simulation data:

```bash
export SUBGRID_DATA_PATH="/path/to/athenak/build/bin"   # raw .bin simulation output files
export SUBGRID_CACHE_PATH="/path/to/local/cache"         # for preprocessed .npy cache files
```

These are read automatically by all `models/` scripts at startup.

---

## Complete Workflow Overview

The CGM subgrid modeling pipeline follows a three-step process:

1. **Generate Training Data**: Run a high-resolution HR simulation using AthenaK
2. **Train the Model**: Use Python ML scripts to train neural networks on the simulation outputs
3. **Integrate into Simulation**: Use the trained model in the subgrid problem to compute closure terms dynamically

### Quick Start Example

```bash
# Step 1: Setup
cd /path/to/SubgridCGMModel
pip install -r requirements.txt
export SUBGRID_DATA_PATH="$(pwd)/athenak/build/bin"
export SUBGRID_CACHE_PATH="$(pwd)/data/cache"
mkdir -p $SUBGRID_CACHE_PATH

# Step 2: Run HR simulation (generates .bin training data)
cd athenak && mkdir -p kh_build && cd kh_build
cmake -S .. -B . -DCMAKE_BUILD_TYPE=Release -DPROBLEM=kh_radiative_cooling
cmake --build . -j$(nproc)
cp ../inputs/subgrid/sg.athinput .
./athenak -i sg.athinput -d outputs/

# Step 3: Train model (reads .bin files and trains network)
cd /path/to/SubgridCGMModel
python models/conv_nn/cnn.py

# Step 4: Run subgrid simulation with trained model
cd athenak && mkdir -p sg_build && cd sg_build
cmake -S .. -B . -DCMAKE_BUILD_TYPE=Release -DPROBLEM=subgrid
cmake --build . -j$(nproc)
cp ../inputs/subgrid/sg.athinput .
./athenak -i sg.athinput -d outputs/
```

---

## 1) HR Simulations (Kelvin-Helmholtz with Radiative Cooling)

### Building the Athena executable

```bash
cd athenak
mkdir -p kh_build && cd kh_build

cmake -S .. -B . \
  -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_CUDA=OFF \
  -DPROBLEM=kh_radiative_cooling

cmake --build . -j$(nproc)
```

### Copy the input file

```bash
cp ../inputs/subgrid/sg.athinput .   # or the appropriate athinput for HR
```

### Running the simulation

```bash
# Fresh run
./athena -i (input_file).athinput -d (output_folder)/

# Restart from checkpoint
./athena -i (input_file).athinput -d (output_folder)/ -r (old_folder)/rst/(restart_file).rst
```

---

## 2) Subgrid Simulations

### Building the Athena executable

```bash
cd athenak
mkdir -p sg_build && cd sg_build

cmake -S .. -B . \
  -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_CUDA=OFF \
  -DPROBLEM=subgrid

cmake --build . -j$(nproc)
```

### Copy the input file

```bash
cp ../inputs/subgrid/sg.athinput .
```

### Running the simulation

```bash
# Fresh run
./athena -i sg.athinput -d (output_folder)/

# Restart from checkpoint
./athena -i sg.athinput -d (output_folder)/ -r (old_folder)/rst/(restart_file).rst
```

---

## 3) Training the Neural Networks

All training scripts live in `models/`. Run from the project root or from within the script's directory.

### What Happens During Training

The training pipeline:
1. Loads binary `.bin` files from the HR simulation (via `SUBGRID_DATA_PATH`)
2. Preprocesses data using [data/data_preprocess.py](../data/data_preprocess.py):
   - Coarse-grains high-resolution fields
   - Computes gradients and Hessians
   - Calculates source terms (e.g., `fmcl` - fluid-mesh coupling losses)
3. Trains the neural network model
4. Saves outputs:
   - Trained model weights → `outputs/model_saves/` (`.pth` files)
   - Normalization arrays → `outputs/model_saves/` (`.npy` files)
   - Training loss curves → `outputs/loss_plots/` (plots)

### Available Models

```bash
# CNN for learning source term predictions (recommended for spatial fields)
python models/conv_nn/cnn.py

# Flux CNN for predicting flux components
python models/conv_nn/flux_cnn.py

# Individual component CNNs
python models/conv_nn/indiv_cnn.py

# PDF CNN for temperature probability distributions
python models/conv_nn/pdf_cnn.py

# Feedforward NN (simpler, faster training)
python models/feedforward_nn/fnn.py

# ConvLSTM for temporal predictions
python models/conv_lstm/clstm.py
```

### Training Output Location

Saved models and loss plots go to `outputs/model_saves/` and `outputs/loss_plots/` respectively.

The trained weights from this step are automatically loaded by the subgrid problem during step 4.

---

## 4) Subgrid Problem Integration

The subgrid problem (`DPROBLEM=subgrid`) is the final integration step. It:

1. Reads your trained model weights from `outputs/model_saves/`
2. Receives coarse-grained simulation fields as input
3. Uses the trained neural network to predict closure terms (source terms, fluxes, PDFs)
4. Feeds these predictions back into the simulation to improve accuracy

This closes the loop: **HR Simulation → Training Data → Model → Low-Resolution Simulation with Closures**

---

## Key Files Reference

| File/Directory | Purpose |
|---|---|
| `athenak/inputs/subgrid/sg.athinput` | Configuration file for KH instability simulations |
| `data/data_preprocess.py` | Loads `.bin` files and preprocesses simulation data |
| `data/bin_convert.py` | Converts binary simulation outputs |
| `models/conv_nn/cnn.py` | Convolutional neural network for source term prediction |
| `models/feedforward_nn/fnn.py` | Fully-connected network model |
| `models/conv_lstm/clstm.py` | Temporal ConvLSTM model |
| `outputs/model_saves/` | Trained model weights (`.pth`) and normalization arrays (`.npy`) |
| `outputs/loss_plots/` | Training and validation loss curves |

---

## Tips & Troubleshooting

- **Data path issues**: Ensure `SUBGRID_DATA_PATH` points to the directory containing `.bin` files from your HR simulation
- **Cache management**: Preprocessed data is cached in `SUBGRID_CACHE_PATH`. Delete cache files if you modify the preprocessing logic
- **Memory usage**: For large simulations (>512×512), reduce batch size in training scripts if out-of-memory errors occur
- **GPU acceleration**: PyTorch automatically uses GPU if `torch.cuda.is_available()` returns True
