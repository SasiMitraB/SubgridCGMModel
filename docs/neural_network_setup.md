# Neural Network Integration in AthenaK Simulations

This report details how the Neural Network (NN) subgrid models are integrated into the `athenak` magnetohydrodynamics (MHD) simulations within the `SubgridCGMModel` project. The integration bridges a highly-parallelized C++ simulation environment (AthenaK) with a Python-based deep learning framework (PyTorch).

## 1. Integration Architecture (C++ / Python Bridge)

The core mechanism linking the `athenak` C++ code to the Python-based Neural Network is the **pybind11** library. This library enables the C++ simulation to embed a Python interpreter and call Python functions directly, allowing seamless data exchange without relying on external file I/O during the simulation run.

### Setup in C++ (`athenak/src/pgen/subgrid.cpp`)
- An embedded Python interpreter (`py::scoped_interpreter`) is instantiated.
- The C++ code imports a Python module named `source_module.py` and extracts a specific function called `source_func`.
- This setup is handled dynamically; the Python environment is initialized once and persists across simulation timesteps to avoid overhead.

## 2. Data Extraction and Transfer

At each evaluation of the physics source terms (`UserSourceTerm`), the simulation extracts the current state of the fluid.

1. **Flattening the Grid:** The multi-dimensional Kokkos views (which hold the simulation state on the GPU or CPU) are flattened. The relevant primitive variables extracted are:
   - Density (`rho`)
   - Pressure (`pres`)
   - X and Y velocities (`ux`, `uy`)
   - Passive scalar/tracer (`ps`)
   - Fractional mass cooled (`fmcl`)
2. **Transfer to Python:** These flattened arrays are wrapped as `py::array_t<double>` (NumPy arrays) without copying the underlying memory where possible. They are then passed as arguments to the Python `source_func`.

## 3. Neural Network Evaluation (`source_module.py`)

The active `source_func` function in Python is responsible for preparing the data, running the NN inference, and post-processing the output.

### 3.1 Pre-processing
- **Temperature Calculation:** The fluid temperature is derived from the pressure and density arrays using the ideal gas law ($T \propto P/\rho$).
- **Tensor Formatting:** The inputs (density, temperature, velocities, and passive scalar) are reshaped into 2D spatial grids to match the CNN's expected input structure (`(1, C, nx, ny)`).
- **Normalization:** The inputs are standardized (zero mean, unit variance) using pre-calculated `input_mean` and `input_std` arrays loaded from disk. 
- The resulting tensor is moved to the appropriate hardware device (CPU or GPU) for PyTorch (`torch`).

### 3.2 Network Architecture and Inference
- The active model uses a **Convolutional Neural Network (CNN)**, specifically defined as `GMM_CNN` (Gaussian Mixture Model CNN). 
- **Inference:** The PyTorch model operates in evaluation mode (`with torch.no_grad():`). It processes the normalized input tensor and outputs parameters for a Gaussian Mixture Model:
  - Mixture weights
  - Means (`mu`)
  - Standard deviations (`sigma`)
- *Note: Previous iterations of the code, retained as comments, show that the project also experimented with predicting source terms directly or predicting subgrid fluxes using architectures like ResUNet.*

### 3.3 Post-processing and Physics Application
- **PDF Construction:** The GMM parameters predicted by the NN are used to construct a continuous Probability Density Function (PDF) of temperatures within each subgrid cell.
- **Cooling Calculation:** Instead of applying a single cooling rate based on the average cell temperature, the code integrates the cooling function (`lambda_cool(T)`) over the predicted temperature PDF:
  $$ \text{cool\_rate} = \int \text{PDF}(T) \cdot \Lambda(T) \cdot n^2 \, dT $$
  This captures the enhanced cooling caused by unresolved dense, cold structures.
- **Smoothing:** An adaptive Gaussian smoothing filter is applied to the resulting source terms to prevent numerical instabilities at sharp boundaries.
- **Formatting for C++:** The final computed source terms for the five conserved variables are flattened, transposed, and returned to C++ as a single NumPy array of shape `(5, N)`.

## 4. Applying the Source Terms in AthenaK

Back in `subgrid.cpp`, the Python function returns the computed source terms. 
- The C++ code unwraps the NumPy array using `py::array_t::unchecked<2>()` for fast access.
- A parallelized Kokkos loop (`par_for`) iterates over the mesh blocks.
- The returned source terms are multiplied by the time step (`bdt`) and added to the corresponding conserved variables (Density, Momentum, Energy, and `fmcl`).

## Summary Flow
1. **AthenaK** extracts fluid state $\rightarrow$ 
2. **pybind11** passes arrays to Python $\rightarrow$ 
3. **PyTorch** CNN infers subgrid Temperature PDF $\rightarrow$ 
4. **Python** calculates exact radiative cooling rate from PDF $\rightarrow$ 
5. **pybind11** returns source terms $\rightarrow$ 
6. **AthenaK** updates the simulation state.
