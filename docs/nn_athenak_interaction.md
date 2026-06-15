# Neural Network & AthenaK Interaction: A Complete Technical Reference

*SubgridCGMModel project — `athenak/` + `models/` + `data/`*

---

## Table of Contents

1. [Project Overview and Physical Motivation](#1-project-overview-and-physical-motivation)
2. [High-Level Architecture](#2-high-level-architecture)
3. [The AthenaK Simulation Engine](#3-the-athenak-simulation-engine)
4. [Data Extraction and Preprocessing Pipeline](#4-data-extraction-and-preprocessing-pipeline)
5. [Neural Network Models](#5-neural-network-models)
6. [The C++/Python Bridge: pybind11 Integration](#6-the-cpython-bridge-pybind11-integration)
7. [The Subgrid Problem Generator (Runtime Interface)](#7-the-subgrid-problem-generator-runtime-interface)
8. [The End-to-End Pipeline Orchestration](#8-the-end-to-end-pipeline-orchestration)
9. [Configuration System and Simulation Parameters](#9-configuration-system-and-simulation-parameters)
10. [Build System and Compilation](#10-build-system-and-compilation)
11. [Data Flow Summary and Worked Example](#11-data-flow-summary-and-worked-example)
12. [Key Design Decisions and Trade-offs](#12-key-design-decisions-and-trade-offs)

---

## 1. Project Overview and Physical Motivation

### 1.1 The Scientific Problem

The **Circumgalactic Medium (CGM)** is a diffuse, multi-phase gas halo surrounding galaxies. Cosmological simulations of the CGM must capture phenomena spanning many orders of magnitude in length scale, from parsec-scale cold cloud condensation to kiloparsec-scale galactic outflows. This imposes a fundamental resolution conflict: the computational cost of fully resolving all scales simultaneously in a global simulation is prohibitive.

The central challenge is the **cooling catastrophe at coarse resolution**. Radiative cooling in a multi-temperature gas is highly non-linear — the cooling rate scales as $\Lambda(T) \propto n^2 f(T)$. When a simulation cell contains a mixture of hot (~10⁶ K) and cold (~10⁴ K) gas at unresolved scales, simply using the mean cell temperature dramatically *underestimates* the actual cooling rate, because the cold, dense clumps dominate the emission.

The subgrid model in this project addresses this precisely: instead of discarding the sub-cell temperature structure, a **neural network predicts the probability distribution of temperatures** (a Temperature PDF) within each coarse simulation cell. The emissivity is then computed by integrating the cooling function $\Lambda(T)$ over this predicted PDF, recovering a much more accurate effective cooling rate for the low-resolution run.

### 1.2 Test Problem: Kelvin-Helmholtz Instability

The physical setup is a 2D **Kelvin-Helmholtz Instability (KHI)** with radiative cooling. This is a paradigmatic problem in CGM physics: a shear interface between a cold dense gas slab and a hot diffuse medium becomes unstable, producing turbulent mixing and condensation of cold clouds — exactly the regime where subgrid physics is critical.

| Phase | Density ($\rho$) | Temperature ($T$) | Velocity ($v_x$) |
|---|---|---|---|
| Cold (inner) | $\rho_\text{cold} = 0.1$ | ~$10^4$ K | $v_{x,\text{cold}} = -2.83$ |
| Hot (outer) | $\rho_\text{hot} = 0.001$ | ~$10^6$ K | $v_{x,\text{hot}} = +28.27$ |

The initial pressure is equal in both phases ($P = 8.63$) ensuring pressure balance. A sinusoidal multi-mode perturbation seeds the instability.

---

## 2. High-Level Architecture

The system is structured as a three-stage pipeline connecting AthenaK simulations, Python-based machine learning, and a runtime coupling layer:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 1: TRAINING DATA GENERATION                          │
│                                                                                  │
│  AthenaK HR Simulation                                                           │
│  (builds/hr_build)                                                               │
│  Problem: kh_radiative_cooling                                                   │
│  Resolution: 256×512 cells                          Binary .bin output files     │
│  Physics: KHI + ISM radiative cooling  ─────────►  (hydro_w, hydro_u)           │
│  Integrator: RK2, PLM, HLLC                         simulation_outputs/hr_build/ │
└─────────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼ data/bin_convert.py + data_preprocess.py
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 2: MACHINE LEARNING TRAINING                         │
│                                                                                  │
│  Python/PyTorch Pipeline                                                         │
│  ┌──────────────────────────────────────────────────┐                           │
│  │  data_preprocess.py                               │                           │
│  │  - Parse binary .bin files (bin_convert.py)       │                           │
│  │  - Compute Temperature from ideal gas law         │                           │
│  │  - Coarse-grain fields (block_reduce, ×64)        │                           │
│  │  - Compute pixel-level Temperature PDFs           │                           │
│  │  - Compute source terms (finite differences)      │                           │
│  └──────────────────────────────────────────────────┘                           │
│                         │                                                        │
│                         ▼                                                        │
│  ┌──────────────────────────────────────────────────┐                           │
│  │  models/conv_nn/pdf_cnn.py (PRIMARY MODEL)       │                           │
│  │  Architecture: Encoder-Decoder CNN               │                           │
│  │  Input: (5, 16, 8) coarse-grained fields         │                           │
│  │  Output: (40, 16, 8) temperature PDF bins        │                           │
│  │  Loss: PDFEmissivityLoss (KL + Emissivity MSE)   │                           │
│  └──────────────────────────────────────────────────┘                           │
│                         │                                                        │
│                         ▼  Saved artifacts                                       │
│              outputs/model_saves/                                                │
│              ├── cnn_(1024,512)_64.pth              (model weights)             │
│              ├── cnn_(1024,512)_64_input_mean.npy   (normalization)             │
│              └── cnn_(1024,512)_64_input_std.npy    (normalization)             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼ pybind11 runtime coupling
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 3: SUBGRID SIMULATION                                │
│                                                                                  │
│  AthenaK SG Simulation                                                           │
│  (builds/subgrid_model)                                                          │
│  Problem: subgrid.cpp                               Each timestep:              │
│  Resolution: 8×16 cells         ◄──────────────►   C++ extracts fluid state    │
│  Physics: KHI + NN source terms   pybind11          Python CNN infers PDF       │
│  Integrator: RK2, PLM, HLLC      (embedded         Emissivity computed         │
│                                   interpreter)      Source terms returned        │
│                                                     Conserved vars updated       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. The AthenaK Simulation Engine

### 3.1 What is AthenaK?

AthenaK (also referred to as AthenaXXX or Athenak in the codebase) is a modern, performance-portable astrophysical MHD code written in C++17, using the **Kokkos** performance portability library for GPU and multi-core CPU execution. It solves the equations of ideal hydrodynamics (or MHD) on a structured Cartesian mesh using:

- **Time integration**: 2nd-order Runge-Kutta (RK2)
- **Spatial reconstruction**: Piecewise Linear Method (PLM)
- **Riemann solver**: HLLC (Harten-Lax-van Leer-Contact)
- **Parallelism**: Kokkos thread-level parallelism (supporting OpenMP and CUDA/HIP)

### 3.2 Data Structures: Kokkos Views

The simulation state is stored in multi-dimensional **Kokkos Views** — abstractions over array memory that can reside on CPU or GPU:

```cpp
// Primitive variables: density, energy, velocities, scalars
DvceArray5D<Real> w0_;  // Shape: [nmb, nvar, nk, nj, ni]

// Conserved variables: rho, momentum, total energy, passive scalars
DvceArray5D<Real> u0_;  // Shape: [nmb, nvar, nk, nj, ni]
```

Here `nmb` is the number of MeshBlocks in the current MeshBlockPack (the fundamental tile of the computation), and `nvar` is the number of fluid variables.

### 3.3 Variable Indexing

The code uses symbolic indices defined in `athena.hpp`:

| Index | Variable |
|---|---|
| `IDN` | Density ($\rho$) |
| `IEN` | Energy density ($e = P / (\gamma - 1)$ for primitives, total $E$ for conserved) |
| `IVX` / `IM1` | x-velocity / x-momentum |
| `IVY` / `IM2` | y-velocity / y-momentum |
| `IVZ` / `IM3` | z-velocity / z-momentum |
| `nfluid` | Index of first passive scalar (tracer `s_00`) |
| `nfluid + 1` | Index of second passive scalar (`frho` — fractional mass of cold gas) |

### 3.4 Problem Generators

AthenaK's simulation physics is defined through **Problem Generators** (`pgen/`) — pluggable C++ translation units selected at CMake compile time via `-DPROBLEM=<name>`. The relevant problem generators for this project are:

- **`kh_radiative_cooling.cpp`**: HR training simulation. Runs pure hydrodynamics with ISM cooling.
- **`subgrid.cpp`**: The SG inference simulation. Identical KHI setup, but replaces the ISM cooling source term with a call to the neural network via pybind11.

### 3.5 ISM Cooling Function

The ISM cooling function is defined in [`athenak/src/srcterms/ismcooling.hpp`](../athenak/src/srcterms/ismcooling.hpp). It implements a piecewise cooling function $\Lambda(T)$:

```
T ≤ 10^4.0 K :  Λ = 0                               (no cooling)
T ≤ 10^4.2 K :  Koyama & Inutsuka (2002) formula
T ≤ 10^8.15 K:  SPEX tabulated curve (Schure et al. 2009) — 102 data points
T > 10^8.15 K:  CGOLS power-law fit:  Λ = 10^(0.45·log T − 26.065)
```

This cooling function is:
1. Evaluated **directly in C++** during HR simulations (using `ISMCoolFn()` on the GPU via `KOKKOS_INLINE_FUNCTION`)
2. **Replicated in Python** (in both `data_preprocess.py` and `pdf_cnn.py`) for offline emissivity computation, training the NN to reproduce its integrated effect

The Python replicas use identical tabulated data (`lhd` array of 102 values) to ensure physical consistency between the training labels and the C++ evaluation.

---

## 4. Data Extraction and Preprocessing Pipeline

### 4.1 Binary Output Format

AthenaK writes simulation state to binary `.bin` files. Two classes of output files are produced:

| File pattern | Content | Variables |
|---|---|---|
| `KH.hydro_w.NNNNN.bin` | Primitive variables | density, velocities, energy, scalars |
| `KH.hydro_u.NNNNN.bin` | Conserved variables | $\rho$, momentum, total energy, passive scalar densities |

The `data/bin_convert.py` utility parses these binary files and reconstructs named arrays (e.g., `"dens"`, `"velx"`, `"vely"`, `"eint"`, `"s_00"`, `"s_01"`).

### 4.2 The `simulation_data` Class

The [`data/data_preprocess.py`](../data/data_preprocess.py) module defines the `simulation_data` class, the central object for all offline data handling:

```python
class simulation_data():
    T_cutoff: float = 1e5    # K  — phase separation threshold
    down_sample: int = 32    # spatial downsampling factor
    total_time: float = 5.0  # Myr — simulation duration
    delta_time: float = 0.01 # Myr — timestep between snapshots
    total_length: float = 40 # pc  — domain size in x
    total_width: float  = 20 # pc  — domain size in y
    resolution: tuple = (512, 256)
    gamma: float = 5.0/3.0
```

It stores time-series arrays of shape `(n_timesteps, nx, ny)` for all primitive and conserved fields.

### 4.3 Binary Parsing and Temperature Derivation

```python
# For each snapshot file:
self.rho[i]  = bin_convert.make_2D_array(file_data, "dens")
self.ux[i]   = bin_convert.make_2D_array(file_data, "velx")
self.uy[i]   = bin_convert.make_2D_array(file_data, "vely")
self.eint[i] = bin_convert.make_2D_array(file_data, "eint")   # internal energy density
self.ps[i]   = bin_convert.make_2D_array(file_data, "s_00")   # tracer scalar
self.frho[i] = bin_convert.make_2D_array(file_data, "s_01")   # cold mass fraction

# Pressure from ideal gas internal energy (γ=5/3 → P = 2/3 * eint)
self.pressure[i] = 2./3. * self.eint[i]

# Temperature from ideal gas law: T = (P/ρ) * (mean molecular mass / k_B)
# Using code units → physical: factor 1.59916e-14 / 1.381e-16
self.temp[i] = (self.pressure[i] * 1.59916e-14 / self.rho[i]) / 1.381e-16
```

The factor `1.59916e-14` converts from code pressure units to CGS (erg/cm³), and `1.381e-16` erg/K is the Boltzmann constant, with mean molecular mass absorbed into the unit system.

### 4.4 Coarse-Graining (Downsampling)

The HR simulation output at high resolution is spatially averaged into coarse-grained (CG) cells that correspond to what the LR simulation would resolve:

```python
def coarse_grain(self, quan: np.ndarray) -> np.ndarray:
    return skimage.measure.block_reduce(quan, (self.down_sample, self.down_sample), np.mean)
```

With `down_sample=64` (used in `pdf_cnn.py`), a `1024×512` HR grid becomes `16×8` CG cells. Each CG cell contains `64×64 = 4096` fine pixels whose statistics are averaged.

### 4.5 Temperature PDF Computation

The ground-truth label for the PDF CNN training is constructed by histogramming the temperatures of all fine pixels within each coarse cell:

```python
def calc_pixel_pdf(self, bins: int = 200) -> np.ndarray:
    ds = self.down_sample
    temp_bins = np.logspace(3.0, 7.0, bins + 1)   # 10³ to 10⁷ K, log-spaced
    
    for each timestep i:
        for each CG cell (j, k):
            block = self.temp[i, j*ds:(j+1)*ds, k*ds:(k+1)*ds]  # 64×64 fine pixels
            hist, _ = np.histogram(block.ravel(), bins=temp_bins)
            pixel_pdf[i, j, k] = hist / hist.sum()               # normalize to p.d.f.
    
    # Transpose: (n_t, nx_cg, ny_cg, bins) → (n_t, bins, nx_cg, ny_cg)
    pixel_pdf = np.transpose(pixel_pdf, (0, 3, 1, 2))
    return pixel_pdf
```

The PDF uses **40 logarithmically spaced bins** from $10^3$ K to $10^7$ K, matching the temperature range of the ISM cooling function. Bin centers are geometric means of adjacent edges:

```python
T_edges   = np.logspace(3.0, 7.0, out_channels + 1)  # 41 edges
T_centers = np.sqrt(T_edges[:-1] * T_edges[1:])       # 40 geometric centers
```

### 4.6 Source Term Computation

For models trained to predict source terms directly (rather than PDFs), `calc_all_source_terms()` computes the rate of change of each conserved variable due to unresolved subgrid physics:

$$S_\phi = \underbrace{\frac{\partial \langle\phi\rangle}{\partial t}}_{\text{time derivative (finite diff)}} + \underbrace{\nabla \cdot F_\phi(\langle\mathbf{u}\rangle, \langle\phi\rangle)}_{\text{resolved flux divergence}}$$

This is evaluated using **central finite differences** in time (with forward/backward differences at boundaries) and explicit divergence operators over the CG grid. The result is a 5-component array: $[S_\rho, S_{m_x}, S_{m_y}, S_E, S_{f_\text{mcl}}]$.

### 4.7 The `fmcl` (Fractional Mass of Cold Gas)

A key subgrid quantity is the **fractional cold mass density** within each CG cell:

```python
def calc_fmcl(self, rho, temp) -> np.ndarray:
    rho_block  = rho.reshape(nx//ds, ds, ny//ds, ds)
    temp_block = temp.reshape(nx//ds, ds, ny//ds, ds)
    
    fc = np.sum(rho_block * (temp_block < self.T_cutoff), axis=(1,3))  # cold mass
    fh = np.sum(rho_block * (temp_block > self.T_cutoff), axis=(1,3))  # hot mass
    fmcl = fc / (fc + fh)
    return fmcl
```

`fmcl` is a scalar field in $[0, 1]$ indicating what fraction of the mass in each CG cell is below the cutoff temperature $T_\text{cutoff} = 10^5$ K. This field is both:
- A **training input** (passed to the CNN alongside density, temperature, etc.)
- A **tracked conserved quantity** in the SG simulation (`s_01` / `frho_index`)

---

## 5. Neural Network Models

The project contains several model architectures in `models/`, each designed for a specific prediction objective.

### 5.1 Primary Model: PDF CNN (`models/conv_nn/pdf_cnn.py`)

This is the production model integrated into the AthenaK subgrid simulation.

#### Architecture (`ConvNN`)

```
Input: (B, 5, nx_cg, ny_cg)   — 5 coarse-grained fields
       Fields: [ρ, T, u_x, u_y, s (tracer)]

Encoder:
  Conv2d(5 → 32, kernel=5, pad=2) → BN → ReLU → Dropout(0.3)
  Conv2d(32 → 64, kernel=5, pad=2) → BN → ReLU → Dropout(0.3)
  Conv2d(64 → 128, kernel=5, pad=2) → BN → ReLU

Decoder:
  Conv2d(128 → 64, kernel=5, pad=2) → BN → ReLU
  Conv2d(64 → 32, kernel=5, pad=2) → BN → ReLU
  Conv2d(32 → 40, kernel=1)          — 1×1 conv to 40 output bins

Output: (B, 40, nx_cg, ny_cg)  — raw logits for 40 temperature bins

Activation: ThresholdedSoftmax → (B, 40, nx_cg, ny_cg) — normalized PDF
```

The encoder-decoder shares the same spatial resolution throughout (no pooling/upsampling), making it a **fully convolutional network** that preserves spatial structure at the coarse-grain scale.

#### Thresholded Softmax Activation

A custom activation (`ThresholdedSoftmax`) is applied to the logits:

```python
class ThresholdedSoftmax(nn.Module):
    def __init__(self, threshold=1e-3, eps=1e-12):
        ...
    def forward(self, logits):
        p = F.softmax(logits, dim=1)          # standard softmax over bin dim
        p = p * (p >= self.threshold).float() # zero out near-zero bins (sparsity)
        return p / (p.sum(dim=1, keepdim=True) + self.eps)  # renormalize
```

This enforces a **sparse PDF**: temperature bins with predicted probability < 0.1% are set to exactly zero. This is motivated by the physical expectation that each simulation cell only occupies a subset of the full temperature range. Sparsemax avoids gradient issues that would arise from a hard argmax, while the re-normalization preserves the probability constraint $\sum_i p_i = 1$.

#### Training Objective: `PDFEmissivityLoss`

The composite loss function combines three terms:

```python
total_loss = pdf_loss + α_emiss * emiss_loss_pixelwise + α_profile * profile_loss
```

1. **PDF Loss** (KL Divergence): Standard KL divergence between predicted and true PDFs:
   $$\mathcal{L}_\text{PDF} = \text{KL}(p_\text{true} \| p_\text{pred}) = \sum_i p_\text{true}^i \log\frac{p_\text{true}^i}{p_\text{pred}^i}$$

2. **Emissivity Loss** (Pixel-wise MSE in log space): Ensures the predicted PDF produces physically correct radiative emissivity:
   $$\varepsilon = \rho^2 \sum_i p_i \Lambda(T_i)$$
   Loss:
   $$\mathcal{L}_\text{emiss} = \text{MSE}\!\left[\log_{10}\varepsilon_\text{pred},\, \log_{10}\varepsilon_\text{true}\right]$$

3. **Profile Loss**: MSE on the column-averaged (x-projected) emissivity profile, constraining large-scale spatial structure:
   $$\mathcal{L}_\text{profile} = \text{MSE}\!\left[\log_{10}\bar{\varepsilon}_\text{pred}(x),\, \log_{10}\bar{\varepsilon}_\text{true}(x)\right]$$

Default weights: $\alpha_\text{emiss} = 10$, $\alpha_\text{profile} = 10$.

#### Training Configuration

| Hyperparameter | Value |
|---|---|
| Resolution (HR) | `(1024, 512)` pixels |
| Downsample factor | `64` |
| CG resolution | `16 × 8` cells |
| Input channels | 5 |
| Output channels | 40 bins |
| Kernel size | 5 |
| Layer sizes | 32 → 64 → 128 → 64 → 32 → 40 |
| Batch size | 64 |
| Learning rate | 1×10⁻³ |
| Weight decay | 1×10⁻³ |
| Dropout | 0.3 |
| Optimizer | Adam |
| Max epochs | 1000 |
| Early stopping | 200-epoch moving average |
| Train/val/test split | 50/25/25 |
| Device | Apple MPS (configurable) |

The normalization statistics (per-channel mean and std) are computed over the training set and saved to disk, ensuring inference uses the same normalization as training.

### 5.2 Secondary Model: Source Term CNN (`models/conv_nn/cnn.py`)

This model predicts the raw subgrid source terms $S_\phi$ directly, rather than the PDF. It is similar in architecture to `pdf_cnn.py` but differs in:

- **Input**: 6 channels — `[ρ, T, u_x, u_y, s, fmcl]`
- **Output**: 1 channel — $S_{f_\text{mcl}}$ (fmcl source term)
- **Loss**: MSE (for a single regression target)
- **Output normalization**: z-score (mean/std saved separately)
- **Resolution**: `(512, 512)` with `downsample=8`, giving `64×64` CG cells
- **No thresholded softmax** — raw continuous regression output

### 5.3 Feedforward Neural Network (`models/feedforward_nn/fnn.py`)

The FNN operates on individual CG grid points (pixels) rather than spatial fields. It uses point-wise features including gradients and Hessians to provide spatial context:

**Feature engineering per pixel:**
- 8 coarse-grained fields: `[ρ, T, P, u_x, u_y, e_int, s, fmcl]`
- 16 first-order gradients: `∂φ/∂x`, `∂φ/∂y` for each field
- 24 second-order Hessian components: `∂²φ/∂x²`, `∂²φ/∂x∂y`, `∂²φ/∂y²` for each field
- Total: 48 scalar features per cell

**Architecture:**

```
Input: 48 scalar features
FC(48 → 256) → BN → ReLU → Dropout(0.4)
FC(256 → 128) → BN → ReLU → Dropout(0.4)
FC(128 → 64) → BN → ReLU → Dropout(0.4)
FC(64 → 1)
Output: scalar S_fmcl
```

**Class-balanced sampling**: Because `fmcl` values are sparse (most cells are either pure hot or pure cold), the training data is resampled to have equal counts from 10 bins of `fmcl` values, using square-root-spaced bin edges. This prevents the network from predicting only the majority class.

**Loss**: Huber loss (robust to outliers, important for heavy-tailed source term distributions).

### 5.4 ConvLSTM Model (`models/conv_lstm/clstm.py`)

A temporal model that processes sequences of simulation snapshots. The architecture:

```python
# Input: (batch, seq_len, channels, height, width)
# For each timestep t:
for each ConvLSTMCell layer:
    combined = cat([x_t, h_prev], dim=1)       # spatial + hidden state
    gates = Conv2d(combined)                    # i, f, o, g gates
    c_next = f*c_prev + i*g                    # cell state update
    h_next = o * tanh(c_next)                  # hidden state
output_t = Conv2d_1x1(h_last_layer)
```

This model is noted as "not tested" in the source code and represents an experimental branch for capturing temporal correlations between snapshots. A ConvLSTM could learn how the Temperature PDF evolves over time, but it introduces significantly higher computational cost at inference time.

### 5.5 Model Variants in `conv_nn/`

The `conv_nn/` directory contains a family of related CNN variants:

| File | Target | Notes |
|---|---|---|
| `cnn.py` | `S_fmcl` (single scalar) | MSE loss, 6 inputs |
| `all_cnn.py` | All 5 source terms | Multi-output (5 channels) |
| `flux_cnn.py` | Subgrid fluxes | Predicts closure fluxes instead of source terms |
| `all_flux_cnn.py` | All 12 flux components | Full flux tensor closure |
| `indiv_cnn.py` | One source term | Individual training per component |
| `log_cnn.py` | Source terms (log space) | Log-transformed training for better dynamic range |
| `pdf_cnn.py` | Temperature PDF | **Primary production model** |

---

## 6. The C++/Python Bridge: pybind11 Integration

### 6.1 How pybind11 Works

**pybind11** is a header-only C++11 library that exposes C++ classes and functions to Python, and vice versa. In this project, it is used in "embedding" mode — the C++ simulation code creates a Python interpreter *inside the C++ process* and calls Python functions directly.

This avoids IPC (inter-process communication) and file-based data exchange, enabling low-latency coupling between the simulation and the neural network.

### 6.2 Build Configuration

pybind11 is added to the CMake build as a subdirectory:

```cmake
# athenak/CMakeLists.txt
add_subdirectory(external/pybind11)
```

The `external/pybind11` directory is a git submodule. When `-DPROBLEM=subgrid` is specified, `subgrid.cpp` is compiled and linked with pybind11, which in turn requires linking against the CPython interpreter shared library.

The relevant headers included in [`subgrid.cpp`](../athenak/src/pgen/subgrid.cpp):

```cpp
#include <pybind11/embed.h>   // py::scoped_interpreter — embed Python in C++
#include <pybind11/numpy.h>   // py::array_t<double>    — zero-copy NumPy array bridge
namespace py = pybind11;
using namespace py::literals;  // for _a literal (keyword arguments)
```

### 6.3 Interpreter Lifetime Management

The Python interpreter is initialized **lazily** — only on the first call to `UserSourceTerm` (typically the first simulation timestep after time zero):

```cpp
// Global persistent pointers (namespace-scoped)
py::scoped_interpreter *pguard    = nullptr;
py::object             *psource_func = nullptr;

void UserSourceTerm(Mesh *pm, const Real bdt) {
    if (pguard == nullptr) {
        pguard       = new py::scoped_interpreter();  // Start Python runtime
        psource_func = new py::object(
            py::module_::import("source_module").attr("source_func")
        );                                            // Import and cache function
    }
    // ... rest of function body ...
}
```

**Why lazy initialization?**
- Avoids Python startup overhead if the source term is never called (e.g., during a restart where `time <= 0`)
- The Python interpreter must be initialized exactly once per process; the `scoped_interpreter` RAII guard prevents double-initialization

**Why heap allocation (via `new`)?**
- `py::scoped_interpreter` destroys the Python runtime in its destructor. By using a raw pointer, the interpreter lifetime is decoupled from `UserSourceTerm`'s stack frame, persisting across all subsequent timestep calls.

**Cleanup**: A finalization callback `SubgridFinalize` is registered as `pgen_final_func` and called by AthenaK at simulation end:

```cpp
void SubgridFinalize(ParameterInput *pin, Mesh *pm) {
    if (psource_func != nullptr) { delete psource_func; psource_func = nullptr; }
    if (pguard != nullptr)       { delete pguard;       pguard = nullptr; }
}
```

### 6.4 The `source_module` Python Module

The C++ code imports a Python module named `source_module` at runtime. This module must be importable from Python's `sys.path` at the time the simulation runs. It defines the function `source_func(dens, press, vx, vy, tracer, fmclrho)` which is the runtime inference entry point.

The `source_module.py` file is expected to:
1. Load the trained CNN weights and normalization statistics from disk
2. Accept 6 NumPy arrays as inputs (flattened mesh fields)
3. Run CNN inference in PyTorch
4. Return a 2D NumPy array of shape `(5, N)` containing source terms

---

## 7. The Subgrid Problem Generator (Runtime Interface)

### 7.1 Problem Initialization (`UserProblem`)

[`athenak/src/pgen/subgrid.cpp`](../athenak/src/pgen/subgrid.cpp) sets up the KHI initial conditions identically to the HR simulation, but registers additional callback functions:

```cpp
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
    user_srcs_func  = UserSourceTerm;  // Called every timestep (stage)
    user_bcs_func   = constant_bcs;   // Called every timestep for boundary conditions
    pgen_final_func = SubgridFinalize; // Called at simulation end
    
    // Read parameters from .athinput
    // Initialize KHI: tanh density/velocity profiles + multi-mode perturbation
    // Convert primitives to conserved
}
```

The KHI initialization (`iprob == 1`) uses smooth hyperbolic tangent profiles:

```cpp
dens = rho0 - rho1 * tanh((x2v - y_cold) / a_char);
vx   = vshear_half + vshear_delta * tanh((x2v - y_cold) / a_char);
// Multi-mode perturbation in vy (wavenumbers k=5,10,18,25,32):
perturb = sin(2π·5·x/L) + sin(2π·10·x/L) + sin(2π·18·x/L) + sin(2π·25·x/L) + sin(2π·32·x/L)
vy = -amp · 2·vshear_delta · perturb · exp(-(x2-y_cold)²/σ²)
```

### 7.2 Boundary Conditions (`constant_bcs`)

The Y-direction boundaries are "user" boundaries that maintain constant inflow conditions:

- **Bottom (inner_x2)**: Cold phase ($\rho = \rho_\text{cold}$, $P/\rho = P/\rho_\text{cold}$, $v_x = v_{x,\text{cold}}$)
- **Top (outer_x2)**: Hot phase ($\rho = \rho_\text{hot}$, $P/\rho = P/\rho_\text{hot}$, $v_x = v_{x,\text{hot}}$)
- Normal velocity ($v_y$): outflow (extrapolated from interior)
- Scalars: $s_\text{cold} = y_0 + y_1 = 1.0$, $s_\text{hot} = y_0 - y_1 = 0.0$

### 7.3 Per-Timestep Source Term Application (`UserSourceTerm`)

This is the core runtime coupling function. It is called by AthenaK **once per Runge-Kutta stage per timestep**, receiving the current time and the substep time $\Delta t_\text{sub}$. The complete data flow:

#### Step 1: Allocate Kokkos Device Views

```cpp
int Ni = ie - is + 1;   // number of active cells in x
int Nj = je - js + 1;   // number of active cells in y
int N2D = Ni * Nj * nmb;

Kokkos::View<double*> dens_d  ("dens_d",   N2D);
Kokkos::View<double*> press_d ("press_d",  N2D);
Kokkos::View<double*> vx_d    ("vx_d",     N2D);
Kokkos::View<double*> vy_d    ("vy_d",     N2D);
Kokkos::View<double*> tracer_d("tracer_d", N2D);
Kokkos::View<double*> fmclrho_d("fmclrho_d", N2D);
```

These are flat 1D arrays sized `(nmb × Ni × Nj)`.

#### Step 2: Extract Fields via Kokkos Parallel Kernel

```cpp
par_for("FlattenSnapshot", DevExeSpace(), 0, nmb-1, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int j, int i) {
        int idx = m*(Nj*Ni) + (i-is)*Nj + (j-js);
        dens_d  (idx) = w0(m, IDN, ks, j, i);
        press_d (idx) = w0(m, IEN, ks, j, i) * gm1;  // internal energy → pressure
        vx_d    (idx) = w0(m, IVX, ks, j, i);
        vy_d    (idx) = w0(m, IVY, ks, j, i);
        tracer_d(idx) = w0(m, tracer_index, ks, j, i);
        fmclrho_d(idx)= w0(m, frho_index, ks, j, i);
    });
```

The index mapping `idx = m*(Nj*Ni) + (i-is)*Nj + (j-js)` lays out the data as `[meshblock][x_index][y_index]`, creating a memory-contiguous 2D map: shape `(nmb*Ni, Nj)`.

Note: for 2D simulations, `ks = indcs.ks` (the single k-index), so the full 3D array is effectively sliced at k=ks.

#### Step 3: Device → Host Memory Copy

```cpp
auto dens_h    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dens_d);
auto press_h   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), press_d);
auto vx_h      = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vx_d);
auto vy_h      = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vy_d);
auto tracer_h  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), tracer_d);
auto fmclrho_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fmclrho_d);
```

`create_mirror_view_and_copy` performs a device-to-host deep copy. On CPU-only builds, this is a no-op. On GPU builds, this is a CUDA/HIP `cudaMemcpy`.

#### Step 4: Wrap as NumPy Arrays and Call Python

```cpp
py::array_t<double> S_arr = (*psource_func)(
    py::array_t<double>({nmb*Ni, Nj}, dens_h.data()),
    py::array_t<double>({nmb*Ni, Nj}, press_h.data()),
    py::array_t<double>({nmb*Ni, Nj}, vx_h.data()),
    py::array_t<double>({nmb*Ni, Nj}, vy_h.data()),
    py::array_t<double>({nmb*Ni, Nj}, tracer_h.data()),
    py::array_t<double>({nmb*Ni, Nj}, fmclrho_h.data())
);
```

`py::array_t<double>({nmb*Ni, Nj}, ptr)` creates a NumPy array that **directly references the existing host memory** — no data copy occurs. Python receives a 2D array of shape `(nmb*Ni, Nj)` for each field.

The Python `source_func` is called synchronously and returns `S_arr` — a NumPy array of shape `(5, N2D)` containing the source terms for all 5 conserved quantities at every cell.

#### Step 5: Apply Source Terms via Kokkos Kernel

```cpp
auto S_buf = S_arr.unchecked<2>();   // Fast unchecked accessor for 2D array

par_for("ApplySrc", DevExeSpace(), 0, nmb-1, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int j, int i) {
        int idx = m*(Nj*Ni) + (i-is)*Nj + (j-js);
        
        // Density update (clamped positive)
        u0(m, IDN, ks, j, i) = fmax(0.0, u0(m, IDN, ks, j, i) + bdt * S_buf(0, idx));
        
        // Momentum updates (unconstrained)
        u0(m, IM1, ks, j, i) += bdt * S_buf(1, idx);
        u0(m, IM2, ks, j, i) += bdt * S_buf(2, idx);
        
        // Energy update
        u0(m, IEN, ks, j, i) += bdt * S_buf(3, idx);
        
        // fmcl density update (clamped positive)
        u0(m, frho_index, ks, j, i) = fmax(0.0, u0(m, frho_index, ks, j, i) + bdt * S_buf(4, idx));
    });
```

This is an explicit Forward Euler integration of the source terms into the conserved variable update:
$$U^{n+1} = U^n + \Delta t \cdot S(U^n)$$

The `fmax(0.0, ...)` clamping prevents nonphysical negative densities from accumulating. Note that `S_buf` is accessed from the host-side Python return value — this means the `par_for` kernel runs on the host if the data never migrates back to device. On a pure-CPU build (typical for development), this is seamless.

---

## 8. The End-to-End Pipeline Orchestration

### 8.1 The `run_pipeline.sh` Script

The complete workflow from data generation to benchmark comparison is orchestrated by [`shell_scripts/run_pipeline.sh`](../shell_scripts/run_pipeline.sh). This script runs six sequential steps:

```
Step 1: HR Simulation        — Generates training data (SKIPPED if outputs exist)
Step 2: Train PDF CNN        — python3 models/conv_nn/pdf_cnn.py
Step 3: Validate model       — python3 data/mocks/pdf_plot.py
Step 4: LR Simulation        — Pure hydro at 32× lower resolution
Step 5: SG Simulation        — Hydro + NN source terms (pybind11 coupling)
Step 6: Benchmark comparison — python3 data/mocks/mock_sg.py
```

The script:
- Creates a timestamped run directory `runs/run_YYYYMMDD_HHMMSS/`
- Tees all output to per-step log files
- Uses `set -euo pipefail` to exit immediately on any failure
- Caches a copy of every `.athinput` used in `runs/<timestamp>/athinputs/`

### 8.2 Athinput Generation

Simulation parameters are centrally managed in [`shell_scripts/config.json`](../shell_scripts/config.json). The script [`shell_scripts/gen_athinput.py`](../shell_scripts/gen_athinput.py) parses this JSON and generates `.athinput` files:

**Three simulation modes:**

| Mode | `nx1 × nx2` | Meshblock | `nscalars` | `ism_cooling` | `user_srcs` |
|---|---|---|---|---|---|
| HR | 256 × 512 | 32 × 512 | 1 | `true` | `false` |
| LR | 8 × 16 | 8 × 16 | 1 | `true` | `false` |
| SG | 8 × 16 | 4 × 16 | 2 | `false` | `true` |

Key differences for the SG simulation:
- **`nscalars = 2`**: The SG run tracks both the standard tracer (`s_00`) and the cold mass fraction density `frho` (`s_01`)
- **`ism_cooling = false`**: The built-in ISM cooling is disabled; cooling is handled by the NN source term
- **`user_srcs = true`**: Enables the `UserSourceTerm` callback in `subgrid.cpp`

---

## 9. Configuration System and Simulation Parameters

### 9.1 Physical Domain and Units

The simulation domain is defined in code units. The conversion factors (from `config.json`) are:

| Quantity | Code value | CGS value | Physical meaning |
|---|---|---|---|
| Length unit | 1 | `3.08568 × 10¹⁸` cm | ~1 pc |
| Mass unit | 1 | `4.91417 × 10³¹` g | |
| Time unit | 1 | `3.15576 × 10¹³` s | ~1 Myr |

The domain extends `[-5, +5]` in x (10 pc) and `[-10, +10]` in y (20 pc). The shear interface is at `y=0` (cold_frac=0.5).

### 9.2 Shear Velocities

```
vx_hot  = 28.2653  code units  ≈ +275 km/s
vx_cold = -2.8265  code units  ≈ -27.5 km/s
Relative shear = vx_hot - vx_cold ≈ 302 km/s
```

These velocities produce a Mach number $\mathcal{M} \sim 1$ for the instability, typical of CGM conditions.

### 9.3 Equation of State and Numerics

```
gamma    = 1.666667   (5/3, monatomic ideal gas)
integrator = rk2      (2nd-order Runge-Kutta)
reconstruct = plm     (Piecewise Linear Method)
rsolver = hllc        (HLLC Riemann solver)
cfl_number = 0.4      (Courant-Friedrichs-Lewy number)
```

---

## 10. Build System and Compilation

### 10.1 CMake Configuration

Building the SG simulation requires enabling pybind11:

```bash
cd athenak
mkdir sg_build && cd sg_build
cmake -S .. -B . \
  -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_CUDA=OFF \
  -DPROBLEM=subgrid         # Selects subgrid.cpp as the problem generator
cmake --build . -j$(nproc)
```

CMake automatically includes pybind11 headers and links the CPython library when `subgrid.cpp` is compiled, because `CMakeLists.txt` has:

```cmake
add_subdirectory(external/pybind11)
# ...
target_link_libraries(athena PUBLIC pybind11::embed)  # implicitly from pybind11
```

### 10.2 Python Environment Requirements

The simulation binary requires the Python environment to be fully configured with all dependencies:

```bash
pip install numpy scipy matplotlib h5py tqdm torch scikit-image pybind11
# or:
pip install -r requirements.txt
```

The `PYTHONPATH` environment variable must include the project root so that `source_module` is importable:

```bash
export PYTHONPATH="/path/to/SubgridCGMModel:${PYTHONPATH}"
```

This is handled automatically by `run_pipeline.sh`:

```bash
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/data:${PYTHONPATH:-}"
```

### 10.3 Model Weight Availability

Before running the SG simulation, the trained model weights and normalization arrays must exist on disk. These are looked up at runtime in `MODEL_SAVES_DIR` (the `MODEL_SAVES_DIR` environment variable, or `outputs/model_saves/` by default):

```
cnn_(1024, 512)_64.pth           — PyTorch model state dict
cnn_(1024, 512)_64_input_mean.npy — Channel-wise input means, shape (1,5,1,1)
cnn_(1024, 512)_64_input_std.npy  — Channel-wise input std devs, shape (1,5,1,1)
```

If these files are missing, the `source_module` will raise a `FileNotFoundError` inside the embedded Python interpreter, which will propagate as an unhandled exception and crash the simulation.

---

## 11. Data Flow Summary and Worked Example

### 11.1 Per-Timestep Data Flow (Runtime)

Here is a concrete worked example of what happens at a single simulation timestep in the SG simulation with mesh `8×16` and 1 MeshBlock (`nmb=1`):

```
Simulation state at time t:
  Active cells: Ni=8, Nj=16, N2D = 1*8*16 = 128

Step 1 — Flatten to device views:
  dens_d   : shape (128,)   — density at each of 128 cells
  press_d  : shape (128,)   — pressure = eint * (γ-1)
  vx_d     : shape (128,)   — x-velocity
  vy_d     : shape (128,)   — y-velocity
  tracer_d : shape (128,)   — scalar tracer (s_00)
  fmclrho_d: shape (128,)   — cold mass fraction density (s_01)

Step 2 — Copy to host:
  dens_h, press_h, vx_h, vy_h, tracer_h, fmclrho_h : shape (128,) on CPU

Step 3 — Wrap as NumPy, reshape to (8, 16):
  6 arrays of shape (8, 16) passed to Python's source_func(...)

Step 4 — Inside Python (source_module.py):
  a) Compute Temperature: T = P/ρ × (unit conversion)
  b) Stack into input tensor: (1, 5, 8, 16)  [batch=1, channels=5, nx=8, ny=16]
  c) Normalize: (x - input_mean) / input_std
  d) CNN forward pass:
       encoder: (1,5,8,16) → (1,32,8,16) → (1,64,8,16) → (1,128,8,16)
       decoder: (1,128,8,16) → (1,64,8,16) → (1,32,8,16) → (1,40,8,16)  [logits]
  e) ThresholdedSoftmax: (1,40,8,16) → probability distribution
  f) Emissivity: ε = ρ² × Σ_i p_i · Λ(T_i)
  g) Format source terms: shape (5, 128)  
     S[0] = d(ρ)/dt     from ε
     S[1] = d(ρu_x)/dt  
     S[2] = d(ρu_y)/dt  
     S[3] = d(E)/dt     from ε = radiative cooling rate
     S[4] = d(f_mcl·ρ)/dt

Step 5 — Apply in C++:
  For each cell idx in [0, 127]:
    u0[IDN]        += bdt * S[0, idx]
    u0[IM1]        += bdt * S[1, idx]
    u0[IM2]        += bdt * S[2, idx]
    u0[IEN]        += bdt * S[3, idx]
    u0[frho_index] += bdt * S[4, idx]
```

### 11.2 Training Data Generation

```
HR simulation: 256×512 grid, 500+ snapshots saved every Δt=0.01 code time
  → KH.hydro_w.00501.bin through KH.hydro_w.NNNNN.bin

data_preprocess.simulation_data.input_data():
  → rho[n_snaps, 1024, 512], temp, pressure, ux, uy, eint, ps

coarse_grain(field, downsample=64):
  → block_reduce((64,64), np.mean)
  → cg_field[n_snaps, 16, 8]

calc_pixel_pdf(bins=40):
  → for each (t, j, k): histogram temp[t, j*64:(j+1)*64, k*64:(k+1)*64] into 40 bins
  → pdf[n_snaps, 40, 16, 8]

Training:
  input_tensor[n_snaps, 5, 16, 8] = stack([cg_rho, cg_temp, cg_ux, cg_uy, cg_ps])
  output_tensor[n_snaps, 40, 16, 8] = pdf (temperature probability distributions)
```

---

## 12. Key Design Decisions and Trade-offs

### 12.1 Embedded Python vs. External Process

Choosing to embed Python inside the AthenaK process (via pybind11) rather than a microservice or pipe-based IPC has significant consequences:

| Aspect | Embedded Python (current) | External Process |
|---|---|---|
| Latency | Very low (function call) | High (IPC, serialization) |
| Memory sharing | Zero-copy (pointer sharing) | Copy required |
| Complexity | Moderate (link deps) | Higher (IPC, sync) |
| GPU isolation | Cannot use PyTorch GPU independently from Kokkos | Can use separate GPU streams |
| Thread safety | Single Python interpreter (GIL) | Fully independent |

For the scale of this project (2D, small grids), the embedded approach is clearly superior. For large 3D MPI runs, the GIL constraint could become a bottleneck.

### 12.2 PDF vs. Direct Source Term Prediction

The PDF-based approach offers significant physical advantages over direct source term prediction:

- **Interpretability**: The predicted PDF is physically meaningful and can be inspected
- **Physical constraints**: The thresholded softmax enforces $\sum p_i = 1$, $p_i \geq 0$
- **Generalizability**: The same PDF can be used to compute any cooling-related quantity (emissivity, heating rates, etc.)
- **Richer loss function**: The emissivity loss directly penalizes physically relevant errors

The trade-off is that the PDF CNN has a larger output space (40 bins per cell vs. 1 scalar), requiring more training data and compute.

### 12.3 Memory Layout

The flattening scheme `idx = m*(Nj*Ni) + (i-is)*Nj + (j-js)` creates a layout where the **y-index is contiguous** in memory (inner loop). When this is reshaped to `(nmb*Ni, Nj)`, it becomes a 2D array where rows are x-positions (across all mesh blocks) and columns are y-positions.

The CNN processes this as a spatial image in `(channels, x, y)` format, consistent with PyTorch's `NCHW` convention.

### 12.4 Normalization Strategy

Input normalization uses **global statistics** computed over the entire training dataset (not per-batch):
- Mean: `input_tensor.mean(dim=(0,2,3), keepdim=True)` — per-channel, averaged over time and space
- Std: `input_tensor.std(dim=(0,2,3), keepdim=True)` — per-channel

These are saved as `.npy` files and loaded at inference time by the embedded Python interpreter. This ensures that the simulation receives inputs in the same normalized space the network was trained on, regardless of how simulation conditions evolve over time.

### 12.5 Conserved vs. Primitive Variables at the Interface

The data extraction in `UserSourceTerm` reads from **primitive variables** (`w0`, via `IDN`, `IEN`, `IVX`, `IVY`) but applies source terms to **conserved variables** (`u0`, via `IDN`, `IM1`, `IM2`, `IEN`). This is standard practice:

- Primitives are stable and physically interpretable for the NN (density is positive, pressure is positive)
- Source terms must be applied to conserved variables to maintain conservation laws
- AthenaK handles the `ConsToPrim` / `PrimToCons` conversions at appropriate points in the timestep

### 12.6 Caching Strategy for Preprocessed Data

The data preprocessing pipeline uses a **file-based cache** keyed by `(resolution, downsample)`:

```python
folder_path = os.path.join(CACHE_PATH, f"sc{resolution}_{downsample}")
if os.path.exists(folder_path):
    # Load from cache (fast)
else:
    # Compute from raw .bin files (slow, ~hours for large simulations)
    # Save to cache
```

This is crucial because computing pixel PDFs via nested loops over timesteps, CG cells, and pixels is an O(N_t × N_x × N_y × ds²) operation, which is dominated by Python loop overhead. The cache avoids recomputation across training runs.

---

## Appendix A: File Reference

| File | Purpose |
|---|---|
| [`athenak/src/pgen/subgrid.cpp`](../athenak/src/pgen/subgrid.cpp) | Main C++/Python interface; problem generator for KHI + NN source terms |
| [`athenak/src/srcterms/ismcooling.hpp`](../athenak/src/srcterms/ismcooling.hpp) | SPEX ISM radiative cooling function (C++, GPU-compatible) |
| [`athenak/CMakeLists.txt`](../athenak/CMakeLists.txt) | Build system; pybind11 inclusion and Kokkos configuration |
| [`data/data_preprocess.py`](../data/data_preprocess.py) | Offline data processing: binary parsing, coarse-graining, PDF computation, source terms |
| [`data/bin_convert.py`](../data/bin_convert.py) | AthenaK binary file parser; extracts named variable arrays |
| [`models/conv_nn/pdf_cnn.py`](../models/conv_nn/pdf_cnn.py) | Primary CNN model: Temperature PDF prediction with emissivity loss |
| [`models/conv_nn/cnn.py`](../models/conv_nn/cnn.py) | CNN for direct fmcl source term prediction |
| [`models/feedforward_nn/fnn.py`](../models/feedforward_nn/fnn.py) | FNN with gradient/Hessian features; pixel-wise source term regression |
| [`models/conv_lstm/clstm.py`](../models/conv_lstm/clstm.py) | Temporal ConvLSTM (experimental) |
| [`shell_scripts/run_pipeline.sh`](../shell_scripts/run_pipeline.sh) | Full pipeline orchestration: HR → train → validate → LR → SG → benchmark |
| [`shell_scripts/config.json`](../shell_scripts/config.json) | Central simulation configuration (resolutions, physics parameters) |
| [`shell_scripts/gen_athinput.py`](../shell_scripts/gen_athinput.py) | Generates `.athinput` files from `config.json` |

---

## Appendix B: Glossary

| Term | Definition |
|---|---|
| **AthenaK** | High-performance C++/Kokkos astrophysical MHD code |
| **CGM** | Circumgalactic Medium — diffuse gas halo surrounding galaxies |
| **CG** | Coarse-grained — spatially block-averaged version of a HR field |
| **CNN** | Convolutional Neural Network |
| **fmcl** | Fractional Mass of Cold Gas — fraction of cell mass below $T_\text{cutoff}=10^5$ K |
| **HR / LR / SG** | High-Resolution / Low-Resolution / SubGrid simulations |
| **KHI** | Kelvin-Helmholtz Instability — shear-driven turbulent mixing |
| **Kokkos** | C++ performance portability library for GPU/CPU parallel computing |
| **MHD** | Magnetohydrodynamics |
| **MeshBlock** | Tile of the computational domain; the unit of parallelism in AthenaK |
| **pybind11** | C++11 header library for embedding Python and calling Python from C++ |
| **PDF** | Probability Density Function — here, distribution of temperatures within a CG cell |
| **PLM** | Piecewise Linear Method — 2nd-order spatial reconstruction scheme |
| **RK2** | 2nd-order Runge-Kutta time integrator |
| **SPEX** | Solar abundance plasma emission code; source of tabulated cooling data |
| **HLLC** | Harten-Lax-van Leer-Contact — approximate Riemann solver |
| **Λ(T)** | Cooling function — energy radiated per unit volume per unit density squared |
| **ε** | Emissivity = ρ² × ⟨Λ(T)⟩_PDF — radiative cooling rate per unit volume |
