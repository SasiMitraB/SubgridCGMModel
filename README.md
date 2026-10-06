# Subgrid CGM Model

A neural-network subgrid model for radiative cooling in multiphase gas.

At coarse resolution, a cell containing both hot (~10⁶ K) and cold (~10⁴ K) gas
is assigned a single mean temperature, and cooling computed from that mean is
badly wrong. This project trains a CNN to predict the **sub-cell temperature PDF**
of each coarse cell from the resolved fields. It then integrates the cooling
function over that PDF (isobaric assumption) and feeds the resulting cooling rate
back into a low-resolution AthenaK run as a source term.

The test problem is a 2D Kelvin–Helmholtz shear layer with radiative cooling.

```
HR AthenaK run ──► coarse-grain ──► train PDF CNN ──► LR AthenaK run with the CNN
(kh_radiative_cooling)  (data/)     (models/conv_nn/)   (subgrid, via embedded Python)
                                                         compared against plain LR
                                                         (ISM cooling) and the HR truth
```

> **Status (Sep 2026):** the repo is being reorganised. Paths and layout below
> describe the code *as it is now*; see [Known issues](#known-issues).

---

## Repository layout

Only the parts that are in active use are listed.

| Path | What it is |
|---|---|
| `athenak/` | AthenaK (git submodule) with local additions: `src/pgen/kh_radiative_cooling.cpp` (HR problem), `src/pgen/subgrid.cpp` (calls the CNN via pybind11). See [CHANGES_TO_ATHENAK.md](CHANGES_TO_ATHENAK.md). |
| `ergane/` | Reader/analysis library for AthenaK binary output (git submodule). |
| `builds/` | CMake build directories, one per executable, plus `rebuild_all.sh`. |
| `builds/subgrid_model/src/source_module.py` | Python module AthenaK imports at runtime; coarse-grains, runs the CNN, returns cooling rates. |
| `models/conv_nn/pdf_cnn.py` | **The model.** Cooling function, data loading, `ConvNN`, losses and training (`__main__`). |
| `data/` | Preprocessing: `data_preprocess.py`, `coarse_grain_utils.py` (shared coarse-graining rules), `downsample_ic.py` (HR snapshot → LR initial condition). |
| `data/mocks/` | Evaluation/plotting scripts: `pdf_plot.py` (PDF benchmark), `mock_sg.py` (subgrid vs LR vs HR diagnostics). |
| `explore_data/` | One-off exploratory and diagnostic scripts. `mock_sg_tiled.py` is used by the tiled pipeline. |
| `shell_scripts/` | Pipeline drivers, `config.json` (simulation parameters), `gen_athinput.py`. |
| `plotting_for_paper/` | Quarto/LaTeX sources for talks and notes. |
| `docs/` | Longer technical notes. Parts are out of date (they predate the current loss and pipeline). |

Generated data (not meant for git): `simulation_outputs/` (AthenaK runs,
very large), `runs/` (one directory per pipeline run), `tiled_outputs/`,
`outputs/` (default model saves and plots), `mocks/`.

---

## Setup

### 1. Clone with submodules

```bash
git clone --recursive git@github.com:SasiMitraB/SubgridCGMModel.git
# or, in an existing clone:
git submodule update --init --recursive
```

### 2. Python environment

Development uses the base miniconda Python (Python 3, PyTorch 2.11, CUDA on an
RTX 4000 Ada). `requirements.txt` is a full `pip freeze`, so treat it as a
record rather than a minimal spec. The essentials are:

```bash
pip install torch numpy scipy matplotlib h5py tqdm scikit-image pybind11 optuna
pip install -e ergane
```

The **subgrid build embeds whichever `python3` is on `PATH` at build time**.
That interpreter must be able to `import torch`, so build and run from the
same environment.

### 3. Build AthenaK

```bash
mkdir -p builds/{hr_build,hr_build_mpi,hr_build_gpu,subgrid_model}   # first time only
bash builds/rebuild_all.sh                  # all four
bash builds/rebuild_all.sh --subgrid-only   # after changing subgrid.cpp
```

| Build | Problem | Notes |
|---|---|---|
| `hr_build` | `kh_radiative_cooling` | Serial CPU. Also used for the plain LR (ISM cooling) runs. |
| `hr_build_mpi` | `kh_radiative_cooling` | Uses `/usr/bin/mpic++`. |
| `hr_build_gpu` | `kh_radiative_cooling` | CUDA, `Kokkos_ARCH_ADA89`. |
| `subgrid_model` | `subgrid` | Links pybind11 and embeds Python. |

Executables end up at `builds/<name>/src/athena`.

---

## Running

### Main pipeline: `shell_scripts/run_pipeline.sh`

Training data is an existing HR run: `simulation_outputs/hr_build_512/bin/`.
One call runs every step:

1. Train the PDF CNN: `models/conv_nn/pdf_cnn.py`.
2. Benchmark predicted PDFs and cooling against the truth: `data/mocks/pdf_plot.py`.
3. Downsample HR snapshot 500 into an LR initial condition: `data/downsample_ic.py`.
4. Run LR with plain ISM cooling from that IC (`hr_build`).
5. Run LR with the CNN subgrid cooling from the same IC (`subgrid_model`).
6. Make diagnostic plots comparing HR, LR and subgrid: `data/mocks/mock_sg.py`.
7. Plot the timestep history.

```bash
bash shell_scripts/run_pipeline.sh
SIM_TLIM=10 bash shell_scripts/run_pipeline.sh   # example override
```

Grid sizes come from `shell_scripts/config.json` (`hr.nx1/nx2` and
`lr.downsample_factor`). Each run writes a self-contained directory:

```
runs/run_<YYYYmmdd_HHMMSS>/
  pipeline.log   logs/step*.log   athinputs/
  model_saves/   loss_plots/      pdf_mocks/   sg_mocks/
```

The LR and subgrid simulation outputs go to `simulation_outputs/lr_build_ism/`
and `simulation_outputs/subgrid_model/`. These are overwritten on every run.

### Other drivers

| Script | Purpose |
|---|---|
| `shell_scripts/run_variable_box_subgrid.sh` | Arbitrary box size (`LX`, `LY`, `CELL_SIZE` in pc). Runs HR, downsamples, then runs subgrid and LR with the CNN tiled over the domain. Writes to `tiled_outputs/`. |
| `shell_scripts/random_subsample_pipeline.sh` | Same as the main pipeline, but trains on random crops across snapshots (`random_snapshot_training.py`). |
| `shell_scripts/run_plots.sh` | Re-runs the diagnostic plots for an existing `runs/` directory. |
| `shell_scripts/run_hyperparam_tuning.sh`, `optimize_subgrid.py` | Optuna search over loss weights, scored by downstream subgrid runs. |
| `shell_scripts/run_resolution_sweep.sh`, `run_box_sweep.sh`, `run_parameter_sweep.sh`, `run_gpu_parameter_sweep_*.sh` | HR simulation sweeps over resolution, box size, or shear velocity × cold fraction. |
| `regenerate_cache.py` | Rebuilds coarse-grained caches after changing `coarse_grain_utils.py`. |

### Training on its own

```bash
export SUBGRID_DATA_PATH=simulation_outputs/hr_build_512/bin
export SUBGRID_CACHE_PATH=simulation_outputs/hr_build_512/cache
export PDF_CNN_RESOLUTION=512,256 PDF_CNN_DOWNSAMPLE=32   # what run_pipeline.sh derives from config.json
python models/conv_nn/pdf_cnn.py --num_epochs 200 --model_save_dir /tmp/models
python models/conv_nn/pdf_cnn.py --help   # all loss weights and training options
```

The weights are saved as `cnn_(<res>)_<ds>.pth`, together with
`*_input_mean.npy` and `*_input_std.npy` normalisation files. The default
directory is `outputs/model_saves/pdf_model_saves/`, or `MODEL_SAVES_DIR` if
set. `source_module.py` loads from the same directory.

---

## The model in brief

- **Inputs**: 5 coarse fields (density, temperature, vₓ, v_y, passive scalar),
  augmented inside the network with 8 mixing-layer features (vorticity,
  temperature and density gradients and their alignment, strain rate, and a
  local variance proxy).
- **Output**: a PDF over log T bins in each coarse cell. A learned gate
  g ∈ [0, 1] blends a sharp peak at the cell temperature (single-phase cells)
  with a broad PDF (mixing cells). Pure hot or cold cells must give exactly
  zero cooling.
- **Training** has two stages: the gate is pretrained first, then the PDF
  branch is trained with the gate frozen.
- **Loss** (`GatedPDFLoss`) has five terms:
  - zoned Wasserstein-1, weighted separately inside and outside the active
    cooling window (log T ∈ [4.1, 5.9] by default)
  - gate BCE
  - mean-temperature consistency
  - log-emissivity MSE
  - leakage of mass out of the active window
- **At runtime** `source_module.py` applies the CNN cooling rate uncapped and
  returns `dt_cool = min(e_int / cooling rate)` as the cooling timestep.
  `subgrid.cpp` keeps `e_int` from dropping below 5% of its value in one stage.

---

## Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `SUBGRID_DATA_PATH`, `SUBGRID_CACHE_PATH` | training | HR `.bin` directory; coarse-grain cache directory. |
| `PDF_CNN_RESOLUTION`, `PDF_CNN_DOWNSAMPLE` | training, runtime | HR grid `"nx2,nx1"` and coarse-graining factor. Together they select the model file. |
| `PDF_CNN_ALPHA_*`, `PDF_CNN_GATE_*` | training | Override loss weights and gate pretraining settings. |
| `LOGT_ACTIVE_START`, `LOGT_ACTIVE_END` | training | Active cooling window in log T. |
| `MODEL_SAVES_DIR`, `LOSS_PLOTS_DIR` | training, runtime | Where models are written and read. |
| `CNN_TILING_MODE`, `TILE_ROWS`, `TILE_COLS`, `TILE_GRID` | runtime | How the CNN is tiled over a domain larger than its training grid. |
| `SIM_NX1`, `SIM_NX2`, `SIM_TLIM`, `HR_IC_SNAPSHOT` | pipeline | Override the LR grid, end time, and IC snapshot. |

`subgrid.cpp` loads the model with `import source_module`. The directory
containing `source_module.py` must therefore be on `PYTHONPATH`. The pipeline
scripts do this by running from `builds/subgrid_model/src/`.

---

## Known issues

These are being worked on:

- The pipeline scripts hardcode `PROJECT_ROOT=/home/sasi/Projects/SubgridCGMModel`,
  and about 30 scripts contain absolute `/home/...` paths. They will not run
  from another location until this is fixed.
- Modules find each other through `sys.path` edits and `PYTHONPATH`, not through
  an installed package.
- `ergane` is registered as a submodule twice (`ergane/` and `src/ergane/`).
  `data/bin_convert.py` is an older copy of `ergane`'s `bin_reader.py`.
- Everything in `models/` except `pdf_cnn.py` is legacy and unused by the
  pipelines: `cnn.py`, `flux_cnn.py`, `all_cnn.py`, `indiv_cnn.py`,
  `all_flux_cnn.py`, `log_cnn.py`, `naive_cnn.py`, `conv_lstm/` and
  `feedforward_nn/`.
- Some generated outputs (plots, videos, `.pth` files, simulation logs) are
  still tracked in git.
- There are no tests yet for the cooling or PDF code.
