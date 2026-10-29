# Plan: `subgrid_ml` pgen — ML subgrid R, Q, C in AthenaK on GPU

## Context

Chapters 1–2 of the notes (`theory/notes/Overleaf/chapters/01-derivation.tex`, `02-implementation.tex`) derive the coarse-grained Euler equations (eq-final). They contain three subgrid terms:
- a momentum flux tensor **R** inside ∇·;
- an energy flux **Q** = ũ_j R_ij + q inside ∇·;
- a cooling correction **C**.

Chapter 2 maps network outputs to these terms:

| Quantity | Map |
|---|---|
| f | σ(z0) |
| S | exp(A)/Tr exp(A) |
| w | √f·sinh\|z\|/\|z\|·z |
| C | g·n²[Λ_full e^{z_C} − Λ_cut] |
| flux gate g_f | f ≥ f_cut |

Trained JAX models exist for R (`models/rvv_tensor_model.py`), the cooling gate (`models/gate_model.py`) and z_C (`models/cooling_model.py`). No q/w model exists yet.

The existing `src/pgen/subgrid.cpp` is not a usable base:
- it is 2D-only;
- it calls Python through pybind and copies device→host every step;
- its lambdas are invalid on the GPU.

The goal is a new pgen that runs a coarse 3D radiative-KH LES (dx = 0.625 pc, the training coarse cell) with hydro **and** inference on the GPU.

**Decisions so far**
- **Q**: build the full q slot now. With no q model configured, q = 0, so Q = ũ·R. A future w model plugs in from the input file.
- **Inference backend**: the network structure is expected to change, possibly to something other than stencil-conv+MLP (U-Net, attention, …). Hand-written Kokkos layers would need rewriting for each new structure, so the network runs in **ONNX Runtime (CUDA execution provider)** instead.

## Design principle: AthenaK passes in raw fields and gets back raw outputs

Each model is exported from JAX as **one ONNX graph** that contains:
1. feature construction (logs, gradients, strain, vorticity),
2. normalisation,
3. the network.

The graph maps **raw coarse fields on a ghost-padded meshblock** to **unconstrained outputs z** on the interior.

AthenaK only:
- (a) packs fixed raw fields into a float32 device buffer;
- (b) runs the session;
- (c) applies the fixed physics maps from the notes (σ, expm, w map, cooling formula) and adds fluxes and sources.

What each kind of change costs:

| Change | Where it happens |
|---|---|
| Architecture, structure, features, normalisation, stencil | Re-export only. |
| New raw input field | One line in the C++ pack kernel. |
| Map or physics change | C++, as it would be under any backend. |

Each `.onnx` carries metadata (`metadata_props`) that AthenaK reads and checks at startup:
- `inputs` (a raw field list from a fixed vocabulary);
- `halo` (ghost cells consumed per side);
- `outputs` (z names);
- `kind`.

**One limitation.** Inference runs per meshblock, so the receptive field must be ≤ ng. A model that needs global context requires a single meshblock per rank (fine for 16×64×16) or a gather step. The pgen enforces `halo + face_extra ≤ ng`.

## Step −1 — save this plan

Copy this plan to the project root as `/home/sasi/Projects/SubgridCGMModel/PLAN_subgrid_ml.md` before any other step, and keep it updated as the decisions in Step 0 are made.

## Step 0 — toolchain spike (do first, it can change details below)

1. **Export path.** Install `onnx`, `onnxruntime-gpu` and `jax2onnx` into the venv. Export the current rvv net with a dynamic batch dimension and dynamic spatial dimensions. Check that ONNX matches JAX to 1e-5 in Python ORT.
   - If jax2onnx cannot handle jax 0.11, fall back to `jax2tf` + `tf2onnx`.
   - If that fails too, write a small exporter that emits the graph with `onnx.helper` from the pickle weights. This is feasible because all current models are Conv3d/MatMul/Gelu.
2. **CUDA pairing.**
   - AthenaK builds against CUDA 12.8 (`builds/build_gpu.sh`).
   - The venv has CUDA 13 libraries and cuDNN 9 (`nvidia_cudnn_cu13`). There is no system cuDNN.
   - The driver is 615.71 on an RTX 4000 Ada, which supports both.
   - Pick **one CUDA major** for Kokkos and ORT. The default is the ORT C/C++ GPU release built for CUDA 12, plus a cuDNN 9 cu12 tarball/wheel in `external/` (not the system). Do not mix cudart 12 and 13 in one process, because sharing streams and IO binding would break.
   - Also check that the bundled Kokkos version builds with that CUDA.
3. **Minimal C++ test.** Allocate a Kokkos CUDA view, wrap it with `Ort::Value::CreateTensor` on `MemoryInfo("Cuda")`, run with IoBinding, read the output back in a Kokkos kernel.

## Design

### 1. Physics formulation (no EOS override)

The conserved E is ⟨E⟩, which includes ρk_sg. Standard ConsToPrim then gives w0(IEN) = ρ·k_max (eq-kmax), so the solver's pressure is P̂ = (γ−1)ρk_max, while the true pressure is P = P̂(1−f). The difference is applied as **face-flux corrections**:

- **Momentum flux.** Add R_ij − (γ−1)ρk_sg δ_ij.
- **Energy flux.** Add Q_i − (γ−1)ρk_sg ũ_i, where Q_i = ũ_j R_ij + g_f q_i.
- **Temperature for cooling and model inputs.** T_CG = T̂(1−f) (eq-temperature).

Cell values are averaged arithmetically to faces.

The Riemann and CFL sound speeds use P̂ ≥ P. That is slightly more diffusive and safe, since f is about 10⁻² in the mixing layer.

### 2. Where it runs in the RK stage

**Task `SubgridML::Evaluate`** is inserted with `tl->InsertTask(&SubgridML::Evaluate, this, phydro->id.flux, phydro->id.sendf)`, following the precedent in `src/srcterms/turb_driver.cpp:300`. It:
- runs the models on `w0`, which is valid in the ghost zones;
- stores f, R, q, g, z_C, C in pgen-owned device arrays;
- adds the corrections into `uflx.x{1,2,3}f`, modelled on `diffusion/viscosity.cpp:96-160`. Doing this before SendFlux keeps SMR/AMR flux correction conservative.

**`user_srcs_func`** runs after RKUpdate on the same `w0` and the stored outputs. It:
- applies E −= bdt·[n²Λ_cut(T_CG) + C], with a floor on e_th = ρk_max(1−f);
- sets `psrc->dtnew = min(dt_cool, dt_Q)` through the fork's `user_cooling` path (`srcterms_newdt.cpp:26-31`).

**Input file**: `<hydro_srcterms> user_cooling=true`, with ism_cooling **off**. Also `<problem> user_srcs=true` and `<mesh> nghost=4`.

### 3. Model chain per stage

1. **R graph.**
   - Inputs: `rho, e (=ρk_max), ux, uy, uz, s0` on the block with its full halo.
   - Outputs: z0..z5 on the interior, extended by (ng − halo).
   - C++ map: f, S, R, k_sg, and T_CG = T̂(1−f).
2. **Gate graph.**
   - Inputs: `rho, T_cg, ux, uy, uz, s0`. Gradients are computed *inside* the graph.
   - Output: a logit, then g = [σ > 0.5].
3. **Cooling graph.**
   - Inputs: the same as the gate graph, plus the `gate` field.
   - Output: z_C on active cells.
4. **q graph** (optional).
   - Inputs: the same as R, plus `f`.
   - Outputs: z6..z8 on the interior ±1.

The exporter wraps each model's `features()` + `forward` + `apply` into one JAX function over raw fields. For rvv it also pre-applies the Galilean weight projection (`rvv_tensor_model.py:148-151`).

The C++ side checks the output extent against the face and stencil needs of the next model, and fails with FATAL if `ng` is too small.

### 4. ONNX Runtime wrapper: `src/pgen/subgrid_ml/onnx_model.hpp/.cpp`

- **`OnnxModel`** holds `Ort::Session` and reads the metadata. It provides `Run(const DvceArray5D<float>& in, DvceArray5D<float>& out)` over a `[nmb, C, nz+2h, ny+2h, nx+2h]` tensor, using IoBinding on Kokkos device pointers with no host copy.
- **Session options**:
  - CUDA EP on the device Kokkos uses;
  - `user_compute_stream` set to Kokkos' CUDA stream where available; otherwise `Kokkos::fence()` before and after Run;
  - a capped `gpu_mem_limit` and arena extension strategy, so ORT does not starve Kokkos.
- **Backend for CPU builds**: the ORT CPU EP with host views. The same code is used, selected by `Kokkos::DefaultExecutionSpace`.
- **One `Ort::Env` per process.** Sessions are created in `UserProblem` and destroyed in `pgen_final_func`, before `Kokkos::finalize`.
- **Packing kernels** (`subgrid_ml/pack.hpp`): a field-name → device-lambda switch writes `Real` w0/derived fields into the float buffer. The fixed vocabulary is `rho, e, ux, uy, uz, s0, T_cg, f, gate`.

### 5. Physics maps: `subgrid_ml/maps.hpp`, KOKKOS_INLINE device functions

- **Shape tensor.** A = Σz_k E^(k), using the basis in `rvv_tensor_model.py:46-51`. A symmetric 3×3 Jacobi eigendecomposition with a fixed sweep count gives S = V diag(softmax λ) Vᵀ.
- **R.** R = 2ρ f k_max S, gated by g_f = [f ≥ f_cut], with `f_cut` from the input file (default 1e-4).
- **q.** q = √ρ·ψ̃·R^{1/2}w, where ψ̃ = k_max[γ−(γ−1)f] and w = √f·sinh|z|/|z|·z. With no model, z = 0, so q = 0.
- **C.** C = g·n²[Λ_full(T_CG)e^{z_C} − Λ_cut(T_CG)], in code units following `srcterms.cpp:156-160`.
  - Add `ISMCoolFnFull` (no window cut) to `src/srcterms/ismcooling.hpp`, and define `ISMCoolFn` = window × Full.
  - The golden test checks that it matches the Python Λ_full (`cooling_model.py:58-76`).

### 6. Exporter: `models/export_athenak.py`

```
python models/export_athenak.py <ckpt.pkl> --kind {rvv,gate,cooling,q} -o athinputs/ml/<kind>.onnx
```

- It builds `raw_fields → z` from each model module's own `features()`/`apply`, so there is no duplicated feature code.
- It writes the metadata, then checks ONNX against JAX in Python ORT.
- It also writes `<kind>_golden.npz`: a random raw-field block, the JAX z values, and the final R/q/C from the Python maps.
- The model files should expose a `raw_forward(params, raw_fields)` helper so each new architecture brings its own export entry point.

### 7. The pgen: `src/pgen/subgrid_ml.cpp`

- **ICs and BCs.** Analytic ICs from `kh_radiative_cooling.cpp` (iprob=1/2), plus `<problem> coarse_ic = <file>`, which overrides iprob (§7a). BCs use `constant_bcs` with GPU-safe captures.
- **`<subgrid>` block.** Paths `rvv_model`, `gate_model`, `cooling_model`, optional `q_model`; `f_cut`; toggles `enable_R/Q/C`; `golden_test`.
- **Grid check.** FATAL if dx ≠ 0.625 pc, because the models carry an implicit dx.
- **Restarts.** Hooks are enrolled and the task is inserted before `if (restart) return;`.
- **Golden test.** `golden_test=true` loads each golden block, runs packing, ORT and the maps, compares the result (rel tol about 1e-4) and exits.
- **History** (`user_hist_func`): Σρk_sg, Σ|C|, total cooling, and the fractions of cells passing g_f and g.

### 7a. Initial conditions from a downsampled high-resolution snapshot

The IC handling lives in its own pieces so that any coarse run can use it. The runs are subgrid_ml with any `enable_*` toggles, including the all-off baseline.

**Python downsampler: `data/make_coarse_ic.py`**

```
python data/make_coarse_ic.py /data/sasi/simulation_outputs/kh3d_256x1024x256/bin/KH.hydro_w.00040.bin \
       --b 16 -o athinputs/ic/kh3d_b16_f040.cic [--diag]
```

- **Reading.** It reads one fine frame through `vis_athenak` (`SimulationData` / `simulation_data/readers.read_file`, the same reader the training code uses). It accepts `hydro_w` bins, which is what the fine run writes.
- **Coarsening.** It coarse-grains **conserved** fields with `data/coarse_grain_utils.block_mean`, the same convention as training (`rvv_tensor_model.py:56-88`):
  - ⟨ρ⟩, ⟨ρv_i⟩, ⟨ρs_n⟩;
  - ⟨E⟩ = ⟨eint⟩ + ⟨½ρ|v|²⟩.

  Because ⟨E⟩ is the full block average, it **includes ρk_sg**. That is exactly the conserved E of eq-final. ConsToPrim then gives e = ρk_max, the model's training input, with no extra handling.
- **Output.** A small self-describing binary `.cic`. Its header holds:
  - magic and version;
  - nx1, nx2, nx3;
  - domain extents;
  - the source frame's time;
  - b;
  - the variable names: `dens, mom1, mom2, mom3, ener, s_00 …`.

  The data follows as float64 arrays in `[var][k][j][i]` order. A 16×64×16 file is under 1 MB.
- **Checks.**
  - The grid must be divisible by b.
  - It warns when the coarse dx ≠ 0.625 pc, the dx the models were trained on.
  - `--b` is free, so other downsampling factors work too.
- **`--diag`.** Also writes an `.npz` with the true f, S, R, q targets and C for that frame, reusing each model's `coarse_frame()`. The a priori check in Verification step 4 compares against this directly.

**C++ loader: `athenak/src/pgen/coarse_ic.hpp`**

It provides `LoadCoarseIC(ParameterInput*, Mesh*, path)`, which is reusable by `kh_radiative_cooling.cpp` as well.

- **Reading.** Every rank reads the file on the host; it is tiny. The loader checks two things:
  - that the header's domain and resolution equal `<mesh>`, and fails with FATAL otherwise;
  - that every variable in the file has a slot. Scalars map to `nhydro + n`, with a FATAL if the file has more scalars than `nscalars`.
- **Mapping.** The data goes to a `DvceArray4D` with `deep_copy`. A `par_for` maps each cell by its global index (from `CellCenterX` and the mesh extents, as in the existing 2D reader at `subgrid.cpp:140-213`).
- **Writes `u0` directly.** It does not go through primitives, so ⟨E⟩ including k_sg is preserved exactly. It then calls `peos->ConsToPrim` on the active cells.
- **Time.** `<problem> coarse_ic_time = true` (optional) sets `pmesh->time` to the frame time, so outputs align with the fine run's times.

The old 2D `init_file` (7-var primitive) path in `kh_radiative_cooling.cpp` is left untouched.

### 8. Output variable `hydro_subgrid`

- Add it to `var_choice` in `src/outputs/outputs.hpp`, bump `NOUTPUT_CHOICES`, and keep the index checks in `basetype_output.cpp:69-110` consistent.
- Emit it from a `DvceArray5D<Real>* subgrid_out` pointer on `Hydro`, which the pgen sets. This follows the `turb_force` precedent (`basetype_output.cpp:614-618`).
- Components: R (6), q (3), f, g, z_C, C.

### 9. Build

- **CMake.** `athenak/CMakeLists.txt` and `src/CMakeLists.txt` get `option(Athena_ENABLE_ONNXRUNTIME)` and `ONNXRUNTIME_ROOT`. When the option is on, they add the include and lib, rpath and `-DATHENA_ONNXRUNTIME`. The pgen fails at compile time with `#error` without it.
- **Build script.** `builds/build_gpu.sh` gets `--onnx DIR` (and a `--cuda` selector if Step 0 picks CUDA 13). The pybind11 setup is left untouched.
- **Run input.** New `athinputs/kh_sgs3D_16x64x16.athinput`: 10×40×10 pc, nghost=4, the training-run units block, and a `<subgrid>` block.

## Files

| File | Change |
|---|---|
| `models/export_athenak.py` | new: JAX → ONNX, metadata, golden data |
| `data/make_coarse_ic.py` | new: fine frame → block-averaged conserved `.cic` (+ `--diag` targets) |
| `athenak/src/pgen/coarse_ic.hpp` | new: reusable `.cic` loader that writes `u0` |
| `models/{rvv_tensor,gate,cooling}_model.py` | add `raw_forward` export entry points |
| `athenak/src/pgen/subgrid_ml.cpp` | new pgen |
| `athenak/src/pgen/subgrid_ml/{onnx_model,pack,maps}.hpp(.cpp)` | new: ORT wrapper, field packing, physics maps |
| `athenak/src/srcterms/ismcooling.hpp` | `ISMCoolFnFull` |
| `athenak/src/outputs/outputs.hpp`, `basetype_output.cpp`, `src/hydro/hydro.hpp` | `hydro_subgrid` output |
| `athenak/CMakeLists.txt`, `src/CMakeLists.txt`, `builds/build_gpu.sh` | optional ORT linkage |
| `athinputs/kh_sgs3D_16x64x16.athinput` | new run input |

## Verification

1. **Step 0 spike**: ONNX matches JAX in Python ORT, and the minimal Kokkos+ORT IoBinding test runs on the GPU.
2. **Golden test on GPU.** Packing, the ORT graphs and the C++ maps together match the JAX/Python R, q, C.
3. **Physics checks** in golden mode:
   - S is SPD with trace 1;
   - Tr R = 2ρk_sg;
   - P > 0;
   - q = 0 without a model;
   - C = 0 where g = 0;
   - a uniform flow adds no flux divergence;
   - a Galilean boost leaves R unchanged.
4. **IC round trip.**
   - Write a `.cic` from a fine frame and start subgrid_ml with all `enable_*` off and `tlim = 0`.
   - The dumped `hydro_u` must equal the `.cic` arrays bit-for-bit.
   - The dumped `hydro_w` must reproduce ⟨ρ⟩, the Favre ũ, and e = ρk_max as computed by `rvv_tensor_model.coarse_frame`.
5. **A priori check.** From the same `.cic`, run one step with output at cycle 0, dump `hydro_subgrid`, and compare with the `--diag` targets and with `benchmarking/plot_*_true_vs_pred.py` predictions on that frame.
6. **A posteriori runs.** Start the 16×64×16 LES from a downsampled frame after 2.5 Myr, with toggles none / R / R+C / R+Q+C, **one at a time** on the GPU. Compare mixing-layer growth and total cooling with later coarse-grained frames of the fine run.
7. **Profiling.** Use the timer output to record the ORT time per stage and the Kokkos↔ORT fence overhead.
