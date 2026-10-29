# Hydro task setup in AthenaK

What goes into building and running the Hydro tasks. Written so someone new to AthenaK (and to grid-based fluid codes) can follow it. [`tasks_architecture.md`](tasks_architecture.md) explains the generic `TaskList` / `TaskID` machinery. This note is the Hydro-specific part. **Section 8** maps the planned ML subgrid model ([`PLAN_subgrid_ml.md`](PLAN_subgrid_ml.md)) onto every step described here: what must change and what is reused as-is.

## 0. Vocabulary you need first

| Term | Plain-English meaning |
|---|---|
| **Hydro** | The module that evolves a gas (density, velocity, thermal energy) using the Euler equations of fluid dynamics. |
| **Cell / grid** | The simulation box is chopped into small boxes called cells. Each cell stores one value of each fluid quantity. |
| **MeshBlock** | A fixed-size chunk of cells (e.g. 32x32x32). The full box is tiled by MeshBlocks, which are the unit of parallelism and of refinement. |
| **MeshBlockPack** | The group of MeshBlocks one MPI rank / GPU handles together. It owns the task lists and the physics modules (`phydro`, `pmhd`, ...). |
| **Ghost zones** | Extra layers of cells (`nghost`) around each MeshBlock holding copies of neighbours' data (or boundary-condition values), so stencils near the edge have something to read. |
| **Conserved variables `u`** | The quantities that are literally conserved: mass density, momentum density (x,y,z), total energy density. The update step works on these. Stored in `u0`. |
| **Primitive variables `w`** | The quantities humans think in: density, velocity (x,y,z), and a thermal-energy variable (pressure is derived from it). Riemann solvers and the timestep work on these. Stored in `w0`. `ConToPrim` converts `u -> w`. |
| **Flux** | The rate at which each conserved quantity crosses a cell face. The change in a cell is (flux in) - (flux out) divided by cell width. |
| **Reconstruction** | Estimating the value of `w` at a cell's *faces* from the cell-centre values. Gives a left state and a right state at each face. |
| **Riemann solver** | Given the left and right states at a face, computes one physically sensible flux for that face. Different solvers trade accuracy against robustness. |
| **EOS (equation of state)** | The rule linking pressure to density and thermal energy, e.g. an ideal gas with adiabatic index gamma. |
| **Time integrator / stage** | AthenaK advances time with a Runge-Kutta (RK) scheme. One time step = a few *stages*, each a partial update. RK2 has 2 stages, RK3 has 3. |
| **CFL / timestep** | The time step must be short enough that information does not cross more than a fraction (`cfl_no`) of a cell per step. `dt ~ cfl * dx / (|v| + sound speed)`. |
| **Source term** | Anything that changes the gas but is not a flux across faces: gravity, cooling, heating, etc. |
| **Boundary conditions (BCs)** | What happens at the edge of the whole box: periodic (wraps around), outflow, reflecting, or user-defined. |
| **MPI** | The library that sends data between processes (ranks). Sending ghost-zone data to neighbouring ranks is the main use here. |
| **SMR / AMR** | Static / adaptive mesh refinement: some regions use smaller cells. The code then needs *restriction* (fine -> coarse) and *prolongation* (coarse -> fine) to pass data between levels. Also called "multilevel". |
| **FOFC** | First-order flux correction. A safety net: where the high-order update would produce negative density/energy, switch to a more diffusive but safe first-order flux. |
| **STS / RKL2** | Super time-stepping. A trick to advance diffusion terms (viscosity, conduction), which would otherwise force tiny time steps, using many cheap sub-stages. |
| **pgen** | "Problem generator": the C++ file in `athenak/src/pgen/` that sets initial conditions (and can register user hooks). |
| **Task** | One named step (a function `(Driver*, int stage)`), e.g. "compute fluxes". |
| **TaskList** | An ordered set of tasks with dependencies, run repeatedly until all are done. |

## Source files

All under `athenak/src/`:

| File | Role |
|---|---|
| [`hydro/hydro.hpp`](../athenak/src/hydro/hydro.hpp) | Declares the `Hydro` class, the struct of task IDs (`HydroTaskIDs`), and the enum of Riemann solvers (`Hydro_RSolver`). The "table of contents" of the module. |
| [`hydro/hydro.cpp`](../athenak/src/hydro/hydro.cpp) | The constructor: reads the input file, picks the EOS, solver and reconstruction, allocates all arrays. |
| [`hydro/hydro_tasks.cpp`](../athenak/src/hydro/hydro_tasks.cpp) | `AssembleHydroTasks` (builds the task graph) and the simple tasks (communication, BCs, conversions). |
| [`hydro/hydro_fluxes.cpp`](../athenak/src/hydro/hydro_fluxes.cpp) | The `Fluxes` task: reconstruct, then Riemann solve, in each direction. |
| [`hydro/hydro_update.cpp`](../athenak/src/hydro/hydro_update.cpp) | The `RKUpdate` task: apply the flux divergence to advance `u`. |
| [`hydro/hydro_newdt.cpp`](../athenak/src/hydro/hydro_newdt.cpp) | The `NewTimeStep` task: computes the largest stable time step. |
| [`hydro/hydro_fofc.cpp`](../athenak/src/hydro/hydro_fofc.cpp) | The first-order flux correction safety net. |
| [`hydro/hydro_sts.cpp`](../athenak/src/hydro/hydro_sts.cpp) | Tasks for super-time-stepping of viscosity and conduction. |
| [`mesh/meshblock_pack.cpp`](../athenak/src/mesh/meshblock_pack.cpp) | Creates the empty task lists and calls `AddPhysics`, which builds Hydro. |
| [`driver/driver.cpp`](../athenak/src/driver/driver.cpp) | The main loop; runs the task lists in order; defines the RK coefficients. |

## 1. Order of construction

At startup, in this order:

1. **Empty task lists are created.** `MeshBlockPack`'s constructor makes 8 empty named `TaskList`s stored in `tl_map` ([meshblock_pack.cpp:45-52](../athenak/src/mesh/meshblock_pack.cpp#L45-L52)). They are just labelled slots that physics modules later fill:
   - `before_timeintegrator`: runs once per cycle before any RK stage. For one-off or operator-split work (e.g. turbulence mode set-up).
   - `after_timeintegrator`: runs once per cycle after all RK stages.
   - `before_stagen`: runs before each RK stage across *all* blocks. For things that must be set up first, like posting MPI receives.
   - `stagen`: runs during each RK stage. This is where the real physics happens.
   - `after_stagen`: runs after each stage across all blocks. For clean-up, like confirming MPI traffic finished.
   - `before_parabolic_stagen`, `parabolic_stagen`, `after_parabolic_stagen`: same three-part pattern, but for one sub-stage of super-time-stepping (only used if you have STS viscosity/conduction).
2. **`AddPhysics` runs** ([meshblock_pack.cpp:105](../athenak/src/mesh/meshblock_pack.cpp#L105)). It builds `Units` first (so other modules can use physical units), then, if the input file has a `<hydro>` block, creates the `Hydro` object.
3. **Hydro tasks are only assembled for single-fluid hydro.** `AssembleHydroTasks` is called only if the input has no `<mhd>`, `<radiation>`, `<adm>` or `<z4c>` block ([meshblock_pack.cpp:122-126](../athenak/src/mesh/meshblock_pack.cpp#L122-L126)). Otherwise the other module builds the combined graph. Hydro's constructor still runs, but its tasks are not added.
4. **Other physics appends tasks** (turbulence forcing, particles, numerical relativity) to the same lists afterwards.

## 2. The `Hydro` constructor ([hydro.cpp:29](../athenak/src/hydro/hydro.cpp#L29))

Tasks only receive `(Driver*, int stage)`, so anything a task needs must already live as a member variable of `Hydro`. The constructor is where all of that gets decided and allocated.

### 2.1 Input-file parameters it reads

`<hydro>` block unless noted.

| Parameter | What it is | Effect |
|---|---|---|
| `eos` (required) | Which equation of state. | `ideal`: ordinary gas with a thermal energy; makes `IdealHydro` (or the SR/GR versions with relativistic coordinates) and `nhydro=5` (density, 3 momenta, energy). `isothermal`: fixed temperature, so no energy equation; `nhydro=4`; not allowed with SR/GR. |
| `nscalars` (default 0) | Number of "passive scalars": extra quantities carried along by the flow without affecting it (tracers, metallicity, ...). | Arrays get `nhydro + nscalars` variables. |
| `nu_iso` / `nu_aniso` | Viscosity coefficients (isotropic / along-field). | Creates a `Viscosity` object and registers it as a "parabolic process" (a diffusion-type term with its own timestep limit). |
| `alpha_iso` / `alpha_aniso` / `alpha_spitzer` | Thermal conduction coefficients. | Creates a `Conduction` object (needs ideal EOS) and registers it as a parabolic process. |
| `<hydro_srcterms>` block | Built-in source terms (gravity, cooling, ...). | Creates `SourceTerms` (`psrc`). |
| `<time>/evolution` | What kind of time evolution the run does. | `stationary`: no time stepping, so none of the flux machinery below is set up. `kinematic`: fluid is just advected, only the `advect` solver is allowed. `dynamic`: full hydrodynamics (llf/hlle/hllc/roe; SR: llf/hlle/hllc; GR: llf/hlle). |
| `fofc` (default false) | Turn on the first-order flux safety net. | Sets `use_fofc`. |
| `reconstruct` (default `plm`) | How face values are estimated from cell centres. | `dc`: piecewise constant (first order, most diffusive). `plm`: piecewise linear (second order, the common default). `ppm4`, `ppmx`: piecewise parabolic (third-ish order). `wenoz`, `teno`: high-order non-oscillatory schemes. Higher order needs more ghost zones. |
| `rsolver` (required if time-evolving) | Which Riemann solver. | `advect`: pure advection. `llf`: very robust, diffusive. `hlle`: a bit sharper. `hllc`: restores the contact wave (sharper density jumps), ideal EOS only. `roe`: linearised solver. Allowed values depend on `evolution` (see above). |
| `<shearing_box>` block | Local model of a rotating disk. | Creates `OrbitalAdvectionCC` and `ShearingBoxCC` objects. Not relevant for most CGM runs. |

### 2.2 Validation done up front

The constructor checks that the choices are consistent and prints `### FATAL ERROR` and exits if not:

- **PLM + FOFC needs `<mesh>/nghost >= 3`**: FOFC needs fluxes one cell further out than usual, which needs one more ghost layer.
- **PPM4 / PPMX / WENOZ / TENO need `nghost >= 3`, and `>= 4` with FOFC**: their stencils are wider.
- **Solver / EOS / relativity mismatches abort**: e.g. `hllc` with isothermal EOS, or a solver that does not exist for SR/GR.

### 2.3 Diffusion mode flags

For each of viscosity and conduction, the constructor records whether it is handled **explicitly** (added to the normal fluxes, limits the time step) or by **STS** (advanced in the separate parabolic lists, doesn't limit the main time step). This gives `has_explicit_viscosity`, `has_sts_viscosity`, `has_explicit_conduction`, `has_sts_conduction`. `has_any_sts_diffusion` is true if either STS flag is set and decides whether the parabolic task lists get filled (section 4).

### 2.4 Memory allocated

Sizes use `ncells = nx + 2*ng` per active dimension (the block plus ghost zones). `nmb = max(nmb_thispack, nmb_maxperrank)` is the number of MeshBlocks, taken as the max so AMR can add blocks without reallocating. `nvar = nhydro + nscalars`.

| Array | What it holds | When allocated |
|---|---|---|
| `u0` | Conserved variables: the main state of the simulation. Shape `(nmb, nvar, n3, n2, n1)`. | always |
| `w0` | Primitive variables, derived from `u0` each stage. Same shape. | always |
| `coarse_u0`, `coarse_w0` | Copies of `u0`/`w0` on a 2x coarser grid, used to pass data across refinement levels. | only with SMR/AMR |
| `pbval_u` | The boundary-value object: send/receive buffers and the routines that exchange ghost zones and apply BCs for `u`. | always |
| `u1` | A second copy of `u`, the "RK register": it remembers the state at the start of the time step so later stages can blend it with the current one. | time-evolving only |
| `uflx.x1f`, `x2f`, `x3f` | The flux through the x-, y-, z-facing faces of every cell. | time-evolving only |
| `wl3d`, `wr3d` | The reconstructed left and right primitive states at each face (the inputs to the Riemann solver). | time-evolving only |
| `u_sts0`, `u_sts1`, `u_sts2`, `u_sts_rhs` | Saved earlier states / first-stage contribution needed by the RKL2 super-time-stepping recurrence. | only with STS diffusion |
| `fofc`, `utest` | Per-cell "needs correction" flag and a scratch copy of a trial updated `u` for FOFC. | declared always; used by FOFC |

The flux and left/right arrays are indexed by the *global* cell index and so are full-size, not `n+1`.

## 3. The main task graph

`Hydro::AssembleHydroTasks` ([hydro_tasks.cpp:48](../athenak/src/hydro/hydro_tasks.cpp#L48)) adds the tasks below. In each `AddTask(fn, this, dep)` call, `dep` is the task that must finish first; `none = TaskID(0)` means "no dependency". The returned ID is saved in `Hydro::id`.

### `before_stagen`: prepare to receive data

| Task | Depends on | What it does and why |
|---|---|---|
| `irecv` (`InitRecv`) | none | Tells MPI "I'm about to receive ghost-zone data from my neighbours", so buffers are ready before anyone sends. Posts receives for `u`; with SMR/AMR also for fluxes; with a shearing box also for orbital advection / shearing data. |

### `stagen`: the physics of one RK stage

A chain: each task depends on the one above it, so they run top to bottom.

| # | Task | Function | What it does and why |
|---|---|---|---|
| 1 | `copyu` | `CopyCons` | Saves the starting state. On stage 1 copies `u0` into `u1` so later stages can use the original. (RK4 additionally accumulates into `u1` on later stages; RK1-3 do nothing after stage 1.) |
| 2 | `flux` | `Fluxes` | Computes the flux through every face. Picks the Riemann solver chosen in the input, runs reconstruction then the solver (section 6), adds any explicit viscous/heat fluxes, then applies FOFC if enabled. |
| 3 | `sendf` | `SendFlux` | SMR/AMR only. Sends fluxes at fine/coarse boundaries to the neighbour, so both sides agree on the flux through the shared face (keeps conservation exact). |
| 4 | `recvf` | `RecvFlux` | SMR/AMR only. Receives those fluxes and overwrites the local ones. |
| 5 | `rkupdt` | `RKUpdate` | The actual time advance. For each cell, computes the flux divergence `(F_out - F_in)/dx` in each direction and sets `u0 = gam0*u0 + gam1*u1 - beta*dt*divF`. `gam0`, `gam1`, `beta` are the RK weights for this stage. |
| 6 | `srctrms` | `HydroSrcTerms` | Adds source terms (gravity, cooling, user hooks) to `u0`. See "Source terms in detail" below. |
| 7 | `sendu_oa` | `SendU_OA` | Shearing-box only (last stage). Sends data for the orbital-advection remap. No-op otherwise. |
| 8 | `recvu_oa` | `RecvU_OA` | Shearing-box only. Receives it and performs the remap. No-op otherwise. |
| 9 | `restu` | `RestrictU` | SMR/AMR only. Averages fine-grid `u0` down onto the coarse array `coarse_u0`, so coarser neighbours can see it. |
| 10 | `sendu` | `SendU` | Packs the edge cells of `u0` and sends them to neighbouring blocks (MPI, or a local copy) to fill their ghost zones. |
| 11 | `recvu` | `RecvU` | Receives those buffers and unpacks them into this block's ghost zones. Retried until the data has arrived. |
| 12 | `sendu_shr` | `SendU_Shr` | Shearing-box only. Sends data for the sheared boundary. |
| 13 | `recvu_shr` | `RecvU_Shr` | Shearing-box only. Receives and remaps it. |
| 14 | `prol` | `Prolongate` | SMR/AMR only. Fills ghost zones of a fine block that touches a coarse neighbour by interpolating from the coarse data. |
| 15 | `bcs` | `ApplyPhysicalBCs` | Fills ghost zones at the outer edge of the whole box (outflow, reflecting, ...) and calls the pgen's `user_bcs_func` if there is one. Skipped if the box is periodic in every direction. Runs after prolongation so corner ghost cells are valid. |
| 16 | `c2p` | `ConToPrim` | Converts `u0` into primitives `w0` over the whole block, ghost zones included. This is what the next stage's reconstruction and the timestep read. |
| 17 | `newdt` | `NewTimeStep` | On the last stage only: computes the largest stable time step for the next cycle (section 5). |

### `after_stagen`: confirm communication is finished

| Task | Depends on | What it does and why |
|---|---|---|
| `csend` (`ClearSend`) | none | Waits until every MPI send posted this stage has completed, so buffers can be reused safely. |
| `crecv` (`ClearRecv`) | `csend` | Checks every receive has completed. The `Recv*` tasks already wait, so this mainly catches bugs in the communication logic. |

These two are stage-aware. Special stage numbers (`-1` for the initial ghost-zone exchange, `-4` for the shearing box) let `Driver::Initialize` reuse them at startup.

### Source terms in detail ([hydro_tasks.cpp:268](../athenak/src/hydro/hydro_tasks.cpp#L268))

`HydroSrcTerms` multiplies by `beta_dt = beta[stage-1]*dt` (this stage's share of the time step) and applies these in order. Each one reads primitives `w0` and adds to `u0`:

1. **Built-in source terms** (`psrc->ApplySrcTerms`): whichever are switched on in `<hydro_srcterms>`: `const_accel` (constant acceleration), `ism_cooling`, `rel_cooling` (cooling curves), `self_gravity`, etc.
2. **Shearing-box terms** (`psbox_u->SourceTermsCC`): Coriolis and tidal forces. Only with a shearing box.
3. **Coordinate source terms** (`pcoord->CoordSrcTerms`): extra terms from curved space-time. GR only.
4. **User source terms** (`pgen->user_srcs_func(pmesh, beta_dt)`): runs only if `<problem>/user_srcs = true`. A function the problem generator registers itself. **This is the hook the subgrid/cooling ML model uses**: [`pgen/subgrid.cpp`](../athenak/src/pgen/subgrid.cpp) sets `user_srcs_func = UserSourceTerm`. It runs after `rkupdt` and before `c2p`, so it edits `u0` after the flux update; it should read only `w0`.
   To make a user source limit the time step, set `<hydro_srcterms>/user_cooling = true` and have the function fill `psrc->dtnew`. `SourceTerms::NewTimeStep` is called from `NewTimeStep`, and `Mesh::NewTimeStep` folds it into the CFL limit.

## 4. Parabolic (STS) task lists

Filled only if `has_any_sts_diffusion` ([hydro_tasks.cpp:79-98](../athenak/src/hydro/hydro_tasks.cpp#L79-L98)). If your run has no STS viscosity/conduction, these lists are empty and never run. They advance diffusion terms in a Strang-split way: half a time step of diffusion before the main RK step, half after.

- **`before_parabolic_stagen`**: `InitRecvParabolic` posts MPI receives for this sub-stage.
- **`parabolic_stagen`** (a chain):
  1. `ClearSTSFlux`: zero the flux arrays so only diffusion fluxes are accumulated.
  2. `STSFluxes`: compute viscous / heat fluxes for the STS-managed terms.
  3. `SendFlux` / `RecvFlux`: same flux sharing at refinement boundaries as in the main graph.
  4. `STSUpdate`: one RKL2 sub-stage, blending the saved states (`u_sts0/1/2`, `u_sts_rhs`) with the new flux divergence.
  5. `RestrictU`, `SendU`, `RecvU`, `Prolongate`, `ApplyPhysicalBCs`, `ConToPrim`: same refresh sequence as the main graph so ghost zones and `w0` are consistent for the next sub-stage.
  6. `STSRefreshTimeStep`: after the final post-sweep sub-stage, recomputes the time-step estimates.
- **`after_parabolic_stagen`**: `ClearSend` then `ClearRecv`, as in the main graph.

Explicit and STS handling are mutually exclusive for each process: `Fluxes` adds only the explicit terms and `STSFluxes` only the STS terms. `STSUpdate` changes only momentum (for viscosity) and energy (for conduction, or viscous heating with an ideal EOS).

## 5. How the tasks are driven (`Driver::Execute`, [driver.cpp:525](../athenak/src/driver/driver.cpp#L525))

One pass of the main loop is one **cycle** (one time step). In order:

1. **STS pre-sweep** (if enabled): for each STS sub-stage, run the three parabolic lists. Uses half the time step.
2. **`before_timeintegrator`** (stage 0): once-per-cycle set-up.
3. **For each RK stage `1 .. nexp_stages`**: run `before_stagen`, then `stagen`, then `after_stagen`.
4. **`after_timeintegrator`** (stage 1): once-per-cycle wrap-up.
5. **STS post-sweep** (if enabled): refreshes the parabolic time step, then runs the second half-sweep.
6. **Bookkeeping**: advance `time` and `ncycle`, write any due outputs, do AMR refine/derefine, then `pmesh->NewTimeStep(tlim)` fixes `dt` for the next cycle.

### How a list is run

`ExecuteTaskList` first resets every non-empty list, then loops calling `DoAvailable` until every pack reports done. `DoAvailable` makes one pass over the list and runs every task that (a) has all its dependencies finished and (b) has not run yet. A task that returns anything other than `TaskStatus::complete` (typically "still waiting for MPI data") is not marked done and is simply retried on the next pass. That retry loop is how communication overlaps with computation.

### Integrator coefficients ([driver.cpp:100-225](../athenak/src/driver/driver.cpp#L100-L225))

`<time>/integrator` (default `rk2`) selects the scheme and fills:

- `nexp_stages`: number of explicit stages per time step.
- `gam0[]`, `gam1[]`: weights for blending the old state (`u1`) and current state (`u0`) at each stage.
- `beta[]`: fraction of `dt` used in each stage's update.

Hydro uses them in `RKUpdate`, `HydroSrcTerms`, `CopyCons` (RK4) and FOFC. Options: `rk1`, `rk2`, `rk3`, `rk4`, plus IMEX variants (`imex2`, `imex2+`, ...) that treat some terms implicitly.

### Time step ([hydro_newdt.cpp:25](../athenak/src/hydro/hydro_newdt.cpp#L25))

Runs only on the last stage. For each active cell it computes `dx / (|v| + sound speed)` per direction, takes the minimum over all cells and active dimensions, and stores it in `Hydro::dtnew`. Special cases:

- Kinematic runs use `dx/|v|` (no sound speed).
- SR uses relativistic sound speeds.
- GR assumes a maximum signal speed of 1 (the speed of light).

It then asks conduction, viscosity and source terms for their own limits (stored in their own `dtnew`). `Mesh::NewTimeStep` multiplies by `cfl_no` and takes the smallest value across everything, including `psrc->dtnew`.

## 6. The flux kernel ([hydro_fluxes.cpp](../athenak/src/hydro/hydro_fluxes.cpp))

The `Fluxes` task, for each active direction (x, then y, then z in 2D/3D), runs:

1. **Reconstruction** (`ReconDispatch<IVX/IVY/IVZ>`): for every cell, estimates the left and right face values of the primitives and stores them in `wl3d` / `wr3d`. The method (dc, plm, ...) is chosen at run time.
2. **Riemann solve** (`SolveFace<rsolver, ivx>`): for every face, takes the left/right states and writes the flux into `uflx.x?f`. The solver is chosen at compile time (a C++ template parameter) because that is faster on GPUs.
3. **Passive scalars**: a scalar's flux is the mass flux times the scalar value from the upwind side (left if the mass flux is positive, else right).

**With FOFC on**, the reconstruction and solve ranges are widened by one cell on each side, since FOFC needs fluxes just outside the active region. `FOFC` then does a trial update of `u`, flags any cell that would end up with negative density/energy (i.e. would need floors), and replaces the fluxes on that cell's faces with the first-order LLF flux. In GR with a black-hole excision region this is triggered automatically for cells near the horizon.

## 7. Gotchas

- **Dependencies are a convention, not checked against data.** Nothing verifies that `c2p` really needs `bcs`. If you add a task, give it the right dependency or you get silent race conditions. `TaskList::InsertTask` splices a task into the chain and rewrites downstream dependencies.
- **64 task IDs per list.** A `TaskID` is a 64-bit mask (`NUMBER_TASKID_BITS = 64`). The hydro `stagen` list uses 17, so there is room, but other physics adds up.
- **State is shared through member variables.** Tasks can't pass arguments, so data flowing from `Fluxes` to `RKUpdate` lives in `uflx` / `u1`.
- **`w0` vs `u0` in source terms.** `w0` is the primitive state from the previous stage's `c2p`, while `u0` is already flux-updated. Source terms deliberately read `w0` and write `u0`.
- **Ghost zones get `ConToPrim` too**, so reconstruction at the next stage has valid primitives in every cell it reads.
- **MHD/radiation change everything.** If `<mhd>` (or `<radiation>`, `<adm>`, `<z4c>`) is in the input, Hydro registers none of the tasks in this note; the other module's task graph is used instead.

## 8. Adding the ML subgrid setup ("subgrid_ml")

Background: [`PLAN_subgrid_ml.md`](PLAN_subgrid_ml.md). The coarse-grained Euler equations contain three extra terms that a neural network predicts from the coarse fields:

- **R**: a subgrid momentum flux tensor (6 independent components).
- **Q**: a subgrid energy flux (`Q = u~ . R + q`, where `q` is an optional extra model; with no `q` model it is just `u~ . R`).
- **C**: a correction to the cooling rate.

Fluxes (R, Q) must be added to the face fluxes; C is a source term. Both are inserted into the *existing* Hydro graph from section 3. Nothing is rebuilt from scratch.

### 8.1 The big picture

- **Do not write a new `AssembleSubgridTasks` and do not touch `AssembleHydroTasks`.** The plan hooks in through the pgen: one task inserted into `stagen`, plus the existing `user_srcs_func` hook. All 17 existing hydro tasks stay, unchanged.
- **One new task**, `SubgridML::Evaluate`, runs between `flux` and `sendf`:
  `tl->InsertTask(&SubgridML::Evaluate, this, phydro->id.flux, phydro->id.sendf)`. Argument order is (function, object, dependency, insert-before-this-task). It therefore depends on `flux` and `sendf` is re-pointed to depend on it. The same pattern is used by turbulence forcing ([turb_driver.cpp:300](../athenak/src/srcterms/turb_driver.cpp#L300)).
- **One existing hook is reused**: `user_srcs_func`, called inside `HydroSrcTerms` (section 3, "Source terms in detail"), applies the cooling correction.
- **The neural nets are not C++.** Each model is exported from JAX as one ONNX graph and run through ONNX Runtime on the GPU. The graph takes raw fields and returns unconstrained numbers `z`. AthenaK only packs inputs, runs the session, applies fixed physics maps (sigmoid, matrix exponential, cooling formula) and adds the result. Changing the network never changes the task graph.
- **Why this works at all:** `RKUpdate` only reads `uflx`. Anything added to `uflx` before `RKUpdate` runs is automatically part of the update. No change to `RKUpdate` is needed.
- **Why it can be done from the pgen:** `AddPhysics` (which builds `Hydro` and fills `tl_map`/`id`) runs inside `BuildTree` ([mesh.cpp:703](../athenak/src/mesh/mesh.cpp#L703)), before the `ProblemGenerator` is constructed ([main.cpp:353](../athenak/src/main.cpp#L353)). So when `UserProblem` runs, `phydro->id.flux` etc. already exist. No existing pgen calls `InsertTask` yet, so this is a new use, but it relies only on facts above.

### 8.2 Step by step: what changes, what is reused

Legend: **Reuse** = no code change. **Config** = only an input-file change. **Change** = code edit in existing AthenaK code. **New** = new code.

#### Construction (sections 1-2)

| Step | Verdict | What to do |
|---|---|---|
| Create the 8 empty task lists (section 1, step 1) | Reuse | Nothing. No new list is needed. |
| `AddPhysics` builds `Hydro` and calls `AssembleHydroTasks` | Reuse | Works for subgrid_ml because it is single-fluid hydro (no `<mhd>`, `<radiation>`, ...). Do **not** add those blocks or the Hydro tasks will not be registered at all. |
| `Hydro` constructor: EOS | Config | `<hydro> eos = ideal`. **No EOS override**: the stock ideal EOS is used. This matters for physics (see `c2p` below). |
| `Hydro` constructor: reconstruction / solver | Config | `reconstruct = plm`, `rsolver = hllc` or `llf` as in the training runs. Any of the existing options works. |
| `Hydro` constructor: ghost-zone check | Config | Set `<mesh> nghost = 4`. The plan requires `halo + face_extra <= ng`, where `halo` is how many ghost cells the network consumes per side. The pgen must check this itself and `FATAL` if it is violated; the constructor does not know about the ML model. |
| `Hydro` constructor: diffusion / STS | Config | Leave viscosity and conduction off. The subgrid terms are not part of the STS path, and the parabolic lists (section 4) stay empty. |
| `Hydro` constructor: `<hydro_srcterms>` | Config | `user_cooling = true`, **`ism_cooling` off**. The built-in cooling would use `T-hat`, not the corrected `T_CG = T-hat (1 - f)`, so the pgen applies cooling itself. `user_cooling = true` makes `SourceTerms::NewTimeStep` return early, so the pgen's own `dtnew` is used. |
| `<problem>` block | Config | `user_srcs = true` so `HydroSrcTerms` calls the pgen's function. Optional `coarse_ic = <file>` for downsampled initial conditions. |
| New `<subgrid>` block | New | Paths to `rvv_model`, `gate_model`, `cooling_model`, optional `q_model`; `f_cut`; toggles `enable_R/Q/C`; `golden_test`. Read by the pgen. |
| Memory (section 2.4): `u0`, `w0`, `u1`, `uflx`, `wl3d`, `wr3d`, `pbval_u` | Reuse | All used as-is. |
| Memory: arrays for the model | New | Owned by the pgen, **not** by `Hydro`: a float32 input buffer `[nmb, channels, n3, n2, n1]` including ghost zones, float32 output buffers for `z`, and `Real` arrays for `f`, `R` (6), `q` (3), `g`, `z_C`, `C`. The networks run in float32 while `Hydro` uses `Real` (usually double), so packing converts. |
| `SubgridML` object lifetime | New | Tasks capture a raw pointer to the object that owns them (`AddTask` stores `obj`). The pgen must keep that object alive for the whole run (file-scope `static`/`unique_ptr`) and free it in `pgen_final_func`, **before** `Kokkos::finalize`, together with the ONNX sessions. |
| Restarts | New | Enroll the task and hooks *before* the pgen's `if (restart) return;`, otherwise a restarted run silently runs without the model. |

#### `before_stagen`

| Task | Verdict | What to do |
|---|---|---|
| `irecv` (`InitRecv`) | Reuse | Buffers are sized from `nghost`, so `nghost = 4` is handled automatically. |

#### `stagen`

| # | Task | Verdict | What to do |
|---|---|---|---|
| 1 | `copyu` | Reuse | Unchanged. |
| 2 | `flux` | Reuse (with a caveat) | Produces the ordinary hydro flux using the (over-estimated, see `c2p`) pressure. Subgrid corrections are **not** added here. Caveat: FOFC runs inside this task, so it neither sees the corrections nor limits them. Keep `fofc` off or accept this. |
| **new** | `Evaluate` | **New** (the main piece of work) | Inserted between `flux` and `sendf`. See "What `Evaluate` does" below. |
| 3 | `sendf` | Reuse | With SMR/AMR it sends the fluxes; because `Evaluate` runs first, the *corrected* flux is what is exchanged, so conservation at refinement boundaries is preserved. With a uniform grid it is a no-op. |
| 4 | `recvf` | Reuse | Same reasoning. |
| 5 | `rkupdt` | Reuse | Reads `uflx` only, so it picks up the corrections automatically. This is the key to the whole design. |
| 6 | `srctrms` (`HydroSrcTerms`) | Reuse the task, **New** the function | The task is unchanged. The new `UserSourceTerm` runs as step 4 of "Source terms in detail". It applies `E -= bdt * [n^2 Lambda_cut(T_CG) + C]` on `w0` and the stored `C`, floors the thermal energy, and writes `psrc->dtnew = min(dt_cool, dt_Q)`. Constraints: its signature is only `(Mesh*, bdt)`, with no stage argument, so it must read the arrays `Evaluate` stored earlier in the *same* stage; it must be GPU-safe (the existing [subgrid.cpp](../athenak/src/pgen/subgrid.cpp) copies device to host every step and is **not** a base to reuse for this). |
| 7-8 | `sendu_oa`, `recvu_oa` | Reuse | No-ops (no shearing box). |
| 9-11 | `restu`, `sendu`, `recvu` | Reuse | Unchanged. |
| 12-13 | `sendu_shr`, `recvu_shr` | Reuse | No-ops. |
| 14 | `prol` | Reuse | Unchanged (SMR/AMR only). |
| 15 | `bcs` | Reuse + **New** BC function | The task is unchanged. The pgen supplies `user_bcs_func` (e.g. `constant_bcs` with captures that are valid in a GPU kernel). |
| 16 | `c2p` | Reuse | The conserved energy is the *coarse-averaged* `E`, which includes the unresolved kinetic energy `rho k_sg`. Stock `ConsToPrim` therefore gives `w0(IEN) = rho k_max`, which is exactly the network's training input. So the solver's pressure `P-hat = (gamma-1) rho k_max` is slightly too high (true `P = P-hat (1 - f)`). `Evaluate` and `UserSourceTerm` correct for this through the flux and temperature corrections, not through the EOS. |
| 17 | `newdt` | Reuse | The sound speed uses `P-hat >= P`: slightly more diffusive and always safe (`f` is about 1e-2 in the mixing layer). The cooling and `Q` limits arrive via `psrc->dtnew` (set in `UserSourceTerm`, honoured because `user_cooling = true`). Only the last stage's value matters for the next step. |

#### `after_stagen`, STS lists, driver

| Part | Verdict | What to do |
|---|---|---|
| `csend`, `crecv` | Reuse | Unchanged. |
| `before/parabolic/after_parabolic_stagen` | Reuse (stay empty) | Not used. |
| `Driver::Execute` loop | Reuse | Unchanged. The task is part of `stagen`, so it is run once per RK stage. |
| Integrator | Config | Any RK scheme works, but note the cost: `Evaluate` runs once per stage (twice per step for `rk2`). Budget two ONNX Runtime calls per model per step. |
| Outputs | Change | New variable `hydro_subgrid` (R, q, f, g, z_C, C): add to `var_choice` in `outputs.hpp`, bump `NOUTPUT_CHOICES`, add a `Hydro::subgrid_out` pointer that the pgen sets (same pattern as `turb_force`), and keep the index checks in `basetype_output.cpp` consistent. |
| History | New | `user_hist_func` for diagnostics (sum of `rho k_sg`, `|C|`, total cooling, gate pass fractions). |
| Cooling function | Change | Add `ISMCoolFnFull` (no temperature window) to [`ismcooling.hpp`](../athenak/src/srcterms/ismcooling.hpp), and redefine `ISMCoolFn` as window times `Full`. The fork currently cuts cooling outside `4.1 < log T <= 5.9`; the correction formula needs the un-windowed version too. |
| Build | Change | `option(Athena_ENABLE_ONNXRUNTIME)` and `ONNXRUNTIME_ROOT` in the two `CMakeLists.txt`, a `--onnx DIR` flag in `build_gpu.sh`. Keep the pybind11 setup untouched. |

### 8.3 What `Evaluate` does in detail

It is the only genuinely new task. Per stage, per MeshBlockPack, it must return `TaskStatus::complete` in one call (it never waits). It runs on every pass, so it must **not** accumulate state between stages. It is safe to add into `uflx` because `Fluxes` rewrites `uflx` from scratch each stage.

1. **Pack** raw fields from `w0` into the float32 buffer, including ghost zones. The channel list comes from a fixed vocabulary (`rho, e, ux, uy, uz, s0, T_cg, f, gate`) and is read from each `.onnx` file's metadata. This is valid because `c2p` fills ghost zones too (section 7).
2. **Run the R graph.** Inputs `rho, e (= rho k_max), ux, uy, uz, s0`; outputs `z0..z5`. Maps give `f = sigmoid(z0)`, the shape tensor `S`, then `R = 2 rho f k_max S`, gated by `f >= f_cut`, and `T_CG = T-hat (1 - f)`.
3. **Run the gate graph** (inputs `rho, T_cg, u, s0`) to get `g = [sigmoid(logit) > 0.5]`.
4. **Run the cooling graph** (same inputs plus `gate`) to get `z_C`, then `C = g n^2 [Lambda_full e^{z_C} - Lambda_cut]`.
5. **(optional) run the q graph** (inputs as for R plus `f`); if no `q_model` is configured, `q = 0` and `Q = u~ . R`.
6. **Add the corrections to the face fluxes** (arithmetic average of the two adjacent cells onto the face), modelled on how viscosity does it in [viscosity.cpp](../athenak/src/diffusion/viscosity.cpp#L96):
   - momentum: add `R_ij - (gamma-1) rho k_sg delta_ij`;
   - energy: add `Q_i - (gamma-1) rho k_sg u~_i`.
7. Store `f, R, q, g, z_C, C` in the pgen-owned arrays for `UserSourceTerm` and for output.

Which parts can be reused: the reconstruction, Riemann solve, the flux arrays, the face loops pattern, boundary exchange, all of `RKUpdate`. What is new: ONNX session wrapper, the packing kernels, the physics maps (matrix exponential via a fixed-sweep Jacobi eigendecomposition, sinh map, cooling formula), and the code that adds to `uflx`.

### 8.4 Things to watch (found while tracing the code)

- **Receptive field vs ghost zones.** Inference runs per MeshBlock, so the network can use at most `ng` cells of context per side minus what is needed to average to faces. Enforce `halo + face_extra <= ng` at startup. A model that needs global context requires one block per rank or a gather step.
- **Models must be re-exportable, not re-implementable.** Because feature construction and normalisation live inside the ONNX graph, a new architecture (U-Net, attention, ...) needs a re-export only. A new *raw input field* needs one line in the packing kernel. A change to the physics maps needs C++.
- **CUDA pairing.** Kokkos (CUDA 12.8) and ONNX Runtime must share one CUDA major version and stream; do the toolchain spike (Step 0 of the plan) first.
- **Stream ordering.** ORT may use its own stream. Either share Kokkos' stream or `Kokkos::fence()` before and after each `Run`, otherwise `Evaluate` reads half-written buffers.
- **Initial state.** The first cycle needs valid `w0` in ghost zones before the first `Evaluate`. `Driver::Initialize` already does a ghost-zone exchange and `ConToPrim`; check this still holds for IC loaded from a `.cic` file (the loader writes `u0` directly and then calls `ConsToPrim`).
- **`InsertTask` returns `TaskID(0)` if the "insert before" task is not found.** Check the return value and `FATAL` on 0, otherwise the model silently never runs.
- **Grid assumption.** The models carry an implicit `dx = 0.625 pc`; `FATAL` at startup if the mesh differs.

## 9. Making it a reusable framework (not a single-problem setup)

Section 8 follows the plan literally: one pgen, three fixed models (R, gate, cooling), plus an optional `q`, on one grid (`dx = 0.625 pc`). That is enough for the first experiment, but several of its choices would have to be torn out to support new architectures, grids and meshes. This section proposes the smallest set of seams that keeps the plan's good ideas (ONNX graph from raw fields, task inserted before `sendf`, source hook) and removes the hard-coding. Nothing here is implemented yet.

### 9.1 What in the plan is hard-wired

| Hard-wired in the plan | Why it blocks reuse |
|---|---|
| The code lives in a **pgen** (`subgrid_ml.cpp`) | A pgen sets up one problem. Using the same models in a CGM box, a turbulence run or a restart from another pgen would mean copying code. |
| Exactly three model kinds (`rvv`, `gate`, `cooling`, `q`) in a fixed order | Adding a fifth model (or changing the dependency between them) means editing C++. |
| Fixed raw-field vocabulary (`rho, e, ux, ..., gate`) | New inputs need a code edit and a rebuild. |
| `dx = 0.625 pc` is `FATAL` | A model can never be run on another resolution, even if it would work. |
| Output always cell-centred, then averaged to faces | A model that predicts face-centred fluxes directly cannot be expressed. |
| Evaluated once per RK stage | No way to evaluate less often or to keep state between steps (recurrent / stateful models). |
| Receptive field must be `<= nghost` | Fine for local stencils; no path for models needing larger or global context. |
| Hydro only (`phydro`, `uflx`, `u0`) | MHD would need a second copy of the code. |
| 3D only in practice | 1D/2D boxes have singleton dimensions that the packing code has to handle. |

### 9.2 Recommended structure: a physics module, not a pgen

AthenaK already has a precedent for a self-contained optional physics module that attaches to the task lists: `TurbulenceDriver`. `AddPhysics` creates it when its input block exists and calls `IncludeAddForcingTask` ([meshblock_pack.cpp:180-192](../athenak/src/mesh/meshblock_pack.cpp#L180-L192)), and the pack owns and deletes it ([meshblock_pack.cpp:72](../athenak/src/mesh/meshblock_pack.cpp#L72)). Do the same:

```
athenak/src/subgrid/
  subgrid.hpp/.cpp          SubgridModule: owned by MeshBlockPack (psubgrid), created if <subgrid> exists
  field_registry.hpp        named device arrays ("rho", "f", "R_xx", ...) with extents/staggering
  model_graph.hpp           reads the model list, builds the dependency graph, runs in order
  backend/inference.hpp     abstract InferenceBackend
  backend/onnx_backend.cpp  ONNX Runtime implementation (default)
  closures/                 physics maps (z -> physical term), registered by name
  sinks/                    where results go: face-flux, cell-source, timestep
```

Benefits:

- **Any pgen can use it.** The pgen only sets initial conditions; the `<subgrid>` block in the input file turns the module on.
- **It is created inside `AddPhysics`**, so it can also read the physics that exists (`phydro` now, `pmhd` later) and the module can be given the right IDs to insert after.
- **Cost:** a small edit to core AthenaK (`meshblock_pack.hpp/.cpp`, about 10 lines). Record it in [`CHANGES_TO_ATHENAK.md`](../athenak/CHANGES_TO_ATHENAK.md) so upstream merges stay manageable. The alternative (keeping everything in a pgen) avoids touching the core but locks the framework to one problem generator.

### 9.3 The four layers

Each layer knows only about the layer below. A new model touches only the layer it belongs to.

| Layer | Job | Written in | Changes when... |
|---|---|---|---|
| **1. Fields** (`FieldRegistry`) | Named device arrays derived from `w0`/`u0`, plus the outputs of other models. Each has a name, shape (with ghost zones) and a **provider** (a small kernel). | C++ (one line per new field) | you need a new raw input, e.g. `T_cg`, `vorticity`. |
| **2. Models** (`InferenceBackend`) | Map named input fields to named output arrays. No physics. | ONNX file (+ metadata) | the architecture, features, normalisation or stencil changes. **No C++ change.** |
| **3. Closures** | Fixed physics maps from raw outputs `z` to physical terms (sigmoid, matrix exponential, cooling formula, ...). | C++, registered by name | the physical formulation changes. |
| **4. Sinks** | Apply a term to the simulation: add to face flux, add to a cell-centred source, or limit the time step. | C++, 3-4 generic ones | you need a new *kind* of coupling (rare). |

An ONNX model never learns anything about the task graph, and the task graph never learns anything about a particular model.

### 9.4 Input file: models declared, not hard-coded

AthenaK input blocks are flat key-value lists, so use a list of names plus one block per name:

```
<subgrid>
models     = rvv, gate, cool        # evaluated in dependency order
hook       = pre_flux_exchange      # where flux sinks run (default)
backend    = onnx

<subgrid_rvv>
path       = ml/rvv.onnx
closure    = rvv_tensor             # maps z0..z5 -> f, S, R
sink       = face_flux              # R, Q corrections
inputs     = rho, e, ux, uy, uz, s0 # optional: overrides ONNX metadata
every      = 1                      # evaluate every N stages/cycles
dx_policy  = error                  # error | warn | input | ignore

<subgrid_cool>
path       = ml/cooling.onnx
closure    = cooling_correction
sink       = cell_source
```

The framework builds the dependency graph from declared inputs and outputs, not from the order of the list: the cooling graph consuming `gate` and `T_cg` (produced by the rvv closure) automatically runs after them. A cycle is a `FATAL`.

### 9.5 What each requirement means in practice

#### Different architectures
- Keep the plan's rule: **features and normalisation live inside the ONNX graph**, so a conv net, U-Net or attention model is only a re-export.
- Define a versioned **metadata schema** in `metadata_props` and have the framework validate it at start-up: `schema_version`, `inputs` (names), `outputs` (names, with a `stagger` field, see below), `halo` (per side, per axis), `ndim`, `trained_dx`, `units`, `scope`, `kind`.
- The abstract `InferenceBackend` (`Load`, `Describe`, `Run(inputs, outputs)`) means ONNX is only one implementation. A native Kokkos model, or a CPU fallback for debugging, can be added behind the same interface without touching anything above. This keeps with the earlier decision that networks are not hand-written Kokkos layers: a second backend is *possible*, not required.

#### Different grids and resolutions
- **Do not hard-code `dx`.** Replace the `FATAL` with `dx_policy`:
  - `error`: refuse (the safe default for models trained at one resolution);
  - `warn`: run and print once;
  - `input`: pass `dx, dy, dz` to the graph as extra scalar inputs, for models trained to take resolution as a feature;
  - `ignore`: for scale-free models.
- Use **dynamic spatial dimensions** in the ONNX graph (already in the plan's spike), so block size is not baked in. The framework reads the block shape from `mb_indcs`.
- Under AMR, blocks on different levels have different `dx` (`mb_size.d_view(m).dx1`). The `input` policy therefore has to send per-block `dx`, not one global number. With `error`, make the check per level and refuse blocks the model was not trained for.
- **Units.** Declare `units` in the metadata (e.g. `density: cm-3`, `length: pc`) and convert from code units through the `<units>` block once, in the field providers. This is what makes a model usable outside the training unit system.

#### Different meshes and dimensionality
- Read `ndim` from the metadata and from `Mesh::multi_d` / `three_d`. If the model is 3D and the mesh is 2D, either refuse or expand singleton axes; make this explicit in the schema (`ndim_policy`).
- **Stagger.** Give every output a location: `cell`, `face1`, `face2`, `face3`. A closure that gets a cell-centred output averages to faces (the plan's default); a model that already predicts face fluxes needs no averaging. This is one metadata field plus a branch in the sink.
- **Batching.** The plan runs one tensor of shape `[nmb, C, nz, ny, nx]`. Add a `max_batch` option so a large `nmb` is split into chunks if GPU memory is tight, and so models that only accept batch size 1 still work.
- **Receptive field.** Support three scopes in the metadata: `block` (default, needs `halo <= ng`), `rank` (one block per rank, e.g. a large U-Net) and `global` (a gather step, not implemented in the first version but reserved in the schema so models are not blocked later). On `ng` itself: `nghost` is read when the `Mesh` is constructed ([mesh.cpp:87](../athenak/src/mesh/mesh.cpp#L87)), which is before `AddPhysics`. If you want `ng` raised automatically from the model metadata, that has to happen in `main.cpp` before the mesh is built (using `pin->SetInteger`, which exists). The simpler first version is to keep `FATAL` with a message that states the required `nghost`.

#### Different physics couplings
- **Sinks** are the only place that touches simulation state. Four are enough to start:

  | Sink | Writes to | Hook point |
  |---|---|---|
  | `face_flux` | `uflx.x1f/x2f/x3f` | task inserted after `flux`, before `sendf` |
  | `cell_source` | `u0` | inside `user_srcs_func` / after `rkupdt` |
  | `timestep` | `dtnew` | read by `NewTimeStep` |
  | `diagnostic` | output arrays / history | outputs |

- Add one more only if a model really needs it, e.g. `reconstruction_input` (modify `w0` before `flux`, a different hook point).
- **Hook points as named positions**, not hard-coded IDs: `pre_flux`, `pre_flux_exchange` (default for flux sinks), `post_update` (default for cell sources), `post_c2p`. The module maps each to a `TaskID` from `phydro->id` today and `pmhd->id` later, so MHD support only changes that mapping.
- **Cadence.** Per model, `every = N` and `stage = first | all | last`. Expensive models do not need to run on every RK stage (section 8.4 noted `rk2` doubles the cost). If a model is skipped, its previous outputs are reused, so output arrays must persist between calls (they already do as module-owned arrays).
- **State.** Reserve a `state` field kind (persistent per-cell arrays that a model reads and writes) for recurrent or memory-carrying models. Not needed for the first release, but it is cheap to keep the field registry capable of it.

#### Validation, so new models can be trusted
- **Generic golden test.** Every exported model ships with `<name>_golden.npz` (raw fields, expected `z`, expected final term). A single `golden_test = true` mode loops over the declared models and compares. The plan builds this for three models; make it model-agnostic and driven by the same list.
- **Generic physics checks** the framework can run for any model that declares its `kind`: flux term is zero for uniform flow, Galilean invariance, positivity of the derived pressure, and so on. Each closure declares which invariants apply.

### 9.6 Build order that preserves optionality

1. **Spike** (plan Step 0): ONNX export, CUDA pairing, minimal Kokkos+ORT test. Unchanged.
2. **Skeleton**: `SubgridModule`, `FieldRegistry`, `InferenceBackend`, ONNX backend, one trivial model (identity or constant) and one sink (`face_flux`). Test with a "model" that returns a known constant so the plumbing is verified before any real model: uniform flow must stay uniform; a constant flux correction must give the analytic result.
3. **Port R** (rvv) as the first real model with the `rvv_tensor` closure. Compare with the golden data.
4. **Add gate, then cooling** as dependent models. This exercises the dependency graph, the `cell_source` sink and the `timestep` sink.
5. **Generalise one axis at a time**: `dx_policy`, then `stagger`, then `every`/`stage`, then `max_batch`. Each is a small, independently testable change.
6. **Later**: MHD (a second hook-point map), stateful models, global scope.

### 9.7 Differences from the current plan

| Plan | Framework version | Impact |
|---|---|---|
| New pgen `subgrid_ml.cpp` owns everything | `SubgridModule` in the pack, any pgen can use it; a thin pgen only handles ICs/BCs | Edits core (`meshblock_pack.*`) |
| Fixed model list and order | `models =` list, dependency graph from metadata | More code up front, no edits per model |
| `dx` must be 0.625 pc | `dx_policy` per model | Same safety by default, but opt-in flexibility |
| Cell-centred outputs only | `stagger` per output | One metadata field |
| Evaluate every stage | `every` / `stage` per model | Lets expensive models run less often |
| ORT wrapper only | `InferenceBackend` interface with ORT implementation | A small abstraction layer |
| Hydro only | Hook-point map (hydro now, MHD later) | No change to models |

The `.cic` initial-condition downsampler, `coarse_ic.hpp` loader, `ISMCoolFnFull`, the exporter, the `hydro_subgrid` output and the verification ladder from the plan are all unaffected and should be kept as they are.
