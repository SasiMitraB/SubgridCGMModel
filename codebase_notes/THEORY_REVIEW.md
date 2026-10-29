# Theory notes review: what to change before the `subgrid_ml` pgen

This review covers the notes in `theory/notes/Overleaf/chapters/`:
- `01-derivation.tex` (Ch. 1)
- `02-implementation.tex` (Ch. 2)
- the supporting chapters `trml-data.tex`, `tbnn-theory.tex` and `hit-data.tex`

I read them against two things. The first is the AthenaK fork: `src/hydro`, `src/eos`, `src/srcterms` and `src/pgen`. The second is the trained models in `models/`. The coupling design they are checked against is in `PLAN_subgrid_ml.md`.

**Scope decisions this review assumes:**
- **No passive scalar** in the final runs or in the final model inputs.
- **The "Expanding Q" chapter (`03-ExpandingQ.tex`) is dropped.** It is not reviewed here. The only related point is removing its `\include`.

The theory is sound as an *a priori* framework. The changes below are mostly about what happens when the equations are solved by a finite-volume code and not just evaluated on filtered data. They are grouped by priority:

- **A. Needed before implementation.** These change the equations the code solves, or the models.
- **B. Errors and inconsistencies in the text.**
- **C. Modelling assumptions to state explicitly.**
- **D. Suggested new sections.**

---

## Summary

| # | Issue | Where | Priority |
|---|---|---|---|
| A1 | The notes never state the equations in the form AthenaK solves. The solver pressure is $\hat P=(\gamma-1)\langle\rho\rangle k_{\max}$, not $\langle P\rangle$. For $\gamma=5/3$ the needed flux corrections reduce to the **deviatoric** part of $\mathsf R$, plus $\mathbf q$. | Ch. 1 §Final set, Ch. 2 | High |
| A2 | The finite-volume meaning of $\mathsf R$ and $\mathbf Q$ is not addressed: volume vs. face averages, the Riemann flux vs. $\langle\mathbf m\rangle\langle\mathbf m\rangle/\langle\rho\rangle$, and implicit numerical dissipation. | Ch. 1 §Filtering | High |
| A3 | All three trained models take the passive scalar $\tilde s$ as an input, and the gate also takes $\nabla\tilde s$. Dropping the scalar means retraining them and checking what is lost. | Ch. 2, `models/*` | High |
| A4 | Ch. 1 states $\mathcal C$ is Galilean invariant, but the cooling and gate models take raw $\tilde{\mathbf v}$ with no Galilean projection. | Ch. 2 §Cooling, `models/cooling_model.py`, `gate_model.py` | High |
| A5 | Time-discrete realizability, wave speeds and timestep limits are not covered. "The network cannot be unphysical" is too strong once $\mathbf q$ and $\mathcal C$ act over a finite $\Delta t$. | Ch. 2 intro | Medium |
| B1 | Notation and units: $\Lambda(T)$ vs. $\Lambda(T_{\rm CG})$; $m_p$ vs. $m_u$; the cooling-rate convention. | Ch. 1 | Low |
| B2 | Placeholder chapters are still included in `main.tex`. One contains an inappropriate caption on a personal photo. The dropped Ch. 3 is still included. | `main.tex`, `tbnn-theory.tex`, `hit-data.tex` | High (before sharing) |
| B3 | It is unspecified whether the flux gate $g_f$ also gates the pressure/temperature split. | Ch. 2 §Gating | Medium |
| C1 | $k_{\rm sg}$ is *diagnostic* (no transport, no explicit dissipation). This should be stated. | Ch. 2 | Medium |
| C2 | There is a train/inference mismatch in $T_{\rm CG}$: training uses the true $\langle P\rangle$, inference uses the predicted $f$. | Ch. 2 §Cooling | Low |
| C3 | Resolution and filter-width dependence: the models carry an implicit $\Delta=0.625$ pc. | Ch. 2 | Medium |
| C4 | The initial transient ($t<2.5$ Myr) is excluded from training, so a posteriori runs must not start from the analytic IC. | Ch. 2 §Distributions | Medium |

---

## A. Needed before implementation

### A1. The equations in the form the code solves

**The current state of the notes.** Eq. (eq-final) evolves $\langle E\rangle$, which *includes* the subgrid kinetic energy $\langle\rho\rangle k_{\rm sg}$. The pressure in it is the true $\langle P\rangle$ from eq. (eq-pressure).

**What AthenaK does.** AthenaK's `ConsToPrim` (`eos/ideal_c2p_hyd.hpp`) knows nothing about $\mathsf R$. From the conserved state it gives

$$
w_{\rm IEN}=\langle E\rangle-\tfrac12\frac{|\langle\mathbf m\rangle|^2}{\langle\rho\rangle}=\langle\rho\rangle k_{\max},
\qquad
\hat P=(\gamma-1)\langle\rho\rangle k_{\max}=\frac{\langle P\rangle}{1-f}.
$$

The Riemann solver, the reconstruction, the CFL sound speed, the floors and the stock cooling all use $\hat P$ and $\hat T$. So the notes need a short "code form" of eq. (eq-final), written in terms of the solver's own fluxes.

**Code form.** With $\hat P-\langle P\rangle=(\gamma-1)\langle\rho\rangle k_{\rm sg}=\tfrac{\gamma-1}{2}\,\mathrm{Tr}\,\mathsf R$:

$$
\begin{aligned}
\partial_t\langle\mathbf m\rangle+\nabla\cdot\Big[\tfrac{\langle\mathbf m\rangle\langle\mathbf m\rangle}{\langle\rho\rangle}+\hat P\,\mathbb 1+\underbrace{\mathsf R-\tfrac{\gamma-1}{2}(\mathrm{Tr}\,\mathsf R)\,\mathbb 1}_{\Delta\mathsf F_m}\Big]&=0,\\
\partial_t\langle E\rangle+\nabla\cdot\Big[(\langle E\rangle+\hat P)\,\tilde{\mathbf v}+\underbrace{\tilde{\mathbf v}\cdot\big(\mathsf R-\tfrac{\gamma-1}{2}(\mathrm{Tr}\,\mathsf R)\mathbb 1\big)+\mathbf q}_{\Delta F_E}\Big]&=-\Big(\tfrac{\langle\rho\rangle}{\mu m_u}\Big)^2\Lambda_{\rm cut}(T_{\rm CG})-\mathcal C .
\end{aligned}
$$

**For $\gamma=5/3$ the correction is exactly the deviatoric part of $\mathsf R$,** since $(\gamma-1)/2=1/3$:

$$
\Delta\mathsf F_m=\mathsf R^{\rm dev}=2\langle\rho\rangle k_{\rm sg}\big(\mathsf S-\tfrac13\mathbb 1\big),\qquad
\Delta F_E=\tilde{\mathbf v}\cdot\mathsf R^{\rm dev}+\mathbf q .
$$

The isotropic part of the subgrid stress behaves as a $\gamma=5/3$ pressure: random isotropic motions are a monatomic "gas". AthenaK's $\hat P$ therefore already carries it exactly.

**Consequences worth writing down:**

1. **Dynamics vs. thermodynamics.**
   - In the fluxes, the trace ($f$) enters only through the amplitude $\langle\rho\rangle k_{\rm sg}$ of $\mathsf S-\mathbb 1/3$ and through $\mathbf q$.
   - In the **thermodynamics** it enters directly: $T_{\rm CG}=\hat T(1-f)$, the cooling, and the model inputs.
   - So errors in $z_0$ matter far more for cooling than for momentum transport, and errors in $z_1..z_5$ matter only where $k_{\rm sg}$ is large. This should shape the loss weighting (Ch. 2 §Distributions already hints at $f$-weighting for $\mathsf S$).
2. **No EOS change needed.** The code adds $\Delta\mathsf F_m$ and $\Delta F_E$ to the face fluxes and keeps the stock `ConsToPrim`.
3. **Positivity of the total stress.** $\hat P\mathbb 1+\mathsf R^{\rm dev}=\langle P\rangle\mathbb 1+\mathsf R$ is positive definite: $\langle P\rangle>0$ by the sigmoid and $\mathsf R\succeq0$ by the expm map. Its smallest eigenvalue is at least $\langle P\rangle=\tfrac23\langle\rho\rangle e_{\rm th}$. This one-line claim fits next to Claim (R-psd).
4. **Floors** (`dfloor`, `pfloor`, `tfloor`) act on $\hat P$ and $\hat T$, not on $\langle P\rangle$ and $T_{\rm CG}$. A cell at the temperature floor can have $T_{\rm CG}$ below it by the factor $1-f$. That is harmless, but it should be stated.

**Suggested text:** a new §"Form solved by the code" at the end of Ch. 1, or the start of Ch. 2. It should give the $\hat P$ definition, the two equations above, and the $\gamma=5/3$ remark.

### A2. Finite-volume meaning of the subgrid terms

Ch. 1 filters with a top-hat of width $\Delta$, the coarse cell. That makes $\langle\cdot\rangle$ exactly the finite-volume cell average, which is a strong point and worth saying explicitly. But the notes stop at the continuous equation $\partial_t\langle U\rangle+\nabla\cdot\langle\mathbf F\rangle=S$, and three issues follow for a finite-volume code.

1. **The exact flux is a face average, not a volume average.**
   - For a top-hat filter, $\partial_x\langle F\rangle$ at a cell centre is exactly $[\bar F_{i+1/2}-\bar F_{i-1/2}]/\Delta$. Here $\bar F$ is the *surface* ($\Delta^2$) average over the face.
   - $\mathsf R$ and $\mathbf Q$ in the notes are *volume* ($\Delta^3$) averages at cell centres. The code interpolates them to faces as $\tfrac12(\mathsf R_i+\mathsf R_{i+1})$, which is equivalent to the central differences used in Ch. 2 §Gating. That is an approximation and should be named as one.
   - Alternative: train on face targets, i.e. surface averages of $\rho\mathbf v\mathbf v$ etc. over coarse faces, computed from the fine grid. This removes the interpolation.
2. **The resolved flux in the code is a Riemann flux.** It is HLLC on PLM-reconstructed states, not $\langle\mathbf m\rangle\langle\mathbf m\rangle/\langle\rho\rangle+\langle P\rangle\mathbb 1$ at the face. The difference is a discretisation error of the same order as the subgrid terms at $\Delta=0.625$ pc, where the layer is about one cell thick (`trml-data.tex` §Convergence).
3. **Implicit LES double counting.**
   - The coarse solver is already dissipative: the 16×64×16 run converges to $\dot\Sigma_{\rm cool}\approx0.95$ against 1.42, with no model at all.
   - The "perfect" closure for a given scheme is (filtered truth at $t+\Delta t$) − (coarse solver step from the filtered state at $t$). That is not the same as the exact $\mathsf R$, $\mathbf Q$, $\mathcal C$. This is the discretisation-consistent / "perfect LES" argument in the data-driven LES literature (e.g. Beck, Flad & Munz 2019, *J. Comput. Phys.*).

**Recommendation for the notes.** Add a §"Discrete closure" that does three things:
- states that the a priori targets are the continuous-filter terms;
- lists the face/volume and Riemann-flux approximations above;
- defines a **solver-in-the-loop residual diagnostic**:
  - take a downsampled fine frame (`data/make_coarse_ic.py` in the plan);
  - run one coarse step without the model;
  - compare with the downsampled next frame.

That residual is the closure the coarse code actually needs. Comparing it with $\nabla\cdot\mathsf R$, $\nabla\cdot\mathbf Q$ and $\mathcal C$ shows how much of the target the solver already supplies. Whether to train on it later is a separate decision. The theory should acknowledge the gap now.

### A3. Removing the passive scalar from the model inputs

**Where the scalar is used now.** Every trained model takes $\tilde s$ (`s0`):
- the R model: `rvv_tensor_model.py:37`;
- the cooling model: `cooling_model.py:43`;
- the gate model: `gate_model.py:45`, which also takes $\nabla\tilde s$.

The fine run's scalar marks initially-cold material. So it is a direct phase label, and probably one of the strongest cues the gate uses to find mixing-layer gas.

**What to change in the notes:**
1. **Model inputs in Ch. 2.** Define them as built only from $\langle\rho\rangle$, $\langle\mathbf m\rangle$ and $\langle E\rangle$ (with $T_{\rm CG}$ derived from these). Then the closure is complete in the variables of eq. (eq-final), and the unknown count stays at **10**.
2. **Equations in Ch. 1.** State that the coarse equations carry no scalar. There is then no scalar subgrid flux to model; the closure is exactly the 10 outputs of eq. (eq-final).
3. **Phase information.** Phase must now come from thermodynamics alone: $\log T_{\rm CG}$, $\log\langle\rho\rangle$, their gradients, and $k_{\max}$. In the TRML that is unambiguous, because the phases are defined by temperature (Ch. 2 §Distributions already partitions by $T_{\rm CG}$). Say so, because it is what justifies dropping $\tilde s$.

**What the scalar's absence costs, which should be measured:**
- Retrain all three models without `s0` (and without $\nabla s_0$ for the gate).
- Report the change in the a priori metrics: the R val losses, the gate F1/recall, the cooling `rmse_in`/`rmse_out`/`total_err`.
- The gate is the most exposed. Near the window edges, a cell's $T_{\rm CG}$ alone may not tell "contains some cold gas" from "uniformly warm". If its recall drops, consider adding $|\nabla\log T_{\rm CG}|$ or the local min/max of $\log T_{\rm CG}$ over the stencil as explicit features.
- The benefit is real, not only simplicity: a run without a scalar cannot drift the models' inputs away from training through numerical scalar diffusion.

### A4. Galilean invariance of $\mathcal C$ and the gate

Ch. 1 §Cooling says $\mathcal C$ is a scalar and "will be invariant". The R model respects its invariance: the velocity weights of the first layer have their stencil mean removed (`rvv_tensor_model.py:148-151`).

The cooling model and the gate model take raw $\tilde u_x,\tilde u_y,\tilde u_z$, with no such projection (`cooling_model.py:43-45`, `gate_model.py:45-52`). The gate's strain and vorticity channels are invariant; the raw velocity channels are not. In practice the training frame is fixed: the layer is roughly at rest by the choice $v_c=-\chi^{-1/2}v_h$. So the models may have learned frame-specific cues, such as "hot gas moves at $+28$". With the scalar removed (A3), this cue becomes more tempting for the network: the bulk velocity also separates the hot stream from the cold one in the training frame. A boosted run, or a layer that drifts, would change $\mathcal C$.

**For the notes:**
- add a short §"Symmetries of the models" in Ch. 2, covering:
  - which outputs must be invariant ($f$, $\mathsf S$, $\mathbf w$, $z_{\mathcal C}$, $g$) or covariant ($\mathsf R$, $\mathbf q$ under rotations);
  - how each is enforced: the Galilean projection, $O_h$ augmentation (approximate, learned equivariance), and the $z\to-z$ reflection;
- state that the cooling and gate models must use the same projection, or feed only velocity differences and gradients.

The retraining for A3 is the natural time to add it. The projection should also live inside the exported ONNX graph for every model.

### A5. Realizability over a finite time step; wave speeds

The Ch. 2 intro says *"The network cannot be unphysical—it can only be inaccurate."* That holds pointwise for $\mathsf R$, $f$ and the *sign* of the total cooling. It does not hold once the terms are integrated over a finite $\Delta t$:

- **Unbounded $\mathbf q$.** $\mathbf q$ is not bounded by anything: $|\mathbf w|\le B$ is not enforced (the Ch. 2 table says so). A large $\nabla\cdot\mathbf q$ can drive $e_{\rm th}$ negative in a neighbouring cell within one step.
- **Unbounded cooling.** $\langle n^2\Lambda\rangle_{\rm model}=g\,n^2\Lambda_{\rm full}e^{z_{\mathcal C}}$ is unbounded above, so cooling can exceed $e_{\rm th}/\Delta t$.
- **Wave speeds.** The effective stress is $\langle P\rangle\mathbb 1+\mathsf R$, which is anisotropic. Along an eigen-direction of $\mathsf S$ with eigenvalue $s_{\max}$, a one-dimensional compression of the subgrid motions behaves like a 1-dof gas ($\gamma=3$). So the signal speed can reach

  $$c_{\rm eff}^2\simeq\frac{\gamma\langle P\rangle}{\langle\rho\rangle}+6\,k_{\rm sg}\,s_{\max},$$

  compared with the solver's $\hat c^{\,2}=\gamma\langle P\rangle/\langle\rho\rangle+\gamma(\gamma-1)k_{\rm sg}$.
  - The difference is small where $f\sim10^{-2}$, but not in general.
  - Because $\mathsf R$ is a *nonlocal* function of the state (a stencil network), hyperbolicity is not guaranteed at all. This is an empirical question for the a posteriori runs.

**Suggested rewording:** "every output is *pointwise* realizable: $\mathsf R\succeq0$, $\langle P\rangle>0$, non-negative total cooling. Stability over a time step needs the constraints $\Delta t\le\min(\Delta/c_{\rm eff},\ e_{\rm th}/|\dot e|,\ \Delta\,\langle\rho\rangle\tilde\psi/|\mathbf q|)$."

Then list these in the "code form" section (A1).

---

## B. Errors and inconsistencies in the text

### B1. Notation and units

- **Eq. (eq-cooling)** writes $\Lambda(T)$ inside the coarse term. Eq. (eq-final) writes $\Lambda(\Tcg)$. Use $\Lambda_{\rm cut}(T_{\rm CG})$ in both, and define $\Lambda_{\rm cut}$ and $\Lambda_{\rm full}$ in Ch. 1 rather than in Ch. 2.
- **$m_p$ vs. $m_u$.**
  - The notes define $n=\rho/(\mu m_p)$ and $P=\rho k_BT/(\mu m_p)$.
  - AthenaK's `Units::temperature_cgs` and the cooling `n_unit` use the atomic mass unit $m_u$, and so does `cooling_model.py:127` ("$\mu m_u$").
  - The difference is 0.7% in $n$ and 1.5% in $n^2\Lambda$.
  - Use $m_u$ throughout. The density *unit* is still $1\,m_p\,{\rm cm^{-3}}$, which is fine.
- **Cooling convention.** State that $n^2\Lambda$ is the code's convention ($n$ is the total particle density), not the $n_en_H\Lambda_N$ normalisation of the Schure et al. table.
- **Scalar mentions in `trml-data.tex`.** It says "A passive scalar tracks the initial phase", and the outputs section lists the scalar. Keep those as a description of the fine runs, but note that the scalar is not used by the closure (A3).
- **Code documentation.** `athenak/CHANGES_TO_ATHENAK.md:46` documents the cut-off as `logt <= 4.1 || logt > 5.9`. The code (`ismcooling.hpp`) and `trml-data.tex` use $1.05\times10^4$ and $0.95\times10^6$ K. Fix that document.

### B2. Chapters to remove from `main.tex`

`main.tex` includes `tbnn-theory.tex`, `hit-data.tex` and `03-ExpandingQ.tex`.

- **`tbnn-theory.tex`** is a placeholder. It contains a caption calling the person in a photo (`images/tbnn_theory/sasi_mitra.jpeg`) "an idiot". Remove it before the PDF goes to anyone.
- **The models are not TBNNs.** Ch. 2's parametrization is a log-Euclidean (matrix-exponential) map, not a tensor-basis expansion. Either write the TBNN chapter as a *contrast* (why the expm map was chosen over Pope's basis) or drop it.
- **`hit-data.tex`** is a TODO stub. Comment out the `\include` until it has content.
- **`03-ExpandingQ.tex`** is being dropped. Remove its `\include`. Also check that no other chapter cross-references it, and that the macros it defines locally (`\eint`, `\Ksgs`, and its `\renewcommand{\avg}`) are not used elsewhere.

### B3. Scope of the flux gate $g_f$

Ch. 2 §Gating sets $\mathsf R\to g_f\mathsf R$ and $\mathbf q\to g_f\mathbf q$. It does not say whether $f$ is also gated in
- $\langle P\rangle=\hat P(1-f)$, and
- $T_{\rm CG}$, which feeds the cooling and the gate/cooling model inputs.

**Recommended answer:** gate only the fluxes, and always use the predicted $f$ in the thermodynamics. For $f<f_{\rm cut}=10^{-4}$ the difference is below $10^{-4}$ anyway. Note that with the A1 code form, gating $\mathsf R$ means gating $\mathsf R^{\rm dev}$; the isotropic part stays in $\hat P$ automatically.

**Also note:** a hard gate switches fluxes on and off in time as cells cross $f_{\rm cut}$, which can inject grid-scale noise. A smooth gate costs nothing, for example a smoothstep in $\log_{10}f$ over $[10^{-5},10^{-4}]$. The a priori error table changes negligibly.

---

## C. Modelling assumptions to state explicitly

### C1. $k_{\rm sg}$ is diagnostic, not transported

In the Ch. 2 design, $f$ (and so $k_{\rm sg}$) is predicted afresh every stage from the instantaneous resolved state on a $3^3$ stencil.

- **No transport.** There is no equation for $\langle\rho\rangle k_{\rm sg}$: no production $\mathsf R:\nabla\tilde{\mathbf v}$, no dissipation $\varepsilon$, no transport.
- **Instant repartitioning.** The split between subgrid kinetic and thermal energy inside $\langle E\rangle$ is redone instantly by the model. Dissipation of $k_{\rm sg}$ into heat is therefore *implicit*.
- **No memory.** Turbulence advected into a cell from upstream is invisible unless the resolved state encodes it.

Many compressible LES closures instead carry the subgrid kinetic energy as a prognostic variable with its own transport equation. The notes should state that this design deliberately does not. The advantages:
- the closure needs no extra conserved field;
- $\langle E\rangle$ stays the true filtered total energy;
- $\langle P\rangle>0$ is guaranteed by construction.

The cost is the loss of memory. A later variant could add a prognostic $\langle\rho\rangle k_{\rm sg}$, with ML closures only for its production and dissipation.

### C2. Model inputs at inference

The gate and cooling models were trained on $\log T$ computed from the *true* $\langle P\rangle$. In the LES, $T_{\rm CG}=\hat T(1-f_{\rm pred})$. The error is about $f$: under 1% in the mixing layer, and negligible in the pure phases. The notes should still state the order of evaluation:
1. the R model gives $f$;
2. $f$ gives $T_{\rm CG}$;
3. $T_{\rm CG}$ feeds the gate, then the cooling model.

A cheap consistency option is to train the gate and cooling models on $T_{\rm CG}$ built from the R model's predicted $f$. This fits the A3 retraining. Once the scalar is gone, $T_{\rm CG}$ is the main phase indicator, so its consistency between training and inference matters more than before.

### C3. Filter width and transfer to other resolutions

The derivation holds for any $\Delta$, but the models do not.
- They see no length scale: gradients are per coarse cell, and $k_{\max}$ and $\rho$ are dimensional in code units.
- All data come from one run, one $\Delta=0.625$ pc, and one $\xi\approx16$.

The notes should say that the closure is trained for **one filter width and one flow**. If transfer is a goal, nondimensionalise the inputs: gradients $\times\Delta$, velocities by a local $\sqrt{k_{\max}}$, temperatures relative to the cooling-window edges. Then train across two or more of the existing coarse-grain factors $b\in\{8,16,32\}$ from the same fine run.

### C4. The initial transient and the choice of initial conditions

Ch. 2 §Distributions excludes $t<2.5$ Myr: the layer is not yet turbulent, and $\mathsf S$ is nearly degenerate. A coarse run started from the analytic $\tanh$ IC at $t=0$ therefore starts **outside the training distribution**.

The notes should say that a posteriori tests start from a *downsampled* fine snapshot at $t\ge2.5$ Myr. That snapshot carries $\langle E\rangle$ including $\langle\rho\rangle k_{\rm sg}$, so that $\hat P$ and $f$ are consistent from the first step. This is the `.cic` route in `PLAN_subgrid_ml.md` §7a; without a scalar it carries only $\langle\rho\rangle$, $\langle\mathbf m\rangle$ and $\langle E\rangle$.

The existing `kh3d_16x64x16` run (`trml-data.tex`, Table "five runs") is the natural no-model baseline: same grid, $\dot\Sigma_{\rm cool}=0.95\pm0.55$ against $1.42\pm0.22$. It should be named in the notes as the thing the closure must improve on.

---

## D. Suggested new sections, in order

1. **Ch. 1 §1.y "Form solved by the code"** (A1). $\hat P$, $\Delta\mathsf F_m=\mathsf R^{\rm dev}$, $\Delta F_E=\tilde{\mathbf v}\cdot\mathsf R^{\rm dev}+\mathbf q$, positivity of the total stress, floors.
2. **Ch. 1 §1.z "Discrete closure"** (A2). Face vs. volume averages, Riemann flux, implicit dissipation, and the solver-in-the-loop residual diagnostic.
3. **Ch. 2 §"Model inputs"** (A3, C2). Inputs built from $(\langle\rho\rangle,\langle\mathbf m\rangle,\langle E\rangle)$ only, with no scalar; the evaluation order $f\to T_{\rm CG}\to g\to z_{\mathcal C}$; why temperature alone identifies the phases in the TRML.
4. **Ch. 2 §"Symmetries of the models"** (A4). Galilean projection for every model, $O_h$ augmentation, reflection.
5. **Ch. 2 §"Time-step constraints"** (A5). $c_{\rm eff}$, the cooling time with the corrected rate, the $\mathbf q$ advection limit. Reword the "cannot be unphysical" claim.
6. **Ch. 2 §Gating:** add the scope statement (B3) and optionally the smooth gate.
7. **Ch. 2, a short statement on diagnostic $k_{\rm sg}$** (C1).
8. **Housekeeping:** remove the `\include`s of `03-ExpandingQ.tex`, `tbnn-theory.tex` and `hit-data.tex` (B2); use $m_u$ throughout; fix `CHANGES_TO_ATHENAK.md`.

### Effect on the implementation plan

- **A1** is already the plan's approach. The plan's correction "$\mathsf R-(\gamma-1)\rho k_{\rm sg}\mathbb 1$" is the same as $\mathsf R^{\rm dev}$ for $\gamma=5/3$. Keep the general-$\gamma$ form in code.
- **A3:** remove `s0` from the plan's packing vocabulary and from every model's inputs. The pgen must run with `nscalars = 0`:
  - drop the scalar writes in the ICs and in `constant_bcs`;
  - remove the `nscalars != 0` fatal check copied from `kh_radiative_cooling.cpp`;
  - make the `.cic` writer and loader skip scalars.
- **A3 + A4 + C2** together mean one retraining round for all three models, run sequentially on the GPU:
  - no `s0`;
  - Galilean projection in the gate and cooling models;
  - optionally, $T_{\rm CG}$ from the predicted $f$.

  The ONNX export path should be built against the retrained models.
- **A5:** the plan's `psrc->dtnew` should include $\Delta/c_{\rm eff}$ as well as the cooling and $\mathbf q$ limits.
- **A2 and C4:** the solver-in-the-loop residual is one extra use of the `.cic` IC path. Run one coarse step and compare with the next downsampled frame.
