# What AthenaK solves, and how the Chapter 2 data comes out of it

This note follows one chain from the continuous Euler equations to the plots in `theory/notes/Overleaf/images/chapter2/`:

```
continuous Euler + cooling
  → finite-volume cell averages                      (§2)
  → PLM reconstruction → HLLC numerical flux         (§4, §5)
  → SSP-RK2 in time, cooling as a source term        (§3, §6)
  → fine-grid snapshots  (256×1024×256, hydro_w .bin) (§8)
  → 16³ block coarse-graining → τ, k_sg, f, w, C     (§9)
  → z₀…z₈ targets, phase splits, figures             (§9)
```

Everything about AthenaK below was read from the source in `athenak/src/` (file and line references are given where it matters). The equations and the HLLC form were also checked numerically (§5.4).

---

## 1. The continuous equations

AthenaK's non-relativistic hydro module (`hydro/`) integrates the Euler equations in conservative form. With the ideal-gas equation of state and optically thin cooling (what the radiative KH run uses):

$$
\begin{aligned}
\partial_t \rho + \nabla\!\cdot(\rho\mathbf v) &= 0\\
\partial_t(\rho\mathbf v) + \nabla\!\cdot(\rho\mathbf v\mathbf v + P\,\mathsf I) &= 0\\
\partial_t E + \nabla\!\cdot\big[(E+P)\mathbf v\big] &= -\,n^2\Lambda(T) + n\Gamma
\end{aligned}
$$

$$
E = \underbrace{\frac{P}{\gamma-1}}_{e_\text{int}} + \tfrac12\rho|\mathbf v|^2,\qquad
P=(\gamma-1)\,e_\text{int},\qquad
T=\frac{\mu m_p}{k_B}\frac{P}{\rho},\qquad
n=\frac{\rho}{\mu m_p}.
$$

* $\gamma = 1.666667$, $\mu=0.62$ (`kh3d_256x1024x256.athinput`).
* The heating $\Gamma$ is the input `hrate`, which is 0 in this run, so the right-hand side is just $-n^2\Lambda(T)$. This is the same $\dot E = n^2\Lambda$ as eq-euler in Chapter 1 of the notes.
* One passive scalar is also advected (`nscalars = 1`). It plays no role in the Chapter 2 data.
* In the code, the stored conserved variables are $\mathbf u = (\rho,\ \rho v_x,\ \rho v_y,\ \rho v_z,\ E)$. The primitives `w0` are $(\rho, v_x, v_y, v_z, e_\text{int})$. The fifth primitive is the internal energy **density**, not the pressure: `IdealGasPressure(eint) = (γ-1)*eint` in `eos/eos.hpp`.

Written as a single conservation law with a source:

$$
\partial_t \mathbf U + \partial_x \mathbf F_x(\mathbf U) + \partial_y \mathbf F_y(\mathbf U) + \partial_z \mathbf F_z(\mathbf U) = \mathbf S(\mathbf U),
$$

$$
\mathbf F_x=\begin{pmatrix}\rho v_x\\ \rho v_x^2+P\\ \rho v_xv_y\\ \rho v_xv_z\\ (E+P)v_x\end{pmatrix},\qquad
\mathbf S=\begin{pmatrix}0\\0\\0\\0\\-n^2\Lambda(T)+n\Gamma\end{pmatrix}.
$$

$\mathbf F_y,\mathbf F_z$ follow by cyclic permutation. The code does exactly this: `ivx, ivy, ivz = IVX + (k, k+1, k+2) mod 3` in every Riemann solver.

---

## 2. Discretisation: finite volume

Divide space into cells of size $\Delta x\times\Delta y\times\Delta z$ (uniform; here $\Delta = 10/256 = 0.039\,$pc). Integrate the PDE over cell $(i,j,k)$ and divide by its volume. The **cell average**

$$
\bar{\mathbf U}_{ijk}(t)=\frac1{\Delta V}\int_{\text{cell}}\mathbf U\,dV
$$

obeys, exactly (divergence theorem),

$$
\frac{d\bar{\mathbf U}_{ijk}}{dt}
= -\frac{\hat{\mathbf F}^x_{i+\frac12}-\hat{\mathbf F}^x_{i-\frac12}}{\Delta x}
 -\frac{\hat{\mathbf F}^y_{j+\frac12}-\hat{\mathbf F}^y_{j-\frac12}}{\Delta y}
 -\frac{\hat{\mathbf F}^z_{k+\frac12}-\hat{\mathbf F}^z_{k-\frac12}}{\Delta z}
 +\bar{\mathbf S}_{ijk},
$$

where $\hat{\mathbf F}^x_{i+\frac12}$ is the flux $\mathbf F_x$ averaged over the face at $x_{i+1/2}$ (and over time). Nothing is approximated yet. **The whole numerical scheme is the choice of how to estimate the face fluxes $\hat{\mathbf F}$ from the cell averages $\bar{\mathbf U}$.** That estimate is the *numerical flux*.

Two facts about this structure matter later:

1. **Conservation.** Each interior face flux appears once with $+$ sign (in the cell on its left) and once with $-$ (in the cell on its right), so mass, momentum and energy are conserved to round-off. The same cancellation is what makes block coarse-graining exact (§9.2).
2. **Unsplit update.** The three directional flux differences are summed into one divergence from the *same* stage state; there is no dimensional splitting and no transverse-flux correction. See `Hydro::RKUpdate` in `hydro/hydro_update.cpp`:

```cpp
divf(i)  = (flx1(i+1) - flx1(i)) / dx1;       // + dF2/dx2 + dF3/dx3
u0 = gam0*u0 + gam1*u1 - beta_dt*divf;        // one SSP-RK stage
```

The cell-average $\bar{\mathbf U}$ is what's stored in `u0`. The `.bin` outputs hold the primitives obtained from it by `ConToPrim`:

$$
e_\text{int} = E - \tfrac12\rho|\mathbf v|^2 = E-\frac{|\rho\mathbf v|^2}{2\rho}
$$

(`eos/ideal_c2p_hyd.hpp`, `w.e = u.e - e_k`), i.e. `dens`, `velx`, `vely`, `velz`, `eint`. These are treated as cell-centred values in the analysis.

---

## 3. Time integration: SSP-RK2 and the timestep

Write the semi-discrete system as $d\bar{\mathbf U}/dt = \mathcal L(\bar{\mathbf U})$, where $\mathcal L$ is the flux divergence plus the source term. `integrator = rk2` (`driver/driver.cpp`) uses the coefficients $(\gamma_0,\gamma_1,\beta)=(0,1,1)$ for stage 1 and $(\tfrac12,\tfrac12,\tfrac12)$ for stage 2, with `u1` a copy of $\bar{\mathbf U}^n$ taken at the start of the step (`CopyCons`):

$$
\begin{aligned}
\mathbf U^{*} &= \mathbf U^n + \Delta t\,\mathcal L(\mathbf U^n)\\
\mathbf U^{n+1} &= \tfrac12\mathbf U^n+\tfrac12\mathbf U^{*}+\tfrac12\Delta t\,\mathcal L(\mathbf U^{*})
\end{aligned}
$$

This is Heun's method, second-order and strong-stability-preserving, so it keeps whatever monotonicity the spatial scheme has under the CFL limit.

**Task order in each stage** (`hydro/hydro_tasks.cpp`):

```
CopyCons → Fluxes → (flux correction across refinement levels) → RKUpdate
         → HydroSrcTerms → (comm / BCs) → ConToPrim → NewTimeStep
```

So the order is: fluxes from the current primitives `w0`, RK update of $\mathbf U$ with the flux divergence, then the source terms added to the *same* `u0` (computed from `w0`, as the comment in `HydroSrcTerms` insists), then conservative → primitive for the next stage.

**Timestep.** Each cell and direction gives $\Delta x_d/(|v_d|+c_s)$ with $c_s=\sqrt{\gamma P/\rho}$, and the step is

$$
\Delta t = \text{cfl}\cdot\min_{\text{cells},\,d}\frac{\Delta x_d}{|v_d|+c_s},\qquad \text{cfl}=0.4,
$$

growing by at most $2\times$ per step (`hydro/hydro_newdt.cpp`, `mesh/mesh.cpp:603–614`). The source-term timestep `psrc->dtnew` is initialised to a huge value and, as far as I could find, never set for ISM cooling. So **there is no cooling-time limit on $\Delta t$**; cooling is explicit and only the CFL constraint bounds the step.

---

## 4. Reconstruction: from cell averages to face states (PLM)

A Riemann solver needs a *left* state $\mathbf w_L$ and a *right* state $\mathbf w_R$ at each face. The cell average gives only one number per cell, so a piecewise-linear profile is built in each cell. `reconstruct = plm` (`reconstruct/plm.hpp`) acts on **primitives** $(\rho,\mathbf v,e_\text{int})$ one variable at a time, one direction at a time.

For cell $i$ and variable $q$:

$$
\delta_- = q_i - q_{i-1},\qquad \delta_+ = q_{i+1}-q_i,
$$

$$
\Delta q_i=\begin{cases}
\dfrac{\delta_-\delta_+}{\delta_-+\delta_+} & \delta_-\delta_+>0\\[2mm]
0 & \text{otherwise (local extremum)}
\end{cases}
$$

$$
q^{L}_{i+\frac12}=q_i+\Delta q_i,\qquad q^{R}_{i-\frac12}=q_i-\Delta q_i .
$$

The $\Delta q_i$ above is **half** of the van Leer harmonic slope $2\delta_-\delta_+/(\delta_-+\delta_+)$, which is what you would multiply by $\pm\tfrac12$ to get a face value. It reduces to the first-order (piecewise-constant) scheme at extrema, which is what keeps the scheme non-oscillatory (TVD). Away from extrema it is second-order accurate.

The face $i+\tfrac12$ therefore gets $\mathbf w_L=\mathbf w^{L}_{i+\frac12}$ (from cell $i$) and $\mathbf w_R=\mathbf w^{R}_{i+\frac12}$ (from cell $i+1$). In `hydro_fluxes.cpp`, one kernel fills the global buffers `wl3d`/`wr3d` for the whole block and a second kernel solves the Riemann problem at each face reading from them.

The fifth reconstructed variable is $e_\text{int}$, so the face pressure is $P=(\gamma-1)e_\text{int}$ and positivity of $e_\text{int}$ is what's needed for positive $P$. Ghost zones: `nghost = 2`, enough for PLM's $\pm1$ stencil at every face.

---

## 5. The numerical flux: HLLC

### 5.1 The Riemann problem at a face

At each face the two reconstructed states define a 1D problem (normal direction $x$ for the $x$-face):

$$
\partial_t\mathbf U+\partial_x\mathbf F(\mathbf U)=0,\qquad
\mathbf U(x,0)=\begin{cases}\mathbf U_L&x<0\\\mathbf U_R&x>0\end{cases}
$$

Its exact solution is self-similar, with three waves: a left wave (shock or rarefaction) with speed $S_L$, a **contact** (and shear for the transverse momenta) moving at $S_*$, and a right wave $S_R$. The numerical flux is the flux along the ray $x/t=0$, $\mathbf F(\mathbf U(0))$.

HLLC (Toro, Spruce & Speares 1994) approximates the fan by three constant states separated by $S_L, S_*, S_R$:

```
        S_L          S_*          S_R
  U_L  │   U*_L     │   U*_R     │  U_R
```

### 5.2 Where the flux formulas come from

Across each wave, the Rankine–Hugoniot condition $\mathbf F_{\!K}^*-\mathbf F_K=S_K(\mathbf U_K^*-\mathbf U_K)$ holds. Across the contact, $u$ and $P$ are continuous: $u^*=S_*$, $P^*_L=P^*_R=P^*$. Solving for the star states gives

$$
\mathbf U_K^*=\rho_K\frac{S_K-u_K}{S_K-S_*}
\begin{pmatrix}1\\ S_*\\ v_K\\ w_K\\ \dfrac{E_K}{\rho_K}+(S_*-u_K)\Big(S_*+\dfrac{P_K}{\rho_K(S_K-u_K)}\Big)\end{pmatrix},\qquad K\in\{L,R\},
$$

with the contact speed and star pressure

$$
S_*=\frac{P_R-P_L+\rho_Lu_L(S_L-u_L)-\rho_Ru_R(S_R-u_R)}{\rho_L(S_L-u_L)-\rho_R(S_R-u_R)},\qquad
P^*=P_L+\rho_L(u_L-S_L)(u_L-S_*).
$$

The HLLC flux is then

$$
\hat{\mathbf F}=
\begin{cases}
\mathbf F_L & S_L\ge0\\
\mathbf F_L+S_L(\mathbf U^*_L-\mathbf U_L)\equiv\mathbf F^*_L & S_L<0\le S_*\\
\mathbf F_R+S_R(\mathbf U^*_R-\mathbf U_R)\equiv\mathbf F^*_R & S_*<0<S_R\\
\mathbf F_R & S_R\le0
\end{cases}
$$

The outer cases are plain upwinding when the whole fan moves one way. The middle two are where the physics is: the contact discontinuity is resolved exactly, since $\rho$, and the transverse velocities $v,w$ in the momentum flux, jump across $S_*$ with $P^*$ continuous. That matters for this project, since the KH shear layer is a *contact + shear* feature that HLLE would smear.

### 5.3 What the AthenaK code does (`hydro/rsolvers/hllc_hyd.hpp`)

**(a) Wave speeds from a pressure estimate** (the "PVRS" estimate, Toro §10.5.2). A cheap middle-state pressure is built from the face states,

$$
P_\text{mid}=\tfrac12\Big[P_L+P_R+(u_L-u_R)\,\tfrac14(\rho_L+\rho_R)(a_L+a_R)\Big],\qquad a=\sqrt{\gamma P/\rho},
$$

$$
q_K=\begin{cases}1&P_\text{mid}\le P_K\\ \sqrt{1+\dfrac{\gamma+1}{2\gamma}\Big(\dfrac{P_\text{mid}}{P_K}-1\Big)}&P_\text{mid}>P_K\end{cases}
$$

$$
S_L=u_L-a_Lq_L,\qquad S_R=u_R+a_Rq_R .
$$

A rarefaction ($q=1$) gets the acoustic speed, and a shock gets the faster shock speed.

**(b) Contact speed and pressure.** With $t_K=P_K+(u_K-S_K)\rho_Ku_K$ and $m_L=\rho_L(u_L-S_L)$, $m_R=-\rho_R(u_R-S_R)$,

$$
S_*=\texttt{am}=\frac{t_L-t_R}{m_L+m_R},\qquad
P^*=\texttt{cp}=\max\!\Big[\frac{m_Lt_R+m_Rt_L}{m_L+m_R},\,0\Big],
$$

which are the closed forms of §5.2 (the clamp $\ge0$ is a floor).

**(c) Fluxes in a shifted frame.** The solver forms, for the clipped speeds $b_-=\min(S_L,-10^{-20})$ and $b_+=\max(S_R,10^{-20})$,

$$
\mathbf D_L=\mathbf F_L-b_-\mathbf U_L,\qquad \mathbf D_R=\mathbf F_R-b_+\mathbf U_R,
$$

e.g. `fl_d = ρ_L(u_L − b_-)`, `fl_mx = ρ_L(u_L − b_-)u_L + P_L`, `fl_e = E_L(u_L − b_-) + P_L u_L`.

**(d) Weights and assembly.** For $S_*\ge0$ the weights are `qc = am/(am−b_-)`, `qd = 0`, `qe = −b_-/(am−b_-)`; for $S_*<0$ they are `qc = 0`, `qd = −am/(b_+−am)`, `qe = b_+/(b_+−am)`. Then

$$
\hat{\mathbf F}=\texttt{qc}\,\mathbf D_L+\texttt{qd}\,\mathbf D_R+\texttt{qe}\,P^*\,\big(0,\,1,\,0,\,0,\,S_*\big)^{\!T}.
$$

Written out for $S_*\ge0$:

$$
\hat{\mathbf F}=\frac{S_*(\mathbf F_L-S_L\mathbf U_L)-S_L\,P^*\,(0,1,0,0,S_*)^T}{S_*-S_L}\;=\;\mathbf F_L+S_L(\mathbf U^*_L-\mathbf U_L).
$$

The $b_\pm$ clipping implements the outer upwind cases. If $S_L>0$ then $b_-=-10^{-20}\approx0$, $\mathbf D_L\to\mathbf F_L$, and the weights go to `qc≈1, qe≈0`, so $\hat{\mathbf F}\approx\mathbf F_L$.

**(e) Passive scalars** are not part of HLLC; they are upwinded on the sign of the mass flux (`hydro_fluxes.cpp`):

$$
\hat F_s=\hat F_\rho\cdot\begin{cases}s_L&\hat F_\rho\ge0\\s_R&\hat F_\rho<0\end{cases}
$$

### 5.4 Check

I checked (c)–(d) against the textbook $\mathbf F_L+S_L(\mathbf U_L^*-\mathbf U_L)$ numerically: the AthenaK form, ported line by line to Python, against the star-state formulas of §5.2, for 50 000 random pairs of states. Whenever the speeds are ordered ($S_L<S_*<S_R$) and the unclamped $P^*>0$, the two agree to $6\times10^{-14}$ (about 92% of random pairs). The remaining cases are the robustness fall-backs, i.e. a clamped $P^*$ or inconsistent PVRS speeds with extreme random jumps; there the code deliberately deviates from the textbook flux. The script is at `/tmp/claude-1007/-home-sasi-Projects-SubgridCGMModel/7a295586-3d7c-46c9-9363-00b02e5a60f2/scratchpad/hllc_check.py`.

### 5.5 Other solvers in the same directory

`rsolvers/` also contains `hlle_hyd.hpp`, `llf_hyd.hpp` (local Lax–Friedrichs), `roe_hyd.hpp` and `advect_hyd.hpp`. All have the form $\hat{\mathbf F}=\tfrac12[\mathbf F_L+\mathbf F_R]-\tfrac12\,\mathsf D\,[\mathbf U_R-\mathbf U_L]$ with different dissipation matrices $\mathsf D$. LLF uses a single scalar speed. HLLE uses $S_L,S_R$ only and averages over the whole fan, with no contact. Roe linearises. HLLC restores the contact to HLLE. The runs use `rsolver = hllc`.

---

## 6. The cooling source term

`srcterms/srcterms.cpp`, `SourceTerms::ISMCooling`, called from `Hydro::HydroSrcTerms` right after `RKUpdate`:

$$
E \;\mathrel{-}=\; \beta\,\Delta t\;\rho\,\big[\rho\,\lambda(T)-\gamma_{\rm heat}\big],\qquad T=\texttt{temp\_unit}\,\frac{(\gamma-1)e_\text{int}}{\rho}
$$

where $\lambda=\Lambda/\text{cooling\_unit}$ is the cooling function in code units. In cgs units this is $-n^2\Lambda(T)$, since $\rho\cdot\rho\lambda\leftrightarrow n^2\Lambda$ once the units are applied. Points to note:

* It uses the **primitives of the current stage** (`w0`), and adds into `u0` after the flux update. So within an RK stage, the energy equation is $\mathcal L_E=-\nabla\!\cdot[(E+P)\mathbf v]-n^2\Lambda$, evaluated on the same stage state. This is a method-of-lines treatment of the source, not operator splitting.
* It only touches the energy. Density and momentum are unchanged.
* $\Lambda(T)$ is `ISMCoolFn` in `srcterms/ismcooling.hpp`: Koyama & Inutsuka (2002) below $10^{4.2}$ K, the SPEX table of Schure et al. (2009) for $4.2<\log T<8.15$ (linear interpolation in $\log T$), and the CGOLS power law above. **This fork adds one change**: $\Lambda=0$ for $T\le1.05\times10^4$ K and $T>0.95\times10^6$ K (`CHANGES_TO_ATHENAK.md`), so the cold and hot phases do not cool away from their initial state. These are exactly the phase boundaries used in the notes, and the notes call this curve $\Lambda_\text{cut}$ (the uncut one is $\Lambda_\text{full}$; see `cooling_curve.py`).

---

## 7. One full step, in pseudocode

```
dt = cfl · min  dx_d / (|v_d| + c_s)                       # §3

for stage in (1, 2):                                       # SSP-RK2
    # w0 = (rho, v, e_int) of the current stage state, ghosts filled
    for d in (x, y, z):
        PLM-reconstruct w0 → (w_L, w_R) at every d-face     # §4
        F̂_d = HLLC(w_L, w_R)  at every d-face              # §5
        scalars: F̂_s = F̂_rho · (s_L or s_R)
    divF = Σ_d (F̂_d[face+] − F̂_d[face−]) / dx_d            # §2
    U ← γ0·U + γ1·U^n − β·dt·divF                           # §3
    E ← E − β·dt·ρ·(ρ·λ(T(w0)) − Γ)                         # §6, uses w0
    apply BCs; w0 ← ConToPrim(U); recompute dt candidate
```

**Boundary conditions** (athinput): periodic in $x$ and $z$; at the lower $y$ face `reflect`, at the upper $y$ face `user` (`constant_bcs` in `pgen/kh_radiative_cooling.cpp`, which holds the state fixed).

---

## 8. The run behind Chapter 2

Everything in Chapter 2 comes from a single run, `kh3d_256x1024x256` (`/data/sasi/simulation_outputs/kh3d_256x1024x256/`; the deck was reconstructed in `athinputs/kh_fid3D_64_cool.athinput` with the resolution changed to this one).

| | |
|---|---|
| Problem generator | `pgen/kh_radiative_cooling.cpp`, `iprob = 2` (tanh shear layer plus white-noise $v_y$ at the interface) |
| Grid | $256\times1024\times256$ cells, box $10\times40\times10$ pc, MeshBlocks $32\times64\times32$ |
| Scheme | `plm` + `hllc`, `rk2`, cfl 0.4, `fofc = false`, $\gamma=1.666667$ |
| Source | `ism_cooling = true`, `hrate = 0` |
| Cold / hot phase | $\rho=0.1$ / $0.001$ (code units), $T\approx10^4$ / $10^6$ K at one pressure; $P/k\approx1625$ K cm$^{-3}$ |
| Shear | $v_x=-2.82$ (cold), $+28.18$ (hot), $\Delta v\approx30$ km/s |
| Units | 1 pc, 1 Myr, $\rho_\text{unit}=1\,m_p\,\text{cm}^{-3}$ |
| Output | `<output2> file_type=bin, variable=hydro_w, dt=0.08` Myr → `dens, velx, vely, velz, eint` per snapshot; `tlim = 10` Myr |

The analysis keeps the snapshots from $t\ge2.5$ Myr (`T_MIN` in `notes_style.py`), the first being the initial transient. That leaves 94 snapshots. They are read through `vis_athenak.SimulationData`, with `Units.from_params` supplying the cgs conversion.

---

## 9. From the fine-grid snapshots to the Chapter 2 figures

### 9.1 What is being measured

The fine run is the "truth", the closest thing to $\rho,\mathbf v,P$ at all scales. The notes ask: if I only knew the **coarse** fields $\bar\rho,\bar{\mathbf m},\bar E$ on a grid $b=16$ times coarser, what do I have to add to the Euler equations (eq-final, Chapter 1) to keep them exact? The answer is the three unknowns $\mathsf R$ (momentum flux), $\mathbf Q$ (energy flux) and $C$ (cooling).

### 9.2 Why block-averaging is the right operation (the telescoping argument)

Take a $b^3$ block of fine cells and sum the finite-volume update of §2 over it. In the $x$ direction,

$$
\sum_{i=1}^{b}\frac{\hat{\mathbf F}^x_{i+\frac12}-\hat{\mathbf F}^x_{i-\frac12}}{\Delta x}
=\frac{\hat{\mathbf F}^x_{b+\frac12}-\hat{\mathbf F}^x_{\frac12}}{\Delta x}
$$

because every interior flux cancels, the same conservation property as in §2. So the block average $\bar{\mathbf U}_I=\frac1{b^3}\sum\bar{\mathbf U}_{ijk}$ obeys *exactly* the coarse finite-volume equation, with a coarse face flux equal to the **average of the $b^2$ fine numerical fluxes lying on the coarse face**. A coarse solver would instead have to evaluate its own flux from the coarse states $\bar{\mathbf U}_I,\bar{\mathbf U}_{I+1}$. The mismatch between those two is what the closure has to supply.

The notes work this out in the continuous filtered form (Chapter 1, `sec:filtering`): with a top-hat kernel $G_\Delta$ (here, the $16^3$ box) and Favre average $\widetilde f=\overline{\rho f}/\bar\rho$,

$$
\mathsf R_{ij}=\bar\rho\,\tau(v_i,v_j)=\bar\rho\,\widetilde{\delta v_i\delta v_j},\qquad
\mathbf Q=\tilde{\mathbf v}\cdot\mathsf R+\mathbf q,\quad q_i=\bar\rho\,\widetilde{\psi\,\delta v_i},\ \psi=h+\tfrac12|\delta\mathbf v|^2,
$$

$$
C=\overline{n^2\Lambda(T)}-\bar n^2\Lambda(T_\text{CG}),\qquad T_\text{CG}=\frac{\mu m_p}{k_B}\frac{\bar P}{\bar\rho},
$$

with the resolved flux $\bar{\mathbf m}\bar{\mathbf m}/\bar\rho+\bar P\,\mathsf I$ (and $(\bar E+\bar P)\bar{\mathbf m}/\bar\rho$ in the energy equation) subtracted off.

### 9.3 Step by step in the code

The per-cell quantities come from three scripts in `theory/notes/plotting_scripts/`. All three read the fine fields through `cell_cache.py`, which computes and caches one `.npz` per frame (`cache/<name>_b16_v<N>/frame_#####.npz`), so replotting does not re-read the run.

1. **Read a frame.** `cell_cache.read_frame` loads `dens, velx, vely, velz, eint` as float64.
2. **Cut into blocks.** With $b=16$, the $256\times1024\times256$ grid becomes $16\times64\times16=16\,384$ coarse cells per frame. With 94 frames that gives the $1\,540\,096$ cells quoted in the text of Chapter 2. `blocks(f, b)` reshapes a field to $(Z,b,Y,b,X,b)$ and `block_mean(f, b)` averages over the $b^3$ cells. Since every fine cell has the same volume, this is exactly the discrete top-hat filter $\overline{f}$.
3. **Coarse resolved fields.** $\bar\rho=\langle\rho\rangle$, Favre velocity $\tilde v_i=\langle\rho v_i\rangle/\bar\rho$, $\bar e_\text{int}=\langle e_\text{int}\rangle$ and $\bar P=(\gamma-1)\bar e_\text{int}$. The temperature is derived from the coarse fields ($T_\text{CG}$), **not** averaged.
4. **Central moments, computed directly.** The fluctuation $\delta v_i=v_i-\tilde v_i$ is formed on the fine cells and then averaged, giving $\tau_{ij}=\langle\rho\,\delta v_i\delta v_j\rangle/\bar\rho$. This is not computed as $\langle\rho v_iv_j\rangle-\langle\rho v_i\rangle\langle\rho v_j\rangle/\bar\rho$, which is the same thing algebraically but can lose positive semi-definiteness to round-off (see the docstring of `momentum_tensor.coarse_cells`).

The scripts and what each produces:

| Script | Per-cell quantities | Chapter 2 outputs |
|---|---|---|
| `momentum_tensor.py` | $\tau_{ij}$, $k_\text{sg}=\tfrac12\operatorname{Tr}\tau$, $e_\text{th}=\bar e_\text{int}/\bar\rho$, $k_\text{max}=e_\text{th}+k_\text{sg}$, $f=k_\text{sg}/k_\text{max}$ | `ksg_fraction`, `rshape_targets`, `rshape_barycentric`, `rshape_anisotropy`, `rshape_orientation` |
| `energy_flux.py` | $\psi=\gamma e_\text{int}/\rho+\tfrac12|\delta\mathbf v|^2$, $\psi_c=\widetilde\psi$, $\mathbf q$, $\mathbf w=\tau^{-1/2}\widetilde{\delta\psi\,\delta\mathbf v}/\psi_c$, $B^2=\tau(\psi,\psi)/\psi_c^2$ | `w_distribution`, `w_vs_f`, `w_vs_targets`, `w_scaling`, `w_targets` |
| `cooling_comparision.py` | $\overline{n^2\Lambda_\text{cut}(T)}$, resolved $\bar n^2\Lambda(T_\text{CG})$, gate $g$, $z_C=\ln r$ | `cooling_correction`, `cooling_zc`, `cooling_zc_cases`, `cooling_zc_factors`, `cooling_targets`, and the `.tex` tables |
| `flux_gate.py` | divergences of $\mathsf R,\mathbf Q$ (gated on $f$) against the resolved flux divergences, on the coarse grid | `flux_gate`, `flux_gate_cut.tex` |
| `cooling_curve.py` | $\Lambda_\text{full}$ against the cut curve of §6 | `cooling_curve` |
| `trml_sharma_figs.py`, `trml_convergence.py` | Sharma et al. (2025) reductions of the same run (via `theory/sharma+2025/`) | `trml_slices`, `trml_profiles`, `trml_pdfs`, `trml_energy`, `trml_convergence` |

Here $h=\gamma e_\text{int}/\rho$ is the specific enthalpy (since $h=e+P/\rho$ and $P/\rho=(\gamma-1)e_\text{int}/\rho$), matching $h=\frac{\gamma}{\gamma-1}P/\rho$ in the notes.

### 9.4 The network targets $z_0\ldots z_8$ and $z_C$

These are the inverse of the output maps of Table `tab:outputs` in Chapter 2, applied to the measured $\tau,f,\mathbf w,C$:

| Target | Definition from the coarse cell | Script |
|---|---|---|
| $z_0$ | $\text{logit}\,f=\ln\frac{f}{1-f}$ (inverse of $f=\sigma(z_0)$) | `momentum_tensor.py` |
| $z_1\ldots z_5$ | $z_k=\big\langle\log\mathsf S-\tfrac13\operatorname{Tr}(\log\mathsf S)\,\mathsf I,\ \mathbb E^{(k)}\big\rangle_F$ with $\mathsf S=\tau/\operatorname{Tr}\tau$ | `momentum_tensor.py` |
| $z_6,z_7,z_8$ | $\mathbf z=\operatorname{arsinh}\!\big(|\mathbf w|/\sqrt f\big)\,\hat{\mathbf w}$ (inverse of $\mathbf w=\sqrt f\sinh|\mathbf z|\,\hat{\mathbf z}$) | `energy_flux.py` (`sinh` map near line 277) |
| $z_C$ | $\ln r$, $r=\overline{n^2\Lambda_\text{cut}}\,/\,[\bar n^2\Lambda_\text{full}(T_\text{CG})]$, only where the gate $g=1$ | `cooling_comparision.py` |

Details that matter when reading the figures:

* **Degenerate cells.** $\mathbf w$ needs $\tau^{-1/2}$, so it is NaN where the smallest eigenvalue of $\mathsf S$ is below $10^{-10}$ (`S_MIN`). No cell from $t\ge2.5$ Myr is degenerate, which is why that frame cut was chosen.
* **Phase split** (`momentum_tensor.PHASES`, using $T_\text{CG}$): hot $\ge0.95\times10^6$ K; mixing $1.05\times10^4$–$0.95\times10^6$ K; cold $<1.05\times10^4$ K. These are the same thresholds as the $\Lambda$ cut-offs of §6.
* **Units.** Energies are converted to cgs ($\text{erg g}^{-1}$, plotted in km² s⁻²) with `Units.from_params(sim.params).velocity_cgs**2`. The dimensionless quantities ($f$, $\mathbf w$, $B$) are not converted.
* **Bounds checked in the data.** $|\mathbf w|\le B$ (Cauchy–Schwarz, eq:bound_on_w) holds to round-off in every cell, because all moments are central moments about the same Favre mean.

### 9.5 Regenerating

```bash
source venv/bin/activate
python theory/notes/plotting_scripts/momentum_tensor.py   # --b 16 --every 1 [--recompute]
python theory/notes/plotting_scripts/energy_flux.py
python theory/notes/plotting_scripts/cooling_comparision.py
python theory/notes/plotting_scripts/flux_gate.py          # needs the energy_flux cache
```

A change to what a script stores per cell requires bumping its `CACHE_VERSION`, which starts a fresh cache directory instead of mixing old and new frames.

### 9.6 Caveats, where the numerical flux and the coarse-graining part ways

* **The fine run is itself under-resolved turbulence.** HLLC with PLM has built-in numerical dissipation, strongest at the grid scale and in sharp gradients. The notes' $\mathsf R$ and $\mathbf Q$ are computed from the fine-grid *cell values*, which already contain that dissipation's effect on the fields. But they do not contain the numerical flux's own dissipative contribution to the fine-scale fluxes, since $\tau_{ij}$ is built from $\delta\mathbf v$ and the physical flux function, not from $\hat{\mathbf F}$. The coarse equation that is *exactly* conserved by the code (§9.2) has face-averaged $\hat{\mathbf F}$, which differs from "$\mathbf F(\mathbf U)$ averaged over the block volume" by this dissipation plus the face-versus-volume averaging difference. Both are second-order small in the fine $\Delta$ but not zero; to my knowledge nothing in the notes accounts for them, so the data defines the *filtered-continuous* closure, not the exact discrete one.
* **Filtering in space only.** The cell values are at the output times; nothing is averaged in time, and the RK2 sub-stages are not seen.
* **Cooling is noisy at the grid scale.** The explicit source (§6) uses cell values of $T$, so $\overline{n^2\Lambda(T)}$ in a block is a sum of fine-cell evaluations. That is the quantity $C$ corrects for. It includes the sharp cut-offs of $\Lambda_\text{cut}$ (any fine cell crossing $1.05\times10^4$ K switches cooling off), which is why the notes need the gate $g$ in `cooling_comparision.py`.
