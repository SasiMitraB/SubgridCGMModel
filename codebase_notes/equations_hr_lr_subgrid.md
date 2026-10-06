# Equations solved in `run_variable_box_subgrid.sh`

The pipeline runs three AthenaK simulations of the same 2D radiative Kelvin–Helmholtz
mixing layer. All three solve the same equations, the compressible Euler equations
plus passive scalars. They differ only in **grid resolution** and in **how the
radiative-cooling sink $\dot{\mathcal{E}}$ is computed**.

| Run     | Build / pgen                              | Cell size $\Delta x$                     | Start                                             | Cooling term                                         |
|---------|-------------------------------------------|------------------------------------------|---------------------------------------------------|------------------------------------------------------|
| HR      | `builds/hr_build_gpu`, `kh_radiative_cooling` | 0.0390625 pc                             | $t=0$, analytic KH profile (`iprob=1`)            | `ism_cooling = true`, $\Lambda(T)$ evaluated per cell |
| LR      | `builds/hr_build`, `kh_radiative_cooling`     | `CELL_SIZE` (0.625 or 1.25 pc) = DS × HR | $t=5$ Myr, coarse-grained HR snapshot (`iprob=2`) | `ism_cooling = true` on the coarse cell mean         |
| Subgrid | `builds/subgrid_model`, `subgrid`             | `CELL_SIZE`                              | $t=5$ Myr, coarse-grained HR snapshot (`iprob=2`) | `ism_cooling = false`; CNN-predicted sub-cell $T$-PDF |

## 1. Common hydrodynamics (all three runs)

$$
\begin{aligned}
\partial_t \rho + \nabla\cdot(\rho\mathbf{v}) &= 0 \\
\partial_t (\rho\mathbf{v}) + \nabla\cdot(\rho\mathbf{v}\mathbf{v} + P\,\mathsf{I}) &= 0 \\
\partial_t E + \nabla\cdot\big[(E+P)\mathbf{v}\big] &= -\dot{\mathcal{E}} \\
\partial_t (\rho s_k) + \nabla\cdot(\rho s_k \mathbf{v}) &= 0
\end{aligned}
$$

where $E = \tfrac{1}{2}\rho v^2 + P/(\gamma-1)$ and $\gamma = 5/3$ (ideal gas). Temperature is
$T = \dfrac{\mu m_u}{k_B}\dfrac{P}{\rho}$ with $\mu = 0.62$, and $n = \rho/(\mu m_u)$.

- **Passive scalars $s_k$.** HR carries 1: the tracer, which is 1 in cold gas and 0 in hot gas.
  LR and Subgrid carry 2: the tracer and the cold-gas mass fraction. Both come from the
  mass-weighted coarse-graining in `data/downsample_ic.py`. Neither has a source term.
- **Numerics.** RK2 time integration, PLM reconstruction, HLLC Riemann solver, CFL 0.4.
- **Units.** 1 pc, 1 Myr, and density in $m_p\,\mathrm{cm^{-3}}$. Heating `hrate = 0`, so $\Gamma = 0$.
- **Boundaries.** $x_1$ is periodic. Inner $x_2$ is reflecting. Outer $x_2$ is fixed to the hot
  phase ($\rho_{\rm hot}$, $P_0$, $v_{x,\rm hot}$) with outflow $v_y$.
- **Timestep.** $\Delta t = \min(\Delta t_{\rm CFL},\ \Delta t_{\rm cool})$, where
  $\Delta t_{\rm cool} = \min_{\rm cells} e_{\rm int}/|\dot{\mathcal{E}}|$.

### Initial condition (HR, `iprob=1`)

The interface sits at $y_c = x_{2,\min} + 0.5\,L_y$:

$$
\rho = \bar\rho - \Delta\rho\,\tanh\!\frac{y-y_c}{a}, \qquad
v_x = \bar v + \Delta v\,\tanh\!\frac{y-y_c}{a}, \qquad P = P_0,
$$

$$
v_y = -2A\,\Delta v\,\exp\!\left[-\left(\frac{y-y_c}{\sigma}\right)^2\right]\sum_{n\in\{5,10,18,25,32\}}\sin\frac{2\pi n x}{L_x}.
$$

Here $\bar\rho,\Delta\rho$ are the mean and half-difference of $\rho_{\rm cold}=0.1$ and $\rho_{\rm hot}=10^{-3}$,
and $\bar v,\Delta v$ are the same for $v_{x,\rm hot}$ and $v_{x,\rm cold}$. $A = 0.05$. The widths
$a$ and $\sigma$ scale with the box height, $a = 0.125\,(L_y/20)$ and $\sigma = 0.5\,(L_y/20)$.

## 2. HR and LR cooling (resolved / cell-mean)

$$
\dot{\mathcal{E}}_{\rm HR,LR} = n^2\,\Lambda(T), \qquad T = T(\rho, P)\ \text{of the cell}.
$$

$\Lambda(T)$ is `ISMCoolFn`: the SPEX table from Schure et al. (2009), with Koyama & Inutsuka (2002)
below $10^{4.2}$ K. It is set to zero for $T \le 1.05\times10^4$ K and $T > 0.95\times10^6$ K.

The equation is identical for HR and LR. In LR, however, $\rho$ and $T$ are coarse-cell means over
$\mathrm{DS}^2$ HR cells (DS = 16 or 32), so $n^2\Lambda(T)$ misses the unresolved mixing-layer
temperatures. In general $\langle n^2\Lambda(T)\rangle \ne \bar n^2\Lambda(\bar T)$.

## 3. Subgrid cooling (CNN-predicted temperature PDF)

The CNN takes each $16\times8$ or $8\times4$ tile of coarse fields $(\bar\rho, \bar T, \bar v_x, \bar v_y, \bar s)$ and
predicts a sub-cell temperature PDF $p_i(\mathbf{x})$ over $N$ log-spaced bins $T_i \in [10^3, 10^7]$ K,
with $\sum_i p_i = 1$. It assumes the sub-cell gas is **isobaric**, $n_i = P/(k_B T_i)$, which gives

$$
\dot{\mathcal{E}}_{\rm SG} = \sum_i p_i\, n_i^2\,\Lambda(T_i)
= \left(\frac{P}{k_B}\right)^2 \sum_i p_i\,\frac{\Lambda(T_i)}{T_i^2}.
$$

- Here $\Lambda$ is the same curve, but it is masked to the active window
  $\log_{10}T \in [4.1, 5.9]$ (`LOGT_ACTIVE_START/END`).
- The source is applied only to the energy. The mass, momentum, and scalar sources are zero.
- The energy update has a floor, $e_{\rm int}^{\rm new} = \max\!\big(0.05\,e_{\rm int},\ e_{\rm int} - \Delta t\,\dot{\mathcal{E}}_{\rm SG}\big)$
  (`subgrid.cpp`).
- $\Delta t_{\rm cool} = \min e_{\rm int}/\dot{\mathcal{E}}_{\rm SG}$ is returned to AthenaK through
  `psrc->dtnew` (`user_cooling = true`).

**Sources:** `athenak/src/srcterms/srcterms.cpp` (`ISMCooling`), `athenak/src/srcterms/ismcooling.hpp`,
`athenak/src/pgen/kh_radiative_cooling.cpp`, `athenak/src/pgen/subgrid.cpp`,
`builds/subgrid_model/src/source_module.py`, `models/conv_nn/pdf_cnn.py` (`isobaric_emissivity_from_pdf`, `lambda_cool`).
