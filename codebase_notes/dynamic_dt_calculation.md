# Calculating the $\Delta t$ dynamically

Let us define $u$ to be the speed at which information propogates in our system. $\Delta x $ is the cell size, and $\Delta t$ is the timestep overwhich we update the system.

It can be shown that for a fluid solver's solution to be. stable, the timestep must be less than the time it takes for information to propogate across a single zone. That is,
$$
\Delta t \le \frac{\Delta x}{u}
$$
This is called the Courant-Freidrichs-Lewy or CFL condition.  A dimensionless quantity called the CFL number is defined. as
$$
\begin{equation}
\mathcal{C} =. \frac{\Delta t u }{\Delta x}
\end{equation}
$$

For stability, we want $\mathcal{C} \le 1$.  The timestep is traditionally determined as 
$$
\Delta t = \mathcal{C} \frac{\Delta x}{ u}
$$

In. general, since. we are solving a non-linear system of equations, it is not possible to run with $\mathcal{C} = 1$, since $u$ and $\mathcal{C}$ changes from zone. to. zone. We instead look at the most. restrictive timestep over all the zones and use that for the entire system.

A simple summary is "No wave, advection + sound speed, may cross more than one grid cell in one timestep." 

An explicit scheme's solution in a cell after one step depends only on neighbours one zone away, and it's stable only if that contains the true domain dependence of the PDE, ie the region reachable by physical signals in time $\Delta t$. Extrapolating to higher dimensions, we can write this as. 
$$
\Lambda \Delta t \leq \Delta x  
$$
where $\Lambda = \max{|\lambda|}$, and $\lambda$ are the. set of eigenvalues of the flux jacobian, representing the characteristic speeds.  For pure advection of a field $\phi$, $$\partial_t \phi + a \partial_x \phi = 0,$$ there is only one eigenvalue, and thus. Equation (1) holds.  

For the Euler equations, linearizing about a state gives us three eigenvalues,
$$
\begin{align}
\lambda_1 &= v \\
\lambda_\pm &= v \pm c_s
\end{align}
$$

Of these, the fastest is $|v| + c_s$, representing an acoustic wave. 

# Implementation in AthenaK

We start with the initial $\mathrm{d} t$, which is computed in the [driver.cpp](../athenak/src/driver/driver.cpp#L462) file. In the `Driver::Initialize()`  It calls each active module's `NewTimeStep()` and then `pmesh->NewTimeStep(tlim)`


The main loop occurs in [`Driver::Execute()`](../athenak/src/driver/driver.cpp#L529). Each cycle runs the RK stages for the integrator set in the athinput file. Advances as `pmesh ->time += pmesh->dt`
Every stage `Hydro::NewTimeStep` is one of the [tasks](./tasks_architecture.md) defined in [`stagen`](../athenak/src/hydro/hydro_tasks.cpp#L71). The `Hydro::NewTimeStep` runs only once per cycle, so it uses the fully-updated primitve state at the end of the cycle.

The file [`hydro_newdt.cpp`](../athenak/src/hydro/hydro_newdt.cpp#L30-140) implements the per-cell hydro CFL calculation.
1. Computes the sound speed `cs = eos.IdealHydroSoundSpeed(rho, p)` and the magximum signal speed in each direction.
`max_dv1 = |v_x| + cs; max_dv2 = |v_y| + cs; max_dv3 = |v_z| + cs`
The candidate timestep per direction is then calculated as `dx_i / max_dv_i`
2. `Kokkos::parellel_reduce` with `Kokkos::Min` finds the minimum over all cells/MeshBlocks in the pack. 
3. `dtnew = min(dt1, dt2, dt3)`,  giving you `Hydro::dtnew`. 


We also have cooling physics, which is handled with `srcterms_newdt.cpp`. The block in [`SourceTerms::NewTimeStep()`](../athenak/src/srcterms/srcterms_newdt.cpp) handles for the flag in `athenak`. If the energy change in a timestep exceeds the energy. itself, you can experience numerical instability. The cooling timestep ensures the internal energy remains bounded. 
$$
\Delta t_\mathrm{cool} = \frac{e_\mathrm{int}}{\dot e_\mathrm{max}}
$$
where $\dot e$ is the net energy loss rate.

For a cell at $(i, j, k)$, the standard ISM cooling law (the SPEX/Schure et al. 2009 cooling curve used by [`ISMCoolFn`](../athenak/src/srcterms/ismcooling.hpp#L19)) is written in terms of **number density** $n$, not mass density $\rho$:
$$
\dot e = n^2 \Lambda(T) - n\,\Gamma_\mathrm{heat}
$$
where $\Lambda(T)$ is the tabulated cooling function (erg cm$^3$ s$^{-1}$), $\Gamma_\mathrm{heat}$ is an optional background heating rate, and
$$
n = \frac{\rho}{\mu\, m_u}
$$
with $\mu$ the mean molecular weight (`<units> mu`, default 1.0) and $m_u$ the atomic mass unit. The temperature itself comes from the ideal-gas relation
$$
T = \frac{e_\mathrm{int}}{\rho}(\gamma - 1)
$$
(converted to cgs via a `temp_unit` factor before being passed into `ISMCoolFn`).

So the per-cell cooling timestep is
$$
\Delta t_\mathrm{cool}^{(i,j,k)} = \frac{e_\mathrm{int}^{(i,j,k)}}{\left| n^{(i,j,k)}\left(n^{(i,j,k)}\Lambda\!\left(T^{(i,j,k)}\right) - \Gamma_\mathrm{heat}\right)\right| + \epsilon}
$$
with $\epsilon = \texttt{FLT\_MIN}$ added purely to avoid a divide-by-zero where cooling is inactive. The global cooling timestep is the minimum over all cells:
$$
\Delta t_\mathrm{cool} = \min_{i,j,k} \Delta t_\mathrm{cool}^{(i,j,k)}
$$

### Why the code multiplies density by itself instead of computing $n$

[`SourceTerms::NewTimeStep()`](../athenak/src/srcterms/srcterms_newdt.cpp#L25-L111) never actually computes $n$ per cell — it works with `w0(IDN)`, which is mass density $\rho$ in **code units**. At first glance the source looks like it's using $\rho^2\Lambda$ instead of $n^2\Lambda$:

```cpp
// srcterms_newdt.cpp
Real temp_unit = pmy_pack->punit->temperature_cgs();
Real n_unit = pmy_pack->punit->density_cgs()/pmy_pack->punit->mu()
              / pmy_pack->punit->atomic_mass_unit_cgs;
Real cooling_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()
                    / n_unit/n_unit;
Real heating_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()
                    / n_unit;
...
Real temp = temp_unit*w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
Real &eint = w0(m,IEN,k,j,i);

Real lambda_cooling = ISMCoolFn(temp)/cooling_unit;
Real gamma_heating = heating_rate/heating_unit;

// add a tiny number
Real cooling_heating = FLT_MIN + fabs(w0(m,IDN,k,j,i) *
                       (w0(m,IDN,k,j,i) * lambda_cooling - gamma_heating));

min_dt = fmin((eint/cooling_heating), min_dt);
```

This is correct — the $\rho \to n$ conversion is folded into the unit-scaling constants instead of being applied explicitly per cell. `n_unit` is the code-density-to-cgs-number-density conversion factor:
$$
n_\mathrm{cgs} = \rho_\mathrm{code}\cdot n_\mathrm{unit}, \qquad n_\mathrm{unit} = \frac{\rho_{\mathrm{cgs/code}}}{\mu\, m_u}
$$
and `cooling_unit` / `heating_unit` are defined with $n_\mathrm{unit}$ baked in:
$$
\texttt{cooling\_unit} = \frac{p_{\mathrm{cgs/code}}/t_{\mathrm{cgs/code}}}{n_\mathrm{unit}^2}, \qquad
\texttt{heating\_unit} = \frac{p_{\mathrm{cgs/code}}/t_{\mathrm{cgs/code}}}{n_\mathrm{unit}}
$$
Substituting the real physical cooling rate and converting to code units (dividing by the energy-density-rate unit $p_\mathrm{cgs}/t_\mathrm{cgs}$):
$$
\dot e_\mathrm{code} = \frac{n_\mathrm{cgs}^2\Lambda(T) - n_\mathrm{cgs}\Gamma}{p_\mathrm{cgs}/t_\mathrm{cgs}}
= \rho_\mathrm{code}^2\underbrace{\left(\frac{\Lambda(T)}{\texttt{cooling\_unit}}\right)}_{\texttt{lambda\_cooling}} - \rho_\mathrm{code}\underbrace{\left(\frac{\Gamma}{\texttt{heating\_unit}}\right)}_{\texttt{gamma\_heating}}
= \rho_\mathrm{code}\left(\rho_\mathrm{code}\cdot\texttt{lambda\_cooling} - \texttt{gamma\_heating}\right)
$$
which is exactly the `cooling_heating` line above. `lambda_cooling` is **not** $\Lambda(T)$ in cgs — it's $\Lambda(T)$ pre-rescaled by $n_\mathrm{unit}^2$ so that multiplying it by $\rho_\mathrm{code}^2$ (not $n_\mathrm{code}^2$) reproduces the correct $n^2\Lambda(T)$ physics. This avoids a per-cell division by $\mu\, m_u$.

`psrc->NewTimeStep()` is called at the end of [`Hydro::NewTimeStep()`](../athenak/src/hydro/hydro_newdt.cpp#L135-L137), right after the CFL calculation, and its result is stored in `SourceTerms::dtnew`.

## Combining $\Delta t_\mathrm{cool}$ and $\Delta t_\mathrm{CFL}$ in `mesh.cpp`

Both `Hydro::dtnew` (CFL) and `SourceTerms::dtnew` (cooling) are per-rank minimums computed independently. They get combined — together with the user-set CFL safety factor $\mathcal{C}$ (`cfl_number` in the athinput, read once into `Mesh::cfl_no` at [`build_tree.cpp:302`](../athenak/src/mesh/build_tree.cpp#L302)) — inside [`Mesh::NewTimeStep()`](../athenak/src/mesh/mesh.cpp#L595-L669):

```cpp
void Mesh::NewTimeStep(const Real tlim) {
  dtold = dt;
  Real dt_cycle = 2.0*dt;               // cap growth to 2x the previous dt
  ...
  if (pmb_pack->phydro != nullptr) {
    dt_cycle = std::min(dt_cycle, (cfl_no)*(pmb_pack->phydro->dtnew) );        // CFL
    if (pmb_pack->phydro->psrc != nullptr) {
      dt_cycle = std::min(dt_cycle, (cfl_no)*(pmb_pack->phydro->psrc->dtnew) );// cooling
    }
  }
  ... // mhd, diffusion/STS, z4c, radiation, particles blocks, same pattern

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, dt_reduction, 2, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
  dt_cycle = dt_reduction[0];            // global minimum across all ranks
#endif

  dt = dt_cycle;
  if ( (time < tlim) && ((time + dt) > tlim) ) {dt = tlim - time;}  // land exactly on tlim
}
```

Note that $\mathcal{C}$ is applied to **both** `dtnew` values identically — there's no separate safety factor for the cooling constraint, even though it isn't a wave-crossing condition in the CFL sense. The combined formula for the timestep of the *next* cycle is
$$
\Delta t = \min\Big\{\,2\Delta t_\mathrm{prev},\;\; \mathcal{C}\cdot\Delta t_\mathrm{CFL},\;\; \mathcal{C}\cdot\Delta t_\mathrm{cool},\;\; \Delta t_\mathrm{other}\,\Big\}
$$
where $\Delta t_\mathrm{other}$ covers MHD/diffusion/radiation/z4c/particles (all inactive for pure hydro + ISM cooling runs), and the $2\Delta t_\mathrm{prev}$ term just prevents $\Delta t$ from more than doubling in one cycle after a narrow dip.

The ratio $\Delta t_\mathrm{cool}/\Delta t_\mathrm{CFL}$ tells you which constraint is binding at a given point in the run: when it drops below 1, cooling — not wave-crossing — is what's limiting the step, which is exactly what you see as the sharp dt dips in the `hr_build` runs (a cell cooling rapidly through the SPEX curve's steep region).
