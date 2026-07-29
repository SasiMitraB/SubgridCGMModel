# Subgrid CGM Model Diagnostic Analysis Report

This report presents a quantitative evaluation of the subgrid model (SG) performance against high-resolution (HR) ground-truth datasets and low-resolution (LR) baseline runs. The diagnostics are ordered from fundamental closure errors to bulk thermodynamic statistics.

---

## 1. Subgrid Emissivity Closure Error & SG Correction KPI

### Formulation
The **Subgrid Emissivity Closure Error** $\delta_\Lambda(y,t)$ measures the fraction of emissivity that is lost when using resolved-scale coarse-grained fields rather than the full-resolution fields:

$$
\delta_\Lambda(y,t) = \frac{\overline{\rho^2 \Lambda(T)} - \bar{\rho}^2\Lambda(\bar{T})}{\overline{\rho^2\Lambda(T)}}
$$

* **Numerator First Term ($\overline{\rho^2 \Lambda(T)}$)**: Coarse-grained ground-truth emissivity product (computed on the high-resolution grid and then downsampled).
* **Numerator Second Term ($\bar{\rho}^2\Lambda(\bar{T})$)**: Resolved-scale emissivity product (emissivity computed directly from coarse-grained density and temperature fields).

The **SG Emissivity Correction KPI** ($F_{\rm corr}$) quantifies the fraction of the missing coarse-grained emissivity that the subgrid model succeeds in recovering:

$$
F_{\text{corr}} = \frac{\Sigma_{c,\text{SG}} - \Sigma_{c,\text{LR}}}{\Sigma_{c,\text{HR-CG}} - \Sigma_{c,\text{LR}}}
$$

where $\Sigma_c = \int \langle n^2 \Lambda(T) \rangle_{X,t} \, dy$ is the integrated emissivity profile.

---

### Diagnostic Figure
![Subgrid Emissivity Closure Error](file:///home/sasi/Projects/SubgridCGMModel/simulation_outputs/plots/diag_subgrid_emissivity_closure.png)

---

### Interpretation
* **Closure Deficit**: $\delta_\Lambda(y)$ peaks in the mixing layers, reaching values above $90\%$. This indicates that the vast majority of cooling in these regions is driven by small-scale density and temperature fluctuations.
* **SG Correction Performance**: The KPI value of **$F_{\rm corr} \approx 0.295$** indicates that the subgrid model recovers only **$29.5\%$** of the missing emissivity. The model remains highly under-luminous, leading directly to the slower cold-gas mass growth rate.

---

## 2. Subgrid Density Variance & Clumping Factor

### Formulation
Because the cooling function is density-squared weighted ($\propto \rho^2$), the primary driver of the emissivity closure error is the **subgrid density variance** $\sigma_\rho^2(y,t)$, defined as:

$$
\sigma_\rho^2(y,t) = \overline{\rho^2} - \bar{\rho}^2
$$

The corresponding **Clumping Factor** $C(y,t)$, which measures the subgrid spatial concentration of gas, is defined as:

$$
C(y,t) = \frac{\overline{\rho^2}}{\bar{\rho}^2}
$$

---

### Diagnostic Figure
![Subgrid Density Variance & Clumping Factor](file:///home/sasi/Projects/SubgridCGMModel/simulation_outputs/plots/diag_density_variance_clumping.png)

---

### Interpretation
* **Spatial Correlation**: The profile of the clumping factor excess ($C(y) - 1$) and density variance $\sigma_\rho^2$ matches the shape and peaks of the emissivity closure error $\delta_\Lambda(y)$ perfectly.
* **Physics Deficit**: This correlation provides direct evidence that the subgrid model fails to capture small-scale clumpiness. Because the model lacks a state variable encoding density variance (clumping factor), it cannot compute the correct subgrid emissivity.
* **Suggested Resolution**: An effective fix would be to model $C$ dynamically as a function of the passive scalar `ps` or resolved turbulence parameters, and scale the cooling term by $(1 + C)$ or an equivalent clumping factor coefficient.

---

## 3. Subgrid Reynolds Stress

### Formulation
The **Subgrid Reynolds Stress** $\tau_{xy}^{\rm sgs}(y,t)$ represents the transport of momentum by subgrid turbulent velocity fluctuations:

$$
\tau_{xy}^{\rm sgs}(y,t) = \overline{\rho u_x u_y} - \bar\rho\,\bar u_x\,\bar u_y
$$

The **Stress Ratio** $R_\tau$ measures the ratio of the resolved momentum transport in the SG run to the high-resolution SGS transport:

$$
R_\tau = \frac{\int \langle |\tau_{xy}^{\rm SG,resolved}| \rangle_t \, dy}{\int \langle |\tau_{xy}^{\rm sgs,HR}| \rangle_t \, dy}
$$

where the resolved fluctuations are computed relative to the horizontal mean:
$$\tau_{xy}^{\rm SG,resolved}(y,t) = \langle \rho u_x u_y \rangle_X - \langle \rho \rangle_X \langle u_x \rangle_X \langle u_y \rangle_X$$

---

### Diagnostic Figure
![Subgrid Reynolds Stress](file:///home/sasi/Projects/SubgridCGMModel/simulation_outputs/plots/diag_subgrid_reynolds_stress.png)

---

### Interpretation
* **Stress Ratio Deficit**: The stress ratio is extremely low ($R_\tau \approx 0.016$).
* **Turbulent Transport Collapse**: The SG model resolved stress is virtually zero everywhere. This indicates that the resolved flow field on the coarse grid does not capture any turbulent momentum flux. An explicit subgrid eddy-viscosity or turbulent diffusion model is required to transport momentum and broaden the mixing layer correctly.

---

## 4. Cold Mass Growth Rate & Cooling Efficiency

### Formulation
The thermodynamic efficiency of gas condensation is measured by the **Effective Cooling Efficiency** $\eta$, which normalizes the bulk mass growth rate ($\dot{M}_{\rm cold}$) by the integrated radiative energy loss ($\Sigma_c$):

$$
\eta = \frac{\dot M_{\rm cold}}{\Sigma_c \cdot A_{\rm domain}}
$$

* **$\dot{M}_{\rm cold}$**: Rate of cold mass growth ($dM_{\rm cold}/dt$), computed from the linear fit slope of cold gas mass ($T < 10^5\text{ K}$) vs. time.
* **$\Sigma_c$**: Time-averaged integrated emissivity profile $\int dy \langle \rho^2 \Lambda(T) \rangle_X$.
* **$A_{\rm domain}$**: The domain cross-sectional area ($L_x \cdot L_y$).

---

### Diagnostic Figure
![Cooling Efficiency](file:///home/sasi/Projects/SubgridCGMModel/simulation_outputs/plots/diag_cold_mass_efficiency.png)

---

### Interpretation
* **Anomalous Efficiency**: The cooling efficiency $\eta$ in SG and LR runs is $\approx 2.6\times$ higher than in the HR runs. 
* **Numerical Phase Mixing**: For a given amount of energy radiated away, the coarse simulations condense $2.6\times$ more gas. This suggests a secondary numerical artifact: low resolution causes artificial phase mixing at the cold-hot interfaces, numerically cooling gas into the cold phase without physical radiative cooling.

---

## 5. Turbulent Kinetic Energy (TKE) Proxy

### Formulation
The subgrid-scale kinetic energy lost during coarse-graining of the high-resolution simulation is compared against the resolved TKE excess in the subgrid simulation relative to the low-resolution run:

$$
{\rm TKE}_{\rm sgs,HR}(y,t) = \tfrac12\left[\overline{\rho u_x^2} - \bar\rho\bar u_x^2 + \overline{\rho u_y^2}-\bar\rho\bar u_y^2\right]
$$

$$
\Delta {\rm KE}_{\rm SG-LR}(y,t) = \tfrac12\bar\rho_{\rm SG}(\bar u_{x,\rm SG}^2+\bar u_{y,\rm SG}^2) - \tfrac12\bar\rho_{\rm LR}(\bar u_{x,\rm LR}^2+\bar u_{y,\rm LR}^2)
$$

---

### Diagnostic Figure
![Turbulent Kinetic Energy](file:///home/sasi/Projects/SubgridCGMModel/simulation_outputs/plots/diag_tke_spectrum_proxy.png)

---

### Interpretation
* **TKE Discrepancy**: The resolved kinetic energy excess ($\Delta \text{KE}$) generated by the SG model does not balance the true SGS turbulent kinetic energy of the high-resolution run. This indicates that the subgrid model does not inject the proper amount of turbulent kinetic energy back into the resolved scale flow.

---

## 6. Mass Flux Divergence Consistency Check

### Formulation
The numerical discretization error introduced by coarse-graining the divergence operator is measured by:

$$
\epsilon_{\rm div}(y,t) = \left| \nabla\cdot(\rho\mathbf u)\big|_{\rm CG} - \nabla\cdot(\rho\mathbf u)\big|_{\rm res}\right|
$$

where the first term is computed on coarse-grained fluxes and the second term is computed on products of coarse-grained primitive variables.

---

### Diagnostic Figure
![Divergence Consistency Check](file:///home/sasi/Projects/SubgridCGMModel/simulation_outputs/plots/diag_divergence_consistency.png)

---

### Interpretation
* **Sanity Check**: The discretization error $\epsilon_{\rm div}(y)$ is small across the domain, confirming that spatial downsampling does not introduce catastrophic mass conservation violation artifacts in the divergence operator on the coarse grid.
