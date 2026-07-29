# Comparative Run Analysis Report
## Comparing High-Resolution, Low-Resolution Baseline, and Subgrid Model Runs

This report compares the effective numerical Reynolds number ($Re_{\rm num}$), numerical viscosity ($\nu_{\rm num}$), and spectral properties across three runs:

1. **hr_build_1024** (High-Resolution reference, $1024\times 512$ grid)

2. **lr_build_ism** (Low-Resolution baseline with ISM cooling, $32\times 16$ grid)

3. **subgrid_model** (Low-Resolution run with the subgrid model active, $32\times 16$ grid)

### Comparison Metrics Table
| Run Name | Grid Resolution | $\Delta x$ [pc] | Fitted Slope $\alpha$ | Knee $k_\nu$ [pc$^{-1}$] | $\nu_{\rm num}$ [cm$^2$ s$^{-1}$] | Effective $Re_{\rm num}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **lr_build_ism** | $20\times16$ | 0.6250 | 1.667 | 2.18 | 8.761e+22 | 61.0 |
| **subgrid_model** | $20\times16$ | 0.6250 | 1.667 | 2.81 | 8.832e+22 | 575.7 |
| **hr_build_1024** | $1024\times 512$ | 0.0195 | 1.659 | 54.95 | 1.721e+21 | 40629.2 |

### Physical Interpretation and Discussion
- **Numerical Viscosity ($\nu_{{\rm num}}$)**:
  - The high-resolution reference run (**hr_build_1024**) has the lowest effective viscosity ($\nu_{\rm num} \approx 1.72e+21\ {\rm cm}^2\ {\rm s}^{-1}$), corresponding to the highest Reynolds number ($Re_{\rm num} \approx 40629.2$), as expected.
  - The low-resolution baseline run (**lr_build_ism**) has a much higher effective viscosity ($\nu_{\rm num} \approx 8.76e+22\ {\rm cm}^2\ {\rm s}^{-1}$) due to the large grid-scale truncation errors at $32\times16$ resolution.
  - The run with the active subgrid model (**subgrid_model**) has an effective viscosity of $\nu_{\rm num} \approx 8.83e+22\ {\rm cm}^2\ {\rm s}^{-1}$.

- **Kinetic Energy Decay Rate ($\bar{\epsilon}$)**:
  - **hr_build_1024** decay rate: $\bar{\epsilon} \approx -1.692e+00$ code units.
  - **lr_build_ism** decay rate: $\bar{\epsilon} \approx -5.549e-01$ code units.
  - **subgrid_model** decay rate: $\bar{\epsilon} \approx 1.558e+00$ code units.

### Diagnostic Figures
1. **Turbulent Energy Spectra Overlay** ([comparison_spectra.png](file:///home/sasi/Projects/SubgridCGMModel/outputs/explore_data/comparison_spectra.png)): Solenoidal kinetic energy spectra showing how the three runs distribute energy across spatial scales.
2. **Kinetic Energy Decay** ([comparison_ke_decay.png](file:///home/sasi/Projects/SubgridCGMModel/outputs/explore_data/comparison_ke_decay.png)): Compares the rate of loss of specific kinetic energy over time.
3. **Enstrophy Spectra Comparison** ([comparison_enstrophy.png](file:///home/sasi/Projects/SubgridCGMModel/outputs/explore_data/comparison_enstrophy.png)): Compares the distribution of enstrophy (vortical activity) across scales.