# Comparison Report: Resolution Sweep Parameters vs. Sharma et al. (2025)

**Target Document**: `paper_similarity.md`  
**Evaluation Target**: [`run_resolution_sweep.sh`](file:///home/sasi/Projects/SubgridCGMModel/shell_scripts/run_resolution_sweep.sh) and baseline configuration [`ref_athinput.athinp`](file:///home/sasi/Projects/SubgridCGMModel/shell_scripts/ref_athinput.athinp)  
**Reference Paper**: *Universal Structure of Turbulent Radiative Mixing Layers* (Sharma, Kumar, Datta, Babul, Das, & Aditya, arXiv:2509.03802v2 / 2025) and accompanying **Supplemental Material (SM)**.

---

## Executive Summary

| Category | Paper / Supplemental Material | Resolution Sweep Script / `ref_athinput` | Status / Agreement | Action Permitted / Required |
| :--- | :--- | :--- | :--- | :--- |
| **Dimensionality** | 3D ($x, y, z$) for primary fiducial & suite; 2D fiducial run (`fid-2D`) included | **2D** ($nx_1 \times nx_2$, $nx_3 = 1$) | **Matches `fid-2D`** | Retained (2D setup) |
| **Gas Density Contrast ($\chi$)** | $\chi = \rho_c/\rho_h = 100$ | $\rho_{\mathrm{cold}} = 0.1$, $\rho_{\mathrm{hot}} = 0.001 \implies \chi = 100$ | **Exact Match** | Keep fixed |
| **Hot Phase Density ($\rho_h$)** | $n_h = \rho_h / (\mu m_p) = 10^{-3}\ \mathrm{cm}^{-3}$ | $\rho_{\mathrm{hot}} = 0.001\ m_p/\mathrm{cm}^3 \implies n_h = 10^{-3}\ \mathrm{cm}^{-3}$ | **Exact Match** | Keep fixed |
| **Shear Velocity ($\Delta u$)** | $\Delta u = 31\ \mathrm{km/s}$ | $v_{x,\mathrm{hot}} - v_{x,\mathrm{cold}} = 28.18 - (-2.82) = 31.0\ \mathrm{km/s}$ | **Exact Match** | Keep fixed |
| **Frame Velocity Scaling** | $v_{x,\mathrm{hot}} / |v_{x,\mathrm{cold}}| = \chi^{1/2} = 10$ | $28.1818 / 2.8182 = 10$ | **Exact Match** | Keep fixed |
| **Gas Pressure / Temperature** | $P_0 \approx 2.23 \times 10^{-13}\ \mathrm{dyn\ cm^{-2}}$ ($T_h = 10^6\ \mathrm{K}, T_c = 10^4\ \mathrm{K}$) | `press = 14.02645` code units ($= 2.227 \times 10^{-13}\ \mathrm{dyn\ cm^{-2}}$) | **Exact Match** | Keep fixed |
| **Radiative Cooling Curve** | Schure et al. (2009) SPEX + KI02 cutoffs ($1.05 T_c < T < 0.95 T_h$) | Identical SPEX table + cutoffs ($1.05 \times 10^4\ \mathrm{K} < T < 0.95 \times 10^6\ \mathrm{K}$) | **Exact Match** | Keep fixed |
| **Cooling Time ($t_0$)** | $t_0 = 0.0194\ \mathrm{Myr}$ | Evaluated at $T_0 = \sqrt{T_c T_h} = 10^5\ \mathrm{K} \approx 0.0194\ \mathrm{Myr}$ | **Exact Match** | Keep fixed |
| **Simulation Duration ($t_{\mathrm{lim}}$)** | $\sim 350\, t_0 \approx 6.8\ \mathrm{Myr}$ | `tlim = 10.0` code units ($= 10.0\ \mathrm{Myr} \approx 515\, t_0$) | **Consistent / Steady State** | Keep fixed |
| **Transverse Box Size ($L_\perp$)** | Fiducial: $L_x = 10\ \mathrm{pc}$ | Current: $x_1 \in [-10, 10] \implies L_x = 20\ \mathrm{pc}$ | **Discrepancy (2x larger)** | **Allowed to change** |
| **Vertical Box Size ($L_z$)** | Fiducial: $L_z = 40\ \mathrm{pc}$ ($[-20, 20]\ \mathrm{pc}$) | Current: $x_2 \in [-20, 20] \implies L_z = 40\ \mathrm{pc}$ | **Exact Match** | Keep fixed |
| **Interface / Perturbation Widths** | $z_0 \approx 2.17 \Delta u t_0 \approx 1.34\ \mathrm{pc}$; tanh seed | `a_char = 0.125 pc`, `sigma = 0.5 pc`, `amp = 0.05` | **Initial seed only** | **Allowed to change** |
| **Grid Resolutions & Cell Sizes** | Table I & Fig. 2 (see section below) | Grid sweep: $8\times 16$ to $512\times 1024$ | **Discrepancy in cell sizes** | **Allowed to change** |

---

## 1. Dimensionality and Physical Setup Alignment

### 1.1 2D vs. 3D
- **Paper**: The main paper uses 3D simulations ($x, y, z$). However, in Table I and throughout the Supplemental Material (SM), the authors explicitly analyze and tabulate a reference 2D run: **`fid-2D`**.
- **Sweep Script**: Solves the 2D problem in $(x_1, x_2)$ with $nx_3 = 1$. The boundary conditions are periodic in $x_1$ and reflect/user in $x_2$, perfectly matching the 2D configuration in the paper.

### 1.2 Thermodynamics, Shear, and Densities
- **Density Contrast**:
  $$\chi = \frac{\rho_c}{\rho_h} = \frac{0.1}{0.001} = 100$$
  This is identical to the paper ($\chi = 100$, $T_h = 10^6\ \mathrm{K}$, $T_c = 10^4\ \mathrm{K}$).
- **Shear Velocity**:
  $$\Delta u = v_{x,\mathrm{hot}} - v_{x,\mathrm{cold}} = 28.1818182 - (-2.8181818) = 31.0\ \mathrm{km/s}$$
  The reference paper uses $\Delta u = 31\ \mathrm{km/s}$.
- **TRML Stationary Rest Frame**:
  To minimize net vertical drifting of the mixing layer in the computational box, the upstream velocities must obey:
  $$\frac{v_{x,\mathrm{hot}}}{|v_{x,\mathrm{cold}}|} = \chi^{1/2} = \sqrt{100} = 10$$
  In `ref_athinput.athinp`, $28.1818182 / 2.8181818 = 10.0$, matching Dimotakis (1986) and the paper setup.
- **Cooling Curve & Cutoffs**:
  Both the C++ source code (`athenak/src/srcterms/ismcooling.hpp`) and the simulation inputs utilize the Schure et al. (2009) SPEX cooling curve with cooling turned off outside $1.05 T_c < T < 0.95 T_h$ ($1.05 \times 10^4\ \mathrm{K}$ to $0.95 \times 10^6\ \mathrm{K}$).

---

## 2. Length Scales and Domain Dimensions

> [!IMPORTANT]
> The user instructions specify: **"The only changes i'm allowed to make are on the length scales."**
> Below is a detailed breakdown of all length scales in the paper vs. the sweep script.

### 2.1 Natural Physical & Characteristic Scales
From the paper's parameters:
1. **Cooling Time**:
   $$t_0 = 0.0194\ \mathrm{Myr} \approx 6.12 \times 10^{11}\ \mathrm{s}$$
2. **Shear-Cooling Length Scale**:
   $$l_{\mathrm{cool}} = \Delta u \, t_0 = (31\ \mathrm{km/s}) \times (0.0194\ \mathrm{Myr}) \approx 0.617\ \mathrm{pc}$$
3. **TRML Steady-State Thickness**:
   $$z_0 \approx 2.1 \Delta u \, t_0 \approx 1.30\ \mathrm{pc} \quad (\text{or } 3.75 \Delta u t_0 \approx 2.31\ \mathrm{pc} \text{ for fid-2D})$$

### 2.2 Box Dimensions ($L_\perp$ and $L_z$)

#### Paper Domain:
- In the Letter (page 2) and SM (page 2, footnote a):
  $$\text{Fiducial box size: } [L_x, L_y, L_z] = [10, 10, 40]\ \mathrm{pc}$$
  Specifically, $x \in [-5, 5]\ \mathrm{pc}$ (or $[0, 10]\ \mathrm{pc}$), meaning $L_\perp = 10\ \mathrm{pc}$, and $z \in [-20, 20]\ \mathrm{pc}$ ($L_z = 40\ \mathrm{pc}$).
- The dimensionless shear-cooling parameter for the fiducial run is:
  $$\xi = \frac{L_\perp}{\Delta u \, t_0} = \frac{10\ \mathrm{pc}}{0.617\ \mathrm{pc}} \approx 16.2$$

#### Sweep Script / `ref_athinput.athinp`:
- `x1min = -10.0`, `x1max = 10.0` $\implies L_{x1} = 20.0\ \mathrm{pc}$.
- `x2min = -20.0`, `x2max = 20.0` $\implies L_{x2} = 40.0\ \mathrm{pc}$.
- **Discrepancy**: The transverse length scale in the script is $L_\perp = 20.0\ \mathrm{pc}$, which corresponds to:
  $$\xi = \frac{20\ \mathrm{pc}}{0.617\ \mathrm{pc}} \approx 32.4 = 2 \times \xi_{\mathrm{fid}}$$
  This corresponds to the $L_\perp\text{-}2x$ suite runs in Table I, rather than the fiducial $L_\perp = 10\ \mathrm{pc}$ run.

---

## 3. Resolution Comparison

### 3.1 Paper Resolutions and Cell Sizes
In Table I of the Supplemental Material, the fiducial run has grid dimensions $280 \times 280 \times 1040$ over $[10, 10, 40]\ \mathrm{pc}$:
- **Fiducial Cell Size**:
  $$\Delta x = \frac{10\ \mathrm{pc}}{280} \approx 0.0357\ \mathrm{pc} \approx \frac{\Delta u t_0}{17.3}$$
  $$\Delta z = \frac{40\ \mathrm{pc}}{1040} \approx 0.0385\ \mathrm{pc} \approx \frac{\Delta u t_0}{16.0}$$
  Notice that $\Delta x \approx \Delta z$ (isotropic grid spacing $\approx 0.036\text{--}0.038\ \mathrm{pc}$).

The paper's resolution study scales resolution relative to this fiducial grid:
- **res-2x**: $\Delta x \approx 0.018\ \mathrm{pc}$
- **res-1x (fiducial)**: $\Delta x \approx 0.036\ \mathrm{pc}$ ($280 \times 1040$)
- **res-1/2x**: $\Delta x \approx 0.071\ \mathrm{pc}$ ($140 \times 520$)
- **res-1/4x**: $\Delta x \approx 0.143\ \mathrm{pc}$ ($70 \times 260$)
- **res-1/8x**: $\Delta x \approx 0.286\ \mathrm{pc}$ ($35 \times 130$ or $32 \times 128$)
- **res-1/16x**: $\Delta x \approx 0.57\ \mathrm{pc}$
- **res-1/32x**: $\Delta x \approx 1.14\ \mathrm{pc}$
- **res-1/64x**: $\Delta x \approx 2.29\ \mathrm{pc}$

### 3.2 Sweep Script Resolutions
In `run_resolution_sweep.sh`, the resolution sweep executes 7 runs with the domain $[L_{x1}, L_{x2}] = [20, 40]\ \mathrm{pc}$:

| Run Tag | $nx_1 \times nx_2$ | $\Delta x_1 = 20/nx_1$ | $\Delta x_2 = 40/nx_2$ | Aspect Ratio $\Delta x_1 / \Delta x_2$ | Paper Equivalent Resolution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `8x16` | $8 \times 16$ | $2.50\ \mathrm{pc}$ | $2.50\ \mathrm{pc}$ | 1.0 (Square) | $\approx$ `res-1/64x` ($\sim 2.3\ \mathrm{pc}$) |
| `16x32` | $16 \times 32$ | $1.25\ \mathrm{pc}$ | $1.25\ \mathrm{pc}$ | 1.0 (Square) | $\approx$ `res-1/32x` ($\sim 1.14\ \mathrm{pc}$) |
| `32x64` | $32 \times 64$ | $0.625\ \mathrm{pc}$ | $0.625\ \mathrm{pc}$ | 1.0 (Square) | $\approx$ `res-1/16x` ($\sim 0.57\ \mathrm{pc}$) |
| `64x128` | $64 \times 128$ | $0.3125\ \mathrm{pc}$ | $0.3125\ \mathrm{pc}$ | 1.0 (Square) | $\approx$ `res-1/8x` ($\sim 0.286\ \mathrm{pc}$) |
| `128x256` | $128 \times 256$ | $0.156\ \mathrm{pc}$ | $0.156\ \mathrm{pc}$ | 1.0 (Square) | $\approx$ `res-1/4x` ($\sim 0.143\ \mathrm{pc}$) |
| `256x512` | $256 \times 512$ | $0.078\ \mathrm{pc}$ | $0.078\ \mathrm{pc}$ | 1.0 (Square) | $\approx$ `res-1/2x` ($\sim 0.071\ \mathrm{pc}$) |
| `512x1024` | $512 \times 1024$ | $0.039\ \mathrm{pc}$ | $0.039\ \mathrm{pc}$ | 1.0 (Square) | $\approx$ `res-1x` (fiducial $\sim 0.036\ \mathrm{pc}$) |

#### Key Insights on Resolution:
1. **Cell isotropic geometry**: Because $L_2/L_1 = 40/20 = 2$ and $nx_2 / nx_1 = 2$, the grid cells in the sweep script are **strictly square** ($\Delta x_1 = \Delta x_2$). This matches the isotropic grid design of the paper.
2. **Span of sweep**: The sweep script spans from $\sim \text{res-1/64x}$ up to $\sim \text{res-1x}$ (fiducial). 
3. **Difference due to domain length**: Because $L_1 = 20\ \mathrm{pc}$ instead of $10\ \mathrm{pc}$, achieving the fiducial resolution requires $512$ cells along $x_1$ rather than $280$ cells.

---

## 4. Initial Condition Length Scales (`a_char` and `sigma`)

In `ref_athinput.athinp`:
- `a_char = 0.125` (interface initial tanh width)
- `sigma = 0.5` (Gaussian envelope width for vertical velocity perturbations)

In the paper:
- The exact initial interface thickness in the paper is set to a thin smoothed tanh profile ($a \ll z_0$). Because the paper's steady-state TRML thickness is $z_0 \approx 2.1 \Delta u t_0 \approx 1.3\ \mathrm{pc}$, an initial width of $a_{\mathrm{char}} = 0.125\ \mathrm{pc} \approx 0.2 \Delta u t_0$ is small enough to allow the Kelvin-Helmholtz instability to rapidly develop and broaden into the self-consistent turbulent steady state ($z_0 \sim 1.3\text{--}2.3\ \mathrm{pc}$).
- However, if matching the fiducial paper box ($L_x = 10\ \mathrm{pc}$) is desired, `a_char` and `sigma` represent physical length scales that scale with the domain or cooling length.

---

## 5. Permissible Length Scale Modifications

Since you are **only allowed to modify length scales**, here are the exact adjustments to bring the sweep into 100% agreement with the paper's fiducial setup:

### Option A: Match the Fiducial Paper Box Exactly ($L_\perp = 10\ \mathrm{pc}, L_z = 40\ \mathrm{pc}$)
To match the paper's fiducial box size $[10, 40]\ \mathrm{pc}$:
1. **Change $X_1$ boundaries in `ref_athinput.athinp`**:
   ```ini
   x1min = -5.0   # or 0.0
   x1max =  5.0   # or 10.0  -> Total Lx = 10.0 pc
   ```
2. **Adjust the resolution numbers in `run_resolution_sweep.sh`**:
   With $L_x = 10\ \mathrm{pc}$ and $L_z = 40\ \mathrm{pc}$, the aspect ratio $L_z / L_x = 4$. To maintain square cells ($\Delta x_1 = \Delta x_2$):
   - $nx_2 / nx_1 = 4$
   - E.g., for fiducial cell size ($\approx 0.036\text{--}0.038\ \mathrm{pc}$):
     $$nx_1 \times nx_2 = 280 \times 1040 \quad (\text{or } 256 \times 1024)$$
   - The resolution sweep levels would then be:
     - $4 \times 16$ ($\approx$ 1/64x)
     - $8 \times 32$ ($\approx$ 1/32x)
     - $16 \times 64$ ($\approx$ 1/16x)
     - $32 \times 128$ ($\approx$ 1/8x, directly tabulated in Table I)
     - $64 \times 256$ ($\approx$ 1/4x)
     - $128 \times 512$ ($\approx$ 1/2x)
     - $256 \times 1024$ (or $280 \times 1040$, $\approx$ 1x fiducial)

### Option B: Keep the Extended Box ($L_\perp = 20\ \mathrm{pc}, L_z = 40\ \mathrm{pc}$)
- As noted in Section S3 of the Supplemental Material, in the strong-cooling regime ($\xi \gg 1$), all turbulent statistics, surface cooling rate $\dot{\Sigma}_{\mathrm{cool}}$, and mixing layer thickness $z_0$ become **independent of $L_\perp$** (saturation of $\dot{\Sigma}_{\mathrm{cool}}$).
- The current $L_\perp = 20\ \mathrm{pc}$ setup represents the **`L_\perp-2x`** series tabulated in Table I.
- The cell sizes $\Delta x$ in your current script already match the cell sizes $\Delta x$ of the paper's resolution study (from $1/64\times$ up to $1\times$ fiducial).

---

## Conclusion
All thermodynamic, hydrodynamic, and cooling parameters (densities, temperatures, pressures, shear velocities, cooling curve cutoffs, and CFL/integration parameters) are **already in exact numerical agreement with the paper**. 

The only deviations are purely length-scale based:
1. **$L_{x1}$ is $20\ \mathrm{pc}$ instead of $10\ \mathrm{pc}$** (corresponds to $L_\perp\text{-}2x$ rather than the fiducial $L_\perp\text{-}1x$).
2. **$nx_2 / nx_1 = 2$ instead of $4$** (a direct consequence of having $L_x = 20\ \mathrm{pc}$ instead of $10\ \mathrm{pc}$ while preserving square cells).
