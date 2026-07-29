import os
import sys
import gc
import json
from pathlib import Path
import numpy as np
import scipy
from scipy.signal.windows import tukey
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Append project root to path
PROJECT_ROOT = Path('/home/sasi/Projects/SubgridCGMModel')
sys.path.append(str(PROJECT_ROOT))

from ergane import SimulationData
import ergane.units

def helmholtz_decompose(ux, uy, dx, dy):
    """
    Perform Helmholtz decomposition on a 2D velocity/momentum field (ux, uy).
    Returns:
      (ux_sol_hat, uy_sol_hat): solenoidal (divergence-free) components in Fourier space
      (ux_comp_hat, uy_comp_hat): compressive (curl-free) components in Fourier space
      kx_grid, ky_grid: wavenumber coordinate grids
      k_mag: magnitude of wavevectors
    """
    ny, nx = ux.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    ky_grid, kx_grid = np.meshgrid(ky, kx, indexing='ij')
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2)
    
    # 2D FFTs (normalized to satisfy Parseval's theorem)
    ux_hat = np.fft.fft2(ux) / (nx * ny)
    uy_hat = np.fft.fft2(uy) / (nx * ny)
    
    dot_prod = ux_hat * kx_grid + uy_hat * ky_grid
    mask = k_mag > 0
    
    ux_comp_hat = np.zeros_like(ux_hat)
    uy_comp_hat = np.zeros_like(uy_hat)
    ux_comp_hat[mask] = (dot_prod[mask] / k_mag[mask]**2) * kx_grid[mask]
    uy_comp_hat[mask] = (dot_prod[mask] / k_mag[mask]**2) * ky_grid[mask]
    
    ux_sol_hat = ux_hat - ux_comp_hat
    uy_sol_hat = uy_hat - uy_comp_hat
    
    return (ux_sol_hat, uy_sol_hat), (ux_comp_hat, uy_comp_hat), kx_grid, ky_grid, k_mag

def compute_spectra(frame, mass_weighted=True, tukey_alpha=0.25, y_min_box=-6.0, y_max_box=6.0):
    """
    Compute 1D energy and enstrophy spectra for a single simulation frame.
    Returns:
      k_centers: 1D wavevector bin centers
      E_1d: total energy spectrum
      E_sol_1d: solenoidal energy spectrum
      E_comp_1d: compressive energy spectrum
      Omega_1d: enstrophy spectrum
      mean_rho: mean density in the sub-box
      delta_v: shear velocity difference across the sub-box
    """
    # 1. Restrict the FFT to a sub-box in y to avoid boundary effects
    y_mask = (frame.yc >= y_min_box) & (frame.yc <= y_max_box)
    yc_sub = frame.yc[y_mask]
    ny_sub = len(yc_sub)
    nx = len(frame.xc)
    
    density_sub = frame.density[y_mask, :]
    velx_sub = frame.velx[y_mask, :]
    vely_sub = frame.vely[y_mask, :]
    
    dx = frame.xc[1] - frame.xc[0]
    dy = frame.yc[1] - frame.yc[0]
    
    mean_rho = np.mean(density_sub)
    
    # Calculate velocity difference across the shear layer
    mean_vx_y = np.mean(velx_sub, axis=1)
    delta_v = np.max(mean_vx_y) - np.min(mean_vx_y)
    
    # 2. Define field to transform
    if mass_weighted:
        ux = np.sqrt(density_sub) * velx_sub
        uy = np.sqrt(density_sub) * vely_sub
    else:
        ux = velx_sub
        uy = vely_sub
        
    # 3. Apply Tukey window along the non-periodic y-axis (axis 0)
    w = tukey(ny_sub, alpha=tukey_alpha)
    C_w = np.mean(w**2)  # Window power correction factor
    ux_windowed = ux * w[:, np.newaxis]
    uy_windowed = uy * w[:, np.newaxis]
    
    # 4. Helmholtz decomposition in Fourier space
    sol_hat, comp_hat, _, _, k_mag = helmholtz_decompose(ux_windowed, uy_windowed, dx, dy)
    ux_sol_hat, uy_sol_hat = sol_hat
    ux_comp_hat, uy_comp_hat = comp_hat
    
    # Total velocity FFTs (normalized)
    ux_hat_norm = np.fft.fft2(ux_windowed) / (nx * ny_sub)
    uy_hat_norm = np.fft.fft2(uy_windowed) / (nx * ny_sub)
    
    # 5. Compute 2D energy and enstrophy densities
    E_2d = 0.5 * (np.abs(ux_hat_norm)**2 + np.abs(uy_hat_norm)**2) / C_w
    E_sol_2d = 0.5 * (np.abs(ux_sol_hat)**2 + np.abs(uy_sol_hat)**2) / C_w
    E_comp_2d = 0.5 * (np.abs(ux_comp_hat)**2 + np.abs(uy_comp_hat)**2) / C_w
    
    # Vorticity hat in Fourier space: i * (kx * uy_hat - ky * ux_hat)
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny_sub, d=dy)
    ky_grid, kx_grid = np.meshgrid(ky, kx, indexing='ij')
    vort_hat = 1j * (kx_grid * uy_hat_norm - ky_grid * ux_hat_norm)
    Omega_2d = 0.5 * np.abs(vort_hat)**2 / C_w
    
    # 6. Azimuthal averaging to 1D bins
    L_x = nx * dx
    L_y_sub = ny_sub * dy
    k_min = max(2 * np.pi / L_x, 2 * np.pi / L_y_sub)
    k_max = np.pi / min(dx, dy)
    dk = min(2 * np.pi / L_x, 2 * np.pi / L_y_sub)
    k_bins = np.arange(k_min, k_max, dk)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    
    bin_indices = np.digitize(k_mag, k_bins)
    
    E_1d = np.zeros(len(k_centers))
    E_sol_1d = np.zeros(len(k_centers))
    E_comp_1d = np.zeros(len(k_centers))
    Omega_1d = np.zeros(len(k_centers))
    
    for i in range(1, len(k_bins)):
        mask = (bin_indices == i)
        if np.any(mask):
            # The 2D-to-1D scale factor is 2 * pi * k (annulus perimeter)
            E_1d[i-1] = np.mean(E_2d[mask]) * 2.0 * np.pi * k_centers[i-1]
            E_sol_1d[i-1] = np.mean(E_sol_2d[mask]) * 2.0 * np.pi * k_centers[i-1]
            E_comp_1d[i-1] = np.mean(E_comp_2d[mask]) * 2.0 * np.pi * k_centers[i-1]
            Omega_1d[i-1] = np.mean(Omega_2d[mask]) * 2.0 * np.pi * k_centers[i-1]
            
    return k_centers, E_1d, E_sol_1d, E_comp_1d, Omega_1d, mean_rho, delta_v

def analyze_resolution_run(resolution_dir, t_min=5.0, t_max=9.0, mass_weighted=True, tukey_alpha=0.25, athinp_path=None):
    """
    Run spectrum and numerical Reynolds analysis on a single resolution folder.
    """
    if athinp_path is not None:
        athinp = athinp_path
    else:
        athinp_files = list(resolution_dir.glob("*.athinput"))
        if not athinp_files:
            raise ValueError(f"No athinput file found in {resolution_dir}")
        athinp = athinp_files[0]
    datafolder = resolution_dir / "bin"
    
    sim = SimulationData(athinp=str(athinp), datafolder=str(datafolder))
    sim.set_units(ergane.units.Units.code())  # Force analysis in clean code units
    
    # Filter times in steady-state window
    times = sim.times
    in_window = (times >= t_min) & (times <= t_max)
    frame_numbers = np.array(sim.frame_numbers)[in_window]
    frame_times = times[in_window]
    
    if len(frame_numbers) == 0:
        raise ValueError(f"No frames found in time window [{t_min}, {t_max}] for {resolution_dir.name}")
    
    # Step 1: Compute kinetic energy decay curve for specific energy dissipation rate
    KE_vals = []
    rho_vals = []
    
    print(f"\nProcessing {resolution_dir.name}: {len(frame_numbers)} frames in window [{t_min}, {t_max}]")
    for num in frame_numbers:
        frame = sim.get_frame(num)
        KE_density = 0.5 * frame.density * (frame.velx**2 + frame.vely**2)
        KE_vals.append(np.mean(KE_density))
        rho_vals.append(np.mean(frame.density))
        
    KE_vals = np.array(KE_vals)
    rho_vals = np.array(rho_vals)
    KE_spec_vals = KE_vals / rho_vals
    
    # Fit linear slope to get specific energy dissipation rate epsilon_decay
    slope, intercept = np.polyfit(frame_times, KE_spec_vals, 1)
    epsilon_decay = -slope
    
    # Step 2: Sample a subset of frames to compute and average 1D spectra
    n_samples = 40
    step = max(1, len(frame_numbers) // n_samples)
    sampled_indices = np.arange(0, len(frame_numbers), step)
    sampled_frame_nums = frame_numbers[sampled_indices]
    
    avg_E = None
    avg_E_sol = None
    avg_E_comp = None
    avg_Omega = None
    avg_k = None
    
    sum_rho = 0.0
    sum_delta_v = 0.0
    count = 0
    
    for num in tqdm(sampled_frame_nums, desc=f"Averaging spectra for {resolution_dir.name}"):
        frame = sim.get_frame(num)
        k_centers, E_1d, E_sol_1d, E_comp_1d, Omega_1d, mean_rho, delta_v = compute_spectra(
            frame, mass_weighted=mass_weighted, tukey_alpha=tukey_alpha
        )
        
        # Specific energy and enstrophy spectra (divided by mean density)
        E_1d_spec = E_1d / mean_rho
        E_sol_1d_spec = E_sol_1d / mean_rho
        E_comp_1d_spec = E_comp_1d / mean_rho
        Omega_1d_spec = Omega_1d / mean_rho
        
        if avg_E is None:
            avg_E = E_1d_spec
            avg_E_sol = E_sol_1d_spec
            avg_E_comp = E_comp_1d_spec
            avg_Omega = Omega_1d_spec
            avg_k = k_centers
        else:
            avg_E += E_1d_spec
            avg_E_sol += E_sol_1d_spec
            avg_E_comp += E_comp_1d_spec
            avg_Omega += Omega_1d_spec
            
        sum_rho += mean_rho
        sum_delta_v += delta_v
        count += 1
        
    avg_E /= count
    avg_E_sol /= count
    avg_E_comp /= count
    avg_Omega /= count
    mean_rho = sum_rho / count
    delta_v = sum_delta_v / count
    
    # Step 3: Identify scale metrics
    # Peak scale (forcing/outer scale)
    idx_peak = np.argmax(avg_E_sol)
    k_L = avg_k[idx_peak]
    L_scale = 2.0 * np.pi / k_L
    
    # Grid Nyquist frequency
    frame_0 = sim.get_frame(frame_numbers[0])
    nx = len(frame_0.xc)
    ny = len(frame_0.yc)
    ny_sub = np.sum((frame_0.yc >= -6.0) & (frame_0.yc <= 6.0))
    dx = frame_0.xc[1] - frame_0.xc[0]
    dy = frame_0.yc[1] - frame_0.yc[0]
    k_Nyquist = np.pi / min(dx, dy)
    
    # Step 4: Fit inertial range
    # Standard fitting band: 1.5 * k_L up to 0.25 * k_Nyquist
    fit_start_k = 1.5 * k_L
    fit_end_k = 0.25 * k_Nyquist
    inertial_mask = (avg_k >= fit_start_k) & (avg_k <= fit_end_k)
    
    if np.sum(inertial_mask) >= 2:
        slope_fit, intercept_fit = np.polyfit(np.log(avg_k[inertial_mask]), np.log(avg_E_sol[inertial_mask]), 1)
        alpha = -slope_fit
        C_K = 1.5  # Kolmogorov constant
        epsilon_fit = (np.exp(intercept_fit) / C_K) ** 1.5
    else:
        alpha = 5.0 / 3.0
        epsilon_fit = np.nan
        slope_fit = -alpha
        intercept_fit = np.nan
        
    # Step 5: Fit dissipation knee wavenumber k_nu (compensated spectrum threshold method)
    E_comp = avg_E_sol * (avg_k ** alpha)
    mean_comp = np.mean(E_comp[inertial_mask]) if np.sum(inertial_mask) > 0 else np.max(E_comp)
    
    drop_fraction = 1.0 / np.e
    threshold = drop_fraction * mean_comp
    
    # Find crossing point at k > 1.5 * k_L
    indices_cross = np.where((avg_k >= 1.5 * k_L) & (E_comp < threshold))[0]
    if len(indices_cross) > 0:
        idx_cross = indices_cross[0]
        if idx_cross > 0:
            k1, k2 = avg_k[idx_cross-1], avg_k[idx_cross]
            y1, y2 = E_comp[idx_cross-1], E_comp[idx_cross]
            k_nu = k1 + (k2 - k1) * (threshold - y1) / (y2 - y1)
        else:
            k_nu = avg_k[idx_cross]
    else:
        # Fallback to a high wavenumber
        k_nu = k_Nyquist * 0.5
        
    # Step 6: Compute numerical viscosity and Reynolds number
    # Use absolute value for safety in case of minor energy fluctuations
    eps_decay_abs = max(1e-20, abs(epsilon_decay))
    nu_num_decay = (eps_decay_abs / (k_nu**4))**(1.0/3.0)
    Re_num_decay = delta_v * L_scale / nu_num_decay
    
    if np.isfinite(epsilon_fit) and epsilon_fit > 0:
        nu_num_fit = (epsilon_fit / (k_nu**4))**(1.0/3.0)
        Re_num_fit = delta_v * L_scale / nu_num_fit
    else:
        nu_num_fit = np.nan
        Re_num_fit = np.nan
        
    # Conversion scaling factor to CGS
    # length_cgs = 3.08568e18 cm, time_cgs = 3.15576e13 s
    nu_scale_cgs = (3.08568e18)**2 / 3.15576e13  # ~ 3.01738e23 cm^2/s
    
    nu_num_decay_cgs = nu_num_decay * nu_scale_cgs
    nu_num_fit_cgs = nu_num_fit * nu_scale_cgs if np.isfinite(nu_num_fit) else np.nan
    
    results = {
        "resolution": resolution_dir.name,
        "nx": nx,
        "ny": ny_sub,
        "dx": dx,
        "dy": dy,
        "delta_v": delta_v,
        "L_scale": L_scale,
        "k_L": k_L,
        "k_Nyquist": k_Nyquist,
        "k_nu": k_nu,
        "alpha": alpha,
        "epsilon_decay": epsilon_decay,
        "epsilon_fit": epsilon_fit,
        "nu_num_decay": nu_num_decay,
        "Re_num_decay": Re_num_decay,
        "nu_num_fit": nu_num_fit,
        "Re_num_fit": Re_num_fit,
        "nu_num_decay_cgs": nu_num_decay_cgs,
        "nu_num_fit_cgs": nu_num_fit_cgs,
        "frame_times": frame_times.tolist(),
        "KE_spec_vals": KE_spec_vals.tolist(),
        "k": avg_k.tolist(),
        "E": avg_E.tolist(),
        "E_sol": avg_E_sol.tolist(),
        "E_comp": avg_E_comp.tolist(),
        "Omega": avg_Omega.tolist(),
        "E_comp_spec": E_comp.tolist(),
        "mean_rho": mean_rho,
        "slope_fit": slope_fit,
        "intercept_fit": intercept_fit,
    }
    
    return results

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(x) for x in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return make_json_serializable(obj.tolist())
    else:
        return obj

def run_comparison():
    output_dir = PROJECT_ROOT / "outputs" / "explore_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ref_athinp = PROJECT_ROOT / "simulation_outputs" / "resolution_test" / "128x64" / "kh_radiative_128x64.athinput"
    
    comparison_dirs = {
        "lr_build_ism": PROJECT_ROOT / "simulation_outputs" / "lr_build_ism",
        "subgrid_model": PROJECT_ROOT / "simulation_outputs" / "subgrid_model",
        "hr_build_1024": PROJECT_ROOT / "simulation_outputs" / "hr_build_1024"
    }
    
    # We will use t_min = 5.2, t_max = 9.8 since lr_build_ism and subgrid_model start at 5.01
    t_min = 5.2
    t_max = 9.8
    
    all_results = {}
    for name, res_dir in comparison_dirs.items():
        try:
            res_data = analyze_resolution_run(
                res_dir, t_min=t_min, t_max=t_max, athinp_path=ref_athinp
            )
            all_results[name] = res_data
        except Exception as e:
            print(f"Error processing {name}: {e}")
            import traceback
            traceback.print_exc()
            
    # Save raw data to JSON for reference
    serializable_results = make_json_serializable(all_results)
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(serializable_results, f, indent=2)
        
    print("\n--- Comparison Results ---")
    print(f"{'Simulation':<15} | {'dx':<10} | {'alpha':<8} | {'k_nu':<8} | {'nu_num_decay (CGS)':<18} | {'Re_num_decay':<12}")
    print("-" * 80)
    for name, data in all_results.items():
        print(f"{name:<15} | {data['dx']:<10.4f} | {data['alpha']:<8.3f} | {data['k_nu']:<8.2f} | {data['nu_num_decay_cgs']:<18.3e} | {data['Re_num_decay']:<12.1f}")
        
    # --- Generate Plots ---
    
    colors = {"lr_build_ism": "tab:red", "subgrid_model": "tab:green", "hr_build_1024": "tab:blue"}
    styles = {"lr_build_ism": "--", "subgrid_model": "-", "hr_build_1024": "-"}
    
    # 1. Figure 1: Energy Spectra Comparison
    plt.figure(figsize=(10, 8))
    for name, data in all_results.items():
        plt.loglog(data["k"], data["E_sol"], label=f"{name} (Solenoidal)", color=colors[name], ls=styles[name], lw=2)
        
    # Reference slope -5/3
    ref_k = np.logspace(0, 1.5, 10)
    ref_y = 1e-1 * (ref_k ** (-5/3))
    plt.loglog(ref_k, ref_y, 'k:', label="Kolmogorov (-5/3)", lw=1.5, alpha=0.7)
    
    plt.xlabel(r"Wavenumber $k$ [pc$^{-1}$]", fontsize=14)
    plt.ylabel(r"Solenoidal Energy Spectrum $E_{\rm sol}(k)$", fontsize=14)
    plt.title("Turbulent Energy Spectra Comparison", fontsize=16, fontweight="bold")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_spectra.png", dpi=200)
    plt.close()
    
    # 2. Figure 2: Kinetic Energy Decay over time
    plt.figure(figsize=(10, 6))
    for name, data in all_results.items():
        plt.plot(data["frame_times"], data["KE_spec_vals"], label=f"{name}", color=colors[name], ls=styles[name], lw=2.5)
        
    plt.xlabel("Time $t$ [Myr]", fontsize=14)
    plt.ylabel(r"Specific Kinetic Energy $\langle {\rm KE} \rangle / \langle \rho \rangle$ [code units]", fontsize=14)
    plt.title("Kinetic Energy Decay Over Time", fontsize=16, fontweight="bold")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_ke_decay.png", dpi=200)
    plt.close()
    
    # 3. Figure 3: Enstrophy Spectra Comparison
    plt.figure(figsize=(10, 8))
    for name, data in all_results.items():
        plt.loglog(data["k"], data["Omega"], label=f"{name} (Enstrophy)", color=colors[name], ls=styles[name], lw=2)
        
    plt.xlabel(r"Wavenumber $k$ [pc$^{-1}$]", fontsize=14)
    plt.ylabel(r"Enstrophy Spectrum $\Omega(k)$", fontsize=14)
    plt.title("Enstrophy Spectra Comparison", fontsize=16, fontweight="bold")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_enstrophy.png", dpi=200)
    plt.close()
    
    # Write a detailed comparison markdown report
    generate_comparison_report(all_results, output_dir)

def generate_comparison_report(all_results, output_dir):
    report_path = output_dir / "comparison_report.md"
    
    report_content = []
    report_content.append("# Comparative Run Analysis Report")
    report_content.append("## Comparing High-Resolution, Low-Resolution Baseline, and Subgrid Model Runs")
    report_content.append("\nThis report compares the effective numerical Reynolds number ($Re_{\\rm num}$), numerical viscosity ($\\nu_{\\rm num}$), and spectral properties across three runs:\n")
    report_content.append("1. **hr_build_1024** (High-Resolution reference, $1024\\times 512$ grid)\n")
    report_content.append("2. **lr_build_ism** (Low-Resolution baseline with ISM cooling, $32\\times 16$ grid)\n")
    report_content.append("3. **subgrid_model** (Low-Resolution run with the subgrid model active, $32\\times 16$ grid)\n")
    
    # Summary Table
    report_content.append("### Comparison Metrics Table")
    report_content.append("| Run Name | Grid Resolution | $\\Delta x$ [pc] | Fitted Slope $\\alpha$ | Knee $k_\\nu$ [pc$^{-1}$] | $\\nu_{\\rm num}$ [cm$^2$ s$^{-1}$] | Effective $Re_{\\rm num}$ |")
    report_content.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for name, data in all_results.items():
        grid_str = "{}\\times{}".format(data['ny'], data['nx']) if name != "hr_build_1024" else "1024\\times 512"
        report_content.append(
            "| **{}** | ${}$ | {:.4f} | {:.3f} | {:.2f} | {:.3e} | {:.1f} |".format(
                name, grid_str, data['dx'], data['alpha'], data['k_nu'], data['nu_num_decay_cgs'], data['Re_num_decay']
            )
        )
        
    report_content.append("\n### Physical Interpretation and Discussion")
    
    lr_nu = all_results["lr_build_ism"]["nu_num_decay_cgs"]
    sg_nu = all_results["subgrid_model"]["nu_num_decay_cgs"]
    hr_nu = all_results["hr_build_1024"]["nu_num_decay_cgs"]
    
    report_content.append("- **Numerical Viscosity ($\\nu_{{\\rm num}}$)**:")
    report_content.append("  - The high-resolution reference run (**hr_build_1024**) has the lowest effective viscosity ($\\nu_{{\\rm num}} \\approx {:.2e}\\ {{\\rm cm}}^2\\ {{\\rm s}}^{{-1}}$), corresponding to the highest Reynolds number ($Re_{{\\rm num}} \\approx {:.1f}$), as expected.".format(hr_nu, all_results["hr_build_1024"]["Re_num_decay"]))
    report_content.append("  - The low-resolution baseline run (**lr_build_ism**) has a much higher effective viscosity ($\\nu_{{\\rm num}} \\approx {:.2e}\\ {{\\rm cm}}^2\\ {{\\rm s}}^{{-1}}$) due to the large grid-scale truncation errors at $32\\times16$ resolution.".format(lr_nu))
    report_content.append("  - The run with the active subgrid model (**subgrid_model**) has an effective viscosity of $\\nu_{{\\rm num}} \\approx {:.2e}\\ {{\\rm cm}}^2\\ {{\\rm s}}^{{-1}}$.".format(sg_nu))
    
    report_content.append("\n- **Kinetic Energy Decay Rate ($\\bar{\\epsilon}$)**:")
    report_content.append("  - **hr_build_1024** decay rate: $\\bar{{\\epsilon}} \\approx {:.3e}$ code units.".format(all_results["hr_build_1024"]["epsilon_decay"]))
    report_content.append("  - **lr_build_ism** decay rate: $\\bar{{\\epsilon}} \\approx {:.3e}$ code units.".format(all_results["lr_build_ism"]["epsilon_decay"]))
    report_content.append("  - **subgrid_model** decay rate: $\\bar{{\\epsilon}} \\approx {:.3e}$ code units.".format(all_results["subgrid_model"]["epsilon_decay"]))
    
    report_content.append("\n### Diagnostic Figures")
    report_content.append("1. **Turbulent Energy Spectra Overlay** ([comparison_spectra.png](file://{}/comparison_spectra.png)): Solenoidal kinetic energy spectra showing how the three runs distribute energy across spatial scales.".format(output_dir))
    report_content.append("2. **Kinetic Energy Decay** ([comparison_ke_decay.png](file://{}/comparison_ke_decay.png)): Compares the rate of loss of specific kinetic energy over time.".format(output_dir))
    report_content.append("3. **Enstrophy Spectra Comparison** ([comparison_enstrophy.png](file://{}/comparison_enstrophy.png)): Compares the distribution of enstrophy (vortical activity) across scales.".format(output_dir))
    
    with open(report_path, "w") as f:
        f.write("\n".join(report_content))
    print(f"Generated comparison report: {report_path}")

if __name__ == "__main__":
    run_comparison()
