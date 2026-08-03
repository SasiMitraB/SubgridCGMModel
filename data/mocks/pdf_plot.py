# Python script to plot the actual and predicted PDFs using a discrete form for the PDFs (n bins in log temp space)

import os
import sys

import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from models.conv_nn.pdf_cnn import snapshot_pred, snapshot_pred_with_gate, lambda_cool, compute_cooling_rate

# Set PyTorch device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    if __name__ == '__main__':
        print(f"Using device: {device}")

# =========================
# RUN TOGGLES
# =========================
RUN_PDF_ANIMATION = False
RUN_COOLING_SCATTER = True
RUN_COOLING_HISTOGRAM = True
RUN_PDF_COMPARE_ANIMATION = True
RUN_COOLING_COMPARE_ANIMATION = True
RUN_FOURWAY_COMPARE_ANIMATION = False
RUN_DENSITY_GATE_ANIMATION = True

# =========================
# SETTINGS
# =========================
DEFAULT_FINE_RESOLUTION = (1024, 512)
DEFAULT_DOWNSAMPLE = 32


def _parse_resolution(value: str, default: tuple[int, int]) -> tuple[int, int]:
    try:
        width_str, height_str = value.split(",")
        return int(width_str.strip()), int(height_str.strip())
    except (ValueError, AttributeError):
        return default


resolution = _parse_resolution(
    os.environ.get("PDF_CNN_RESOLUTION", "1024,512"), DEFAULT_FINE_RESOLUTION
)
downsample = int(os.environ.get("PDF_CNN_DOWNSAMPLE", str(DEFAULT_DOWNSAMPLE)))
bins = 40


# =========================
# IMPORT YOUR CLASS
# =========================
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from data_preprocess import simulation_data

# =========================
# QUANTITATIVE METRICS  (Change #4)
# =========================
def print_metrics(true, pred, label):
    """
    Log-space bias, RMSE, and Pearson correlation for cooling rate arrays.
    Only pixels where both true and pred are positive are included.
    """
    mask = (true > 0) & (pred > 0)
    num_pixels = mask.sum()

    print(f"\n--- {label} ---")
    print(f"  Pixels used : {num_pixels} / {true.size}")

    if num_pixels < 2:
        print("  Log-Bias    : N/A (insufficient positive pixels)")
        print("  Log-RMSE    : N/A (insufficient positive pixels)")
        print("  Correlation : N/A (insufficient positive pixels)")
        return

    log_true = np.log10(true[mask])
    log_pred = np.log10(pred[mask])

    bias = np.mean(log_pred - log_true)
    rmse = np.sqrt(np.mean((log_pred - log_true) ** 2))
    corr, _ = pearsonr(log_true, log_pred)

    print(f"  Log-Bias    : {bias:+.3f} dex")
    print(f"  Log-RMSE    :  {rmse:.3f} dex")
    print(f"  Correlation :  {corr:.4f}")


# Define PDF bins and log temperature centers for background color calculations
temp_bins = np.logspace(3.0, 7.0, bins + 1)
log_temp_centers = 0.5 * (np.log10(temp_bins[:-1]) + np.log10(temp_bins[1:]))
cmap = plt.get_cmap("inferno")
norm = colors.Normalize(vmin=3.0, vmax=7.0)

folder_path = f"/home/sasi/Projects/SubgridCGMModel/simulation_outputs/hr_build_1024/cache/sc{resolution}_{downsample}"

PDF_MOCKS_DIR = os.environ.get("PDF_MOCKS_DIR", "mocks/pdf")
os.makedirs(PDF_MOCKS_DIR, exist_ok=True)

mp4_path = os.path.join(PDF_MOCKS_DIR, "pdf_animation.mp4")
first_frame_path = os.path.join(PDF_MOCKS_DIR, "pdf_snapshot_t0.png")
temp_frame_path = os.path.join(PDF_MOCKS_DIR, "temp_snapshot_t0.png")


# =========================
# PARALLEL ANIMATION HELPERS & WORKERS
# =========================
def chunk_frames(nt, num_workers):
    chunks = [[] for _ in range(num_workers)]
    for frame in range(nt):
        chunks[frame % num_workers].append(frame)
    return [c for c in chunks if len(c) > 0]


def generate_parallel_animation(worker_func, nt, output_path, fps, num_workers=16, extra_args=None):
    import tempfile
    import shutil
    import subprocess
    import time
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm

    temp_dir = tempfile.mkdtemp()
    
    # Chunk the frames
    chunks = chunk_frames(nt, num_workers)
    
    if extra_args is None:
        extra_args = ()
        
    print(f"Rendering {nt} frames in parallel using {len(chunks)} workers...")
    
    # Run the worker function in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for fl in chunks:
            futures.append(executor.submit(worker_func, fl, temp_dir, nt, *extra_args))
        
        # Track progress using progress files
        completed_frames = set()
        with tqdm(total=nt, desc=f"Rendering {os.path.basename(output_path)}") as pbar:
            workers_done = False
            while len(completed_frames) < nt:
                # Check if all futures have completed
                if not workers_done:
                    workers_done = all(fut.done() for fut in futures)
                    # Raise immediately if any worker threw an exception
                    if workers_done:
                        for fut in futures:
                            fut.result()

                try:
                    files = os.listdir(temp_dir)
                    for f in files:
                        if f.startswith("progress_") and f.endswith(".txt"):
                            parts = f.split("_")
                            if len(parts) > 1:
                                frame_num = int(parts[1].split(".")[0])
                                if frame_num not in completed_frames:
                                    completed_frames.add(frame_num)
                                    pbar.update(1)
                except FileNotFoundError:
                    pass

                if workers_done and len(completed_frames) < nt:
                    # Workers are all done but we haven't seen all progress files yet.
                    # Give the filesystem a moment to flush, then scan a few more times
                    # before giving up (avoids an infinite loop on lost writes).
                    time.sleep(0.05)
                elif not workers_done:
                    time.sleep(0.1)
            
    print(f"Stitching frames together into {output_path} using ffmpeg...")
    
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(temp_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    shutil.rmtree(temp_dir)


def worker_pdf_compare(frames_list, temp_dir, nt, nx, ny, nb, temp_pdf, conv_temp_pdf, true_cool, true_iso_cool, cnn_cool, log_temp_centers, active_bin_start, active_bin_end, snapshot_compare_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np
    import os

    fig2 = plt.figure(figsize=(ny * 4.0, nx * 1.8))
    gs2 = fig2.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], top=0.90, wspace=0.15)

    # Add section titles for True vs Predicted groups
    fig2.text(
        0.24,
        0.92,
        "TRUE PDFs (Simulation)",
        fontsize=36,
        ha="center",
        va="center",
        weight="bold",
    )
    fig2.text(
        0.72,
        0.92,
        "PREDICTED PDFs (CNN Model)",
        fontsize=36,
        ha="center",
        va="center",
        weight="bold",
    )

    cmap = plt.get_cmap("inferno")
    norm = colors.Normalize(vmin=3.0, vmax=7.0)

    # ---- LEFT (TRUE) ----
    true_axes = np.empty((nx, ny), dtype=object)
    sub_gs_left = gs2[0].subgridspec(nx, ny)

    for i in range(nx):
        for j in range(ny):
            ax = fig2.add_subplot(sub_gs_left[i, j])
            true_axes[i, j] = ax

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("grey")
                spine.set_linewidth(0.3)

            ax.set_xticks([])
            ax.set_yticks([])

    # ---- RIGHT (PRED) ----
    pred_axes = np.empty((nx, ny), dtype=object)
    sub_gs_right = gs2[1].subgridspec(nx, ny)

    for i in range(nx):
        for j in range(ny):
            ax = fig2.add_subplot(sub_gs_right[i, j])
            pred_axes[i, j] = ax

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("grey")
                spine.set_linewidth(0.3)

            ax.set_xticks([])
            ax.set_yticks([])

    # ---- COLORBAR AXIS ----
    cbar_ax2 = fig2.add_subplot(gs2[2])
    sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm2.set_array([])
    cbar2 = fig2.colorbar(sm2, cax=cbar_ax2)
    cbar2.set_label(r"Expectation Value of \log_{10} Temperature", fontsize=36)
    cbar2.ax.tick_params(labelsize=28)

    true_lines, pred_lines = np.empty((nx, ny), dtype=object), np.empty((nx, ny), dtype=object)
    true_texts, pred_texts = np.empty((nx, ny), dtype=object), np.empty((nx, ny), dtype=object)

    for i in range(nx):
        for j in range(ny):
            (lt,) = true_axes[i, j].plot([], [], lw=1)
            (lp,) = pred_axes[i, j].plot([], [], lw=1, color="r")

            true_axes[i, j].set_xlim(0, nb - 1)
            true_axes[i, j].set_yscale("log")
            true_axes[i, j].set_ylim(1e-5, 1.1)

            pred_axes[i, j].set_xlim(0, nb - 1)
            pred_axes[i, j].set_yscale("log")
            pred_axes[i, j].set_ylim(1e-5, 1.1)

            ttxt = true_axes[i, j].text(
                0.95,
                0.95,
                "",
                transform=true_axes[i, j].transAxes,
                fontsize=6,
                color="black",
                ha="right",
                va="top",
            )

            ptxt = pred_axes[i, j].text(
                0.95,
                0.95,
                "",
                transform=pred_axes[i, j].transAxes,
                fontsize=6,
                color="black",
                ha="right",
                va="top",
            )

            true_lines[i, j] = lt
            pred_lines[i, j] = lp
            true_texts[i, j] = ttxt
            pred_texts[i, j] = ptxt

            # Draw active cooling window span ONCE!
            true_axes[i, j].axvspan(
                active_bin_start,
                active_bin_end,
                color="green",
                alpha=0.12,
                lw=0,
            )
            pred_axes[i, j].axvspan(
                active_bin_start,
                active_bin_end,
                color="green",
                alpha=0.12,
                lw=0,
            )

    x = np.arange(nb)

    # Render loop
    for frame in frames_list:
        true_pdf = temp_pdf[frame]
        pred_pdf = conv_temp_pdf[frame]

        for i in range(nx):
            for j in range(ny):
                ii = nx - 1 - i

                y_true = true_pdf[:, ii, j]
                y_pred = pred_pdf[:, ii, j]

                exp_val_true = np.sum(y_true * log_temp_centers)
                exp_val_pred = np.sum(y_pred * log_temp_centers)

                y_true_plot = y_true + 1e-8
                y_pred_plot = y_pred + 1e-8

                true_lines[i, j].set_data(x, y_true_plot)
                pred_lines[i, j].set_data(x, y_pred_plot)

                bg_true = cmap(norm(exp_val_true))
                bg_pred = cmap(norm(exp_val_pred))
                true_axes[i, j].set_facecolor(bg_true)
                pred_axes[i, j].set_facecolor(bg_pred)

                lum_true = 0.299 * bg_true[0] + 0.587 * bg_true[1] + 0.114 * bg_true[2]
                true_color = "white" if lum_true < 0.5 else "black"
                true_lines[i, j].set_color(true_color)
                true_texts[i, j].set_color(true_color)

                lum_pred = 0.299 * bg_pred[0] + 0.587 * bg_pred[1] + 0.114 * bg_pred[2]
                pred_color = "white" if lum_pred < 0.5 else "black"
                pred_lines[i, j].set_color(pred_color)
                pred_texts[i, j].set_color(pred_color)

                tc = true_cool[frame, ii, j]
                tic = true_iso_cool[frame, ii, j]
                pc = cnn_cool[frame, ii, j]

                true_texts[i, j].set_text(f"F:{tc:.1e}\nI:{tic:.1e}")
                pred_texts[i, j].set_text(f"{pc:.1e}")

        fig2.suptitle(f"True vs Predicted PDFs | t = {frame}", fontsize=48, y=0.96)

        if frame == 0:
            fig2.savefig(snapshot_compare_path, dpi=300)

        # dpi=40 is fast and sharp enough
        fig2.savefig(os.path.join(temp_dir, f"frame_{frame:04d}.png"), dpi=40)
        
        # Write progress marker
        with open(os.path.join(temp_dir, f"progress_{frame}.txt"), "w") as f:
            pass

    plt.close(fig2)


def worker_cooling_compare(frames_list, temp_dir, nt, cool_vmin, cool_vmax, true_iso_cool, cnn_cool, snapshot_cooling_compare_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np
    import os

    fig3 = plt.figure(figsize=(12, 6))
    gs3 = fig3.add_gridspec(
        1,
        3,
        width_ratios=[1, 1, 0.05],
        top=0.85,
        bottom=0.15,
        left=0.08,
        right=0.90,
        wspace=0.15,
    )

    # Add section titles
    fig3.text(
        0.28,
        0.90,
        "TRUE ISOBARIC COOLING (Simulation)",
        fontsize=16,
        ha="center",
        va="center",
        weight="bold",
    )
    fig3.text(
        0.70,
        0.90,
        "PREDICTED COOLING (CNN Model)",
        fontsize=16,
        ha="center",
        va="center",
        weight="bold",
    )

    norm_cool = colors.LogNorm(vmin=cool_vmin, vmax=cool_vmax)
    cmap_cool = plt.get_cmap("viridis")

    # ---- LEFT (TRUE) ----
    ax_t = fig3.add_subplot(gs3[0])
    ax_t.set_title("True Isobaric", fontsize=14)
    # ---- RIGHT (PRED) ----
    ax_p = fig3.add_subplot(gs3[1])
    ax_p.set_title("CNN Prediction", fontsize=14)

    # Plot initial frames
    im_t = ax_t.imshow(
        np.clip(true_iso_cool[0], cool_vmin, None),
        origin="lower",
        cmap=cmap_cool,
        norm=norm_cool,
    )
    im_p = ax_p.imshow(
        np.clip(cnn_cool[0], cool_vmin, None),
        origin="lower",
        cmap=cmap_cool,
        norm=norm_cool,
    )

    for ax in [ax_t, ax_p]:
        ax.set_xlabel("Y (pixels)", fontsize=12)
        ax.set_ylabel("X (pixels)", fontsize=12)

    # ---- COLORBAR AXIS ----
    cbar_ax3 = fig3.add_subplot(gs3[2])
    sm3 = plt.cm.ScalarMappable(cmap=cmap_cool, norm=norm_cool)
    sm3.set_array([])
    cbar3 = fig3.colorbar(sm3, cax=cbar_ax3)
    cbar3.set_label("Cooling Rate (erg / cm$^3$ / s)", fontsize=14)
    cbar3.ax.tick_params(labelsize=10)

    for frame in frames_list:
        im_t.set_data(np.clip(true_iso_cool[frame], cool_vmin, None))
        im_p.set_data(np.clip(cnn_cool[frame], cool_vmin, None))

        fig3.suptitle(f"Cooling Rate Comparison | t = {frame}", fontsize=18, y=0.96)

        if frame == 0:
            fig3.savefig(snapshot_cooling_compare_path, dpi=300)

        fig3.savefig(os.path.join(temp_dir, f"frame_{frame:04d}.png"), dpi=100)
        
        # Write progress marker
        with open(os.path.join(temp_dir, f"progress_{frame}.txt"), "w") as f:
            pass

    plt.close(fig3)


def worker_density_gate(frames_list, temp_dir, nt, rho_vmin, rho_vmax, sim_data_rho, cg_rho, cnn_gate_maps, snapshot_density_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    fig5, axes5 = plt.subplots(1, 3, figsize=(12, 7))
    ax_fine, ax_cg, ax_gate = axes5

    im_fine = ax_fine.imshow(
        np.log10(sim_data_rho[0] + 1e-10), 
        origin="lower", cmap="viridis", vmin=rho_vmin, vmax=rho_vmax
    )
    ax_fine.set_title("True Density (Before CG) [$\log_{10}$]", fontsize=14)
    fig5.colorbar(im_fine, ax=ax_fine, fraction=0.046, pad=0.04)

    im_cg = ax_cg.imshow(
        np.log10(cg_rho[0] + 1e-10), 
        origin="lower", cmap="viridis", vmin=rho_vmin, vmax=rho_vmax
    )
    ax_cg.set_title("Coarse-Grained Density [$\log_{10}$]", fontsize=14)
    fig5.colorbar(im_cg, ax=ax_cg, fraction=0.046, pad=0.04)

    im_gate = ax_gate.imshow(
        cnn_gate_maps[0], 
        origin="lower", cmap="inferno", vmin=0.0, vmax=1.0
    )
    ax_gate.set_title("CNN Gate Output $g(x,y)$", fontsize=14)
    fig5.colorbar(im_gate, ax=ax_gate, fraction=0.046, pad=0.04)

    for ax in axes5:
        ax.set_xlabel("Y (cells)")
        ax.set_ylabel("X (cells)")

    title5 = fig5.suptitle(f"Density & Gate Comparison | t = 0", fontsize=18, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], w_pad=1.0)

    for frame in frames_list:
        im_fine.set_data(np.log10(sim_data_rho[frame] + 1e-10))
        im_cg.set_data(np.log10(cg_rho[frame] + 1e-10))
        im_gate.set_data(cnn_gate_maps[frame])
        
        title5.set_text(f"Density & Gate Comparison | t = {frame}")

        if frame == 0:
            fig5.savefig(snapshot_density_path, dpi=300)

        fig5.savefig(os.path.join(temp_dir, f"frame_{frame:04d}.png"), dpi=100)
        
        # Write progress marker
        with open(os.path.join(temp_dir, f"progress_{frame}.txt"), "w") as f:
            pass

    plt.close(fig5)


def batch_predict_with_gate(sim_data, downsample, resolution, device):
    from models.conv_nn.pdf_cnn import ConvNN, MODEL_SAVE_DIR, in_channels, layer_size1, layer_size2, layer_size3, layer_size4, out_channels, kernel_size
    import torch
    import numpy as np
    from tqdm import tqdm
    import os

    nt = sim_data.rho.shape[0]
    shape = (resolution[0] // downsample, resolution[1] // downsample)
    nx, ny = shape
    
    # 1. Load model once
    model_path = os.path.join(MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}.pth")
    cnn_model = ConvNN(
        in_channels,
        layer_size1,
        layer_size2,
        layer_size3,
        layer_size4,
        out_channels,
        kernel_size,
    ).to(device)
    cnn_model.load_state_dict(torch.load(model_path, map_location=device))
    cnn_model.eval()
    
    # 2. Load normalization stats once
    input_mean = torch.tensor(
        np.load(os.path.join(MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}_input_mean.npy")),
        dtype=torch.float32,
    ).to(device)
    input_std = torch.tensor(
        np.load(os.path.join(MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}_input_std.npy")),
        dtype=torch.float32,
    ).to(device)
    
    conv_temp_pdf = np.zeros((nt, out_channels, nx, ny))
    cnn_gate_maps = np.zeros((nt, nx, ny))
    cnn_vort_maps = np.zeros((nt, nx, ny))
    
    print("Running batch predictions...")
    with torch.no_grad():
        for i in tqdm(range(nt), desc="Predicting CNN temperature PDFs"):
            # Coarse-grain inputs
            cg_rho = sim_data.coarse_grain(sim_data.rho[i])
            cg_temp = sim_data.coarse_grain(sim_data.temp[i])
            cg_ux = sim_data.coarse_grain(sim_data.ux[i])
            cg_uy = sim_data.coarse_grain(sim_data.uy[i])
            cg_ps = sim_data.coarse_grain(sim_data.ps[i])
            
            # Stack and build input tensor
            stack = np.stack([cg_rho, cg_temp, cg_ux, cg_uy, cg_ps], axis=0).astype(np.float32)
            input_tensor = torch.from_numpy(stack).unsqueeze(0).to(device) # (1, C, nx, ny)
            
            # Normalize
            input_tensor = (input_tensor - input_mean) / input_std
            
            # Predict
            x_enriched = cnn_model.mixing(input_tensor)
            mixing_feats = x_enriched[:, -cnn_model._N_MIXING :, :, :]
            gate_raw = cnn_model.gate_branch(mixing_feats)
            
            features = cnn_model.encoder(x_enriched)
            logits = cnn_model.decoder(features)
            pdf_tensor = cnn_model.pdf_activation(logits, gate_raw)
            
            conv_temp_pdf[i] = pdf_tensor[0].cpu().numpy()
            cnn_gate_maps[i] = gate_raw[0, 0].cpu().numpy()
            cnn_vort_maps[i] = mixing_feats[0, 0].cpu().numpy()
            
    return conv_temp_pdf, cnn_gate_maps, cnn_vort_maps

if __name__ == '__main__':
    
    
    # =========================
    # LOAD DATA
    # =========================
    print("Loading data...")
    
    sim_data = simulation_data()
    sim_data.down_sample = downsample
    sim_data.resolution = resolution
    
    sim_data.rho = np.load(f"{folder_path}/rho.npy")
    sim_data.temp = np.load(f"{folder_path}/temp.npy")
    sim_data.pressure = np.load(f"{folder_path}/pressure.npy")
    sim_data.ux = np.load(f"{folder_path}/ux.npy")
    sim_data.uy = np.load(f"{folder_path}/uy.npy")
    sim_data.eint = np.load(f"{folder_path}/eint.npy")
    sim_data.ps = np.load(f"{folder_path}/ps.npy")
    
    print("Data loaded.")
    
    # Derive the Code's internal Temperature Unit factor ONCE globally
    # T_phys = T_unit * (P_code / rho_code), so T_unit = T_phys * (rho_code / P_code)
    # (If your code uses v0 = 1 km/s and mu = 0.62, this will perfectly equal ~75.0 K)
    mu = 0.62
    T_unit = float(np.median(sim_data.temp[0] * sim_data.rho[0] / sim_data.pressure[0]))
    print(f"Derived internal T_unit = {T_unit:.2f} K")
    
    
    # =========================
    # COMPUTE PDF
    # =========================
    print("Computing pixel PDFs...")
    
    temp_pdf = sim_data.calc_pixel_pdf(bins=bins)
    temp_pdf /= temp_pdf.sum(axis=1, keepdims=True) + 1e-12
    
    nt, nb, nx, ny = temp_pdf.shape
    print(f"Shape: nt={nt}, bins={nb}, nx={nx}, ny={ny}")
    
    # ─── Predict CNN temperature PDFs, gate, and vorticity (optimized batch) ───
    conv_temp_pdf, cnn_gate_maps, cnn_vort_maps = batch_predict_with_gate(
        sim_data, downsample, resolution, device
    )
    conv_temp_pdf /= conv_temp_pdf.sum(axis=1, keepdims=True) + 1e-12
    
    
    # =========================
    # COARSE-GRAIN TEMP
    # =========================
    print("Computing coarse-grained temperature...")
    
    cg_rho = np.zeros((nt, nx, ny))
    cg_temp = np.zeros((nt, nx, ny))
    
    for t in tqdm(range(nt), desc="Coarse-graining temperature & density"):
        cg_rho[t] = sim_data.coarse_grain(sim_data.rho[t])
        cg_temp[t] = sim_data.coarse_grain(sim_data.temp[t])
    
    
    # =========================
    # FIGURE SETUP
    # =========================
    if RUN_PDF_ANIMATION:
        fig = plt.figure(figsize=(ny * 2.2, nx * 1.8))
    
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
    
        # ---- PDF GRID ----
        pdf_axes = np.empty((nx, ny), dtype=object)
        sub_gs = gs[0].subgridspec(nx, ny)
    
        for i in range(nx):
            for j in range(ny):
                ax = fig.add_subplot(sub_gs[i, j])
                pdf_axes[i, j] = ax
    
                # square borders
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color("grey")
                    spine.set_linewidth(0.3)
    
                ax.set_xticks([])
                ax.set_yticks([])
    
        # ---- TEMP PANEL ----
        temp_ax = fig.add_subplot(gs[1])
    
        log_temp0 = np.log10(cg_temp[0] + 1e-8)
        temp_im = temp_ax.imshow(log_temp0, origin="lower", cmap="inferno", norm=norm)
    
        temp_ax.set_title(r"$\log_{10}$ Temp (CG)")
        cbar = plt.colorbar(temp_im, ax=temp_ax, fraction=0.046)
        cbar.set_label(r"$\log_{10}$ Temperature / Expectation Value", fontsize=24)
        cbar.ax.tick_params(labelsize=18)
    
        # =========================
        # INIT LINES
        # =========================
        x = np.arange(nb)
    
        lines = []
        for i in range(nx):
            row = []
            for j in range(ny):
                (line,) = pdf_axes[i, j].plot([], [], lw=1)
                pdf_axes[i, j].set_xlim(0, nb - 1)
                pdf_axes[i, j].set_ylim(0, 1)
                row.append(line)
            lines.append(row)
    
        # =========================
        # INIT FUNCTION
        # =========================
        def init():
            for i in range(nx):
                for j in range(ny):
                    lines[i][j].set_data([], [])
                    pdf_axes[i, j].set_facecolor("black")
            return sum(lines, [])
    
        # =========================
        # UPDATE FUNCTION
        # =========================
        def update(frame):
    
            pdf = temp_pdf[frame]
    
            for i in range(nx):
                for j in range(ny):
                    # FIX: flip vertically to match imshow
                    ii = nx - 1 - i
    
                    y = pdf[:, ii, j]
    
                    # Compute expectation value of log10 temperature
                    exp_val = np.sum(y * log_temp_centers)
    
                    y = y / (y.max() + 1e-8)
    
                    lines[i][j].set_data(x, y)
    
                    # Set background color of subplot based on expectation value
                    bg_color = cmap(norm(exp_val))
                    pdf_axes[i, j].set_facecolor(bg_color)
    
                    # Determine dynamic contrasting line color (luminance-based)
                    lum = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
                    lines[i][j].set_color("white" if lum < 0.5 else "black")
    
            # update temp (log scale CG)
            log_temp = np.log10(cg_temp[frame] + 1e-8)
            temp_im.set_data(log_temp)
    
            fig.suptitle(f"t = {frame}", fontsize=48)
    
            # Save first frame
            if frame == 0:
                fig.savefig(first_frame_path, dpi=300)
                plt.imsave(temp_frame_path, log_temp, cmap="inferno")
                print(f"Saved PDF snapshot → {first_frame_path}")
                print(f"Saved temp snapshot → {temp_frame_path}")
    
            return sum(lines, []) + [temp_im]
    
        # =========================
        # ANIMATION
        # =========================
        print("Creating animation...")
    
        anim = animation.FuncAnimation(fig, update, frames=nt, init_func=init, blit=False)
    
        print("Saving MP4...")
        with tqdm(total=nt, desc="Saving MP4") as pbar:
            anim.save(
                mp4_path,
                writer="ffmpeg",
                fps=10,
                progress_callback=lambda i, n: pbar.update(1),
            )
    
        print(f"Saved animation → {mp4_path}")
    
        plt.close()
    
    
    # =========================
    # TRUE vs PRED PDF COMPARISON
    # =========================
    # =====================================================================
    # COOLING COMPUTATION BLOCK  (Changes #1, #2)
    # Three separate cooling fields to isolate model error vs closure error
    # =====================================================================
    print("Computing cooling rates (True Fine, True Isobaric, CNN Isobaric)...")
    
    # Shared constants — identical across all three computations
    mu = 0.62
    kb = 1.380649e-16
    unit_fix = 1.975e27
    
    # ---- PDF bin centres (geometric mean of edges) ----
    temp_bins = np.logspace(3, 7, nb + 1)
    temp_centers = np.sqrt(temp_bins[:-1] * temp_bins[1:])  # (nb,)
    
    # ---- Active cooling window for shading (Change #3) ----
    active_bin_start = np.searchsorted(temp_centers, 10**4.5)
    active_bin_end = np.searchsorted(temp_centers, 10**5.5)
    
    # ------------------------------------------------------------------
    # (A) Cool_True_Fine : fine-grid truth averaged to coarse blocks
    #     Uses compute_cooling_rate() in scalar mode — no approximation.
    # ------------------------------------------------------------------
    print("  (A) True Fine cooling...")
    true_cool = np.zeros((nt, nx, ny))
    for t in tqdm(range(nt), desc="True fine cooling"):
        rho = sim_data.rho[t]
        temp = sim_data.temp[t]
        n = rho / mu
        lam = lambda_cool(temp, mask=True)
        fine_cool = lam * n**2 * unit_fix
        true_cool[t] = sim_data.coarse_grain(fine_cool)
    
    
    # ------------------------------------------------------------------
    # (B) Coarse-grain pressure (shared by both isobaric variants)
    # ------------------------------------------------------------------
    cg_pressure = np.zeros((nt, nx, ny))
    for t in tqdm(range(nt), desc="Coarse-graining pressure"):
        cg_pressure[t] = sim_data.coarse_grain(sim_data.pressure[t])
    
    # ------------------------------------------------------------------
    # (C) Cool_True_Isobaric : use TRUE PDF + isobaric n(T)=P/(kb*T)
    #     If this disagrees badly with (A) → isobaric assumption is wrong.
    # ------------------------------------------------------------------
    print("  (C) True Isobaric cooling (using simulation PDF)...")
    true_iso_cool = np.zeros((nt, nx, ny))
    for t in tqdm(range(nt), desc="True isobaric cooling"):
        true_iso_cool[t] = compute_cooling_rate(
            temp_pdf[t],  # (nb, nx, ny)  – true PDF
            temp_centers,  # (nb,)
            pressure=cg_pressure[t],
            is_pdf=True,
            is_isobaric=True,
            T_unit=T_unit,
        )
    
    # ------------------------------------------------------------------
    # (D) Cool_CNN_Isobaric : use CNN PDF + isobaric n(T)=P/(kb*T)
    #     Compared to (C) this isolates the CNN's contribution only.
    # ------------------------------------------------------------------
    print("  (D) CNN Isobaric cooling (using CNN PDF)...")
    cnn_cool = np.zeros((nt, nx, ny))
    for t in tqdm(range(nt), desc="CNN isobaric cooling"):
        cnn_cool[t] = compute_cooling_rate(
            conv_temp_pdf[t],  # (nb, nx, ny)  – CNN PDF
            temp_centers,  # (nb,)
            pressure=cg_pressure[t],
            is_pdf=True,
            is_isobaric=True,
            T_unit=T_unit,
        )
    
    print("Cooling computation done.")
    
    # =========================
    # METRICS  (Change #4)
    # =========================
    print("\n=== Quantitative Benchmarking Metrics ===")
    # Flatten across all pixels and timesteps for a global assessment
    print_metrics(
        true_cool.flatten(),
        true_iso_cool.flatten(),
        "Physics Closure Error  (True Fine vs True Isobaric)",
    )
    print_metrics(
        true_iso_cool.flatten(),
        cnn_cool.flatten(),
        "CNN Prediction Error   (True Isobaric vs CNN Isobaric)",
    )
    print_metrics(
        true_cool.flatten(),
        cnn_cool.flatten(),
        "Total Error            (True Fine vs CNN Isobaric)",
    )
    
    # =========================
    # COARSE-GRAIN TEMPERATURE
    # =========================
    if RUN_COOLING_SCATTER:
        # For a single timestep, scatter true_iso_cool vs cnn_cool
        # coloured by: (a) CNN active-window mass, (b) true active-window mass
        t = 0
        active_mass_true = temp_pdf[t, active_bin_start:active_bin_end].sum(
            axis=0
        )  # (nx,ny)
        active_mass_cnn = conv_temp_pdf[t, active_bin_start:active_bin_end].sum(axis=0)
    
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sc = axes[0].scatter(active_mass_true.flat, active_mass_cnn.flat, s=2, alpha=0.3)
        axes[0].set_xlabel("True active-window mass")
        axes[0].set_ylabel("CNN active-window mass")
        axes[0].set_title("PDF mass in cooling window")
    
        sc2 = axes[1].scatter(
            true_iso_cool[t].flat,
            cnn_cool[t].flat,
            c=active_mass_cnn.flat,
            s=2,
            alpha=0.3,
            cmap="hot",
        )
        plt.colorbar(sc2, ax=axes[1], label="CNN active-window mass")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_title("Cooling scatter coloured by CNN window mass")
        plt.tight_layout()
        plt.show()
    
        cg_temp = np.zeros((nt, nx, ny))
    
        for t in tqdm(range(nt), desc="Coarse-graining temperature (global scatter)"):
            cg_temp[t] = sim_data.coarse_grain(sim_data.temp[t])
    
        # ============================================================
        # GLOBAL COOLING RATE DIAGNOSTICS: 2D Density & Residuals
        # ============================================================
        print("Creating global cooling diagnostic plots...")
    
        from matplotlib.colors import LogNorm
    
        SCATTER_MIN = 1e0  # Points with x OR y below this threshold are excluded entirely
    
        # Flatten the three cooling fields (raw values; no eps clipping)
        temp_flat = cg_temp.flatten()
        flat_true = true_cool.flatten()
        flat_true_iso = true_iso_cool.flatten()
        flat_cnn = cnn_cool.flatten()
    
        def add_running_median(ax, xv, yv, n_bins=25):
            logx = np.log10(xv)
            bin_edges = np.linspace(logx.min(), logx.max(), n_bins + 1)
            bin_idx = np.digitize(logx, bin_edges)
            xs, meds, lo, hi = [], [], [], []
            for b in range(1, n_bins + 1):
                sel = bin_idx == b
                if sel.sum() < 5:
                    continue
                xs.append(10 ** (0.5 * (bin_edges[b-1] + bin_edges[b])))
                yb = yv[sel]
                meds.append(np.median(yb))
                lo.append(np.percentile(yb, 16))
                hi.append(np.percentile(yb, 84))
            if len(xs) > 0:
                ax.plot(xs, meds, color="cyan", lw=2, label="Median")
                ax.fill_between(xs, lo, hi, color="cyan", alpha=0.2, label="16–84%")
                ax.legend(fontsize=8, loc="upper left")
    
        fig_sc, axes_sc = plt.subplots(2, 2, figsize=(14, 12))
        ax_closure = axes_sc[0, 0]
        ax_pred = axes_sc[0, 1]
        ax_total = axes_sc[1, 0]
        ax_resid = axes_sc[1, 1]
    
        # Panel 1: Physics Closure Error (True Fine vs True Isobaric)
        mask1 = (flat_true >= SCATTER_MIN) & (flat_true_iso >= SCATTER_MIN)
        xm1, ym1 = flat_true[mask1], flat_true_iso[mask1]
        tm1 = np.clip(temp_flat[mask1], 1e3, None)
    
        hb1 = ax_closure.hexbin(
            xm1, ym1,
            C=tm1,
            reduce_C_function=np.median,
            xscale="log", yscale="log",
            gridsize=60,
            cmap="plasma",
            norm=LogNorm(vmin=1e3, vmax=1e8),
            mincnt=1,
            rasterized=True,
        )
        if len(xm1):
            _lim = [min(xm1.min(), ym1.min()), max(xm1.max(), ym1.max())]
            ax_closure.plot(_lim, _lim, "r--", lw=1)
        ax_closure.set_xscale("log")
        ax_closure.set_yscale("log")
        ax_closure.set_xlabel("True Fine")
        ax_closure.set_ylabel("True Isobaric")
        ax_closure.set_title(f"Physics Closure Error\n({mask1.sum():,} / {len(flat_true):,} points)")
        plt.colorbar(hb1, ax=ax_closure, label="Median Temperature (K)")
        add_running_median(ax_closure, xm1, ym1)
    
        # Panel 2: CNN Prediction Error (True Isobaric vs CNN Isobaric)
        mask2 = (flat_true_iso >= SCATTER_MIN) & (flat_cnn >= SCATTER_MIN)
        xm2, ym2 = flat_true_iso[mask2], flat_cnn[mask2]
    
        hb2 = ax_pred.hexbin(
            xm2, ym2,
            xscale="log", yscale="log",
            gridsize=60,
            bins="log",
            cmap="viridis",
            mincnt=1,
            rasterized=True,
        )
        if len(xm2):
            _lim = [min(xm2.min(), ym2.min()), max(xm2.max(), ym2.max())]
            ax_pred.plot(_lim, _lim, "r--", lw=1)
        ax_pred.set_xscale("log")
        ax_pred.set_yscale("log")
        ax_pred.set_xlabel("True Isobaric")
        ax_pred.set_ylabel("CNN Isobaric")
        ax_pred.set_title(f"CNN Prediction Error\n({mask2.sum():,} / {len(flat_true_iso):,} points)")
        plt.colorbar(hb2, ax=ax_pred, label="log$_{10}$(count)")
        add_running_median(ax_pred, xm2, ym2)
    
        # Panel 3: Total Error (True Fine vs CNN Isobaric)
        mask3 = (flat_true >= SCATTER_MIN) & (flat_cnn >= SCATTER_MIN)
        xm3, ym3 = flat_true[mask3], flat_cnn[mask3]
    
        hb3 = ax_total.hexbin(
            xm3, ym3,
            xscale="log", yscale="log",
            gridsize=60,
            bins="log",
            cmap="viridis",
            mincnt=1,
            rasterized=True,
        )
        if len(xm3):
            _lim = [min(xm3.min(), ym3.min()), max(xm3.max(), ym3.max())]
            ax_total.plot(_lim, _lim, "r--", lw=1)
        ax_total.set_xscale("log")
        ax_total.set_yscale("log")
        ax_total.set_xlabel("True Fine")
        ax_total.set_ylabel("CNN Isobaric")
        ax_total.set_title(f"Total Error\n({mask3.sum():,} / {len(flat_true):,} points)")
        plt.colorbar(hb3, ax=ax_total, label="log$_{10}$(count)")
        add_running_median(ax_total, xm3, ym3)
    
        # Panel 4: CNN Residuals (log10(CNN / True) vs True Isobaric)
        mask4 = (flat_true_iso >= SCATTER_MIN) & (flat_cnn >= SCATTER_MIN)
        xm4, ym4 = flat_true_iso[mask4], np.log10(flat_cnn[mask4] / flat_true_iso[mask4])
    
        hb4 = ax_resid.hexbin(
            xm4, ym4,
            xscale="log",
            gridsize=60,
            bins="log",
            cmap="viridis",
            mincnt=1,
            rasterized=True,
        )
        ax_resid.axhline(0, color="r", linestyle="--", lw=1)
        ax_resid.set_xscale("log")
        ax_resid.set_xlabel("True Isobaric")
        ax_resid.set_ylabel(r"$\log_{10}$(CNN Isobaric / True Isobaric)")
        ax_resid.set_title(f"CNN Residuals\n({mask4.sum():,} / {len(flat_true_iso):,} points)")
        plt.colorbar(hb4, ax=ax_resid, label="log$_{10}$(count)")
        add_running_median(ax_resid, xm4, ym4)
    
        fig_sc.suptitle("Cooling Rate Comparisons & Diagnostics (All Pixels, All Timesteps)", fontsize=16)
        fig_sc.tight_layout()
        fig_sc.savefig(
            os.path.join(PDF_MOCKS_DIR, "pdf_cooling_scatter_threeway.png"), dpi=200
        )
        plt.show()
        print("Saved multi-panel cooling diagnostic plot.")
    
    
    # ============================================================
    # HISTOGRAM: Zero-Fraction + Positive-Only log10 (Change #5)
    # ============================================================
    if RUN_COOLING_HISTOGRAM:
        print("Creating improved histogram plots...")
    
        _fields = {
            "True Fine": true_cool.flatten(),
            "True Isobaric": true_iso_cool.flatten(),
            "CNN Isobaric": cnn_cool.flatten(),
        }
        _colors = ["steelblue", "darkorange", "mediumseagreen"]
    
        fig_hist, (ax_zero, ax_pos) = plt.subplots(1, 2, figsize=(14, 5))
    
        # ---- Panel 1: Zero-fraction bar chart ----
        zero_fracs = [np.mean(v == 0.0) * 100 for v in _fields.values()]
        bars = ax_zero.bar(_fields.keys(), zero_fracs, color=_colors, width=0.5)
        for bar, frac in zip(bars, zero_fracs):
            ax_zero.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{frac:.1f}%",
                ha="center",
                va="bottom",
                fontsize=11,
            )
        ax_zero.set_ylabel("Fraction of pixels with cooling = 0 (%)")
        ax_zero.set_title("Zero-Cooling Fraction")
        ax_zero.set_ylim(0, max(zero_fracs) * 1.25 + 1)
    
        # ---- Panel 2: log10(cooling) distribution for positive pixels ----
        for (label, vals), col in zip(_fields.items(), _colors):
            pos = vals[vals > 0]
            if len(pos) == 0:
                continue
            log_pos = np.log10(pos)
            ax_pos.hist(
                log_pos,
                bins=80,
                density=True,
                histtype="step",
                linewidth=2,
                label=label,
                color=col,
            )
    
        ax_pos.set_xlabel(r"$\log_{10}$(Cooling Rate)")
        ax_pos.set_ylabel("Probability Density")
        ax_pos.set_title(r"Distribution of Positive $\log_{10}$(Cooling)")
        ax_pos.legend()
        ax_pos.set_yscale("log")
    
        fig_hist.suptitle("Cooling Rate Histograms", fontsize=14)
        fig_hist.tight_layout()
        fig_hist.savefig(os.path.join(PDF_MOCKS_DIR, "pdf_cooling_histogram.png"), dpi=200)
        plt.show()
        print("Saved histogram plot.")
    
    # =========================
    # TRUE vs PRED PDF ANIMATION
    # =========================
    if RUN_PDF_COMPARE_ANIMATION:
        print("Creating TRUE vs PRED PDF comparison animation in parallel...")
    
        mp4_path_compare = os.path.join(PDF_MOCKS_DIR, "pdf_compare_animation.mp4")
        snapshot_compare_path = os.path.join(PDF_MOCKS_DIR, "pdf_compare_t0.png")
    
        generate_parallel_animation(
            worker_pdf_compare,
            nt,
            mp4_path_compare,
            fps=10,
            num_workers=16,
            extra_args=(
                nx,
                ny,
                nb,
                temp_pdf,
                conv_temp_pdf,
                true_cool,
                true_iso_cool,
                cnn_cool,
                log_temp_centers,
                active_bin_start,
                active_bin_end,
                snapshot_compare_path,
            )
        )
        print(f"Saved comparison animation → {mp4_path_compare}")
    
    
    # ============================================================
    # COOLING COMPARISON ANIMATION (NEW)
    # ============================================================
    if RUN_COOLING_COMPARE_ANIMATION:
        print("Creating cooling comparison animation in parallel...")
    
        mp4_path_cooling_compare = os.path.join(
            PDF_MOCKS_DIR, "pdf_cooling_compare_animation.mp4"
        )
        snapshot_cooling_compare_path = os.path.join(
            PDF_MOCKS_DIR, "pdf_cooling_compare_t0.png"
        )
    
        # Compute global vmin/vmax for cooling rates
        all_pos_cool = np.concatenate(
            [true_iso_cool[true_iso_cool > 0], cnn_cool[cnn_cool > 0]]
        )
        if len(all_pos_cool) > 0:
            cool_vmin = max(np.percentile(all_pos_cool, 1), 1e-10)
            cool_vmax = np.percentile(all_pos_cool, 99)
        else:
            cool_vmin = 1e-28
            cool_vmax = 1e-18
    
        generate_parallel_animation(
            worker_cooling_compare,
            nt,
            mp4_path_cooling_compare,
            fps=10,
            num_workers=16,
            extra_args=(
                cool_vmin,
                cool_vmax,
                true_iso_cool,
                cnn_cool,
                snapshot_cooling_compare_path,
            )
        )
        print(f"Saved cooling comparison animation → {mp4_path_cooling_compare}")
    
    
    # ============================================================
    # FOUR-WAY COMPARISON ANIMATION
    # Panels: True Cooling | Predicted Cooling | Vorticity | Gate
    # ============================================================
    if RUN_FOURWAY_COMPARE_ANIMATION:
        print("Creating four-way comparison animation (cooling / vorticity / gate)...")
    
        mp4_path_fourway = os.path.join(PDF_MOCKS_DIR, "pdf_fourway_compare_animation.mp4")
        snapshot_fourway_path = os.path.join(PDF_MOCKS_DIR, "pdf_fourway_compare_t0.png")
    
        # ---- Shared colormap + norm for cooling panels ----
        all_pos_cool4 = np.concatenate([true_cool[true_cool > 0], cnn_cool[cnn_cool > 0]])
        if len(all_pos_cool4) > 0:
            cool4_vmin = max(np.percentile(all_pos_cool4, 1), 1e-10)
            cool4_vmax = np.percentile(all_pos_cool4, 99)
        else:
            cool4_vmin, cool4_vmax = 1e-28, 1e-18
        norm_cool4 = colors.LogNorm(vmin=cool4_vmin, vmax=cool4_vmax)
        cmap_cool4 = plt.get_cmap("magma")
    
        # ---- Vorticity: symmetric linear norm across all timesteps ----
        vort_abs_max = np.percentile(np.abs(cnn_vort_maps), 99)
        vort_abs_max = max(vort_abs_max, 1e-30)
        norm_vort = colors.Normalize(vmin=-vort_abs_max, vmax=vort_abs_max)
        cmap_vort = plt.get_cmap("RdBu_r")
    
        # ---- Gate: always in [0, 1] ----
        norm_gate = colors.Normalize(vmin=0.0, vmax=1.0)
        cmap_gate = plt.get_cmap("viridis")
    
        # ---- Build figure: 4 image panels + 4 colorbars ----
        fig4 = plt.figure(figsize=(22, 6))
        fig4.patch.set_facecolor("#0d0d0d")
    
        # gridspec: 4 image cols + 4 narrow cbar cols
        gs4 = fig4.add_gridspec(
            1,
            8,
            width_ratios=[1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05],
            top=0.82,
            bottom=0.12,
            left=0.05,
            right=0.97,
            wspace=0.05,
        )
    
        panel_specs = [
            (
                gs4[0],
                gs4[1],
                "True Cooling",
                cmap_cool4,
                norm_cool4,
                "Cooling Rate\n(erg / cm³ / s)",
            ),
            (
                gs4[2],
                gs4[3],
                "Predicted Cooling",
                cmap_cool4,
                norm_cool4,
                "Cooling Rate\n(erg / cm³ / s)",
            ),
            (gs4[4], gs4[5], "Vorticity ω", cmap_vort, norm_vort, "Vorticity (code units)"),
            (gs4[6], gs4[7], "Gate g(x,y)", cmap_gate, norm_gate, "Gate value ∈ (0, 1)"),
        ]
    
        axes4, ims4, cbars4 = [], [], []
        for img_spec, cbar_spec, title, cmap_p, norm_p, cbar_lbl in panel_specs:
            ax = fig4.add_subplot(img_spec)
            ax.set_facecolor("#0d0d0d")
            ax.tick_params(colors="white", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333333")
            ax.set_title(title, fontsize=13, color="white", pad=6, fontweight="bold")
            axes4.append(ax)
    
            cax = fig4.add_subplot(cbar_spec)
            axes4.append(cax)  # store for reference
    
            sm = plt.cm.ScalarMappable(cmap=cmap_p, norm=norm_p)
            sm.set_array([])
            cb = fig4.colorbar(sm, cax=cax)
            cb.set_label(cbar_lbl, fontsize=9, color="white")
            cb.ax.tick_params(colors="white", labelsize=7)
            cbars4.append(cb)
    
        # Unpack only the image axes (indices 0, 2, 4, 6)
        ax_tc, _, ax_pc, _, ax_vt, _, ax_gt, _ = [fig4.axes[k] for k in range(8)]
    
        def _clip_cool(arr, frame):
            return np.clip(arr[frame], cool4_vmin, None)
    
        # Initial imshow frames
        im_tc4 = ax_tc.imshow(
            _clip_cool(true_cool, 0), origin="lower", cmap=cmap_cool4, norm=norm_cool4
        )
        im_pc4 = ax_pc.imshow(
            _clip_cool(cnn_cool, 0), origin="lower", cmap=cmap_cool4, norm=norm_cool4
        )
        im_vt4 = ax_vt.imshow(
            cnn_vort_maps[0], origin="lower", cmap=cmap_vort, norm=norm_vort
        )
        im_gt4 = ax_gt.imshow(
            cnn_gate_maps[0], origin="lower", cmap=cmap_gate, norm=norm_gate
        )
    
        for ax in [ax_tc, ax_pc, ax_vt, ax_gt]:
            ax.set_xlabel("Y (cells)", fontsize=9, color="white")
            ax.set_ylabel("X (cells)", fontsize=9, color="white")
    
        title4 = fig4.suptitle(
            "True Cooling | Predicted Cooling | Vorticity | Gate   —   t = 0",
            fontsize=15,
            color="white",
            y=0.97,
            fontweight="bold",
        )
    
        def init_fourway():
            im_tc4.set_data(_clip_cool(true_cool, 0))
            im_pc4.set_data(_clip_cool(cnn_cool, 0))
            im_vt4.set_data(cnn_vort_maps[0])
            im_gt4.set_data(cnn_gate_maps[0])
            return [im_tc4, im_pc4, im_vt4, im_gt4]
    
        def update_fourway(frame):
            im_tc4.set_data(_clip_cool(true_cool, frame))
            im_pc4.set_data(_clip_cool(cnn_cool, frame))
            im_vt4.set_data(cnn_vort_maps[frame])
            im_gt4.set_data(cnn_gate_maps[frame])
            title4.set_text(
                f"True Cooling | Predicted Cooling | Vorticity | Gate   —   t = {frame}"
            )
            if frame == 0:
                fig4.savefig(snapshot_fourway_path, dpi=300, facecolor=fig4.get_facecolor())
                print(f"Saved four-way snapshot → {snapshot_fourway_path}")
            return [im_tc4, im_pc4, im_vt4, im_gt4]
    
        anim4 = animation.FuncAnimation(
            fig4, update_fourway, frames=nt, init_func=init_fourway, blit=False
        )
    
        print("Saving four-way comparison MP4...")
        with tqdm(total=nt, desc="Saving four-way MP4") as pbar:
            anim4.save(
                mp4_path_fourway,
                writer="ffmpeg",
                fps=10,
                progress_callback=lambda i, n: pbar.update(1),
            )
    
        print(f"Saved four-way comparison animation → {mp4_path_fourway}")
        plt.close(fig4)
    
    
    # ============================================================
    # DENSITY (FINE vs COARSE) & GATE COMPARISON ANIMATION
    # Panels: Fine Density (log) | Coarse Density (log) | Gate Output
    # ============================================================
    if RUN_DENSITY_GATE_ANIMATION:
        print("Creating density and gate comparison animation in parallel...")
    
        mp4_path_density = os.path.join(PDF_MOCKS_DIR, "pdf_density_gate_animation.mp4")
        snapshot_density_path = os.path.join(PDF_MOCKS_DIR, "pdf_density_gate_t0.png")
    
        # Determine consistent colorbar limits for the density plots using the coarse data
        pos_rho = cg_rho[cg_rho > 0]
        if len(pos_rho) > 0:
            rho_vmin = np.log10(max(np.percentile(pos_rho, 1), 1e-10))
            rho_vmax = np.log10(np.percentile(pos_rho, 99))
        else:
            rho_vmin, rho_vmax = -5, 5
    
        generate_parallel_animation(
            worker_density_gate,
            nt,
            mp4_path_density,
            fps=24,
            num_workers=16,
            extra_args=(
                rho_vmin,
                rho_vmax,
                sim_data.rho,
                cg_rho,
                cnn_gate_maps,
                snapshot_density_path,
            )
        )
        print(f"Saved density/gate animation → {mp4_path_density}")
