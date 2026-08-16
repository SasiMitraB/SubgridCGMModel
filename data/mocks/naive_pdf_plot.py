# Python script to plot the actual and predicted PDFs using a discrete form for the PDFs (n bins in log temp space)

import os
import sys
from multiprocessing.shared_memory import SharedMemory

import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from models.conv_nn.naive_cnn import snapshot_pred
from models.conv_nn.pdf_cnn import lambda_cool, compute_cooling_rate

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
RUN_DENSITY_GATE_ANIMATION = False
RUN_GATE_ENTROPY_DIAGNOSTIC = False

# =========================
# SETTINGS
# =========================
DEFAULT_FINE_RESOLUTION = (1024, 512)
DEFAULT_DOWNSAMPLE = 64


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

CACHE_PATH = os.environ.get(
    "SUBGRID_CACHE_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../simulation_outputs/hr_build_512/cache")),
)
folder_path = os.path.join(CACHE_PATH, f"sc{resolution}_{downsample}")

PDF_MOCKS_DIR = os.environ.get("PDF_MOCKS_DIR", "mocks/pdf_naive")
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


def get_positive_percentiles(arr1, arr2, p_lo=1, p_hi=99):
    """Compute percentile range across positive values of two arrays without full concatenation."""
    pos1 = arr1[arr1 > 0]
    pos2 = arr2[arr2 > 0]
    if len(pos1) == 0 and len(pos2) == 0:
        return 1e-30, 1e-20   # physical CGS fallback (erg/cm³/s)
    if len(pos1) == 0:
        return max(np.percentile(pos2, p_lo), 1e-40), np.percentile(pos2, p_hi)
    if len(pos2) == 0:
        return max(np.percentile(pos1, p_lo), 1e-40), np.percentile(pos1, p_hi)

    p_lo_val = min(np.percentile(pos1, p_lo), np.percentile(pos2, p_lo))
    p_hi_val = max(np.percentile(pos1, p_hi), np.percentile(pos2, p_hi))
    return max(p_lo_val, 1e-40), p_hi_val


def _worker_wrapper(worker_func, frames_list, temp_dir, nt, *args):
    unpacked_args = []
    shm_to_close = []
    for arg in args:
        if isinstance(arg, tuple) and len(arg) == 4 and arg[0] == "_SHM_":
            _, shm_name, shape, dtype_str = arg
            shm = SharedMemory(name=shm_name)
            arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
            shm_to_close.append(shm)
            unpacked_args.append(arr)
        else:
            unpacked_args.append(arg)
    try:
        return worker_func(frames_list, temp_dir, nt, *unpacked_args)
    finally:
        for shm in shm_to_close:
            shm.close()


def get_best_video_codec():
    """Detect available video codec for ffmpeg."""
    try:
        res = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if "h264_nvenc" in res.stdout:
            return "h264_nvenc"
        if "libx264" in res.stdout:
            return "libx264"
    except Exception:
        pass
    return "mpeg4"


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

    # Wrap large numpy arrays (> 1 MB) into SharedMemory to avoid 16x RAM duplication across workers
    shm_list = []
    processed_extra_args = []
    for arg in extra_args:
        if isinstance(arg, np.ndarray) and arg.nbytes > 1024 * 1024:
            shm = SharedMemory(create=True, size=arg.nbytes)
            shared_arr = np.ndarray(arg.shape, dtype=arg.dtype, buffer=shm.buf)
            shared_arr[:] = arg[:]
            shm_list.append(shm)
            processed_extra_args.append(("_SHM_", shm.name, arg.shape, str(arg.dtype)))
        else:
            processed_extra_args.append(arg)

    print(f"Rendering {nt} frames in parallel using {len(chunks)} workers...")
    
    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for fl in chunks:
                futures.append(
                    executor.submit(_worker_wrapper, worker_func, fl, temp_dir, nt, *processed_extra_args)
                )
            
            # Track progress using progress files
            completed_frames = set()
            stuck_counter = 0
            with tqdm(total=nt, desc=f"Rendering {os.path.basename(output_path)}") as pbar:
                workers_done = False
                while len(completed_frames) < nt:
                    if not workers_done:
                        workers_done = all(fut.done() for fut in futures)
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
                        stuck_counter += 1
                        time.sleep(0.05)
                        if stuck_counter > 100:  # 5 second timeout if all workers finished but markers missing
                            print(f"Warning: workers completed but missing {nt - len(completed_frames)} frame progress markers.")
                            break
                    elif not workers_done:
                        time.sleep(0.1)
                
        print(f"Stitching frames together into {output_path} using ffmpeg...")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        codec = get_best_video_codec()
        
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(temp_dir, "frame_%04d.png"),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", codec,
            "-pix_fmt", "yuv420p",
            output_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res.returncode != 0:
            cmd_fb = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(temp_dir, "frame_%04d.png"),
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "mpeg4",
                "-pix_fmt", "yuv420p",
                output_path
            ]
            res_fb = subprocess.run(cmd_fb, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if res_fb.returncode != 0:
                raise RuntimeError(f"ffmpeg failed with exit code {res.returncode}:\n{res.stderr.decode()}")
    finally:
        for shm in shm_list:
            shm.close()
            shm.unlink()
        shutil.rmtree(temp_dir)


def worker_pdf_compare(frames_list, temp_dir, nt, nx, ny, nb, temp_pdf, conv_temp_pdf, true_iso_cool, cnn_cool, log_temp_centers, active_bin_start, active_bin_end, snapshot_compare_path):
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

                tic = true_iso_cool[frame, ii, j]
                pc = cnn_cool[frame, ii, j]

                true_texts[i, j].set_text(f"{tic:.1e}")
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


def render_density_gate_sequential(nt, temp_dir, rho_vmin, rho_vmax, sim_data_rho, cg_rho, cnn_gate_maps, snapshot_density_path):
    """Render every density-gate frame sequentially in a single process.

    Using parallel workers here causes each worker to receive a full copy of
    sim_data_rho, cg_rho, and cnn_gate_maps (all large arrays), which balloons
    RAM and freezes Python.  Sequential rendering reuses a single figure and
    never duplicates the arrays.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm

    fig5, axes5 = plt.subplots(1, 3, figsize=(12, 7))
    ax_fine, ax_cg, ax_gate = axes5

    im_fine = ax_fine.imshow(
        np.log10(sim_data_rho[0] + 1e-10),
        origin="lower", cmap="viridis", vmin=rho_vmin, vmax=rho_vmax
    )
    ax_fine.set_title(r"True Density (Before CG) [$\log_{10}$]", fontsize=14)
    fig5.colorbar(im_fine, ax=ax_fine, fraction=0.046, pad=0.04)

    im_cg = ax_cg.imshow(
        np.log10(cg_rho[0] + 1e-10),
        origin="lower", cmap="viridis", vmin=rho_vmin, vmax=rho_vmax
    )
    ax_cg.set_title(r"Coarse-Grained Density [$\log_{10}$]", fontsize=14)
    fig5.colorbar(im_cg, ax=ax_cg, fraction=0.046, pad=0.04)

    im_gate = ax_gate.imshow(
        cnn_gate_maps[0],
        origin="lower", cmap="inferno", vmin=0.0, vmax=1.0
    )
    ax_gate.set_title(r"CNN Gate Output $g(x,y)$", fontsize=14)
    fig5.colorbar(im_gate, ax=ax_gate, fraction=0.046, pad=0.04)

    for ax in axes5:
        ax.set_xlabel("Y (cells)")
        ax.set_ylabel("X (cells)")

    title5 = fig5.suptitle("Density & Gate Comparison | t = 0", fontsize=18, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], w_pad=1.0)

    for frame in tqdm(range(nt), desc="Rendering density gate frames"):
        im_fine.set_data(np.log10(sim_data_rho[frame] + 1e-10))
        im_cg.set_data(np.log10(cg_rho[frame] + 1e-10))
        im_gate.set_data(cnn_gate_maps[frame])
        title5.set_text(f"Density & Gate Comparison | t = {frame}")

        if frame == 0:
            fig5.savefig(snapshot_density_path, dpi=300)

        fig5.savefig(os.path.join(temp_dir, f"frame_{frame:04d}.png"), dpi=100)

    plt.close(fig5)


def batch_predict_with_gate(sim_data, downsample, resolution, device):
    from models.conv_nn.naive_cnn import NaiveConvNN, MODEL_SAVE_DIR, in_channels, layer_size1, layer_size2, layer_size3, layer_size4, out_channels, kernel_size
    import torch
    import numpy as np
    from tqdm import tqdm
    import os

    nt = sim_data.rho.shape[0]
    shape = (resolution[0] // downsample, resolution[1] // downsample)
    nx, ny = shape
    
    # 1. Load model once
    model_path = os.path.join(MODEL_SAVE_DIR, f"cnn_{resolution}_{downsample}.pth")
    state_dict = torch.load(model_path, map_location=device)
    ckpt_ksize = kernel_size
    if "encoder.0.weight" in state_dict:
        ckpt_ksize = state_dict["encoder.0.weight"].shape[-1]

    cnn_model = NaiveConvNN(
        in_channels,
        layer_size1,
        layer_size2,
        layer_size3,
        layer_size4,
        out_channels,
        ckpt_ksize,
    ).to(device)
    cnn_model.load_state_dict(state_dict)
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
    cnn_gate_maps = np.ones((nt, nx, ny))
    cnn_vort_maps = np.zeros((nt, nx, ny))
    
    print("Running batch predictions (Naive CNN)...")
    with torch.no_grad():
        for i in tqdm(range(nt), desc="Predicting Naive CNN temperature PDFs"):
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
            pdf_tensor = cnn_model.predict_pdf(input_tensor)
            conv_temp_pdf[i] = pdf_tensor[0].cpu().numpy()
            
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
    
        plt.close(fig)
    
    
    # =========================
    # TRUE vs PRED PDF COMPARISON
    # =========================
    # =====================================================================
    # COOLING COMPUTATION BLOCK
    # Outputs are converted to physical CGS units (erg / cm³ / s)
    # =====================================================================
    print("Computing cooling rates (True PDF, CNN PDF)...")
    
    # Shared constants from athinput
    kb   = 1.3807e-16            # Boltzmann constant  [erg / K]
    m_H  = 1.67262e-24           # Hydrogen mass        [g]
    mu   = 0.62                  # Mean molecular weight
    unit_fix = 1.975e27          # Code-unit energy-rate factor  (rho_0*L_0)/(m_H^2*v_0^3)
    
    # Code-unit -> physical number-density conversion from kh_radiative.athinput:
    #   L_cgs = 3.08568e18 cm (1 pc), T_cgs = 3.15576e13 s (1 Myr), M_cgs = 4.91417e31 g
    _L_cgs   = 3.08568e18         # 1 pc in cm
    _T_cgs   = 3.15576e13         # 1 Myr in s
    _M_cgs   = 4.91417e31         # code mass unit [g]
    _RHO_cgs = _M_cgs / _L_cgs**3 # code density unit [g/cm³]  (~1.67262e-24 g/cm³)
    n_to_cm3 = _RHO_cgs / (mu * m_H)  # cm⁻³ per code density unit (~1.6129 cm⁻³)
    
    # Factor: compute_cooling_rate output (code units) -> erg/cm³/s
    _code_to_cgs = 1.0 / unit_fix
    
    # ---- PDF bin centres (geometric mean of edges) ----
    temp_centers = np.sqrt(temp_bins[:-1] * temp_bins[1:])  # (nb,)
    
    # ---- Active cooling window for shading ----
    logT_active_start = float(os.environ.get("LOGT_ACTIVE_START", "4.2"))
    logT_active_end = float(os.environ.get("LOGT_ACTIVE_END", "6.0"))
    active_bin_start = np.searchsorted(temp_centers, 10**logT_active_start)
    active_bin_end = np.searchsorted(temp_centers, 10**logT_active_end)
    
    # ------------------------------------------------------------------
    # (B) Coarse-grain pressure  (kept for reference; not used in cooling)
    # ------------------------------------------------------------------
    cg_pressure = np.zeros((nt, nx, ny))
    for t in tqdm(range(nt), desc="Coarse-graining pressure"):
        cg_pressure[t] = sim_data.coarse_grain(sim_data.pressure[t])
    
    # ------------------------------------------------------------------
    # (C) Cool_True_PDF : use TRUE PDF, n = rho_cg / mu  (no isobaric assumption)
    #     n² × Σᵢ PDF(Tᵢ) Λ(Tᵢ)
    # ------------------------------------------------------------------
    print("  (C) True PDF cooling (using simulation PDF)...")
    true_iso_cool = np.zeros((nt, nx, ny))
    for t in tqdm(range(nt), desc="True PDF cooling"):
        true_iso_cool[t] = compute_cooling_rate(
            temp_pdf[t],   # (nb, nx, ny)  – true PDF
            temp_centers,  # (nb,)
            is_pdf=True,
            rho_cg=cg_rho[t],   # (nx, ny)  – coarse-grained code density
        )
    
    # ------------------------------------------------------------------
    # (D) Cool_CNN_PDF : use CNN PDF, n = rho_cg / mu  (no isobaric assumption)
    # ------------------------------------------------------------------
    print("  (D) CNN PDF cooling (using CNN PDF)...")
    cnn_cool = np.zeros((nt, nx, ny))
    for t in tqdm(range(nt), desc="CNN PDF cooling"):
        cnn_cool[t] = compute_cooling_rate(
            conv_temp_pdf[t],  # (nb, nx, ny)  – CNN PDF
            temp_centers,      # (nb,)
            is_pdf=True,
            rho_cg=cg_rho[t],  # (nx, ny)  – coarse-grained code density
        )
    
    print("Cooling computation done.")
    
    # ------------------------------------------------------------------
    # Convert both cooling fields from code units -> physical erg/cm³/s
    # ------------------------------------------------------------------
    print("  Converting to physical units (erg / cm³ / s)...")
    true_iso_cool *= _code_to_cgs
    cnn_cool      *= _code_to_cgs
    print("  Conversion done.")

    # Free memory for fine-grid simulation arrays that are no longer needed
    del sim_data.temp, sim_data.pressure, sim_data.ux, sim_data.uy, sim_data.eint, sim_data.ps
    
    # =========================
    # METRICS
    # =========================
    print("\n=== Quantitative Benchmarking Metrics ===")
    print_metrics(
        true_iso_cool.flatten(),
        cnn_cool.flatten(),
        "CNN Prediction Error   (True PDF vs CNN PDF)",
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
        plt.close(fig)
    

    
        # ============================================================
        # GLOBAL COOLING RATE DIAGNOSTICS: 2D Density & Residuals
        # ============================================================
        print("Creating global cooling diagnostic plots...")
    
        from matplotlib.colors import LogNorm
    
        SCATTER_MIN = 1e-40  # Exclude masked-zero cells; physical CGS values are ~1e-27 to 1e-17
    
        # Flatten the two cooling fields (raw values; no eps clipping)
        temp_flat = cg_temp.flatten()
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
    
        fig_sc, axes_sc = plt.subplots(1, 2, figsize=(14, 6))
        ax_pred = axes_sc[0]
        ax_resid = axes_sc[1]
    
        # Panel 1: CNN Prediction Error (True PDF vs CNN PDF)
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
        ax_pred.set_xlabel("True PDF Cooling (erg / cm$^3$ / s)")
        ax_pred.set_ylabel("CNN PDF Cooling (erg / cm$^3$ / s)")
        ax_pred.set_title(f"CNN Prediction Error\n({mask2.sum():,} / {len(flat_true_iso):,} points)")
        plt.colorbar(hb2, ax=ax_pred, label="log$_{10}$(count)")
        add_running_median(ax_pred, xm2, ym2)
    
        # Panel 2: CNN Residuals (log10(CNN / True) vs True PDF)
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
        ax_resid.set_xlabel("True PDF Cooling (erg / cm$^3$ / s)")
        ax_resid.set_ylabel(r"$\log_{10}$(CNN / True PDF)")
        ax_resid.set_title(f"CNN Residuals\n({mask4.sum():,} / {len(flat_true_iso):,} points)")
        plt.colorbar(hb4, ax=ax_resid, label="log$_{10}$(count)")
        add_running_median(ax_resid, xm4, ym4)
    
        fig_sc.suptitle("Cooling Rate Comparisons & Diagnostics (All Pixels, All Timesteps)", fontsize=16)
        fig_sc.tight_layout()
        fig_sc.savefig(
            os.path.join(PDF_MOCKS_DIR, "pdf_cooling_scatter_twoway.png"), dpi=200
        )
        plt.show()
        plt.close(fig_sc)
        print("Saved cooling diagnostic plot.")
    
    
    # ============================================================
    # HISTOGRAM: Zero-Fraction + Positive-Only log10 (Change #5)
    # ============================================================
    if RUN_COOLING_HISTOGRAM:
        print("Creating improved histogram plots...")
    
        _fields = {
            "True PDF": true_iso_cool.flatten(),
            "CNN PDF": cnn_cool.flatten(),
        }
        _colors = ["darkorange", "mediumseagreen"]
    
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
    
        ax_pos.set_xlabel(r"$\log_{10}$(Cooling Rate [erg / cm$^3$ / s])")
        ax_pos.set_ylabel("Probability Density")
        ax_pos.set_title(r"Distribution of Positive $\log_{10}$(Cooling)")
        ax_pos.legend()
        ax_pos.set_yscale("log")
    
        fig_hist.suptitle("Cooling Rate Histograms", fontsize=14)
        fig_hist.tight_layout()
        fig_hist.savefig(os.path.join(PDF_MOCKS_DIR, "pdf_cooling_histogram.png"), dpi=200)
        plt.show()
        plt.close(fig_hist)
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
        cool_vmin, cool_vmax = get_positive_percentiles(true_iso_cool, cnn_cool)
    
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
        cool4_vmin, cool4_vmax = get_positive_percentiles(true_iso_cool, cnn_cool)
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
    
        # ---- Build figure: 3 image panels + 3 colorbars ----
        fig4 = plt.figure(figsize=(18, 6))
        fig4.patch.set_facecolor("#0d0d0d")
    
        # gridspec: 3 image cols + 3 narrow cbar cols
        gs4 = fig4.add_gridspec(
            1,
            6,
            width_ratios=[1, 0.05, 1, 0.05, 1, 0.05],
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
                "Predicted Cooling",
                cmap_cool4,
                norm_cool4,
                "Cooling Rate\n(erg / cm³ / s)",
            ),
            (gs4[2], gs4[3], "Vorticity ω", cmap_vort, norm_vort, "Vorticity (code units)"),
            (gs4[4], gs4[5], "Gate g(x,y)", cmap_gate, norm_gate, "Gate value ∈ (0, 1)"),
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
            axes4.append(cax)
    
            sm = plt.cm.ScalarMappable(cmap=cmap_p, norm=norm_p)
            sm.set_array([])
            cb = fig4.colorbar(sm, cax=cax)
            cb.set_label(cbar_lbl, fontsize=9, color="white")
            cb.ax.tick_params(colors="white", labelsize=7)
            cbars4.append(cb)
    
        # Unpack only the image axes (indices 0, 2, 4)
        ax_pc, _, ax_vt, _, ax_gt, _ = [fig4.axes[k] for k in range(6)]
    
        def _clip_cool(arr, frame):
            return np.clip(arr[frame], cool4_vmin, None)
    
        # Initial imshow frames
        im_pc4 = ax_pc.imshow(
            _clip_cool(cnn_cool, 0), origin="lower", cmap=cmap_cool4, norm=norm_cool4
        )
        im_vt4 = ax_vt.imshow(
            cnn_vort_maps[0], origin="lower", cmap=cmap_vort, norm=norm_vort
        )
        im_gt4 = ax_gt.imshow(
            cnn_gate_maps[0], origin="lower", cmap=cmap_gate, norm=norm_gate
        )
    
        for ax in [ax_pc, ax_vt, ax_gt]:
            ax.set_xlabel("Y (cells)", fontsize=9, color="white")
            ax.set_ylabel("X (cells)", fontsize=9, color="white")
    
        title4 = fig4.suptitle(
            "Predicted Cooling | Vorticity | Gate   —   t = 0",
            fontsize=15,
            color="white",
            y=0.97,
            fontweight="bold",
        )
    
        def init_fourway():
            im_pc4.set_data(_clip_cool(cnn_cool, 0))
            im_vt4.set_data(cnn_vort_maps[0])
            im_gt4.set_data(cnn_gate_maps[0])
            return [im_pc4, im_vt4, im_gt4]
    
        def update_fourway(frame):
            im_pc4.set_data(_clip_cool(cnn_cool, frame))
            im_vt4.set_data(cnn_vort_maps[frame])
            im_gt4.set_data(cnn_gate_maps[frame])
            title4.set_text(
                f"Predicted Cooling | Vorticity | Gate   —   t = {frame}"
            )
            if frame == 0:
                fig4.savefig(snapshot_fourway_path, dpi=300, facecolor=fig4.get_facecolor())
                print(f"Saved three-panel snapshot → {snapshot_fourway_path}")
            return [im_pc4, im_vt4, im_gt4]
    
        anim4 = animation.FuncAnimation(
            fig4, update_fourway, frames=nt, init_func=init_fourway, blit=False
        )
    
        print("Saving three-panel comparison MP4...")
        with tqdm(total=nt, desc="Saving three-panel MP4") as pbar:
            anim4.save(
                mp4_path_fourway,
                writer="ffmpeg",
                fps=10,
                progress_callback=lambda i, n: pbar.update(1),
            )
    
        print(f"Saved three-panel comparison animation → {mp4_path_fourway}")
        plt.close(fig4)
    
    
    # ============================================================
    # DENSITY (FINE vs COARSE) & GATE COMPARISON ANIMATION
    # Panels: Fine Density (log) | Coarse Density (log) | Gate Output
    # ============================================================
    if RUN_DENSITY_GATE_ANIMATION:
        import tempfile, shutil, subprocess
        print("Creating density and gate comparison animation (sequential)...")

        mp4_path_density = os.path.join(PDF_MOCKS_DIR, "pdf_density_gate_animation.mp4")
        snapshot_density_path = os.path.join(PDF_MOCKS_DIR, "pdf_density_gate_t0.png")

        # Determine consistent colorbar limits from the coarse data
        pos_rho = cg_rho[cg_rho > 0]
        if len(pos_rho) > 0:
            rho_vmin = np.log10(max(np.percentile(pos_rho, 1), 1e-10))
            rho_vmax = np.log10(np.percentile(pos_rho, 99))
        else:
            rho_vmin, rho_vmax = -5, 5

        _temp_dir = tempfile.mkdtemp()
        try:
            render_density_gate_sequential(
                nt, _temp_dir,
                rho_vmin, rho_vmax,
                sim_data.rho, cg_rho, cnn_gate_maps,
                snapshot_density_path,
            )

            print(f"Stitching density gate frames → {mp4_path_density}")
            cmd = [
                "ffmpeg", "-y",
                "-framerate", "24",
                "-i", os.path.join(_temp_dir, "frame_%04d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                mp4_path_density,
            ]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if res.returncode != 0:
                print(f"ffmpeg error: {res.stderr.decode()}")
            else:
                print(f"Saved density/gate animation → {mp4_path_density}")
        finally:
            shutil.rmtree(_temp_dir)


# ============================================================
# GATE / ENTROPY vs RESIDUAL DIAGNOSTIC
# Tests the hypothesis that CNN overprediction at low cooling
# is driven by miscalibration of the mixing-layer gate against
# the true PDF's Shannon entropy.
# ============================================================

if RUN_GATE_ENTROPY_DIAGNOSTIC:
    print("Creating gate/entropy vs residual diagnostic plots...")

    def add_running_median_linear(ax, xv, yv, n_bins=25, x_range=None):
        """Same as add_running_median but for a LINEAR x-axis in [0, 1]
        (gate values, normalized entropy) instead of log-spaced x."""
        if x_range is None:
            x_range = (xv.min(), xv.max())
        bin_edges = np.linspace(x_range[0], x_range[1], n_bins + 1)
        bin_idx = np.digitize(xv, bin_edges)
        xs, meds, lo, hi = [], [], [], []
        for b in range(1, n_bins + 1):
            sel = bin_idx == b
            if sel.sum() < 5:
                continue
            xs.append(0.5 * (bin_edges[b - 1] + bin_edges[b]))
            yb = yv[sel]
            meds.append(np.median(yb))
            lo.append(np.percentile(yb, 16))
            hi.append(np.percentile(yb, 84))
        if len(xs) > 0:
            ax.plot(xs, meds, color="cyan", lw=2, label="Median")
            ax.fill_between(xs, lo, hi, color="cyan", alpha=0.2, label="16-84%")
            ax.legend(fontsize=8, loc="upper left")

    # ---- 1. Compute normalized entropy of the TRUE pdf, per pixel/timestep ----
    # H = -sum_i p_i log(p_i), normalized to [0, 1] by log(n_bins)
    eps_ent = 1e-12
    true_entropy = -np.sum(
        temp_pdf * np.log(temp_pdf + eps_ent), axis=1
    )  # (nt, nx, ny)
    true_entropy_norm = true_entropy / np.log(nb)  # (nt, nx, ny)

    # ---- 2. Also compute predicted-PDF entropy, for comparison ----
    pred_entropy = -np.sum(
        conv_temp_pdf * np.log(conv_temp_pdf + eps_ent), axis=1
    )  # (nt, nx, ny)
    pred_entropy_norm = pred_entropy / np.log(nb)

    # ---- 3. Residual: log10(CNN / True Isobaric), masked to positive pairs ----
    SCATTER_MIN_GE = 1e-40  # physical CGS units: ~1e-27 to 1e-17 erg/cm³/s
    mask_ge = (true_iso_cool >= SCATTER_MIN_GE) & (cnn_cool >= SCATTER_MIN_GE)

    resid_flat = np.log10(cnn_cool[mask_ge] / true_iso_cool[mask_ge])
    gate_flat = cnn_gate_maps[mask_ge]
    true_ent_flat = true_entropy_norm[mask_ge]
    pred_ent_flat = pred_entropy_norm[mask_ge]

    print(f"  Gate/entropy diagnostic using {mask_ge.sum():,} / {mask_ge.size:,} points")

    fig_ge, axes_ge = plt.subplots(2, 2, figsize=(14, 12))
    ax_resid_gate = axes_ge[0, 0]
    ax_resid_ent = axes_ge[0, 1]
    ax_gate_vs_ent = axes_ge[1, 0]
    ax_gate_hist = axes_ge[1, 1]

    # Panel 1: Residual vs Gate value
    hb_rg = ax_resid_gate.hexbin(
        gate_flat, resid_flat,
        gridsize=60,
        bins="log",
        cmap="viridis",
        mincnt=1,
        rasterized=True,
    )
    ax_resid_gate.axhline(0, color="r", linestyle="--", lw=1)
    ax_resid_gate.set_xlabel("Gate value $g(x,y)$")
    ax_resid_gate.set_ylabel(r"$\log_{10}$(CNN Isobaric / True Isobaric)")
    ax_resid_gate.set_title(f"Residual vs Gate\n({mask_ge.sum():,} points)")
    plt.colorbar(hb_rg, ax=ax_resid_gate, label="log$_{10}$(count)")
    add_running_median_linear(ax_resid_gate, gate_flat, resid_flat, x_range=(0, 1))

    # Panel 2: Residual vs True-PDF Entropy
    hb_re = ax_resid_ent.hexbin(
        true_ent_flat, resid_flat,
        gridsize=60,
        bins="log",
        cmap="viridis",
        mincnt=1,
        rasterized=True,
    )
    ax_resid_ent.axhline(0, color="r", linestyle="--", lw=1)
    ax_resid_ent.set_xlabel(r"Normalized true-PDF entropy $H/\log(n_{bins})$")
    ax_resid_ent.set_ylabel(r"$\log_{10}$(CNN Isobaric / True Isobaric)")
    ax_resid_ent.set_title(f"Residual vs True Entropy\n({mask_ge.sum():,} points)")
    plt.colorbar(hb_re, ax=ax_resid_ent, label="log$_{10}$(count)")
    add_running_median_linear(ax_resid_ent, true_ent_flat, resid_flat, x_range=(0, 1))

    # Panel 3: Gate vs True Entropy (checks whether the gate is actually
    # tracking the quantity it was trained to predict — a sharp diagonal
    # here means the gate is well-calibrated; scatter/saturation means it isn't)
    hb_ge = ax_gate_vs_ent.hexbin(
        true_ent_flat, gate_flat,
        gridsize=60,
        bins="log",
        cmap="viridis",
        mincnt=1,
        rasterized=True,
    )
    ax_gate_vs_ent.axvline(
        float(os.environ.get("GATE_ENTROPY_THRESHOLD", "0.05")),
        color="r", linestyle="--", lw=1, label="entropy_threshold",
    )
    ax_gate_vs_ent.set_xlabel(r"Normalized true-PDF entropy $H/\log(n_{bins})$")
    ax_gate_vs_ent.set_ylabel("Gate value $g(x,y)$")
    ax_gate_vs_ent.set_title(f"Gate Calibration Check\n({mask_ge.sum():,} points)")
    ax_gate_vs_ent.legend(fontsize=8)
    plt.colorbar(hb_ge, ax=ax_gate_vs_ent, label="log$_{10}$(count)")
    add_running_median_linear(ax_gate_vs_ent, true_ent_flat, gate_flat, x_range=(0, 1))

    # Panel 4: Gate value distribution, split by whether residual is
    # over- or under-predicting, to see if overprediction pixels cluster
    # at a particular (intermediate) gate value
    over_mask = resid_flat > 0.1     # CNN overpredicts by > ~0.1 dex
    under_mask = resid_flat < -0.1   # CNN underpredicts by > ~0.1 dex
    near_mask = ~over_mask & ~under_mask

    ax_gate_hist.hist(
        gate_flat[over_mask], bins=50, range=(0, 1), density=True,
        histtype="step", linewidth=2, color="crimson", label="Overpredict (>0.1 dex)",
    )
    ax_gate_hist.hist(
        gate_flat[under_mask], bins=50, range=(0, 1), density=True,
        histtype="step", linewidth=2, color="royalblue", label="Underpredict (<-0.1 dex)",
    )
    ax_gate_hist.hist(
        gate_flat[near_mask], bins=50, range=(0, 1), density=True,
        histtype="step", linewidth=2, color="grey", label="Near-accurate", alpha=0.7,
    )
    ax_gate_hist.set_xlabel("Gate value $g(x,y)$")
    ax_gate_hist.set_ylabel("Probability Density")
    ax_gate_hist.set_title("Gate distribution by prediction error bucket")
    ax_gate_hist.legend(fontsize=9)

    fig_ge.suptitle("Gate & Entropy Calibration Diagnostics", fontsize=16)
    fig_ge.tight_layout()
    fig_ge.savefig(
        os.path.join(PDF_MOCKS_DIR, "pdf_gate_entropy_diagnostic.png"), dpi=200
    )
    plt.show()
    plt.close(fig_ge)
    print("Saved gate/entropy diagnostic plot.")

    # ---- Quick numeric summary, printed for convenience ----
    print("\n=== Gate/Entropy Diagnostic Summary ===")
    print(f"  Fraction of points with gate < 0.1  : {(gate_flat < 0.1).mean()*100:.1f}%")
    print(f"  Fraction of points with gate > 0.9  : {(gate_flat > 0.9).mean()*100:.1f}%")
    print(f"  Median residual | gate < 0.1        : {np.median(resid_flat[gate_flat < 0.1]):+.3f} dex")
    print(f"  Median residual | 0.1 <= gate <= 0.9: {np.median(resid_flat[(gate_flat >= 0.1) & (gate_flat <= 0.9)]):+.3f} dex")
    print(f"  Median residual | gate > 0.9         : {np.median(resid_flat[gate_flat > 0.9]):+.3f} dex")
    gate_entropy_corr, _ = pearsonr(gate_flat, true_ent_flat)
    print(f"  Pearson corr(gate, true entropy)    : {gate_entropy_corr:.4f}")