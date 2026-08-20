#!/usr/bin/env bash
# =============================================================================
# run_gpu_parameter_sweep_2048x1024_2xlength.sh
# Location: shell_scripts/run_gpu_parameter_sweep_2048x1024_2xlength.sh
#
# Runs a parameter sweep of Kelvin-Helmholtz simulations on GPU at:
#   Grid resolution : nx1 = 1024, nx2 = 2048 (32 meshblocks of 256x256)
#   Physical domain : X1 in [-10.0, 10.0] (Lx = 20 pc, 2x baseline)
#                     X2 in [-20.0, 20.0] (Ly = 40 pc, 2x baseline)
#   Scale parameters: a_char = 0.25 (2x baseline)
#                     sigma  = 1.0  (2x baseline)
#   Accelerator     : NVIDIA GPU via CUDA (Kokkos Ada89 backend)
#
# Parameter Sweep Grid:
#   - cold_frac       : (0.33 0.67)
#   - shear_velocity  : (31 40)
#
# Outputs:
#   <PROJECT_ROOT>/simulation_outputs/hr_gpu_sweep_1024x2048_2xlength/
#       ├── vshear_31_coldfrac_0.33/
#       ├── vshear_31_coldfrac_0.67/
#       ├── vshear_40_coldfrac_0.33/
#       └── vshear_40_coldfrac_0.67/
# =============================================================================

set -euo pipefail

# =============================================================================
# PARAMETERS (TINKER AT TOP OF SCRIPT)
# =============================================================================

# Parameter sweep values
SHEAR_VELOCITIES=(31 40)
COLD_FRACS=(0.33 0.67)

# Simulation time limit in Myr (default: 10.0)
TLIM="${TLIM:-10.00}"

# GPU device selection
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# =============================================================================
# CUDA Environment
# =============================================================================
for cuda_dir in /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4 /usr/local/cuda-12 /usr/local/cuda; do
    if [ -d "$cuda_dir/bin" ]; then
        export CUDA_ROOT="$cuda_dir"
        export CUDA_HOME="$cuda_dir"
        export PATH="$cuda_dir/bin:$PATH"
        export LD_LIBRARY_PATH="$cuda_dir/lib64:${LD_LIBRARY_PATH:-}"
        break
    fi
done

# =============================================================================
# Paths and Grid Configuration
# =============================================================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_SRC="${PROJECT_ROOT}/builds/hr_build_gpu/src"
ATHENA="${BUILD_SRC}/athena"
REF_ATHINPUT="${PROJECT_ROOT}/builds/hr_build/src/kh_radiative_128.athinput"

BASE_OUT_DIR="${PROJECT_ROOT}/simulation_outputs/hr_gpu_sweep_1024x2048_2xlength"

# Grid resolution & decomposition (1024 x 2048)
NX1=1024
NX2=2048
MB_NX1=$((NX1 / 4))  # 256
MB_NX2=$((NX2 / 8))  # 256 (32 meshblocks total)

# Physical domain (2x baseline length scales)
X1MIN="-10.0"
X1MAX="10.0"
X2MIN="-20.0"
X2MAX="20.0"
A_CHAR="0.25"
SIGMA="1.0"

# =============================================================================
# Sanity checks
# =============================================================================
if [[ ! -x "${ATHENA}" ]]; then
    echo "ERROR: Athena GPU binary not found or not executable: ${ATHENA}" >&2
    exit 1
fi

if [[ ! -f "${REF_ATHINPUT}" ]]; then
    echo "ERROR: Reference athinput not found: ${REF_ATHINPUT}" >&2
    exit 1
fi

mkdir -p "${BASE_OUT_DIR}"

# =============================================================================
# Helper: Generate patched athinput for a specific parameter pair
# =============================================================================
make_athinput() {
    local target_file="$1"
    local job_name="$2"
    local vx_hot="$3"
    local vx_cold="$4"
    local cold_frac="$5"

    python3 - <<PY "${REF_ATHINPUT}" "${target_file}" "${NX1}" "${NX2}" "${MB_NX1}" "${MB_NX2}" "${X1MIN}" "${X1MAX}" "${X2MIN}" "${X2MAX}" "${A_CHAR}" "${SIGMA}" "${TLIM}" "${job_name}" "${vx_hot}" "${vx_cold}" "${cold_frac}"
import sys, re, pathlib

(
    src_path, out_path_str,
    nx1, nx2, mb_nx1, mb_nx2,
    x1min, x1max, x2min, x2max,
    a_char, sigma, tlim,
    job_name, vx_hot, vx_cold, cold_frac
) = sys.argv[1:18]

src = pathlib.Path(src_path).read_text()
out_path = pathlib.Path(out_path_str)

in_job = in_mesh = in_mb = in_prob = in_time = False
lines_out = []

for line in src.splitlines():
    stripped = line.strip()
    if stripped == '<job>':
        in_job = True; in_mesh = in_mb = in_prob = in_time = False
    elif stripped == '<mesh>':
        in_mesh = True; in_job = in_mb = in_prob = in_time = False
    elif stripped == '<meshblock>':
        in_mb = True; in_job = in_mesh = in_prob = in_time = False
    elif stripped == '<problem>':
        in_prob = True; in_job = in_mesh = in_mb = in_time = False
    elif stripped == '<time>':
        in_time = True; in_job = in_mesh = in_mb = in_prob = False
    elif stripped.startswith('<') and stripped.endswith('>'):
        in_job = in_mesh = in_mb = in_prob = in_time = False

    # Job section
    if in_job and re.match(r'\s*basename\s*=', line):
        line = re.sub(r'(\s*basename\s*=\s*)\S+', r'\g<1>' + job_name, line)

    # Mesh section updates
    elif in_mesh:
        if re.match(r'\s*nx1\s*=', line):
            line = re.sub(r'(\s*nx1\s*=\s*)\d+', r'\g<1>' + nx1, line)
        elif re.match(r'\s*nx2\s*=', line):
            line = re.sub(r'(\s*nx2\s*=\s*)\d+', r'\g<1>' + nx2, line)
        elif re.match(r'\s*x1min\s*=', line):
            line = re.sub(r'(\s*x1min\s*=\s*)[-\d.]+', r'\g<1>' + x1min, line)
        elif re.match(r'\s*x1max\s*=', line):
            line = re.sub(r'(\s*x1max\s*=\s*)[-\d.]+', r'\g<1>' + x1max, line)
        elif re.match(r'\s*x2min\s*=', line):
            line = re.sub(r'(\s*x2min\s*=\s*)[-\d.]+', r'\g<1>' + x2min, line)
        elif re.match(r'\s*x2max\s*=', line):
            line = re.sub(r'(\s*x2max\s*=\s*)[-\d.]+', r'\g<1>' + x2max, line)

    # MeshBlock section updates
    elif in_mb:
        if re.match(r'\s*nx1\s*=', line):
            line = re.sub(r'(\s*nx1\s*=\s*)\d+', r'\g<1>' + mb_nx1, line)
        elif re.match(r'\s*nx2\s*=', line):
            line = re.sub(r'(\s*nx2\s*=\s*)\d+', r'\g<1>' + mb_nx2, line)

    # Time section updates
    elif in_time:
        if re.match(r'\s*tlim\s*=', line):
            line = re.sub(r'(\s*tlim\s*=\s*)[-\d.]+', r'\g<1>' + tlim, line)

    # Problem section updates
    elif in_prob:
        if re.match(r'\s*a_char\s*=', line):
            line = re.sub(r'(\s*a_char\s*=\s*)[-\d.]+', r'\g<1>' + a_char, line)
        elif re.match(r'\s*sigma\s*=', line):
            line = re.sub(r'(\s*sigma\s*=\s*)[-\d.]+', r'\g<1>' + sigma, line)
        elif re.match(r'\s*vx_hot\s*=', line):
            line = re.sub(r'(\s*vx_hot\s*=\s*)[-\d.]+', r'\g<1>' + vx_hot, line)
        elif re.match(r'\s*vx_cold\s*=', line):
            line = re.sub(r'(\s*vx_cold\s*=\s*)[-\d.]+', r'\g<1>' + vx_cold, line)
        elif re.match(r'\s*cold_frac\s*=', line):
            line = re.sub(r'(\s*cold_frac\s*=\s*)[-\d.]+', r'\g<1>' + cold_frac, line)

    lines_out.append(line)

out_path.write_text('\n'.join(lines_out) + '\n')
print(f"Generated input file: {out_path}")
PY
}

# =============================================================================
# Run Sweep Loop
# =============================================================================
TOTAL_SIMS=$((${#SHEAR_VELOCITIES[@]} * ${#COLD_FRACS[@]}))
CURRENT_SIM=0

echo "========================================================================"
echo " Starting GPU Parameter Sweep (2048x1024, 2x Physical Length Scales)"
echo " Total Simulations : ${TOTAL_SIMS}"
echo " Shear Velocities  : ${SHEAR_VELOCITIES[*]}"
echo " Cold Gas Fracs    : ${COLD_FRACS[*]}"
echo " Time Limit        : ${TLIM} Myr"
echo " GPU Device        : ${CUDA_VISIBLE_DEVICES}"
echo " Base Output Dir   : ${BASE_OUT_DIR}"
echo "========================================================================"
echo ""

for v_shear in "${SHEAR_VELOCITIES[@]}"; do
    for cold_frac in "${COLD_FRACS[@]}"; do
        CURRENT_SIM=$((CURRENT_SIM + 1))

        # Calculate vx_hot and vx_cold in zero-momentum frame
        # v_shear = vx_hot - vx_cold; vx_cold = -0.1 * vx_hot => v_shear = 1.1 * vx_hot
        vx_hot=$(python3 -c "print(f'{$v_shear / 1.1:.7f}')")
        vx_cold=$(python3 -c "print(f'{-0.1 * ($v_shear / 1.1):.7f}')")

        pair_name="vshear_${v_shear}_coldfrac_${cold_frac}"
        sim_dir="${BASE_OUT_DIR}/${pair_name}"
        athinput="${sim_dir}/kh_radiative_${pair_name}.athinput"
        log_file="${sim_dir}/${pair_name}.log"

        mkdir -p "${sim_dir}"

        echo "------------------------------------------------------------------------"
        echo "[${CURRENT_SIM}/${TOTAL_SIMS}] Starting Simulation: ${pair_name}"
        echo "  Shear Velocity : ${v_shear} (vx_hot=${vx_hot}, vx_cold=${vx_cold})"
        echo "  Cold Fraction  : ${cold_frac}"
        echo "  Output Dir     : ${sim_dir}"
        echo "  Log File       : ${log_file}"
        echo "------------------------------------------------------------------------"

        # Generate athinput
        make_athinput "${athinput}" "${pair_name}" "${vx_hot}" "${vx_cold}" "${cold_frac}"

        # Launch simulation on GPU
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching Athena on GPU ${CUDA_VISIBLE_DEVICES} ..."
        
        cd "${BUILD_SRC}"

        "${ATHENA}" \
            -i "${athinput}" \
            -d "${sim_dir}" \
            2>&1 | tee "${log_file}"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Simulation finished: ${pair_name}"

        # ---------------------------------------------------------------------
        # Render visualization animation into the same output folder
        # ---------------------------------------------------------------------
        mp4_save_path="${sim_dir}/${pair_name}.mp4"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Rendering animation to: ${mp4_save_path} ..."

        if [[ -f "${PROJECT_ROOT}/venv/bin/python" ]]; then
            PYTHON_EXEC="${PROJECT_ROOT}/venv/bin/python"
        else
            PYTHON_EXEC="python3"
        fi

        "${PYTHON_EXEC}" - <<PY || echo "WARNING: Visualization generation failed for ${pair_name}, continuing..."
import sys, os
sys.path.append('${PROJECT_ROOT}')
import ergane

athinp = '${athinput}'
datafolder = '${sim_dir}'
save_path = '${mp4_save_path}'

print(f"  [ergane] Loading simulation data from {datafolder} ...")
sim_data = ergane.SimulationData(athinp=athinp, datafolder=datafolder)

print(f"  [ergane] Rendering animation ...")
mpl_viz = sim_data.visualize(backend="matplotlib", interval=80)

print(f"  [ergane] Saving to {save_path} ...")
mpl_viz.ani.save(save_path, writer="ffmpeg", fps=60, dpi=150)
print(f"  [ergane] Successfully saved: {save_path}")
PY

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished Simulation [${CURRENT_SIM}/${TOTAL_SIMS}]: ${pair_name}"
        echo ""
    done
done

echo "========================================================================"
echo " All ${TOTAL_SIMS} GPU simulations and visualizations completed!"
echo " Outputs and MP4 animations stored in: ${BASE_OUT_DIR}"
echo "========================================================================"
