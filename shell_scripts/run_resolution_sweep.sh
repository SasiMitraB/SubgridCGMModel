#!/usr/bin/env bash
# =============================================================================
# run_resolution_sweep.sh — hr_build_gpu resolution sweep
# Location: shell_scripts/run_resolution_sweep.sh
#
# Runs the KH radiative simulation at multiple resolutions using the GPU-enabled
# Athena build:
#   128 ×  256
#   256 ×  512
#   512 × 1024
#   1024 × 2048
#
# For each resolution an athinput is generated from the 128 reference file
# with nx1/nx2 and meshblock sizes patched inline, then Athena is executed
# on the selected GPU device.
#
# Output directories:
#   <PROJECT_ROOT>/simulation_outputs/hr_gpu_128x256/
#   <PROJECT_ROOT>/simulation_outputs/hr_gpu_256x512/
#   <PROJECT_ROOT>/simulation_outputs/hr_gpu_512x1024/
#   <PROJECT_ROOT>/simulation_outputs/hr_gpu_1024x2048/
#
# Logs for each run land in the same output directory as <tag>.log.
#
# Environment overrides:
#   CUDA_VISIBLE_DEVICES — GPU device ID to use (default: 0)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# CUDA Environment
# ---------------------------------------------------------------------------
for cuda_dir in /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4 /usr/local/cuda-12 /usr/local/cuda; do
    if [ -d "$cuda_dir/bin" ]; then
        export CUDA_ROOT="$cuda_dir"
        export CUDA_HOME="$cuda_dir"
        export PATH="$cuda_dir/bin:$PATH"
        export LD_LIBRARY_PATH="$cuda_dir/lib64:${LD_LIBRARY_PATH:-}"
        break
    fi
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_SRC="${PROJECT_ROOT}/builds/hr_build_gpu/src"
ATHENA="${BUILD_SRC}/athena"

# Reference athinput (128×256 baseline kept in hr_build/src)
REF_ATHINPUT="${PROJECT_ROOT}/builds/hr_build/src/kh_radiative_128.athinput"

SIM_OUTPUTS="${PROJECT_ROOT}/simulation_outputs"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if [[ ! -x "${ATHENA}" ]]; then
    echo "ERROR: Athena GPU executable not found or not executable: ${ATHENA}" >&2
    exit 1
fi

if [[ ! -f "${REF_ATHINPUT}" ]]; then
    echo "ERROR: Reference athinput not found: ${REF_ATHINPUT}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Helper: generate a patched athinput for a given resolution
#
# Usage: make_athinput <nx1> <nx2> <mb_nx1> <mb_nx2> <output_path>
#
# Meshblock decomposition: 4 tiles in X1, 8 tiles in X2 = 32 total meshblocks.
#   mb_nx1 = nx1 / 4
#   mb_nx2 = nx2 / 8
# ---------------------------------------------------------------------------
make_athinput() {
    local nx1="$1"
    local nx2="$2"
    local mb_nx1="$3"
    local mb_nx2="$4"
    local out="$5"

    python3 - <<PY "${REF_ATHINPUT}" "${out}" "${nx1}" "${nx2}" "${mb_nx1}" "${mb_nx2}"
import sys, re, pathlib

src    = pathlib.Path(sys.argv[1]).read_text()
out    = pathlib.Path(sys.argv[2])
nx1    = sys.argv[3]; nx2    = sys.argv[4]
mb_nx1 = sys.argv[5]; mb_nx2 = sys.argv[6]

in_mesh = in_mb = False
lines_out = []
for line in src.splitlines():
    stripped = line.strip()
    if stripped == '<mesh>':
        in_mesh = True; in_mb = False
    elif stripped == '<meshblock>':
        in_mb = True; in_mesh = False
    elif stripped.startswith('<') and stripped.endswith('>'):
        in_mesh = in_mb = False

    if in_mesh and re.match(r'\s*nx1\s*=', line):
        line = re.sub(r'(\s*nx1\s*=\s*)\d+', r'\g<1>' + nx1, line)
    elif in_mesh and re.match(r'\s*nx2\s*=', line):
        line = re.sub(r'(\s*nx2\s*=\s*)\d+', r'\g<1>' + nx2, line)
    elif in_mb and re.match(r'\s*nx1\s*=', line):
        line = re.sub(r'(\s*nx1\s*=\s*)\d+', r'\g<1>' + mb_nx1, line)
    elif in_mb and re.match(r'\s*nx2\s*=', line):
        line = re.sub(r'(\s*nx2\s*=\s*)\d+', r'\g<1>' + mb_nx2, line)

    lines_out.append(line)

out.write_text('\n'.join(lines_out) + '\n')
print(f"  Written: {out}")
PY
}

# ---------------------------------------------------------------------------
# Helper: run one simulation
#
# Usage: run_sim <tag> <nx1> <nx2> <mb_nx1> <mb_nx2>
# ---------------------------------------------------------------------------
run_sim() {
    local tag="$1"
    local nx1="$2"
    local nx2="$3"
    local mb_nx1="$4"
    local mb_nx2="$5"

    local out_dir="${SIM_OUTPUTS}/hr_gpu_${tag}"
    local athinput="${out_dir}/kh_radiative_${tag}.athinput"
    local log_file="${out_dir}/${tag}.log"

    echo ""
    echo "============================================================"
    echo " Starting resolution: ${tag}  (nx1=${nx1}, nx2=${nx2})"
    echo " CUDA Device : GPU ${CUDA_VISIBLE_DEVICES}"
    echo " Meshblock   : ${mb_nx1} x ${mb_nx2}"
    echo " Output dir  : ${out_dir}"
    echo " Log file    : ${log_file}"
    echo "============================================================"

    mkdir -p "${out_dir}"

    # Generate patched athinput
    make_athinput "${nx1}" "${nx2}" "${mb_nx1}" "${mb_nx2}" "${athinput}"

    # Confirm mesh settings
    echo "  Resolved mesh settings:"
    grep -E '^\s*nx[123]\s*=' "${athinput}" | sed 's/^/    /'

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching Athena on GPU ${CUDA_VISIBLE_DEVICES} for ${tag} ..."

    cd "${BUILD_SRC}"

    "${ATHENA}" \
        -i "${athinput}" \
        -d "${out_dir}" \
        2>&1 | tee "${log_file}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: ${tag}"
}

# ---------------------------------------------------------------------------
# Resolution sweep
#
# Runs across 7 resolutions:
#   16  ×   8  (nx1=8,    nx2=16)
#   32  ×  16  (nx1=16,   nx2=32)
#   64  ×  32  (nx1=32,   nx2=64)
#  128  ×  64  (nx1=64,   nx2=128)
#  256  × 128  (nx1=128,  nx2=256)
#  512  × 256  (nx1=256,  nx2=512)
# 1024  × 512  (nx1=512,  nx2=1024)
#
# Meshblock decomposition: 4×8 = 32 total meshblocks.
#   mb_nx1 = nx1 / 4   (4 tiles in X1)
#   mb_nx2 = nx2 / 8   (8 tiles in X2)
# ---------------------------------------------------------------------------

# --- 16 x 8 (nx1=8, nx2=16) ---  mb: 4x4  (2x4=8 MBs, mb_nx >= 4)
run_sim "8x16"        8    16   4              4

# --- 32 x 16 (nx1=16, nx2=32) ---  mb: 4x4  (4x8=32 MBs)
run_sim "16x32"      16    32   $((16 / 4))    $((32 / 8))

# --- 64 x 32 (nx1=32, nx2=64) ---  mb: 8x8  (4x8=32 MBs)
run_sim "32x64"      32    64   $((32 / 4))    $((64 / 8))

# --- 128 x 64 (nx1=64, nx2=128) ---  mb: 16x16  (4x8=32 MBs)
run_sim "64x128"     64   128   $((64 / 4))    $((128 / 8))

# --- 256 x 128 (nx1=128, nx2=256) ---  mb: 32x32  (4x8=32 MBs)
run_sim "128x256"   128   256   $((128 / 4))   $((256 / 8))

# --- 512 x 256 (nx1=256, nx2=512) ---  mb: 64x64  (4x8=32 MBs)
run_sim "256x512"   256   512   $((256 / 4))   $((512 / 8))

# --- 1024 x 512 (nx1=512, nx2=1024) ---  mb: 128x128  (4x8=32 MBs)
run_sim "512x1024"  512  1024   $((512 / 4))   $((1024 / 8))

echo ""
echo "============================================================"
echo " All resolution sweep runs completed successfully."
echo " Outputs:"
echo "   ${SIM_OUTPUTS}/hr_gpu_8x16/"
echo "   ${SIM_OUTPUTS}/hr_gpu_16x32/"
echo "   ${SIM_OUTPUTS}/hr_gpu_32x64/"
echo "   ${SIM_OUTPUTS}/hr_gpu_64x128/"
echo "   ${SIM_OUTPUTS}/hr_gpu_128x256/"
echo "   ${SIM_OUTPUTS}/hr_gpu_256x512/"
echo "   ${SIM_OUTPUTS}/hr_gpu_512x1024/"
echo "============================================================"

# ---------------------------------------------------------------------------
# Post-processing: Profile Plots & Visualizations
# ---------------------------------------------------------------------------
if [[ -f "${PROJECT_ROOT}/venv/bin/python" ]]; then
    PYTHON_EXEC="${PROJECT_ROOT}/venv/bin/python"
else
    PYTHON_EXEC="python3"
fi

echo ""
echo "============================================================"
echo " Generating profile comparison plots ..."
echo " Running: ${PROJECT_ROOT}/explore_data/plot_profiles.py"
echo "============================================================"
"${PYTHON_EXEC}" "${PROJECT_ROOT}/explore_data/plot_profiles.py"

echo ""
echo "============================================================"
echo " Generating simulation animations (saving to run folders) ..."
echo " Running: ${PROJECT_ROOT}/explore_data/visualize_hr_mpi_sims.py"
echo "============================================================"
"${PYTHON_EXEC}" "${PROJECT_ROOT}/explore_data/visualize_hr_mpi_sims.py"

echo ""
echo "============================================================"
echo " All resolution sweep runs, plots, and animations completed!"
echo "============================================================"

