#!/usr/bin/env bash
# =============================================================================
# run_box_sweep.sh
# Location: shell_scripts/run_box_sweep.sh
#
# Runs Kelvin-Helmholtz simulations across all resolutions seen in
# run_resolution_sweep:
#   8x16, 16x32, 32x64, 64x128, 128x256, 256x512, 512x1024
#
# For each resolution:
#   1. Box 10pc x 20pc  (x in [-5, 5], y in [-10, 10], a_char=0.125, sigma=0.5)
#   2. Box 20pc x 40pc  (x in [-10, 10], y in [-20, 20], a_char=0.25, sigma=1.0)
#   3. Box 30pc x 60pc  (x in [-15, 15], y in [-30, 30], a_char=0.375, sigma=1.5)
#
# Followed by running explore_data/plot_box_profiles.py to generate profiles
# plotted vs y / box_height, with a separate plot for each resolution (comparing all 3 box sizes).
#
# Environment Overrides:
#   CUDA_VISIBLE_DEVICES - GPU device ID (default: 0)
#   TLIM                 - Simulation time limit (default: 10.00)
#   KEEP_SIM_FILES       - 0 to delete simulation outputs after plotting each resolution (default: 0)
#   ONLY_PLOTS           - 1 to only generate plots without running Athena (default: 0)
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
# Paths and configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_SRC="${PROJECT_ROOT}/builds/hr_build_gpu/src"
ATHENA="${BUILD_SRC}/athena"
REF_ATHINPUT="${PROJECT_ROOT}/shell_scripts/ref_athinput.athinp"
SWEEP_ROOT="${PROJECT_ROOT}/simulation_outputs/box_sweep"

TLIM="${TLIM:-10.00}"
KEEP_SIM_FILES="${KEEP_SIM_FILES:-0}"
ONLY_PLOTS="${ONLY_PLOTS:-0}"

# ---------------------------------------------------------------------------
# Python executable
# ---------------------------------------------------------------------------
if [[ -f "${PROJECT_ROOT}/venv/bin/python" ]]; then
    PYTHON_EXEC="${PROJECT_ROOT}/venv/bin/python"
else
    PYTHON_EXEC="python3"
fi

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if [[ "${ONLY_PLOTS}" -eq 0 ]]; then
    if [[ ! -x "${ATHENA}" ]]; then
        echo "ERROR: Athena GPU binary not found or not executable: ${ATHENA}" >&2
        exit 1
    fi
    if [[ ! -f "${REF_ATHINPUT}" ]]; then
        echo "ERROR: Reference athinput not found: ${REF_ATHINPUT}" >&2
        exit 1
    fi
fi

mkdir -p "${SWEEP_ROOT}"

# ---------------------------------------------------------------------------
# Helper: generate patched athinput for given resolution and box parameters
# ---------------------------------------------------------------------------
make_athinput() {
    local out_path="$1"
    local nx1="$2"
    local nx2="$3"
    local mb_nx1="$4"
    local mb_nx2="$5"
    local x1min="$6"
    local x1max="$7"
    local x2min="$8"
    local x2max="$9"
    local a_char="${10}"
    local sigma="${11}"
    local tlim="${12}"

    python3 - <<PY "${REF_ATHINPUT}" "${out_path}" "${nx1}" "${nx2}" "${mb_nx1}" "${mb_nx2}" "${x1min}" "${x1max}" "${x2min}" "${x2max}" "${a_char}" "${sigma}" "${tlim}"
import sys, re, pathlib

(
    src_path, out_path_str,
    nx1, nx2, mb_nx1, mb_nx2,
    x1min, x1max, x2min, x2max,
    a_char, sigma, tlim
) = sys.argv[1:14]

src = pathlib.Path(src_path).read_text()
out_path = pathlib.Path(out_path_str)

in_mesh = in_mb = in_prob = in_time = False
lines_out = []

for line in src.splitlines():
    stripped = line.strip()
    if stripped == '<mesh>':
        in_mesh = True; in_mb = in_prob = in_time = False
    elif stripped == '<meshblock>':
        in_mb = True; in_mesh = in_prob = in_time = False
    elif stripped == '<problem>':
        in_prob = True; in_mesh = in_mb = in_time = False
    elif stripped == '<time>':
        in_time = True; in_mesh = in_mb = in_prob = False
    elif stripped.startswith('<') and stripped.endswith('>'):
        in_mesh = in_mb = in_prob = in_time = False

    # Mesh section
    if in_mesh:
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

    # MeshBlock section
    elif in_mb:
        if re.match(r'\s*nx1\s*=', line):
            line = re.sub(r'(\s*nx1\s*=\s*)\d+', r'\g<1>' + mb_nx1, line)
        elif re.match(r'\s*nx2\s*=', line):
            line = re.sub(r'(\s*nx2\s*=\s*)\d+', r'\g<1>' + mb_nx2, line)

    # Time section
    elif in_time:
        if re.match(r'\s*tlim\s*=', line):
            line = re.sub(r'(\s*tlim\s*=\s*)[-\d.]+', r'\g<1>' + tlim, line)

    # Problem section
    elif in_prob:
        if re.match(r'\s*a_char\s*=', line):
            line = re.sub(r'(\s*a_char\s*=\s*)[-\d.]+', r'\g<1>' + a_char, line)
        elif re.match(r'\s*sigma\s*=', line):
            line = re.sub(r'(\s*sigma\s*=\s*)[-\d.]+', r'\g<1>' + sigma, line)

    lines_out.append(line)

out_path.write_text('\n'.join(lines_out) + '\n')
print(f"  Generated athinput: {out_path}")
PY
}

# ---------------------------------------------------------------------------
# Helper: run simulation for a given resolution and box size
# ---------------------------------------------------------------------------
run_case() {
    local res_tag="$1"
    local nx1="$2"
    local nx2="$3"
    local mb_nx1="$4"
    local mb_nx2="$5"
    local box_tag="$6"
    local x1min="$7"
    local x1max="$8"
    local x2min="$9"
    local x2max="${10}"
    local a_char="${11}"
    local sigma="${12}"

    local sim_name="hr_gpu_${res_tag}_${box_tag}"
    local out_dir="${SWEEP_ROOT}/${sim_name}"
    local athinput="${out_dir}/kh_radiative_${res_tag}_${box_tag}.athinput"
    local log_file="${out_dir}/${res_tag}_${box_tag}.log"

    echo ""
    echo "========================================================================"
    echo " Simulation: ${sim_name}"
    echo " Resolution : nx1=${nx1}, nx2=${nx2} (Meshblock: ${mb_nx1}x${mb_nx2})"
    echo " Domain X1  : [${x1min}, ${x1max}] pc"
    echo " Domain X2  : [${x2min}, ${x2max}] pc"
    echo " Scales     : a_char=${a_char}, sigma=${sigma}"
    echo " Out Dir    : ${out_dir}"
    echo "========================================================================"

    mkdir -p "${out_dir}"

    # Generate athinput
    make_athinput "${athinput}" "${nx1}" "${nx2}" "${mb_nx1}" "${mb_nx2}" \
                  "${x1min}" "${x1max}" "${x2min}" "${x2max}" \
                  "${a_char}" "${sigma}" "${TLIM}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching Athena on GPU ${CUDA_VISIBLE_DEVICES} ..."
    cd "${BUILD_SRC}"

    "${ATHENA}" \
        -i "${athinput}" \
        -d "${out_dir}" \
        2>&1 | tee "${log_file}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: ${sim_name}"
}

# ---------------------------------------------------------------------------
# Resolution Pipeline: Run -> Plot -> Delete
# ---------------------------------------------------------------------------
RESOLUTIONS=(
    "8x16 8 16 4 4"
    "16x32 16 32 4 4"
    "32x64 32 64 8 8"
    "64x128 64 128 16 16"
    "128x256 128 256 32 32"
    "256x512 256 512 64 64"
)

for entry in "${RESOLUTIONS[@]}"; do
    read -r res_tag nx1 nx2 mb1 mb2 <<< "${entry}"

    echo ""
    echo "########################################################################"
    echo " Processing Resolution: ${res_tag} (nx1=${nx1}, nx2=${nx2})"
    echo "########################################################################"

    if [[ "${ONLY_PLOTS}" -eq 0 ]]; then
        # 1. Run Box 10pc x 20pc
        run_case "${res_tag}" "${nx1}" "${nx2}" "${mb1}" "${mb2}" \
                 "box_10x20" "-5.0" "5.0" "-10.0" "10.0" "0.125" "0.5"

        # 2. Run Box 20pc x 40pc
        run_case "${res_tag}" "${nx1}" "${nx2}" "${mb1}" "${mb2}" \
                 "box_20x40" "-10.0" "10.0" "-20.0" "20.0" "0.25" "1.0"

        # 3. Run Box 30pc x 60pc
        run_case "${res_tag}" "${nx1}" "${nx2}" "${mb1}" "${mb2}" \
                 "box_30x60" "-15.0" "15.0" "-30.0" "30.0" "0.375" "1.5"
    fi

    # 4. Generate profile plots comparing the 3 boxes for this resolution
    echo ""
    echo "========================================================================"
    echo " Generating vertical profiles for resolution: ${res_tag} ..."
    echo "========================================================================"
    "${PYTHON_EXEC}" "${PROJECT_ROOT}/explore_data/plot_box_profiles.py" "${res_tag}"

    # 5. Delete simulation output files for this resolution to free disk space
    if [[ "${KEEP_SIM_FILES}" -eq 0 ]]; then
        echo ""
        echo "========================================================================"
        echo " Cleaning up simulation data for resolution ${res_tag} ..."
        echo "========================================================================"
        for box in "box_10x20" "box_20x40" "box_30x60"; do
            target_dir="${SWEEP_ROOT}/hr_gpu_${res_tag}_${box}"
            if [[ -d "${target_dir}" ]]; then
                echo "  Removing: ${target_dir}"
                rm -rf "${target_dir}"
            fi
        done
        echo "  Cleanup complete for ${res_tag}."
    fi
done

echo ""
echo "========================================================================"
echo " All resolution runs, plots, and cleanups completed successfully!"
echo " Output plots are located in: ${PROJECT_ROOT}/outputs/box_sweep_profiles/"
echo "========================================================================"

