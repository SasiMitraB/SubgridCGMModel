#!/usr/bin/env bash
# =============================================================================
# run_resolution_sweep.sh — hr_build_mpi resolution sweep
# Location: shell_scripts/run_resolution_sweep.sh
#
# Runs the KH radiative simulation at three resolutions using the MPI-enabled
# Athena build:
#   128 ×  256
#   256 ×  512
#   512 × 1024
#
# For each resolution an athinput is generated from the 128 reference file
# with nx1/nx2 and meshblock sizes patched inline, then mpirun is invoked.
#
# Output directories:
#   <PROJECT_ROOT>/simulation_outputs/hr_mpi_128x256/
#   <PROJECT_ROOT>/simulation_outputs/hr_mpi_256x512/
#   <PROJECT_ROOT>/simulation_outputs/hr_mpi_512x1024/
#
# Logs for each run land in the same output directory as <tag>.log.
#
# Environment overrides:
#   MPI_NP   — number of MPI ranks (default: 4)
#              nx2 must be divisible by MPI_NP for the default decomposition.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT="/home/sasi/Projects/SubgridCGMModel"
BUILD_SRC="${PROJECT_ROOT}/builds/hr_build_mpi/src"
ATHENA="${BUILD_SRC}/athena"

# Reference athinput (128×256 baseline kept in hr_build/src)
REF_ATHINPUT="${PROJECT_ROOT}/builds/hr_build/src/kh_radiative_128.athinput"

SIM_OUTPUTS="${PROJECT_ROOT}/simulation_outputs"

# Number of MPI ranks — override via: MPI_NP=8 ./run_resolution_sweep.sh
MPI_NP="${MPI_NP:-4}"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if [[ ! -x "${ATHENA}" ]]; then
    echo "ERROR: Athena executable not found or not executable: ${ATHENA}" >&2
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
# Meshblock sizes are chosen so that each MPI rank owns exactly one meshblock.
# For MPI_NP ranks the default split tiles ranks along X2:
#   mb_nx1 = nx1,  mb_nx2 = nx2 / MPI_NP
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

    local out_dir="${SIM_OUTPUTS}/hr_mpi_${tag}"
    local athinput="${out_dir}/kh_radiative_${tag}.athinput"
    local log_file="${out_dir}/${tag}.log"

    echo ""
    echo "============================================================"
    echo " Starting resolution: ${tag}  (nx1=${nx1}, nx2=${nx2})"
    echo " MPI ranks : ${MPI_NP}"
    echo " Meshblock : ${mb_nx1} x ${mb_nx2}"
    echo " Output dir: ${out_dir}"
    echo " Log file  : ${log_file}"
    echo "============================================================"

    mkdir -p "${out_dir}"

    # Generate patched athinput
    make_athinput "${nx1}" "${nx2}" "${mb_nx1}" "${mb_nx2}" "${athinput}"

    # Confirm mesh settings
    echo "  Resolved mesh settings:"
    grep -E '^\s*nx[123]\s*=' "${athinput}" | sed 's/^/    /'

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching mpirun for ${tag} ..."

    cd "${BUILD_SRC}"

    mpirun -np "${MPI_NP}" "${ATHENA}" \
        -i "${athinput}" \
        -d "${out_dir}" \
        2>&1 | tee "${log_file}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: ${tag}"
}

# ---------------------------------------------------------------------------
# Resolution sweep
#
# Meshblock decomposition: MPI ranks tiled along the X2 (vertical) axis.
#   mb_nx1 = nx1  (single tile in X1)
#   mb_nx2 = nx2 / MPI_NP
#
# Requires: nx2 % MPI_NP == 0 for all three resolutions.
# With the default MPI_NP=4: 256/4=64, 512/4=128, 1024/4=256 — all integer.
# ---------------------------------------------------------------------------

# --- 128 x 256 ---
run_sim "128x256"   128  256  128 $((256  / MPI_NP))

# --- 256 x 512 ---
run_sim "256x512"   256  512  256 $((512  / MPI_NP))

# --- 512 x 1024 ---
run_sim "512x1024"  512 1024  512 $((1024 / MPI_NP))

echo ""
echo "============================================================"
echo " All three resolutions completed successfully."
echo " Outputs:"
echo "   ${SIM_OUTPUTS}/hr_mpi_128x256/"
echo "   ${SIM_OUTPUTS}/hr_mpi_256x512/"
echo "   ${SIM_OUTPUTS}/hr_mpi_512x1024/"
echo "============================================================"
