#!/usr/bin/env bash
# =============================================================================
# run_sim_2048x1024_2xlength.sh
# Location: shell_scripts/run_sim_2048x1024_2xlength.sh
#
# Runs the Kelvin-Helmholtz simulation with radiative cooling at:
#   Grid resolution : nx1 = 1024, nx2 = 2048 (32 meshblocks of 256x256)
#   Physical domain : X1 in [-10.0, 10.0] (Lx = 20 pc, 2x baseline)
#                     X2 in [-20.0, 20.0] (Ly = 40 pc, 2x baseline)
#   Scale parameters: a_char = 0.25 (2x baseline)
#                     sigma  = 1.0  (2x baseline)
#   MPI ranks       : 16 ranks (default, overridable via MPI_NP)
#
# Output directory  : <PROJECT_ROOT>/simulation_outputs/hr_mpi_1024x2048_2xlength/
# Log file          : <output_dir>/1024x2048_2xlength.log
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_SRC="${PROJECT_ROOT}/builds/hr_build_mpi/src"
ATHENA="${BUILD_SRC}/athena"
REF_ATHINPUT="${PROJECT_ROOT}/builds/hr_build/src/kh_radiative_128.athinput"

TAG="1024x2048_2xlength"
OUT_DIR="${PROJECT_ROOT}/simulation_outputs/hr_mpi_${TAG}"
ATHINPUT="${OUT_DIR}/kh_radiative_${TAG}.athinput"
LOG_FILE="${OUT_DIR}/${TAG}.log"

# MPI configuration (default: 16 ranks)
MPI_NP="${MPI_NP:-16}"

# Grid resolution & decomposition
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

# Simulation time limit
TLIM="${TLIM:-10.00}"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if [[ ! -x "${ATHENA}" ]]; then
    echo "ERROR: Athena MPI binary not found or not executable: ${ATHENA}" >&2
    exit 1
fi

if [[ ! -f "${REF_ATHINPUT}" ]]; then
    echo "ERROR: Reference athinput not found: ${REF_ATHINPUT}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

# ---------------------------------------------------------------------------
# Generate patched athinput with 2x physical length scales
# ---------------------------------------------------------------------------
echo "========================================================================"
echo " Generating patched athinput with 2x physical length scales ..."
echo "========================================================================"

python3 - <<PY "${REF_ATHINPUT}" "${ATHINPUT}" "${NX1}" "${NX2}" "${MB_NX1}" "${MB_NX2}" "${X1MIN}" "${X1MAX}" "${X2MIN}" "${X2MAX}" "${A_CHAR}" "${SIGMA}" "${TLIM}"
import sys, re, pathlib

src_path = sys.argv[1]
out_path = pathlib.Path(sys.argv[2])
nx1, nx2 = sys.argv[3], sys.argv[4]
mb_nx1, mb_nx2 = sys.argv[5], sys.argv[6]
x1min, x1max = sys.argv[7], sys.argv[8]
x2min, x2max = sys.argv[9], sys.argv[10]
a_char, sigma = sys.argv[11], sys.argv[12]
tlim = sys.argv[13]

src = pathlib.Path(src_path).read_text()
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

    # Mesh section updates
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

    # Problem section updates (2x physical length parameters)
    elif in_prob:
        if re.match(r'\s*a_char\s*=', line):
            line = re.sub(r'(\s*a_char\s*=\s*)[-\d.]+', r'\g<1>' + a_char, line)
        elif re.match(r'\s*sigma\s*=', line):
            line = re.sub(r'(\s*sigma\s*=\s*)[-\d.]+', r'\g<1>' + sigma, line)

    lines_out.append(line)

out_path.write_text('\n'.join(lines_out) + '\n')
print(f"Generated input file: {out_path}")
PY

# ---------------------------------------------------------------------------
# Display setup summary
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo " Athena KH Simulation Configuration"
echo "========================================================================"
echo " Tag            : ${TAG}"
echo " Resolution     : nx1=${NX1}, nx2=${NX2}"
echo " Meshblock Size : ${MB_NX1} x ${MB_NX2} (32 MeshBlocks total)"
echo " MPI Processes  : ${MPI_NP} ranks"
echo " Domain X1      : [${X1MIN}, ${X1MAX}] (Lx = $(python3 -c "print(${X1MAX} - (${X1MIN}))") pc)"
echo " Domain X2      : [${X2MIN}, ${X2MAX}] (Ly = $(python3 -c "print(${X2MAX} - (${X2MIN}))") pc)"
echo " a_char (width) : ${A_CHAR}"
echo " sigma (perturb): ${SIGMA}"
echo " tlim           : ${TLIM} Myr"
echo " Output Dir     : ${OUT_DIR}"
echo " Log File       : ${LOG_FILE}"
echo " Executable     : ${ATHENA}"
echo "========================================================================"
echo ""

# ---------------------------------------------------------------------------
# Launch Athena simulation
# ---------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching Athena via mpirun (${MPI_NP} ranks)..."

cd "${BUILD_SRC}"

mpirun -np "${MPI_NP}" "${ATHENA}" \
    -i "${ATHINPUT}" \
    -d "${OUT_DIR}" \
    2>&1 | tee "${LOG_FILE}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Simulation completed successfully!"
echo "Outputs stored in: ${OUT_DIR}"
