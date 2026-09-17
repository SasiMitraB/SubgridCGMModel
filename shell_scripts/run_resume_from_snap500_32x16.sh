#!/usr/bin/env bash
# =============================================================================
# run_resume_from_snap500_32x16.sh
#
# Automated Pipeline for Kelvin-Helmholtz Tiled Subgrid CGM Evolution:
#
# 1. Accepts CELL_SIZE (1.25 or 0.625 pc) and domain size BOX_LENGTH_X, BOX_LENGTH_Y.
#    Validates that domain lengths fit an integer number of 10x20 pc tiles:
#      N_tile_x = BOX_LENGTH_X / 10
#      N_tile_y = BOX_LENGTH_Y / 20
# 2. Computes High Resolution (HR) and Low Resolution (LR) grids:
#      HR res = 0.0390625 pc (downsampled by 32 for 1.25 pc, or 16 for 0.625 pc)
#      e.g. 10x20 pc @ 1.25 pc: HR 256x512,  LR 8x16,   1x1 tile
#      e.g. 20x40 pc @ 1.25 pc: HR 512x1024, LR 16x32,  2x2 tiles
#      e.g. 30x60 pc @ 1.25 pc: HR 768x1536, LR 24x48,  3x3 tiles
#      e.g. 40x80 pc @ 1.25 pc: HR 1024x2048, LR 32x64, 4x4 tiles
# 3. Runs HR simulation using builds/hr_build_gpu for 10 Myr (tlim=10.0).
# 4. Cleans up snapshots before 5 Myr (< 00500) to save disk space.
# 5. Downsamples snapshot 500 (at t=5 Myr) to target LR resolution.
# 6. Generates tailored athinput files for subgrid_model and hr_build (LR).
# 7. Runs subgrid_model (CNN tiled inference) and hr_build (ISM cooling).
# 8. Runs explore_data/mock_sg_tiled.py to produce diagnostic plots and animations.
#
# Usage:
#   bash shell_scripts/run_resume_from_snap500_32x16.sh [subgrid|hr|both|postprocess]
#
# Environment variables:
#   CELL_SIZE            - 1.25 or 0.625 (default: 1.25)
#   BOX_LENGTH_X         - Domain length in X1 [pc] (default: 20)
#   BOX_LENGTH_Y         - Domain length in X2 [pc] (default: 40)
#   BOX_SIZE             - Optional shorthand like "20x40"
#   SKIP_HR_RUN          - 1 to skip running HR simulation if snapshots exist (default: 0)
#   KEEP_PRE5MYR_HR      - 1 to preserve HR snapshots before 5 Myr (default: 0)
#   TLIM                 - Simulation time limit (default: 10.0)
#   CUDA_VISIBLE_DEVICES - GPU device ID for HR GPU build (default: 0)
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_ACTIVATE="${PROJECT_ROOT}/venv/bin/activate"
if [ -f "${VENV_ACTIVATE}" ]; then
    source "${VENV_ACTIVATE}"
fi

# CUDA Environment Setup
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

# -----------------------------------------------------------------------------
# 1. Grid Cell Size & Box Dimension Parameters
# -----------------------------------------------------------------------------
CELL_SIZE="${CELL_SIZE:-0.625}"

# Support BOX_SIZE shorthand (e.g. "20x40" or "10x20") if explicit lengths not passed
if [ -n "${BOX_SIZE:-}" ] && [ -z "${BOX_LENGTH_X:-}" ] && [ -z "${BOX_LENGTH_Y:-}" ]; then
    BOX_LENGTH_X="$(echo "${BOX_SIZE}" | awk -Fx '{print $1}')"
    BOX_LENGTH_Y="$(echo "${BOX_SIZE}" | awk -Fx '{print $2}')"
fi
BOX_LENGTH_X="${BOX_LENGTH_X:-20}"
BOX_LENGTH_Y="${BOX_LENGTH_Y:-40}"

# -----------------------------------------------------------------------------
# 2. Validate Integer Number of 10x20 pc Tiles & Compute Resolutions
# -----------------------------------------------------------------------------
RES_INFO="$(python3 - <<PY
import sys, math

cell_size = float("${CELL_SIZE}")
box_x = float("${BOX_LENGTH_X}")
box_y = float("${BOX_LENGTH_Y}")

# Tile size is always 10 pc x 20 pc
TILE_LEN_X = 10.0
TILE_LEN_Y = 20.0

n_tile_x = box_x / TILE_LEN_X
n_tile_y = box_y / TILE_LEN_Y

# Validate integer number of tiles
if not (math.isclose(n_tile_x, round(n_tile_x), abs_tol=1e-5) and n_tile_x >= 1.0):
    sys.stderr.write(f"ERROR: BOX_LENGTH_X ({box_x} pc) is not an integer multiple of tile width (10 pc). n_tiles_x={n_tile_x}\n")
    sys.exit(1)

if not (math.isclose(n_tile_y, round(n_tile_y), abs_tol=1e-5) and n_tile_y >= 1.0):
    sys.stderr.write(f"ERROR: BOX_LENGTH_Y ({box_y} pc) is not an integer multiple of tile height (20 pc). n_tiles_y={n_tile_y}\n")
    sys.exit(1)

n_tile_x = int(round(n_tile_x))
n_tile_y = int(round(n_tile_y))

# Downsample factor DS and HR resolution
if math.isclose(cell_size, 0.625, abs_tol=1e-3):
    ds = 16
elif math.isclose(cell_size, 1.25, abs_tol=1e-3) or math.isclose(cell_size, 1.26, abs_tol=1e-3):
    ds = 32
else:
    sys.stderr.write(f"ERROR: Unsupported CELL_SIZE: {cell_size}. Expected 0.625 or 1.25.\n")
    sys.exit(1)

hr_dx = cell_size / ds  # exactly 0.0390625 pc
hr_nx1 = int(round(box_x / hr_dx))
hr_nx2 = int(round(box_y / hr_dx))

lr_nx1 = int(round(box_x / cell_size))
lr_nx2 = int(round(box_y / cell_size))

# Tile size in cells for CNN
tile_rows = int(round(TILE_LEN_Y / cell_size))
tile_cols = int(round(TILE_LEN_X / cell_size))

x1min = -box_x / 2.0
x1max =  box_x / 2.0
x2min = -box_y / 2.0
x2max =  box_y / 2.0

# Instability perturbation scale factors proportional to length scale
a_char = 0.125 * (box_x / 10.0)
sigma  = 0.5   * (box_x / 10.0)

# Meshblock decomposition for HR GPU run (each MB typically 64x64 or 128x128)
mb_nx1 = min(128, hr_nx1 // 4 if hr_nx1 >= 256 else hr_nx1 // 2)
mb_nx2 = min(128, hr_nx2 // 8 if hr_nx2 >= 512 else hr_nx2 // 4)
if mb_nx1 < 4: mb_nx1 = hr_nx1
if mb_nx2 < 4: mb_nx2 = hr_nx2

print(f"{ds} {hr_nx1} {hr_nx2} {lr_nx1} {lr_nx2} {n_tile_y} {n_tile_x} {tile_rows} {tile_cols} {x1min} {x1max} {x2min} {x2max} {a_char} {sigma} {mb_nx1} {mb_nx2}")
PY
)"

read -r DS HR_NX1 HR_NX2 LR_NX1 LR_NX2 N_TILE_ROWS N_TILE_COLS TILE_ROWS TILE_COLS X1MIN X1MAX X2MIN X2MAX A_CHAR SIGMA HR_MB_NX1 HR_MB_NX2 <<< "${RES_INFO}"

NX1="${LR_NX1}"
NX2="${LR_NX2}"
TILE_GRID="${N_TILE_ROWS},${N_TILE_COLS}"

if [ "${CELL_SIZE}" = "0.625" ]; then
    DEFAULT_MODEL_SAVES="${PROJECT_ROOT}/runs/run_random_crop_20260909_164548/model_saves"
    DEFAULT_NORM_PREFIX="cnn_(512, 256)_32"
else
    DEFAULT_MODEL_SAVES="${PROJECT_ROOT}/runs/run_random_crop_20260904_191402/model_saves"
    DEFAULT_NORM_PREFIX="cnn_(512, 256)_32"
fi

# Directory structure
BOX_TAG="box_${BOX_LENGTH_X}x${BOX_LENGTH_Y}"
HR_SIM_DIR="${PROJECT_ROOT}/simulation_outputs/hr_gpu_${HR_NX1}x${HR_NX2}_${BOX_TAG}"
HR_ATHINPUT="${HR_SIM_DIR}/kh_radiative_${HR_NX1}x${HR_NX2}_${BOX_TAG}.athinput"
HR_BIN_DIR="${HR_SIM_DIR}/bin"
SNAP500_BIN="${HR_BIN_DIR}/KH.hydro_u.00500.bin"

IC_DIR="${PROJECT_ROOT}/simulation_outputs/downsampled_ic"
IC_FILE="${IC_DIR}/ic_snap500_${BOX_TAG}_${NX2}x${NX1}.bin"

SG_OUTPUT_DIR="${PROJECT_ROOT}/simulation_outputs/subgrid_${BOX_TAG}_${NX2}x${NX1}_from_snap500"
LR_OUTPUT_DIR="${PROJECT_ROOT}/simulation_outputs/hr_build_${BOX_TAG}_${NX2}x${NX1}_from_snap500"

ATHINPUT_DIR="${PROJECT_ROOT}/simulation_outputs/athinputs_${BOX_TAG}_${NX2}x${NX1}"
SG_ATHINPUT="${ATHINPUT_DIR}/subgrid_${NX2}x${NX1}.athinput"
LR_ATHINPUT="${ATHINPUT_DIR}/hr_build_${NX2}x${NX1}.athinput"

REF_ATHINPUT_SRC="${PROJECT_ROOT}/shell_scripts/ref_athinput.athinp"

# CNN model configuration
export CELL_SIZE="${CELL_SIZE}"
export DS="${DS}"
export MODEL_SAVES_DIR="${MODEL_SAVES_DIR:-${DEFAULT_MODEL_SAVES}}"
export NORM_PREFIX="${NORM_PREFIX:-${DEFAULT_NORM_PREFIX}}"
export PDF_CNN_RESOLUTION="512,256"
export PDF_CNN_DOWNSAMPLE="32"
export CNN_TILING_MODE="tiled"
export TILE_GRID="${TILE_GRID}"
export TILE_ROWS="${TILE_ROWS}"
export TILE_COLS="${TILE_COLS}"
export LOGT_ACTIVE_START="4.1"
export LOGT_ACTIVE_END="5.9"
export COOL_TFLOOR="1.0e4"

# Simulation controls
TLIM="${TLIM:-10.0}"
NLIM="${NLIM:--1}"
SKIP_HR_RUN="${SKIP_HR_RUN:-0}"
KEEP_PRE5MYR_HR="${KEEP_PRE5MYR_HR:-0}"
RUN_MODE="${1:-both}"  # subgrid, hr, both, or postprocess

mkdir -p "${IC_DIR}" "${ATHINPUT_DIR}" "${SG_OUTPUT_DIR}" "${LR_OUTPUT_DIR}" "${HR_SIM_DIR}"

write_manifest() {
    local target_file="$1"
    mkdir -p "$(dirname "${target_file}")"
    {
        echo "======================================================================"
        echo " SubgridCGM Simulation & Tiled CNN Pipeline Manifest"
        echo "======================================================================"
        echo "Generated at             : $(date)"
        echo "Timestamp ID             : ${TIMESTAMP}"
        echo "Script                   : ${BASH_SOURCE[0]}"
        echo "Project root             : ${PROJECT_ROOT}"
        echo "Execution mode           : ${RUN_MODE}"
        echo ""
        echo "--- Physical Domain & Box Geometry ---"
        echo "Box Length X1 (width)    : ${BOX_LENGTH_X} pc"
        echo "Box Length X2 (height)   : ${BOX_LENGTH_Y} pc"
        echo "Box aspect ratio (Y/X)   : $(python3 -c "print(${BOX_LENGTH_Y}/${BOX_LENGTH_X})")"
        echo "Domain extents X1        : [${X1MIN}, ${X1MAX}] pc"
        echo "Domain extents X2        : [${X2MIN}, ${X2MAX}] pc"
        echo ""
        echo "--- Physical Scales & Initial Perturbations ---"
        echo "Tanh shear layer width   : a_char = ${A_CHAR} pc"
        echo "Gaussian perturb sigma   : sigma = ${SIGMA} pc"
        echo "Perturbation amplitude   : amp = 0.05"
        echo "Shear velocities         : vx_hot = +28.1818182 km/s, vx_cold = -2.8181818 km/s"
        echo "Densities                : rho_hot = 0.001, rho_cold = 0.1"
        echo "Initial pressure         : press = 14.02645 (2.22685e-13 dyne/cm^2)"
        echo "Cold gas fraction        : cold_frac = 0.5"
        echo ""
        echo "--- High Resolution (HR) Simulation Setup ---"
        echo "HR grid dimensions       : nx1=${HR_NX1} (width/X1), nx2=${HR_NX2} (height/X2)"
        echo "HR cell size (dx=dy)     : 0.0390625 pc (3.08568e+18 cm * 0.0390625)"
        echo "HR meshblock resolution  : ${HR_MB_NX1} x ${HR_MB_NX2}"
        echo "HR total meshblocks      : $(( (HR_NX1 / HR_MB_NX1) * (HR_NX2 / HR_MB_NX2) ))"
        echo "HR simulation directory  : ${HR_SIM_DIR}"
        echo "HR athinput              : ${HR_ATHINPUT}"
        echo "HR snapshot 500 (5 Myr)  : ${SNAP500_BIN}"
        echo "HR GPU executable        : ${PROJECT_ROOT}/builds/hr_build_gpu/src/athena"
        echo "CUDA device              : GPU ${CUDA_VISIBLE_DEVICES}"
        echo ""
        echo "--- Low Resolution (LR) & Downsampling Setup ---"
        echo "Coarse grid cell size    : ${CELL_SIZE} pc"
        echo "Downsample factor (DS)   : ${DS} (HR -> LR)"
        echo "Coarse grid dimensions   : nx1=${NX1} (width/X1), nx2=${NX2} (height/X2)"
        echo "Downsampled IC binary    : ${IC_FILE}"
        echo "LR athinput (ISM cooling): ${LR_ATHINPUT}"
        echo "LR simulation directory  : ${LR_OUTPUT_DIR}"
        echo "LR build executable      : ${PROJECT_ROOT}/builds/hr_build/src/athena"
        echo ""
        echo "--- Subgrid Model & CNN Tiling Setup ---"
        echo "Subgrid athinput         : ${SG_ATHINPUT}"
        echo "Subgrid output directory : ${SG_OUTPUT_DIR}"
        echo "Subgrid executable       : ${PROJECT_ROOT}/builds/subgrid_model/src/athena"
        echo "CNN tiling mode          : ${CNN_TILING_MODE}"
        echo "Base physical tile size  : 10 pc (X1) x 20 pc (X2)"
        echo "Tile grid layout         : ${TILE_GRID} (N_tile_y, N_tile_x)"
        echo "Total CNN tiles          : $(( N_TILE_ROWS * N_TILE_COLS ))"
        echo "Tile shape in grid cells : ${TILE_ROWS} rows (X2) x ${TILE_COLS} cols (X1)"
        echo "Model saves dir          : ${MODEL_SAVES_DIR}"
        echo "Model norm prefix        : ${NORM_PREFIX}"
        echo "Model weights file       : ${MODEL_SAVES_DIR}/${NORM_PREFIX}.pth"
        echo "Model mean file          : ${MODEL_SAVES_DIR}/${NORM_PREFIX}_input_mean.npy"
        echo "Model std file           : ${MODEL_SAVES_DIR}/${NORM_PREFIX}_input_std.npy"
        echo "PDF CNN resolution       : ${PDF_CNN_RESOLUTION}"
        echo "PDF CNN downsample       : ${PDF_CNN_DOWNSAMPLE}"
        echo "Active log10(T) range    : [${LOGT_ACTIVE_START}, ${LOGT_ACTIVE_END}]"
        echo "Cooling floor temp       : ${COOL_TFLOOR} K"
        echo ""
        echo "--- Simulation Time Controls ---"
        echo "Simulation time limit    : ${TLIM} Myr"
        echo "Cycle limit (nlim)       : ${NLIM}"
        echo "HR start snapshot        : 500 (t = 5.0 Myr)"
        echo "Subgrid / LR restart time: 5.0 Myr (evolves up to ${TLIM} Myr)"
        echo "======================================================================"
    } > "${target_file}"
    echo "Manifest written to: ${target_file}"
}

# Write initial manifest into the simulation run output directory
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
SIM_MANIFEST="${PROJECT_ROOT}/simulation_outputs/manifest_${BOX_TAG}_${NX2}x${NX1}_${TIMESTAMP}.txt"
write_manifest "${SIM_MANIFEST}"

echo "======================================================================"
echo " SUBGRID CGM AUTOMATED PIPELINE: CELL_SIZE=${CELL_SIZE} pc, BOX=${BOX_LENGTH_X}x${BOX_LENGTH_Y} pc"
echo " Date: $(date)"
echo " Project root: ${PROJECT_ROOT}"
echo " Physical Domain: X1 in [${X1MIN}, ${X1MAX}] pc, X2 in [${X2MIN}, ${X2MAX}] pc"
echo " Perturbation scales: a_char=${A_CHAR}, sigma=${SIGMA}"
echo " High Resolution (HR): nx1=${HR_NX1} (width), nx2=${HR_NX2} (height), dx=0.0390625 pc"
echo " Target Low Res (LR) : nx1=${NX1} (width), nx2=${NX2} (height), dx=${CELL_SIZE} pc (DS=${DS})"
echo " CNN Tiling Setup    : ${N_TILE_ROWS}x${N_TILE_COLS} tiles of size ${TILE_ROWS}x${TILE_COLS} cells (each 10x20 pc)"
echo " Model saves dir     : ${MODEL_SAVES_DIR}"
echo " Manifest saved      : ${SIM_MANIFEST}"
echo " Mode                : ${RUN_MODE}"
echo "======================================================================"

# -----------------------------------------------------------------------------
# STEP 0: Generate HR Athinput & Run HR Simulation (10 Myr) on GPU
# -----------------------------------------------------------------------------
if [ "${RUN_MODE}" != "postprocess" ]; then
    echo ""
    echo "======================================================================"
    echo "[Step 0] High Resolution Simulation (${HR_NX1}x${HR_NX2}, tlim=${TLIM} Myr)"
    echo "======================================================================"

    # Generate HR athinput
    python3 - <<PY
import re, pathlib

ref_path = pathlib.Path("${REF_ATHINPUT_SRC}")
hr_path  = pathlib.Path("${HR_ATHINPUT}")
lines = ref_path.read_text().splitlines()

in_mesh = in_mb = in_prob = in_time = in_hydro_src = False
new_lines = []

for line in lines:
    stripped = line.strip()
    if stripped == '<mesh>':
        in_mesh = True; in_mb = in_prob = in_time = in_hydro_src = False
    elif stripped == '<meshblock>':
        in_mb = True; in_mesh = in_prob = in_time = in_hydro_src = False
    elif stripped == '<problem>':
        in_prob = True; in_mesh = in_mb = in_time = in_hydro_src = False
    elif stripped == '<time>':
        in_time = True; in_mesh = in_mb = in_prob = in_hydro_src = False
    elif stripped == '<hydro_srcterms>':
        in_hydro_src = True; in_mesh = in_mb = in_prob = in_time = False
    elif stripped.startswith('<') and stripped.endswith('>'):
        in_mesh = in_mb = in_prob = in_time = in_hydro_src = False

    if in_mesh:
        if re.match(r'^\s*nx1\s*=', line):
            line = f"nx1       = ${HR_NX1}        # Number of zones in X1-direction"
        elif re.match(r'^\s*nx2\s*=', line):
            line = f"nx2       = ${HR_NX2}        # Number of zones in X2-direction"
        elif re.match(r'^\s*x1min\s*=', line):
            line = f"x1min     = ${X1MIN}        # minimum value of X1"
        elif re.match(r'^\s*x1max\s*=', line):
            line = f"x1max     = ${X1MAX}        # maximum value of X1"
        elif re.match(r'^\s*x2min\s*=', line):
            line = f"x2min     = ${X2MIN}       # minimum value of X2"
        elif re.match(r'^\s*x2max\s*=', line):
            line = f"x2max     = ${X2MAX}        # maximum value of X2"
    elif in_mb:
        if re.match(r'^\s*nx1\s*=', line):
            line = f"nx1       = ${HR_MB_NX1}        # Number of cells in each MeshBlock, X1-dir"
        elif re.match(r'^\s*nx2\s*=', line):
            line = f"nx2       = ${HR_MB_NX2}        # Number of cells in each MeshBlock, X2-dir"
    elif in_time:
        if re.match(r'^\s*tlim\s*=', line):
            line = f"tlim       = ${TLIM}        # time limit"
    elif in_prob:
        if re.match(r'^\s*a_char\s*=', line):
            line = f"a_char    = ${A_CHAR}        # width of tanh profile"
        elif re.match(r'^\s*sigma\s*=', line):
            line = f"sigma     = ${SIGMA}        # width of gaussian profile"

    new_lines.append(line)

hr_path.write_text('\n'.join(new_lines) + '\n')
print(f"  Created HR athinput: {hr_path}")
PY

    # Check if HR run needed
    if [ "${SKIP_HR_RUN}" -eq 1 ] && [ -f "${SNAP500_BIN}" ]; then
        echo "Skipping HR Athena run (SKIP_HR_RUN=1 and snapshot 500 exists: ${SNAP500_BIN})"
    else
        HR_GPU_EXE="${PROJECT_ROOT}/builds/hr_build_gpu/src/athena"
        if [ ! -x "${HR_GPU_EXE}" ]; then
            echo "ERROR: Athena GPU executable not found at: ${HR_GPU_EXE}" >&2
            exit 1
        fi

        echo "Running HR simulation on GPU ${CUDA_VISIBLE_DEVICES} ..."
        (
            cd "${PROJECT_ROOT}/builds/hr_build_gpu/src"
            "${HR_GPU_EXE}" -i "${HR_ATHINPUT}" -d "${HR_SIM_DIR}"
        )
        echo "HR simulation completed. Outputs in: ${HR_SIM_DIR}"
    fi

    # Clean up snapshots before 5 Myr to save space
    if [ "${KEEP_PRE5MYR_HR}" -eq 0 ]; then
        echo "Cleaning up HR snapshots before 5 Myr (< snapshot 500) to save disk space..."
        find "${HR_BIN_DIR}" -maxdepth 1 -type f \( -name "KH.hydro_u.00[0-4]*.bin" -o -name "KH.hydro_w.00[0-4]*.bin" \) -delete 2>/dev/null || true
        echo "Cleanup complete. Preserved snapshot 500 and subsequent snapshots."
    fi
fi

# -----------------------------------------------------------------------------
# STEP 1: Downsample snapshot 500 to LR binary IC
# -----------------------------------------------------------------------------
if [ ! -f "${SNAP500_BIN}" ]; then
    echo "ERROR: Snapshot 500 not found at: ${SNAP500_BIN}" >&2
    exit 1
fi

echo ""
echo "======================================================================"
echo "[Step 1] Downsampling snapshot 500 -> ${IC_FILE} ..."
echo "======================================================================"
python3 "${PROJECT_ROOT}/data/downsample_ic.py" \
    --input "${SNAP500_BIN}" \
    --nx1 "${NX1}" \
    --nx2 "${NX2}" \
    --output "${IC_FILE}"

# -----------------------------------------------------------------------------
# STEP 2: Generate .athinput files for subgrid_model and hr_build (LR)
# -----------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "[Step 2] Generating .athinput files for Subgrid and LR ..."
echo "======================================================================"

python3 - <<PY
import re, pathlib

ref_path = pathlib.Path("${HR_ATHINPUT}")
sg_path  = pathlib.Path("${SG_ATHINPUT}")
lr_path  = pathlib.Path("${LR_ATHINPUT}")
ic_file  = "${IC_FILE}"
nx1_val  = "${NX1}"
nx2_val  = "${NX2}"
tlim_val = "${TLIM}"
nlim_val = "${NLIM}"

content = ref_path.read_text()

def patch_athinput(text, is_subgrid=False):
    lines = text.splitlines()
    in_mesh = in_mb = in_hydro_src = in_problem = in_time = in_hydro = False
    new_lines = []
    
    for line in lines:
        stripped = line.strip()
        if stripped == '<mesh>':
            in_mesh = True; in_mb = in_hydro_src = in_problem = in_time = in_hydro = False
        elif stripped == '<meshblock>':
            in_mb = True; in_mesh = in_hydro_src = in_problem = in_time = in_hydro = False
        elif stripped == '<hydro_srcterms>':
            in_hydro_src = True; in_mesh = in_mb = in_problem = in_time = in_hydro = False
        elif stripped == '<problem>':
            in_problem = True; in_mesh = in_mb = in_hydro_src = in_time = in_hydro = False
        elif stripped == '<time>':
            in_time = True; in_mesh = in_mb = in_hydro_src = in_problem = in_hydro = False
        elif stripped == '<hydro>':
            in_hydro = True; in_mesh = in_mb = in_hydro_src = in_problem = in_time = False
        elif stripped.startswith('<') and stripped.endswith('>'):
            in_mesh = in_mb = in_hydro_src = in_problem = in_time = in_hydro = False

        if in_mesh and re.match(r'^\s*nx1\s*=', line):
            line = f"nx1       = {nx1_val}        # Number of zones in X1-direction"
        elif in_mesh and re.match(r'^\s*nx2\s*=', line):
            line = f"nx2       = {nx2_val}        # Number of zones in X2-direction"
        elif in_mb and re.match(r'^\s*nx1\s*=', line):
            line = f"nx1       = {nx1_val}        # Number of cells in each MeshBlock, X1-dir"
        elif in_mb and re.match(r'^\s*nx2\s*=', line):
            line = f"nx2       = {nx2_val}        # Number of cells in each MeshBlock, X2-dir"
        elif in_time and re.match(r'^\s*tlim\s*=', line):
            line = f"tlim       = {tlim_val}        # time limit"
        elif in_time and re.match(r'^\s*nlim\s*=', line):
            line = f"nlim       = {nlim_val}         # cycle limit"
        elif in_hydro and re.match(r'^\s*nscalars\s*=', line):
            line = "nscalars    = 2         # number of passive scalars in hydro (tracer + cold frac)"
        elif in_hydro_src and re.match(r'^\s*ism_cooling\s*=', line):
            if is_subgrid:
                line = "ism_cooling = false"
            else:
                line = "ism_cooling = true"

        new_lines.append(line)

    # In problem block, set iprob = 2, init_file = ic_file, and user_srcs
    final_lines = []
    for line in new_lines:
        final_lines.append(line)
        if line.strip() == '<problem>':
            final_lines.append(f"iprob     = 2         # initialize from binary file")
            final_lines.append(f"init_file = {ic_file}")
            if is_subgrid:
                final_lines.append("user_srcs = true")
            else:
                final_lines.append("user_srcs = false")

    clean_lines = []
    seen_iprob = False
    for line in final_lines:
        if re.match(r'^\s*iprob\s*=', line):
            if seen_iprob:
                continue
            seen_iprob = True
        clean_lines.append(line)

    return '\n'.join(clean_lines) + '\n'

sg_path.write_text(patch_athinput(content, is_subgrid=True))
lr_path.write_text(patch_athinput(content, is_subgrid=False))
print(f"  Created: {sg_path}")
print(f"  Created: {lr_path}")
PY

# -----------------------------------------------------------------------------
# STEP 3: Evolve with subgrid_model
# -----------------------------------------------------------------------------
if [ "${RUN_MODE}" = "both" ] || [ "${RUN_MODE}" = "subgrid" ]; then
    echo ""
    echo "======================================================================"
    echo "[Step 3] Evolving with builds/subgrid_model (Tiled CNN Subgrid Model)"
    echo "======================================================================"
    SG_EXE="${PROJECT_ROOT}/builds/subgrid_model/src/athena"
    if [ ! -f "${SG_EXE}" ]; then
        echo "ERROR: Athena executable not found at: ${SG_EXE}" >&2
        exit 1
    fi

    # Ensure base conda python environment and site-packages are exposed to embedded Python
    CONDA_PY="/home/sasi/miniconda3/bin/python3"
    export PYTHONHOME="$("${CONDA_PY}" -c "import sys; print(sys.prefix)")"
    PY_SYS_PATH="$("${CONDA_PY}" -c "import sys; print(':'.join(p for p in sys.path if p))")"
    export PYTHONPATH="${PROJECT_ROOT}/builds/subgrid_model/src:${PROJECT_ROOT}:${PROJECT_ROOT}/models/conv_nn:${PY_SYS_PATH}:${PYTHONPATH:-}"
    export LD_LIBRARY_PATH="/home/sasi/miniconda3/lib:${LD_LIBRARY_PATH:-}"

    echo "Running Athena subgrid model..."
    rm -rf "${SG_OUTPUT_DIR}"/*
    (
        cd "${PROJECT_ROOT}/builds/subgrid_model/src"
        "${SG_EXE}" -i "${SG_ATHINPUT}" -d "${SG_OUTPUT_DIR}"
    )
    echo "Subgrid model evolution completed. Outputs in: ${SG_OUTPUT_DIR}"
fi

# -----------------------------------------------------------------------------
# STEP 4: Evolve with hr_build (LR ISM cooling)
# -----------------------------------------------------------------------------
if [ "${RUN_MODE}" = "both" ] || [ "${RUN_MODE}" = "hr" ]; then
    echo ""
    echo "======================================================================"
    echo "[Step 4] Evolving with builds/hr_build (Standard ISM Cooling)"
    echo "======================================================================"
    HR_EXE="${PROJECT_ROOT}/builds/hr_build/src/athena"
    if [ ! -f "${HR_EXE}" ]; then
        echo "ERROR: Athena executable not found at: ${HR_EXE}" >&2
        exit 1
    fi

    echo "Running Athena hr_build (LR)..."
    rm -rf "${LR_OUTPUT_DIR}"/*
    (
        cd "${PROJECT_ROOT}/builds/hr_build/src"
        "${HR_EXE}" -i "${LR_ATHINPUT}" -d "${LR_OUTPUT_DIR}"
    )
    echo "LR evolution completed. Outputs in: ${LR_OUTPUT_DIR}"
fi

# -----------------------------------------------------------------------------
# STEP 5: Generate Comparison Plots & Diagnostics (mock_sg_tiled.py)
# -----------------------------------------------------------------------------
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
PLOTS_DIR="${PROJECT_ROOT}/tiled_outputs/run_tiled_${BOX_TAG}_${NX2}x${NX1}_${TIMESTAMP}"
mkdir -p "${PLOTS_DIR}"

# Write Run Manifest
MANIFEST="${PLOTS_DIR}/manifest.txt"
write_manifest "${MANIFEST}"

echo ""
echo "======================================================================"
echo "[Step 5] Generating Full Diagnostic Suite & Animations via mock_sg_tiled.py"
echo " Plots directory: ${PLOTS_DIR}"
echo " Manifest saved : ${MANIFEST}"
echo "======================================================================"

export NX1="${NX1}"
export NX2="${NX2}"
export BOX_LENGTH_X="${BOX_LENGTH_X}"
export BOX_LENGTH_Y="${BOX_LENGTH_Y}"
export TILE_GRID="${TILE_GRID}"

python3 "${PROJECT_ROOT}/explore_data/mock_sg_tiled.py" \
    --output-dir "${PLOTS_DIR}" \
    --hr-athinput "${HR_ATHINPUT}" \
    --hr-bin "${HR_BIN_DIR}" \
    --sg-athinput "${SG_ATHINPUT}" \
    --sg-bin "${SG_OUTPUT_DIR}/bin" \
    --lr-athinput "${LR_ATHINPUT}" \
    --lr-bin "${LR_OUTPUT_DIR}/bin"

# -------- NEW: Plot dt vs time for all runs --------
echo ""
echo "======================================================================"
echo "[Step 6] Plotting dt vs time for HR, Subgrid, and LR runs"
echo "======================================================================"

python3 - <<PLOT_PY
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Define runs with their history files
runs_info = {
    "HR (${HR_NX1}x${HR_NX2})": "${HR_BIN_DIR}",
    "Subgrid (${NX1}x${NX2})": "${SG_OUTPUT_DIR}/bin",
    "LR (${NX1}x${NX2})": "${LR_OUTPUT_DIR}/bin",
}

# Extract dt data from .hst files
dt_data = {}
for label, bin_dir in runs_info.items():
    bin_path = Path(bin_dir)
    hst_files = list(bin_path.glob("*.hydro.hst"))

    if not hst_files:
        print(f"Warning: No .hst file found in {bin_dir}")
        continue

    hst_file = hst_files[0]  # Take first .hst file
    print(f"Reading {label} from {hst_file}")

    try:
        data = np.loadtxt(hst_file, comments='#')
        time = data[:, 0]
        dt = data[:, 1]
        dt_data[label] = (time, dt)
    except Exception as e:
        print(f"Error reading {hst_file}: {e}")

if dt_data:
    # Individual plots for each run
    for label, (time, dt) in dt_data.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time, dt, marker='o', markersize=2, label=label, alpha=0.7)
        ax.set_xlabel("Time (Myr)", fontsize=11)
        ax.set_ylabel("Timestep dt", fontsize=11)
        ax.set_title(f"Timestep Evolution: {label}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        # Save individual plot to run output directory
        if "HR" in label and not "Subgrid" in label:
            out_dir = Path("${HR_SIM_DIR}")
        elif "Subgrid" in label:
            out_dir = Path("${SG_OUTPUT_DIR}")
        else:
            out_dir = Path("${LR_OUTPUT_DIR}")

        out_dir.mkdir(parents=True, exist_ok=True)
        plot_file = out_dir / "dt_vs_time.png"
        plt.savefig(plot_file, dpi=150)
        print(f"Saved individual plot: {plot_file}")
        plt.close()

    # Comparison plot
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['blue', 'green', 'red']
    for (label, (time, dt)), color in zip(dt_data.items(), colors):
        ax.plot(time, dt, marker='o', markersize=2, label=label, alpha=0.7, color=color)

    ax.set_xlabel("Time (Myr)", fontsize=11)
    ax.set_ylabel("Timestep dt", fontsize=11)
    ax.set_title("Timestep Evolution: HR vs Subgrid vs LR", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')

    # Save comparison plot to PLOTS_DIR
    comparison_plot = Path("${PLOTS_DIR}") / "dt_vs_time_comparison.png"
    plt.savefig(comparison_plot, dpi=150)
    print(f"Saved comparison plot: {comparison_plot}")
    plt.close()
else:
    print("No dt vs time data available to plot.")
PLOT_PY

echo "dt vs time plots saved to individual run directories and comparison plot to ${PLOTS_DIR}"

echo ""
echo "======================================================================"
echo " ALL TASKS COMPLETED SUCCESSFULLY!"
echo " Plots & Animations saved in: ${PLOTS_DIR}"
echo " Manifest file: ${MANIFEST}"
echo "======================================================================"
