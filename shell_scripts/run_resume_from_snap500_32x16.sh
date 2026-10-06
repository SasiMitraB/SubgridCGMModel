#!/usr/bin/env bash
# =============================================================================
# run_resume_from_snap500_32x16.sh
#
# Initializes a 32x16 simulation using the 500th snapshot from:
#   /home/sasi/Projects/SubgridCGMModel/simulation_outputs/hr_gpu_512x1024
#
# 1. Downsamples snapshot 500 to 32x16 resolution (nx1=16, nx2=32)
#    preserving length scales: x1 in [-5.0, 5.0], x2 in [-20.0, 20.0].
# 2. Runs builds/subgrid_model with tiled CNN inference (batching 4 tiles of 16x8).
# 3. Runs builds/hr_build with standard ISM cooling.
# 4. Executes explore_data/mock_sg_tiled.py to produce all diagnostic plots
#    and animations into a run_tiled32x16_<timestamp> directory.
#
# Usage:
#   bash shell_scripts/run_resume_from_snap500_32x16.sh [subgrid|hr|both]
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

# Snapshots and Output Paths
SNAP500_BIN="${PROJECT_ROOT}/simulation_outputs/hr_gpu_512x1024/bin/KH.hydro_u.00500.bin"

# -----------------------------------------------------------------------------
# Configuration: Grid Cell Size Option
# -----------------------------------------------------------------------------
# Option 1: CELL_SIZE="0.625" (0.625 pc -> DS=16, grid: 64x32, nx1=32, nx2=64)
CELL_SIZE="${CELL_SIZE:-0.625}"

if [ "${CELL_SIZE}" = "0.625" ]; then
    DS=16
    NX1=32
    NX2=64
    DEFAULT_MODEL_SAVES="${PROJECT_ROOT}/runs/run_random_crop_20260909_164548/model_saves"
    DEFAULT_NORM_PREFIX="cnn_(512, 256)_32"
    TILE_GRID="4,4"
    TILE_ROWS="16"
    TILE_COLS="8"
elif [ "${CELL_SIZE}" = "1.25" ] || [ "${CELL_SIZE}" = "1.26" ]; then
    DS=32
    NX1=16
    NX2=32
    DEFAULT_MODEL_SAVES="${PROJECT_ROOT}/runs/run_random_crop_20260904_191402/model_saves"
    DEFAULT_NORM_PREFIX="cnn_(512, 256)_32"
    TILE_GRID="4,4"
    TILE_ROWS="8"
    TILE_COLS="4"
else
    echo "ERROR: Unsupported CELL_SIZE: ${CELL_SIZE}. Expected 0.625 or 1.25." >&2
    exit 1
fi

IC_DIR="${PROJECT_ROOT}/simulation_outputs/downsampled_ic"
IC_FILE="${IC_DIR}/ic_snap500_${NX2}x${NX1}.bin"

SG_OUTPUT_DIR="${PROJECT_ROOT}/simulation_outputs/subgrid_${NX2}x${NX1}_from_snap500"
HR_OUTPUT_DIR="${PROJECT_ROOT}/simulation_outputs/hr_build_${NX2}x${NX1}_from_snap500"

ATHINPUT_DIR="${PROJECT_ROOT}/simulation_outputs/athinputs_${NX2}x${NX1}"
SG_ATHINPUT="${ATHINPUT_DIR}/subgrid_${NX2}x${NX1}.athinput"
HR_ATHINPUT="${ATHINPUT_DIR}/hr_build_${NX2}x${NX1}.athinput"

REF_ATHINPUT="${PROJECT_ROOT}/simulation_outputs/hr_gpu_512x1024/kh_radiative_512x1024.athinput"

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

# Simulation controls (can be overridden via environment variables)
# HR runs 0-10 Myr; SG/LR restart from t=5 and run for 5 Myr (internal time) to reach t=10 Myr (absolute)
HR_TLIM="${HR_TLIM:-10.0}"
SG_LR_TLIM="${SG_LR_TLIM:-5.05}"  # Run for ~5 Myr internal time from restart
NLIM="${NLIM:--1}"     # Cycle limit (-1 for unlimited)

RUN_MODE="${1:-both}"  # subgrid, hr, both, or postprocess

mkdir -p "${IC_DIR}" "${ATHINPUT_DIR}" "${SG_OUTPUT_DIR}" "${HR_OUTPUT_DIR}"

echo "======================================================================"
echo " RESUME FROM SNAPSHOT 500 (CELL SIZE: ${CELL_SIZE} pc, ${NX2}x${NX1} RESOLUTION)"
echo " Date: $(date)"
echo " Project root: ${PROJECT_ROOT}"
echo " Source snapshot: ${SNAP500_BIN}"
echo " Cell size: ${CELL_SIZE} pc (DS=${DS})"
echo " Target coarse resolution: nx1=${NX1} (width), nx2=${NX2} (height)"
echo " Model saves dir: ${MODEL_SAVES_DIR}"
echo " Norm prefix: ${NORM_PREFIX}"
echo " Mode: ${RUN_MODE}"
echo "======================================================================"

# -----------------------------------------------------------------------------
# STEP 1: Downsample snapshot 500 to 32x16 binary IC
# -----------------------------------------------------------------------------
if [ ! -f "${SNAP500_BIN}" ]; then
    echo "ERROR: Snapshot 500 not found at: ${SNAP500_BIN}" >&2
    exit 1
fi

echo "[Step 1] Downsampling snapshot 500 -> ${IC_FILE} ..."
python3 "${PROJECT_ROOT}/data/downsample_ic.py" \
    --input "${SNAP500_BIN}" \
    --nx1 "${NX1}" \
    --nx2 "${NX2}" \
    --output "${IC_FILE}"

# -----------------------------------------------------------------------------
# STEP 2: Generate .athinput files for subgrid_model and hr_build
# -----------------------------------------------------------------------------
echo "[Step 2] Generating .athinput files ..."

python3 - <<PY
import re, pathlib

ref_path = pathlib.Path("${REF_ATHINPUT}")
sg_path  = pathlib.Path("${SG_ATHINPUT}")
hr_path  = pathlib.Path("${HR_ATHINPUT}")
ic_file  = "${IC_FILE}"
nx1_val  = "${NX1}"
nx2_val  = "${NX2}"
hr_tlim_val = "${HR_TLIM}"
sg_lr_tlim_val = "${SG_LR_TLIM}"
nlim_val = "${NLIM}"

content = ref_path.read_text()

def patch_athinput(text, is_subgrid=False, is_hr_build=False):
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
            # Use different tlim for HR vs SG/LR (HR runs 0-10, SG/LR run 0-5 internal)
            tlim_to_use = hr_tlim_val if is_hr_build else sg_lr_tlim_val
            line = f"tlim       = {tlim_to_use}        # time limit"
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
        # subgrid model reports its own cooling-limited dt via psrc->dtnew
        # (set directly in UserSourceTerm()), so tell SourceTerms to trust it
        # instead of resetting dtnew to float_max every cycle.
        if in_hydro_src and is_subgrid and re.match(r'^\s*ism_cooling\s*=', line):
            new_lines.append("user_cooling = true")

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

    # Remove original iprob line if present lower in problem block
    clean_lines = []
    seen_iprob = False
    for line in final_lines:
        if re.match(r'^\s*iprob\s*=', line):
            if seen_iprob:
                continue
            seen_iprob = True
        clean_lines.append(line)

    return '\n'.join(clean_lines) + '\n'

sg_path.write_text(patch_athinput(content, is_subgrid=True, is_hr_build=False))
hr_path.write_text(patch_athinput(content, is_subgrid=False, is_hr_build=False))  # hr_path is actually LR (hr_build)
print(f"  Created: {sg_path}")
print(f"  Created: {hr_path}")
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
# STEP 4: Evolve with hr_build (ISM cooling)
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

    echo "Running Athena hr_build..."
    rm -rf "${HR_OUTPUT_DIR}"/*
    (
        cd "${PROJECT_ROOT}/builds/hr_build/src"
        "${HR_EXE}" -i "${HR_ATHINPUT}" -d "${HR_OUTPUT_DIR}"
    )
    echo "hr_build evolution completed. Outputs in: ${HR_OUTPUT_DIR}"
fi

# -----------------------------------------------------------------------------
# STEP 5: Generate Comparison Plots & Diagnostics (mock_sg_tiled.py)
# -----------------------------------------------------------------------------
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
PLOTS_DIR="${PROJECT_ROOT}/tiled_outputs/run_tiled${NX2}x${NX1}_${TIMESTAMP}"
mkdir -p "${PLOTS_DIR}"

# -----------------------------------------------------------------------------
# Write Run Manifest
# -----------------------------------------------------------------------------
MANIFEST="${PLOTS_DIR}/manifest.txt"
{
    echo "======================================================================"
    echo " SubgridCGM Tiled Resume from Snap500 Run Manifest"
    echo "======================================================================"
    echo "Timestamp                : ${TIMESTAMP} ($(date))"
    echo "Script                   : ${BASH_SOURCE[0]}"
    echo "Project root             : ${PROJECT_ROOT}"
    echo "Output / Plots dir       : ${PLOTS_DIR}"
    echo "Execution mode           : ${RUN_MODE}"
    echo ""
    echo "--- Grid & Resolution ---"
    echo "Cell size                : ${CELL_SIZE} pc"
    echo "Downsample factor (DS)   : ${DS}"
    echo "Coarse grid dimensions   : nx1=${NX1} (width/X1), nx2=${NX2} (height/X2)"
    echo "Domain extents           : x1 in [-5.0, 5.0] pc, x2 in [-10.0, 10.0] pc"
    echo "Reference HR grid        : 512x1024 (nx1=512, nx2=1024, res=0.0390625 pc)"
    echo ""
    echo "--- Model & Tiling Setup ---"
    echo "Model saves dir          : ${MODEL_SAVES_DIR}"
    echo "Norm prefix              : ${NORM_PREFIX}"
    echo "Model weights file       : ${MODEL_SAVES_DIR}/${NORM_PREFIX}.pth"
    echo "Model mean file          : ${MODEL_SAVES_DIR}/${NORM_PREFIX}_input_mean.npy"
    echo "Model std file           : ${MODEL_SAVES_DIR}/${NORM_PREFIX}_input_std.npy"
    echo "Tiling mode              : ${CNN_TILING_MODE}"
    echo "Tile grid                : ${TILE_GRID}"
    echo "Tile size                : ${TILE_ROWS} rows x ${TILE_COLS} cols"
    echo "PDF CNN resolution       : ${PDF_CNN_RESOLUTION}"
    echo "PDF CNN downsample       : ${PDF_CNN_DOWNSAMPLE}"
    echo "Active log10(T) range    : [${LOGT_ACTIVE_START}, ${LOGT_ACTIVE_END}]"
    echo ""
    echo "--- Simulation & Initial Condition Setup ---"
    echo "Source HR snapshot 500   : ${SNAP500_BIN}"
    echo "Downsampled IC binary    : ${IC_FILE}"
    echo "Reference athinput       : ${REF_ATHINPUT}"
    echo "Subgrid athinput         : ${SG_ATHINPUT}"
    echo "HR/LR build athinput     : ${HR_ATHINPUT}"
    echo "Subgrid output dir       : ${SG_OUTPUT_DIR}"
    echo "HR/LR build output dir   : ${HR_OUTPUT_DIR}"
    echo "HR time limit (tlim)     : ${HR_TLIM} Myr (absolute: 0-10 Myr)"
    echo "SG/LR time limit (tlim)  : ${SG_LR_TLIM} Myr (internal: 0-5 Myr, physical: 5-10 Myr after restart)"
    echo "Cycle limit (nlim)       : ${NLIM}"
    echo "Subgrid executable       : ${PROJECT_ROOT}/builds/subgrid_model/src/athena"
    echo "HR build executable      : ${PROJECT_ROOT}/builds/hr_build/src/athena"
    echo "======================================================================"
} > "${MANIFEST}"

echo ""
echo "======================================================================"
echo "[Step 5] Generating Full Diagnostic Suite & Animations via mock_sg_tiled.py"
echo " Plots directory: ${PLOTS_DIR}"
echo " Manifest saved : ${MANIFEST}"
echo "======================================================================"

python3 "${PROJECT_ROOT}/explore_data/mock_sg_tiled.py" \
    --output-dir "${PLOTS_DIR}" \
    --hr-athinput "${REF_ATHINPUT}" \
    --hr-bin "${PROJECT_ROOT}/simulation_outputs/hr_gpu_512x1024/bin" \
    --sg-athinput "${SG_ATHINPUT}" \
    --sg-bin "${SG_OUTPUT_DIR}/bin" \
    --lr-athinput "${HR_ATHINPUT}" \
    --lr-bin "${HR_OUTPUT_DIR}/bin"

echo ""
echo "======================================================================"
echo " ALL TASKS COMPLETED SUCCESSFULLY!"
echo " Plots & Animations saved in: ${PLOTS_DIR}"
echo " Manifest file: ${MANIFEST}"
echo "======================================================================"
