#!/usr/bin/env bash
# =============================================================================
# run_variable_box_subgrid.sh
#
# Variable-domain-size version of run_resume_from_snap500_32x16.sh.
#
# Given a physical box size (LX x LY, in pc):
#   1. Picks the CNN subgrid model matching CELL_SIZE (0.625 or 1.25 pc).
#   2. Builds a fresh full-resolution HR .athinput at a FIXED HR cell size
#      (0.0390625 pc, i.e. the cell size the CNN models were designed for),
#      with nx1/nx2 derived automatically from LX/LY so the physical cell
#      size never changes as the box grows or shrinks.
#   3. Runs that HR problem (iprob=1, standard KH instability) for 10 Myr.
#   4. Downsamples the 5 Myr snapshot to the CELL_SIZE coarse grid and uses
#      it as the initial condition (iprob=2) for a Subgrid (CNN) run and a
#      plain LR (ISM cooling) run, both restarted at t=5 Myr and evolved to
#      t=10 Myr.
#   5. Adjusts the CNN tile grid to fit the coarse domain exactly, and fails
#      fast if the box size cannot be tiled evenly.
#   6. Runs explore_data/mock_sg_tiled.py to produce the diagnostic plots.
#
# Usage:
#   LX=20 LY=40 CELL_SIZE=0.625 bash shell_scripts/run_variable_box_subgrid.sh [full_hr|subgrid|lr|both|all|postprocess]
#
# Environment variables:
#   LX, LY       - Physical box size in pc (default: 20, 40). Aspect ratio
#                  need not match the 1:2 baseline, but LX/LY must divide
#                  evenly by both the HR cell size and CELL_SIZE, and the
#                  resulting coarse grid must divide evenly by the CNN's
#                  native tile shape (see below).
#   CELL_SIZE    - Coarse/CNN grid cell size: 0.625 or 1.25 pc (default: 0.625)
#   HR_TLIM      - HR simulation time limit in Myr (default: 10.0)
#   RESTART_TIME_MYR - Time (Myr) at which the HR snapshot is taken as the
#                  Subgrid/LR initial condition (default: 5.0)
#   SG_LR_TLIM   - Subgrid/LR internal tlim, i.e. time evolved *after*
#                  restart (default: 5.05, so absolute end time is ~10 Myr)
#   NLIM         - Cycle limit (default: -1)
#   HR_FULL_EXE_DIR - Build dir for the full-resolution HR run
#                  (default: builds/hr_build_gpu)
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

# -----------------------------------------------------------------------------
# CUDA environment (needed for the full-resolution HR run on GPU)
# -----------------------------------------------------------------------------
for cuda_dir in /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4 /usr/local/cuda-12 /usr/local/cuda; do
    if [ -d "${cuda_dir}/bin" ]; then
        export CUDA_ROOT="${cuda_dir}"
        export CUDA_HOME="${cuda_dir}"
        export PATH="${cuda_dir}/bin:${PATH}"
        export LD_LIBRARY_PATH="${cuda_dir}/lib64:${LD_LIBRARY_PATH:-}"
        break
    fi
done
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

REF_ATHINPUT="${PROJECT_ROOT}/simulation_outputs/hr_gpu_512x1024/kh_radiative_512x1024.athinput"
if [ ! -f "${REF_ATHINPUT}" ]; then
    echo "ERROR: Reference athinput not found: ${REF_ATHINPUT}" >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# STEP 0: Physical box size & CNN model selection
# -----------------------------------------------------------------------------
LX="${LX:-10}"
LY="${LY:-20}"
CELL_SIZE="${CELL_SIZE:-0.625}"

if [ "${CELL_SIZE}" = "0.625" ]; then
    DS=16
    TILE_ROWS=16
    TILE_COLS=8
    DEFAULT_MODEL_SAVES="${PROJECT_ROOT}/runs/run_random_crop_20260909_164548/model_saves"
    DEFAULT_NORM_PREFIX="cnn_(512, 256)_32"
elif [ "${CELL_SIZE}" = "1.25" ] || [ "${CELL_SIZE}" = "1.26" ]; then
    DS=32
    TILE_ROWS=8
    TILE_COLS=4
    DEFAULT_MODEL_SAVES="${PROJECT_ROOT}/runs/run_random_crop_20260904_191402/model_saves"
    DEFAULT_NORM_PREFIX="cnn_(512, 256)_32"
else
    echo "ERROR: Unsupported CELL_SIZE: ${CELL_SIZE}. Expected 0.625 or 1.25." >&2
    exit 1
fi

MODEL_SAVES_DIR="${MODEL_SAVES_DIR:-${DEFAULT_MODEL_SAVES}}"
NORM_PREFIX="${NORM_PREFIX:-${DEFAULT_NORM_PREFIX}}"

# Fixed fine-grid cell size the CNN models were trained/designed against
# (CELL_SIZE / DS is the same 0.0390625 pc for both 0.625@DS16 and 1.25@DS32).
HR_CELL_SIZE="0.0390625"

# Baseline box (the original KH setup) used to scale the shear-layer width
# (a_char) and perturbation width (sigma) so the instability keeps the same
# relative shape when the box height changes. Both scale with the X2 extent,
# since a_char/sigma are applied along x2 in the problem generator.
BASELINE_LY="20.0"
BASELINE_A_CHAR="0.125"
BASELINE_SIGMA="0.5"

# -----------------------------------------------------------------------------
# STEP 0b: Derive grid resolutions, domain extents, meshblocks, and tile grid.
# Fails fast (before running any simulation) if the box size doesn't produce
# an integer HR/coarse resolution or an integer number of CNN tiles.
# -----------------------------------------------------------------------------
export LX_ENV="${LX}" LY_ENV="${LY}" CELL_SIZE_ENV="${CELL_SIZE}" DS_ENV="${DS}"
export HR_CELL_SIZE_ENV="${HR_CELL_SIZE}" BASELINE_LY_ENV="${BASELINE_LY}"
export BASELINE_A_CHAR_ENV="${BASELINE_A_CHAR}" BASELINE_SIGMA_ENV="${BASELINE_SIGMA}"
export TILE_ROWS_ENV="${TILE_ROWS}" TILE_COLS_ENV="${TILE_COLS}"

CONFIG_ASSIGNMENTS="$(python3 - <<'PY'
import os
import sys


def require_integer(value, description):
    rounded = round(value)
    if abs(value - rounded) > 1e-6 * max(1.0, abs(value)):
        sys.stderr.write(
            f"ERROR: {description} is not an integer number of cells "
            f"(got {value:.6f}). Choose a box length that divides evenly.\n"
        )
        sys.exit(1)
    return int(rounded)


def pick_divisor(n, max_val=128):
    """Largest divisor of n that is <= max_val (always succeeds; 1 divides everything)."""
    for d in range(min(max_val, n), 0, -1):
        if n % d == 0:
            return d
    return 1


lx = float(os.environ["LX_ENV"])
ly = float(os.environ["LY_ENV"])
cell_size = float(os.environ["CELL_SIZE_ENV"])
ds = int(os.environ["DS_ENV"])
hr_cell_size = float(os.environ["HR_CELL_SIZE_ENV"])
baseline_ly = float(os.environ["BASELINE_LY_ENV"])
baseline_a_char = float(os.environ["BASELINE_A_CHAR_ENV"])
baseline_sigma = float(os.environ["BASELINE_SIGMA_ENV"])
tile_rows = int(os.environ["TILE_ROWS_ENV"])
tile_cols = int(os.environ["TILE_COLS_ENV"])

hr_nx1 = require_integer(lx / hr_cell_size, f"HR nx1 (LX={lx} / HR cell size={hr_cell_size})")
hr_nx2 = require_integer(ly / hr_cell_size, f"HR nx2 (LY={ly} / HR cell size={hr_cell_size})")
nx1 = require_integer(lx / cell_size, f"Coarse nx1 (LX={lx} / CELL_SIZE={cell_size})")
nx2 = require_integer(ly / cell_size, f"Coarse nx2 (LY={ly} / CELL_SIZE={cell_size})")

if hr_nx1 != nx1 * ds or hr_nx2 != nx2 * ds:
    sys.stderr.write(
        f"ERROR: HR resolution ({hr_nx1}x{hr_nx2}) is not exactly DS={ds} times "
        f"the coarse resolution ({nx1}x{nx2}). This should not happen for "
        f"consistent LX/LY/CELL_SIZE inputs.\n"
    )
    sys.exit(1)

if nx2 % tile_rows != 0 or nx1 % tile_cols != 0:
    sys.stderr.write(
        f"ERROR: Coarse grid ({nx2} rows x {nx1} cols) cannot be tiled evenly "
        f"by the CNN's native tile shape ({tile_rows} rows x {tile_cols} cols) "
        f"for CELL_SIZE={cell_size}. Choose LX/LY so that "
        f"nx2={nx2} is a multiple of {tile_rows} and nx1={nx1} is a multiple of {tile_cols}.\n"
    )
    sys.exit(1)

n_tile_rows = nx2 // tile_rows
n_tile_cols = nx1 // tile_cols

mb_nx1 = pick_divisor(hr_nx1, 128)
mb_nx2 = pick_divisor(hr_nx2, 128)

x1min = -lx / 2.0
x1max = lx / 2.0
x2min = -ly / 2.0
x2max = ly / 2.0

scale = ly / baseline_ly
a_char = baseline_a_char * scale
sigma = baseline_sigma * scale

print(f"HR_NX1={hr_nx1}")
print(f"HR_NX2={hr_nx2}")
print(f"NX1={nx1}")
print(f"NX2={nx2}")
print(f"HR_MB_NX1={mb_nx1}")
print(f"HR_MB_NX2={mb_nx2}")
print(f"X1MIN={x1min:.6f}")
print(f"X1MAX={x1max:.6f}")
print(f"X2MIN={x2min:.6f}")
print(f"X2MAX={x2max:.6f}")
print(f"A_CHAR={a_char:.6f}")
print(f"SIGMA={sigma:.6f}")
print(f"N_TILE_ROWS={n_tile_rows}")
print(f"N_TILE_COLS={n_tile_cols}")
PY
)"
eval "${CONFIG_ASSIGNMENTS}"

TILE_GRID="${N_TILE_ROWS},${N_TILE_COLS}"

# -----------------------------------------------------------------------------
# CNN model configuration (exported for source_module.py and mock_sg_tiled.py)
# -----------------------------------------------------------------------------
export CELL_SIZE="${CELL_SIZE}"
export DS="${DS}"
export MODEL_SAVES_DIR="${MODEL_SAVES_DIR}"
export NORM_PREFIX="${NORM_PREFIX}"
export PDF_CNN_RESOLUTION="512,256"
export PDF_CNN_DOWNSAMPLE="32"
export CNN_TILING_MODE="tiled"
export TILE_GRID="${TILE_GRID}"
export TILE_ROWS="${TILE_ROWS}"
export TILE_COLS="${TILE_COLS}"
export LOGT_ACTIVE_START="4.1"
export LOGT_ACTIVE_END="5.9"
export COOL_TFLOOR="1.0e4"

# -----------------------------------------------------------------------------
# Simulation time controls
# -----------------------------------------------------------------------------
HR_TLIM="${HR_TLIM:-10.0}"
RESTART_TIME_MYR="${RESTART_TIME_MYR:-5.0}"
SG_LR_TLIM="${SG_LR_TLIM:-5.05}"
NLIM="${NLIM:--1}"
export RESTART_TIME_MYR

# Bin output cadence, read from the reference athinput's <output3> dt so the
# snapshot index used for restart always matches whatever the template says.
BIN_DT="$(python3 - <<PY
import re
text = open("${REF_ATHINPUT}").read()
m = re.search(r'<output3>.*?dt\s*=\s*([0-9.eE+-]+)', text, re.DOTALL)
print(m.group(1) if m else "0.01")
PY
)"
SNAP_IDX="$(python3 -c "print(f'{round(${RESTART_TIME_MYR} / ${BIN_DT}):05d}')")"

RUN_MODE="${1:-all}"  # full_hr, subgrid, lr, both, all, postprocess

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BOX_TAG="box_${LX}x${LY}"
RES_TAG="${NX2}x${NX1}"

HR_FULL_EXE_DIR="${HR_FULL_EXE_DIR:-${PROJECT_ROOT}/builds/hr_build_gpu}"
LR_EXE_DIR="${PROJECT_ROOT}/builds/hr_build"
SG_EXE_DIR="${PROJECT_ROOT}/builds/subgrid_model"

HR_FULL_OUTPUT_DIR="${PROJECT_ROOT}/simulation_outputs/hr_gpu_${HR_NX1}x${HR_NX2}_${BOX_TAG}"
HR_FULL_BIN_DIR="${HR_FULL_OUTPUT_DIR}/bin"
SNAP_BIN="${HR_FULL_BIN_DIR}/KH.hydro_u.${SNAP_IDX}.bin"

IC_DIR="${PROJECT_ROOT}/simulation_outputs/downsampled_ic"
IC_FILE="${IC_DIR}/ic_snap${SNAP_IDX}_${BOX_TAG}_${RES_TAG}.bin"

SG_OUTPUT_DIR="${PROJECT_ROOT}/simulation_outputs/subgrid_${BOX_TAG}_${RES_TAG}_from_snap${SNAP_IDX}"
LR_OUTPUT_DIR="${PROJECT_ROOT}/simulation_outputs/hr_build_${BOX_TAG}_${RES_TAG}_from_snap${SNAP_IDX}"

ATHINPUT_DIR="${PROJECT_ROOT}/simulation_outputs/athinputs_${BOX_TAG}_${RES_TAG}"
HR_FULL_ATHINPUT="${ATHINPUT_DIR}/kh_full_${HR_NX1}x${HR_NX2}.athinput"
SG_ATHINPUT="${ATHINPUT_DIR}/subgrid_${RES_TAG}.athinput"
LR_ATHINPUT="${ATHINPUT_DIR}/hr_build_${RES_TAG}.athinput"

mkdir -p "${IC_DIR}" "${ATHINPUT_DIR}" "${SG_OUTPUT_DIR}" "${LR_OUTPUT_DIR}" "${HR_FULL_OUTPUT_DIR}"

echo "======================================================================"
echo " VARIABLE-BOX SUBGRID PIPELINE"
echo " Date: $(date)"
echo " Project root: ${PROJECT_ROOT}"
echo " Box size: ${LX} x ${LY} pc  (X1 in [${X1MIN}, ${X1MAX}], X2 in [${X2MIN}, ${X2MAX}])"
echo " a_char=${A_CHAR}, sigma=${SIGMA}"
echo " HR resolution   : nx1=${HR_NX1}, nx2=${HR_NX2} (dx=${HR_CELL_SIZE} pc, meshblock ${HR_MB_NX1}x${HR_MB_NX2})"
echo " Coarse resolution: nx1=${NX1}, nx2=${NX2} (dx=${CELL_SIZE} pc, DS=${DS})"
echo " CNN tile grid   : ${TILE_GRID} tiles of ${TILE_ROWS}x${TILE_COLS} cells"
echo " Model saves dir : ${MODEL_SAVES_DIR}"
echo " Restart snapshot: index ${SNAP_IDX} (t=${RESTART_TIME_MYR} Myr)"
echo " Mode            : ${RUN_MODE}"
echo "======================================================================"

# -----------------------------------------------------------------------------
# STEP 1: Generate .athinput files (full-res HR, coarse Subgrid, coarse LR)
# -----------------------------------------------------------------------------
echo "[Step 1] Generating .athinput files ..."

python3 - <<PY
import re, pathlib

ref_path = pathlib.Path("${REF_ATHINPUT}")
hr_full_path = pathlib.Path("${HR_FULL_ATHINPUT}")
sg_path = pathlib.Path("${SG_ATHINPUT}")
lr_path = pathlib.Path("${LR_ATHINPUT}")

hr_nx1, hr_nx2 = "${HR_NX1}", "${HR_NX2}"
hr_mb_nx1, hr_mb_nx2 = "${HR_MB_NX1}", "${HR_MB_NX2}"
nx1, nx2 = "${NX1}", "${NX2}"
x1min, x1max = "${X1MIN}", "${X1MAX}"
x2min, x2max = "${X2MIN}", "${X2MAX}"
a_char, sigma = "${A_CHAR}", "${SIGMA}"
hr_tlim = "${HR_TLIM}"
sg_lr_tlim = "${SG_LR_TLIM}"
nlim_val = "${NLIM}"
ic_file = "${IC_FILE}"

content = ref_path.read_text()


def walk_sections(lines):
    section = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('<') and stripped.endswith('>'):
            section = stripped[1:-1]
        yield section, line


def patch_full_hr(text):
    """Fresh full-resolution HR run: same iprob=1 KH setup as the reference,
    just re-sized to the new box (nx1/nx2, extents, meshblock, tlim, a_char/sigma)."""
    out = []
    for section, line in walk_sections(text.splitlines()):
        if section == 'mesh':
            if re.match(r'^\s*nx1\s*=', line):
                line = f"nx1       = {hr_nx1}        # Number of zones in X1-direction"
            elif re.match(r'^\s*nx2\s*=', line):
                line = f"nx2       = {hr_nx2}        # Number of zones in X2-direction"
            elif re.match(r'^\s*x1min\s*=', line):
                line = f"x1min     = {x1min}        # minimum value of X1"
            elif re.match(r'^\s*x1max\s*=', line):
                line = f"x1max     = {x1max}        # maximum value of X1"
            elif re.match(r'^\s*x2min\s*=', line):
                line = f"x2min     = {x2min}        # minimum value of X2"
            elif re.match(r'^\s*x2max\s*=', line):
                line = f"x2max     = {x2max}        # maximum value of X2"
        elif section == 'meshblock':
            if re.match(r'^\s*nx1\s*=', line):
                line = f"nx1       = {hr_mb_nx1}        # Number of cells in each MeshBlock, X1-dir"
            elif re.match(r'^\s*nx2\s*=', line):
                line = f"nx2       = {hr_mb_nx2}        # Number of cells in each MeshBlock, X2-dir"
        elif section == 'time':
            if re.match(r'^\s*tlim\s*=', line):
                line = f"tlim       = {hr_tlim}        # time limit"
            elif re.match(r'^\s*nlim\s*=', line):
                line = f"nlim       = {nlim_val}         # cycle limit"
        elif section == 'problem':
            if re.match(r'^\s*a_char\s*=', line):
                line = f"a_char    = {a_char}        # width of tanh profile"
            elif re.match(r'^\s*sigma\s*=', line):
                line = f"sigma     = {sigma}        # width of gaussian profile"
        out.append(line)
    return '\n'.join(out) + '\n'


def patch_coarse(text, is_subgrid):
    """Coarse Subgrid/LR run initialized from the downsampled HR snapshot
    (iprob=2): re-sized to the box, single-meshblock coarse grid, restarted
    at RESTART_TIME_MYR and evolved for SG_LR_TLIM internal time."""
    out = []
    for section, line in walk_sections(text.splitlines()):
        if section == 'mesh':
            if re.match(r'^\s*nx1\s*=', line):
                line = f"nx1       = {nx1}        # Number of zones in X1-direction"
            elif re.match(r'^\s*nx2\s*=', line):
                line = f"nx2       = {nx2}        # Number of zones in X2-direction"
            elif re.match(r'^\s*x1min\s*=', line):
                line = f"x1min     = {x1min}        # minimum value of X1"
            elif re.match(r'^\s*x1max\s*=', line):
                line = f"x1max     = {x1max}        # maximum value of X1"
            elif re.match(r'^\s*x2min\s*=', line):
                line = f"x2min     = {x2min}        # minimum value of X2"
            elif re.match(r'^\s*x2max\s*=', line):
                line = f"x2max     = {x2max}        # maximum value of X2"
        elif section == 'meshblock':
            if re.match(r'^\s*nx1\s*=', line):
                line = f"nx1       = {nx1}        # Number of cells in each MeshBlock, X1-dir"
            elif re.match(r'^\s*nx2\s*=', line):
                line = f"nx2       = {nx2}        # Number of cells in each MeshBlock, X2-dir"
        elif section == 'time':
            if re.match(r'^\s*tlim\s*=', line):
                line = f"tlim       = {sg_lr_tlim}        # time limit"
            elif re.match(r'^\s*nlim\s*=', line):
                line = f"nlim       = {nlim_val}         # cycle limit"
        elif section == 'hydro':
            if re.match(r'^\s*nscalars\s*=', line):
                line = "nscalars    = 2         # number of passive scalars in hydro (tracer + cold frac)"
        elif section == 'hydro_srcterms':
            if re.match(r'^\s*ism_cooling\s*=', line):
                line = "ism_cooling = false" if is_subgrid else "ism_cooling = true"
        elif section == 'problem':
            if re.match(r'^\s*a_char\s*=', line):
                line = f"a_char    = {a_char}        # width of tanh profile"
            elif re.match(r'^\s*sigma\s*=', line):
                line = f"sigma     = {sigma}        # width of gaussian profile"
        out.append(line)
        # subgrid model reports its own cooling-limited dt via psrc->dtnew
        # (set directly in UserSourceTerm()), so tell SourceTerms to trust it
        # instead of resetting dtnew to float_max every cycle.
        if section == 'hydro_srcterms' and is_subgrid and re.match(r'^\s*ism_cooling\s*=', line):
            out.append("user_cooling = true")

    # Insert iprob=2 / init_file / user_srcs right after <problem>, and drop
    # the original iprob=1 line further down.
    final_lines = []
    for line in out:
        final_lines.append(line)
        if line.strip() == '<problem>':
            final_lines.append("iprob     = 2         # initialize from binary file")
            final_lines.append(f"init_file = {ic_file}")
            final_lines.append("user_srcs = true" if is_subgrid else "user_srcs = false")

    clean_lines = []
    seen_iprob = False
    for line in final_lines:
        if re.match(r'^\s*iprob\s*=', line):
            if seen_iprob:
                continue
            seen_iprob = True
        clean_lines.append(line)

    return '\n'.join(clean_lines) + '\n'


hr_full_path.write_text(patch_full_hr(content))
sg_path.write_text(patch_coarse(content, is_subgrid=True))
lr_path.write_text(patch_coarse(content, is_subgrid=False))
print(f"  Created: {hr_full_path}")
print(f"  Created: {sg_path}")
print(f"  Created: {lr_path}")
PY

# -----------------------------------------------------------------------------
# STEP 2: Run the fresh full-resolution HR simulation for HR_TLIM Myr
# -----------------------------------------------------------------------------
if [ "${RUN_MODE}" = "full_hr" ] || [ "${RUN_MODE}" = "all" ]; then
    echo ""
    echo "======================================================================"
    echo "[Step 2] Evolving full-resolution HR (${HR_NX1}x${HR_NX2}, tlim=${HR_TLIM} Myr)"
    echo "======================================================================"
    HR_FULL_EXE="${HR_FULL_EXE_DIR}/src/athena"
    if [ ! -x "${HR_FULL_EXE}" ]; then
        echo "ERROR: HR executable not found or not executable: ${HR_FULL_EXE}" >&2
        exit 1
    fi

    (
        cd "${HR_FULL_EXE_DIR}/src"
        "${HR_FULL_EXE}" -i "${HR_FULL_ATHINPUT}" -d "${HR_FULL_OUTPUT_DIR}"
    )
    echo "Full-resolution HR evolution completed. Outputs in: ${HR_FULL_OUTPUT_DIR}"
fi

# -----------------------------------------------------------------------------
# STEP 3: Downsample the restart snapshot, then evolve Subgrid and/or LR
# -----------------------------------------------------------------------------
if [ "${RUN_MODE}" = "subgrid" ] || [ "${RUN_MODE}" = "lr" ] || [ "${RUN_MODE}" = "both" ] || [ "${RUN_MODE}" = "all" ]; then
    if [ ! -f "${SNAP_BIN}" ]; then
        echo "ERROR: Restart snapshot not found at: ${SNAP_BIN}" >&2
        echo "       Run with mode 'full_hr' or 'all' first to produce it." >&2
        exit 1
    fi

    echo ""
    echo "======================================================================"
    echo "[Step 3] Downsampling snapshot ${SNAP_IDX} -> ${IC_FILE} ..."
    echo "======================================================================"
    python3 "${PROJECT_ROOT}/data/downsample_ic.py" \
        --input "${SNAP_BIN}" \
        --nx1 "${NX1}" \
        --nx2 "${NX2}" \
        --output "${IC_FILE}"

    if [ "${RUN_MODE}" = "subgrid" ] || [ "${RUN_MODE}" = "both" ] || [ "${RUN_MODE}" = "all" ]; then
        echo ""
        echo "======================================================================"
        echo "[Step 4] Evolving builds/subgrid_model (Tiled CNN Subgrid Model)"
        echo "======================================================================"
        SG_EXE="${SG_EXE_DIR}/src/athena"
        if [ ! -f "${SG_EXE}" ]; then
            echo "ERROR: Athena executable not found at: ${SG_EXE}" >&2
            exit 1
        fi

        # Ensure base conda python environment and site-packages are exposed to embedded Python
        CONDA_PY="/home/sasi/miniconda3/bin/python3"
        export PYTHONHOME="$("${CONDA_PY}" -c "import sys; print(sys.prefix)")"
        PY_SYS_PATH="$("${CONDA_PY}" -c "import sys; print(':'.join(p for p in sys.path if p))")"
        export PYTHONPATH="${SG_EXE_DIR}/src:${PROJECT_ROOT}:${PROJECT_ROOT}/models/conv_nn:${PY_SYS_PATH}:${PYTHONPATH:-}"
        export LD_LIBRARY_PATH="/home/sasi/miniconda3/lib:${LD_LIBRARY_PATH:-}"

        rm -rf "${SG_OUTPUT_DIR}"/*
        (
            cd "${SG_EXE_DIR}/src"
            "${SG_EXE}" -i "${SG_ATHINPUT}" -d "${SG_OUTPUT_DIR}"
        )
        echo "Subgrid model evolution completed. Outputs in: ${SG_OUTPUT_DIR}"
    fi

    if [ "${RUN_MODE}" = "lr" ] || [ "${RUN_MODE}" = "both" ] || [ "${RUN_MODE}" = "all" ]; then
        echo ""
        echo "======================================================================"
        echo "[Step 5] Evolving builds/hr_build (Standard ISM Cooling, coarse LR)"
        echo "======================================================================"
        LR_EXE="${LR_EXE_DIR}/src/athena"
        if [ ! -f "${LR_EXE}" ]; then
            echo "ERROR: Athena executable not found at: ${LR_EXE}" >&2
            exit 1
        fi

        rm -rf "${LR_OUTPUT_DIR}"/*
        (
            cd "${LR_EXE_DIR}/src"
            "${LR_EXE}" -i "${LR_ATHINPUT}" -d "${LR_OUTPUT_DIR}"
        )
        echo "LR evolution completed. Outputs in: ${LR_OUTPUT_DIR}"
    fi
fi

# -----------------------------------------------------------------------------
# STEP 6: Generate Comparison Plots & Diagnostics (mock_sg_tiled.py)
# -----------------------------------------------------------------------------
if [ "${RUN_MODE}" != "full_hr" ]; then
    TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
    PLOTS_DIR="${PROJECT_ROOT}/tiled_outputs/run_tiled_${BOX_TAG}_${RES_TAG}_${TIMESTAMP}"
    mkdir -p "${PLOTS_DIR}"

    MANIFEST="${PLOTS_DIR}/manifest.txt"
    {
        echo "======================================================================"
        echo " Variable-Box Subgrid Pipeline Run Manifest"
        echo "======================================================================"
        echo "Timestamp                : ${TIMESTAMP} ($(date))"
        echo "Script                   : ${BASH_SOURCE[0]}"
        echo "Project root             : ${PROJECT_ROOT}"
        echo "Execution mode           : ${RUN_MODE}"
        echo ""
        echo "--- Box Geometry ---"
        echo "Box length X1 (width)    : ${LX} pc"
        echo "Box length X2 (height)   : ${LY} pc"
        echo "Domain extents X1        : [${X1MIN}, ${X1MAX}] pc"
        echo "Domain extents X2        : [${X2MIN}, ${X2MAX}] pc"
        echo "Shear layer width a_char : ${A_CHAR} pc"
        echo "Perturbation sigma       : ${SIGMA} pc"
        echo ""
        echo "--- High Resolution (HR) ---"
        echo "HR grid                  : nx1=${HR_NX1} (X1), nx2=${HR_NX2} (X2)"
        echo "HR cell size             : ${HR_CELL_SIZE} pc"
        echo "HR meshblock             : ${HR_MB_NX1} x ${HR_MB_NX2}"
        echo "HR athinput              : ${HR_FULL_ATHINPUT}"
        echo "HR output directory      : ${HR_FULL_OUTPUT_DIR}"
        echo "HR executable            : ${HR_FULL_EXE_DIR}/src/athena"
        echo "HR time limit            : ${HR_TLIM} Myr"
        echo ""
        echo "--- Coarse (Subgrid / LR) ---"
        echo "Coarse cell size (CNN)   : ${CELL_SIZE} pc"
        echo "Downsample factor (DS)   : ${DS}"
        echo "Coarse grid              : nx1=${NX1} (X1), nx2=${NX2} (X2)"
        echo "Restart snapshot         : index ${SNAP_IDX} (t=${RESTART_TIME_MYR} Myr)"
        echo "Downsampled IC binary    : ${IC_FILE}"
        echo "Subgrid athinput         : ${SG_ATHINPUT}"
        echo "Subgrid output directory : ${SG_OUTPUT_DIR}"
        echo "LR athinput              : ${LR_ATHINPUT}"
        echo "LR output directory      : ${LR_OUTPUT_DIR}"
        echo "Subgrid/LR internal tlim : ${SG_LR_TLIM} Myr (absolute end ~$(python3 -c "print(${RESTART_TIME_MYR} + ${SG_LR_TLIM})") Myr)"
        echo "Cycle limit (nlim)       : ${NLIM}"
        echo ""
        echo "--- CNN Model & Tiling ---"
        echo "Model saves dir          : ${MODEL_SAVES_DIR}"
        echo "Norm prefix              : ${NORM_PREFIX}"
        echo "Tile grid (rows,cols)    : ${TILE_GRID}"
        echo "Tile shape (cells)       : ${TILE_ROWS} rows x ${TILE_COLS} cols"
        echo "Active log10(T) range    : [${LOGT_ACTIVE_START}, ${LOGT_ACTIVE_END}]"
        echo "Cooling floor temp       : ${COOL_TFLOOR} K"
        echo "======================================================================"
    } > "${MANIFEST}"

    echo ""
    echo "======================================================================"
    echo "[Step 6] Generating Full Diagnostic Suite & Animations via mock_sg_tiled.py"
    echo " Plots directory: ${PLOTS_DIR}"
    echo " Manifest saved : ${MANIFEST}"
    echo "======================================================================"

    python3 "${PROJECT_ROOT}/explore_data/mock_sg_tiled.py" \
        --output-dir "${PLOTS_DIR}" \
        --hr-athinput "${HR_FULL_ATHINPUT}" \
        --hr-bin "${HR_FULL_BIN_DIR}" \
        --sg-athinput "${SG_ATHINPUT}" \
        --sg-bin "${SG_OUTPUT_DIR}/bin" \
        --lr-athinput "${LR_ATHINPUT}" \
        --lr-bin "${LR_OUTPUT_DIR}/bin"

    echo ""
    echo "======================================================================"
    echo " ALL TASKS COMPLETED SUCCESSFULLY!"
    echo " Plots & Animations saved in: ${PLOTS_DIR}"
    echo " Manifest file: ${MANIFEST}"
    echo "======================================================================"
else
    echo ""
    echo "Mode 'full_hr' complete. Re-run with mode 'subgrid', 'lr', 'both', or 'all' to continue."
fi
