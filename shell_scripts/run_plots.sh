#!/usr/bin/env bash
# =============================================================================
# run_plots.sh — Re-run diagnostic plots (mock_sg.py & pdf_plot.py) for a run
# =============================================================================
#
# Usage:
#   bash shell_scripts/run_plots.sh                         # uses the most recent run_* directory
#   bash shell_scripts/run_plots.sh run_random_crop_20260827_214817   # uses a specific run name
#   bash shell_scripts/run_plots.sh /full/path/to/run_dir             # uses a specific run path
#
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/home/sasi/Projects/SubgridCGMModel"

# ---------------------------------------------------------------------------
# 1. Resolve run directory (latest by default, or explicit argument)
# ---------------------------------------------------------------------------
if [[ $# -ge 1 ]]; then
    if [[ -d "$1" ]]; then
        RUN_DIR="$1"
    elif [[ -d "${PROJECT_ROOT}/runs/$1" ]]; then
        RUN_DIR="${PROJECT_ROOT}/runs/$1"
    else
        echo "ERROR: Run directory not found: $1" >&2
        exit 1
    fi
else
    # Find the most recently modified run directory in runs/
    RUN_DIR=$(python3 -c "import glob, os; runs = sorted(glob.glob('${PROJECT_ROOT}/runs/run_*'), key=os.path.getmtime, reverse=True); print(runs[0] if runs else '')")
    if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
        echo "ERROR: No run_* directories found under ${PROJECT_ROOT}/runs/" >&2
        exit 1
    fi
fi

echo "========================================================================"
echo "RUNNING DIAGNOSTIC PLOTS"
echo "========================================================================"
echo "Using run directory: ${RUN_DIR}"

# ---------------------------------------------------------------------------
# 2. Parse configuration from manifest.txt if available
# ---------------------------------------------------------------------------
MANIFEST="${RUN_DIR}/manifest.txt"

if [[ -f "${MANIFEST}" ]]; then
    echo "Found manifest: ${MANIFEST}"
    
    # Extract resolution and downsample info if present
    MANIFEST_RES=$(grep -E '^\s*Resolution \(fine\)\s*:' "${MANIFEST}" | head -1 | sed -E 's/.*:\s*([0-9]+,[0-9]+).*/\1/' || true)
    MANIFEST_DS=$(grep -E '^\s*Downsample factor\s*:' "${MANIFEST}" | head -1 | sed -E 's/.*:\s*([0-9]+).*/\1/' || true)
    MANIFEST_EVAL_REF=$(grep -E '^\s*Eval reference\s*:' "${MANIFEST}" | head -1 | awk -F': ' '{print $2}' || true)
    MANIFEST_EVAL_RES=$(grep -E '^\s*Eval resolution\s*:' "${MANIFEST}" | head -1 | sed -E 's/.*:\s*([0-9]+,[0-9]+).*/\1/' || true)
    MANIFEST_EVAL_DS=$(grep -E '^\s*Eval resolution\s*:' "${MANIFEST}" | head -1 | sed -E 's/.*ds=([0-9]+).*/\1/' || true)
    MANIFEST_SIM_NX2=$(grep -E '^\s*Athena simulation\s*:' "${MANIFEST}" | head -1 | sed -E 's/.*nx2=([0-9]+).*/\1/' || true)
    MANIFEST_SIM_NX1=$(grep -E '^\s*Athena simulation\s*:' "${MANIFEST}" | head -1 | sed -E 's/.*nx1=([0-9]+).*/\1/' || true)
    
    export PDF_CNN_RESOLUTION="${PDF_CNN_RESOLUTION:-${MANIFEST_RES:-2048,1024}}"
    export PDF_CNN_DOWNSAMPLE="${PDF_CNN_DOWNSAMPLE:-${MANIFEST_DS:-32}}"
    export HR_SIM_OUTPUT="${HR_SIM_OUTPUT:-${MANIFEST_EVAL_REF:-${PROJECT_ROOT}/simulation_outputs/hr_build_512}}"
    export HR_EVAL_RESOLUTION="${HR_EVAL_RESOLUTION:-${MANIFEST_EVAL_RES:-512,256}}"
    export HR_EVAL_DOWNSAMPLE="${HR_EVAL_DOWNSAMPLE:-${MANIFEST_EVAL_DS:-32}}"
    export SIM_NX2="${SIM_NX2:-${MANIFEST_SIM_NX2:-32}}"
    export SIM_NX1="${SIM_NX1:-${MANIFEST_SIM_NX1:-16}}"
    export SG_RESOLUTION="${SG_RESOLUTION:-${SIM_NX2},${SIM_NX1}}"
else
    echo "Notice: manifest.txt not found. Using environment or default parameters."
    export PDF_CNN_RESOLUTION="${PDF_CNN_RESOLUTION:-2048,1024}"
    export PDF_CNN_DOWNSAMPLE="${PDF_CNN_DOWNSAMPLE:-32}"
    export HR_SIM_OUTPUT="${HR_SIM_OUTPUT:-${PROJECT_ROOT}/simulation_outputs/hr_build_512}"
    export HR_EVAL_RESOLUTION="${HR_EVAL_RESOLUTION:-512,256}"
    export HR_EVAL_DOWNSAMPLE="${HR_EVAL_DOWNSAMPLE:-32}"
    export SIM_NX2="${SIM_NX2:-32}"
    export SIM_NX1="${SIM_NX1:-16}"
    export SG_RESOLUTION="${SG_RESOLUTION:-32,16}"
fi

export SUBGRID_DATA_PATH="${HR_SIM_OUTPUT}/bin"
export SUBGRID_CACHE_PATH="${HR_SIM_OUTPUT}/cache"

# ---------------------------------------------------------------------------
# 3. Export mock output directories
# ---------------------------------------------------------------------------
export SG_MOCKS_DIR="${RUN_DIR}/sg_mocks"
export PDF_MOCKS_DIR="${RUN_DIR}/pdf_mocks"
export MODEL_SAVES_DIR="${RUN_DIR}/model_saves"
export LOSS_PLOTS_DIR="${RUN_DIR}/loss_plots"

mkdir -p "${SG_MOCKS_DIR}" "${PDF_MOCKS_DIR}" "${RUN_DIR}/logs"

# ---------------------------------------------------------------------------
# 4. Activate venv and set PYTHONPATH
# ---------------------------------------------------------------------------
VENV_ACTIVATE="${PROJECT_ROOT}/venv/bin/activate"
if [[ -f "${VENV_ACTIVATE}" ]]; then
    # shellcheck source=/dev/null
    source "${VENV_ACTIVATE}"
    echo "Activated venv: ${VENV_ACTIVATE}"
else
    echo "WARNING: venv not found at ${VENV_ACTIVATE} — using system Python"
fi

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/data:${PYTHONPATH:-}"

echo "Configuration:"
echo "  PDF_CNN_RESOLUTION : ${PDF_CNN_RESOLUTION}"
echo "  PDF_CNN_DOWNSAMPLE : ${PDF_CNN_DOWNSAMPLE}"
echo "  HR_SIM_OUTPUT      : ${HR_SIM_OUTPUT}"
echo "  HR_EVAL_RESOLUTION : ${HR_EVAL_RESOLUTION}"
echo "  SG_RESOLUTION      : ${SG_RESOLUTION}"
echo "  MODEL_SAVES_DIR    : ${MODEL_SAVES_DIR}"
echo "  SG_MOCKS_DIR       : ${SG_MOCKS_DIR}"
echo "  PDF_MOCKS_DIR      : ${PDF_MOCKS_DIR}"
echo "========================================================================"
echo ""

# ---------------------------------------------------------------------------
# 5. Run diagnostic plot scripts
# ---------------------------------------------------------------------------
LOG_FILE="${RUN_DIR}/logs/step6_diagnostic_plots.log"

cd "${PROJECT_ROOT}/data/mocks"

echo "--- 1. Running mock_sg.py (Diagnostic comparisons) ---" | tee "${LOG_FILE}"
python3 mock_sg.py 2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "--- 2. Running pdf_plot.py (PDF CNN benchmark) ---" | tee -a "${LOG_FILE}"
python3 pdf_plot.py 2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "========================================================================"
echo "Plots completed successfully!"
echo "Outputs saved to:"
echo "  - SG Mocks  : ${SG_MOCKS_DIR}"
echo "  - PDF Mocks : ${PDF_MOCKS_DIR}"
echo "========================================================================"
