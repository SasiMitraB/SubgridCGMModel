#!/usr/bin/env bash
# =============================================================================
# run_plots.sh — Re-run mock_sg.py diagnostic plots for the latest pipeline run
#
# Usage:
#   sh shell_scripts/run_plots.sh            # uses the most recent run_* directory
#   sh shell_scripts/run_plots.sh run_20260623_134844   # use a specific run
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/Volumes/PortableSSD/Projects/SubgridCGMModel"

# ---------------------------------------------------------------------------
# Resolve run directory (latest by default, or explicit arg)
# ---------------------------------------------------------------------------
if [[ $# -ge 1 ]]; then
    RUN_DIR="${PROJECT_ROOT}/runs/$1"
    if [[ ! -d "${RUN_DIR}" ]]; then
        echo "ERROR: Run directory not found: ${RUN_DIR}" >&2
        exit 1
    fi
else
    RUN_DIR="$(ls -dt "${PROJECT_ROOT}/runs/run_"* 2>/dev/null | head -1)"
    if [[ -z "${RUN_DIR}" ]]; then
        echo "ERROR: No run_* directories found under ${PROJECT_ROOT}/runs/" >&2
        exit 1
    fi
fi

echo "Using run directory: ${RUN_DIR}"

# ---------------------------------------------------------------------------
# Export the SG_MOCKS_DIR that mock_sg.py reads to decide where to save plots
# ---------------------------------------------------------------------------
export SG_MOCKS_DIR="${RUN_DIR}/sg_mocks"
mkdir -p "${SG_MOCKS_DIR}"

# ---------------------------------------------------------------------------
# Activate venv and set PYTHONPATH
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

# ---------------------------------------------------------------------------
# Run mock_sg.py from data/mocks/ so relative imports resolve correctly
# ---------------------------------------------------------------------------
LOG_FILE="${RUN_DIR}/logs/step5_diagnostic_plots.log"
mkdir -p "${RUN_DIR}/logs"

echo "Saving plots to : ${SG_MOCKS_DIR}"
echo "Log             : ${LOG_FILE}"
echo ""

cd "${PROJECT_ROOT}/data/mocks"
python3 mock_sg.py 2>&1 | tee "${LOG_FILE}"

echo ""
echo "Done. Plots saved to: ${SG_MOCKS_DIR}"
