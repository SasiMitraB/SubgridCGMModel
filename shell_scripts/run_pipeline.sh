#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — Full SubgridCGM pipeline
#
# Steps:
#   1. Train the PDF CNN model              (models/conv_nn/pdf_cnn.py)
#   2. Benchmark PDF CNN model              (data/mocks/pdf_plot.py)
#   3. Low-resolution simulation 5 Myr      (16×8 grid; ISM cooling)
#      Outputs to: simulation_outputs/lr_build
#   4. lr_build — restart from 5 Myr rst   (hr_build/src/athena; ISM cooling)
#      Uses:  simulation_outputs/lr_build/rst/KH.00005.rst
#      Outputs to: simulation_outputs/lr_build_ism
#   5. subgrid_model — restart from same rst (subgrid_model/src/athena; CNN)
#      Uses:  simulation_outputs/lr_build/rst/KH.00005.rst
#      Outputs to: simulation_outputs/subgrid_model
#   6. Diagnostic plots                     (data/mocks/mock_sg.py)
#
# RESTART FILE POLICY
# -----------------------------------
# Steps 3 and 4 both branch from the SAME 5 Myr restart file produced
# by Step 2:
#   simulation_outputs/lr_build/rst/KH.00005.rst
#
# The restart file index 00005 corresponds to tlim=5.0 with rst_dt=1.0
# (one restart file written per simulated Myr).
#
# ATHINPUT CACHING
# -----------------------------------
# Every athinput actually used during the run is copied into:
#   runs/run_<timestamp>/athinputs/
#
# LOGS
# -----------------------------------
# All stdout/stderr for every step is tee-d into:
#   runs/run_<timestamp>/logs/<step_N_name>.log
# A master pipeline log is written to:
#   runs/run_<timestamp>/pipeline.log
#
# EXIT POLICY
# -----------------------------------
# The script exits immediately on any non-zero return (set -e).
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 0.  Project root and paths
# ---------------------------------------------------------------------------
PROJECT_ROOT="/home/sasi/Projects/SubgridCGMModel"

# ---- Canonical HR output (training data source; not re-run here) ----
HR_SIM_OUTPUT="${PROJECT_ROOT}/simulation_outputs/hr_build_1024"
HR_BIN_DIR="${HR_SIM_OUTPUT}/bin"

# ---- LR 5 Myr base simulation (Step 2) ----
LR_OUTPUT_DIR="${PROJECT_ROOT}/simulation_outputs/lr_build"
LR_RST_5MYR="${LR_OUTPUT_DIR}/rst/KH.00005.rst"

# ---- lr_build restart (Step 3) — ISM cooling from 5 Myr ----
LR_BUILD_OUTPUT_DIR="${PROJECT_ROOT}/simulation_outputs/lr_build_ism"

# ---- subgrid_model restart (Step 4) — CNN from 5 Myr ----
SG_OUTPUT_DIR="${PROJECT_ROOT}/simulation_outputs/subgrid_model"

# ---- Per-run timestamped directory ----
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="${PROJECT_ROOT}/runs/run_${TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
ATHINPUT_CACHE_DIR="${RUN_DIR}/athinputs"   # all athinputs used this run

export MODEL_SAVES_DIR="${RUN_DIR}/model_saves"
export LOSS_PLOTS_DIR="${RUN_DIR}/loss_plots"
export PDF_MOCKS_DIR="${RUN_DIR}/pdf_mocks"
export SG_MOCKS_DIR="${RUN_DIR}/sg_mocks"

mkdir -p \
    "${LOG_DIR}" \
    "${ATHINPUT_CACHE_DIR}" \
    "${MODEL_SAVES_DIR}" \
    "${LOSS_PLOTS_DIR}" \
    "${PDF_MOCKS_DIR}" \
    "${SG_MOCKS_DIR}" \
    "${LR_OUTPUT_DIR}" \
    "${LR_BUILD_OUTPUT_DIR}" \
    "${SG_OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
MASTER_LOG="${RUN_DIR}/pipeline.log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "${msg}" | tee -a "${MASTER_LOG}"
}

separator() {
    log "$(printf '=%.0s' {1..72})"
}

# ---------------------------------------------------------------------------
# Helper: run a named step, tee stdout+stderr to a per-step log file.
# Usage:  run_step  <N>  <label>  <cmd> [args…]
# ---------------------------------------------------------------------------
run_step() {
    local step_num="$1"
    local step_name="$2"
    shift 2
    local log_file="${LOG_DIR}/step${step_num}_${step_name}.log"

    separator
    log "STEP ${step_num}: ${step_name}"
    log "Command  : $*"
    log "Log file : ${log_file}"
    separator

    # Pipe through tee while preserving the command's exit code via pipefail
    if "$@" 2>&1 | tee "${log_file}"; then
        log "STEP ${step_num} COMPLETED OK: ${step_name}"
    else
        log "STEP ${step_num} FAILED: ${step_name} — see ${log_file}"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Helper: generate an athinput file from config.json
# ---------------------------------------------------------------------------
CONFIG_JSON="${PROJECT_ROOT}/shell_scripts/config.json"
GEN_ATHINPUT="${PROJECT_ROOT}/shell_scripts/gen_athinput.py"

generate_athinput() {
    local step="$1"
    local output="$2"
    log "Generating ${step} athinput -> ${output}"
    python3 "${GEN_ATHINPUT}" --config "${CONFIG_JSON}" --step "${step}" --output "${output}"
}

# ---------------------------------------------------------------------------
# Activate project virtual environment
# ---------------------------------------------------------------------------
VENV_ACTIVATE="${PROJECT_ROOT}/venv/bin/activate"
if [[ -f "${VENV_ACTIVATE}" ]]; then
    # shellcheck source=/dev/null
    source "${VENV_ACTIVATE}"
    log "Activated venv : ${VENV_ACTIVATE}"
else
    log "WARNING: venv not found at ${VENV_ACTIVATE} — using system Python"
fi

# PYTHONPATH so all in-repo modules are importable
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/data:${PYTHONPATH:-}"

# Environment variables for pdf_cnn.py's data/cache paths
export SUBGRID_DATA_PATH="${HR_BIN_DIR}"
export SUBGRID_CACHE_PATH="${HR_SIM_OUTPUT}/cache"

# Derive CNN resolution/downsample from config.json (single source of truth)
read -r PDF_CNN_RESOLUTION PDF_CNN_DOWNSAMPLE < <(python3 - <<PY
import json
c = json.load(open("${CONFIG_JSON}"))
hr = c["hr"]
ds = c["lr"]["downsample_factor"]
print(f"{hr['nx2']},{hr['nx1']} {ds}")
PY
)
export PDF_CNN_RESOLUTION PDF_CNN_DOWNSAMPLE

# Active cooling window log10(T) bounds (default: 4.2 to 6.0)
export LOGT_ACTIVE_START="${LOGT_ACTIVE_START:-4.2}"
export LOGT_ACTIVE_END="${LOGT_ACTIVE_END:-6.0}"

# ---------------------------------------------------------------------------
# Write a manifest of all key paths for this run
# ---------------------------------------------------------------------------
MANIFEST="${RUN_DIR}/manifest.txt"
{
    echo "============================================================"
    echo " SubgridCGM Pipeline Run"
    echo "============================================================"
    echo "Timestamp          : ${TIMESTAMP}"
    echo "Run directory      : ${RUN_DIR}"
    echo "Project root       : ${PROJECT_ROOT}"
    echo ""
    echo "--- Active Window Bounds ---"
    echo "LOGT_ACTIVE_START  : ${LOGT_ACTIVE_START}"
    echo "LOGT_ACTIVE_END    : ${LOGT_ACTIVE_END}"
    echo ""
    echo "--- Source scripts / configs ---"
    echo "Config JSON        : ${CONFIG_JSON}"
    echo "LR athinput (gen)  : ${ATHINPUT_CACHE_DIR}/lr_sim.athinput"
    echo "lr_build athinput  : ${ATHINPUT_CACHE_DIR}/lr_build_sim.athinput"
    echo "SG athinput (gen)  : ${ATHINPUT_CACHE_DIR}/sg_sim.athinput"
    echo "CNN trainer        : ${PROJECT_ROOT}/models/conv_nn/pdf_cnn.py"
    echo "PDF benchmark      : ${PROJECT_ROOT}/data/mocks/pdf_plot.py"
    echo "Diagnostic plots   : ${PROJECT_ROOT}/data/mocks/mock_sg.py"
    echo ""
    echo "--- Simulation outputs ---"
    echo "LR sim (0→5 Myr)   : ${LR_OUTPUT_DIR}"
    echo "lr_build (5→10 Myr): ${LR_BUILD_OUTPUT_DIR}  [ISM cooling restart]"
    echo "subgrid_model      : ${SG_OUTPUT_DIR}         [CNN restart]"
    echo "5 Myr restart file : ${LR_RST_5MYR}"
    echo ""
    echo "--- Model / plot outputs ---"
    echo "Model weights      : ${MODEL_SAVES_DIR}"
    echo "Loss plots         : ${LOSS_PLOTS_DIR}"
    echo "PDF mock outputs   : ${PDF_MOCKS_DIR}"
    echo "SG  mock outputs   : ${SG_MOCKS_DIR}"
    echo ""
    echo "--- Cached athinputs (this run) ---"
    echo "  ${ATHINPUT_CACHE_DIR}/lr_sim.athinput"
    echo "  ${ATHINPUT_CACHE_DIR}/lr_build_sim.athinput"
    echo "  ${ATHINPUT_CACHE_DIR}/sg_sim.athinput"
} > "${MANIFEST}"

log "Run directory  : ${RUN_DIR}"
log "Manifest       : ${MANIFEST}"
separator

# ===========================================================================
# STEP 1 — Train the PDF CNN
# ===========================================================================
separator
log "STEP 1: train_pdf_cnn"
log "Training data  : ${HR_BIN_DIR}"
separator

run_step 1 "train_pdf_cnn" \
    python3 "${PROJECT_ROOT}/models/conv_nn/pdf_cnn.py"

# ===========================================================================
# STEP 2 — Benchmark PDF CNN model (pdf_plot.py)
#
# Runs benchmarking and metric calculations on trained PDF CNN model.
# Output plots and animations land in PDF_MOCKS_DIR.
# ===========================================================================
separator
log "STEP 2: benchmark_pdf_cnn  (pdf_plot.py)"
log "Benchmarking script : ${PROJECT_ROOT}/data/mocks/pdf_plot.py"
separator

run_step 2 "benchmark_pdf_cnn" \
    bash -c "cd '${PROJECT_ROOT}/data/mocks' && python3 pdf_plot.py"

# ===========================================================================
# STEP 3 — Low-resolution simulation: 0 → 5 Myr  (16×8 grid, ISM cooling)
#
# Grid:  nx1=8, nx2=16  (16×8 cells — 32× downsampled from HR 256×512)
# tlim:  5.0  (5 Myr)
# rst_dt: 1.0 → restart files written at t=1,2,3,4,5 Myr
#
# Output restart files land in:
#   simulation_outputs/lr_build/rst/KH.000{01..05}.rst
# The 5 Myr file KH.00005.rst is the branch point for Steps 4 & 5.
# ===========================================================================
LR_ATHINPUT="${ATHINPUT_CACHE_DIR}/lr_sim.athinput"
generate_athinput "lr" "${LR_ATHINPUT}"

separator
log "STEP 3: lr_simulation  (16×8 grid, 0 → 5 Myr)"
log "LR athinput mesh settings:"
grep -E '^\s*nx[12]\s*=' "${LR_ATHINPUT}" | tee -a "${MASTER_LOG}"
log "LR athinput tlim:"
grep -E '^\s*tlim\s*=' "${LR_ATHINPUT}" | tee -a "${MASTER_LOG}"
log "LR athinput press and mu settings:"
grep -E '^\s*(press|mu)\s*=' "${LR_ATHINPUT}" | tee -a "${MASTER_LOG}"
separator

run_step 3 "lr_simulation_5myr" \
    bash -c "
        set -euo pipefail
        cd '${PROJECT_ROOT}/builds/hr_build/src'
        ./athena -i '${LR_ATHINPUT}' -d '${LR_OUTPUT_DIR}'
    "

# Verify the 5 Myr restart file was produced
if [[ ! -f "${LR_RST_5MYR}" ]]; then
    log "ERROR: Expected 5 Myr restart file not found: ${LR_RST_5MYR}"
    log "       Check the LR simulation output in ${LR_OUTPUT_DIR}/rst/"
    exit 1
fi
log "5 Myr restart file confirmed: ${LR_RST_5MYR}"

# ===========================================================================
# STEP 4 — lr_build: restart from 5 Myr with ISM cooling (no CNN)
#
# Uses:    hr_build/src/athena   (ISM-cooling build, no neural-network source)
# Restart: simulation_outputs/lr_build/rst/KH.00005.rst  (t = 5 Myr)
# tlim:    10.0  (continues from 5 → 10 Myr)
# Output:  simulation_outputs/lr_build_ism/
# ===========================================================================
LR_BUILD_ATHINPUT="${ATHINPUT_CACHE_DIR}/lr_build_sim.athinput"
generate_athinput "lr_build" "${LR_BUILD_ATHINPUT}"

separator
log "STEP 4: lr_build  (ISM cooling restart from ${LR_RST_5MYR})"
log "lr_build athinput mesh settings:"
grep -E '^\s*nx[12]\s*=' "${LR_BUILD_ATHINPUT}" | tee -a "${MASTER_LOG}"
log "lr_build athinput tlim:"
grep -E '^\s*tlim\s*=' "${LR_BUILD_ATHINPUT}" | tee -a "${MASTER_LOG}"
log "lr_build athinput press and mu settings:"
grep -E '^\s*(press|mu)\s*=' "${LR_BUILD_ATHINPUT}" | tee -a "${MASTER_LOG}"
separator

run_step 4 "lr_build_ism_restart" \
    bash -c "
        set -euo pipefail
        cd '${PROJECT_ROOT}/builds/hr_build/src'
        ./athena \
            -i '${LR_BUILD_ATHINPUT}' \
            -d '${LR_BUILD_OUTPUT_DIR}' \
            -r '${LR_RST_5MYR}'
    "

# ===========================================================================
# STEP 5 — subgrid_model: restart from same 5 Myr rst with CNN source terms
#
# Uses:    subgrid_model/src/athena  (CNN-enabled build)
# Restart: simulation_outputs/lr_build/rst/KH.00005.rst  (t = 5 Myr)
# tlim:    10.0  (continues from 5 → 10 Myr)
# Output:  simulation_outputs/subgrid_model/
#
# The subgrid_model build requires PYTHONPATH to find source_module.py.
# ===========================================================================
SG_ATHINPUT="${ATHINPUT_CACHE_DIR}/sg_sim.athinput"
generate_athinput "sg" "${SG_ATHINPUT}"

separator
log "STEP 5: subgrid_model  (CNN restart from ${LR_RST_5MYR})"
log "SG athinput mesh settings:"
grep -E '^\s*nx[12]\s*=' "${SG_ATHINPUT}" | tee -a "${MASTER_LOG}"
log "SG athinput tlim:"
grep -E '^\s*tlim\s*=' "${SG_ATHINPUT}" | tee -a "${MASTER_LOG}"
log "SG athinput press and mu settings:"
grep -E '^\s*(press|mu)\s*=' "${SG_ATHINPUT}" | tee -a "${MASTER_LOG}"
separator

run_step 5 "subgrid_model_cnn_restart" \
    bash -c "
        set -euo pipefail
        cd '${PROJECT_ROOT}/builds/subgrid_model/src'

        # Activate venv and set PYTHONPATH for the embedded Python source module
        source '${VENV_ACTIVATE}'
        VENV='${PROJECT_ROOT}/venv'
        SITE_PACKAGES=\"\$VENV/lib/python3.14/site-packages\"
        export PYTHONPATH=\"\$PWD:\$SITE_PACKAGES\${PYTHONPATH:+:\$PYTHONPATH}\"

        ./athena \
            -i '${SG_ATHINPUT}' \
            -d '${SG_OUTPUT_DIR}' \
            -r '${LR_RST_5MYR}'
    "

# ===========================================================================
# STEP 6 — Diagnostic plots (mock_sg.py)
#
# Compares lr_build (ISM) and subgrid_model (CNN) simulation outputs.
# Run from data/mocks/ so relative output paths resolve correctly.
# ===========================================================================
separator
log "STEP 6: diagnostic_plots  (mock_sg.py)"
separator

run_step 6 "diagnostic_plots" \
    bash -c "cd '${PROJECT_ROOT}/data/mocks' && python3 mock_sg.py"

# ===========================================================================
# Done — summary
# ===========================================================================
separator
log "ALL PIPELINE STEPS COMPLETED SUCCESSFULLY"
separator
log ""
log "Run directory    : ${RUN_DIR}"
log "Master log       : ${MASTER_LOG}"
log "Manifest         : ${MANIFEST}"
log ""
log "Cached athinputs (this run):"
log "  LR (0→5 Myr)    → ${ATHINPUT_CACHE_DIR}/lr_sim.athinput"
log "  lr_build (ISM)  → ${ATHINPUT_CACHE_DIR}/lr_build_sim.athinput"
log "  subgrid_model   → ${ATHINPUT_CACHE_DIR}/sg_sim.athinput"
log ""
log "Step logs:"
for f in "${LOG_DIR}"/step*.log; do
    log "  ${f}"
done
log ""
log "Key output directories:"
log "  LR sim (0→5 Myr)      : ${LR_OUTPUT_DIR}"
log "  lr_build (ISM restart) : ${LR_BUILD_OUTPUT_DIR}"
log "  subgrid_model (CNN)    : ${SG_OUTPUT_DIR}"
log "  Model weights          : ${MODEL_SAVES_DIR}"
log "  PDF mock               : ${PDF_MOCKS_DIR}"
log "  SG mock                : ${SG_MOCKS_DIR}"
separator
