#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — Full SubgridCGM pipeline (Downsampled IC Mode)
#
# Steps:
#   1. Train the PDF CNN model              (models/conv_nn/pdf_cnn.py)
#   2. Benchmark PDF CNN model              (data/mocks/pdf_plot.py)
#   3. Downsample HR snapshot to coarse IC  (data/downsample_ic.py)
#   4. Low-resolution simulation (ISM)      (hr_build/src/athena; ISM cooling)
#      Starts from: downsampled IC file (iprob=2)
#      Outputs to:  simulation_outputs/lr_build_ism
#   5. subgrid_model simulation (CNN)       (subgrid_model/src/athena; CNN)
#      Starts from: same downsampled IC file (iprob=2)
#      Outputs to:  simulation_outputs/subgrid_model
#   6. Diagnostic plots                     (data/mocks/mock_sg.py)
#
# INITIAL CONDITION POLICY
# -----------------------------------
# Both LR (ISM cooling) and Subgrid (CNN) start directly from the SAME
# downsampled high-resolution snapshot (iprob=2, init_file).
# No 0→5 Myr pre-run or restart file is used.
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
HR_SIM_OUTPUT="${PROJECT_ROOT}/simulation_outputs/hr_build_512"
HR_BIN_DIR="${HR_SIM_OUTPUT}/bin"

# ---- High-Resolution Snapshot for Initial Condition Downsampling ----
HR_IC_SNAPSHOT="${HR_IC_SNAPSHOT:-${PROJECT_ROOT}/simulation_outputs/hr_build_512/bin/KH.hydro_w.00500.bin}"

# ---- Target Low-Resolution Simulation Grid (nx1=width, nx2=height) ----
export SIM_NX1="${SIM_NX1:-8}"
export SIM_NX2="${SIM_NX2:-16}"
export SIM_MB_NX1="${SIM_MB_NX1:-${SIM_NX1}}"
export SIM_MB_NX2="${SIM_MB_NX2:-${SIM_NX2}}"
export SIM_TLIM="${SIM_TLIM:-5.0}"
export RESTART_TIME_MYR="${RESTART_TIME_MYR:-5.0}"

# ---- Output Directories ----
LR_BUILD_OUTPUT_DIR="${PROJECT_ROOT}/simulation_outputs/lr_build_ism"
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

# Path where the downsampled initial condition binary will be stored
DOWNSAMPLED_IC_FILE="${DOWNSAMPLED_IC_FILE:-${RUN_DIR}/ic_downsampled_${SIM_NX1}x${SIM_NX2}.bin}"

mkdir -p \
    "${LOG_DIR}" \
    "${ATHINPUT_CACHE_DIR}" \
    "${MODEL_SAVES_DIR}" \
    "${LOSS_PLOTS_DIR}" \
    "${PDF_MOCKS_DIR}" \
    "${SG_MOCKS_DIR}" \
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
# Helper: generate an athinput file from config.json with IC override
# ---------------------------------------------------------------------------
CONFIG_JSON="${PROJECT_ROOT}/shell_scripts/config.json"
GEN_ATHINPUT="${PROJECT_ROOT}/shell_scripts/gen_athinput.py"

generate_athinput() {
    local step="$1"
    local output="$2"
    log "Generating ${step} athinput -> ${output}"
    python3 "${GEN_ATHINPUT}" \
        --config "${CONFIG_JSON}" \
        --step "${step}" \
        --output "${output}" \
        --nx1 "${SIM_NX1}" \
        --nx2 "${SIM_NX2}" \
        --mb_nx1 "${SIM_MB_NX1}" \
        --mb_nx2 "${SIM_MB_NX2}" \
        --iprob 2 \
        --init_file "${DOWNSAMPLED_IC_FILE}" \
        --tlim "${SIM_TLIM}"
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

# Active cooling window log10(T) bounds (default: 4.1 to 5.9)
export LOGT_ACTIVE_START="${LOGT_ACTIVE_START:-4.1}"
export LOGT_ACTIVE_END="${LOGT_ACTIVE_END:-5.9}"

# ---------------------------------------------------------------------------
# Write a manifest of all key paths for this run
# ---------------------------------------------------------------------------
MANIFEST="${RUN_DIR}/manifest.txt"
{
    echo "============================================================"
    echo " SubgridCGM Pipeline Run (Downsampled IC Mode)"
    echo "============================================================"
    echo "Timestamp          : ${TIMESTAMP}"
    echo "Run directory      : ${RUN_DIR}"
    echo "Project root       : ${PROJECT_ROOT}"
    echo ""
    echo "--- Grid & Initial Condition ---"
    echo "HR IC Snapshot     : ${HR_IC_SNAPSHOT}"
    echo "Downsampled IC File: ${DOWNSAMPLED_IC_FILE}"
    echo "Grid Nx1 x Nx2     : ${SIM_NX1} x ${SIM_NX2}"
    echo "Simulation tlim    : ${SIM_TLIM} Myr"
    echo ""
    echo "--- Active Window Bounds ---"
    echo "LOGT_ACTIVE_START  : ${LOGT_ACTIVE_START}"
    echo "LOGT_ACTIVE_END    : ${LOGT_ACTIVE_END}"
    echo ""
    echo "--- Source scripts / configs ---"
    echo "Config JSON        : ${CONFIG_JSON}"
    echo "lr_build athinput  : ${ATHINPUT_CACHE_DIR}/lr_build_sim.athinput"
    echo "SG athinput (gen)  : ${ATHINPUT_CACHE_DIR}/sg_sim.athinput"
    echo "CNN trainer        : ${PROJECT_ROOT}/models/conv_nn/pdf_cnn.py"
    echo "PDF benchmark      : ${PROJECT_ROOT}/data/mocks/pdf_plot.py"
    echo "IC Downsampler     : ${PROJECT_ROOT}/data/downsample_ic.py"
    echo "Diagnostic plots   : ${PROJECT_ROOT}/data/mocks/mock_sg.py"
    echo ""
    echo "--- Simulation outputs ---"
    echo "lr_build (ISM)     : ${LR_BUILD_OUTPUT_DIR}"
    echo "subgrid_model (CNN): ${SG_OUTPUT_DIR}"
    echo ""
    echo "--- Model / plot outputs ---"
    echo "Model weights      : ${MODEL_SAVES_DIR}"
    echo "Loss plots         : ${LOSS_PLOTS_DIR}"
    echo "PDF mock outputs   : ${PDF_MOCKS_DIR}"
    echo "SG  mock outputs   : ${SG_MOCKS_DIR}"
    echo ""
    echo "--- Cached athinputs (this run) ---"
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
# STEP 3 — Downsample HR Snapshot to Coarse Initial Condition File
# ===========================================================================
separator
log "STEP 3: downsample_ic  (${SIM_NX1}x${SIM_NX2} from ${HR_IC_SNAPSHOT})"
log "Output IC file : ${DOWNSAMPLED_IC_FILE}"
separator

run_step 3 "downsample_ic" \
    python3 "${PROJECT_ROOT}/data/downsample_ic.py" \
        --input "${HR_IC_SNAPSHOT}" \
        --nx1 "${SIM_NX1}" \
        --nx2 "${SIM_NX2}" \
        --output "${DOWNSAMPLED_IC_FILE}"

# ===========================================================================
# STEP 4 — lr_build: Low-Resolution Simulation with ISM cooling (no CNN)
#
# Starts directly from downsampled initial condition (iprob=2, init_file).
# Uses:    hr_build/src/athena
# Output:  simulation_outputs/lr_build_ism/
# ===========================================================================
LR_BUILD_ATHINPUT="${ATHINPUT_CACHE_DIR}/lr_build_sim.athinput"
generate_athinput "lr_build" "${LR_BUILD_ATHINPUT}"

separator
log "STEP 4: lr_build_ism  (ISM cooling, starting from ${DOWNSAMPLED_IC_FILE})"
log "lr_build athinput mesh settings:"
grep -E '^\s*nx[12]\s*=' "${LR_BUILD_ATHINPUT}" | tee -a "${MASTER_LOG}"
log "lr_build athinput problem settings:"
grep -E '^\s*(iprob|init_file)\s*=' "${LR_BUILD_ATHINPUT}" | tee -a "${MASTER_LOG}"
log "lr_build athinput tlim:"
grep -E '^\s*tlim\s*=' "${LR_BUILD_ATHINPUT}" | tee -a "${MASTER_LOG}"
separator

# Clean previous simulation outputs if any
rm -rf "${LR_BUILD_OUTPUT_DIR}"/*

run_step 4 "lr_build_ism" \
    bash -c "
        set -euo pipefail
        cd '${PROJECT_ROOT}/builds/hr_build/src'
        ./athena \
            -i '${LR_BUILD_ATHINPUT}' \
            -d '${LR_BUILD_OUTPUT_DIR}'
    "

# ===========================================================================
# STEP 5 — subgrid_model: Simulation with CNN source terms
#
# Starts directly from SAME downsampled initial condition (iprob=2, init_file).
# Uses:    subgrid_model/src/athena  (CNN-enabled build)
# Output:  simulation_outputs/subgrid_model/
# ===========================================================================
SG_ATHINPUT="${ATHINPUT_CACHE_DIR}/sg_sim.athinput"
generate_athinput "sg" "${SG_ATHINPUT}"

separator
log "STEP 5: subgrid_model_cnn  (CNN subgrid, starting from ${DOWNSAMPLED_IC_FILE})"
log "SG athinput mesh settings:"
grep -E '^\s*nx[12]\s*=' "${SG_ATHINPUT}" | tee -a "${MASTER_LOG}"
log "SG athinput problem settings:"
grep -E '^\s*(iprob|init_file)\s*=' "${SG_ATHINPUT}" | tee -a "${MASTER_LOG}"
log "SG athinput tlim:"
grep -E '^\s*tlim\s*=' "${SG_ATHINPUT}" | tee -a "${MASTER_LOG}"
separator

# Clean previous simulation outputs if any
rm -rf "${SG_OUTPUT_DIR}"/*

run_step 5 "subgrid_model_cnn" \
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
            -d '${SG_OUTPUT_DIR}'
    "

# ===========================================================================
# STEP 6 — Diagnostic plots (mock_sg.py)
#
# Compares lr_build_ism and subgrid_model simulation outputs from t=0.
# Run from data/mocks/ so relative output paths resolve correctly.
# ===========================================================================
separator
log "STEP 6: diagnostic_plots  (mock_sg.py)"
separator

run_step 6 "diagnostic_plots" \
    bash -c "
        export START_FRAME=0
        export RESTART_TIME_MYR='${RESTART_TIME_MYR}'
        export SIM_NX1='${SIM_NX1}'
        export SIM_NX2='${SIM_NX2}'
        cd '${PROJECT_ROOT}/data/mocks' && python3 mock_sg.py
    "

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
log "Initial condition used:"
log "  Source snapshot: ${HR_IC_SNAPSHOT}"
log "  Downsampled IC : ${DOWNSAMPLED_IC_FILE} (${SIM_NX1}x${SIM_NX2})"
log ""
log "Cached athinputs (this run):"
log "  lr_build (ISM)  → ${ATHINPUT_CACHE_DIR}/lr_build_sim.athinput"
log "  subgrid_model   → ${ATHINPUT_CACHE_DIR}/sg_sim.athinput"
log ""
log "Step logs:"
for f in "${LOG_DIR}"/step*.log; do
    log "  ${f}"
done
log ""
log "Key output directories:"
log "  lr_build (ISM)         : ${LR_BUILD_OUTPUT_DIR}"
log "  subgrid_model (CNN)    : ${SG_OUTPUT_DIR}"
log "  Model weights          : ${MODEL_SAVES_DIR}"
log "  PDF mock               : ${PDF_MOCKS_DIR}"
log "  SG mock                : ${SG_MOCKS_DIR}"
separator
