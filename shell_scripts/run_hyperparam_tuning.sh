#!/usr/bin/env bash
# =============================================================================
# run_hyperparam_tuning.sh — SubgridCGM Hyperparameter Tuning & Pipeline
#
# Steps:
#   1. Base low-resolution simulation 0 → 5 Myr (16×8 grid; produces restart rst)
#   2. Generate Athinput configurations
#   3. Closed-loop Optuna Bayesian optimization (optimize_subgrid.py):
#      - Tunes alpha_emiss, alpha_gate, alpha_mean_temp
#      - Evaluates downstream Athena simulation physical error metrics
#      - Saves best model weights to MODEL_SAVES_DIR
#   4. Benchmark best PDF CNN model (data/mocks/pdf_plot.py)
#   5. lr_build — restart from 5 Myr rst (ISM cooling 5 → 10 Myr)
#   6. subgrid_model — restart from 5 Myr rst with best CNN model
#   7. Diagnostic plots & comparison (data/mocks/mock_sg.py)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 0.  Project root and paths
# ---------------------------------------------------------------------------
PROJECT_ROOT="/home/sasi/Projects/SubgridCGMModel"

# ---- Number of Optuna optimization trials ----
N_OPTUNA_TRIALS="${N_OPTUNA_TRIALS:-60}"
EPOCHS_PER_TRIAL="${EPOCHS_PER_TRIAL:-500}"

# ---- Canonical HR output (training data source; not re-run here) ----
HR_SIM_OUTPUT="${PROJECT_ROOT}/simulation_outputs/hr_build_1024"
HR_BIN_DIR="${HR_SIM_OUTPUT}/bin"

# ---- LR 5 Myr base simulation ----
LR_OUTPUT_DIR="${PROJECT_ROOT}/simulation_outputs/lr_build"
LR_RST_5MYR="${LR_OUTPUT_DIR}/rst/KH.00005.rst"

# ---- lr_build restart — ISM cooling from 5 Myr ----
LR_BUILD_OUTPUT_DIR="${PROJECT_ROOT}/simulation_outputs/lr_build_ism"

# ---- subgrid_model restart — CNN from 5 Myr ----
SG_OUTPUT_DIR="${PROJECT_ROOT}/simulation_outputs/subgrid_model"

# ---- Per-run timestamped directory ----
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="${PROJECT_ROOT}/runs/run_optuna_${TIMESTAMP}"
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

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/data:${PYTHONPATH:-}"

# Environment variables for pdf_cnn.py's data/cache paths
export SUBGRID_DATA_PATH="${HR_BIN_DIR}"
export SUBGRID_CACHE_PATH="${HR_SIM_OUTPUT}/cache"

read -r PDF_CNN_RESOLUTION PDF_CNN_DOWNSAMPLE < <(python3 - <<PY
import json
c = json.load(open("${CONFIG_JSON}"))
hr = c["hr"]
ds = c["lr"]["downsample_factor"]
print(f"{hr['nx2']},{hr['nx1']} {ds}")
PY
)
export PDF_CNN_RESOLUTION PDF_CNN_DOWNSAMPLE

export LOGT_ACTIVE_START="${LOGT_ACTIVE_START:-4.1}"
export LOGT_ACTIVE_END="${LOGT_ACTIVE_END:-5.9}"

# ---------------------------------------------------------------------------
# Generate all Athinputs
# ---------------------------------------------------------------------------
LR_ATHINPUT="${ATHINPUT_CACHE_DIR}/lr_sim.athinput"
LR_BUILD_ATHINPUT="${ATHINPUT_CACHE_DIR}/lr_build_sim.athinput"
SG_ATHINPUT="${ATHINPUT_CACHE_DIR}/sg_sim.athinput"

generate_athinput "lr" "${LR_ATHINPUT}"
generate_athinput "lr_build" "${LR_BUILD_ATHINPUT}"
generate_athinput "sg" "${SG_ATHINPUT}"

# ---------------------------------------------------------------------------
# Write a manifest of all key paths for this run
# ---------------------------------------------------------------------------
MANIFEST="${RUN_DIR}/manifest.txt"
{
    echo "============================================================"
    echo " SubgridCGM Hyperparameter Tuning & Pipeline Run"
    echo "============================================================"
    echo "Timestamp          : ${TIMESTAMP}"
    echo "Run directory      : ${RUN_DIR}"
    echo "Project root       : ${PROJECT_ROOT}"
    echo "Optuna Trials      : ${N_OPTUNA_TRIALS}"
    echo "Epochs per trial   : ${EPOCHS_PER_TRIAL}"
    echo ""
    echo "--- Source scripts / configs ---"
    echo "Config JSON        : ${CONFIG_JSON}"
    echo "LR athinput (gen)  : ${LR_ATHINPUT}"
    echo "lr_build athinput  : ${LR_BUILD_ATHINPUT}"
    echo "SG athinput (gen)  : ${SG_ATHINPUT}"
    echo "Optuna Optimizer   : ${PROJECT_ROOT}/shell_scripts/optimize_subgrid.py"
    echo "CNN trainer        : ${PROJECT_ROOT}/models/conv_nn/pdf_cnn.py"
    echo "PDF benchmark      : ${PROJECT_ROOT}/data/mocks/pdf_plot.py"
    echo "Diagnostic plots   : ${PROJECT_ROOT}/data/mocks/mock_sg.py"
    echo ""
    echo "--- Simulation outputs ---"
    echo "LR sim (0→5 Myr)   : ${LR_OUTPUT_DIR}"
    echo "lr_build (5→10 Myr): ${LR_BUILD_OUTPUT_DIR}"
    echo "subgrid_model      : ${SG_OUTPUT_DIR}"
    echo "5 Myr restart file : ${LR_RST_5MYR}"
    echo ""
    echo "--- Model / plot outputs ---"
    echo "Model weights      : ${MODEL_SAVES_DIR}"
    echo "Loss plots         : ${LOSS_PLOTS_DIR}"
    echo "PDF mock outputs   : ${PDF_MOCKS_DIR}"
    echo "SG  mock outputs   : ${SG_MOCKS_DIR}"
} > "${MANIFEST}"

log "Run directory  : ${RUN_DIR}"
log "Manifest       : ${MANIFEST}"
separator

# ===========================================================================
# STEP 1 — Low-resolution simulation: 0 → 5 Myr (if not already present)
# ===========================================================================
if [[ ! -f "${LR_RST_5MYR}" ]]; then
    separator
    log "STEP 1: lr_simulation  (16×8 grid, 0 → 5 Myr)"
    separator
    run_step 1 "lr_simulation_5myr" \
        bash -c "
            set -euo pipefail
            cd '${PROJECT_ROOT}/builds/hr_build/src'
            ./athena -i '${LR_ATHINPUT}' -d '${LR_OUTPUT_DIR}'
        "
else
    log "STEP 1: Existing 5 Myr restart file found at ${LR_RST_5MYR} (skipping rerun)."
fi

# ===========================================================================
# STEP 2 — Closed-loop Optuna Bayesian Hyperparameter Optimization
# ===========================================================================
OPTUNA_DB="${RUN_DIR}/cnn_hyperparams.db"

separator
log "STEP 2: optuna_hyperparameter_optimization"
log "Trials: ${N_OPTUNA_TRIALS}, Epochs per trial: ${EPOCHS_PER_TRIAL}"
log "Database: ${OPTUNA_DB}"
separator

run_step 2 "optuna_hyperparameter_optimization" \
    python3 "${PROJECT_ROOT}/shell_scripts/optimize_subgrid.py" \
        --n_trials "${N_OPTUNA_TRIALS}" \
        --epochs_per_trial "${EPOCHS_PER_TRIAL}" \
        --athinput "${SG_ATHINPUT}" \
        --restart_file "${LR_RST_5MYR}" \
        --storage "sqlite:///${OPTUNA_DB}" \
        --study_name "cnn_hyperparams_${TIMESTAMP}" \
        --output_best_dir "${MODEL_SAVES_DIR}"

# ===========================================================================
# STEP 3 — Benchmark Best PDF CNN Model (pdf_plot.py)
# ===========================================================================
separator
log "STEP 3: benchmark_pdf_cnn (pdf_plot.py)"
separator

run_step 3 "benchmark_pdf_cnn" \
    bash -c "cd '${PROJECT_ROOT}/data/mocks' && python3 pdf_plot.py"

# ===========================================================================
# STEP 4 — lr_build: restart from 5 Myr with ISM cooling (no CNN)
# ===========================================================================
separator
log "STEP 4: lr_build (ISM cooling restart from ${LR_RST_5MYR})"
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
# STEP 5 — subgrid_model: restart from 5 Myr using optimized CNN weights
# ===========================================================================
separator
log "STEP 5: subgrid_model (CNN restart with optimized model)"
separator

run_step 5 "subgrid_model_cnn_restart" \
    bash -c "
        set -euo pipefail
        cd '${PROJECT_ROOT}/builds/subgrid_model/src'

        source '${VENV_ACTIVATE}'
        VENV='${PROJECT_ROOT}/venv'
        SITE_PACKAGES=\"\$VENV/lib/python3.10/site-packages\"
        export PYTHONPATH=\"\$PWD:\$SITE_PACKAGES\${PYTHONPATH:+:\$PYTHONPATH}\"
        export MODEL_SAVES_DIR='${MODEL_SAVES_DIR}'

        ./athena \
            -i '${SG_ATHINPUT}' \
            -d '${SG_OUTPUT_DIR}' \
            -r '${LR_RST_5MYR}'
    "

# ===========================================================================
# STEP 6 — Diagnostic plots (mock_sg.py)
# ===========================================================================
separator
log "STEP 6: diagnostic_plots (mock_sg.py)"
separator

run_step 6 "diagnostic_plots" \
    bash -c "cd '${PROJECT_ROOT}/data/mocks' && python3 mock_sg.py"

# ===========================================================================
# Summary
# ===========================================================================
separator
log "HYPERPARAMETER TUNING AND PIPELINE EXECUTION COMPLETED"
separator
log "Run directory    : ${RUN_DIR}"
log "Master log       : ${MASTER_LOG}"
log "Optuna Database  : ${OPTUNA_DB}"
log "Best Model Dir   : ${MODEL_SAVES_DIR}"
log "SG Mocks Output  : ${SG_MOCKS_DIR}"
separator
