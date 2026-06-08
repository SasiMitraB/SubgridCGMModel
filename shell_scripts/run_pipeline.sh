#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — Full SubgridCGM pipeline
#
# Steps:
#   1. High-resolution Athena simulation  (skipped if master HR output exists)
#   2. Train the PDF CNN model            (models/conv_nn/pdf_cnn.py)
#   3. Validate the model                 (data/mocks/pdf_plot.py)
#   4. Low-resolution Athena simulation   (HR athinput downsampled 32×)
#   5. Subgrid-model Athena simulation    (builds/subgrid_model)
#   6. Benchmark mock comparison          (data/mocks/mock_sg.py)
#
# HIGH-RESOLUTION SIMULATION POLICY
# -----------------------------------
# The HR simulation is expensive. The script keeps a single canonical set of
# HR outputs under:
#
#   simulation_outputs/hr_build/   (bin/, hst/, rst/, cache/)
#
# Before running Step 1, we check whether that directory already contains
# binary output files (*.bin).  If it does, Step 1 is SKIPPED and the
# existing outputs are reused.  Delete (or rename) the directory to force a
# fresh HR run.
#
# ATHINPUT CACHING
# -----------------------------------
# Every athinput actually used during the run is copied into:
#   runs/run_<timestamp>/athinputs/
# so you have a complete, self-contained record of what was run.
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
# The script exits immediately on any non-zero return (set -e) so it is
# obvious which step failed.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 0.  Project root and paths
# ---------------------------------------------------------------------------
PROJECT_ROOT="/Volumes/PortableSSD/Projects/SubgridCGMModel"

# ---- Canonical HR output (never changes across runs) ----
HR_SIM_OUTPUT="${PROJECT_ROOT}/simulation_outputs/hr_build"
HR_BIN_DIR="${HR_SIM_OUTPUT}/bin"

# ---- Per-run timestamped directory ----
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="${PROJECT_ROOT}/runs/run_${TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
ATHINPUT_CACHE_DIR="${RUN_DIR}/athinputs"   # all athinputs used this run

export MODEL_SAVES_DIR="${RUN_DIR}/model_saves"
export LOSS_PLOTS_DIR="${RUN_DIR}/loss_plots"
export PDF_MOCKS_DIR="${RUN_DIR}/pdf_mocks"
export SG_MOCKS_DIR="${RUN_DIR}/sg_mocks"

mkdir -p "${LOG_DIR}" "${ATHINPUT_CACHE_DIR}" "${MODEL_SAVES_DIR}" "${LOSS_PLOTS_DIR}" "${PDF_MOCKS_DIR}" "${SG_MOCKS_DIR}"

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
    echo "--- Source scripts / configs ---"
    echo "Config JSON        : ${CONFIG_JSON}"
    echo "HR athinput (gen)  : ${ATHINPUT_CACHE_DIR}/hr_sim.athinput"
    echo "LR athinput (gen)  : ${ATHINPUT_CACHE_DIR}/lr_sim.athinput"
    echo "SG athinput (gen)  : ${ATHINPUT_CACHE_DIR}/sg_sim.athinput"
    echo "CNN trainer        : ${PROJECT_ROOT}/models/conv_nn/pdf_cnn.py"
    echo "PDF mock           : ${PROJECT_ROOT}/data/mocks/pdf_plot.py"
    echo "SG mock            : ${PROJECT_ROOT}/data/mocks/mock_sg.py"
    echo ""
    echo "--- Output directories ---"
    echo "HR sim output      : ${HR_SIM_OUTPUT}  [canonical / shared across runs]"
    echo "LR sim output      : ${PROJECT_ROOT}/simulation_outputs/lr_build"
    echo "SG sim output      : ${PROJECT_ROOT}/simulation_outputs/subgrid_model"
    echo "Model weights      : ${MODEL_SAVES_DIR}"
    echo "Loss plots         : ${LOSS_PLOTS_DIR}"
    echo "PDF mock outputs   : ${PDF_MOCKS_DIR}"
    echo "SG  mock outputs   : ${SG_MOCKS_DIR}"
    echo ""
    echo "--- Cached athinputs (this run) ---"
    echo "  ${ATHINPUT_CACHE_DIR}/hr_sim.athinput"
    echo "  ${ATHINPUT_CACHE_DIR}/lr_sim.athinput"
    echo "  ${ATHINPUT_CACHE_DIR}/sg_sim.athinput"
} > "${MANIFEST}"

log "Run directory  : ${RUN_DIR}"
log "Manifest       : ${MANIFEST}"
separator

# ===========================================================================
# STEP 1 — High-resolution simulation
#
# This is guarded: if bin files already exist under HR_BIN_DIR, the simulation
# is skipped and the existing outputs are reused.
# ===========================================================================
separator
log "STEP 1: hr_simulation"

HR_ATHINPUT="${ATHINPUT_CACHE_DIR}/hr_sim.athinput"
generate_athinput "hr" "${HR_ATHINPUT}"

# Clean beforehand
dot_clean -m "${PROJECT_ROOT}"

if compgen -G "${HR_BIN_DIR}/*.bin" > /dev/null 2>&1 || \
   compgen -G "${HR_BIN_DIR}/*.hydro_w.*" > /dev/null 2>&1; then
    log "HR output already exists at: ${HR_BIN_DIR}"
    log "SKIPPING HR simulation — delete ${HR_BIN_DIR} to force a re-run."
    log "STEP 1 SKIPPED (using cached HR outputs)"
else
    log "No existing HR outputs found — running HR simulation."
    mkdir -p "${HR_SIM_OUTPUT}"
    run_step 1 "hr_simulation" \
        bash "${PROJECT_ROOT}/builds/hr_build/src/sim_still_steady_state.sh" "${HR_ATHINPUT}"
fi
dot_clean -m "${PROJECT_ROOT}"
separator

# ===========================================================================
# STEP 2 — Train the PDF CNN
# ===========================================================================

run_step 2 "train_pdf_cnn" \
    python3 "${PROJECT_ROOT}/models/conv_nn/pdf_cnn.py"

dot_clean -m "${PROJECT_ROOT}"

# ===========================================================================
# STEP 3 — Validate the model (pdf_plot.py)
#
# pdf_plot.py uses relative paths rooted at data/mocks/, so we run it from
# that directory.
# ===========================================================================
run_step 3 "validate_pdf_model" \
    bash -c "cd '${PROJECT_ROOT}/data/mocks' && python3 pdf_plot.py"

dot_clean -m "${PROJECT_ROOT}"

# ===========================================================================
# STEP 4 — Low-resolution simulation
#
# Derived from the canonical HR athinput with a 32× downsample:
#   HR  nx1=256, nx2=512  →  LR  nx1=8, nx2=16
#   HR  meshblock nx1=32  →  LR  meshblock nx1=8
#   HR  meshblock nx2=512 →  LR  meshblock nx2=16
#
# The generated athinput is written to the athinput cache directory and is
# also the file passed to Athena (-i flag), so the cache IS the source of
# truth for this run's LR config.
# ===========================================================================
LR_ATHINPUT="${ATHINPUT_CACHE_DIR}/lr_sim.athinput"
LR_OUTPUT_DIR="${PROJECT_ROOT}/simulation_outputs/lr_build"
mkdir -p "${LR_OUTPUT_DIR}"

generate_athinput "lr" "${LR_ATHINPUT}"

log "LR athinput mesh settings:"
grep -E '^\s*nx[12]\s*=' "${LR_ATHINPUT}" | tee -a "${MASTER_LOG}"

run_step 4 "lr_simulation" \
    bash -c "
        set -euo pipefail
        cd '${PROJECT_ROOT}/builds/hr_build/src'
        dot_clean -m '${PROJECT_ROOT}'
        ./athena -i '${LR_ATHINPUT}' -d '${LR_OUTPUT_DIR}'
        dot_clean -m '${PROJECT_ROOT}'
    "

dot_clean -m "${PROJECT_ROOT}"

# ===========================================================================
# STEP 5 — Subgrid-model simulation (neural-network source terms)
# ===========================================================================
SG_ATHINPUT="${ATHINPUT_CACHE_DIR}/sg_sim.athinput"
generate_athinput "sg" "${SG_ATHINPUT}"

run_step 5 "subgrid_model_simulation" \
    bash "${PROJECT_ROOT}/builds/subgrid_model/src/run_simulation.sh" "${SG_ATHINPUT}"

dot_clean -m "${PROJECT_ROOT}"

# ===========================================================================
# STEP 6 — Benchmark mock (mock_sg.py)
#
# Compares HR (coarse-grained), LR, and SG simulation outputs.
# Run from data/mocks/ so relative output paths resolve correctly.
# ===========================================================================
run_step 6 "benchmark_mock_sg" \
    bash -c "cd '${PROJECT_ROOT}/data/mocks' && python3 mock_sg.py"

dot_clean -m "${PROJECT_ROOT}"

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
log "  HR  → ${ATHINPUT_CACHE_DIR}/hr_sim.athinput"
log "  LR  → ${ATHINPUT_CACHE_DIR}/lr_sim.athinput"
log "  SG  → ${ATHINPUT_CACHE_DIR}/sg_sim.athinput"
log ""
log "Step logs:"
for f in "${LOG_DIR}"/step*.log; do
    log "  ${f}"
done
log ""
log "Key output directories:"
log "  HR sim   : ${HR_SIM_OUTPUT}  [canonical, shared]"
log "  LR sim   : ${LR_OUTPUT_DIR}"
log "  SG sim   : ${PROJECT_ROOT}/simulation_outputs/subgrid_model"
log "  Model    : ${MODEL_SAVES_DIR}"
log "  PDF mock : ${PDF_MOCKS_DIR}"
log "  SG mock  : ${SG_MOCKS_DIR}"
separator
