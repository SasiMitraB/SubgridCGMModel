#!/usr/bin/env bash
# =============================================================================
# random_subsample_pipeline.sh — SubgridCGM pipeline with Snapshot-Split Random Crop Training
#
# Training Data Source (Random Crop Pool):
#   simulation_outputs/hr_gpu_sweep_1024x2048_2xlength/vshear_31_coldfrac_0.67
#
# Evaluation & Benchmark Reference:
#   simulation_outputs/hr_build_512
#
# Steps:
#   1. Train the PDF CNN with random snapshot crops (random_snapshot_training.py)
#   2. Benchmark PDF CNN model against hr_build_512 (data/mocks/pdf_plot.py)
#   3. Low-resolution simulation 5 Myr              (16×8 grid; ISM cooling)
#      Outputs to: simulation_outputs/lr_build
#   4. lr_build — restart from 5 Myr rst           (hr_build/src/athena; ISM cooling)
#      Uses:  simulation_outputs/lr_build/rst/KH.00005.rst
#      Outputs to: simulation_outputs/lr_build_ism
#   5. subgrid_model — restart from same rst         (subgrid_model/src/athena; CNN)
#      Uses:  simulation_outputs/lr_build/rst/KH.00005.rst
#      Outputs to: simulation_outputs/subgrid_model
#   6. Diagnostic plots against hr_build_512        (data/mocks/mock_sg.py)
#
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 0.  Project root and paths
# ---------------------------------------------------------------------------
PROJECT_ROOT="/home/sasi/Projects/SubgridCGMModel"

# ---- High-Resolution Training Data Source (Sweep) ----
HR_TRAIN_OUTPUT="${PROJECT_ROOT}/simulation_outputs/hr_gpu_sweep_1024x2048_2xlength/vshear_31_coldfrac_0.67"
HR_TRAIN_BIN_DIR="${HR_TRAIN_OUTPUT}/bin"
HR_TRAIN_CACHE_DIR="${HR_TRAIN_OUTPUT}/cache"

# ---- High-Resolution Evaluation Reference (hr_build_512) ----
HR_EVAL_OUTPUT="${PROJECT_ROOT}/simulation_outputs/hr_build_512"
HR_EVAL_BIN_DIR="${HR_EVAL_OUTPUT}/bin"
HR_EVAL_CACHE_DIR="${HR_EVAL_OUTPUT}/cache"
HR_EVAL_RESOLUTION="512,256"
HR_EVAL_DOWNSAMPLE="32"

# ---- Training Grid & Downsampling Configuration ----
export PDF_CNN_RESOLUTION="${PDF_CNN_RESOLUTION:-2048,1024}"
export PDF_CNN_DOWNSAMPLE="${PDF_CNN_DOWNSAMPLE:-64}"
export CROP_H="${CROP_H:-1024}"
export CROP_W="${CROP_W:-512}"

# ---- Training Hyperparameters ----
export NUM_EPOCHS="${NUM_EPOCHS:-1000}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export LEARNING_RATE="${LEARNING_RATE:-1e-3}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
export TRAIN_FRAC="${TRAIN_FRAC:-0.60}"
export VAL_FRAC="${VAL_FRAC:-0.20}"
export N_CROPS_TRAIN="${N_CROPS_TRAIN:-8}"
export N_CROPS_VAL="${N_CROPS_VAL:-4}"
export N_CROPS_TEST="${N_CROPS_TEST:-4}"
export EMA_ALPHA="${EMA_ALPHA:-0.9}"
export SEED="${SEED:-42}"

# Loss weights
export PDF_CNN_ALPHA_ACTIVE_KL="${PDF_CNN_ALPHA_ACTIVE_KL:-10.0}"
export PDF_CNN_ALPHA_INACTIVE_KL="${PDF_CNN_ALPHA_INACTIVE_KL:-10.0}"
export PDF_CNN_ALPHA_GATE="${PDF_CNN_ALPHA_GATE:-50.0}"
export PDF_CNN_ALPHA_MEAN_TEMP="${PDF_CNN_ALPHA_MEAN_TEMP:-10.0}"
export PDF_CNN_ALPHA_EMISS="${PDF_CNN_ALPHA_EMISS:-10.0}"
export PDF_CNN_ALPHA_LEAK="${PDF_CNN_ALPHA_LEAK:-10.0}"

# ---- LR 5 Myr base simulation (Step 3) ----
LR_OUTPUT_DIR="${PROJECT_ROOT}/simulation_outputs/lr_build"
LR_RST_5MYR="${LR_OUTPUT_DIR}/rst/KH.00005.rst"

# ---- lr_build restart (Step 4) — ISM cooling from 5 Myr ----
LR_BUILD_OUTPUT_DIR="${PROJECT_ROOT}/simulation_outputs/lr_build_ism"

# ---- subgrid_model restart (Step 5) — CNN from 5 Myr ----
SG_OUTPUT_DIR="${PROJECT_ROOT}/simulation_outputs/subgrid_model"

# ---- Per-run timestamped directory ----
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="${PROJECT_ROOT}/runs/run_random_crop_${TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
ATHINPUT_CACHE_DIR="${RUN_DIR}/athinputs"

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
    "${SG_OUTPUT_DIR}" \
    "${HR_TRAIN_CACHE_DIR}"

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

# Active cooling window log10(T) bounds
export LOGT_ACTIVE_START="${LOGT_ACTIVE_START:-4.1}"
export LOGT_ACTIVE_END="${LOGT_ACTIVE_END:-5.9}"

# ---------------------------------------------------------------------------
# Write a manifest of all key paths for this run
# ---------------------------------------------------------------------------
MANIFEST="${RUN_DIR}/manifest.txt"
{
    echo "============================================================"
    echo " SubgridCGM Random Crop Pipeline Run"
    echo "============================================================"
    echo "Timestamp          : ${TIMESTAMP}"
    echo "Run directory      : ${RUN_DIR}"
    echo "Project root       : ${PROJECT_ROOT}"
    echo "Training data      : ${HR_TRAIN_BIN_DIR}"
    echo "Training cache     : ${HR_TRAIN_CACHE_DIR}"
    echo "Eval reference     : ${HR_EVAL_OUTPUT}"
    echo "Resolution (train) : ${PDF_CNN_RESOLUTION}"
    echo "Downsample (train) : ${PDF_CNN_DOWNSAMPLE}"
    echo "Crop dimensions    : (${CROP_H}, ${CROP_W})"
    echo "Eval resolution    : ${HR_EVAL_RESOLUTION} (ds=${HR_EVAL_DOWNSAMPLE})"
    echo "Epochs             : ${NUM_EPOCHS}"
    echo "Batch size         : ${BATCH_SIZE}"
    echo "Learning rate      : ${LEARNING_RATE}"
    echo "Snapshot split     : train=${TRAIN_FRAC}, val=${VAL_FRAC}, test=$(python3 -c "print(round(1.0-${TRAIN_FRAC}-${VAL_FRAC}, 2))")"
    echo "Crops per snap     : train=${N_CROPS_TRAIN}, val=${N_CROPS_VAL}, test=${N_CROPS_TEST}"
    echo "EMA Alpha          : ${EMA_ALPHA}"
    echo ""
    echo "--- Loss Weights ---"
    echo "alpha_active_kl    : ${PDF_CNN_ALPHA_ACTIVE_KL}"
    echo "alpha_inactive_kl  : ${PDF_CNN_ALPHA_INACTIVE_KL}"
    echo "alpha_gate         : ${PDF_CNN_ALPHA_GATE}"
    echo "alpha_mean_temp    : ${PDF_CNN_ALPHA_MEAN_TEMP}"
    echo "alpha_emiss        : ${PDF_CNN_ALPHA_EMISS}"
    echo "alpha_leak         : ${PDF_CNN_ALPHA_LEAK}"
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
} > "${MANIFEST}"

log "Run directory  : ${RUN_DIR}"
log "Manifest       : ${MANIFEST}"
separator

# ===========================================================================
# STEP 1 — Train the PDF CNN with Random Snapshot Crops
# ===========================================================================
separator
log "STEP 1: train_random_snapshot_cnn"
log "Training data  : ${HR_TRAIN_BIN_DIR}"
log "Cache path     : ${HR_TRAIN_CACHE_DIR}"
separator

run_step 1 "train_random_snapshot_cnn" \
    python3 "${PROJECT_ROOT}/random_snapshot_training.py" \
        --data_path "${HR_TRAIN_BIN_DIR}" \
        --cache_path "${HR_TRAIN_CACHE_DIR}" \
        --resolution "${PDF_CNN_RESOLUTION}" \
        --crop_h "${CROP_H}" \
        --crop_w "${CROP_W}" \
        --downsample "${PDF_CNN_DOWNSAMPLE}" \
        --train_frac "${TRAIN_FRAC}" \
        --val_frac "${VAL_FRAC}" \
        --n_crops_train "${N_CROPS_TRAIN}" \
        --n_crops_val "${N_CROPS_VAL}" \
        --n_crops_test "${N_CROPS_TEST}" \
        --ema_alpha "${EMA_ALPHA}" \
        --epochs "${NUM_EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --learning_rate "${LEARNING_RATE}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --seed "${SEED}" \
        --alpha_active_kl "${PDF_CNN_ALPHA_ACTIVE_KL}" \
        --alpha_inactive_kl "${PDF_CNN_ALPHA_INACTIVE_KL}" \
        --alpha_gate "${PDF_CNN_ALPHA_GATE}" \
        --alpha_mean_temp "${PDF_CNN_ALPHA_MEAN_TEMP}" \
        --alpha_emiss "${PDF_CNN_ALPHA_EMISS}" \
        --alpha_leak "${PDF_CNN_ALPHA_LEAK}" \
        --model_save_dir "${MODEL_SAVES_DIR}" \
        --loss_plot_dir "${LOSS_PLOTS_DIR}"

# Provide alias checkpoint names for eval resolution (512, 256) ds=32
train_model=$(find "${MODEL_SAVES_DIR}" -name "cnn_*.pth" | head -n 1)
train_mean=$(find "${MODEL_SAVES_DIR}" -name "cnn_*_input_mean.npy" | head -n 1)
train_std=$(find "${MODEL_SAVES_DIR}" -name "cnn_*_input_std.npy" | head -n 1)

if [[ -n "${train_model}" && -f "${train_model}" ]]; then
    cp -f "${train_model}" "${MODEL_SAVES_DIR}/cnn_(512, 256)_32.pth"
    cp -f "${train_mean}" "${MODEL_SAVES_DIR}/cnn_(512, 256)_32_input_mean.npy"
    cp -f "${train_std}" "${MODEL_SAVES_DIR}/cnn_(512, 256)_32_input_std.npy"
    log "Aliases created for (512, 256)_32 in ${MODEL_SAVES_DIR}"
fi

# ===========================================================================
# STEP 2 — Benchmark PDF CNN model (pdf_plot.py) against hr_build_512
# ===========================================================================
separator
log "STEP 2: benchmark_pdf_cnn  (pdf_plot.py against ${HR_EVAL_OUTPUT})"
log "Benchmarking script : ${PROJECT_ROOT}/data/mocks/pdf_plot.py"
separator

run_step 2 "benchmark_pdf_cnn" \
    bash -c "
        set -euo pipefail
        cd '${PROJECT_ROOT}/data/mocks'
        export HR_SIM_OUTPUT='${HR_EVAL_OUTPUT}'
        export SUBGRID_DATA_PATH='${HR_EVAL_BIN_DIR}'
        export SUBGRID_CACHE_PATH='${HR_EVAL_CACHE_DIR}'
        export PDF_CNN_RESOLUTION='${HR_EVAL_RESOLUTION}'
        export PDF_CNN_DOWNSAMPLE='${HR_EVAL_DOWNSAMPLE}'
        python3 pdf_plot.py
    "

# ===========================================================================
# STEP 3 — Low-resolution simulation: 0 → 5 Myr  (16×8 grid, ISM cooling)
# ===========================================================================
LR_ATHINPUT="${ATHINPUT_CACHE_DIR}/lr_sim.athinput"
generate_athinput "lr" "${LR_ATHINPUT}"

separator
log "STEP 3: lr_simulation  (16×8 grid, 0 → 5 Myr)"
log "LR athinput mesh settings:"
grep -E '^\s*nx[12]\s*=' "${LR_ATHINPUT}" | tee -a "${MASTER_LOG}" || true
log "LR athinput tlim:"
grep -E '^\s*tlim\s*=' "${LR_ATHINPUT}" | tee -a "${MASTER_LOG}" || true
separator

run_step 3 "lr_simulation_5myr" \
    bash -c "
        set -euo pipefail
        cd '${PROJECT_ROOT}/builds/hr_build/src'
        ./athena -i '${LR_ATHINPUT}' -d '${LR_OUTPUT_DIR}'
    "

if [[ ! -f "${LR_RST_5MYR}" ]]; then
    log "ERROR: Expected 5 Myr restart file not found: ${LR_RST_5MYR}"
    exit 1
fi
log "5 Myr restart file confirmed: ${LR_RST_5MYR}"

# ===========================================================================
# STEP 4 — lr_build: restart from 5 Myr with ISM cooling (no CNN)
# ===========================================================================
LR_BUILD_ATHINPUT="${ATHINPUT_CACHE_DIR}/lr_build_sim.athinput"
generate_athinput "lr_build" "${LR_BUILD_ATHINPUT}"

separator
log "STEP 4: lr_build  (ISM cooling restart from ${LR_RST_5MYR})"
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
# ===========================================================================
SG_ATHINPUT="${ATHINPUT_CACHE_DIR}/sg_sim.athinput"
generate_athinput "sg" "${SG_ATHINPUT}"

separator
log "STEP 5: subgrid_model  (CNN restart from ${LR_RST_5MYR})"
separator

run_step 5 "subgrid_model_cnn_restart" \
    bash -c "
        set -euo pipefail
        cd '${PROJECT_ROOT}/builds/subgrid_model/src'

        source '${VENV_ACTIVATE}'
        VENV='${PROJECT_ROOT}/venv'
        SITE_PACKAGES=\"\$VENV/lib/python3.14/site-packages\"
        export PYTHONPATH=\"\$PWD:\$SITE_PACKAGES\${PYTHONPATH:+:\$PYTHONPATH}\"
        export PDF_CNN_RESOLUTION='${HR_EVAL_RESOLUTION}'
        export PDF_CNN_DOWNSAMPLE='${HR_EVAL_DOWNSAMPLE}'

        ./athena \
            -i '${SG_ATHINPUT}' \
            -d '${SG_OUTPUT_DIR}' \
            -r '${LR_RST_5MYR}'
    "

# ===========================================================================
# STEP 6 — Diagnostic plots (mock_sg.py) against hr_build_512
# ===========================================================================
separator
log "STEP 6: diagnostic_plots  (mock_sg.py against ${HR_EVAL_OUTPUT})"
separator

run_step 6 "diagnostic_plots" \
    bash -c "
        set -euo pipefail
        cd '${PROJECT_ROOT}/data/mocks'
        export HR_SIM_OUTPUT='${HR_EVAL_OUTPUT}'
        export SUBGRID_DATA_PATH='${HR_EVAL_BIN_DIR}'
        export SUBGRID_CACHE_PATH='${HR_EVAL_CACHE_DIR}'
        export PDF_CNN_RESOLUTION='${HR_EVAL_RESOLUTION}'
        export PDF_CNN_DOWNSAMPLE='${HR_EVAL_DOWNSAMPLE}'
        python3 mock_sg.py
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
log "Key output directories:"
log "  LR sim (0→5 Myr)      : ${LR_OUTPUT_DIR}"
log "  lr_build (ISM restart) : ${LR_BUILD_OUTPUT_DIR}"
log "  subgrid_model (CNN)    : ${SG_OUTPUT_DIR}"
log "  Model weights          : ${MODEL_SAVES_DIR}"
log "  PDF mock               : ${PDF_MOCKS_DIR}"
log "  SG mock                : ${SG_MOCKS_DIR}"
separator
