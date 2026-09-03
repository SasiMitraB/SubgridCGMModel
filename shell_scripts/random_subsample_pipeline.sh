#!/usr/bin/env bash
# =============================================================================
# random_subsample_pipeline.sh — SubgridCGM Pipeline with Random Crop Training
# =============================================================================
#
# Steps:
#   1. Train PDF CNN on random snapshot crops (random_snapshot_training.py)
#   2. Benchmark PDF CNN model (pdf_plot.py)
#   3. Low-resolution simulation 0 -> 5 Myr (athena; ISM cooling)
#   4. lr_build — restart from 5 Myr rst (athena; ISM cooling, 5 -> 10 Myr)
#   5. subgrid_model — restart from 5 Myr rst (athena; CNN subgrid, 5 -> 10 Myr)
#   6. Diagnostic comparison plots & animations (mock_sg.py)
#
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ROOT="/home/sasi/Projects/SubgridCGMModel"

# ---- 1. High-Resolution Training Data Sources ----
# Use both datasets from hr_gpu_sweep_1024x2048_2xlength for training
HR_SWEEP_BASE="${PROJECT_ROOT}/simulation_outputs/hr_gpu_sweep_1024x2048_2xlength"
HR_TRAIN_OUTPUT_1="${HR_TRAIN_OUTPUT_1:-${HR_SWEEP_BASE}/vshear_31_coldfrac_0.33}"
HR_TRAIN_OUTPUT_2="${HR_TRAIN_OUTPUT_2:-${HR_SWEEP_BASE}/vshear_31_coldfrac_0.67}"

HR_TRAIN_RUNS=(
    "${HR_TRAIN_OUTPUT_1}"
    "${HR_TRAIN_OUTPUT_2}"
)

HR_TRAIN_BIN_DIRS=()
HR_TRAIN_CACHE_DIRS=()
for run_dir in "${HR_TRAIN_RUNS[@]}"; do
    HR_TRAIN_BIN_DIRS+=("${run_dir}/bin")
    HR_TRAIN_CACHE_DIRS+=("${run_dir}/cache")
done

# Legacy aliases
HR_TRAIN_OUTPUT="${HR_TRAIN_RUNS[0]}"
HR_TRAIN_BIN_DIR="${HR_TRAIN_BIN_DIRS[0]}"
HR_TRAIN_CACHE_DIR="${HR_TRAIN_CACHE_DIRS[0]}"

# Full Fine-Grid Training Resolution (height H, width W)
# Format: "H,W" -> e.g. "2048,1024"
export PDF_CNN_RESOLUTION="${PDF_CNN_RESOLUTION:-2048,1024}"

# Coarse-graining downsample factor
# e.g., downsample=32 with 2048x1024 gives full coarse grid of 64x32
#       downsample=64 with 2048x1024 gives full coarse grid of 32x16
export PDF_CNN_DOWNSAMPLE="${PDF_CNN_DOWNSAMPLE:-64}"

# Coarse Random Crop Dimensions (height H_cg, width W_cg in coarse cells)
# 16x8 random crops drawn from the 32x16 coarse grid
export CROP_H_CG="${CROP_H_CG:-16}"
export CROP_W_CG="${CROP_W_CG:-8}"

# Derived Fine Crop Dimensions (for random_snapshot_training.py)
export CROP_H=$(( CROP_H_CG * PDF_CNN_DOWNSAMPLE ))
export CROP_W=$(( CROP_W_CG * PDF_CNN_DOWNSAMPLE ))

# ---- 2. Low-Resolution Simulation Grid (Athena) ----
# Mesh resolution for the LR simulation and restarts (nx2=height, nx1=width)
export SIM_NX2="${SIM_NX2:-${CROP_H_CG}}"         # e.g., 16
export SIM_NX1="${SIM_NX1:-${CROP_W_CG}}"         # e.g., 8
export SIM_MB_NX2="${SIM_MB_NX2:-${SIM_NX2}}"     # MeshBlock height (e.g., 16)
export SIM_MB_NX1="${SIM_MB_NX1:-${SIM_NX1}}"     # MeshBlock width  (e.g., 8)

# Domain boundaries and problem configuration (matches 2xlength sweep domain)
export DOMAIN_X1MIN="${DOMAIN_X1MIN:--5.0}"
export DOMAIN_X1MAX="${DOMAIN_X1MAX:-5.0}"
export DOMAIN_X2MIN="${DOMAIN_X2MIN:--10.0}"
export DOMAIN_X2MAX="${DOMAIN_X2MAX:-10.0}"
export PROBLEM_SIGMA="${PROBLEM_SIGMA:-1.0}"
export PROBLEM_A_CHAR="${PROBLEM_A_CHAR:-0.25}"
export PROBLEM_COLD_FRAC="${PROBLEM_COLD_FRAC:-0.5}"
export SIM_TLIM_LR="${SIM_TLIM_LR:-5.0}"
export SIM_TLIM_RESTART="${SIM_TLIM_RESTART:-10.0}"
export RESTART_TIME_MYR="${RESTART_TIME_MYR:-${SIM_TLIM_LR}}"
export START_FRAME="${START_FRAME:-500}"
export HR_START_FRAME="${HR_START_FRAME:-500}"

# ---- 3. Evaluation & Benchmark Reference ----
# Benchmarks model against hr_build_512 reference
HR_EVAL_OUTPUT="${HR_EVAL_OUTPUT:-${PROJECT_ROOT}/simulation_outputs/hr_build_512}"
HR_EVAL_BIN_DIR="${HR_EVAL_OUTPUT}/bin"
HR_EVAL_CACHE_DIR="${HR_EVAL_OUTPUT}/cache"
HR_EVAL_RESOLUTION="${HR_EVAL_RESOLUTION:-512,256}"
HR_EVAL_DOWNSAMPLE="${HR_EVAL_DOWNSAMPLE:-32}"

# ---- 4. Training Hyperparameters ----
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
export PDF_CNN_ALPHA_ACTIVE_WASSERSTEIN="${PDF_CNN_ALPHA_ACTIVE_WASSERSTEIN:-${PDF_CNN_ALPHA_ACTIVE_KL:-10.0}}"
export PDF_CNN_ALPHA_INACTIVE_WASSERSTEIN="${PDF_CNN_ALPHA_INACTIVE_WASSERSTEIN:-${PDF_CNN_ALPHA_INACTIVE_KL:-10.0}}"
export PDF_CNN_ALPHA_ACTIVE_KL="${PDF_CNN_ALPHA_ACTIVE_WASSERSTEIN}"
export PDF_CNN_ALPHA_INACTIVE_KL="${PDF_CNN_ALPHA_INACTIVE_WASSERSTEIN}"
export PDF_CNN_ALPHA_GATE="${PDF_CNN_ALPHA_GATE:-0.0}"
export PDF_CNN_ALPHA_MEAN_TEMP="${PDF_CNN_ALPHA_MEAN_TEMP:-10.0}"
export PDF_CNN_ALPHA_EMISS="${PDF_CNN_ALPHA_EMISS:-10.0}"
export PDF_CNN_ALPHA_LEAK="${PDF_CNN_ALPHA_LEAK:-10.0}"
export PDF_CNN_GATE_EPOCHS="${PDF_CNN_GATE_EPOCHS:-200}"
export PDF_CNN_GATE_LR="${PDF_CNN_GATE_LR:-1e-3}"

# Active cooling window log10(T) bounds
export LOGT_ACTIVE_START="${LOGT_ACTIVE_START:-4.1}"
export LOGT_ACTIVE_END="${LOGT_ACTIVE_END:-5.9}"

# =============================================================================
# DIRECTORIES & LOGGING SETUP
# =============================================================================

# ---- Simulation Output Directories ----
LR_OUTPUT_DIR="${PROJECT_ROOT}/simulation_outputs/lr_build"
LR_RST_5MYR="${LR_OUTPUT_DIR}/rst/KH.00005.rst"
LR_BUILD_OUTPUT_DIR="${PROJECT_ROOT}/simulation_outputs/lr_build_ism"
SG_OUTPUT_DIR="${PROJECT_ROOT}/simulation_outputs/subgrid_model"

# ---- Per-run Timestamped Output Directory ----
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
    "${SG_OUTPUT_DIR}"

for c_dir in "${HR_TRAIN_CACHE_DIRS[@]}"; do
    mkdir -p "${c_dir}"
done

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
# Helper: generate an athinput file with dynamic dimensions
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
        --x1min "${DOMAIN_X1MIN}" \
        --x1max "${DOMAIN_X1MAX}" \
        --x2min "${DOMAIN_X2MIN}" \
        --x2max "${DOMAIN_X2MAX}" \
        --sigma "${PROBLEM_SIGMA}" \
        --a_char "${PROBLEM_A_CHAR}" \
        --cold_frac "${PROBLEM_COLD_FRAC}"
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

# Parse resolution components for logging
RES_H=$(echo "${PDF_CNN_RESOLUTION}" | cut -d',' -f1 | tr -d ' ')
RES_W=$(echo "${PDF_CNN_RESOLUTION}" | cut -d',' -f2 | tr -d ' ')
COARSE_FULL_H=$(( RES_H / PDF_CNN_DOWNSAMPLE ))
COARSE_FULL_W=$(( RES_W / PDF_CNN_DOWNSAMPLE ))

# ---------------------------------------------------------------------------
# Write run manifest
# ---------------------------------------------------------------------------
MANIFEST="${RUN_DIR}/manifest.txt"
{
    echo "============================================================"
    echo " SubgridCGM Random Subsample Pipeline Run"
    echo "============================================================"
    echo "Timestamp          : ${TIMESTAMP}"
    echo "Run directory      : ${RUN_DIR}"
    echo "Project root       : ${PROJECT_ROOT}"
    echo "Training data      : ${HR_TRAIN_BIN_DIRS[*]}"
    echo "Training cache     : ${HR_TRAIN_CACHE_DIRS[*]}"
    echo "Eval reference     : ${HR_EVAL_OUTPUT}"
    echo "Resolution (fine)  : ${PDF_CNN_RESOLUTION} (H=${RES_H}, W=${RES_W})"
    echo "Downsample factor  : ${PDF_CNN_DOWNSAMPLE}"
    echo "Full coarse grid   : (${COARSE_FULL_H}x${COARSE_FULL_W})"
    echo "Random crop size   : Coarse=(${CROP_H_CG}x${CROP_W_CG}), Fine=(${CROP_H}x${CROP_W})"
    echo "Athena simulation  : nx2=${SIM_NX2}, nx1=${SIM_NX1} (meshblock: ${SIM_MB_NX2}x${SIM_MB_NX1})"
    echo "Domain extents     : x1=[${DOMAIN_X1MIN}, ${DOMAIN_X1MAX}], x2=[${DOMAIN_X2MIN}, ${DOMAIN_X2MAX}]"
    echo "Simulation times   : LR 0→${SIM_TLIM_LR} Myr, Restart ${RESTART_TIME_MYR}→${SIM_TLIM_RESTART} Myr (start_frame=${START_FRAME}, hr_start=${HR_START_FRAME})"
    echo "Eval resolution    : ${HR_EVAL_RESOLUTION} (ds=${HR_EVAL_DOWNSAMPLE})"
    echo "Epochs             : ${NUM_EPOCHS}"
    echo "Batch size         : ${BATCH_SIZE}"
    echo "Learning rate      : ${LEARNING_RATE}"
    echo "Snapshot split     : train=${TRAIN_FRAC}, val=${VAL_FRAC}, test=$(python3 -c "print(round(1.0-${TRAIN_FRAC}-${VAL_FRAC}, 2))")"
    echo "Crops per snap     : train=${N_CROPS_TRAIN}, val=${N_CROPS_VAL}, test=${N_CROPS_TEST}"
    echo "EMA Alpha          : ${EMA_ALPHA}"
    echo ""
    echo "--- Loss Weights ---"
    echo "alpha_active_wass  : ${PDF_CNN_ALPHA_ACTIVE_WASSERSTEIN}"
    echo "alpha_inact_wass   : ${PDF_CNN_ALPHA_INACTIVE_WASSERSTEIN}"
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

log "Run directory      : ${RUN_DIR}"
log "Training grid      : Fine=${PDF_CNN_RESOLUTION} -> Downsample=${PDF_CNN_DOWNSAMPLE} -> Full coarse=${COARSE_FULL_H}x${COARSE_FULL_W}"
log "Random crop size   : Coarse=${CROP_H_CG}x${CROP_W_CG} (Fine=${CROP_H}x${CROP_W})"
log "Athena simulation  : ${SIM_NX2}x${SIM_NX1} grid"
log "Manifest           : ${MANIFEST}"
separator

# ===========================================================================
# STEP 1 — Train the PDF CNN with Random Snapshot Crops
# ===========================================================================
separator
log "STEP 1: train_random_snapshot_cnn"
log "Training data  : ${HR_TRAIN_BIN_DIRS[*]}"
log "Cache paths    : ${HR_TRAIN_CACHE_DIRS[*]}"
separator

run_step 1 "train_random_snapshot_cnn" \
    python3 "${PROJECT_ROOT}/random_snapshot_training.py" \
        --data_path "${HR_TRAIN_BIN_DIRS[@]}" \
        --cache_path "${HR_TRAIN_CACHE_DIRS[@]}" \
        --resolution "${PDF_CNN_RESOLUTION}" \
        --crop_h "${CROP_H}" \
        --crop_w "${CROP_W}" \
        --crop_h_cg "${CROP_H_CG}" \
        --crop_w_cg "${CROP_W_CG}" \
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
        --alpha_active_wasserstein "${PDF_CNN_ALPHA_ACTIVE_WASSERSTEIN}" \
        --alpha_inactive_wasserstein "${PDF_CNN_ALPHA_INACTIVE_WASSERSTEIN}" \
        --alpha_gate "${PDF_CNN_ALPHA_GATE}" \
        --alpha_mean_temp "${PDF_CNN_ALPHA_MEAN_TEMP}" \
        --alpha_emiss "${PDF_CNN_ALPHA_EMISS}" \
        --alpha_leak "${PDF_CNN_ALPHA_LEAK}" \
        --gate_epochs "${PDF_CNN_GATE_EPOCHS}" \
        --gate_learning_rate "${PDF_CNN_GATE_LR}" \
        --model_save_dir "${MODEL_SAVES_DIR}" \
        --loss_plot_dir "${LOSS_PLOTS_DIR}"

# Provide alias checkpoint names for eval and legacy resolution naming
# Filter out gate-only checkpoint (*_gate.pth)
train_model="${MODEL_SAVES_DIR}/cnn_(${PDF_CNN_RESOLUTION//,/,\ })_${PDF_CNN_DOWNSAMPLE}.pth"
train_mean="${MODEL_SAVES_DIR}/cnn_(${PDF_CNN_RESOLUTION//,/,\ })_${PDF_CNN_DOWNSAMPLE}_input_mean.npy"
train_std="${MODEL_SAVES_DIR}/cnn_(${PDF_CNN_RESOLUTION//,/,\ })_${PDF_CNN_DOWNSAMPLE}_input_std.npy"

if [[ ! -f "${train_model}" ]]; then
    train_model=$(find "${MODEL_SAVES_DIR}" -name "cnn_*.pth" ! -name "*gate*" | head -n 1)
    train_mean=$(find "${MODEL_SAVES_DIR}" -name "cnn_*_input_mean.npy" ! -name "*gate*" | head -n 1)
    train_std=$(find "${MODEL_SAVES_DIR}" -name "cnn_*_input_std.npy" ! -name "*gate*" | head -n 1)
fi

if [[ -n "${train_model}" && -f "${train_model}" ]]; then
    # Aliases for evaluation resolution
    cp -f "${train_model}" "${MODEL_SAVES_DIR}/cnn_(${HR_EVAL_RESOLUTION//,/,\ })_${HR_EVAL_DOWNSAMPLE}.pth" 2>/dev/null || true
    cp -f "${train_mean}" "${MODEL_SAVES_DIR}/cnn_(${HR_EVAL_RESOLUTION//,/,\ })_${HR_EVAL_DOWNSAMPLE}_input_mean.npy" 2>/dev/null || true
    cp -f "${train_std}" "${MODEL_SAVES_DIR}/cnn_(${HR_EVAL_RESOLUTION//,/,\ })_${HR_EVAL_DOWNSAMPLE}_input_std.npy" 2>/dev/null || true

    # Aliases for standard configurations
    cp -f "${train_model}" "${MODEL_SAVES_DIR}/cnn_(512, 256)_32.pth" 2>/dev/null || true
    cp -f "${train_mean}" "${MODEL_SAVES_DIR}/cnn_(512, 256)_32_input_mean.npy" 2>/dev/null || true
    cp -f "${train_std}" "${MODEL_SAVES_DIR}/cnn_(512, 256)_32_input_std.npy" 2>/dev/null || true

    cp -f "${train_model}" "${MODEL_SAVES_DIR}/cnn_(1024, 512)_64.pth" 2>/dev/null || true
    cp -f "${train_mean}" "${MODEL_SAVES_DIR}/cnn_(1024, 512)_64_input_mean.npy" 2>/dev/null || true
    cp -f "${train_std}" "${MODEL_SAVES_DIR}/cnn_(1024, 512)_64_input_std.npy" 2>/dev/null || true
    log "Checkpoint aliases created in ${MODEL_SAVES_DIR}"
fi

# ===========================================================================
# STEP 2 — Benchmark PDF CNN model (pdf_plot.py)
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
        export MODEL_SAVES_DIR='${MODEL_SAVES_DIR}'
        export PDF_MOCKS_DIR='${PDF_MOCKS_DIR}'
        python3 pdf_plot.py
    "

# ===========================================================================
# STEP 3 — Low-resolution simulation: 0 → 5 Myr  (ISM cooling)
# ===========================================================================
LR_ATHINPUT="${ATHINPUT_CACHE_DIR}/lr_sim.athinput"
generate_athinput "lr" "${LR_ATHINPUT}"

separator
log "STEP 3: lr_simulation  (${SIM_NX2}×${SIM_NX1} grid, 0 → ${SIM_TLIM_LR} Myr)"
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

# Clean previous simulation outputs if any to prevent stale files from polluting restart frames
rm -rf "${LR_BUILD_OUTPUT_DIR:?}"/*
mkdir -p "${LR_BUILD_OUTPUT_DIR}"

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

# Clean previous simulation outputs if any to prevent stale files from polluting restart frames
rm -rf "${SG_OUTPUT_DIR:?}"/*
mkdir -p "${SG_OUTPUT_DIR}"

run_step 5 "subgrid_model_cnn_restart" \
    bash -c "
        set -euo pipefail
        cd '${PROJECT_ROOT}/builds/subgrid_model/src'

        source '${VENV_ACTIVATE}'
        VENV='${PROJECT_ROOT}/venv'
        SITE_PACKAGES=\"\$VENV/lib/python3.14/site-packages\"
        export PYTHONPATH=\"\$PWD:\$SITE_PACKAGES\${PYTHONPATH:+:\$PYTHONPATH}\"
        export PDF_CNN_RESOLUTION='${PDF_CNN_RESOLUTION}'
        export PDF_CNN_DOWNSAMPLE='${PDF_CNN_DOWNSAMPLE}'
        export TILE_ROWS='${CROP_H_CG}'
        export TILE_COLS='${CROP_W_CG}'
        export CROP_H_CG='${CROP_H_CG}'
        export CROP_W_CG='${CROP_W_CG}'
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
log "STEP 6: diagnostic_plots  (mock_sg.py against ${HR_EVAL_OUTPUT})"
separator

run_step 6 "diagnostic_plots" \
    bash -c "
        set -euo pipefail
        cd '${PROJECT_ROOT}/data/mocks'
        export START_FRAME='${START_FRAME}'
        export RESTART_TIME_MYR='${RESTART_TIME_MYR}'
        export HR_START_FRAME='${HR_START_FRAME}'
        export HR_SIM_OUTPUT='${HR_EVAL_OUTPUT}'
        export SUBGRID_DATA_PATH='${HR_EVAL_BIN_DIR}'
        export SUBGRID_CACHE_PATH='${HR_EVAL_CACHE_DIR}'
        export PDF_CNN_RESOLUTION='${PDF_CNN_RESOLUTION}'
        export PDF_CNN_DOWNSAMPLE='${PDF_CNN_DOWNSAMPLE}'
        export HR_EVAL_RESOLUTION='${HR_EVAL_RESOLUTION}'
        export HR_EVAL_DOWNSAMPLE='${HR_EVAL_DOWNSAMPLE}'
        export SIM_NX2='${SIM_NX2}'
        export SIM_NX1='${SIM_NX1}'
        export CROP_H_CG='${CROP_H_CG}'
        export CROP_W_CG='${CROP_W_CG}'
        export SG_RESOLUTION='${SIM_NX2},${SIM_NX1}'
        export MODEL_SAVES_DIR='${MODEL_SAVES_DIR}'
        export SG_MOCKS_DIR='${SG_MOCKS_DIR}'
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
log "  LR sim (0→5 Myr)       : ${LR_OUTPUT_DIR}"
log "  lr_build (ISM restart) : ${LR_BUILD_OUTPUT_DIR}"
log "  subgrid_model (CNN)    : ${SG_OUTPUT_DIR}"
log "  Model weights          : ${MODEL_SAVES_DIR}"
log "  PDF mock               : ${PDF_MOCKS_DIR}"
log "  SG mock                : ${SG_MOCKS_DIR}"
separator
