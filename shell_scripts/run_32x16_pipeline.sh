#!/usr/bin/env bash
# =============================================================================
# run_32x16_pipeline.sh
# ---------------------
# Full pipeline for 32x16 Tiled Subgrid CNN Simulation vs LR vs HR reference:
#
# Steps:
#   1. LR simulation (32x16, 0 -> 5 Myr, ISM cooling) -> produces KH.00005.rst
#   2. LR ISM restart (32x16, 5 -> 10 Myr, ISM cooling)
#   3. SG Tiled restart (32x16, 5 -> 10 Myr, CNN 4-tile subgrid)
#   4. Benchmark with mock_sg_32x16.py (comparing HR, SG-tiled, and LR)
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/home/sasi/Projects/SubgridCGMModel"
OUTPUT_BASE="${PROJECT_ROOT}/simulation_outputs/subgrid_32x16_vshear31_cf033"

LR_OUTPUT_DIR="${OUTPUT_BASE}/lr_build"
LR_BUILD_OUTPUT_DIR="${OUTPUT_BASE}/lr_build_ism"
SG_OUTPUT_DIR="${OUTPUT_BASE}/sg_tiled"
MOCKS_DIR="${OUTPUT_BASE}/mocks"

LR_RST_5MYR="${LR_OUTPUT_DIR}/rst/KH.00005.rst"

# Athinputs
ATHINPUT_DIR="${PROJECT_ROOT}/athinputs/subgrid_32x16"
LR_ATHINPUT="${ATHINPUT_DIR}/lr_vshear31_cf033.athinput"
LR_ISM_ATHINPUT="${ATHINPUT_DIR}/lr_ism_vshear31_cf033.athinput"
SG_ATHINPUT="${ATHINPUT_DIR}/sg_tiled_vshear31_cf033.athinput"

# Reference HR simulation (2xlength domain, vshear=31, coldfrac=0.33)
HR_SIM_DIR="${PROJECT_ROOT}/simulation_outputs/hr_gpu_sweep_1024x2048_2xlength/vshear_31_coldfrac_0.33"

# ML Model Configuration
export MODEL_SAVES_DIR="${PROJECT_ROOT}/runs/run_optuna_20260816_215537/model_saves"
export PDF_CNN_RESOLUTION="1024,512"
export PDF_CNN_DOWNSAMPLE="64"

# Mock SG Configuration
export HR_SIM_OUTPUT="${HR_SIM_DIR}"
export SG_BIN_PATH="${SG_OUTPUT_DIR}/bin"
export LR_BIN_PATH="${LR_BUILD_OUTPUT_DIR}/bin"
export SG_MOCKS_DIR="${MOCKS_DIR}"

mkdir -p "${LR_OUTPUT_DIR}" "${LR_BUILD_OUTPUT_DIR}" "${SG_OUTPUT_DIR}" "${MOCKS_DIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

separator() {
    echo "========================================================================"
}

separator
log "STARTING 32x16 TILED SUBGRID PIPELINE"
log "Output directory: ${OUTPUT_BASE}"
log "HR Reference    : ${HR_SIM_DIR}"
log "Model saves     : ${MODEL_SAVES_DIR}"
separator

# ---------------------------------------------------------------------------
# STEP 1: LR Simulation (0 -> 5 Myr, ISM Cooling)
# ---------------------------------------------------------------------------
separator
log "STEP 1: Running LR 32x16 Simulation (0 -> 5 Myr, ISM cooling)"
separator
cd "${PROJECT_ROOT}/builds/hr_build/src"
./athena -i "${LR_ATHINPUT}" -d "${LR_OUTPUT_DIR}"

if [[ ! -f "${LR_RST_5MYR}" ]]; then
    log "ERROR: 5 Myr restart file not found at ${LR_RST_5MYR}"
    exit 1
fi
log "Confirmed 5 Myr restart file: ${LR_RST_5MYR}"

# ---------------------------------------------------------------------------
# STEP 2: LR ISM Restart (5 -> 10 Myr, ISM Cooling)
# ---------------------------------------------------------------------------
separator
log "STEP 2: Running LR ISM Restart (5 -> 10 Myr)"
separator
cd "${PROJECT_ROOT}/builds/hr_build/src"
./athena \
    -i "${LR_ISM_ATHINPUT}" \
    -d "${LR_BUILD_OUTPUT_DIR}" \
    -r "${LR_RST_5MYR}"

# ---------------------------------------------------------------------------
# STEP 3: Subgrid Tiled CNN Restart (5 -> 10 Myr)
# ---------------------------------------------------------------------------
separator
log "STEP 3: Running Subgrid Tiled CNN Restart (5 -> 10 Myr)"
separator
cd "${PROJECT_ROOT}/builds/subgrid_model/src"

# Setup Python environment for pybind11 in subgrid.cpp
VENV="${PROJECT_ROOT}/venv"
SITE_PACKAGES="${VENV}/lib/python3.14/site-packages"
export PYTHONPATH="${PWD}:${SITE_PACKAGES}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

./athena \
    -i "${SG_ATHINPUT}" \
    -d "${SG_OUTPUT_DIR}" \
    -r "${LR_RST_5MYR}"

# ---------------------------------------------------------------------------
# STEP 4: Benchmark Diagnostics (mock_sg_32x16.py)
# ---------------------------------------------------------------------------
separator
log "STEP 4: Running Diagnostics & Benchmarking with mock_sg_32x16.py"
separator
cd "${PROJECT_ROOT}/data/mocks"
python3 mock_sg_32x16.py

separator
log "32x16 PIPELINE COMPLETE! Plots and animations saved to ${MOCKS_DIR}"
separator
