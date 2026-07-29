#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/sasi/Projects/SubgridCGMModel"
BUILD_DIR="${PROJECT_ROOT}/builds/subgrid_vertical"
VENV="${PROJECT_ROOT}/venv"

echo "=== Step 1: Configuring and Building Athena for subgrid_vertical ==="
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

PYTHON_BIN="${VENV}/bin/python3"

cmake -S ../../athenak -B . \
  -DCMAKE_BUILD_TYPE=Release \
  -DPROBLEM=subgrid_vertical \
  -DPython_EXECUTABLE="${PYTHON_BIN}" \
  -DPython_ROOT_DIR="${VENV}"

cmake --build . -j$(nproc)

echo "=== Step 2: Preparing execution directory ==="
# Copy source_module.py and sg_vertical.athinput into src directory where the executable is
cp "${PROJECT_ROOT}/builds/subgrid_model/src/source_module.py" "${BUILD_DIR}/src/source_module.py"
cp "${BUILD_DIR}/sg_vertical.athinput" "${BUILD_DIR}/src/sg_vertical.athinput"

ISM_ATHINPUT="${BUILD_DIR}/src/sg_vertical_ismcooling.athinput"
python3 - <<'PY' "${BUILD_DIR}/src/sg_vertical.athinput" "${ISM_ATHINPUT}"
import pathlib
import re
import sys

source_path = pathlib.Path(sys.argv[1])
target_path = pathlib.Path(sys.argv[2])
text = source_path.read_text()
text = re.sub(r"(^\s*ism_cooling\s*=\s*)false\b", r"\1true", text, flags=re.MULTILINE)
text = re.sub(r"(^\s*user_srcs\s*=\s*)true\b", r"\1false", text, flags=re.MULTILINE)
target_path.write_text(text)
PY

echo "=== Step 3: Running vertical KHI simulation with usersourceterms ==="
cd "${BUILD_DIR}/src"

# Activate virtual environment
if [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"
fi

# Dynamically resolve site-packages directory in the venv
PY_DIR=$(ls -d "${VENV}/lib"/python3.* | head -n 1)
SITE_PACKAGES="${PY_DIR}/site-packages"

export PYTHONPATH="${BUILD_DIR}/src:${SITE_PACKAGES}:${PYTHONPATH:-}"

# Create simulation outputs directory if it does not exist
#mkdir -p "${PROJECT_ROOT}/simulation_outputs/subgrid_vertical_run_usersourceterms"

#./athena -i sg_vertical.athinput -d "${PROJECT_ROOT}/simulation_outputs/subgrid_vertical_run_usersourceterms"

echo "=== Step 4: Running vertical KHI simulation with ISMCooling ==="
mkdir -p "${PROJECT_ROOT}/simulation_outputs/subgrid_vertical_run_ismcooling"

./athena -i sg_vertical_ismcooling.athinput -d "${PROJECT_ROOT}/simulation_outputs/subgrid_vertical_run_ismcooling"

echo "=== Simulation and comparison finished successfully! ==="
