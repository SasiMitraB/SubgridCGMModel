cd /Volumes/PortableSSD/Projects/SubgridCGMModel/builds/subgrid_model

dot_clean -m /Volumes/PortableSSD/Projects/SubgridCGMModel

VENV="/Volumes/PortableSSD/Projects/SubgridCGMModel/venv"
PYTHON_BIN="$VENV/bin/python3"

cmake -S ../../athenak -B . \
	-DCMAKE_BUILD_TYPE=Release \
	-DPROBLEM=subgrid \
	-DPython_EXECUTABLE="$PYTHON_BIN" \
	-DPython_ROOT_DIR="$VENV"
cmake --build . -j$(nproc)

dot_clean -m /Volumes/PortableSSD/Projects/SubgridCGMModel

