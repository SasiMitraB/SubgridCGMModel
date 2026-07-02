cd /home/sasi/Projects/SubgridCGMModel/builds/subgrid_model

VENV="/home/sasi/Projects/SubgridCGMModel/venv"
PYTHON_BIN="$VENV/bin/python3"

cmake -S ../../athenak -B . \
	-DCMAKE_BUILD_TYPE=Release \
	-DPROBLEM=subgrid \
	-DPython_EXECUTABLE="$PYTHON_BIN" \
	-DPython_ROOT_DIR="$VENV"
cmake --build . -j$(nproc)

