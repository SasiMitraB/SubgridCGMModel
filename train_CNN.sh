export SUBGRID_DATA_PATH='/Volumes/PortableSSD/Projects/SubgridCGMModel/simulation_outputs/hr_build_steady_state/bin'
export SUBGRID_CACHE_PATH='/Volumes/PortableSSD/Projects/SubgridCGMModel/simulation_outputs/hr_build_steady_state/cache'

source /Volumes/PortableSSD/Projects/SubgridCGMModel/venv/bin/activate

dot_clean -m /Volumes/PortableSSD/Projects/SubgridCGMModel

python3 models/conv_nn/log_cnn.py

dot_clean -m /Volumes/PortableSSD/Projects/SubgridCGMModel