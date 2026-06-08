export SUBGRID_DATA_PATH='/Volumes/PortableSSD/Projects/SubgridCGMModel/simulation_outputs/hr_build/bin'
export SUBGRID_CACHE_PATH='/Volumes/PortableSSD/Projects/SubgridCGMModel/simulation_outputs/hr_build/cache'

source /Volumes/PortableSSD/Projects/SubgridCGMModel/venv/bin/activate

dot_clean -m /Volumes/PortableSSD/Projects/SubgridCGMModel

python3 models/conv_nn/pdf_cnn.py

dot_clean -m /Volumes/PortableSSD/Projects/SubgridCGMModel