export SUBGRID_DATA_PATH='/Volumes/PortableSSD/Projects/SubgridCGMModel/simulation_outputs/hr_build/bin'
export SUBGRID_CACHE_PATH='/Volumes/PortableSSD/Projects/SubgridCGMModel/simulation_outputs/hr_build/cache'

source /Volumes/PortableSSD/Projects/SubgridCGMModel/venv/bin/activate

dot_clean -m /Volumes/PortableSSD/Projects/SubgridCGMModel

start_time=$SECONDS
python3 models/conv_nn/pdf_cnn.py
duration=$(( SECONDS - start_time ))
echo "Python script took $((duration / 60))m $((duration % 60))s to run."


dot_clean -m /Volumes/PortableSSD/Projects/SubgridCGMModel