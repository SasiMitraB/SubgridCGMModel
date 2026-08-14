#!/bin/bash
export SUBGRID_DATA_PATH='/home/sasi/Projects/SubgridCGMModel/simulation_outputs/hr_build_1024/bin'
export SUBGRID_CACHE_PATH='/home/sasi/Projects/SubgridCGMModel/simulation_outputs/hr_build_1024/cache'
export PDF_CNN_RESOLUTION='1024,512'
export PDF_CNN_DOWNSAMPLE='64'

. /home/sasi/Projects/SubgridCGMModel/venv/bin/activate

start_time=$SECONDS
python3 models/conv_nn/pdf_cnn.py
duration=$(( SECONDS - start_time ))
echo "Python script took $((duration / 60))m $((duration % 60))s to run."

python3 data/mocks/pdf_plot.py