#!/bin/bash
export SUBGRID_DATA_PATH='/home/sasi/Projects/SubgridCGMModel/simulation_outputs/hr_build_512/bin'
export SUBGRID_CACHE_PATH='/home/sasi/Projects/SubgridCGMModel/simulation_outputs/hr_build_512/cache'
export PDF_CNN_RESOLUTION='512,256'
export PDF_CNN_DOWNSAMPLE='32'

. /home/sasi/Projects/SubgridCGMModel/venv/bin/activate

start_time=$SECONDS
python3 -u models/conv_nn/naive_cnn.py
duration=$(( SECONDS - start_time ))
echo "Python script took $((duration / 60))m $((duration % 60))s to run."
python3 -u data/mocks/naive_pdf_plot.py
