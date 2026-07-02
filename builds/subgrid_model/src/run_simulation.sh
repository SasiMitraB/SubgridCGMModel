cd /home/sasi/Projects/SubgridCGMModel/builds/subgrid_model/src

source /home/sasi/Projects/SubgridCGMModel/venv/bin/activate

VENV="/home/sasi/Projects/SubgridCGMModel/venv"
SITE_PACKAGES="$VENV/lib/python3.14/site-packages"
export PYTHONPATH="$PWD:$SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"

./athena -i "${1:-neural_network.athinput}" -d /home/sasi/Projects/SubgridCGMModel/simulation_outputs/subgrid_model