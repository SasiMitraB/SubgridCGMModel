cd /Volumes/PortableSSD/Projects/SubgridCGMModel/builds/subgrid_model/src

dot_clean -m /Volumes/PortableSSD/Projects/SubgridCGMModel

source /Volumes/PortableSSD/Projects/SubgridCGMModel/venv/bin/activate

dot_clean -m /Volumes/PortableSSD/Projects/SubgridCGMModel

VENV="/Volumes/PortableSSD/Projects/SubgridCGMModel/venv"
SITE_PACKAGES="$VENV/lib/python3.14/site-packages"
export PYTHONPATH="$PWD:$SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"

./athena -i "${1:-neural_network.athinput}" -d /Volumes/PortableSSD/Projects/SubgridCGMModel/simulation_outputs/subgrid_model

dot_clean -m /Volumes/PortableSSD/Projects/SubgridCGMModel