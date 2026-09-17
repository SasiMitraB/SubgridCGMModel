cd /home/sasi/Projects/SubgridCGMModel/builds/subgrid_model/src

if [ -f /home/sasi/Projects/SubgridCGMModel/venv/bin/activate ]; then
    source /home/sasi/Projects/SubgridCGMModel/venv/bin/activate
fi

SITE_PACKAGES="$(python3 -c 'import site; print(":".join(site.getsitepackages()))')"
export PYTHONPATH="$PWD:$SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"

./athena -i "${1:-neural_network.athinput}" -d /home/sasi/Projects/SubgridCGMModel/simulation_outputs/subgrid_model