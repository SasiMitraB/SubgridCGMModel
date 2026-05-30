cd /Volumes/PortableSSD/Projects/SubgridCGMModel/builds/hr_build/src

mkdir -p /Volumes/PortableSSD/Projects/SubgridCGMModel/simulation_outputs/hr_build_steady_state

./athena -i sim_from_steady_state.athinput -d /Volumes/PortableSSD/Projects/SubgridCGMModel/simulation_outputs/hr_build_steady_state -r /Volumes/PortableSSD/Projects/SubgridCGMModel/simulation_outputs/hr_build/rst/KH.00005.rst

dot_clean -m /Volumes/PortableSSD/Projects/SubgridCGMModel