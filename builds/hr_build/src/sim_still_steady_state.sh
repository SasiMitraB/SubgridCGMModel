cd /Volumes/PortableSSD/Projects/SubgridCGMModel/builds/hr_build/src

dot_clean -m /Volumes/PortableSSD/Projects/SubgridCGMModel

./athena -i "${1:-sim_till_steady_state.athinput}" -d /Volumes/PortableSSD/Projects/SubgridCGMModel/simulation_outputs/hr_build

dot_clean -m /Volumes/PortableSSD/Projects/SubgridCGMModel