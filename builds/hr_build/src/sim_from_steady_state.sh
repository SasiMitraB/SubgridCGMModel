cd /home/sasi/Projects/SubgridCGMModel/builds/hr_build/src

mkdir -p /home/sasi/Projects/SubgridCGMModel/simulation_outputs/lr_build_steady_state

./athena -i sim_from_steady_state.athinput -d /home/sasi/Projects/SubgridCGMModel/simulation_outputs/hr_build_steady_state -r /home/sasi/Projects/SubgridCGMModel/simulation_outputs/lr_build/rst/KH.00005.rst