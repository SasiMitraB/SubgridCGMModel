import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ergane

hr_sim_data = ergane.SimulationData(
    athinp='/Volumes/PortableSSD/Projects/SubgridCGMModel/builds/hr_build/src/sim_from_steady_state.athinput',
    datafolder='/Volumes/PortableSSD/Projects/SubgridCGMModel/simulation_outputs/subgrid_model'
)

hr_sim_data.visualize(backend='matplotlib').show()