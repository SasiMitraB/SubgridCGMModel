import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ergane

hr_sim_data = ergane.SimulationData(
    athinp='/Volumes/PortableSSD/Projects/SubgridCGMModel/builds/hr_build/src/kh_radiative_512.athinput',
    datafolder='/Volumes/PortableSSD/Projects/SubgridCGMModel/simulation_outputs/kh_radiative_32'
)

print(hr_sim_data.fields_available)
hr_sim_data.visualize(backend='fastplotlib').show()
