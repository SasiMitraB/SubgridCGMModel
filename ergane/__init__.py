"""
ergane
~~~~~~~~~~~~
A library for loading, inspecting, and visualising AthenaK / Athena++
simulation output data.

Quick start
-----------
>>> from ergane import SimulationData
>>> sim = SimulationData(athinp="kh2d/kh2d-sin.athinput",
...                      datafolder="kh2d/outputs")
>>> sim.density[300]
>>> sim.visualize(fields=["density", "pressure"]).show()
"""

from .simulation_data import SimulationData, Frame
from .visualization import Visualization
from .units import Units
from .athinput_parser import parse_athinput
from .vtk_reader import parse_athena_vtk, read_vtk_time

__all__ = [
    "SimulationData",
    "Frame",
    "Visualization",
    "Units",
    "parse_athinput",
    "parse_athena_vtk",
    "read_vtk_time",
]
