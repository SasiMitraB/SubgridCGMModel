#!/bin/bash
#SBATCH --job-name=athena
#SBATCH --output=athena.out
#SBATCH --error=athena.err
#SBATCH --partition=p.test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
# #SBATCH --time=08:00:00

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

export PYTHONPATH=$PWD/python:$PYTHONPATH

# ---- main run ----
# srun ./athena -i kh_cooling_pcunits.athinput -d src16_8/

# ---- restart run ----
srun ./athena -i kh_cooling_pcunits.athinput -d sc16_8/ -r src16_8/rst/KH.00005.rst
