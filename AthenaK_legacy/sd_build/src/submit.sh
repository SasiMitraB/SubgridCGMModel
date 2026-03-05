#!/bin/bash
#SBATCH --job-name=athena
#SBATCH --output=athena.out
#SBATCH --error=athena.err
#SBATCH --partition=p.test
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

export PYTHONPATH=$PWD/python:$PYTHONPATH

#./athena -i sg.athinput -d rc16_8/
./athena -i sg.athinput -d lc16_8/ -r rc16_8/rst/KH.00005.rst
