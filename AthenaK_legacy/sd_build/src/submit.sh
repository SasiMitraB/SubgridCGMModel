#!/bin/bash
#SBATCH --job-name=athena
#SBATCH --output=athena.out
#SBATCH --error=athena.err
#SBATCH --partition=p.test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:30:00

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

export PYTHONPATH=$PWD/python:$PYTHONPATH

#./athena -i sg.athinput -d rh16_8/
./athena -i sg.athinput -d ch_16_8/ -r rh16_8/rst/KH.00005.rst
