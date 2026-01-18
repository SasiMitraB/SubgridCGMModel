#!/bin/bash
#SBATCH --job-name=athena
#SBATCH --output=athena.out
#SBATCH --error=athena.err
#SBATCH --partition=p.test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:30:00

module purge
module load gcc/10

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

export PYTHONPATH=$PWD/python:$PYTHONPATH

# ---- main run ----
# ./athena -i kh_cooling_pcunits.athinput -d rc512_256/

# ---- restart run ----
./athena -i kh_cooling_pcunits.athinput -d c2512_256/ -r rc512_256/rst/KH.00005.rst
