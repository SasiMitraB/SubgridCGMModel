#!/bin/bash
#SBATCH --job-name=cnn_train
#SBATCH --partition=p.test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:30:00
#SBATCH --output=job.log
#SBATCH --error=job.err

echo "Starting CNN training..."
python3 -u all_flux_cnn.py

echo "Job complete."
