#!/bin/bash
#SBATCH --job-name=sg_plot
# #SBATCH --partition=p.test	
#SBATCH --partition=p.gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:30:00
#SBATCH --output=job.log
#SBATCH --error=job.err

echo "Starting SG Anim..."
python3 -u mocks/mock_sg.py

echo "Done"
