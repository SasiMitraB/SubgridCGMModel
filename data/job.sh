#!/bin/bash
# #SBATCH --job-name=pdf_plot
# #SBATCH --partition=p.24h
# #SBATCH --partition=p.test	
#SBATCH --partition=p.gpu
#SBATCH --gres=gpu:2
# #SBATCH --partition=p.gpu.ampere
# #SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:30:00
#SBATCH --output=job.log
#SBATCH --error=job.err

rm mocks/pdf/log_pdf_animation.mp4
rm mocks/pdf/log_pdf_compare_animation.mp4

echo "Starting SG Anim..."
python3 -u mocks/log_plot.py

echo "Converting gif to mp4..."
module load ffmpeg
ffmpeg -i mocks/pdf/pdf_animation.gif -movflags faststart -pix_fmt yuv420p mocks/pdf/log_pdf_animation.mp4
ffmpeg -i mocks/pdf/pdf_compare_animation.gif -movflags faststart -pix_fmt yuv420p mocks/pdf/log_pdf_compare_animation.mp4

echo "Deleting GIFs..."
rm mocks/pdf/pdf_animation.gif mocks/pdf/log_pdf_animation.gif
rm mocks/pdf/pdf_animation.gif mocks/pdf/log_pdf_compare_animation.gif

echo "Conversion done."

