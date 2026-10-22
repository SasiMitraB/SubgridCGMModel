#!/usr/bin/env bash
# Train the cooling-correction stencil model and benchmark it on the held-out frames.
# Everything lands in outputs/cooling_<timestamp>/: train.log, cooling_model.pkl,
# true_vs_pred.png (z inside / outside the window and C; validation) and
# true_vs_pred_train.png, plus mid-slice animations true_vs_pred_slice{,_train}.mp4 and loss.png.
#
# Usage: ./run_cooling_model.sh [training args, e.g. --every 2 --epochs 200 --stencil 3
#                                --gate-model outputs/gate_<timestamp>/gate_model.pkl]
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="$root/outputs/cooling_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out"
cd "$root"

echo "== training  ->  $out"
source /home/sasi/Projects/SubgridCGMModel/venv/bin/activate
python models/cooling_model.py --out "$out/cooling_model.pkl" "$@" 2>&1 | tee "$out/train.log"

echo "== benchmarking"
python benchmarking/plot_true_vs_pred.py --model "$out/cooling_model.pkl" --split val \
    --out "$out/true_vs_pred.png" --anim-out "$out/true_vs_pred_slice.mp4" \
    --loss-out "$out/loss.png" 2>&1 | tee -a "$out/train.log"
python benchmarking/plot_true_vs_pred.py --model "$out/cooling_model.pkl" --split train \
    --out "$out/true_vs_pred_train.png" --anim-out "$out/true_vs_pred_slice_train.mp4" \
    --loss-out "" 2>&1 | tee -a "$out/train.log"

echo "== done: $out"
