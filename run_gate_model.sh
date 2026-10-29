#!/usr/bin/env bash
# Train the mixing-layer gate stencil model and benchmark it on the held-out frames.
# Everything lands in outputs/gate_<timestamp>/: train.log, gate_model.pkl,
# true_vs_pred.png (confusion matrix, probabilities, precision-recall, ROC, recall vs f_mix;
# validation), its *_train.png counterpart, true_vs_pred{,_train}_fmix.{png,csv} (TN / FP / FN / TP
# against f_mix), mid-slice animations true_vs_pred_slice{,_train}.mp4 and loss.png.
#
# Usage: ./run_gate_model.sh [training args, e.g. --every 2 --epochs 200 --min-frac 1e-3]
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="$root/outputs/gate_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out"
cd "$root"

echo "== training  ->  $out"
source /home/sasi/Projects/SubgridCGMModel/venv/bin/activate
python models/gate_model.py --stencil 3 --out "$out/gate_model.pkl" "$@" 2>&1 | tee "$out/train.log"

echo "== benchmarking"
python benchmarking/plot_gate_true_vs_pred.py --model "$out/gate_model.pkl" --split val \
    --out "$out/true_vs_pred.png" --anim-out "$out/true_vs_pred_slice.mp4" \
    --loss-out "$out/loss.png" 2>&1 | tee -a "$out/train.log"
python benchmarking/plot_gate_true_vs_pred.py --model "$out/gate_model.pkl" --split train \
    --out "$out/true_vs_pred_train.png" --anim-out "$out/true_vs_pred_slice_train.mp4" \
    --loss-out "" 2>&1 | tee -a "$out/train.log"

echo "== done: $out"
