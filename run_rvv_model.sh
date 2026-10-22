#!/usr/bin/env bash
# Train the R (subgrid momentum flux) stencil model and benchmark it on the held-out frames.
# Everything lands in outputs/rvv_<timestamp>/: train.log, rvv_tensor_model.pkl,
# true_vs_pred.png (f, S; validation), tensor_true_vs_pred.png (2 f S_ij; validation),
# temperature_true_vs_pred.png (T_CG; validation),
# their *_train.png counterparts, mid-slice animations true_vs_pred_slice{,_train}.mp4
# and loss.png.
#
# Usage: ./run_rvv_model.sh [training args, e.g. --every 2 --epochs 200 --w-shape 10]
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="$root/outputs/rvv_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out"
cd "$root"

echo "== training  ->  $out"
source /home/sasi/Projects/SubgridCGMModel/venv/bin/activate
python models/rvv_tensor_model.py --stencil 3 --out "$out/rvv_tensor_model.pkl" "$@" 2>&1 | tee "$out/train.log"

echo "== benchmarking"
python benchmarking/plot_rvv_true_vs_pred.py --model "$out/rvv_tensor_model.pkl" --split val \
    --out "$out/true_vs_pred.png" --tensor-out "$out/tensor_true_vs_pred.png" \
    --temp-out "$out/temperature_true_vs_pred.png" \
    --anim-out "$out/true_vs_pred_slice.mp4" --loss-out "$out/loss.png" 2>&1 | tee -a "$out/train.log"
python benchmarking/plot_rvv_true_vs_pred.py --model "$out/rvv_tensor_model.pkl" --split train \
    --out "$out/true_vs_pred_train.png" --tensor-out "$out/tensor_true_vs_pred_train.png" \
    --temp-out "$out/temperature_true_vs_pred_train.png" \
    --anim-out "$out/true_vs_pred_slice_train.mp4" --loss-out "" 2>&1 | tee -a "$out/train.log"

echo "== done: $out"
