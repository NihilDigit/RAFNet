#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DEVICE="${DEVICE:-cpu}"
REPEATS="${REPEATS:-3}"
WARMUP="${WARMUP:-1}"
SEED="${SEED:-best}"
FEATURES_DIR="${FEATURES_DIR:-}"

MODELS=(
  "convnext_only_loss_tuned"
  "grouprec_convnext_gated_loss_tuned"
  "grouprec_convnext_gated_c_spatial_graph_loss_tuned"
)

for model in "${MODELS[@]}"; do
  echo "== Tier2 diagnostics: $model =="
  CMD=(pixi run python prism/scripts/tier2_diagnostics.py
    --model "$model" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --repeats "$REPEATS" \
    --warmup "$WARMUP")
  if [[ -n "$FEATURES_DIR" ]]; then
    CMD+=(--features-dir "$FEATURES_DIR")
  fi
  "${CMD[@]}"
done

echo "Tier2 outputs written under: results/evaluation/tier2/"
