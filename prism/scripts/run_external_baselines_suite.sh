#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}" # reserved for future methods; current baselines are CPU-based
SEEDS="${SEEDS:-41,42,43,44,45}"
OUT_DIR="${OUT_DIR:-results/external_baselines}"
INCLUDE_XGBOOST="${INCLUDE_XGBOOST:-0}"

METHODS=(
  convnext_logreg
  grouprec_logreg
  convnext_linear_svm
  grouprec_linear_svm
  concat_logreg
  concat_linear_svm
  concat_mlp
  late_fusion_logreg
  lmf_logreg
)

if [[ "$INCLUDE_XGBOOST" == "1" ]]; then
  METHODS+=(xgboost_concat)
fi

METHODS_CSV=$(IFS=, ; echo "${METHODS[*]}")

echo "========================================================================"
echo "PRISM External Baseline Suite"
echo "========================================================================"
echo "GPU (unused now): $GPU"
echo "Seeds: $SEEDS"
echo "Methods: $METHODS_CSV"
echo "Output: $OUT_DIR"

time PYTHONUNBUFFERED=1 pixi run python prism/scripts/run_external_baselines.py \
  --methods "$METHODS_CSV" \
  --seeds "$SEEDS" \
  --output-dir "$OUT_DIR"

echo
echo "Done. Results:"
echo "  - $OUT_DIR/final_results.json"
echo "  - $OUT_DIR/final_results.csv"
