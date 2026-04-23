#!/usr/bin/env bash
set -euo pipefail

# One-click main paper suite (Table 1 subset — no C+Spatial, no GroupRec-3D).
# 6 settings × 5 seeds = 30 trainings, then unified aggregation.
#
# Settings:
#  1) grouprec_only
#  2) convnext_only
#  3) grouprec_convnext_mlp
#  4) grouprec_convnext_gated
#  5) grouprec_convnext_gated_loss_tuned
#  6) convnext_only (with config override: convnext_only_loss_tuned.yaml)
#
# For the full suite including C+Spatial and 3D variants, use run_repro.sh.
#
# Usage:
#   pixi run repro-main          # via pixi task (recommended)
#   bash prism/scripts/run_repro_main.sh
#
# Optional env overrides:
#   GPU=0
#   SEEDS="41,42,43,44,45"
#   CONDA_CUDA="11.1"
#   RETRY=1
#   RUN_EVAL=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

GPU="${GPU:-0}"
SEEDS="${SEEDS:-41,42,43,44,45}"
CONDA_CUDA="${CONDA_CUDA:-11.1}"
RETRY="${RETRY:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

# spec format: "output_model_name|train_model_arg|optional_config_override"
SPECS=(
  "grouprec_only|grouprec_only|"
  "convnext_only|convnext_only|"
  "grouprec_convnext_mlp|grouprec_convnext_mlp|"
  "grouprec_convnext_gated|grouprec_convnext_gated|"
  "grouprec_convnext_gated_loss_tuned|grouprec_convnext_gated_loss_tuned|"
  "convnext_only_loss_tuned|convnext_only|prism/configs/convnext_only_loss_tuned.yaml"
)

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="results/paper_runs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/paper_full_suite_${RUN_TS}.log"

IFS=',' read -r -a SEED_ARR <<< "$SEEDS"
TOTAL=$(( ${#SPECS[@]} * ${#SEED_ARR[@]} ))
DONE=0
FAIL=0

echo "====================================================================" | tee -a "$LOG_FILE"
echo "Paper Full Suite Runner" | tee -a "$LOG_FILE"
echo "ROOT_DIR: $ROOT_DIR" | tee -a "$LOG_FILE"
echo "GPU: $GPU" | tee -a "$LOG_FILE"
echo "SEEDS: $SEEDS" | tee -a "$LOG_FILE"
echo "TOTAL TASKS: $TOTAL (6 settings × ${#SEED_ARR[@]} seeds)" | tee -a "$LOG_FILE"
echo "LOG: $LOG_FILE" | tee -a "$LOG_FILE"
echo "====================================================================" | tee -a "$LOG_FILE"

run_train_once() {
  local model_arg="$1"
  local seed="$2"
  local cfg_override="$3"

  if [[ -n "$cfg_override" ]]; then
    CONDA_OVERRIDE_CUDA="$CONDA_CUDA" \
      pixi run python prism/scripts/train.py \
      --model "$model_arg" \
      --config-override "$cfg_override" \
      --seed "$seed" \
      --gpu "$GPU"
  else
    CONDA_OVERRIDE_CUDA="$CONDA_CUDA" \
      pixi run python prism/scripts/train.py \
      --model "$model_arg" \
      --seed "$seed" \
      --gpu "$GPU"
  fi
}

for spec in "${SPECS[@]}"; do
  IFS='|' read -r out_model model_arg cfg_override <<< "$spec"
  for seed in "${SEED_ARR[@]}"; do
    DONE=$((DONE + 1))
    echo "" | tee -a "$LOG_FILE"
    echo "[$DONE/$TOTAL] out_model=$out_model model_arg=$model_arg seed=$seed" | tee -a "$LOG_FILE"
    if [[ -n "$cfg_override" ]]; then
      echo "  using config override: $cfg_override" | tee -a "$LOG_FILE"
    fi

    set +e
    run_train_once "$model_arg" "$seed" "$cfg_override" 2>&1 | tee -a "$LOG_FILE"
    status=$?
    set -e

    if [[ $status -ne 0 ]]; then
      echo "  -> failed (exit=$status), retry=${RETRY}" | tee -a "$LOG_FILE"
      if [[ "$RETRY" -ge 1 ]]; then
        set +e
        run_train_once "$model_arg" "$seed" "$cfg_override" 2>&1 | tee -a "$LOG_FILE"
        status=$?
        set -e
      fi
    fi

    if [[ $status -ne 0 ]]; then
      FAIL=$((FAIL + 1))
      echo "  -> final: FAIL out_model=$out_model seed=$seed" | tee -a "$LOG_FILE"
    else
      echo "  -> final: OK out_model=$out_model seed=$seed" | tee -a "$LOG_FILE"
    fi
  done
done

echo "" | tee -a "$LOG_FILE"
echo "====================================================================" | tee -a "$LOG_FILE"
echo "Training completed: total=$TOTAL fail=$FAIL ok=$((TOTAL - FAIL))" | tee -a "$LOG_FILE"
echo "====================================================================" | tee -a "$LOG_FILE"

if [[ "$RUN_EVAL" == "1" ]]; then
  echo "" | tee -a "$LOG_FILE"
  echo "[EVAL] Aggregating paper models..." | tee -a "$LOG_FILE"
  PAPER_MODELS="grouprec_only,convnext_only,convnext_only_loss_tuned,grouprec_convnext_mlp,grouprec_convnext_gated,grouprec_convnext_gated_loss_tuned"
  set +e
  CONDA_OVERRIDE_CUDA="$CONDA_CUDA" \
    pixi run python prism/scripts/evaluate.py --models "$PAPER_MODELS" 2>&1 | tee -a "$LOG_FILE"
  eval_status=$?
  set -e
  if [[ $eval_status -ne 0 ]]; then
    echo "[EVAL] failed (exit=$eval_status)" | tee -a "$LOG_FILE"
  else
    echo "[EVAL] done. See results/evaluation/final_results.csv" | tee -a "$LOG_FILE"
  fi
fi

echo "" | tee -a "$LOG_FILE"
echo "Run log saved to: $LOG_FILE" | tee -a "$LOG_FILE"

if [[ $FAIL -ne 0 ]]; then
  exit 1
fi

