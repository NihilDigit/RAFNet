#!/usr/bin/env bash
set -euo pipefail

# One-click full reproduction runner (NCST-focused).
# Produces pinned features + pinned config overrides + multi-seed training + aggregation.
# Covers all paper models including C+Spatial and GroupRec-3D variants.
#
# Usage:
#   pixi run repro               # via pixi task (recommended)
#   bash prism/scripts/run_repro.sh
#
# Optional env:
#   SUITE=all|main|3d
#   GPU=0
#   SEEDS=41,42,43,44,45
#   CONDA_CUDA=11.1
#   RUN_EXTRACT=1
#   RUN_TRAIN=1
#   RUN_EVAL=1
#   CONVNEXT_BS=1
#   G3D_BS=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SUITE="${SUITE:-all}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-41,42,43,44,45}"
CONDA_CUDA="${CONDA_CUDA:-11.1}"
RUN_EXTRACT="${RUN_EXTRACT:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
CONVNEXT_BS="${CONVNEXT_BS:-1}"
G3D_BS="${G3D_BS:-1}"

DATA_ROOT="data/datasets/ncst_classroom"
IMAGES_DIR="data/datasets/ncst_classroom/images"

EMB_ROOT="output/repro/ncst_grouprec_convnext"
G3D_ROOT="output/repro/ncst_grouprec3d"
G3DM_ROOT="output/repro/ncst_grouprec3d_convnext_merged"
REPRO_CFG_DIR="output/repro/configs"

MAIN_MODELS=(
  grouprec_only
  convnext_only_loss_tuned
  grouprec_convnext_mlp
  grouprec_convnext_gated
  grouprec_convnext_gated_loss_tuned
  grouprec_convnext_cross_attention_loss_tuned
  grouprec_convnext_gated_c_spatial_graph_loss_tuned
)

MODELS_3D=(
  grouprec3d_only
  grouprec3d_convnext_gated_loss_tuned
  grouprec3d_convnext_gated_c_spatial_graph_loss_tuned
)

expect_count() {
  case "$1" in
    train) echo 35259 ;;
    val) echo 8250 ;;
    test) echo 8035 ;;
    *) echo 0 ;;
  esac
}

count_labels() {
  local p="$1"
  pixi run python - "$p" <<'PY'
import pickle, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    print(-1)
    raise SystemExit(0)
with p.open("rb") as f:
    x = pickle.load(f)
print(len(x.get("labels", [])))
PY
}

check_dir_counts() {
  local dir="$1"
  local with_ctx="${2:-0}"
  local ok=1
  for sp in train val test; do
    local base="${dir}/${sp}_features.pkl"
    local n
    n="$(count_labels "$base")"
    local exp
    exp="$(expect_count "$sp")"
    if [[ "$n" != "$exp" ]]; then
      echo "[WARN] invalid split ${sp}: ${base} got=${n} expected=${exp}"
      ok=0
    fi
    if [[ "$with_ctx" == "1" ]]; then
      local c="${dir}/${sp}_features_with_context.pkl"
      if [[ ! -f "$c" ]]; then
        echo "[WARN] missing context split file: $c"
        ok=0
      fi
    fi
  done
  echo "$ok"
}

run_model() {
  local model="$1"
  local cfg="${REPRO_CFG_DIR}/${model}.yaml"
  IFS=',' read -r -a seeds_arr <<< "$SEEDS"
  local i=0
  local total=${#seeds_arr[@]}
  for seed in "${seeds_arr[@]}"; do
    i=$((i + 1))
    echo "[$i/$total] model=${model} seed=${seed}"
    CONDA_OVERRIDE_CUDA="$CONDA_CUDA" \
      pixi run python prism/scripts/train.py \
      --model "$model" \
      --config-override "$cfg" \
      --seed "$seed" \
      --gpu "$GPU"
  done
}

echo "===================================================================="
echo "Reviewer Repro Runner"
echo "suite=${SUITE} seeds=${SEEDS} gpu=${GPU}"
echo "===================================================================="

if [[ "$RUN_EXTRACT" == "1" ]]; then
  echo "== [A1] Extract NCST grouprec+convnext features =="
  pixi run python prism/scripts/extract_features.py \
    --data-root "$DATA_ROOT" \
    --images-dir "$IMAGES_DIR" \
    --modalities grouprec convnext \
    --batch-size 1 \
    --output-root "$EMB_ROOT" \
    --tag repro_emb

  echo "== [A2] Build context sidecar (merged) =="
  pixi run python prism/scripts/build_feature_context_sidecar.py \
    --features-dir "${EMB_ROOT}/latest" \
    --data-root "$DATA_ROOT" \
    --mode merged

  echo "== [A3] Extract NCST GroupRec3D features =="
  pixi run python prism/scripts/extract_grouprec3d_features.py \
    --data-root "$DATA_ROOT" \
    --images-dir "$IMAGES_DIR" \
    --output-root "$G3D_ROOT" \
    --run-tag repro_g3d \
    --batch-size "$G3D_BS" \
    --device "cuda:${GPU}"

  echo "== [A4] Merge ConvNeXt + GroupRec3D features =="
  pixi run python prism/scripts/merge_grouprec3d_convnext_features.py \
    --base-features-dir "${EMB_ROOT}/latest" \
    --grouprec3d-features-dir "${G3D_ROOT}/latest" \
    --output-dir "${G3DM_ROOT}/latest" \
    --clean-output
  pixi run python prism/scripts/merge_grouprec3d_convnext_features.py \
    --base-features-dir "${EMB_ROOT}/latest" \
    --grouprec3d-features-dir "${G3D_ROOT}/latest" \
    --output-dir "${G3DM_ROOT}/latest" \
    --with-context
fi

echo "== [B] Validate feature signatures =="
if [[ "$(check_dir_counts "${EMB_ROOT}/latest" 1)" != "1" ]]; then
  echo "[ERROR] invalid embedding features under ${EMB_ROOT}/latest"
  exit 1
fi
if [[ "$(check_dir_counts "${G3D_ROOT}/latest" 0)" != "1" ]]; then
  echo "[ERROR] invalid grouprec3d features under ${G3D_ROOT}/latest"
  exit 1
fi
if [[ "$(check_dir_counts "${G3DM_ROOT}/latest" 1)" != "1" ]]; then
  echo "[ERROR] invalid merged grouprec3d+convnext features under ${G3DM_ROOT}/latest"
  exit 1
fi

echo "== [C] Generate pinned reproducibility configs =="
pixi run python prism/scripts/make_repro_configs.py \
  --out-dir "$REPRO_CFG_DIR" \
  --emb-features-dir "${EMB_ROOT}/latest" \
  --g3d-features-dir "${G3D_ROOT}/latest" \
  --g3d-merged-features-dir "${G3DM_ROOT}/latest"

if [[ "$RUN_TRAIN" == "1" ]]; then
  if [[ "$SUITE" == "all" || "$SUITE" == "main" ]]; then
    echo "== [D1] Train main suite =="
    for model in "${MAIN_MODELS[@]}"; do
      run_model "$model"
    done
  fi
  if [[ "$SUITE" == "all" || "$SUITE" == "3d" ]]; then
    echo "== [D2] Train 3D compatibility suite =="
    for model in "${MODELS_3D[@]}"; do
      run_model "$model"
    done
  fi
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  models_csv="grouprec_only,convnext_only_loss_tuned,grouprec_convnext_mlp,grouprec_convnext_gated,grouprec_convnext_gated_loss_tuned,grouprec_convnext_cross_attention_loss_tuned,grouprec_convnext_gated_c_spatial_graph_loss_tuned,grouprec3d_only,grouprec3d_convnext_gated_loss_tuned,grouprec3d_convnext_gated_c_spatial_graph_loss_tuned"
  echo "== [E] Aggregate =="
  pixi run python prism/scripts/evaluate.py --models "$models_csv"
  echo "[DONE] results/evaluation/final_results.{json,csv}"
fi

echo "Repro run finished."

