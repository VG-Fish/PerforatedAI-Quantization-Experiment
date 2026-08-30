#!/usr/bin/env bash
# Run the text and molecular replacements as one serialized, fully PQAT MPS sweep.
set -euo pipefail

bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd "$bundle_dir/../.." && pwd)"
cd "$repo_dir"

seed_value="${SEED:-0}"
# TextCNN and MPNN use pre-materialized tensor datasets. A single loader process
# removes worker scheduling noise and startup overhead, while MPS executes all
# model work. The package itself seeds Python/NumPy/Torch/MPS before every arm.
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/dqb-mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/private/tmp/dqb-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/private/tmp/dqb-uv-cache}"
export DQB_DATA_NUM_WORKERS=0
export PYTHONHASHSEED="$seed_value"
export PYTORCH_MPS_FAST_MATH=0
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"

run_name="${RUN_NAME:-textcnn_mpnn_pqat}"
run_root="$bundle_dir/$run_name/seed_$seed_value"
results_root="$run_root/results"
comparison_root="$run_root/comparison"
logging_dir="$run_root/logs"

mkdir -p "$logging_dir"
uv run python -c 'import torch; assert torch.backends.mps.is_available(), "MPS is required for this sweep"; assert torch.empty(1, device="mps").device.type == "mps"'
echo "[$(date '+%Y-%m-%d %H:%M:%S')] waiting for active Dynamic12 MPS workers"
while pgrep -f '[d]qb run|[q]ueue_nondendritic_resnet_mps.sh' >/dev/null 2>&1; do
  sleep "${QUEUE_POLL_SECONDS:-30}"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting deterministic TextCNN + MPNN PQAT sweep on MPS"
uv run dqb run \
  --models textcnn mpnn \
  --conditions base_fp32 base_q8 base_q4 base_q2 base_q1_58 base_q1 dendrites_fp32 dendrites_q8 dendrites_q4 dendrites_q2 dendrites_q1_58 dendrites_q1 \
  --seed "$seed_value" \
  --model-scale "${MODEL_SCALE:-1.0}" \
  --allow-PQAT \
  --dynamic-dendritic-training \
  --results-root "$results_root" \
  --comparison-root "$comparison_root" \
  --logging-dir "$logging_dir" \
  --jobs 1

uv run python experiments/dynamic12/scripts/verify_pqat.py \
  --results-root "$results_root" \
  --models textcnn mpnn \
  --conditions base_fp32 base_q8 base_q4 base_q2 base_q1_58 base_q1 dendrites_fp32 dendrites_q8 dendrites_q4 dendrites_q2 dendrites_q1_58 dendrites_q1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] TextCNN + MPNN sweep and PQAT verification complete"
