#!/usr/bin/env bash
# Calibrate SAINT's complete classifier-head target after existing MPS queues.
# Its dendritic PQAT descendants remain deliberately unqueued until this source
# artifact proves that PAI retained an insertion in the final topology.
set -euo pipefail

bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd "$bundle_dir/../.." && pwd)"
cd "$repo_dir"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/dqb-mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/private/tmp/dqb-cache}"
export DQB_DATA_NUM_WORKERS="${DQB_DATA_NUM_WORKERS:-0}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"

seed_value="${SEED:-0}"
run_name="${RUN_NAME:-saint_head_fixed100}"
run_root="$bundle_dir/$run_name/seed_$seed_value"
results_root="$run_root/results"
comparison_root="$run_root/comparison"
logging_dir="$run_root/logs"

mkdir -p "$logging_dir"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] waiting for active Dynamic12 MPS queues"
# The nondendritic ResNet queue is already waiting. Keeping this queue behind
# it prevents a check-then-start race when the active PointNet worker exits.
while pgrep -f '[d]qb run|[q]ueue_nondendritic_resnet_mps.sh' >/dev/null 2>&1; do
  sleep "${QUEUE_POLL_SECONDS:-30}"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting SAINT classifier-head calibration on MPS"
uv run dqb run \
  --models saint_adult \
  --conditions base_fp32 dendrites_fp32 \
  --seed "$seed_value" \
  --model-scale "${MODEL_SCALE:-1.0}" \
  --dynamic-dendritic-training \
  --results-root "$results_root" \
  --comparison-root "$comparison_root" \
  --logging-dir "$logging_dir" \
  --jobs 1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] SAINT calibration complete; inspect dendrite_audit.csv before PQAT"
