#!/usr/bin/env bash
# Queue the nondendritic ResNet control behind any already-running dqb workers.
# The active priority run owns the shared result tree, so this script waits for
# those workers to exit before starting the standard ResNet's six base/PQAT arms.
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
run_name="${RUN_NAME:-priority_replications}"
run_root="$bundle_dir/$run_name/seed_$seed_value"
results_root="$run_root/results"
comparison_root="$run_root/comparison"
logging_dir="$run_root/logs"

mkdir -p "$logging_dir"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] waiting for active dqb workers"
while pgrep -f '[d]qb run' >/dev/null 2>&1; do
  sleep "${QUEUE_POLL_SECONDS:-30}"
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting nondendritic ResNet control on MPS"

uv run dqb run \
  --models resnet18_cifar10 \
  --conditions base_fp32 base_q8 base_q4 base_q2 base_q1_58 base_q1 \
  --seed "$seed_value" \
  --model-scale "${MODEL_SCALE:-1.0}" \
  --allow-PQAT \
  --results-root "$results_root" \
  --comparison-root "$comparison_root" \
  --logging-dir "$logging_dir" \
  --jobs 1

uv run python experiments/dynamic12/scripts/verify_pqat.py \
  --results-root "$results_root" \
  --models resnet18_cifar10 \
  --conditions base_fp32 base_q8 base_q4 base_q2 base_q1_58 base_q1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] nondendritic ResNet control and PQAT verification complete"
