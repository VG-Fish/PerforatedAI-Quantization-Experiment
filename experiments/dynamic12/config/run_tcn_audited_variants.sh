#!/usr/bin/env bash
# Compare PAI targets only after the retained-dendrite audit is available.
# Each variant is isolated because its PAI state and result artifacts differ.
set -euo pipefail

bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd "$bundle_dir/../.." && pwd)"
cd "$repo_dir"

export DQB_DATA_NUM_WORKERS="${DQB_DATA_NUM_WORKERS:-0}"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"

read -r -a seed_values <<< "${SEEDS:-0 1 2}"
read -r -a variant_values <<< "${PAI_VARIANTS:-default tcn_head_output tcn_head_both}"
read -r -a condition_values <<< "${CONDITION_KEYS:-base_fp32 dendrites_fp32}"

model_scale="${MODEL_SCALE:-0.75}"
run_prefix="${RUN_PREFIX:-tcn_audited}"

for pai_variant in "${variant_values[@]}"; do
  for seed_value in "${seed_values[@]}"; do
    run_root="$bundle_dir/${run_prefix}_${pai_variant}/seed_$seed_value"
    uv run dqb run \
      --models tcn_forecaster \
      --conditions "${condition_values[@]}" \
      --seed "$seed_value" \
      --model-scale "$model_scale" \
      --pai-variant "$pai_variant" \
      --dynamic-dendritic-training \
      --results-root "$run_root/results" \
      --comparison-root "$run_root/comparison" \
      --logging-dir "$run_root/logs" \
      --jobs 1
  done
done
