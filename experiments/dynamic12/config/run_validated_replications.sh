#!/usr/bin/env bash
# Replicate only the configurations that remain informative after the QAT fix.
set -euo pipefail

bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd "$bundle_dir/../.." && pwd)"
cd "$repo_dir"

export DQB_DATA_NUM_WORKERS="${DQB_DATA_NUM_WORKERS:-0}"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"

read -r -a model_values <<< "${MODEL_KEYS:-resnet18_cifar10 saint_adult pointnet_modelnet40}"
read -r -a condition_values <<< "${CONDITION_KEYS:-base_fp32 dendrites_fp32 base_q8 dendrites_q8 base_q2 dendrites_q2}"
read -r -a seed_values <<< "${SEEDS:-0 1 2}"

model_scale="${MODEL_SCALE:-1.0}"
run_name="${RUN_NAME:-priority_replications}"

for seed_value in "${seed_values[@]}"; do
  run_root="$bundle_dir/$run_name/seed_$seed_value"
  uv run dqb run \
    --models "${model_values[@]}" \
    --conditions "${condition_values[@]}" \
    --seed "$seed_value" \
    --model-scale "$model_scale" \
    --allow-PQAT \
    --dynamic-dendritic-training \
    --results-root "$run_root/results" \
    --comparison-root "$run_root/comparison" \
    --logging-dir "$run_root/logs" \
    --jobs 2
done
