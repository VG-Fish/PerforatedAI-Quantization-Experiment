#!/usr/bin/env bash
# Run every informative FP32 dependency and quantized PQAT condition.
#
# The ResNet comparison is deliberately paired rather than treated as a
# second dynamic-PAI sweep:
#   * resnet18_cifar10 supplies the base_fp32/base_q* control arms.
#   * resnet18_hf_perforated_cifar10 supplies the already-perforated (dendritic)
#     counterpart, stored under its base_fp32/base_q* keys.
#
# The generic MODEL_KEYS/CONDITION_KEYS override remains available for broad
# exploratory sweeps.  With no overrides, the paired ResNet groups below avoid
# comparing the HF checkpoint to a newly discovered second PAI graph.
set -euo pipefail

bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd "$bundle_dir/../.." && pwd)"
cd "$repo_dir"

export DQB_DATA_NUM_WORKERS="${DQB_DATA_NUM_WORKERS:-0}"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"

read -r -a model_values <<< "${MODEL_KEYS:-}"
read -r -a condition_values <<< "${CONDITION_KEYS:-}"
read -r -a base_resnet_models <<< "${BASE_RESNET_MODEL_KEYS:-resnet18_cifar10}"
read -r -a perforated_resnet_models <<< "${PERFORATED_RESNET_MODEL_KEYS:-resnet18_hf_perforated_cifar10}"
read -r -a priority_models <<< "${PRIORITY_MODEL_KEYS:-saint_adult pointnet_modelnet40}"

base_conditions=(base_fp32 base_q8 base_q4 base_q2 base_q1_58 base_q1)
priority_conditions=(
  base_fp32 base_q8 base_q4 base_q2 base_q1_58 base_q1
  dendrites_fp32 dendrites_q8 dendrites_q4 dendrites_q2 dendrites_q1_58 dendrites_q1
)
all_conditions=("${priority_conditions[@]}")
read -r -a seed_values <<< "${SEEDS:-0 1 2}"

model_scale="${MODEL_SCALE:-1.0}"
run_name="${RUN_NAME:-priority_replications}"

for seed_value in "${seed_values[@]}"; do
  run_root="$bundle_dir/$run_name/seed_$seed_value"
  run_group() {
    local group_models=()
    read -r -a group_models <<< "$1"
    shift
    local group_conditions=("$@")
    uv run dqb run \
      --models "${group_models[@]}" \
      --conditions "${group_conditions[@]}" \
      --seed "$seed_value" \
      --model-scale "$model_scale" \
      --allow-PQAT \
      --dynamic-dendritic-training \
      --results-root "$run_root/results" \
      --comparison-root "$run_root/comparison" \
      --logging-dir "$run_root/logs" \
      --jobs 2

    uv run python experiments/dynamic12/scripts/verify_pqat.py \
      --results-root "$run_root/results" \
      --models "${group_models[@]}" \
      --conditions "${group_conditions[@]}"
  }

  if [[ -n "${MODEL_KEYS:-}" || -n "${CONDITION_KEYS:-}" ]]; then
    [[ -n "${MODEL_KEYS:-}" ]] || model_values=(resnet18_cifar10 resnet18_hf_perforated_cifar10 saint_adult pointnet_modelnet40)
    [[ -n "${CONDITION_KEYS:-}" ]] || condition_values=("${all_conditions[@]}")
    run_group "${model_values[*]}" "${condition_values[@]}"
  else
    # Keep the standard and published-perforated ResNet arms separate.  The
    # HF key is already perforated, so its base_* records are the dendritic
    # counterpart for this paired comparison.
    run_group "${base_resnet_models[*]}" "${base_conditions[@]}"
    run_group "${perforated_resnet_models[*]}" "${base_conditions[@]}"
    run_group "${priority_models[*]}" "${priority_conditions[@]}"
  fi
done
