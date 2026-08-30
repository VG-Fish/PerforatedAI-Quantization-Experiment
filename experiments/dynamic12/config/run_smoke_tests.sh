#!/usr/bin/env bash
# Fast preflight for Dynamic12's three priority models.  Uses one real batch
# per model; it does not train epochs or mutate the benchmark result tree.
set -euo pipefail

bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd "$bundle_dir/../.." && pwd)"
cd "$repo_dir"

export DQB_DATA_NUM_WORKERS="${DQB_DATA_NUM_WORKERS:-0}"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"

uv run python "$bundle_dir/scripts/smoke_models.py" "$@"
uv run python "$bundle_dir/scripts/smoke_pai_targets.py" "$@"
