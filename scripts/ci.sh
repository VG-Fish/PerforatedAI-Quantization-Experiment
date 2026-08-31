#!/usr/bin/env bash
# The full check suite, in the order that fails fastest and cheapest.
#
# Everything here is offline and CPU-only: no dataset is downloaded, no model is
# built, and no result tree is written. Run it before pushing; CI runs exactly
# this script so a green local run means a green CI run.
set -euo pipefail

cd "$(dirname "$0")/.."

if command -v uv >/dev/null 2>&1; then
    run() { uv run "$@"; }
else
    # Fall back to the project virtualenv for environments without uv.
    run() { ".venv/bin/$1" "${@:2}"; }
fi

echo "== ty check =="
run ty check

echo "== pytest =="
run pytest

echo "== generated documentation is current =="
run dqb docs --check

echo "All checks passed."
