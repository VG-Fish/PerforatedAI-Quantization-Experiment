#!/usr/bin/env bash
# Train every base_fp32 baseline, split across four parallel `dqb run` processes,
# and report progress while they run.
#
# Sequential, this sweep is ~24h: ResNet-18 alone costs ~155s/epoch x 200 epochs.
# Contention between concurrent runs on this machine is mild (ResNet-18 held
# 3.80 -> 3.61 batch/s with a second job alongside), because training is
# MPS-compute-bound rather than data-bound, so splitting cuts wall-clock close to
# linearly. Streams are grouped so the two long-pole CIFAR models run alone.
#
# --ignore-saved-models is deliberate, not a convenience: the model definitions
# changed (Cora ego-graph construction), so any surviving record would compare a
# new implementation against an old checkpoint.
#
# Usage:
#   ./run_base_sweep.sh              launch, then print progress every 60s
#   ./run_base_sweep.sh -i 120       ...at a different interval
#   ./run_base_sweep.sh --fresh      delete stale epoch checkpoints first
#   ./run_base_sweep.sh --detach     launch and exit immediately
#   ./run_base_sweep.sh --status     report on already-running streams, then exit
#
# Ctrl-C stops the progress display only; training keeps running (each stream
# ignores SIGINT/SIGHUP). Use `pkill -f 'dqb run'` to actually stop training.
set -uo pipefail

cd "$(dirname "$0")"

RESULTS_DIR="updated_models"
LOG_DIR="logs_tuned/streams"
INTERVAL=60
MODE="watch"
FRESH=0

STREAM_CIFAR=(resnet18_cifar10 mobilenetv2_cifar10)
STREAM_NLP=(gcn distilbert m5 textcnn tabnet saint_adult mpnn
            attentivefp_freesolv gin_imdbb vae_mnist)
STREAM_HEAVY=(capsnet_mnist snn_nmnist pointnet_modelnet40)
STREAM_LIGHT=(lenet5 lstm_forecaster tcn_forecaster gru_forecaster
              lstm_autoencoder actor_critic dqn_lunarlander ppo_bipedalwalker)
ALL_MODELS=("${STREAM_CIFAR[@]}" "${STREAM_NLP[@]}" "${STREAM_HEAVY[@]}" "${STREAM_LIGHT[@]}")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --detach)   MODE="detach"; shift ;;
    --status)   MODE="status"; shift ;;
    --wait)     MODE="watch";  shift ;;
    --fresh)    FRESH=1; shift ;;
    -i|--interval) INTERVAL="${2:-60}"; shift 2 ;;
    -h|--help)  sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------- progress ---
# Parsed in Python rather than shell: the interesting state lives in tqdm bars
# that are rewritten in place with \r, and the postfix carries the running best
# metric. Extracting that with grep/sed is far more fragile than it looks.
report() {
  python3 - "$LOG_DIR" "$RESULTS_DIR" <<'PY'
import re, sys, time
from pathlib import Path

log_dir, results_dir = Path(sys.argv[1]), sys.argv[2]
ANSI  = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
EPOCH = re.compile(r"^(\S+) \| (\S+):\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+) \[([^<\]]+)<([^,\]]+),\s*(.*)\]\s*$")
DONE  = re.compile(r"\[done\] (\S+) / (\S+) [—-]+ (.+?): (-?[\d.]+)")
START = re.compile(r"\[train\] (\S+) / (\S+) [—-]+ starting")
BEST  = re.compile(r"best_\w+=(-?[\d.]+)")

ORDER = ["cifar", "nlp_graph", "heavy", "light"]
logs = sorted(log_dir.glob("*.log"), key=lambda p: ORDER.index(p.stem) if p.stem in ORDER else 99)
if not logs:
    print("  (no stream logs yet)")
    raise SystemExit

total_done = 0
print(f"  {'stream':<10} {'model':<22} {'epoch':>10} {'%':>4}  {'remaining':>10}  {'best':>9}")
print(f"  {'-'*10} {'-'*22} {'-'*10} {'-'*4}  {'-'*10}  {'-'*9}")

for log in logs:
    text = ANSI.sub("", log.read_text(errors="replace")).replace("\r", "\n")
    lines = text.splitlines()

    done = DONE.findall(text)
    total_done += len(done)

    current, last_epoch = None, None
    for line in lines:
        m = START.search(line)
        if m:
            current, last_epoch = m.group(1), None
        m = EPOCH.match(line.strip())
        if m:
            last_epoch = m
    finished = {d[0] for d in done}
    if current in finished:
        current = None

    if current is None:
        state = "all queued models finished" if done else "starting…"
        print(f"  {log.stem:<10} {state}")
    elif last_epoch is None:
        print(f"  {log.stem:<10} {current:<22} {'—':>10} {'—':>4}  {'warming up':>10}  {'—':>9}")
    else:
        _, _, pct, cur, tot, _elapsed, remaining, postfix = last_epoch.groups()
        best = BEST.search(postfix)
        print(f"  {log.stem:<10} {current:<22} {cur+'/'+tot:>10} {pct+'%':>4}  "
              f"{remaining:>10}  {(best.group(1) if best else '—'):>9}")

    for model, _cond, metric, value in done:
        print(f"  {'':<10}   ✓ {model:<20} {metric}: {value}")

print()
print(f"  {total_done}/23 models complete   ({time.strftime('%H:%M:%S')})   -> results/{results_dir}")
PY
}

running() { pgrep -f "dqb run" >/dev/null 2>&1; }

watch_loop() {
  trap 'echo; echo "detached — training continues. stop with: pkill -f \"dqb run\""; exit 0' INT
  while :; do
    echo
    echo "=============================================================================="
    report
    running || { echo; echo "all streams finished."; break; }
    sleep "$INTERVAL"
  done
}

# ------------------------------------------------------------------ status ---
if [[ "$MODE" == "status" ]]; then
  report
  running || echo "  (no dqb run processes active)"
  exit 0
fi

# ------------------------------------------------------------------ launch ---
# shellcheck disable=SC1091
source .venv/bin/activate
mkdir -p "$LOG_DIR"

if running; then
  echo "refusing to launch: 'dqb run' is already active — two runs would race on the" >&2
  echo "same results/$RESULTS_DIR paths. Use --status to watch, or pkill -f 'dqb run'." >&2
  exit 1
fi

# --ignore-saved-models does NOT prevent epoch-level resume: training.py calls
# _load_epoch_checkpoint() unconditionally whenever the output dir exists, and
# never consults that flag. The flag only stops a finished record.json from
# causing a skip. So after any model-definition change, a leftover
# epoch_checkpoint.pt would silently continue an old-architecture run.
stale=()
for model in "${ALL_MODELS[@]}"; do
  ckpt="results/$RESULTS_DIR/$model/base_fp32/epoch_checkpoint.pt"
  [[ -f "$ckpt" ]] && stale+=("$model")
done

if (( ${#stale[@]} )); then
  if (( FRESH )); then
    echo "removing ${#stale[@]} stale epoch checkpoint(s): ${stale[*]}"
    for model in "${stale[@]}"; do
      rm -rf "results/$RESULTS_DIR/$model/base_fp32"
    done
  else
    echo "WARNING: ${#stale[@]} model(s) have an epoch_checkpoint.pt and WILL resume" >&2
    echo "  mid-run rather than start clean: ${stale[*]}" >&2
    echo "  --ignore-saved-models does not override this. If the model definitions" >&2
    echo "  changed since those checkpoints, the run would be invalid." >&2
    echo "  Re-run with --fresh to delete them first, or accept the resume." >&2
    echo >&2
  fi
fi

stream() {
  local name="$1"; shift
  # The subshell ignores SIGINT/SIGHUP so Ctrl-C in the watcher (and closing the
  # terminal) leaves training running.
  ( trap '' INT HUP
    exec dqb run \
      --results-root results \
      --results-directory "$RESULTS_DIR" \
      --logging-dir logs_tuned \
      --conditions base_fp32 \
      --ignore-saved-models \
      --models "$@" \
      >"$LOG_DIR/$name.log" 2>&1
  ) &
  printf '  %-10s pid %-7s %s\n' "$name" "$!" "$*"
}

echo "launching 4 parallel streams -> results/$RESULTS_DIR"
: >"$LOG_DIR/cifar.log"; : >"$LOG_DIR/nlp_graph.log"
: >"$LOG_DIR/heavy.log"; : >"$LOG_DIR/light.log"

stream cifar     "${STREAM_CIFAR[@]}"
stream nlp_graph "${STREAM_NLP[@]}"
stream heavy     "${STREAM_HEAVY[@]}"
stream light     "${STREAM_LIGHT[@]}"

cat <<EOF

logs:    tail -f $LOG_DIR/*.log
status:  ./run_base_sweep.sh --status
stop:    pkill -f 'dqb run'
EOF

if [[ "$MODE" == "detach" ]]; then
  exit 0
fi

echo
echo "progress every ${INTERVAL}s — Ctrl-C detaches without stopping training"
watch_loop
