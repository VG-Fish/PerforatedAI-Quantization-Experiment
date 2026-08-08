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
#   ./run_base_sweep.sh logs_run7    ...writing all logs under logs_run7/
#   ./run_base_sweep.sh -l logs_run7 same thing, explicit flag
#   ./run_base_sweep.sh -i 120       ...at a different interval
#   ./run_base_sweep.sh --fresh      delete stale epoch checkpoints first
#   ./run_base_sweep.sh --detach     launch and exit immediately
#   ./run_base_sweep.sh --status     report on already-running streams, then exit
#
# Everything the sweep prints is also written under the log directory (default
# logs_tuned/), so a detached run can be reconstructed after the terminal is gone:
#   <log-dir>/                     dqb's own run_*.txt logs (its --logging-dir)
#   <log-dir>/streams/<name>.log   raw per-stream stdout, tqdm bars and all
#   <log-dir>/sweep_progress.log   every progress table this script has printed,
#                                  appended across launches and --status queries
# --status reads whichever log directory it is given, so pointing it at an old
# sweep's directory replays that sweep's final state.
#
# Ctrl-C stops the progress display only; training keeps running (each stream
# ignores SIGINT/SIGHUP). Use `pkill -f 'dqb run'` to actually stop training.
#
# When the watcher sees the last stream exit it rebuilds manifest.csv and the
# comparison reports (see rebuild_reports). Four concurrent `dqb run` processes
# each write those files from their own records at exit, so without the rebuild
# the sweep ends holding whichever quarter of the results finished last. Detached
# and Ctrl-C'd runs never reach that point — rebuild them by hand:
#   dqb compare --manifest --results-root results --results-directory updated_models
set -uo pipefail

cd "$(dirname "$0")"

RESULTS_DIR="updated_models"
LOG_ROOT="logs_tuned"
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
    -l|--log-dir)  LOG_ROOT="${2:?--log-dir needs a directory}"; shift 2 ;;
    # Anchored on the comment text rather than line numbers so editing the
    # header above cannot silently make --help print the wrong block. The second
    # sed only prints lines it could strip a '#' from, so the `set` line that
    # ends the range is dropped and any paragraph added to the header before it
    # is picked up automatically.
    -h|--help)  sed -n '/^# Usage:/,/^set -uo pipefail/p' "$0" \
                  | sed -n 's/^#\{1,\} \{0,1\}//p'; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *)  LOG_ROOT="$1"; shift ;;
  esac
done

LOG_ROOT="${LOG_ROOT%/}"
LOG_DIR="$LOG_ROOT/streams"
# Deliberately a level above LOG_DIR: anything ending in .log inside LOG_DIR is a
# candidate stream log to the reporter below, and a progress file sitting there
# gets reported as a fifth stream stuck at "starting…".
PROGRESS_LOG="$LOG_ROOT/sweep_progress.log"

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
# Named explicitly rather than globbed: the only files that parse as stream logs
# are the four this script writes, and a glob would pick up anything else.
logs = [p for p in (log_dir / f"{name}.log" for name in ORDER) if p.exists()]
if not logs:
    print("  (no stream logs yet)")
    raise SystemExit

total_done = 0
print(f"  {'stream':<10} {'model':<22} {'epoch':>10} {'%':>4}  {'remaining':>10}  {'best':>9}")
print(f"  {'-'*10} {'-'*22} {'-'*10} {'-'*4}  {'-'*10}  {'-'*9}")

for log in logs:
    # These files belong to the stream subshells, not to us, and the watcher
    # rereads them every INTERVAL for hours. A log that is rotated, truncated or
    # removed between the listing above and this read must not take the watcher
    # down mid-sweep.
    try:
        text = ANSI.sub("", log.read_text(errors="replace")).replace("\r", "\n")
    except OSError as exc:
        print(f"  {log.stem:<10} (unreadable: {exc.strerror})")
        continue
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

# Every progress table is appended to PROGRESS_LOG as well as printed. The
# per-stream logs are unreadable after the fact — tqdm rewrites one line with \r
# for hours — so this file is the only durable record of how the sweep advanced,
# and it is what makes --detach usable.
emit() { tee -a "$PROGRESS_LOG"; }

report_block() {
  {
    echo
    echo "=============================================================================="
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    report
  } | emit
}

# Each `dqb run` writes manifest.csv and the comparison reports from *its own*
# records when it exits, so with four concurrent streams the last one to finish
# overwrites the other three. The per-model record.json files are never touched
# by that race, so a single rebuild across all of them restores the correct
# manifest and reports. Only reached when the watcher saw every stream exit —
# --detach and --status leave this to the operator, since neither knows when (or
# whether) training finished.
rebuild_reports() {
  {
    echo
    echo "rebuilding manifest.csv and comparison reports from all per-model records"
    echo "  (each stream had overwritten them with only its own share)"
  } | emit
  if dqb compare --manifest \
       --results-root results \
       --results-directory "$RESULTS_DIR" \
       --logging-dir "$LOG_ROOT" 2>&1 | emit
  then
    echo "  rebuilt -> results/$RESULTS_DIR/manifest.csv" | emit
  else
    # Non-fatal on purpose: training results are already on disk, and a failed
    # rebuild must not make the sweep look like it lost them.
    {
      echo "  WARNING: rebuild failed. Results are intact; rerun by hand with:"
      echo "    dqb compare --manifest --results-root results --results-directory $RESULTS_DIR"
    } | emit
  fi
}

watch_loop() {
  trap 'echo; echo "detached — training continues. stop with: pkill -f \"dqb run\""; exit 0' INT
  while :; do
    report_block
    running || { { echo; echo "all streams finished."; } | emit; rebuild_reports; break; }
    sleep "$INTERVAL"
  done
}

mkdir -p "$LOG_DIR"

# ------------------------------------------------------------------ status ---
if [[ "$MODE" == "status" ]]; then
  report_block
  running || echo "  (no dqb run processes active)" | emit
  exit 0
fi

# ------------------------------------------------------------------ launch ---
# shellcheck disable=SC1091
source .venv/bin/activate

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
    # Logged, not just printed: which checkpoints were discarded is the one fact
    # that decides whether a later result is a clean run or a resumed one.
    echo "removing ${#stale[@]} stale epoch checkpoint(s): ${stale[*]}" | emit
    for model in "${stale[@]}"; do
      rm -rf "results/$RESULTS_DIR/$model/base_fp32"
    done
  else
    # Same reasoning as the --fresh branch: still stderr for the operator, but
    # also on disk, so a resumed run cannot later be mistaken for a clean one.
    {
      echo "WARNING: ${#stale[@]} model(s) have an epoch_checkpoint.pt and WILL resume"
      echo "  mid-run rather than start clean: ${stale[*]}"
      echo "  --ignore-saved-models does not override this. If the model definitions"
      echo "  changed since those checkpoints, the run would be invalid."
      echo "  Re-run with --fresh to delete them first, or accept the resume."
      echo
    } | tee -a "$PROGRESS_LOG" >&2
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
      --logging-dir "$LOG_ROOT" \
      --conditions base_fp32 \
      --ignore-saved-models \
      --models "$@" \
      >"$LOG_DIR/$name.log" 2>&1
  ) &
  printf '  %-10s pid %-7s %s\n' "$name" "$!" "$*" | emit
}

{
  echo
  echo "=== launch $(date '+%Y-%m-%d %H:%M:%S') -> results/$RESULTS_DIR, logs -> $LOG_ROOT/"
} | emit
: >"$LOG_DIR/cifar.log"; : >"$LOG_DIR/nlp_graph.log"
: >"$LOG_DIR/heavy.log"; : >"$LOG_DIR/light.log"

stream cifar     "${STREAM_CIFAR[@]}"
stream nlp_graph "${STREAM_NLP[@]}"
stream heavy     "${STREAM_HEAVY[@]}"
stream light     "${STREAM_LIGHT[@]}"

cat <<EOF | emit

logs:     tail -f $LOG_DIR/*.log
progress: tail -f $PROGRESS_LOG
status:   ./run_base_sweep.sh -l $LOG_ROOT --status
stop:     pkill -f 'dqb run'
EOF

if [[ "$MODE" == "detach" ]]; then
  cat <<EOF | emit
rebuild:  dqb compare --manifest --results-root results --results-directory $RESULTS_DIR
          (run once every stream has exited — detached runs skip the automatic
           rebuild, so manifest.csv holds only the last stream's records)
EOF
  exit 0
fi

echo
echo "progress every ${INTERVAL}s — Ctrl-C detaches without stopping training"
echo "(also appended to $PROGRESS_LOG)"
watch_loop
