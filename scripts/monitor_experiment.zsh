#!/bin/zsh
# Persistent local monitor for one detached DQB experiment launch.
# Emits a status line every 30 minutes and stops only matching worker processes
# when a fatal Python/storage error appears in the selected stream logs.

set -u

if (( $# < 3 || $# > 4 )); then
  print -u2 "usage: $0 RESULTS_DIRECTORY LOG_DIRECTORY REPORT_LOG [PROJECT_ROOT]"
  exit 64
fi

RESULTS_DIRECTORY="$1"
LOG_DIRECTORY="$2"
REPORT_LOG="$3"
PROJECT_ROOT="${4:-${0:A:h:h}}"
cd "$PROJECT_ROOT"
PROBLEM_LOG="information/problems/live-monitor-events.md"
REPORT_INTERVAL=1800
POLL_INTERVAL=60
FATAL_PATTERN='Traceback \(most recent call last\)|lost sys\.stderr|RuntimeError:.*(No space left|ios_base::clear|inline_container)|OSError:.*No space left|ENOSPC|Segmentation fault|Fatal Python error'

typeset -A seen_lines
typeset -A seen_sizes

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %z'
}

notify() {
  local title="$1"
  local message="$2"
  osascript -e "display notification \"${message}\" with title \"${title}\"" 2>>"$REPORT_LOG" || true
}

matching_worker_pids() {
  ps -axo pid=,command= | while IFS= read -r line; do
    case "$line" in
      *"/dqb run --worker"*"--results-directory ${RESULTS_DIRECTORY}"*)
        print -- "${line%% *}"
        ;;
    esac
  done
}

stop_workers() {
  local pid
  for pid in ${(f)$(matching_worker_pids)}; do
    kill -TERM "$pid" 2>>"$REPORT_LOG" || true
  done
  sleep 10
  for pid in ${(f)$(matching_worker_pids)}; do
    kill -KILL "$pid" 2>>"$REPORT_LOG" || true
  done
}

write_status() {
  local completed free_kb free_gb active
  completed=$(find "experiment_results/${RESULTS_DIRECTORY}" -name record.json 2>/dev/null | wc -l | tr -d ' ')
  free_kb=$(df -k . | awk 'NR == 2 {print $4}')
  free_gb=$(( free_kb / 1048576 ))
  active=$(matching_worker_pids | wc -l | tr -d ' ')
  print "[$(timestamp)] progress: completed=${completed}/60, active_workers=${active}, disk_free=${free_gb}GiB" | tee -a "$REPORT_LOG"
  notify "Dendritic quantization progress" "${completed}/60 complete; ${free_gb} GB disk free; ${active} workers active."
}

record_fatal() {
  local file="$1"
  local context="$2"
  print "[$(timestamp)] FATAL monitor event in ${file}" | tee -a "$REPORT_LOG"
  {
    print "\n## $(timestamp) — fatal worker error"
    print
    print "- Stream: \`${file}\`"
    print "- Action: alerted locally and sent SIGTERM/SIGKILL only to workers for \`${RESULTS_DIRECTORY}\`."
    print
    print "\`\`\`text"
    print -r -- "$context"
    print "\`\`\`"
  } >> "$PROBLEM_LOG"
  notify "Dendritic quantization stopped" "Fatal worker error detected. Workers for ${RESULTS_DIRECTORY} were stopped; see ${PROBLEM_LOG}."
  stop_workers
  exit 1
}

for file in "$LOG_DIRECTORY"/streams/stream_*.log(N); do
  seen_lines[$file]=$(wc -l < "$file")
done

write_status
last_report=$SECONDS

while true; do
  for file in "$LOG_DIRECTORY"/streams/stream_*.log(N); do
    current_lines=$(wc -l < "$file")
    previous_lines=${seen_lines[$file]:-0}
    if (( current_lines > previous_lines )); then
      new_text=$(sed -n "$((previous_lines + 1)),${current_lines}p" "$file")
      if print -r -- "$new_text" | rg -q "$FATAL_PATTERN"; then
        record_fatal "$file" "$(print -r -- "$new_text" | tail -n 80)"
      fi
      seen_lines[$file]=$current_lines
    fi
  done

  if (( SECONDS - last_report >= REPORT_INTERVAL )); then
    write_status
    last_report=$SECONDS
  fi
  sleep "$POLL_INTERVAL"
done
