# Full-code deep dive — 2026-08-28

Scope: every line of `src/dendritic_benchmark/` (12 modules, 14,318 lines), read
end-to-end in one pass, looking for correctness bugs ahead of the next training
run. Uncommitted §8 fixes (training.py) were already in the tree and are treated
as part of the reviewed state.

## Bugs found and fixed in this pass

### 1. Collapse-guard log message off by one (training.py, `_run_training_epochs`)

`_record_best_epoch` stores `state.best_epoch = epoch + 1` (1-indexed), but the
`[collapse]` message printed `epoch {state.best_epoch + 1}` — every collapse
line in every log to date names the best epoch one higher than it was. Cosmetic
in the artifact files (record.json was always right), but these lines are
exactly what gets read during post-mortems, including the §8 investigation.
**Fixed:** the message now prints `state.best_epoch` unchanged.

### 2. Latency benchmark silently measured the wrong architecture for `dendrites_*` (benchmark.py, `_load_model_state`)

`benchmark_condition` builds a *plain* `build_model()` skeleton and loaded
`model.pt` with a shape-compatibility filter + `strict=False`. A dendritic
`model.pt` carries PerforatedAI wrapper key names, so almost nothing matched:
the "dendritic" latency rows in `benchmarks/` were an unperforated,
mostly-randomly-initialized base network — i.e. base-architecture latency
labelled as dendritic. **Fixed:** the loader now counts unloadable checkpoint
tensors and unfilled model tensors (ignoring `tracker_string`) and refuses the
load with an explicit message, so those conditions report
`failed to load model state` instead of a wrong number. Any existing
`benchmarks/*/dendrites_*.json` latency rows should be discarded.

### 3. §9 root-caused — actor_critic's "missing" history rows (`_persist_over_budget_snapshot`, by design)

Not a code bug, but a resolved open item: dynamic dendritic runs split their
history at the recipe budget — `history.csv` holds epochs `1..max_epochs`,
everything later goes to `continued_until_complete/history.csv`. Verified on
disk: actor_critic dendrites_fp32 has epochs 1–60 canonical + 61–145 in the
sidecar (145 total, matching the log and `best_epoch=117`). This retroactively
corrects §8's actor_critic row: the run was **not** cut off mid-phase — it
completed 4 dendrite cycles and ended normally. MEASUREMENT_CAVEATS §8/§9 and
the status table were updated; §8's true casualty count is **4 of 7**
(mpnn, vae_mnist, tcn_forecaster, saint_adult), with lenet5 killed in an
ordinary plateau and gcn + actor_critic completing their schedules. That also
narrows §8's invalidation: **gcn's and actor_critic's dendritic numbers stand.**

## Known-sharp edges confirmed but left as designed (documented, not changed)

- **`paramCounts.csv` is flat by construction** (one final count repeated per
  history row) — already documented in §8; do not read it as a growth curve.
- **`results/PAI/<save_name>/` is never cleared between runs** — its
  `switch_epochs.csv`/`param_counts.csv` mix runs; use each run's own
  history.csv (+ `continued_until_complete/`).
- **Epoch-budget asymmetry**: dynamic dendritic arms train until PAI completes
  (`itertools.count()`), base arms use the fixed recipe budget. A real
  confound, flagged in §8; a design decision for the user, not a code fix.
- **PPO truncation bootstrap** (`PPORolloutSource._advantages_and_returns`)
  bootstraps a truncated episode from the *next* buffer entry (the reset
  state), the standard CleanRL/SB3-style simplification.
- **`lenet5` under `DOING_HISTORY` never switches** (mode-`n` EMA trap,
  see memory/pai-running-average-blocks-switch): not addressed by the §8 fix.
  Making lenet5 dendritic requires a `_MODEL_DENDRITIC_FIXED_SWITCH_INTERVALS`
  entry (as DistilBERT has) — an experiment-design change, so not made
  unilaterally; lenet5 is dropped from the next run's selection instead.
- **Collapse-killed runs leave `epoch_checkpoint.pt` behind**, so a re-launch
  resumes the dead run unless `--fresh` is passed. `run_dynamic7.sh` re-runs
  should use `--fresh` for previously collapsed conditions (the §8 rerun must
  anyway, since those results are invalid).

## Modules read clean (no defects found)

`specs.py`, `log_utils.py` (output-root validation is sound), `compat.py`
(quantization kernels re-verified against §1/§6: scale by `|qmin|`, b1.58
absmean scale, binary zero-preservation, deterministic strided calibration
subsample), `models.py` (SAINT's batch-dim transpose is intentional intersample
attention; MPNN/AttentiveFP/GIN padding masks correct; bridge-finding ring
detection in data.py correct), `data.py` (chronological forecast splits,
train-only normalization, episode-level RL splits, transductive Cora, Planetoid
split all correct), `results.py`, `plots.py`, `cli.py`, `pipeline.py`
(worker respawn/log-rotation/plan accounting correct).
