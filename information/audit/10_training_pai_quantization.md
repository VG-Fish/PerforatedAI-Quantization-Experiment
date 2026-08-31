# Deep review: training, PAI compatibility, and quantization

**Scope:** `src/dendritic_benchmark/training.py`, `compat.py`, related tests/docs.  
**Status:** analysis only; no source changes or deletions.

## Findings

### A. `training.py` is an orchestration monolith

The module is 4,413 lines and mixes metric/loss implementations, dataset-shape inference, FP32/QAT/PTQ projection, PAI setup and log parsing, optimizer construction, checkpoint compatibility, batch execution, memory guards, LR scheduling, dynamic dendrite updates, and artifact persistence. The function inventory (for example `_compute_all_metrics` 971, `_forward` 1043, `_quantize_tensor` 1262, `_dendrite_audit` 1806, `_run_epoch_batches` 2698, `_run_training_epochs` 3731, `train_and_evaluate` 4155) shows multiple independent policies in one module.

**Simplify:** extract stable services with typed boundaries: `metrics.py` (loss/metric registry), `quantization.py` (one projection/QAT implementation), `pai_runtime.py` (configuration/lifecycle/logs), `checkpointing.py`, `epoch_engine.py`, and `artifacts.py`. Keep `train_and_evaluate` as a thin coordinator. This makes the experiment factor (dendrite × quantization) visible and testable.

### B. Duplicate compatible-state loading

`training.py::_load_compatible_best_state` (3194), `pipeline.py::_split_compatible_state`/`_load_compatible_state` (318–394), and `benchmark.py::_load_model_state` each implement variants of shape/key filtering. This is a scientific risk: permissive `strict=False` restores “some compatible state” rather than proving the recorded best epoch/topology was loaded. Consolidate one shared loader returning a structured report (`loaded`, `missing`, `unexpected`, `shape_mismatches`, `source_revision`) and make reportability fail closed when required tensors are absent.

### C. PAI schedule workaround is now partly obsolete

`compat.py::_configure_interval_pai_schedule` (294–317) and its explanatory comments justify fixed intervals because of the zero-seeded history EMA. The August 30 audit says `initial_history_after_switches = 8` was added at `compat.py:406`; `experiments/dynamic12/README.md` and `MEASUREMENT_CAVEATS.md` document the fix. The old rationale is now misleading and should be rewritten or deleted after verifying all callers. Fixed schedules are empirically invalid (GRU 8→1, VAE 20→8, SAINT 100→19, TCN never fired).

**Recommendation:** make HISTORY the sole default, retain fixed mode only as an explicitly labeled diagnostic, and record requested vs observed switch epochs. Do not delete fixed support until remaining historical artifacts are labeled.

### D. Dendrite learning-rate policy is fragmented

`_scheduled_learning_rate` (3661), `_dendrite_learning_rate` (3697), `_apply_lr_schedule` (3716), optimizer param grouping (2011–2084), and recipe overrides in `pipeline.py` jointly determine LR. The August 30 audit finds late candidates trained after the backbone cosine collapsed; only three models opt into the floor. Make a single schedule object produce backbone and dendrite rates and persist the effective per-group LR in every epoch row. Add a validation that a newly initialized dendrite receives a nonzero configured rate for its adaptation window.

### E. Candidate phases are governed by a hard cap, not convergence

`MAX_DENDRITE_PHASE_EPOCHS = 8` (around line 75) and `_pai_dendrite_phase_stalled` (3283) can terminate MPNN candidates while PB correlation is still rising (August 30 audit §5.4). Make the cap a timeout/diagnostic, not a silent convergence decision; persist termination reason and candidate score trajectory. Avoid interpreting PB correlation as a generalization proxy.

### F. Reporting status fails closed incorrectly

`results.py` excludes `no_retained_insertion`, `inherited_no_retained_insertion`, `unverified`, and `inherited_unverified`, but omits `legacy_unchecked` from `_NON_REPORTABLE_DENDRITE_STATUSES` (lines 24–29). `_legacy_dendrite_audit_status` assigns that status when raw and final parameter counts cannot be confirmed (164–227), so stale legacy rows become `reportable=True`. Add a distinct `invalid/unknown` status and require explicit `verified_retained` (plus current quantization revision) for dendritic claims. Add tests for every status transition.

### G. Compatibility and fallback paths need an inventory

`compat.py` contains many `_call_if_available`/optional-setter fallbacks (170–284), output/debugger suppression (628–752), old save-name flattening and resume checks (919–1105), plus environment aliases (60–103). These are useful only for supported PerforatedAI versions. Pin and test the supported PAI API, then delete branches for versions no longer supported; otherwise move them to a small adapter with an explicit compatibility matrix. Never catch an import/runtime failure and continue with an unlabeled dense result.

### H. Static dead-code signals

- `compat.py:pai_root` (797) is reported unused by both Vulture and deadcode and has no repository references: **high-confidence delete candidate**, after checking external consumers.
- `data.py:_Bond.ring_closure` (148) is assigned but not read by the benchmark; it is mentioned only in parser documentation (1403): candidate removal or make it part of a graph feature intentionally.
- `training.py` dataclass fields flagged as unused (`best_metric_value`, `train_seconds`, audit fields at 189–201) are serialized/reporting API, not safe automatic deletions.
- Nearly every `nn.Module.forward` is reported unused by static tools; these are framework entrypoints and must not be deleted.

## Exact cleanup candidates

- **[DELETE CANDIDATE]** `compat.py:pai_root` after external API check.
- **[REWRITE/DELETE OBSOLETE COMMENT]** the pre-fix fixed-interval rationale in `_configure_interval_pai_schedule` and duplicated migration prose once HISTORY is canonical.
- **[CONSOLIDATE]** three compatible-state loaders and repeated artifact validity checks.
- **[DELETE AFTER REPLACEMENT]** output/debugger suppression and old-version setters not covered by the supported PAI matrix.
- **[KEEP UNTIL REPLACED]** resume/checkpoint compatibility and raw PAI sidecars; they are needed to diagnose stale topology.

## Required tests before implementation

1. Property tests for reportability: no missing/legacy/stale topology can be reportable.
2. Schedule tests comparing requested/observed switch intervals and nonzero dendrite LR.
3. Checkpoint tests that fail on missing required tensors and expose a mismatch report.
4. Paired deterministic tests proving quantization projection is applied exactly once.

