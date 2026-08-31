# Dynamic Dendritic Run — Migration Guide

<!-- status-banner -->
> **Status: historical migration record.** The migration is complete: HISTORY scheduling is the default, fixed-interval switching is a labelled diagnostic, and every change listed below shipped. What survives here is why the move was made and what PerforatedAI 3.2.3 forced along the way. The proposed diffs, the recommended first invocation, and the ordering/rollback plan were removed in the 2026-08-31 cleanup — they cited line numbers that no longer exist, and the tuning rationale argued for values (`max_dendrites=6`, `reset_best_score_on_switch=True`) that the shipped configuration does not use. `compat.py::PAI_DYNAMIC_SCHEDULE_DEFAULTS` is the live schedule.

This began as a plan to switch from the bounded (`DOING_FIXED_SWITCH`) dendritic pipeline to a dynamic (`DOING_HISTORY`) run driven by PerforatedAI's own completion signal, and to stop suppressing PAI's training graphs.

> **Status note (2026-08-06).** The per-model numbers in §1 — the win/regression
> buckets, the best-epoch-vs-switch observations, the "baseline was 13%" note on
> PointNet — come from the `results/` set produced *before* the baseline-quality
> pass. That pass changed every model's learning-rate schedule, CapsNet's loss,
> three models' input widths, and the molecular/graph/text preprocessing (see
> "Baseline Quality" in [DOCUMENTATION.md](DOCUMENTATION.md)). AttentiveFP alone
> moved from RMSE 2.14 to 0.85. The *structural* argument still holds —
> fixed-schedule switching firing before the base network converges is a property
> of the schedule, not of any one baseline — but the specific deltas are stale and
> should be recomputed from a fresh run before being cited.

---

## 1. Motivation — what the bounded runs actually did

All `dendrites_fp32` runs up to this point used the **bounded** pipeline: `_configure_bounded_pai_schedule` in `compat.py`, since removed. Every model got exactly `min(4, active_epochs // 4)` dendrite cycles regardless of merit, and no run ever reached PAI's natural `training_complete` signal (`pai_training_complete` is `False` on every row of every `history.csv`).

Per-model outcome (23 models, base_fp32 vs dendrites_fp32):

| Bucket | Count | Notable |
|---|---|---|
| Wins (Δ > 0.5% rel) | 14 | lstm_autoencoder +11.4%, tcn_forecaster +9.7%, actor_critic +7.0%, resnet18 +5.7%, mpnn +5.2% |
| Regressions | 3 | **distilbert −42%** (only 4 epochs, killed by first switch at ep 3); **pointnet_modelnet40 −66%** (baseline was 13% — broken before dendrites); **dqn_lunarlander −0.9%** |
| Wash (\|Δ\| < 0.5%) | 6 | capsnet, lenet5, saint_adult, snn_nmnist, tabnet, textcnn |

Two structural problems fall out of that table:

1. **Best epoch often lands before or at the first switch** (gcn ep 2 vs switch 40, capsnet ep 3 vs switch 6, snn_nmnist ep 5 vs switch 10, textcnn ep 3 vs switch 4, distilbert ep 2 vs switch 3). Fixed-schedule switching fires while the base network is still improving, so the dendrite phase inherits weights that hadn't converged.
2. **Everyone gets 4 dendrites, no more, no less** — no exploration of whether 2 was enough or 6 would have helped.

Dynamic mode solves both, but the dynamic config as it stood was under-tuned for real use (`max_dendrites=100`, no `improvement_threshold`, no history lookback).

---

## 2. What the migration changed

Six changes, all of them shipped. The first four tightened the dynamic path's
configuration, the fifth turned on PAI graph output, the sixth scoped the first sweep.

| # | Change | Where it lives now |
|---|---|---|
| 1 | Tighten `_configure_dynamic_pai_schedule` — a real `max_dendrites`, a decaying `improvement_threshold` ladder, a history lookback long enough to detect a plateau, and a smaller candidate-weight multiplier | `compat.py::PAI_DYNAMIC_SCHEDULE_DEFAULTS` and `_configure_dynamic_pai_schedule` |
| 2 | Enable dashboard events so the Training View receives `epoch`, `switch`, and `run_start` | `compat.py::_configure_pai_trackers` |
| 3 | Stop suppressing PAI's own training graphs (`making_graphs=True`) | `compat.py::perforate_model` |
| 4 | Stop force-zeroing `weight_decay` for dendrite runs, which was stripping regularisation from the base network too | `pipeline.py` |
| 5 | Surface PAI's graphs next to the benchmark artifacts instead of leaving them in the PAI tree | `training.py` |
| 6 | Hold `distilbert` out of the first dynamic sweep — a 3-epoch recipe cannot plateau against a lookback of 8 | run scoping, not code |

The one knob that did *not* survive review: `reset_best_score_on_switch` stayed `False`.
The shipped `max_dendrites` default is 3, with per-model overrides in `pipeline.py`, not
the 6 this plan proposed.

---

## 3. PerforatedAI 3.2.3 — `save_name` may not contain `/`

PerforatedAI ≥ 3.2.3 validates `save_name` and calls `sys.exit(1)` when it contains a path separator:

```
Warning: save_name 'PAI/lenet5_dendrites_fp32' contains '/'. Relative paths are not implemented yet.
```

`SystemExit` is not an `Exception`, so it slipped through `perforate_model`'s `except Exception` handler and killed the run with no traceback — the log simply stopped after that warning.

PAI resolves `save_name` against the process working directory, so `PAI/{save_name}/` is now reached by changing directory rather than by nesting the name:

- `_pai_flat_save_name()` collapses the save name to a single separator-free segment.
- `pai_working_directory()` is a re-entrant context manager that `chdir`s into `PAI/` for the duration of each PAI library call, and restores the previous directory on the way out.
- Every call that touches PAI's filesystem — `perforate_model`, `load_system`, `setup_optimizer`, `add_validation_score` — runs inside that context manager.

The resulting on-disk layout is byte-for-byte the pre-3.2.3 one, so existing `PAI/{save_name}/switch_*.pt` checkpoints stay loadable.

Two related 3.2.3 changes fall out of this:

- The perforation config moved from a single `PAI/PAI_config.json` to a per-run `PAI/{save_name}/{save_name}_config.json`. `_snapshot_pai_config` reads the new path; without that fix it found nothing and the per-condition `*_PAI_config.json` snapshots stopped being written.
- `perforate_model` now converts a PAI `SystemExit` into a `RuntimeError`, so any future library-level abort surfaces as a real failure instead of a silent process death.

---
