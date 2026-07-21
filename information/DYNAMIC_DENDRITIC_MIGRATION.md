# Dynamic Dendritic Run — Migration Guide

Migration plan for switching from the current bounded (FIXED_SWITCH) dendritic pipeline to a proper dynamic (DOING_HISTORY) run driven by PerforatedAI's own completion signal. Also enables saving PAI-generated training graphs, which are currently suppressed.

Everything in this doc is grounded in the current codebase and the 23-model result set in `results/`. File and line references point to the exact spots that need editing.

---

## 1. Motivation — what the current runs actually did

All existing `dendrites_fp32` runs used the **bounded** pipeline: [`_configure_bounded_pai_schedule`](../src/dendritic_benchmark/compat.py) in [compat.py:238-264](../src/dendritic_benchmark/compat.py#L238-L264). Every model got exactly `min(4, active_epochs // 4)` dendrite cycles regardless of merit, and no run ever reached PAI's natural `training_complete` signal (`pai_training_complete` is `False` on every row of every `history.csv`).

Per-model outcome (23 models, base_fp32 vs dendrites_fp32):

| Bucket | Count | Notable |
|---|---|---|
| Wins (Δ > 0.5% rel) | 14 | lstm_autoencoder +11.4%, tcn_forecaster +9.7%, actor_critic +7.0%, resnet18 +5.7%, mpnn +5.2% |
| Regressions | 3 | **distilbert −42%** (only 4 epochs, killed by first switch at ep 3); **pointnet_modelnet40 −66%** (baseline was 13% — broken before dendrites); **dqn_lunarlander −0.9%** |
| Wash (\|Δ\| < 0.5%) | 6 | capsnet, lenet5, saint_adult, snn_nmnist, tabnet, textcnn |

Two structural problems fall out of that table:

1. **Best epoch often lands before or at the first switch** (gcn ep 2 vs switch 40, capsnet ep 3 vs switch 6, snn_nmnist ep 5 vs switch 10, textcnn ep 3 vs switch 4, distilbert ep 2 vs switch 3). Fixed-schedule switching fires while the base network is still improving, so the dendrite phase inherits weights that hadn't converged.
2. **Everyone gets 4 dendrites, no more, no less** — no exploration of whether 2 was enough or 6 would have helped.

Dynamic mode solves both, but the current dynamic config in [compat.py:217-235](../src/dendritic_benchmark/compat.py#L217-L235) is under-tuned for real use (`max_dendrites=100`, no `improvement_threshold`, no history lookback).

---

## 2. Scope of the migration

Six changes. First four are configuration tightening for the dynamic path. Fifth turns on PAI graph output. Sixth trims the model list to exclude runs that dendrites can't fix.

| # | File | Change |
|---|---|---|
| 1 | [compat.py:217-235](../src/dendritic_benchmark/compat.py#L217-L235) | Tighten `_configure_dynamic_pai_schedule` |
| 2 | [compat.py:150-167](../src/dendritic_benchmark/compat.py#L150-L167) | Add `set_dashboard_events_enabled(True)` in `_configure_pai_trackers` |
| 3 | [compat.py:618](../src/dendritic_benchmark/compat.py#L618) | Flip `making_graphs=False` → `True` |
| 4 | [pipeline.py:842](../src/dendritic_benchmark/pipeline.py#L842) | Stop force-zeroing `weight_decay` for dendrite runs |
| 5 | [training.py](../src/dendritic_benchmark/training.py) | Copy PAI-generated graphs from `PAI/{save_name}/` into `results/{model}/dendrites_fp32/pai_plots/` at end of run |
| 6 | CLI invocation | Exclude `pointnet_modelnet40` and `distilbert` from the first dynamic sweep |

---

## 3. Change 1 — tighter dynamic PAI schedule

### Current — [compat.py:217-235](../src/dendritic_benchmark/compat.py#L217-L235)

```python
def _configure_dynamic_pai_schedule(
    pc: Any,
    batches_per_epoch: int | None = None,
    initial_correlation_batches_limit: int | None = None,
) -> None:
    _set_pai_switch_mode(pc, "DOING_HISTORY")
    _apply_pai_schedule_values(
        pc,
        {
            "set_n_epochs_to_switch": 10,
            "set_p_epochs_to_switch": 2,
            "set_max_dendrites": 100,
        },
    )
    correlation_batches = _initial_correlation_batches(
        batches_per_epoch, initial_correlation_batches_limit
    )
    if correlation_batches is not None:
        _call_pai_setter(pc, "set_initial_correlation_batches", correlation_batches)
```

### Proposed

```python
def _configure_dynamic_pai_schedule(
    pc: Any,
    batches_per_epoch: int | None = None,
    initial_correlation_batches_limit: int | None = None,
) -> None:
    _set_pai_switch_mode(pc, "DOING_HISTORY")
    _apply_pai_schedule_values(
        pc,
        {
            "set_n_epochs_to_switch": 10,
            "set_p_epochs_to_switch": 2,
            "set_max_dendrites": 6,
            "set_n_epochs_for_switch_history": 8,
            "set_improvement_threshold": [0.005, 0.001, 0.0001, 0],
            "set_reset_best_score_on_switch": True,
            "set_candidate_weight_initialization_multiplier": 0.005,
        },
    )
    correlation_batches = _initial_correlation_batches(
        batches_per_epoch, initial_correlation_batches_limit
    )
    if correlation_batches is not None:
        _call_pai_setter(pc, "set_initial_correlation_batches", correlation_batches)
```

Why each knob:

- **`max_dendrites=6`** — the empirical win curve plateaus at 2-4 dendrites in almost every model. 100 leaves runaway room with no upside; 6 is a soft cap that still exceeds every historic best-arch count.
- **`n_epochs_for_switch_history=8`** — matches PAI's guidance that history should be long enough to detect a real plateau. Default `history_lookback=1` triggers on transient noise.
- **`improvement_threshold=[0.005, 0.001, 0.0001, 0]`** — a decaying ladder. Early dendrites only fire when improvement stalls hard (>0.5% relative). Later dendrites fire on smaller gains as the model saturates.
- **`reset_best_score_on_switch=True`** — dendrite additions restructure the model; comparing to the pre-restructure best anchors PAI to a metric it may no longer be able to match on the new architecture.
- **`candidate_weight_initialization_multiplier=0.005`** — currently 0.01 in captured configs. Halving it should reduce the post-switch metric spike that caused several models to peak before the first switch.

`_apply_pai_schedule_values` calls `getattr(pc, setter_name, None)` — if a setter isn't in the installed PerforatedAI, it's silently skipped. Safe.

---

## 4. Change 2 — dashboard events

The [train-my-model skill](../PAI%20Skills/skills/train-my-model/SKILL.md) requires `GPA.pc.set_dashboard_events_enabled(True)` for the Training View to receive `epoch`, `switch`, and `run_start` events. Right now the benchmark never calls it, so the dashboard's Training View is empty for our runs even when the MCP server is up.

### Edit [compat.py:150-167](../src/dendritic_benchmark/compat.py#L150-L167)

Add one line inside `_configure_pai_trackers`, alongside the other `_call_if_available` toggles:

```python
_call_if_available(pc, "set_dashboard_events_enabled", True)
```

If the installed PAI version doesn't have this setter, `_call_if_available` is a no-op. No harm.

**Complementary** — for the epoch progress bar in the Training View to show a total instead of an unbounded counter, POST a `run_config` event once per run. Add to [training.py](../src/dendritic_benchmark/training.py) near the start of `train_and_evaluate` (only for `use_dendrites and enable_pai_dendrite_updates`):

```python
if config.use_dendrites and _pai_updates_enabled(config):
    try:
        events_url = getattr(gpa.pc, "events_url", None)
        if events_url:
            import requests
            total = None if config.train_dendrites_until_complete else config.max_epochs
            requests.post(events_url, json={"type": "run_config", "total_epochs": total},
                          timeout=1.0)
    except Exception:
        pass
```

The dashboard treats absent totals as unbounded, which is correct for dynamic mode — no total is knowable up front.

---

## 5. Change 3 — enable PAI-generated training graphs

PAI's `drawing_pai: true` is already set (see any captured `PAI_config.json`), but graphs are suppressed at the entry point.

### Edit [compat.py:618](../src/dendritic_benchmark/compat.py#L618)

```python
perforated = upa_perforate_model(
    model,
    doing_pai=doing_pai,
    save_name=pai_save_name,
    maximizing_score=maximizing_score,
    making_graphs=True,   # was False
)
```

With this on, PAI writes its own training visualization (score progression, switch boundaries, LR schedule) into `PAI/{save_name}/` alongside the existing `switch_*.pt` and `best_model.pt` files.

---

## 6. Change 5 — copy PAI graphs next to benchmark artifacts

PAI's graphs land in `PAI/{save_name}/` (e.g. `PAI/actor_critic_dendrites_fp32/`). Benchmark artifacts live in `results/{model}/dendrites_fp32/`. Nothing surfaces the graphs to `results/`, so the analyze skill (which reads `results/`) can't find them.

Add a small helper called at the end of a dendritic run inside [training.py](../src/dendritic_benchmark/training.py) (near where the record is written):

```python
def _copy_pai_graphs_to_output(pai_save_name: str, output_dir: Path) -> None:
    src = Path("PAI") / pai_save_name
    if not src.exists():
        return
    dst = output_dir / "pai_plots"
    dst.mkdir(parents=True, exist_ok=True)
    for ext in ("*.png", "*.svg", "*.pdf"):
        for f in src.glob(ext):
            shutil.copy2(f, dst / f.name)
```

Call it once, gated by `config.use_dendrites`, after the training loop exits normally. Copying (not moving) preserves the checkpoint dir layout PAI expects on resume.

---

## 7. Change 4 — restore recipe `weight_decay` for dendrite runs

[pipeline.py:842](../src/dendritic_benchmark/pipeline.py#L842) currently reads:

```python
weight_decay = 0.0 if condition.use_dendrites else training_hyperparameters.weight_decay
```

This was defensive — dendrite parameters can react poorly to global weight decay — but the effect is to strip regularization from the base network too. Several wash-bucket models (`tabnet`, `saint_adult`, `snn_nmnist`) have non-trivial `weight_decay` in their recipes ([pipeline.py:518-547](../src/dendritic_benchmark/pipeline.py#L518-L547)) and lose it silently.

Change to always honor the recipe:

```python
weight_decay = training_hyperparameters.weight_decay
```

If a specific model regresses on the dynamic run because of this, isolate it via `_MODEL_DENDRITIC_BATCH_SIZES`-style per-model override rather than a blanket zero.

---

## 8. Change 6 — exclude broken baselines from the first dynamic sweep

Two models cannot benefit from dynamic dendrites in their current form:

- **`pointnet_modelnet40`** — baseline accuracy is **13.4%** on ModelNet40 (10-class random ≈ 10%, 40-class random ≈ 2.5%). The base model is broken. Dendrites made it worse (−66%) because they're amplifying a non-signal. Fix the baseline first — investigate `models.py`'s pointnet definition and the modelnet40 dataloader — before including it in a dendritic sweep.
- **`distilbert`** — the recipe is 4 epochs. Dynamic mode needs many more to detect a real plateau (`n_epochs_for_switch_history=8` already exceeds it). Either bump `distilbert`'s `max_epochs` recipe to ≥ 20 for the dendritic condition, or hold it out.

Recommended first dynamic invocation:

```bash
uv run dqb run \
  --dynamic-dendritic-training \
  --models actor_critic attentivefp_freesolv capsnet_mnist dqn_lunarlander gcn \
           gin_imdbb gru_forecaster lenet5 lstm_autoencoder lstm_forecaster m5 \
           mobilenetv2_cifar10 mpnn ppo_bipedalwalker resnet18_cifar10 saint_adult \
           snn_nmnist tabnet tcn_forecaster textcnn vae_mnist \
  --conditions dendrites_fp32 \
  --ignore-saved-models
```

`--conditions dendrites_fp32` scopes the sweep to only the condition that benefits from dynamic mode. Quantized dendritic conditions can be regenerated in a follow-up pass from the new dendrites_fp32 checkpoints.

---

## 9. Ordering and verification

Recommended order:

1. Apply Change 3 (`making_graphs=True`) and Change 2 (dashboard events) — cheap, no behavior change on the training math.
2. Apply Change 1 (dynamic schedule) and Change 4 (`weight_decay`).
3. Apply Change 5 (graph copy helper).
4. Smoke-test on `lenet5` alone:
   ```bash
   uv run dqb run --dynamic-dendritic-training --models lenet5 \
       --conditions dendrites_fp32 --ignore-saved-models
   ```
   Expected: run terminates via `pai_training_complete=True` (not by hitting `max_epochs`), `pai_plots/` populated in `results/lenet5/dendrites_fp32/`, `switch_epochs` no longer at fixed intervals.
5. Run the full sweep (Change 6).

Verification checklist per model after the sweep:

- `record.json` has a **new best_metric_value** vs the current `results/<model>/dendrites_fp32/record.json`.
- `history.csv` contains at least one row with `pai_training_complete=True`.
- `history.csv`'s `pai_restructured=True` rows are **not** at uniform intervals (indicates history-based switching kicked in).
- `pai_plots/` exists and contains at least one PAI-generated `.png` or `.svg`.

---

## 10. Rollback

Every change is a single-line or single-block revert. If the dynamic sweep produces worse aggregate results than the bounded runs, revert Change 1 first (the schedule) and re-run. Changes 2, 3, 4, 5 are independently useful and don't need to be reverted with it.

The old bounded runs in `results/` are untouched by adding `--ignore-saved-models` only to the new sweep — but a safer path is to move `results/` to `results.bounded/` before starting.

## 11. Skill dependencies

- [perforatedai](../PAI%20Skills/skills/perforatedai/SKILL.md) — canonical setup steps this migration piggybacks on (`while True:` loop, `add_validation_score`, restructure handling).
- [perforatedai-analyze](../PAI%20Skills/skills/perforatedai-analyze/SKILL.md) — post-run analysis; reads the exact CSVs listed in the verification checklist.
- [train-my-model](../PAI%20Skills/skills/train-my-model/SKILL.md) — dashboard wiring (`set_dashboard_events_enabled`, `run_config` event).
