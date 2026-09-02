# Diagnosis: `PerforatedAI did not write its final-clean inference artifact`

**Date:** 2026-09-02
**Failing condition:** `mnist_pai / dendrites_fp32` (also reproduces on `lenet5` with `max_epochs: 1`)
**Verdict:** **(iii) — something else**, whose closest neighbour is (ii).
This is a **regression introduced by commit `9de8880` ("fixed benchmark error", 2026-09-01 22:07)**.
It is not a property of `mnist_pai`, not a property of its epoch budget, and not
fixable by lengthening the budget. `_export_final_pai_artifact` gates the
benchmark's own artifact on a PAI **end-of-training** event that the benchmark's
**documented default mode structurally never produces**.

---

## A. What writes the final-clean artifact, and why it is absent

### The write side is PerforatedAI's, not the benchmark's

`_export_final_pai_artifact` does not *create* anything. It only copies a file
PAI is expected to have already written:

`src/dendritic_benchmark/training.py:1657-1666`
```python
def _export_final_pai_artifact(pai_save_name: str, checkpoint_path: Path) -> None:
    """Copy PAI's final-clean serialization to the benchmark artifact path."""
    pai_path = pai_save_path(pai_save_name) / "final_clean_pai.pt"
    if not pai_path.is_file():
        raise RuntimeError(
            "PerforatedAI did not write its final-clean inference artifact."
        )
    shutil.copy2(pai_path, checkpoint_path)
```

Nothing in this repository ever writes `final_clean_pai.pt`. Verified:

- `grep -rn "final_clean_pai" src/` — four hits, all **read** paths
  (`training.py:1662`, `training.py:1525`, `pipeline.py:434`, `pipeline.py:448`).
- The only writer the benchmark drives is
  `save_pai_system(model, pai_save_name, PAI_ARTIFACT_NAME)` at
  `training.py:4886-4890`, and `PAI_ARTIFACT_NAME = "dqb_artifact"`
  (`compat.py:40`) — it writes `dqb_artifact.pt`, never `final_clean_pai.pt`.

The file is produced *inside* the PAI library, by
`perforatedai.tracker_perforatedai.process_final_network`, whose docstring reads:

> "When the max number of dendrites has been hit load the best_model and return"

i.e. it runs on PAI's own **training-complete** transition
(`tracker_perforatedai.TRAINING_COMPLETE`), which the benchmark surfaces as the
third return value of `pai_tracker.add_validation_score(...)`
(`training.py:3858-3871`).

### Empirical confirmation from the stored PAI trees

`final_clean_pai.pt` is present **iff** the run reached PAI's own completion
signal — not iff a switch fired:

| PAI dir | switch_N.pt | `final_clean_pai.pt` | benchmark `termination_reason` |
|---|---:|---|---|
| `showcase5_capacity_seed0/PAI/m5_…` | 4 | YES | `pai_training_complete` |
| `showcase5_capacity_seed0/PAI/mpnn_…` | 6 | YES | `pai_training_complete` |
| `showcase5_capacity_seed0/PAI/actor_critic_…` | 4 | YES | `pai_training_complete` |
| `showcase5_capacity_seed0/PAI/lstm_forecaster_…` | 4 | YES | `pai_training_complete` |
| `showcase5_capacity_seed0/PAI/pointnet_modelnet40_…` | 2 | YES | `pai_training_complete` |
| `dendritic_quantization_launch5_…/PAI/distilbert_…` | **0** | **no** | **`epoch_budget`** |
| `dendritic_quantization_launch5_…/PAI/resnet18_cifar10_…` | 2 | **no** | (run never finished — only `epoch_checkpoint.pt`) |
| `audit_repair_mpnn_actor_seed0/PAI/mpnn_…` | 6 | **no** | (run never finished — only `epoch_checkpoint.pt`) |
| `.smoke/results/PAI/mnist_pai_dendrites_fp32_*` | **0** | **no** | `epoch_budget` (the failure) |

Note the `resnet18_cifar10` row: **two switches fired and the file still is not
there.** A completed switch is *not* the trigger. Only PAI's end-of-search is.

### Why it is absent for `mnist_pai`

`.smoke/results/PAI/mnist_pai_dendrites_fp32_b3b56310fe7f/…switch_epochs.csv`
contains only its header row; `…param_counts.csv` contains a single row
`0,1199882`. Zero switches, zero dendrites, therefore no `process_final_network`,
therefore no `final_clean_pai.pt`, therefore the raise.

The run log shows why PAI never switched
(`.smoke/logs/run_20260902_005915.txt:345-347`):
```
Checking PAI switch with mode n, switch mode DOING_HISTORY, epoch 9,
  last improved epoch 4, total epochs 9, n: 10, p: 2, num_cycles: 0
Returning False - no triggers to switch have been hit
```
and at `:364` the benchmark itself then disables PAI for the last 20% of the
budget: `[pai] live dendrite updates frozen; continuing with a standard optimizer.`
(`_pai_updates_frozen`, `training.py:3897-3903`, with
`freeze_dendrite_updates_fraction=0.20` set at `pipeline.py:2363`).

### Answering the specific question

> does it require at least one completed switch / at least one retained dendrite?

It requires **strictly more than that**: it requires PAI to have *finished its
entire dendritic search* and emitted `TRAINING_COMPLETE`. One switch is not
enough; ten switches are not enough. In the default fixed-budget mode the
benchmark **guarantees this can never happen at the end of a run**, because it
kills the tracker for the final 20% of epochs
(`_freeze_pai_live_updates`, `training.py:3930-3944`, which calls
`clear_pai_tracker_state()` and sets `pai_tracker = None` at `training.py:4319-4322`).

### `_prepare_final_clean_pai_model` is *not* the failing path

`_prepare_final_clean_pai_model` (`training.py:1572-1644`) calls
`perforatedai.utils_perforatedai.prepare_final_model(model)` **in process**. It
does not touch the filesystem and works for a zero-dendrite model. It was the
pre-regression accounting source and is now dead weight in the training path —
`train_and_evaluate` no longer calls it; only the test-facing wrapper
`_final_clean_pai_parameter_stats` (`training.py:1646-1654`) does.

---

## B. Do the shipped `dendrites_fp32` results complete a switch inside `max_epochs`?

**No stored result ever completed PAI's search inside a fixed epoch budget.**
Every one that produced a final artifact ran in the open-ended
`--dynamic-dendritic-training` / `--pai-capacity-check` mode.

From all 19 `pai_summary.json` files under `experiment_results/`:

| model | switch epochs | canonical epochs | continued epochs | `training_complete` epoch | termination |
|---|---|---:|---:|---|---|
| actor_critic | 45, 52, 61, 68 | 60 | **80** | 140 | `pai_training_complete` |
| mpnn | 16, 24, 76, 84, 94, 102 | 155 | 0 | 155 | `pai_training_complete` |
| lstm_forecaster | 9, 17 | 60 | **41** | 101 | `pai_training_complete` |
| m5 | 37, 45 | 40 | **84** | 124 | `pai_training_complete` |
| pointnet_modelnet40 | 9, 17 | 76 | 0 | 76 | `pai_training_complete` |
| **distilbert** | **(none)** | **3** | 0 | — | **`epoch_budget`** |

Two things fall out of this table:

1. **The `continued_epoch_count` column is the tell.** actor_critic's recipe
   budget was 60 epochs; it needed **140** to complete. m5's was 40; it needed
   **124**. lstm_forecaster's was 60; it needed **101**. These runs finished only
   because `train_dendrites_until_complete` let them ignore `max_epochs`
   entirely (`_epoch_progress` switches to `itertools.count()`,
   `training.py:3960-3978`) and dump the overflow into
   `continued_until_complete/`. Several also had `max_dendrites` cut to 1–3 via
   `--pai-override` (`pai_summary.dynamic_schedule`) purely to make completion
   reachable; the library default is **100**.

2. **`distilbert` is the direct counterexample that settles the diagnosis.**
   It is the one stored result that terminated on `epoch_budget` with zero
   switches — structurally identical to `mnist_pai`. Its PAI tree has **no**
   `final_clean_pai.pt`, and yet the run **succeeded**:
   `experiment_results/dendritic_quantization_launch5_seed0_20260831_1750/distilbert/dendrites_fp32/`
   contains `model.pt` (270 MB), `metrics.json`, `history.csv`,
   `artifact_manifest.json`, `pai_summary.json`, `record.json`. Its
   `record.json` records the *intended* outcome for this situation:
   ```json
   "dendrite_audit_status": "no_retained_insertion",
   "dendrite_audit_reason": "raw PAI switch log has no candidate-insertion switch"
   ```
   That artifact was produced on 2026-09-01 06:15 — **before** commit `9de8880`.
   Its `model.pt` begins `50 4b 03 04` (a PyTorch zip) and its first pickle key
   is `tracker_string`: it was written by `torch.save(plain_model.state_dict())`,
   the code path `9de8880` removed.

So "budget too short" is not a regression — **it is the normal, documented path**,
and it used to work.

---

## C. Which characterization is right

### Not (i) — "the 20-epoch recipe is too short; lengthen the budget"

Three independent reasons:

1. **Lengthening does not reach the gate.** The gate is
   `TRAINING_COMPLETE`, not "a dendrite was added". With the library defaults
   actually in force for `mnist_pai` (`max_dendrites: 100`,
   `n_epochs_to_switch: 10`, `p_epochs_to_switch: 2` — see
   `.smoke/results/mnist_pai/dendrites_fp32/PAI_config.json`), reaching
   completion would take on the order of the 100–170 epochs the stored dynamic
   runs needed *with `max_dendrites` cut to 1–3*. `resnet18_cifar10` had two
   switches and still no file.
2. **The benchmark actively prevents it.** `freeze_dendrite_updates_fraction=0.20`
   (`pipeline.py:2363`) tears the tracker down for the last 20% of *any* fixed
   budget (`training.py:3897-3903`, `4319-4322`), so PAI cannot signal
   completion at the end of a fixed-budget run no matter how long it is.
3. **The default is documented as fixed-budget.** `cli.py:384-390`:
   > `--dynamic-dendritic-training`: "Use PerforatedAI's open-ended dynamic FP32
   > dendritic training mode. **By default, dendritic FP32 runs use the same
   > fixed epoch budget as the matching non-dendritic run and freeze dendrite
   > insertion for the final 20% of epochs.**"

   `train_dendrites_until_complete` is only ever true when that flag (or
   `--pai-capacity-check`) is passed: `pipeline.py:2356-2359`. `mnist_pai`'s own
   recipe comment states the intent explicitly (`pipeline.py:1470-1478`):
   *"Upstream runs open-ended until PAI reports completion; 20 epochs is the
   fixed budget this benchmark gives it (02_OPEN_DECISIONS.md D9)."*

   Under (i), **every** default-mode dendritic run in the repo is broken —
   which is exactly what is observed, since `lenet5` fails identically.

### Not exactly (ii), but (ii) is the right instinct

(ii) says the guard is wrong to raise "when zero dendrites were ever added".
That is true but under-states the scope in two ways:

- The failure is not conditioned on zero dendrites. It is conditioned on PAI not
  finishing its search. A run with **four retained dendrites** that stops on the
  epoch budget fails identically.
- The design already has a first-class, non-fatal answer for the zero-dendrite
  case. `_dendrite_audit` (`training.py:2339-2440`) returns
  `"no_retained_insertion"` when `switch_count < 2`, and the caller merely
  **prints** it (`training.py:4978-4987`). A dendritic run that adds nothing is
  an *expected labelled outcome*, not an error — `distilbert` is the proof.
  `9de8880` made that outcome unreachable by crashing before the audit runs.

### (iii) — the actual defect: an unconditional gate added by `9de8880`

`git log -S "_export_final_pai_artifact"` yields exactly one commit: `9de8880`.
The diff changed the dendritic artifact write from

```python
# BEFORE (worked for distilbert, epoch_budget, 0 switches)
_final_clean_stats = (
    _final_clean_pai_parameter_stats(_plain_model) if use_dendrites else None
)
...
torch.save(plain_model.state_dict(), checkpoint_path)   # unconditional
```

to

```python
# AFTER — training.py:4944-4954, 5015
final_checkpoint_path = output_dir / _MODEL_PT
_final_clean_stats = None
if use_dendrites:
    if not pai_save_name:
        raise RuntimeError("dendritic final export requires a PAI save name")
    _export_final_pai_artifact(pai_save_name, final_checkpoint_path)   # ← hard gate
    _final_clean_stats = _final_pai_artifact_stats(final_checkpoint_path)
...
    checkpoint_already_written=use_dendrites,   # ← suppresses the torch.save fallback
```
(`_persist_stage_artifacts`, `training.py:1822-1835`.)

The motivation is legitimate and documented in
`information/results_analysis/2026-09-01-mpnn-actor-critic-audit-repair.md:44-56`:
`prepare_final_model()`'s `model.parameters()` over-counts residual PAI
scaffolding (MPNN by 7,494; actor-critic by 384), and PAI's own final-clean
serialization is the only source that agrees with `param_counts.csv`. But every
artifact in that analysis is a **capacity diagnostic**, i.e. a
`--pai-capacity-check` run, which by construction always reaches
`TRAINING_COMPLETE`. The fix was validated only on the path where the file is
guaranteed and then applied to *all* dendritic conditions, including the default
fixed-budget path where it is guaranteed **absent**. No test covers
`_export_final_pai_artifact` — the two tests added in `9de8880`
(`tests/test_dynamic_pai_followup.py:283-315`) exercise only the in-process
`_final_clean_pai_parameter_stats`.

### A second, still-latent defect in the same commit

Even when `final_clean_pai.pt` *does* exist, the copy changes the on-disk format
of `model.pt` from a PyTorch zip to **safetensors** (verified: the header of
`experiment_results/showcase5_capacity_seed0/PAI/m5_…/final_clean_pai.pt` is
`f8 0e 00 …{"bn1.main_module.num_batches_tracked":{"dtype":"I64",…`).
But the consumer of a dendritic `model.pt` is:

`src/dendritic_benchmark/pipeline.py:255-266`
```python
def _load_state(self, model: Any, checkpoint_path: Path) -> Any:
    ...
    state = torch.load(checkpoint_path, map_location=choose_device(), weights_only=True)
```
reached from `_prepare_condition_model` → `_load_source_checkpoint`
(`pipeline.py:1104-1112`) for every `dendrites_q*` condition. `torch.load`
cannot read safetensors, and `grep -rn safetensors src/` shows no reader
anywhere on this path (`models.py` uses it only for HF downloads). Also, the
sibling stage writers `_persist_post_pqat_snapshot` and
`_persist_over_budget_snapshot` still default to `checkpoint_already_written=False`,
so `continued_until_complete/model.pt` stays a torch zip while the canonical
`model.pt` beside it is safetensors — two different formats under one name.
So `9de8880` breaks the FP32→quantized dendritic chain as soon as the first
failure is worked around by running with `--dynamic-dendritic-training`.

---

## D. What controls `n_epochs_to_switch`

**Nothing per-model. It is the installed PerforatedAI library default in every
default run.**

Chain:

1. `pipeline.py:901-906`
   ```python
   def _pai_dynamic_schedule(self, model_key: str) -> PAIDynamicSchedule | None:
       """Return only explicit caller overrides; PAI owns the default policy."""
       _ = model_key                      # model_key deliberately unused
       if self._pai_override is None:
           return None
       return self._pai_override.apply_to_schedule(None)
   ```
   `self._pai_override` is populated only from the CLI `--pai-override <json>`
   (`cli.py:461-470`, `cli.py:902-905`), which is rejected unless exactly one
   `--models` is selected (`cli.py:930-939`). There is no per-model table.
   `_pai_fixed_switch_interval` (`pipeline.py:890-899`) is likewise
   `model_key`-independent and returns the `--pai-fixed-switch-interval`
   diagnostic flag only.

2. `compat.py:334-378` `_configure_dynamic_pai_schedule` — with
   `schedule=None`, nothing is set beyond `set_initial_correlation_batches`.
   `_configure_pai_training_schedule` (`compat.py:380-395`) first calls
   `_restore_pai_library_schedule_defaults` (`compat.py:317-332`), which
   deliberately re-reads a **fresh `GPA.PAIConfig()`** and copies the eleven
   fields in `_PAI_LIBRARY_SCHEDULE_FIELDS` (`compat.py:301-315`) back onto the
   global config, so an override from an earlier model in the same process
   cannot leak forward.

3. Defaults actually applied, read back from the run's own snapshot
   `.smoke/results/mnist_pai/dendrites_fp32/PAI_config.json`:

   | setting | value |
   |---|---|
   | `switch_mode` | `1` (= `DOING_HISTORY`) |
   | `n_epochs_to_switch` | **10** |
   | `p_epochs_to_switch` | **2** |
   | `history_lookback` | 1 |
   | `initial_history_after_switches` | 0 |
   | `max_dendrites` | **100** |
   | `improvement_threshold` | `[0.001, 0.0001, 0.0]` |
   | `first_fixed_switch_num` / `fixed_switch_num` | 1 / 250 (inert in HISTORY mode) |
   | `retain_all_dendrites` | false |

   These match `perforatedai 3.2.7`'s `PAIConfig()` exactly; the benchmark sets
   none of them.

Consequence for the budget arithmetic: `mnist_pai`'s recipe is 20 epochs
(`pipeline.py:1475-1478`), of which `ceil(20 × 0.20) = 4` are dendrite-frozen,
leaving 16 live epochs against `n_epochs_to_switch = 10` and
`max_dendrites = 100`. The smoke run used `--recipe-override {"max_epochs": 14}`
(`.smoke/override14.json`), leaving 11 live epochs. Reaching *one* switch was
already marginal; reaching `TRAINING_COMPLETE` was never on the table.

---

## Minimal correct fix (NOT implemented)

Restore the invariant "a dendritic run always publishes an artifact", and treat
PAI's final-clean file as a **preferred, optional** source of the same artifact:

1. **`training.py:1657-1666`** — make `_export_final_pai_artifact` return
   `bool` (copy and return `True` when `final_clean_pai.pt` exists, return
   `False` otherwise) instead of raising.
2. **`training.py:4944-4954`** — branch on that result:
   - exported → `_final_clean_stats = _final_pai_artifact_stats(final_checkpoint_path)`
     and `checkpoint_already_written=True` (today's behaviour, now only where
     the file genuinely exists);
   - not exported → `_final_clean_stats = _final_clean_pai_parameter_stats(_plain_model)`
     and `checkpoint_already_written=False`, i.e. exactly the pre-`9de8880`
     path that produced `distilbert`'s valid `no_retained_insertion` artifact.
     Print a one-line note rather than raising; `_dendrite_audit`
     (`training.py:2339-2440`) already labels the outcome correctly and will
     not award `verified_retained` without the raw switch/architecture
     agreement, so the audit gate described in
     `2026-09-01-mpnn-actor-critic-audit-repair.md:73-86` is not weakened.
3. **`training.py:5015` / `pipeline.py:255-266`** — before this can ship for
   real, decide the `model.pt` format question. Two artifacts under one filename
   with two incompatible encodings is not viable: either
   `_load_state` must sniff the safetensors magic and dispatch, or the
   final-clean export should be written to a **sibling** name (e.g.
   `final_clean_pai.safetensors`) used purely for parameter/topology accounting
   while `model.pt` stays a `torch.save` state dict. The latter is smaller and
   keeps the `dendrites_fp32 → dendrites_q*` chain working, and it still
   delivers everything the audit-repair analysis asked for, since that analysis
   needs the *counts*, not the file location.
4. **Tests** — add coverage for the branch that has never had any: a dendritic
   run whose PAI directory contains no `final_clean_pai.pt` must still write
   `model.pt` + `metrics.json` and record `dendrite_audit_status =
   no_retained_insertion`.

### Sanity check on the fix

`mnist_pai / dendrites_fp32` at 14 epochs would then publish an artifact whose
topology is identical to the dense model, with
`dendrite_audit_status: "no_retained_insertion"` — the honest scientific record
that PAI's HISTORY schedule found no plateau to switch on inside a budget where
validation accuracy was still climbing (0.9910 @ epoch 5 → 0.9902 @ epoch 10).
That is a finding about the schedule, and it belongs in `metrics.json`, not in a
stack trace. Whether `mnist_pai` *should* get a longer budget or a tuned
`n_epochs_to_switch` is a separate, legitimate question for
`02_OPEN_DECISIONS.md` D9 — but it is a science question, not the cause of this
crash.

---

## Cited files

- `src/dendritic_benchmark/training.py` — 1525, 1572-1644, 1646-1654, **1657-1666**, 1670-1717, 1822-1835, 2339-2440, 3858-3871, 3897-3903, 3930-3944, 3960-3978, 4319-4322, 4376-4381, 4886-4890, **4944-4954**, 4978-4987, 5015
- `src/dendritic_benchmark/compat.py` — 33, 40, 301-315, 317-332, 334-378, 380-395, 809-811, 824-846
- `src/dendritic_benchmark/pipeline.py` — 255-266, 434, 448, 890-899, **901-906**, 1104-1112, 1470-1478, 2356-2363
- `src/dendritic_benchmark/cli.py` — 384-390, 461-470, 902-905, 930-939
- `tests/test_dynamic_pai_followup.py` — 283-315 (the only tests added by `9de8880`)
- `information/results_analysis/2026-09-01-mpnn-actor-critic-audit-repair.md` — 44-56, 73-86
- `.smoke/logs/run_20260902_005915.txt` — 345-347, 364, 402-455
- `.smoke/results/mnist_pai/dendrites_fp32/PAI_config.json`
- `.smoke/results/PAI/mnist_pai_dendrites_fp32_b3b56310fe7f/…switch_epochs.csv`, `…param_counts.csv`
- `experiment_results/dendritic_quantization_launch5_seed0_20260831_1750/distilbert/dendrites_fp32/{record.json,pai_summary.json,model.pt}`
- `experiment_results/showcase5_capacity_seed0/**/pai_summary.json`

---

## Fix applied (2026-09-02)

Implemented as specified in "Minimal correct fix" above. The invariant
"a dendritic run always publishes an artifact" is restored, and PAI's
final-clean file is now a *preferred, optional* accounting source rather than a
precondition for publishing.

### What changed

| File:line | Change |
|---|---|
| `src/dendritic_benchmark/training.py:58-73` | New `_FINAL_CLEAN_PAI_PT` (`final_clean_pai.pt`, PAI's own file) and `_FINAL_CLEAN_PAI_EXPORT` (`final_clean_pai.safetensors`, the benchmark's sibling copy) constants, with the reason the two names must stay distinct. |
| `src/dendritic_benchmark/training.py:1538-1542` | Comment on `_artifact_path`'s legacy extensionless fallback list, recording that the new sibling is deliberately *not* a candidate artifact path. |
| `src/dendritic_benchmark/training.py:1676-1709` | `_export_final_pai_artifact` now returns `bool` (copied / PAI wrote nothing) instead of raising. Docstring carries the evidence from sections A and B above. |
| `src/dendritic_benchmark/training.py:1864-1885` | `_persist_stage_artifacts` lost its `checkpoint_already_written` parameter; `model.pt` is once again an unconditional `torch.save(state_dict())`. Nothing passes `True` any more, because the export no longer targets `model.pt`. |
| `src/dendritic_benchmark/training.py:2018-2021` | The sibling export joins `additional_files` in the artifact manifest, so when it exists it is sealed as the evidence behind the reported counts. |
| `src/dendritic_benchmark/training.py:4998-5040` | `train_and_evaluate` branches on the bool: exported → `_final_pai_artifact_stats(sibling)`; not exported → `_final_clean_pai_parameter_stats(_plain_model)` plus a one-line `[pai] …` note. Long comment records why both sources exist and why the fallback cannot fake a retained dendrite. |
| `src/dendritic_benchmark/pipeline.py:447-450` | Matching comment on `_artifact_path`'s legacy list. |
| `tests/test_p2_artifact_properties.py:265-418` | New `test_a_dendritic_run_publishes_an_artifact_without_pais_final_export`: a PAI directory with a header-only `switch_epochs.csv` and no `final_clean_pai.pt` must still yield `model.pt` (readable by `torch.load(..., weights_only=True)`), `metrics.json`, `no_retained_insertion` in both `pai_summary.json` and the sealed manifest, and no safetensors sibling. Offline, CPU-only. |

`9de8880`'s intent is preserved: when PAI *does* write its final-clean export
(capacity-check and dynamic runs), the artifact's parameter counts and topology
hash still come from that file's tensor header, which is the only source that
agrees with `param_counts.csv`. Only the file's *destination* moved.

`_dendrite_audit` was not touched. `_require_verified_dendritic_pqat_source`
(`pipeline.py:1905-1929`, landed 2026-08-30, two days before `9de8880`) was not
touched either.

### The file-format split (item 3)

Resolved with the diagnosis's recommended option, not the sniffing option.
`model.pt` is *always* a `torch.save` state dict; PAI's safetensors export is
copied beside it as `final_clean_pai.safetensors` and read only for
parameter/topology accounting. Verified on the fixed run:

```
$ xxd -l 4 .smoke/fix/lenet5/dendrites_fp32/model.pt
00000000: 504b 0304                                PK..
$ python -c "import torch; s=torch.load('.smoke/fix/lenet5/dendrites_fp32/model.pt',
    map_location='cpu', weights_only=True); print(len(s), list(s)[:4])"
107 ['tracker_string', 'features.0.main_module.weight',
     'features.0.main_module.bias', 'features.3.main_module.weight']
```

Same encoding and same leading `tracker_string` key as distilbert's
pre-regression artifact (section B).

### Verification

```
$ .venv/bin/ty check
All checks passed!

$ .venv/bin/pytest
FAILED tests/test_dynamic12_hf_pqat.py::Dynamic12HFPQATTests::test_post_run_verifier_rejects_missing_pqat_stage
FAILED tests/test_p2_docs.py::DocumentationIndexTests::test_superseded_documents_name_their_replacement
2 failed, 145 passed, 1558 subtests passed in 5.38s
```
Both failures pre-date this work and reference files that do not exist.

```
$ .venv/bin/dqb docs --check
[01:23:35] [docs] information/CURRENT_GUIDE.md is current.
```

#### End-to-end

```
$ .venv/bin/dqb run --models lenet5 --conditions dendrites_fp32 dendrites_q8 \
    --recipe-override .smoke/override14.json --results-root .smoke/fix \
    --comparison-root .smoke/fixcmp --logging-dir .smoke/fixlogs
[01:24:23] [done] lenet5 / base_fp32 — Accuracy: 0.9867
[pai] live dendrite updates frozen; continuing with a standard optimizer.
[pai] lenet5 | dendrites_fp32: PerforatedAI did not reach training-complete, so it
  wrote no final-clean inference artifact; the benchmark serialized its own
  prepared model and counted that instead.
[audit] lenet5 | dendrites_fp32: no_retained_insertion — raw PAI switch log has no
  candidate-insertion switch
[01:25:27] [done] lenet5 / dendrites_fp32 — Accuracy: 0.9905
```

`.smoke/fix/lenet5/dendrites_fp32/` now contains `model.pt` (0.51 MB) and
`metrics.json`; `record.json` reads

```json
"metric_value": 0.9904999732971191,
"best_metric_value": 0.9887999892234802,
"best_epoch": 12,
"param_count": 61706,
"nonzero_params": 61706,
"dendrite_audit_status": "no_retained_insertion",
"dendrite_audit_reason": "raw PAI switch log has no candidate-insertion switch"
```

which is precisely the outcome the "Sanity check on the fix" section predicted:
the dense topology, published, honestly labelled.

**`dendrites_q8` is then refused, by a different and older guard.**
`_require_verified_dendritic_pqat_source` (`pipeline.py:1905-1929`) raises

> `lenet5 / dendrites_q8 requires a verified retained dendrites_fp32 source
> before PQAT; found no_retained_insertion.`

This is not the regression and not a consequence of the fix: the guard landed in
`facfbf4` (2026-08-30), it fires only *after* the FP32 arm has successfully
published, and it is doing its documented job — "a quantized dendritic arm only
fine-tunes the saved FP32 graph; it cannot create a retained dendrite itself."
Section B's `distilbert` precedent agrees: its shipped launch directory has
`base_q1 … base_q8` and `dendrites_fp32`, and **no** `dendrites_q*` at all. It
was deliberately left alone (the brief forbids weakening the audit gate, and
this is the same gate one level up).

To prove the `_load_state` path from item 3 end to end anyway, the same command
was rerun with `--pai-fixed-switch-interval 3` so the FP32 arm retains a
dendrite and clears the guard:

```
$ .venv/bin/dqb run --models lenet5 --conditions dendrites_fp32 dendrites_q8 \
    --recipe-override .smoke/override14.json --pai-fixed-switch-interval 3 \
    --results-root .smoke/fixsw --comparison-root .smoke/fixswcmp \
    --logging-dir .smoke/fixswlogs
EXIT=0
[01:30:27] [done] lenet5 / dendrites_fp32 — Accuracy: 0.9892
[01:30:28] [done] lenet5 / dendrites_q8   — Accuracy: 0.9891
```

| condition | param_count | nonzero | audit status |
|---|---:|---:|---|
| `dendrites_fp32` | 120,840 | 120,840 | `verified_retained` |
| `dendrites_q8` | 120,840 | 119,289 | `inherited_verified_retained` |

Both `model.pt` files begin `50 4b 03 04`. `dendrites_q8` rebuilt itself from the
dendritic FP32 `model.pt` through `pipeline._load_state`, which is the exact
call the safetensors split would have broken — and the FP32 arm still earned
`verified_retained` only because PAI's raw switch log and raw
`param_counts.csv` independently agreed on 120,840 against a 61,706-parameter
dense reference, i.e. the audit gate is intact.
