# Implementation review of the P2 override/manifest work

<!-- status-banner -->
> **Status: current (2026-08-31).** A strict verification of whether the code changes required by [03_execution_matrix.md](03_execution_matrix.md) were implemented correctly, plus the defects found and fixed. Records the state of the runner *before* the first sweep; it is not a run report and contains no experimental results.

**Reviewed:** the uncommitted working tree on top of `b5ccdd1` ("created optimization plan"), covering
`cli.py`, `pipeline.py`, `plans.py`, `quantization.py`, `training.py`, and
`tests/test_dynamic_pai_followup.py` / `tests/test_p2_overrides.py`.
**Method:** read every changed line against the requirements in
[03_execution_matrix.md](03_execution_matrix.md) and [00_assessment.md](00_assessment.md),
then checked the claims that depend on data — CSV column names, module names, and existing
artifacts — against the files actually on disk rather than against the code's comments.

## Verdict

The required objects were built and wired correctly, but shipped with four defects. One of
them would have silently defeated the entire mechanism it was written to enable: every sweep
trial would have discarded and retrained its own artifact on each invocation. All four are
fixed and covered by regression tests. `./scripts/ci.sh` is green (it was red on arrival, for
an unrelated pre-existing reason recorded below).

## What was verified as correct

| requirement (03_execution_matrix.md) | status | evidence |
|---|---|---|
| immutable `RecipeOverride` / `PAIOverride` objects | correct | `plans.py`; frozen dataclasses, unknown JSON keys rejected by name |
| complete field coverage | correct | all 17 `ModelTrainingRecipe` fields; all 7 `PAIDynamicSchedule` fields |
| "initial-history must equal lookback" EMA guard | correct | enforced in `PAIOverride.__post_init__`, not left to a downstream warning |
| override exposed via CLI | correct | `--recipe-override` / `--pai-override`, both documented in the generated guide |
| an override is one trial for one model | correct | guarded in both `cli._handle_run` and `BenchmarkRunner.run`; the CLI copy is required because a `--jobs>1` run never calls `run()` on the coordinator's runner |
| DistilBERT batch-4 LR scaling preserved across N trials | correct | the override is applied *before* `with_batch_size`, so 2e-5 @ 32 still becomes 2.5e-6 @ 4 |
| M5 gets a model-specific target selection + coverage test | correct | `.conv4`/`.fc1` perforated, the other seven parameter-bearing modules tracked; zero parameters left untyped |
| DistilBERT NP1 as a target/track-only registry branch + coverage test | correct | `distilbert_classifier_only` variant; the test builds from `DistilBertConfig` so it stays offline |
| manifest: `recipe_override`, `pai_override`, effective post-batch-scaling recipe, all target/track IDs, first candidate/retention/completion epochs, p-phase ceiling, final-clean topology hash, paired-control identity, quantizer revision | all present | written to `metrics.json` and to the artifact manifest's identity/telemetry |
| topology hash is architecture, not weights | correct | hashes name + owning-module class + shape only |
| param-count/switch-epoch join by `Switch Number` | correct | column headers confirmed against real logs under `results/dynamic5/PAI/` and `results/top10/PAI/` |

## Defects found and fixed

### 1. `PAIOverride.to_dict()` emitted tuples — every override run would retrain itself

**Severity: high.** `plans.py:218`

`to_dict()` is written to `metrics.json` and read back by
`BenchmarkRunner._condition_metadata_current` to decide whether a saved artifact still matches
the requested configuration. JSON has no tuple type, so a recorded
`[".fc1"]` could never compare equal to an expected `(".fc1",)`. Any `PAIOverride` that set
`module_ids_to_perforate`, `track_only_module_ids`, or `improvement_threshold` — that is, every
target-set trial and every `H(...)` row in the execution matrix — would judge its own
freshly-written artifact stale and retrain it from scratch on the next invocation, overwriting
the previous trial's result.

`PAIDynamicSchedule.to_dict()` already converted its one sequence field to a list for exactly
this reason; `PAIOverride` did not inherit the precaution.

**Fix:** `_PAI_SEQUENCE_OVERRIDE_FIELDS` (`plans.py:143`) now names the three sequence fields
once, and `to_dict()` emits them as lists. Regression test asserts
`json.loads(json.dumps(d)) == d`, and a second test asserts `to_dict()` round-trips back
through `from_json_file`. An ad-hoc check confirmed all six values
`_condition_metadata_current` compares now survive the round trip.

### 2. An override could change PAI targets with no coverage check

**Severity: high.** `pipeline.py:918`

00_assessment.md: "Perforation targets must cover every parameter either as perforated or
tracked; otherwise PAI may enter a debugger during candidate training." 03_execution_matrix.md:
"an alternate target requires matching track-only coverage and a structural smoke test, not
merely an ID edit."

Nothing enforced this at runtime. Worse, this benchmark *suppresses* the evidence: PAI's
"Parameter does not have parameter_type attribute" warning is swallowed by
`compat._consume_pai_debugger_message` and its `pdb` call is neutralized by
`_suppress_pai_debugger`. A `--pai-override` naming an incomplete target set therefore produces
a silently mistyped run — no warning, no crash, and a result that looks ordinary. GP1
(`.readout.0` + `.readout_gate` only) and AP1 (`.fc1` only) are both exactly this shape.

**Fix:** `_reject_uncovered_override_parameters` raises before any training starts, naming the
orphaned parameters. Scoped deliberately to the override path — see the open finding below for
why a blanket guard was not viable.

### 3. `m5`'s target set changed without an artifact revision

**Severity: medium-high.** `pipeline.py:122`

M5's targets moved from type-selecting *every* `Linear`/`Conv1d` to the AP0 pair. The new
`module_ids_to_perforate` manifest field is compared "only when recorded" — a deliberate and
correct choice, since artifacts predating the field would otherwise all be invalidated at once
— which means an M5 dendritic artifact trained under the old blanket target set reads back as
`None` and is treated as matching. It would have been silently reused under targets it was
never trained with.

`_MODEL_ARTIFACT_REVISIONS` is the existing mechanism for precisely this, and
`resnet18_cifar10`, `saint_adult` and `pointnet_modelnet40` each got one when their targets
changed. M5 did not.

**Fix:** added `"m5": "optimization_ap0_late_target_v1"`. No M5 artifacts exist on disk today,
so this was latent rather than active corruption.

### 4. `first_candidate_epoch` discarded an epoch the log plainly held

**Severity: low.** `training.py:1838`

The milestone read only the `{switch_number: epoch}` map, so a log without a usable
`Switch Number` column reported `None` even though `switch_epochs` held the value. It also
keyed off the lowest *epoch* rather than the lowest switch number, which answers a slightly
different question than the one the field's name asks.

**Fix:** keys off `min(switch_number)`, and falls back to the ordered `switch_epochs` list.

### 5. Two smaller hardening fixes

- **Empty sequence overrides** (`plans.py:202`). An empty `module_ids_to_perforate` — a `[]`
  typo — is not "no override": `_perforation_modules_to_perforate` falls back to
  type-selecting every `Linear`/`Conv1d`/`Conv2d` when the ID list is empty, so a typo would
  have *widened* the target set to the blanket wrapping 01_initial_five_plan.md forbids as a
  primary comparison. Now rejected at construction.
- **`source_commit` on a dirty tree** (`pipeline.py:810`). The field recorded a bare `HEAD`
  even with uncommitted changes, naming a commit whose checkout does not reproduce the run —
  the exact irreproducibility the field exists to prevent, and most sweep work happens on a
  dirty tree. Now suffixed `-dirty`; an unavailable `git status` is treated as dirty rather
  than as clean.

### 6. Unrelated pre-existing test failure

`information/CAPACITY_MATCHED_DENSE_CONTROLS.md` (added in `2c1d9db`, before this work) had no
status banner and no `HISTORICAL_INDEX.md` entry, so `tests/test_p2_docs.py` was failing on
arrival. Fixed, and `information/optimization/` itself is now indexed — `HISTORICAL_INDEX.md`
claims to cover every document under `information/` but omitted this directory.

## Open findings — not changed, decision required

### MPNN, a launch-cohort model, runs with untyped parameters today

Checked by building each registered model and testing every parameter against the union of its
perforate, track-only, and parameter-track ID lists:

| model | uncovered parameters |
|---|---|
| `mpnn` | `node_encoder.0/2.*`, `layers.0.edge_mlp.*`, and others |
| `gru_forecaster` | `head.0.weight`, `head.0.bias` |
| `vae_mnist` | `encoder.1/3.*`, `mu.*`, and others |
| `m5`, `distilbert` (both variants), `tcn_forecaster`, `pointnet_modelnet40`, `resnet18_cifar10`, `saint_adult` | none |

This is pre-existing, not introduced by the P2 work, and the suppressed warning means nothing
surfaces it. But `mpnn` is model 5 of the launch cohort and GP0 is defined as "current six IDs",
so the protocol's own stress-test model would run with parameters PAI cannot type. Repairing it
means changing three models' default target sets and bumping three artifact revisions — a change
to the launch-cohort configuration, which is a decision for the operator, not a review fix.
This is also why the new coverage guard is scoped to the override path: a blanket guard would
refuse to run three currently-registered models.

### Smaller gaps

- `paired_control_identity` is recorded as a hardcoded `None`. The matched dense-continuation
  and capacity-matched controls of 00_assessment.md's validity protocol step 4 do not exist in
  the runner; [../CAPACITY_MATCHED_DENSE_CONTROLS.md](../CAPACITY_MATCHED_DENSE_CONTROLS.md) is
  a proposal, not an implementation.
- `before_pqat/` snapshots carry no `topology_hash` — it is computed after training, and the
  pre-PQAT snapshot is written earlier. "The same FP32 source topology for PTQ and PQAT"
  therefore cannot be verified from the manifests alone.
- Dense (non-dendritic) artifacts carry no `topology_hash` at all, since it is derived from
  PAI's final-clean model. A capacity-matched dense control is exactly where comparing
  topologies would pay off.
- `RecipeOverride` cannot set a field *to* `None`, because `None` means "unset". A trial that
  wants to disable gradient clipping, the step schedule, or the LR-schedule horizon
  (`grad_clip_norm`, `lr_decay_every`, `lr_schedule_epochs`) cannot express it.
- `source_commit` is recorded but deliberately excluded from staleness comparison. This is a
  sound engineering call — otherwise any commit would force a full retrain — but it is a
  documented deviation from "the artifact identity must include… the source commit".

## Still true before the first sweep

Nothing in this review changes the standing position of
[00_assessment.md](00_assessment.md): no dendritic quantization claim is reportable yet, the
537 historical records remain manifest-`unknown`, and the advancement checklist in
[01_initial_five_plan.md](01_initial_five_plan.md) is still entirely unchecked. This work makes
a sweep *expressible and auditable*; it does not start one.
