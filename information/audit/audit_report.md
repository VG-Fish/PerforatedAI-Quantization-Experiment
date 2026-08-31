# Codebase cleanup audit report

**Audited:** 2026-08-30 · **Last reconciled against the tree:** 2026-08-31
**Scope:** repository cleanup in support of `information/DENDRITE_EFFECT_AUDIT_2026-08-30.md`

The audit ran as five separate reviews (investigation map, training/PAI/quantization,
pipeline/CLI/artifacts, models/data/scope, and cross-cutting static analysis). Their
findings are merged here, with everything the P0–P2 passes closed removed rather than
restated: what is written below is what is still true. The original five documents are
in this file's git history.

## Executive conclusion

The codebase was never blocked by a lack of features. It was blocked by too many
overlapping execution paths and too much ambiguity about which artifact is
authoritative. The experiment's question is narrow, but the repository carries a
24-model/12-condition framework, several dynamic experiment generations, broad
data/cache support, per-model policy branches, historical reports, and ~48 GB of local
generated material. So the urgent work was validity infrastructure first, decomposition
of the orchestration monoliths second — and deleting historical evidence before it was
indexed would have made the dendrite/quantization question harder to answer, not easier.

P0–P2 closed the validity infrastructure and the documentation/evidence work. The
decomposition is half done: `training.py` and `pipeline.py` are still the two modules
that mix the most unrelated policy.

## Priority ledger

| priority | objective | status |
|---|---|---|
| P0 | Make reportability fail closed and bind every result to one immutable run namespace | **done** — `artifacts.py` manifests; `results.py` reads a manifest verdict instead of inferring validity from the filesystem; `legacy_unchecked` is no longer reportable |
| P0 | Remove the obsolete fixed-switch default; instrument requested vs observed schedules | **done** — HISTORY is the default, fixed mode is a labelled diagnostic, and `dendrite_audit.csv` carries requested/observed switch epochs and a termination reason |
| P0 | Consolidate checkpoint/state loading | **done** — `checkpointing.load_state_dict_checked` is the single loader for `pipeline.py`, `training.py`, and `benchmark.py`, and returns a missing/unexpected/shape-mismatch report |
| P1 | Split `training.py` and `pipeline.py` by policy boundary | **partial** — `quantization.py`, `plans.py`, `model_adapters.py`, `checkpointing.py`, and `workers.py` are extracted; the metric registry, PAI lifecycle, recipe table, per-model PAI targeting, and epoch engine are not |
| P1 | Reduce model/data key coupling and default experiment breadth | **partial** — `DEFAULT_MODEL_KEYS` narrows a bare `dqb run` to five evidence-backed models; the per-model branches in `pipeline.py` and `data.py` remain |
| P1 | Address Sonar correctness/security findings | **partial** — the unchecked loop bound, implicit reductions, float equalities, redundant exception, and unused parameters are fixed; the complexity findings track the split above |
| P2 | Consolidate docs and index generated history | **done** — `CURRENT_GUIDE.md` is generated from the registries, every hand-written document is banded and indexed, and `EVIDENCE_INDEX.md` plus `RETENTION_POLICY.md` exist |
| P2 | Expand tests and type/tooling configuration | **done** — 88 tests and 716 subtests, `ty check` clean, dev/audit dependency groups, `scripts/ci.sh` in CI |

## What P0–P2 built

**Validity.** `artifacts.py` writes an `artifact_manifest.json` (schema version 1) that
binds one immutable artifact identity to the SHA-256 of every file it owns. Reporting
consumes that verdict and nothing else. A dendritic arm is reportable only at
`verified_retained`/`inherited_verified_retained`; a quantized arm additionally needs the
current quantization-evaluation revision.

**Boundaries.** `quantization.py` owns PTQ dispatch and the QAT shadow-state lifecycle.
`plans.py` owns immutable recipe/condition/source-checkpoint types and the
`ExperimentPlan`, including a dataset/preprocessing revision in every artifact identity.
`model_adapters.py` is the registry for task kind, primary metric, constructor
capabilities, and default scope. `workers.py` owns restart and process-group termination.
`checkpointing.py` owns state loading.

**Documentation and evidence.** `docs.py` renders `CURRENT_GUIDE.md` from `specs.py`,
`model_adapters.py`, `artifacts.py`, `statistics.py`, and the `cli.py` argparse registry;
`dqb docs --check` fails in CI when the checked-in guide drifts. `evidence.py` plus
`dqb evidence_index` write `evidence_index.json` and `EVIDENCE_INDEX.md`. The first index
found **537 training records across 26.6 GB, every one reporting `unknown`** — all stored
evidence predates the artifact manifest and cannot be made reportable without being
re-run. That is why nothing was deleted, and it is what `RETENTION_POLICY.md` governs.

**Statistics.** `statistics.py` owns the seed-paired effect estimate: paired differences
signed by metric direction, the dense control's own seed spread as the noise floor, an
exact Student-t p-value, and a verdict that is `insufficient_seeds` below three paired
seeds however large the difference looks. Arms whose artifact does not validate are
dropped from the statistics rather than contributing a stale number.

## Still open

### Training, PAI, and quantization

- **`training.py` is still an orchestration monolith** (4,406 lines). It holds metric and
  loss implementations, dataset-shape inference, PAI setup and log parsing, optimizer
  construction, batch execution, memory guards, LR scheduling, dynamic dendrite updates,
  and artifact persistence. Extract `metrics.py`, `pai_runtime.py`, `epoch_engine.py`, and
  a training-artifact writer, leaving `train_and_evaluate` a thin coordinator.
- **The dendrite learning-rate policy is fragmented** across `_scheduled_learning_rate`,
  `_dendrite_learning_rate`, `_apply_lr_schedule`, optimizer param grouping, and recipe
  overrides in `pipeline.py`. The effective per-group rate is now persisted, but a single
  schedule object should produce both the backbone and the dendrite rate. Only three
  models opt into `dendrite_lr_min_factor`, so a late dendrite still inherits a collapsed
  cosine everywhere else (`MEASUREMENT_CAVEATS.md` §11).
- **Candidate phases end on a hard cap, not on convergence.** `MAX_DENDRITE_PHASE_EPOCHS
  = 8` can terminate an MPNN candidate while its PB correlation is still rising
  (`DENDRITE_EFFECT_AUDIT_2026-08-30.md` §5.4). The termination reason is recorded now;
  the cap itself should read as a timeout, not a convergence decision.
- **`compat.py` (1,227 lines) has no supported-version contract.** The
  `_call_if_available`/optional-setter fallbacks, output and debugger suppression, and
  old save-name flattening are only meaningful against specific PerforatedAI releases.
  Pin and test the supported API, then delete the branches for versions no longer
  supported. Never catch an import or runtime failure and continue with an unlabelled
  dense result.

### Pipeline, CLI, and artifact lifecycle

- **`BenchmarkRunner` (2,439 lines) still owns too much**: model construction, per-model
  PAI target selection, recipe policy, artifact compatibility, dependency scheduling,
  result reuse, and report generation, with dozens of model-key branches inline. The
  `ExperimentPlan` type exists; the runner does not yet consume it as its only input.
- **Orphan PAI trees are indexed but not retired.** `results/PAI/<save_name>/` is still
  not owned by a run namespace, so stale `switch_*.pt` and score rows from an earlier run
  can sit beside a live one. `RETENTION_POLICY.md` covers what has to exist before they
  are removed; the structural fix is a run-owned PAI directory.
- **`dqb clean` infers its deletion targets from a global append-only history**
  (`.dqb/command_config.json`), so a recorded path can outlive the run that created it.
  Per-run manifests plus a dry-run/confirm workflow would replace it.
- **Worker orchestration parses text logs.** `run_parallel` and its helpers maintain log
  roots, PID/worker detection, stream parsing, progress rendering, and stop patterns.
  Emit structured JSON events and keep the human-readable log as a derived view.
- **CLI surface is operational and scientific at once.** `--detach` skips final reports
  and needs a manual `compare`; `--status` replays old logs. Neither is an experimental
  factor and both belong outside the scientific runner.

### Models, data, and scope

- **Model-key coupling is repeated across modules.** A key branches independently in
  pipeline target selection, track-only lists, recipe overrides, training
  forward/loss/metric dispatch, and data bundle construction. A new model needs edits in
  several places and can silently omit a factor. Fold these into the `model_adapters.py`
  registry: constructor, data builder, forward adapter, loss/metric, supported
  conditions, PAI targets, default recipe.
- **The PAI variants are permanent branches in the main runner.** `gru_gate_ablation`,
  `tcn_head_output`, `tcn_head_both`, `vae_latent`, and `mpnn_capacity` are valid
  diagnostics only while their hypotheses and controls stay documented; they should be
  experiment specs, not runner branches.
- **`data.py` (2,839 lines) mixes unrelated domains and owns cache policy.** Vision,
  speech, audio, time series, text, graph, RL rollouts, ModelNet40, ISIC, and medical
  loaders share one module with the cache paths. Split the domain adapters and centralise
  cache metadata (dataset version, preprocessing revision, seed, split hash) so a cache
  filename carries every preprocessing factor.
- **Unvalidated models stay available but unmarked.** Exploratory models are out of the
  default roster but have no smoke test or condition matrix declaring what they support.

### Results, plots, and reporting

- **`plots.py` (722 lines) is a general-purpose chart library**, with custom overlap
  detection and several heatmap/bar/scatter variants beyond what the canonical report
  uses. Keep the report's plots; move exploratory annotation elsewhere. A chart of zeroed
  placeholders must never look like real data without a validity legend.

## Redundant/dead code ledger

Only items still present in the tree:

- `src/dendritic_benchmark/compat.py` `pai_root()` — no caller anywhere in the
  repository (`set_pai_root` is the one that is used); Vulture and deadcode agree.
  Delete after checking for external consumers.
- `src/dendritic_benchmark/data.py` `_Bond.ring_closure` — assigned by the SMILES parser
  and never read. Delete, or promote it to a tested graph feature.
- Ignored/generated `.DS_Store`, `.scannerwork`, `.ruff_cache`, `.pytest_cache`,
  `.uv-cache`, `.venv`, egg-info, PID files, and dashboard scratch files — removed by a
  clean checkout, not by a source change. Old rollout cache versions
  (`heuristic_rollouts.pt`, `_v2`) fall under `data/`, which `RETENTION_POLICY.md`
  classifies as disposable cache.

Not dead, despite what the analysers say: every `nn.Module.forward` (PyTorch dispatches
them dynamically) and the `TrainingRecord` fields flagged as unused (serialised reporting
API — removing one is a schema migration).

## Architecture simplification target

```text
specs / model adapters / data adapters
                 ↓
          immutable ExperimentPlan
                 ↓
       artifact-owned Runner/Worker
          ↙       ↓        ↘
   PAI adapter  epoch engine  quantization adapter
          ↘       ↓        ↙
          validated ArtifactStore
                 ↓
       report/statistics/plots
```

The plan carries model revision, dataset/preprocessing revision, condition/source
dependency, seed, quantization revision, PAI variant/schedule, and output namespace.
Reporting accepts an artifact-store verdict and never discovers validity by looking for
files.

## Scientific guardrails for further work

1. Never call a dendritic arm reportable without raw candidate-insertion evidence, a
   retained parameter/topology increase, and a current quantization-evaluation revision.
2. Keep FP32 dendrite and dense controls paired by seed, initialisation, data order, and
   training horizon; add a more-training control and capacity-matched dense controls.
3. Persist the noise floor, seed spread, requested/observed switch epochs, effective
   dendrite LR, phase termination reason, and test outcome in the standard audit CSV.
4. Treat PB correlation as an internal training diagnostic, not evidence of
   generalisation.
5. Do not stack another PAI conversion on the already perforated HF ResNet; keep its
   comparison role explicit.
6. Keep stale artifacts quarantined until their provenance is captured; delete only after
   a manifest or checksum preserves what was learned.

## Definition of done

Achieved: one artifact manifest and validator govern resume, comparison, benchmark, and
plot eligibility; one checkpoint loader reports every missing, unexpected, and
shape-mismatched key; HISTORY scheduling is the default with fixed mode an opt-in
diagnostic carrying requested/observed telemetry; quantization is applied exactly once
with mandatory stage metadata; current docs are generated and verified from the specs and
the CLI while historical docs are indexed and labelled; `ty check`, the tests, the smoke
matrix, and the documentation check run in CI.

Outstanding: one authoritative model/condition/plan registry replacing the scattered key
branches; core modules each below a maintainable size or with a single clear
responsibility; the remaining Sonar complexity findings resolved or explicitly justified;
and only after those, removal of the ledger's approved generated files and legacy
branches.

## Static-analysis baseline

- **Tests:** `pytest` — 88 tests, 716 subtests, passing. The matrix smoke test walks all
  24 models × 12 conditions through recipe resolution, condition planning, constructor
  kwargs, dependency ordering, and PAI namespace minting without training anything.
- **Typing:** `ty check` — 0 diagnostics. The version is pinned in the `dev` dependency
  group and the interpreter is pinned to the package's 3.12 support floor.
- **Sonar:** the last full scan was `2026-08-31T02:12:41Z`, commit
  `49c8d63a03c094a07984214df715d66699b27e0f` (analysis `e8998202-ef29-448e-a3f7-272ab649b9ec`):
  47 issues, 33 open — 21 critical, 11 major, 1 minor; 27 code smells, 5 bugs, 1
  vulnerability. That inventory predates the P1 and P2 commits, which fixed the bugs, the
  vulnerability, and the unused parameters; the open complexity findings (S3776 in
  `pipeline.py`, `data.py`, and `training.py`) track the module split above. Re-scan
  before treating any count here as current.
