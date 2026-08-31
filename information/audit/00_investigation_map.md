# Cleanup audit: investigation map

**Date:** 2026-08-30  
**Status:** initial reconnaissance; no cleanup has been applied  
**Write boundary:** audit notes in `information/audit/` and the requested final report are the only intended repository changes during this audit.

## Experimental north star

The repository exists to answer one narrow question: **do dendrites make perforated models perform better under quantization?** Cleanup should make that comparison easier to run, reproduce, and trust. Features or artifacts that do not support model construction, controlled training, quantization, dendrite insertion, measurement, or reproducibility need an explicit justification to remain.

The existing `information/DENDRITE_EFFECT_AUDIT_2026-08-30.md` materially changes the cleanup priority. It reports that the quantized dendritic arms are currently non-reportable, stale artifacts can contaminate runs, fixed-switch scheduling is based on an obsolete workaround, and `legacy_unchecked` results are incorrectly reportable. Cleanup must preserve the evidence needed to resolve those issues; deleting historical evidence before a reproducible replacement exists would be premature.

## Repository shape

- Tracked application surface: 12 package modules, 2 test modules, Dynamic12 entrypoints/configuration, top-level documentation, and packaging metadata.
- Core package size: about 19,000 lines when tests and current experiment helpers are included.
- Largest modules: `training.py` (4,413 lines), `data.py` (2,814), `pipeline.py` (2,564), `models.py` (1,867), `compat.py` (1,296), and `cli.py` (932).
- Working directory size: about 48 GB. Major local categories are `data/` (~20 GB), `archive/` (~15 GB), `experiments/` (~9.4 GB), `results/` (~2.3 GB), and logs (~129 MB across the top-level log roots).
- Git currently reports a clean tracked worktree. Most bulky/generated material is ignored rather than versioned.
- No repository-local `AGENTS.md` was found.

## Rough areas for deep review

### 1. Training engine, quantization, and PAI lifecycle — highest risk

`training.py` combines generic metric calculation, PPO/RL behavior, quantization and QAT/PQAT shadow-state handling, artifact serialization, PAI log interpretation, optimizer construction, batch execution, memory monitoring, checkpoint/resume compatibility, dendrite updates, LR scheduling, and top-level training orchestration. This is too many reasons for one 4,413-line module to change. Review for:

- dead compatibility paths and redundant state-loading implementations;
- opportunities to isolate quantization, PAI lifecycle, persistence, metrics, and the epoch engine;
- duplicated configuration/state concepts;
- comments explaining obsolete fixed-interval behavior;
- unsafe permissive fallbacks that turn invalid evidence into reportable results;
- tests that couple to private functions because no stable subsystem boundary exists.

### 2. Runner/pipeline, CLI, benchmark, and artifact lifecycle — highest risk

`pipeline.py`, `cli.py`, and `benchmark.py` collectively own planning, configuration, model-state projection, job execution, cleanup, reporting, and latency benchmarking. Review for:

- duplicate compatible-state loading in `pipeline.py`, `training.py`, and `benchmark.py`;
- stale `results/PAI/<save_name>/` directories and ambiguous ownership of clearing/resume behavior;
- command sprawl and features that do not serve the core experiment;
- enormous per-model conditional tables embedded in methods;
- duplicated flags/config defaults across CLI and runner layers;
- destructive behavior hidden behind ordinary execution paths;
- a smaller architecture centered on an explicit experiment plan and immutable run identity.

### 3. Models, data, and specifications — high complexity

`models.py` and `data.py` hold roughly 25 unrelated architectures and their loaders in monolithic registries/functions. Review for:

- models/datasets that are unsupported, unvalidated, redundant, or low-value for the dendrite-under-quantization question;
- repeated factory branches and model-key conditionals that should be declarative adapters;
- hidden coupling between model keys, batch shapes, losses, metrics, PAI targets, and pipeline hyperparameters;
- cached/generated dataset artifacts and obsolete rollout versions;
- stale comments and compatibility code for model families no longer intended to run.

### 4. Results, plots, experiment helpers, and statistical validity — highest scientific risk

`results.py`, `plots.py`, and Dynamic12 scripts turn records into claims. Review for:

- the known `legacy_unchecked` reportability defect;
- separate sources of truth for raw records, PAI sidecars, summaries, and plots;
- scripts that reproduce logic already present in the package;
- legacy plots and stale comparisons that look authoritative but are invalid;
- missing noise-floor, seed, capacity-control, and within-run fields required by the August 30 effect audit;
- one auditable result schema with provenance and explicit validity gates.

### 5. Tests, typing, dependencies, and static analysis — medium/high risk

Only two tracked test files cover a broad package; both focus on recent Dynamic12 follow-ups and import many private symbols. Review for:

- core paths with no behavioral tests;
- redundant test scaffolding and brittle white-box tests;
- runtime dependencies that are actually development-only (`pytest`, `vulture`, `bandit`, `deadcode`);
- missing declared tooling/configuration for the required `ty check` workflow;
- Sonar scope/configuration issues (current properties contain duplicate `sonar.projectKey`, broad exclusions, obsolete report settings, and Python versions beyond the package's declared support floor).

### 6. Documentation, historical experiments, generated files, and repository hygiene — high bloat

The top-level docs contain overlapping dated audits, migration notes, run reports, caveats, model reference material, and remaining-fixes lists. Locally, Dynamic9–12 run directories and several older top-level result/log roots coexist. Review for:

- superseded documents to delete or move to a clearly labeled historical archive;
- claims contradicted by the August 30 dendrite-effect audit;
- old Dynamic9/10/11 scripts and outputs after extracting reproducibility metadata;
- redundant `.DS_Store`, caches, `.scannerwork`, egg-info, old rollout variants, duplicate skill installations, logs, stale PID files, and old archives;
- a retention policy that distinguishes source, reproducible configuration, canonical evidence, disposable cache, and historical archive.

## Deep-review assignments

The next phase will assign independent reviews for:

1. training/PAI/quantization internals;
2. pipeline/CLI/benchmark architecture and artifact lifecycle;
3. models/data/specifications and removable feature surface.

The primary review will cover results/reporting/statistics, repository hygiene, docs, tests/dependencies, `ty check`, and Sonar issue extraction, then cross-check all findings across subsystem boundaries.

## Evidence still to collect

- Complete function/class/import maps and cross-module coupling.
- Static type failures from `ty check`.
- A fresh Sonar scan and complete paginated issue inventory fetched with `curl` and transformed/checked with `jq`.
- Dead-code candidates from reference searches and available analyzers, manually checked for dynamic/CLI use.
- Test results and collection coverage (without modifying caches/configuration where avoidable).
- Exact file/directory deletion candidates, with disposition and prerequisites.
- Documentation contradictions and a proposed canonical documentation set.

## Reconnaissance update (2026-08-30)

- `pytest -q`: 35 passed.
- `ty check`: 13 diagnostics; see `40_crosscutting_static_docs.md` for the complete categorized output.
- SonarCloud issue API (complete `ps=500` page via `curl` + `jq`): 47 total, 33 open, 14 closed; 21 open critical issues. After explicit user authorization, a fresh authenticated scan completed successfully at `2026-08-31T02:12:41Z` for commit `49c8d63a03c094a07984214df715d66699b27e0f` (analysis ID `e8998202-ef29-448e-a3f7-272ab649b9ec`).
- Vulture/deadcode agree on one likely unused helper (`compat.py:pai_root`) and one unused parser field (`data.py:ring_closure`), while flagging framework-dispatched `forward` methods and serialized dataclass fields as false-positive candidates.
- The three assigned deep reviews hit the agent usage limit; the primary agent completed equivalent reviews and wrote `10_training_pai_quantization.md`, `20_pipeline_cli_artifacts.md`, and `30_models_data_scope.md`.

The final synthesis is `information/audit_report.md`. No source, configuration, test, dependency, generated-result, dataset, log, archive, or documentation file outside `information/audit/` was modified.
