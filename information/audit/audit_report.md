# Codebase cleanup audit report

**Date:** 2026-08-30  
**Scope:** repository cleanup before implementing the recommendations in `information/DENDRITE_EFFECT_AUDIT_2026-08-30.md`  
**Mode:** analysis only. No source/config/dependency/test/result deletion or cleanup was performed. The only writes are the audit notes and this report under `information/`.

## Executive conclusion

The codebase is not blocked by a lack of features; it is blocked by too many overlapping execution paths and too much ambiguity about which artifact is authoritative. The experiment’s scientific question is narrow, but the repository contains a 24-model/12-condition framework, multiple dynamic experiment generations, broad data/cache support, duplicate checkpoint loaders, per-model policy branches, historical reports, and 48 GB of local generated material. The most urgent cleanup is therefore validity infrastructure, followed by decomposition of the orchestration monoliths. Deleting historical evidence before it is indexed would make the dendrite/quantization question harder to answer.

The detailed evidence is split into:

- [initial investigation map](audit/00_investigation_map.md)
- [training/PAI/quantization review](audit/10_training_pai_quantization.md)
- [pipeline/CLI/artifact review](audit/20_pipeline_cli_artifacts.md)
- [models/data/scope review](audit/30_models_data_scope.md)
- [static analysis/docs/results review](audit/40_crosscutting_static_docs.md)

## Priority order

| priority | cleanup objective | why it comes first | proposed outcome |
|---|---|---|---|
| P0 | Make reportability fail closed and bind every result to one immutable run namespace | The August 30 audit found stale PAI trees, invalid TCN/GRU evidence, and `legacy_unchecked` rows marked reportable; quantized dendritic evidence is currently non-reportable | One manifest-owned artifact tree; explicit `verified_retained`, `invalid`, `unknown`, and quantization-revision statuses; no filesystem inference |
| P0 | Remove the obsolete fixed-switch default and instrument requested vs observed schedules | Fixed intervals are not honored (GRU 8→1, VAE 20→8, SAINT 100→19, TCN never fired) and the prior HISTORY workaround is documented as fixed | HISTORY as default; fixed mode diagnostic-only; observed switch/termination reasons persisted |
| P0 | Consolidate checkpoint/state loading | Three permissive shape-filter implementations can restore a structurally compatible but scientifically wrong epoch/topology | One loader returning mismatch details; required tensors/topology must validate before evaluation/reporting |
| P1 | Split `training.py` and `pipeline.py` by policy boundary | 4,413-line training module and 2,564-line pipeline module mix unrelated concerns and make factor changes risky | Typed modules for metrics, quantization, PAI lifecycle, checkpoints, artifacts, epoch engine, plans, and workers |
| P1 | Reduce model/data key coupling and default experiment breadth | A model requires edits across model, data, loss, metric, PAI, recipe, CLI, and docs; 24 models multiply invalid combinations | Declarative model adapters; small evidence-backed default roster; exploratory models opt-in |
| P1 | Address Sonar correctness/security findings | 33 open issues include 5 bugs, 1 vulnerability, and 13 complexity findings | Refactor by policy boundary, validate loop bounds/paths, fix float comparisons and reductions, remove unused parameters |
| P2 | Consolidate docs and archive generated history | Five large overlapping guides plus dated audits contradict current state; local output is 48 GB | One generated current guide plus indexed historical evidence and documented retention policy |
| P2 | Expand tests and type/tooling configuration | 35 passing tests cover only a narrow Dynamic12/private-helper slice; `ty` reports 13 diagnostics | Matrix smoke tests, artifact-validity/property tests, three-seed statistical tests, supported ty/Sonar CI config |

## Architecture simplification target

The desired dependency direction is:

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

The plan should contain model revision, dataset/preprocessing revision, condition/source dependency, seed, quantization revision, PAI variant/schedule, and output namespace. Reporting should accept only an artifact-store verdict, never discover validity by looking for arbitrary files.

## Redundant/dead code ledger

### High-confidence candidates

- `src/dendritic_benchmark/compat.py:797` `pai_root`: reported unused by both Vulture and deadcode; no repository references. Delete only after checking external consumers.
- `src/dendritic_benchmark/data.py:148` `_Bond.ring_closure`: assigned but never read; either delete or promote to a tested graph feature.
- `src/dendritic_benchmark/training.py:2518`: redundant exception class caught by a broader exception (Sonar S5713).
- `src/dendritic_benchmark/pipeline.py:1575` `ignore_saved` and `:2348` `stop_pattern`: Sonar reports unused parameters; verify calls then remove.
- Ignored/generated `.DS_Store`, caches, `.scannerwork`, `.ruff_cache`, `.pytest_cache`, `.uv-cache`, `.venv`, egg-info, PID files, and dashboard scratch files: delete from a clean checkout/retention process, not from this audit.

### Keep, but simplify or relabel

- `nn.Module.forward` methods flagged by deadcode/Vulture are framework entrypoints and are not dead.
- `TrainingRecord` fields flagged as unused are serialized/reporting API; remove only with a schema migration.
- PAI resume/checkpoint compatibility, raw PAI sidecars, and canonical Dynamic12 evidence are needed to reproduce the August 30 validity findings.
- Dynamic9/10/11 reports and variants are historical evidence, but should be marked superseded/diagnostic and excluded from current instructions.

## File/comment/doc disposition ledger

| item | disposition | prerequisite |
|---|---|---|
| fixed-interval workaround comments in `compat.py` and migration prose | rewrite/delete obsolete rationale | HISTORY behavior and observed switch logging verified |
| `information/DYNAMIC9_PAI_GRAPH_AUDIT.md` | retain as historical, mark superseded | add current-state index |
| `information/DYNAMIC8_RUN_*.md`, `DYNAMIC9_RUN_*.md`, `CODE_REVIEW_*.md`, `REMAINING_FIXES.md` | retain provenance, archive from current guide | extract unresolved actions and checksums |
| `information/DOCUMENTATION.md`, `MODEL_REFERENCE.md`, `CLI_DIAGRAMS.md`, top-level `README.md` | consolidate/generated current docs | make specs and CLI option registry authoritative |
| `information/DYNAMIC_DENDRITIC_MIGRATION.md` | mark completed vs proposed; remove stale proposals | settle implementation choices |
| Dynamic9–11 generated results/logs/comparisons | archive or delete after evidence manifest | checksum, provenance index, canonical Dynamic12 selection |
| `archive/*.zip` | retain only if checksums and purpose are indexed; otherwise delete | confirm no unique evidence |
| `data/` | externalize/cache; do not version as source | document dataset versions and preprocessing hashes |
| duplicate `.claude/skills` and `PAI Skills/` | choose one source of truth | verify install/runtime consumers |

## Static-analysis baseline

### Tests

`pytest -q` passed **35 tests**. This is a baseline only, not broad coverage.

### Typing

`ty 0.0.1-alpha.3` found **13 diagnostics**. The likely real issue is `data.py:931` (`list[LiteralString]` returned under `list[str]`); optional `perforatedai` imports at `models.py:1741–1742` need an adapter/import-discovery decision. Remaining diagnostics involve dynamic module loading, `TemporaryDirectory`, and possibly-unbound tensors. Record a supported ty version and fix/annotate deliberately.

### Sonar

A fresh authenticated SonarCloud scan completed successfully for commit `49c8d63a03c094a07984214df715d66699b27e0f` (analysis ID `e8998202-ef29-448e-a3f7-272ab649b9ec`, processed `2026-08-31T02:12:41Z`). It contains **47 issues**, of which **33 are open**: 21 critical, 11 major, 1 minor; 27 code smells, 5 bugs, and 1 vulnerability. All 47 were fetched from the issue API with authenticated `curl` (`ps=500`) and enumerated with `jq`; the full grouped inventory and every open file/line/rule are in [40_crosscutting_static_docs.md](audit/40_crosscutting_static_docs.md).

## Scientific guardrails for the next implementation pass

1. Never call a dendritic arm reportable without raw candidate-insertion evidence, a retained parameter/topology increase, and a current quantization-evaluation revision.
2. Keep FP32 dendrite and dense controls paired by seed, initialization, data order, and a matched training horizon; add a more-training control and capacity-matched dense controls.
3. Persist the noise floor, seed spread, requested/observed switch epochs, effective dendrite LR, phase termination reason, and test outcome in the standard audit CSV.
4. Treat PB correlation as an internal training diagnostic, not as evidence of generalization.
5. Do not stack another PAI conversion on the already perforated HF ResNet; keep its comparison role explicit.
6. Keep stale artifacts quarantined until their provenance is captured; delete only after a manifest/checksum preserves what was learned.

## Definition of done for cleanup implementation

- One authoritative model/condition/plan registry replaces scattered key branches.
- One artifact manifest and validator govern resume, comparison, benchmark, and plot eligibility.
- One checkpoint loader reports all missing/unexpected/shape-mismatched keys.
- HISTORY scheduling is default; fixed mode is opt-in diagnostic with requested/observed telemetry.
- Quantization is applied exactly once and stage metadata is mandatory.
- Core package modules are each below a maintainable size or have clear single responsibilities.
- Open Sonar bugs/vulnerability and high-value complexity issues are resolved or explicitly justified.
- `ty check`, tests, smoke matrix, and statistical validation run in CI.
- Current docs are generated/verified from specs and CLI; historical docs are indexed and labeled.
- Only after the above: remove the ledger’s approved generated files and legacy branches.
