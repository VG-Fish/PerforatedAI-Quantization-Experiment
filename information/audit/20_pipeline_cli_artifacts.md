# Deep review: pipeline, CLI, benchmark, and artifact lifecycle

**Scope:** `pipeline.py`, `cli.py`, `benchmark.py`, `log_utils.py`, entrypoints, tests/docs.  
**Status:** analysis only; no source changes or deletions.

## Findings

### A. Runner responsibility is too broad

`BenchmarkRunner` (pipeline.py:291 onward) owns model construction, per-model target selection (597–770), recipe/hyperparameter policy, artifact compatibility (1382–1490), dependency scheduling, result reuse, and report generation. The same class embeds dozens of model-key branches. Split into declarative `ExperimentPlan`/`ModelAdapter`/`ArtifactStore` services. A plan should fully identify model revision, condition, source dependency, PAI variant, seed, schedule, and output namespace before execution.

### B. Sonar confirms high-complexity hotspots

Open Sonar issues include cognitive complexity at `pipeline.py:611` (16), `1411` (37), `1617` (17), `2097` (33), and `training.py` callers. These are not cosmetic: target selection, metadata validity, and worker progress are policy boundaries. Extract pure predicates and table-driven configuration before changing behavior. Sonar also reports unused parameters `ignore_saved` at 1575 and `stop_pattern` at 2348; verify call graphs, then remove them.

### C. Stale artifact lifecycle remains the central validity hazard

The August 30 audit identifies never-cleared `results/PAI/<save_name>/` directories and TCN stale folders with only two score rows but leftover `switch_N.pt` checkpoints. Current safeguards are spread across `--fresh` (CLI 442–450), `_condition_metadata_current` (1411–1490), PAI root setting, and result-side audit inference. Replace with an atomic run namespace containing a manifest, immutable run ID, source revision, and an owned PAI directory. A run must refuse to reuse an existing namespace unless an explicit resume token matches the manifest.

**[DELETE CANDIDATE after migration]** orphan PAI trees, stale `switch_*.pt`, PID files, and generated report folders that are not referenced by a manifest. Preserve a tar/zip or checksum manifest first if they are evidence for the August audit.

### D. Duplicate state and validity logic

Pipeline validates metadata, training validates checkpoints, benchmark re-filters state, and results infers legacy status. Centralize artifact validation and return a machine-readable verdict. Report/plot code must consume only that verdict; it should not infer validity from filesystem paths.

### E. CLI command and option bloat

The CLI exposes `run`, `download_data`, `compare`, `generate_graphs`, `benchmark_models`, and `clean`, with many options (`--detach`, `--status`, `--fresh`, model scale, PAI variants, PQAT, seed, jobs, interval). `README.md`, `CLI_DIAGRAMS.md`, and `DOCUMENTATION.md` duplicate these descriptions. Retain the core commands needed to run paired experiments and analyze them; consider moving latency benchmarking, graph regeneration, and clean bookkeeping to separate optional tooling. Generate help/docs from one option registry.

`--detach` intentionally skips final reports and requires a manual compare; `--status` replays old logs. These are operational features, not experimental factors, and should be isolated from the scientific runner.

### F. Clean command has hidden mutation risk

`cli.py:74–188` records absolute paths in `.dqb/command_config.json` and later removes them. It has safety checks, but generated-path ownership is inferred from prior invocations and can outlive the run that created it. Replace with per-run manifests and a dry-run/confirmation workflow; never infer deletion targets from a global append-only history. **Do not execute cleanup during this audit.**

### G. Benchmark can measure a different architecture

The August 28 code review fixed latency loading for dendritic artifacts. This remains a boundary to lock down: benchmark input must reference the exact validated `model.pt` and artifact manifest, not reconstruct a model from a model key. Persist model revision, topology/audit status, quantization revision, and input-shape provenance in every latency row.

### H. Worker orchestration duplicates shell/process state

`run_parallel` and helpers 1924–2467 maintain log-root variants, PID/worker detection, stream parsing, progress rendering, stop patterns, termination, and final reports. Simplify to one process supervisor with structured JSON events. Delete text-log parsing once all workers emit a stable event schema; retain a human-readable log as a derived view.

## Exact cleanup candidates

- **[REMOVE PARAMETER]** `ignore_saved` and `stop_pattern` after call-site verification (Sonar S1172).
- **[CONSOLIDATE]** model/condition compatibility predicates and state loading into one artifact validator.
- **[REWRITE]** per-model target branches into declarative specs/adapters; constants for repeated `.head.0`, `.head.3`, and `manifest.csv` (Sonar S1192).
- **[REFACTOR]** worker progress and report-generation complexity (Sonar S3776); simplify regex at pipeline.py:1998 (S8786).
- **[DELETE AFTER MANIFEST MIGRATION]** unreferenced PAI checkpoints, duplicate Dynamic9–11 generated output trees, stale PID/log roots.
- **[KEEP UNTIL EVIDENCE PACKAGED]** canonical Dynamic12 results and raw PAI sidecars used by the August 30 audit.

