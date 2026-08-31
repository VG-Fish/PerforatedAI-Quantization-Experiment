# Cross-cutting audit: results, docs, tests, dependencies, and static analysis

**Status:** analysis only; no source changes or deletions.

## Required command evidence

### Tests

`.venv/bin/pytest -q` passed: **35 passed**. This is a narrow suite: two tracked test files, heavily concentrated on Dynamic12 follow-up behavior and private helpers. It is not evidence that all 24 model/data paths or all 12 conditions work.

### `ty check`

`ty 0.0.1-alpha.3` found **13 diagnostics**:

- `data.py:931` return annotation mismatch (`list[LiteralString]` vs `list[str]`): likely a real typing issue.
- `models.py:1741–1742` unresolved optional `perforatedai` imports: configure third-party import discovery or isolate the optional adapter.
- `plot_dynamic_seven.py:65` and several `TemporaryDirectory` calls: old ty overload limitations, but annotate/upgrade rather than blanket-ignore.
- `test_dynamic12_hf_pqat.py:40`, `161`, `169`, `tests/test_dynamic_pai_followup.py:68–70`, `574`: possibly-unbound/dynamic-module typing; improve guards and typed loader protocols.

Record the exact output in CI and establish a supported ty version. Do not “fix” by weakening annotations globally.

### SonarCloud and `curl` + `jq`

After explicit user authorization, `sonar-scanner` authenticated from the project environment, analyzed all 12 indexed source files, uploaded the report, and completed successfully. SonarCloud processed the task at `2026-08-31T02:12:41Z` for revision `49c8d63a03c094a07984214df715d66699b27e0f`; analysis ID: `e8998202-ef29-448e-a3f7-272ab649b9ec`. The refreshed issue inventory was fetched with authenticated `curl` and inspected with `jq` from `/api/issues/search` using `ps=500`.

That complete page contains **47 issues: 33 open, 14 closed**. Open breakdown: 21 CRITICAL, 11 MAJOR, 1 MINOR; 27 code smells, 5 bugs, 1 vulnerability. By file: `pipeline.py` 11, `data.py` 9, `training.py` 8, `compat.py` 2, `models.py` 2, `cli.py` 1. By rule: S3776 complexity 13, S1192 duplicated literals 6, S1244 float equality 5, with S1172 unused parameters, S3358 nested conditional, S5713 redundant exception, S5754 exception flow, S6929 missing reduction dim, S8786 regex backtracking, and S6680 user-controlled loop bound.

Open issue inventory (all 33):

| file | lines/rules | action |
|---|---|---|
| `pipeline.py` | 611, 1411, 1617, 2097 / S3776; 641, 662, 1761 / S1192; 739 / S3358; 1575, 2348 / S1172; 1998 / S8786 | extract target/metadata/worker policies; constants; remove unused params; safe regex |
| `data.py` | 1272, 1332, 1400, 1479 / S3776; 426, 1779 / S6929; 1474 / S1244; 1682 / S1192; 2325 / S6680 | split builders; explicit dims; tolerant float comparisons; validate bounds |
| `training.py` | 861, 1806, 3194, 3731, 4155 / S3776; 966 / S1244 twice; 2518 / S5713 | extract metrics/audit/checkpoint/epoch orchestration; use tolerance; remove redundant exception |
| `compat.py` | 854 / S1192; 900 / S5754 | constant for module path; re-raise/stop as expected |
| `models.py` | 1659 / S1192; 1734 / S1244 | constant for torchvision import; avoid float equality |
| `cli.py` | 784 / S1244 | avoid exact float comparison |

Closed issues should remain in history but not be reintroduced: prior unsafe filesystem paths, unsafe loading, wrong argument types, and redundant complexity/literals.

## Results/reporting and scientific validity

`results.py:164–248` infers legacy statuses from filesystem sidecars; this is fragile and currently lets `legacy_unchecked` through as reportable. The report schema should require a signed/hashed run manifest and explicit validity status. Add noise-floor, seed, switch requested/observed, phase termination, and capacity-control fields to `dendrite_audit.csv`; otherwise the August 30 conclusions require bespoke analysis.

`plots.py` is a general-purpose 722-line chart library with custom overlap detection and multiple heatmap/bar/scatter variants. Keep only plots used by the canonical report; move exploratory annotations to a separate analysis package. Never generate a chart with zero placeholders that visually resembles missing data without a validity legend.

## Documentation contradictions and deletion markers

- **[SUPERSEDED / ARCHIVE]** `information/DYNAMIC9_PAI_GRAPH_AUDIT.md` claims perforation works on every judged model; the August 30 effect audit explicitly supersedes its scientific verdict. Keep as historical evidence, label it superseded, and remove it from “current guidance.”
- **[CONSOLIDATE]** `DOCUMENTATION.md` (1,205 lines), `MODEL_REFERENCE.md` (1,270), `CLI_DIAGRAMS.md` (475), `README.md`, and dated run/caveat/fix docs repeat command and architecture facts. Generate a short current guide from specs/CLI and retain dated audits in a historical index.
- **[REWRITE]** `DYNAMIC_DENDRITIC_MIGRATION.md` contains proposed/obsolete migration steps; mark completed steps and delete proposals after implementation decisions are settled.
- **[ARCHIVE]** `DYNAMIC8_RUN`, `DYNAMIC9_RUN`, `CODE_REVIEW`, and `REMAINING_FIXES` contain valuable provenance but should not read as current instructions. Add status banners and a canonical “current state” page.
- **[DELETE CANDIDATE]** ignored `.DS_Store`, `.scannerwork`, `.ruff_cache`, `.pytest_cache`, `.uv-cache`, `.venv`, `.egg-info`, `.perforated_tools`, old PID files, and generated log roots from a clean checkout/retention process; do not delete during this audit.
- **[DELETE/ARCHIVE CANDIDATE]** `archive/*.zip`, top-level `logs*`, `comparison/`, `results/`, and Dynamic9–11 generated trees after checksums and evidence manifests are stored. `data/` is a cache, not source; document external storage rather than versioning it.
- **[REVIEW]** duplicate `.claude/skills` and `PAI Skills/` installations; retain one source of truth and package the rest separately.

## Test/dependency cleanup

The package declares runtime and developer tools together (`pytest`, `vulture`, `bandit`, `deadcode` alongside torch/data/PAI). Split optional `dev`/`audit` extras and pin a supported Python/ty/Sonar workflow. Add smoke tests generated from `MODEL_SPECS × CONDITION_SPECS`, artifact validity tests, and at least three-seed statistical tests for the models used to claim dendrite effects.
