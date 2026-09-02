# Documentation index

Every document under `information/`, what state it is in, and what it may still be
cited for. Three of these files are generated; the rest are hand-written and keep
their original numbers on purpose, because they are the evidence of what was true
when they were written. A hand-written document is trimmed only when a section of it
duplicates a generated file or proposes work that has since shipped; the trim is
recorded in that document's own banner.

**Statuses**

- **current (generated)** — rendered from the code by `uv run dqb docs` or
  `uv run dqb evidence_index`. Never edit by hand.
- **current** — hand-written and still true.
- **historical** — accurate for the run or date it describes; not current instructions.
- **superseded** — a specific claim in it has been overturned; the replacement is named.

## Current

| document | status | cite it for |
|---|---|---|
| [CURRENT_GUIDE.md](CURRENT_GUIDE.md) | current (generated) | model roster, condition grid, reportability rules, command reference |
| [EVIDENCE_INDEX.md](EVIDENCE_INDEX.md) | current (generated) | what generated evidence exists on disk, in which run namespace, with which manifest verdict |
| [RETENTION_POLICY.md](RETENTION_POLICY.md) | current | which generated trees may be archived or deleted, and what must exist first |
| [MEASUREMENT_CAVEATS.md](MEASUREMENT_CAVEATS.md) | current (dated line numbers) | the eleven measurement caveats, their root causes, and which results each one invalidates |
| [CAPACITY_MATCHED_DENSE_CONTROLS.md](CAPACITY_MATCHED_DENSE_CONTROLS.md) | current (proposal) | the topology-matched dense-control design — not yet implemented in the runner |

## Upstream base models (2026-09-02)

`information/base_examples/` is the working record of porting the five PerforatedAI
`examples/base_examples` models into this benchmark. Read it in numeric order.

| document | status | cite it for |
|---|---|---|
| [UPSTREAM_BASE_MODELS_CHANGE_SUMMARY.md](UPSTREAM_BASE_MODELS_CHANGE_SUMMARY.md) | historical (2026-09-02) | the handoff written at the end of the port: what was added, and the recipe table as of that day |
| [base_examples/01_UPSTREAM_AUDIT.md](base_examples/01_UPSTREAM_AUDIT.md) | current | the pinned upstream commit and, per example, its architecture, targets, recipe and reported numbers |
| [base_examples/02_OPEN_DECISIONS.md](base_examples/02_OPEN_DECISIONS.md) | current | which departures from upstream are the user's decision (D1, D2, D5) and which are the implementation's |
| [base_examples/03_IMPLEMENTATION_RECORD.md](base_examples/03_IMPLEMENTATION_RECORD.md) | historical | what the first implementation pass built, module by module |
| [base_examples/04_DIAGNOSIS_pai_final_artifact.md](base_examples/04_DIAGNOSIS_pai_final_artifact.md) | historical | why the final PAI artifact export was missing, and the fix |
| [base_examples/05_STATUS_AND_HANDOFF.md](base_examples/05_STATUS_AND_HANDOFF.md) | historical | the state of the port at that handoff |
| [base_examples/06_DIAGNOSIS_control_conditions_abort_the_sweep.md](base_examples/06_DIAGNOSIS_control_conditions_abort_the_sweep.md) | historical | why an unsupported control condition used to kill a whole sweep, and the skip-and-record fix |
| [base_examples/IMPLEMENTATION_FINDINGS.md](base_examples/IMPLEMENTATION_FINDINGS.md) | historical | the discovery pass that preceded any code change |

## Problems and analyses

| document | status | cite it for |
|---|---|---|
| [problems/2026-09-01-distilbert-no-retained-dendrite.md](problems/2026-09-01-distilbert-no-retained-dendrite.md) | historical (2026-09-01) | why DistilBERT finished with no retained dendrite |
| [problems/2026-09-01-storage-exhaustion-and-monitoring.md](problems/2026-09-01-storage-exhaustion-and-monitoring.md) | historical (2026-09-01) | the disk exhaustion that stopped that sweep and what was added to catch it |
| [problems/live-monitor-events.md](problems/live-monitor-events.md) | historical (append-only log) | raw worker errors as they were emitted; not a diagnosis |
| [results_analysis/2026-09-01-mpnn-actor-critic-audit-repair.md](results_analysis/2026-09-01-mpnn-actor-critic-audit-repair.md) | historical (2026-09-01) | the audit verdict on the two non-reportable dendritic artifacts from that sweep |

## Historical

| document | status | cite it for | do not cite it for |
|---|---|---|---|
| [DOCUMENTATION.md](DOCUMENTATION.md) | historical | experiment rationale, PerforatedAI integration walkthrough, quantization background, the baseline-quality record | the roster or the condition grid |
| [MODEL_REFERENCE.md](MODEL_REFERENCE.md) | historical | per-model hyperparameters, preprocessing, and PAI targeting narrative | metrics, metric directions, or which models are in the default roster |
| [CLI_DIAGRAMS.md](CLI_DIAGRAMS.md) | historical | command flowcharts and the generated-output directory layout | flags, defaults, or the list of subcommands |
| [REMAINING_FIXES.md](REMAINING_FIXES.md) | historical (2026-08-07) | why the baseline-quality pass changed what it changed | the outstanding-work list; the audit ledger owns that now |
| [DYNAMIC_DENDRITIC_MIGRATION.md](DYNAMIC_DENDRITIC_MIGRATION.md) | historical | the reasoning behind moving from fixed-interval to HISTORY scheduling, and the PerforatedAI 3.2.3 `save_name` constraint | schedule values — `compat.py::PAI_DYNAMIC_SCHEDULE_DEFAULTS` is the live one |

## Superseded

| document | superseded by | what specifically was overturned |
|---|---|---|
| [MODEL_SELECTION.md](MODEL_SELECTION.md) | [CURRENT_GUIDE.md](CURRENT_GUIDE.md) | criterion 1's exclusion of `pointnet_modelnet40` and `resnet18_cifar10`, which rested on an expired PerforatedAI token that has since been renewed. Its cost table and diversity reasoning still hold. |

## Rules

1. A document is added here in the same change that adds it to `information/`.
2. Hand-written documents carry a one-line status banner directly under their title;
   `tests/test_p2_docs.py` fails if one is missing or if a document is not indexed here.
3. Generated documents are never edited by hand. `uv run dqb docs --check` fails when
   the checked-in guide no longer matches the registries.
4. A historical document is never rewritten to agree with the current state. If its
   claim is overturned, it moves to **Superseded** with the replacement named.
