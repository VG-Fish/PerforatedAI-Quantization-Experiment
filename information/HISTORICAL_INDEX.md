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
| [DENDRITE_EFFECT_AUDIT_2026-08-30.md](DENDRITE_EFFECT_AUDIT_2026-08-30.md) | current | the standing verdict on whether the dendrite effect beats noise and more training |
| [MEASUREMENT_CAVEATS.md](MEASUREMENT_CAVEATS.md) | current (dated line numbers) | the eleven measurement caveats, their root causes, and which results each one invalidates |
| [audit/audit_report.md](audit/audit_report.md) | current | the cleanup priority ledger, what P0–P2 built, what is still open, and the dead-code ledger |
| [CAPACITY_MATCHED_DENSE_CONTROLS.md](CAPACITY_MATCHED_DENSE_CONTROLS.md) | current (proposal) | the topology-matched dense-control design — not yet implemented in the runner |

## Optimization plan (2026-08-31)

`information/optimization/` is the working memory for the next selection/optimization
loop. Read it in the order its README gives; it is a plan, not a record of results.

| document | status | cite it for |
|---|---|---|
| [optimization/README.md](optimization/README.md) | current | the launch cohort, the reading order, and the reportability rule |
| [optimization/00_assessment.md](optimization/00_assessment.md) | current | machine profile, model-selection reasoning, and the five-step validity protocol |
| [optimization/01_initial_five_plan.md](optimization/01_initial_five_plan.md) | current | per-model starting recipes, tuning spaces, gates, and the advancement checklist |
| [optimization/02_research_and_sources.md](optimization/02_research_and_sources.md) | current | external sources consulted and the explicit non-conclusions |
| [optimization/03_execution_matrix.md](optimization/03_execution_matrix.md) | current | the exact sweep trials, current-code limits, and required manifest fields |
| [optimization/04_implementation_review.md](optimization/04_implementation_review.md) | current | whether the runner actually implements 03's requirements, the defects fixed, and what is still open before a sweep |

## Historical

| document | status | cite it for | do not cite it for |
|---|---|---|---|
| [DOCUMENTATION.md](DOCUMENTATION.md) | historical | experiment rationale, PerforatedAI integration walkthrough, quantization background, the baseline-quality record | the roster or the condition grid |
| [MODEL_REFERENCE.md](MODEL_REFERENCE.md) | historical | per-model hyperparameters, preprocessing, and PAI targeting narrative | metrics, metric directions, or which models are in the default roster |
| [CLI_DIAGRAMS.md](CLI_DIAGRAMS.md) | historical | command flowcharts and the generated-output directory layout | flags, defaults, or the list of subcommands |
| [REMAINING_FIXES.md](REMAINING_FIXES.md) | historical (2026-08-07) | why the baseline-quality pass changed what it changed | the outstanding-work list; the audit ledger owns that now |
| [CODE_REVIEW_2026-08-28.md](CODE_REVIEW_2026-08-28.md) | historical (2026-08-28) | the bugs found and fixed in that pass | current open findings |
| [DYNAMIC8_RUN_2026-08-28.md](DYNAMIC8_RUN_2026-08-28.md) | historical run report | what dynamic8 ran and observed | reportable results — it predates the artifact manifest |
| [DYNAMIC9_RUN_2026-08-28.md](DYNAMIC9_RUN_2026-08-28.md) | historical run report | what dynamic9 ran and observed | reportable results — it predates the artifact manifest |
| [DYNAMIC_DENDRITIC_MIGRATION.md](DYNAMIC_DENDRITIC_MIGRATION.md) | historical | the reasoning behind moving from fixed-interval to HISTORY scheduling, and the PerforatedAI 3.2.3 `save_name` constraint | schedule values — `compat.py::PAI_DYNAMIC_SCHEDULE_DEFAULTS` is the live one |

## Superseded

| document | superseded by | what specifically was overturned |
|---|---|---|
| [DYNAMIC9_PAI_GRAPH_AUDIT.md](DYNAMIC9_PAI_GRAPH_AUDIT.md) | [DENDRITE_EFFECT_AUDIT_2026-08-30.md](DENDRITE_EFFECT_AUDIT_2026-08-30.md) | "perforation is working correctly on every model far enough along to judge". Dendrites *are* inserted and retained — that part stands — but the audit never tested the score change against run noise. |
| [MODEL_SELECTION.md](MODEL_SELECTION.md) | [CURRENT_GUIDE.md](CURRENT_GUIDE.md) | criterion 1's exclusion of `pointnet_modelnet40` and `resnet18_cifar10`, which rested on an expired PerforatedAI token that has since been renewed. Its cost table and diversity reasoning still hold. |

## Rules

1. A document is added here in the same change that adds it to `information/`.
2. Hand-written documents carry a one-line status banner directly under their title;
   `tests/test_p2_docs.py` fails if one is missing or if a document is not indexed here.
3. Generated documents are never edited by hand. `uv run dqb docs --check` fails when
   the checked-in guide no longer matches the registries.
4. A historical document is never rewritten to agree with the current state. If its
   claim is overturned, it moves to **Superseded** with the replacement named.
