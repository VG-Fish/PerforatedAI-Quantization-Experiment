# Optimization handoff — 2026-08-31

## Purpose and boundary

This directory is the working memory for the next selection/optimization loop.
The user has authorized **analysis and a proposed plan only** at this stage:

- Do not change the model registry, datasets, training recipes, quantizers, or run scripts yet.
- Do not launch training.
- The eventual workflow is: independently optimize a base model and a dendritic/perforated version, train the dendritic arm until PerforatedAI returns `training_complete` / `is_complete`, then run PTQ and PQAT and compare their quantization behaviour.
- Dendrites are not a preliminary standalone step. The base and perforated versions are the two objects being optimized and measured.

Read these files in order:

1. [00_assessment.md](00_assessment.md) — current state, evidence quality, machine constraints, and non-negotiable validity rules.
2. [01_initial_five_plan.md](01_initial_five_plan.md) — recommended first five models, proposed tuning spaces, gates, and exact next actions.
3. [02_research_and_sources.md](02_research_and_sources.md) — external research, PerforatedAI ecosystem review, and sources consulted.
4. [03_execution_matrix.md](03_execution_matrix.md) — exact per-model hyperparameter trials, current-code limits, and decision rules.
5. [04_implementation_review.md](04_implementation_review.md) — whether the runner actually implements what 03 requires, the defects found and fixed, and the gaps still open before a sweep. Read this before trusting any statement in 03 about what the current code does.

## Current decision

Keep the current 24-model registry unchanged for now. Use the five-model launch cohort below, prove the protocol on it, and make the **25th registry slot** a medical-segmentation U-Net only after the protocol produces reportable results. `TinyUNet`/ISIC support already exists in code and data plumbing, but is intentionally not registered today.

| order | model key | domain | role in this loop |
|---|---|---|---|
| 1 | `resnet18_cifar10` | edge vision | protocol anchor; upstream has a directly relevant ResNet pre-FC design |
| 2 | `m5` | keyword spotting/audio | compact 1D-convolution deployment case |
| 3 | `distilbert` | NLP | modern pretrained-transformer case; head-only dendrite scope is explicit |
| 4 | `saint_adult` | tabular risk/operations | attention-based structured-data case |
| 5 | `mpnn` | molecular property prediction | graph regression / drug-discovery case |

`pointnet_modelnet40` is the first expansion/replication candidate rather than part of the launch five: it is the only current run with a within-run dendrite gain clearing its estimated noise floor, but has only one seed and has a materially higher cost. `lenet5` remains a smoke-test/control model, not a commercial-coverage model.

## The decisive rule

No result becomes a dendritic quantization claim merely because a file exists or because a single run improved. A reportable claim requires:

1. Three paired seeds and a fixed, held-out test set.
2. A valid retained-dendrite audit for every perforated source artifact.
3. A matched dense continuation control for extra training time.
4. A capacity-matched dense control after the retained dendrite size is known.
5. The same FP32 source topology for PTQ and PQAT, plus recorded bit-width, granularity, evaluator revision, and before/after-PQAT snapshots. The current quantizer is weight-only and has no calibration-data input; any future activation-calibrated method needs a new versioned protocol.

The existing artifact-manifest/reportability system is a useful starting point, but historical records are `unknown` because they predate manifests; do not reuse them as evidence.
