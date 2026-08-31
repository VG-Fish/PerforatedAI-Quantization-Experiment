# Initial five-model optimization plan

## Decision

Run the five models in this order, beginning with a single fully audited `resnet18_cifar10` seed. Do not activate the next model until the preceding model has a valid base source, a verified retained/perforated source (or a documented negative result), and a dry-run of all PTQ/PQAT descendants.

| order | model / dataset | deployment domain | initial objective |
|---:|---|---|---|
| 1 | ResNet-18 / CIFAR-10 | edge visual inspection, retail, manufacturing | establish a clean CNN + PAI + quantization protocol |
| 2 | M5 / SpeechCommands v0.02 | wake-word and command recognition | test 1D convolution and small-footprint audio |
| 3 | DistilBERT / GLUE SST-2 | support routing, sentiment and document triage | test a pretrained NLP encoder with head-only dendrites |
| 4 | SAINT / Adult Income | structured-data scoring and operations | test tabular attention with a compact, late classifier target |
| 5 | MPNN / MoleculeNet ESOL | molecular-property prediction | test graph regression and recurrent/gated message passing |

The dataset choices are deliberately familiar and reproducible rather than application-specific. Do not replace them now: a benchmark change at the same time as protocol repair would prevent attribution. In a later application phase, replace Adult with a documented tabular benchmark such as CoverType or HIGGS and ESOL with a scaffold-split industrially relevant molecular task; do so as a *new, versioned task*, never by silently replacing data under an existing key.

## Common two-stage tuning design

### Stage A — base-model selection

1. Hold split, preprocessing, metric, and test set fixed.
2. Screen the bounded prior-informed tuning space below on validation only with one seed and approximately 25% of the full budget.
3. Promote the two best stable configurations to full budget. Choose a single recipe before accessing test metrics.
4. Re-run the winner for seeds `0, 1, 2`; save FP32 checkpoint, curve, parameter count, and score distribution.

The aim is a strong, stable baseline—not the highest lucky validation peak. When two variants overlap within validation noise, choose the smaller/faster one and document the tie rule.

### Stage B — perforated-model selection

1. Reuse the selected base recipe and exact seed/split. The base model must include any structural layer present in both arms (for example, ResNet's `pre_fc`).
2. Begin with the current conservative late-target configuration. Sweep only PAI choices below, using validation-only selection.
3. A candidate is valid only if PAI produces insertion evidence, ends with topology/parameter growth, and returns its completion flag.
4. Re-run the selected perforated recipe for the same three seeds, then run dense-continuation and capacity-matched controls.
5. Freeze FP32 sources; only then create PTQ and PQAT descendants.

### PAI tuning grid

Use HISTORY/plateau mode first. Fixed switch intervals are a diagnostic ablation only: current evidence shows their observed switch epochs did not match configuration. Start with one dendrite; expand to two only after the one-dendrite arm passes matched controls.

| knob | values to screen | rule |
|---|---|---|
| PAI targets | conservative late target; one alternate late target where listed | never default-wrap all eligible modules as the primary comparison |
| `max_dendrites` | 1; then 2 if justified | capacity controls must match the final topology |
| plateau/history window | current default; +50% window | add only after dense validation has flattened |
| `p_epochs_to_switch` | current default; 6; 8 | this is a PAI setting, but `training.py::MAX_DENDRITE_PHASE_EPOCHS=8` is the real hard cap; values above 8 do not lengthen a candidate phase |
| dendrite LR floor | 0.0, 0.05, 0.10 times base LR | dendrite parameter group only; backbone trajectory stays matched |
| candidate init multiplier | 0.01, 0.03, 0.10 | select for stable validation and retained topology, never PB score alone |

## Per-model starting recipes and bounded tuning spaces

These are starting hypotheses, not finalized hyperparameters. Current repository recipes are the first point in each space because they have already been selected from relevant published/common recipes and prior diagnosis.

### 1. ResNet-18 / CIFAR-10

**Why first:** the repo implements an apples-to-apples `pre_fc` layer in the base model and conservative single-target PAI setup; PerforatedAI's upstream ResNet example uses this pre-FC design.

| item | start | validation-only screen |
|---|---|---|
| base | SGD/Nesterov, 200 epochs, batch 128, LR 0.1, WD 5e-4, cosine, five-epoch warmup, label smoothing 0.1 | LR `{0.05, 0.1, 0.2}` × WD `{2e-4, 5e-4}`; smoothing `{0, 0.1}` among survivors |
| PAI target | `.pre_fc` only | retain `.pre_fc` as the primary/only target; no residual-block wrapping in the first study |
| PAI schedule | HISTORY, one dendrite, 10 candidate-phase epochs | history-window +50%; dendrite LR floor `{0, .05, .10}` |
| gate | stable CIFAR-10 curve and a documented strong accuracy tolerance before PAI | if it misses materially, repair base training before any dendrite trial |

Do not use `resnet18_hf_perforated_cifar10` as the paired arm. It is an external topology/reference checkpoint, but differs in pretraining and topology. The causal pair begins from the same local architecture/initialization policy.

### 2. M5 / SpeechCommands

**Why second:** keyword spotting is a direct compact deployment case and gives a temporal-convolution stressor at manageable cost.

| item | start | validation-only screen |
|---|---|---|
| base | Adam, 40 epochs, batch 128, LR 1e-2, WD 1e-4, StepLR at 20 × 0.1 | LR `{3e-3, 1e-2}` × WD `{1e-5, 1e-4}`; preserve labels/split |
| PAI target | last two Conv1d layers plus classifier; earlier layers tracked | `{conv3+conv4+fc, conv4+fc, fc only}` after structural inspection |
| PAI schedule | HISTORY, one dendrite | history-window +50%; init multiplier `{.01, .03, .1}` |
| gate | remove historical late-epoch oscillation and establish stable held-out accuracy | repeat low-bit quantizer smoke test; old M5 dendritic low-bit results are invalid |

### 3. DistilBERT / SST-2

**Why third:** contemporary NLP representative; the current test split avoids phrase-level leakage from SST-2 train data.

| item | start | validation-only screen |
|---|---|---|
| base | AdamW, 3 epochs, batch 32, LR 2e-5, WD .01, linear decay, clip 1.0 | LR `{1e-5, 2e-5, 3e-5}`; base batch 16/32 only if MPS changes effective behavior |
| PAI target | `.model.pre_classifier` and `.model.classifier` | head pair versus `classifier` only; encoder remains tracked, not perforated |
| PAI runtime | dendritic batch 4, correlation cap 4 batches, cleanup every 128 batches | no full-encoder scope on this machine without separately approved memory study |
| gate | credible SST-2 fine-tuning range without selection on test | label result **head-only perforation**; it cannot establish a full-Transformer claim |

The current generic quantizer and one-epoch PQAT budget are a feasibility result, not a deployment claim, until export and backend-compatible operator coverage are verified. It is a weight-only custom projection, not observer-calibrated activation PTQ.

### 4. SAINT / Adult Income

**Why fourth:** tests tabular attention plus a small late classifier whose capacity growth is controllable. Adult is a reproducible proxy only; do not make lending, employment, or fairness claims from it.

| item | start | validation-only screen |
|---|---|---|
| base | AdamW, 200 epochs, batch 256, LR 1e-4, WD 1e-5, cosine floor .02, warmup 5, clip 1.0 | LR `{1e-4, 3e-4, 6e-4}` × WD `{1e-6, 1e-5}`; dropout only after LR is fixed |
| PAI target | complete `.head` classifier (+4,418 parameters/dendrite) | `.head` only; row/column QKV remains tracked |
| PAI schedule | HISTORY (not fixed-100), one dendrite | history `{default, +50%}` and floor `{0, .05, .1}` |
| gate | reproducible validation/test behavior | PB correlation = 0 or no retained topology is inconclusive |

### 5. MPNN / ESOL

**Why fifth:** molecule property prediction is commercial graph coverage, but comes last because the historical PAI candidate phase was cut off before convergence. It is a protocol stress test, not a place to chase early best scores.

| item | start | validation-only screen |
|---|---|---|
| base | Adam, 200 epochs, batch 32, LR 1e-3, WD 1e-5, cosine floor .02, clip 5.0 | hidden width `{64, 96, 128}` × head dropout `{.1, .2}`; retain train-only target standardization |
| split | current reproducible split for first clean comparison | separate scaffold-split replication before transferability claims |
| PAI target | late gated readout and regression head first | readout/head-only versus existing default after exact eligibility check |
| PAI schedule | HISTORY, one dendrite, candidate phases `{8, 12}` | invalid if a safeguard terminates a candidate while correlation/validation still improves |
| gate | reasonable ESOL RMSE without severe train/validation divergence | capacity control mandatory because growth can be large relative to data |

## Quantization experiment after FP32 gates

For finalized base and perforated source topology, run:

`FP32 → PTQ(Q8, Q4, Q2, Q1.58, Q1) → PQAT(Q8, Q4, Q2, Q1.58, Q1)`.

Use identical bit definition, granularity, clipping rule, evaluator revision, and fixed test data within a pair. The current system is weight-only and has no calibration-example knob; if that changes later, version it and fix a train-only calibration subset. `Q2` must use the current four-level robust-scale kernel; `Q1`/`Q1.58` must retain scale metadata. Historical results lacking these properties are invalid.

1. Verify base and perforated FP32 source artifacts independently.
2. Snapshot PTQ evaluation **before** PQAT and snapshot PQAT afterward; never overwrite PTQ.
3. Confirm only one intended projection/quantizer is applied.
4. Record serialized size and correct-backend latency in addition to metric. Fake quantization is not deployment latency.
5. Report PTQ loss and PQAT recovery separately. Dendritic benefit is paired normalized-retention difference, not merely a better absolute post-quantization score.

## Advancement checklist

- [ ] Inspect current working tree and `information/CURRENT_GUIDE.md`; this plan is not permission to edit registry.
- [ ] Create a new isolated run namespace; never reuse `results/PAI/<name>`.
- [ ] Perform one ResNet base/dendritic dry run through restructuring, checkpointing, manifest finalization, PTQ and PQAT metadata.
- [ ] Validate all source artifacts with hash verification.
- [ ] Execute full ResNet three-seed FP32 and controls before quantization.
- [ ] Run all bit widths with PTQ and PQAT only after FP32 sources pass.
- [ ] Write an evidence report; advance to M5 only when conclusion is reportable or explicitly negative/inconclusive.
- [ ] After all five, decide whether to register `unet_isic` as #25. It needs a working protocol and baseline performance/cost screen, not merely the empty slot.
