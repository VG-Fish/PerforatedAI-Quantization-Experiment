# Assessment: what is known before the next optimization loop

**Date:** 2026-08-31  
**Scope:** analysis only; no model, dataset, or code changes are authorized.

## Executive assessment

The repository is not starting from zero. It has 24 registered models, a complete six-level PTQ grid (Q8/Q4/Q2/Q1.58/Q1 plus FP32), PQAT staging, model and artifact metadata, and 537 historical records. It also has a well-founded reason not to interpret most of those records as scientific evidence yet:

- `information/DENDRITE_EFFECT_AUDIT_2026-08-30.md` found no replicated positive dendrite result. PointNet cleared a *within-run* noise floor once; LeNet-5 was marginal; the remaining usable runs were at/below their noise floors. No quantified dendritic arm in the audited combined run was reportable.
- `information/EVIDENCE_INDEX.md` says all 537 historical records have manifest status `unknown`. They retain diagnostic value, but are not valid experimental evidence under the repository's current manifest rules.
- The historical bugs were not cosmetic: stale PAI folders could merge run state; the best-epoch restore could combine incompatible dendritic topology with older backbone weights; early low-bit quantizers lacked correct scaling; and fixed PAI intervals did not fire when configured. Consult `information/MEASUREMENT_CAVEATS.md` before treating any earlier score as a comparison target.

The correct next move is therefore **not** to broaden to 25 models or to tune quantization first. It is to make five base/perforated pairs reproducible, well-converged, and auditable, then apply the same quantization protocol. This will provide both a credible first answer and an execution template for later models.

## Local hardware profile

Measured using `system_profiler`:

| resource | observed value | planning consequence |
|---|---:|---|
| machine | MacBook Pro, Mac15,7 | Apple Silicon/MPS-first development |
| SoC | Apple M3 Pro | use MPS for training; validate final numerical paths on CPU when needed |
| CPU | 12 cores (6 performance, 6 efficiency) | avoid more than 2–3 simultaneous training workers |
| GPU | 18-core Apple GPU, Metal supported | suitable for compact CNNs, SAINT, MPNN, PointNet; transformer batch sizes remain small |
| unified memory | 36 GB LPDDR5 | enough for one medium model; do not run multiple MPS-heavy arms concurrently |

Recommended scheduler policy: one MPS training job at a time, with at most one CPU-side evaluation/calibration job only after it is verified not to contend. Use `-j 1` for dendritic work. Benchmark latency separately at batch 1 and a throughput batch, on the actual intended backend; MPS training support does not make every integer inference operator a meaningful MPS deployment benchmark.

## What the installed PAI guidance and upstream source require

The local `PAI Skills/` were reviewed. The relevant operational requirements are consistent with the current upstream repository:

- Wrap a constructed model with `UPA.perforate_model`, register/rebuild the optimizer after restructuring, report validation metrics through `GPA.pai_tracker.add_validation_score`, and continue until its completion flag is true. The official minimal loop is in PerforatedAI's [README](https://github.com/PerforatedAI/PerforatedAI).
- Module IDs must use leading-dot notation. Perforation targets must cover every parameter either as perforated or tracked; otherwise PAI may enter a debugger during candidate training.
- A retained dendrite is an architectural change, not merely a metric event. Record candidate-insertion and retain/reject switches, final parameter count, topology, and PAI configuration with the source artifact.
- PAI's current public library is not the full PB algorithm: its own README states that the commercial `perforatedbp` component delivers the headline PB results. The experiment must record the exact installed package versions, licence state, and whether PB is actually active. Do not generalize a public-library result to a full-PB claim.

The local `perforatedai-analyze` skill's PB-score heuristics should be treated as diagnostics only. The repository's own audit found no useful rank relation between peak PB correlation and validation gain; a high correlation did not predict generalization.

## Existing model roster and the selection decision

`src/dendritic_benchmark/specs.py` currently lists 24 models. They already cover vision, audio, text, graphs, tabular data, time series, anomaly detection, RL, point clouds, generative models, and spiking networks. Adding models before the five-pair protocol works would multiply invalid artifacts, not information.

The launch cohort prioritizes diverse deployment domains, established online benchmarks, model families that PAI can target, and tractability on 36 GB of unified memory:

| model | why it belongs now | known risk / required guard |
|---|---|---|
| `resnet18_cifar10` | widely recognized CNN control; code already mirrors PerforatedAI's pre-FC targeting in both arms | the published HF perforated transfer is not a causal paired control; train local base and perforated versions from matched initialization instead |
| `m5` / SpeechCommands | direct on-device keyword-spotting case; low-cost convolutional quantization target | historical M5 dendritic low-bit artifacts are invalid; rerun from clean namespace |
| `distilbert` / SST-2 | commercially common pretrained text encoder; established fine-tuning recipe | MPS head-only dendrites test only `.pre_classifier`/`.classifier`, not the entire Transformer; state this in every report |
| `saint_adult` | attention-based tabular architecture, a category with broad risk/operations applications | Adult is a standard proxy, not a fairness claim; retain sensitive-feature and subgroup reporting as future work |
| `mpnn` / ESOL | molecular regression gives graph and drug-discovery coverage | prior candidate phases hit an 8-epoch cap while PB correlation was still rising; do not accept a run unless its candidate/retain phase actually completes |

### Deliberately deferred models

- **PointNet/ModelNet40:** first expansion candidate because one HISTORY-mode run cleared the internal noise floor. It needs three seeds and a capacity-matched control before it is evidence. At roughly 36 seconds/epoch, it would slow protocol debugging.
- **LeNet-5:** retain as a fast integration and quantizer smoke test only. It is not sufficient commercial coverage.
- **TCN/GRU/VAE:** defer until the loop has reproduced a valid topology-aware best checkpoint; their historical fixed-switch and structure-restoration outcomes are unsuitable as decision evidence.
- **HF perforated ResNet:** useful as an external topology/reference check, but not a substitute for matched local base/perforated training because its ImageNet pretraining, retained topology, and CIFAR adaptation confound the comparison.

## Validity protocol (must precede any optimization claim)

1. **Freeze the experimental unit.** For each `(model, dataset, seed)` make one immutable split manifest before tuning. Stratify classification; use scaffold/group splits for molecular data where feasible; never choose a recipe on test results. Record package lock, device, seed, dataset version/checksum, transforms, and input shape.
2. **Optimize the base model honestly.** Use validation only. A successive-halving screen may use one seed and a reduced budget; the finalist must be re-run at full budget on three seeds. The base quality gate is a predeclared tolerance to a cited/common reference or documented strong baseline, not an arbitrary absolute score.
3. **Optimize the perforated model without moving the goalposts.** Start from the selected base recipe and the same seed/split. Tune the PAI target set, plateau/history settings, maximum dendrites, candidate-init scale, and dendrite-only LR floor on validation only. Keep the backbone schedule identical to the base arm. Train until PAI says complete; do **not** replace this with a hard epoch cap. If it stalls, resume/diagnose and label it inconclusive rather than calling it complete.
4. **Separate three causes of an apparent gain.** For every final perforated configuration add a matched dense continuation control (same extra optimizer steps) and a capacity-matched dense control (same approximate parameter count).
5. **Quantize only finalized FP32 source artifacts.** The current implementation is a custom *weight-only* projection: it quantizes every parameter on CPU and has no activation observers or calibration set. Run this implemented PTQ/PQAT protocol at Q8, Q4, Q2, Q1.58, and Q1, retaining stage snapshots and measuring both absolute quality and normalized retention. If activation quantization/calibration is added later, version the experiment and record a fixed train-only calibration subset; it is a different benchmark, not a replacement for existing weight-only data.

For a maximize metric, `retention_b = q_metric_b / fp32_metric`; for a minimize metric, `retention_b = fp32_metric / q_metric_b`. Report the paired dendritic-minus-base difference in retention, confidence intervals across paired seeds, parameter count, serialized size, latency, calibration count, and PQAT gain over PTQ. Three seeds establish minimal reportability, not a stable SOTA ranking.
