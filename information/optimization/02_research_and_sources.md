# Research notes and sources

## What was reviewed

### Local project

- `PAI Skills/skills/`: comparison, dashboard, integration, analysis, distributed, training, and visualization guidance. The integration guidance requires PAI's open-ended completion loop and optimizer reinitialization after structural change; its analysis guidance is diagnostic and must not override this repo's noise/control evidence.
- `information/CURRENT_GUIDE.md`, `EVIDENCE_INDEX.md`, `DENDRITE_EFFECT_AUDIT_2026-08-30.md`, `MEASUREMENT_CAVEATS.md`, `MODEL_REFERENCE.md`, Dynamic12 documentation, model specs, pipeline recipes, and results trees.
- Machine profile: M3 Pro, 12 CPU cores, 18 GPU cores, 36 GB unified memory.

### PerforatedAI ecosystem

- [PerforatedAI public library](https://github.com/PerforatedAI/PerforatedAI) was cloned into the read-only research sandbox at commit `0a5967b` and its README, API structure, examples, and agent guidance were reviewed. It offers a minimal PyTorch integration loop, an ImageNet ResNet-18 pre-FC example, PyTorch Lightning/Hugging Face examples, TD3, and Edge Impulse keyword spotting. It distinguishes the open dendritic library from the separately licensed Perforated Backpropagation component.
- [PerforatedAI-Examples](https://github.com/PerforatedAI/PerforatedAI-Examples) was cloned and reviewed. It contains SOTA examples (HIST, TrimNet, mTAN), library integrations, and examples spanning U-Net segmentation, DenseNet, MobileNet, ResNet, and vision transformers. Its stated workflow reinforces architecture-specific validation and per-architecture best-score tracking.
- [PerforatedAI's organization](https://github.com/PerforatedAI) currently lists 10 public repositories. The review included the main library, examples, Transformers forks, tabular fork, PyG fork, Studio installer, community projects, and LPCV efficient-video project. The latter supplies a useful deployment pattern—baseline versus perforated throughput, parameter count, and accuracy—but is too compute/data intensive for an M3 Pro launch cohort. The tabular/PyG forks reinforce that those domains are in scope; they do not by themselves validate dendritic gains.
- [Perforated's public site](https://www.perforatedai.com/) frames the product as a data-efficient PyTorch workflow intended for existing benchmarks and production deployment. Its [Thoro semantic-segmentation paper](https://www.perforatedai.com/Perforated_Thoro_Paper.pdf) describes alternating neuron and dendrite phases and validates medical segmentation as a sensible future #25 domain, but its task/compute profile argues against introducing it before this protocol is stable.

## Research conclusions that affect the experiment

### PTQ and PQAT/QAT are separate measurements

PyTorch defines QAT as fake quantization in the training/fine-tuning forward pass followed by conversion; it is intended to recover degradation caused by PTQ. See [PyTorch's current QAT workflow](https://docs.pytorch.org/ao/stable/workflows/qat.html) and [PT2E QAT guide](https://docs.pytorch.org/ao/stable/pt2e_quantization/pt2e_quant_qat.html). Therefore the benchmark must preserve a pre-PQAT PTQ snapshot and cannot call a PQAT result "PTQ." Use the exact same quantizer settings at prepare and conversion; otherwise PQAT recovery is not attributable.

### Calibration and granularity must be fixed, not opportunistic

Quantization parameters are estimated from observed ranges in standard observer-based PTQ. Calibration data would therefore belong in a future activation-quantization manifest and must come from training data only. PyTorch's guidance also identifies per-channel weight quantization as a standard default for weights; see [Quantization in Practice](https://pytorch.org/blog/quantization-in-practice/). **The current repository does not implement that workflow:** `quantization.py` applies a custom parameter-only projection with no calibration examples or activation observers. The current Q2 robust-scale correction and binary/ternary scale correction are version boundaries; old results cannot be pooled with new results.

### Low bit-width is the meaningful stress regime

The expected ordering is not a hypothesis to "prove": Q8 is a sanity-check tier, Q4 provides an important deployment trade-off, and Q2/Q1.58/Q1 are severe stress tests where QAT/PQAT is more likely to recover loss. The trial must nevertheless measure every tier, because model families can fail differently. PyTorch explains fake quantization and its recovery role in [QAT for LLMs](https://docs.pytorch.org/blog/quantization-aware-training/); this supports the separation of PTQ loss from PQAT recovery, not a promised outcome for these models.

### Model selection is domain coverage, not a leaderboard claim

| model | common benchmark | commercial analogue | caveat |
|---|---|---|---|
| ResNet-18 | CIFAR-10 | edge image classification | protocol benchmark, not production vision |
| M5 | SpeechCommands | keyword spotting | 12-label task is simpler than open-world speech |
| DistilBERT | SST-2 | text routing/sentiment | head-only PAI scope on this machine |
| SAINT | Adult | structured scoring/operations | not a fairness/deployment dataset |
| MPNN | ESOL | molecular screening | random split does not establish scaffold generalization |

The next expansion model is PointNet/ModelNet40 (digital twins, robotic perception, industrial 3D). The proposed #25 is U-Net/ISIC skin-lesion segmentation because the repo already has model/data support and upstream work demonstrates segmentation relevance; it remains deferred until the protocol yields reportable artifacts.

## Explicit non-conclusions

- No public PerforatedAI claim, example graph, or single historical benchmark run is evidence that this repository's dendritic models will quantize better.
- No PB correlation threshold establishes a causal or generalization benefit.
- The current five are a launch cohort, not the final portfolio and not an assertion that they are the five globally best architectures.
- No current run supports claims about SOTA accuracy, 90% compression, or quantization robustness for this project.

## Re-entry procedure

When a future agent resumes:

1. Read `README.md`, `00_assessment.md`, and `01_initial_five_plan.md` here.
2. Verify `information/CURRENT_GUIDE.md` and the git working tree; code may have changed since this plan.
3. Read the latest `DENDRITE_EFFECT_AUDIT` and `MEASUREMENT_CAVEATS` before interpreting historical output.
4. Confirm installed PerforatedAI and `perforatedbp` versions/licence against the run manifest before launching a perforated arm.
5. Start only the first unchecked advancement item; record a dated decision note here when deviating from this plan.
