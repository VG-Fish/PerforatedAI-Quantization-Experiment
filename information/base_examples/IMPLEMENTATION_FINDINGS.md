# PerforatedAI Base-Example Integration Findings

Status: discovery and design only; no benchmark model or training code has been changed.

## Requested scope

The next benchmark roster should be based directly on five PerforatedAI examples:

- `examples/transfer_learning`: CIFAR-100 only.
- `examples/base_examples/mnist`.
- `examples/base_examples/pytorch_unet`.
- `examples/base_examples/segmentation-image-resolution`: the main, full-resolution setup only.
- `examples/base_examples/resnet`: the knowledge-distillation plus perforation setup.

The experiment must add quantization without silently replacing the upstream architectures,
datasets, preprocessing, objectives, or PerforatedAI behavior that made these examples useful.

## Source pin

Research is based on PerforatedAI/PerforatedAI commit
`0a5967b4574d4b280b31d6ef30beffcd4e4308ea` (upstream `main`, committed
2026-08-23, "updated imagenet example"). The exact revision should be recorded in the eventual
experiment metadata so later upstream changes cannot alter the meaning of a comparison.

## Early repository findings

- This benchmark is registry-driven and currently describes itself as a 24-model, 12-condition
  experiment. Adding the five requested upstream examples will affect more than model classes:
  model specs, dataset construction, training-policy branches, artifact metadata, generated docs,
  CLI validation/completion, and offline matrix tests all need to remain synchronized.
- The benchmark already distinguishes FP32 source training, post-training quantization (PTQ), and
  optional PerforatedAI quantization-aware fine-tuning (PQAT). The integration should reuse that
  condition/artifact machinery while allowing example-specific trainers where the upstream loss or
  lifecycle cannot be expressed faithfully by the generic classification loop.
- The requested models span classification and semantic segmentation, and the ResNet request adds
  a third experimental factor (knowledge distillation). A single accuracy-only adapter is therefore
  not sufficient; task-aware metrics, losses, batches, checkpoint state, and inference inputs must
  be explicit.
- Quantization must be applied after the upstream model topology is assembled, including any
  retained dendrites. Dense, perforated, and perforated-plus-KD source checkpoints must not be
  conflated.

## Questions being resolved from source

1. Which exact architecture and pretrained checkpoint the CIFAR-100 transfer-learning example uses.
2. Which MNIST topology, optimizer, schedule, and PerforatedAI settings produced the published gain.
3. Whether the two segmentation examples need separate datasets and metric contracts, and which
   full-resolution configuration is the intended `segmentation-image-resolution` target.
4. How the ResNet teacher, student, distillation loss, perforation phases, and evaluation interact.
5. Which portions fit the current generic runner and which require a task/trainer extension point.

Further verified findings and the recommended implementation sequence will be appended below as the
source audit progresses.
