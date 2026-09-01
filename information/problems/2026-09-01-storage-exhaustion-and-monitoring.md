# Storage exhaustion during the five-model launch

## Summary

The initial five-model, two-worker run stopped on 2026-08-31 when the macOS data volume reached 100% capacity. At failure, the run had completed five conditions. ResNet-18 stopped during `base_fp32` after epoch 182/200; DistilBERT completed `base_q1_58` and failed while saving the `base_q1` pre-PQAT snapshot.

## Evidence

- Available space at the time of failure: 479 MiB; filesystem capacity: 100%.
- Result directory size at that point: 8.5 GiB.
- The DistilBERT worker raised `RuntimeError: ios_base::clear: unspecified iostream_category error`, followed by PyTorch's `inline_container.cc` unexpected-position error while serializing a snapshot.
- Completed DistilBERT PQAT conditions retain a 1.0 GiB epoch checkpoint plus one or more 255 MiB model snapshots. This checkpoint pattern was the dominant storage consumer.

## Diagnosis

This was a storage-write failure, not an API-key, PAI-feature, model-training, or data-loading failure. PyTorch checkpoint serialization failed after the disk became full. The ResNet worker's incomplete traceback is consistent with the same shared storage exhaustion.

## Recovery already performed

The user freed storage, then the experiment was restarted with the same result directory and seed. Existing `record.json` artifacts are skipped and the ResNet epoch checkpoint resumes from epoch 182. The new stream logs are in `logs/dendritic_quantization_launch5_seed0_20260831_17502/`; results remain in the shared original result directory.

## Monitoring and stop policy

`scripts/monitor_experiment.zsh` is used as a persistent local monitor for this run. It emits a progress report and desktop notification every 30 minutes. It scans appended stream output once per minute for fatal Python tracebacks, explicit disk-full errors, PyTorch serialization errors, segmentation faults, and fatal interpreter errors.

On a detected fatal error it writes the new stream context to this file, sends a desktop alert, and stops only workers whose command line matches this run's `--results-directory`. It deliberately does not stop a worker for non-fatal warnings, slow batches, or ordinary validation regressions.

## Next actions if storage becomes constrained again

1. Preserve completed `record.json` files and their final `model.pt` artifacts.
2. Remove only recoverable `epoch_checkpoint.pt` files for conditions that already have `record.json`, after reviewing the exact targets.
3. Resume the run without `--fresh` so incomplete conditions continue from their checkpoints.
