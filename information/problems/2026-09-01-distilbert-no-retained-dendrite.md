# DistilBERT dendritic run produced no retained topology

## Incident

At 06:16 on 2026-09-01, the persistent monitor stopped the two active workers after DistilBERT attempted to start `dendrites_q8`. The pipeline correctly rejected that condition because the prerequisite `dendrites_fp32` artifact was not a verified dendritic topology.

## Evidence

- `distilbert/dendrites_fp32/record.json` records `dendrite_audit_status: "no_retained_insertion"` and the reason: `raw PAI switch log has no candidate-insertion switch`.
- The final dendritic parameter count is **66,955,010**, exactly equal to the completed dense FP32 reference. No additional topology survived training.
- The raw PAI configuration requested history mode with `n_epochs_to_switch: 10`, `p_epochs_to_switch: 2`, `history_lookback: 8`, and `initial_history_after_switches: 8`.
- DistilBERT's training recipe has only **3 epochs**. Its log reports `Returning False - no triggers to switch have been hit` after epochs 1 and 2, then freezes live dendrite updates before epoch 3.

## Root cause

The selected PAI history schedule cannot reach an insertion event within a three-epoch DistilBERT run: its N phase alone is configured for ten epochs. Consequently, this condition measures ordinary dense training under PAI wrapping, not a topology-modified network. Its reported test accuracy (0.9083) is **not** a valid dendritic result.

This is an experiment-configuration incompatibility, not an API-key, dashboard, data-loading, or storage failure. The dashboard connection warning was non-fatal; PAI initialized successfully but its switch trigger never fired.

## Why stopping was correct

The dendritic PQAT conditions inherit their topology from `dendrites_fp32`. Continuing to `dendrites_q8`, `dendrites_q4`, `dendrites_q2`, `dendrites_q1_58`, or `dendrites_q1` without a verified retained insertion would create invalid comparisons. The guard in `BenchmarkRunner._require_verified_dendritic_pqat_source` rejected the first descendant and the monitor stopped the companion ResNet worker as requested for a serious integrity failure.

## Scope

- The five completed dense DistilBERT conditions remain usable as dense baselines.
- The completed `distilbert/dendrites_fp32` artifact must be treated as non-reportable for dendritic claims.
- ResNet-18 was interrupted during `dendrites_fp32` at epoch 118/200. Its final dendritic status has not been established.
- No dendritic DistilBERT PQAT descendant was produced.

## Required correction before rerun

The initial diagnostic should use PAI's `testing_dendrite_capacity=True`
capacity check, which is designed to add three dendrites within seven epochs.
It verifies the model wrappers, target modules, output dimensions, and
optimizer reconstruction separately from the benchmark's production policy.
After that succeeds, run the real DistilBERT FP32 dendritic source with a
PAI-controlled continuation (`training_complete`), rather than ending after
the three-epoch dense fine-tuning budget or freezing updates before PAI reaches
a candidate phase. Keep the intended HISTORY schedule and record the added
continuation epochs as part of the dendritic condition; its dense comparison
needs a correspondingly explicit time-matched control.

Inspect the raw switch log, PB-score rows, and final parameter/architecture
evidence. Only after the source reports `verified_retained` should its PQAT
descendants be launched. Do not delete the current artifacts until the
corrected plan specifies which stale/incomplete condition directories should
be removed or regenerated.
