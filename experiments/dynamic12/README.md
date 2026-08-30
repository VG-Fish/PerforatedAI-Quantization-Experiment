# Dynamic12

The priority sweep pairs a **nondendritic ResNet-18 control** with
**PerforatedAI's published ResNet-18 transferred to CIFAR-10**, alongside
SAINT (Adult) and PointNet (ModelNet40). The ResNet backbone and five-branch
pre-FC graph in the perforated arm come from
`perforated-ai/resnet-18-perforated-gd` on Hugging Face; it is not a newly
initialized local approximation. Earlier
bundles in this directory are kept as-is and are *not* part of this sweep:

| directory | what it holds |
|---|---|
| `combined/seed_0` | historical refreshed-model results merged with the unchanged models from Dynamic10 |
| `validated_replications/seed_{0,1}` | the older GRU/VAE post-QAT-fix replications |
| `tcn_audited_*/seed_{0,1,2}` | the TCN target-variant comparison |
| `priority_replications/seed_*` | **the new sweep** (written by `run_validated_replications.sh`) |

## Preflight

```bash
experiments/dynamic12/config/run_smoke_tests.sh
```

It uses a single real training batch for ResNet-18, SAINT, and PointNet; checks
forward, loss, backward, optimizer, FP32/Q8/Q4/Q2/channel-ternary inference, and
a one-batch ternary-QAT single-projection check; then checks every configured
PAI target variant and the target's inferred output dimensions. It makes no
benchmark artifacts and does not start PAI candidate training.

To restrict it, pass the same model argument to both checks:

```bash
experiments/dynamic12/config/run_smoke_tests.sh --models resnet18_hf_perforated_cifar10
```

The smoke test does **not** cover the training loop. For that, patch
`BenchmarkRunner._training_hyperparameters` down to a few epochs and run the
real runner into a scratch results root — see `REMAINING_FIXES.md` §4.

## Perforation targets

Each model perforates a deliberately small, late set; everything else is
track-only. Full rationale in `information/MODEL_REFERENCE.md`.

| model | perforated | params added per dendrite |
|---|---|---|
| `resnet18_cifar10` | none in the base arm (control) | — |
| `resnet18_hf_perforated_cifar10` | published `.pre_fc` graph (already perforated) | no additional graph |
| `saint_adult` | complete `.head` classifier | 4,418 (+2.1%) |
| `pointnet_modelnet40` | `.conv3.0`, `.head.0` | 656,896 (+18.9%) |

The Hugging Face ResNet uses the published weights and topology: a 512→512
`pre_fc` projection after global pooling with five saved branches (the main
projection plus four retained dendrite paths). Dynamic12 center-crops the
learned ImageNet 7×7 stem to 3×3, removes max-pooling, and replaces only the
1000-way classifier with a 10-way CIFAR classifier. Its `base_*` name means
"the loaded checkpoint without another PAI conversion"; it is already the
perforated model. Consequently, `dendrites_*` is skipped for this key rather
than stacking a second, scientifically misleading dendritic graph.

For the requested ResNet comparison, the runner treats
`resnet18_cifar10/base_*` as the nondendritic control and
`resnet18_hf_perforated_cifar10/base_*` as the dendritic/perforated
counterpart. The latter keeps the `base_*` storage keys because those are the
conditions supported by the loaded checkpoint; the comparison role is recorded
in this document rather than relabeling artifacts in a way that would break
the condition dependency chain. Both arms run FP32 plus Q8, Q4, Q2, Q1.58,
and Q1, and every quantized arm receives PQAT.

Every parameter must fall in the perforate or track list — one in neither gets
no `parameter_type`, which PAI warns on each p-phase step and follows with
`pdb.set_trace`. `tests/test_dynamic_pai_followup.py` asserts this structurally.

## Giving a late dendrite a live learning rate

The learning rate is a pure function of the absolute epoch index, so a dendrite
inserted near the end of the budget inherits whatever the anneal has left. For
the historical scratch `resnet18_cifar10` experiment that was **exactly zero**:
its cosine uses the default
`lr_min_factor=0.0`, so every epoch from 200 on — the entire window the dynamic
cap exists to provide — ran at `lr=0`. A short run made this concrete: 13 of 19
epochs at `lr=0.0`, validation flat inside 0.004, no dendrite phase ever
entered. SAINT's floor was 2% of base and PointNet's 2.8%; neither is a rate a
freshly initialized module can train at.

Worse, the plateau `DOING_HISTORY` detects at a cosine tail *is* the anneal, not
capacity saturation — so dendrites were being inserted for the wrong reason at
the worst possible moment.

Two changes address this:

- `dendrite_lr_min_factor` (per recipe, `0.1` for all three priority models)
  floors **only the retained dendrite parameters**, which PAI's optimizer now
  gets as their own param group. The backbone keeps the identical schedule its
  `base_fp32` control runs, so a dendritic gain cannot be an artifact of a warm
  restart the control never received. It defaults to `0.0` — an exact no-op —
  so no other model's stored results change meaning. This does not apply to
  the published HF ResNet, whose topology is loaded rather than discovered.
- The dynamic epoch cap is derived from the schedule instead of a flat `+16`,
  which could not fit even one dendrite: a switch costs its candidate phase
  (bounded by `MAX_DENDRITE_PHASE_EPOCHS = 8`) plus an adaptation window. For
  the priority models this is `+28` rather than `+16`.

The run log prints `[pai-lr] dendrite parameter group: ...` on the epoch the
group first exists. If that line never appears, no dendrite was retained.

What is measured so far: the mechanism is verified (a real optimizer step at
epoch 228 moves the dendrite and not the backbone), and the floor is verified
inert while the schedule stays above it. It is **not** measured to change any
outcome on either model tested: a SAINT A/B in the binding regime came out
inside run-to-run noise, and dendrite retention there came from the epoch cap,
not the floor. A follow-up ResNet-18 A/B (0.0 vs 0.01 — a categorical rather
than marginal difference) went further: per-epoch validation accuracy was
**bit-identical** between floor-on and floor-off for every epoch of the
p-phase, the only window the two schedules actually differ. The floor is
mechanically real but inert-so-far on both cases tried. See
MEASUREMENT_CAVEATS §11.

With both changes in place the scratch ResNet and the two dynamically
perforated priority models retain a dendrite on a short run, each adding
exactly the documented per-dendrite cost:

| model | base -> dendritic params | added |
|---|---|---|
| `resnet18_cifar10` | 11,436,618 -> 11,699,274 | +262,656 |
| `pointnet_modelnet40` | 3,471,473 -> 4,128,369 | +656,896 |
| `saint_adult` | 211,906 -> 216,324 | +4,418 |

Those runs' accuracies are **not** effect sizes: under
`--dynamic-dendritic-training` the dendritic arm trains past the base arm's
budget, so the arms are not matched.

## The sweep

```bash
# Three-seed replications with PQAT at every quantization level.
experiments/dynamic12/config/run_validated_replications.sh
```

Defaults: seeds 0–2. The script queues three groups per seed: the standard
ResNet's six `base_*` control arms, the HF ResNet's six already-perforated
`base_*` counterpart arms, and SAINT/PointNet with all 12 base/dendritic
condition keys. Every quantized condition uses `--allow-PQAT`; SAINT and
PointNet also use `--dynamic-dendritic-training`. Results go to
`priority_replications/seed_$SEED`. Set `MODEL_KEYS` and `CONDITION_KEYS` for
the previous single-sweep behavior, or use `BASE_RESNET_MODEL_KEYS`,
`PERFORATED_RESNET_MODEL_KEYS`, `PRIORITY_MODEL_KEYS`, `SEEDS`, and `RUN_NAME`
to change the paired defaults.

**This is a long run.** The HF ResNet transfer uses 50 FP32 epochs plus up to
10 PQAT epochs for each of five quantizers. PointNet and SAINT additionally
train a dynamic dendritic source before its five descendants. Run seed 0 first
when validating a new machine.

The runner rejects a saved quantized artifact when its QAT flag, epoch count,
quantization revision, or `before_pqat/` and `after_pqat/` stage metadata do
not match the requested PQAT run. After every seed, `verify_pqat.py` checks all
requested quantized artifacts and fails the script if any arm was PTQ-only,
skipped training, or lacks either stage snapshot.

If a priority run is already active, queue the standard ResNet control without
interrupting it:

```bash
experiments/dynamic12/config/queue_nondendritic_resnet_mps.sh
```

The queue waits for existing `dqb` workers, then runs seed 0's six standard
ResNet `base_*` arms on MPS and verifies their PQAT stage metadata.

SAINT's current calibration queue uses one complete classifier-head target and
a fixed first PAI switch at epoch 100. It runs only `base_fp32` and
`dendrites_fp32`; the five dendritic PQAT descendants are intentionally blocked
until the source record has both a raw candidate-insertion switch and a larger
final retained topology.

## Reading the output

The runner writes `comparison/dendrite_audit.csv`. Records with no raw PAI
candidate-insertion switch or no retained parameter increase are marked
inconclusive and excluded from dendrite comparisons. QAT artifacts without the
single-projection revision are similarly excluded until recomputed.
