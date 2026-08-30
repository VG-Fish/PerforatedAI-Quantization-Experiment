# Dynamic12

The priority sweep is **ResNet-18 (CIFAR-10), SAINT (Adult), and PointNet
(ModelNet40)**, replacing Dynamic11/12's TCN, GRU, and VAE. Earlier bundles in
this directory are kept as-is and are *not* part of the priority sweep:

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
experiments/dynamic12/config/run_smoke_tests.sh --models resnet18_cifar10
```

The smoke test does **not** cover the training loop. For that, patch
`BenchmarkRunner._training_hyperparameters` down to a few epochs and run the
real runner into a scratch results root — see `REMAINING_FIXES.md` §4.

## Perforation targets

Each model perforates a deliberately small, late set; everything else is
track-only. Full rationale in `information/MODEL_REFERENCE.md`.

| model | perforated | params added per dendrite |
|---|---|---|
| `resnet18_cifar10` | `.pre_fc` | 262,656 (+2.3%) |
| `saint_adult` | `.row_blocks.{0,1}.attn.qkv`, `.head.1` | 29,120 (+14.2%) |
| `pointnet_modelnet40` | `.conv3.0`, `.head.0` | 656,896 (+18.9%) |

ResNet-18 mirrors upstream PerforatedAI exactly: their published model
(`LPA.ResNetPAIPreFC`, `perforated-ai/resnet-18-perforated-gd`) adds a 512→512
`pre_fc` projection after global pooling and perforates only that, tracking the
residual backbone. The projection is in **both** arms here so a dendritic win
is not confounded with the extra dense layer. Upstream retains 5 dendrites;
this benchmark caps at 1 (`_MODEL_DYNAMIC_PAI_SCHEDULES`) to keep a
parameter-matched dense control practical.

Every parameter must fall in the perforate or track list — one in neither gets
no `parameter_type`, which PAI warns on each p-phase step and follows with
`pdb.set_trace`. `tests/test_dynamic_pai_followup.py` asserts this structurally.

## Giving a late dendrite a live learning rate

The learning rate is a pure function of the absolute epoch index, so a dendrite
inserted near the end of the budget inherits whatever the anneal has left. For
ResNet-18 that was **exactly zero**: its cosine uses the default
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
  so no other model's stored results change meaning.
- The dynamic epoch cap is derived from the schedule instead of a flat `+16`,
  which could not fit even one dendrite: a switch costs its candidate phase
  (bounded by `MAX_DENDRITE_PHASE_EPOCHS = 8`) plus an adaptation window. For
  the priority models this is `+28` rather than `+16`.

The run log prints `[pai-lr] dendrite parameter group: ...` on the epoch the
group first exists. If that line never appears, no dendrite was retained.

What is measured so far: the mechanism is verified (a real optimizer step at
epoch 228 moves the dendrite and not the backbone), and the floor is verified
inert while the schedule stays above it. It is **not** yet measured to change
any outcome — a SAINT A/B in the binding regime came out inside run-to-run
noise, and dendrite retention there came from the epoch cap, not the floor.
See MEASUREMENT_CAVEATS §11.

With both changes in place all three priority models retain a dendrite on a
short run, each adding exactly the documented per-dendrite cost:

| model | base -> dendritic params | added |
|---|---|---|
| `resnet18_cifar10` | 11,436,618 -> 11,699,274 | +262,656 |
| `pointnet_modelnet40` | 3,471,473 -> 4,128,369 | +656,896 |
| `saint_adult` | 211,906 -> 241,026 | +29,120 |

Those runs' accuracies are **not** effect sizes: under
`--dynamic-dendritic-training` the dendritic arm trains past the base arm's
budget, so the arms are not matched.

## The sweep

```bash
# Three-seed FP32/Q8/Q2 replications for the priority architectures.
experiments/dynamic12/config/run_validated_replications.sh
```

Defaults: seeds 0–2, conditions `base_fp32 dendrites_fp32 base_q8 dendrites_q8
base_q2 dendrites_q2`, `--allow-PQAT`, `--dynamic-dendritic-training`,
`--jobs 2`, writing to `priority_replications/seed_$SEED`. Override with the
`MODEL_KEYS`, `CONDITION_KEYS`, `SEEDS`, and `RUN_NAME` environment variables.

**This is a long run.** ResNet-18 alone is ~8.6 FP32-hours per arm at its
200-epoch budget, and the dendritic arm is unbounded under
`--dynamic-dendritic-training`; PointNet is ~3.0. Budget on the order of a day
per seed, and consider `SEEDS=0` first.

`--allow-PQAT` means the `q8`/`q2` arms are quantization-aware fine-tunes, not
pure PTQ. No stored result yet reflects the 2026-08-11 PQAT shadow-weight fix,
so this sweep is its first validation.

## Reading the output

The runner writes `comparison/dendrite_audit.csv`. Records with no raw PAI
candidate-insertion switch or no retained parameter increase are marked
inconclusive and excluded from dendrite comparisons. QAT artifacts without the
single-projection revision are similarly excluded until recomputed.
