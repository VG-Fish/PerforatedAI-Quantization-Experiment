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
