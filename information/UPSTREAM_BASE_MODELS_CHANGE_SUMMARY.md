# Upstream-equivalent PerforatedAI base models: implementation handoff

<!-- status-banner -->
> **Status: historical (2026-09-02).** The handoff written at the end of the
> port, amended the same day by an upstream-fidelity pass. Seven things it
> describes changed:
>
> 1. The Carvana Dice term reduces over the whole batch, per upstream's
>    `multiclass_dice_coeff(..., reduce_batch_first=True)`.
> 2. The Carvana dense arms own their own `ReduceLROnPlateau`; only the
>    dendritic arm hands the schedule to PAI.
> 3. Container targets report a channels-first node axis, upstream's
>    `set_output_dimensions([-1, 0, -1, -1])`. Blast radius measured across all
>    29 registered models: **39 targets, all of them in `unet_carvana` (18) and
>    `unet_supervisely` (21)**. No pre-existing model's node axis moves, so no
>    stored result is invalidated. `resnet18_kd_cifar100` is untouched -- its
>    `.pre_fc` is a `Linear` with a rank-2 output.
> 4. The PAI extra scores use upstream's own labels (`train`, `Train Acc 1`).
> 5. `_SuperviselyDataset.__getitem__` checks `cv2.imread` for `None` before
>    reversing the channel axis, so an unreadable pair raises naming both files
>    instead of `TypeError`.
> 6. **Perforated Backpropagation stays on.** `_configure_pai_runtime_options`
>    restored `perforated_backpropagation` from a fresh `PAIConfig()`
>    *attribute* (`False`) rather than the live `get_...()` (`True`), so every
>    model that passes no runtime kwargs -- 27 of 29, including `mnist_pai`,
>    `unet_carvana` and `unet_supervisely` -- had PB switched off at
>    perforation and trained gradient-descent dendrites instead. This is the
>    benchmark's independent variable; see `compat._pai_live_config_value`.
> 7. **`p_epochs_to_switch` no longer leaks between models in a worker.** Same
>    root cause: it exists only on the live config, not on a fresh
>    `PAIConfig()`, so `_restore_pai_library_schedule_defaults` skipped it and
>    `resnet18_kd_cifar100`'s `40` (or `unet_carvana`'s `25`) stayed in force
>    for every later model in the same process, against a library default of
>    `2`. A worker takes `--models` as a list, so this was reachable.
>
> The KD student's `pre_fc` architecture and `[".pre_fc"]` target, described
> here, are **correct** and unchanged. Cite this document for what the port
> added; `information/base_examples/01_UPSTREAM_AUDIT.md` and the code are
> current for the targets, recipes, and upstream reference numbers.

**Date:** 2026-09-02  
**Purpose:** Give the next agent a complete record of the PerforatedAI base-model port, the later upstream-equivalence corrections, the retention diagnosis, and the verification state.

## Executive summary

The repository now contains five PerforatedAI base-example models and the benchmark machinery needed to train, quantize, and audit them:

| Local key | Upstream source | Dataset | Reported metric |
| --- | --- | --- | --- |
| `mnist_pai` | `examples/base_examples/mnist` | MNIST | accuracy, maximize |
| `resnet18_hf_perforated_cifar100` | `examples/transfer_learning` | CIFAR-100 | accuracy, maximize |
| `resnet18_kd_cifar100` | `examples/base_examples/resnet` | CIFAR-100 | accuracy, maximize |
| `unet_carvana` | `examples/base_examples/pytorch_unet` | Carvana | Dice, maximize |
| `unet_supervisely` | `examples/base_examples/segmentation-image-resolution` | Supervisely Person | mIoU, maximize |

The external source was cloned and inspected at PerforatedAI commit `0a5967b4574d4b280b31d6ef30beffcd4e4308ea`. The initial port added the model/data/training/CLI integration. A second pass then corrected several deliberate local substitutions so these five entries use upstream optimizer, scheduler ownership, validation signals, data splits, stems, augmentation, and PAI schedule behavior.

The most important correction is the one connected to the original “dendritic FP32 is not retained” symptom. The benchmark previously applied its own epoch-indexed scheduler after PAI had inserted a candidate. For MNIST, insertion around epoch 47 meant the candidate received a learning rate near zero, so it could not improve and was not retained. The current implementation registers upstream schedulers with PAI; PAI now steps the scheduler on validation and recreates/replays it when the topology changes. The outer loop does not overwrite a PAI-owned schedule.

## Source and investigation record

The work used:

- `information/`, especially `information/base_examples/01_UPSTREAM_AUDIT.md`, `02_OPEN_DECISIONS.md`, `03_IMPLEMENTATION_RECORD.md`, `04_DIAGNOSIS_pai_final_artifact.md`, and `06_DIAGNOSIS_control_conditions_abort_the_sweep.md`;
- the local `PAI Skills` guidance for tracker, optimizer, scheduler, and artifact ownership;
- the upstream GitHub examples listed above, cloned through the repository-search workflow;
- the installed PerforatedAI 3.2.7 API. In particular, the installed tracker exposes `set_scheduler(scheduler)`, `set_optimizer_instance(optimizer_instance)`, and `setup_optimizer(net, opt_args, sched_args=None, parameters=None)`.

The earlier investigation was treated as provisional, as requested. Every recorded departure was rechecked against the upstream source before being changed. Some user decisions remain intentional: CIFAR-100 was selected for the KD entry, official Carvana data is required, and the teacher remains an out-of-band prerequisite rather than a new matrix stage.

## Initial port: registry and public model surface

### Model registry and adapters

`src/dendritic_benchmark/specs.py` now registers all five keys with dataset, display name, metric, and direction. `src/dendritic_benchmark/model_adapters.py` adds matching task adapters. The five are default-enabled, so they appear in the normal roster in addition to the previously registered models; selecting model keys explicitly remains supported.

The HF transfer model is marked as pre-perforated. Its checkpoint already contains a trained PAI graph, so the matrix excludes a second `dendrites_*` conversion. `condition_supported_by_model` was generalized from one hard-coded CIFAR-10 key to `PRE_PERFORATED_MODEL_KEYS`. The control-condition planner was also hardened: controls whose declared `dendrites_fp32` source is absent are recorded as unavailable instead of aborting a whole multi-hour sweep or quantizing a random model.

### Model implementations in `models.py`

- `MnistPAINet` is the upstream MNIST CNN: 1→32 and 32→64 3×3 convolutions, max pool, dropout 0.25, 9216→128 linear, dropout 0.5, 128→10 head, and `log_softmax` output.
- The Carvana implementation restates the quarter-width milesial/Pytorch-UNet architecture, including `DoubleConv`, down blocks, up blocks, and the two-channel output. Conv+BatchNorm pairs have stable `.block1`/`.block2` names so the PAI target set can match upstream’s `PAISequential` pairs without importing PAI into the dense model class.
- The Supervisely implementation restates the upstream MobileNetV2 encoder/decoder, including `_make_divisible`, the exact `InvertedResidual` stage layout, decoder blocks, skip connections, and two-class output. Class names and module IDs are preserved because upstream selects `InvertedResidual`, `DecoderBlock`, `Linear`, and `Conv2d` by type/name.
- The KD student uses torchvision ResNet-18, a dropout-0.2 head, and the local `ResNet18PreFC` wrapper. The wrapper now accepts `identity_initialize_pre_fc`; existing CIFAR-10 behavior keeps identity initialization, while the KD upstream path uses default `nn.Linear` initialization, matching upstream `ResNetPAI`.
- The KD teacher uses torchvision ResNet-50 with ImageNet V2 weights and a fresh 100-way head. The teacher now retains the stock 7×7 stride-2 stem and max-pool, as upstream does; the previous local 3×3 CIFAR adaptation was removed.
- The HF transfer loader still uses the verified `perforated-ai/resnet-18-perforated-gd` checkpoint. The upstream model-card alias and transfer examples refer to the same published family. The loader remains low-level and checksum-pinned because the installed high-level helper double-converted this legacy checkpoint under PerforatedAI 3.2.6/3.2.7. The CIFAR-100 transfer path now retains the stock ImageNet stem and max-pool; the pre-existing CIFAR-10 entry can still request its separate CIFAR stem adaptation through `adapt_cifar_stem=True`.

The KD architecture change invalidates the previous local teacher checkpoint. `KD_TEACHER_CIFAR100_FILENAME` is now `resnet50_cifar100_teacher_upstream_stem_v2.pt`; the old 3×3-stem file is left untouched but will no longer be loaded accidentally.

## Initial port: data and preprocessing

`src/dendritic_benchmark/data.py` adds the five data builders and bumps `DATA_PIPELINE_REVISION` to `upstream_base_examples_2026_09_02_v2`, invalidating artifacts made with the old preprocessing.

### MNIST

`VisionDatasets.mnist_pai` uses the complete 60,000-image training set, the complete 10,000-image test set as both validation and test, `ToTensor`, and `Normalize((0.1307,), (0.3081,))`. Training batches are 64 and evaluation batches are 1,000, matching `mnist_perforatedai.py`. It is separate from the older benchmark MNIST builder, which retains its own 55k/5k split and optional shift augmentation.

### CIFAR-100 transfer

The transfer builder uses `RandomResizedCrop(32)` and horizontal flip for training, `Resize(32)` plus `CenterCrop(32)` for evaluation, and ImageNet channel statistics. It uses the full CIFAR-100 training set and the full CIFAR-100 test set for both validation and test, matching the transfer-learning example.

### CIFAR-100 KD

The KD builder uses `RandomCrop(32, padding=4)`, horizontal flip, and CIFAR-100 statistics. It keeps the upstream stratified 25% training-label subset and the class-balanced validation/test half split. The reported MixUp/CutMix settings are now present in `_kd_mixup_cutmix_collate`: each batch receives one randomly selected transform, MixUp alpha 0.2 or CutMix alpha 0.6, with one-hot soft targets. Classification metrics convert soft targets to their dominant class only for reporting; the loss receives the soft labels.

### Carvana

`_CarvanaDataset` matches upstream `BasicDataset` at scale 0.5: RGB image tensors in [0,1], bicubic image resize, nearest-neighbor mask resize, and binary long mask targets. The official Kaggle `train_hq`/`train_masks` data is required; the builder gives the rules URL rather than silently using a re-encoded mirror. The split is now upstream’s seeded 90% train / 10% validation split (seed 0), with the same validation dataset reused for the benchmark’s final test pass. Batch size is 1 and validation uses `drop_last=True`.

### Supervisely

The cv2-backed loader restates the upstream resize, centered padding, affine augmentation order, noise, crop, flip, and binary mask conversion. The fixed upstream train and valid pair files are used directly. The entire valid list is used for both validation and test, rather than the earlier local 50/50 split. `opencv-python-headless` was added to `pyproject.toml`; `_cv2()` disables OpenCV’s internal thread pool so four DataLoader workers do not oversubscribe the machine.

### Loader extensions

`_make_loader` and `_bundle_from_splits` now support an evaluation batch size, `drop_last`, and a training `collate_fn`. Existing callers retain their previous defaults. The per-model fallback batch-size table records upstream values: MNIST 64, transfer 128, KD 32, Carvana 1, Supervisely 16.

## Initial port: objectives and metrics

`src/dendritic_benchmark/training.py` adds upstream-aligned objectives:

- MNIST uses `NLLLoss`, not `CrossEntropyLoss`, because the model already emits log probabilities.
- Carvana uses the upstream binary cross-entropy plus soft Dice objective.
- Supervisely uses the upstream multiclass Dice objective.
- mIoU is calculated from two-class softmax logits, and Dice/mIoU are dispatched by model key instead of assuming every segmentation model is the older one-channel ISIC task.
- KD is training-only: `_kd_teacher_logits` loads and freezes the external teacher, `_kd_loss` computes `(1-alpha)*CE + alpha*KL*T²` with alpha 0.4 and temperature 4.0, and validation/test remain plain cross-entropy. The teacher is module-level cached, never a student child, so it is not perforated, quantized, or serialized with the student.
- Soft MixUp/CutMix labels are accepted by classification metric code while preserving integer-label behavior for every other model.

## PAI target sets and runtime configuration

The pipeline records exact target and track-only selections in artifact metadata and checks for uncovered parameters. The current upstream-oriented selections are:

- MNIST: PAI’s type-selected Conv2d/Linear defaults.
- KD: `.pre_fc` is perforated; ResNet block/stem/head modules are tracked as upstream’s `convert-count 0` arrangement.
- Carvana: the 19 Conv2d/BatchNorm pairs plus output Conv2d are perforated; ConvTranspose2d modules are track-only.
- Supervisely: upstream’s InvertedResidual/DecoderBlock/Linear/Conv2d targets are represented by the matching stable IDs; the specified stem/final transition track-only modules remain track-only.

`compat.py` gained runtime options for `weight_decay_accepted`, `cap_at_n`, `test_saves`, `perforated_backpropagation`, and the PAI forward function. The pipeline supplies the KD settings (history mode, cap-at-N where supported, test saves, PB, ReLU), Carvana weight-decay acceptance, and safe library defaults for the other examples. PAI global schedule fields are reset from a fresh library configuration before applying an example-specific schedule, preventing one model’s settings from leaking into the next model in the same process.

## Upstream-equivalence correction: optimizers and schedules

`plans.py` and `TrainingConfig` now support the additional schedule/optimizer contracts:

- `OptimizerName` includes `rmsprop`; both ordinary and PAI optimizer builders construct RMSprop with the configured learning rate, momentum, and weight decay.
- `LRScheduleName` includes `plateau` and `poly`.
- Recipes can specify ReduceLROnPlateau factor/patience/mode, polynomial power, PAI scheduler ownership, whether PAI should construct the optimizer or receive an instance, and a restructure learning-rate multiplier.
- Checkpoint save/load includes a trainer-owned scheduler state when one exists.

The five recipes are now:

| Model | Optimizer and rate | Schedule | PAI-specific behavior |
| --- | --- | --- | --- |
| MNIST | Adadelta, lr 1.0 | StepLR(1, 0.7) | PAI owns and replays scheduler |
| HF transfer | SGD, lr 0.1, momentum 0.9, wd 1e-4 | cosine, 200 epochs, label smoothing 0.1 | pre-perforated; no second search |
| KD | SGD, lr 0.0125, momentum 0.9, wd 1e-3 | StepLR(30, 0.1) | PAI owns scheduler; rebuilt optimizers use 10× rate |
| Carvana | RMSprop, lr 1e-5, momentum 0.999, wd 1e-8 | ReduceLROnPlateau(mode=max, patience=5) | PAI owns scheduler and optimizer setup |
| Supervisely | SGD, lr 0.01, momentum 0.9, wd 1e-8 | per-batch polynomial power 0.9 | PAI receives an optimizer instance; trainer owns poly schedule |

### The scheduler ownership fix

Before this correction, `_setup_pai_optimizer` called `tracker.set_optimizer` but never called `tracker.set_scheduler`; the outer `_apply_lr_schedule` then recomputed a learning rate from the original global epoch. That is incompatible with PAI’s `PARAM_VALS_BY_UPDATE_EPOCH` behavior because a new dendrite starts late in the run but receives a rate intended for an old parameter group.

The corrected flow is:

1. Build the upstream optimizer and scheduler specification.
2. Call `tracker.set_optimizer(...)` and `tracker.set_scheduler(...)` before `setup_optimizer` for MNIST, KD, and Carvana.
3. Let `pai_tracker.add_validation_score` step the registered scheduler, as upstream does.
4. On a restructure, rebuild the optimizer and scheduler through the same path. KD applies its upstream 10× post-restructure learning-rate multiplier.
5. Skip benchmark-side epoch scheduling whenever PAI owns the schedule.

Supervisely intentionally takes the other upstream path: `_setup_pai_optimizer` calls `set_optimizer_instance`, and `_run_epoch_batches` applies `lr = init_lr * (1 - current_iteration / maximum_iterations) ** 0.9` after each batch. `lr_schedule_restart_epoch` resets the polynomial phase after a PAI restructure and is reconstructed from PAI switch state on resume.

### Carvana validation clock

Upstream Carvana evaluates five times per epoch, and those validation calls are also the PAI and ReduceLROnPlateau clocks. `_run_epoch_batches` now supports an intra-epoch callback. The Carvana path invokes validation at five evenly spaced batch intervals, passes each Dice score to PAI, adopts a replacement model/optimizer after a switch, and resumes candidate-graph mode. Dense Carvana runs use the same callback to step ReduceLROnPlateau five times per epoch. The final epoch-level validation remains the benchmark’s reported row.

## Upstream PAI growth policies and score signals

`BenchmarkRunner._pai_dynamic_schedule` now supplies the policies found in the corresponding examples rather than relying on a generic local schedule:

- MNIST leaves PAI defaults untouched.
- Carvana uses max two dendrites, 25 normal epochs, and 25 PAI epochs.
- KD uses max five dendrites, 40 normal epochs, 40 PAI epochs, initial history after two switches, thresholds `(0.001, 0.0001, 0.0)`, candidate multiplier 0.1, and PB enabled.
- Supervisely uses the upstream fixed switch at 80.

The pipeline now enables `train_dendrites_until_complete` for the four dynamically trained upstream models, so the benchmark’s 20% freeze and ordinary fixed-epoch stop cannot prevent PAI from reaching a retained topology. The pre-perforated HF transfer model remains a controls-only entry because it already contains a trained graph.

PAI’s decision score is separated from the benchmark’s reporting score:

- MNIST and KD accuracy are multiplied by 100 before being passed to PAI, matching upstream’s percentage scores.
- Supervisely passes validation loss to PAI with `maximizing_score=False`, while the benchmark still reports/checkpoints mIoU as a maximizing metric.
- Carvana passes Dice as a maximizing score.

## Artifact, sweep, and CLI changes

- The five model revisions are marked `upstream_base_equivalence_v2`; the data revision is bumped, so stale artifacts cannot silently masquerade as equivalent results.
- The existing final-clean PAI artifact checks, switch evidence audit, source-topology checks, and unavailable-control recording remain strict. A dendritic FP32 source without retained-insertion evidence still cannot feed PQAT descendants.
- `dqb pretrain_kd_teacher` was added as the explicit external teacher-pretraining command. Its defaults now match upstream’s 90 epochs, batch 32, SGD lr 0.0125, StepLR(30, 0.1), momentum 0.9, weight decay 1e-3, and label smoothing 0.1.
- `.gitignore` excludes local run/smoke output. `information/CURRENT_GUIDE.md` was regenerated after the registry/recipe changes. `uv.lock` and `pyproject.toml` include the segmentation dependency update.

## Tests added or updated

`tests/test_upstream_base_equivalence.py` covers:

- all five recipe contracts;
- model-specific PAI growth schedules and the Supervisely fixed switch;
- PAI-owned StepLR registration;
- Supervisely optimizer-instance setup and loss-based PAI signal;
- MixUp/CutMix soft-target shape and normalization;
- stock ImageNet KD stems and non-identity upstream pre-FC initialization;
- five intra-epoch callback invocations.

Existing tests were updated where the public roster, optimizer/schedule literals, artifact properties, unavailable controls, or training-pass return contract changed. `ty check` passes, and all focused suites pass.

## Verification state

Commands run successfully:

```text
uv run ty check
All checks passed!

uv run pytest -q tests/test_upstream_base_equivalence.py \
  tests/test_dynamic_pai_followup.py \
  tests/test_p2_overrides.py \
  tests/test_p1_architecture.py \
  tests/test_p2_matrix_smoke.py
................................................................. [100%]
```

The complete suite currently has three unrelated/pre-existing failures:

1. `tests/test_dynamic12_hf_pqat.py::test_post_run_verifier_rejects_missing_pqat_stage` imports the absent `experiments/dynamic12/scripts/verify_pqat.py`.
2. `tests/test_p2_docs.py::GeneratedGuideTests::test_the_checked_in_guide_matches_the_registries` was stale immediately after the recipe edits; `uv run dqb docs` regenerated `information/CURRENT_GUIDE.md`, so rerun this check after the final working-tree state.
3. `tests/test_p2_docs.py::DocumentationIndexTests::test_superseded_documents_name_their_replacement` reads the absent historical file `information/DYNAMIC9_PAI_GRAPH_AUDIT.md`.

The first and third failures are missing-repository-file failures, not failures in the five-model implementation. The guide failure was a generated-document synchronization issue and should be rechecked after this handoff file is added.

## Data and execution caveats for the next agent

- Carvana remains gated by Kaggle competition-rule acceptance. The builder intentionally fails with the acceptance URL when official data is absent.
- CIFAR-100 must be available before the transfer and KD entries can run. The KD student also requires the newly named upstream-stem teacher checkpoint; run `dqb pretrain_kd_teacher` to produce it.
- Existing artifacts made before `upstream_base_equivalence_v2` are intentionally stale. Do not copy their metrics into a new comparison without retraining or explicitly documenting the old recipe.
- The worktree was already dirty before this final documentation request and contains the broad five-model port plus prior experiment/docs changes. Do not use a destructive reset to isolate this work. Review `git status` and preserve unrelated user edits.
- `information/base_examples/02_OPEN_DECISIONS.md` contains the historical decisions that motivated earlier local departures (batch-size scaling, split isolation, omitted MixUp/CutMix, CIFAR stem adaptation, fixed epoch budgets, and mIoU as the PAI score). This implementation pass supersedes those technical departures for the five base models. D1 (CIFAR-100 selection), D2 (official Carvana data), and D5 (out-of-band teacher) remain intentional decisions.

## Files changed in the working tree

The implementation touches the following tracked areas:

- `src/dendritic_benchmark/specs.py`, `model_adapters.py`: registry and task metadata;
- `src/dendritic_benchmark/models.py`: five model families, checkpoint loading, KD teacher, and ResNet stem/init behavior;
- `src/dendritic_benchmark/data.py`: all five data builders, exact preprocessing/splits, loader options, and CV2 support;
- `src/dendritic_benchmark/training.py`: objectives, metrics, KD, RMSprop, PAI scheduler ownership, polynomial scheduling, checkpoint scheduler state, and intra-epoch validation;
- `src/dendritic_benchmark/compat.py`: PAI global runtime settings and reset behavior;
- `src/dendritic_benchmark/plans.py`: recipe/override types for the new optimizer and schedule fields;
- `src/dendritic_benchmark/pipeline.py`: recipes, PAI targets/policies, score directions, open-ended dynamic training, revisions, and runtime wiring;
- `src/dendritic_benchmark/cli.py`: teacher pretraining command and upstream defaults;
- `pyproject.toml`, `uv.lock`, `.gitignore`, and `information/CURRENT_GUIDE.md`;
- `tests/test_upstream_base_equivalence.py` plus updates to the existing dynamic, architecture, artifact, and matrix tests;
- `information/base_examples/` source audit, decisions, implementation, diagnosis, status, and handoff notes.

This file is a handoff summary, not a claim that every old decision document is still current. The source of truth for the implemented behavior is the code and the tests listed above.
