# What was actually built, and where

Companion to `01_UPSTREAM_AUDIT.md` (what upstream does) and
`02_OPEN_DECISIONS.md` (why we diverge where we do). This file is the map from
those to the code.

## Registered model keys

| Key | Upstream example | Dataset | Metric |
| --- | --- | --- | --- |
| `mnist_pai` | `base_examples/mnist` | MNIST | Accuracy (max) |
| `resnet18_hf_perforated_cifar100` | `transfer_learning` | CIFAR-100 | Accuracy (max) |
| `resnet18_kd_cifar100` | `base_examples/resnet` (KD) | CIFAR-100 | Accuracy (max) |
| `unet_carvana` | `base_examples/pytorch_unet` | Carvana | Dice (max) |
| `unet_supervisely` | `base_examples/segmentation-image-resolution` | Supervisely Person | mIoU (max) |

The roster is now 29 models. All five are `default_enabled`.

## Files touched

### `specs.py`
Five `ModelSpec`s. `HF_PERFORATED_RESNET18_KEY` grew a sibling and both moved
into a new `PRE_PERFORATED_MODEL_KEYS` frozenset;
`condition_supported_by_model` now tests membership rather than one key, so the
CIFAR-100 transfer entry gets the same `dendrites_*` exclusion the CIFAR-10 one
already had. That exclusion is not cosmetic: the checkpoint arrives with a
trained dendrite graph, and converting it again stacks a second search graph.

### `model_adapters.py`
Five adapters. Two declare `task_kind="segmentation"` with distinct
`primary_metric_key`s (`dice`, `miou`) -- the first time two segmentation
models with *different* reported metrics coexist here.

### `plans.py` / `training.py`
`OptimizerName` gained `"adadelta"`, wired into both `_build_optimizer` and
`_optimizer_class` (the latter is what PAI's `setup_optimizer` re-creates the
optimizer through after a restructure, so missing it would have failed only
after the first dendrite switch).

### `models.py`
* `MnistPAINet` -- upstream's `Net`, verbatim.
* `UNetDoubleConv` / `UNetDown` / `UNetUp` / `CarvanaUNet` -- milesial U-Net at
  upstream's `multFactor = 0.25`. The conv+BN pairs upstream wraps in
  `GPA.PAISequential` are plain `nn.Sequential` named `.block1` / `.block2`, so
  the base model builds without importing `perforatedai` and the perforation
  targets can still name exactly those pairs.
* `_make_divisible` / `_conv_bn` / `_conv_1x1_bn` / `InvertedResidual` /
  `MobileNetV2Backbone` / `DecoderBlock` / `SupervisleyUNet` -- the
  AntiAegis MobileNetV2-U-Net, restated rather than adapted from torchvision so
  the stage table, block internals and module ids all match what upstream's
  target set names.
* `_build_resnet18_kd_cifar100` -- ImageNet ResNet-18, dropout head,
  CIFAR stem, wrapped in the existing `ResNet18PreFC`.
* `build_kd_teacher_resnet50` / `kd_teacher_checkpoint_path`.

**Verified parameter count.** `resnet18_kd_cifar100` builds to 11,482,788
parameters. Upstream's README reports 11,490,981 for its Food-101 baseline.
The difference decomposes exactly: 11,176,512 (backbone less `fc`) + 262,656
(`pre_fc`) + 51,300 (100-way head; upstream's is 51,813 for 101 classes)
- 7,680 (3x3 CIFAR stem in place of the 7x7 ImageNet stem). Nothing unaccounted
for, which is the strongest single check that the architecture is upstream's.

### `data.py`
* `_stratified_subset_indices` / `_stratified_halves` -- upstream's
  `stratified_subset_by_class` and `split_eval_dataset_stratified`.
* `_cifar100_splits` and the two entry points `_build_cifar100_transfer`
  (RandomResizedCrop, ImageNet stats, full labels) and `_build_cifar100_kd`
  (RandomCrop+pad, CIFAR stats, 25% stratified subset).
* `_CarvanaDataset` / `_build_carvana` -- `BasicDataset` semantics at
  `--scale 0.5`. Raises with the rules URL when the gated data is absent.
* `_supervisely_resize` / `_SuperviselyDataset` / `_build_supervisely` --
  upstream's `SegmentationDataset` including augmentation order, the long-side
  resize with centred padding, and the `label[label > 0] = 1` binarization.
* `_BATCH_SIZES`, `dataset_exists` sentinels and `build_task_bundle` dispatch.

`opencv-python-headless` was added to `pyproject.toml`: upstream's segmentation
dataloader is written against cv2 (`INTER_LINEAR`/`INTER_NEAREST`,
`warpAffine`), and approximating it with PIL would silently change the
augmentation.

### `training.py`
* `_softmax_dice_loss`, `_miou_from_logits`, `_multiclass_dice_from_logits`,
  `CarvanaUNetLoss`, `SuperviselyDiceLoss` -- each a restatement of the
  upstream function it is named after.
* `_binary_or_multi_loss` branches: `NLLLoss` for `mnist_pai` (its head is
  `log_softmax`, so `CrossEntropyLoss` would log-softmax twice), and the two
  segmentation objectives.
* `_compute_all_metrics` no longer assumes segmentation means "unet_isic with a
  1-channel sigmoid head"; the two new models are 2-channel softmax and report
  different metrics, so each is dispatched explicitly.
* `_PRIMARY_METRIC_KEY` is now derived from `ALL_MODEL_KEYS` instead of a
  hand-retyped roster, which is how it stayed correct across a 24 -> 29 change.
* Knowledge distillation: `KD_ALPHA`, `KD_TEMPERATURE`, `_kd_teacher`,
  `_kd_teacher_logits`, `_kd_loss`. The teacher is cached module-level, never a
  child module, so PAI never perforates it, quantization never projects it, and
  it never enters a checkpoint. `_run_training_batch` is the only caller, which
  is what confines KD to the training loss exactly as upstream does.

### `pipeline.py`
Five recipes, three perforation target sets (`resnet18_kd_cifar100` reuses
`[".pre_fc"]`; the two U-Nets name their blocks), two track-only lists, and
cost estimates for worker balancing.

**Coverage verified.** `_uncovered_parameter_names` returns empty for all four
ID-selected models. This matters more than it looks: a parameter in neither the
perforate nor the track list gets no `parameter_type`, and this repo suppresses
the warning that would otherwise say so -- the failure mode is a silently
mistyped run, not a crash.

### `cli.py`
`dqb pretrain_kd_teacher` -- the one-time ResNet-50 fine-tune the student
requires.

### `tests/`
`test_p2_matrix_smoke.py` now derives the valid optimizer and LR-schedule sets
from their `Literal`s rather than retyping them, and takes its exclusion list
from `PRE_PERFORATED_MODEL_KEYS`. `test_p1_architecture.py`'s expected default
roster gained the five keys.

## Check status

`ty check` clean. `dqb docs --check` clean after regeneration. `pytest`: all
green except two failures that **pre-date this work** and are unrelated to it --
verified by stashing every change and re-running:

* `test_dynamic12_hf_pqat.py::test_post_run_verifier_rejects_missing_pqat_stage`
  -- imports `experiments/dynamic12/scripts/verify_pqat.py`; there is no
  `experiments/` directory in the repo.
* `test_p2_docs.py::test_superseded_documents_name_their_replacement`
  -- reads `information/DYNAMIC9_PAI_GRAPH_AUDIT.md`, which does not exist.

## Data status

| Dataset | State |
| --- | --- |
| MNIST | already cached |
| Supervisely Person | downloaded (4.3 GB archive) and extracted; 2134 train / 266 val / 267 test pairs verified loading at 320px |
| CIFAR-100 | downloading |
| Carvana | **blocked** on Kaggle competition rule acceptance (HTTP 401) |

## Smoke results so far

`mnist_pai` `base_fp32` at **one** epoch: 98.03% test accuracy
(`best_metric_value` 0.9762 on validation). Upstream's example reaches ~98%
after one epoch and ~99.1% by epoch 14, so the architecture, the NLL loss, the
Adadelta(1.0) + StepLR(1, 0.7) recipe and the un-augmented MNIST pipeline are
all behaving as the source does. `base_q8` and `base_q1` also completed.

**A methodology note worth keeping.** The first dendritic smoke was run at
`max_epochs=1` and failed with "PerforatedAI did not write its final-clean
inference artifact." That was *not* a defect in the new model: running the
existing, evidence-backed `lenet5` through the identical 1-epoch override
reproduced the same error exactly. PAI needs enough epochs to complete a switch
cycle before it emits a final artifact, so any dendritic smoke has to be given a
realistic budget. Checking the control before opening a bug saved a diagnosis
that would have chased the wrong thing.
