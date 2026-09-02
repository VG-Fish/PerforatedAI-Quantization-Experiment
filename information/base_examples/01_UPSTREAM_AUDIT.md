# Upstream audit of the five requested PerforatedAI examples

Source pin: `PerforatedAI/PerforatedAI` @ `0a5967b4574d4b280b31d6ef30beffcd4e4308ea`
(upstream `main`, 2026-08-23). Verified by clone, not by memory.

Everything below is read out of that tree. Where the benchmark cannot express an
upstream choice faithfully, the departure is named explicitly rather than
silently absorbed.

**Where that pin actually is.** The checkout on this machine,
`/Users/vishy/Desktop/PerforatedAI`, is a *fork* -- `origin` is
`VG-Fish/PerforatedAI`, the branch is `updated_examples`, and it sits at
`0cc9c32` (2026-08-18). It does **not** contain the pin above; `git cat-file -t
0a5967b4...` there fails with "no such commit", which reads at first glance like
the pin was invented. It was not: `git ls-remote upstream main` resolves to
exactly `0a5967b4...`, and fetching that commit into a scratch bare repo and
diffing it against the local tree shows all five audited directories
(`base_examples/{mnist,pytorch_unet,segmentation-image-resolution,resnet}` and
`transfer_learning`) are **byte-identical**. So the audit below describes the
pinned upstream code even though the working tree it was read from is a fork.
Re-verify with:

    git init --bare /tmp/pin.git
    git -C /tmp/pin.git fetch --depth=1 \
        https://github.com/PerforatedAI/PerforatedAI.git 0a5967b4574d4b280b31d6ef30beffcd4e4308ea
    git -C /tmp/pin.git archive FETCH_HEAD | tar -x -C /tmp/pin_tree
    diff -rq /tmp/pin_tree/examples/base_examples/mnist \
             /Users/vishy/Desktop/PerforatedAI/examples/base_examples/mnist

**Scope.** `examples/base_examples/` holds **six** directories at the pin, not
four: `mnist`, `pytorch_unet`, `resnet`, `segmentation-image-resolution`,
`transformer`, and `yolo-pascal`. The five entries audited here are the five
that were requested; `transformer` (WikiText LM, needs `wandb`) and
`yolo-pascal` (YOLOv11n on VOC2007, needs `ultralytics` + a VOC download) were
not, and are not in the benchmark. They existed at the pin -- all six landed in
the same restructure commit `2a763aa` on 2026-08-13 -- so their absence is a
scope boundary, not an oversight to be fixed by re-reading upstream.

---

## 1. `examples/base_examples/mnist`

**File:** `mnist_perforatedai.py` (baseline `mnist.py` is the same net without PAI).

| Aspect | Upstream value |
| --- | --- |
| Architecture | `Net(width)`: Conv2d(1,32,3,1) -> ReLU -> Conv2d(32,64,3,1) -> ReLU -> MaxPool2d(2) -> Dropout(0.25) -> flatten -> Linear(9216,128) -> ReLU -> Dropout(0.5) -> Linear(128,10) -> `log_softmax` |
| Dataset | `torchvision.datasets.MNIST`, `ToTensor` + `Normalize((0.1307,),(0.3081,))`. No augmentation. |
| Loss | `F.nll_loss` over log-probabilities |
| Optimizer | `Adadelta`, `lr=1.0` |
| Scheduler | `StepLR(step_size=1, gamma=0.7)` |
| Batch | train 64, test 1000 |
| Epochs | 10000 (i.e. run until PAI reports `training_complete`) |
| PAI config | only `set_testing_dendrite_capacity(False)`; `UPA.perforate_model(model)` with **default type-based** module selection |

Notes that matter for integration:

* This is **not** the registry's existing `lenet5`. Different topology (32/64
  3x3 convs vs LeNet's 6/16 5x5), different optimizer family, different loss.
  It must be a new model key, not a re-tune of `lenet5`.
* The output layer is `log_softmax`, so the criterion is `NLLLoss`, not
  `CrossEntropyLoss`. Feeding log-probs to `CrossEntropyLoss` would apply a
  second log-softmax and quietly flatten the gradient.
* `Adadelta` is not in the benchmark's `OptimizerName` literal today.
* `add_extra_score(train_acc, 'train')` each epoch and
  `add_validation_score(test_acc, model)` each epoch: a maximizing accuracy
  metric, which is the benchmark's default direction.

## 2. `examples/transfer_learning` (CIFAR-100 only)

**File:** `train_from_hf_sweep.py` is the script whose CIFAR-100 path is
complete; `train_flowers_from_hf.py` is flowers-only.

| Aspect | Upstream value |
| --- | --- |
| Architecture | `torchvision resnet18(weights=None, num_classes=1000)` -> `LPA.ResNetPAIPreFC(base)` -> `UPA.from_hf_pretrained(model, hf_repo_id)` -> `model.fc = Linear(in_features, 100)` |
| HF repos used in README | `perforated-ai/resnet-18-perforated-cascor`, `perforated-ai/resnet-18-perforated` |
| Dataset | `CIFAR100`, img_size **32** |
| Train transform | `RandomResizedCrop(32, bilinear)` + `RandomHorizontalFlip` + `ToTensor` + ImageNet-stat `Normalize` |
| Eval transform | `Resize(32)` + `CenterCrop(32)` + `ToTensor` + ImageNet-stat `Normalize` (`val_resize_size = img_size` because `img_size <= 32`) |
| Optimizer | `SGD(lr, momentum=0.9, weight_decay=1e-4)` — hard-coded in `train_single_run`, so the config's `weight_decay: 5e-4` is **dead** |
| Scheduler | `CosineAnnealingLR(T_max=epochs - warmup, eta_min=0)`, warmup 0 for CIFAR-100 |
| Config (`get_dataset_config('cifar100')`) | epochs 200, batch 128, lr 0.1, label_smoothing 0.1, lr_warmup_epochs 0 |
| Loss | `CrossEntropyLoss(label_smoothing=0.1)` |

Notes:

* The dendrites come **pre-trained in the checkpoint**; this script never calls
  `perforate_model` or the tracker. It is transfer of a published perforated
  backbone, exactly like the registry's existing
  `resnet18_hf_perforated_cifar10`. The CIFAR-100 entry is therefore a close
  sibling of an entry the benchmark already runs correctly.
* Because the graph arrives already perforated, `dendrites_*` conditions are
  not a meaningful second control — the same exclusion
  `condition_supported_by_model` already applies to
  `resnet18_hf_perforated_cifar10` must apply here.

## 3. `examples/base_examples/pytorch_unet`

**File:** `train_perforatedai.py`, model in `unet/unet_model.py`.

| Aspect | Upstream value |
| --- | --- |
| Architecture | `UNet(n_channels=3, n_classes=2, bilinear=False)` with `multFactor = 0.25` -> channel widths 16/32/64/128/256 |
| PAI wrapping | each `DoubleConv` uses two `GPA.PAISequential([Conv2d, BatchNorm2d])` blocks |
| Dataset | Carvana Image Masking Challenge; `dir_img = ./data/train_hq`, `dir_mask = ./data/train_masks` |
| Image scale | `--scale 0.5` |
| Val split | `--validation 10.0` (10%) |
| Loss | `CrossEntropyLoss` + `dice_loss(softmax(logits), one_hot(target), multiclass=True)` |
| Metric | validation **Dice** (`multiclass_dice_coeff` over the foreground channel only) — maximize |
| Optimizer | `RMSprop(lr=1e-5, weight_decay=1e-8, momentum=0.999)` |
| Scheduler | `ReduceLROnPlateau(mode='max', patience=5)` |
| Batch | 1 |
| Grad clip | `clip_grad_norm_(..., 1.0)` |
| PAI config | `DOING_HISTORY`, `n/p_epochs_to_switch=25`, `nodeIndex=1`, `output_dimensions=[-1,0,-1,-1]`, `unwrapped_modules_confirmed(True)`, `weight_decay_accepted(True)`, `testing_dendrite_capacity(False)`, `max_dendrites(2)`, `append_module_names_to_track(['ConvTranspose2d'])` |

**Upstream publishes no number for this example.** Its `best_arch_scores.csv`
is an empty file and its README reports no Dice, accuracy, or loss -- unlike the
other three, which all state at least one figure. There is therefore nothing for
`unet_carvana` to reproduce: fidelity here can only be argued from the recipe
(optimizer, schedule, loss, split, scale, PAI config), never from a score.

**Access blocker (verified):** the Kaggle competition endpoint returns HTTP 401
for the configured credentials — the account has not accepted the Carvana
competition rules. `datasets/list?search=carvana` works, so the key itself is
valid. See `02_OPEN_DECISIONS.md`.

## 4. `examples/base_examples/segmentation-image-resolution` (full resolution)

**File:** `train.py`, config `config/config_UNet.json`, model `models/UNet.py`.

| Aspect | Upstream value |
| --- | --- |
| Architecture | `UNet(backbone='mobilenetv2', num_classes=2, pretrained_backbone=None)`; MobileNetV2 encoder + 4 `DecoderBlock`s (`ConvTranspose2d(k=4,s=2,p=1)` + concat shortcut + `InvertedResidual`) |
| Dataset | Supervisely Person, pairs files `dataset/train_supervisely.txt` / `dataset/valid_supervisely.txt` |
| Resolution | `resize: 320` — this is the "full" setting the request asks for (`--resolution-scale` default) |
| Train aug | `noise_std 3`, `crop_range [0.90, 1.0]`, `flip_hor 0.5`, `rotate 0.0`, `normalize`, RGB |
| Loss | `dice_loss` (softmax over 2 channels, scatter one-hot, `1 - 2*inter/(sum+sum)`) |
| Metric | `miou` (argmax -> one-hot -> intersection/union, mean over batch **and both classes**) |
| Optimizer | `SGD(lr=1e-2, momentum=0.9, weight_decay=1e-8)` |
| Scheduler | `StepLR(step_size=100, gamma=1.0)` — constant in an 80-epoch run |
| Batch | 16 |
| Epochs | 80 |
| Monitor | `valid_loss`, mode `min`; `perforate_model(..., maximizing_score=False)` |
| PAI config | `set_module_names_to_perforate(['InvertedResidual','DecoderBlock','Linear','Conv2d'])`, `output_dimensions=[-1,0,-1,-1]`, `append_module_ids_to_track(['.backbone.features.0','.backbone.features.18.1'])`, `DOING_FIXED_SWITCH`, `fixed_switch_num=80`, `first_fixed_switch_num=80` |

Reported full-resolution baseline: valid mIoU **0.8420**, valid loss 0.0978.

**Access (verified good):** the README's Google Drive id
`1Y1atvePuMx1pyIOVJNGgJ_jVNBy_Bds8` resolves to
`Human Segmentation Dataset.zip (4.3G)` and is downloadable without auth via the
`drive.usercontent.google.com` confirm-token flow.

**Caveat:** the checked-in pairs files are `train_mask.txt` / `valid_mask.txt`
(6626 / 737 pairs) with absolute `/root/HumanSeg_data/...` paths, not the
`*_supervisely.txt` the config names. Pair files must be regenerated from the
extracted archive (`dataset/create_pairs.py`) with local paths.

**Why the regenerated split is smaller, and why 0.8420 is out of reach.** The
regenerated lists hold 2134 train / 533 valid = **2667** pairs against
upstream's 6626 / 737 = **7363**. That is not a bug in the regeneration.
`create_pairs.py` globs *two* dataset roots and concatenates them:

```python
image_files  = sorted(glob(".../HumanSeg/EG/data_for_run/images/*.*"))
image_files += sorted(glob(".../HumanSeg/Supervisely/data_for_run/images/*.*"))
```

Only the second root ships in the downloadable 4.3 GB archive -- it extracts to
`supervisely_person_clean_2667_img`, and 2134 + 533 = 2667 exactly, matching
`dataset/supervisely_all_pairs.txt`'s line count. The `EG` half (~4,696 pairs)
lives behind an absolute path on the original author's machine and is not
published anywhere in the repo.

So `unet_supervisely` trains on 36% of upstream's data. Upstream's reported
valid mIoU **0.8420** and valid loss 0.0978 are measured on a different and
larger corpus and are therefore **not a reproduction target** for this
benchmark; comparing against them measures the missing dataset, not the port.
Fidelity for this entry is argued from the recipe, not from the score.

## 5. `examples/base_examples/resnet` (KD + perforation)

**File:** `train_perforated_resnet_KD.py`, wrapper `resnet_double.py`.

| Aspect | Upstream value |
| --- | --- |
| Student | `torchvision resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')`, `fc -> Sequential(Dropout(0.2), Linear(512, num_classes))`, then `custom_resnet.ResNetPAI(model)` |
| Teacher | `resnet50(weights=IMAGENET1K_V2)` with `fc -> Linear(2048, num_classes)`, **fine-tuned on the same dataset first** via `--pre-train-teacher`, then frozen and `eval()` |
| KD loss | `KD_ALPHA = 0.4`, `KD_TEMPERATURE = 4.0`; `loss = 0.6*CE + 0.4 * KL(log_softmax(s/T), softmax(t/T)) * T^2` |
| Dataset (README table) | Food-101, `--train-label-fraction 0.25`, 224px |
| Dataset (also supported) | CIFAR-100, 32px, `RandomCrop(32,pad=4)` + flip, CIFAR normalization |
| Eval split | test set split stratified 50/50 into val and test (`--val-test-split-seed 42`) |
| Optimizer | `SGD(lr=0.0125, momentum=0.9, weight_decay=1e-4)`; README overrides `--wd 0.001` |
| Scheduler | `steplr`, `lr_step_size=30`, `lr_gamma=0.1` |
| Batch / epochs | 32 / 90 |
| Aug (README) | `label_smoothing 0.1`, `mixup_alpha 0.2`, `cutmix_alpha 0.6`, `random_erase 0.2`, `auto_augment ta_wide`, `dropout 0.2` |
| PAI config | `DOING_HISTORY`, `n/p_epochs_to_switch=40`, `cap_at_n(True)`, `initial_history_after_switches(2)`, `test_saves(True)`, `testing_dendrite_capacity(False)`, `append_module_names_to_perforate(['BasicBlock','Bottleneck'])`, `improvement_threshold=[0.001,0.0001,0]`, `candidate_weight_initialization_multiplier=0.1`, `pai_forward_function=relu`, dendrite_mode 2 -> `max_dendrites(5)` + `perforated_backpropagation(True)` |
| Track list | `--convert-count 0` -> tracks `.layer1..4`, plus `.conv1`, `.bn1`, `.fc`; so **`.pre_fc` is the sole perforated module** |
| Extra scores | `add_extra_score` x5: `"Train Acc 1"`, `"Train Acc 5"` (train), `"Val Acc 5"` (validate), `"Test Acc 1"`, `"Test Acc 5"` (test) |

**Known, bounded gap: only `"Train Acc 1"` is emitted.** The benchmark's
per-epoch metric contract is top-1 accuracy on a train/validation split, with no
top-5 and no separate test pass, so `best_arch_scores.csv` here carries one
extra-score column where upstream's carries five. Closing it would mean adding
top-5 to the shared classification metric path for all 20-odd classification
models and a third evaluation pass, changing every stored record for a column
that is diagnostic -- upstream's reported result is top-1. `mnist`'s single
`'train'` extra score *is* reproduced exactly; `pytorch_unet` and
`segmentation-image-resolution` call `add_extra_score` not at all, which the
port also matches.

Reported Food-101 top-1: baseline 74.92, perforated 77.38, KD 77.35,
**KD + perforated 78.60**. The directory's `best_arch_scores.csv` records a
different, shorter run -- two architectures rather than the README's three/two
dendrites -- and its dataset is not stated:

| Params | Max valid | Train Acc 1 | Test Acc 1 |
| --- | --- | --- | --- |
| 11,490,981 | 75.2396 | 56.7464 | 75.2396 |
| 11,753,637 | 76.9505 | 64.2142 | 77.4495 |

That CSV is the second, independent confirmation of the architecture: its first
row is the same 11,490,981 the README reports, and 11,753,637 - 11,490,981 =
**262,656**, one `pre_fc` dendrite exactly. Neither number is a target for this
benchmark, which trains CIFAR-100 (decision D1), not Food-101.

**Which `ResNetPAI`.** This directory defines the class *twice*, in `resnet.py`
and in `resnet_double.py`, and they are not the same architecture. The KD
script builds the second one -- line 29 is `import resnet_double as
custom_resnet`, and line 1222 is `model = custom_resnet.ResNetPAI(model)`.

* `resnet_double.ResNetPAI` (the one used): leaves `conv1`/`bn1` separate --
  its `b1 = GPA.PAISequential([...])` line is commented out -- inserts
  `pre_fc = nn.Linear(512, 512)`, and forwards
  `fc(relu(pre_fc(flatten(avgpool(...)))))`.
* `resnet.py:1151 ResNetPAI` (**not** used here): folds `conv1`+`bn1` into
  `self.b1 = GPA.PAISequential([conv1, bn1])` and has no `pre_fc` at all.

Reading `resnet.py` by mistake moves the perforation target from `.pre_fc` onto
the stem, and nothing in the code catches it. The parameter count does:
upstream's README reports **11,490,981** for its Food-101 baseline, a 101-way
ResNet-18 is 11,228,325, and the difference is exactly `pre_fc`'s
512x512 + 512 = 262,656. `tests/test_upstream_base_equivalence.py` asserts that
arithmetic for this reason.

So `ResNetPAI._forward_impl` is
`fc(relu(pre_fc(flatten(avgpool(layer4(...))))))` -- byte-for-byte the forward
path of the benchmark's existing `models.ResNet18PreFC`, and its
`--convert-count 0` target set is exactly the existing
`_default_module_ids_to_perforate` `[".pre_fc"]` for `resnet18_cifar10`. The
architecture and perforation target need no new code; what is new is the **KD
teacher, the KD loss, and the dataset**.

Two departures the benchmark must declare:

* `ResNet18PreFC` identity-initializes `pre_fc`; upstream uses default
  `nn.Linear` init. The KD entry passes `identity_initialize_pre_fc=False` to
  match upstream; the existing `resnet18_cifar10` entry keeps identity
  initialization for the reason its docstring gives (numerical equality with
  stock ResNet-18 at step zero).
* Upstream's student starts from ImageNet weights; the benchmark's
  `resnet18_cifar10` starts from scratch. The KD entry uses ImageNet
  weights to match the reported table.
