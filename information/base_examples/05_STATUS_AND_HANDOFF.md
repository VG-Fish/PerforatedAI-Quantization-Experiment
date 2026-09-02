# Status and handoff — PerforatedAI base-example port

Written 2026-09-02. Read `01_UPSTREAM_AUDIT.md`, `02_OPEN_DECISIONS.md`,
`03_IMPLEMENTATION_RECORD.md` and `04_DIAGNOSIS_pai_final_artifact.md` first;
this file is the current state and the remaining work, not a repeat of them.

## The task

Port five PerforatedAI examples with published dendritic gains into this
quantization benchmark, then run the sweep:

`transfer_learning` (CIFAR-100 only), `base_examples/mnist`,
`base_examples/pytorch_unet`, `base_examples/segmentation-image-resolution`
(full resolution only), `base_examples/resnet` (KD + perforation).

Source pinned at `PerforatedAI/PerforatedAI@0a5967b`.

## Done

Implementation is complete for all five. Registry is 29 models. `ty check`
clean, `dqb docs --check` clean, `pytest` green apart from two failures that
pre-date this work (see below).

Model keys: `mnist_pai`, `resnet18_hf_perforated_cifar100`,
`resnet18_kd_cifar100`, `unet_carvana`, `unet_supervisely`.

### Verified working end to end

* **`mnist_pai`** — `base_fp32` reaches **98.03%** test accuracy in one epoch;
  a 14-epoch dendritic run reached **99.10%** validation. Upstream's example is
  ~98% at one epoch and ~99.1% by epoch 14. `base_q8` and `base_q1` PTQ both
  produce artifacts.
* **`unet_supervisely`** — `base_fp32` two epochs → **mIoU 0.6255** test;
  `base_q8` PTQ → 0.6292. Upstream reports 0.8420 at 80 epochs, so a 0.63 at
  two epochs is on trajectory. Loss, metric, quantization and artifact paths all
  exercised.
* **`resnet18_kd_cifar100`** — builds to 11,482,788 parameters, which
  decomposes *exactly* against upstream's reported 11,490,981 (100-way vs
  101-way head, 3x3 CIFAR stem vs 7x7 ImageNet stem). Forward verified.
* PAI parameter coverage verified empty-uncovered for all four ID-selected
  models — a parameter in neither list gets no `parameter_type` and this repo
  suppresses the warning, so the failure mode is a silently mistyped run.

### A measured performance fix worth keeping

The Supervisely loader was data-bound, not compute-bound: **2.37 s/batch** at
`num_workers=0` against **0.47 s/batch** at 4 (8 was *worse* — OpenCV
oversubscription). Added `data._cv2()`, which disables OpenCV's internal thread
pool once per process, and set `num_workers=4` on both segmentation loaders.
End-to-end the bundle now yields **0.317 s/batch** — 7.5x. That is the
difference between a ~7h and a ~1.5h training arm, times four trained arms.

## Blocked / in flight

1. **`unet_carvana` has no data.** The Kaggle competition endpoint returns
   HTTP 401: the account has not accepted the rules. The key itself is valid
   (`datasets/list` works). **Action for the user:** accept at
   https://www.kaggle.com/c/carvana-image-masking-challenge/rules, then
   `dqb download_data --models unet_carvana`. `_build_carvana` raises with this
   URL rather than silently substituting the PNG-re-encoded mirror, which would
   change the pixels a Dice score is measured on.

2. **CIFAR-100 download is slow** (~70 kB/s from the torchvision mirror,
   ~35 min for 169 MB). Gates `resnet18_hf_perforated_cifar100` and
   `resnet18_kd_cifar100`. No action needed beyond waiting.

3. **A repo-wide dendritic regression is being fixed.** See
   `04_DIAGNOSIS_pai_final_artifact.md`. Commit `9de8880` made
   `_export_final_pai_artifact` raise when PerforatedAI has not written
   `final_clean_pai.pt` — a file PAI writes only at its own `TRAINING_COMPLETE`
   transition, which a fixed-epoch-budget run (the documented default, with the
   final 20% of epochs frozen) structurally never reaches. **This breaks every
   default-mode `dendrites_fp32` run in the repository, not just the new
   models** — reproduced on `lenet5`. A second latent defect in the same commit
   turns `model.pt` into safetensors while `pipeline._load_state` still uses
   `torch.load`, which would break every `dendrites_q*` condition. A fix agent
   is implementing the four-item remedy the diagnosis specifies.

## Results so far (2026-09-02 07:20)

Results root `experiment_results/base_examples_launch1/`, `--seed 0`.

### `resnet18_hf_perforated_cifar100` — COMPLETE

Its matrix is the six `base_*` conditions only (D3, D14).

| condition | test acc |
| --- | ---: |
| base_fp32 | **0.7314** |
| base_q8 | 0.7290 |
| base_q4 | 0.5882 |
| base_q2 | 0.0106 |
| base_q1_58 | 0.0100 |
| base_q1 | 0.0086 |

q8 is nearly free (-0.24pp); q4 costs 14.3pp; q2 and below collapse to chance
(1% on 100 classes).

### `mnist_pai` — base complete, dendritic running

Parameter count 1,199,882, byte-identical to row 0 of upstream's own
`best_arch_scores.csv`.

| condition | test acc |
| --- | ---: |
| base_fp32 | 0.9901 |
| base_q8 | 0.9901 |
| base_q4 | **0.9904** |
| base_q2 | 0.9825 |
| base_q1_58 | 0.9376 |
| base_q1 | 0.4952 |

Lossless to q4, -0.8pp at q2, -5.3pp at q1.58, collapse at q1. `dendrites_fp32`
is running open-ended and **has switched** — `switch_epochs.csv` records
switches at epochs 3 and 47, so a dendrite was retained and the `dendrites_q*`
family will not be refused by `_require_verified_dendritic_pqat_source`.

### `unet_supervisely` — base complete, dendritic running

| condition | test mIoU |
| --- | ---: |
| base_fp32 | **0.8612** |
| base_q8 | 0.8624 |
| base_q4 | 0.3890 |
| base_q2 | 0.3890 |
| base_q1_58 | 0.3890 |
| base_q1 | 0.3890 |

**FP32 beats upstream's reported full-resolution baseline of 0.8420.** q8 is
free. **Flag for follow-up:** q4 through q1 all return *exactly* 0.3890. An
identical value across four different bit widths is the signature of a
degenerate all-background prediction, whose mIoU is a dataset constant, not of
four independent measurements. Physically plausible for segmentation under
aggressive quantization, but it should be confirmed rather than assumed --
check whether the argmax is constant before reporting these as four points.

`dendrites_fp32` is at epoch 26; its first fixed switch is at epoch 80 (D13).

### `resnet18_kd_cifar100` — running

Teacher finished at **82.26% CIFAR-100 val accuracy** (30 epochs, CIFAR-adapted
stem per D12), cached at `data/kd_teachers/resnet50_cifar100_teacher.pt`.

Distillation verified genuinely active, not silently skipped: the teacher emits
`(4, 100)` logits, the loss moves 5.5648 (CE only) -> 3.7407 (with the KD term)
at `alpha=0.4, T=4.0`, and a non-KD model correctly gets `None`. `base_fp32` is
training at ~5.5 batch/s over 391 batches/epoch for 90 epochs.

### `unet_carvana` — still blocked

Kaggle competition rules not accepted; `competitions/data/list` returns 401.

## Runs launched (2026-09-02, ~01:38)

Results root: `experiment_results/base_examples_launch1/`. One `dqb run` per
model, because `--pai-override` is only honoured when exactly one model is
selected. All use `--seed 0` and `--jobs 1`.

Progress below is as of the last poll; the per-model tables are appended as
conditions complete.

### `mnist_pai`

```
dqb run --models mnist_pai --dynamic-dendritic-training \
  --pai-override .runs/pai_mnist.json \
  --results-root experiment_results/base_examples_launch1/mnist_pai ...
```

All six base conditions complete; `dendrites_fp32` running open-ended.
**Parameter count 1,199,882 — byte-identical to row 0 of upstream's own
`best_arch_scores.csv`.**

| condition | test acc | best val |
| --- | ---: | ---: |
| base_fp32 | 0.9901 | 0.9892 |
| base_q8 | 0.9901 | 0.9901 |
| base_q4 | 0.9904 | 0.9904 |
| base_q2 | 0.9825 | 0.9825 |
| base_q1_58 | 0.9376 | 0.9376 |
| base_q1 | 0.4952 | 0.4952 |

Lossless to q4, -0.8pp at q2, -5.3pp at q1.58, collapse at q1. Upstream's
published FP32 is 99.16 with one dendrite lifting it to 99.33.

### `resnet18_hf_perforated_cifar100`

No `--pai-override` and no `--dynamic-dendritic-training`: this checkpoint is
already perforated, so it has no `dendrites_*` conditions at all. 112 s/epoch x
200 epochs, so `base_fp32` alone is ~6.2 h.

**Watch this one.** Upstream's own CIFAR-100 config says
`'use_pretrained': False,  # Train from scratch` while `load_model_from_hf`
loads the HF checkpoint unconditionally — so `lr=0.1` is a from-scratch rate
applied to a pretrained backbone, which is why epoch 1 is near chance. That is
faithfully what upstream's code does, but if it has not passed ~50% by epoch 40
it is worth revisiting. Trajectory so far: 0.0258 (ep 1) -> 0.2276 (ep 6).

Its twelve control conditions are structurally impossible (no dendritic arm to
fork from) and are now recorded as unavailable rather than fatal — see
`06_DIAGNOSIS_control_conditions_abort_the_sweep.md`.

### `unet_supervisely`

```
dqb run --models unet_supervisely --dynamic-dendritic-training \
  --pai-override .runs/pai_unet_supervisely.json --pai-fixed-switch-interval 80 ...
```

Launched 01:53. `base_fp32` 80 epochs at ~1.2 s/batch x 134 batches under
four-way MPS contention (~2.7 min/epoch, ~3.6 h). The 0.317 s/batch measured
for the loader fix was on an otherwise idle machine; the gap is contention, not
a regression.

### `pretrain_kd_teacher`

ResNet-50, CIFAR-adapted stem (D12), 30 epochs, ~2.5 min/epoch.
**81.82% val at epoch 15**, checkpointing to
`data/kd_teachers/resnet50_cifar100_teacher.pt`. `resnet18_kd_cifar100` cannot
start until this finishes.

## A second repo-wide defect, found and fixed on 2026-09-02

`06_DIAGNOSIS_control_conditions_abort_the_sweep.md` and `02_OPEN_DECISIONS.md`
D14. In one sentence: the default sweep is 24 conditions, twelve of them are the
two controls, and a control that is *structurally impossible* for a model raised
out of the condition loop and killed the whole run — plus, had the crash simply
been removed, the quantized descendants of the missing control would have
quantized a **randomly initialised** model and published the number. Both halves
are fixed; `./scripts/ci.sh` is green apart from the two pre-existing failures.

The three in-flight runs hold the pre-fix module in memory and will still abort
at their control condition. They were deliberately not restarted: staleness is
keyed off `model_revision` / `dataset_revision` / recipe / quantization
revisions, none of which this fix touches, so relaunching the same command
afterwards resumes from the saved `record.json`s and picks up only what is left.

## Remaining work, in order

1. When `pretrain_kd_teacher` finishes: smoke `resnet18_kd_cifar100`
   (`--conditions base_fp32 base_q8 --recipe-override .smoke/override2.json`)
   to confirm the KD loss path runs, then launch its full sweep with
   `--dynamic-dendritic-training --pai-override .runs/pai_resnet_kd.json`.
2. Relaunch each of the three in-flight runs once it aborts at its control
   condition, so the post-fix code records the skips instead.
3. `unet_carvana` stays blocked on Kaggle rules acceptance
   (`competitions/data/list/carvana-image-masking-challenge` still 401 at
   02:00). Only ~15 GB free on this disk: check `df -h .` before downloading,
   and delete the zips the moment they are extracted.
4. Replace the guessed `_MODEL_COST_HOURS` entries with measurements.

## Two pre-existing test failures — not from this work

Verified by stashing every change and re-running against clean `HEAD`:

* `test_dynamic12_hf_pqat.py::test_post_run_verifier_rejects_missing_pqat_stage`
  imports `experiments/dynamic12/scripts/verify_pqat.py`; there is no
  `experiments/` directory in the repo at all.
* `test_p2_docs.py::test_superseded_documents_name_their_replacement` reads
  `information/DYNAMIC9_PAI_GRAPH_AUDIT.md`, which does not exist.

Both are dangling references to deleted files. Worth cleaning up, but out of
scope here and deliberately left alone.

## Methodology note

The first dendritic smoke ran at `max_epochs=1` and failed. Before opening a bug
I ran the *existing* `lenet5` through the identical override and got the
identical error — which is what turned "my new model is broken" into "the
repository's dendritic path is broken", and pointed the diagnosis at the right
commit. Check the control before you open the bug.
