# Open decisions on the base-example port, and the answer taken

Every entry is a question I would otherwise have put to the user. Each states
the options, argues them, and records the choice actually implemented. A choice
here is a *declared departure* from upstream unless it says otherwise: the point
is that no substitution is silent.

Status legend: **[USER]** answered directly by the user; **[TAKEN]** decided
here under the standing instruction to pick the best option and move on.

---

## D1. Dataset for the ResNet KD entry — **[USER]**

**Question.** The resnet example's reported table is Food-101 @224px with a 25%
label subset and a separately fine-tuned ResNet-50 teacher. The same script also
supports CIFAR-100 @32px natively.

**Argument for Food-101.** It is the configuration behind the published numbers
(74.92 baseline / 77.38 perforated / 77.35 KD / 78.60 KD+perforated). Any other
dataset means the benchmark cannot claim to reproduce that table.

**Argument for CIFAR-100.** Cost. 18,938 train images at 224px, ~90 epochs, on
MPS, for each of ~4 fully-trained arms, plus a ResNet-50 teacher pretrain that is
roughly 3x the per-epoch cost of the student. Rough estimate ~30-40 h before any
quantization condition runs, against ~2-3 h for CIFAR-100. It also needs ~10 GB
of the 21 GB free on this disk. CIFAR-100 additionally lines the KD entry up with
the transfer-learning entry (D3), making KD-vs-no-KD and pretrained-vs-scratch
comparable across two ResNet arms of the same dataset.

**Answer: CIFAR-100 @32px.** The experiment's question is whether perforation,
KD and quantization stack, which needs the arms to be mutually consistent, not to
match an absolute published number.

## D2. Carvana data access — **[USER]**

**Question.** Verified: the configured Kaggle key works for `datasets/list` but
`competitions/data/list/carvana-image-masking-challenge` returns HTTP 401 — the
account has not accepted the competition rules. A mirror
(`ipythonx/carvana-image-masking-png`, 868 MB) downloads with no acceptance.

**Argument for the mirror.** Available immediately, no user action.

**Argument for the official data.** The mirror is a PNG re-encode, not the
original `train_hq` JPEGs the example's `dir_img` points at. Pixel data differs,
which contaminates a segmentation Dice number against a published one.

**Answer: user accepts the rules; download the official competition data.**
Until acceptance lands, `unet_carvana` stays registered but its dataset builder
raises with the acceptance URL rather than silently falling back to the mirror.

## D3. Which HF checkpoint for the CIFAR-100 transfer entry — **[TAKEN]**

**Question.** The transfer_learning README shows
`perforated-ai/resnet-18-perforated-cascor` and
`perforated-ai/resnet-18-perforated`. The benchmark already pins
`perforated-ai/resnet-18-perforated-gd` with a SHA-256 and a working
lower-level loader, plus a recorded bug: the model card's high-level
`UPA.from_hf_pretrained` double-converts this checkpoint under PerforatedAI
3.2.6 and then fails strict state loading.

**Argument for `-cascor`.** Literal fidelity to the README command.

**Argument for `-gd`.** It is the checkpoint this repo has already verified end
to end, checksum-pinned, and whose loader bug is already worked around in
`models._hf_perforated_resnet18_checkpoint`. Introducing a second, unverified
repo risks re-hitting exactly the double-conversion failure that is already
documented, and would make the CIFAR-10 and CIFAR-100 transfer entries
non-comparable — they would differ in *both* dataset and backbone.

**Answer: reuse `-gd`.** Recorded as a departure. The CIFAR-100 entry then
differs from the existing CIFAR-10 entry in exactly one variable, the dataset,
which is the more useful comparison. If the `-cascor` weights are wanted later
they are a one-line factory change plus a new checksum.

## D4. Mixup / CutMix on the KD entry — **[TAKEN]**

**Question.** The resnet README's commands pass `--mixup-alpha 0.2 --cutmix-alpha
0.6`, applied through `collate_fn` and therefore dataset-independent — they would
apply to CIFAR-100 too. (`--auto-augment` and `--random-erase` would *not*: the
script's CIFAR-100 branch in `load_data` builds its own transform list and
ignores them.)

**Argument for including them.** Closer to the reported recipe.

**Argument against.** Mixup/CutMix replace integer class targets with mixed
soft targets. That changes the target contract for the *entire* metric harness:
`_classification_metrics`, `_compute_all_metrics`, the accumulator, the anomaly
and regression branches' shared helpers, and the accuracy alias all assume
`targets` are class indices. Supporting soft targets is a broad, risky change to
code shared by 24 working models, for an augmentation that is not part of the
question being asked. The upstream CIFAR-100 path already declines two of the
four README augmentations, so a partial-augmentation CIFAR-100 recipe is not a
novel configuration.

**Answer: omit mixup and cutmix; keep `label_smoothing 0.1`, `dropout 0.2` and
`--wd 0.001`.** Declared departure. Identical across every arm, so it cannot
bias the perforation/KD/quantization contrast.

## D5. Where the KD teacher lives — **[TAKEN]**

**Question.** The benchmark pipeline has no concept of a pretraining stage, and
KD needs a ResNet-50 fine-tuned on CIFAR-100 first.

**Options.** (a) A pipeline stage. (b) A one-time out-of-band script producing a
checksummed checkpoint under `data/`, loaded at training time. (c) Skip
fine-tuning and distil from a bare ImageNet ResNet-50.

**Answer: (b).** (c) is not the upstream method and a 1000-way ImageNet head
distils nothing useful into a 100-way CIFAR student. (a) would put a
model-specific, one-off stage into the generic model x condition matrix that all
24 other models pay for. (b) matches the precedent already set by the HF
checkpoint download: an external, checksum-pinned artifact resolved by the model
factory. The teacher is held in a module-level cache in `training.py`, never as
a child module, so PAI, quantization and `state_dict` never see it.

**Corollary taken:** KD applies to the training loss only, exactly as upstream's
`train_one_epoch` does it — validation and test stay plain cross-entropy. This
falls out of passing teacher logits from `_run_training_batch` alone.

## D6. Carvana image set and batch size — **[TAKEN]**

**Question.** Upstream uses `dir_img = ./data/train_hq` at `--scale 0.5` with
`--batch-size 1` and RMSprop `lr=1e-5`.

**Argument for batch 1.** Literal fidelity; the LR is tuned to it.

**Argument against.** 5,088 images at batch 1 is 5,088 optimizer steps per
epoch. At MPS throughput that is a multi-hour epoch, and the benchmark runs
~24 conditions per model.

**Answer: keep `--scale 0.5` and RMSprop, raise batch to 4 with the LR scaled by
the existing `ModelTrainingRecipe.with_batch_size` per-sample rule.** The repo
already treats batch size as an MPS-tuning knob with an explicit LR-scaling
contract (`_BATCH_SIZES`, `with_batch_size`), so this uses machinery that exists
rather than inventing a departure. Recorded.

## D7. Third split for the two segmentation examples — **[TAKEN]**

**Question.** `pytorch_unet` splits train/val only (`--validation 10.0`).
`segmentation-image-resolution` ships fixed train/valid pair files. The benchmark
requires train/val/test, because PAI reads validation to decide switches and the
reported number must come from data PAI never saw.

**Answer.** Carvana: split 80/10/10 from the single labelled set with the
repo's seeded `_split_dataset`, so val stays the 10% upstream used. Supervisely:
keep upstream's train pairs as train, and split its *valid* pairs 50/50 into val
and test with a fixed seed — the same stratified-halving idea the resnet example
uses on its own eval set (`split_eval_dataset_stratified`), so it is upstream's
own convention rather than a new one.

## D8. Metric direction for the Supervisely entry — **[TAKEN]**

**Question.** Upstream monitors `valid_loss` (mode `min`) and calls
`perforate_model(..., maximizing_score=False)`, but the README reports and
compares **validation mIoU** (higher better).

**Answer: report mIoU, direction `maximize`.** The README's comparison table is
mIoU, and this repo has a recorded failure mode
(`pai-zero-seeded-ema-best-tracking`) where non-positive-maximize metrics
produced epoch-1 restores and corrupted dendritic numbers — a maximized,
strictly-positive metric is the safer contract. Declared departure from
upstream's `maximizing_score=False`.

## D9. Epoch budget vs upstream's "run until PAI says complete" — **[TAKEN]**

**Question.** The mnist and pytorch_unet examples set `--epochs 10000` and stop
on `training_complete`.

**Answer: fixed `max_epochs` per recipe, as every other model in this benchmark
does.** The runner already has `train_dendrites_until_complete` and a dynamic
over-budget path for the dendritic arm; an unbounded budget would make the dense
control and the dendritic arm non-comparable on training length, which is exactly
what the `base_more_training` control exists to rule out.

## D10. Should the five new models be the *only* default roster — **[TAKEN]**

**Question.** The instruction was "ignore all the models in the project and focus
on quantizing these". `DEFAULT_MODEL_KEYS` currently selects 5 evidence-backed
models for a bare `dqb run`.

**Argument for replacing.** Literal reading of "ignore all the models".

**Argument for adding.** Removing `default_enabled` from the existing five
changes what a bare `dqb run` means for every stored result and every doc that
describes the roster, and deletes information the user did not ask to delete.
The five new models can be selected explicitly and are also default-enabled, so
"focus on these" is satisfied without destroying the old default.

**Answer: mark the five new models `default_enabled=True` and leave the existing
flags alone.** Runs for this task are launched with the five keys named
explicitly, so the wider default never comes into play in practice.

## D11. Adadelta under PAI's optimizer re-setup — **[TAKEN]**

**Question.** The mnist example uses `Adadelta(lr=1.0)` + `StepLR(1, 0.7)`.
PAI's `setup_optimizer` re-creates the optimizer on every restructure, and this
repo funnels that through `_optimizer_class` / `_optimizer_args`, which knew only
adam/adamw/sgd.

**Answer: add `adadelta` to `OptimizerName` and to both builders.** Substituting
Adam at lr=1.0 would diverge, and Adam at 1e-3 would not be the upstream recipe.
`config.momentum` is deliberately unread for Adadelta; noted in the code so it
does not read as an omission.

## D12. CIFAR stem on the KD *teacher* — **[TAKEN]**

**Question.** Upstream's `pretrain_teacher` builds the teacher as
`resnet50(weights=IMAGENET1K_V2)` with only `fc` replaced — no stem change,
because its reported runs are Food-101 at 224px. D1 moved us to CIFAR-100 at
32px. Does the teacher get the same CIFAR stem adaptation the student got?

**Argument against.** Upstream does not touch the teacher's stem, and every
departure widens the gap from the published setup.

**Argument for.** Measured on this machine, an unmodified ResNet-50 fed 32px
input produces feature maps of 8 -> 8 -> 4 -> 2 -> **1x1** across `layer1..4`.
The last two stages run on 2x2 and 1x1 maps and the pooling the architecture is
designed around never happens. This repository already has that exact failure
on record for `mobilenetv2_cifar10`, where the same 32x-downsampling mistake
cost ~2.5 accuracy points against published CIFAR-10 baselines.

It matters more for a teacher than for an ordinary classifier. The teacher's
only job in this experiment is to emit soft targets; a teacher hobbled by a
resolution mismatch would make the KD arm measure the mismatch rather than
distillation, which is precisely the thing the experiment is being asked to
report on. It would also be internally inconsistent: the student already
carries the CIFAR stem (`_build_resnet18_kd_cifar100`), so leaving the teacher
un-adapted would compare a correctly-resolved student against a crippled
teacher.

**Answer: adapt the teacher's stem identically** — centre-crop the learned 7x7
kernel to 3x3, stride 1, drop the max-pool. Verified: `layer4` now yields 4x4.
Declared departure, and one that follows from D1 rather than standing alone —
if the dataset were Food-101, this decision would reverse.

## D13. Making dendrites actually get retained — **[TAKEN]**

**Question.** In the benchmark's default mode a dendritic run gets the same
fixed epoch budget as its dense control and freezes dendrite insertion for the
final 20% of epochs, with PAI's library defaults (`DOING_HISTORY`,
`n_epochs_to_switch=10`, `max_dendrites=100`). Measured: `mnist_pai` ran 14
epochs with validation still climbing and **never switched**. And a
`dendrites_fp32` source with no retained dendrite is refused by
`_require_verified_dendritic_pqat_source` (`pipeline.py:1905-1929`), so every
`dendrites_q*` condition is skipped. That is the entire "perforation +
quantization" half of what this task exists to measure.

**Argument for leaving the default alone.** It is the documented default, and
the dense/dendritic arms stay matched on training length -- which is what the
`base_more_training` control exists to check.

**Argument for open-ended dynamic training.** Three things point the same way:

1. It is what **upstream does**. Every one of these five examples runs until PAI
   reports completion -- mnist with `--epochs 10000`, pytorch_unet with
   `--epochs 500000`, the others looping on `training_complete`.
2. It is what **this repository does**. Of the stored dendritic artifacts, every
   single one that completed ran open-ended and overshot its recipe budget
   (60 -> 140, 40 -> 124, 60 -> 101). The one that did not, `distilbert`, has
   zero switches and no `dendrites_q*` results at all.
3. Without it there is **no result to report** for half the conditions.

**The cost** is an unbounded run: PAI's default `max_dendrites` is 100.

**Answer: `--dynamic-dendritic-training`, bounded per model by a
`--pai-override` carrying that example's own upstream PAI configuration.**
Faithful *and* terminating, because upstream itself caps the search:

| Model | `max_dendrites` | other | upstream source |
| --- | ---: | --- | --- |
| `mnist_pai` | 1 | library defaults | `best_arch_scores.csv` has exactly two rows, 1,199,882 -> 2,399,764 params: the published architecture is one dendrite |
| `resnet18_kd_cifar100` | 5 | `n=p=40`, `improvement_threshold=[0.001,0.0001,0]`, `candidate_weight_initialization_multiplier=0.1` | `train_perforated_resnet_KD.main`, dendrite_mode 2 |
| `unet_carvana` | 2 | `n=p=25` | `set_max_dendrites(2)`, `set_n/p_epochs_to_switch(25)` |
| `unet_supervisely` | 2 | `--pai-fixed-switch-interval 80` | `DOING_FIXED_SWITCH`, `fixed_switch_num=80`, `first_fixed_switch_num=80` |

`resnet18_hf_perforated_cifar100` is not in the table: its checkpoint arrives
already perforated, so it has no `dendrites_*` conditions at all (D3, and
`PRE_PERFORATED_MODEL_KEYS`).

Two notes on fidelity. First, `mnist_pai`'s cap of 1 is read off upstream's
*result*, not its config -- the example sets no `max_dendrites` and would search
to 100. Capping at the published architecture is the bounded reading of "what
upstream got". Second, `unet_supervisely` uses fixed switching where this
repository's stated policy is that "HISTORY is the scientific default for every
model"; upstream explicitly chose `DOING_FIXED_SWITCH` for this example, and
following the example wins here.

`initial_history_after_switches` is deliberately **not** set for the KD model
even though upstream sets it to 2: `PAIOverride.__post_init__` requires it to
move together with `history_lookback`, to avoid the zero-seeded EMA bug this
project already has on record. Changing the lookback is a larger departure than
leaving both at their defaults.

## D14. What a sweep should do when a control condition is impossible — **[TAKEN]**

**Question.** The default `dqb run` selects all 24 conditions, twelve of which
are the `base_more_training_*` / `capacity_dense_*` controls. Three situations
make a control genuinely impossible: a pre-perforated checkpoint has no
`dendrites_fp32` for the controls to fork from; `capacity_dense_*` supports
only Linear retained branches, so no conv-perforated model can ever have it;
and `dendrites_q*` is refused after a `no_retained_insertion` FP32 arm. Before
this decision, each of the three raised out of one condition and killed the
whole multi-hour sweep. Full evidence in
`06_DIAGNOSIS_control_conditions_abort_the_sweep.md`.

**Argument for leaving it fatal.** A loud crash cannot be ignored, and it was —
accidentally — the only thing preventing a much worse outcome: with the source
missing, `_prepare_condition_model` falls through to the freshly *initialised*
model, so `capacity_dense_q8` would have quantized random weights and published
the number as a real result.

**Argument for recording and continuing.** The impossibility is a property of
the architecture, known before the run starts, and permanent — no amount of
retrying changes it. `capacity_control.py`'s own module docstring already
states the intended contract: "callers must record that status rather than
widen a model as a substitute control". The caller simply never recorded it.
And the repository already applies exactly this reasoning one step later, in
`_write_final_reports`: "training results are already on disk, and a failed
report build must not make the run look like it lost them."

**Answer: record and continue, and refuse the descendants.** A condition whose
declared `source_key` produced no artifact is skipped rather than trained, which
closes the fabricated-result path above; `UnsupportedTopology` and the new
`ConditionPrerequisiteUnmet` are caught only around condition *preparation*, so
a genuine training failure still stops the run; and every skip is written to
`<results>/<model>/unavailable_conditions.json` with its root cause, because a
skipped condition leaves no `record.json` and otherwise "impossible for this
architecture" and "nobody ran it" are indistinguishable in the result tree.

Neither audit gate was weakened. `_require_verified_dendritic_pqat_source` keeps
its exact condition and message; only its blast radius changed from "abort the
sweep" to "skip five conditions". The consequence for the reported table is that
`capacity_dense_*` exists for exactly one of the five new models
(`resnet18_kd_cifar100`, whose sole perforated module is the `.pre_fc` Linear);
`base_more_training_*` — the control D9 actually depends on — is unaffected for
the four dendritic models.

## D14. Control conditions for a pre-perforated model — **[TAKEN]**

**Found the hard way.** `resnet18_hf_perforated_cifar100` trained all six base
conditions (~7h) and then died in `_prepare_control_model` with
`UnsupportedTopology: capacity controls require dendrites_fp32`.

**Why.** `PRE_PERFORATED_MODEL_KEYS` excluded only `dendrites_*`. But both
*control* families are defined relative to a `dendrites_fp32` run:
`base_more_training` resumes that run's saved pre-candidate fork for the same
extra epochs, and `capacity_dense` widens the dense model to match the topology
that run ended at. With no `dendrites_fp32` in the matrix there is no fork to
resume and no topology to match.

**Answer: exclude both control families for pre-perforated models too**, in
`condition_supported_by_model`. A pre-perforated model's matrix is exactly the
six `base_*` conditions.

The refusal was already correct — `capacity_control` raised rather than
inventing a control — it just happened at *train* time instead of *plan* time.
Moving it to the planner is the whole fix. Regression test added:
`test_pre_perforated_models_keep_exactly_the_six_base_conditions`.

This latent bug also applied to the pre-existing `resnet18_hf_perforated_cifar10`,
which had simply never been run with its control conditions selected.

## D15. `mnist_pai`'s dendrite could not train — **[TAKEN]**

**Found by a failed run.** `mnist_pai / dendrites_q8` was refused:
`requires a verified retained dendrites_fp32 source; found no_retained_insertion`.

**But the dendrite was inserted.** PAI's own `param_counts.csv` for that run
reads `0,1199882 / 1,1199882 / 2,2399764` — the second row is exactly
upstream's published one-dendrite architecture, to the parameter. The dendritic
arm also *scored* better than the dense one (test 0.9916 against 0.9901, best
val 0.9912 against 0.9892).

**Why it was not retained.** The recipe is upstream's `Adadelta(lr=1.0)` +
`StepLR(step_size=1, gamma=0.7)`. That schedule gives:

| epoch | 9 | 20 | 47 | 120 |
| --- | ---: | ---: | ---: | ---: |
| lr | 4.0e-2 | 8.0e-4 | **5.2e-8** | 2.6e-19 |

The open-ended run inserted its candidate at **epoch 47**, where the learning
rate is 5.2e-8. A freshly initialized dendrite at that rate cannot move.
Validation therefore never beat the pre-dendrite best at epoch 9, PAI's
completion step restored *that* model, and the audit — correctly — found no
dendrite in the final topology.

**Answer: `dendrite_lr_min_factor=0.1`**, the same remedy this repository
already applies to `resnet18_cifar10` and `saint_adult` for the same cause. It
floors the learning rate for *retained dendrite parameters only*; the backbone
keeps the identical schedule its `base_fp32` control runs, so the paired
comparison is untouched.

Considered and rejected: lengthening the budget (the run already went 120
epochs, 72 of them post-insertion, at a rate below 1e-8 — more epochs at zero
is still zero); and changing the base schedule (that would desynchronise the
dendritic arm from its dense control, which is the one thing the
`base_more_training` control exists to prevent).

This is more acute here than for the two models that already carry the fix: a
step decay of 0.7/epoch falls roughly 1000x faster than their cosines.
