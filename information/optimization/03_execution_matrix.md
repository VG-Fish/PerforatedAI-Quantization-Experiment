# Executable hyperparameter matrix — initial five

This file replaces ambiguity with a bounded runbook. It tells the next agent what to change, where the value currently lives, which values to try, how many runs that creates, and when to stop. It does **not** authorize implementation or training by itself.

## Non-negotiable implementation facts

| fact | current location | consequence |
|---|---|---|
| base recipes are hard-coded | `BenchmarkRunner._training_hyperparameters()` in `pipeline.py` | do not hand-edit a selected recipe between runs; add a versioned override/config mechanism before a sweep |
| only global `--model-scale` exists | CLI / `ExperimentPlan.model_scale` | it changes MPNN width, but does not change ResNet, M5, DistilBERT, or SAINT architecture |
| dynamic PAI defaults | `compat.py::PAI_DYNAMIC_SCHEDULE_DEFAULTS` | `max_dendrites=3`, `n_epochs_to_switch=10`, `history_lookback=8`, `initial_history_after_switches=8`, `p_epochs_to_switch=2`, candidate init `.005` |
| PAI p-phase ceiling | `training.py::MAX_DENDRITE_PHASE_EPOCHS` | hard-capped at 8 epochs; changing `p_epochs_to_switch` above 8 cannot extend a candidate phase |
| PAI targets | `pipeline.py::_perforation_module_ids_to_perforate` | an alternate target requires matching track-only coverage and a structural smoke test, not merely an ID edit |
| current PTQ/PQAT | `quantization.py` | custom, parameter-only projection; it is not activation-observer/calibration PTQ |
| PQAT budget | `pipeline.py::_pqat_epoch_budget` | `ceil(base_epochs × .30)`, clamped to 1–10: ResNet/M5/SAINT/MPNN 10, DistilBERT 1 |

Before any sweep, implement a small immutable `RecipeOverride` and `PAIOverride` object, put both in artifact identity/metadata, and expose their JSON path via CLI. Do not use repeated ad-hoc changes to `pipeline.py`; they make artifacts irreproducible. The artifact identity must include every value in the tables below, the module target list, and the source commit.

## Shared selection and stopping rules

### Base sweep

- A **pilot** uses seed `0`, the stated pilot epoch budget, and validation only.
- Eliminate a trial if loss is non-finite, it fails a structural/shape smoke test, or its final-five-epoch validation mean is worse than the leader by more than 2 percentage points (classification) or 10% (RMSE).
- Promote the two remaining trials to the listed full budget using seed `0`. Select highest validation metric (or lowest RMSE); a tie inside 0.25 points / 1% RMSE selects the lower learning rate, then lower parameter count.
- Lock one recipe, then run seeds `0,1,2`. Test is read only after the recipe is locked.

### PAI sweep

- Begin from the locked base recipe. Do **not** retune base LR/weight decay in the PAI sweep; only PAI settings and its dendrite-only LR floor can change.
- Every PAI pilot must use an isolated `pai_save_name` and log: effective schedule, first candidate epoch, retain/reject epoch, completion reason, raw/final parameter count, and final-clean topology hash.
- Reject as *inconclusive* (not negative) if no candidate is inserted, no dendrite is retained, the completion flag is not reached, or a candidate is still improving when the eight-epoch guard ends.
- Promote the configuration with the best validation result that also passes topology/completion gates. A validation improvement below its predeclared noise floor is not a win; it may still be quantized as a feasibility case.
- The chosen PAI recipe is rerun on seeds `0,1,2`, followed by matched dense continuation and capacity-matched dense controls.

## Exact base hyperparameter trials

The notation below is `(learning_rate, weight_decay, other change)`. All fields not named retain the **start recipe** exactly. Each model therefore has a finite first sweep rather than a vague range.

| model | start recipe, exact | pilot | base trials | full budget |
|---|---|---:|---|---:|
| ResNet-18 | SGD+Nesterov, batch 128, 200 ep, momentum .9, cosine, 5 warmup ep, label smoothing .1 | 50 ep | `R0=(.10,5e-4,LS=.1)`, `R1=(.05,5e-4,.1)`, `R2=(.20,5e-4,.1)`, `R3=(.10,2e-4,.1)`, `R4=(.10,5e-4,LS=0)` | 200 ep |
| M5 | Adam, batch 128, 40 ep, step at epoch 20 × .1 | 10 ep | `A0=(1e-2,1e-4)`, `A1=(3e-3,1e-4)`, `A2=(1e-2,1e-5)`, `A3=(3e-3,1e-5)` | 40 ep |
| DistilBERT | AdamW, batch 32, 3 ep, LR 2e-5, WD .01, linear decay, clip 1 | full | `N0=(1e-5,.01)`, `N1=(2e-5,.01)`, `N2=(3e-5,.01)` | 3 ep |
| SAINT | AdamW, batch 256, 200 ep, LR 1e-4, WD 1e-5, cosine min .02, 5 warmup, clip 1 | 50 ep | `T0=(1e-4,1e-5)`, `T1=(3e-4,1e-5)`, `T2=(6e-4,1e-5)`, `T3=(1e-4,1e-6)`, `T4=(3e-4,1e-6)` | 200 ep |
| MPNN | Adam, batch 32, 200 ep, LR 1e-3, WD 1e-5, cosine min .02, clip 5 | 50 ep | `G0=(scale=1.00,width=96,1e-3,1e-5)`, `G1=(.75,72,1e-3,1e-5)`, `G2=(.667,65,1e-3,1e-5)`, `G3=(1.00,96,3e-4,1e-5)`, `G4=(1.00,96,1e-3,1e-4)` | 200 ep |

For MPNN, width is `ceil(96 × model_scale)` in `models.py`, hence `.75 → 72` and `.667 → 65`. Record the actual constructed width in metadata. The selected compact base must be used by both dense and perforated arms.

Do **not** list dropout/depth/head-count trials for these five until the model factory exposes them. M5, SAINT, and DistilBERT currently have no such constructor settings in `BenchmarkRunner._model_kwargs`; pretending they are available would cause the next agent to tune nothing.

## Exact PAI configurations and targets

`H(n,h,i,p,d,c,t)` means HISTORY mode, `n_epochs_to_switch=n`, `history_lookback=h`, `initial_history_after_switches=i`, `p_epochs_to_switch=p`, `max_dendrites=d`, candidate-init multiplier `c`, and improvement thresholds `t`. The initial-history value must equal the lookback when lookback changes, to avoid the known zero-seeded EMA bug.

### ResNet-18

Current targets are exactly `.pre_fc`; current tracked modules are `.conv1`, `.bn1`, `.layer1`–`.layer4`, `.fc`. Preserve both lists.

| ID | exact PAI override | dendrite LR floor | promote only if |
|---|---|---:|---|
| RP0 | `H(10,8,8,10,1,.005,[.005,.002])` | .10 × base LR | current clean reference |
| RP1 | `H(15,12,12,8,1,.005,[.005,.002])` | .10 × base LR | candidate begins after genuine base plateau |
| RP2 | `H(15,12,12,8,1,.001,[.005,.002])` | .10 × base LR | RP1 insertion destabilizes validation |
| RP3 | `H(15,12,12,8,1,.010,[.005,.002])` | .10 × base LR | RP1 candidate is inert but completes cleanly |

Run RP0 then RP1. Run RP2/RP3 only for their stated diagnosis. No residual-block target is in scope in this loop.

### M5

Current code type-selects **all** `Linear`/`Conv1d` modules because it has no M5 ID-specific branch. That is not an acceptable primary experiment because capacity could grow in every temporal layer. Before an M5 PAI run, add a model-specific selection and a test analogous to the SAINT/PointNet coverage tests. Candidate targets must be exact `.conv3`, `.conv4`, `.fc1`; all other parameter-bearing modules, including BatchNorm layers, must be explicitly perforated or tracked.

| ID | target IDs | exact override | dendrite LR floor |
|---|---|---|---:|
| AP0 | `.conv4`, `.fc1` | `H(10,8,8,6,1,.005,[.005,.002])` | .05 |
| AP1 | `.fc1` | `H(10,8,8,6,1,.005,[.005,.002])` | .05 |
| AP2 | `.conv3`, `.conv4`, `.fc1` | `H(15,12,12,8,1,.005,[.005,.002])` | .05 |

Order: AP0, then AP1 only if AP0 adds excessive parameters or becomes unstable, then AP2 only if AP0 retains a dendrite with a credible validation signal.

### DistilBERT

Current targets are exactly `.model.pre_classifier` and `.model.classifier`. The base transformer is track-only through `.model.distilbert`, and `.model.base_model` is excluded from PAI saving to avoid a duplicate pointer. The dendritic batch is 4; `with_batch_size` scales LR from `2e-5` at batch 32 to `2.5e-6` at batch 4. Preserve this scaling for all N trials.

| ID | targets | exact override | correlation batches | floor |
|---|---|---|---:|---:|
| NP0 | pre-classifier + classifier | `H(10,8,8,6,1,.005,[.005,.002])` | 4 | .05 |
| NP1 | classifier only | `H(10,8,8,6,1,.005,[.005,.002])` | 4 | .05 |
| NP2 | pre-classifier + classifier | `H(15,12,12,8,1,.001,[.005,.002])` | 4 | .05 |

NP1 requires a new target/track-only registry branch and coverage test. Do not expand to encoder linear layers on M3 Pro; it has been restricted to the head to avoid MPS memory pressure.

### SAINT

Current target is the complete `.head` sequence. Keep `.feature_embed`, `.column_blocks`, all row-block attention/FFN/norm modules, and `.column_embedding` tracked exactly as current code specifies. Do not target QKV first: row attention is batch-coupled and the current head experiment is the controlled +4,418-parameter/dendrite test.

| ID | targets | exact override | dendrite LR floor |
|---|---|---|---:|
| TP0 | `.head` | `H(10,8,8,10,1,.005,[.005,.002])` | .10 |
| TP1 | `.head` | `H(15,12,12,8,1,.005,[.005,.002])` | .10 |
| TP2 | `.head` | `H(15,12,12,8,1,.001,[.005,.002])` | .10 |

TP0 is the exact current schedule. Use TP1 if its candidate arrives before the dense curve stabilizes; use TP2 only if TP1 insertion spikes validation. A `.column_blocks.1` expansion is deferred until TP0/TP1 passes all controls and requires a new explicit target/coverage test.

### MPNN

The exact current target set is `.readout.0`, `.readout_gate`, `.layers.2.update.hidden_gates`, `.layers.2.update.input_gates`, `.layers.3.update.hidden_gates`, and `.layers.3.update.input_gates`. It is the only launch model with a known candidate-phase cap issue. Present code uses `max_dendrites=3`, `p_epochs_to_switch=2`, and the absolute p-phase ceiling of 8; prior correlation was still increasing at that ceiling.

| ID | targets | exact override | floor | decision |
|---|---|---|---:|---|
| GP0 | current six IDs | `H(15,12,12,8,1,.005,[.005,.002])` | .05 | primary single-dendrite health test |
| GP1 | `.readout.0`, `.readout_gate` | `H(15,12,12,8,1,.005,[.005,.002])` | .05 | use if GP0 growth is too large or unstable |
| GP2 | current six IDs | `H(15,12,12,8,1,.001,[.005,.002])` | .05 | use only if GP0 insertion spikes |

If GP0 ends because `MAX_DENDRITE_PHASE_EPOCHS=8` while PB correlation is still rising, **do not select GP0 or change `p_epochs_to_switch` to 12**. First make the ceiling a versioned `TrainingConfig`/CLI setting, test a 12-epoch candidate phase on one seed, and give its artifact a new training revision. That is a protocol repair, not a hyperparameter tweak.

## Quantization/PQAT configuration matrix

The present implementation has no calibration, activation bit-width, group size, or observer hyperparameter. The complete supported matrix is therefore:

| condition | exact code path | tune now? | required record |
|---|---|---|---|
| Q8 | `symmetric_quantize_tensor(..., 8)`, tensor granularity | no | before/after metric, topology hash |
| Q4 | `symmetric_quantize_tensor(..., 4)`, tensor granularity | no | same |
| Q2 | signed four-level robust 99.9th-percentile scale | no | quantizer revision and scale semantics |
| Q1.58 | scaled ternary, tensor granularity | no | scale semantics; no second projection |
| Q1 | scaled binary | no | scale semantics; no second projection |
| PQAT | full-precision shadow, hard projection before forward and after every step | no bit-specific LR override yet | `before_pqat/` and `after_pqat/` stage metadata |

Use the locked base recipe's optimizer/LR for current PQAT and existing budgets: 10 epochs for ResNet/M5/SAINT/MPNN and one epoch for DistilBERT. Do not call these values optimal. A later PQAT tuning study must add an explicit bit-specific `(LR multiplier, epochs)` field and compare `{0.1,0.3,1.0} × {current,2×current}` on **Q4 then Q2 only**, after the weight-only protocol is validated. It must save each trial under a new quantization revision.

## Required per-run manifest fields beyond current metadata

`recipe_override`, `pai_override`, effective training recipe after batch-size scaling, all target and track-only IDs, first candidate/retention/completion epochs, p-phase ceiling, final-clean topology hash, paired-control identity, and weight-only quantizer revision. The next agent should add these before the first sweep, then regenerate `information/CURRENT_GUIDE.md` and tests.
