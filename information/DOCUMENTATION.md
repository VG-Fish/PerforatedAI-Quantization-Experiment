# Dendritic Quantization Benchmark: Documentation

<!-- status-banner -->
> **Status: historical reference.** Written before the 2026-08-30 audits. The model roster and condition grid here are superseded by the generated [CURRENT_GUIDE.md](CURRENT_GUIDE.md); the experiment rationale, PerforatedAI integration notes, and quantization background are still the fullest account and are cited as such. Where the two disagree, the generated guide is the current state. The module walkthrough and command guide this file used to carry were removed once the CLI's own registry became authoritative — see the closing note.

This document holds the experiment plan and the extended model proposals that shaped the benchmark, plus the PerforatedAI integration walkthrough, the quantization background, and the baseline-quality record.

---

# Part 1: Benchmark Experiment Plan

## Overview
This experiment investigates whether quantized dendritic models (created via Perforated Backpropagation) outperform non-dendritic counterparts across diverse fields. The current runnable suite contains **24 registered models** spanning complexities from ~25K to ~66M parameters. Models support up to **12 experimental conditions**; the published Hugging Face PerforatedAI ResNet is already dendritic, so its redundant `dendrites_*` aliases are intentionally skipped.

The hardware target is an **Apple M3 Pro** chip using PyTorch's MPS backend, with a total budget of 12–48 hours. All quantization uses `torchao` (PyTorch-native), and dendrites are added via the `PerforatedAI` library.

***
## Original 10 Benchmark Models
| # | Model | Domain | Dataset | Complexity | ~Params | Est. Base Hrs |
|---|-------|--------|---------|------------|---------|---------------|
| 1 | **LeNet-5** | Image Classification | MNIST | Tiny | 60K | 0.5h |
| 2 | **M5 (1D-CNN)** | Audio Classification | SpeechCommands | Tiny | 25K | 1h |
| 3 | **LSTM Univariate** | Time-Series Forecasting | ETTh1 | Tiny | 52K | 1h |
| 4 | **TextCNN** | NLP / Text Classification | AG News | Small | 873K | 0.5h |
| 5 | **GCN** | Graph / Node Classification | Cora | Tiny | 92K | 0.5h |
| 6 | **TabNet** | Tabular Classification | Adult Income | Tiny | 39K | 1h |
| 7 | **MPNN** | Drug Discovery / Molecular | ESOL (MoleculeNet) | Small | 353K | 1h |
| 8 | **Actor-Critic** | Reinforcement Learning | CartPole-v1 | Tiny | 18K | 0.3h |
| 9 | **LSTM Autoencoder** | Anomaly Detection (ECG) | MIT-BIH | Tiny | 71K | 1.5h |
| 10 | **DistilBERT (fine-tune)** | NLP / Seq Classification | SST-2 | Large | 66M | 10h |

> Models 1–9 are intentionally lightweight to allow all 12 conditions to complete within the 12–48h budget. Model 10 (DistilBERT) serves as the large-model anchor.

***
## Experimental Conditions (12 per model)
Each model is trained/evaluated in the following 12 conditions. Metrics recorded for every condition: **accuracy (or task-equivalent metric)**, **parameter count**, and **model file size on disk**. The benchmark now isolates only two experimental factors within a model: quantization level and whether the model uses dendrites.

| # | Condition Label | Description |
|---|----------------|-------------|
| 1 | **Base FP32** | Vanilla model, no modifications, full float32 precision |
| 2 | **Base + Q8** | Post-training quantization to 8-bit via `torchao` |
| 3 | **Base + Q4** | Post-training quantization to 4-bit |
| 4 | **Base + Q2** | Post-training quantization to 2-bit |
| 5 | **Base + Q1.58** | Ternary quantization {−1, 0, +1} (BitNet-style) |
| 6 | **Base + Q1** | Binary quantization {−1, +1} |
| 7 | **+Dendrites** | Base model with dendritic compartments via Perforated Backpropagation, FP32 |
| 8 | **+Dendrites + Q8** | Dendritic model post-training quantized to 8-bit |
| 9 | **+Dendrites + Q4** | Dendritic model post-training quantized to 4-bit |
| 10 | **+Dendrites + Q2** | Dendritic model post-training quantized to 2-bit |
| 11 | **+Dendrites + Q1.58** | Dendritic model ternary quantization |
| 12 | **+Dendrites + Q1** | Dendritic model binary quantization |

With `--allow-PQAT`, all quantized conditions run quantization-aware
fine-tuning after an initial PTQ snapshot is saved.

***
## Output Graphs (Per Model)
For each registered model, generate **3 comparison bar charts** — one for each metric — with its supported conditions on the x-axis:

### Graph Set A: Accuracy (or Task Metric)
- Y-axis: Accuracy % (classification), MAE/MSE (regression/forecasting), Action Accuracy (behaviour-cloned RL: `actor_critic`, `dqn_lunarlander`), Episodic Return (on-policy RL: `ppo_bipedalwalker`), AUC (anomaly), ELBO (VAE)
- X-axis: All 12 conditions
- Color coding: Base conditions in blue family, Dendrite conditions in green family

### Graph Set B: Parameter Count
- Y-axis: Number of non-zero parameters (after pruning)
- X-axis: All 12 conditions
- Highlights the structural compression achieved by pruning + quantization

### Graph Set C: Model File Size (MB)
- Y-axis: Saved model size in MB (using `torch.save` or ONNX export)
- X-axis: All 12 conditions
- Shows real storage savings across the quantization spectrum

***
## Cross-Model Comparison Graphs
After all individual runs, produce the following **cross-domain comparison plots**:

### Cross-Graph 1: Accuracy Retention Heatmap (model × condition)
- Rows = models/domains, Columns = conditions
- Cell value = accuracy as % of the Base FP32 baseline (retention ratio)

### Cross-Graph 2: Size Reduction vs. Accuracy Tradeoff (scatter)
- X-axis: File size reduction ratio vs. Base FP32
- Y-axis: Accuracy retention (%)
- One point per available (model × condition) combination

### Cross-Graph 3: "Dendrite Delta" Bar Chart (per domain)
- For each domain: side-by-side bars of `Base FP32` vs `+Dendrites FP32` accuracy

### Cross-Graph 4: Best Quantization Level per Domain (heatmap)
- Rows = domains, Columns = quantization levels (FP32, Q8, Q4, Q2, Q1.58, Q1)
- Cell = best accuracy among Base and Dend+Prune variants at that bit level

***
## Training Plan (M3 Pro, 12–48h Budget)

### Phase 1 — Tiny/Small Models (Models 1–9): ~12–20h total
Run sequentially overnight. All 12 conditions per model. Use `doing_pai=False` in `perforate_model` for all base conditions (conditions 1–6) to skip dendrite overhead entirely.

### Phase 2 — Large Model (DistilBERT, Model 10): ~25–30h total
Run in isolation. For Q1 and Q1.58, use QAT (quantization-aware training) via `torchao` rather than PTQ for better accuracy retention.

### Recommended Execution Order per Model
1. **Base FP32** → train with `doing_pai=False`; save checkpoint
2. **Base + Q8/Q4/Q2/Q1.58/Q1** → load Base FP32 checkpoint; apply PTQ/QAT via `torchao`; evaluate
3. **+Dendrites FP32** → retrain from scratch with `doing_pai=True`; by default use the same fixed epoch budget as Base FP32, with PAI insertion active for the first 80% and frozen for the last 20%
4. **+Dendrites FP32** → use the completed fixed-budget dendritic checkpoint as the source state for all dendritic quantized evaluations
5. **+Dendrites+Q8 through Q1** → load the completed dendritic FP32 checkpoint; apply quantization, or run short PQAT fine-tuning when `--allow-PQAT` is enabled

### PerforatedAI Output Files
The benchmark passes PerforatedAI save names under `PAI/`, so library-created
checkpoints and sidecars stay in the `PAI/` tree. The library writes these
automatically to the `save_name/` folder:
- `best_model` — best checkpoint by validation score
- `final_clean_pai` — inference-optimized checkpoint (when enabled by the library)
- `latest` — most recent checkpoint; use to resume if training crashes
- `best_arch_scores.csv` — best test scores + parameter counts per dendrite cycle
- `paramCounts.csv` — parameter count at each epoch
- `Scores.csv` — validation + extra scores per epoch

The library's active `PAI/PAI_config.json` is also snapshotted after each
perforation as `PAI/<model>_<condition>_PAI_config.json` and, for the run
artifact, as `results/<model>/<condition>/PAI_config.json`.

This benchmark suite itself saves the best model state it evaluated to `results/<model>/<condition>/model.pt` and uses that file for comparisons and file-size reporting. For dendritic FP32 runs, the default mode treats the configured `max_epochs` value as a hard budget matching the base model. Before a new target set is used in production, `--pai-capacity-check --conditions dendrites_fp32` enables PAI's built-in seven-epoch capacity diagnostic; the benchmark leaves PAI live until it reports completion and marks the resulting artifact as diagnostic, so it cannot be reused as a production result. `--dynamic-dendritic-training` restores the open-ended production completion mode; if PerforatedAI needs more epochs to reach `training_complete=True`, those later epochs are saved separately in `results/<model>/<condition>/continued_until_complete/`. If PerforatedAI changes bookkeeping tensor shapes during live restructuring, the restore step reloads only shape-compatible tensors and leaves incompatible tracker metadata at the current model value.

***
## PyTorch Implementation Notes

### PerforatedAI Integration

#### Step 1 — Imports
```python
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA
```

#### Step 2 — Convert the Model
```python
model = YourModel()
model = UPA.perforate_model(
    model,
    doing_pai=True,
    save_name='PAI',
    making_graphs=True,
    maximizing_score=True
)
```

For custom integrations outside this benchmark, non-standard layers can be registered explicitly when their outputs are compatible with PerforatedAI:
```python
GPA.pc.append_modules_to_perforate([nn.MultiheadAttention])
GPA.pc.append_module_names_to_perforate(['encoder_block'])
GPA.pc.append_module_ids_to_perforate(['.layer1.0.conv1'])
```

This benchmark keeps the runtime path on tensor-returning `nn.Linear`,
`nn.Conv1d`, and `nn.Conv2d` modules. Recurrent and attention-style benchmark
models are written with explicit Linear gate/projection layers so those
parameters are still eligible for dendritic insertion without directly wrapping
tuple-returning LSTM/GRU/MultiheadAttention modules. The compatibility wrapper
also forwards the selected runtime device into `GPA.pc.set_device(...)`.

#### Step 3 — Optimizer & Scheduler Setup
```python
GPA.pai_tracker.set_optimizer(torch.optim.Adam)
GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
optimArgs = {'params': model.parameters(), 'lr': learning_rate}
schedArgs = {'mode': 'max', 'patience': 5}
optimizer, PAIscheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
```

#### Step 4 — Validation Loop Hook
The benchmark wires PerforatedAI's tracker into the optimizer and validation
loop for FP32 dendritic training. By default, dendritic FP32 runs use
`range(max_epochs)`: `add_validation_score(...)` is called only during the
first 80% of epochs, PAI switch hyperparameters are shortened for that active
window, and dendrite insertion is frozen for the final 20% of epochs.
`--pai-capacity-check --conditions dendrites_fp32` first exercises PAI's
built-in seven-epoch integration diagnostic. It also uses the open-ended loop,
but its explicitly marked diagnostic artifact must not be used for a production
comparison. `--dynamic-dendritic-training` then switches to the same
open-ended loop for production, keeping `add_validation_score(...)` active
until PerforatedAI reports `training_complete=True`; any epochs after the
canonical budget are isolated under `continued_until_complete/`.

```python
epoch = -1
while True:
    epoch += 1
    # train and validate as normal
    model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
        score, model
    )
    model.to(device)

    if restructured:
        optimizer, PAIscheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
    if training_complete:
        break
```

#### Step 5 — Training Loop Structure
```python
for epoch in range(max_epochs):
    # non-dendritic and PTQ/PQAT runs use a fixed epoch budget

while not training_complete:
    # with --pai-capacity-check or --dynamic-dendritic-training
    # epochs greater than max_epochs are saved in continued_until_complete/
```

#### MPS Device Handling (M3 Pro)
```python
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
GPA.pc.set_device(device)
model = model.to(device)
torch.set_float32_matmul_precision('high')
```

The implementation keeps CUDA-only pinned-memory transfers disabled, reuses
persistent DataLoader workers, uses larger per-model batch sizes to amortize
Python dispatch, and applies `torch.compile(..., backend='aot_eager')` for
non-dendritic MPS models when PyTorch supports it. Dendritic models are not
compiled because PerforatedAI may restructure modules during the live phase.
Long dendritic runs periodically clear PerforatedAI processor buffers and the
accelerator cache after completed batches to prevent late-epoch MPS memory
pressure.

### Quantization via torchao
```python
import torchao
# 8-bit
torchao.quantize_(model, torchao.quantization.int8_weight_only())
# 4-bit
torchao.quantize_(model, torchao.quantization.int4_weight_only())
# 2-bit / 1.58-bit / 1-bit — use QAT for best results
from torchao.quantization.prototype.qat import Int8ActInt4WeightQATQuantizer
```

`torchao` quantization currently targets CPU/CUDA for kernel dispatch; on M3 Pro, run the quantization step on CPU and then move back to MPS for evaluation.

### Pruning
```python
import torch.nn.utils.prune as prune
parameters_to_prune = [(module, 'weight') for module in model.modules()
                        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d))]
prune.global_unstructured(parameters_to_prune,
                           pruning_method=prune.L1Unstructured, amount=0.40)
prune.remove(module, 'weight')  # make permanent before quantization
```

For the current benchmark, pruning is not part of the primary condition grid so the dendritic/non-dendritic comparison stays clean.

***
## Baseline Quality: Why the FP32 Recipes Track Published Setups

Every question this benchmark asks is a *comparison* against `base_fp32`. An
under-trained baseline does not just lower one number — it inflates the measured
"dendrite delta" and the measured quantization robustness, because a model still
far from its own optimum has slack that either intervention can take up. So the
FP32 recipes deliberately track the reference training setup for each
model/dataset pair rather than a shared default.

The 2026-08-06 pass re-derived those recipes from the published sources and
fixed four classes of problem found by reading `results/<model>/base_fp32/`:

**1. No learning-rate schedule.** Every model but PointNet trained at a flat
rate. ResNet-18 sat at lr 0.05 for all 90 epochs, train loss stuck at 0.18 and
validation oscillating around 88% — the classic signature of a run that needs
annealing, not more epochs. MobileNetV2 showed the same at 150 epochs. The
`lr_schedule` field (`constant` / `step` / `cosine` / `linear`, all with optional
warmup) now exists for this; see MODEL_REFERENCE.md for what each model uses.

**2. A loss that did not match the architecture.** CapsNet's forward returns
per-class capsule *lengths* in `[0, 1)`, but the loop applied `CrossEntropyLoss`
to them. A softmax over a range narrower than one nat produces a tiny, nearly
uniform gradient: train loss moved 1.578 → 1.468 across 30 whole epochs. It now
uses the margin loss from Sabour et al., which is what routing-by-agreement was
designed against.

**3. Unnormalised regression targets.** ESOL's log solubilities span −11.6..+1.6
and FreeSolv's hydration energies −25.5..+3.4 kcal/mol. Trained on raw values,
AttentiveFP spent its first ten epochs at train MSE ≈ 14.8 — exactly FreeSolv's
target variance, i.e. it had learned the mean and nothing else — and finished at
RMSE 2.14 against MoleculeNet's ~1.15. Targets are now z-scored from
training-split statistics, with `TaskBundle.target_offset` / `target_scale`
carrying the transform so reported RMSE stays in the dataset's own units.

**4. Featurisation that discarded the structure being measured.** The SMILES
parser skipped every non-alphabetic character, so `(` and `)` vanished and any
branched molecule was flattened into a single chain — wrong topology for 76% of
ESOL and 70% of FreeSolv. It also dropped bond orders, ring closures and bracket
atoms. Separately, three models pooled over padded node slots: IMDB-BINARY graphs
average 19.8 nodes padded to 96, so GIN's `mean(dim=1)` divided the graph
embedding by ~5 and mixed in whatever the layer biases produced for empty slots.
MPNN and AttentiveFP masked on `adjacency.sum(-1) > 0`, which is true everywhere
because the featuriser writes a self-loop into every row — a no-op mask. All
three now pool on the feature block, which is zero for padding.

Two more changes are ordinary under-training fixes rather than bugs: DistilBERT
fine-tuned at 1e-4, five times the canonical 2e-5 for BERT-family models, and
AG News was truncated to 64 tokens with a 5k vocabulary.

### Reference points for `base_fp32`

These are the published numbers the recipes aim at, not results:

| Key | Metric | Reference | Source |
|---|---|---|---|
| `lenet5` | Accuracy | ~99.1% | LeCun et al. 1998 |
| `capsnet_mnist` | Accuracy | 99.5% | Sabour et al. 2017 |
| `resnet18_cifar10` | Accuracy | ~94–95% | He et al.; PyTorch Lightning CIFAR-10 baseline |
| `mobilenetv2_cifar10` | Accuracy | ~94.1% | Sandler et al.; CIFAR-adapted reimplementations |
| `distilbert` | Accuracy | 91.3% (SST-2 dev) | `distilbert-base-uncased-finetuned-sst-2-english` |
| `gcn` | Accuracy | 81.5% (public split) | Kipf & Welling 2017 |
| `gin_imdbb` | Accuracy | 75.1% ± 5.1 | Xu et al. 2019 |
| `mpnn` | RMSE | ~0.58 | MoleculeNet (ESOL, random split) |
| `attentivefp_freesolv` | RMSE | ~1.15 | MoleculeNet (FreeSolv) |
| `tabnet` | Accuracy | 85.7% | Arık & Pfister 2021 |
| `saint_adult` | Accuracy | ~86% | Somepalli et al. 2021 |
| `pointnet_modelnet40` | Accuracy | 89.2% | Qi et al. 2017 |
| `ppo_bipedalwalker` | Episodic Return | 300 = solved; ~213 at 5M steps | Gymnasium; SB3 RL Zoo |
| `actor_critic` | Episodic Return | 500 = max | CartPole-v1 |
| `dqn_lunarlander` | Episodic Return | 200 = solved | LunarLander-v3 |

Note that this benchmark's splits are not always the reference splits — Cora now
uses the 140-label Planetoid split and full-graph transductive propagation, but
IMDB-BINARY uses one held-out split rather than 10-fold CV, and `distilbert`
tests on 611 of the GLUE dev set's 872 rows (the other 261 are its validation
split) — so the numbers are targets to be near, not thresholds to hit exactly.
IMDB-BINARY's test split in particular is 150 graphs, where a single graph moves
accuracy by 0.67%.

Two budgets are knowingly short of their reference and should be read as such:
`ppo_bipedalwalker` trains 1.6M environment steps against the RL Zoo's 5M, and
`pointnet_modelnet40` 200 epochs against Qi et al.'s 250. For `actor_critic` and
`dqn_lunarlander` the return is a recorded extra column, not what they train on
— their headline metric is agreement with a scripted policy and has no published
counterpart.

### What has actually been re-measured

A subset was retrained on the new recipes and compared against the old
`results/` records. The rest of the roster is changed but **not yet re-measured**
— treat the recipes for those as reasoned from the reference setups, not
verified here.

| Key | Metric | Old | New | Read |
|---|---|---|---|---|
| `attentivefp_freesolv` | RMSE | 2.1378 | **0.8457** | 2.5× better; past MoleculeNet's ~1.15 |
| `lenet5` | Accuracy | 0.9839 | **0.9923** | Clears LeCun's ~99.05% |
| `textcnn` | Accuracy | 0.9103 | **0.9155** | Larger vocab/sequence + annealed 30 epochs |
| `tabnet` | Accuracy | 0.8507 | 0.8530 | Flat — at the dataset ceiling, see below |
| `saint_adult` | Accuracy | 0.8573 | 0.8566 | Flat — at the dataset ceiling, see below |
| `mpnn` | RMSE | 0.7708 | 0.8117 → **0.6665** | See below |
| `gcn` | Accuracy | 0.7862 | 0.7641 | See below |
| `gin_imdbb` | Accuracy | 0.7933 | 0.7467 | See below |

**MPNN.** The first re-run at 300 epochs *regressed* to 0.8117. The cause is
visible in the curve: with the richer featuriser and only ~790 training
molecules the model reached train RMSE 0.29 against val 0.80 — it had memorised
the training set. The same recipe at 200 epochs tested at 0.6665, better than
both, so the budget is now 200. A regularisation sweep (weight decay 1e-4/1e-3,
dropout 0.3) was started and not finished; it is the obvious next thing to try
if MPNN is worth pushing toward MoleculeNet's 0.58.

**The two Adult Income models did not move**, despite doubling the epoch budget
and adding annealing: TabNet 0.8507 → 0.8530, SAINT 0.8573 → 0.8566. Both were
already within ~0.6% of their published numbers (85.7% and ~86%), so this is the
expected outcome — the recipes were not what was holding them back. What is
holding them back is the feature encoding: `_build_adult` maps each categorical
column to an *ordinal* integer and then standardises it, which asserts a false
ordering over unordered categories (workclass, occupation, native-country). The
reference implementations one-hot or embed those. This was left alone
deliberately — one-hot takes Adult from 14 to ~108 columns, and SAINT's
row/column attention is quadratic in column count, so it would cost roughly 60×
per step for a likely sub-1% gain. It is the right next lever for these two if
tabular accuracy matters more than runtime.

**GCN and GIN** both came out lower on test while training more healthily, and
neither difference is larger than its split's noise:

- GCN's *old* best epoch was epoch 1 of 200 — a checkpoint from before the model
  had learned anything, which happened to generalise. By epoch 200 the old run
  had driven train loss to 0.027 (pure memorisation) and validation down to
  0.766. The new run peaks at epoch 17 with a genuine optimum, holds a mean
  validation of 0.791 across its last 20 epochs against the old run's 0.761, and
  ends at train loss 0.232. Test moved 0.786 → 0.764 on a ~406-node split whose
  standard error is about 2.1%, i.e. within one sigma.
- GIN's train loss now reaches 0.573 (was 0.622) and its mean validation over
  the last 20 epochs is 0.729 (was 0.704); best validation is identical at
  0.7667. Test moved 0.793 → 0.747 on a 150-graph split — seven graphs. The new
  figure sits on Xu et al.'s 75.1%; the old one sat above it.

The masked pooling and degree featurisation behind those two are correctness
fixes independent of the score (mean-pooling 96 slots when 20 are real is wrong
whatever it measures), so they stay. But a single held-out split cannot
distinguish a 2-6% move from noise on datasets this small, and neither of these
should be reported as a regression or an improvement without repeated seeds.

**Dendritic path.** `gin_imdbb / dendrites_fp32` was run end-to-end against the
new architecture to confirm PerforatedAI still perforates a model whose input
width changed (8 → 10 features): it completed with no errors, grew 39,302 →
116,492 parameters, wrote the full artifact set (`best_arch_scores.csv`,
`paramCounts.csv`, `pai_plots/`), and reached a best validation of 0.7800 against
the baseline's 0.7667. That is a pipeline check, not a result — the other
architecture changes (MPNN/AttentiveFP at 20 node features, TextCNN's 20k vocab)
have not had their dendritic conditions exercised.

## Key Research Hypotheses
1. Do dendritic models consistently outperform base models in accuracy before quantization?
2. Does the dendrite + pruning combination produce better accuracy-per-byte than base + quantization alone?
3. Are certain domains (e.g., graph, tabular) more tolerant of extreme quantization (Q1–Q2) in dendritic form?
4. Does the accuracy gap between dendritic and non-dendritic models widen or narrow at extreme bit depths?
5. Is the file size / parameter count reduction from Dend+Prune+Q1 competitive with the accuracy loss vs. Base FP32?

---

# Part 2: 15 New Models & Extended Experiments (Round 2)

> **Superseded (2026-08-08) — every number in the Executive Summary and in
> "Key Findings from Round 1" below is pre-fix.** They come from runs that
> predate the 2026-08-06 baseline-quality pass and the 2026-08-07 comparability
> fixes, and several are known to measure the wrong thing:
>
> - **Actor-Critic's "0.815 → 0.931 Reward" is not a reward.** The RL models were
>   scored as negative action MAE against a heuristic policy. `actor_critic` and
>   `dqn_lunarlander` now report *Action Accuracy*, and `ppo_bipedalwalker` was
>   converted to real on-policy PPO reporting *Episodic Return*. The +14.3%
>   "policy optimization benefits greatly" reading has no basis in the new metric.
> - **The forecasters' MAEs were 3–5× optimistic** — the windows leaked across
>   the split. That invalidates the LSTM Forecaster row and the entire "Q4
>   Rescue" table built on it.
> - **GCN was restructured** to full-graph transductive Cora, so 79.61% → 79.12%
>   describes a model that no longer exists.
> - **DistilBERT's validation set was carved out of GLUE SST-2's train split**
>   and leaked; its 82.80% is not a clean number.
>
> The conclusions drawn from these tables — which domains gain from dendrites,
> where Q4 rescue happens, whether transformers absorb dendritic capacity — are
> **not established** and must not be cited until the retraining sweep
> (`REMAINING_FIXES.md` §3.4) has run. "What has actually been re-measured"
> above is the only measured, post-fix table in this document.

## Executive Summary
The first 10-model round reveals three distinct behavioral clusters. **Dendrites deliver large, consistent gains** in reinforcement learning (Actor-Critic: +14.3%), molecular property prediction (MPNN: +15.6%), and audio classification (M5: +4.5%), while also rescuing Q4 accuracy in time-series forecasting (LSTM Forecaster's Q4 normalized score jumps from 45.6% → 97.0% — a +51.4 point rescue). **Dendrites are neutral-to-mildly-harmful** for transformers (DistilBERT: −1.1%), graph convolutions (GCN: −0.6%), tabular attention (TabNet: −0.1%), and text CNNs (TextCNN: −0.1%). **Q2 universally collapses** (≤35% normalized score for almost every model regardless of dendrites), while Q4 is the critical threshold where dendrites matter most.

***
## Key Findings from Round 1

### Dendrite Delta at FP32
| Model | Domain | Base FP32 Score | Dendrites FP32 | Δ (pp) | Interpretation |
|---|---|---|---|---|---|
| MPNN | Molecular | 1.036 RMSE | 0.896 RMSE | **+15.6%** | Strongest beneficiary |
| Actor-Critic | RL | 0.815 Reward | 0.931 Reward | **+14.3%** | Policy optimization benefits greatly |
| M5 (1D-CNN) | Audio | 80.99% | 84.60% | **+4.5%** | Temporal feature hierarchies deepen |
| LSTM Autoencoder | Anomaly | 77.25% AUC | 78.09% AUC | **+1.1%** | Modest gain |
| LSTM Forecaster | Time-Series | MAE 0.0702 | MAE 0.0695 | **+1.0%** | Small FP32 gain, but enormous Q4 rescue |
| LeNet-5 | Image | 98.84% | 99.05% | +0.2% | Near-saturated |
| TextCNN | NLP | 90.49% | 90.39% | −0.1% | Saturated embedding model |
| TabNet | Tabular | 85.04% | 84.93% | −0.1% | Attention mechanism doesn't gain |
| GCN | Graph | 79.61% | 79.12% | **−0.6%** | Sparse adjacency may conflict with dendritic routing |
| DistilBERT | Large NLP | 82.80% | 81.88% | **−1.1%** | Large transformer absorbs optimization capacity |

### Q4 Rescue: The Most Actionable Finding
| Model | Base Q4 (norm%) | Dend+Prune+Q4 (norm%) | Q4 Rescue (Δpp) |
|---|---|---|---|
| **LSTM Forecaster** | 45.65 | 97.01 | **+51.4** |
| **MPNN** | 99.07 | 112.31 | **+13.2** |
| **Actor-Critic** | 99.51 | 110.88 | **+11.4** |
| TextCNN | 98.02 | 99.75 | +1.7 |
| GCN | 99.69 | 99.69 | 0.0 |
| TabNet | 99.64 | 98.92 | −0.7 |
| DistilBERT | 97.92 | 96.12 | −1.8 |
| LeNet-5 | 99.92 | 97.63 | −2.3 |
| LSTM Autoencoder | 123.50 | 114.71 | −8.8 |
| M5 | 69.48 | 50.87 | **−18.6** |

***
## 15 New Models for Round 2

### Group A: Deeper RL (Probe the RL Dendrite Win)

#### Model 11 — DQN (LunarLander-v2)
| Field | Value |
|---|---|
| **Key** | `dqn_lunarlander` |
| **Domain** | Reinforcement Learning — harder continuous state space |
| **Dataset/Env** | `gymnasium LunarLander-v2` |
| **Architecture** | 3-layer MLP Q-network + target Q-network, replay buffer (50K), ε-greedy |
| **Metric** | Mean episodic reward (solved ≥ 200) |
| **PAI Notes** | Perforate the Q-network MLP only (not target network) |

#### Model 12 — PPO Policy Network (BipedalWalker-v3)
| Field | Value |
|---|---|
| **Key** | `ppo_bipedalwalker` |
| **Domain** | Reinforcement Learning — continuous action space |
| **Dataset/Env** | `gymnasium BipedalWalker-v3` |
| **Architecture** | Shared backbone MLP + separate actor/critic heads, diagonal Gaussian policy, GAE(λ) advantage estimation |
| **Training** | Real on-policy PPO — one epoch = one iteration of 2048 env steps + 10 minibatch passes. The only model in the suite with no cached dataset. |
| **Metric** | Mean episodic return (solved ≥ 300), and here it is the selection metric, not a recorded extra |
| **PAI Notes** | Perforate the shared backbone; `.actor_mean` and `.critic` stay track-only — a dendrite switching in on either head invalidates the live rollout buffer (clip range on one, advantage baseline on the other) |

### Group B: Molecular/Graph Depth (Probe the MPNN Win)

#### Model 13 — AttentiveFP (FreeSolv)
| Field | Value |
|---|---|
| **Key** | `attentivefp_freesolv` |
| **Domain** | Drug Discovery / Molecular Property Prediction |
| **Dataset** | FreeSolv (642 molecules, hydration free energy regression) |
| **Architecture** | Multi-layer graph attention with node/edge features + global readout |
| **Metric** | RMSE (kcal/mol) |
| **PAI Notes** | GRU-style graph updates are implemented with explicit Linear gates, so the default Linear/Conv perforation registration applies. |

#### Model 14 — GIN (IMDB-B, Graph Classification)
| Field | Value |
|---|---|
| **Key** | `gin_imdbb` |
| **Domain** | Graph Classification (Social Networks) |
| **Dataset** | IMDB-Binary — 1000 graphs, binary classification |
| **Architecture** | 4-layer Graph Isomorphism Network with MLP aggregators, global mean pooling |
| **Metric** | Accuracy (10-fold CV) |

### Group C: Time-Series Depth (Probe the Q4 Rescue)

#### Model 15 — TCN Forecaster (ETTm1)
| Field | Value |
|---|---|
| **Key** | `tcn_forecaster` |
| **Domain** | Time-Series Forecasting — convolutional (non-RNN) |
| **Dataset** | ETTm1 (15-min intervals, 7 features) |
| **Architecture** | Dilated causal 1D convolutions with residual blocks, multi-step output head |
| **Metric** | MAE |
| **Scientific Rationale** | Critical control: tests if the LSTM Q4 rescue is an RNN recurrence property or general |

#### Model 16 — GRU Forecaster (Weather Dataset)
| Field | Value |
|---|---|
| **Key** | `gru_forecaster` |
| **Domain** | Time-Series Forecasting — RNN variant |
| **Dataset** | Weather (21 meteorological features) |
| **Architecture** | 2-layer GRU forecaster implemented with explicit Linear update/reset/new gates, FC projection to multi-step output |
| **Metric** | MAE |
| **PAI Notes** | Default Linear/Conv perforation registration applies to the gate projections. |

### Group D: Entirely New Domains

#### Model 17 — PointNet (ModelNet40, 3D Classification)
| Field | Value |
|---|---|
| **Key** | `pointnet_modelnet40` |
| **Domain** | 3D Point Cloud Classification |
| **Dataset** | ModelNet40 (12,311 CAD models, 40 classes), sampled uniformly over mesh faces to 1024 points and cached |
| **Architecture** | T-Net input/feature transform, shared MLP on per-point features, global max pooling |
| **Metric** | Accuracy (%) |

#### Model 18 — VAE (MNIST, Generative)
| Field | Value |
|---|---|
| **Key** | `vae_mnist` |
| **Domain** | Generative Modeling / Unsupervised Representation Learning |
| **Dataset** | MNIST (60K images) |
| **Architecture** | FC encoder → (μ, logσ²), reparameterization trick, FC decoder; ELBO loss |
| **Metric** | ELBO (higher = better) |

#### Model 19 — Spiking Neural Network (N-MNIST)
| Field | Value |
|---|---|
| **Key** | `snn_nmnist` |
| **Domain** | Neuromorphic Computing / Event-Driven Classification |
| **Dataset** | N-MNIST (event-camera MNIST, 60K samples) |
| **Architecture** | Conv-LIF → Conv-LIF → FC-LIF SNN, T=10 timesteps, PyTorch surrogate-gradient spike activation |
| **Metric** | Accuracy (%) |
| **Scientific Rationale** | Most biologically motivated experiment — biological dendrites and spiking neurons coexist |

<!--
#### Model 20 — Tiny U-Net (ISIC Skin Lesion Segmentation)
| Field | Value |
|---|---|
| **Key** | `unet_isic` |
| **Domain** | Medical Image Segmentation / Dense Prediction |
| **Dataset** | ISIC 2018 Task 1 (2,594 dermoscopy images, binary lesion mask) |
| **Architecture** | 4-level encoder-decoder with skip connections (16→32→64→128 channels) |
| **Metric** | Dice coefficient |
-->

### Group E: Architecture Interaction Studies

#### Model 21 — ResNet-18 (CIFAR-10)
| Field | Value |
|---|---|
| **Key** | `resnet18_cifar10` |
| **Domain** | Image Classification — residual architecture |
| **Dataset** | CIFAR-10 (50K/10K, 10 classes) |
| **Architecture** | Standard ResNet-18 with modified first conv for 32×32 input |
| **Metric** | Accuracy (%) |
| **Dynamic12 role** | Nondendritic `base_fp32` control and `base_q*` PQAT arms |

#### Model 21a — Published Perforated ResNet-18 (CIFAR-10 transfer)
| Field | Value |
|---|---|
| **Key** | `resnet18_hf_perforated_cifar10` |
| **Domain** | Image Classification — residual, already perforated |
| **Dataset** | CIFAR-10 (transfer from ImageNet weights) |
| **Architecture** | Hugging Face `perforated-ai/resnet-18-perforated-gd`; learned stem adapted to 32×32 and classifier replaced with 10 outputs |
| **Metric** | Accuracy (%) |
| **Dynamic12 role** | Already-perforated/dendritic counterpart to `resnet18_cifar10` |
| **PAI Notes** | The published pre-FC graph has five branches. Its `base_fp32` and five `base_q*` records are the distinct perforated counterpart arms; a second PAI conversion is not performed. |

#### Model 22 — MobileNetV2 (CIFAR-10)
| Field | Value |
|---|---|
| **Key** | `mobilenetv2_cifar10` |
| **Domain** | Image Classification — efficient depthwise-separable |
| **Dataset** | CIFAR-10 |
| **Architecture** | MobileNetV2 inverted residual bottlenecks, modified for 32×32 inputs |
| **Metric** | Accuracy (%) |

#### Model 23 — SAINT (Adult Income, Tabular Transformer)
| Field | Value |
|---|---|
| **Key** | `saint_adult` |
| **Domain** | Tabular Classification — self + inter-sample attention |
| **Dataset** | Adult Income |
| **Architecture** | Feature embedding, column-wise self-attention, row-wise inter-sample attention |
| **Metric** | Accuracy (%) |
| **PAI Notes** | Attention uses explicit Linear Q/K/V/output projections, so default Linear/Conv perforation registration applies. |

#### Model 24 — Capsule Network (CapsNet, MNIST)
| Field | Value |
|---|---|
| **Key** | `capsnet_mnist` |
| **Domain** | Image Classification — equivariant dynamic routing |
| **Dataset** | MNIST |
| **Architecture** | Conv feature detector → PrimaryCaps → DigitCaps with routing-by-agreement (3 iterations) |
| **Metric** | Accuracy (%) |
| **Scientific Rationale** | Unique combination of routing-by-agreement and PAI's cascade-correlation dendrite addition |

***
## Registered Model Roster
| # | Key | Domain | Dataset | ~Params |
|---|---|---|---|---|
| 1 | `lenet5` | Image (tiny CNN) | MNIST | 62K |
| 2 | `m5` | Audio (1D-CNN) | SpeechCommands | 25K |
| 3 | `lstm_forecaster` | Time-Series (RNN) | ETTh1 | 52K |
| 4 | `textcnn` | NLP (Text CNN) | AG News | ~2.8M |
| 5 | `gcn` | Graph (Conv) | Cora | 92K |
| 6 | `tabnet` | Tabular (Seq Att) | Adult Income | 39K |
| 7 | `mpnn` | Molecular (GNN) | ESOL | 355K |
| 8 | `actor_critic` | RL (CartPole) | CartPole-v1 | 18K |
| 9 | `lstm_autoencoder` | Anomaly Detect | MIT-BIH ECG | 71K |
| 10 | `distilbert` | Large NLP (Xfmr) | SST-2 | ~67M |
| 11 | `dqn_lunarlander` | RL (Q-net) | LunarLander-v2 | 69K |
| 12 | `ppo_bipedalwalker` | RL (continuous) | BipedalWalker-v3 | 20K |
| 13 | `attentivefp_freesolv` | Molecular (Att-GNN) | FreeSolv | 612K |
| 14 | `gin_imdbb` | Graph Classif. | IMDB-B | 39K |
| 15 | `tcn_forecaster` | Time-Series (TCN) | ETTm1 | 99K |
| 16 | `gru_forecaster` | Time-Series (GRU) | Weather | 74K |
| 17 | `pointnet_modelnet40` | 3D Point Cloud | ModelNet40 | ~3.5M |
| 18 | `vae_mnist` | Generative (VAE) | MNIST | ~1.1M |
| 19 | `snn_nmnist` | Neuromorphic SNN | N-MNIST | 60K |
<!-- | 20 | `unet_isic` | Medical Seg. | ISIC 2018 | ~1.9M | -->
| 21 | `resnet18_cifar10` | Image (ResNet) | CIFAR-10 | ~11.2M |
| 21a | `resnet18_hf_perforated_cifar10` | Image (published perforated ResNet) | CIFAR-10 transfer | ~12.5M |
| 22 | `mobilenetv2_cifar10` | Image (Efficient) | CIFAR-10 | ~2.2M |
| 23 | `saint_adult` | Tabular (Xfmr) | Adult Income | 205K |
| 24 | `capsnet_mnist` | Image (CapsNet) | MNIST | ~6.8M |

***
## Additional Experiments Beyond New Models

### Experiment A — Pruning Rate Sweep (MPNN & Actor-Critic)
Re-run dendritic quantized conditions at sparsity levels {10%, 20%, 30%, 40%, 50%, 60%, 70%} for MPNN and Actor-Critic to find the Pareto-optimal prune rate for each bit-width.

### Experiment B — Dendrite Cycle Count Ablation
For Actor-Critic and MPNN, vary `GPA.pc.set_max_dendrites(N)` across N ∈ {1, 2, 3, 4, 5} and record Q4/Q8 normalized score at each cycle count.

### Experiment C — QAT-Integrated Dendritic Training (for Q2 Rescue)
For MPNN, Actor-Critic, and LSTM Forecaster, run QAT inside the PAI dendritic training loop — project weights to Q4/Q2 representations during forward passes while allowing full-precision gradient accumulation.

### Experiment D — Structured vs. Unstructured Pruning Comparison
For MPNN, Actor-Critic, and LSTM Forecaster, compare L1 unstructured global pruning at 40% sparsity against L2 structured (channel-level) pruning at equivalent parameter reduction.

### Experiment E — Inference Latency Benchmarking on M3 Pro
For the 5 best Q4 conditions per model, measure actual wall-clock inference latency on M3 Pro (batch size 1 and 32) using `torch.utils.benchmark.Timer`. Add `dqb benchmark_models` command to `cli.py`.

### Experiment F — Dataset Difficulty Scaling (LSTM Forecaster)
Run the full 12-condition suite for LSTM Forecaster on ETTh1, ETTh2, ETTm1, ETTm2, and Weather to test whether the +51.4pp Q4 rescue magnitude scales with dataset complexity.

### Experiment G — Anomaly Detection Regularization Study
For LSTM Autoencoder, add Gaussian noise injection at σ ∈ {0.01, 0.05, 0.10, 0.20} during training to validate whether the Q4 inversion (+123.5% AUC) is due to implicit noise injection.

### Experiment H — Cross-Architecture Tabular Comparison (SAINT vs TabNet)
Once SAINT is trained in Round 2, plot a side-by-side comparison of all 12 conditions between SAINT, TabNet, and XGBoost baseline on Adult Income to determine whether the null result is domain-specific or architecture-specific.

***
## Conclusion
Round 1 produced three clean scientific findings: (1) dendritic models provide the largest gains in domains with strong temporal dynamics and continuous optimization landscapes (RL, molecular, audio); (2) dendrites rescue Q4 accuracy specifically in RNN-based time-series, suggesting recurrent hidden-state precision is the mechanism; (3) Q2 is a near-universal floor not addressable by dendrites alone, requiring QAT from the start. The expanded registered suite stress-tests each of these findings across new architectures and domains, providing broad cross-domain evidence about dendritic quantization robustness.

---

Three further parts of this document were removed in the 2026-08-31 documentation
cleanup: a walkthrough of each module (Part 3), a usage guide for every `dqb` subcommand
(Part 4), and a latency-benchmarking guide (Part N). All three restated facts that are now
generated from the code — see [CURRENT_GUIDE.md](CURRENT_GUIDE.md) for the model roster,
the condition grid, the reportability rules, and the full command reference, and
[CLI_DIAGRAMS.md](CLI_DIAGRAMS.md) for the per-command flowcharts and output layout. The
module map predated the P1 split and no longer matched the package, and the latency
guide's dendrite-overhead advice rested on the mislabelled dendritic latency rows that
[CODE_REVIEW_2026-08-28.md](CODE_REVIEW_2026-08-28.md) §2 withdrew.
