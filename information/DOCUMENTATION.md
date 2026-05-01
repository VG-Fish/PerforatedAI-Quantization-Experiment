# Dendritic Quantization Benchmark: Documentation

This document consolidates all project documentation: the experiment plan, extended model proposals, script architecture, and CLI usage guide.

---

# Part 1: 10-Model, 10-Domain Experiment Plan

## Overview
This experiment investigates whether quantized dendritic models (created via Perforated Backpropagation) outperform non-dendritic counterparts across diverse fields. Ten models are selected — one per domain — spanning complexities from ~50K to ~66M parameters. Each model is trained and evaluated under **12 experimental conditions**, yielding 120 total training runs.

The hardware target is an **Apple M3 Pro** chip using PyTorch's MPS backend, with a total budget of 12–48 hours. All quantization uses `torchao` (PyTorch-native), and dendrites are added via the `PerforatedAI` library.

***
## The 10 Benchmark Models
| # | Model | Domain | Dataset | Complexity | ~Params | Est. Base Hrs |
|---|-------|--------|---------|------------|---------|---------------|
| 1 | **LeNet-5** | Image Classification | MNIST | Tiny | 60K | 0.5h |
| 2 | **M5 (1D-CNN)** | Audio Classification | SpeechCommands | Tiny | 300K | 1h |
| 3 | **LSTM Univariate** | Time-Series Forecasting | ETTh1 | Tiny | 200K | 1h |
| 4 | **TextCNN** | NLP / Text Classification | AG News | Tiny | 500K | 0.5h |
| 5 | **GCN** | Graph / Node Classification | Cora | Tiny | 180K | 0.5h |
| 6 | **TabNet** | Tabular Classification | Adult Income | Small | 1M | 1h |
| 7 | **MPNN** | Drug Discovery / Molecular | ESOL (MoleculeNet) | Small | 400K | 1h |
| 8 | **Actor-Critic** | Reinforcement Learning | CartPole-v1 | Tiny | 50K | 0.3h |
| 9 | **LSTM Autoencoder** | Anomaly Detection (ECG) | MIT-BIH | Small | 800K | 1.5h |
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

***
## Output Graphs (Per Model)
For each of the 10 models, generate **3 comparison bar charts** — one for each metric — with all 12 conditions on the x-axis:

### Graph Set A: Accuracy (or Task Metric)
- Y-axis: Accuracy % (classification), MAE/MSE (regression/forecasting), Reward (RL), AUC (anomaly), ELBO (VAE)
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

### Cross-Graph 1: Accuracy Retention Heatmap (10 × 13)
- Rows = models/domains, Columns = conditions
- Cell value = accuracy as % of the Base FP32 baseline (retention ratio)

### Cross-Graph 2: Size Reduction vs. Accuracy Tradeoff (scatter)
- X-axis: File size reduction ratio vs. Base FP32
- Y-axis: Accuracy retention (%)
- One point per (model × condition) combination — 130 points total

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
3. **+Dendrites FP32** → retrain from scratch with `doing_pai=True`; use `DOING_FIXED_SWITCH` to bound time
4. **+Dendrites FP32** → use the dendritic checkpoint as the source state for all dendritic quantized evaluations
5. **+Dendrites+Q8 through Q1** → load dendritic FP32 checkpoint; apply quantization in sequence

### PerforatedAI Output Files
The library writes these automatically to the `save_name/` folder:
- `best_model` — best checkpoint by validation score
- `final_clean_pai` — inference-optimized checkpoint (when enabled by the library)
- `latest` — most recent checkpoint; use to resume if training crashes
- `best_arch_scores.csv` — best test scores + parameter counts per dendrite cycle
- `paramCounts.csv` — parameter count at each epoch
- `Scores.csv` — validation + extra scores per epoch

The library's active `PAI/PAI_config.json` is also snapshotted after each
perforation as `PAI/<model>_<condition>_PAI_config.json` and, for the run
artifact, as `results/<model>/<condition>/PAI_config.json`.

This benchmark suite itself saves the best model state it evaluated to `results/<model>/<condition>/model.pt` and uses that file for comparisons and file-size reporting.

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

For non-standard layers (GRUs, Transformers, custom blocks):
```python
GPA.pc.append_modules_to_perforate([nn.MultiheadAttention])
GPA.pc.append_module_names_to_perforate(['encoder_block'])
GPA.pc.append_module_ids_to_perforate(['.layer1.0.conv1'])
```

#### Step 3 — Optimizer & Scheduler Setup
```python
GPA.pai_tracker.set_optimizer(torch.optim.Adam)
GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
optimArgs = {'params': model.parameters(), 'lr': learning_rate}
schedArgs = {'mode': 'max', 'patience': 5}
optimizer, PAIscheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
```

#### Step 4 — Validation Loop Hook
The benchmark now enables live dendrite restructuring for dendritic training
runs by wiring PerforatedAI's tracker into the optimizer and validation loop.
Dynamic insertion is active for the first 80% of the configured epochs, then
the final 20% trains with the current dendrite layout frozen so weights can
settle before final evaluation.

```python
if epoch < int(max_epochs * 0.8):
    model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
        score, model
    )
    model.to(device)

    if training_complete:
        break
    elif restructured:
        optimizer, PAIscheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
```

#### Step 5 — Training Loop Structure
```python
epoch = -1
while True:
    epoch += 1
    # train as normal
    # validation loop ends with add_validation_score call above
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
## Key Research Hypotheses
1. Do dendritic models consistently outperform base models in accuracy before quantization?
2. Does the dendrite + pruning combination produce better accuracy-per-byte than base + quantization alone?
3. Are certain domains (e.g., graph, tabular) more tolerant of extreme quantization (Q1–Q2) in dendritic form?
4. Does the accuracy gap between dendritic and non-dendritic models widen or narrow at extreme bit depths?
5. Is the file size / parameter count reduction from Dend+Prune+Q1 competitive with the accuracy loss vs. Base FP32?

---

# Part 2: 15 New Models & Extended Experiments (Round 2)

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
| **Architecture** | Shared backbone MLP + separate actor/critic heads, GAE advantage estimation |
| **Metric** | Mean episodic reward (solved ≥ 300) |
| **PAI Notes** | Perforate shared backbone; heads can remain standard |

### Group B: Molecular/Graph Depth (Probe the MPNN Win)

#### Model 13 — AttentiveFP (FreeSolv)
| Field | Value |
|---|---|
| **Key** | `attentivefp_freesolv` |
| **Domain** | Drug Discovery / Molecular Property Prediction |
| **Dataset** | FreeSolv (642 molecules, hydration free energy regression) |
| **Architecture** | Multi-layer graph attention with node/edge features + global readout |
| **Metric** | RMSE (kcal/mol) |
| **PAI Notes** | Must add `GPA.pc.append_modules_to_perforate([nn.GRUCell])` before `perforate_model` |

#### Model 14 — GIN (IMDB-B, Graph Classification)
| Field | Value |
|---|---|
| **Key** | `gin_imdbb` |
| **Domain** | Graph Classification (Social Networks) |
| **Dataset** | IMDB-Binary — 1000 graphs, binary classification |
| **Architecture** | 3-layer Graph Isomorphism Network with MLP aggregators, global mean pooling |
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
| **Architecture** | 2-layer bidirectional GRU, FC projection to multi-step output |
| **Metric** | MAE |
| **PAI Notes** | Must declare `GPA.pc.append_modules_to_perforate([nn.GRU])` |

### Group D: Entirely New Domains

#### Model 17 — PointNet (ModelNet40, 3D Classification)
| Field | Value |
|---|---|
| **Key** | `pointnet_modelnet40` |
| **Domain** | 3D Point Cloud Classification |
| **Dataset** | ModelNet40 (12,311 CAD models, 40 classes) |
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

#### Model 19 — Spiking Neural Network (SpikingJelly, N-MNIST)
| Field | Value |
|---|---|
| **Key** | `snn_nmnist` |
| **Domain** | Neuromorphic Computing / Event-Driven Classification |
| **Dataset** | N-MNIST (event-camera MNIST, 60K samples) |
| **Architecture** | Conv-LIF → Conv-LIF → FC-LIF SNN, T=10 timesteps, surrogate gradient (ATan) |
| **Metric** | Accuracy (%) |
| **Scientific Rationale** | Most biologically motivated experiment — biological dendrites and spiking neurons coexist |

#### Model 20 — Tiny U-Net (ISIC Skin Lesion Segmentation)
| Field | Value |
|---|---|
| **Key** | `unet_isic` |
| **Domain** | Medical Image Segmentation / Dense Prediction |
| **Dataset** | ISIC 2018 Task 1 (2,594 dermoscopy images, binary lesion mask) |
| **Architecture** | 4-level encoder-decoder with skip connections (16→32→64→128 channels) |
| **Metric** | Dice coefficient |

### Group E: Architecture Interaction Studies

#### Model 21 — ResNet-18 (CIFAR-10)
| Field | Value |
|---|---|
| **Key** | `resnet18_cifar10` |
| **Domain** | Image Classification — residual architecture |
| **Dataset** | CIFAR-10 (50K/10K, 10 classes) |
| **Architecture** | Standard ResNet-18 with modified first conv for 32×32 input |
| **Metric** | Accuracy (%) |

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
| **PAI Notes** | Use `GPA.pc.append_modules_to_perforate([nn.MultiheadAttention])` |

#### Model 24 — Capsule Network (CapsNet, MNIST)
| Field | Value |
|---|---|
| **Key** | `capsnet_mnist` |
| **Domain** | Image Classification — equivariant dynamic routing |
| **Dataset** | MNIST |
| **Architecture** | Conv feature detector → PrimaryCaps → DigitCaps with routing-by-agreement (3 iterations) |
| **Metric** | Accuracy (%) |
| **Scientific Rationale** | Unique combination of routing-by-agreement and PAI's cascade-correlation dendrite addition |

#### Model 25 — ConvLSTM (Moving MNIST, Spatiotemporal)
| Field | Value |
|---|---|
| **Key** | `convlstm_movingmnist` |
| **Domain** | Spatiotemporal Sequence Prediction |
| **Dataset** | Moving MNIST (10K sequences, predict next 10 frames from 10 input frames) |
| **Architecture** | 2-layer ConvLSTM (64 filters, 3×3 kernel), frame decoder, MSE reconstruction loss |
| **Metric** | SSIM |

***
## Complete 25-Model Roster
| # | Key | Domain | Dataset | ~Params |
|---|---|---|---|---|
| 1 | `lenet5` | Image (tiny CNN) | MNIST | 60K |
| 2 | `m5` | Audio (1D-CNN) | SpeechCommands | 35K |
| 3 | `lstm_forecaster` | Time-Series (RNN) | ETTh1 | 17K |
| 4 | `textcnn` | NLP (Text CNN) | AG News | 739K |
| 5 | `gcn` | Graph (Conv) | Cora | 92K |
| 6 | `tabnet` | Tabular (Seq Att) | Adult Income | 5K |
| 7 | `mpnn` | Molecular (GNN) | ESOL | 34K |
| 8 | `actor_critic` | RL (CartPole) | CartPole-v1 | 5K |
| 9 | `lstm_autoencoder` | Anomaly Detect | MIT-BIH ECG | 34K |
| 10 | `distilbert` | Large NLP (Xfmr) | SST-2 | 839K |
| 11 | `dqn_lunarlander` | RL (CNN Q-net) | LunarLander-v2 | ~50K |
| 12 | `ppo_bipedalwalker` | RL (continuous) | BipedalWalker-v3 | ~80K |
| 13 | `attentivefp_freesolv` | Molecular (Att-GNN) | FreeSolv | ~120K |
| 14 | `gin_imdbb` | Graph Classif. | IMDB-B | ~30K |
| 15 | `tcn_forecaster` | Time-Series (TCN) | ETTm1 | ~200K |
| 16 | `gru_forecaster` | Time-Series (GRU) | Weather | ~25K |
| 17 | `pointnet_modelnet40` | 3D Point Cloud | ModelNet40 | ~3.5M |
| 18 | `vae_mnist` | Generative (VAE) | MNIST | ~400K |
| 19 | `snn_nmnist` | Neuromorphic SNN | N-MNIST | ~100K |
| 20 | `unet_isic` | Medical Seg. | ISIC 2018 | ~7M |
| 21 | `resnet18_cifar10` | Image (ResNet) | CIFAR-10 | ~11M |
| 22 | `mobilenetv2_cifar10` | Image (Efficient) | CIFAR-10 | ~2.2M |
| 23 | `saint_adult` | Tabular (Xfmr) | Adult Income | ~500K |
| 24 | `capsnet_mnist` | Image (CapsNet) | MNIST | ~8M |
| 25 | `convlstm_movingmnist` | Spatiotemporal | Moving MNIST | ~500K |

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
For the 5 best Q4 conditions per model, measure actual wall-clock inference latency on M3 Pro (batch size 1 and 32) using `torch.utils.benchmark.Timer`. Add `dqb bench` command to `cli.py`.

### Experiment F — Dataset Difficulty Scaling (LSTM Forecaster)
Run the full 13-condition suite for LSTM Forecaster on ETTh1, ETTh2, ETTm1, ETTm2, and Weather to test whether the +51.4pp Q4 rescue magnitude scales with dataset complexity.

### Experiment G — Anomaly Detection Regularization Study
For LSTM Autoencoder, add Gaussian noise injection at σ ∈ {0.01, 0.05, 0.10, 0.20} during training to validate whether the Q4 inversion (+123.5% AUC) is due to implicit noise injection.

### Experiment H — Cross-Architecture Tabular Comparison (SAINT vs TabNet)
Once SAINT is trained in Round 2, plot a side-by-side comparison of all 12 conditions between SAINT, TabNet, and XGBoost baseline on Adult Income to determine whether the null result is domain-specific or architecture-specific.

***
## Conclusion
Round 1 produced three clean scientific findings: (1) dendritic models provide the largest gains in domains with strong temporal dynamics and continuous optimization landscapes (RL, molecular, audio); (2) dendrites rescue Q4 accuracy specifically in RNN-based time-series, suggesting recurrent hidden-state precision is the mechanism; (3) Q2 is a near-universal floor not addressable by dendrites alone, requiring QAT from the start. The 15 new models stress-test each of these findings across new architectures and domains. The complete 25-model suite with 12 conditions yields 300 training runs, providing strong statistical power for cross-domain claims about dendritic quantization robustness.

---

# Part 3: Script Architecture Guide

## Entry Point

### `src/dendritic_benchmark/cli.py`
The package exposes the `dqb` entry point via `pyproject.toml`.
- `uv run dqb run` constructs a `BenchmarkRunner` and executes the selected models and conditions.
- `uv run dqb compare` loads saved `best_model_stats.csv` files (falling back to `record.json`) and rebuilds per-model and cross-model comparison outputs.
- `uv run dqb generate_graphs` regenerates training plots from existing `history.csv` files.
- `--results-root` controls where per-model results are read from or written to.
- `--comparison-root` controls where comparison outputs are written for `run` and `compare`.
- `.env` credentials are loaded before running so PerforatedAI aliases and API variables are mirrored into the environment.

## Experiment Configuration

### `src/dendritic_benchmark/specs.py`
Declares benchmark model and condition metadata.
- `MODEL_SPECS` defines the 25 benchmark models and their evaluation metrics.
- `CONDITION_SPECS` defines the 13 experimental conditions, including source checkpoint dependencies, quantization settings, dendrite usage, pruning, and QAT.
- `model_by_key()` and `condition_by_key()` look up valid keys and raise on unknown values.

## Pipeline Orchestration

### `src/dendritic_benchmark/pipeline.py`
`BenchmarkRunner` orchestrates the full benchmark run.
- Creates the results and comparison directories.
- Expands requested condition keys into a dependency-resolved order.
- Builds dataset bundles via `data.py` and models via `models.py`.
- Loads source checkpoints when a condition depends on a previous result.
- Perforates models with PerforatedAI for dendritic conditions.
- Calls `train_and_evaluate()` for every model-condition pair.
- When `--allow-PQAT` is enabled, quantized conditions save a PTQ snapshot under `before_pqat/`, fine-tune for a short model-aware PQAT budget, and save the post-PQAT artifacts under `after_pqat/`.
- Writes per-condition training records and regenerates comparison outputs during the run.
- Skips dataset loading entirely for models where all conditions are already recorded.

The run method also computes `max_epochs` from `_base_epoch_budget()` so baseline and dendritic FP32 conditions receive larger budgets than post-training quantized conditions.

## Real Data

### `src/dendritic_benchmark/data.py`
Builds task bundles for each benchmark task and caches datasets under `data/` by default. Set `DQB_DATA_ROOT` to move the cache location. DataLoaders are tuned for Apple Silicon MPS: `pin_memory=False` (unified memory), `persistent_workers=True`, per-model batch sizes chosen for GPU saturation without OOM.

## Models

### `src/dendritic_benchmark/models.py`
Defines compact PyTorch model implementations for all 25 benchmark models and exposes `build_model()` to construct each architecture by key. Includes all model classes inline and a `MODEL_FACTORIES` dict mapping keys to constructors. ResNet-18 and MobileNetV2 use `torchvision.models` with CIFAR-10 adaptations.

## Compatibility Helpers

### `src/dendritic_benchmark/compat.py`
Isolates optional dependencies and PerforatedAI integration.
- Safely imports PyTorch and optional `dotenv`.
- Detects MPS, CUDA, or CPU at runtime.
- Mirrors PerforatedAI token and email aliases from environment variables.
- Wraps and configures models with PerforatedAI when available. Suppresses repeated `[PAI Config] Saved` messages by patching `builtins.print`.
- Provides simple symmetric, ternary, and binary quantization helpers.

## Training and Evaluation

### `src/dendritic_benchmark/training.py`
Runs each individual condition.
- Moves the model to the selected device.
- Sets high float32 matmul precision on MPS to improve Apple Silicon throughput where supported.
- Applies L1 unstructured global pruning for pruned conditions.
- Applies quantization for low-bit conditions and optionally QAT-style weight projection during the training loop.
- Compiles non-dendritic MPS models with `torch.compile(..., backend='aot_eager')` when available.
- Uses `GPA.pai_tracker.set_optimizer`, `setup_optimizer`, and `add_validation_score` for dendritic runs; live restructuring stops for the last 20% of epochs.
- Trains for the condition-specific epoch budget, or skips training for post-training quantization conditions (printing a skip-reason banner).
- Evaluates validation and test metrics with a rich set of per-task metrics (accuracy, MAE, RMSE, AUC, Dice, SSIM, ELBO, etc.).
- Saves the best model state to `model.pt` and writes `best_model_stats.csv` for the `compare` command.
- Writes `metrics.json`, `history.csv`, and returns a normalized `TrainingRecord`.
- For dendritic runs, also writes `best_arch_scores.csv` and `paramCounts.csv`.

## Results and Reports

### `src/dendritic_benchmark/results.py`
Reads and writes benchmark records and generates final visualizations.
- `save_training_record()` writes `record.json` and `record.csv` for each condition.
- `load_training_records()` prefers `results/*/*/best_model_stats.csv` and falls back to `record.json` for older runs.
- `write_manifest()` writes `results/manifest.csv`.
- `write_model_reports()` generates per-model metric, parameter, and size comparison charts.
- `write_comparison_reports()` generates cross-model heatmaps, a tradeoff scatter plot, a dendrite delta chart, and `summary.csv`.
- `generate_training_graphs()` regenerates training plots from `history.csv` and architecture evolution plots from `best_arch_scores.csv`.

## Plotting

### `src/dendritic_benchmark/plots.py`
Creates charts using Matplotlib's non-interactive `Agg` backend.
- Bar charts with adaptive label sizing and optional hatch patterns for PTQ conditions.
- Line charts plotting metric and loss series over epochs.
- Heatmaps and scatter plots for cross-model comparison outputs.
- Plot label overlap detection uses rendered text extents to avoid collisions.

## Typical Flows

### Full benchmark
```bash
uv run dqb run
```
1. `cli.py` parses the command and global options.
2. `pipeline.py` loops over selected models and conditions.
3. `data.py` downloads missing datasets and builds task bundles.
4. `models.py` builds the requested model architecture.
5. `training.py` trains, evaluates, and saves artifacts.
6. `results.py` writes records and reports.
7. `plots.py` writes SVG charts.

### Regenerate plots after training
```bash
uv run dqb compare --manifest
```
1. `cli.py` loads saved record files.
2. `results.py` optionally rewrites `manifest.csv`.
3. `results.py` rewrites per-model charts from saved records.
4. `results.py` rewrites comparison charts in the comparison directory.

### Generate training graphs
```bash
uv run dqb generate_graphs --results-root results
```
1. `cli.py` loads the results root.
2. `results.py` scans existing `history.csv` files.
3. `results.py` recreates `plots/` directories and writes metric and loss curves.
4. For dendritic conditions, also generates architecture evolution plots from `best_arch_scores.csv`.

---

# Part 4: Script Usage Guide

## Prerequisites

Create or activate the `uv` environment first:

```bash
uv venv .venv
```

Then run commands through `uv` so the project uses the managed environment:

```bash
uv run dqb --help
```

If you have PerforatedAI beta credentials, place them in a local `.env` file at the repo root. The CLI will load them automatically.

```dotenv
PAITOKEN=your-token-here
PAIEMAIL=you@example.com
```

The loader also recognizes the following aliases: `PERFORATEDAI_API_KEY`, `PERFORATEDAI_TOKEN`, `PERFORATEDBP_API_KEY`, `PERFORATEDBP_TOKEN`, `PERFORATEDAI_EMAIL`, `PERFORATEDBP_EMAIL`, `PAITOKEN`, `PAIEMAIL`.

## Commands

### `dqb run`
Runs the benchmark pipeline for selected models and conditions.

```bash
uv run dqb run
uv run dqb run --models lenet5 textcnn
uv run dqb run --conditions base_fp32 base_q8 dendrites_fp32
uv run dqb run --results-root results
uv run dqb run --comparison-root comparison
uv run dqb run --allow-PQAT
uv run dqb run --ignore-saved-models
```

### `dqb download_data`
Pre-downloads and caches datasets so that `run` works offline.

```bash
uv run dqb download_data
uv run dqb download_data --models lenet5 mpnn
uv run dqb download_data --strict
```

### `dqb compare`
Rebuilds comparison outputs from previously saved records.

```bash
uv run dqb compare
uv run dqb compare --manifest
uv run dqb compare --results-root results --comparison-root comparison
```

### `dqb generate_graphs`
Generates training curves from saved result histories without retraining.

```bash
uv run dqb generate_graphs
uv run dqb generate_graphs --results-root results
uv run dqb generate_graphs --regenerate-graphs
```

## Output Layout

```text
results/
  <model>/
    <condition>/
      record.json
      record.csv
      metrics.json
      history.csv
      model.pt
      PAI_config.json        # dendritic runs only
      best_model_stats.csv
      before_pqat/           # PQAT-enabled quantized runs only
      after_pqat/            # PQAT-enabled quantized runs only
      best_arch_scores.csv    # dendritic runs only
      paramCounts.csv         # dendritic runs only
      plots/
        training_curve.svg
        primary_metric.svg
        loss_curves.svg
        architecture_evolution.svg  # dendritic runs only
comparison/
  accuracy_retention_heatmap.svg
  size_tradeoff_scatter.svg
  dendrite_delta.svg
  best_quantization_heatmap.svg
  summary.csv
```

## Model Keys
`lenet5`, `m5`, `lstm_forecaster`, `textcnn`, `gcn`, `tabnet`, `mpnn`, `actor_critic`, `lstm_autoencoder`, `distilbert`, `dqn_lunarlander`, `ppo_bipedalwalker`, `attentivefp_freesolv`, `gin_imdbb`, `tcn_forecaster`, `gru_forecaster`, `pointnet_modelnet40`, `vae_mnist`, `snn_nmnist`, `unet_isic`, `resnet18_cifar10`, `mobilenetv2_cifar10`, `saint_adult`, `capsnet_mnist`, `convlstm_movingmnist`

## Condition Keys
`base_fp32`, `base_q8`, `base_q4`, `base_q2`, `base_q1_58`, `base_q1`, `dendrites_fp32`, `dendrites_q8`, `dendrites_q4`, `dendrites_q2`, `dendrites_q1_58`, `dendrites_q1`

## Common Workflows

```bash
# Smoke test
uv run dqb run --models lenet5 --conditions base_fp32 base_q8

# Download all data before a long run
uv run dqb download_data

# Regenerate plots from existing results
uv run dqb compare --manifest

# Re-run specific models ignoring cached results
uv run dqb run --models mpnn actor_critic --ignore-saved-models
```

## Notes
- Real datasets are downloaded automatically into `data/` by default. Set `DQB_DATA_ROOT` to keep the cache outside the repository.
- Dendritic runs are expected to produce additional PerforatedAI sidecar artifacts.
- If you add new models or conditions, update `src/dendritic_benchmark/specs.py` and rerun the CLI.
- Conditions are executed in dependency order — omitting an upstream condition will cause its dependents to be skipped.
- `generate_graphs` only rebuilds per-condition training plots. Use `compare` to create or refresh `comparison/`.
