# Dendritic Quantization Benchmark: 15 New Models & Extended Experiments
## Executive Summary
The first 10-model round reveals three distinct behavioral clusters. **Dendrites deliver large, consistent gains** in reinforcement learning (Actor-Critic: +14.3%), molecular property prediction (MPNN: +15.6%), and audio classification (M5: +4.5%), while also rescuing Q4 accuracy in time-series forecasting (LSTM Forecaster's Q4 normalized score jumps from 45.6% → 97.0% — a +51.4 point rescue). **Dendrites are neutral-to-mildly-harmful** for transformers (DistilBERT: −1.1%), graph convolutions (GCN: −0.6%), tabular attention (TabNet: −0.1%), and text CNNs (TextCNN: −0.1%). **Q2 universally collapses** (≤35% normalized score for almost every model regardless of dendrites), while Q4 is the critical threshold where dendrites matter most. A surprising inversion is observed for the LSTM Autoencoder on anomaly detection: Q4 actually *improves* over FP32 (123.5% normalized AUC at base Q4), suggesting quantization noise acts as a structural regularizer for reconstruction-based anomaly detection.[^1]

***
## Part 1: Key Findings from Round 1
### Dendrite Delta at FP32
| Model | Domain | Base FP32 Score | Dendrites FP32 | Δ (pp) | Interpretation |
|---|---|---|---|---|---|
| MPNN | Molecular | 1.036 RMSE | 0.896 RMSE | **+15.6%** | Strongest beneficiary — graph message passing amplified by dendrites |
| Actor-Critic | RL | 0.815 Reward | 0.931 Reward | **+14.3%** | Policy optimization benefits greatly from dendritic temporal routing |
| M5 (1D-CNN) | Audio | 80.99% | 84.60% | **+4.5%** | Temporal feature hierarchies deepen with dendritic compartments |
| LSTM Autoencoder | Anomaly | 77.25% AUC | 78.09% AUC | **+1.1%** | Modest gain; quantization effects dominate |
| LSTM Forecaster | Time-Series | MAE 0.0702 | MAE 0.0695 | **+1.0%** | Small FP32 gain, but enormous Q4 rescue |
| LeNet-5 | Image | 98.84% | 99.05% | +0.2% | Near-saturated — little room for improvement |
| TextCNN | NLP | 90.49% | 90.39% | −0.1% | Saturated embedding model; no dendrite benefit |
| TabNet | Tabular | 85.04% | 84.93% | −0.1% | Attention mechanism doesn't gain from dendrites |
| GCN | Graph | 79.61% | 79.12% | **−0.6%** | Sparse adjacency-based aggregation may conflict with dendritic routing |
| DistilBERT | Large NLP | 82.80% | 81.88% | **−1.1%** | Large transformer self-attention absorbs optimization capacity |
### Q4 Rescue: The Most Actionable Finding
The Q4 rescue magnitude — (Dend+Prune+Q4 normalized %) − (Base Q4 normalized %) — reveals the real-world value of dendritic quantization:[^1]

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

[^1]

The LSTM Forecaster Q4 rescue is extraordinary: base quantized to Q4 barely forecasts (45.6% of FP32 quality), but the dendritic+pruned Q4 model operates near FP32 parity (97.0%). This suggests that for RNN-based time-series, the dendritic compartments preserve temporal dynamics that scalar quantization destroys. Conversely, M5 (audio 1D-CNN) actually performs *worse* under dendritic Q4 vs. base Q4 (−18.6pp), likely because its depthwise convolutional features are disrupted by the dendritic wrappers at low bit-widths.[^1]
### Q2: Universal Collapse Floor
Q2 quantization collapses nearly universally. The only models with Q2 normalized scores above 50% are LSTM Autoencoder (112.1% base Q2 — another inversion), Actor-Critic (92.7%), GCN (84.9%), and DistilBERT (63.0%). Dendritic models provide a slight Q2 rescue only for MPNN (+7.6pp) and LSTM Forecaster (+2.6pp), both RNN/GNN architectures where the Q2 collapse is due to numerical instability rather than information compression. The Q2 floor appears to be a hardware-level artifact of torchao's non-native sub-4-bit path on MPS.[^1][^2]

***
## Part 2: 15 New Models for Round 2
The selection is organized into five strategic groups, each targeting a specific gap or follow-up hypothesis from Round 1. All are straightforwardly implementable in PyTorch, are registered by adding entries to `specs.py`, and are sized for the 12–48h M3 Pro budget.[^3][^4]

***
### Group A: Deeper RL (Probe the RL Dendrite Win)
Actor-Critic on CartPole showed a +14.3% FP32 dendrite gain and solid Q4 rescue (+11.4pp). The key question: does this generalize to harder environments and CNN-based policies?

#### Model 11 — DQN (LunarLander-v2)

| Field | Value |
|---|---|
| **Key** | `dqn_lunarlander` |
| **Domain** | Reinforcement Learning — harder continuous state space |
| **Dataset/Env** | `gymnasium LunarLander-v2` |
| **Complexity** | Small (~50K params) |
| **Architecture** | 3-layer MLP Q-network + target Q-network, replay buffer (50K), ε-greedy, τ-soft update |
| **Metric** | Mean episodic reward (solved ≥ 200) |
| **Est. Training Time** | ~1h per condition (500 episodes) |
| **PAI Notes** | Perforate the Q-network MLP only (not target network); target net is a copy |
| **Scientific Rationale** | LunarLander has 8 continuous state dims vs 4 for CartPole — tests RL dendrite scaling with state complexity. DQN also adds a replay buffer, testing if the memory mechanism interacts with dendritic structure. |

[^5][^6]

#### Model 12 — PPO Policy Network (BipedalWalker-v3)

| Field | Value |
|---|---|
| **Key** | `ppo_bipedalwalker` |
| **Domain** | Reinforcement Learning — continuous action space |
| **Dataset/Env** | `gymnasium BipedalWalker-v3` |
| **Complexity** | Small (~80K params) |
| **Architecture** | Shared backbone MLP + separate actor/critic heads, GAE advantage estimation |
| **Metric** | Mean episodic reward (solved ≥ 300) |
| **Est. Training Time** | ~2h per condition |
| **PAI Notes** | Perforate shared backbone; heads can remain standard |
| **Scientific Rationale** | Tests dendrites in continuous action RL (PPO vs discrete-action A2C). BipedalWalker is significantly harder — do dendrites maintain the +14% advantage under a harder optimization landscape? |

***
### Group B: Molecular/Graph Depth (Probe the MPNN Win)
MPNN on ESOL showed the strongest dendrite gain of any model (+15.6% FP32, +13.2pp Q4 rescue). Testing two follow-ups:

#### Model 13 — AttentiveFP (FreeSolv, Molecular Solvation)

| Field | Value |
|---|---|
| **Key** | `attentivefp_freesolv` |
| **Domain** | Drug Discovery / Molecular Property Prediction |
| **Dataset** | FreeSolv (642 molecules, hydration free energy regression) via `torch_geometric.datasets.MoleculeNet` |
| **Complexity** | Small (~120K params) |
| **Architecture** | Multi-layer graph attention with node/edge features + global readout via attention timesteps |
| **Metric** | RMSE (kcal/mol) |
| **Est. Training Time** | ~0.5h per condition |
| **PAI Notes** | AttentiveFP uses `nn.GRUCell` internally — must add `GPA.pc.append_modules_to_perforate([nn.GRUCell])` before `perforate_model` |
| **Scientific Rationale** | AttentiveFP is architecturally richer than MPNN with explicit attention mechanisms. Tests whether the molecular dendrite win is architecture-agnostic or MPNN-specific. Also tests GRUCell dendritic wrapping, a novel architecture interaction. |

[^7][^8]

#### Model 14 — GIN (IMDB-B, Graph Classification)

| Field | Value |
|---|---|
| **Key** | `gin_imdbb` |
| **Domain** | Graph Classification (Social Networks) |
| **Dataset** | IMDB-Binary via `torch_geometric.datasets.TUDataset` — 1000 graphs, binary classification |
| **Complexity** | Tiny (~30K params) |
| **Architecture** | 3-layer Graph Isomorphism Network with MLP aggregators, global mean pooling |
| **Metric** | Accuracy (10-fold CV) |
| **Est. Training Time** | ~0.3h per condition |
| **PAI Notes** | GIN MLP layers are standard `nn.Linear` — perforation is straightforward |
| **Scientific Rationale** | GIN is provably more expressive than GCN (can distinguish all graph structures that WL-isomorphism can). Tests if the GCN dendrite stagnation is a GNN expressiveness limitation — perhaps more expressive GNNs leave less room for dendritic improvement. |

***
### Group C: Time-Series Depth (Probe the Q4 Rescue)
The LSTM Forecaster's +51pp Q4 rescue is the most striking single result. Two key follow-ups probe whether this is RNN-specific or general to time-series:

#### Model 15 — TCN Forecaster (ETTm1)

| Field | Value |
|---|---|
| **Key** | `tcn_forecaster` |
| **Domain** | Time-Series Forecasting — convolutional (non-RNN) |
| **Dataset** | ETTm1 (same family as ETTh1, 15-min intervals, 7 features) |
| **Complexity** | Small (~200K params) |
| **Architecture** | Dilated causal 1D convolutions with residual blocks (`pytorch-tcn` package), input projection, multi-step output head |
| **Metric** | MAE |
| **Est. Training Time** | ~1h per condition |
| **PAI Notes** | TCN residual blocks contain standard `Conv1d` and `Linear` layers — standard perforation works; `weight_norm` should be removed before quantization |
| **Scientific Rationale** | Critical control: tests if the LSTM Q4 rescue is an RNN recurrence property (preserving hidden state precision) or a general time-series property. If TCN shows no Q4 rescue, recurrence is the key mechanism. If it does, temporal inductive bias broadly benefits dendrites at Q4. |

[^9][^10]

#### Model 16 — GRU Forecaster (Weather Dataset)

| Field | Value |
|---|---|
| **Key** | `gru_forecaster` |
| **Domain** | Time-Series Forecasting — RNN variant |
| **Dataset** | Weather (ETT companion, 21 meteorological features, Informer benchmark split) |
| **Complexity** | Tiny (~25K params) |
| **Architecture** | 2-layer bidirectional GRU, FC projection to multi-step output |
| **Metric** | MAE |
| **Est. Training Time** | ~0.5h per condition |
| **PAI Notes** | Must declare `GPA.pc.append_modules_to_perforate([nn.GRU])` before `perforate_model` |
| **Scientific Rationale** | GRU vs. LSTM comparison: both are gated RNNs, but GRU has no separate cell state. If GRU also shows a Q4 rescue, the cell-state hypothesis is incorrect and the rescue is a gated-unit property. |

[^11]

***
### Group D: Entirely New Domains
These five models add modalities not present in Round 1 — 3D point clouds, generative models, neuromorphic computing, dense prediction, and spatiotemporal data.

#### Model 17 — PointNet (ModelNet40, 3D Classification)

| Field | Value |
|---|---|
| **Key** | `pointnet_modelnet40` |
| **Domain** | 3D Point Cloud Classification |
| **Dataset** | ModelNet40 (12,311 CAD models, 40 classes) via `torch_geometric` |
| **Complexity** | Small (~3.5M params) |
| **Architecture** | T-Net input/feature transform, shared MLP on per-point features, global max pooling, classification MLP |
| **Metric** | Accuracy (%) |
| **Est. Training Time** | ~3h per condition |
| **PAI Notes** | Set-based MLP operations — `nn.Linear` layers in shared MLP are perforatable; T-Net is a small sub-network that can be independently perforated |
| **Scientific Rationale** | PointNet operates on unordered sets — a completely different inductive bias from CNNs, RNNs, or GNNs. Tests if dendritic compartments benefit permutation-invariant architectures, and whether point-cloud spatial features are more or less quantization-robust than 2D image features. |

[^12][^13]

#### Model 18 — VAE (MNIST, Generative)

| Field | Value |
|---|---|
| **Key** | `vae_mnist` |
| **Domain** | Generative Modeling / Unsupervised Representation Learning |
| **Dataset** | MNIST (60K images) |
| **Complexity** | Tiny (~400K params) |
| **Architecture** | FC encoder → (μ, logσ²), reparameterization trick, FC decoder; ELBO loss (reconstruction + KL divergence) |
| **Metric** | ELBO (lower = worse; higher = better) and FID on 1K samples |
| **Est. Training Time** | ~0.5h per condition |
| **PAI Notes** | Perforate encoder and decoder FC layers separately; the stochastic reparameterization node is not a learnable layer and does not need perforating |
| **Scientific Rationale** | Generative ELBO objective is fundamentally different from discriminative cross-entropy. Tests how dendrites interact with continuous latent spaces and whether dendritic structure produces more disentangled latent codes (measurable via interpolation quality). The LSTM Autoencoder result (Q4 inversion) motivates asking: does the VAE show the same regularization effect? |

[^14][^15]

#### Model 19 — Spiking Neural Network (SpikingJelly, N-MNIST)

| Field | Value |
|---|---|
| **Key** | `snn_nmnist` |
| **Domain** | Neuromorphic Computing / Event-Driven Classification |
| **Dataset** | N-MNIST (event-camera MNIST, 60K samples, 2-channel spike trains) via `spikingjelly.datasets` |
| **Complexity** | Tiny (~100K params) |
| **Architecture** | Conv-LIF → Conv-LIF → FC-LIF SNN with leaky integrate-and-fire neurons, T=10 timesteps, surrogate gradient (ATan) |
| **Metric** | Accuracy (%) |
| **Est. Training Time** | ~1h per condition |
| **PAI Notes** | Standard `nn.Conv2d` and `nn.Linear` inside SNN can be perforated; LIF neuron states are managed by SpikingJelly and are separate from PAI — no conflict. Use surrogate gradient-compatible quantization |
| **Scientific Rationale** | **The most biologically motivated experiment in the entire suite.** Biological dendrites and spiking neurons coexist in real neurons. Testing whether PerforatedAI's artificial dendrites amplify or conflict with spiking dynamics is a uniquely neuroscientific question. SNNs are also naturally sparse and low-power, making quantization interactions particularly meaningful. |

[^16][^17][^18]

#### Model 20 — Tiny U-Net (ISIC Skin Lesion Segmentation)

| Field | Value |
|---|---|
| **Key** | `unet_isic` |
| **Domain** | Medical Image Segmentation / Dense Prediction |
| **Dataset** | ISIC 2018 Task 1 (2,594 dermoscopy images, binary lesion mask) — auto-downloadable |
| **Complexity** | Medium (~7M params, encoder-decoder) |
| **Architecture** | 4-level encoder-decoder with skip connections, 2× depth reduction vs standard U-Net (16→32→64→128 channels) |
| **Metric** | Dice coefficient |
| **Est. Training Time** | ~4h per condition |
| **PAI Notes** | Skip connections cross encode-decode path — use `PAISequential` to wrap each encoder block's Conv-BN-ReLU triplet together. Decoder upsampling blocks are perforatable independently |
| **Scientific Rationale** | Tests dendrites on a dense prediction task (every pixel classified). U-Net has both local (encoder) and global (skip-connected decoder) feature reuse — uniquely tests whether dendritic compartments benefit skip-connection architectures. Also a medium-complexity anchor between the tiny models and DistilBERT. |

***
### Group E: Architecture Interaction Studies
These five models test how dendrites interact with specific structural patterns: residual connections, depthwise convolutions, inter-sample attention, dynamic routing, and spatiotemporal convolutions.

#### Model 21 — ResNet-18 (CIFAR-10)

| Field | Value |
|---|---|
| **Key** | `resnet18_cifar10` |
| **Domain** | Image Classification — residual architecture |
| **Dataset** | CIFAR-10 (50K/10K, 10 classes) via `torchvision` |
| **Complexity** | Medium (~11M params) |
| **Architecture** | Standard ResNet-18 with modified first conv (3×3 kernel, stride 1 for 32×32 input), `torchvision.models.resnet18(num_classes=10)` |
| **Metric** | Accuracy (%) — baseline ~93% achievable[^19] |
| **Est. Training Time** | ~2h per condition |
| **PAI Notes** | Residual `BasicBlock` contains two `Conv2d` + `BatchNorm` — perforate each block as a unit using `PAISequential`; skip connection projection layers should remain unperforated |
| **Scientific Rationale** | ResNet skip connections create gradient highways that may interact with PAI's cascade-correlation training (which alternates neuron vs dendrite phases). Tests whether residual architectures benefit from, or are disrupted by, dendritic compartments. Expected baseline quality is higher than LeNet-5, so the "saturation ceiling" is a fairer comparison. |

[^20][^19]

#### Model 22 — MobileNetV2 (CIFAR-10)

| Field | Value |
|---|---|
| **Key** | `mobilenetv2_cifar10` |
| **Domain** | Image Classification — efficient depthwise-separable |
| **Dataset** | CIFAR-10 |
| **Complexity** | Small (~2.2M params) |
| **Architecture** | MobileNetV2 inverted residual bottlenecks, modified for 32×32 inputs |
| **Metric** | Accuracy (%) |
| **Est. Training Time** | ~1.5h per condition |
| **PAI Notes** | Depthwise `Conv2d` (groups=in_channels) can be perforated but produces many very thin layers — set `min_channels_to_perforate=16` in PAI config to avoid wrapping 1-channel depthwise ops |
| **Scientific Rationale** | Depthwise separable convolutions factorize standard convolutions into spatial and channel mixing. Tests if dendritic compartments that operate on full-rank weight matrices benefit or conflict with factored weights. MobileNetV2 also uses ReLU6 activations and linear bottlenecks — different quantization sensitivity profile than standard ReLU networks. |

#### Model 23 — SAINT (Adult Income, Tabular Transformer)

| Field | Value |
|---|---|
| **Key** | `saint_adult` |
| **Domain** | Tabular Classification — self + inter-sample attention |
| **Dataset** | Adult Income (same as TabNet — allows direct cross-model comparison) |
| **Complexity** | Small (~500K params, depth=2) |
| **Architecture** | Feature embedding, column-wise self-attention transformer block, row-wise inter-sample attention block, classification head |
| **Metric** | Accuracy (%) |
| **Est. Training Time** | ~1h per condition |
| **PAI Notes** | Use `GPA.pc.append_modules_to_perforate([nn.MultiheadAttention])` for both self-attention and inter-sample attention blocks; embedding layers can remain standard |
| **Scientific Rationale** | TabNet's dendritic stagnation was potentially an architectural mismatch — TabNet's sequential attention is quite different from how PAI expects to perforate a model. SAINT uses standard `nn.MultiheadAttention` (natively perforatable) and is tested on the same dataset as TabNet, enabling a controlled comparison. Hypothesis: attention-over-features (SAINT) benefits more from dendrites than sequential attention (TabNet). |

[^21][^22][^23][^24][^25]

#### Model 24 — Capsule Network (CapsNet, MNIST)

| Field | Value |
|---|---|
| **Key** | `capsnet_mnist` |
| **Domain** | Image Classification — equivariant dynamic routing |
| **Dataset** | MNIST |
| **Complexity** | Small (~8M params) |
| **Architecture** | Conv feature detector → PrimaryCaps → DigitCaps with routing-by-agreement (3 iterations), reconstruction decoder for regularization |
| **Metric** | Accuracy (%) |
| **Est. Training Time** | ~1.5h per condition |
| **PAI Notes** | CapsNet's routing-by-agreement uses no learnable weights in the routing step — only the capsule transformation matrices (`W`) in DigitCaps are perforatable; Conv layer is also standard |
| **Scientific Rationale** | CapsNet is architecturally the most distinct model in the entire benchmark — it uses dynamic routing instead of backpropagation-only weight updates for part-whole relationships. The interplay between routing-by-agreement and PAI's cascade-correlation dendrite addition is entirely unexplored. Both are biologically motivated alternatives to standard backprop, making this a uniquely interesting combination. |

#### Model 25 — ConvLSTM (Moving MNIST, Spatiotemporal)

| Field | Value |
|---|---|
| **Key** | `convlstm_movingmnist` |
| **Domain** | Spatiotemporal Sequence Prediction |
| **Dataset** | Moving MNIST (10K sequences of 2 moving digits, predict next 10 frames from 10 input frames) |
| **Complexity** | Small (~500K params) |
| **Architecture** | 2-layer ConvLSTM (64 filters, 3×3 kernel), frame decoder, MSE reconstruction loss |
| **Metric** | SSIM (structural similarity on predicted frames) |
| **Est. Training Time** | ~2h per condition |
| **PAI Notes** | ConvLSTM's recurrent kernel is a `nn.Conv2d` internally — declare `GPA.pc.append_modules_to_perforate([ConvLSTM2d])` for custom cell; gate weight tensors are the primary targets |
| **Scientific Rationale** | Bridges the two most dendrite-friendly domains: temporal sequences (like LSTM Forecaster) and spatial structure (like LeNet-5). If dendrites benefit both independently, the spatiotemporal case should show the strongest combined benefit. This is also the only prediction/generation task that involves spatial coherence as a metric, not just classification. |

***
## Part 3: Complete 25-Model Extended Roster
| # | Key | Domain | Dataset | ~Params | Est. Time/cond |
|---|---|---|---|---|---|
| 1 | `lenet5` ✓ | Image (tiny CNN) | MNIST | 60K | 0.5h |
| 2 | `m5` ✓ | Audio (1D-CNN) | SpeechCommands | 35K | 41min |
| 3 | `lstm_forecaster` ✓ | Time-Series (RNN) | ETTh1 | 17K | 10sec |
| 4 | `textcnn` ✓ | NLP (Text CNN) | AG News | 739K | 3min |
| 5 | `gcn` ✓ | Graph (Conv) | Cora | 92K | 10sec |
| 6 | `tabnet` ✓ | Tabular (Seq Att) | Adult Income | 5K | 13sec |
| 7 | `mpnn` ✓ | Molecular (GNN) | ESOL | 34K | 5sec |
| 8 | `actor_critic` ✓ | RL (CartPole) | CartPole-v1 | 5K | 20sec |
| 9 | `lstm_autoencoder` ✓ | Anomaly Detect | MIT-BIH ECG | 34K | ~1min |
| 10 | `distilbert` ✓ | Large NLP (Xfmr) | SST-2 | 839K | ~1h |
| 11 | `dqn_lunarlander` **NEW** | RL (CNN Q-net) | LunarLander-v2 | ~50K | ~1h |
| 12 | `ppo_bipedalwalker` **NEW** | RL (continuous) | BipedalWalker-v3 | ~80K | ~2h |
| 13 | `attentivefp_freesolv` **NEW** | Molecular (Att-GNN) | FreeSolv | ~120K | ~0.5h |
| 14 | `gin_imdbb` **NEW** | Graph Classif. | IMDB-B | ~30K | ~0.3h |
| 15 | `tcn_forecaster` **NEW** | Time-Series (TCN) | ETTm1 | ~200K | ~1h |
| 16 | `gru_forecaster` **NEW** | Time-Series (GRU) | Weather | ~25K | ~0.5h |
| 17 | `pointnet_modelnet40` **NEW** | 3D Point Cloud | ModelNet40 | ~3.5M | ~3h |
| 18 | `vae_mnist` **NEW** | Generative (VAE) | MNIST | ~400K | ~0.5h |
| 19 | `snn_nmnist` **NEW** | Neuromorphic SNN | N-MNIST | ~100K | ~1h |
| 20 | `unet_isic` **NEW** | Medical Seg. | ISIC 2018 | ~7M | ~4h |
| 21 | `resnet18_cifar10` **NEW** | Image (ResNet) | CIFAR-10 | ~11M | ~2h |
| 22 | `mobilenetv2_cifar10` **NEW** | Image (Efficient) | CIFAR-10 | ~2.2M | ~1.5h |
| 23 | `saint_adult` **NEW** | Tabular (Xfmr) | Adult Income | ~500K | ~1h |
| 24 | `capsnet_mnist` **NEW** | Image (CapsNet) | MNIST | ~8M | ~1.5h |
| 25 | `convlstm_movingmnist` **NEW** | Spatiotemporal | Moving MNIST | ~500K | ~2h |

**Estimated total Round 2 training time** (13 conditions × 15 new models): ~65–90h at M3 Pro throughput. Recommended to run the tiny/small models (11–16, 18–19, 22–23) in a single overnight batch (~20–25h), then PointNet, U-Net, ResNet18, CapsNet, ConvLSTM, and the two RL models on separate days.

***
## Part 4: Additional Experiments Beyond New Models
These experiments extract more insight from the existing 10 models without requiring full re-training and can be interleaved with Round 2 runs.
### Experiment A — Pruning Rate Sweep (MPNN & Actor-Critic)
**What:** Re-run `dendrites_pruned` and all dendritic quantized conditions at sparsity levels {10%, 20%, 30%, 40% (current), 50%, 60%, 70%} for MPNN and Actor-Critic.

**Why:** The current 40% fixed pruning rate was chosen heuristically. Given that MPNN shows a +13.2pp Q4 rescue, the Pareto-optimal prune rate for each bit-width may differ substantially. A sweep reveals whether more aggressive pruning *further enhances* the Q4 rescue (sparsity + quantization synergy) or degrades it.

**Implementation:** Modular — add a `pruning_amount` parameter to `CONDITION_SPECS` in `specs.py`. Run as separate condition keys (`dendrites_prune20`, `dendrites_prune60`, etc.).
### Experiment B — Dendrite Cycle Count Ablation
**What:** For Actor-Critic and MPNN, vary `GPA.pc.set_max_dendrites(N)` across N ∈ {1, 2, 3, 4, 5} and record the Q4/Q8 normalized score at each cycle count.[^2]

**Why:** The PAI library adds dendrites in cycles — each cycle grows the network and re-trains. More cycles add parameters but also increase quantization error. This experiment finds the minimum cycle count that still achieves the Q4 rescue, potentially reducing Round 3 training time substantially.

**Implementation:** Override `set_fixed_switch_num` for each cycle count; compare `best_arch_scores.csv` across conditions.
### Experiment C — QAT-Integrated Dendritic Training (for Q2 Rescue)
**What:** For MPNN, Actor-Critic, and LSTM Forecaster, run QAT (quantization-aware training using `torchao`'s QAT path) *inside* the PAI dendritic training loop — project weights to Q4/Q2 representations during forward passes while still allowing full-precision gradient accumulation.

**Why:** The current pipeline uses PTQ: train in FP32, then quantize afterward. QAT allows the dendritic structure to adapt to quantization constraints *during formation*, which may unlock Q2 rescue currently unavailable via PTQ. The `torchao` QAT path supports this pattern.[^2]

**Implementation:** In `compat.py`, apply `torchao`'s fake-quantization wrapper before calling `perforate_model`, then proceed with the normal PAI loop.
### Experiment D — Structured vs. Unstructured Pruning Comparison
**What:** For MPNN, Actor-Critic, and LSTM Forecaster, compare the current L1 unstructured global pruning at 40% sparsity against L2 structured (channel-level) pruning at equivalent parameter reduction.

**Why:** Unstructured sparsity produces irregular weight tensors that are hard to accelerate on hardware, while structured pruning removes entire channels (true speedup on MPS/CPU). The Q4 rescue may be partly attributable to *which weights* are pruned — structured pruning changes the network topology more fundamentally, potentially interacting differently with dendritic compartments.

**Implementation:** Add `structured` flag to `CONDITION_SPECS`; use `prune.ln_structured` with `n=2, dim=0` for channel pruning in `training.py`.
### Experiment E — Inference Latency Benchmarking on M3 Pro
**What:** For the 5 best Q4 conditions per model, measure actual wall-clock inference latency on M3 Pro (batch size 1 and 32) using `torch.utils.benchmark.Timer`.

**Why:** The current metrics (accuracy, param count, file size) do not capture real-world speedup. A Q4 dendritic model with 97% accuracy of FP32 is only valuable in deployment if it is actually faster. On Apple Silicon, `torchao` Q4 dispatches to Metal Performance Shaders kernels that may or may not be faster than FP32 for small models.[^2]

**Implementation:** Add `dqb bench` command to `cli.py` that loads `model.pt` files and runs `Timer` sweeps; no retraining required.
### Experiment F — Dataset Difficulty Scaling (LSTM Forecaster)
**What:** Run the full 13-condition suite for LSTM Forecaster on four progressively harder datasets: ETTh1 (current, hourly, 1 var), ETTh2, ETTm1 (15-min), ETTm2, and the multivariate Weather dataset.

**Why:** The +51.4pp Q4 rescue on ETTh1 is remarkable. Does this rescue magnitude *increase* with dataset complexity (harder sequences need more representational capacity, which dendrites provide) or *decrease* (harder datasets amplify all approximation errors, including quantization)? This is a dataset-difficulty ablation of the central finding.

**Implementation:** Add `etth2`, `ettm1`, `ettm2`, `weather` as dataset keys in `data.py`, reuse the `lstm_forecaster` model key.
### Experiment G — Anomaly Detection Regularization Study
**What:** For LSTM Autoencoder, add Gaussian noise injection at σ ∈ {0.01, 0.05, 0.10, 0.20} during training and measure AUC — comparing to the Q4 quantization noise effect.

**Why:** The Q4 inversion (123.5% normalized AUC) suggests quantization noise acts as a regularizer for reconstruction-based anomaly detection. This experiment validates the hypothesis directly: if Gaussian noise at the right σ level also improves AUC, then quantization is functioning as implicit noise injection, not as a fundamentally different mechanism.

**Implementation:** Add a `noise_sigma` parameter to the training loop in `training.py`; run as separate condition keys outside the main 13-condition suite.
### Experiment H — Cross-Architecture Tabular Comparison (SAINT vs TabNet)
**What:** Once SAINT (`saint_adult`) is trained in Round 2, plot a side-by-side comparison of all 13 conditions between SAINT, TabNet, and XGBoost baseline performance on Adult Income.

**Why:** Both SAINT and TabNet failed to benefit from dendrites in Round 1, but they use fundamentally different mechanisms (sequential feature attention vs. transformer attention). If SAINT also shows no dendrite benefit, the null result is robust for tabular data. If SAINT benefits, the TabNet failure was architectural, not domain-specific.

***
## Part 5: `specs.py` Integration Notes
To add the 15 new models, append to `MODEL_SPECS` in `src/dendritic_benchmark/specs.py`:[^4]

```python
# Example entry for dqn_lunarlander
ModelSpec(
    key="dqn_lunarlander",
    display_name="DQN (LunarLander)",
    domain="Reinforcement Learning",
    dataset_key="lunarlander_v2",
    metric_name="Reward",
    metric_mode="max",     # higher is better
    param_budget_epochs=500,  # episodes, not epochs
    pai_modules=[nn.Linear],
)
```

Models with non-standard metrics require updates to `results.py`'s `normalize_score()` function:
- **VAE**: ELBO is negative — invert sign for normalization
- **U-Net**: Dice is already  bounded — use directly[^26]
- **SSIM** (ConvLSTM): already  — use directly[^26]
- **SNN**: standard accuracy — no changes needed

For environment-based RL (DQN, PPO), `data.py` should initialize the `gymnasium` environment rather than downloading a dataset; the `train_and_evaluate` function in `training.py` needs an `is_rl_env=True` branch that runs episode rollouts instead of dataloader batches.[^4]

***
## Conclusion
Round 1 produced three clean scientific findings: (1) dendritic models provide the largest gains in domains with strong temporal dynamics and continuous optimization landscapes (RL, molecular, audio); (2) dendrites rescue Q4 accuracy specifically in RNN-based time-series, suggesting recurrent hidden-state precision is the mechanism; (3) Q2 is a near-universal floor not addressable by dendrites alone, requiring QAT from the start. The 15 new models are selected to stress-test each of these findings across new architectures and domains — the RL depth (DQN, PPO), molecular attention (AttentiveFP), time-series architecture independence (TCN, GRU), and entirely new modalities (3D, generative, neuromorphic, spatiotemporal, medical) — while the 8 additional experiments extract maximum insight from existing results without redundant compute. The complete 25-model suite with 13 conditions yields 325 training runs, providing strong statistical power for cross-domain claims about dendritic quantization robustness.

---

## References

1. [summary.csv](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/74179011/3329b3d0-a6c9-49ec-bd41-b527e917bd52/summary.csv?AWSAccessKeyId=ASIA2F3EMEYE5C2BQYZV&Signature=pxcc8mE3iGR0i8GBBb6xzSuXWNE%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEEkaCXVzLWVhc3QtMSJIMEYCIQC9dlKi3cW1LvVqWRpFQpHk0i7HiqWdRe848bw7WaiN7gIhAJqXQyE528l4bBVUaaO4to4b0cwEPxrIQNLWCEYk%2FdUWKvMECBIQARoMNjk5NzUzMzA5NzA1Igy8QcfuUk1lK1SKQBoq0ATp3JoufaRHVxfoZQURbEK2aVr9McSXKM4kWNJ2g56P4oMdbI4OiNqBxbh0HGrIQTkbroNgqdrl6ornrb86viBCoq8O7rvuWMu3Xz57beWNkzX%2FcYFW5e5Bs1cCNDvsZnj7G7hIIXVTKiLCPERYAARWUIqg1yO8NSscAWLA8lNmcZ3K4%2FPwsSJdyxPNFBBYPZz2aKX96cnVk0Lq362ZkI%2FJWGEw7ycmtEXsNnolCRYzkTlIdCzzPFB%2Bgn5cHVL8p0NRS91ZCuJru8AfBplkDxAT5I2OnKN0zssZMIkRnWYLeV00T5nhnrIgw4JfxFrWgtRfkNx5tQErrCvEgmglnKR%2FP8hVqZpCSGN5aae6tC%2FsHIKMNtLk%2BYG20n1lwJSxPkgv0eh0IoFeNfkthF%2FGk0LFaWksJGiG8KqIqu5D%2FHylpjIPeuRKfZ6Fy2GHVK0HbhJkbF048QZv%2B7IpSpa2yq6y%2B5xrUj5ox2QBFeRXvy0Pin64zId0G7HO%2FfuOkE1IHt7jocpEKQwlVTl4E%2Fb7mEIvM1EqrllveWNNtW3KqjXBi%2FBETIHfn4xzNf6kYj8HTVnyDvghIcEXJXR%2BTwW3HMsEJvfPYX3CSJP2eFipqnUA1ilYQlmBtJYGNsz%2BzIfbFE8fUM3dAwDpm7s8CYFkLEoZrlK%2FaELcPPWyfnalpSDvBPr2TjNjze1%2Bjw24rb27v3HvTI9k9qUVD4mOBY8JGOxcO5V30BcGw9xLG6WB2AJ72kJUgDFmfWkaRcrW371euPqNHxs3H1OMnPcPNYpBzSsKMJGIzs8GOpcBX19tZMHrlySFzlY5lcfI%2FFVYx5W6rl7zh2D%2Fu%2F0D%2FjK6cAfT%2Bq88LFBm56VDDcjUHTqvZzTn4woOCGeYRy%2BTJgLFS7M1wiP6mufa44CyxVAONG0hoM%2B55qMw62daiAKhasJHqmTWeiqr7Ebm%2Bilf8NUb4CVuOdTEuCQLslFQ%2BYEvsB2PzFMjjanw6UyfbZ7GArUVbHVqhA%3D%3D&Expires=1777570276) - model_key,condition_key,metric_name,metric_value,normalized_score_percent,size_reduction_percent,fil...

2. [Dendritic-Quantization-Benchmark-Plan.md](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/74179011/d44cf2b0-99a1-4ca4-9c02-fcd984288af0/Dendritic-Quantization-Benchmark-Plan.md?AWSAccessKeyId=ASIA2F3EMEYE5C2BQYZV&Signature=6SmohQ7%2BVUMLzbDaRy668a3Trvk%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEEkaCXVzLWVhc3QtMSJIMEYCIQC9dlKi3cW1LvVqWRpFQpHk0i7HiqWdRe848bw7WaiN7gIhAJqXQyE528l4bBVUaaO4to4b0cwEPxrIQNLWCEYk%2FdUWKvMECBIQARoMNjk5NzUzMzA5NzA1Igy8QcfuUk1lK1SKQBoq0ATp3JoufaRHVxfoZQURbEK2aVr9McSXKM4kWNJ2g56P4oMdbI4OiNqBxbh0HGrIQTkbroNgqdrl6ornrb86viBCoq8O7rvuWMu3Xz57beWNkzX%2FcYFW5e5Bs1cCNDvsZnj7G7hIIXVTKiLCPERYAARWUIqg1yO8NSscAWLA8lNmcZ3K4%2FPwsSJdyxPNFBBYPZz2aKX96cnVk0Lq362ZkI%2FJWGEw7ycmtEXsNnolCRYzkTlIdCzzPFB%2Bgn5cHVL8p0NRS91ZCuJru8AfBplkDxAT5I2OnKN0zssZMIkRnWYLeV00T5nhnrIgw4JfxFrWgtRfkNx5tQErrCvEgmglnKR%2FP8hVqZpCSGN5aae6tC%2FsHIKMNtLk%2BYG20n1lwJSxPkgv0eh0IoFeNfkthF%2FGk0LFaWksJGiG8KqIqu5D%2FHylpjIPeuRKfZ6Fy2GHVK0HbhJkbF048QZv%2B7IpSpa2yq6y%2B5xrUj5ox2QBFeRXvy0Pin64zId0G7HO%2FfuOkE1IHt7jocpEKQwlVTl4E%2Fb7mEIvM1EqrllveWNNtW3KqjXBi%2FBETIHfn4xzNf6kYj8HTVnyDvghIcEXJXR%2BTwW3HMsEJvfPYX3CSJP2eFipqnUA1ilYQlmBtJYGNsz%2BzIfbFE8fUM3dAwDpm7s8CYFkLEoZrlK%2FaELcPPWyfnalpSDvBPr2TjNjze1%2Bjw24rb27v3HvTI9k9qUVD4mOBY8JGOxcO5V30BcGw9xLG6WB2AJ72kJUgDFmfWkaRcrW371euPqNHxs3H1OMnPcPNYpBzSsKMJGIzs8GOpcBX19tZMHrlySFzlY5lcfI%2FFVYx5W6rl7zh2D%2Fu%2F0D%2FjK6cAfT%2Bq88LFBm56VDDcjUHTqvZzTn4woOCGeYRy%2BTJgLFS7M1wiP6mufa44CyxVAONG0hoM%2B55qMw62daiAKhasJHqmTWeiqr7Ebm%2Bilf8NUb4CVuOdTEuCQLslFQ%2BYEvsB2PzFMjjanw6UyfbZ7GArUVbHVqhA%3D%3D&Expires=1777570276) - # Dendritic Quantization Benchmark: 10-Model, 10-Domain Experiment Plan
## Overview
This experiment ...

3. [SCRIPT_USAGE.md](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/74179011/bdcddb5e-e0d6-4c7e-a37c-5fd9b57e9a4e/SCRIPT_USAGE.md?AWSAccessKeyId=ASIA2F3EMEYE5C2BQYZV&Signature=NA6j4TmxnpOdS9JnS2cpzWTbHh8%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEEkaCXVzLWVhc3QtMSJIMEYCIQC9dlKi3cW1LvVqWRpFQpHk0i7HiqWdRe848bw7WaiN7gIhAJqXQyE528l4bBVUaaO4to4b0cwEPxrIQNLWCEYk%2FdUWKvMECBIQARoMNjk5NzUzMzA5NzA1Igy8QcfuUk1lK1SKQBoq0ATp3JoufaRHVxfoZQURbEK2aVr9McSXKM4kWNJ2g56P4oMdbI4OiNqBxbh0HGrIQTkbroNgqdrl6ornrb86viBCoq8O7rvuWMu3Xz57beWNkzX%2FcYFW5e5Bs1cCNDvsZnj7G7hIIXVTKiLCPERYAARWUIqg1yO8NSscAWLA8lNmcZ3K4%2FPwsSJdyxPNFBBYPZz2aKX96cnVk0Lq362ZkI%2FJWGEw7ycmtEXsNnolCRYzkTlIdCzzPFB%2Bgn5cHVL8p0NRS91ZCuJru8AfBplkDxAT5I2OnKN0zssZMIkRnWYLeV00T5nhnrIgw4JfxFrWgtRfkNx5tQErrCvEgmglnKR%2FP8hVqZpCSGN5aae6tC%2FsHIKMNtLk%2BYG20n1lwJSxPkgv0eh0IoFeNfkthF%2FGk0LFaWksJGiG8KqIqu5D%2FHylpjIPeuRKfZ6Fy2GHVK0HbhJkbF048QZv%2B7IpSpa2yq6y%2B5xrUj5ox2QBFeRXvy0Pin64zId0G7HO%2FfuOkE1IHt7jocpEKQwlVTl4E%2Fb7mEIvM1EqrllveWNNtW3KqjXBi%2FBETIHfn4xzNf6kYj8HTVnyDvghIcEXJXR%2BTwW3HMsEJvfPYX3CSJP2eFipqnUA1ilYQlmBtJYGNsz%2BzIfbFE8fUM3dAwDpm7s8CYFkLEoZrlK%2FaELcPPWyfnalpSDvBPr2TjNjze1%2Bjw24rb27v3HvTI9k9qUVD4mOBY8JGOxcO5V30BcGw9xLG6WB2AJ72kJUgDFmfWkaRcrW371euPqNHxs3H1OMnPcPNYpBzSsKMJGIzs8GOpcBX19tZMHrlySFzlY5lcfI%2FFVYx5W6rl7zh2D%2Fu%2F0D%2FjK6cAfT%2Bq88LFBm56VDDcjUHTqvZzTn4woOCGeYRy%2BTJgLFS7M1wiP6mufa44CyxVAONG0hoM%2B55qMw62daiAKhasJHqmTWeiqr7Ebm%2Bilf8NUb4CVuOdTEuCQLslFQ%2BYEvsB2PzFMjjanw6UyfbZ7GArUVbHVqhA%3D%3D&Expires=1777570276) - # Script Usage Guide

This repository exposes a small command-line interface for running the dendrit...

4. [SCRIPT_ARCHITECTURE.md](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/74179011/43bd0d09-9c4b-4bcb-ba8c-e11185147f2e/SCRIPT_ARCHITECTURE.md?AWSAccessKeyId=ASIA2F3EMEYE5C2BQYZV&Signature=iTzy861PbawC8qj9mQmQOlMI9Yo%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEEkaCXVzLWVhc3QtMSJIMEYCIQC9dlKi3cW1LvVqWRpFQpHk0i7HiqWdRe848bw7WaiN7gIhAJqXQyE528l4bBVUaaO4to4b0cwEPxrIQNLWCEYk%2FdUWKvMECBIQARoMNjk5NzUzMzA5NzA1Igy8QcfuUk1lK1SKQBoq0ATp3JoufaRHVxfoZQURbEK2aVr9McSXKM4kWNJ2g56P4oMdbI4OiNqBxbh0HGrIQTkbroNgqdrl6ornrb86viBCoq8O7rvuWMu3Xz57beWNkzX%2FcYFW5e5Bs1cCNDvsZnj7G7hIIXVTKiLCPERYAARWUIqg1yO8NSscAWLA8lNmcZ3K4%2FPwsSJdyxPNFBBYPZz2aKX96cnVk0Lq362ZkI%2FJWGEw7ycmtEXsNnolCRYzkTlIdCzzPFB%2Bgn5cHVL8p0NRS91ZCuJru8AfBplkDxAT5I2OnKN0zssZMIkRnWYLeV00T5nhnrIgw4JfxFrWgtRfkNx5tQErrCvEgmglnKR%2FP8hVqZpCSGN5aae6tC%2FsHIKMNtLk%2BYG20n1lwJSxPkgv0eh0IoFeNfkthF%2FGk0LFaWksJGiG8KqIqu5D%2FHylpjIPeuRKfZ6Fy2GHVK0HbhJkbF048QZv%2B7IpSpa2yq6y%2B5xrUj5ox2QBFeRXvy0Pin64zId0G7HO%2FfuOkE1IHt7jocpEKQwlVTl4E%2Fb7mEIvM1EqrllveWNNtW3KqjXBi%2FBETIHfn4xzNf6kYj8HTVnyDvghIcEXJXR%2BTwW3HMsEJvfPYX3CSJP2eFipqnUA1ilYQlmBtJYGNsz%2BzIfbFE8fUM3dAwDpm7s8CYFkLEoZrlK%2FaELcPPWyfnalpSDvBPr2TjNjze1%2Bjw24rb27v3HvTI9k9qUVD4mOBY8JGOxcO5V30BcGw9xLG6WB2AJ72kJUgDFmfWkaRcrW371euPqNHxs3H1OMnPcPNYpBzSsKMJGIzs8GOpcBX19tZMHrlySFzlY5lcfI%2FFVYx5W6rl7zh2D%2Fu%2F0D%2FjK6cAfT%2Bq88LFBm56VDDcjUHTqvZzTn4woOCGeYRy%2BTJgLFS7M1wiP6mufa44CyxVAONG0hoM%2B55qMw62daiAKhasJHqmTWeiqr7Ebm%2Bilf8NUb4CVuOdTEuCQLslFQ%2BYEvsB2PzFMjjanw6UyfbZ7GArUVbHVqhA%3D%3D&Expires=1777570276) - # Script Architecture Guide

This guide explains how the benchmark code is organized and how data mo...

5. [Pytorch implementation of DQN on openai's lunar lander ... - GitHub](https://github.com/Jason-CKY/lunar_lander_DQN) - Pytorch implementation of deep Q-learning on the openAI lunar lander environment · 1: fire left orie...

6. [Deep Q-Network (DQN) on LunarLander-v2 - Chan`s Jupyter](https://goodboychan.github.io/python/reinforcement_learning/pytorch/udacity/2021/05/07/DQN-LunarLander.html) - In this post, We will take a hands-on-lab of Simple Deep Q-Network (DQN) on openAI LunarLander-v2 en...

7. [pytorch_geometric/examples/attentive_fp.py at master · pyg-team/pytorch_geometric](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/attentive_fp.py) - Graph Neural Network Library for PyTorch. Contribute to pyg-team/pytorch_geometric development by cr...

8. [torch_geometric.nn.models.AttentiveFP](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.AttentiveFP.html)

9. [pytorch-tcn - PyPI](https://pypi.org/project/pytorch-tcn/1.1.0/) - This python package provides a flexible and comprehensive implementation of temporal convolutional n...

10. [GitHub - paul-krug/pytorch-tcn: (Realtime) Temporal Convolutions in ...](https://github.com/paul-krug/pytorch-tcn) - The TCN class provides a flexible and comprehensive implementation of temporal convolutional neural ...

11. [Multi-step time series forecasting - PyTorch Forums](https://discuss.pytorch.org/t/multi-step-time-series-forecasting/172290) - I'm currently developing a multi-step time series forecasting model by using a GRU (or also a bidire...

12. [GitHub - yanx27/Pointnet_Pointnet2_pytorch: PointNet and PointNet++ implemented by pytorch (pure python) and on ModelNet, ShapeNet and S3DIS.](https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/master) - PointNet and PointNet++ implemented by pytorch (pure python) and on ModelNet, ShapeNet and S3DIS. - ...

13. [PointNet and PointNet++ implemented by pytorch (pure ...](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) - PointNet and PointNet++ implemented by pytorch (pure python) and on ModelNet, ShapeNet and S3DIS. - ...

14. [Assessing a Variational Autoencoder on MNIST using Pytorch](https://maurocamaraescudero.netlify.app/post/assessing-a-variational-autoencoder-on-mnist-using-pytorch/) - Learn how one can split the VAE into an encoder and decoder to perform various tasks such as: testin...

15. [My VAE PyTorch Implementation for MNIST - vision](https://discuss.pytorch.org/t/my-vae-pytorch-implementation-for-mnist/224125) - python. import torch. import torch.nn.functional as F. from torch import nn. import torch. from torc...

16. [GitHub - ridgerchu/spikingjelly-1: SpikingJelly is an open-source deep learning framework for Spiking Neural Network (SNN) based on PyTorch.](https://github.com/ridgerchu/spikingjelly-1) - SpikingJelly is an open-source deep learning framework for Spiking Neural Network (SNN) based on PyT...

17. [spikingjelly](https://pypi.org/project/spikingjelly/0.0.0.0.2/) - A Spiking Neural Networks simulator built on PyTorch.

18. [fangwei123456/spikingjelly](https://github.com/fangwei123456/spikingjelly) - SpikingJelly is an open-source deep learning framework for Spiking Neural Network (SNN) based on PyT...

19. [kuangliu/pytorch-cifar: 95.47% on CIFAR10 with PyTorch - GitHub](https://github.com/kuangliu/pytorch-cifar) - Train CIFAR10 with PyTorch. I'm playing with PyTorch on the CIFAR10 dataset. Prerequisites Training ...

20. [️ Training ResNet-18 for CIFAR-10 Image Classification - GitHub](https://github.com/deepmancer/resnet-cifar-classification) - This project implements ResNet-18 from scratch in PyTorch and trains it on the CIFAR-10 dataset to a...

21. [SAINT; Improved Neural Networks for Tabular Data via Row ...](https://seunghan96.github.io/tab/cl/(Tab_paper3)SAINT/) - SAINT ( Self-Attention and Intersample Attention Transformer )Permalink leverages several mechanisms...

22. [SAINT: Improved Neural Networks for Tabular Data via Row ... - arXiv](https://arxiv.org/abs/2106.01342) - We devise a hybrid deep learning approach to solving tabular data problems. Our method, SAINT, perfo...

23. [SAINT: Improved Neural Networks for Tabular Data](https://arxiv.org/pdf/2106.01342.pdf)

24. [SAINT: Improved Neural Networks for Tabular Data via Row ...](https://openreview.net/forum?id=nL2lDlsrZU) - SAINT uses both self-attention among variables and inter-sample attention among different samples. U...

25. [somepago/saint: The official PyTorch implementation of recent paper](https://github.com/somepago/saint) - The official PyTorch implementation of recent paper - SAINT: Improved Neural Networks for Tabular Da...

26. [liuchongming1999/Dendritic-integration-inspired-CNN-NeurIPS-2024](https://github.com/liuchongming1999/Dendritic-integration-inspired-CNN-NeurIPS-2024) - This repository is the official implementation of Dendritic Integration Inspired Artificial Neural N...

