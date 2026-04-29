# Dendritic Quantization Benchmark: 10-Model, 10-Domain Experiment Plan
## Overview
This experiment investigates whether quantized dendritic models (created via Perforated Backpropagation) outperform non-dendritic counterparts across diverse fields. Ten models are selected — one per domain — spanning complexities from ~50K to ~66M parameters. Each model is trained and evaluated under **13 experimental conditions**, yielding 130 total training runs.

The hardware target is an **Apple M3 Pro** chip using PyTorch's MPS backend, with a total budget of 12–48 hours. All quantization uses `torchao` (PyTorch-native), pruning uses `torch.nn.utils.prune`, and dendrites are added via the `PerforatedAI` library.[^1][^2][^3][^4][^5][^6][^7][^8]

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

> Models 1–9 are intentionally lightweight to allow all 13 conditions to complete within the 12–48h budget. Model 10 (DistilBERT) serves as the large-model anchor.

***
## Experimental Conditions (13 per model)
Each model is trained/evaluated in the following 13 conditions. Metrics recorded for every condition: **accuracy (or task-equivalent metric)**, **parameter count**, and **model file size on disk**.

| # | Condition Label | Description |
|---|----------------|-------------|
| 1 | **Base FP32** | Vanilla model, no modifications, full float32 precision |
| 2 | **Base + Q8** | Post-training quantization to 8-bit via `torchao` |
| 3 | **Base + Q4** | Post-training quantization to 4-bit |
| 4 | **Base + Q2** | Post-training quantization to 2-bit |
| 5 | **Base + Q1.58** | Ternary quantization {−1, 0, +1} (BitNet-style)[^9][^10] |
| 6 | **Base + Q1** | Binary quantization {−1, +1} |
| 7 | **+Dendrites** | Base model with dendritic compartments via Perforated Backpropagation[^11][^6], FP32 |
| 8 | **+Dend + Prune** | Dendritic model with L1 unstructured pruning (30–50% sparsity)[^1][^2] |
| 9 | **+Dend + Prune + Q8** | Pruned dendritic model, then quantized to 8-bit |
| 10 | **+Dend + Prune + Q4** | Pruned dendritic model, then quantized to 4-bit |
| 11 | **+Dend + Prune + Q2** | Pruned dendritic model, then quantized to 2-bit |
| 12 | **+Dend + Prune + Q1.58** | Pruned dendritic model, ternary quantization |
| 13 | **+Dend + Prune + Q1** | Pruned dendritic model, binary quantization |

***
## Output Graphs (Per Model)
For each of the 10 models, generate **3 comparison bar charts** — one for each metric — with all 13 conditions on the x-axis:
### Graph Set A: Accuracy (or Task Metric)
- Y-axis: Accuracy % (classification), MAE/MSE (regression/forecasting), Reward (RL), AUC (anomaly), ELBO (VAE)
- X-axis: All 13 conditions
- Color coding: Base conditions in blue family, Dendrite conditions in green family
### Graph Set B: Parameter Count
- Y-axis: Number of non-zero parameters (after pruning)
- X-axis: All 13 conditions
- Highlights the structural compression achieved by pruning + quantization
### Graph Set C: Model File Size (MB)
- Y-axis: Saved model size in MB (using `torch.save` or ONNX export)
- X-axis: All 13 conditions
- Shows real storage savings across the quantization spectrum

***
## Cross-Model Comparison Graphs
After all individual runs, produce the following **cross-domain comparison plots**:
### Cross-Graph 1: Accuracy Retention Heatmap (10 × 13)
- Rows = models/domains, Columns = conditions
- Cell value = accuracy as % of the Base FP32 baseline (retention ratio)
- Reveals which domains are most robust to quantization + dendrites
### Cross-Graph 2: Size Reduction vs. Accuracy Tradeoff (scatter)
- X-axis: File size reduction ratio vs. Base FP32
- Y-axis: Accuracy retention (%)
- One point per (model × condition) combination — 130 points total
- Color = domain, shape = whether dendrites are present
### Cross-Graph 3: "Dendrite Delta" Bar Chart (per domain)
- For each domain: side-by-side bars of `Base FP32` vs `+Dendrites FP32` accuracy
- Directly shows the raw dendritic benefit before quantization
### Cross-Graph 4: Best Quantization Level per Domain (heatmap)
- Rows = domains, Columns = quantization levels (FP32, Q8, Q4, Q2, Q1.58, Q1)
- Cell = best accuracy among Base and Dend+Prune variants at that bit level
- Highlights which domains can tolerate extreme quantization

***
## Training Plan (M3 Pro, 12–48h Budget)
The M3 Pro MPS backend supports PyTorch training natively. Dendritic training takes longer than base training because the PAI training loop alternates between neuron and dendrite phases — in practice the total number of epochs is higher than a standard run since the loop continues until `training_complete` is returned. Quantization (post-training or QAT) adds minimal overhead. Pruning + retraining adds ~1 epoch.[^3][^4][^6]

**Important:** The `PerforatedAI` training loop is self-terminating. You should **not** set a fixed epoch count; instead use a `while True` loop and let `training_complete` break it. For budget control, the `DOING_FIXED_SWITCH` mode can be used to bound the number of epochs per phase:[^6]
```python
GPA.pc.set_switch_mode(GPA.pc.DOING_FIXED_SWITCH)
GPA.pc.set_fixed_switch_num(20)       # epochs per dendrite phase
GPA.pc.set_first_fixed_switch_num(40) # allow longer initial neuron phase
```
### Phase 1 — Tiny/Small Models (Models 1–9): ~12–20h total
Run sequentially overnight. All 13 conditions per model. Use `doing_pai=False` in `perforate_model` for all base conditions (conditions 1–6) to skip dendrite overhead entirely.
### Phase 2 — Large Model (DistilBERT, Model 10): ~25–30h total
Run in isolation. For Q1 and Q1.58, use QAT (quantization-aware training) via `torchao` rather than PTQ for better accuracy retention.[^7][^8]
### Recommended Execution Order per Model
1. **Base FP32** → train with `doing_pai=False`; save checkpoint (`final_clean_pai` output or manual `torch.save`)
2. **Base + Q8/Q4/Q2/Q1.58/Q1** → load Base FP32 checkpoint; apply PTQ/QAT via `torchao`; evaluate; no retraining needed for Q8/Q4; use QAT loop for Q2/Q1.58/Q1
3. **+Dendrites FP32** → retrain from scratch with `doing_pai=True`; use `save_name` matching model ID; use `DOING_FIXED_SWITCH` to bound time
4. **+Dend+Prune** → load best dendritic checkpoint (`best_model` or `final_clean_pai`); apply L1 unstructured global pruning at 40% sparsity; fine-tune 5 epochs using `set_optimizer_instance`; make pruning permanent before saving
5. **+Dend+Prune+Q8 through Q1** → load pruned dendritic checkpoint; apply Q8→Q4→Q2→Q1.58→Q1 in sequence; evaluate each; use QAT for Q2 and below
### Reading PerforatedAI Output Files
The library writes these automatically to the `save_name/` folder:[^6]
- `best_model` — best checkpoint by validation score (use for quantization experiments)
- `final_clean_pai` — best model optimized for inference; smaller file; open-source compatible
- `latest` — most recent checkpoint; use this to resume if training crashes
- `best_arch_scores.csv` — **primary results CSV**: best test scores + parameter counts per dendrite cycle
- `paramCounts.csv` — parameter count at each epoch (useful for the param count metric)
- `Scores.csv` — validation + extra scores per epoch

For the **model file size** metric (condition 3 in every graph), always measure `final_clean_pai` for dendritic conditions and a plain `torch.save(model.state_dict(), path)` for base conditions.

***
## PyTorch Implementation Notes
### PerforatedAI Integration
Install from PyPI or source:

```bash
pip install perforatedai
```

#### Step 1 — Imports
```python
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA
```

#### Step 2 — Convert the Model (immediately after initialization, before `.to(device)` or DataParallel)
```python
model = YourModel()
model = UPA.perforate_model(
    model,
    doing_pai=True,       # set False to run without adding dendrites (baseline ablation)
    save_name='PAI',      # change per experiment so outputs don't collide
    making_graphs=True,
    maximizing_score=True # set False when tracking loss instead of accuracy
)
```
`perforate_model` automatically wraps `nn.Linear` and `nn.Conv*` layers with PAI dendritic modules. For more complex architectures (GRUs, Transformers, custom blocks) you must explicitly declare which modules to wrap *before* calling `perforate_model`:[^5][^6]
```python
# By module class type (handles duplicate names or lora layers safely)
GPA.pc.append_modules_to_perforate([nn.MultiheadAttention])
# By name string (requires all names to be unique, no '.' in name)
GPA.pc.append_module_names_to_perforate(['encoder_block'])
# By exact path in the model (e.g. model.layer1.conv1)
GPA.pc.append_module_ids_to_perforate(['.layer1.0.conv1'])
```
Normalization layers must be inside a wrapped block. If they are standalone, wrap them together:
```python
GPA.pc.PAISequential([linear_layer, batch_norm_layer])
```

#### Step 3 — Optimizer & Scheduler Setup
The recommended method enables automatic learning-rate sweeps when dendrites are added. **Remove any external `scheduler.step()` if you use this — stepping is handled internally:**
```python
GPA.pai_tracker.set_optimizer(torch.optim.Adam)
GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
optimArgs = {'params': model.parameters(), 'lr': learning_rate}
schedArgs = {'mode': 'max', 'patience': 5}  # patience must be < epochs per switch
optimizer, PAIscheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
```
If the optimizer is managed by a framework (e.g. Hugging Face Trainer), use the simpler fallback instead:
```python
GPA.pai_tracker.set_optimizer_instance(optimizer)
```
> **Note on weight decay:** The docs warn that weight decay can sometimes harm dendritic learning. If results are poor, try removing it.[^6]

#### Step 4 — Validation Loop Hook
At the end of every validation epoch, call `add_validation_score`. This is the function that actually triggers dendrite addition/switching:
```python
model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
    score, model  # use model.module if DataParallel
)
model.to(device)  # re-send to device after potential restructuring

if training_complete:
    break  # training is done; best model already loaded
elif restructured:
    # Reinitialize optimizer after dendrites are added
    optimizer, PAIscheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
```

To track additional metrics on the same graph (e.g. train accuracy):
```python
GPA.pai_tracker.add_extra_score(train_score, 'Train')
# To log to CSV only (not on graph — use when units differ):
GPA.pai_tracker.add_extra_score_without_graphing(test_loss, 'Test Loss')
```

#### Step 5 — Training Loop Structure
Because the pai_tracker controls when training ends, replace a fixed-epoch loop with an open-ended one:
```python
# Before:
for epoch in range(1, num_epochs + 1):
    ...
# After:
epoch = -1
while True:
    epoch += 1
    ...  # train as normal
    # validation loop ends with add_validation_score call above
```

#### Step 6 — Initial Capacity Test
On first run, the library defaults to `GPA.pc.set_testing_dendrite_capacity(True)`, which rapidly adds 3 dendrite sets to verify the architecture and GPU/MPS memory can handle larger networks. Once the output message confirms the test completed, set this to `False` for real experiments.[^6]

#### MPS Device Handling (M3 Pro)
PerforatedAI does not auto-detect MPS — set the device explicitly:[^6]
```python
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
GPA.pc.set_device(device)  # required for non-CUDA devices
model = model.to(device)
```

#### DDP / Multi-GPU Compatibility
When using `DistributedDataParallel`, pre-initialize gradients to zeros before `loss.backward()` to avoid DDP allreduce errors caused by PAI's selective (Cascade Correlation) training:[^6]
```python
optimizer.zero_grad()
if distributed:
    for param in model.parameters():
        if param.requires_grad and param.grad is None:
            param.grad = torch.zeros_like(param)
loss.backward()
optimizer.step()
```

#### Accessing Wrapped Module Attributes
After conversion, modules are wrapped as `PAINeuronLayer`. Access their attributes via `.mainModule`:
```python
# Before: model.your_layer.SOME_ATTR
# After:
model.your_layer.mainModule.SOME_ATTR
# Or delegate via UPA for private/dunder attributes:
UPA.apply_method_delegation_to_model(model, 'method_name', module_type)
```
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
For sub-4-bit (Q2, Q1.58, Q1), `torchao`'s QAT path is recommended over PTQ for non-LLM models.[^7][^8][^12]
### Pruning
```python
import torch.nn.utils.prune as prune
# Global L1 unstructured pruning at 40%
parameters_to_prune = [(module, 'weight') for module in model.modules()
                        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d))]
prune.global_unstructured(parameters_to_prune,
                           pruning_method=prune.L1Unstructured, amount=0.40)
prune.remove(module, 'weight')  # make permanent before quantization
```
Always apply pruning **before** quantization so the quantizer sees the final sparsified weights.[^1][^2][^13]
### Metrics Collection
For base conditions, collect metrics manually. For dendritic conditions, `best_arch_scores.csv` already tracks parameter counts and best scores per cycle — read it directly:
```python
import os, pandas as pd

# Manual collection (all conditions)
results = {
    "condition": condition_name,
    "accuracy": eval_accuracy,
    "param_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
    "nonzero_params": sum((p != 0).sum().item() for p in model.parameters()),
    "file_size_mb": os.path.getsize(saved_path) / (1024 ** 2)
}

# For dendritic conditions — read PAI's own output
best_arch = pd.read_csv(f'{save_name}/best_arch_scores.csv')
param_count_per_cycle = pd.read_csv(f'{save_name}/paramCounts.csv')
```

To track extra scores for the `best_arch_scores.csv` (e.g. test accuracy alongside validation):
```python
GPA.pai_tracker.add_extra_score(train_accuracy, 'Train')
GPA.pai_tracker.add_extra_score_without_graphing(test_accuracy, 'Test Accuracy')
```
### MPS Backend Setup (M3 Pro)
```python
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
GPA.pc.set_device(device)  # required — PerforatedAI does not auto-detect MPS
model = model.to(device)
```
`torchao` quantization currently targets CPU/CUDA for kernel dispatch; on M3 Pro, run the quantization step on CPU and then move back to MPS for evaluation.[^14][^3]

***
## Key Research Hypotheses to Test
1. **Do dendritic models consistently outperform base models in accuracy before quantization?** Expected: yes, based on Perforated Backpropagation paper results.[^11][^15]
2. **Does the dendrite + pruning combination produce better accuracy-per-byte than base + quantization alone?** This is the central hypothesis.
3. **Are certain domains (e.g., graph, tabular) more tolerant of extreme quantization (Q1–Q2) in dendritic form?**
4. **Does the accuracy gap between dendritic and non-dendritic models widen or narrow at extreme bit depths (Q1, Q1.58)?**
5. **Is the file size / parameter count reduction from Dend+Prune+Q1 competitive with the accuracy loss vs. Base FP32?**

---

## Repository Implementation

The benchmark command-line implementation is located under `src/dendritic_benchmark/`.

- `cli.py` exposes the `dqb` entry point and parses global options.
- `specs.py` defines model and condition keys used by the benchmark.
- `pipeline.py` orchestrates dataset loading, checkpoint chaining, model construction, and condition execution.
- `training.py` runs each condition, evaluates metrics, and saves model artifacts.
- `results.py` writes `record.json`/`record.csv`, regenerates manifest files, and creates summary plots.
- `plots.py` renders SVG charts with Matplotlib.
- `compat.py` isolates optional dependencies, device selection, and PerforatedAI wrapping.

## References

1. [torch.nn.utils.prune.l1_unstructured — PyTorch 2.11 documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.prune.l1_unstructured.html) - Prunes tensor corresponding to parameter called name in module by removing the specified amount of (...

2. [Pruning Tutorial - PyTorch documentation](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html) - In this tutorial, you will learn how to use torch.nn.utils.prune to sparsify your neural networks, a...

3. [MPS Support - PyTorch - Mintlify](https://mintlify.wiki/pytorch/pytorch/hardware/mps) - Apple Metal Performance Shaders GPU acceleration for PyTorch on Mac

4. [Accelerated PyTorch training on Mac - Metal - Apple Developer](https://developer.apple.com/metal/pytorch/) - PyTorch uses the new Metal Performance Shaders (MPS) backend for GPU training acceleration.

5. [Examples Repository for PyTorch Extension - Perforated ... - GitHub](https://github.com/PerforatedAI/PerforatedAI-Examples) - Because Perforated Backpropagationtm is not open source, a license is required to use our software. ...

6. [PerforatedAI/PerforatedAI: Add Dendrites to your PyTorch Project](https://github.com/PerforatedAI/PerforatedAI) - Add biologically-inspired dendritic optimization to your PyTorch neural networks with just a few lin...

7. [Quantization Overview — torchao 0.17 documentation](https://docs.pytorch.org/ao/stable/contributing/quantization_overview.html) - No matter what quantization we are doing, in the end we will be using some low precision dtypes to r...

8. [Quantization-Aware Training in TorchAO (II) - PyTorch](https://pytorch.org/blog/quantization-aware-training-in-torchao-ii/) - Achieve on par accuracy with a 3-bit per-row model compared to a 4-bit per-group baseline, while usi...

9. [LabStrangeLoop/bitnet: Train and evaluate 1.58 bits Neural Networks](https://github.com/AlarioAI/bitnet) - Implementation of BitNet b1.58 (ternary quantization) applied to ResNet architectures, with systemat...

10. [Fine-tuning LLMs to 1.58bit: extreme quantization made easy](https://huggingface.co/blog/1_58_llm_extreme_quantization) - BitNet is an architecture introduced by Microsoft Research that uses extreme quantization, represent...

11. [A Neuroscience Inspired Extension to Artificial Neural Networks - arXiv](https://arxiv.org/abs/2501.18018) - The paper explores a novel system of "Perforated" backpropagation empowering the artificial neurons ...

12. [TorchAO: PyTorch-Native Training-to-Serving Model Optimization](https://arxiv.org/abs/2507.16099) - We present TorchAO, a PyTorch-native model optimization framework leveraging quantization and sparsi...

13. [Neural Network Pruning: How to Accelerate Inference with Minimal ...](https://arikpoz.github.io/posts/2025-04-10-neural-network-pruning-how-to-accelerate-inference-with-minimal-accuracy-loss/) - PyTorch provides a built-in pruning utility under torch.nn.utils.prune . This API supports both unst...

14. [matmul() using PyTorch's MPS backend is faster than Apple's MLX](https://kevinmartinjose.com/2025/04/21/matmul-using-pytorchs-mps-backend-is-faster-than-apples-mlx/) - Disclaimer: I do not know why PyTorch + MPS is faster (yet) I recently came across Apple’s MLX frame...

15. [[Revue de papier] Perforated Backpropagation: A Neuroscience Inspired Extension to Artificial Neural Networks](https://www.themoonlight.io/fr/review/perforated-backpropagation-a-neuroscience-inspired-extension-to-artificial-neural-networks) - The paper titled "Perforated Backpropagation: A Neuroscience Inspired Extension to Artificial Neural...

