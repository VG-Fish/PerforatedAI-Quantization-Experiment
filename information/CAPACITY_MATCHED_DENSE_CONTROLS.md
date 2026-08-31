# Automated topology-matched dense controls

<!-- status-banner -->
> **Status: current protocol proposal (2026-08-31).** Describes the control design step 4 of the validity protocol in [optimization/00_assessment.md](optimization/00_assessment.md) requires. Not yet implemented in the runner: no `dqb` flag builds these controls today.

## Purpose

This protocol tests a narrower and more useful question than “does the
perforated model beat the original base model?”

> Given the same base checkpoint, insertion time, added parameter budget,
> optimizer treatment, training horizon, and quantization procedure, does a
> retained PAI dendritic branch outperform an ordinary dense branch placed in
> exactly the same location?

The primary comparator must be **topology matched**, not merely a wider model.
A global width multiplier can happen to match the total parameter count while
changing every layer, feature dimension, and optimization path. That is a
useful secondary sensitivity check, but it is not the closest non-dendritic
counterfactual.

## The three models

For every model key and seed, construct these arms from one dense base
architecture:

| Arm | Definition | What it answers |
|---|---|---|
| `base` | The unmodified base model. | Is there any benefit over the original model? |
| `dendrites` | The base model with PAI's retained branches. | The observed PAI result. |
| `capacity_dense` | The base model with ordinary PyTorch branches in the same locations and with the same tensor shapes/parameter count as the retained PAI branches. It contains no `PAINeuronModule`, `PAIDendriteModule`, PAI tracker, PB scoring, candidate state, or PAI switching. | Is the result better than equally placed extra ordinary capacity? |

The `capacity_dense` arm is the required control. A `width_matched_dense` arm
may additionally widen the original layers until it reaches the same parameter
budget, but it must be labelled secondary because it is less structurally
similar.

## What “as similar as possible” means

The control starts from the exact same base factory and changes only the
capacity introduced by a retained dendrite.

For every retained PAI branch, the automated extractor records:

- the wrapped base-module path, for example `.conv1.linear`;
- the branch input and output tensor shapes;
- operation type, activation, bias, merge operation, and any mixing-weight
  shape used in the forward pass;
- trainable parameter tensor names, shapes, dtypes, and count;
- the insertion checkpoint, epoch, optimizer/scheduler state, RNG state, and
  data-order state; and
- the raw PAI switch and architecture-log references that prove retention.

The control factory rebuilds the unwrapped base module and adds a normal
PyTorch residual/parallel branch with that recorded specification. Conceptually:

```text
base:            y = F(x)
dendritic:       y = merge(F(x), PAI_dendrite(x))
capacity_dense:  y = merge(F(x), ordinary_dense_branch(x))
```

`ordinary_dense_branch` must use the same mathematical branch shape and merge
location as the retained dendrite, but be implemented solely from ordinary
`torch.nn` modules. The control is deliberately given PAI's selected location;
this favours the control and avoids claiming that PAI's placement search is an
advantage over a control that was not allowed to use it.

Do **not** replace this with an arbitrary larger hidden size, extra unrelated
layer, global channel multiplier, or copied final dendritic weights. Those
change more than “dendrites versus equally placed capacity.”

## Automated workflow

### 1. Discover a retained topology

Run the normal `dendrites_fp32` arm. Generate a control only when all of these
hold:

1. `dendrite_audit_status == "verified_retained"`;
2. the raw PAI `switch_epochs.csv` has the expected switch history;
3. raw PAI `param_counts.csv` reports a larger final topology; and
4. the final checkpoint's non-bookkeeping parameter count matches that raw
   final count.

If PAI tried a candidate and rejected it, no capacity control is generated:
there is no retained architecture to match. If the audit is unverified, retain
the artifacts for debugging but do not train or report the control.

### 2. Capture the pre-branch fork point

At the start of the retained candidate phase, persist a dedicated
`capacity_control_fork.pt`. It must include:

- unwrapped dense model state;
- optimizer and scheduler state;
- Python, NumPy, Torch, and accelerator RNG states;
- data-loader/sampler order state; and
- the PAI topology specification and candidate initializer specification.

The fork is taken before candidate optimization. It is not a final PAI model
snapshot. Starting a control from trained dendrite weights would leak the
dendritic result into the control.

### 3. Build the matched dense model

Create `capacity_dense_fp32` by loading the forked dense backbone and applying
the extracted branch specification with ordinary modules. Initialize each
ordinary branch with the same distribution, seed, dtype, and scale used for
the PAI candidate at that fork. Do not copy post-training branch weights.

Record two exact counts:

```text
base_trainable_params
dendritic_trainable_params
capacity_dense_trainable_params
```

After excluding PAI runtime bookkeeping such as `tracker_string`, the last two
counts must be equal. Any mismatch is an invalid control, not an approximation
to report. The artifact manifest should also bind the control to the source
dendritic artifact ID and topology-specification hash.

### 4. Match optimization, not just parameters

Resume `capacity_dense` from the same fork point and use:

- the same optimizer type, hyperparameters, weight decay, clipping, and
  optimizer reset/rebuild event as the dendritic arm;
- the same backbone learning-rate schedule;
- the same learning-rate floor applied to the newly added capacity; this is
  essential when a late PAI insertion would otherwise receive a near-zero
  learning rate;
- the same batches in the same order from the fork onward; and
- the same number of post-fork epochs as the dendritic arm's observed run,
  including any dynamic continuation.

The dense control must not call PAI, collect PB scores, add candidates, or
switch topology. It is a fixed ordinary architecture after the fork.

Also run `base_more_training`: the original dense model resumed from the same
fork, with no additional branch, for that identical post-fork epoch count and
schedule. This distinguishes a dendrite/capacity effect from merely receiving
more updates.

### 5. Quantize every comparable architecture

For `base`, `base_more_training`, `dendrites`, and `capacity_dense`, derive
Q8, Q4, Q2, Q1.58, and Q1 artifacts from each arm's own FP32 checkpoint. Apply
the same PTQ evaluation, PQAT budget, optimizer settings, calibration data,
and quantization-evaluation revision to every arm. Never quantize the
dendritic checkpoint twice or use it as a source for the dense control.

## Required implementation pieces

Implement this as a first-class benchmark feature rather than ad-hoc tuning
scripts.

1. Add a `RetainedTopologySpec`/`DenseBranchSpec` serializer, preferably in a
   new `src/dendritic_benchmark/capacity_control.py` module. It owns PAI-wrapper
   inspection, control construction, exact-count checks, and provenance.
2. Add a training hook immediately before retained-candidate optimization to
   write `capacity_control_fork.pt`; this makes the fork reproducible rather
   than reverse-engineered from later PAI snapshots.
3. Add condition keys such as `capacity_dense_fp32`, `capacity_dense_q8`, …,
   and `base_more_training_fp32`, with explicit artifact dependencies. Their
   records need `control_kind`, `control_of_artifact_id`, `fork_checkpoint_sha256`,
   `topology_spec_sha256`, and all three parameter counts.
4. Extend the manifest/audit writer with `capacity_control_status`:
   `not_requested`, `generated`, `invalid_parameter_mismatch`,
   `invalid_provenance`, or `current`.
5. Extend paired statistics to report `dendrites - capacity_dense` and
   `capacity_dense - base`, direction-corrected for both accuracy and
   loss-style metrics.

The implementation must fail closed: if PAI wrappers cannot be mapped to an
ordinary branch exactly, mark that model/control unsupported and do not silently
fall back to width scaling.

## Example: current GCN configuration

The clean seed-0 GCN artifact has:

```text
base parameters:       92,231
retained PAI parameters: 184,391
capacity to match:      92,160 additional trainable parameters
PAI-targeted modules:   .conv1.linear and .conv2.linear
```

The GCN control must therefore begin with the same 92,231-parameter GCN and
add ordinary branches at those two recorded paths whose combined trainable
count is exactly 92,160. A generic wider GCN with about 184k parameters is not
the primary control: it changes message-passing dimensions throughout the
network instead of only replacing the retained PAI branches.

The extractor, not a hand-written GCN formula, determines the branch tensor
shapes and merge operation from the retained PAI artifact. That keeps the
method valid when PAI changes a target's internal branch layout.

## Analysis and decision rule

For each seed and precision, compute direction-corrected effects:

```text
dendrite_effect      = score(dendrites) - score(base)
capacity_effect      = score(capacity_dense) - score(base)
dendrite_specificity = score(dendrites) - score(capacity_dense)
training_effect      = score(base_more_training) - score(base)
```

For quantized models, also compare within-arm degradation from the corresponding
FP32 artifact:

```text
quantization_drop(arm, q) = score(arm_q) - score(arm_fp32)
robustness_advantage(q) = quantization_drop(dendrites, q)
                          - quantization_drop(capacity_dense, q)
```

Report “dendrites outperform added capacity” only when all conditions below
are satisfied:

1. three or more paired seeds pass artifact, dendrite, control, and
   quantization audits;
2. the mean `dendrite_specificity` is positive in the metric's correct
   direction and exceeds the paired-seed noise floor;
3. its paired confidence interval excludes zero (and the configured paired
   significance test passes); and
4. the result is not explained by `base_more_training`.

If `capacity_dense` matches the dendritic arm, the correct conclusion is:
“PAI found a useful place to add capacity, but this experiment does not show a
benefit beyond equally placed ordinary capacity.” If `base_more_training`
matches both, the correct conclusion is: “The apparent benefit is compatible
with extra training.”

## Non-negotiable fairness checks

- Same seed, dense initialization, split, data order, and evaluation set.
- Same pre-branch checkpoint SHA-256 for dendritic, capacity, and
  more-training continuations.
- Same insertion epoch and post-insertion horizon.
- Equal trainable parameter counts for dendritic and capacity branches,
  excluding runtime bookkeeping only.
- Same quantization/PQAT recipe and source-artifact isolation.
- No validation/test data used to choose a width, branch shape, or winner after
  observing the result.
- Report failed or unsupported controls explicitly; do not substitute an
  approximate model without a label.

This protocol gives the dense control every reasonable advantage while holding
the base model nearly identical to the dendritic model. Any remaining,
replicated `dendrites - capacity_dense` advantage is therefore meaningful
evidence for the dendritic mechanism rather than a generic “bigger model”
effect.
