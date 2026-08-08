# Remaining Comparability Fixes — Handoff

Status as of 2026-08-07. This document is a handoff to an agent picking up work
on making this benchmark's FP32 baselines comparable to published results.

> **Everything in §3 has since been implemented.** §3.1–§3.4 were done in the
> 2026-08-07 pass, plus a fifth item (§3.5, `pointnet_modelnet40`) that was not
> in the original list. Each subsection below keeps its original diagnosis and
> records what was actually built at the end. The one thing still outstanding is
> the retraining sweep in §3.4 — **no results under `results/` reflect any of
> this yet.**

Read `information/MODEL_REFERENCE.md` first — its "Shared Notes" section is the
source of truth for per-model configuration and already documents everything
that has been fixed.

---

## 1. Why any of this matters

This repo benchmarks **PerforatedAI dendritic augmentation against
quantization**. Every conclusion it produces is a comparison of a dendritic arm
against an FP32 baseline. If a baseline is trained on a leaky split, or on a
different task than the published number it is being read against, the
comparison is measuring something other than what it claims.

Two consequences that are easy to forget:

- **PerforatedAI's dendrite-switch decisions read the validation metric.** A
  validation signal that is inflated or leaky does not just misreport accuracy;
  it changes when (or whether) dendrites get added. Validation integrity is a
  correctness issue for the dendritic arm, not just a reporting nicety.
- The suite's own `metric_value` is the **test** metric (after best-val weights
  are restored) and `best_metric_value` is the **validation** metric. They are
  different splits. A gap between them is not a bug by itself.

---

## 2. What has already been fixed

All of this is **committed at `49b90c1` ("added model fixes")**, except item 7
which is uncommitted in the working tree.

| # | Model(s) | Defect | Fix |
|---|---|---|---|
| 1 | `lstm/tcn/gru_forecaster` | `random_split` over sliding windows (adjacent windows share `seq_len-1` timesteps) + normalisation fitted on the whole file | `_chronological_forecast_bundle` in `data.py`: contiguous splits, train-only statistics, published window geometry |
| 2 | `mobilenetv2_cifar10` | only the stem stride adapted for 32×32 → 16× downsampling, 2×2 final map | `features[2]` depthwise stride 2→1 (8×, 4×4), guarded against torchvision stage-table drift |
| 3 | `gcn` | 70/15/15 split = 1895 training labels vs Planetoid's 140 | `_planetoid_style_split`: 20/class, 500 val, 1000 test |
| 4 | `tabnet`, `saint_adult` | 8 nominal Adult columns ordinal-coded then z-scored | `TabularColumnEmbedding` in `models.py`; loader standardises numerics only |
| 5 | `mpnn`, `attentivefp_freesolv` | messages ignored bond order entirely; `in_ring` set only on ring-closure atoms (63% of ESOL ring atoms unlabelled) | bridge-detection ring perception + dense `[N,N,6]` edge tensor into messages and attention |
| 6 | `actor_critic`, `dqn_lunarlander`, `ppo_bipedalwalker` | cloned heuristics that scored −519 and −120 in-environment; no return was ever measured | Gymnasium reference heuristics; `_evaluate_episodic_return` records a real return |
| 7 | same three | rollout caches `random_split` by **timestep**, so step *t* and *t+1* straddled train/test | `_split_by_episode`: whole episodes assigned to one split |

Incidental fixes in the same commit: `benchmark.py`'s latency input shapes had
drifted from the featurisers (Cora 50 vs 64 nodes, molecules 9 vs 20 features)
and would have crashed on the new embedding lookups; `_require_dependency` used
`__import__`, which returns the *top-level* package for a dotted name.

### Measured effect of the RL fix

Use these as regression references — they were measured, not estimated:

| model | train metric before → after | env return before → after | heuristic ceiling | published |
|---|---|---|---|---|
| `actor_critic` | 0.8300 → 0.9273 | 292.7 → **500.0 ± 0.0** | 500.0 | 500 = max |
| `dqn_lunarlander` | 0.9882 → 0.9845 | −522.9 → **+245.3 ± 95.9** | 230.5 | 200 = solved |
| `ppo_bipedalwalker` | −0.0004 → −0.0308 | −119.5 → **−79.6 ± 12.6** | 89.2 | 300 = solved |

Those numbers predate fix 7 (episode split), so the train metrics will shift
down slightly when re-measured — that is the expected direction, since the leak
was inflating them.

---

## 3. Remaining work

### 3.1 Convert `ppo_bipedalwalker` to real PPO — **done**

This was the main outstanding task. The design research below was completed
first; the implementation followed it, and what was built is recorded at the end
of this subsection.

**Why it is needed.** `ppo_bipedalwalker` is behaviour cloning, and it clones a
*stateful* policy. `BipedalWalkerHeuristics` carries the swing leg and gait phase
between steps, so one observation maps to different actions depending on hidden
state a feedforward net cannot observe. The clone reaches ≈ −80 against the
heuristic's +90 and **will not close that gap by training longer**. Real PPO is
the way to a number comparable to the 300-point solved threshold.

**Current state of the model (`models.py:615`).** The PPO scaffolding exists but
is inert — verified by a backward pass, not inferred:

- `forward()` returns only `tanh(actor_mean(backbone(x)))`.
- `.critic` and `actor_log_std` are on no forward path used in training →
  **zero gradient**, 133 of 20361 params (0.7%).
- `value_function()` has no caller anywhere in the repo.
- `pipeline.py:407` `_perforation_track_only_module_ids` marks `.critic` and
  `.actor_mean` track-only for PAI. **This must be revisited** once the critic
  is actually trained.

`ActorCritic` has the same disease (`.value` gets no gradient — `_loss_for_model`
does `criterion(outputs[0], targets)`), but it is staying as behaviour cloning.

**Design decisions already reached** (adopt or consciously override):

1. **Gaussian policy, no tanh squashing.** Sample `a ~ N(mean, exp(log_std))`,
   clip to `[-1, 1]` when stepping the env, compute log-prob on the *unclipped*
   sample. This is Stable-Baselines3's default for `Box` action spaces
   (`squash_output=False`) and avoids the tanh Jacobian correction in the
   log-prob. Do **not** keep the current `tanh` on the mean.
2. **One epoch = one PPO iteration.** Collect `n_steps` with the current policy,
   compute GAE advantages and returns, then run K minibatch passes over that
   buffer. This maps onto the existing epoch loop without restructuring it.
3. **Batch becomes** `(obs, action, old_logprob, advantage, return)`.
4. **Validation metric = mean episodic return.** This is the natural PPO
   selection metric and it is leak-free by construction.
5. `forward(obs)` returns `(mean, log_std, value)`.

**Integration seams** (all line numbers current as of this document):

| what | where | change needed |
|---|---|---|
| epoch batch loop | `training.py:1811` `_run_epoch_batches` | at epoch start, if `bundle.on_policy` is set, refresh `bundle.train_loader` from the current policy |
| progress bar | `training.py:1582` `_training_batch_progress` | reads `bundle.train_loader`; refresh **before** this is called |
| batch unpacking | `training.py:805` `_forward` | 5-tuple branch for `ppo_bipedalwalker` |
| loss | `training.py:336` `_binary_or_multi_loss` and `_loss_for_model` | clipped surrogate + value loss + entropy bonus; remove `ppo_bipedalwalker` from the `MSELoss` set |
| metrics | `training.py:731` `_compute_all_metrics` | policy/value loss, explained variance, KL |
| primary metric | `training.py:285` `_PRIMARY_METRIC_KEYS` | currently `"reward_proxy"`; point at the episodic return |
| validation | `training.py:2561` `_run_validation_pass` → `training.py:1441` `_eval_on_loader` | special-case: run rollouts instead of iterating a val loader |
| bundle | `data.py:145` `TaskBundle` | add `on_policy: Any | None = None` |
| rollout source | `data.py` near `_build_bipedalwalker` | new class: holds env id, `n_steps`, `gamma`, `gae_lambda`; `collect(model, device)` returns a fresh `DataLoader` |
| PAI track-only | `pipeline.py:407` | `.critic` is now trained — re-evaluate |

`_evaluate_episodic_return` (`training.py:642`) already exists and is reusable
for the validation metric. It takes `(model_key, model, device, episodes, seed)`
and reads `_RL_ENVIRONMENTS` (`training.py:630`). Note it calls `model(batch)`
and takes `output[0]` when the result is a tuple — that stays compatible with a
`(mean, log_std, value)` return, but **the mean must be clipped** to the action
space before stepping.

**Caveats to carry forward, not to hide.** This was flagged to the user and they
chose to proceed anyway:

- PPO's objective is non-stationary and its returns are high-variance. The
  dendritic-vs-quantization comparison for this model becomes much noisier than
  for the supervised models, and probably needs multiple seeds to say anything.
- PAI's switch logic will be reading a noisy episodic return rather than a smooth
  validation loss. Watch for dendrites being added on rollout noise.
- Every other model in the suite is supervised. `ppo_bipedalwalker` becomes the
  only one whose training data depends on its own weights.

#### What was built

All five design decisions above were adopted unchanged.

| piece | where |
|---|---|
| `PPORolloutSource` — persistent env, GAE(λ), reward scaling, returns a fresh `DataLoader` | `data.py` |
| `_RepeatedPermutationSampler` — folds PPO's K passes into one loader iteration, so the epoch loop did not have to change | `data.py` |
| `RunningObsNorm`, orthogonal init, `forward → (mean, log_std, value)` | `models.py` `PPOPolicy` |
| `_ppo_terms` / `_ppo_loss` / `_ppo_metrics` — clipped surrogate, value loss, entropy, Schulman `approx_kl`, clip fraction, explained variance | `training.py` |
| `_refresh_on_policy_batches` — rebuilds `bundle.train_loader` at epoch start | `training.py` |
| `_rollout_evaluation` — validation and test are episodic rollouts, disjoint seeds | `training.py` |

Details worth not rediscovering:

- **Observation statistics are folded forward one iteration.** `obs_norm` is
  updated with the *previous* rollout's observations, before the current one
  starts, then left alone. Updating mid-iteration would mean the stored
  `old_log_prob` was computed under different normalisation than the surrogate
  recomputes, so the importance ratio would not start at exactly 1.
- **The candidate graph is switched off during collection.** A rollout is 2048
  single-observation forward passes; left recording, PerforatedAI would correlate
  dendrite candidates against batch-of-one activations that no optimizer step
  ever backpropagates through.
- **Budget is 800 iterations (~1.6M env steps, ~2h base / ~4h dendritic).** Not
  arbitrary: an untrained policy stands still and scores ≈ −9, and a policy that
  has started moving but still falls scores ≈ −100. A run cut off inside that
  trough selects its *first* epoch as the best checkpoint and ends up comparing
  two arms' untrained weights. Verified at 4 and 10 epochs that the trough is
  real (−9.2 → −94 → −100 over the first four iterations).
- **PAI track-only was revisited**, as this document asked. `.critic` and
  `.actor_mean` stay track-only but for a new reason — the old one (`.critic`
  gets no gradient) is void now that it is trained. A dendrite switching in
  changes its module's output as a step; on `.actor_mean` that moves the policy
  outside the 0.18 clip range against the log-probs the live buffer was recorded
  under, and on `.critic` it changes the value baseline those advantages were
  computed against. Dendrites go on `.backbone` only — confirmed by a
  10-epoch dendritic run: 20361 → 60041 params, insertions on `.backbone.0` and
  `.backbone.2` only, PAI switch fires, no warnings.

### 3.2 `gcn` is still not mechanically comparable — **done (restructured)**

The label budget now matches Kipf & Welling (140 train / 500 val / 1000 test),
but the mechanism does not:

| | Kipf transductive | this suite |
|---|---|---|
| receptive field | full 2708-node graph | 2 hops, capped at 64 nodes |
| batching | one graph, one step | 32 independent ego graphs per step |
| unlabelled nodes | propagate information | only appear inside some ego graph |
| high-degree node | all neighbours present | truncated to 64 |

So **81.5% is still not the number to expect**. The current setup is essentially
GraphSAGE-style inductive classification — legitimate, but not what 81.5%
measures. `_CoraEgoDataset` is at `data.py:948`; `GCN.forward` reads out
`x[:, 0]` (the centre node).

Making it truly transductive means full-graph forward passes with a train mask,
which does not fit the DataLoader-per-batch architecture the pipeline is built
on. **This is a scope decision for the user, not an obvious fix.** Either:
- accept it and label the number "inductive ego-graph Cora, not comparable to
  Kipf's 81.5%" (currently what `MODEL_REFERENCE.md` does), or
- restructure for full-graph training, which touches the same epoch-loop seams
  as the PPO work.

#### What was built

The user chose the restructure. It turned out **not** to need epoch-loop changes:
a full-graph "batch" is a dataset of length 1, so the existing loader machinery
carries it unchanged.

- `_CoraEgoDataset` → `_CoraTransductiveDataset`, yielding
  `(x_all, adjacency, node_indices, labels)` — the whole 2708-node graph plus the
  split's node indices. Three instances share the same graph and differ only in
  indices. `_BATCH_SIZES["gcn"] = 1`.
- `GCN.forward` returns all-node logits; `_forward` in `training.py` selects the
  split's rows with `index_select`.
- `GraphConv` rewritten to Kipf's form: `Â(XW) + b` with symmetric
  degree normalisation. **Operand order matters** — `Â(XW)` is 0.72 GFLOP against
  10.5 for `(ÂX)W`, which is what makes full-graph training affordable at all.
- Moving the bias out of `nn.Linear` broke PerforatedAI (four
  `Parameter does not have parameter_type attribute in n mode` warnings: the
  owning `GraphConv` cannot be tracked because its child `.linear` is
  perforated). Fixed with PAI's own documented remedy,
  `append_parameter_ids_to_track`, plumbed through `compat.py` and
  `_perforation_parameter_ids_to_track` in `pipeline.py`. **Parameter ids need a
  leading dot** (`".conv1.bias"`) — PAI validates them through the same checker
  as module ids.
- `benchmark.py` caps the latency sweep at batch 1 for `gcn`; a 32-batch
  full-graph adjacency is 939 MB.

Measured: **test 0.7990 ± 0.005 over 4 seeds**, 200 epochs in 3.3s, against
Kipf's 81.5% and the previous inductive setup's 76.4%. Now genuinely the same
mechanism, so 81.5% *is* the number to read it against.

### 3.3 DistilBERT's validation split leaks — **done (dev-set holdout)**

`TextDataSets.sst2` (`data.py:916`) carves validation out of GLUE SST-2's
`train` split at `data.py:935`:

```python
train_ds, val_ds, _ = _split_dataset(train_full, train_ratio=0.9, val_ratio=0.1)
```

SST-2's `train` is **phrase-level**: Stanford parsed each sentence into a
constituency tree and labelled every subtree, so one sentence contributes many
rows (the full sentence, plus overlapping sub-phrases). A random 10% therefore
puts a phrase in validation whose parent sentence sits in training.

Measured: val 95.19% against test 90.48% — the validation number is ~4.7 points
optimistic. **Test is fine** (it is the real GLUE dev set, and 90.48% ≈ the
published ~91%), so the reported metric is not wrong.

The reason to care is PAI: the switch logic reads validation. A signal that is
4.7 points inflated and moves for the wrong reasons is a poor plateau detector.

Possible fixes, none free:
- Split validation off by **parent sentence** rather than by row. SST-2 ships the
  tree structure, but the HuggingFace `glue/sst2` config does not expose it —
  would need the raw Stanford treebank.
- Hold out a slice of the GLUE dev set for validation and test on the rest.
  Shrinks the test set and departs from the standard protocol.
- Accept and document. This is what is currently done.

#### What was built

The user chose the dev-set holdout. `TextDataSets.sst2` now trains on the **full**
67349-row GLUE train split (it previously threw 10% of it away to make a leaky
validation set) and carves validation out of the 872-row GLUE dev set with
`_stratified_holdout`: 261 validation / 611 test at
`SST2_DEV_VALIDATION_RATIO = 0.3`, per-class so the small split cannot drift
off the label balance.

The trade the user accepted: the test set shrinks 872 → 611, which widens its
confidence interval by about ×1.2, and it is no longer the standard GLUE dev
protocol. In exchange the validation signal PerforatedAI switches on comes from
sentences that share no parent tree with anything in training — which was the
point, since a 4.7-point-inflated signal is a poor plateau detector.

### 3.4 Everything needs retraining — **rebuild wired in; sweep still owed**

`results/updated_models` is **pre-fix for every model listed in §2**. Those
records were produced by the code as it stood before `49b90c1`.

> **Update (2026-08-08): `results/` is now empty.** The pre-fix trees were
> archived to `archive/old_models_pre-fix_20260808.zip` and
> `archive/results_dynamic_pre-fix_20260808.zip` (verified, then removed), and
> `results/updated_models` is gone as well — it held one post-fix record, a
> `gcn` `base_fp32` at 0.796 Accuracy, worth 31 seconds to reproduce. So there
> is no longer any partial state to reconcile: the sweep below starts from
> nothing, and `--fresh` has nothing to clear.

At the time of writing the original sweep is **still running**
`pointnet_modelnet40` (unaffected by these changes). Once it exits:

```bash
uv run dqb run --conditions base_fp32 --fresh --logging-dir logs_fixed
```

`--fresh` matters: `--ignore-saved-models` does **not** prevent epoch-level
checkpoint resume. `training.py` calls `_load_epoch_checkpoint()` whenever the
output directory exists and never consults that flag, so a leftover
`epoch_checkpoint.pt` would silently continue an old-architecture run. The
script warns about this and refuses to pretend otherwise.

**Known defect in the parallel sweep** (pre-existing, not introduced here):
`pipeline.py` writes `manifest.csv` and the comparison reports from *its own*
records at exit, so with four concurrent `dqb run` streams the last one to
finish overwrites the others. Per-model `record.json` files are intact, so
nothing is lost. Rebuild afterwards with:

```bash
dqb compare --manifest --results-root results --results-directory updated_models
```

Wiring this into `run_base_sweep.sh` at the point `watch_loop` breaks on "all
streams finished" was deliberately **not** done — it changes what the script does
at completion and is the user's call.

#### What was built

The user asked for it to be wired in. `run_base_sweep.sh` gained
`rebuild_reports()`, called from `watch_loop` at the point it breaks on "all
streams finished". It runs the `dqb compare --manifest` above and, on failure,
prints the exact command to rerun by hand rather than leaving a half-written
manifest unexplained — the per-model records are intact either way. The
`--detach` exit path prints the same command as a hint, since detaching means
nothing is left watching for the streams to end. The behaviour is described in
the script header, which is what `--help` prints.

**Update (2026-08-08): the defect above is fixed, and the shell script is gone.**
The parallel streams now live inside `dqb run` itself (`--jobs`, default 4), so
there is no longer a `run_base_sweep.sh` or a `dqb sweep`. Workers run with
`write_reports=False` and the coordinator writes `manifest.csv` and the
comparison reports once, from every record on disk, after the last worker exits
— so the overwrite race cannot happen rather than being repaired afterwards.
`--detach` still skips that step, and prints the `dqb compare --manifest`
command to run by hand.

#### Still owed: the sweep itself

**Nothing under `results/` reflects any fix in §3.** The retraining is the one
piece of this handoff that has not been done. Run it with the command above.
Two things changed since this section was written:

- `pointnet_modelnet40` is no longer "unaffected" — see §3.5. The sweep that was
  running while this document was written finished against the old dataloader,
  so that record is stale too.
- Budgets moved: `ppo_bipedalwalker` 120 → 800 epochs (~2h base, ~4h dendritic),
  `pointnet_modelnet40` 100 → 200 epochs but ~5.5h → ~1.2h. `gcn` is 3 seconds.

### 3.5 `pointnet_modelnet40` reads mesh vertices, not the mesh — **done**

Not in the original list; added on request. The stale claim to ignore is
`DYNAMIC_DENDRITIC_MIGRATION.md` §8's "baseline accuracy is 13.4% … the base
model is broken". By 2026-08-07 the model reached **0.7937 validation accuracy**
against Qi et al.'s 89.2%; the 13.4% predated the corrupted-eval fix
(`_move_batch_to_device`), the feature-transform orthogonality penalty, and the
step schedule. The model definition is faithful to the paper, T-Net identity
initialisation included. The remaining gap was entirely in the loader.

**The defect.** `_ModelNet40Dataset.__getitem__` read the OFF file's *vertex
list*, ignored its *faces* completely, and took 1024 evenly spaced vertex
indices with `torch.linspace`. ModelNet40 is CAD geometry, so vertex density
tracks how a model was authored, not how large its surface is — a tabletop can
be four vertices while a moulding detail is several thousand. Evenly spaced
indices therefore sample the file's authoring order. Measured over all 12311
meshes: median 3843 vertices, but **24% have fewer than 1024** and 8% fewer than
256, and those were padded by cyclically repeating the same handful of corner
points until the tensor was full. Qi et al. train on
`modelnet40_ply_hdf5_2048`, which is sampled uniformly from the faces.

**Three fixes, all in `data.py`:**

1. **Area-weighted surface sampling.** `_read_off_mesh` now parses faces too
   (tokenised, because 2862 of the files glue the counts onto the `OFF`
   keyword, and because not every face is a triangle); `_sample_mesh_surface`
   picks triangles proportional to area and a uniform barycentric point in each.
   Every mesh yields a real surface sample at any vertex count.
2. **Caching.** Parsing costs ~19 ms/mesh, so 9843 training meshes were ~3
   minutes of single-threaded text parsing **per epoch** — the dominant cost of
   this model, ahead of the network. Clouds are sampled once into
   `data/modelnet40/surface_points_v1_{split}_2048.pt` (~300 MB total, ~4 min to
   build) and indexed from memory thereafter. 2048 points cached, 1024 used:
   training draws a fresh random subset each epoch, evaluation takes a fixed
   prefix. This is the reference protocol.
3. **Rotation axis.** The training augmentation rotated about **Y**. The raw OFF
   meshes are **Z-up** — verified by mean extent: person `[14, 23, 42]`, bottle
   `[68, 70, 241]`, chair `[154, 161, 291]`. Rotating about Y tips objects onto
   their sides during training only, discarding a cue the never-rotated
   evaluation splits still carry. (The distributed HDF5 clouds *are* Y-up, which
   is where the mistake comes from; the conversion rotated them.) Now Z.

Measured at 10 epochs, against the old loader at the same epochs:

| | old loader | new loader |
|---|---|---|
| epoch 1 val accuracy | 0.3862 | 0.3547 |
| epoch 9 val accuracy | 0.6128 | **0.7337** |
| seconds/epoch | ~200 | **~36** |

The old run needed ~30 epochs to reach 0.7337. Budget went 100 → 200 epochs and
still costs a third of the old wall clock.

---

## 4. Practical notes for whoever picks this up

**Environment.** `.venv/bin/python`, Apple Silicon MPS, `dqb` CLI installed from
`src/` (editable). `source .venv/bin/activate` then `dqb run --help`.

**There is no epoch-limit CLI flag.** To smoke-test a model end to end, patch the
recipe. A working harness lives at
`<scratchpad>/verify_models.py` — it monkeypatches
`BenchmarkRunner._training_hyperparameters` down to N epochs and runs the real
pipeline into a scratch results root. Recreate it if the scratchpad is gone; the
pattern is:

```python
BenchmarkRunner._training_hyperparameters = patched   # returns recipe with max_epochs=N
runner.run(model_keys=[key], condition_keys=["base_fp32"], ignore_saved=True)
```

**The `__main__` guard in that harness is load-bearing.** macOS spawns DataLoader
workers, and each worker re-imports `__main__`. Without the guard every worker
restarts the whole verification run, and it surfaces as a confusing
`DataLoader worker (pid ...) exited unexpectedly` for *every* model. If you see
that error across the board, check the guard before suspecting memory.

**Rollout caches are versioned** (`HEURISTIC_ROLLOUTS_FILENAME`, currently
`heuristic_rollouts_v3.pt`). Bump the version for **both** a change of labelling
policy and a change of payload schema — v3 added per-step episode ids, and
reusing a v2 file raised `KeyError: 'episode'`. `dataset_exists` sentinels in
`data.py` reference the same constant, so bumping it correctly reports "not
cached" and triggers a rebuild.

**Verification status of §2.** All 22 models build and forward on-device; the 9
models changed by fixes 1–5 pass 2-epoch end-to-end pipeline runs; the 3 RL
models pass 3-epoch runs after the episode split. The forecasters' 2-epoch
numbers already land in the published neighbourhood (LSTM 0.296 vs Informer
0.247, GRU 0.303 vs Autoformer 0.336), which is the main evidence that the
leakage fix worked — the pre-fix numbers were 3–5× optimistic.

**Verification status of §3.** Short end-to-end pipeline runs, not full budgets:
`gcn` across all five conditions (`base_fp32`, `base_q8`, `base_q4`,
`dendrites_fp32`, `dendrites_q4`) with no PAI warnings and dendrites inserting
(92231 → 368711 params); `ppo_bipedalwalker` 4 epochs base and 10 epochs
dendritic; `pointnet_modelnet40` 10 epochs base. `ruff check src/` is clean. The
numbers those runs produced are quoted in the subsections above. **None of this
is a full-budget result** — that is what the §3.4 sweep is for.

**Do not trust `dendritic_pai_graphs` CSV `best_epoch` values** without checking
them against the PNG; 5 of 14 disagreed when this was measured. A known
pre-existing issue, unrelated to the current work. The directory it referred to
(`results/dendritic_pai_graphs`) no longer exists and is in neither archive, so
this stands as a caution about the graphs PerforatedAI emits, not as a pointer
to files you can still inspect.
