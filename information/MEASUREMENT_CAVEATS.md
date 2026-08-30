# Measurement caveats: root causes and fixes

Three measurement caveats were recorded (without being fixed) during the `dynamic5`
run — see `experiments/dynamic5/reference/BENCHMARKS.md` §"The quantization kernels"
and `experiments/dynamic5/report.md` §6.6. This document traces each one to the
exact code responsible and lays out fix options. Line numbers are current as of
commit `675105a` (branch `dynamic5-baselines`). Caveats §4 and §5 were found live
during the `dynamic7` run itself, on 2026-08-28.

Status at a glance:

| # | caveat | root cause found | fix applied |
|---|---|---|---|
| 1 | `q2` collapses to 3 knife-edge levels | yes — `compat.py` kernel math | **yes** — superseded by §6's MSE-optimal calibration, see §1 |
| 2 | `tcn_forecaster` `q1`/`q1_58` overflow to ~1e9–1e10 | **superseded** — was blamed on architecture; the real cause was a missing scale factor, see §6 | **yes, via §6** — original diagnosis and prediction both corrected in §2 |
| 3 | `dendrites_fp32` vs `dendrites_q*` param-count mismatch | yes — two independent, unsynchronized checkpoint systems | **yes** — see §3 |
| 4 | `gcn/dendrites_q8` transient checkpoint-reconstruction crash | partial — mechanism guarded, exact race not confirmed | **yes, via §5** — the "transient" framing was wrong, see §5 |
| 5 | `actor_critic`/`m5` ship a phantom randomly-initialized dendrite in every `dendrites_q*` arm | yes — `_split_compatible_state` only checks one direction | **yes** — see §5 |
| 6 | `q1`/`q1_58` had **no scale factor at all**; `q4`/`q8` calibrated on outliers | yes — `compat.py` kernel math | **yes** — see §6 |
| 7 | run-to-run variance (3.4pp) exceeds the effect being measured (0.3pp) | yes — nothing was seeded | **partly** — `--seed` added, but error bars still need multiple seeds, see §7 |
| 8 | the collapse guard killed 4 of 7 dendritic runs mid-dendrite-phase (plus `lenet5` in an ordinary plateau); only `gcn` and `actor_critic` completed their schedules | yes — a dendrite phase freezes validation by construction, and the rescue path was unreachable | **yes** — see §8 (counts corrected after §9) |
| 9 | `actor_critic dendrites_fp32` `history.csv` shows 60 of 145 real epochs | yes — `_persist_over_budget_snapshot` splits over-budget rows into `continued_until_complete/` **by design** | **n/a** — reading rule documented, see §9 |
| 10 | PAI's zero-seeded running average corrupts best-model tracking for every non-positive-maximize metric | yes — the first-score seeding branch never ran at `initial_history_after_switches=0` | **yes** — see §10 |
| 11 | a dendrite inserted late trains at a learning rate at or near zero, so it cannot earn its parameters | yes — the LR is a pure function of the absolute epoch index, and the dynamic tail sits past the anneal | **yes** — see §11 |

**Anything measured at `q1`, `q1_58`, `q2`, or `q4` before 2026-08-28 is invalid**
(§6), and every `dendrites_q*` number for `actor_critic` and `m5` in `dynamic7` is
invalid independently of that (§5). A full rerun is required; see §7.

**Superseding all of the above: every `dendrites_*` number produced before the §8
fix is invalid except `gcn`'s and `actor_critic`'s**, at every bit width. The
invalidated arms trained for a fraction of their base arm's epochs and never
received a working dendrite, so they do not measure dendrites at all. `gcn` and
`actor_critic` completed their full dendrite schedules (§8, §9) and their numbers
stand. `base_*` columns are unaffected. See §8.

---

## 1. `q2` is not really 2-bit and is knife-edge sensitive

**Where:** `src/dendritic_benchmark/compat.py`, `symmetric_quantize_tensor` (~line 984).

```python
def symmetric_quantize_tensor(tensor, bit_width):
    if bit_width >= 16: return tensor.clone()
    if bit_width <= 1: return tensor.sign().clamp(min=-1, max=1)
    levels = 2**bit_width - 1
    max_abs = tensor.abs().max()
    if max_abs == 0: return tensor.clone()
    scale = max_abs / (levels // 2)
    return torch.round(tensor / scale).clamp(-(levels//2), levels//2) * scale
```

**Root cause.** At `bit_width=2`: `levels = 2**2 - 1 = 3`, so `levels // 2 == 1` (integer
division) and `scale = max_abs / 1 = max_abs`. The kernel becomes
`round(w / max_abs).clamp(-1, 1) * max_abs` — exactly three output values,
`{-max_abs, 0, +max_abs}`. Two separate defects fall out of that one integer-division
edge case:

- **It's ternary, not 2-bit.** A genuine signed 2-bit integer has 4 levels
  (`{-2,-1,0,1}` or similar); this kernel only ever produces 3. It is strictly a worse
  quantizer than `q1_58` (which is *also* 3 levels, `{-1,0,+1}`, but scaled by
  `std(w)*0.5` — a robust statistic). Measured on every model in `dynamic5`: `q1_58`
  beat `q2` in all 5 cases (e.g. lenet5 0.7495 vs 0.2916).
- **The scale is one outlier weight.** `max_abs` is a single value out of tens of
  thousands of weights, and the survival test `|w| ≥ 0.5·max_abs` inherits that
  fragility. Two `lenet5 base_fp32` checkpoints with statistically indistinguishable
  weight distributions (std within 2%) produced retained-weight fractions of 2.83%
  and 9.92% purely because their single largest weight differed, and the `q2` score
  swung from 0.9588 to 0.2916 — a 67-point spread with no change in "how good" the
  network actually is. `q4` doesn't have this problem: `levels // 2 = 7`, so the scale
  is `max_abs / 7`, an order of magnitude less sensitive to any one weight.

**Fix chosen and implemented** (user chose "fix it properly" — both of the two
options below, applied together, `bit_width == 2` only):

- **(a) Make it actually 2-bit.** `qmin = -2`, `qmax = 1` — the standard signed
  integer range (`{-2,-1,0,1}`, matching two's-complement int2). Real 4 codes
  instead of 3.
- **(b) Base the scale on a robust statistic instead of the true max.** Scale is now
  `tensor.abs().float().quantile(0.999) / 2` — the 99.9th percentile of `|w|`,
  divided by the largest signed-code magnitude, not `tensor.abs().max()`. A single
  outlier weight can still get clamped to the extreme code, but it can no longer set
  the scale for the entire tensor.

```python
if bit_width == 2:
    qmin = -2
    qmax = 1
    robust_max = tensor.abs().float().quantile(0.999)
    if robust_max == 0:
        return tensor.clone()
    scale = robust_max / max(abs(qmin), abs(qmax))
    return torch.clamp(torch.round(tensor / scale), qmin, qmax) * scale
```

`bit_width` 4 and 8 are **untouched** — same formula, verified byte-identical output
(`torch.equal` on 10k-element random tensors). Their level counts (15, 255) were
never knife-edge-fragile the way 3 was, so there was nothing to fix there, and
changing them would have broken comparability with every stored `q4`/`q8` result for
no measured benefit.

**Verified** (synthetic tensors, not yet against a re-run model):
- Level count: 4 distinct outputs at `bit_width=2` on ordinary symmetric random tensors
  (was ≤3).
- Outlier robustness: two "statistically identical" distributions differing only in
  their single largest weight (replicating the `top10`/`dynamic5` `lenet5`
  `max|w|=0.4799` vs `0.3487` case) now produce **identical** survival fractions
  (ratio 1.00×, was ~3.5×).
- A true outlier (`+5.0` or `-5.0`, vs. a background std of 0.02) now gets clamped
  to the nearest extreme code instead of setting `scale = 5.0` and crushing the rest
  of the tensor to 0.
- Edge cases (all-zero, single-element, all-negative, 4-element bias tensor) all
  still return finite, correctly-shaped output.
- `q4`/`q8` outputs unchanged (`torch.equal` true against the pre-fix formula).

**Consequence, stated plainly:** every stored `q2` number in `top10` and `dynamic5`
is now produced by a different kernel than the 7-model run will use. `q2` results
from before this fix and after it are **not comparable** — this was the tradeoff the
user explicitly accepted in choosing to fix it now, on the reasoning that a new
7-model experiment is a clean version boundary to absorb that break rather than
carrying the flawed kernel forward again.

---

## 2. `tcn_forecaster` overflows at `q1`/`q1_58` — ~~moot, model is being dropped~~

> **CORRECTED 2026-08-28. Both the diagnosis and the prediction below are wrong.**
>
> The section reasons that binary/ternary quantization "pins every weight to
> `±scale`", making the overflow an inherent consequence of `tcn_forecaster`'s
> depth. It was not. `binary_quantize_tensor` and `ternary_quantize_tensor`
> carried **no scale factor at all** — they returned a bare `{-1, 0, +1}`
> indicator, so a layer trained to `std ≈ 0.005` came back with every surviving
> weight at magnitude exactly `1.0`. The `sqrt(fan_in)` gain argument is real,
> but it was compounding on top of a ~200x amplification that should never have
> been there. Depth determined *which* model exploded first, not *whether* one
> would. See §6 for the kernel and the fix.
>
> The prediction — "none of `textcnn`, `m5`, or `mpnn` have the specific
> combination that broke `tcn_forecaster`, so this is not expected to recur" —
> was falsified by the `dynamic7` data for two of those three models:
>
> | model | fp32 | `q1_58` | `q1` | |
> |---|---|---|---|---|
> | `mpnn` (RMSE ↓) | 0.7204 | **617.0** | **1030.7** | same overflow, ~7 orders smaller than `tcn_forecaster` only because it is 4 layers deep, not 10 |
> | `m5` (Acc ↑) | 0.9360 | 0.1135 | 0.0822 | collapsed to 12-class chance (≈0.083) rather than overflowing |
> | `textcnn` (Acc ↑) | 0.9162 | 0.8538 | 0.8445 | genuinely graceful — the one case the prediction got right |
>
> The section's closing instruction was right in spirit and wrong in remedy: it
> proposed working *around* the blow-up (drop the conditions, clamp the output,
> clip per layer). The correct remedy was to fix the kernel, which makes all
> three workarounds unnecessary. Its final sentence — "it should be checked once
> real `q1`/`q1_58` numbers come back, not assumed" — is the part that held up,
> and is why this was caught.
>
> Original text preserved below for the record.

**Where:** interaction between `TCNForecaster` (`models.py` ~line 947) and
`binary_quantize_tensor`/`ternary_quantize_tensor` (`compat.py` ~line 990-1005).

**Root cause.** This run's `TCNForecaster` fix (receptive field 61→125) added a 5th
`TemporalBlock`, taking the network from 8 to 10 `Conv1d` layers, and kept the RevIN
denormalization (`out * std + mean`) at the output. Binary/ternary quantization pins
every weight to `±scale` (no near-zero values survive), so each `Conv1d`'s effective
gain becomes roughly `sqrt(fan_in) ≈ sqrt(64·3) ≈ 14`. Ten such layers compound
multiplicatively: `14^10 ≈ 3×10^11`, matching the observed `base_q1 ≈ 2×10^10` order
of magnitude, and RevIN's final multiply by the input window's `std` compounds it
further. `base_q1_58` and `base_q1` were 2.31e9 and 2.06e10 in `dynamic5`, against
`top10`'s pre-fix 1.04 and 3.66 (the old, shallower architecture was still bad — MAE
far above the 0.41 FP32 baseline — but not numerically explosive).

**Why no fix is being written here:** the user's plan for the 7-model run removes
`tcn_forecaster` entirely, so this caveat has no model left to apply to. The general
lesson is worth keeping for whichever of the 3 replacement models turns out to be
deep or uses a multiplicative output normalization (RevIN-style): **binary/ternary
quantization on a deep or multiplicatively-renormalized network needs either (a)
excluding `q1`/`q1_58` from that architecture's reported conditions, (b) per-layer
rather than per-tensor clipping so gain doesn't compound across depth, or (c) a
sanity clamp on the final output before scoring.** None of `textcnn` (4 parallel
conv branches, no compounding depth), `m5` (4 sequential conv layers, no output
renormalization), or `mpnn` (4 message-passing steps with GRU-gated updates, which
bound activations the way LSTM/GRU gates generally do) have the specific combination
that broke `tcn_forecaster`, so this is not expected to recur in the new 7-model set
— but it should be checked once real `q1`/`q1_58` numbers come back, not assumed.

---

## 3. `dendrites_fp32` vs `dendrites_q*` param-count mismatch — root cause and fix

This is the one with a real, unambiguous bug underneath it, not just a design
trade-off. It also explains a second symptom that was **not** previously connected to
it: in `results/top10`, `lstm_forecaster`, `tabnet`, `mpnn`, and `gru_forecaster` all
show their dendritic arm's **test** metric (`metric_value`) *regressing* relative to
`base_fp32`, while their `best_metric_value` (the validation score dendrite-switch
decisions were actually made on) is flat or *better* than the base arm. `textcnn` is
the one extra model that does **not** show this symptom — its `metric_value` and
`best_metric_value` agree, which turns out to mean its dendrite structure simply
never changed after its best epoch. That is the same bug in two runs of the pipeline.

### The two independent, unsynchronized checkpoint systems

**System A — the benchmark's own best-epoch tracking.** `training.py::_record_best_epoch`
(~line 2590) snapshots `model.state_dict()` (a plain value dict) every time validation
improves:

```python
state.best_metric = val_metric
state.best_epoch = epoch + 1
state.best_state = {k: v.detach().cpu().clone() for k, v in _unwrap_compiled(model).state_dict().items()}
```

At the end of training, `_load_compatible_best_state(model, best_state)`
(~line 2666) loads that snapshot back into the **live, final-epoch** model:

```python
for key, value in best_state.items():
    ...
    if current_shape is None or source_shape is None or current_shape != source_shape:
        skipped.append(key)   # <-- silently dropped, not reset, not warned loudly
        continue
    compatible_state[key] = value
missing, unexpected = plain_model.load_state_dict(compatible_state, strict=False)
```

If PAI added a dendrite **after** the recorded best epoch — training continued past
it without beating that validation score again, which is normal, expected behaviour
for a plateau-triggered switch scheduler — the live model has parameter tensors
(the new dendrite's weights) that don't exist, or have a different shape, in
`best_state`. Those get **skipped**, not reset to a pre-dendrite value: they stay
whatever the live, continued-training run left them at. The result saved to
`model.pt` (`training.py:1374`, right after this restore) and reported as
`dendrites_fp32`'s `metric_value` is therefore a **hybrid**: old best-epoch values
for everything whose shape didn't change, plus leftover post-best-epoch training
values for anything that did. This is a plausible explanation for the test-metric
regression seen in `lstm_forecaster`/`tabnet`/`mpnn`/`gru_forecaster`: the "restored
best model" isn't actually the best model.

**System B — PAI's own switch checkpoints**, independent of the above. Every time
PAI performs an n→p switch (adds a dendrite), its own library code writes
`switch_N.pt` under the PAI save directory (`compat.py::save_pai_system`, backed by
`perforatedai.utils_perforatedai.save_system`). For a **post-training-quantization
condition** (`dendrites_q8`, `..._q4`, etc. — 10 of the 12 conditions, which never
train), `pipeline.py::_load_source_checkpoint` (~line 253) has to reconstruct the
dendrite *structure* from scratch before it can load any weights into it, since a
freshly-instantiated model has zero dendrites. It does this by asking for the
**latest** switch checkpoint, unconditionally:

```python
pai_checkpoint_name = latest_pai_switch_checkpoint(source_save_name)   # compat.py:880 — max(switch_N)
if pai_checkpoint_name is not None:
    model = load_pai_system_checkpoint(model, source_save_name, pai_checkpoint_name)
    ...
model = self._load_state(model, checkpoint_path, strict=False)   # loads model.pt onto that structure
```

`latest_pai_switch_checkpoint` (`compat.py:880`) globs `switch_*.pt` and returns
whichever has the highest number — i.e. the dendrite structure as of PAI's **last**
switch event, full stop. Nothing here asks "which structure did the best epoch (or
even the final epoch) actually have?" — it's simply the maximum. `model.pt` (System
A's hybrid output) is then loaded onto that skeleton with `strict=False`, so any
further shape mismatch is, again, silently dropped.

**Net effect:** up to three different dendrite structures can be in play for a
single model — (1) the structure at the recorded best epoch, (2) the structure the
live model ended training with, (3) whatever PAI's last switch checkpoint captured
— and nothing in the pipeline asserts any two of them agree. `dendrites_fp32`'s
`param_count` reflects (2) (the live model's structure, values partially from (1)).
`dendrites_q*`'s `param_count` reflects (3), with values loaded from a file that was
itself built against (2). When (2) and (3) coincide — no switch happened between the
last checkpoint and the end of training, the common case — everything lines up and
nothing looks wrong. When they don't, the param-count and (silently) the weight
values diverge, and nothing flags it.

### Fix implemented

Two changes, both defensive rather than semantic — they turn *silent* corruption
into either a correct result or a loud failure, they don't change what any
condition is defined to mean:

**Fix A (`training.py::_load_compatible_best_state`).** If restoring `best_state`
onto the live model would skip any tensor for a **shape** mismatch (as opposed to a
harmless key-set difference, e.g. `_is_ignorable_state_key` entries), that means the
dendrite structure changed after the best epoch and the best-epoch model can no
longer be faithfully reconstructed from a value-only snapshot. In that case: don't
apply the partial/hybrid restore at all — evaluate and persist the **final live
model** instead (structurally self-consistent, safe to reconstruct downstream), and
record which epoch was actually used rather than reporting a stale `best_epoch`/
`best_metric_value` for a structure that no longer exists in the saved artifact. Log
this loudly (`print`, same convention as the existing `[state] skipped ...` message)
so it's visible in run logs, not just inferable after the fact from a metrics
mismatch.

**Fix B (`pipeline.py::_load_source_checkpoint`).** After loading `checkpoint_path`
onto the `latest_pai_switch_checkpoint`-reconstructed skeleton, check whether
anything was skipped. With Fix A in place, `model.pt` is always structurally
self-consistent (it's either a clean best-epoch model or a clean final model, never
a hybrid), so a skip here means the switch-checkpoint skeleton doesn't match
`model.pt`'s own shapes — which should now be treated as a hard error surfaced at
condition-run time (aborts that condition with a clear message identifying the
mismatched keys), not a silently-degraded result shipped into `record.json`.

Both are implemented in this session; see the diff on branch
`dynamic5-baselines` (or whatever branch the 7-model run is launched from) for the
exact change. Neither changes any already-*consistent* result — a model whose
dendrite structure never changed after its best epoch (like `textcnn`, and like most
models most of the time, since PAI's plateau scheduler is specifically designed to
stop adding dendrites once they stop helping) produces byte-identical output before
and after.

### What this predicts for re-running the affected `top10` models

If `lstm_forecaster`, `tabnet`, `mpnn`, and `gru_forecaster` are re-run with the fix,
one of two things happens to each: either the dendrite structure turns out to be
stable after the best epoch after all (in which case nothing changes — the fix is a
no-op for that record), or it wasn't, in which case `dendrites_fp32`'s `metric_value`
should move *toward* `best_metric_value` (since it's no longer a corrupted hybrid) —
i.e. these four models' dendritic gain may currently be **understated**, not
overstated. `mpnn` is one of the three models chosen for the new 7-model run
(see `MODEL_SELECTION.md`), which makes it a natural test case: its `top10` numbers
disagree with this prediction only if the fix turns out not to fire for it.

---

## 4. `gcn / dendrites_q8` — Fix B fired correctly, but the crash it caught was
transient, and the auto-restart safety net wasn't watching

Observed live during the `dynamic7` launch, 2026-08-28. **Not a data-quality
problem** — the stored `gcn` numbers are fine (see below) — but a real gap in how
this kind of crash gets recovered from, worth recording before it's forgotten.

**What happened.** `gcn`'s `dendrites_fp32` training (stream_3) hit a `DOING_HISTORY`
switch trigger at epoch 117 that added a new candidate dendrite, and in that same
epoch PAI's own scoring immediately judged the candidate not to have helped and
declared training complete:

```
Returning True - History and last improved is hit
The newest added dendrites did not improve system and 2 > 2 so returning training_complete.
```

The condition finished and reported `dendrites_fp32 — Accuracy: 0.7990` at
`10:42:17`. The very next condition, `dendrites_q8`, needs to reconstruct the
dendrite structure from PAI's own checkpoint system before it can quantize
`gcn`'s trained weights (`pipeline.py::_load_source_checkpoint`), and that
reconstruction disagreed with the raw final state saved to `model.pt` — exactly
the mismatch Fix B (§3) was written to catch:

```
RuntimeError: [state] source-checkpoint structure does not match the PAI
switch-checkpoint-reconstructed model -- refusing a partial load. Mismatched
tensors: conv1.linear.dendrite_module.dendrites_to_candidates.0,
conv2.linear.dendrite_module.dendrites_to_candidates.0.
```

The worker process for stream_3 exited on this uncaught exception, 4 of `gcn`'s
12 conditions still unrun.

**Fix B behaved exactly as intended.** This is not a case where the fix was wrong
to fire — a genuine structural disagreement existed between the two checkpoint
systems (§3's root cause, applied here to a newly-created, never-tested candidate
dendrite rather than a stale best-epoch snapshot), and refusing the partial load
instead of silently producing a `dendrites_q8` record with the wrong `param_count`
is the whole point of that fix.

**How it actually got recovered — manually, not automatically.** `run_20260828_104616.txt`
shows a *separate* `dqb run --models gcn --dynamic-dendritic-training …` invocation
launched at `10:46:16`, about 4 minutes after the crash, which skipped the 7
already-recorded conditions and successfully trained/quantized the remaining 5 —
`dendrites_q8` through `dendrites_q1` all completed within about 10 seconds
(`10:46:16`–`10:46:26`), with no error. This was Codex noticing the dead stream and
relaunching it, not this codebase's own recovery path.

**Why the built-in auto-restart didn't catch it.** `pipeline.py::_watch` (~line
1709) already has exactly this kind of recovery — up to `_MAX_WORKER_RESTARTS = 5`
automatic respawns per stream, 30s apart, added 2026-08-10 for a different failure
mode (silent SIGKILLs). It didn't fire here: `run_progress.log` has no
`"restarting"`/`"crashed"`/`"exited with code"` line anywhere in it, and as of this
check the top-level `dqb run --jobs 7` watcher process isn't running at all — both
surviving workers (`m5`, `textcnn`) have `PPID 1` (reparented to `launchd`), meaning
their original parent already exited. `run_progress.log`'s last entry is frozen at
`11:44:56` while the workers keep training well past that timestamp. So the restart
safety net exists in the code but wasn't an active supervisor for this launch by
the time (or possibly before) the crash happened — **if `m5` or `textcnn` crash
before finishing, nothing will restart them either; each would need the same kind
of manual single-model relaunch `gcn` got.**

**Open question — why did the identical on-disk files fail once and then
succeed once, unmodified?** `dendrites_fp32`'s `record.json`/`model.pt` were not
regenerated between the two attempts (same `10:42:17` mtime both times), so the
retry reconstructed from and loaded exactly the same bytes that had just failed.
A deterministic structural incompatibility baked into those files would have
failed again identically; it didn't. The mismatch was therefore tied to something
about reconstructing the structure *within the same long-lived worker process that
had just finished writing it*, not to the files themselves — a fresh process
reading them cold, four minutes later, worked immediately. That "identical run,
different outcome" signature is the same one already on record for
[[mps-nonblocking-eval-race]] (PointNet's corrupted eval input): a leading but
**unconfirmed** hypothesis is an async write/materialization race in the
checkpoint round-trip (MPS tensor data, or a buffered file write, not fully durable
by the time the same process reads it back), not a bug in the comparison logic
itself. This wasn't investigated further this session — root-causing it would mean
instrumenting `save_pai_system`/`load_pai_system_checkpoint` (`compat.py`) around
this exact epoch boundary on a reproduction run, which didn't happen here.

**Is the stored `gcn` data trustworthy?** Yes. The successful retry went through
the identical Fix B compatibility check and passed it cleanly — no partial load
occurred, so `gcn`'s `dendrites_q8`–`dendrites_q1` records reflect the same
structure `dendrites_fp32` actually reported, not a silently-degraded
reconstruction.

**Follow-up worth doing, not done here:**
- If this recurs, wrap `_load_source_checkpoint`'s checkpoint-reconstruction step
  in a small in-process retry (a few attempts, short backoff) before letting the
  exception propagate — turns a manual-relaunch recovery into a self-healing one,
  consistent with the philosophy `_watch`'s stream-level restart already uses one
  level up.
- Separately: `_watch`'s auto-restart is only a safety net while its parent process
  is alive. Confirm whether this launch's watcher exited intentionally (e.g. `dqb
  run --detach` semantics, or a terminal that was closed) or crashed itself — if
  the latter, the watcher's own crash resilience needs the same kind of scrutiny
  applied to the workers it supervises.

---

## 5. `actor_critic` and `m5` shipped a phantom, randomly-initialized dendrite in every `dendrites_q*` arm

**Found:** 2026-08-28, while analyzing the partial `dynamic7` results.
**Status: fixed.** This also supersedes §4's "transient" framing — see the end of
this section.

### Symptom

Six of the seven `dynamic7` models agree on `param_count` between their FP32
dendritic arm and their five quantized dendritic arms. Two do not:

| model | `dendrites_fp32` | `dendrites_q*` | |
|---|---|---|---|
| `gcn` | 369,066 | 369,066 | consistent |
| `lenet5` | 185,354 | 185,354 | consistent |
| `saint_adult` | 299,266 | 299,266 | consistent |
| `mpnn` | 1,427,336 | 1,427,336 | consistent |
| **`actor_critic`** | **52,617** | **71,059** | **+35%** |
| **`m5`** | **50,456** | **75,696** | **+50%** |

This is the same *symptom* as §3, but §3's fix does not cover it and its guard
does not catch it. §3 fixed the case where the source checkpoint has tensors the
target cannot accept. This is the mirror case: the target has tensors the source
never supplies.

### Root cause

`pipeline.py::_split_compatible_state` iterated the **source** state dict only:

```python
for key, value in state.items():        # source keys only
    current_value = current_state.get(key)
    if not _is_compatible_state_value(current_value, value):
        skipped.append(key)
        continue
    compatible_state[key] = value
```

A target parameter absent from `state` is never visited, so it is never reported.
`_load_compatible_state` then calls `load_state_dict(compatible_state,
strict=False)` — and `strict=False` is exactly the flag that makes missing keys
silent. Those parameters keep whatever `perforate_model` / `load_pai_system`
initialized them to. They are then quantized and scored as if trained.

Dumping the actual tensors confirms it. `actor_critic`'s quantized arms carry a
complete second dendrite that its FP32 arm does not have:

```
only in dendrites_q4, per perforated layer:
  backbone.0.dendrite_module.layers.1.{weight,bias}      <- second dendrite
  backbone.0.dendrite_module.dendrites_to_candidates.0   <- candidate wiring
  backbone.0.dendrite_module.dendrites_to_dendrites.{0,1}
  backbone.0.dendrites_to_top.1
```

and `layers.1.weight` has `std = 0.00501` against the trained `layers.0.weight`'s
`std = 0.00491` — an untrained draw from the same initializer, not a trained
weight. `m5`'s case is worse: its FP32 arm has the dendrite *bookkeeping buffers*
(`dendrite_values.0.*`) but **no dendrite weights at all**, while its quantized
arms have a full `layers.0` + `dendrites_to_top.0`. The two arms are not the same
model with different precision; they are different architectures.

### Why the structures diverged

Two checkpoints are written at different points and were assumed to agree:

- `model.pt` — written from the model as it stands after the best-epoch restore
  decision (`training.py`, the `if best_state is not None:` block) and before
  quantization. This is what supplies the **weights**.
- the `PAI_RESUME_NAME` snapshot — written *inside* the epoch loop by
  `_save_pai_resume_state`. This is what supplies the **structure**.

If the final epoch added a candidate dendrite, or the best-state restore declined
a structure change (which §3's Fix A makes it do deliberately), the two describe
different architectures. Whichever way they differ decides the failure mode:

- snapshot has **fewer** tensors than `model.pt` → §3's guard fires, the run
  crashes loudly. This is what happened to `gcn` in §4.
- snapshot has **more** tensors → nothing fires, the extras stay at init, and a
  plausible-looking wrong number is written to `record.json`.

So §4 and §5 are the same bug seen from opposite sides. **§4's "transient"
framing was wrong**: the `gcn` crash was not a race, and the manual relaunch did
not "self-heal" it — the relaunch simply resumed from a point where the two
checkpoints happened to agree. Nothing about it was nondeterministic. The open
question §4 recorded ("identical unmodified on-disk files failed once then
succeeded once") is answered: the files were not the inputs that differed; the
in-memory PAI structure at snapshot time was.

### Fix implemented

Pin the structure to the artifact rather than to the epoch loop.

1. **`compat.py`** — new `PAI_ARTIFACT_NAME = "dqb_artifact"` snapshot name.
2. **`training.py`** — immediately after the best-epoch restore decision and
   before any quantization, i.e. at the one instant that is guaranteed to
   describe `model.pt`:
   ```python
   if use_dendrites and config.pai_save_name:
       save_pai_system(_unwrap_compiled(model), config.pai_save_name, PAI_ARTIFACT_NAME)
   ```
3. **`pipeline.py::_source_pai_checkpoint_name`** — prefer `PAI_ARTIFACT_NAME`
   ahead of `PAI_RESUME_NAME`, `"latest"`, `"best_model"`, `"final_clean_pai"`,
   and the `switch_N` fallback. Older results without the new snapshot fall
   through to the existing chain unchanged.
4. **`pipeline.py::_split_compatible_state`** — report mismatches in **both**
   directions, so a target key the source never supplies is an error rather than
   an invisible default:
   ```python
   for key in current_state:
       if _is_ignorable_state_key(key) or key in state:
           continue
       skipped.append(key)
   ```
5. **`pipeline.py::_load_compatible_state`** — the raised error now separates the
   two directions ("N target tensor(s) the source never supplies (would stay at
   init: ...)" vs "N shape mismatch(es)"), because the remedies differ.

(1)–(3) make the structures agree by construction; (4)–(5) are the guard for when
something still gets them out of step. Both are needed: a guard alone would turn
these two models from silently-wrong into loudly-failing, which is better but
still not a result.

### Verification

A full 12-condition `gcn` run on the fixed code, in a scratch results root,
happened to reproduce the exact triggering condition — the log contains:

```
[state] best-epoch structure does not match the final trained structure (a
dendrite was likely added after the best epoch) -- keeping the final model
instead of a partial restore.
```

That is §3's Fix A declining a hybrid restore, which is precisely what used to
leave the PAI snapshot describing a structure `model.pt` never had. On the fixed
code all six dendritic conditions came back at an identical **461,652**
parameters, and `dqb_artifact.pt` was written for each. Under the old code this
same sequence produced either §4's crash or §5's phantom dendrite.

### Scope of invalidated data

Every `dendrites_q8`/`q4`/`q2`/`q1_58`/`q1` record for **`actor_critic`** and
**`m5`** in `dynamic7`, and any earlier run whose `dendrites_q*` `param_count`
disagrees with its own `dendrites_fp32`. The FP32 dendritic arms are unaffected —
`model.pt` was always self-consistent; only the reconstruction was wrong.

This is not a rounding-level effect. It plausibly accounts for most of the
"dendrites degrade worse under quantization" signal in the partial results: the
two phantom-dendrite models are also the two with by far the worst quantization
retention at `q4` (`actor_critic` −0.094, `m5` −0.231 relative to their own
baselines), while the four structurally-consistent models sit within ±0.02 of
zero. That correlation is exactly what this bug predicts, and it means the
headline comparison could not be read off the current data even if §6 had not
also been true.

---

## 6. `q1`/`q1_58` had no scale factor; `q4`/`q8` were calibrated on outliers

**Found:** 2026-08-28. **Status: fixed.** This is the root cause behind §2 and
subsumes §1.

### The `q1`/`q1_58` defect

```python
def ternary_quantize_tensor(tensor):
    threshold = tensor.std(unbiased=False) * 0.5
    pos = (tensor > threshold).to(tensor.dtype)
    neg = (tensor < -threshold).to(tensor.dtype)
    return pos - neg                      # <- returns {-1, 0, +1}. No scale.

def binary_quantize_tensor(tensor):
    return torch.where(tensor >= 0, torch.ones_like(tensor), -torch.ones_like(tensor))
                                          # <- returns {-1, +1}. No scale.
```

Both return bare sign indicators. A layer whose weights have `std ≈ 0.005` comes
back with every weight at magnitude `1.0` — roughly 200x amplification, per
layer, compounding multiplicatively with depth. This is not "aggressive
quantization"; the quantized network is not an approximation of the trained one
at all. Published binary/ternary schemes all carry a per-tensor scale precisely
to prevent this: XNOR-Net uses `α = mean(|W|)`, BitNet b1.58 uses the same
absmean scale.

Two consequences beyond the magnitude blow-up:

- **`sign(0) = +1`.** An all-zero parameter became an all-ones parameter. `m5`'s
  `conv1.dendrites_to_top.0` is exactly that — a genuinely zeroed dendrite output
  gate, which binarization turned fully on.
- **PQAT inherited it.** `_qat_project_for_forward` calls the same kernels, so
  every PQAT fine-tune optimized against these weights too.

### The `q4`/`q8` defect

`symmetric_quantize_tensor` calibrated on `tensor.abs().max()`. One outlier
weight then sets the step size for the whole tensor: if the largest weight is 20x
the next largest, every ordinary weight collapses onto one or two codes. §1
identified this at `q2` and special-cased *only* `q2` with a `quantile(0.999)`
scale, leaving `q4` and `q8` on the outlier-sensitive path. That is why `m5`'s
`base_q4` fell 21pp (0.9360 → 0.7244) while every other model was fine at 4-bit:
`m5`'s first conv has `absmax 1.90` against `std 0.29`.

### Fix implemented

All in `compat.py`:

- **`ternary_quantize_tensor`** → BitNet b1.58 absmean:
  `s = mean(|W|)`, return `clamp(round(W/s), -1, 1) * s`.
- **`binary_quantize_tensor`** → XNOR-Net: `mean(|W|) * sign(W)`, and `s == 0`
  now returns zeros instead of ones.
- **`symmetric_quantize_tensor`** → one uniform signed-integer grid for every
  width (`qmin = -2**(b-1)`, `qmax = 2**(b-1)-1`), so §1's `q2` special case
  disappears into the general rule rather than sitting beside it. The scale
  divides by `|qmin|`, not `qmax` — dividing by `qmax` is what produced §1's
  three-level collapse.
- **`_calibrate_scale`** → replaces both `abs().max()` and the fixed 0.999
  percentile with an **MSE-optimal clip search**: try clip ratios from 1.00 down
  to 0.40 and keep the scale minimizing `||q(W) − W||²`. A fixed percentile is
  arbitrary in the other direction — at 8-bit there are 256 codes, the step is
  already fine, and clipping a genuinely large weight costs more than it buys.
  Searching per tensor *and* per bit width lands near 1.0 where codes are
  plentiful and clips hard at 2-bit where they are not, so no constant has to be
  right for every layer in the suite. Tensors above 65,536 elements are
  subsampled for the search; the chosen scale applies to the full tensor.

### Verification: both kernels, identical trained weights

Run-to-run variance is large enough here to swamp a kernel comparison across two
training runs (see §7), so both kernels were applied to the *same* stored
`base_fp32` checkpoint and evaluated on the same test set.

`mpnn`, RMSE ↓ — the regression case, where a missing scale cannot hide:

| | `fp32` | `q8` | `q4` | `q2` | `q1_58` | `q1` |
|---|---|---|---|---|---|---|
| old kernels | 0.3453 | 0.3464 | 0.4100 | 1.7367 | **295.71** | **493.97** |
| new kernels | — | 0.3456 | 0.3902 | **0.6774** | **0.9877** | **0.9867** |

The blow-up is gone: `q1`/`q1_58` land at ~0.99 RMSE — clearly degraded from
0.345, which is what a 1-bit model *should* look like — instead of 300–500x the
FP32 error. `q2` improves 2.6x. (Absolute values differ from `record.json`
because this harness scores raw model output without the pipeline's target
denormalization; both kernels are scored identically, so the comparison holds.)

`gcn`, Accuracy ↑ — the classification case:

| | `fp32` | `q8` | `q4` | `q2` | `q1_58` | `q1` |
|---|---|---|---|---|---|---|
| old kernels | 0.7960 | 0.7960 | 0.8040 | 0.7830 | 0.4170 | 0.5350 |
| new kernels | — | 0.7960 | 0.7930 | 0.7770 | **0.6290** | 0.4960 |

`q1_58` gains 21pp; `q4`/`q2` move by about 1pp in the other direction, which is
inside this model's noise (§7 measures `gcn`'s run-to-run `fp32` spread at
3.4pp). Note that the missing scale was *less* catastrophic for `gcn` than for
`mpnn`: a classifier's `argmax` is invariant to a uniform rescaling of the
logits, so a sign-only network can still rank classes correctly. That invariance
is why the defect survived this long — it is nearly invisible on accuracy
metrics and fatal on regression metrics.

Per-tensor sanity: on `m5`'s real `conv1.main_module.weight` (`std 0.282`,
`absmax 1.904`), relative reconstruction error `||q(W)−W|| / ||W||` is 0.013 /
0.186 / 0.500 / 0.728 / 0.811 at `q8`/`q4`/`q2`/`q1_58`/`q1` — monotone in bit
width, with scale preserved at every width (`q1` returns `±0.165 = mean(|W|)`,
not `±1.0`). All-zero tensors return zeros at every width.

### Scope of invalidated data

**Every `q1`, `q1_58`, `q2`, and `q4` number ever recorded by this benchmark**,
across `top10`, `dynamic5`, and `dynamic7`. `q8` numbers are affected in
principle but were near-lossless under both kernels (all seven `dynamic7` models
sit within 0.002 of their FP32 score at `q8`), so `q8` and `fp32` conclusions
stand.

---

## 7. What has to be re-run

§5 invalidates `actor_critic` and `m5`'s dendritic quantized arms; §6 invalidates
every `q4`/`q2`/`q1_58`/`q1` cell for every model and both arms. Between them,
the only `dynamic7` numbers that survive are the `fp32` and `q8` columns — which
is most of the *models'* cost but almost none of the *experiment's* claim, since
the claim is about low-bit behaviour.

The `fp32` and `q8` results that do survive:

| model | metric | `base_fp32` | `dendrites_fp32` | dendrite advantage |
|---|---|---|---|---|
| `actor_critic` | Action Acc ↑ | 0.9000 | 0.9907 | **+9.07pp** |
| `gcn` | Acc ↑ | 0.7960 | 0.7990 | +0.30pp |
| `saint_adult` | Acc ↑ | 0.8585 | 0.8601 | +0.16pp |
| `lenet5` | Acc ↑ | 0.9910 | 0.9925 | +0.15pp |
| `m5` | Acc ↑ | 0.9360 | 0.9327 | −0.33pp |
| `mpnn` | RMSE ↓ | 0.7204 | 0.7989 | −10.9% (worse) |
| `textcnn` | Acc ↑ | 0.9162 | (still training) | — |

A caution for the rerun that is independent of any bug: **four of these effects
are smaller than run-to-run noise, and this is now measured rather than
suspected.** The `gcn` verification run above is a second independent sample of
the same model under the same config:

| | first `dynamic7` run | verification run | swing |
|---|---|---|---|
| `gcn base_fp32` | 0.7960 | 0.7620 | **3.4pp** |
| `gcn dendrites_fp32` | 0.7990 | 0.7890 | 1.0pp |
| implied dendrite advantage | +0.30pp | +2.70pp | **2.4pp** |
| dendrites added by PAI | 3 (369,066 params) | 4 (461,652 params) | — |

No quantization is involved in any of those numbers. The `fp32` baseline alone
moves 3.4pp between runs — **eleven times the +0.30pp dendrite advantage the
first run reported** — and the advantage itself swings by 2.4pp, because
`DOING_HISTORY` is plateau-triggered and the two runs did not even settle on the
same architecture (3 dendrites vs 4). On Cora's 1000-node test set, +0.30pp is
three nodes.

`lenet5`'s +0.15pp is fifteen MNIST images against a 99.1% ceiling; `saint_adult`
and `m5` are the same order. **`actor_critic`'s +9.07pp is the only effect in the
table large enough to survive a single seed.**

### `--seed` (added 2026-08-28)

`dqb run --seed N` now exists, and `compat.seed_everything` is applied at two
points: before `build_task_bundle` (dataset splits and shuffle order are drawn
there) and before every individual condition. The second placement is the one
that matters most — it makes a model's `base_*` and `dendrites_*` arms draw the
**same initial weights**, so the two arms become a paired comparison and a
difference between them is attributable to the dendrites rather than to a
different lottery ticket. The flag passes through `_run_passthrough`, so parallel
workers inherit it instead of each running unseeded.

Verified end to end, not just at the RNG: `gcn base_fp32` run twice as two
separate process launches with `--seed 4242` returned
`metric=0.814000, best=0.776000, best_epoch=48` both times, bit-identical. The
two unseeded runs of the same condition returned 0.7960 and 0.7620.

`torch.use_deterministic_algorithms` is deliberately **not** enabled: MPS has no
deterministic implementation for several ops these models use, so it would fail
outright rather than silently disagree. Seeding removes the large run-to-run
swings; what remains is nondeterministic reduction order, worth far less than the
3.4pp that was the actual problem.

**A seed makes a run reproducible; it does not make an effect real.** A single
seeded run still cannot tell you whether `lenet5`'s +0.15pp is signal. Error bars
need several seeds — `config/run_dynamic7.sh` reads `SEED` from the environment
for exactly this, at ~28 min per replicate for the 5-model set.

---

## 8. The collapse guard killed four of seven dendritic runs mid-dendrite-phase
*(originally reported as "six of seven"; corrected after §9 reclassified
`actor_critic` as a completed run and the table below reclassified `lenet5`'s
kill as an ordinary plateau outside any dendrite phase)*

**Found:** 2026-08-28, investigating why `mpnn`'s dendritic arm was *worse* than its
base arm at fp32 (RMSE 0.7921 vs 0.7162, and RMSE is minimize — so the dendrites
genuinely lost).

**Where:** `src/dendritic_benchmark/training.py` — `_training_collapsed`
(`_COLLAPSE_GUARD_EPOCHS = 12`) and `TrainingConfig.max_dendrite_phase_epochs`.

### The observation

`mpnn`'s two arms did not get the same amount of training:

| arm | epochs | best epoch | test RMSE ↓ |
|---|---|---|---|
| `base_fp32` | **200** | 180 | **0.7162** |
| `dendrites_fp32` | **24** | 3 | 0.7921 |

The dendritic arm stopped after 24 epochs. Epochs 13–24 have a validation metric
that is *bit-identical* — `0.8645049929618835`, twelve times, across **every**
`val_*` column (loss, MAE, MAPE, R², max_error, …) — while every `train_*` column
kept moving. And the frozen value is exactly epoch 2's.

### Root cause

PAI switched to dendrite mode at epoch 12 (`Returning True - History and last
improved is hit`, then `Module … calling set mode p` for all 21 wrapped modules).
**In mode `p` the parent network is frozen and the candidate dendrites are not yet
wired into the output**, so the network's predictions cannot change: a bit-frozen
validation metric is the *expected, healthy* signature of a dendrite phase, not a
symptom of anything.

`_training_collapsed` cannot tell that apart from a dead network. It was written for
the 2026-07-29 DistilBERT divergence (constant majority-class output for 39 epochs)
and fires on any 12 consecutive identical validation values that are worse than the
best seen. A dendrite phase is exactly that. So at epoch 24 it fired:

```
[collapse] mpnn | dendrites_fp32: validation RMSE frozen at 0.864505 for 12 epochs,
worse than the best 0.751856 from epoch 4 — stopping this condition rather than
training a dead network.
```

The guard against this already existed and **could never fire**:
`max_dendrite_phase_epochs = 50` forces a switch once a phase overruns, but the
collapse guard trips at 12. 12 < 50, so the run was always killed 38 epochs before
the rescue. `grep -c "Forcing the switch"` across every log in the repo returns
**0** — the path had never once executed.

### Why PAI never left mode `p` on its own

PAI leaves the phase when no node's correlation has improved for
`p_epochs_to_switch` (2) epochs. With `running_average_pb = True` and
`history_lookback = 8`, the comparison is against an EMA. Feeding an EMA a constant
makes it converge *asymptotically* — improving by a smaller but always non-zero
amount every epoch — so the patience counter resets forever. The logs show it
directly: `last improved epoch` equals `epoch` on every single mode-`p` check.
This is the same mechanism already recorded for HISTORY mode in `compat.py`
(~line 257) and measured there at 91 of 92 dendrite-mode switch checks.

### Scope — this is not an `mpnn` bug

Counting restructure events in each run's own `history.csv` (the per-run record;
the `results/PAI/*/…param_counts.csv` and `…switch_epochs.csv` files are **not**
usable for this — PAI's save directory is never cleared between runs, so those
files carry switch epochs from earlier launches that contradict the current one):

| model | base epochs | dendrite epochs | restructures | frozen tail | outcome |
|---|---|---|---|---|---|
| `mpnn` | 200 | **24** | 1 (ep 12) | 12 | collapse-killed in first phase |
| `vae_mnist` | 50 | **24** | 1 (ep 12) | 12 | collapse-killed in first phase |
| `tcn_forecaster` | 80 | **24** | 1 (ep 12) | 12 | collapse-killed in first phase |
| `saint_adult` | 200 | **61** | 1 (ep 49) | 12 | collapse-killed in first phase |
| `actor_critic` | 60 | **145** (60 + 85 in `continued_until_complete/`) | 4 cycles | — | **completed** — see §9; the 60-row `history.csv` is a by-design split, not a truncation |
| `gcn` | 200 | 121 | 9 | 2 | **completed** — 4 full cycles, `pai_training_complete` |
| `lenet5` | 40 | 40 | 0 | 12 | collapse-killed, but **not** in a dendrite phase |

Four of seven (`mpnn`, `vae_mnist`, `tcn_forecaster`, `saint_adult`) entered
their first dendrite phase and never escaped it, so they never completed a single
dendrite cycle. `gcn` and `actor_critic` both escaped mode `p` unaided and
completed multi-cycle schedules; `gcn`'s phases lasted 3 epochs each (switches at
50→53, 72→75, 93→96, 107→110).

`lenet5` is a different case and the fix does not address it: it never switched at
all in 40 epochs, so its frozen validation (0.9912 accuracy, 12 epochs) is an
ordinary plateau on a small validation split, which is what the guard is *for*.

**Do not read `results/<model>/<cond>/paramCounts.csv` as a growth curve.** It is
written after training by repeating the single final `param_count` once per history
row (`training.py` ~line 1455), so it is flat by construction and says nothing about
when dendrites were added. An earlier draft of this section drew the wrong
conclusion from it.

### Fix

1. **`max_dendrite_phase_epochs: 50 → 8`.** Above the longest phase that has ever
   ended on its own (3, `gcn`) and below `_COLLAPSE_GUARD_EPOCHS`, so the forced
   switch now happens while the run is still alive.
2. **`_training_collapsed` exempts dendrite-phase epochs.** A new per-epoch
   `pai_dendrite_phase` flag is recorded in `history_row` (read from the tracker
   *before* `add_validation_score`, so it describes the phase the epoch actually ran
   under) and written to `history.csv`. The guard skips any window containing one.

The two together are what makes this safe. Dynamic runs iterate `itertools.count()`
— **unbounded** — and until now the collapse guard was the only thing stopping a
stuck run, so exempting those epochs on its own would have traded a truncated run
for an infinite one. Fix 1 bounds the phase, which restores termination; fix 2 then
stops the guard misfiring during the bounded phase.

### Verified

`mpnn dendrites_fp32` re-run from scratch at `--seed 0` with the fix
(`results/_fixverify`):

| | epochs | restructures | forced switches | collapses | test RMSE ↓ |
|---|---|---|---|---|---|
| before | 24 | 1 | 0 | 1 | 0.7921 |
| after | **50** | **5** (ep 12, 20, 31, 39, 50) | **2** (both at the 8-epoch limit) | **0** | **0.6976** |
| `base_fp32` | 200 | — | — | — | 0.7162 |

The phase structure is now visible in `history.csv`: epochs 13–20 and 32–39 carry
`pai_dendrite_phase=True` with validation frozen at 0.8645 (exempted, as intended),
and validation resumes moving at epochs 21 and 40 once the dendrite is switched in.
PAI reached `pai_training_complete` at epoch 50 instead of never.

**This does not mean dendrites help `mpnn`.** Its best validation epoch is still
epoch 3 — before any dendrite existed — across all 50 epochs, and PAI's own stop
reason is `noImprove`. The kept model's `num_cycles` is 0 with all
`dendrite_values.*.initialized` at zero, i.e. PAI tried two dendrites, neither
improved validation, and it discarded both (`retain_all_dendrites=False`). The
0.6976 is a 50-epoch network whose dendrites were rejected, beating a 200-epoch
base arm — which is a variance result, not a dendrite result. What the fix buys is
that the question can now be *asked*; the answer for `mpnn` is still "no".

Note also that the two arms are compared at different epoch counts by design
(`train_dendrites_until_complete` runs the dendritic arm until PAI finishes, 50
epochs here, versus the base arm's fixed 200). For a model like `mpnn`, whose base
arm's best epoch is 180, that asymmetry is a real confound independent of this bug.

### What this invalidates

**Every `dendrites_*` number in the 2026-08-28 seed-0 run except `gcn`'s and
`actor_critic`'s.** Not because they are mismeasured, but because they do not
measure what the column claims: four of the seven dendritic arms trained for a
fraction of their base arm's epochs and never completed a single dendrite cycle.
`mpnn`'s "dendrites are worse" is 24 epochs versus 200. `lenet5`'s "+26pp at q2"
is a 40-epoch model that never switched at all. Since the `q*` arms all load the
`dendrites_fp32` checkpoint, the dendritic half of the grid for those five models
inherits this. It is a larger invalidation than §5 or §6 and requires a rerun of
their dendritic conditions.

`gcn` and `actor_critic` are the two dendritic arms whose mechanism ran to
completion (§9 initially hid `actor_critic`'s). `gcn` is the one the
model-selection notes in `experiments/dynamic7/config/run_dynamic7.sh` demoted
to a smoke test for its 3.4pp noise floor (§7) — but note that was an
*unseeded, cross-run* noise floor; the seeded paired-arm design measures the
within-run delta, which three paired runs put at +1.9/+2.9/+2.2pp.

`base_*` columns are unaffected — they never enter this path.

---

## 9. `actor_critic dendrites_fp32` — `history.csv` undercounts the real run (RESOLVED — by design, not a bug)

**Found:** 2026-08-28, while re-checking each stuck model's raw log against §8's table.
**Root-caused:** 2026-08-28, during the full-code review.

`results/actor_critic/dendrites_fp32/history.csv` has exactly 60 rows, one
`pai_restructured=True` at epoch 58, and no collapse — read at face value this
looked like "ran out of its epoch budget 2 epochs into the dendrite phase," and
that is what got reported. It's wrong. The worker log for the same run
(`logs2/streams/stream_4.log`, `13:11:03`–`13:18:21`) shows continuous epoch
numbering to at least 101 (`Checking PAI switch ... total epochs 100, ... num_cycles:
4`), i.e. the run added dendrites across (at least) 4 cycles and kept training
well past epoch 60. `record.json`'s `best_epoch=117` is consistent with the log,
not the CSV. There is no `[collapse]` message for this run anywhere in the logs —
it ended in a plain `[done]`, so it was not cut short by §8's bug.

**Root cause — `_persist_over_budget_snapshot` (training.py), and it is
deliberate.** For dynamic dendritic runs (`train_dendrites_until_complete`),
the epoch history is split at the recipe's nominal `max_epochs` on save:
rows `1..max_epochs` become the canonical `history.csv`, and every row past
the budget is written to `continued_until_complete/history.csv` in the same
condition directory. Verified on disk: actor_critic's canonical file holds
epochs 1–60 (60 = the recipe budget) and
`continued_until_complete/history.csv` holds epochs 61–145 — 145 total,
consistent with the log and with `best_epoch=117`. Nothing was lost; the
"missing" rows were always one directory down.

Two reading rules follow:

1. **Epoch-count diagnosis on a dynamic dendritic run must concatenate
   `history.csv` + `continued_until_complete/history.csv`.** The canonical
   file alone understates any run that outlived its budget.
2. **`record.json`/`metrics.json` can name a `best_epoch` that is not in
   the canonical `history.csv` at all** (117 > 60 here) — that is the same
   split, not corruption. It also means the canonical record's test metric
   was produced by a model trained *past* the budget, which is one more face
   of the base-vs-dendritic epoch-budget asymmetry already flagged in §8.

`mpnn`, `vae_mnist`, `saint_adult`, and `tcn_forecaster` did not show the
mismatch simply because §8's collapse guard killed them *before* their
budgets, so they never had over-budget rows to split off.

## 10. PAI's zero-seeded running average corrupts best-model tracking for every non-positive-maximize metric (FOUND + FIXED 2026-08-28)

**Symptom that exposed it:** in dynamic8, `vae_mnist dendrites_fp32` finished
with val ELBO −92.71 (better than base's −92.85 val) yet *tested* at −94.15
(worse than base's −92.16) — a 1.4-nat val→test inversion where base shows a
+0.7-nat val→test gain.

**Mechanism.** PAI's score tracking is an EMA that, in our configuration, is
seeded at zero with weight `1/history_lookback` (=1/8; verified exactly:
vae's first running score −11.8696 = −94.9572/8, tcn's 0.04089 = 0.32713/8,
mpnn's 0.10351 = 0.82810/8). The upstream source
(`update_running_accuracy`) only seeds the running average from the first
*real* score while `epochs_since_cycle_switch <
initial_history_after_switches` — and that config defaults to **0**, so the
seeding branch never runs. At PAI's default `history_lookback=1` this is
harmless (the EMA weight is 1.0, running = raw); it was *our own override*
of `history_lookback` to 8 (added to stop switching on transient noise)
that armed the zero-seeded warm-up. Zero is a *better-than-anything-real*
score for every metric that is not positive-maximize:

- ELBO under maximize: real scores ≈ −92; the EMA descends from ~0 toward
  −92, i.e. PAI sees the score "worsen" every epoch of the entire run.
- MAE/RMSE under minimize: real scores ≈ 0.3–0.9; the EMA ascends from ~0
  toward them — again perpetual "worsening".

Consequences, all observed in dynamic8's PAI artifacts:

1. **Best model pinned at epoch ~1.** The EMA peak is during warm-up, so the
   gated best-model save never fires again. Final `best_arch_scores.csv` for
   vae/tcn/mpnn lists *only the base architecture* at its epoch-0/1 raw score
   (−94.827 / 0.32713 / 0.82810).
2. **`find_best_lr` no-improvement restarts** (the archived
   `*noImprove_lr_0*` snapshots).
3. **Restore-best-on-complete hands back a barely-trained network.** At the
   completion switch PAI restored its "best" — the epoch~1-era state — and
   rewound its own score history (final `Scores.csv` for vae/tcn/mpnn contains
   only epochs 0–1). The harness's own best-state restore then correctly
   *declined* (structure mismatch, §3's no-hybrid rule) for vae and tcn, so
   the PAI-restored early state is what got tested and shipped.

Accuracy-style metrics (positive, maximize) sit in the one benign quadrant:
the warm-up EMA looks *worse* than real scores, so best-tracking stays sane —
that is why `gcn` and `actor_critic` (full-length rising `Scores.csv`, three
architectures in `best_arch_scores.csv`) were unaffected. The same warm-up
shape is, however, exactly the mode-`n` rising-EMA switch-blocking trap that
kept `lenet5`/`distilbert` from ever switching (memory:
pai-running-average-blocks-switch) — one root mechanism, two failure modes.

**What this invalidates in dynamic8:**

- `vae_mnist dendrites_fp32` (and all `dendrites_q*` derived from its
  checkpoint): tested model is an epoch~1-era restore (test −94.15 ≈ epoch-1
  val −94.83 + the +0.7 val→test offset). The genuinely trained dendritic
  model (val −92.71, *better* than base) was never tested. The "dendrites
  hurt the VAE" reading is an artifact.
- `tcn_forecaster dendrites_*`: tested model is a PAI-restored **pre-dendrite**
  base-arch early state; the 0.3013-vs-0.3094 "win" is an early-model/test-
  period accident, not dendrites. (Separately real: tcn's dendritic run never
  improved val at any point — best 0.3269@21 vs 0.3271@1.)
- `mpnn dendrites_*`: harness best_epoch=3 — before any dendrite existed —
  restored cleanly onto PAI's rewound base structure; the 0.6976 "win"
  contains zero trained dendrites. (Its recorded param_count=713668 still
  reflects dendrite tensors physically present in the artifact — do not read
  dynamic8 mpnn/vae/tcn dendritic param or latency numbers as dendritic.)
- `gcn` and `actor_critic` dendritic numbers **stand** (benign quadrant,
  best-state restores succeeded on matching structures).

**Fix (2026-08-28, compat.py):** `set_initial_history_after_switches: 8`
(matching `set_history_lookback: 8`) in `_configure_dynamic_pai_schedule` —
PAI then seeds the running average from the first real score and warms up
with a cumulative mean over the first 8 epochs after every cycle switch,
before transitioning to the EMA. The bounded/fixed-switch path needs no
change (it leaves `history_lookback` at 1, where running = raw).

A first fix attempt — `set_running_average_pb: False` — was **wrong and is
reverted**: a fresh vae_mnist verification run under it reproduced the
broken result bit-for-bit (test −94.15, best val −92.71@50), and its switch
log still showed the EMA warm-up value as global best
(`global_best=-22.2394`). That flag does not gate the validation running
average. Lesson recorded because the "fix" initially looked verified from
gcn's raw-looking switch-log values, which a late-stage EMA also produces.

The real fix was verified three ways: mechanically (upstream
`update_running_accuracy` source: the first-score seeding branch runs iff
`epochs_since_cycle_switch < initial_history_after_switches`); empirically
in a pre-launch verification re-run (vae_mnist dendrites_fp32 tested ELBO
−91.55, up from the broken run's −94.15 and beating base's −92.49); and
again inside dynamic9's own production numbers (mpnn raw 0.876 → running
0.839 at epoch 2; old bug would have given ≈0.11). Predicted side-benefit
confirmed: lenet5 entered its first-ever dendrite phase under dynamic9 (4
switch events, first dendrite by epoch 9) — the rising-EMA switch blocker
no longer manufactures a monotone rise from zero. Full record:
`information/DYNAMIC9_RUN_2026-08-28.md`.

**Reading rule:** for any stored run, check
`results/PAI/<model>_<cond>/<...>best_arch_scores.csv` — if it lists only the
base architecture at an epoch-0/1 score while the harness history shows later
improvement, that condition's final model is a warm-up-era restore and its
test metric measures the wrong network.

---

## 11. A late-inserted dendrite trains at a learning rate at or near zero (FOUND + FIXED 2026-08-30)

### The observation

A short `resnet18_cifar10 dendrites_fp32` run, reading `history.csv` together
with `continued_until_complete/history.csv`:

```
epoch  6  lr=0.1   train=0.8333  val=0.8127
epoch  7  lr=0.0   train=0.8527  val=0.8392
epoch  8  lr=0.0   train=0.8524  val=0.8410
...
epoch 19  lr=0.0   train=0.8524  val=0.8414
```

**13 of 19 epochs at `lr` exactly `0.0`** — about two-thirds of a 1,972-second
run. Validation sits inside a 0.0036 band throughout and `pai_dendrite_phase`
is `False` for every one of them. Nothing was learning.

### Root cause

Three settings compose badly, and none is wrong on its own:

1. `resnet18_cifar10` uses `lr_schedule="cosine"` with `lr_min_factor`
   defaulting to `0.0` (`pipeline.py`, `ModelTrainingRecipe`).
2. `_scheduled_learning_rate` (`training.py`) clamps `progress` at `1.0`, so
   from `max_epochs` onward `lr = floor = base * 0.0 = 0.0` exactly. It is a
   pure function of the absolute epoch index — deliberately, so that a
   checkpoint resume and PAI's optimizer rebuild both land on the same curve.
3. `_dynamic_training_epoch_cap` returned `max_epochs + 16`.

So the 16 extra epochs that exist *solely* to let a dendrite be inserted and
adapt all ran at `lr = 0`. It is bad well before the cliff, too:

| epoch | ResNet-18 lr | % of base |
|---|---|---|
| 150 | 0.015364 | 15.4% |
| 180 | 0.002573 | 2.6% |
| 190 | 0.000647 | 0.65% |
| 200–216 | **0.000000** | **0%** |

SAINT's cosine floor is 2% of base (2e-6) and PointNet's step decay reaches
0.7^10 = 2.8% by epoch 200 and holds there. Neither is a rate at which a
freshly initialized module trains.

### The second-order problem

`DOING_HISTORY` fires on "no improvement for `n_epochs_to_switch` epochs with
`history_lookback` history". A cosine-to-zero tail produces exactly that
signature *regardless of what the network is doing* — the plateau it detects is
the anneal, not capacity saturation. Dendrites were therefore being inserted
for the wrong reason, at the one moment they could not train. This is a
coherent explanation for `no_retained_insertion` on both ResNet-18 and SAINT.

### Fix

- **`dendrite_lr_min_factor`** (new, per recipe). PAI's optimizer is now built
  with two parameter groups; the dendrite group's rate is floored at
  `learning_rate * dendrite_lr_min_factor`. The backbone group keeps the
  identical schedule its `base_fp32` control runs, so a dendritic gain cannot
  be an artifact of a warm restart the control never received — which is why
  this is a targeted floor rather than a whole-model schedule restart.

  Dendrite-side parameters are identified structurally, from module layout
  rather than `parameter_type` (which only separates `neuron` from `ignored`
  and does not single out newly inserted tensors). Read off a real perforated
  PointNet checkpoint: `<mod>.dendrites_to_top.<i>` and
  `<mod>.dendrite_module.layers.<i>.*` are dendrite-side;
  `<mod>.dendrite_module.parent_module.*` is the frozen shadow copy PAI carries
  for candidate scoring and stays on the backbone schedule.

  It **defaults to `0.0`, an exact no-op**. Only the three dynamic12 priority
  models opt in (at `0.1`), so no stored result for any other model changes
  meaning. `mobilenetv2_cifar10` has the same cosine-to-zero recipe and the
  same latent problem; it is deliberately left alone until it is re-run.

- **`_dynamic_training_epoch_cap`** is derived from the PAI schedule instead of
  a flat `+16`, which could not fit even one dendrite: a switch costs its
  candidate phase (bounded by `MAX_DENDRITE_PHASE_EPOCHS = 8`, not by the
  configured `p_epochs_to_switch = 10`) plus an adaptation window. For the
  priority models the cap is now `+28`.

The dynamic-schedule defaults moved into a single
`PAI_DYNAMIC_SCHEDULE_DEFAULTS` table in `compat.py`, read by both the PAI
configuration path and the cap, so the two cannot drift.

### What the fix is, and is not, measured to do

Verified mechanically. With a real `optimizer.step()` at epoch 228 of a
200-epoch ResNet-18 schedule -- inside the dynamic tail, where the backbone
rate is exactly 0.0 -- the dendrite and its dendrite-to-neuron mixing weight
move; the backbone and PAI's frozen shadow copy do not. Covered by
`test_optimizer_step_moves_dendrites_after_the_backbone_freezes`.

Verified inert when it should be inert. A SAINT A/B at a 12-epoch budget was
bit-identical between floor-on and floor-off, because the switch landed while
the schedule was still at 4e-5, above the 1e-5 floor. `max(scheduled, floor)`
never bound.

**Not** measured to change outcomes. A second SAINT A/B at a 6-epoch budget,
where the tail *is* pinned below the floor, came out inside the noise:

| | base | dendritic | audit |
|---|---|---|---|
| floor off (0.0) | 0.8456 | 0.8506 | `verified_retained` |
| floor on (0.1) | 0.8444 | 0.8501 | `verified_retained` |

The two dendritic arms differ by 0.0005, but the two *base* arms differ by
0.0012 despite identical configuration and seed, so run-to-run nondeterminism
is about twice the effect being read. Two things follow. First, dendrite
retention here is attributable to the epoch-cap change (+28 vs +16), **not** to
the learning-rate floor -- floor-off retains a dendrite too. Second, SAINT is a
weak test of this fix: its floor case is 2e-6 against 1e-5, both negligible
rates, whereas ResNet-18's is 0.0 against 0.01, which is categorical rather
than a factor of five.

**The ResNet-18 A/B (2026-08-30) is the stronger test, and it came out null
too.** Same 8-epoch dendrite budget, fast-switch, seed 0, floor 0.0 vs 0.01 --
a 10^4x ratio rather than SAINT's 5x:

| | dendrites_fp32 best | best_epoch |
|---|---|---|
| floor off (0.0) | 0.9036 | 22 |
| floor on (0.01) | 0.9039 | 30 |

`history.csv` makes this exact rather than approximate: per-epoch validation
accuracy is **bit-identical between the two runs for every epoch of the
p-phase** (epochs 20-27, the only window where the schedules actually differ --
lr=0.0 vs lr=0.01 on the dendrite group, confirmed from the raw history rows).
The dendrite itself trained at a genuinely different rate in each run --
`_apply_lr_schedule` is doing what it says -- and it changed nothing
observable. The 0.0003 final-accuracy gap traces to a `best_epoch` pick (30 vs
22) after the two runs' post-p-phase RNG streams decorrelated, not to the
floor.

So: across the two models this has been tested on, one weak test and one
categorical test, the learning-rate floor has not moved a single outcome. The
fix remains mechanically verified -- the dendrite parameter group really does
train at a different, non-collapsed rate -- but it should currently be read as
**inert-so-far**, not as a performance improvement. The most likely reading is
that an 8-epoch p-phase, even at a live rate, is too short for a
freshly-initialized dendrite's training rate to matter next to *whether* it
gets retained at all (the epoch-cap change). A longer p-phase budget, or a
model whose dendrite has more to learn than ResNet's single wide `.pre_fc`
projection, would be a better next test than assuming the current null
generalizes.

### Scope of invalidated data

None. No stored result used `dendrite_lr_min_factor`, and the default
reproduces the previous schedule exactly. What this *does* mean is that any
prior dendritic run whose switch landed in an annealed tail was measuring a
dendrite that could not train — the arm is not wrong, but it is not evidence
against dendrites either.

**Reading rule:** the run log prints
`[pai-lr] dendrite parameter group: N tensors / M parameters held at a floor of
F` on the epoch the group first exists. If that line never appears, no dendrite
was retained and the arm carries no dendritic signal.
