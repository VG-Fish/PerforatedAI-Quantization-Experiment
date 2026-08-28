# Measurement caveats: root causes and fixes

Three measurement caveats were recorded (without being fixed) during the `dynamic5`
run — see `experiments/dynamic5/reference/BENCHMARKS.md` §"The quantization kernels"
and `experiments/dynamic5/report.md` §6.6. This document traces each one to the
exact code responsible and lays out fix options. Line numbers are current as of
commit `675105a` (branch `dynamic5-baselines`).

Status at a glance:

| # | caveat | root cause found | fix applied |
|---|---|---|---|
| 1 | `q2` collapses to 3 knife-edge levels | yes — `compat.py` kernel math | **yes** — user chose "fix it properly", see §1 |
| 2 | `tcn_forecaster` `q1`/`q1_58` overflow to ~1e9–1e10 | yes — architecture-specific | **moot** — model is being dropped, see §2 |
| 3 | `dendrites_fp32` vs `dendrites_q*` param-count mismatch | yes — two independent, unsynchronized checkpoint systems | **yes** — see §3 |

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

## 2. `tcn_forecaster` overflows at `q1`/`q1_58` — moot, model is being dropped

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
