# Diagnosis: an inapplicable control condition aborts the entire sweep

**Date:** 2026-09-02
**Found by:** static reading plus a direct reproduction, *before* it fired.
**Status:** fixed the same day; `./scripts/ci.sh` green apart from the two
pre-existing failures.

This is the same shape of defect as `04_DIAGNOSIS_pai_final_artifact.md`: an
outcome the design already has a name for is raised as a fatal error from
inside one condition, and takes the whole multi-hour run down with it. It was
found by asking "what happens to `resnet18_hf_perforated_cifar100` when
`base_fp32` finishes?", six hours before the answer would have arrived on its
own.

---

## A. The default sweep is 24 conditions, and 12 of them are controls

`BenchmarkRunner._expand_condition_keys(None)` returns every entry of
`CONDITION_SPECS` — **24**, not the 12 the `dqb run` help text still describes.
Twelve of those are the two controls added later:

| control family | `source_key` | what it needs |
| --- | --- | --- |
| `base_more_training_*` | `dendrites_fp32` | the dendritic arm's `capacity_control_fork.pt` and its post-fork epoch count |
| `capacity_dense_*` | `dendrites_fp32` | the same fork **plus** an exactly reproducible retained topology |

Both go through `_prepare_control_model` (`pipeline.py:1158-1207`), which
raises `UnsupportedTopology` in four places, and `_control_post_fork_epochs`
adds a fifth (`pipeline.py:2255`).

**No stored run in `experiment_results/` contains a single control-condition
directory.** Verified with
`find experiment_results -maxdepth 3 -type d \( -name 'capacity_dense_*' -o -name 'base_more_training_*' \)`
— empty. Every shipped result predates these conditions being in the default
set, so the path they take through a real sweep had never been exercised.

## B. Three ways a control is legitimately impossible

### 1. A pre-perforated checkpoint has no dendritic arm to match

`resnet18_hf_perforated_cifar100` and `resnet18_hf_perforated_cifar10` are in
`PRE_PERFORATED_MODEL_KEYS`, so `condition_supported_by_model` removes all six
`dendrites_*` conditions — the run log says so at startup:

```
[conditions] resnet18_hf_perforated_cifar100 already contains published
dendrites; skipping non-distinct conditions: dendrites_fp32, dendrites_q8, …
```

It does **not** remove the twelve controls, whose declared source is exactly
the `dendrites_fp32` that was just removed. Confirmed:

```
>>> [c.key for c in CONDITION_SPECS if condition_supported_by_model(
...     "resnet18_hf_perforated_cifar100", c.key)]
18 keys — the six base_*, the six base_more_training_*, the six capacity_dense_*
```

So `base_more_training_fp32` is the 7th condition attempted, immediately after
the six `base_*` arms, and `_prepare_control_model` raises
`UnsupportedTopology("capacity controls require dendrites_fp32")`.

### 2. `capacity_dense_*` only supports **Linear** retained branches

`capacity_control.py:104-110` rejects a branch whose weight is not 2-D, and
says why:

> The first protocol is ResNet's `.pre_fc` Linear. Conv branches need PAI's
> spatial mixer semantics, which are not inferred safely from a state dict, so
> reject them rather than approximate them.

Reproduced directly:

```
conv branch  ((32,1,3,3) weight, (32,) mixer)
  -> UnsupportedTopology: .conv1: only Linear branches are currently supported
linear branch ((512,512) weight, (512,) mixer)
  -> kind='linear'
```

Which of the five new models can ever have this control:

| model | perforated modules | `capacity_dense_*` |
| --- | --- | --- |
| `mnist_pai` | none named → type-selected Conv2d **and** Linear | **impossible** |
| `resnet18_kd_cifar100` | `.pre_fc` (Linear) | possible |
| `unet_carvana` | 18 `.block1`/`.block2` conv+BN pairs, `.outc` | **impossible** |
| `unet_supervisely` | 17 `InvertedResidual`s, 4 decoders, 3 convs | **impossible** |
| `resnet18_hf_perforated_cifar100` | n/a (pre-perforated) | **impossible** (case 1) |

This is not a bug in the extractor. It is the extractor's documented contract:
"Other layouts raise `UnsupportedTopology`; **callers must record that status
rather than widen a model as a substitute control**" (`capacity_control.py:1-11`).
The caller was the part that did not hold up its end.

### 3. `dendrites_q*` after a `no_retained_insertion` FP32 arm

`_require_verified_dendritic_pqat_source` (`pipeline.py:1904-1928`) refuses a
PQAT descendant whose FP32 source did not earn `verified_retained`. Correct,
and `04_DIAGNOSIS`'s section on it is right that it must not be weakened — but
it was raising a bare `RuntimeError` out of the condition loop, so a dendritic
arm that honestly reports "no dendrite was retained" killed the sweep instead
of costing it five conditions.

## C. What actually happened at the raise

Nothing caught it. Traced frame by frame:

`_prepare_control_model` → `_run_condition` → `_train_pending_condition` →
`_process_one_model_spec` → `run()` → `cli._handle_run`. `grep -n "except "
src/dendritic_benchmark/pipeline.py` shows every handler in the file is a
narrow `OSError` / `json.JSONDecodeError` / `TypeError` around a file read; the
only broad one is `except Exception` in `_write_final_reports`
(`pipeline.py:2874`), which is *after* training and whose comment states the
principle this diagnosis is applying:

> Non-fatal on purpose: training results are already on disk, and a failed
> report build must not make the run look like it lost them.

Concrete cost, had it been left alone:

| run | dies at | after | loses |
| --- | --- | --- | --- |
| `resnet18_hf_perforated_cifar100` | `base_more_training_fp32` | ~6.5 h | manifest + reports; 12 conditions never reported as skipped |
| `mnist_pai` | `capacity_dense_fp32` | after all 18 other conditions | same |
| `unet_supervisely` | `capacity_dense_fp32` | ~15 h | same |
| `unet_carvana` | `capacity_dense_fp32` | — | same |

## D. The worse failure hiding behind it

Skipping a control is not enough on its own. `_prepare_condition_model`
(`pipeline.py:1107-1133`) loads a source checkpoint only

```python
if condition.source_key in saved_dirs:
    ...
if not condition.use_dendrites:
    return model          # ← the freshly *initialised* model
```

`capacity_dense_q8` has `use_dendrites=False`, so with `capacity_dense_fp32`
absent from `saved_dirs` it would fall straight through this branch, quantize a
**randomly initialised** network, and publish the number as a result — with a
valid artifact manifest and a `record.json` indistinguishable from a real one.

So the crash was, accidentally, the only thing preventing a fabricated result.
Any fix that merely stops the crash without also refusing the descendants makes
the repository *less* trustworthy, not more.

## E. The fix

`src/dendritic_benchmark/pipeline.py`:

| Change | Why |
| --- | --- |
| New `ConditionPrerequisiteUnmet(RuntimeError)` | Lets the sweep loop tell "this arm has no admissible source" from "training failed", and stop only for the second. Subclasses `RuntimeError` so `test_dendritic_pqat_requires_a_verified_fp32_source`'s `assertRaisesRegex(RuntimeError, …)` is unaffected. |
| `_require_verified_dendritic_pqat_source` raises it instead of `RuntimeError` | Same message, same strictness, recognisable type. |
| Sweep loop: skip any pending condition whose `source_key` is not in `saved_dirs` | Closes section D. This is the load-bearing half of the fix. |
| Sweep loop: `except (UnsupportedTopology, ConditionPrerequisiteUnmet)` around `_train_pending_condition` | Both types are raised only while *preparing* a condition, never from inside training, so nothing partial has been written. Deliberately narrow — a genuine training failure still stops the run. |
| New `_record_unavailable_conditions` → `<results>/<model>/unavailable_conditions.json` | A skipped condition leaves no `record.json`; without this file, "impossible for this architecture" and "nobody ran it" look identical in the result tree. |

Reason strings are chained one level, not nested: `capacity_dense_q1` reports
`source capacity_dense_fp32 unavailable: dendrites_fp32: only Linear branches
are currently supported`, not three copies of the same sentence.

## F. Verification

`tests/test_dynamic_pai_followup.py::UnavailableConditionsAreRecordedNotFatalTests`
drives the real `_process_one_model_spec` over all 24 conditions with
`_train_pending_condition` stubbed to raise from one chosen condition:

* refusing `capacity_dense_fp32` → the six `base_*`, six `dendrites_*` and six
  `base_more_training_*` still train; all six `capacity_dense_*` are recorded
  as unavailable and **none** of the five quantized children runs.
* refusing `dendrites_fp32` → only the six `base_*` arms train; the other 18
  are recorded.
* `_require_verified_dendritic_pqat_source` raises `ConditionPrerequisiteUnmet`.

```
$ .venv/bin/ty check
All checks passed!
$ .venv/bin/pytest
2 failed, 148 passed, 1558 subtests passed        # the two pre-existing failures
$ .venv/bin/dqb docs --check
[docs] information/CURRENT_GUIDE.md is current.
```

## G. Consequence for the reported results

`capacity_dense_*` is **permanently unavailable** for `mnist_pai`,
`unet_carvana`, `unet_supervisely` and `resnet18_hf_perforated_cifar100`, and
`base_more_training_*` is permanently unavailable for the last of those. That
is a property of the control's protocol, not of this port, and widening the
extractor to conv branches is explicitly refused upstream of here. The
topology-matched-dense control is therefore available for exactly one of the
five new models, `resnet18_kd_cifar100`. `base_more_training_*` remains
available for the other four dendritic models, and is the control that answers
the "is it just more training?" question — which is the one D9 in
`02_OPEN_DECISIONS.md` cares about.

## H. Effect on the runs already in flight

The three `dqb run` processes launched at ~01:38 hold the pre-fix
`pipeline.py` in memory, so they will still abort at their control condition.
They were deliberately **not** restarted: every completed condition has already
written its `record.json`, and `_condition_record_usable` keys staleness off
`model_revision` / `dataset_revision` / recipe / quantization revisions — none
of which this fix touches — so relaunching the same command afterwards resumes
from disk and only picks up what is left. Letting them run costs one traceback;
restarting them would have cost the hours already spent.
