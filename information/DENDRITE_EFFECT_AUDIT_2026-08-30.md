# Dendrite effect audit: does perforation beat noise and more training?

**Date:** 2026-08-30
**Scope:** every `dendrites_fp32` run under `experiments/dynamic12/`
**Question:** not "were dendrites added?" but "did they *help*, by more than this run's own
noise and more than simply training longer?"

This supersedes the verdict in `DYNAMIC9_PAI_GRAPH_AUDIT.md` ("perforation is working
correctly on every model far enough along to judge"). That audit was correct about what it
tested — dendrites *are* being inserted and retained. It did not test whether the resulting
score change is distinguishable from noise. This document does.

**Headline: of nine dynamic12 dendritic runs, one (PointNet) shows a gain that clears its
own run's noise floor with margin. One (LeNet-5) is marginal. The rest are inside noise,
zero, or invalid. Nothing has been replicated across seeds.**

---

## 1. Method

### 1.1 Why PAI's own CSVs, not the summary tables

`comparison/summary.csv` compares `dendrites_fp32` against `base_fp32` — two separate runs.
That difference contains the dendrite effect *plus* seed noise *plus* any epoch-budget
difference. Since dendritic arms train until `training_complete` (by design), the budgets
differ, and the comparison cannot isolate the dendrite.

PAI's `<save>_best_arch_scores.csv` avoids all of this. Row 0 is the best validation score
the network reached **with zero dendrites, inside the same run** — same seed, same data
order, same initialization. Later rows are the best per dendritic architecture. The
difference is therefore a **within-run, same-seed** measurement of the dendrite alone.

### 1.2 The correct null

Both compared numbers are *maxima over windows of a noisy series*. The max of `m` draws
from pure noise rises with `m` even when nothing improves, so comparing a max to a max and
calling the difference "improvement" is biased upward. The null is the expected best of `m`
noise draws:

```
sigma      = sd of successive validation differences over the 12 converged
             epochs immediately before the first switch, divided by sqrt(2)
noise_floor = sigma * sqrt(2 * ln m)
m           = number of validation epochs after the first switch
```

A gain only counts if it exceeds `noise_floor`.

### 1.3 Direction handling

MAE / RMSE models are `minimize`; Accuracy / ELBO models are `maximize`. All gains below
are **direction-corrected**: positive always means "the dendrite helped." Read directly,
PAI's column header (`Max Valid Scores`) is a fixed label and is misleading for
minimize-metrics — MPNN's 0.75761 -> 0.73112 is an *improvement* in RMSE, not a regression.

### 1.4 Files used (so every number here is reproducible)

| file | columns used |
|---|---|
| `results/PAI/<save>/<save>_best_arch_scores.csv` | `Param Counts`, `Max Valid Scores` |
| `results/PAI/<save>/<save>Scores.csv` | `Epochs`, `Validation Scores` |
| `results/PAI/<save>/<save>switch_epochs.csv` | `Switch Number`, `Switch Epoch` |
| `results/PAI/<save>/<save>param_counts.csv` | `Switch Number`, `Param Count` |
| `results/PAI/<save>/<save>Best PBScores.csv` | all `Best ever ...` columns |
| `results/PAI/<save>/<save>learning_rate.csv` | `learning_rate` |
| `results/<model>/<cond>/record.json` | `metric_value` (test), `best_metric_value` (val), `metric_direction` |
| `results/PAI/<save>_PAI_config.json` | `switch_mode`, `fixed_switch_num`, `n_epochs_to_switch`, `p_epochs_to_switch`, `max_dendrites` |

`switch_mode`: `1 = DOING_HISTORY` (PAI decides on a detected plateau),
`2 = DOING_FIXED_SWITCH` (forced interval).

---

## 2. Result: dendrite gain vs. the run's own noise floor

| run | mode | gain | noise floor | ratio | verdict |
|---|---|---|---|---|---|
| `priority_replications/seed_0` PointNet | HISTORY | **+0.03557** | 0.02194 | **1.62x** | clears floor |
| `combined/seed_0` LeNet-5 | HISTORY | +0.00280 | 0.00208 | 1.35x | marginal |
| `combined/seed_0` GCN | HISTORY | +0.00800 | 0.00869 | 0.92x | at/below floor |
| `combined/seed_0` actor-critic | HISTORY | +0.00373 | 0.02036 | 0.18x | well below floor |
| `combined/seed_0` MPNN | HISTORY | +0.02649 | 0.19071 | 0.14x | well below floor |
| `combined/seed_0` VAE | FIXED (20) | +0.08781 | 0.48347 | 0.18x | well below floor |
| `saint_head_fixed100/seed_0` SAINT | FIXED (100) | +0.00000 | 0.00508 | — | no gain |
| `combined/seed_0` GRU | FIXED (8) | +0.00064 | n/a | — | **invalid, see 2.1** |
| `combined/seed_0` TCN | FIXED (6) | none retained | n/a | — | **stale folder, see 2.2** |

Both runs that clear or approach their floor are in HISTORY mode. No fixed-interval model
produces a usable positive result.

### 2.1 GRU is invalid, not negative

GRU's first switch fired at **epoch 1**, leaving a single pre-switch epoch. `best_arch_scores`
row 0 is therefore an *untrained network*, not a converged baseline, and the within-run
comparison is meaningless. Its nominal +0.00064 validation gain is contradicted by test,
where the dendritic arm is 0.00042 **worse** (0.20876 vs 0.20834).

This is the fixed interval misfiring: configured 8, fired at 1.

### 2.2 TCN in `combined/seed_0` should not be used

That folder has 2 rows in `Scores.csv`, an empty `switch_epochs.csv`, and a PB correlation
of 0.0 — yet two `switch_N.pt` checkpoints sit in the same directory. This is the
never-cleared `results/PAI/<save_name>/` problem recorded in `CODE_REVIEW_2026-08-28.md`.
The `tcn_audited_*` runs (section 4) are the clean source for TCN.

---

## 3. The "more training" control

For **actor-critic**, a matched control window exists: over the `m` epochs immediately
before the first switch, plain continued training improved validation by **+0.05748 — 15x
the dendrite's +0.00373.**

Its PAI graph shows validation reaching 0.990 at **epoch 40, before any dendrite existed**.
The headline +9.2pp that `actor_critic` shows over `base_fp32` is therefore almost entirely
pre-dendrite training, not a dendrite effect.

For every other model **no matched control window exists**: the dendrite window is longer
than the entire pre-dendrite training period, so the run contains no comparable interval.
That is itself a finding — dendrites are being inserted so early, relative to total run
length, that the run cannot support an internal control.

---

## 4. TCN: the only 3-seed data in the corpus

Three perforation targets x three seeds. MAE, lower is better.

| target | seed | base val | dend val | val gain | base test | dend test | test gain |
|---|---|---|---|---|---|---|---|
| `.head.0` (default) | 0 | 0.33734 | 0.33734 | +0.00000 | 0.31613 | 0.30935 | +0.00678 *(no dendrite)* |
| `.head.0` | 1 | 0.33393 | 0.33393 | +0.00000 | 0.31497 | 0.32784 | **-0.01287** |
| `.head.0` | 2 | 0.33791 | 0.33791 | +0.00000 | 0.31097 | 0.33013 | **-0.01916** |
| `.head.0+.head.3` | 0 | 0.33734 | 0.33518 | +0.00215 | 0.31613 | 0.31352 | +0.00262 |
| `.head.0+.head.3` | 1 | 0.33393 | 0.33190 | +0.00203 | 0.31497 | 0.31406 | +0.00090 |
| `.head.0+.head.3` | 2 | 0.33791 | 0.33791 | +0.00000 | 0.31097 | 0.31507 | -0.00409 |
| `.head.3` (output) | 0 | 0.33566 | 0.33566 | +0.00000 | 0.30824 | 0.30777 | +0.00046 *(no dendrite)* |
| `.head.3` | 1 | 0.33393 | 0.33393 | +0.00000 | 0.31497 | 0.30678 | +0.00819 *(no dendrite)* |
| `.head.3` | 2 | 0.33791 | 0.33723 | +0.00069 | 0.31097 | 0.32452 | **-0.01355** |

| quantity | value |
|---|---|
| baseline validation spread across all 9 runs (**no dendrites involved**) | 0.00398 (sd 0.00174) |
| dendrite validation gain, the 6 runs that retained one | **+0.00081 +/- 0.00094** |
| dendrite **test** gain, same 6 runs | **-0.00769 +/- 0.00802** |

Three things follow:

1. **The mean dendrite effect is smaller than the baseline seed sd.** Changing only the
   seed moves validation more than adding a dendrite does.
2. **On held-out test the dendrite makes TCN worse**, by roughly 10x the size of the
   validation "gain."
3. **A run that retained no dendrite at all still showed a +0.00678 test improvement** from
   pure run-to-run variation — larger than every genuine dendrite validation gain in the
   entire corpus.

Rows 2 and 3 also show dendrites **retained at exactly 0.00000 validation gain**, which
then cost 1.3-1.9pp of test MAE. `improvement_threshold` is not rejecting a zero-gain
candidate.

---

## 5. Mechanism evidence

### 5.1 Per-run configuration and outcome

| run | mode | interval | n_ep | p_ep | maxD | switch epochs | param ladder | LR first->last | PB peak |
|---|---|---|---|---|---|---|---|---|---|
| actor-critic | HISTORY | — | 10 | 6 | 2 | 40, 47, 56, 63 | 10,083 -> 19,395 -> 28,707 | 3.0e-4 -> **1.5e-5** | 0.037 |
| GCN | HISTORY | — | 10 | 6 | 1 | 19, 26 | 69,175 -> 138,295 | 1.0e-2 -> 5.9e-3 | 0.119 |
| LeNet-5 | HISTORY | — | 10 | 2 | 1 | 9, 17 | 35,105 -> 68,568 | 1.0e-2 -> 2.4e-4 | 0.148 |
| MPNN | HISTORY | — | 10 | 2 | 3 | 18, 26, 48, 56, 82, 90 | 201,962 -> 405,869 | 1.0e-3 -> 2.0e-4 | **0.346** |
| PointNet | HISTORY | — | 10 | 10 | 1 | 9, 17 | 3,471,473 -> 4,128,369 | 1.0e-3 -> 2.4e-4 | 0.134 |
| VAE | FIXED | 20 | 20 | 6 | 1 | **8**, 16 | 770,000 -> 1,071,840 | 1.0e-3 -> 6.8e-4 | 0.207 |
| GRU | FIXED | 8 | 8 | 6 | 1 | **1**, 9 | 130,080 -> 137,040 | 3.0e-4 -> 7.7e-5 | 0.160 |
| TCN | FIXED | 6 | 6 | 2 | 1 | **none** | 79,272 (flat) | 1.0e-3 -> 1.0e-3 | 0.000 |
| SAINT | FIXED | 100 | 100 | 10 | 1 | **19**, 27 | 211,906 -> 216,324 | **2.0e-5 -> 1.0e-5** | 0.000 |

Each retained dendrite appears as **two** rows in `switch_epochs.csv` (candidate-in, then
incorporate/reject).

### 5.2 Fixed intervals are not producing the intervals they specify

| model | configured interval | observed first switch |
|---|---|---|
| GRU | 8 | **1** |
| VAE | 20 | **8** |
| SAINT | 100 | **19** |
| TCN | 6 | **never fired** (and 3 of 6 audited runs) |

HISTORY-mode models fired at plausible plateau points by comparison: actor-critic 40,
GCN 19, LeNet-5 9, MPNN 18/48/82, PointNet 9.

The rationale for the fixed-interval workaround is stated in
`_configure_interval_pai_schedule` (`src/dendritic_benchmark/compat.py:294-317`):

> HISTORY mode compares a running average (an EMA over `history_lookback` epochs) that
> starts at 0 and only ever climbs toward the current score. [...] `epoch_last_improved` is
> refreshed continuously and `n_epochs_to_switch` never counts down.

**That is the zero-seeded-EMA bug, and it was fixed on 2026-08-28** by setting
`initial_history_after_switches = 8` (`compat.py:406`, `MEASUREMENT_CAVEATS.md` section 10).
The comment documenting that fix sits about 90 lines below the docstring that still
justifies the workaround. The premise is obsolete.

### 5.3 Dendrites arrive after the learning rate has collapsed

`actor_critic`'s LR falls from 3.0e-4 to ~1.5e-5 at epoch 41 and stays flat. Both of its
dendrites (epochs 40-47 and 56-63) therefore trained at roughly **5% of the initial rate**.
SAINT is worse: its entire run sits at 2.0e-5 -> 1.0e-5, and its dendrite candidate reached
a PB correlation of **exactly 0.0** — completely inert.

`dendrite_lr_min_factor` (`training.py:3697`) exists precisely for this and defaults to
`0.0`. It is enabled (`0.1`) for only three models — PointNet, ResNet-18, SAINT
(`pipeline.py:1193`, `:1231`, `:1258`). **PointNet, the one model that clears its noise
floor, is one of them.** Suggestive, but not proof: the repo's own A/Bs found the floor
*inert* on ResNet-18 and SAINT (`MEASUREMENT_CAVEATS.md` section 11).

Note also that four of the five per-model justifications for fixed intervals
(`pipeline.py:90-109`) are really about LR, not plateau detection — the VAE candidate
"trained at the LR floor," the GRU search "began after the cosine had effectively bottomed
out," SAINT's late candidate "was never kept."

### 5.4 Candidates are cut off before their correlation converges

MPNN's `Best PBScores.csv` is still climbing (0.26 -> 0.35) when each p-phase ends. Its
phases run exactly 8 epochs — `MAX_DENDRITE_PHASE_EPOCHS = 8` (`training.py:75`) — meaning
they are terminated by the stall guard, not by convergence. MPNN's configured
`p_epochs_to_switch` is 2, so the phase length is set entirely by the cap.

### 5.5 PB correlation does not predict validation gain

| model | PB peak | gain vs noise floor |
|---|---|---|
| MPNN | 0.346 | 0.14x — inside noise |
| VAE | 0.207 | 0.18x — inside noise |
| GRU | 0.160 | invalid |
| LeNet-5 | 0.148 | 1.35x — marginal |
| PointNet | 0.134 | **1.62x — clears** |
| GCN | 0.119 | 0.92x — at floor |
| actor-critic | 0.037 | 0.18x — inside noise |
| SAINT | 0.000 | no gain |

The rank correlation is essentially nil. The `perforatedai-analyze` heuristic that
"> 0.02 = good placement" does not hold here: the two highest-correlation modules in the
corpus (MPNN 0.346, VAE 0.207) produce no measurable validation gain, while PointNet
succeeds at 0.134. PB correlation measures how well the candidate fits the residual
learning signal on *training* data; it does not measure generalization.

---

## 6. A reporting defect found during this audit

`results.py:24-29` defines:

```python
_NON_REPORTABLE_DENDRITE_STATUSES = {
    "no_retained_insertion", "inherited_no_retained_insertion",
    "unverified", "inherited_unverified",
}
```

`legacy_unchecked` is **absent**, so `_record_is_reportable` (`results.py:245`) returns
`True` for it. And `_legacy_dendrite_audit_status` (`results.py:197-227`) assigns
`legacy_unchecked` in exactly the case where it *cannot* confirm `raw_params == final_params`.

Net effect in `experiments/dynamic12/combined/seed_0/comparison/dendrite_audit.csv`:
`actor_critic`, `gcn`, `lenet5` and `mpnn` are all `legacy_unchecked` **and**
`reportable=True` — the four models whose recorded parameter counts disagree with PAI's own
architecture log. The root cause of that disagreement is the never-cleared
`results/PAI/<save_name>/` directory (`CODE_REVIEW_2026-08-28.md`), so the arch log can come
from a different run than the record.

Separately, all 40 quantized dendritic rows in that run are `stale_double_projection` and
already excluded — the quantization x dendrite question, which is the point of the
experiment, currently has no reportable data in `combined/seed_0`.

---

## 7. Limits of this audit

These constrain how far the conclusions can be pushed.

- **PointNet and LeNet-5 are single-seed.** Clearing a within-run noise floor is not the
  same as clearing seed noise. TCN is the only model where seed noise can be measured, and
  it wipes the effect out entirely. Neither result should be called confirmed until
  replicated.
- **The GRU row is invalid, not negative** (section 2.1), and TCN in `combined/seed_0` comes
  from a stale folder (section 2.2). Neither is evidence against dendrites; both are
  evidence that those runs cannot be read.
- **The matched-window "more training" control computed for actor-critic only.** For every
  other model no comparable interval exists inside the run, so section 3's conclusion is
  established for one model and merely plausible elsewhere.
- **`noise_floor` assumes roughly independent, roughly stationary validation noise** in the
  pre-switch window. Validation curves are autocorrelated, which makes the true floor
  somewhat *higher* than estimated — so this test is, if anything, generous to the dendrite.
- **This audit says nothing about quantization robustness**, the experiment's actual
  question. Every quantized dendritic arm in `combined/seed_0` is currently non-reportable.

## 8. What would settle it

1. **Multi-seed the two models that pass.** PointNet and LeNet-5 at 3 seeds, using the
   `tcn_audited_*` layout that already works.
2. **Run the capacity-matched dense controls that already exist and have never been run:**
   `experiments/dynamic12/tuning/tune_gru_capacity.py` (widths `[48, 51]`) and
   `tune_vae_capacity.py` (scales `[0.75, 1.0]`). Width 51 is approximately the
   137,040-param GRU dendritic artifact; scale 1.0 is approximately the 1,071,840-param VAE
   artifact. They answer "dendrite, or just more parameters?" directly. Both currently print
   to stdout and persist nothing.
3. **Let PAI choose its own switch points** — remove the fixed intervals, whose justifying
   bug is fixed and which are not honoring their configured intervals anyway (section 5.2).
4. **Add a noise-floor column to `dendrite_audit.csv`** so a flat result is visible in the
   standard report rather than requiring this analysis to be redone by hand.
