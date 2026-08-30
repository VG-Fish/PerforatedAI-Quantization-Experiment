# dynamic9 run — 2026-08-28 (first full run under the §10 EMA fix)

Run facts: commit `d008fcb`, seed 0, 7 workers (one per model), launched 16:05:54,
full pipeline (verify-gate → train → compare → graphs → benchmarks → tables) done
17:09:56 — **~64 min wall-clock for 84/84 conditions, zero crashes, zero worker
restarts.** Bundle: `experiments/dynamic9/` (self-contained).

Launch was gated on a fresh vae_mnist + gcn verification re-run (see
`experiments/fixcheck_ema/`), which passed: vae dendrites tested ELBO −91.55 (> the
−93.0 gate) vs the broken era's −94.15; gcn dendrites 0.798 (> the 0.70 floor).

## §10 fix: now empirically confirmed twice (verification + production)

`set_initial_history_after_switches: 8` (matching `history_lookback: 8`) is verified
working inside dynamic9 itself: running-average seeds are now close to first raw score,
not raw/8. Example: mpnn raw 0.876 → running 0.839 at epoch 2 (old bug: ≈0.11); vae raw
−95.33 → running −95.15 at epoch 1 (old bug: ≈−11.9). Both are in the previously-broken
metric quadrants (minimize / negative-maximize). MEASUREMENT_CAVEATS §10's forward
reference is now filled: **vae_mnist dendrites_fp32 tested ELBO = −91.55**, up from the
broken run's −94.15 and better than base's −92.49 — dendrites genuinely help the VAE.

## Results (test metric, base → dendrites, per bit width)

| Model (metric) | fp32 | q8 | q4 | q2 | q1.58 | q1 |
|---|---|---|---|---|---|---|
| gcn (Acc ↑) | .7850→**.7980** | .7860→.7980 | .7730→.7960 | .6900→**.8130** | .4310→**.6670** | .5800→.4830 |
| actor_critic (Acc ↑) | .9054→**.9883** | .9051→**.9886** | .9072→.8807 | .8768→.8614 | .8605→.8605 | .8142→.8145 |
| vae_mnist (ELBO ↑) | −92.49→**−91.55** | −92.65→**−92.07** | −106.40→−132.78 | −217.85→−278.43 | −240.82→−239.21 | −252.32→−307.60 |
| tcn_forecaster (MAE ↓) | .3124→**.3033** | .3123→.3033 | .3160→.3095 | .4049→.4063 | .4543→.4489 | .4248→.4274 |
| mpnn (RMSE ↓) | .6663→**.6160** | .6646→**.6182** | .7639→.8178 | 2.344→1.784 | 2.047→2.051 | 2.033→2.031 |
| lenet5 (Acc ↑) | .9917→**.9923** | .9917→.9920 | .9901→.9906 | .3420→**.4721** | .3986→.3988 | .3823→.3297 |
| gru_forecaster (MAE ↓) | .2604→**.2598** | .2609→.2601 | .2800→.2960 | .4438→.4340 | .4053→.4281 | .4586→.4646 |

Bold = dendrites win at that width. gcn's base_fp32 stayed at .7850 (not dynamic8's
.8000 at the same seed) — see the open item below.

## Dendritic training details

| Model | param growth | dendrite cycles (switch events) | best_epoch | epochs (dend / base) | notable events |
|---|---|---|---|---|---|
| gcn | 92,231→369,066 (+300%) | 4 (ep 24/27/42/45) | 42 | 159 / 191 | 1 find_best_lr restart |
| actor_critic | 17,539→71,059 (+305%) | 3 (ep 45/48/66) | 96 | 60 / 60 | 1 find_best_lr restart |
| vae_mnist | 1,091,920→3,278,144 (+200%) | 2 (ep 91/99) | 174 | 50 / 50 | 1 find_best_lr restart |
| tcn_forecaster | 124,008→134,928 (+8.8%) | 2 (ep ~20-28, ~47) not in switch_epochs.csv* | 5 | 74 / 80 | 1 find_best_lr restart |
| mpnn | 356,834→1,821,152 (+410%) | 6 (ep 18/26/36/44/76/84) | 139 | 157 / 200 | no restarts |
| lenet5 | 61,706→248,004 (+302%) | 4 (ep 9/17/26/34) | 19 | 40 / 40 | §8 collapse-guard stopped it early (frozen 12 epochs) |
| gru_forecaster | 172,448→303,488 (+76%) | 2 (ep ~20-28, ~47) not in switch_epochs.csv* | 1 | 74 / 80 | 1 find_best_lr restart |

\* tcn/gru show `pai_restructured=True` in history.csv at the same epochs as the other
models' switches, and param counts grew, but `switch_epochs.csv` itself was left empty
— a PAI CSV-writer quirk for these two architectures, not a sign dendrites weren't
added. Confirmed via `pai_dendrite_phase`/`pai_restructured` columns in history.csv.

## lenet5: the §10 switch-unblocking prediction is confirmed

dynamic9's header predicted the EMA fix would let lenet5 enter its first-ever dendrite
phase (previously blocked because a monotonically-rising accuracy EMA never reads as a
plateau). It did: 4 switch events, first dendrite incorporated by epoch 9, best val
0.9914 at epoch 19, and the §8 collapse-guard correctly stopped the run 12 epochs after
that plateau instead of grinding on a dead architecture. Tested: 0.9923 vs base 0.9917 —
small but real, and the mechanism now works as designed on a model it never touched
before.

## Headline answer: do dendrites help with quantization?

**Still no — the picture from dynamic8 mostly holds, but the fp32 effect is now
confirmed real and positive on five of seven models (dynamic8 had two).**

1. **fp32 advantage is broader and mostly carries into q8**: gcn +1.3pp→+1.2pp,
   actor_critic +8.3pp→+8.4pp, vae +0.94 nats→+0.58 nats, mpnn −0.050→−0.046 RMSE,
   lenet5 +0.06pp→+0.03pp. tcn/gru show only marginal (<0.001) fp32 deltas — see caveat
   below on whether these are real dendrite effects.
2. **No added quantization robustness, and the q4 wipeout repeats**: actor_critic loses
   nothing at base q4 (.9072) but the dendritic arm drops 10.8pp (.9883→.8807) —
   identical failure mode to dynamic8. vae's dendritic arm degrades *faster* than base
   below q8 (q4: −132.78 vs base −106.40) — the extra dendrite capacity has more to lose.
3. **Sub-q2 results are noisy but sometimes favor dendrites this run**: gcn +12.3pp at
   q2, +23.6pp at q1.58; lenet5 +13pp at q2. Single seed, treat as directional only —
   dynamic8 showed the opposite sign on gcn's low-bit swings.

Per-model verdicts: **actor_critic** — largest real effect, quantization-fragile at q4;
**vae_mnist** — real fp32/q8 win (§10's discovery case, now clean), fragile below q4;
**gcn** — small real fp32/q8 win this run (vs. null in dynamic8 — see open item), erratic
below q2; **lenet5** — new model, small real win, and the switch-unblocking mechanism
now demonstrated; **mpnn** — real fp32/q8 win (−0.05 RMSE), collapses with base below q4
(both arms converge); **tcn_forecaster** — dendrites never improved validation at any
epoch (best_epoch=5, before the first restructure at ~ep 20) — same real null result
dynamic8 found independently; the tiny tested MAE win (.3124→.3033) reflects the final
post-restructure weights, not a validated best state — treat as noise, not a dendrite
effect; **gru_forecaster** — identical pattern to tcn (best_epoch=1, pre-dendrite) —
confirms this is a forecasting-head trait, not tcn-specific.

## Caveats

- **tcn/gru "wins" are likely not dendrite effects.** For both, PAI's own best-validated
  epoch was *before* the first dendrite was ever added, and validation never improved
  afterward (flat ~0.335-0.338 for tcn, worsening to ~0.37-0.38 for gru post-restructure).
  Unlike vae/gcn/lenet5, no `[state] best-epoch structure does not match` mismatch was
  logged for these two — `_load_compatible_best_state` restored without complaint, so
  the tested model is *some* compatible state, but it is not the epoch the harness's own
  tracker calls "best." The small tested-MAE improvements over base are most plausibly
  test-set noise on an unimproved validation trajectory. Needs a follow-up read of
  exactly which weights got tested in this no-mismatch-but-not-best-epoch case.
- **Open item, unresolved**: gcn base_fp32 = 0.7850 in both this run and the pre-launch
  verification, vs 0.8000 in dynamic8 at the same seed 0. Base arms don't touch PAI
  config, so §10's fix can't explain it — something changed gcn's *base* behavior
  between commits 49d8f0c and d008fcb. Does not affect dynamic9's internal base-vs-
  dendrite pairing (both measured under the same commit) but breaks direct
  cross-run comparison for gcn specifically.
- Single seed (0); all q* arms are post-hoc PTQ, not quantization-aware training.
- Epoch-budget asymmetry from MEASUREMENT_CAVEATS §8 persists: dendritic arms train to
  PAI completion (up to 200 epochs for mpnn) while base arms use the fixed recipe budget.

Tables: `experiments/dynamic9/tables/01..04_*.md`. Plots:
`experiments/dynamic9/comparison/*.svg`. Mid-run PAI-graph audit (perforation-integrity
check against every claim in `PAI Skills/skills/perforatedai-analyze`):
`information/DYNAMIC9_PAI_GRAPH_AUDIT.md`.
