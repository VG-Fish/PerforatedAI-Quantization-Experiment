# dynamic8 run — 2026-08-28 (first post-§8-fix run)

Run facts: commit `49d8f0c`, seed 0, 5 workers (one per model), launched 14:24:56,
full pipeline (train → compare → graphs → benchmarks → tables) done 14:44:44 —
**~20 min wall-clock for 60/60 conditions, zero crashes, zero worker restarts.**
Bundle: `experiments/dynamic8/` (self-contained: results, comparison, benchmarks,
tables, logs).

## Fix verifications (both 2026-08-28 fixes observed working live)

- **§8 collapse-guard fix**: zero `[collapse]` lines in any stream. The forced
  phase switch (`max_dendrite_phase_epochs=8`) fired exactly 2× each on
  precisely the three previously-stuck models (tcn_forecaster, vae_mnist,
  mpnn) and 0× on gcn and actor_critic — the two whose schedules always
  self-terminated. The mechanism discriminates exactly as designed.
- **benchmark.py loader fix**: every `dendrites_*` latency benchmark now
  refuses with "does not match the plain model architecture … refusing to
  benchmark the wrong architecture" instead of silently measuring an
  unperforated skeleton. Base-condition latency rows in
  `experiments/dynamic8/benchmarks/` are valid; dendritic rows are
  (correctly) absent.
- **Seeding**: mpnn base_fp32 RMSE 0.7162 is bit-identical to the previous
  seed-0 run — exact reproducibility confirmed.

## Results (test metric, base → dendrites, per bit width)

| Model (metric) | fp32 | q8 | q4 | q2 | q1.58 | q1 |
|---|---|---|---|---|---|---|
| gcn (Acc ↑) | .8000→.8010 | .7990→.8010 | .8090→.8010 | .7940→.8090 | .4420→.5860 | .5720→.2550 |
| actor_critic (Acc ↑) | .9054→**.9898** | .9051→**.9870** | .9072→.8816 | .8768→.8614 | .8605→.8608 | .8142→.8142 |
| vae_mnist (ELBO ↑) | −92.16→−94.15 | −92.35→−94.41 | −108.5→−106.5 | −218.5→−218.1 | −254.5→−243.3 | −253.6→−249.6 |
| tcn_forecaster (MAE ↓) | .3094→**.3013** | .3094→.3014 | .3126→.3058 | .4196→.4131 | .4610→.4581 | .4430→.4347 |
| mpnn (RMSE ↓) | .7162→**.6976** | .7183→.6968 | .7674→.7334 | 2.009→2.044 | 2.052→2.053 | 2.048→2.049 |

## Dendritic training details

| Model | param growth | epochs (canonical + over-budget) | best_epoch | train s (dend vs base) |
|---|---|---|---|---|
| gcn | +300% (~3 dendrites) | 121 | 82 | 150 vs 20 |
| actor_critic | +305% (~3 dendrites) | 60 + 85 (§9 split) | 117 | 392 vs 48 |
| vae_mnist | +100% (1 dendrite) | 50 | 50 | 798 vs 297 |
| tcn_forecaster | +8.8% (head-only) | 50 | 21 | 315 vs 478 |
| mpnn | +100% (1 dendrite) | 50 | 3 | 497 vs 323 |

Notes: actor_critic is the only run that trained past its budget (145 epochs
total, history split per `_persist_over_budget_snapshot`). vae's best_epoch=50
is the final epoch — it may still have been improving. mpnn's best_epoch=3 with
a final RMSE better than base suggests its dendrite benefit came from an early
restructure; treat with care. tcn's dendritic arm was *faster* than base
(head-only perforation, and the dynamic schedule ended at the same 50 epochs).

## Headline answer: do dendrites help with quantization?

**Mostly no — dendrites improve some models, and that advantage survives q8
intact, but they do not confer robustness to quantization itself.**

1. **fp32 advantage carries into q8 essentially unchanged, everywhere**
   (actor_critic +8.4pp→+8.2pp; tcn and mpnn margins intact).
2. **No extra retention below q8, and one sharp counter-example**: at q4 the
   actor_critic base model loses nothing (.9054→.9072) while the dendritic
   model loses 10.8pp (.9898→.8816), flipping the comparison negative. By q1
   both arms produce the identical number (.8142) — binarization erases the
   dendrites completely. tcn's margin is flat across widths (the fp32 gain
   just rides along, no slope change); mpnn's holds at q4 then both arms
   collapse together at q2.
3. **Below q2 nothing is meaningful**: vae and mpnn collapse in both arms;
   gcn's sub-2-bit swings (+14.4pp at q1.58, −31.7pp at q1, base q1 beating
   base q1.58) are noise on a 1000-node test set, single seed.

Per-model verdicts: actor_critic — large real dendrite effect, quantization-
fragile; tcn_forecaster — small consistent effect (−0.008 MAE) at 8.8% param
cost, carries at all widths; mpnn — modest effect (−0.019 RMSE) up to q4;
gcn — null at this seed (+0.1pp, vs +1.9/+2.9/+2.2pp in earlier paired runs;
within noise); vae_mnist — dendrites *hurt* at fp32/q8 (−2 nats) despite
doubling params.

## Caveats

- Single seed (0). Error bars require SEED=1,2 replicate runs into separate
  results roots (~20 min each).
- All q* arms are **post-hoc** quantization of the trained checkpoint (PTQ
  retention). This says nothing about quantization-aware training; the PQAT
  shadow-weight fix (2026-08-11) still has no stored re-run.
- Dynamic dendritic arms train to PAI completion while base arms use the fixed
  recipe budget (epoch-budget asymmetry, MEASUREMENT_CAVEATS §8) — actor_critic
  dendrites got 145 epochs vs base's 60. gcn's dendritic arm got 121 vs base's
  own budget too. The +8.4pp is therefore dendrites *plus* extra epochs.

Tables: `experiments/dynamic8/tables/01..04_*.md`. Plots:
`experiments/dynamic8/comparison/*.svg`.

---

## VALIDITY REVISION (2026-08-28, later the same day — see MEASUREMENT_CAVEATS §10)

A deeper audit found PAI's zero-seeded running average (EMA) corrupts
best-model tracking for every metric that is not positive-maximize. This
**invalidates three of the five dendritic stories above**:

- **vae_mnist**: the tested "dendritic" model was PAI's epoch~1-era restore.
  The real trained dendritic model reached val −92.71 vs base val −92.85 —
  dendrites likely *helped*, not hurt. "Dendrites hurt the VAE" is withdrawn.
- **tcn_forecaster**: the tested model was a PAI-restored pre-dendrite early
  state; the consistent −0.008 MAE "win" is not attributable to dendrites.
  (Also real: its dendritic run never improved validation at any epoch.)
- **mpnn**: the tested model was the epoch-3 snapshot — before any dendrite
  existed. The −0.019 RMSE "win" contains zero trained dendrites.
- **gcn, actor_critic**: unaffected (benign metric quadrant); their numbers
  — including the actor_critic q4 wipeout — stand.

The headline "dendrites don't add quantization robustness" therefore now
rests only on gcn and actor_critic. Fixed in compat.py
(`set_initial_history_after_switches: 8` to match `history_lookback: 8`;
a first attempt with `set_running_average_pb: False` reproduced the broken
vae result bit-for-bit and was reverted — §10 has the full story); vae and
gcn verification re-runs were launched the same day (results recorded below
when complete). lenet5 and
gru_forecaster selected as the two additions for the next run — one tests the
fix's switch-unblocking prediction (maximize regime), one the repaired
minimize regime.
