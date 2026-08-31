# dynamic9 mid-run PAI graph audit — 2026-08-28 (~16:50, run still in progress)

<!-- status-banner -->
> **Status: superseded (2026-08-30).** Its finding that dendrites are inserted and retained still stands; its verdict that perforation is therefore *working* does not — it never tested the effect against noise. See [DENDRITE_EFFECT_AUDIT_2026-08-30.md](DENDRITE_EFFECT_AUDIT_2026-08-30.md).

Prompted by the user's concern that "the models are not being properly perforated."
Method: the seven skills in `PAI Skills/skills/` (chiefly `perforatedai-analyze`) applied
to `experiments/dynamic9/results/PAI/*` — switch_epochs, param_counts, best_arch_scores,
Best PBScores, Scores.csv, and the rendered PNGs.

## Verdict: perforation is working correctly on every model far enough along to judge

The skill's criteria for "properly perforated," applied per model:

| Criterion (from perforatedai-analyze) | gcn (done) | actor_critic (~ep 78) | vae/mpnn/lenet5 (early) |
|---|---|---|---|
| Dendrites actually added (switch_epochs non-empty) | ✅ ep 24/27, 42/45 | ✅ ep 45/48, 66 | n/a — still in first neuron phase |
| Param growth on restructure | ✅ 92k→184k→277k | ✅ 17.5k→34.9k | n/a |
| PBScore correlation > 0.02 ("good placement") | ✅ .conv1 0.067–0.077, .conv2 0.050–0.069 | ⚠️ .backbone.2 0.022–0.027 good; .policy 0.018, .backbone.0 0.014 marginal (both above the <0.01 "wasted params" bar) | n/a |
| best_arch_scores improves with dendrites | ✅ 0.756 → 0.770 (+1.4pp), then 2nd dendrite 0.748 → correctly discarded, 1-dendrite arch restored | ✅ 0.9918 → 0.9925 and climbing | n/a |
| Running average sane (post-§10-fix) | ✅ EMA tracks raw | ✅ EMA tracks raw | ✅ **key evidence below** |

## The §10 fix verified inside dynamic9 itself

First running-average values now ≈ first raw scores, not raw/8:
- mpnn (minimize): raw 0.876 → running 0.839 at epoch 2 (bug would have given ≈0.11)
- vae (maximize, negative): raw −95.33 → running −95.15 (bug would have given ≈−11.9)

Both are metrics in the previously-corrupted quadrants. The zero-seeded warm-up is gone.

## Per-model notes

- **gcn — complete, healthy.** Two dendrites explored, one kept (second failed the
  improvement threshold → completion + restore of the 1-dendrite arch). The 4
  `noImprove_lr_*` archives are find_best_lr LR-sweep restarts at cycle starts, NOT the
  skill's "no dendrites were ever added" red flag (that applies only when switch_epochs
  is empty — it isn't). Skill-recommended tuning for future runs: max_dendrites ≈ 2 for
  gcn. Borderline: the ep-24 switch came while raw val was still nudging up
  (0.744→0.756 over ep 21–24) — mildly premature by the skill's standard, but the
  8-epoch history window judged plateau and the post-switch gain was real.
  Tested: dendrites_fp32 0.7980 vs base 0.7850; dendritic quantization inheritance is
  strong (q2 0.8130, q1_58 0.6670 vs base 0.4310; q1 flips negative 0.4830 vs 0.5800).
- **actor_critic — mid-run, healthy.** Dendrite 1 incorporated (ep 45/48), second cycle
  underway (ep 66). Optimization note for future runs: if .backbone.0 stays ~0.014,
  consider excluding it (`append_module_ids_to_track([".backbone.0"])`) per the skill's
  module-selection guidance.
- **lenet5 — no switch yet (~ep 19), as expected.** The red line at ep 9 in its PNG is
  the end of the initial LR sweep (score-track restart), not a dendrite switch — no
  beforeSwitch_* files exist. Raw val still improving (best 0.9914@18), so the
  10-epochs-since-improvement trigger correctly hasn't fired. The §10 switch-unblocking
  prediction remains open until val genuinely plateaus.
- **vae, mpnn — just started (ep ~6 / ~3).** Empty switch_epochs is expected here.
- **tcn, gru — dendritic arms not yet started** (streams still on base conditions).

## Open item carried forward

gcn base_fp32 = 0.7850 in both fixcheck and dynamic9 vs 0.8000 in dynamic8 (same seed).
Base arms don't touch the PAI config, so the §10 fix can't explain it; something between
dynamic8's commit 49d8f0c and b9452c7 changed base gcn behavior. Investigate after the
run; affects cross-run comparability, not dynamic9's internal base-vs-dendrites pairing.

## PNG reading caution (standing)

PAI PNGs are redrawn live and have previously plotted fewer epochs than trained
(dendritic-graphs best_epoch mismatch). CSVs next to each PNG are ground truth.
