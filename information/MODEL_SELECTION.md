# Model selection for the 7-model run

> **SUPERSEDED for the Dynamic12 priority sweep (2026-08-29).** That sweep runs
> `resnet18_cifar10`, `saint_adult`, and `pointnet_modelnet40` — two of which
> this document rules out. Criterion 1 excluded `pointnet_modelnet40` and
> `resnet18_cifar10` because an expired PerforatedAI token had blocked their
> low-bit dendritic conditions. **That block is gone**: the token in `.env`
> perforates all three cleanly, verified 2026-08-29 by calling
> `compat.perforate_model` on each. The criterion-1 exclusion is therefore void,
> and with it the reason those two were not candidates.
>
> Nothing else here is retracted — the cost table, the domain-diversity
> argument, and the `top10` signal-quality analysis all still read correctly for
> the 7-model run they were written for. See `experiments/dynamic12/README.md`
> for the priority sweep's own scope and `information/MODEL_REFERENCE.md` for
> the three models' perforation targets.


The user's plan: drop `tcn_forecaster` from the current 5 (`lenet5`, `gcn`,
`actor_critic`, `saint_adult`, `tcn_forecaster`) and add 3 new models from the
23 in `specs.py::MODEL_SPECS`, for 7 total. This documents why `textcnn`, `m5`,
and `mpnn` were chosen over the other 19 candidates.

**Kept:** `lenet5`, `gcn`, `actor_critic`, `saint_adult`
**Dropped:** `tcn_forecaster` (per the user's instruction — also see
`MEASUREMENT_CAVEATS.md` #2, its q1/q1_58 overflow, now moot)
**Added:** `textcnn`, `m5`, `mpnn`

## Criteria, in priority order

1. **No known unresolved defect that would make the result untrustworthy or
   incomplete.** Ruled out on this alone: `pointnet_modelnet40` and
   `resnet18_cifar10` (PAI license expiry blocked their low-bit dendrite
   conditions as of the last check — see the `pai-license-expired-blocks-lowbit-dendrites`
   memory; unresolved as of this write-up), `lstm_autoencoder` (regression
   flagged in `model-quality-followups-aug6`, not yet root-caused).
2. **Architectural / domain diversity from the 4 kept models.** The 4 kept
   already cover vision-CNN (`lenet5`), citation-graph GNN (`gcn`),
   behaviour-cloning RL (`actor_critic`), and tabular attention (`saint_adult`).
   Picking 3 more models in the *same* domains would answer the same question
   three more times rather than extending the benchmark's coverage.
3. **Tractable compute cost**, given a parallel `dqb run --jobs N` sweep on
   this machine. `pipeline.py`'s own per-model FP32-hour estimates
   (`_MODEL_COST_HOURS`, line ~1315) are the source for this:

   | model | est. FP32 hours | note |
   |---|---|---|
   | `resnet18_cifar10` | 8.6 | ruled out by criterion 1 anyway |
   | `mobilenetv2_cifar10` | 8.0 | too expensive |
   | `snn_nmnist` | 4.4 | too expensive for this pass |
   | `capsnet_mnist` | 3.8 | too expensive for this pass (see below) |
   | `pointnet_modelnet40` | 3.0 | ruled out by criterion 1 |
   | `gru_forecaster` | 2.5 | see criterion 4 — also disqualified |
   | `distilbert` | 2.0 | too expensive |
   | `m5` | 1.0 | **picked** |
   | `ppo_bipedalwalker` | 0.3 | domain overlap with `actor_critic` (RL) |
   | `saint_adult` | 0.2 | already kept |
   | `vae_mnist` | 0.2 | generative — see note below |
   | `dqn_lunarlander` | 0.2 | domain overlap with `actor_critic` (RL) |
   | `lstm_autoencoder` | 0.2 | ruled out by criterion 1 |
   | *(unlisted → 0.1h default; actuals below come from `results/top10`)* | | |
   | `textcnn` | ~0.7h dendritic total (1826s train + quant evals) | **picked** |
   | `tabnet` | ~0.35h | domain overlap with `saint_adult` (Adult Income) |
   | `mpnn` | ~0.8h | **picked** |
   | `lstm_forecaster` | ~1.2h (dendritic training alone was 3929s) | see criterion 4 |

4. **Signal quality, where a prior result exists.** `results/top10` (a prior
   `DOING_FIXED_SWITCH` run, post every fix in `REMAINING_FIXES.md`) already has
   records for 5 of the 18 remaining candidates: `lstm_forecaster`, `textcnn`,
   `tabnet`, `mpnn`, `gru_forecaster`. Their FP32 dendritic deltas:

   | model | base metric | dendrites metric | test delta | best_metric_value delta |
   |---|---|---|---|---|
   | `textcnn` | 0.9141 acc | 0.9162 acc | **+0.21pp** | +0.16pp (agrees) |
   | `tabnet` | 0.8458 acc | 0.8214 acc | −2.44pp | **+0.21pp (disagrees)** |
   | `mpnn` | 0.7273 RMSE | 0.7806 RMSE | worse (higher) | **better (0.7719→0.7321)** |
   | `gru_forecaster` | 0.2678 MAE | 0.2892 MAE | worse (higher) | worse (0.3356→0.3487) |
   | `lstm_forecaster` | 0.2944 MAE | 0.3357 MAE | worse (higher) | ~flat (0.1736→0.1754) |

   `textcnn` is the only one of these five whose test delta and validation
   delta *agree* — the signature (established in `MEASUREMENT_CAVEATS.md` #3)
   of a model whose dendrite structure never changed after its best epoch, so
   the old best-epoch/final-structure bug never had anything to corrupt.
   `tabnet`, `mpnn`, `gru_forecaster`, and `lstm_forecaster` all show the
   opposite pattern: validation says dendrites are flat-to-better, the
   (bugged) test metric says worse. That is exactly the corruption signature,
   not evidence that dendrites hurt these models — but it does mean their
   `top10` numbers cannot be used as-is to judge them.

## Why `mpnn` over `tabnet` / `gru_forecaster` / `lstm_forecaster`

All four showed the corruption signature, so signal quality doesn't
distinguish them; domain and cost do:

- `tabnet` is Adult Income — the same dataset `saint_adult` already covers.
  Adding it would test "does this specific tabular architecture respond to
  dendrites" rather than extend domain coverage.
- `gru_forecaster` (2.5h) and `lstm_forecaster` (~1.2h, dendritic training
  alone took 3929s in `top10`) are the two most expensive of the four, and
  time-series forecasting is a domain already represented by the model being
  *dropped* (`tcn_forecaster`) — keeping a forecaster in the 7 wasn't judged a
  priority once `tcn_forecaster` leaves.
- `mpnn` is graph-level regression over molecules (ESOL, RMSE) — a materially
  different task from `gcn`'s node classification (different metric direction,
  graph pooling instead of per-node output, edge-feature message passing, and
  a `DendriticGRUCell`-gated update rule per `information/MODEL_REFERENCE.md`
  §7, so dendrites interact with a recurrent gate rather than a plain linear
  layer). It is also a direct test case for the caveat #3 fix: if re-run
  shows its test delta move toward its validation delta, that's independent
  confirmation the fix works, on a model this session didn't touch to make it
  work.

## Why `m5` and `textcnn` over the untested remainder

`m5` (SpeechCommands, 1D-CNN) and `textcnn` (AG News, parallel-conv text
classifier) are the two cheapest architecturally-distinct domains left
(audio, text) with no flagged defects anywhere in `information/`. `textcnn`
additionally already has a clean, uncorrupted `top10` result to sanity-check
the new run against. Domains not covered by any of the 7 after this pick:
generative modeling (`vae_mnist`), spiking nets (`snn_nmnist`), point clouds
(`pointnet_modelnet40`), transformer NLP (`distilbert`), and heavier vision
CNNs (`resnet18_cifar10`, `mobilenetv2_cifar10`) — left out on cost or the
criterion-1 license block, not because they're uninteresting.

## Caveat this pick doesn't resolve

`vae_mnist`'s `ELBO` metric and `actor_critic`'s `Action Accuracy` both needed
an explicit "this number isn't what it looks like" note in `dynamic5`'s docs
(see `reference/SOTA.md`). None of `textcnn`/`m5`/`mpnn` need one — all three
report a metric that means exactly what its name says (accuracy or RMSE on a
held-out test set) — so this pick also avoids adding a fourth comparability
caveat to document and remember when reading the results table.
