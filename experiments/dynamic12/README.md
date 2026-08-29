# Dynamic12

`combined/seed_0` contains the three refreshed model results plus the unchanged
models merged from Dynamic10. Before launching another full seed, run:

```bash
experiments/dynamic12/config/run_smoke_tests.sh
```

It uses a single real training batch for TCN, GRU, and VAE; checks forward,
loss, backward, optimizer, FP32/Q8/Q4/Q2/channel-ternary inference, and a
one-batch ternary-QAT single-projection check; then
checks every configured PAI target variant and the target's inferred output
dimensions. It makes no benchmark artifacts and does not start PAI candidate
training.

To restrict the smoke test, pass the same model argument to both checks:

```bash
experiments/dynamic12/config/run_smoke_tests.sh --models tcn_forecaster
```

The runner writes `comparison/dendrite_audit.csv`. Records with no raw PAI
candidate-insertion switch or no retained parameter increase are marked
inconclusive and excluded from dendrite comparisons. QAT artifacts without the
single-projection revision are similarly excluded until recomputed.

Long follow-ups are intentionally separate from smoke checks:

```bash
# FP32 audited TCN target variants, seeds 0–2.
experiments/dynamic12/config/run_tcn_audited_variants.sh

# Recompute the credible GRU/VAE comparisons after the QAT correction.
experiments/dynamic12/config/run_validated_replications.sh

# Dense controls: GRU decoder 51 ~= the 137,040-parameter dendritic arm;
# VAE scale 1.0 ~= the 1.07M-parameter dendritic arm.
uv run python experiments/dynamic12/tuning/tune_gru_capacity.py
uv run python experiments/dynamic12/tuning/tune_vae_capacity.py
```
