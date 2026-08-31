# Deep review: models, data, specifications, and scope

**Scope:** `models.py`, `data.py`, `specs.py`, Dynamic12 helpers/docs.  
**Status:** analysis only; no source changes or deletions.

## Findings

### A. Model roster is broad relative to the experiment question

`specs.py` registers 24 model keys spanning vision, audio, text, graph, RL, forecasting, generative, spiking, segmentation, tabular, and capsule tasks. `models.py` is 1,867 lines and `data.py` 2,814 lines. A broad domain survey is useful once, but every model multiplies quantization kernels, PAI target policies, metrics, data loaders, tests, and documentation. The August 30 audit says only PointNet has a single-seed effect clearing its own noise floor; TCN has the only three-seed evidence and washes out the effect.

**Recommendation:** define a supported core (models with current Dynamic12 paired/PQAT evidence) and an explicitly archived exploratory roster. Do not remove a model solely because it lacks results; mark it `unsupported/unvalidated` and exclude it from default runs until a smoke test and condition matrix exist. Candidate default scope: PointNet, LeNet-5, TCN, and the matched dense controls required by the audit; retain ResNet/SAINT where they answer transfer/late-LR questions.

### B. Hidden model-key coupling is repeated across modules

Model keys branch independently in pipeline target selection (612–675), track-only lists (677–770), recipe overrides (around 1272–1297), training forward/loss/metric dispatch (1043–1070, 1220–1230), and data bundle construction (2776 onward). A new model requires edits in many places and can silently omit a factor. Replace with a `ModelAdapter` registry containing constructor, data builder, forward adapter, loss/metric, supported conditions, PAI targets, and default recipe.

### C. Duplicate/legacy architecture variants create bloat

Dynamic11/12 retain explicit variants (`gru_gate_ablation`, `tcn_head_output`, `tcn_head_both`, `vae_latent`, `mpnn_capacity`) and old Dynamic9–11 result trees. These are valid diagnostics only if their hypotheses and comparison controls remain documented. Make variants plugins/experiment specs rather than permanent branches in the main runner. **[DELETE CANDIDATE]** variant code and output after extracting results and confirming no current script invokes it; keep a compact manifest of historical conclusions.

### D. Data module mixes unrelated domains and cache policy

`data.py` groups vision, speech, audio, time series, text, graph, RL rollouts, ModelNet40, ISIC, and medical datasets in one module. It also owns cache paths and generated/augmented data. Split domain adapters and centralize cache metadata (dataset version, preprocessing revision, seed, split hash). Cache filenames must include every preprocessing factor; undocumented old rollout versions (`heuristic_rollouts.pt`, `_v2`, `_v3`) are **[DELETE CANDIDATES]** after checking which version `build_task_bundle` selects and archiving checksums.

### E. Static dead-code findings need framework-aware triage

Vulture/deadcode report `compat.py:pai_root` as unused (high-confidence; see audit 10), `data.py:ring_closure` as unused, and every `nn.Module.forward` as unused (false positives because PyTorch dispatches them dynamically). Dataclass fields flagged in `TrainingRecord` are serialized API. Remove only the helper/field candidates confirmed by call-graph and artifact-schema searches.

### F. Sonar/data correctness hotspots

Sonar reports cognitive complexity at `data.py:1272` (33), `1332` (33), `1400` (29), and `1479` (21), plus missing `dim` at 426/1779 (S6929), float equality at 1474 (S1244), repeated `gymnasium[box2d]` at 1682 (S1192), and a critical user-controlled loop bound at 2325 (S6680). Refactor loader/split builders into small pure functions, make reduction dimensions explicit, use tolerance/`isclose` where intended, centralize dependency literals, and validate bounds before indexing/iteration. These are correctness and security work, not merely style cleanup.

## Exact cleanup/deletion markers

- **[DELETE CANDIDATE]** `compat.py:pai_root` after external API check.
- **[DELETE OR PROMOTE]** `_Bond.ring_closure`: remove if parser never uses ring features; otherwise expose it in a tested graph feature.
- **[DELETE AFTER ARCHIVE]** duplicate rollout files and obsolete Dynamic9/10/11 variant outputs.
- **[RETAIN, EXCLUDE BY DEFAULT]** unvalidated models until their data/condition support is explicit.
- **[REFACTOR]** monolithic data builders and model factory into adapters/registries.

