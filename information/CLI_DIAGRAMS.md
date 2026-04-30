# CLI Command Diagrams

Mermaid flowcharts for all `uv run dqb` commands.

---

## Global Options

All commands share these top-level flags:

| Flag | Default | Description |
|---|---|---|
| `--results-root DIR` | `results` | Root directory for per-model result folders |
| `--comparison-root DIR` | `comparison` | Root directory for cross-model comparison outputs |
| `--logging-dir DIR` | `logs` | Directory for timestamped log files |

---

## `uv run dqb run`

Trains models across all (or a subset of) conditions and saves results.

```bash
uv run dqb run
uv run dqb run --models lenet5 textcnn
uv run dqb run --conditions base_fp32 base_q8 dendrites_fp32
uv run dqb run --results-root results
uv run dqb run --comparison-root comparison
uv run dqb run --ignore-saved-models
```

```mermaid
flowchart TD
    A([uv run dqb run]) --> B[Parse args --models, --conditions, --ignore-saved-models]
    B --> C[Load .env credentials via compat.py]
    C --> D[BenchmarkRunner pipeline.py]
    D --> E[Create results/ and comparison/ directories]
    E --> F[Resolve conditions in dependency order]
    F --> G{For each model × condition}
    G --> H{record.json already exists?}
    H -- "Yes, not --ignore-saved-models" --> I[Skip — load existing record]
    H -- "No or --ignore-saved-models" --> J[data.py build_task_bundle]
    J --> K[models.py build_model]
    K --> L{Dendritic condition?}
    L -- Yes --> M[compat.py perforate_model via PAI]
    L -- No --> N[Standard model]
    M --> O[training.py train_and_evaluate]
    N --> O
    O --> P{Pruning condition?}
    P -- Yes --> Q[Apply L1 unstructured global pruning 40%]
    P -- No --> R{Quantization condition?}
    Q --> R
    R -- Q8/Q4/Q2/Q1.58/Q1 --> S[Apply torchao PTQ or QAT projection]
    R -- FP32 --> T[Train full epochs with Adam + scheduler]
    S --> T
    T --> U[Evaluate val + test metrics]
    U --> V[Save artifacts: model.pt, metrics.json, history.csv, plots/]
    V --> W{Dendritic run?}
    W -- Yes --> X[Also save: best_model, final_clean_pai, best_arch_scores.csv, paramCounts.csv]
    W -- No --> Y[results.py save_training_record record.json + record.csv]
    X --> Y
    Y --> G
    G -- Done --> Z[results.py write_model_reports write_comparison_reports]
    Z --> ZZ([End])

    style A fill:#2d6a4f,color:#fff
    style ZZ fill:#2d6a4f,color:#fff
    style H fill:#457b9d,color:#fff
    style L fill:#457b9d,color:#fff
    style P fill:#457b9d,color:#fff
    style R fill:#457b9d,color:#fff
    style W fill:#457b9d,color:#fff
```

### Condition Dependency Chain

Conditions must be run in the order below — omitting an upstream condition causes its dependents to be skipped.

```mermaid
flowchart LR
    A[base_fp32] --> B[base_q8]
    A --> C[base_q4]
    A --> D[base_q2]
    A --> E[base_q1_58]
    A --> F[base_q1]
    A --> G[dendrites_fp32]
    G --> H[dendrites_pruned]
    H --> I[dendrites_pruned_q8]
    H --> J[dendrites_pruned_q4]
    H --> K[dendrites_pruned_q2]
    H --> L[dendrites_pruned_q1_58]
    H --> M[dendrites_pruned_q1]

    style A fill:#1d3557,color:#fff
    style G fill:#2d6a4f,color:#fff
    style H fill:#2d6a4f,color:#fff
```

---

## `uv run dqb download_data`

Pre-downloads and caches all datasets so that `run` can work offline.

```bash
uv run dqb download_data
uv run dqb download_data --models lenet5 mpnn
uv run dqb download_data --strict
```

```mermaid
flowchart TD
    A([uv run dqb download_data]) --> B[Parse args --models, --strict]
    B --> C[Resolve DQB_DATA_ROOT env var or default ./data]
    C --> D{For each selected model}
    D --> E{dataset_exists model_key?}
    E -- Yes --> F[Skip — already cached]
    E -- No --> G[data.py build_task_bundle download + prepare]
    G --> H{Success?}
    H -- Yes --> I[Log done + elapsed time]
    H -- "No and --strict" --> J[Raise exception and abort]
    H -- "No and not --strict" --> K[Log FAILED continue to next model]
    F --> D
    I --> D
    K --> D
    D -- Done --> L[Print summary: downloaded, cached, failed counts]
    L --> ZZ([End])

    style A fill:#2d6a4f,color:#fff
    style ZZ fill:#2d6a4f,color:#fff
    style E fill:#457b9d,color:#fff
    style H fill:#457b9d,color:#fff
    style J fill:#e63946,color:#fff
```

---

## `uv run dqb compare`

Rebuilds all comparison outputs from previously saved `record.json` files without retraining.

```bash
uv run dqb compare
uv run dqb compare --manifest
uv run dqb compare --results-root results --comparison-root comparison
```

```mermaid
flowchart TD
    A([uv run dqb compare]) --> B[Parse args --manifest]
    B --> C[results.py load_training_records scan results/*/*/record.json]
    C --> D{--manifest flag set?}
    D -- Yes --> E[results.py write_manifest results/manifest.csv]
    D -- No --> F
    E --> F{For each ModelSpec in MODEL_SPECS}
    F --> G[Filter records for this model]
    G --> H{Any records found?}
    H -- Yes --> I[results.py write_model_reports accuracy + param + size bar charts per model]
    H -- No --> F
    I --> F
    F -- Done --> J[results.py write_comparison_reports all records]
    J --> K[accuracy_retention_heatmap.svg 10x13 normalized scores]
    J --> L[size_tradeoff_scatter.svg 130 model×condition points]
    J --> M[dendrite_delta.svg Base FP32 vs Dendrites FP32]
    J --> N[best_quantization_heatmap.svg best accuracy per bit level]
    J --> O[summary.csv]
    K & L & M & N & O --> ZZ([End])

    style A fill:#2d6a4f,color:#fff
    style ZZ fill:#2d6a4f,color:#fff
    style D fill:#457b9d,color:#fff
    style H fill:#457b9d,color:#fff
```

---

## `uv run dqb generate_graphs`

Renders per-epoch training-curve plots from saved result histories without retraining.

```bash
uv run dqb generate_graphs
uv run dqb generate_graphs --results-root results
uv run dqb generate_graphs --regenerate-graphs
```

```mermaid
flowchart TD
    A([uv run dqb generate_graphs]) --> B[Parse args --regenerate-graphs]
    B --> C[results.py generate_training_graphs results_root, regenerate flag]
    C --> D{Walk results/ for each condition folder}
    D --> E{history.csv exists?}
    E -- No --> D
    E -- Yes --> F{Graph files already exist?}
    F -- "Yes, not --regenerate-graphs" --> D
    F -- "No or --regenerate-graphs" --> G[plots/ Create or recreate plots directory]
    G --> H[plots.py Render training_curve.svg primary_metric.svg loss_curves.svg]
    H --> I{Dendritic condition?}
    I -- Yes --> J[plots.py Render architecture_evolution.svg from best_arch_scores.csv]
    I -- No --> D
    J --> D
    D -- Done --> ZZ([End])

    style A fill:#2d6a4f,color:#fff
    style ZZ fill:#2d6a4f,color:#fff
    style E fill:#457b9d,color:#fff
    style F fill:#457b9d,color:#fff
    style I fill:#457b9d,color:#fff
```

---

## Output Directory Layout

```mermaid
flowchart TD
    R[results/] --> M1[model_key/]
    M1 --> C1[condition_key/]
    C1 --> RJ[record.json]
    C1 --> RC[record.csv]
    C1 --> MJ[metrics.json]
    C1 --> HC[history.csv]
    C1 --> MP[model.pt]
    C1 --> PL[plots/ training_curve.svg primary_metric.svg loss_curves.svg]
    C1 --> DD[dendritic only: best_model final_clean_pai best_arch_scores.csv paramCounts.csv architecture_evolution.svg]

    CO[comparison/] --> H1[accuracy_retention_heatmap.svg]
    CO --> H2[size_tradeoff_scatter.svg]
    CO --> H3[dendrite_delta.svg]
    CO --> H4[best_quantization_heatmap.svg]
    CO --> H5[summary.csv]

    RM[results/manifest.csv]

    style DD fill:#2d6a4f,color:#fff
```

---

## Command Summary

```mermaid
flowchart LR
    CLI([uv run dqb]) --> RUN[run Train models and save results]
    CLI --> DL[download_data Pre-cache datasets offline-safe]
    CLI --> CMP[compare Rebuild comparison plots from records]
    CLI --> GG[generate_graphs Render training curves from history]

    RUN -->|writes| RES[(results/ records + plots)]
    RUN -->|writes| COM[(comparison/ charts + summary)]
    DL -->|writes| DAT[(data/ dataset cache)]
    CMP -->|reads| RES
    CMP -->|writes| COM
    GG -->|reads| RES
    GG -->|writes| RES

    style CLI fill:#1d3557,color:#fff
    style RES fill:#457b9d,color:#fff
    style COM fill:#457b9d,color:#fff
    style DAT fill:#457b9d,color:#fff
```
