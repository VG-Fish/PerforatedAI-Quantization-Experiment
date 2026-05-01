# CLI Command Diagrams

Mermaid flowcharts for all `uv run dqb` commands.

---

## Shared Options

These flags are shared by all commands:

| Flag | Default | Description |
|---|---|---|
| `--results-root DIR` | `results` | Root directory for per-model result folders (also used by `benchmark_models` to locate trained models) |
| `--results-directory NAME` | _unset_ | Optional subdirectory under `--results-root`; when set, results path becomes `<results-root>/<results-directory>` |
| `--logging-dir DIR` | `logs` | Directory for timestamped log files |

`--comparison-root DIR` is available only on `uv run dqb run` and `uv run dqb compare`.

`--benchmark-root DIR` is available only on `uv run dqb benchmark_models`.

---

## `uv run dqb run`

Trains models across all (or a subset of) conditions and saves results.

```bash
uv run dqb run
uv run dqb --results-directory experiment_a run
uv run dqb run --models lenet5 textcnn
uv run dqb run --conditions base_fp32 base_q8 dendrites_fp32
uv run dqb run --results-root results
uv run dqb run --comparison-root comparison
uv run dqb run --allow-PQAT
uv run dqb run --dynamic-dendritic-training
uv run dqb run --ignore-saved-models
```

```mermaid
flowchart TD
    A([uv run dqb run]) --> B["Parse args<br>--models, --conditions,<br>--ignore-saved-models,<br>--allow-PQAT,<br>--dynamic-dendritic-training"]
    B --> C["Load .env credentials<br>via compat.py"]
    C --> D["BenchmarkRunner<br>pipeline.py"]
    D --> E["Create results/ and<br>comparison/ directories"]
    E --> F["Resolve conditions in<br>dependency order"]
    F --> G{"For each<br>model + condition"}
    G --> |While Training|H{"record.json<br>already exists?"}
    H -->|Yes, <br>--ignore-saved-models<br> not set| I["Skip load existing record"]
    H -->|No, or <br>--ignore-saved-models<br> set| J["data.py<br>build_task_bundle"]
    J --> K["models.py<br>build_model"]
    K --> L{"Dendritic<br>condition?"}
    L -->|Yes| M["compat.py perforate_model<br>via PAI"]
    L -->|No| N["Standard model"]
    M --> O["training.py<br>train_and_evaluate<br>FP32 dendrites fixed budget by default;<br>dynamic flag trains until PAI complete"]
    N --> O
    O --> R{"Quantization<br>condition?"}
    R -->|Q8/Q4/Q2/Q1.58/Q1| S["Load source checkpoint and<br>apply PTQ snapshot"]
    R -->|FP32| T["Train full epochs with<br>model-specific recipe"]
    S --> S2{"--allow-PQAT?"}
    S2 -->|No| U
    S2 -->|Yes| S3["Save before_pqat/<br>then fine-tune for a<br>model-aware PQAT budget<br>then save after_pqat/"]
    S3 --> U
    T --> U["Evaluate val + test metrics"]
    U --> V["Save artifacts:<br>model.pt (best), best_model_stats.csv,<br>metrics.json, history.csv, plots/"]
    V --> W{"Dendritic<br>run?"}
    W -->|Yes| X["Also save:<br>PAI_config.json<br>best_arch_scores.csv<br>and paramCounts.csv"]
    W -->|No| Y["results.py:<br>save_training_record<br>record.json + record.csv"]
    X --> Y
    Y --> G
    G -->|Training Complete?| Z["results.py<br>write_model_reports<br>write_comparison_reports"]
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
    G --> H[dendrites_q8]
    G --> I[dendrites_q4]
    G --> J[dendrites_q2]
    G --> K[dendrites_q1_58]
    G --> L[dendrites_q1]

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
    A([uv run dqb download_data]) --> B["Parse args<br>--models, --strict"]
    B --> C["Resolve DQB_DATA_ROOT<br>env var or default ./data"]
    C --> D{"For each<br>selected model"}
    D --> E{"dataset_exists<br>model_key?"}
    
    %% Branch 1: Exists
    E -->|Yes| F["Skip — already cached"]
    F --> M{Next Model}

    %% Branch 2: Needs download
    E -->|No| G["data.py<br>build_task_bundle<br>download + prepare"]
    G --> H{"Success?"}
    
    %% Success
    H -->|Yes| I["Log done + elapsed time"]
    I --> M
    
    %% Failure options
    H -->|No and --strict set| J["Raise exception abort"]
    J --> ZZ([End])
    
    H -->|No and --strict not set| K["Log FAILED<br>continue next model"]
    K --> M

    %% Loop back
    M -->|More Models| D
    M -->|Done| L["Print summary:<br>downloaded, cached,<br>failed counts"]
    L --> ZZ

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
uv run dqb --results-directory experiment_a compare
uv run dqb compare --manifest
uv run dqb compare --results-root results --comparison-root comparison
```

```mermaid
flowchart TD
    A([uv run dqb compare]) --> B["Parse args --manifest"]
    B --> C["results.py:<br>load_training_records<br>scan:<br>results/\*/\*/record.json"]
    C --> D{"--manifest<br>flag set?"}
    D -->|Yes| E["results.py write_manifest<br>results/manifest.csv"]
    D -->|No| F{"For each ModelSpec<br>in MODEL_SPECS"}
    E --> F
    F --> G["Filter records for this model"]
    G --> H{"Any records<br>found?"}
    H -->|Yes| I["results.py write_model_reports<br>accuracy + param + size<br>bar charts per model"]
    H -->|No| F
    I --> F
    F -->|Done| J["results.py<br>write_comparison_reports<br>all records"]
    J --> K["accuracy_retention_heatmap.svg<br>10x13 normalized scores"]
    J --> L["size_tradeoff_scatter.svg<br>130 model×condition points"]
    J --> M["dendrite_delta.svg<br>Base FP32 vs Dendrites FP32"]
    J --> N["best_quantization_heatmap.svg<br>best accuracy per bit level"]
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
uv run dqb --results-directory experiment_a generate_graphs
uv run dqb generate_graphs --results-root results
uv run dqb generate_graphs --regenerate-graphs
```

```mermaid
flowchart TD
    A([uv run dqb generate_graphs]) --> B["Parse args<br>--regenerate-graphs"]
    B --> C["results.py<br>generate_training_graphs<br>results_root,<br>regenerate flag"]
    C --> C2["Comparison outputs are not managed here;<br>use dqb compare for comparison/"]
    C2 --> D{"Walk results/<br>for each condition folder"}
    D --> E{"history.csv<br>exists?"}
    E -->|No| D
    E -->|Yes| F{"Graph files<br>already exist?"}
    F -->|Yes, or<br>--regenerate-graphs not set| D
    F -->|No, or<br>--regenerate-graphs set| G["plots/<br>Create or recreate plots dir"]
    G --> H["plots.py: Render<br>training_curve.svg<br>primary_metric.svg<br>loss_curves.svg"]
    H --> I{"Dendritic<br>condition?"}
    I -->|Yes| J["plots.py: Render<br>architecture_evolution.svg<br>from best_arch_scores.csv"]
    I -->|No| D
    J --> D
    D -->|Done| ZZ([End])

    style A fill:#2d6a4f,color:#fff
    style ZZ fill:#2d6a4f,color:#fff
    style E fill:#457b9d,color:#fff
    style F fill:#457b9d,color:#fff
    style I fill:#457b9d,color:#fff
```

---

## `uv run dqb benchmark_models`

Measures wall-clock inference latency for all trained models using `torch.utils.benchmark.Timer`.

```bash
uv run dqb benchmark_models
uv run dqb --results-directory experiment_a benchmark_models
uv run dqb benchmark_models --models lenet5 m5
uv run dqb benchmark_models --conditions base_fp32 base_q4 dendrites_q4
uv run dqb benchmark_models --batch-sizes 1 8 32
uv run dqb benchmark_models --num-runs 20
uv run dqb benchmark_models --benchmark-root my_benchmarks
```

```mermaid
flowchart TD
    A([uv run dqb benchmark_models]) --> B["Parse args<br>--models, --conditions,<br>--batch-sizes,<br>--num-runs,<br>--benchmark-root"]
    B --> C["Create benchmark directory"]
    C --> D["Capture system info<br>CPU, device, PyTorch version"]
    D --> E["Write computer_info.json"]
    E --> F{"For each<br>selected model"}
    F --> G{"For each<br>selected condition"}
    G --> H["Load trained model from<br>results/model/condition/model.pt"]
    H --> I{"Model loaded<br>successfully?"}
    I -->|No| J["Log error skip condition"]
    I -->|Yes| K["Generate sample inputs<br>matching model shape"]
    K --> L["For each batch size"]
    L --> M["Run torch.utils.benchmark.Timer<br>for num_runs iterations"]
    M --> N["Collect latency stats<br>mean, median, stdev"]
    N --> O["Write results to<br>condition_key.json"]
    O --> P["Aggregate to<br>latency_summary.csv"]
    P --> L
    L -->|Done| G
    J --> G
    G -->|Done| Q["Write model<br>latency_summary.csv"]
    Q --> F
    F -->|Done| R["Write manifest.csv<br>all benchmarks"]
    R --> ZZ([End])

    style A fill:#2d6a4f,color:#fff
    style ZZ fill:#2d6a4f,color:#fff
    style I fill:#457b9d,color:#fff
    style L fill:#457b9d,color:#fff
```

### Output Files

Results are organized by model and condition:

- `benchmarks/computer_info.json` — System specification snapshot
- `benchmarks/manifest.csv` — Cross-model summary of all latency measurements
- `benchmarks/{model}/latency_summary.csv` — Per-model aggregated latencies
- `benchmarks/{model}/{condition}.json` — Full results for one model + condition

---

## Output Directory Layout

```text
.
├── benchmarks/                          # created by dqb benchmark_models
│   ├── computer_info.json
│   ├── manifest.csv
│   └── model_key/
│       ├── latency_summary.csv
│       └── condition_key.json
├── comparison/
│   ├── accuracy_retention_heatmap.svg
│   ├── best_quantization_heatmap.svg
│   ├── dendrite_delta.svg
│   ├── size_tradeoff_scatter.svg
│   └── summary.csv
├── logs/
│   └── command_timestamp.txt
├── PAI/
│   ├── PAI_config.json
│   ├── model_key_condition_key_PAI_config.json
│   └── model_key_condition_key/
└── results/
    └── [results-directory]/                # optional, from --results-directory
        ├── manifest.csv
        └── model_key/
            └── condition_key/
                ├── history.csv
                ├── metrics.json
                ├── model.pt
                ├── PAI_config.json              # dendritic only
                ├── best_model_stats.csv
                ├── record.csv
                ├── record.json
                ├── before_pqat/                 # quantized runs only when --allow-PQAT is enabled
                ├── after_pqat/                  # quantized runs only when --allow-PQAT is enabled
                ├── continued_until_complete/    # dynamic FP32 dendritic epochs beyond max_epochs
                ├── plots/
                │   ├── architecture_evolution.svg  # dendritic only
                │   ├── loss_curves.svg
                │   ├── primary_metric.svg
                │   └── training_curve.svg
                ├── best_arch_scores.csv          # dendritic only
                └── paramCounts.csv               # dendritic only
```

---

## Command Summary

```mermaid
flowchart TD
    CLI([uv run dqb])
    
    GG["generate_graphs<br>Render training curves"]
    CMP["compare<br>Rebuild comparison plots"]
    RUN["run<br>Train models and<br>save results"]
    DL["download_data<br>Pre-cache datasets"]
    BEN["benchmark_models<br>Measure inference<br>latency"]

    RES[("results/<br>records + plots")]
    COM[("comparison/<br>charts + summary")]
    DAT[("data/<br>dataset cache")]
    BEN_OUT[("benchmarks/<br>latency + system info")]

    CLI --> GG
    CLI --> CMP
    CLI --> RUN
    CLI --> DL
    CLI --> BEN

    GG <-->|reads & writes| RES
    CMP -->|reads| RES
    CMP -->|writes| COM
    RUN -->|writes| RES
    RUN -->|writes| COM
    DL -->|writes| DAT
    BEN -->|reads| RES
    BEN -->|writes| BEN_OUT

    style CLI fill:#1d3557,color:#fff
    style RES fill:#457b9d,color:#fff
    style COM fill:#457b9d,color:#fff
    style DAT fill:#457b9d,color:#fff
    style BEN_OUT fill:#457b9d,color:#fff
```
