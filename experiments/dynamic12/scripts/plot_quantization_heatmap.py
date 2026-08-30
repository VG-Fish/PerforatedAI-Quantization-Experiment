"""Best-quantization heatmap over a chosen model set, including quantized arms.

The runner's own `best_quantization_heatmap.svg` drops every record that
`_record_is_reportable` rejects. For `experiments/dynamic12/combined/seed_0`
that is all 40 quantized conditions (`stale_double_projection` — they were
measured under the superseded double-projection QAT evaluation path), which
leaves a heatmap whose five quantization columns are entirely empty.

This script keeps those arms so the quantized behaviour is visible, and labels
them as stale on the figure itself. It reuses the runner's own retention math
(`_process_model_comparison`) rather than recomputing it, so the numbers are
identical to what a re-run would produce once the arms are recomputed.

Columns with no record at all for any model are dropped rather than drawn as
0% — for this run that is Q1, which was never run and is not the same claim as
"quantizing to 1 bit scored zero".

Usage:
    uv run python experiments/dynamic12/scripts/plot_quantization_heatmap.py \
        --run experiments/dynamic12/combined/seed_0 \
        --models lenet5 gcn mpnn actor_critic vae_mnist
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dendritic_benchmark.plots import winner_heatmap
from dendritic_benchmark.results import (
    _baseline_lookup,
    _process_model_comparison,
    load_training_records,
)
from dendritic_benchmark.specs import CONDITION_SPECS, MODEL_SPECS

# Same grouping the runner uses; index i of each row is the i-th group here.
QUANTIZATION_GROUPS: list[list[str]] = [
    ["base_fp32", "dendrites_fp32"],
    ["base_q8", "dendrites_q8"],
    ["base_q4", "dendrites_q4"],
    ["base_q2", "dendrites_q2"],
    ["base_q1_58", "dendrites_q1_58"],
    ["base_q1", "dendrites_q1"],
]
COLUMN_LABELS = ["FP32", "Q8", "Q4", "Q2", "Q1.58", "Q1"]

DEFAULT_MODELS = ["lenet5", "gcn", "mpnn", "actor_critic", "vae_mnist"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path,
                        default=Path("experiments/dynamic12/combined/seed_0"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    records = load_training_records(args.run / "results")
    if not records:
        raise SystemExit(f"no records under {args.run / 'results'}")

    wanted = set(args.models)
    specs = [spec for spec in MODEL_SPECS if spec.key in wanted]
    missing = wanted - {spec.key for spec in specs}
    if missing:
        raise SystemExit(f"unknown model keys: {sorted(missing)}")

    # Deliberately NOT filtered through _record_is_reportable: including the
    # stale quantized arms is the whole point of this figure.
    baselines = _baseline_lookup(records)
    condition_order = [spec.key for spec in CONDITION_SPECS]

    quant_rows: list[list[float]] = []
    other_rows: list[list[float]] = []
    winners: list[list[int]] = []
    for spec in specs:
        _ret, quant_row, other_row, winner_row, _s, _t = _process_model_comparison(
            spec.key, records, baselines, condition_order, QUANTIZATION_GROUPS
        )
        quant_rows.append(quant_row)
        other_rows.append(other_row)
        winners.append(winner_row)

    present = {
        index
        for index, group in enumerate(QUANTIZATION_GROUPS)
        if any(
            record["model_key"] in wanted and record["condition_key"] in group
            for record in records
        )
    }
    dropped = [COLUMN_LABELS[i] for i in range(len(COLUMN_LABELS)) if i not in present]
    keep = sorted(present)
    labels = [COLUMN_LABELS[i] for i in keep]
    quant_rows = [[row[i] for i in keep] for row in quant_rows]
    other_rows = [[row[i] for i in keep] for row in other_rows]
    winners = [[row[i] for i in keep] for row in winners]

    # winner_heatmap renders the subtitle as part of the title at fontsize 18,
    # so it has to be hand-wrapped or it stretches the whole figure.
    caveat = "Q1 omitted (never run)" if dropped else ""
    if dropped and dropped != ["Q1"]:
        caveat = f"{'/'.join(dropped)} omitted (never run)"
    subtitle = "\n".join(
        line
        for line in (
            "Which variant achieves the best retention per quantization level (%)",
            "Quantized arms are stale_double_projection (pre-2026-08-11 QAT eval):",
            " · ".join(
                part for part in ("shown for shape, not as final numbers", caveat)
                if part
            ),
        )
        if line
    )

    out = args.out or (
        args.run / "comparison" / "best_quantization_heatmap_five_models.svg"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    winner_heatmap(
        out,
        "Best Quantization Level per Domain",
        [spec.display_name for spec in specs],
        labels,
        winners,
        quant_rows,
        subtitle=subtitle,
        metric_labels=[spec.metric_name for spec in specs],
        other_score_matrix=other_rows,
    )
    print(f"{len(specs)} models x {len(labels)} levels -> {out}")
    if dropped:
        print(f"  omitted (no records): {', '.join(dropped)}")
    header = "  " + "model".ljust(14) + "".join(lab.rjust(9) for lab in labels)
    print(header)
    for spec, row, win in zip(specs, quant_rows, winners):
        cells = "".join(
            f"{value:8.1f}{'D' if flag else 'B'}" for value, flag in zip(row, win)
        )
        print("  " + spec.display_name.ljust(14) + cells)
    print("  (value = best-variant retention vs base FP32; D = dendrites won, B = base)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
