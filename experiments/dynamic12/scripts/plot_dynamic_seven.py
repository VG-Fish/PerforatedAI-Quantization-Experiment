"""Single-figure summary of the seven dynamic-dendritic models.

Reads the stored `record.json` for each model's `base_fp32` and
`dendrites_fp32` arms and draws one direction-corrected bar per model, so
accuracy, MAE, RMSE, and ELBO land on a comparable axis where positive always
means "the dendritic arm did better".

The bars are keyed to `comparison/dendrite_audit.csv`, which is the point of
the figure: only two of the seven arms carry a verified retained dendrite, and
one of them (`tcn_forecaster`) has an identical parameter count to its base, so
its apparent gain cannot be a dendrite effect at all. A plain seven-bar
"improvement" chart would imply evidence that is not there.

Usage:
    uv run python experiments/dynamic12/scripts/plot_dynamic_seven.py \
        --run experiments/dynamic12/combined/seed_0
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.patches import Patch

# Shared with src/dendritic_benchmark/plots.py so this figure sits beside the
# runner's own SVGs without looking like it came from somewhere else.
BACKGROUND = "#fbfaf7"
GRID = "#d9d2c3"
TEXT = "#16202a"
MUTED = "#52606d"

# Audit status drives the colour. The distinction is the whole point: an
# unverified or absent dendrite is not a weaker result, it is a different claim.
STATUS_STYLE: dict[str, tuple[str, str]] = {
    "verified_retained": ("#2f855a", "verified retained dendrite"),
    "legacy_unchecked": ("#a0aec0", "unverified (pre-audit run)"),
    "no_retained_insertion": ("#c05621", "no dendrite retained"),
}
FALLBACK_STYLE = ("#a0aec0", "unverified")


def _load_record(run: Path, model: str, condition: str) -> dict[str, Any] | None:
    path = run / "results" / model / condition / "record.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _audit_statuses(run: Path) -> dict[str, str]:
    path = run / "comparison" / "dendrite_audit.csv"
    if not path.is_file():
        return {}
    with path.open() as handle:
        return {
            row["model_key"]: row["dendrite_audit_status"]
            for row in csv.DictReader(handle)  # type: ignore[no-matching-overload]
            if row.get("condition_key") == "dendrites_fp32"
        }


def _relative_change(base: float, dendritic: float, direction: str) -> float:
    """Percent change with "better" always positive, whatever the metric."""
    if base == 0.0:
        return 0.0
    delta = (dendritic - base) if direction == "maximize" else (base - dendritic)
    return 100.0 * delta / abs(base)


def collect(run: Path) -> list[dict[str, Any]]:
    statuses = _audit_statuses(run)
    results_dir = run / "results"
    rows: list[dict[str, Any]] = []
    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        model = model_dir.name
        base = _load_record(run, model, "base_fp32")
        dendritic = _load_record(run, model, "dendrites_fp32")
        if base is None or dendritic is None:
            continue
        base_params = int(base["param_count"])
        rows.append(
            {
                "model": model,
                "metric": base["metric_name"],
                "base": float(base["metric_value"]),
                "dendritic": float(dendritic["metric_value"]),
                "change": _relative_change(
                    float(base["metric_value"]),
                    float(dendritic["metric_value"]),
                    base["metric_direction"],
                ),
                "param_change": (
                    100.0 * (int(dendritic["param_count"]) - base_params) / base_params
                    if base_params
                    else 0.0
                ),
                "status": statuses.get(model, "legacy_unchecked"),
            }
        )
    return rows


def render(rows: list[dict[str, Any]], out: Path) -> None:
    rows = sorted(rows, key=lambda r: r["change"])
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    positions = range(len(rows))
    colors = [STATUS_STYLE.get(r["status"], FALLBACK_STYLE)[0] for r in rows]
    # A retained dendrite is a solid claim; anything else is drawn hollow so it
    # never reads as evidence at a glance.
    bars = ax.barh(
        list(positions),
        [r["change"] for r in rows],
        color=[
            c if r["status"] == "verified_retained" else BACKGROUND
            for c, r in zip(colors, rows)
        ],
        edgecolor=colors,
        linewidth=1.6,
        height=0.62,
    )

    lo = min(0.0, min(r["change"] for r in rows))
    hi = max(0.0, max(r["change"] for r in rows))
    span = (hi - lo) or 1.0
    offset = span * 0.015
    for bar, row in zip(bars, rows):
        change = row["change"]
        ax.text(
            change + (offset if change >= 0 else -offset),
            bar.get_y() + bar.get_height() / 2,
            f"{change:+.2f}%",
            va="center",
            ha="left" if change >= 0 else "right",
            fontsize=8.5,
            color=TEXT,
        )

    ax.set_yticks(list(positions))
    # Parameter cost rides in the tick label rather than in a left-hand gutter:
    # widening xlim to make room for it would draw axis ticks over empty space
    # and read as data that is not there.
    ax.set_yticklabels(
        [
            f"{r['model']}\n{r['metric']} · {r['param_change']:+.0f}% params"
            for r in rows
        ],
        fontsize=8.5,
        color=TEXT,
    )
    ax.axvline(0.0, color=TEXT, linewidth=1.0)
    # Extra left pad only when a bar is negative, so its value label has room
    # without colliding with the tick labels.
    ax.set_xlim(lo - span * (0.20 if lo < 0 else 0.02), hi + span * 0.16)
    ax.set_xlabel(
        "dendritic vs base, direction-corrected (positive = dendritic better)",
        fontsize=9,
        color=TEXT,
    )
    ax.set_title(
        "Dynamic12 · seven dynamic-dendritic models\n"
        "only two arms carry a verified retained dendrite",
        fontsize=11,
        color=TEXT,
        loc="left",
    )
    ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT, length=0)

    present = [s for s in STATUS_STYLE if any(r["status"] == s for r in rows)]
    ax.legend(
        handles=[
            Patch(
                facecolor=STATUS_STYLE[s][0] if s == "verified_retained" else BACKGROUND,
                edgecolor=STATUS_STYLE[s][0],
                linewidth=1.6,
                label=STATUS_STYLE[s][1],
            )
            for s in present
        ],
        loc="lower right",
        fontsize=7.5,
        frameon=False,
    )
    fig.text(
        0.005,
        -0.01,
        "tcn_forecaster's dendritic arm has the same parameter count as its base, "
        "so its gain is not a dendrite effect.\n"
        "Unverified arms predate the dendrite audit; their bars are not evidence "
        "for or against dendrites.",
        fontsize=7,
        color=MUTED,
        va="top",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format=out.suffix.lstrip(".") or "svg", facecolor=BACKGROUND,
                bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=Path("experiments/dynamic12/combined/seed_0"),
        help="run directory holding results/ and comparison/",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = collect(args.run)
    if not rows:
        raise SystemExit(f"no base_fp32/dendrites_fp32 pairs found under {args.run}")
    out = args.out or args.run / "comparison" / "dynamic_seven_summary.svg"
    render(rows, out)
    print(f"{len(rows)} models -> {out}")
    for row in sorted(rows, key=lambda r: -r["change"]):
        print(
            f"  {row['model']:<16} {row['metric']:<16} "
            f"{row['change']:+7.2f}%  {row['param_change']:+7.1f}% params  "
            f"{row['status']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
