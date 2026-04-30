from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox

BACKGROUND = "#fbfaf7"
GRID = "#d9d2c3"
TEXT = "#16202a"
MUTED = "#52606d"
BASE_BLUE = "#2b6cb0"
DENDRITE_GREEN = "#2f855a"
PALETTE = [
    BASE_BLUE,
    "#3182ce",
    "#4299e1",
    DENDRITE_GREEN,
    "#38a169",
    "#68d391",
    "#c05621",
    "#dd6b20",
    "#805ad5",
    "#319795",
]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _wrap_label(label: str, width: int = 12) -> str:
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False)) or label


def _palette(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def _setup_figure(width: float, height: float) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.tick_params(colors=TEXT)
    for spine in ax.spines.values():
        spine.set_color(TEXT)
    return fig, ax


def _draw(fig: Figure) -> None:
    fig.canvas.draw()


def _text_bboxes(fig: Figure, artists: list[Any]) -> list[Bbox]:
    _draw(fig)
    renderer = getattr(fig.canvas, "get_renderer")()
    return [
        artist.get_window_extent(renderer=renderer).expanded(1.04, 1.12)
        for artist in artists
        if artist.get_visible() and artist.get_text()
    ]


def _has_overlaps(fig: Figure, artists: list[Any]) -> bool:
    bboxes = _text_bboxes(fig, artists)
    for index, bbox in enumerate(bboxes):
        if any(bbox.overlaps(other) for other in bboxes[index + 1 :]):
            return True
    return False


def _autosize_axis_labels(fig: Figure, ax: Axes, *, min_font: int = 7) -> None:
    """Rotate and shrink x tick labels until the renderer no longer sees collisions."""
    tick_labels = list(ax.get_xticklabels())
    if not tick_labels:
        return

    candidates: list[tuple[int, int, Literal["left", "center", "right"]]] = [
        (0, 10, "center"),
        (25, 9, "right"),
        (45, 8, "right"),
        (60, 7, "right"),
    ]
    for rotation, font_size, horizontal_alignment in candidates:
        for label in tick_labels:
            label.set_rotation(rotation)
            label.set_fontsize(max(min_font, font_size))
            label.set_horizontalalignment(horizontal_alignment)
            label.set_rotation_mode("anchor")
        _draw(fig)
        if not _has_overlaps(fig, tick_labels):
            return


def _annotate_bars_without_overlap(
    fig: Figure, ax: Axes, bars: Any, values: list[float]
) -> None:
    _draw(fig)
    renderer = getattr(fig.canvas, "get_renderer")()
    accepted: list[Bbox] = []
    skipped = 0
    y_min, y_max = ax.get_ylim()
    y_offset = (y_max - y_min) * 0.012

    for bar, value in zip(bars, values):
        if not math.isfinite(value):
            skipped += 1
            continue
        label = ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_offset,
            f"{value:.3g}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=TEXT,
        )
        _draw(fig)
        bbox = label.get_window_extent(renderer=renderer).expanded(1.08, 1.18)
        if any(bbox.overlaps(existing) for existing in accepted):
            label.remove()
            skipped += 1
        else:
            accepted.append(bbox)

    if skipped:
        ax.text(
            0.995,
            0.985,
            f"{skipped} value labels hidden to avoid overlap",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color=MUTED,
        )


def _save(fig: Figure, path: Path) -> None:
    _ensure_parent(path)
    fig.savefig(
        path,
        format=path.suffix.lstrip(".") or "svg",
        facecolor=BACKGROUND,
        bbox_inches="tight",
    )
    plt.close(fig)


def bar_chart(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    y_label: str,
    colors: list[str] | None = None,
    hatches: list[str | None] | None = None,
) -> None:
    """Draw a bar chart.

    ``hatches`` is an optional per-bar hatch pattern list (same length as
    ``values``).  Pass ``"////"`` for bars where training was skipped (PTQ)
    and ``None`` for normally-trained bars.  When at least one hatch is
    present a small legend entry is added to the chart.
    """
    fig_width = max(12.0, 0.72 * len(labels) + 4.0)
    fig, ax = _setup_figure(fig_width, 7.2)
    x_positions = list(range(len(values)))
    bar_colors = [
        colors[index] if colors and index < len(colors) else _palette(index)
        for index in x_positions
    ]
    effective_hatches = hatches if hatches is not None else [None] * len(values)
    bars = ax.bar(
        x_positions, values, color=bar_colors, edgecolor="white", linewidth=0.8
    )

    # Apply per-bar hatching for PTQ (no-training) conditions.
    for bar, hatch in zip(bars, effective_hatches):
        if hatch:
            bar.set_hatch(hatch)
            bar.set_edgecolor(MUTED)  # hatch lines are drawn in the edge colour

    ax.set_title(title, fontsize=18, color=TEXT, pad=18)
    ax.set_ylabel(y_label, color=TEXT)
    ax.set_xticks(x_positions, [_wrap_label(label, width=13) for label in labels])
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.75)
    ax.margins(x=0.01)

    max_value = max(values) if values else 1.0
    max_value = max(max_value, 1.0)
    ax.set_ylim(top=max_value * 1.18)
    _autosize_axis_labels(fig, ax)
    _annotate_bars_without_overlap(fig, ax, bars, values)

    # Add a legend entry when any bar has been hatched.
    if any(h is not None for h in effective_hatches):
        legend_handles = [
            Patch(
                facecolor="lightgray",
                hatch="////",
                edgecolor=MUTED,
                label="PTQ — no re-training (checkpoint quantized & evaluated)",
            )
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=False,
            labelcolor=TEXT,
            fontsize=9,
        )

    _save(fig, path)


def grouped_bar_chart(
    path: Path,
    title: str,
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    y_label: str,
) -> None:
    fig_width = max(12.0, 0.82 * len(labels) + 4.0)
    fig, ax = _setup_figure(fig_width, 7.2)
    x_positions = list(range(len(labels)))
    group_width = 0.78
    bar_width = group_width / max(1, len(series))
    all_values: list[float] = []

    for series_index, (series_name, values, color) in enumerate(series):
        offset = -group_width / 2 + bar_width / 2 + series_index * bar_width
        plotted_values = [
            values[index] if index < len(values) else 0.0 for index in x_positions
        ]
        all_values.extend(plotted_values)
        ax.bar(
            [position + offset for position in x_positions],
            plotted_values,
            width=bar_width,
            label=series_name,
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )

    ax.set_title(title, fontsize=18, color=TEXT, pad=18)
    ax.set_ylabel(y_label, color=TEXT)
    ax.set_xticks(x_positions, [_wrap_label(label, width=12) for label in labels])
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.75)
    ax.legend(
        loc="upper left",
        frameon=False,
        labelcolor=TEXT,
        ncols=min(3, max(1, len(series))),
    )
    ax.set_ylim(top=max(all_values or [1.0]) * 1.18)
    _autosize_axis_labels(fig, ax)
    _save(fig, path)


def heatmap(
    path: Path,
    title: str,
    row_labels: list[str],
    col_labels: list[str],
    matrix: list[list[float]],
    subtitle: str | None = None,
) -> None:
    rows = len(row_labels)
    cols = len(col_labels)
    fig_width = max(12.0, 1.02 * cols + 4.0)
    fig_height = max(6.6, 0.52 * rows + 3.2)
    fig, ax = _setup_figure(fig_width, fig_height)
    values = matrix or [[0.0]]
    image = ax.imshow(values, aspect="auto", cmap="RdYlBu_r")

    ax.set_title(
        title if subtitle is None else f"{title}\n{subtitle}",
        fontsize=18,
        color=TEXT,
        pad=18,
    )
    ax.set_xticks(range(cols), [_wrap_label(label, width=10) for label in col_labels])
    ax.set_yticks(range(rows), row_labels)
    ax.tick_params(axis="x", labeltop=True, labelbottom=False, colors=TEXT, pad=4)
    ax.tick_params(axis="y", colors=TEXT)
    ax.set_xticks([index - 0.5 for index in range(1, cols)], minor=True)
    ax.set_yticks([index - 0.5 for index in range(1, rows)], minor=True)
    ax.grid(which="minor", color=BACKGROUND, linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    cell_values = [value for row in matrix for value in row]
    span = (max(cell_values) - min(cell_values)) if cell_values else 0
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            color = (
                "white"
                if span and value < (min(cell_values) + span * 0.42)
                else "#111827"
            )
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )

    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    colorbar.ax.tick_params(colors=TEXT)
    _autosize_axis_labels(fig, ax)
    _save(fig, path)


def winner_heatmap(
    path: Path,
    title: str,
    row_labels: list[str],
    col_labels: list[str],
    winner_matrix: list[list[int]],
    score_matrix: list[list[float]],
    subtitle: str | None = None,
) -> None:
    """Categorical heatmap: 0 = base wins (blue), 1 = dendrites wins (green)."""
    rows = len(row_labels)
    cols = len(col_labels)
    fig_width = max(12.0, 1.02 * cols + 4.0)
    fig_height = max(6.6, 0.52 * rows + 3.2)
    fig, ax = _setup_figure(fig_width, fig_height)

    import numpy as np
    from matplotlib.colors import ListedColormap

    data = np.array(winner_matrix, dtype=float)
    cmap = ListedColormap([BASE_BLUE, DENDRITE_GREEN])
    ax.imshow(data, aspect="auto", cmap=cmap, vmin=-0.5, vmax=1.5)

    ax.set_title(
        title if subtitle is None else f"{title}\n{subtitle}",
        fontsize=18,
        color=TEXT,
        pad=18,
    )
    ax.set_xticks(range(cols), [_wrap_label(label, width=10) for label in col_labels])
    ax.set_yticks(range(rows), row_labels)
    ax.tick_params(axis="x", labeltop=True, labelbottom=False, colors=TEXT, pad=4)
    ax.tick_params(axis="y", colors=TEXT)
    ax.set_xticks([index - 0.5 for index in range(1, cols)], minor=True)
    ax.set_yticks([index - 0.5 for index in range(1, rows)], minor=True)
    ax.grid(which="minor", color=BACKGROUND, linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_index, (winner_row, score_row) in enumerate(zip(winner_matrix, score_matrix)):
        for col_index, (winner, score) in enumerate(zip(winner_row, score_row)):
            label = "Base" if winner == 0 else "Dendrites"
            ax.text(
                col_index,
                row_index,
                f"{label}\n{score:.1f}%",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white",
                fontweight="bold",
            )

    legend_patches = [
        Patch(facecolor=BASE_BLUE, label="Base"),
        Patch(facecolor=DENDRITE_GREEN, label="Dendrites"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower right",
        framealpha=0.85,
        fontsize=9,
        title="Best variant",
        title_fontsize=9,
    )
    _autosize_axis_labels(fig, ax)
    _save(fig, path)


def _place_scatter_labels(
    fig: Figure, ax: Axes, annotations: list[tuple[float, float, str]]
) -> int:
    _draw(fig)
    renderer = getattr(fig.canvas, "get_renderer")()
    accepted = [
        tick.get_window_extent(renderer=renderer)
        for tick in ax.get_xticklabels() + ax.get_yticklabels()
    ]
    accepted.extend(
        artist.get_window_extent(renderer=renderer)
        for artist in (ax.title, ax.xaxis.label, ax.yaxis.label)
        if artist.get_visible() and artist.get_text()
    )
    legend = ax.get_legend()
    if legend is not None:
        accepted.append(legend.get_window_extent(renderer=renderer))
    offsets = [(7, 7), (7, -9), (-7, 7), (-7, -9), (11, 0), (-11, 0)]
    hidden = 0

    for x_value, y_value, label in annotations:
        placed = False
        for x_offset, y_offset in offsets:
            text = ax.annotate(
                label,
                (x_value, y_value),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha="left" if x_offset >= 0 else "right",
                va="bottom" if y_offset >= 0 else "top",
                fontsize=6.5,
                color=TEXT,
            )
            _draw(fig)
            bbox = text.get_window_extent(renderer=renderer).expanded(1.08, 1.18)
            if any(bbox.overlaps(existing) for existing in accepted):
                text.remove()
                continue
            accepted.append(bbox)
            placed = True
            break
        if not placed:
            hidden += 1

    return hidden


def line_chart(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    x_values: list[int | float],
    y_values: list[float],
) -> None:
    """Create a line chart with optional dual y-axis support."""
    fig, ax = _setup_figure(10.0, 6.0)

    if not x_values or not y_values:
        ax.set_title(title, fontsize=16, color=TEXT, pad=18)
        ax.set_xlabel(x_label, color=TEXT)
        ax.set_ylabel(y_label, color=TEXT)
        _save(fig, path)
        return

    # Plot the line
    ax.plot(
        x_values,
        y_values,
        color=BASE_BLUE,
        linewidth=2.0,
        marker="o",
        markersize=4,
        markerfacecolor=BASE_BLUE,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label=y_label,
    )

    ax.set_title(title, fontsize=16, color=TEXT, pad=18)
    ax.set_xlabel(x_label, color=TEXT)
    ax.set_ylabel(y_label, color=TEXT)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.75)
    ax.tick_params(colors=TEXT)

    # Set x-axis to show integer ticks if appropriate
    if all(isinstance(x, int) or x.is_integer() for x in x_values):
        ax.set_xticks(sorted(set(x_values)))

    _save(fig, path)


def multi_line_chart(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    x_values: list[int | float],
    series: list[tuple[str, list[float], str | None]],
) -> None:
    fig, ax = _setup_figure(10.6, 6.2)

    if not x_values or not series:
        ax.set_title(title, fontsize=16, color=TEXT, pad=18)
        ax.set_xlabel(x_label, color=TEXT)
        ax.set_ylabel(y_label, color=TEXT)
        _save(fig, path)
        return

    finite_values: list[float] = []
    for index, (label, y_values, color) in enumerate(series):
        if not y_values:
            continue
        plotted_color = color or _palette(index)
        ax.plot(
            x_values[: len(y_values)],
            y_values,
            color=plotted_color,
            linewidth=2.0,
            marker="o",
            markersize=3.6,
            markerfacecolor=plotted_color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=label,
        )
        finite_values.extend(value for value in y_values if math.isfinite(value))

    ax.set_title(title, fontsize=16, color=TEXT, pad=18)
    ax.set_xlabel(x_label, color=TEXT)
    ax.set_ylabel(y_label, color=TEXT)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.75)
    ax.tick_params(colors=TEXT)
    if all(isinstance(x, int) or x.is_integer() for x in x_values):
        ax.set_xticks(sorted(set(x_values)))
    if finite_values:
        value_min = min(finite_values)
        value_max = max(finite_values)
        span = max(value_max - value_min, 1e-6)
        ax.set_ylim(value_min - 0.06 * span, value_max + 0.12 * span)
    if len(series) > 1:
        ax.legend(loc="best", frameon=False, labelcolor=TEXT)

    _save(fig, path)


def scatter(
    path: Path,
    title: str,
    points: list[dict[str, Any]],
    x_label: str = "X",
    y_label: str = "Y",
) -> None:
    fig, ax = _setup_figure(13.5, 8.5)
    if not points:
        ax.set_title(title, fontsize=18, color=TEXT, pad=18)
        ax.set_xlabel(x_label, color=TEXT)
        ax.set_ylabel(y_label, color=TEXT)
        _save(fig, path)
        return

    # "ptq" shape = post-training quantization (no gradient updates).
    circles = [point for point in points if point.get("shape", "circle") == "circle"]
    squares = [point for point in points if point.get("shape") == "square"]
    ptq_points = [point for point in points if point.get("shape") == "ptq"]
    for group, marker, label in (
        (circles, "o", "Base"),
        (squares, "s", "Dendritic"),
        (ptq_points, "x", "PTQ — no re-training"),
    ):
        if not group:
            continue
        scatter_kwargs: dict[str, Any] = dict(
            c=[point.get("color", BASE_BLUE) for point in group],
            marker=marker,
            alpha=0.88,
            label=label,
        )
        if marker == "x":
            scatter_kwargs["s"] = 70
            scatter_kwargs["linewidths"] = 1.8
        else:
            scatter_kwargs["s"] = 44
            scatter_kwargs["edgecolors"] = "white"
            scatter_kwargs["linewidths"] = 0.6
        ax.scatter(
            [float(point["x"]) for point in group],
            [float(point["y"]) for point in group],
            **scatter_kwargs,
        )

    ax.set_title(title, fontsize=18, color=TEXT, pad=18)
    ax.set_xlabel(x_label, color=TEXT)
    ax.set_ylabel(y_label, color=TEXT)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.75)
    ax.legend(loc="upper left", frameon=False, labelcolor=TEXT)
    ax.margins(x=0.06, y=0.08)

    annotations = [
        (float(point["x"]), float(point["y"]), str(point.get("label", "")))
        for point in points
        if point.get("label")
    ]
    hidden = _place_scatter_labels(fig, ax, annotations)
    if hidden:
        ax.text(
            0.995,
            0.015,
            f"{hidden} dense labels hidden after overlap detection",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color=MUTED,
        )

    _save(fig, path)
