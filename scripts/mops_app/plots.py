#!/usr/bin/env python3
"""Figure builders for the MOPS app — matplotlib and plotly, same semantics.

`plot_timebins` is the notebook function from `5_tokens-analysis.ipynb`,
ported with three fixes (documented in its docstring). `plotly_timebins`
is its interactive twin: it draws the same series, plus the hits-per-bin
volume on a secondary axis, which is what makes a drop in a line readable
("is the word rarer, or is the corpus smaller?").

Both take a `TemporalResult` (see `mops_app.data`), so the app never has to
reshape anything to switch renderer.
"""
from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .data import METRICS, TemporalResult

#: Colour cycle shared by both renderers so a toggled figure keeps its colours.
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
           "#8c564b", "#17becf", "#e377c2", "#7f7f7f", "#bcbd22"]


def _rgba(hex_color: str, alpha: float) -> str:
    """`#1f77b4` + alpha -> `rgba(31,119,180,a)`, for plotly fill colours."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _format_value(value: float, metric: str) -> str:
    """Human-readable tick/hover formatting that respects the metric's scale."""
    if metric in ("count",):
        return f"{value:,.0f}"
    if metric == "share":
        return f"{value:.2%}"
    return f"{value:,.1f}"


# =============================================================================
# MATPLOTLIB (notebook port)
# =============================================================================

def plot_timebins(result: TemporalResult, ax=None, *, x: str = "center",
                  log_y: bool = False, colors=None, figsize=(10, 6),
                  show_bin_sizes: bool = False, title: str | None = None,
                  grid: bool = True, fontsize: int = 10, **plot_kwargs):
    """Line plot over time bins: one line per word, one point per bin.

    Ported from `5_tokens-analysis.ipynb`. Three fixes against the notebook
    version:

    1. it consumes a `TemporalResult` instead of the notebook's
       `timebins_freq_data` list of dicts, so it shares its input with the
       plotly twin (the notebook's `metric` callable/`(word, value)`-pairs
       handling is therefore gone — normalisation now happens once, in
       `words_over_time`);
    2. `x` accepts only the three positions that make sense here
       ("center", "start", "end") — the notebook's "label" mode duplicated
       `bin_labels` and produced a categorical axis that the app does not use;
    3. it can overlay the hits-per-bin volume on a twin axis
       (`show_bin_sizes`), which the notebook did not have.

    Parameters
    ----------
    result : TemporalResult
    ax : matplotlib Axes, optional
    x : {"center", "start", "end"}
        X position of a bin: its midpoint year (default), first or last year.
    log_y : bool
        Logarithmic y-axis; non-positive values are skipped, not plotted.
    colors : dict[word, color] | list[color] | None
    show_bin_sizes : bool
        Faint bars on a secondary axis with the number of hits per bin.
    plot_kwargs
        Forwarded to `ax.plot` for every line (e.g. `linewidth=2`).

    Returns
    -------
    (pd.DataFrame, matplotlib Axes)
        The plotted values (index = bin labels, columns = words).
    """
    if x not in ("center", "start", "end"):
        raise ValueError("x must be 'center', 'start' or 'end'")
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    words = result.words
    xpos = ([c for c in result.centres] if x == "center"
            else [b[0] for b in result.bins] if x == "start"
            else [b[1] for b in result.bins])

    if show_bin_sizes:
        ax2 = ax.twinx()
        ax2.bar(xpos, result.bin_sizes.values, width=(result.bins[0][1] - result.bins[0][0]) * 0.6,
                color="lightgrey", alpha=0.45, zorder=0, label="concordances")
        ax2.set_ylabel("concordances per bin", fontsize=fontsize, color="grey")
        ax2.tick_params(labelsize=fontsize, colors="grey")
        ax2.set_zorder(ax.get_zorder() + 1)
        ax2.patch.set_visible(False)

    for i, word in enumerate(words):
        y = np.asarray(result.matrix[word], dtype=float)
        if log_y:
            y = np.where(y > 0, y, np.nan)          # log(0) is undefined
        line_kw = dict(marker="o", linestyle="-", label=word, zorder=3)
        if isinstance(colors, dict):
            line_kw["color"] = colors.get(word, PALETTE[i % len(PALETTE)])
        elif colors is not None:
            line_kw["color"] = colors[i % len(colors)]
        else:
            line_kw["color"] = PALETTE[i % len(PALETTE)]
        line_kw.update(plot_kwargs)
        ax.plot(xpos, y, **line_kw)

    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("year")
    ax.set_ylabel(METRICS.get(result.metric, result.metric))
    ax.set_title(title or f"{len(words)} words over time "
                          f"({result.bins[0][0]}\u2013{result.bins[-1][1]}, "
                          f"{result.bins[0][1] - result.bins[0][0] + 1}-year bins)")
    if len(words) > 1:
        ax.legend(fontsize=fontsize)
    ax.grid(grid, alpha=0.3)
    ax.tick_params(labelsize=fontsize)
    fig = ax.get_figure()
    fig.tight_layout()
    return result.matrix, ax


# =============================================================================
# PLOTLY (interactive default)
# =============================================================================

def plotly_timebins(result: TemporalResult, *, log_y: bool = False,
                    show_bin_sizes: bool = True, height: int = 560,
                    title: str | None = None) -> go.Figure:
    """Interactive twin of `plot_timebins`.

    Same series and colours, plus: hover with the exact value *and* the raw
    count/share for that bin, the hits-per-bin volume as faint bars on a second
    axis, and a shaded band under the line so overlapping series stay legible.
    """
    fig = go.Figure()

    # faint volume bars first, so the lines draw on top
    if show_bin_sizes:
        fig.add_trace(go.Bar(
            x=list(result.centres),
            y=result.bin_sizes.values,
            name="concordances",
            yaxis="y2",
            marker_color="rgba(150,150,150,0.35)",
            hovertemplate="bin centre %{x:.0f}<br>%{y:,} concordances<extra></extra>",
        ))

    for i, word in enumerate(result.words):
        values = result.matrix[word]
        counts = result.counts[word]
        color = PALETTE[i % len(PALETTE)]
        customdata = np.stack([counts.values, result.bin_sizes.values], axis=-1)
        fig.add_trace(go.Scatter(
            x=list(result.centres),
            y=values.values,
            name=word,
            mode="lines+markers",
            line=dict(width=2, color=color),
            marker=dict(size=7),
            fill="tozeroy",
            fillcolor=_rgba(color, 0.06),
            customdata=customdata,
            hovertemplate=(f"<b>{word}</b><br>bin centre %{{x:.0f}}<br>"
                           f"{METRICS.get(result.metric, result.metric)}: "
                           f"%{{y:.4g}}<br>occurrences: %{{customdata[0]:,.0f}}"
                           f"<br>in %{{customdata[1]:,.0f}} concordances<extra></extra>"),
        ))

    fig.update_layout(
        title=title or "Temporal frequencies",
        height=height,
        hovermode="x unified",
        xaxis=dict(title="year", showspikes=True, spikemode="across"),
        yaxis=dict(title=METRICS.get(result.metric, result.metric), type="log" if log_y else "linear"),
        yaxis2=dict(title="concordances", overlaying="y", side="right",
                    showgrid=False, range=[0, float(result.bin_sizes.max()) * 3 or 1],
                    showticklabels=show_bin_sizes),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0),
        margin=dict(l=60, r=60, t=60, b=60),
        plot_bgcolor="white",
    )
    return fig


def plotly_topic_trends(matrix: pd.DataFrame, labels: dict[int, str], *,
                        metric_label: str, colors: dict[str, str] | None = None,
                        bin_docs: pd.Series | None = None, log_y: bool = False,
                        on_secondary: Sequence[int] = (),
                        secondary_axis_title: str | None = None,
                        height: int = 560, title: str | None = None) -> go.Figure:
    """Topic share over time, one line per topic, optionally over a volume backdrop.

    The topic analogue of `plotly_timebins`, and it keeps the same two devices that
    make that figure readable:

    * **the volume backdrop** — documents per bin, as faint bars on a second axis.
      For topics this matters more than for word frequencies: the corpus grows ~8x
      across the century, so a line that stays flat in share is a *rising* count, and
      a bin with 200 documents cannot support the same reading as one with 50,000.
    * **exact hover values**, including the bin's document count, so a spike can be
      checked against how much data is behind it.

    Parameters
    ----------
    matrix, labels, colors
        `matrix` is bins x topics (index = bin label, in the order they should be
        drawn); `labels` maps topic id -> rendered label (see
        `topics_data.topic_label`); `colors` maps *rendered label* -> colour, which is
        how the outlier topic is drawn grey without this module importing
        `topics_data`.
    on_secondary
        Topic ids to draw against a **right-hand axis** instead of the left one. This
        exists because of a measured problem, not for decoration: the outlier topic
        holds ~56% of the corpus while a real topic holds ~3%, so on a shared axis the
        outlier pins the top of the range and every other line is flattened into a row
        of dots along zero — the selection becomes invisible. Giving it its own axis
        lets the primary axis auto-scale to the topics that were asked for, while the
        outlier stays on screen and readable. The volume backdrop then moves to a third
        axis (unlabelled) rather than competing for the right one.

    Attention
        A secondary axis means two scales in one frame. The axis title says so, and the
        caller is expected to say so too — an unlabelled dual axis is a way to make a
        small number look like a large one.
    """
    fig = go.Figure()

    secondary_ids = [t for t in matrix.columns if int(t) in set(on_secondary)]
    # A secondary axis is only meaningful if something is left on the primary one.
    primary_ids = [t for t in matrix.columns if t not in secondary_ids]
    use_secondary = bool(secondary_ids) and bool(primary_ids)
    if not use_secondary:
        secondary_ids = []

    # Volume backdrop first, so lines draw on top. It needs its own axis whichever
    # arrangement we are in: documents per bin run to ~130,000 against shares of ~0.05.
    volume_axis = "y2"
    if use_secondary and bin_docs is not None:
        volume_axis = "y3"
    if bin_docs is not None:
        fig.add_trace(go.Bar(
            x=[str(b) for b in matrix.index],
            y=[float(bin_docs.get(b, 0)) for b in matrix.index],
            name="documents in bin",
            yaxis=volume_axis,
            marker_color="rgba(150,150,150,0.30)",
            hovertemplate="%{x}<br>%{y:,} documents in this bin<extra></extra>",
        ))

    counts = ([float(bin_docs.get(b, 0)) for b in matrix.index]
              if bin_docs is not None else None)
    customdata = np.array([[c] for c in counts], dtype=float) if counts else None

    for i, topic in enumerate(matrix.columns):
        label = labels.get(int(topic), f"t{topic}")
        color = (colors or {}).get(label, PALETTE[i % len(PALETTE)])
        fig.add_trace(go.Scatter(
            x=[str(b) for b in matrix.index],
            y=matrix[topic].astype(float).values,
            name=label,
            mode="lines+markers",
            yaxis="y2" if topic in secondary_ids else "y",
            line=dict(width=2, color=color),
            marker=dict(size=7),
            fill="tozeroy",
            fillcolor=_rgba(color, 0.05),
            customdata=customdata,
            hovertemplate=(f"<b>{label}</b><br>%{{x}}<br>{metric_label}: %{{y:.4g}}"
                           + ("<br>of %{customdata[0]:,.0f} documents in bin"
                              if customdata is not None else "")
                           + "<extra></extra>"),
        ))

    layout: dict = dict(
        title=title or "Topic share over time",
        height=height,
        hovermode="x unified",
        xaxis=dict(title="", showspikes=True, spikemode="across", categoryorder="array",
                   categoryarray=[str(b) for b in matrix.index]),
        yaxis=dict(title=metric_label, type="log" if log_y else "linear"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0),
        margin=dict(l=60, r=60, t=60, b=60),
        plot_bgcolor="white",
    )
    if use_secondary:
        # Right axis: the separated topics. Labelled, because it is a different scale.
        names = ", ".join(labels.get(int(t), f"t{t}") for t in secondary_ids)
        layout["yaxis2"] = dict(
            title=secondary_axis_title or f"{metric_label} — {names}",
            overlaying="y", side="right", showgrid=False,
            type="log" if log_y else "linear")
        # Third axis: the backdrop, deliberately untitled and unlabelled — it is a
        # visual density cue, and its numbers are in the hover.
        layout["yaxis3"] = dict(
            overlaying="y", side="right", showgrid=False, showticklabels=False,
            range=[0, float(bin_docs.max()) * 3] if bin_docs is not None and len(bin_docs)
            else [0, 1])
    else:
        layout["yaxis2"] = dict(
            title="documents in bin", overlaying="y", side="right", showgrid=False,
            range=[0, float(bin_docs.max()) * 3] if bin_docs is not None and len(bin_docs)
            else [0, 1],
            showticklabels=bin_docs is not None)
    fig.update_layout(**layout)
    return fig
