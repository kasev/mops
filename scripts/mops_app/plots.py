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
