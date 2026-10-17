"""Page: topic model — what is said near a mention of Paul.

The page reads the artifacts written by `scripts/fit_paul_topics.py` (design record:
`TOPICS.md` at the repo root — read it before quoting any topic number). It never
fits anything (`TOPICS.md` D11).

Two things it does that the other pages do not, and both are deliberate:

* **It aggregates from `docs.parquet`, not from the precomputed tables.** The
  precomputed `topics_over_time.parquet` / `topics_per_class.parquet` describe the
  *unfiltered* fit; a user's discipline/language/tier filters have to be applied
  before anything is counted, so the page recomputes. `check_mops_app.py` verifies
  that with no filters the two agree.
* **It states the outlier share prominently and never hides it.** A third of the
  documents sit in `Topic -1`, which is the binding constraint on what the model can
  claim (`TOPICS.md` §7). A page that buried that would invite exactly the overclaim
  the record forbids.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from .. import cache, data
from .. import plots as pplots
from .. import topics_data as td

NO_MODEL = """
### No topic model on disk

The topic model is fitted by a script, not by the app:

```bash
cd scripts
../mops_venv/bin/python fit_paul_topics.py        # ~11 min on the RTX A4000
```

It needs the topic stack in `requirements-topics.txt`. The design, the frozen
parameters and the limitations are in **`TOPICS.md`** — read that before quoting any
topic number.
"""


def _sidebar(docs: pd.DataFrame) -> dict:
    """Filters, as a cache key and a `filter_docs` call in one dict."""
    with st.sidebar:
        st.header("Topic model")
        run = st.selectbox("Fitted run", td.available_runs(), index=0,
                           help="`full` is the production fit (all documents). A sample run "
                                "is useful for comparison but is not the published model.")

        st.divider()
        st.header("Corpus subset")
        y_min, y_max = int(docs["year"].min()), int(docs["year"].max())
        year_range = st.slider("Year range", y_min, y_max, (y_min, y_max), step=1)
        languages = st.multiselect("Languages", sorted(docs["language"].dropna().unique()),
                                   default=sorted(docs["language"].dropna().unique()),
                                   help="JSTOR codes. English is both 'en' and 'eng'.")
        tiers = st.multiselect("Tiers", sorted(docs["tier"].dropna().unique()),
                               default=sorted(docs["tier"].dropna().unique()),
                               help="A = explicitly identified Paul (the trustworthy tier). "
                                    "C = bare 'Paul', which contains other people called Paul.")

        with st.expander("More filters", expanded=False):
            disciplines = st.multiselect("Discipline (journal)",
                                         sorted(docs["journal-discipline"].dropna().unique()))
            confessions = st.multiselect("Confession (journal)",
                                         sorted(docs["journal-confession"].dropna().unique()))
            journals = st.multiselect("Journals", sorted(docs["journal"].dropna().unique()))

        st.divider()
        st.header("Outlier topic (−1)")
        # This control is **display only** and deliberately cannot change a denominator.
        # Hiding outliers used to renormalise the time series as a side effect, so the same
        # topic in the same subset changed share (0.023 -> 0.060) while the axis still
        # claimed "outliers included". Excluding them from the denominator is a *different
        # question*, and it has its own entry in the metric selector (`share_modelled`),
        # where the label can say so.
        show_outlier = st.radio(
            "Show it", ["yes — grey, as unclustered", "no — hide from figures"],
            index=0, label_visibility="collapsed",
            help="Hidden means hidden, not removed: shares below still include it, so "
                 "switching this cannot move another topic's values. To see the "
                 "composition of the clustered part only, change the **metric** instead.")

    return {"run": run, "year_range": tuple(year_range), "languages": languages,
            "tiers": tiers, "disciplines": disciplines, "confessions": confessions,
            "journals": journals, "show_outlier": show_outlier.startswith("yes")}


def _colour_kwargs(labels: list[str]) -> dict:
    """plotly colour map with the outlier topic grey (see `td.topic_color_map`)."""
    return {"color_discrete_map": td.topic_color_map(labels)}


def _drop_outlier_rows(frame: pd.DataFrame, show: bool) -> pd.DataFrame:
    """Remove the outlier row from a *display* frame when the user asked to hide it.

    Never used on the documents themselves: dropping the rows before aggregation is
    what would change the shares (see the sidebar note).
    """
    if show:
        return frame
    return frame[frame.index != td.OUTLIER_TOPIC]


def _topic_picker(docs: pd.DataFrame, info: pd.DataFrame, *, columns: int = 3,
                  default: int = 5) -> list[int]:
    """Checkbox grid for choosing exactly which topics to draw.

    Checkboxes rather than a slider because temporal work is usually about following
    *specific* topics — the ones a hypothesis is about — not the largest ones. A
    "top N" control cannot express "t1, t8 and t22", which is the common case.

    Topics are ordered by size in the current subset (so the grid is stable and the
    big ones are where your eye already is) and the first `default` start ticked, so
    the control is useful before it is touched. The outlier topic is *not* in the
    grid: its visibility is the sidebar's job, so there is one source of truth for it.
    """
    sizes = docs["topic"].value_counts()
    candidates = [int(t) for t in sizes.index if t != td.OUTLIER_TOPIC]
    if not candidates:
        return []

    # Fill *down* each column, not across: with three columns this puts the largest
    # topics in the first column in size order, so scanning down it is scanning by
    # size. Filling row-wise would put every third topic in the first column (t0, t3,
    # t6 …), which reads as unordered.
    cols = st.columns(columns)
    rows_per_col = -(-len(candidates) // columns)   # ceil
    chosen: list[int] = []
    for i, topic in enumerate(candidates):
        col = cols[min(i // rows_per_col, columns - 1)]
        # A stable per-topic key: the selection therefore survives a change of metric
        # or year window, which is what you want when comparing views of one topic.
        if col.checkbox(td.topic_label(topic, info, width=26),
                        value=(i < default), key=f"topic_pick_{topic}"):
            chosen.append(topic)
    return chosen


def _trend(docs: pd.DataFrame, info: pd.DataFrame, metric: str, topics: list[int],
           show_outlier: bool, *, volume: bool = True, log_y: bool = False) -> None:
    """Topic share over time for a *given* set of topics.

    The topic list is decided by the caller (checkbox grid or "largest N"), because
    this function should not have an opinion about selection. Two things it does own:

    * the outlier topic is appended last when visible, so it always sits at the end of
      the legend and in grey, rather than being sorted among the real topics by size;
    * **the outlier is moved to its own right-hand axis when it would otherwise hide
      everything else.** It holds ~56% of the corpus against ~3% for a real topic, so
      on a shared axis it pins the top of the range and the topics a reader selected
      collapse into a flat row along zero. That is not a hypothetical: it is what the
      figure did before this was added.
    """
    matrix = td.topic_trends(docs, bin_length=data.DEFAULT_BIN_LENGTH, metric=metric)
    if matrix.empty:
        st.info("No documents in this subset.")
        return

    order = [t for t in topics if t in matrix.columns]
    if show_outlier and td.OUTLIER_TOPIC in matrix.columns:
        order = order + [td.OUTLIER_TOPIC]
    if not order:
        st.info("No topics selected — tick at least one above.")
        return

    # Documents per bin, for the volume backdrop and the per-bin table. It is the same
    # `count` metric summed over topics, so it cannot disagree with the shares.
    bin_docs = td.topic_trends(docs, bin_length=data.DEFAULT_BIN_LENGTH,
                               metric="count").sum(axis=1)

    # Would the outlier swamp the topics the reader asked for? Compare against the
    # largest *other* series, so the test is about this selection rather than about the
    # corpus in general (a reader may deliberately select only large topics).
    on_secondary: list[int] = []
    if show_outlier and td.OUTLIER_TOPIC in order and len(order) > 1:
        others = [t for t in order if t != td.OUTLIER_TOPIC]
        peak_other = float(matrix[others].to_numpy().max()) if others else 0.0
        peak_outlier = float(matrix[td.OUTLIER_TOPIC].max())
        # 3x is a deliberately blunt threshold: anything beyond it puts the small
        # series inside the bottom few percent of the axis, where their shape is lost.
        if peak_other > 0 and peak_outlier > 3 * peak_other:
            on_secondary = [td.OUTLIER_TOPIC]

    labels = {int(t): td.topic_label(int(t), info, width=34) for t in order}
    fig = pplots.plotly_topic_trends(
        matrix[order], labels, metric_label=td.TOPIC_METRICS[metric],
        colors=td.topic_color_map([labels[int(t)] for t in order]),
        bin_docs=bin_docs if volume else None, log_y=log_y,
        on_secondary=on_secondary,
        secondary_axis_title=f"outlier share — right axis ({td.TOPIC_METRICS[metric]})"
        if on_secondary else None,
        title=f"{len(order)} topics — {td.TOPIC_METRICS[metric]}")
    st.plotly_chart(fig, use_container_width=True)

    if on_secondary:
        st.caption(
            f"⚠️ The outlier topic is on the **right axis** — it is ~{peak_outlier / peak_other:.0f}× "
            f"the largest selected topic, so on a shared axis it would flatten everything "
            f"else. The two scales are not comparable; the left axis is the one your "
            f"selection is read on.")
    if not show_outlier:
        st.caption("The outlier topic is hidden from the figure; it is still in the "
                   "denominator, so these values are unchanged from the default view.")
    return matrix[order], bin_docs


def _trend_values(matrix: pd.DataFrame, bin_docs: pd.Series, info: pd.DataFrame,
                  metric: str, run: str) -> None:
    """The plotted numbers as a table, plus the per-bin document counts.

    Kept next to the chart rather than behind a toggle: a share is only readable
    against the number of documents behind it, and the earliest bins are thin.
    """
    with st.expander("Values behind the figure", expanded=False):
        table = matrix.copy()
        table.columns = [td.topic_label(int(c), info, width=60) for c in table.columns]
        st.dataframe(table.round(5), use_container_width=True)
        st.caption("Documents per bin — the denominator, and the reason a sparse early "
                   "bin should be read cautiously:")
        st.dataframe(bin_docs.rename("documents").to_frame().T, use_container_width=True)
        st.download_button(
            "Download these series (CSV)",
            data=table.to_csv().encode("utf-8"),
            file_name=f"mops_topic_trends_{run}_{metric}.csv", mime="text/csv")


def _map(docs: pd.DataFrame, info: pd.DataFrame, top_n: int, show_outlier: bool) -> None:
    """The 2-D document map, if the fit produced coordinates."""
    if not {"x", "y"} <= set(docs.columns):
        st.info("This run has no 2-D map — it was fitted without the app projection.")
        return
    sizes = docs["topic"].value_counts()
    keep = set(sizes.index[:top_n])
    if show_outlier:
        keep |= {td.OUTLIER_TOPIC}
    sample = td.map_sample(docs[docs["topic"].isin(keep)])
    sample = sample.assign(label=sample["topic"].map(
        lambda t: td.topic_label(int(t), info, width=30)))
    labels = sorted(sample["label"].unique())
    fig = px.scatter(sample, x="x", y="y", color="label", opacity=0.35,
                     hover_data=["year"], labels={"x": "", "y": "", "label": ""},
                     **_colour_kwargs(labels))
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.35),
                      height=560)
    st.plotly_chart(fig, use_container_width=True)


def render() -> None:
    """Draw the topics page."""
    st.title("Topics around Paul")
    st.caption(
        "Each document is one concordance line — the ±10-token window around a mention of "
        "Paul, lemmatised. Topics are what recurs in those windows: a description of the "
        "vocabulary near the name, not of articles or arguments. Labels are generated from "
        "the lemmas (c-TF-IDF), so they read as term lists rather than names."
    )

    if not td.is_available(td.TOPIC_RUN):
        st.warning(NO_MODEL)
        return

    ui = _sidebar(cache.topic_docs())
    docs = cache.topic_docs(ui["run"])

    # `show_outlier` is deliberately NOT passed to `filter_docs`: the documents keep
    # their outliers so every share below means the same thing whether or not the
    # topic is drawn. Only the figures drop it. (Passing it here is the bug this
    # replaced — see the sidebar comment.)
    spec = {k: v for k, v in ui.items() if k not in ("run", "show_outlier")}
    subset = td.filter_docs(docs, **spec)
    info = cache.topics_info(ui["run"])
    show_outlier = ui["show_outlier"]

    if subset.empty:
        st.warning("No documents match these filters.")
        return

    outlier_share = float((subset["topic"] == td.OUTLIER_TOPIC).mean())
    n_topics = int(subset.loc[subset["topic"] != td.OUTLIER_TOPIC, "topic"].nunique())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", f"{len(subset):,}")
    c2.metric("Topics present", f"{n_topics}")
    c3.metric("Outlier share", f"{outlier_share:.1%}")
    c4.metric("Journals", f"{subset['journal'].nunique():,}")

    # The one number that governs every reading of this page: say it out loud.
    st.info(f"**{outlier_share:.1%} of these documents are in the outlier topic.** "
            f"{td.OUTLIER_NOTE}")

    report = cache.topic_report(ui["run"])
    if ui["run"] == td.TOPIC_RUN:
        st.caption(
            f"Production fit ({report['started'][:10]}): {report['topics']['n_topics']} topics, "
            f"`min_cluster_size={report['topics']['min_topic_size']}` "
            f"({report['topics']['min_topic_size'] / report['documents']['fitted']:.3%} of "
            f"{report['documents']['fitted']:,}), "
            f"`min_samples={report['topics']['min_samples']}`, "
            f"`cluster_selection_epsilon={report['topics']['cluster_selection_epsilon']}`. "
            f"Encoder `{report['decisions']['D6_embedding_model'].split('/')[-1]}`. "
            f"See **TOPICS.md** for the decisions and the limitations."
        )

    st.divider()
    tab_sizes, tab_time, tab_map, tab_class = st.tabs(
        ["Topics", "Over time", "Map", "By journal group"])

    with tab_sizes:
        st.markdown("##### Largest topics in the current subset")
        st.caption("Document counts are recomputed for your filters; `topics_info`'s own "
                   "`Count` column describes the unfiltered fit and is not what is shown. "
                   "The two share columns are different questions: `share of subset` "
                   "includes the unclustered documents, `share of clustered` does not.")
        table = td.topic_sizes(subset, info)
        table = table[table["Topic"].isin(
            [td.OUTLIER_TOPIC] + list(table.loc[table["Topic"] != td.OUTLIER_TOPIC,
                                                "Topic"].head(40)))]
        if not show_outlier:
            table = table[table["Topic"] != td.OUTLIER_TOPIC]
        table["terms"] = table["Topic"].map(lambda t: td.topic_label(int(t), info, width=100))
        st.dataframe(table[["terms", "n_docs", "share", "share_modelled"]]
                     .rename(columns={"n_docs": "documents",
                                      "share": "share of subset",
                                      "share_modelled": "share of clustered"}),
                     use_container_width=True, hide_index=True)

    with tab_time:
        st.markdown("##### Which topics?")
        st.caption("Follow the topics a question is *about*, rather than the largest ones — "
                   "a share can rise while the counts behind it fall, so read the lines "
                   "against the volume backdrop.")

        mode = st.radio("Selection", ["Pick topics (checkboxes)", "Largest N by size"],
                        horizontal=True, label_visibility="collapsed")

        ctrl = st.columns([2, 1, 1])
        with ctrl[0]:
            metric = st.selectbox(
                "Metric", list(td.TOPIC_METRICS),
                index=list(td.TOPIC_METRICS).index(td.DEFAULT_TOPIC_METRIC),
                format_func=lambda m: td.TOPIC_METRICS[m])
        with ctrl[1]:
            volume = st.toggle("Volume backdrop", value=True,
                               help="Faint bars: documents in each bin, on the right axis. "
                                    "The corpus grows ~8x, so a flat share is a rising count.")
        with ctrl[2]:
            log_y = st.toggle("Log y-axis", value=False,
                              help="Useful when a small topic's shape is the point.")

        if mode.startswith("Pick"):
            chosen = _topic_picker(subset, info)
        else:
            top_n = st.slider("Topics shown", 1, 25, 8)
            chosen = [int(t) for t in subset["topic"].value_counts().index
                      if t != td.OUTLIER_TOPIC][:top_n]

        st.caption("Values are normalised within each bin, so a corpus that grows does not "
                   "move the lines by itself. **To see the clustered part only, change the "
                   "metric** — `share_modelled` divides by the clustered documents, which is "
                   "a different question from `share`, and the two can move in opposite "
                   "directions when the outlier share changes.")

        drawn = _trend(subset, info, metric, chosen, show_outlier,
                       volume=volume, log_y=log_y)
        if drawn:
            matrix, bin_docs = drawn
            _trend_values(matrix, bin_docs, info, metric, ui["run"])

    with tab_map:
        top_n = st.slider("Topics coloured", 3, 25, 12, key="map_top_n")
        _map(subset, info, top_n, show_outlier)

    with tab_class:
        col1, col2 = st.columns([3, 1])
        with col2:
            class_type = st.selectbox(
                "Grouping", list(td.TOPIC_CLASS_TYPES),
                format_func=lambda c: td.TOPIC_CLASS_TYPES[c])
            top_n = st.slider("Topics shown", 3, 20, 8, key="class_top_n")
        long = td.topic_by_class(subset, class_type)
        sizes = subset["topic"].value_counts()
        topics = [t for t in sizes.index if t != td.OUTLIER_TOPIC][:top_n]
        if show_outlier:
            topics = topics + [td.OUTLIER_TOPIC]
        long = long[long["topic"].isin(topics)]
        long["topic"] = long["topic"].map(lambda t: td.topic_label(int(t), info, width=30))
        labels = sorted(long["topic"].unique())
        fig = px.bar(long, x="share_of_class", y="class_value", color="topic",
                     orientation="h", barmode="stack",
                     labels={"share_of_class": "share of the group", "class_value": "",
                             "topic": ""},
                     **_colour_kwargs(labels))
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.4),
                          height=520)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("`(unannotated)` is a real group: a third of the journals carry no "
                   "discipline/confession annotation, and dropping them would make the "
                   "other groups look complete.")

    with st.expander("Topic labels as a table", expanded=False):
        terms = info.copy()
        terms["terms"] = terms["Topic"].map(
            lambda t: td.topic_label(int(t), info, width=200))
        st.dataframe(terms[["Topic", "Count", "terms"]]
                     .rename(columns={"Count": "docs (unfiltered fit)"}),
                     use_container_width=True, hide_index=True)

    st.download_button(
        "Download this subset's topics (CSV)",
        data=td.topic_sizes(subset, info).to_csv(index=False).encode("utf-8"),
        file_name=f"mops_topics_{ui['run']}.csv", mime="text/csv")
