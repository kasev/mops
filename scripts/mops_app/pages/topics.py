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
        include_outliers = st.toggle(
            "Include the outlier topic", value=True,
            help="Topic -1 (HDBSCAN could not cluster the document). Off = composition of "
                 "the modelled part only.")

        with st.expander("More filters", expanded=False):
            disciplines = st.multiselect("Discipline (journal)",
                                         sorted(docs["journal-discipline"].dropna().unique()))
            confessions = st.multiselect("Confession (journal)",
                                         sorted(docs["journal-confession"].dropna().unique()))
            journals = st.multiselect("Journals", sorted(docs["journal"].dropna().unique()))

    return {"run": run, "year_range": tuple(year_range), "languages": languages,
            "tiers": tiers, "disciplines": disciplines, "confessions": confessions,
            "journals": journals, "include_outliers": include_outliers}


def _trend(docs: pd.DataFrame, info: pd.DataFrame, metric: str, top_n: int) -> None:
    """Topic frequency over time, as a share of each bin (D12)."""
    matrix = td.topic_trends(docs, bin_length=data.DEFAULT_BIN_LENGTH, metric=metric)
    if matrix.empty:
        st.info("No documents in this subset.")
        return

    # Draw the largest topics by size; 90 lines are not readable.
    sizes = docs["topic"].value_counts()
    keep = [t for t in sizes.index[:top_n] if t in matrix.columns]
    plot = matrix[keep]
    long = (plot.rename_axis("bin").reset_index()
            .melt(id_vars="bin", var_name="topic", value_name="value"))
    long["topic"] = long["topic"].map(lambda t: td.topic_label(int(t), info, width=34))

    fig = px.line(long, x="bin", y="value", color="topic", markers=True,
                  labels={"bin": "", "value": td.TOPIC_METRICS[metric], "topic": ""})
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.45))
    st.plotly_chart(fig, use_container_width=True)


def _map(docs: pd.DataFrame, info: pd.DataFrame, top_n: int) -> None:
    """The 2-D document map, if the fit produced coordinates."""
    if not {"x", "y"} <= set(docs.columns):
        st.info("This run has no 2-D map — it was fitted without the app projection.")
        return
    sizes = docs["topic"].value_counts()
    keep = set(sizes.index[:top_n]) | {td.OUTLIER_TOPIC}
    sample = td.map_sample(docs[docs["topic"].isin(keep)])
    sample = sample.assign(label=sample["topic"].map(
        lambda t: td.topic_label(int(t), info, width=30)))
    fig = px.scatter(sample, x="x", y="y", color="label", opacity=0.35,
                     hover_data=["year"], labels={"x": "", "y": "", "label": ""})
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

    spec = {k: v for k, v in ui.items() if k != "run"}
    subset = td.filter_docs(docs, **spec)
    info = cache.topics_info(ui["run"])

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
                   "`Count` column describes the unfiltered fit and is not what is shown.")
        table = td.topic_sizes(subset, info)
        table = table[table["Topic"].isin(
            [td.OUTLIER_TOPIC] + list(table.loc[table["Topic"] != td.OUTLIER_TOPIC,
                                                "Topic"].head(40)))]
        table["terms"] = table["Topic"].map(lambda t: td.topic_label(int(t), info, width=100))
        st.dataframe(table[["terms", "n_docs", "share", "share_modelled"]]
                     .rename(columns={"n_docs": "documents",
                                      "share": "share of subset",
                                      "share_modelled": "share of clustered"}),
                     use_container_width=True, hide_index=True)

    with tab_time:
        col1, col2 = st.columns([3, 1])
        with col2:
            metric = st.selectbox("Metric", list(td.TOPIC_METRICS),
                                  index=list(td.TOPIC_METRICS).index(td.DEFAULT_TOPIC_METRIC),
                                  format_func=lambda m: td.TOPIC_METRICS[m])
            top_n = st.slider("Topics shown", 3, 20, 8)
        st.caption("Values are normalised within each bin, so a corpus that grows does not "
                   "move the lines by itself. `share` includes the outlier topic in the "
                   "denominator, which is what makes it comparable across filters.")
        _trend(subset, info, metric, top_n)

    with tab_map:
        top_n = st.slider("Topics coloured", 3, 25, 12, key="map_top_n")
        _map(subset, info, top_n)

    with tab_class:
        col1, col2 = st.columns([3, 1])
        with col2:
            class_type = st.selectbox(
                "Grouping", list(td.TOPIC_CLASS_TYPES),
                format_func=lambda c: td.TOPIC_CLASS_TYPES[c])
            top_n = st.slider("Topics shown", 3, 20, 8, key="class_top_n")
        long = td.topic_by_class(subset, class_type)
        sizes = subset["topic"].value_counts()
        keep = list(sizes.index[:top_n])
        long = long[long["topic"].isin(keep)]
        long["topic"] = long["topic"].map(lambda t: td.topic_label(int(t), info, width=30))
        fig = px.bar(long, x="share_of_class", y="class_value", color="topic",
                     orientation="h", barmode="stack",
                     labels={"share_of_class": "share of the group", "class_value": "",
                             "topic": ""})
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
