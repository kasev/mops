"""Page: concordance (KWIC) browser.

The quantitative pages count occurrences; this one reads them. Given a word it
lists the collisions actually quoted — each with its window and, on demand, its
sentence and the surrounding article text.

Three search modes (see `mops_app.data.SEARCH_MODES`) because they serve
different jobs: `lemma` follows the temporal page (same words, read instead of
counted), `kwic` is a free-text search for phrases, and `pattern` audits what a
scan regex actually matched — the workflow behind `paul_scan/paul_eval.tsv`.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import cache, data

#: Quick-start queries per mode.
EXAMPLE_QUERIES = {
    "lemma": ["justification", "hellenistic", "gentile", "righteousness", "apostle"],
    "kwic": ["apostle Paul", "Paul of Tarsus", "Pauline epistles", "writes to the Romans"],
    "pattern": ["A6_saint_paul", "A4_tarsus", "B1_pauline_adj"],
}


def _filters(tokens: pd.DataFrame) -> dict:
    """Facet controls; empty means "no restriction" for every field."""
    with st.expander("Corpus filters", expanded=False):
        y_min, y_max = int(tokens["year"].min()), int(tokens["year"].max())
        year_range = st.slider("Year range", y_min, y_max, data.DEFAULT_YEAR_RANGE, step=1)
        tiers = st.multiselect("Tiers", options=list(data.TIERS), default=list(data.TIERS),
                               format_func=lambda t: data.TIERS[t])
        languages = st.multiselect("Languages", options=data.available(tokens, "language"),
                                   default=list(data.ENGLISH_CODES))
        col1, col2 = st.columns(2)
        with col1:
            disciplines = st.multiselect("Discipline (journal)",
                                         data.available(tokens, "journal-discipline"))
            exclude_flags = st.multiselect(
                "Exclude flagged hits", data.flag_counts(tokens).index.tolist()[:25],
                help="e.g. 'excluded_name' removes the non-apostle Pauls, "
                     "'citation_apparatus' removes bibliography lines.")
        with col2:
            confessions = st.multiselect("Confession (journal)",
                                         data.available(tokens, "journal-confession"))
            journals = st.multiselect("Journals", data.available(tokens, "journal"))
    return {"year_range": tuple(year_range), "tiers": tiers, "languages": languages,
            "disciplines": disciplines, "confessions": confessions,
            "journals": journals, "exclude_flags": exclude_flags}


def _details(row: pd.Series) -> None:
    """Full sentence, scan metadata and the article's own wording for one hit."""
    st.markdown("#### Selected hit")
    st.markdown(f"**{data.kwic_text(row)}**")
    st.caption(f"{row['year']} · {row['journal']} · `{row['pattern']}` "
               f"· tier {row['tier']} · {row['language']}"
               f"{' · flags: ' + row['flags'] if pd.notna(row['flags']) else ''}")

    if pd.notna(row.get("sentence")):
        st.markdown(f"> {str(row['sentence']).strip()}")

    snippet = data.read_fulltext_context(row["item_id"], int(row["char_start"]))
    if snippet:
        st.caption("Original text around this offset:")
        st.markdown(snippet)
    else:
        st.caption(f"No full text on disk for `{row['item_id']}`.")
    st.caption(f"item_id `{row['item_id']}` · char offset {int(row['char_start'])}")


def render() -> None:
    """Draw the concordance page."""
    st.title("Concordance browser")
    st.caption(
        "Read the collisions instead of counting them. Pick a mode, type a term, and the "
        "page lists every matching concordance line with its window. Select a row to see the "
        "full sentence and the article's own wording around the match."
    )

    tokens = cache.tokens()

    with st.form("concordance_search"):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            mode = st.radio("Search by", list(data.SEARCH_MODES), horizontal=True,
                            format_func=lambda m: m,
                            help=" · ".join(f"{m}: {d}" for m, d in data.SEARCH_MODES.items()))
        with col2:
            query = st.text_input("Query", value=EXAMPLE_QUERIES["lemma"][0])
        with col3:
            limit = st.selectbox("Show at most", [100, 500, 2_000, 10_000], index=1)

        st.caption("Examples: " + " · ".join(f"`{q}`" for q in EXAMPLE_QUERIES.get(mode, [])))
        filters = _filters(tokens)
        submitted = st.form_submit_button("Search", type="primary")

    if not query.strip():
        st.info("Type a term to search for.")
        return

    result = cache.concordance(filters, mode, query)
    if submitted:
        st.toast(f"{len(result):,} concordance lines", icon="🔍")

    if result.empty:
        st.warning(f"No concordance line matches `{query}` in this subset.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Concordance lines", f"{len(result):,}")
    c2.metric("Distinct articles", f"{result['item_id'].nunique():,}")
    c3.metric("Years", f"{int(result['year'].min())}–{int(result['year'].max())}")

    with st.expander("Distribution over time", expanded=False):
        st.bar_chart(result.groupby("year").size().rename("hits"), height=200)

    shown = result.head(limit)
    table = shown.assign(KWIC=[
        data.kwic_text(row) for _, row in shown.iterrows()
    ])[["year", "journal", "language", "tier", "KWIC"]]

    event = st.dataframe(
        table, use_container_width=True, height=460, hide_index=True,
        on_select="rerun", selection_mode="single-row",
        column_config={
            "KWIC": st.column_config.TextColumn("KWIC (⟦…⟧ = the match)", width="large"),
            "year": st.column_config.NumberColumn(width="small"),
            "tier": st.column_config.TextColumn(width="small"),
        })

    if event.selection.rows:
        st.divider()
        _details(shown.iloc[event.selection.rows[0]])

    st.download_button(
        "Download all matches (CSV)",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name=f"mops_concordance_{mode}_{query.replace(' ', '_')}.csv",
        mime="text/csv")
