"""streamlit-cached wrappers around the loaders in `mops_app.data`.

Streamlit caches by argument, so wrapping the loaders here (where they take no
arguments) means each parquet is read at most once per server process, and every
page shares the same frames. `mops_app.data` itself stays streamlit-free.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from . import data
from . import topics_data as td


@st.cache_data(show_spinner="Loading the concordance table …")
def tokens() -> pd.DataFrame:
    """The slim app table (`tokens.parquet`) — hit metadata + `filtered_lemmata`."""
    return data.load_tokens()


@st.cache_data(show_spinner="Loading concordance lines …")
def hits() -> pd.DataFrame:
    """The flat scan output (`hits.parquet`) — KWIC text, no NLP tokens."""
    return data.load_hits()


@st.cache_data(show_spinner="Aggregating …")
def temporal(spec: dict) -> tuple[object, int]:
    """Filter + aggregate the token table for one temporal-plot request.

    `spec` is a plain dict of hashable values, so it doubles as the cache key:

        {
          "words": [...], "bin_length": 20, "year_range": (1900, 2019),
          "metric": "per_1000",
          "filters": {"languages": ["en", "eng"], "tiers": ["A", "B"], ...},
        }

    Returns `(TemporalResult, n_filtered_hits)`.
    """
    filtered = data.filter_tokens(tokens(), **spec["filters"])
    result = data.words_over_time(
        filtered, spec["words"],
        bin_length=spec["bin_length"],
        year_range=spec["year_range"],
        metric=spec["metric"],
    )
    return result, len(filtered)


@st.cache_data(show_spinner=False)
def top_lemmata(filters: dict, n: int = 60) -> pd.Series:
    """Most frequent context lemmas for a filter set (discovery aid)."""
    return data.top_lemmata(data.filter_tokens(tokens(), **filters), n)


@st.cache_data(show_spinner="Searching concordances …")
def concordance(filters: dict, mode: str, query: str) -> pd.DataFrame:
    """Matching concordance lines with their KWIC text (see `data.search_concordance`)."""
    # Only the lemma mode needs the lemma bags; the other modes read the keys
    # alone, which keeps that path from pulling in the whole tokens table.
    tok = tokens() if mode == "lemma" else data.load_tokens(columns=data.JOIN_KEY)
    return data.search_concordance(tok, hits(), query=query, mode=mode, filters=filters)


# --- topic model (scripts/fit_paul_topics.py; see TOPICS.md) -------------------

@st.cache_data(show_spinner="Loading the topic model …")
def topic_docs(run: str = td.TOPIC_RUN) -> pd.DataFrame:
    """Per-document topics + 2-D map coordinates (`docs.parquet`)."""
    return td.load_docs(run=run)


@st.cache_data(show_spinner=False)
def topics_info(run: str = td.TOPIC_RUN) -> pd.DataFrame:
    """Topic table with the c-TF-IDF terms (`topics_info.parquet`)."""
    return td.load_topics_info(run=run)


@st.cache_data(show_spinner=False)
def topic_report(run: str = td.TOPIC_RUN) -> dict:
    """The run's provenance record (parameters, versions, counts)."""
    return td.load_report(run=run)
