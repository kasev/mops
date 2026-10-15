"""Page: temporal frequency lineplots.

Turns the notebook's "Temporal lineplots" section into an interactive page:
pick words, a bin size, a year window and a corpus subset, and see how often
each word occurs in the context of a Paul reference over time.

The page owns only the UI; all counting lives in `mops_app.data.words_over_time`,
so the numbers here and in `5_tokens-analysis.ipynb` come from the same logic
(the port is checked against the notebook's recorded output — see
`scripts/mops_app/README.md`).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import cache, data
from ..plots import plot_timebins, plotly_timebins

#: Ready-made word groups. These mirror the vocabulary the study is about
#: (christology, justification, Jewish/Judaism, and the Pauline adjective),
#: and each one lists its morphological variants explicitly because matching
#: is exact — `jew` does not match `jewish`.
PRESETS = {
    "Jewish / Judaism": ["jewish", "judaism", "jew"],
    "Justification / righteousness": ["justification", "righteousness", "faith", "grace"],
    "Christology": ["christ", "christology", "jesus"],
    "Pauline vocabulary": ["pauline", "apostle", "epistle", "epistles"],
    "Early Christianity": ["hellenistic", "gentile", "gentiles", "mission"],
}

DEFAULT_WORDS = PRESETS["Jewish / Judaism"]


def _apply_preset() -> None:
    """Copy the chosen preset into the words box (runs before the rerun)."""
    preset = st.session_state.get("temporal_preset")
    if preset:
        st.session_state["temporal_words"] = "\n".join(PRESETS[preset])


def _sidebar(tokens: pd.DataFrame) -> dict:
    """Draw the controls and return the request spec (also the cache key)."""
    with st.sidebar:
        st.header("Words")

        if "temporal_words" not in st.session_state:
            st.session_state["temporal_words"] = "\n".join(DEFAULT_WORDS)
        st.selectbox("Preset groups", ["—"] + list(PRESETS),
                     key="temporal_preset", on_change=_apply_preset,
                     help="Loads a ready-made group into the box below.")
        raw = st.text_area("One lemma per line", key="temporal_words", height=130,
                           help="Exact match, lowercased. Add variants yourself "
                                "(jew / jewish / judaism are three different entries).")
        words = [w.strip().lower() for w in raw.splitlines() if w.strip()]

        st.divider()
        st.header("Time")
        y_min, y_max = int(tokens["year"].min()), int(tokens["year"].max())
        year_range = st.slider("Year range", y_min, y_max, data.DEFAULT_YEAR_RANGE, step=1)
        bin_length = st.slider("Bin length (years)", 1, 50, data.DEFAULT_BIN_LENGTH, step=1,
                               help="20 = bidecades (the notebook's choice), 10 = decades.")

        st.divider()
        st.header("Values")
        metric = st.selectbox(
            "Metric", list(data.METRICS), index=list(data.METRICS).index(data.DEFAULT_METRIC),
            format_func=lambda m: f"{data.METRICS[m]}  ({m})")
        log_y = st.toggle("Logarithmic y-axis", value=False,
                          help="Useful when one word is an order of magnitude rarer.")
        show_volume = st.toggle("Show concordance volume", value=True,
                                help="Faint bars: how many Paul references the bin contains.")

        st.divider()
        renderer = st.radio("Renderer", ["Interactive (plotly)", "Static (matplotlib)"],
                            horizontal=False)

    return {"words": words, "bin_length": bin_length, "year_range": tuple(year_range),
            "metric": metric, "renderer": renderer, "log_y": log_y,
            "show_volume": show_volume}


def _filters(tokens: pd.DataFrame, year_range: tuple[int, int]) -> dict:
    """Corpus filters, collapsed into an expander on the main area."""
    with st.expander("Corpus filters", expanded=False):
        tiers = st.multiselect(
            "Tiers", options=list(data.TIERS), default=list(data.TIERS),
            format_func=lambda t: data.TIERS[t],
            help="A = explicitly identified Paul, B = Pauline/Paulus/'of Paul', "
                 "C = bare 'Paul' resolved by context (noisy).")
        languages = st.multiselect(
            "Languages", options=data.available(tokens, "language"),
            default=list(data.ENGLISH_CODES),
            help="JSTOR codes. English is both 'en' and 'eng'.")

        col1, col2 = st.columns(2)
        with col1:
            disciplines = st.multiselect("Discipline (journal)", data.available(tokens, "journal-discipline"),
                                        help="Selecting any value drops journals the "
                                             "annotation sheet does not cover.")
            drop_patterns = st.multiselect("Drop patterns", data.available(tokens, "pattern"),
                                          help="Remove systematically noisy regex patterns.")
        with col2:
            confessions = st.multiselect("Confession (journal)", data.available(tokens, "journal-confession"))
            exclude_flags = st.multiselect(
                "Exclude flagged hits", data.flag_counts(tokens).index.tolist()[:25],
                help="Context flags from the scan, e.g. 'excluded_name' (a different "
                     "Paul) or 'citation_apparatus' (like 'Paul' in a bibliography).")

        journals = st.multiselect("Journals", data.available(tokens, "journal"))
        drop_empty = st.checkbox("Drop hits with no content words in the window", value=False)

    return {"tiers": tiers, "languages": languages, "disciplines": disciplines,
            "confessions": confessions, "journals": journals,
            "drop_patterns": drop_patterns, "exclude_flags": exclude_flags,
            "year_range": tuple(year_range), "drop_empty": drop_empty}


def render() -> None:
    """Draw the temporal frequencies page."""
    st.title("Temporal frequencies")
    st.caption(
        "How often do words occur in the context of a Paul reference, per time bin? "
        "Values are normalised by the number of concordances in the bin, so a growing "
        "corpus does not by itself move the line. Filtering by tier and by the "
        "annotation of the journal is what makes the corpus comparable over time."
    )

    tokens = cache.tokens()
    ui = _sidebar(tokens)
    filters = _filters(tokens, ui["year_range"])

    if not ui["words"]:
        st.info("Enter at least one lemma in the sidebar.")
        return

    result, n_hits = cache.temporal({
        "words": ui["words"], "bin_length": ui["bin_length"],
        "year_range": ui["year_range"], "metric": ui["metric"], "filters": filters,
    })

    if n_hits == 0:
        st.warning("No concordances match these filters — widen the year range, the tiers "
                   "or the language selection.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Concordances", f"{n_hits:,}")
    c2.metric("Bins", f"{len(result.bins)} × {ui['bin_length']}y")
    c3.metric("Words found", f"{len(ui['words']) - len(result.missing)} / {len(ui['words'])}")

    if result.missing:
        st.warning("Never found in this subset, so drawn as a flat zero line: "
                   + ", ".join(f"`{w}`" for w in result.missing))

    if filters["disciplines"] or filters["confessions"]:
        annotated = tokens["journal-discipline"].notna().mean()
        st.info(f"Annotated journals are {annotated:.0%} of the table; the "
                f"{1 - annotated:.0%} belonging to unannotated journals are excluded "
                f"while a discipline or confession filter is active.")

    if ui["renderer"].startswith("Interactive"):
        st.plotly_chart(
            plotly_timebins(result, log_y=ui["log_y"], show_bin_sizes=ui["show_volume"],
                            title=f"{len(ui['words'])} words in the context of Paul"),
            use_container_width=True)
    else:
        fig, _ = plot_timebins(result, log_y=ui["log_y"], show_bin_sizes=ui["show_volume"],
                               figsize=(11, 6))
        st.pyplot(fig, use_container_width=True)

    with st.expander("Values", expanded=False):
        tabs = st.tabs(["metric", "counts", "concordances per bin"])
        with tabs[0]:
            st.dataframe(result.matrix.round(5), use_container_width=True)
        with tabs[1]:
            st.dataframe(result.counts, use_container_width=True)
        with tabs[2]:
            st.dataframe(result.bin_sizes.rename("concordances").to_frame().T,
                         use_container_width=True)
        st.download_button(
            "Download (CSV)",
            data=result.matrix.to_csv().encode("utf-8"),
            file_name=f"mops_temporal_{ui['metric']}_{ui['year_range'][0]}-{ui['year_range'][1]}.csv",
            mime="text/csv")

    with st.expander("What occurs around Paul in this selection?", expanded=False):
        st.caption("Most frequent context lemmas in the filtered subset — a quick way to "
                   "find words worth plotting.")
        st.dataframe(cache.top_lemmata(filters).rename("occurrences").to_frame(),
                     use_container_width=True)
