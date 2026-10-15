#!/usr/bin/env python3
"""MOPS web app — entry point for the `/mops/` streamlit service.

Serves the interactive companions to the "120 years of Pauline scholarship"
study: how the vocabulary around references to Paul moves across two centuries
of journal articles, and the concordances behind it.

Run
---
    cd /home/jupyter-vojta/notebooks/mops/scripts
    ../mops_venv/bin/python -m streamlit run mops_jstor.py \
        --server.address localhost --server.port 8061 --server.baseUrlPath mops

Deployment
----------
Managed by the systemd unit `mops.service`, which runs exactly the command
above, behind nginx:

    location /mops/ { proxy_pass http://127.0.0.1:8061/mops/; ... }

`--server.baseUrlPath mops` is what makes the app emit its own asset URLs under
`/mops/…`; without it streamlit would serve them from the site root and the
proxy would need path rewriting. Keep the two in sync.

Data
----
Read from `data/large_files/paul_scan/` — see `mops_app/data.py` for the layout
and `prepare_jstor_app_data.py` for how `tokens.parquet` is built. All reads are
lazy and cached, so the app starts without touching the corpus.
"""
from __future__ import annotations

import sys
from pathlib import Path

# streamlit runs this file as a script, so make `mops_app` importable regardless
# of the interpreter's sys.path (the systemd unit sets WorkingDirectory to this
# directory, but a bare `streamlit run /abs/path/mops_jstor.py` does not).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="MOPS — Pauline scholarship over time",
    page_icon="📚",
    layout="wide",
)

from mops_app import data  # noqa: E402
from mops_app.pages import concordance, temporal  # noqa: E402

ABOUT = """
# MOPS

Interactive companion to the study of **120 years of Pauline scholarship** — how the
language of modern religious and theological journals talks about Paul, and how that
language changes between 1900 and today.

The app reads a concordance of every reference to Paul in the JSTOR 2026-04 delivery
of Philosophy and Religion journals: 426,917 hits in 69,709 articles, from 316 journals,
each one detected by a tiered regex scan and stored with its keyword-in-context window.

| page | what it does |
| --- | --- |
| **Temporal frequencies** | how often words occur near a Paul reference, per time bin, with the corpus subset under your control |
| **Concordance browser** | the quoted lines themselves — search by context lemma, by text, or by scan pattern |

## Reading the numbers

*Every count here is normalised, and the normalisation is on the page.* A word can
become rarer in the corpus, or the corpus can simply shrink; the "concordance volume"
bars are there so you can tell those apart.

Two properties of the underlying scan matter for any figure taken from it:

* **Tiers.** `A` = Paul explicitly identified ("apostle Paul", "Paul of Tarsus",
  "Pauline epistles"); `B` = strong ("Pauline", "Paulus", "of Paul"); `C` = bare
  "Paul" resolved by context only. Tier C is the largest and by far the noisiest —
  the corpus is full of people called Paul — so the tier filter is not decoration:
  a claim should state which tiers back it.
* **Flags.** Hits carry context flags (`excluded_name`, `citation_apparatus`,
  `place_or_building`, …). Nothing is dropped silently; the flags are filters you
  can turn on, and the counts always describe the subset you selected.
"""


def _about() -> None:
    """Landing page: what this is, how to read it, where the data comes from."""
    st.markdown(ABOUT)

    with st.expander("Where the data comes from", expanded=False):
        st.markdown(f"""
* **Corpus** — JSTOR/ITHAKA 2026-04 delivery, Philosophy ∪ Religion, 384,807 full texts,
  materialised as one `.txt` per article at `{data.FULLTEXTS}`.
* **Detection** — `scripts/paul_concordance.py`: 24 tiered regex patterns over the whole
  corpus, with an exclusion list kept as data (`paul_scan/paul_exclusions.txt`).
* **Context** — each hit keeps a ±42-character window, the abbreviation-aware sentence,
  and (for this app) the lemmatised content words within 10 tokens of the match.
* **App table** — `data/large_files/paul_scan/tokens.parquet`, rebuilt with
  `python3 scripts/prepare_jstor_app_data.py`.
        """)

    with st.expander("Caveats", expanded=False):
        st.markdown("""
* The scan detects **mentions of the name**, not mentions of the person: `Paul VI`,
  Paul Ricœur and the apostle are told apart by context and flags, not by reading.
* Only ~65% of the hits belong to journals the annotation sheet covers, so a
  discipline/confession filter also narrows the corpus.
* Lemma matching is **exact**: plotting `jew` does not include `jewish`.
        """)


def _temporal() -> None:
    temporal.render()


def _concordance() -> None:
    concordance.render()


st.navigation([
    st.Page(_about, title="About", icon="📚", default=True),
    st.Page(_temporal, title="Temporal frequencies", icon="📈"),
    st.Page(_concordance, title="Concordance browser", icon="🔍"),
]).run()
