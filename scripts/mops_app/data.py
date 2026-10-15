#!/usr/bin/env python3
"""Analysis layer of the MOPS app: loaders, filters, temporal aggregation.

Nothing in this module imports streamlit, so the same functions can be used
from a notebook (see `5_tokens-analysis.ipynb`, whose "Temporal lineplots"
section this ports) and from the running web service.

Data
----
Three parquet files in `data/large_files/paul_scan/`, all built from the Paul
concordance scan (`scripts/paul_concordance.py`):

* `hits.parquet`      — flat scan output (no NLP): year, journal, language,
  tier, pattern, flags, char offsets, `kwic_left`/`match`/`kwic_right`,
  `sentence`. Built by `3_paul-hits-explore.ipynb`.
* `hits_nlp.parquet`  — the above + `kwic_nlp` (spaCy tokens of the hit's
  sentences) + journal annotations. Built by `4_concordance-tokens-explore.ipynb`.
* `tokens.parquet`    — the slim app table: hit metadata + `filtered_lemmata`
  (the lemmatised, POS/stopword-filtered context bag). Built by
  `scripts/prepare_jstor_app_data.py`. **This is what the app loads.**

Conventions
-----------
* "hit" = one concordance line (one occurrence of a Paul pattern). "bin" = a
  time slice of `bin_length` years. "concordances" = the number of hits in the
  currently filtered subset — the denominator of every frequency metric, exactly
  as in the notebook (`n / concs_n * 20`).
* `language` values are the raw JSTOR codes, so English is *two* codes
  (`en`, `eng`) and both must always be selected together.
* The `flags` column is a **string** and `df.flags` is shadowed by pandas'
  own `DataFrame.flags`, so it must always be accessed as `df["flags"]`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# =============================================================================
# PATHS AND CONSTANTS
# =============================================================================

MOPS = Path(__file__).resolve().parents[2]          # .../notebooks/mops
SCAN_DIR = MOPS / "data" / "large_files" / "paul_scan"
TOKENS_PATH = SCAN_DIR / "tokens.parquet"
HITS_PATH = SCAN_DIR / "hits.parquet"

#: Shared JSTOR corpus (read-only; outside the repo, used for full-text snippets).
JSTOR = Path("/srv/data/jstor/jstor2026-04")
FULLTEXTS = JSTOR / "fulltexts_txts"

#: The study window. The corpus itself covers 1829–2026, but the notebook and
#: every published figure use 1900–2019, so that stays the default.
DEFAULT_YEAR_RANGE = (1900, 2019)

#: JSTOR language codes that mean "English" — both must travel together.
ENGLISH_CODES = ("en", "eng")

DEFAULT_BIN_LENGTH = 20

#: Frequency metrics. `per_20` reproduces the notebook formula exactly
#: (`count / n_concordances * 20`); `per_1000` is the readable variant
#: (`* 1000`); `share` is the share of all content-word tokens in the bin;
#: `count` is the raw number of occurrences.
METRICS = {
    "count": "raw count of occurrences",
    "per_1000": "occurrences per 1,000 concordances",
    "per_20": "occurrences per 20 concordances (notebook formula)",
    "share": "share of all content-word tokens in the bin",
}
DEFAULT_METRIC = "per_1000"

#: Human-readable gloss for the tier values assigned by `paul_concordance.py`.
TIERS = {"A": "A — explicit", "B": "B — strong", "C": "C — bare 'Paul'"}


# =============================================================================
# LOADERS
# =============================================================================

def load_tokens(columns: list[str] | None = None, path: Path | None = None) -> pd.DataFrame:
    """Load the slim app table (`tokens.parquet`).

    Parameters
    ----------
    columns : list[str], optional
        Subset of columns (the `filtered_lemmata` column dominates the read
        cost, so leaving it out makes metadata-only pages instant).
    path : Path, optional
        Override the default location (useful for tests).
    """
    path = path or TOKENS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — build it with: "
            f"python3 scripts/prepare_jstor_app_data.py")
    return pd.read_parquet(path, columns=columns, engine="pyarrow")


def load_hits(columns: list[str] | None = None, path: Path | None = None) -> pd.DataFrame:
    """Load the flat scan output (`hits.parquet`) — KWIC text, no NLP tokens."""
    path = path or HITS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — build it from 3_paul-hits-explore.ipynb")
    return pd.read_parquet(path, columns=columns, engine="pyarrow")


def read_fulltext_context(item_id: str, char_start: int, window: int = 600) -> str:
    """Return ±`window` characters of the original article text around a hit.

    The scan matches on a *length-preserving* normalised copy of the text, so
    `char_start` from the scan indexes the file on disk verbatim — the snippet
    is the article's own wording, not the normalised form.
    """
    path = FULLTEXTS / f"{item_id}.txt"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    lo = max(0, char_start - window)
    hi = min(len(text), char_start + window)
    return (("… " if lo else "") + text[lo:hi].strip().replace("\n", " ") + (" …" if hi < len(text) else ""))


# =============================================================================
# FACETS AND FILTERS
# =============================================================================

def available(tokens: pd.DataFrame, column: str) -> list[str]:
    """Sorted non-null values of a facet column (language, tier, journal...)."""
    return sorted(v for v in tokens[column].dropna().unique())


def journal_table(tokens: pd.DataFrame) -> pd.DataFrame:
    """Per-journal hit counts joined with the hand-curated annotations.

    One row per journal: number of hits, distinct articles, year span and the
    discipline/confession codes mapped in from the `journals_annotated` sheet.
    A missing annotation is `NaN` (the sheet does not cover every journal).
    """
    annotations = (tokens.drop_duplicates("journal")
                  .set_index("journal")[["journal-cat", "journal-discipline", "journal-confession"]])
    table = tokens.groupby("journal").agg(
        hits=("item_id", "size"),
        articles=("item_id", "nunique"),
        year_min=("year", "min"),
        year_max=("year", "max"),
    )
    return table.join(annotations).sort_values("hits", ascending=False)


def filter_tokens(
    tokens: pd.DataFrame,
    *,
    year_range: tuple[int, int] | None = None,
    languages: list[str] | None = None,
    tiers: list[str] | None = None,
    disciplines: list[str] | None = None,
    confessions: list[str] | None = None,
    journals: list[str] | None = None,
    drop_patterns: list[str] | None = None,
    exclude_flags: list[str] | None = None,
    drop_empty: bool = False,
) -> pd.DataFrame:
    """Apply the app's filters to the hit table.

    Every argument is optional and an empty/`None` value means "no restriction",
    so calling with no arguments returns the input unchanged. This is the one
    place the notebook's `hits_filtering` + `journaltype_filtering` pair lives;
    they are merged because they operated on the same frame and the notebook
    called one from the other with an inconsistent signature.

    Parameters
    ----------
    year_range : (start, end), inclusive.
    languages : JSTOR codes to keep (use `ENGLISH_CODES` for English).
    tiers : "A"/"B"/"C" — see `TIERS`.
    disciplines, confessions : values of `journal-discipline` /
        `journal-confession`. Selecting any value **drops** journals the sheet
        does not annotate, which is what the notebook did; leave empty to keep
        everything.
    journals : explicit journal names.
    drop_patterns : pattern ids (e.g. `A6_saint_paul`) to remove — lets a user
        remove a systematically noisy pattern.
    exclude_flags : remove hits carrying any of these context flags (e.g.
        `citation_apparatus`).
    drop_empty : remove hits with no content words in the window (they would
        otherwise be an all-zero column in the temporal view).
    """
    df = tokens
    if year_range is not None:
        df = df[df["year"].between(year_range[0], year_range[1])]
    if languages:
        df = df[df["language"].isin(languages)]
    if tiers:
        df = df[df["tier"].isin(tiers)]
    if disciplines:
        df = df[df["journal-discipline"].isin(disciplines)]
    if confessions:
        df = df[df["journal-confession"].isin(confessions)]
    if journals:
        df = df[df["journal"].isin(journals)]
    if drop_patterns:
        df = df[~df["pattern"].isin(drop_patterns)]
    if exclude_flags:
        # `flags` is a semicolon-separated list of context flags (verified on
        # the data: e.g. `capital_after;possibly_given_name;excluded_name`),
        # so a plain substring test is sufficient and matches what the
        # notebook does when it looks for `citation_apparatus` etc.
        flags = df["flags"].fillna("")
        mask = pd.Series(False, index=df.index)
        for flag in exclude_flags:
            mask |= flags.str.contains(flag, regex=False)
        df = df[~mask]
    if drop_empty:
        df = df[df["filtered_lemmata"].apply(len) > 0]
    return df


# =============================================================================
# TIME BINS
# =============================================================================

def bin_edges(year_start: int, year_end: int, bin_length: int) -> list[tuple[int, int]]:
    """Half-open-free year bins covering `year_start`..`year_end` inclusive.

    The notebook hard-codes `range(1900, 2020, 20)`; this generalises it and
    always covers the last year (e.g. 1900–2019 + 20 → 1900-1919 … 2000-2019,
    and 1900–2010 + 20 → …, 2000-2010 for the trailing partial bin).
    """
    if bin_length < 1:
        raise ValueError("bin_length must be >= 1")
    edges = []
    start = year_start
    while start <= year_end:
        edges.append((start, min(start + bin_length - 1, year_end)))
        start += bin_length
    return edges


def bin_label(edge: tuple[int, int]) -> str:
    """`(1900, 1919)` → `"1900–1919"` (en dash, as in the notebook output)."""
    return f"{edge[0]}\u2013{edge[1]}"


def _assign_bins(df: pd.DataFrame, year_start: int, year_end: int, bin_length: int) -> pd.DataFrame:
    """Add a `_bin` column (index into `bin_edges`) to rows inside the window."""
    df = df[df["year"].between(year_start, year_end)].copy()
    df["_bin"] = ((df["year"] - year_start) // bin_length).astype("int64")
    n_bins = len(bin_edges(year_start, year_end, bin_length))
    return df[df["_bin"] < n_bins]


# =============================================================================
# TEMPORAL AGGREGATION
# =============================================================================

def _non_zero(series: pd.Series) -> pd.Series:
    """Float view of a denominator with zeros turned into NaN.

    Used by `words_over_time` whenever a bin can be empty (e.g. a narrow year
    window that predates a journal's coverage, or after a discipline filter).
    Dividing by NaN yields NaN, which the caller fills with 0 — doing it this
    way avoids `replace(0, pd.NA)`, whose result is object-dtype and therefore
    raises `TypeError: float() argument must be ... not 'NAType'` on
    `astype(float)`.
    """
    values = series.astype(float)
    return values.where(values > 0)


@dataclass
class TemporalResult:
    """Everything a temporal lineplot needs, in one object.

    Attributes
    ----------
    matrix : pd.DataFrame
        index = bin label, columns = requested words, values = the chosen
        metric. This is the object to plot.
    counts : pd.DataFrame
        Same shape, raw occurrence counts (what `matrix` is derived from).
    bin_sizes : pd.Series
        Hits per bin — the denominator of the frequency metrics.
    centres : pd.Series
        Bin midpoint year, for plotting against a numeric axis.
    metric, words, bins : the inputs, echoed back for labelling.
    missing : dict[str, int]
        Words that never occurred in the filtered subset (a zero line is still
        drawn, but the page warns about them).
    """

    matrix: pd.DataFrame
    counts: pd.DataFrame
    bin_sizes: pd.Series
    centres: pd.Series
    metric: str
    words: list[str]
    bins: list[tuple[int, int]]
    missing: dict[str, int] = field(default_factory=dict)


def words_over_time(
    df: pd.DataFrame,
    words: list[str],
    *,
    bin_length: int = DEFAULT_BIN_LENGTH,
    year_range: tuple[int, int] = DEFAULT_YEAR_RANGE,
    metric: str = DEFAULT_METRIC,
    lowercase: bool = True,
) -> TemporalResult:
    """Count `words` per time bin and normalise, replacing the notebook's
    `from_words_to_freqs` / `get_count_tups_selected` pair.

    Parameters
    ----------
    df : pd.DataFrame
        A **filtered** token frame (use `filter_tokens` first) with a
        `filtered_lemmata` column.
    words : list[str]
        Lemmas to count. Matching is exact (case-insensitive by default):
        `jew` does not match `jewish` — the notebook behaves the same way, which
        is why its examples always list the morphological variants explicitly.
    bin_length : int
        Years per bin (20 → bidecades, 10 → decades).
    year_range : (int, int)
        Inclusive window that defines both the bins and the rows kept.
    metric : str
        Key of `METRICS`.
    lowercase : bool
        Lowercase the lemmas before matching (the notebook's default).

    Returns
    -------
    TemporalResult
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {sorted(METRICS)}")

    year_start, year_end = year_range
    bins = bin_edges(year_start, year_end, bin_length)
    labels = [bin_label(b) for b in bins]
    centres = pd.Series([(a + b) / 2 for a, b in bins], index=labels, name="centre")

    wanted = [w.lower() for w in words] if lowercase else list(words)
    df = _assign_bins(df, year_start, year_end, bin_length)

    # hits per bin = the notebook's `concs_n`
    bin_sizes = (df.groupby("_bin").size()
                 .reindex(range(len(bins)), fill_value=0)
                 .set_axis(labels))

    # flatten the list column once; the pre-filter frame also gives the true
    # denominator "all content words in the bin" that the `share` metric needs
    exploded = df[["_bin", "filtered_lemmata"]].explode("filtered_lemmata")
    exploded = exploded[exploded["filtered_lemmata"].notna()]
    if lowercase:
        exploded["filtered_lemmata"] = exploded["filtered_lemmata"].str.lower()
    tokens_per_bin = (exploded.groupby("_bin").size()
                      .reindex(range(len(bins)), fill_value=0)
                      .set_axis(labels))
    exploded = exploded[exploded["filtered_lemmata"].isin(wanted)]

    counts = (exploded.groupby(["_bin", "filtered_lemmata"]).size()
              .unstack(fill_value=0)
              .reindex(index=range(len(bins)), columns=wanted, fill_value=0)
              .fillna(0)
              .astype("int64")
              .set_axis(labels))

    missing = {w: int(counts[w].sum()) for w in wanted if counts[w].sum() == 0}

    counts_f = counts.astype(float)
    if metric == "count":
        matrix = counts_f
    elif metric == "share":
        matrix = counts_f.div(_non_zero(tokens_per_bin), axis=0)
    else:
        scale = 1000 if metric == "per_1000" else 20
        matrix = counts_f.mul(scale).div(_non_zero(bin_sizes), axis=0)
    matrix = matrix.fillna(0.0)

    return TemporalResult(matrix=matrix, counts=counts, bin_sizes=bin_sizes,
                          centres=centres, metric=metric, words=wanted,
                          bins=bins, missing=missing)


def top_lemmata(df: pd.DataFrame, n: int = 50) -> pd.Series:
    """Most frequent context lemmas in a filtered frame (for suggestions)."""
    exploded = df["filtered_lemmata"].explode().dropna().str.lower()
    return exploded.value_counts().head(n)


def pattern_counts(df: pd.DataFrame) -> pd.Series:
    """Hits per regex pattern id — used to spot systematically noisy patterns."""
    return df["pattern"].value_counts()


def flag_counts(df: pd.DataFrame) -> pd.Series:
    """Hits per individual context flag.

    `flags` holds a semicolon-separated string of zero or more flags per hit
    (`capital_after;excluded_name;...`), so it is split before counting. Note
    that only about a fifth of all hits carry any flag at all.
    """
    flags = df["flags"].fillna("").str.split(";").explode().str.strip()
    return flags[flags != ""].value_counts()


# =============================================================================
# CONCORDANCE SEARCH
# =============================================================================

#: The join key linking a meaning row in `tokens.parquet` to its concordance
#: line in `hits.parquet`. Both files hold the same 426,917 hits, and
#: `prepare_jstor_app_data.py` asserts this pair is unique when building.
JOIN_KEY = ["item_id", "char_start"]

#: Columns of the concordance table the pages display.
CONCORDANCE_COLUMNS = JOIN_KEY + ["year", "journal", "language", "tier", "pattern",
                                  "flags", "kwic_left", "match", "kwic_right", "sentence"]

#: Search modes for `search_concordance`.
SEARCH_MODES = {
    "lemma": "the lemma occurs within 10 words of the match",
    "kwic": "the text occurs inside the quoted window",
    "pattern": "the scan's regex pattern id fired",
}


def search_concordance(
    tokens: pd.DataFrame,
    hits: pd.DataFrame,
    *,
    query: str,
    mode: str = "lemma",
    filters: dict | None = None,
) -> pd.DataFrame:
    """Find concordance lines and return them with their KWIC text.

    Three ways in, because they answer different questions:

    * `lemma` — "show me every hit whose neighbourhood contains *justification*".
      Needs `tokens.parquet` (the lemma bags), and joins the matching keys to
      `hits.parquet` for the display text. This is the mode that mirrors the
      temporal page: the same words, but read in their sentences instead of
      counted.
    * `kwic` — a free-text search of the ±42-character window (`hits.parquet`
      alone). Good for phrases and for checking what a regex pattern caught.
    * `pattern` — "show me everything `A6_saint_paul` matched". Used to audit a
      pattern's precision, which is how `paul_eval.tsv` was labelled.

    Parameters
    ----------
    tokens, hits : the two frames (see `load_tokens` / `load_hits`).
    query : search term; case-insensitive in every mode.
    mode : key of `SEARCH_MODES`.
    filters : keyword arguments for `filter_tokens`, applied to both frames.

    Returns
    -------
    pd.DataFrame
        `CONCORDANCE_COLUMNS`, sorted by year, index reset.
    """
    if mode not in SEARCH_MODES:
        raise ValueError(f"mode must be one of {sorted(SEARCH_MODES)}")
    filters = dict(filters or {})
    # year_range is applied to `tokens` for lemma matching; `hits` is filtered
    # with the same spec so both sides describe the same subset.
    hits_f = filter_tokens(hits, **filters)
    query_l = query.strip().lower()
    if not query_l:
        return hits_f.iloc[0:0][[c for c in CONCORDANCE_COLUMNS if c in hits_f.columns]]

    if mode == "pattern":
        matched = hits_f[hits_f["pattern"].str.lower().str.contains(query_l, regex=False)]
    elif mode == "kwic":
        window = (hits_f["kwic_left"].fillna("") + " "
                  + hits_f["match"].fillna("") + " "
                  + hits_f["kwic_right"].fillna("")).str.lower()
        matched = hits_f[window.str.contains(query_l, regex=False)]
    else:  # lemma
        tokens_f = filter_tokens(tokens, **filters)
        exploded = tokens_f[JOIN_KEY + ["filtered_lemmata"]].explode("filtered_lemmata")
        exploded = exploded[exploded["filtered_lemmata"].notna()]
        keys = (exploded[exploded["filtered_lemmata"].str.lower() == query_l][JOIN_KEY]
                .drop_duplicates())
        matched = hits_f.merge(keys, on=JOIN_KEY, how="inner")

    present = [c for c in CONCORDANCE_COLUMNS if c in matched.columns]
    return matched[present].sort_values("year").reset_index(drop=True)


def kwic_text(row: pd.Series, width: int | None = None) -> str:
    """Render one concordance row as `left ⟦match⟧ right`.

    `width` truncates each side to that many characters (for narrow tables);
    the full window is ~42 characters per side.
    """
    def cut(text: str) -> str:
        text = "" if pd.isna(text) else str(text).strip()
        return text if width is None or len(text) <= width else text[:width] + "…"

    return f"{cut(row['kwic_left'])} ⟦{str(row['match']).strip()}⟧ {cut(row['kwic_right'])}"
