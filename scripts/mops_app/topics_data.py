#!/usr/bin/env python3
"""Topic-model layer of the MOPS app: artifact loaders and topic aggregation.

Like `mops_app.data`, this module **never imports streamlit**, so a notebook can
reuse exactly what the page renders.

Where the artifacts come from
-----------------------------
`scripts/fit_paul_topics.py` writes them to `data/large_files/paul_scan/topics/`
(gitignored). The design record is **`TOPICS.md`** at the repo root — read it
before citing any topic number. The app only ever *reads* these files; fitting
belongs to the script, not to a web request (`TOPICS.md` D11).

Two different things live in that directory and it matters which is which:

* `docs.parquet` — one row per fitted document: its topic, its 2-D map
  coordinates and the filterable metadata. **The page aggregates from this**,
  because a user's filters (discipline, language, tier, year) have to be applied
  before counting.
* `topics_info/over_time/per_class.parquet` — aggregates of the *unfiltered* fit,
  precomputed so a notebook (or a reader who wants the model as fitted) does not
  have to recompute them. The page does not plot them; `check_mops_app.py` uses
  `over_time` to verify the page's own aggregation reproduces them.

One warning is repeated on the page itself, deliberately: the outlier share is the
binding constraint on this model, and a page that quietly hid it would invite
exactly the overclaim `TOPICS.md` §7 forbids.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import SCAN_DIR, bin_edges, bin_label

# =============================================================================
# PATHS
# =============================================================================

TOPICS_DIR = SCAN_DIR / "topics"

#: Which fitted run the app serves. `full` is the production fit (all 379,859
#: documents); `sample100k` etc. exist for pilots and are reachable via `run=`.
TOPIC_RUN = "full"

#: The sentence every page that shows topic numbers has to show somewhere.
OUTLIER_NOTE = (
    "A large share of documents sits in the outlier topic (`-1`): HDBSCAN could "
    "not place them in any cluster, which is a finding about the corpus rather "
    "than a failure to clean up. They are kept and reported, never reassigned "
    "(TOPICS.md D9), so no topic share here is a share of *all* discourse."
)

#: Metrics for the trend and composition views.
TOPIC_METRICS = {
    "share": "share of the bin's documents (outliers included)",
    "share_modelled": "share of the clustered documents (outlier topic omitted)",
    "count": "documents",
}
DEFAULT_TOPIC_METRIC = "share"

#: The class groupings `per_class.parquet` uses, and the page offers.
TOPIC_CLASS_TYPES = {
    "journal-discipline": "Discipline (journal)",
    "journal-confession": "Confession (journal)",
    "language": "Language",
}

OUTLIER_TOPIC = -1

#: The outlier topic is drawn in a neutral grey rather than a categorical colour.
#: Two reasons, and the second is the important one:
#:
#: 1. it is not a topic, so it should not consume a slot in the categorical palette;
#: 2. it is **always visible**. Hiding it, or folding it into a real topic, is what
#:    would let a reader forget that more than half the corpus is unclustered. Grey
#:    says "measured, but of a different kind" — which is exactly its status (D9).
OUTLIER_COLOR = "#9ca3af"

#: Plotly's default qualitative palette, copied so this module needs no plotly import
#: (it stays importable from a bare pandas/numpy kernel).
CATEGORICAL_COLORS = (
    "#636efa", "#ef553b", "#00cc96", "#ab63fa", "#ffa15a", "#19d3f3",
    "#ff6692", "#b6e880", "#ff97ff", "#fecb52",
)


def is_outlier(label: str) -> bool:
    """Whether a *rendered* topic label (see `topic_label`) denotes the outlier topic."""
    return label.startswith("Outlier")


def topic_color_map(labels: list[str]) -> dict[str, str]:
    """Colour per rendered label, with the outlier topic grey and topics categorical.

    Categorical colours are assigned in the order given, skipping the outlier, so the
    same palette position always means the same relative rank and a reader can tell
    the outlier apart at a glance in any figure.
    """
    out, i = {}, 0
    for label in labels:
        if is_outlier(label):
            out[label] = OUTLIER_COLOR
        else:
            out[label] = CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
            i += 1
    return out


def paths_for(run: str = TOPIC_RUN) -> dict[str, Path]:
    """Artifact paths for one fitted run (mirrors `fit_paul_topics.paths_for`).

    The last three are the cached *stages* of the fit rather than things the app
    reads: the embeddings (D7), the 5-d clustering space and the 2-d map. They are
    listed here so a notebook can report or inspect them without hard-coding
    filenames — and because the caching of those three is what makes a re-cluster
    cost ~20 s instead of a full refit.
    """
    stem = "" if run == "full" else f"_{run}"
    return {
        "docs": TOPICS_DIR / f"docs{stem}.parquet",
        "topics_info": TOPICS_DIR / f"topics_info{stem}.parquet",
        "over_time": TOPICS_DIR / f"topics_over_time{stem}.parquet",
        "per_class": TOPICS_DIR / f"topics_per_class{stem}.parquet",
        "report": TOPICS_DIR / f"report{stem}.json",
        "embeddings": TOPICS_DIR / f"embeddings{stem}.npy",
        "reduced5": TOPICS_DIR / f"reduced5{stem}.npy",
        "xy": TOPICS_DIR / f"xy{stem}.npy",
    }


def available_runs() -> list[str]:
    """Run names present on disk, `full` first (a pilot must not shadow it)."""
    found = []
    if TOPICS_DIR.exists():
        for path in sorted(TOPICS_DIR.glob("report*.json")):
            name = path.stem.replace("report", "").lstrip("_")
            found.append(name or "full")
    return sorted(found, key=lambda r: (r != "full", r))


def is_available(run: str = TOPIC_RUN) -> bool:
    """Whether the page can render at all — checked before any load."""
    return paths_for(run)["docs"].exists()


# =============================================================================
# LOADERS
# =============================================================================

def _missing(path: Path, run: str) -> FileNotFoundError:
    return FileNotFoundError(
        f"{path} not found — fit it with:\n"
        f"    cd scripts && ../mops_venv/bin/python fit_paul_topics.py"
        + ("" if run == "full" else f" --sample {run.replace('sample', '')}")
        + "\n(see TOPICS.md for the frozen parameters)")


def load_docs(columns: list[str] | None = None, run: str = TOPIC_RUN) -> pd.DataFrame:
    """Per-document topics and map coordinates (`docs.parquet`).

    One row per fitted concordance hit: `doc_id`, `topic`, the 2-D map position
    (`x`, `y`) and the filterable metadata (`year`, `journal`, `language`,
    `tier`, `flags`, the journal annotations). `doc_id` is a dense 0..n-1 index,
    asserted by `fit_paul_topics.py --verify`.
    """
    path = paths_for(run)["docs"]
    if not path.exists():
        raise _missing(path, run)
    return pd.read_parquet(path, columns=columns, engine="pyarrow")


def load_topics_info(run: str = TOPIC_RUN) -> pd.DataFrame:
    """Topic table: `Topic`, `Count`, `top_terms` (a real list), `is_outlier`.

    `top_terms_str` is the underscore-free rendering of the same list, which is
    what a table wants; `top_terms` is kept so callers can join or filter on the
    terms instead of re-parsing a string.
    """
    path = paths_for(run)["topics_info"]
    if not path.exists():
        raise _missing(path, run)
    return pd.read_parquet(path, engine="pyarrow")


def load_topics_over_time(run: str = TOPIC_RUN) -> pd.DataFrame:
    """Precomputed per-bin topic frequencies for the **unfiltered** fit."""
    path = paths_for(run)["over_time"]
    if not path.exists():
        raise _missing(path, run)
    return pd.read_parquet(path, engine="pyarrow")


def load_topics_per_class(run: str = TOPIC_RUN) -> pd.DataFrame:
    """Precomputed topic distribution per journal annotation / language."""
    path = paths_for(run)["per_class"]
    if not path.exists():
        raise _missing(path, run)
    return pd.read_parquet(path, engine="pyarrow")


def load_report(run: str = TOPIC_RUN) -> dict:
    """The run's provenance record — parameters, versions, counts, decisions."""
    path = paths_for(run)["report"]
    if not path.exists():
        raise _missing(path, run)
    return json.loads(path.read_text(encoding="utf-8"))


# =============================================================================
# LABELS
# =============================================================================

def topic_label(topic: int, info: pd.DataFrame, width: int = 48) -> str:
    """`-1` -> `"Outlier"`; otherwise `"t12 — jewish, jews, judaism, …"`.

    Topics are identified by number everywhere (that is what `docs.parquet`
    stores); the terms are attached for reading, never used to key anything.
    """
    if topic == OUTLIER_TOPIC:
        return "Outlier (−1)"
    row = info.loc[info["Topic"] == topic]
    if row.empty:
        return f"t{topic}"
    terms = str(row.iloc[0].get("top_terms_str", ""))[:width]
    return f"t{topic} — {terms}" if terms else f"t{topic}"


def labels(info: pd.DataFrame) -> dict[int, str]:
    """All topic labels at once, for `format_func` in a multiselect."""
    return {int(t): topic_label(int(t), info) for t in info["Topic"]}


# =============================================================================
# AGGREGATION (recomputed, because the user's filters come first)
# =============================================================================

def filter_docs(
    docs: pd.DataFrame,
    *,
    year_range: tuple[int, int] | None = None,
    languages: list[str] | None = None,
    tiers: list[str] | None = None,
    disciplines: list[str] | None = None,
    confessions: list[str] | None = None,
    journals: list[str] | None = None,
    exclude_flags: list[str] | None = None,
    topics: list[int] | None = None,
    include_outliers: bool = True,
) -> pd.DataFrame:
    """Apply the page's filters to the document table.

    Same contract as `data.filter_tokens`: every argument optional, an empty or
    `None` value meaning "no restriction". `flags` is again a semicolon-separated
    string and must be read as `df["flags"]`.
    """
    df = docs
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
    if exclude_flags:
        flags = df["flags"].fillna("")
        mask = pd.Series(False, index=df.index)
        for flag in exclude_flags:
            mask |= flags.str.contains(flag, regex=False)
        df = df[~mask]
    if topics:
        df = df[df["topic"].isin(topics)]
    if not include_outliers:
        df = df[df["topic"] != OUTLIER_TOPIC]
    return df


def topic_sizes(docs: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """Per-topic document counts for the filtered subset, with their terms.

    The `Count` column in `topics_info.parquet` describes the *unfiltered* fit,
    so it cannot be joined blindly onto a filtered table — this recomputes it and
    attaches the label terms, keeping the two next to each other so a discrepancy
    is visible rather than implied.
    """
    counts = docs["topic"].value_counts().rename("n_docs")
    table = info[["Topic", "top_terms_str"]].copy()
    table["n_docs"] = table["Topic"].map(counts).fillna(0).astype("int64")
    table = table[table["n_docs"] > 0].sort_values("n_docs", ascending=False)
    total = len(docs)
    table["share"] = table["n_docs"] / total if total else float("nan")
    modelled = int((docs["topic"] != OUTLIER_TOPIC).sum())
    table["share_modelled"] = (table["n_docs"] / modelled
                               if modelled else float("nan"))
    return table.reset_index(drop=True)


def topic_trends(
    docs: pd.DataFrame,
    *,
    bin_length: int = 20,
    year_range: tuple[int, int] = (1900, 2019),
    metric: str = DEFAULT_TOPIC_METRIC,
) -> pd.DataFrame:
    """Topic frequency per time bin, as a bins × topics frame (D12).

    Recomputed from the filtered documents rather than read from
    `topics_over_time.parquet`, because that artifact describes the unfiltered
    fit. `Metric` semantics:

    * `share` — documents in this topic ÷ all documents in the bin. This is the
      default because it is the only one that cannot rise merely because the
      outlier share changed.
    * `share_modelled` — documents in this topic ÷ *clustered* documents in the
      bin, for readers who want the composition of the modelled part alone.
    * `count` — raw documents.

    Bins with no documents yield 0, not NaN, so a sparse early bin draws a zero
    line instead of breaking the plot (`data._non_zero` documents the same trap
    in the temporal view).
    """
    if metric not in TOPIC_METRICS:
        raise ValueError(f"metric must be one of {sorted(TOPIC_METRICS)}")

    edges = bin_edges(year_range[0], year_range[1], bin_length)
    order = {bin_label(e): i for i, e in enumerate(edges)}
    df = docs[["year", "topic"]].copy()
    df = df[df["year"].between(year_range[0], year_range[1])]
    if df.empty:
        return pd.DataFrame(index=pd.Index([], name="bin"))

    idx = ((df["year"] - year_range[0]) // bin_length).astype("int64")
    df = df[idx < len(edges)]
    idx = idx[idx < len(edges)]
    df["bin"] = idx.map(dict(enumerate(bin_label(e) for e in edges)))

    grouped = df.groupby(["bin", "topic"], observed=True).size().rename("count").reset_index()
    per_bin = grouped.groupby("bin", observed=True)["count"].transform("sum")
    modelled = (df[df["topic"] != OUTLIER_TOPIC].groupby("bin", observed=True).size())

    if metric == "count":
        grouped["value"] = grouped["count"].astype(float)
    elif metric == "share":
        grouped["value"] = grouped["count"] / per_bin.replace(0, pd.NA).astype("float64")
    else:
        denom = grouped["bin"].map(modelled).astype("float64")
        grouped["value"] = (grouped["count"] / denom.where(denom > 0)).astype("float64")

    matrix = (grouped.pivot(index="bin", columns="topic", values="value")
              .reindex([bin_label(e) for e in edges if bin_label(e) in set(grouped["bin"])])
              .fillna(0.0))
    matrix.index.name = "bin"
    matrix.columns = [int(c) for c in matrix.columns]
    # The outlier topic is not one of the clustered documents, so it has no
    # `share_modelled` — including it would divide it by a total that excludes it
    # and yield a value above 1, which is not a share of anything.
    if metric == "share_modelled" and OUTLIER_TOPIC in matrix.columns:
        matrix = matrix.drop(columns=OUTLIER_TOPIC)
    return matrix.sort_index(key=lambda ix: ix.map(order))


def topic_by_class(docs: pd.DataFrame, class_type: str) -> pd.DataFrame:
    """Topic × class composition for one grouping, recomputed from the subset.

    Returns a long frame (`class_value`, `topic`, `count`, `share_of_class`).
    Rows whose class value is missing are labelled `(unannotated)` rather than
    dropped, because ~35% of journals carry no annotation and dropping them would
    make the discipline shares look complete when they are not.
    """
    if class_type not in docs.columns:
        raise ValueError(f"{class_type} is not a column of docs.parquet")
    frame = docs[["topic", class_type]].copy()
    frame[class_type] = frame[class_type].fillna("(unannotated)")
    grouped = (frame.groupby(["topic", class_type], observed=True).size()
               .rename("count").reset_index()
               .rename(columns={class_type: "class_value"}))
    totals = grouped.groupby("class_value", observed=True)["count"].transform("sum")
    grouped["share_of_class"] = grouped["count"] / totals
    grouped["class_type"] = class_type
    return grouped[["class_type", "class_value", "topic", "count", "share_of_class"]]


def map_sample(docs: pd.DataFrame, max_points: int = 60_000, seed: int = 42) -> pd.DataFrame:
    """A random subsample of the 2-D map, for drawing (`x`, `y`, `topic`).

    380k scatter points are technically plottable but they make the browser
    stutter for no analytic gain — the shape of the space is legible long before
    that. Sampling is seeded so the picture does not change between reruns.
    """
    if len(docs) <= max_points:
        return docs[["x", "y", "topic", "year"]]
    return docs[["x", "y", "topic", "year"]].sample(max_points, random_state=seed)


# =============================================================================
# COMPARING SUBSETS
# =============================================================================

def outlier_shares(subsets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Outlier share per labelled subset — **check this before comparing anything**.

    The outlier share is not a constant of the corpus: it is 62% for tier A and
    53% for tier C, and it differs between disciplines. That matters because it is
    the denominator of `share_modelled`, so a subset with a larger unclustered
    remainder has its remaining topics *inflated* by the normalisation alone. Two
    subsets can therefore differ in every topic share without differing in what
    is actually being discussed. Printing this table next to any comparison is how
    that gets noticed rather than mistaken for a finding.
    """
    rows = []
    for label, df in subsets.items():
        if df.empty:
            rows.append({"subset": label, "n_docs": 0, "n_outlier": 0,
                         "outlier_share": float("nan"), "n_modelled": 0})
            continue
        n_out = int((df["topic"] == OUTLIER_TOPIC).sum())
        rows.append({"subset": label, "n_docs": int(len(df)), "n_outlier": n_out,
                     "outlier_share": n_out / len(df), "n_modelled": len(df) - n_out})
    return pd.DataFrame(rows)


def subset_shares(subsets: dict[str, pd.DataFrame], *,
                  metric: str = DEFAULT_TOPIC_METRIC,
                  topics: list[int] | None = None) -> pd.DataFrame:
    """Share of each topic within each labelled subset (topics × subsets).

    The table a "how does this subset differ" question actually needs: every
    column is normalised *within its own subset*, so columns are comparable even
    when the subsets differ wildly in size.

    Parameters
    ----------
    subsets : dict[str, pd.DataFrame]
        Label -> filtered document frame. Build these with `filter_docs`.
    metric : one of `TOPIC_METRICS`.
        * `share` — of all documents in the subset (outliers included);
        * `share_modelled` — of the subset's *clustered* documents;
        * `count` — raw documents.
    topics : optional row selection/order.

    Returns
    -------
    pd.DataFrame
        Index = topic, columns = subset labels, values = the metric. A topic with
        no documents in a subset gets 0.0, which is the correct share, not a
        missing value. With `share_modelled` the outlier row is **omitted**: it is
        not one of the clustered documents, so it has no share of them (including
        it would divide it by a total that excludes it and give a value above 1).
    """
    if metric not in TOPIC_METRICS:
        raise ValueError(f"metric must be one of {sorted(TOPIC_METRICS)}")

    columns = {}
    for label, df in subsets.items():
        if df.empty:
            continue
        counts = df["topic"].value_counts()
        if metric == "count":
            values = counts.astype(float)
        else:
            total = (len(df) if metric == "share"
                     else int((df["topic"] != OUTLIER_TOPIC).sum()))
            values = counts / total if total else counts.astype(float) * float("nan")
        columns[label] = values

    frame = pd.DataFrame(columns).fillna(0.0)
    frame.index.name = "topic"
    if metric == "share_modelled" and OUTLIER_TOPIC in frame.index:
        frame = frame.drop(index=OUTLIER_TOPIC)
    if topics is not None:
        frame = frame.reindex([t for t in topics if t in frame.index])
    return frame.sort_index()


def diff_from_reference(shares: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Subtract one column of `subset_shares` from every other (signed differences).

    Positive = the topic takes a larger share of the subset than of the reference.
    Kept as a separate step so the raw shares stay on screen next to the
    differences: a difference is only interpretable against the base rate, and a
    +2pp gap on a topic that is 0.3% of the corpus is a different claim from the
    same gap on a topic that is 20%.
    """
    if reference not in shares.columns:
        raise ValueError(f"reference {reference!r} not among {list(shares.columns)}")
    return shares.sub(shares[reference], axis=0).drop(columns=reference)


def trends_by_subset(subsets: dict[str, pd.DataFrame], *,
                     bin_length: int = 20,
                     year_range: tuple[int, int] = (1900, 2019),
                     metric: str = DEFAULT_TOPIC_METRIC) -> pd.DataFrame:
    """Topic share over time, per subset, as a long frame.

    Long (`subset`, `bin`, `topic`, `value`) rather than a dict of matrices
    because the comparison is the point: this is what answers "does this topic's
    trajectory differ between the subsets", and it feeds a grouped plot directly.

    Each subset's series is normalised within itself (see `topic_trends`), so the
    lines answer "of this subset, what fraction was topic X at this time" — never
    "how many documents", which would mostly measure the subset's own growth.
    """
    frames = []
    for label, df in subsets.items():
        if df.empty:
            continue
        matrix = topic_trends(df, bin_length=bin_length, year_range=year_range,
                              metric=metric)
        if matrix.empty:
            continue
        long = (matrix.rename_axis("topic", axis=1).stack().rename("value")
                .reset_index().rename(columns={"level_0": "bin", "level_1": "topic"}))
        long["subset"] = label
        frames.append(long[["subset", "bin", "topic", "value"]])
    if not frames:
        return pd.DataFrame(columns=["subset", "bin", "topic", "value"])
    order = {label: i for i, label in enumerate(subsets)}
    out = pd.concat(frames, ignore_index=True)
    out["_subset_order"] = out["subset"].map(order)
    return (out.sort_values(["_subset_order", "bin", "topic"])
            .drop(columns="_subset_order").reset_index(drop=True))


def trajectory_gap(trends: pd.DataFrame, topic: int) -> pd.DataFrame:
    """How far apart the subsets' trajectories are for one topic, per pair.

    Reports, for each subset pair, the mean absolute difference of the topic's
    share across the time bins. This is a *descriptive* magnitude — it ranks which
    subsets disagree most about a topic's history — and it is deliberately not a
    significance test: the documents are overlapping ±10-token windows, so they
    are not independent draws and any p-value computed from them would be
    meaningless (TOPICS.md §7, limitation 1).
    """
    sub = trends[trends["topic"] == topic]
    if sub.empty:
        return pd.DataFrame(columns=["subset_a", "subset_b", "mean_abs_gap", "max_abs_gap"])
    wide = sub.pivot(index="bin", columns="subset", values="value")
    rows = []
    labels = list(wide.columns)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            gap = (wide[a] - wide[b]).abs()
            rows.append({"subset_a": a, "subset_b": b,
                         "mean_abs_gap": float(gap.mean()),
                         "max_abs_gap": float(gap.max())})
    return (pd.DataFrame(rows).sort_values("mean_abs_gap", ascending=False)
            .reset_index(drop=True))


# =============================================================================
# HOW DIFFERENT ARE TWO DISTRIBUTIONS?
# =============================================================================

#: Distance measures for comparing two topic distributions.
#: `js` is the Jensen–Shannon **distance** (the square root of the divergence, so it
#: is a true metric in [0, 1] when `base=2`); `tv` is total variation
#: (= 1/2 * sum|p-q|); `cosine` is 1 - cosine similarity.
DISTANCES = {
    "js": "Jensen–Shannon distance (0 = identical, 1 = disjoint support)",
    "tv": "total variation (share of mass that must move)",
    "cosine": "cosine distance (1 - similarity)",
}


def _js(p: "np.ndarray", q: "np.ndarray") -> float:
    from scipy.spatial.distance import jensenshannon

    # jensenshannon returns the distance, and with base=2 it is bounded by 1.
    return float(jensenshannon(p, q, base=2))


def _tv(p: "np.ndarray", q: "np.ndarray") -> float:
    return float(np.abs(p - q).sum() / 2)


def _cosine(p: "np.ndarray", q: "np.ndarray") -> float:
    denom = np.linalg.norm(p) * np.linalg.norm(q)
    return float(1 - (p @ q) / denom) if denom else float("nan")


_DISTANCE_FUNCS = {"js": _js, "tv": _tv, "cosine": _cosine}


def distribution_distance(shares: pd.DataFrame, reference: str, *,
                          method: str = "js") -> pd.DataFrame:
    """Distance of every column of `subset_shares` from one reference column.

    One number per subset answering "how different is this subset's overall topic
    mix from the corpus as a whole", which is a different (and coarser) question
    than per-topic differences — two subsets can swap topics around while keeping
    the same distance from the reference.

    **These are descriptive magnitudes, not tests.** The documents are overlapping
    ±10-token windows and therefore not independent draws, so no significance
    statement can be attached to them (TOPICS.md §7). Read them as ordinals: which
    subsets are further from the whole, and by how much, on a fixed scale.

    Parameters
    ----------
    shares : pd.DataFrame
        Output of `subset_shares`. Columns should sum to 1 for a distance to be
        interpretable; both `share` and `share_modelled` satisfy that.
    reference : column name to measure against (usually `"overall"`).
    method : key of `DISTANCES`.

    Returns
    -------
    pd.DataFrame
        `subset`, `distance`, `method`, sorted descending (the reference itself is
        omitted).
    """
    if method not in _DISTANCE_FUNCS:
        raise ValueError(f"method must be one of {sorted(_DISTANCE_FUNCS)}")
    if reference not in shares.columns:
        raise ValueError(f"reference {reference!r} not among {list(shares.columns)}")
    func = _DISTANCE_FUNCS[method]
    ref = shares[reference].to_numpy(dtype=float)
    rows = []
    for label in shares.columns:
        if label == reference:
            continue
        vec = shares[label].to_numpy(dtype=float)
        rows.append({"subset": label, "distance": func(ref, vec), "method": method})
    return pd.DataFrame(rows).sort_values("distance", ascending=False).reset_index(drop=True)


def distance_over_time(trends: pd.DataFrame, reference: str, *,
                       method: str = "js",
                       min_docs: int = 0,
                       docs_per_bin: pd.Series | None = None) -> pd.DataFrame:
    """Distance from the reference subset's topic mix, computed **per time bin**.

    The trajectory version of `distribution_distance`: it answers whether the
    subsets have grown more or less alike over time, which a single overall number
    cannot show.

    `min_docs` guards against the noise that motivates it: a bin in which a subset
    has only a handful of documents yields a wildly unstable share vector, and
    without a floor those bins dominate the picture. Pass `docs_per_bin` (a
    `(subset, bin) -> n` series, e.g. from `trends_by_subset(metric="count")`) to
    enable the mask; bins below the floor are returned as NaN rather than dropped,
    so a plot shows a gap instead of silently bridging it.
    """
    if method not in _DISTANCE_FUNCS:
        raise ValueError(f"method must be one of {sorted(_DISTANCE_FUNCS)}")
    if trends.empty:
        return pd.DataFrame(columns=["bin", "subset", "distance"])

    func = _DISTANCE_FUNCS[method]
    wide = trends.pivot_table(index=["subset", "bin"], columns="topic", values="value",
                             aggfunc="sum").fillna(0.0)
    if reference not in wide.index.get_level_values("subset"):
        raise ValueError(f"reference {reference!r} not among the subsets")

    ref = wide.xs(reference, level="subset")
    rows = []
    for (label, bin_), row in wide.iterrows():
        if label == reference:
            continue
        if min_docs and docs_per_bin is not None:
            n = docs_per_bin.get((label, bin_), 0)
            if n < min_docs:
                rows.append({"bin": bin_, "subset": label, "distance": float("nan"),
                             "n_docs": int(n)})
                continue
        rows.append({"bin": bin_, "subset": label,
                     "distance": func(ref.loc[bin_].to_numpy(dtype=float),
                                      row.to_numpy(dtype=float))})
    out = pd.DataFrame(rows)
    order = {b: i for i, b in enumerate(ref.index)}
    out["_order"] = out["bin"].map(order)
    return (out.sort_values(["subset", "_order"]).drop(columns="_order")
            .reset_index(drop=True))
