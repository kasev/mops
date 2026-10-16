#!/usr/bin/env python3
"""fit_paul_topics.py — BERTopic over the lemmatised Paul concordance.

The implementation of the design frozen in **`TOPICS.md`** (read that first; every
non-obvious choice below cites the decision it comes from). This script is the
only thing that fits a model; the web app only ever reads the artifacts.

Data flow
---------
`paul_scan/tokens.parquet`  (426,917 hits: metadata + `filtered_lemmata`)
  -> filter to the scope of D2/D3          (languages, years)
  -> drop documents with no content lemma  (D4, forced — empty docs break c-TF-IDF)
  -> `docs` = " ".join(filtered_lemmata)   (D5 — lemmatised text, NOT the raw KWIC)
  -> embed the NATURAL window on GPU       (D6/D7 — asymmetric on purpose: lemmas
     label the topic, natural text carries the semantics; cached to .npy)
  -> UMAP(5d) -> HDBSCAN -> c-TF-IDF       (D8 — unsupervised baseline)
  -> aggregate over time / per class        (D12 — shares, not counts)
  -> `topics/` artifacts (D11)

Determinism
-----------
Every source of randomness is seeded (`SEED`), the row order is canonical
(sorted by `item_id, char_start` before sampling), and the whole resolution of
parameters is written to `topics/report.json`. Re-running with the same
arguments and the same `tokens.parquet` reproduces the same artifacts.

Usage
-----
    python3 fit_paul_topics.py --sample 100000 --resume --sweep   # D8 grid (TOPICS.md §8)
    python3 fit_paul_topics.py                    # full fit (~11 min on the RTX A4000)
    python3 fit_paul_topics.py --verify           # check an existing run, fit nothing
    python3 fit_paul_topics.py --resume           # reuse cached embeddings, re-cluster

Both expensive stages are cached, so only the first run of a given (model, scope) pays for
them: `embeddings*.npy` (D7) and `reduced5*.npy` (the 5-d UMAP space). Changing an HDBSCAN
parameter, or swapping the embedding model to compare encoders, costs at most one of the two.

`--min-topic-size-frac` is the safe way to move `min_cluster_size` between a pilot and the full
fit: it is a proportion of the corpus, and carrying a fixed count across sample sizes is what
produced the degenerate 2-topic pilot (TOPICS.md D8.1).

`--sample` and `--model` create separate artifact namespaces so a pilot never overwrites a
production run, and so two encoders can be compared on identical documents. `--sweep` writes its
evidence to `pilot_sweep<slug>.json` instead of to the run artifacts, because two pilots of the
same size share a slug and would otherwise overwrite each other's results.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

# --- environment must be set BEFORE torch/transformers import -----------------
# Keep HuggingFace downloads inside the project (D11) even if the caller forgot
# to export HF_HOME, so a fit is reproducible and never fills ~/.cache.
MOPS = Path(__file__).resolve().parents[1]
HF_CACHE = MOPS / "data" / "large_files" / "hf_cache"
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Reuse the app's binning so topic trends and the Temporal page agree by
# construction rather than by coincidence (mops_app.data has no streamlit import).
from mops_app.data import bin_edges, bin_label  # noqa: E402

# =============================================================================
# CONFIGURATION — the defaults reproduce TOPICS.md §4
# =============================================================================

SCAN = MOPS / "data" / "large_files" / "paul_scan"
TOKENS = SCAN / "tokens.parquet"
HITS = SCAN / "hits.parquet"
TOPICS = SCAN / "topics"

#: D2 — both English codes travel together.
DEFAULT_LANGUAGES = ("eng", "en", "fre", "ger")
#: D3 — the study window.
DEFAULT_YEAR_RANGE = (1900, 2019)
#: D6 — multilingual; D5 means we also need the natural windows.
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#: D8 — HDBSCAN's min_cluster_size, fixed by the FULL-SCALE sweep (TOPICS.md D8.1). It is a
#: *proportion* knob: 760 is 0.200% of the 379,859-document fit set, chosen for a median topic
#: of ~1,742 documents — enough that a 20-year bin still holds ~290 of them, which is what makes
#: the per-bin trends of D12 readable. See `pilot_sweep.json` for all 16 configurations.
DEFAULT_MIN_TOPIC_SIZE = 760
#: D8 — HDBSCAN's min_samples. `None` means min_cluster_size, which is very conservative; the
#: sweep found 1 strictly better than 5 at every size (200/0.1: 43.0% outliers at 1 vs 53.7% at 5).
DEFAULT_MIN_SAMPLES = 1
#: D8 — HDBSCAN's cluster_selection_epsilon. **Left at BERTopic's default of 0.0.**
#:
#: This default was briefly set to 0.1 on the strength of the 100k pilot, where it lowered the
#: outlier share (51% -> 43%) at a constant topic count. At full scale that is a **trap**:
#: epsilon is an ABSOLUTE distance in the reduced space, so it does not transfer between sample
#: sizes, and on 379,859 documents it merges clusters into a single ~200k-document blob (52.6% of
#: the corpus) while *lowering* the reported outlier share to 33% — i.e. it makes the headline
#: metric look better while destroying the model. Measured at full scale: 0.0/0.01/0.02/0.03 are
#: identical; 0.05 barely differs; 0.1 collapses. Only raise this after sweeping it at the scale
#: you intend to fit (see `--sweep` and `TOPICS.md` D8.1, which records this as an error caught by
#: the full-scale sweep).
DEFAULT_CLUSTER_SELECTION_EPSILON = 0.0
#: D12 — 20-year bins, matching the app.
BIN_LENGTH = 20
#: Fixed everywhere.
SEED = 42

#: D8 — UMAP for the clustering space, and a second one purely for the app's map.
UMAP_PARAMS = dict(n_components=5, metric="cosine", n_neighbors=15,
                   min_dist=0.0, random_state=SEED)
UMAP_2D_PARAMS = dict(n_components=2, metric="cosine", n_neighbors=15,
                      min_dist=0.1, random_state=SEED)

#: Columns carried into `docs.parquet` for the app to filter and display on.
DOC_COLUMNS = [
    "item_id", "char_start", "year", "bidecade", "journal", "language", "tier",
    "pattern", "flags", "journal-cat", "journal-discipline", "journal-confession",
]
#: Class groupings for the per-class aggregation (D11).
CLASS_COLUMNS = ["journal-discipline", "journal-confession", "language"]


# =============================================================================
# HELPERS
# =============================================================================

def log(msg: str) -> None:
    """Timestamped progress line (buffered stdout is unhelpful for a long fit)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def namespaces(sample: int | None, model: str) -> tuple[str, str]:
    """Return `(run_name, slug)` so pilots and model variants never collide.

    A pilot run and the production run write to different files; so do different
    embedding models. `run_name` labels the files, `slug` is the filename part.
    """
    parts = []
    if sample:
        parts.append(f"sample{_short_count(sample)}")
    if model != DEFAULT_MODEL:
        parts.append(Path(model.rstrip("/")).name)
    slug = "_".join(parts) if parts else "full"
    return ("production" if not parts else slug), slug


def _short_count(n: int) -> str:
    """`25000` -> `25k`, for readable filenames."""
    return f"{n // 1000}k" if n >= 1000 and n % 1000 == 0 else str(n)


def paths_for(slug: str) -> dict[str, Path]:
    """All output paths for a run, in one place (single source of truth)."""
    stem = "" if slug == "full" else f"_{slug}"
    return {
        "docs": TOPICS / f"docs{stem}.parquet",
        "embeddings": TOPICS / f"embeddings{stem}.npy",
        "reduced5": TOPICS / f"reduced5{stem}.npy",
        "xy": TOPICS / f"xy{stem}.npy",
        "topics_info": TOPICS / f"topics_info{stem}.parquet",
        "over_time": TOPICS / f"topics_over_time{stem}.parquet",
        "per_class": TOPICS / f"topics_per_class{stem}.parquet",
        "model": TOPICS / f"topic_model{stem}",
        "report": TOPICS / f"report{stem}.json",
    }


# =============================================================================
# 1. DOCUMENTS
# =============================================================================

def build_documents(languages, year_range, sample, drop_flags=None) -> pd.DataFrame:
    """Assemble the fit table: one row per concordance line (D1).

    Applies D2 (languages), D3 (years) and D4 (drop empty lemma bags), then joins
    the *natural* window from `hits.parquet` for the embedding step. The returned
    frame carries both `doc` (lemmatised, for c-TF-IDF) and `natural` (for the
    encoder), plus `doc_id` giving the row order that `embeddings.npy` follows.
    """
    if not TOKENS.exists():
        raise SystemExit(f"missing {TOKENS} — build it with "
                         f"python3 scripts/prepare_jstor_app_data.py")

    log(f"reading {TOKENS.name} …")
    tok = pd.read_parquet(TOKENS, engine="pyarrow")
    log(f"  {len(tok):,} hits")

    n0 = len(tok)
    tok = tok[tok["language"].isin(languages)]
    log(f"  D2 languages {'/'.join(languages)}: {len(tok):,} ({len(tok) / n0:.2%})")

    tok = tok[tok["year"].between(year_range[0], year_range[1])]
    log(f"  D3 years {year_range[0]}-{year_range[1]}: {len(tok):,}")

    tok["n_lemmas"] = tok["filtered_lemmata"].apply(len)
    before = len(tok)
    tok = tok[tok["n_lemmas"] > 0]
    log(f"  D4 dropped {before - len(tok):,} with no content lemma "
        f"({(before - len(tok)) / before:.2%}); {len(tok):,} remain")

    if drop_flags:
        # Topical scope choice (D10): the default keeps everything, but the
        # apparatus can be excluded to quantify how much structure it carries.
        flags = tok["flags"].fillna("")
        mask = pd.Series(False, index=tok.index)
        for flag in drop_flags:
            mask |= flags.str.contains(flag, regex=False)
        tok = tok[~mask]
        log(f"  D10 excluding flags {drop_flags}: {len(tok):,} remain")

    # Canonical order makes sampling and the embedding cache deterministic.
    tok = tok.sort_values(["item_id", "char_start"], kind="stable")

    if sample and sample < len(tok):
        tok = tok.sample(n=sample, random_state=SEED).sort_values(
            ["item_id", "char_start"], kind="stable")
        log(f"  sampled {len(tok):,} (random_state={SEED})")

    tok = tok.reset_index(drop=True)
    tok["doc_id"] = np.arange(len(tok))

    # D5: `doc` is what c-TF-IDF sees; D6: `natural` is what the encoder sees.
    tok["doc"] = tok["filtered_lemmata"].apply(lambda xs: " ".join(xs))

    log(f"reading {HITS.name} for the natural windows …")
    windows = pd.read_parquet(
        HITS, columns=["item_id", "char_start", "kwic_left", "match", "kwic_right"],
        engine="pyarrow")
    tok = tok.merge(windows, on=["item_id", "char_start"], how="left",
                    validate="m:1", indicator=True)
    # Test join COVERAGE, not NaN-ness: ~140 hits legitimately have an empty
    # kwic_left/kwic_right (the match sits at the very start or end of the text),
    # so a NaN check would fail on good data. `match` is never null.
    unmatched = int((tok["_merge"] == "left_only").sum())
    if unmatched:
        raise SystemExit(f"{unmatched:,} documents did not join to {HITS.name} — "
                         f"tokens.parquet and hits.parquet are out of sync")
    tok = tok.drop(columns=["_merge"])
    tok["natural"] = (tok["kwic_left"].fillna("") + " " + tok["match"].fillna("")
                      + " " + tok["kwic_right"].fillna("")).str.strip()

    keep = [c for c in DOC_COLUMNS if c in tok.columns]
    return tok[["doc_id", "doc", "natural", "n_lemmas"] + keep]


# =============================================================================
# 2. EMBEDDINGS
# =============================================================================

#: Some encoders require an instruction prefix — the E5 family documents
#: `query: ` for retrieval and `passage: ` for the texts being retrieved, and
#: measured worse without it. Applied to the text handed to the *encoder* only,
#: never to the lemmatised `doc` that c-TF-IDF sees, where the prefix would be
#: counted as a topic word and would appear in every topic label.
PREFIX_RULES = ((r"e5-", "passage: "), (r"e5_", "passage: "))


def embedding_prefix(model_name: str) -> str:
    """The instruction prefix a model expects, or `""` for the many that want none."""
    name = model_name.lower()
    for needle, prefix in PREFIX_RULES:
        if needle in name:
            return prefix
    return ""


def embed(docs: list[str], model_name: str, device: str, batch_size: int,
          cache: Path, resume: bool) -> np.ndarray:
    """Embed the natural windows on the GPU, caching to `cache` (D7).

    The cache is keyed by nothing but the file itself; `--resume` trusts it when
    the row count matches, which is what makes re-clustering free.
    """
    if resume and cache.exists():
        emb = np.load(cache, mmap_mode="r")
        if emb.shape[0] == len(docs):
            log(f"reusing {cache.name} {emb.shape} (--resume)")
            return np.asarray(emb)
        log(f"ignoring {cache.name}: {emb.shape[0]:,} rows vs {len(docs):,} docs")

    import torch
    from sentence_transformers import SentenceTransformer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"embedding {len(docs):,} docs with {model_name} on {device}")
    if device == "cuda":
        # NVML can be broken on this host while CUDA itself works; say so rather
        # than let a stray warning look like a failure.
        log(f"  torch {torch.__version__} | cuda build {torch.version.cuda} | "
            f"{torch.cuda.get_device_name(0)}")

    model = SentenceTransformer(model_name, device=device)
    dim = model.get_sentence_embedding_dimension()
    log(f"  dimension {dim} | batch {batch_size} | HF_HOME={os.environ['HF_HOME']}")

    prefix = embedding_prefix(model_name)
    if prefix:
        log(f"  prefixing every document with {prefix!r} (this model requires it)")
        docs = [prefix + d for d in docs]

    t0 = time.perf_counter()
    emb = model.encode(docs, batch_size=batch_size, convert_to_numpy=True,
                       show_progress_bar=False, normalize_embeddings=False)
    dt = time.perf_counter() - t0
    emb = np.asarray(emb, dtype=np.float32)
    log(f"  embedded in {dt:.1f}s ({len(docs) / dt:,.0f} docs/s) -> {emb.shape}")

    np.save(cache, emb)
    log(f"  cached to {cache.name} ({cache.stat().st_size / 1e6:,.0f} MB)")
    return emb


# =============================================================================
# 3. FIT
# =============================================================================

class PrecomputedSpace:
    """A stand-in for UMAP that returns an already-computed 5-d reduction.

    BERTopic always runs its `umap_model` inside `fit_transform`, so a cached
    reduction can only be reused by handing it a model whose `fit_transform`
    returns a stored array instead of fitting again (see `reduce_space`). Defined
    at module level so the saved model stays picklable.
    """

    def __init__(self, reduced: np.ndarray):
        self.reduced = np.asarray(reduced, dtype=np.float32)

    def fit(self, X, y=None):  # noqa: N803 - sklearn-style signature
        return self

    def fit_transform(self, X, y=None):  # noqa: N803
        return self.reduced

    def transform(self, X):  # noqa: N803
        return self.reduced


def reduce_space(embeddings: np.ndarray, cache: Path, resume: bool) -> np.ndarray:
    """UMAP 384-d -> 5-d, cached to `cache`.

    D7 caches the embeddings so that re-tuning the clustering does not re-embed.
    The same argument holds one step further down the pipeline: UMAP is the
    expensive half of the clustering (~5 min at 348k) and is seeded, so its output
    is reproducible and worth keeping. Caching it turns a parameter sweep from one
    UMAP run per configuration into one UMAP run in total, and guarantees that a
    sweep and the production fit cluster the *same* space.
    """
    if resume and cache.exists():
        reduced = np.load(cache, mmap_mode="r")
        if reduced.shape[0] == len(embeddings):
            log(f"reusing {cache.name} {reduced.shape} (--resume)")
            return np.asarray(reduced, dtype=np.float32)
        log(f"ignoring {cache.name}: {reduced.shape[0]:,} rows vs {len(embeddings):,}")

    import umap

    log("reducing 384d -> 5d (UMAP, the clustering space) …")
    t0 = time.perf_counter()
    reduced = np.asarray(umap.UMAP(**UMAP_PARAMS).fit_transform(embeddings),
                         dtype=np.float32)
    log(f"  done in {time.perf_counter() - t0:.1f}s -> {reduced.shape}")
    np.save(cache, reduced)
    log(f"  cached to {cache.name} ({cache.stat().st_size / 1e6:,.0f} MB)")
    return reduced


def fit(docs: list[str], reduced: np.ndarray, *, min_topic_size: int,
        min_samples: int | None, cluster_selection_epsilon: float,
        cluster_selection_method: str, nr_topics, model_name: str,
        verbose: bool = True) -> tuple[object, np.ndarray]:
    """HDBSCAN -> c-TF-IDF, unsupervised (D8), on a precomputed 5-d space.

    HDBSCAN is constructed explicitly rather than left to BERTopic's default so
    the parameters are visible in `report.json` and in this file. `prediction_data`
    is off: we never classify new documents, and it would cost memory for nothing.

    `min_samples` matters more than it looks. Left at `None` it becomes
    `min_cluster_size`, which is very conservative — the 25k pilot then put ~55% of
    documents in the outlier topic. Lowering it makes HDBSCAN assign documents that
    sit near a cluster but not densely inside it, which is the parameter to tune
    before ever considering post-hoc outlier reassignment (D9).

    `cluster_selection_epsilon` merges clusters whose separation is below the given
    distance, which is the second lever on the outlier share; the pilot measured it
    at 0.1 on a 100k sample (see TOPICS.md D8).
    """
    import hdbscan
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        core_dist_n_jobs=-1,                 # the `hdbscan` package's parallelism knob
        prediction_data=False,
    )
    vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words=None, min_df=1)
    ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)

    topic_model = BERTopic(
        embedding_model=model_name,          # kept for provenance on save() only
        umap_model=PrecomputedSpace(reduced),
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf,
        top_n_words=10,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        calculate_probabilities=False,       # omitted deliberately: memory/time cost
        verbose=verbose,
    )

    log("clustering (HDBSCAN -> c-TF-IDF) …")
    t0 = time.perf_counter()
    topics, _ = topic_model.fit_transform(docs, embeddings=reduced)
    log(f"  fitted in {time.perf_counter() - t0:.1f}s")
    return topic_model, np.asarray(topics)


def reduce_2d(reduced5: np.ndarray, cache: Path, resume: bool) -> np.ndarray:
    """Second UMAP, 5d -> 2d, only for the app's document map (D8).

    Cached for exactly the reason the clustering space is (D7): it depends only on
    the 5-d space already on disk, so re-clustering with different HDBSCAN
    parameters must not pay for it. That matters more here than it looks — measured
    at 379,859 documents this is the **slowest single stage of the whole pipeline**
    (>28 min), several times the 5-d reduction (326 s), although both build one
    neighbour graph over the same rows. The reason is that the cost is set by the
    *intrinsic* dimension of the input, not its column count: the 384-d embeddings
    lie on a low-dimensional manifold, while the 5-d output of the first UMAP is
    genuinely 5-dimensional and yields a much denser, harder neighbour problem.
    """
    if resume and cache.exists():
        xy = np.load(cache, mmap_mode="r")
        if xy.shape[0] == len(reduced5):
            log(f"reusing {cache.name} {xy.shape} (--resume)")
            return np.asarray(xy, dtype=np.float32)
        log(f"ignoring {cache.name}: {xy.shape[0]:,} rows vs {len(reduced5):,}")

    import umap

    log("reducing 5d -> 2d for the app map …")
    t0 = time.perf_counter()
    xy = np.asarray(umap.UMAP(**UMAP_2D_PARAMS).fit_transform(reduced5),
                    dtype=np.float32)
    log(f"  done in {time.perf_counter() - t0:.1f}s")
    np.save(cache, xy)
    log(f"  cached to {cache.name} ({cache.stat().st_size / 1e6:,.0f} MB)")
    return xy


# =============================================================================
# 4. AGGREGATION
# =============================================================================

def over_time(docs: pd.DataFrame, bin_length: int) -> pd.DataFrame:
    """Topic frequency per time bin, as counts AND shares (D12).

    `share` divides by the bin's own document count, which is what makes a topic
    trend readable against a corpus that grows ~8x; `share_modelled` excludes the
    outlier topic, for readers who want the composition of the clustered part.
    """
    y0, y1 = int(docs["year"].min()), int(docs["year"].max())
    edges = bin_edges(y0, y1, bin_length)
    labels = [bin_label(e) for e in edges]
    docs = docs.copy()
    docs["bin"] = ((docs["year"] - y0) // bin_length).astype("int64")
    docs = docs[docs["bin"] < len(edges)]
    docs["bin"] = docs["bin"].map(dict(enumerate(labels)))

    grouped = (docs.groupby(["bin", "topic"], observed=True).size()
               .rename("count").reset_index())
    per_bin = grouped.groupby("bin", observed=True)["count"].transform("sum")
    grouped["n_docs_in_bin"] = per_bin
    grouped["share"] = grouped["count"] / per_bin.replace(0, np.nan)

    modelled = docs[docs["topic"] != -1]
    n_mod = (modelled.groupby("bin", observed=True).size()
             .rename("n_modelled_in_bin"))
    grouped = grouped.merge(n_mod, left_on="bin", right_index=True, how="left")
    grouped["share_modelled"] = grouped["count"] / grouped["n_modelled_in_bin"].replace(0, np.nan)

    order = {lab: i for i, lab in enumerate(labels)}
    grouped["_order"] = grouped["bin"].map(order)
    return (grouped.sort_values(["_order", "topic"])
            .drop(columns="_order").reset_index(drop=True)
            [["bin", "topic", "count", "n_docs_in_bin", "n_modelled_in_bin",
              "share", "share_modelled"]])


def per_class(docs: pd.DataFrame) -> pd.DataFrame:
    """Topic distribution across the journal annotations and language (D11).

    Long format with a `class_type` column so one parquet serves discipline,
    confession and language. Rows whose class value is missing are reported as
    `"(unannotated)"` rather than dropped — ~35% of journals are unannotated, and
    hiding them would make the discipline shares look complete.
    """
    out = []
    for col in CLASS_COLUMNS:
        if col not in docs.columns:
            continue
        frame = docs[["topic", col]].copy()
        frame[col] = frame[col].fillna("(unannotated)")
        grouped = (frame.groupby(["topic", col], observed=True).size()
                   .rename("count").reset_index())
        totals = grouped.groupby(col, observed=True)["count"].transform("sum")
        grouped["n_in_class"] = totals
        grouped["share_of_class"] = grouped["count"] / totals.replace(0, np.nan)
        grouped["class_type"] = col
        out.append(grouped.rename(columns={col: "class_value"}))
    return pd.concat(out, ignore_index=True)[
        ["class_type", "class_value", "topic", "count", "n_in_class", "share_of_class"]] \
        if out else pd.DataFrame()


def topics_info(topic_model, docs: pd.DataFrame) -> pd.DataFrame:
    """`get_topic_info()` plus explicit term lists and long-form labels.

    BERTopic's `Name` packs terms into one underscore string; `top_terms` is kept
    as a real list so the app (`st.dataframe`) and any downstream code do not have
    to re-parse it.
    """
    info = topic_model.get_topic_info().copy()
    terms = {}
    for t in info["Topic"]:
        if t == -1:
            terms[t] = []
            continue
        terms[t] = [w for w, _ in (topic_model.get_topic(t) or [])]
    info["top_terms"] = info["Topic"].map(terms)
    info["top_terms_str"] = info["top_terms"].apply(lambda ws: ", ".join(ws))
    info["is_outlier"] = info["Topic"] == -1
    counts = docs["topic"].value_counts().rename("n_docs")
    info = info.merge(counts, left_on="Topic", right_index=True, how="left")
    return info.sort_values("Topic").reset_index(drop=True)


# =============================================================================
# 5. ORCHESTRATION
# =============================================================================

def run(args: argparse.Namespace) -> int:
    """Do a complete fit and write every artifact (D11)."""
    import sklearn
    import umap

    TOPICS.mkdir(parents=True, exist_ok=True)
    run_name, slug = namespaces(args.sample, args.model)
    out = paths_for(slug)
    started = time.time()

    log(f"=== run '{run_name}' -> {out['docs'].name} ===")
    log(f"HF_HOME={os.environ['HF_HOME']}")

    docs = build_documents(tuple(args.languages), tuple(args.year_range),
                           args.sample, args.drop_flags)
    if args.dry_run:
        log(f"DRY RUN: would fit {len(docs):,} documents; no artifacts written")
        log(f"  languages {'/'.join(args.languages)} | years {args.year_range[0]}-"
            f"{args.year_range[1]} | model {args.model}")
        log(f"  sample={args.sample} min_topic_size={args.min_topic_size} "
            f"nr_topics={args.nr_topics} drop_flags={args.drop_flags}")
        log(f"  would write: {out['docs'].name}, {out['report'].name}, …")
        return 0

    t_embed = time.perf_counter()
    embeddings = embed(docs["natural"].tolist(), args.model, args.device,
                       args.batch_size, out["embeddings"], args.resume)
    embed_secs = time.perf_counter() - t_embed

    # The clustering space is cached on its own (D7): a re-cluster with different
    # HDBSCAN parameters, or a sweep, must not cost another UMAP run.
    reduced = reduce_space(embeddings, out["reduced5"], args.resume)

    # min_cluster_size is a proportion of the corpus (see DEFAULT_MIN_TOPIC_SIZE), so
    # a fraction is resolved against this run's actual document count.
    min_topic_size = args.min_topic_size
    if args.min_topic_size_frac:
        min_topic_size = max(2, round(args.min_topic_size_frac * len(docs)))
        log(f"  min_topic_size {min_topic_size:,} = {args.min_topic_size_frac:.3%} "
            f"of {len(docs):,} documents")

    topic_model, topics = fit(docs["doc"].tolist(), reduced,
                              min_topic_size=min_topic_size,
                              min_samples=args.min_samples,
                              cluster_selection_epsilon=args.cluster_selection_epsilon,
                              cluster_selection_method=args.cluster_selection_method,
                              nr_topics=args.nr_topics, model_name=args.model)
    docs["topic"] = topics

    xy = reduce_2d(reduced, out["xy"], args.resume)
    docs["x"] = xy[:, 0]
    docs["y"] = xy[:, 1]

    log("aggregating …")
    info = topics_info(topic_model, docs)
    time_df = over_time(docs, BIN_LENGTH)
    class_df = per_class(docs)

    outlier_share = float((docs["topic"] == -1).mean())
    n_topics = int((info["Topic"] != -1).sum())
    log(f"  {n_topics} topics | {outlier_share:.1%} outliers")

    docs_out = docs[["doc_id", "topic", "x", "y", "n_lemmas"] + DOC_COLUMNS]
    docs_out.to_parquet(out["docs"], engine="pyarrow", index=False)
    info.to_parquet(out["topics_info"], engine="pyarrow", index=False)
    time_df.to_parquet(out["over_time"], engine="pyarrow", index=False)
    if not class_df.empty:
        class_df.to_parquet(out["per_class"], engine="pyarrow", index=False)

    log(f"saving the BERTopic model to {out['model'].name} …")
    # save_embedding_model=False: the encoder is a HuggingFace repo id, and writing it out
    # would copy ~470 MB into the project for no benefit — the model name is recorded in
    # report.json and the embeddings are cached separately as .npy.
    topic_model.save(str(out["model"]), serialization="safetensors",
                     save_ctfidf=True, save_embedding_model=False)

    report = {
        "run_name": run_name,
        "slug": slug,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_seconds": round(time.time() - started, 1),
        "embedding_seconds": round(embed_secs, 1),
        "host": platform.node(),
        "python": platform.python_version(),
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"),
        "documents": {
            "fitted": int(len(docs)),
            "articles": int(docs["item_id"].nunique()),
            "journals": int(docs["journal"].nunique()),
            "year_min": int(docs["year"].min()),
            "year_max": int(docs["year"].max()),
            "languages": {k: int(v) for k, v in docs["language"].value_counts().items()},
            "lemmas_per_doc_mean": round(float(docs["n_lemmas"].mean()), 2),
        },
        "decisions": {
            "D1_unit": "+/-10-token lemmatised window per concordance hit",
            "D2_languages": list(args.languages),
            "D3_year_range": list(args.year_range),
            "D4_dropped_no_content_lemma": True,
            "D5_docs": "lemmatised window (c-TF-IDF)",
            "D5_embeddings": "natural KWIC window",
            "D6_embedding_model": args.model,
            "D8_mode": "unsupervised (no seed topics, no zero-shot)",
            "D9_outliers": "retained as topic -1, never reassigned",
            "D10_dropped_flags": args.drop_flags or [],
            "D12": "shares per bin, not raw counts",
        },
        "topics": {
            "n_topics": n_topics,
            "outlier_topic_docs": int((docs["topic"] == -1).sum()),
            "outlier_share": round(outlier_share, 4),
            "min_topic_size": min_topic_size,
            "min_topic_size_frac": (round(min_topic_size / len(docs), 5)
                                    if args.min_topic_size_frac else None),
            "min_samples": args.min_samples,
            "cluster_selection_epsilon": args.cluster_selection_epsilon,
            "cluster_selection_method": args.cluster_selection_method,
            "nr_topics": args.nr_topics,
            "bin_length": BIN_LENGTH,
        },
        "umap": {k: str(v) for k, v in UMAP_PARAMS.items()},
        "umap_2d": {k: str(v) for k, v in UMAP_2D_PARAMS.items()},
        "hdbscan": {"min_cluster_size": min_topic_size,
                    "min_samples": args.min_samples,
                    "cluster_selection_epsilon": args.cluster_selection_epsilon,
                    "cluster_selection_method": args.cluster_selection_method,
                    "prediction_data": False},
        "vectorizer": {"ngram_range": [1, 1], "min_df": 1},
        "embedding_prefix": embedding_prefix(args.model),
        "versions": {"bertopic": _version("bertopic"), "sentence-transformers": _version("sentence-transformers"),
                     "transformers": _version("transformers"), "torch": _version("torch"),
                     "numpy": np.__version__, "pandas": pd.__version__,
                     "umap-learn": _version("umap-learn"), "scikit-learn": sklearn.__version__,
                     "hdbscan": _version("hdbscan")},
        "seed": SEED,
        "artifacts": {k: str(v) for k, v in out.items()},
    }
    out["report"].write_text(json.dumps(report, indent=2), encoding="utf-8")

    log(f"wrote {out['docs'].name} ({out['docs'].stat().st_size / 1e6:,.0f} MB), "
        f"{out['topics_info'].name}, {out['over_time'].name}, {out['per_class'].name}")
    log(f"  report: {out['report'].name} | total {report['runtime_seconds']:.0f}s")
    _print_topics(info)
    return 0


def _version(pkg: str) -> str:
    import importlib.metadata as md

    try:
        return md.version(pkg)
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return "?"


def _print_topics(info: pd.DataFrame, n: int = 25) -> None:
    """Print the topic table — the thing a human actually has to judge."""
    log(f"--- {len(info)} topics (largest first) ---")
    for _, r in info.sort_values("Count", ascending=False).head(n).iterrows():
        tag = "OUTLIER" if r["Topic"] == -1 else f"t{r['Topic']}"
        terms = str(r.get("top_terms_str", ""))[:66]
        print(f"    {tag:>8} n={r['Count']:>6,}  {terms}")


# =============================================================================
# 6. VERIFY
# =============================================================================

def verify(slug: str = "full") -> int:
    """Check an existing run's artifacts for internal consistency.

    Independent of the fit: catches a truncated write, a stale embedding cache
    reused for the wrong run, or a topic id in `docs.parquet` that `topics_info`
    does not describe.
    """
    out = paths_for(slug)
    ok = True

    def check(cond: bool, msg: str) -> None:
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'} {msg}")
        ok = ok and bool(cond)

    print(f"verifying run '{slug}'")
    for key in ("docs", "topics_info", "over_time", "report"):
        check(out[key].exists(), f"{out[key].name} exists")
    if not out["docs"].exists():
        return 1

    docs = pd.read_parquet(out["docs"], engine="pyarrow")
    info = pd.read_parquet(out["topics_info"], engine="pyarrow")
    ttime = pd.read_parquet(out["over_time"], engine="pyarrow")
    report = json.loads(out["report"].read_text(encoding="utf-8"))

    check(len(docs) == report["documents"]["fitted"],
          f"docs rows {len(docs):,} == report {report['documents']['fitted']:,}")
    check(docs["doc_id"].is_unique and docs["doc_id"].min() == 0
          and docs["doc_id"].max() == len(docs) - 1,
          "doc_id is a dense 0..n-1 index")
    check(set(docs["topic"].unique()) <= set(info["Topic"]),
          "every topic in docs.parquet is described by topics_info")
    check(int((docs["topic"] == -1).sum()) == report["topics"]["outlier_topic_docs"],
          "outlier count matches the report")

    if out["embeddings"].exists():
        n = np.load(out["embeddings"], mmap_mode="r").shape[0]
        check(n == len(docs), f"embeddings rows {n:,} == docs rows {len(docs):,}")

    # A stale cache reused for the wrong run is the failure this catches: the
    # clustering space must describe exactly the fitted documents.
    if out["reduced5"].exists():
        rd = np.load(out["reduced5"], mmap_mode="r")
        check(rd.shape[0] == len(docs) and rd.shape[1] == int(UMAP_PARAMS["n_components"]),
              f"reduced 5-d space {rd.shape} matches {len(docs):,} documents")

    if out["xy"].exists():
        xy = np.load(out["xy"], mmap_mode="r")
        check(xy.shape == (len(docs), 2),
              f"2-d coordinates {xy.shape} match {len(docs):,} documents")
        check(bool(np.isfinite(xy).all()), "2-d coordinates are all finite")

    # shares must reconstruct the per-bin totals
    per_bin = ttime.groupby("bin")["share"].sum()
    check(bool(np.allclose(per_bin, 1.0, atol=1e-6)),
          "over-time shares sum to 1 per bin")

    # the join the app relies on
    if HITS.exists():
        hits = pd.read_parquet(HITS, columns=["item_id", "char_start"], engine="pyarrow")
        joined = docs[["item_id", "char_start"]].merge(
            hits.drop_duplicates(["item_id", "char_start"]), how="inner",
            on=["item_id", "char_start"])
        check(len(joined) == len(docs),
              f"all {len(docs):,} documents join to hits.parquet ({len(joined):,})")

    check(all(k in report for k in ("decisions", "versions", "topics", "artifacts")),
          "report carries decisions, versions, topics and artifact paths")

    print("VERIFY OK" if ok else "VERIFY FAILED")
    return 0 if ok else 1


# =============================================================================
# 7. PILOT SWEEP — the evidence for D8's min_cluster_size (TOPICS.md §8)
# =============================================================================

#: Grid for the min_cluster_size decision. HDBSCAN's parameters are *proportions*
#: of the corpus, so the sweep runs on a sample and the chosen fraction is scaled
#: up. These sizes span 0.05%-0.4% of a 100k sample.
SWEEP_MIN_TOPIC_SIZES = (50, 100, 200, 400)
SWEEP_MIN_SAMPLES = (1, 5)
SWEEP_EPSILONS = (0.0, 0.1)
#: `leaf` selection gives finer topics than `eom`. It is run at one size only, as a
#: documented alternative to the primary grid rather than as a second full grid.
SWEEP_LEAF_SIZE = 200


def sweep(args: argparse.Namespace) -> int:
    """Fit a parameter grid and write the evidence to `pilot_sweep<slug>.json`.

    Why a side-file rather than the run artifacts: pilots of the same size share a
    slug, so they write the same filenames and overwrite each other — only the last
    grid point would survive. That is precisely how the earlier `min_samples`
    finding (43% outliers at 1 versus 54% at 5) ended up recorded in a code comment
    instead of a record. Here every configuration, including the rejected ones, is
    kept with its numbers, so the choice made in `TOPICS.md` §8 can be re-checked
    without re-running anything.

    The sweep fits the *same* pipeline as a production run: same documents, same
    embeddings, same cached 5-d UMAP space. Only HDBSCAN and the c-TF-IDF labels
    are recomputed, which is exactly the scope of the parameters being tuned.
    """
    TOPICS.mkdir(parents=True, exist_ok=True)
    _run_name, slug = namespaces(args.sample, args.model)
    out = paths_for(slug)
    started = time.time()

    log(f"=== sweep '{slug}' ===")
    docs = build_documents(tuple(args.languages), tuple(args.year_range),
                           args.sample, args.drop_flags)
    embeddings = embed(docs["natural"].tolist(), args.model, args.device,
                       args.batch_size, out["embeddings"], args.resume)
    reduced = reduce_space(embeddings, out["reduced5"], args.resume)
    texts = docs["doc"].tolist()
    n = len(texts)

    grid = [dict(min_topic_size=mcs, min_samples=ms,
                 cluster_selection_epsilon=eps, cluster_selection_method="eom")
            for mcs in args.sweep_min_topic_sizes
            for ms in args.sweep_min_samples
            for eps in args.sweep_epsilons]
    if args.sweep_leaf_size:
        grid.append(dict(min_topic_size=args.sweep_leaf_size, min_samples=1,
                         cluster_selection_epsilon=0.1,
                         cluster_selection_method="leaf"))

    results = []
    for cfg in grid:
        model, topics = fit(texts, reduced, nr_topics=None, model_name=args.model,
                            verbose=False, **cfg)
        # `get_topic_info()` already carries the per-topic document counts, so the
        # sweep does not need the merged documents frame for every configuration.
        info = model.get_topic_info()
        info["top_terms_str"] = info["Topic"].map(
            lambda t: ", ".join(w for w, _ in (model.get_topic(t) or [])))
        sizes = info.loc[info["Topic"] != -1, "Count"].sort_values(ascending=False)
        biggest = info.sort_values("Count", ascending=False).head(8)
        entry = {
            **cfg,
            "min_topic_size_share": round(cfg["min_topic_size"] / n, 5),
            "n_topics": int(len(sizes)),
            "outlier_share": round(float((topics == -1).mean()), 4),
            "largest_topic": int(sizes.iloc[0]) if len(sizes) else 0,
            "median_topic_size": int(sizes.median()) if len(sizes) else 0,
            "topics_over_1pct": int((sizes / n > 0.01).sum()),
            "top_terms": {int(r["Topic"]): r["top_terms_str"] for _, r in
                          biggest.iterrows() if r["Topic"] != -1},
        }
        results.append(entry)
        log(f"  mcs={cfg['min_topic_size']:>4} ms={cfg['min_samples']} "
            f"eps={cfg['cluster_selection_epsilon']:.1f} "
            f"{cfg['cluster_selection_method']:>4} -> {entry['n_topics']:>4} topics | "
            f"outliers {entry['outlier_share']:>6.1%} | largest {entry['largest_topic']:,}")
        del model, info

    payload = {
        "slug": slug,
        "sample": args.sample,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_seconds": round(time.time() - started, 1),
        "documents": int(n),
        "languages": list(args.languages),
        "year_range": list(args.year_range),
        "embedding_model": args.model,
        "space": "UMAP 384d->5d (cached; identical for every configuration)",
        "umap": {k: str(v) for k, v in UMAP_PARAMS.items()},
        "grid": results,
    }
    path = TOPICS / f"pilot_sweep{'' if slug == 'full' else '_' + slug}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"wrote {path.name} ({len(results)} configurations, "
        f"{payload['runtime_seconds']:.0f}s)")
    return 0


# =============================================================================
# CLI
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=None,
                    help="fit a random subsample (writes to a separate artifact namespace)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model")
    ap.add_argument("--languages", nargs="+", default=list(DEFAULT_LANGUAGES),
                    help="JSTOR language codes (D2; 'en' and 'eng' both mean English)")
    ap.add_argument("--year-from", type=int, default=DEFAULT_YEAR_RANGE[0])
    ap.add_argument("--year-to", type=int, default=DEFAULT_YEAR_RANGE[1])
    ap.add_argument("--min-topic-size", type=int, default=DEFAULT_MIN_TOPIC_SIZE,
                    help="HDBSCAN min_cluster_size (D8) — fixed by the pilot sweep; "
                         "380 is 0.100%% of the 379,859-document fit set")
    ap.add_argument("--min-topic-size-frac", type=float, default=None,
                    help="set min_cluster_size as a fraction of the documents actually "
                         "fitted (e.g. 0.001); overrides --min-topic-size and is the "
                         "safe way to carry the D8 decision from a pilot to the full fit")
    ap.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES,
                    help="HDBSCAN min_samples; the sweep found 1 better than 5 at every "
                         "size (D8)")
    ap.add_argument("--cluster-selection-epsilon", type=float,
                    default=DEFAULT_CLUSTER_SELECTION_EPSILON,
                    help="HDBSCAN cluster_selection_epsilon. An ABSOLUTE distance in the reduced "
                         "space, so it does NOT transfer between sample sizes: 0.1 is benign at "
                         "100k but merges a 200k-document blob at 380k. Sweep it at the target "
                         "scale before raising it (D8.1)")
    ap.add_argument("--cluster-selection-method", default="eom", choices=["eom", "leaf"],
                    help="HDBSCAN cluster_selection_method; 'leaf' yields finer topics")
    ap.add_argument("--nr-topics", default=None,
                    help="merge down to this many topics ('auto' to let BERTopic decide)")
    ap.add_argument("--drop-flags", nargs="*", default=None,
                    help="exclude hits carrying these scan flags (D10)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--resume", action="store_true",
                    help="reuse the cached embeddings and only re-cluster")
    ap.add_argument("--sweep", action="store_true",
                    help="fit the D8 parameter grid and write pilot_sweep<slug>.json "
                         "(a side-file — the run artifacts are not touched)")
    ap.add_argument("--sweep-min-topic-sizes", nargs="+", type=int,
                    default=list(SWEEP_MIN_TOPIC_SIZES))
    ap.add_argument("--sweep-min-samples", nargs="+", type=int,
                    default=list(SWEEP_MIN_SAMPLES))
    ap.add_argument("--sweep-epsilons", nargs="+", type=float,
                    default=list(SWEEP_EPSILONS))
    ap.add_argument("--sweep-leaf-size", type=int, default=SWEEP_LEAF_SIZE,
                    help="one `cluster_selection_method='leaf'` run at this size "
                         "(0 to skip)")
    ap.add_argument("--dry-run", action="store_true", help="build documents and report, fit nothing")
    ap.add_argument("--verify", action="store_true", help="check an existing run and exit")
    ap.add_argument("--verify-slug", default="full", help="which run --verify inspects")
    args = ap.parse_args(argv)
    args.year_range = (args.year_from, args.year_to)
    if args.nr_topics not in (None, "auto"):
        args.nr_topics = int(args.nr_topics)

    if args.verify:
        return verify(args.verify_slug)
    if args.sweep:
        return sweep(args)
    try:
        return run(args)
    except KeyboardInterrupt:
        log("interrupted — artifacts are written only at the end of a run, so nothing is partial")
        return 130


if __name__ == "__main__":
    sys.exit(main())
