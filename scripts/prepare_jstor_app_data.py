#!/usr/bin/env python3
"""Reduce the Paul-scan NLP parquet to the slim table the streamlit app serves.

Why this exists
---------------
`data/large_files/paul_scan/hits_nlp.parquet` (426,917 x 20) is the *analysis*
artifact of `4_concordance-tokens-explore.ipynb`: it carries, per hit, the raw
KWIC, the sentence, and `kwic_nlp` — a `list<list<struct>>` holding a full spaCy
token record (text, lemma, pos, tag, dep, is_alpha, is_stop, is_match) for every
token of the hit's sentences. That is ~90 nested values per row.

The app only ever needs two derived things from that structure:

* `filtered_lemmata` — the lemmatised, POS/stopword-filtered bag of words around
  the hit, i.e. exactly what `5_tokens-analysis.ipynb` feeds into its frequency
  dicts and temporal lineplots;
* the flat hit metadata (year, journal, language, tier, pattern, flags,
  annotations) that the app filters on.

Reading the nested column is the slow part of every app start, so it is
flattened once, here, into `tokens.parquet` (flat columns only). The two
`extract_*` functions are ported *verbatim* from the notebook so the flattened
table reproduces the notebook numbers exactly.

`journal-cat` / `journal-discipline` / `journal-confession` are already mapped
into `hits_nlp.parquet` (from the `journals_annotated` Google Sheet), so the
app needs no Google credentials and no second data source. About 35% of rows
belong to journals the sheet does not cover; those keep `NaN` annotations, and
the app treats "no filter selected" as "keep everything", so they are included
by default — the same behaviour as the notebook.

Usage
-----
    python3 prepare_jstor_app_data.py              # write tokens.parquet
    python3 prepare_jstor_app_data.py --dry-run    # report, write nothing
    python3 prepare_jstor_app_data.py --verify     # check an existing output
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# The scan outputs are mops-specific and live inside the project (gitignored).
MOPS = Path(__file__).resolve().parents[1]
SCAN = MOPS / "data" / "large_files" / "paul_scan"
SRC = SCAN / "hits_nlp.parquet"
OUT = SCAN / "tokens.parquet"

#: Columns kept in `tokens.parquet`, in order. Everything else in the source
#: (KWIC windows, sentence, char offsets, `kwic_nlp`, `kwic_tokens`, `doc_len`)
#: is dropped: the app reads those straight from `hits.parquet` when needed.
#: `char_start` is kept as the join key back to `hits.parquet` — a lemma match
#: found here can then be displayed with its KWIC text.
KEEP_COLUMNS = [
    "item_id",
    "char_start",
    "year",
    "bidecade",
    "journal",
    "language",
    "tier",
    "pattern",
    "flags",
    "journal-cat",
    "journal-discipline",
    "journal-confession",
    "filtered_lemmata",
]

#: POS tags kept by the notebook's context filter.
KEEP_POS = ("NOUN", "VERB", "ADJ", "PROPN")

#: Columns read from the source parquet (a subset of its 20, to keep the read
#: cheap — the nested `kwic_nlp` is the only expensive one).
READ_COLUMNS = [
    "item_id",
    "char_start",
    "year",
    "bidecade",
    "journal",
    "language",
    "tier",
    "pattern",
    "flags",
    "journal-cat",
    "journal-discipline",
    "journal-confession",
    "kwic_nlp",
]


# =============================================================================
# NOTEBOOK PORT (verbatim: 4_concordance-tokens-explore / 5_tokens-analysis)
# =============================================================================

def extract_kwic_10target10(nlp_tokens: list) -> list:
    """Flatten the hit's sentences and cut 10 tokens around the match span.

    Ported verbatim from `5_tokens-analysis.ipynb`. `nlp_tokens` is the
    per-hit `kwic_nlp` value: a list of sentences, each a list of token dicts.
    The match itself may be several tokens ("apostle Paul"), so the window
    spans from 10 tokens before the *first* match token to 10 tokens after
    the *last*. Returns `[]` for malformed rows instead of raising.
    """
    try:
        tokens_flat = [token for sent_data in nlp_tokens for token in sent_data]
        target_indices = [n for n, token in enumerate(tokens_flat) if token["is_match"]]
        return tokens_flat[target_indices[0] - 10:target_indices[-1] + 11]
    except Exception:
        return []


def extract_filtered_lemmatized(kwic_10target10: list) -> list:
    """Lemmatise + filter the context window to content words.

    Ported verbatim from `5_tokens-analysis.ipynb`: keep alphabetic, non-stop
    tokens whose POS is a content tag, excluding the target match itself.
    Comments in the notebook note two *disabled* conditions (dropping
    "be/have/do" and lemmas shorter than three characters); they are kept
    disabled here so this stays byte-comparable with the notebook.
    """
    try:
        return [
            token["lemma"] for token in kwic_10target10
            if (
                token["is_alpha"]
                and not token["is_stop"]
                and not token["is_match"]
                and token["pos"] in KEEP_POS
            )
        ]
    except Exception:
        return []


# =============================================================================
# BUILD
# =============================================================================

def build(src: Path = SRC) -> pd.DataFrame:
    """Read the nested source and return the flat `tokens` frame.

    The two stages are the notebook's: flatten `kwic_nlp` to a 10+target+10
    window, then reduce that window to lemmas. Both are applied row-wise; on
    426,917 rows this is a few seconds.
    """
    if not src.exists():
        raise SystemExit(f"source not found: {src}\n"
                         f"build it with 4_concordance-tokens-explore.ipynb first")
    print(f"reading {src} ...")
    df = pd.read_parquet(src, columns=READ_COLUMNS, engine="pyarrow")
    print(f"  {len(df):,} rows x {df.shape[1]} columns")

    df["kwic_10target10"] = df["kwic_nlp"].apply(extract_kwic_10target10)
    df["filtered_lemmata"] = df["kwic_10target10"].apply(extract_filtered_lemmatized)
    df = df.drop(columns=["kwic_nlp", "kwic_10target10"])
    out = df[KEEP_COLUMNS]
    key = ["item_id", "char_start"]
    if out.duplicated(key).any():
        raise SystemExit(f"key {key} is not unique — the concordance join would be wrong")
    print(f"  join key {key} is unique over {len(out):,} rows")
    return out


def report(df: pd.DataFrame) -> None:
    """Print the audit numbers that make a silent regression visible."""
    n_lemmas = df["filtered_lemmata"].apply(len)
    print(f"\n  rows              {len(df):,}")
    print(f"  items             {df['item_id'].nunique():,}")
    print(f"  journals          {df['journal'].nunique():,}")
    print(f"  years             {int(df['year'].min())}–{int(df['year'].max())}")
    print(f"  empty lemma rows  {int((n_lemmas == 0).sum()):,} "
          f"({(n_lemmas == 0).mean():.2%})  [expect a few % — hits with no content words]")
    print(f"  lemmas per hit    mean {n_lemmas.mean():.1f}, max {int(n_lemmas.max())} "
          f"[expect ~mean 8–12, max ~20]")
    print(f"  annotated rows    discipline {df['journal-discipline'].notna().sum():,} "
          f"/ confession {df['journal-confession'].notna().sum():,} "
          f"({df['journal-discipline'].notna().mean():.1%})")


def verify(path: Path = OUT) -> int:
    """Check an existing `tokens.parquet`; non-zero exit on a defect."""
    if not path.exists():
        print(f"MISSING: {path}")
        return 1
    ok = True
    df = pd.read_parquet(path, engine="pyarrow")
    print(f"{path}  {len(df):,} rows x {df.shape[1]} columns")
    print(f"  columns: {list(df.columns)}")
    if list(df.columns) != KEEP_COLUMNS:
        print("  FAIL: column set/order differs from KEEP_COLUMNS")
        ok = False
    nested = [c for c in df.columns if df[c].dtype == object and isinstance(df[c].iloc[0], list)]
    if set(nested) != {"filtered_lemmata"}:
        print(f"  FAIL: unexpected list columns {nested} (only filtered_lemmata expected)")
        ok = False
    if df["filtered_lemmata"].apply(len).max() > 21:
        print("  FAIL: a window longer than 21 tokens — did the 10+target+10 cut break?")
        ok = False
    report(df)
    print("VERIFY OK" if ok else "VERIFY FAILED")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=SRC)
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--dry-run", action="store_true", help="build and report, write nothing")
    ap.add_argument("--verify", action="store_true", help="check an existing output and exit")
    args = ap.parse_args(argv)

    if args.verify:
        return verify(args.output)

    t0 = time.time()
    df = build(args.source)
    elapsed = time.time() - t0
    report(df)

    if args.dry_run:
        print(f"\ndry run: would write {args.output} ({len(df):,} rows) in {elapsed:.1f} s")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, engine="pyarrow", index=False)
    size_mb = args.output.stat().st_size / 1e6
    print(f"\nwrote {args.output} ({size_mb:.1f} MB, {len(df):,} rows) in {elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
