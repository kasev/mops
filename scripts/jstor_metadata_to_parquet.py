#!/usr/bin/env python3
"""Flatten the 13.7 GB JSTOR metadata JSONL into partitioned Parquet.

Why this exists
---------------
`jstor_metadata_2026-04-28.jsonl` is the *unfiltered* ITHAKA delivery: one JSON
object per line, 12,616,405 lines, 13.7 GB, 24 top-level fields (one of them,
`identifiers`, is a fixed-schema nested object). Every question about scope
("how many articles? how many languages? which disciplines?") currently costs a
scan of the JSONL, and `grep -c` over it takes ~40 s per pattern (single core,
C locale). Parsing the whole file once into columnar Parquet turns those
questions into sub-second reads.

It is a *faithful* flattening, not a filter: no row is dropped and no field is
renamed except the expansion of `identifiers` (documented below). The subset
used for the Pauline-scholarship study is a *view* on top of this Parquet
(see `jstor_metadata_2026-04-28-filtered.csv`), not a separate universe.

Row identity
------------
One line == one line of the JSONL == one `item_id`. This is the *item* level,
which is finer than the "article" level: a journal issue's table of contents,
a single book chapter, an appendix of a research report and an audio recording
are each their own item. That is why the raw item count (12.6M) is far larger
than the 384,829 filtered *articles* used by the corpus study.

Schema produced (per row)
-------------------------
* all top-level fields verbatim (`content_type`, `discipline_names`,
  `languages`, `is_part_of`, `creators`, ...) except the `identifiers` struct;
* the 8 keys of `identifiers` promoted to columns of the same name
  (`print_isbn`, `online_isbn`, `print_issn`, `online_issn`, `ssid`, `catsid`,
  `journal_code`, `aluka_doi`). They are used to join against external
  bibliographies, so they stay as separate columns rather than one struct; the
  parent struct is then dropped (its keys are all now columns, and its
  alphabetical field order made an exact Arrow struct cast brittle);
* derived convenience columns, each documented at `add_derived_columns`:
  `year`, `n_creators`, `n_languages`, `n_disciplines`, `primary_discipline`,
  `primary_language`, `is_journal_article`, `has_journal_code`.

Normalisation decisions (and why)
---------------------------------
* `published_date` is kept verbatim *and* a `year` int is derived. The raw value
  is not a clean date ("2011-01-0", "2011", sometimes with a trailing dash), so
  the year is taken with a leading 4-digit regex and left null (NaN) when it
  fails rather than guessed. Nothing is coerced away: `published_date` is the
  record of truth, `year` is a convenience.
* Missing collections are `null` in the JSONL; pandas turns those into `None`
  inside object columns. No sentinel strings are invented and no `None` is
  rewritten to `[]`, so "field absent" stays distinguishable from "field empty".
* List-valued fields (`languages`, `discipline_names`, `collections`,
  `publishers`, `creators`) are written as Arrow lists, which preserves their
  multiplicity -- an item with two disciplines is not silently truncated to one.
  `creators` is a list of `{first_name, last_name, order}` structs and is kept
  intact; `n_creators` is a derived count for cheap filtering.
* No deduplication and no `review_required` filtering are applied: `grep` shows
  11,046,561 of 12,616,405 rows carry `review_required:true`, so filtering on it
  would silently shrink the corpus. Rows are written exactly as delivered; the
  reader decides what to exclude.

Output layout
-------------
    metadata_parquet/part-00000.parquet   (1,000,000 rows each)
    metadata_parquet/part-00001.parquet
    ...
    metadata_parquet/_manifest.json       row counts, part list, source stats

Partitioning by row count rather than by year/discipline keeps the writer
memory-bounded (~1M rows while the pyarrow table is alive, on a 125 GB /
32-core box) and lets an interrupted run resume: existing parts are skipped
and the next part index is derived from the manifest.

Usage
-----
    python3 jstor_metadata_to_parquet.py                 # build (resumable)
    python3 jstor_metadata_to_parquet.py --verify        # check row counts only
    python3 jstor_metadata_to_parquet.py --rows 500000   # smaller parts
    python3 jstor_metadata_to_parquet.py --dry-run       # one part, then stop

Reading the result
------------------
    import pyarrow.dataset as ds
    d = ds.dataset("metadata_parquet", format="parquet")
    d.to_table(filter=ds.field("content_type") == "article").num_rows
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
# The delivery directory /srv/data/jstor/jstor2026-04/ is root-owned and read-only
# for the research account, so the source is read there and the Parquet is written
# into the repo's gitignored data/large_files/ (on the 595 GB-free home volume).
JSTOR_DIR = Path("/srv/data/jstor/jstor2026-04")
SOURCE = JSTOR_DIR / "jstor_metadata_2026-04-28.jsonl"
OUT_DIR = HERE.parent / "data" / "large_files" / "jstor_metadata_parquet"
CHUNK_ROWS = 100_000           # pandas parse granularity
DEFAULT_PART_ROWS = 1_000_000  # parquet file granularity

IDENTIFIER_KEYS = [
    "print_isbn", "online_isbn", "print_issn", "online_issn",
    "ssid", "catsid", "journal_code", "aluka_doi",
]

# Scalar text columns are pinned to `object` at parse time. Without this, pandas
# infers each 100k-row chunk's dtype independently: a chunk whose `issue_volume`
# happens to contain only numeric-looking strings is silently cast to float64
# (`"1"` -> `1.0`), which both corrupts the value and makes that part's Arrow
# schema disagree with the others. Pinning only these scalar columns (not the
# whole frame) leaves the list/struct columns to infer as real lists/dicts.
#
# `object` rather than `str`: pandas' `dtype=str` re-fills missing values with
# the *literal string* "None" (turning JSON null into a real value), whereas
# `object` keeps null as None/NaN and still blocks the numeric inference.
TEXT_COLS = [
    "item_id", "ithaka_doi", "title", "is_part_of", "creators_string",
    "published_date", "issue_number", "issue_volume", "c5_data_type",
    "c5_section_type", "content_type", "content_subtype", "ccda_resource_type",
    "ccda_resource_subtype", "licensing_status", "url",
]
BOOL_COLS = ["review_required", "contributed_content"]
PARSE_DTYPE = {**{c: "object" for c in TEXT_COLS},
               **{c: bool for c in BOOL_COLS}}

# Canonical schema every part is cast to. Guarantees identical schemas across
# parts (so pyarrow.dataset can union them) even when a single part happens to
# have an all-null column, which would otherwise be inferred as Arrow `null`.
TARGET_SCHEMA = pa.schema(
    [(c, pa.string()) for c in TEXT_COLS]
    + [("review_required", pa.bool_()), ("contributed_content", pa.bool_())]
    + [("creators", pa.list_(pa.struct(
        [("first_name", pa.string()), ("last_name", pa.string()),
         ("order", pa.int64())])))
       ]
    + [(c, pa.list_(pa.string()))
       for c in ("publishers", "languages", "discipline_names", "collections")]
    + [(k, pa.string()) for k in IDENTIFIER_KEYS]
    + [("year", pa.int16()), ("n_creators", pa.int16()),
       ("n_languages", pa.int16()), ("n_disciplines", pa.int16()),
       ("primary_discipline", pa.string()), ("primary_language", pa.string()),
       ("is_journal_article", pa.bool_()), ("has_journal_code", pa.bool_())]
)
SCHEMA_ORDER = [f.name for f in TARGET_SCHEMA]

_YEAR_RE = re.compile(r"(\d{4})")


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach convenience columns; every one is additive, none overwrites source data.

    * `year`            -- first 4-digit run in `published_date`, else NaN.
    * `n_creators`      -- len(creators) or 0, for cheap "has an author" filters.
    * `n_languages` / `n_disciplines` -- multiplicity of the list fields.
    * `primary_discipline` / `primary_language` -- first element, so grouped
      counts work without exploding the lists (the full lists remain available).
    * `is_journal_article` -- `content_type == "article"` (the corpus-study view).
    * `has_journal_code`   -- `identifiers.journal_code` is present; this is the
      link to JSTOR's journal-level metadata.
    """
    df = df.copy()  # we may be handed a slice; avoid SettingWithCopyWarning
    ids = df["identifiers"]
    for k in IDENTIFIER_KEYS:
        df[k] = ids.apply(lambda d, k=k: None if not isinstance(d, dict) else d.get(k))
        # Force string dtype: an all-null column would otherwise be inferred as
        # Arrow `null` in one part and `string` in another, and pyarrow.dataset
        # refuses to union parts whose schemas disagree.
        df[k] = df[k].astype("string")
    # Drop the nested struct: its Arrow field order is alphabetical (pandas
    # sorts dict keys) and its all-null `aluka_doi` child is typed `null`, both
    # of which make an exact struct cast brittle. Every key now lives in its own
    # column above, so nothing is lost and the schema is stable across parts.
    df = df.drop(columns=["identifiers"])

    df["year"] = pd.to_numeric(
        df["published_date"].astype("string").str.extract(_YEAR_RE)[0],
        errors="coerce",
    ).astype("Int16")

    def _n(x):
        return 0 if x is None or isinstance(x, float) else len(x)

    df["n_creators"] = df["creators"].apply(_n).astype("int16")
    df["n_languages"] = df["languages"].apply(_n).astype("int16")
    df["n_disciplines"] = df["discipline_names"].apply(_n).astype("int16")

    def _first(x):
        return x[0] if x is not None and not isinstance(x, float) and len(x) else None

    df["primary_discipline"] = df["discipline_names"].apply(_first)
    df["primary_language"] = df["languages"].apply(_first)
    df["is_journal_article"] = df["content_type"].eq("article")
    df["has_journal_code"] = df["journal_code"].notna()
    return df


def iter_parts(source: Path, part_rows: int, chunk_rows: int,
               part_index: int, start_row: int):
    """Yield (part_index, arrow_table, n_rows) for each full part-sized batch."""
    reader = pd.read_json(source, lines=True, chunksize=chunk_rows,
                          dtype=PARSE_DTYPE)
    buf: list[pd.DataFrame] = []
    n_buf = 0
    skipped = 0
    for chunk in reader:
        if skipped < start_row:                     # resume: drop already-written rows
            take = min(chunk_rows, start_row - skipped)
            chunk = chunk.iloc[take:]
            skipped += take
            if chunk.empty:
                continue
        buf.append(chunk)
        n_buf += len(chunk)

        def _emit(rows_df: pd.DataFrame) -> pa.Table:
            t = pa.Table.from_pandas(add_derived_columns(rows_df),
                                     preserve_index=False)
            return t.select(SCHEMA_ORDER).cast(TARGET_SCHEMA)

        while n_buf >= part_rows:
            all_rows = pd.concat(buf, ignore_index=True)
            head, tail = all_rows.iloc[:part_rows], all_rows.iloc[part_rows:]
            yield part_index, _emit(head), len(head)
            part_index += 1
            buf = [tail] if not tail.empty else []
            n_buf = len(tail)
    if n_buf:
        all_rows = pd.concat(buf, ignore_index=True)
        yield part_index, _emit(all_rows), len(all_rows)


def count_source_rows(source: Path) -> int:
    n = 0
    with source.open("rb") as fh:
        for _ in fh:
            n += 1
    return n


def load_manifest(out_dir: Path) -> dict:
    p = out_dir / "_manifest.json"
    return json.loads(p.read_text()) if p.exists() else {"parts": []}


def verify(out_dir: Path) -> int:
    man = load_manifest(out_dir)
    parts = sorted(out_dir.glob("part-*.parquet"))
    total = sum(pq.ParquetFile(p).metadata.num_rows for p in parts)
    src = count_source_rows(SOURCE)
    print(f"parts on disk : {len(parts)} ({total:,} rows)")
    print(f"manifest      : {len(man.get('parts', []))} parts, "
          f"{man.get('rows_written', 0):,} rows")
    print(f"source lines  : {src:,}")
    ok = total == src == man.get("rows_written")
    print("OK: row counts agree" if ok else "MISMATCH: row counts disagree")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--rows", type=int, default=DEFAULT_PART_ROWS,
                    help="rows per parquet part (default 1,000,000)")
    ap.add_argument("--chunk", type=int, default=CHUNK_ROWS,
                    help="pandas JSON parse chunk (default 100,000)")
    ap.add_argument("--verify", action="store_true", help="check counts, write nothing")
    ap.add_argument("--dry-run", action="store_true", help="write one part, then stop")
    args = ap.parse_args()

    if args.verify:
        return verify(args.out)

    args.out.mkdir(parents=True, exist_ok=True)
    man = load_manifest(args.out)
    done = {p["part"]: p for p in man.get("parts", [])}
    start_row = man.get("rows_written", 0)
    next_part = max(done, default=-1) + 1

    if start_row:
        print(f"resuming after {start_row:,} rows / {next_part} parts")

    t0 = time.time()
    written = start_row
    n_parts = len(done)
    for idx, table, n in iter_parts(args.source, args.rows, args.chunk,
                                    next_part, start_row):
        path = args.out / f"part-{idx:05d}.parquet"
        pq.write_table(table, path, compression="snappy")
        written += n
        n_parts += 1
        done[idx] = {"part": idx, "file": path.name, "rows": n,
                     "cols": table.num_columns}
        el = time.time() - t0
        print(f"part {idx:05d}: {n:>9,} rows  {written:>11,} total  "
              f"{el:6.0f}s  {written/el:,.0f} rows/s", flush=True)
        (args.out / "_manifest.json").write_text(json.dumps(
            {"source": str(args.source), "parts": [done[k] for k in sorted(done)],
             "rows_written": written, "rows_per_part": args.rows}, indent=2))
        if args.dry_run:
            print("dry-run: stopping after one part")
            break

    print(f"\nwrote {written:,} rows in {n_parts} parts under {args.out} "
          f"in {time.time()-t0:.0f}s")
    if args.dry_run:
        return 0
    return verify(args.out)


if __name__ == "__main__":
    sys.exit(main())
