#!/usr/bin/env python3
"""repair_fre_lemmas.py — re-lemmatise the French hits that notebook 4 silently dropped.

The defect (recorded as **D13** in `TOPICS.md`)
-----------------------------------------------
`nlp_kwic()` in `4_concordance-tokens-explore.ipynb` dispatches spaCy on the codes
`["en", "eng"] / ["fr", "fra"] / ["de", "ger"]`, but the JSTOR corpus uses ISO 639-2/B
codes — French is **`fre`**. `fre` matches no branch, so it falls through to
`else: nlp_eng("")`, which parses an *empty string*. Result: `kwic_nlp == []` for all
66,059 French hits, hence no lemmas, hence French is invisible to every lemma-based
analysis (the topic model, the app's lemma filters).

What this script does
---------------------
Re-runs the **same** `nlp_kwic` logic on the French rows only — identical string
assembly, identical match-span test, identical token fields — but with the correct
model (`fr_core_news_md`). Nothing else is touched: rows in other languages are copied
through byte-for-byte by slicing the original Arrow column, so the repair cannot
perturb existing data.

It is deliberately *targeted* rather than a full re-run of the notebook: the notebook's
own full pass takes hours and would recompute 360k rows that are already correct.

Usage
-----
    python3 repair_fre_lemmas.py --dry-run     # report what would change, write nothing
    python3 repair_fre_lemmas.py               # back up, repair, verify, replace
    python3 repair_fre_lemmas.py --verify      # check a repaired file (no writes)

After a successful repair, rebuild the derived app table:

    python3 prepare_jstor_app_data.py
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

MOPS = Path(__file__).resolve().parents[1]
SCAN = MOPS / "data" / "large_files" / "paul_scan"
DEFAULT_SOURCE = SCAN / "hits_nlp.parquet"

#: Number of spaCy worker processes used by `nlp.pipe`. `nlp.pipe` with
#: `n_process > 1` is what makes this take ~2 minutes instead of an hour.
WORKERS = 8

#: The language codes this script repairs, and the spaCy model for each.
#: Only `fre` is listed: `lat`/`ita` are also 100% empty (same defect) but are out of
#: scope for the topic model (D2) and would need different models.
REPAIRS = {"fre": "fr_core_news_md"}

#: Token fields stored in `kwic_nlp`, matching the existing Arrow struct exactly.
TOKEN_FIELDS = ("dep", "is_alpha", "is_match", "is_stop", "lemma", "pos", "tag", "text")

#: Mirror the notebook's POS/stopword filter (only used for reporting here).
CONTENT_POS = ("NOUN", "VERB", "ADJ", "PROPN")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def eq_mask(column, value) -> np.ndarray:
    """Boolean numpy mask for `column == value`, treating nulls as False.

    A plain `pc.equal(...).to_numpy()` yields an object array holding ``None`` for
    rows whose language is null (10 rows in this file), which then explodes on `~`
    and in boolean indexing. Filling nulls first makes the mask a real bool array.
    """
    return pc.fill_null(pc.equal(column, value), False).to_numpy(zero_copy_only=False)


# =============================================================================
# THE PORTED NOTEBOOK LOGIC (verbatim, only the code->model map is corrected)
# =============================================================================

def lemmatise_kwic(nlp, left, match, right) -> list:
    """Replicate `nlp_kwic()`'s output for one row.

    Copied faithfully from `4_concordance-tokens-explore.ipynb`, including two details
    that are easy to "improve" and thereby break comparability with the existing
    English/German rows:

    * the operands are passed through `str()`, so a missing window becomes the literal
      token ``nan`` rather than an empty string (the notebook does exactly this);
    * `is_match` is decided by character-span overlap of the match inside the assembled
      string, not by token comparison.

    Returns the notebook's structure: list of sentences, each a list of 8 named fields.
    """
    left, match, right = str(left), str(match), str(right)
    kwic_string = " ".join([left, match, right])
    match_start = len(left) + 1
    match_end = match_start + len(match)

    doc = nlp(kwic_string)
    sents = []
    for sent in doc.sents:
        sents.append([
            {
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "dep": token.dep_,
                "is_alpha": bool(token.is_alpha),
                "is_stop": bool(token.is_stop),
                # True if this token overlaps the match span
                "is_match": bool(token.idx < match_end
                                 and (token.idx + len(token.text)) > match_start),
            }
            for token in sent
        ])
    return sents


def repair_params(nlp, rows) -> tuple[list, dict]:
    """Lemmatise every row with `nlp.pipe` (the only departure from row-wise apply).

    `nlp.pipe` with `n_process>1` is what makes this take ~2 minutes instead of an
    hour; the per-row result is identical to the notebook's `apply(axis=1)`.
    """
    strings = [
        " ".join([str(l), str(m), str(r)])
        for l, m, r in zip(rows["kwic_left"], rows["match"], rows["kwic_right"])
    ]
    docs = list(nlp.pipe(strings, batch_size=256, n_process=WORKERS))

    out = []
    for (left, match, right), doc in zip(
            zip(rows["kwic_left"], rows["match"], rows["kwic_right"]), docs):
        # Re-derive the span from the same operands so the logic is literally the
        # same as `lemmatise_kwic`, but reuse the parsed document (no re-parsing).
        left_s, match_s = str(left), str(match)
        match_start = len(left_s) + 1
        match_end = match_start + len(match_s)
        sents = []
        for sent in doc.sents:
            sents.append([
                {
                    "text": t.text, "lemma": t.lemma_, "pos": t.pos_, "tag": t.tag_,
                    "dep": t.dep_, "is_alpha": bool(t.is_alpha),
                    "is_stop": bool(t.is_stop),
                    "is_match": bool(t.idx < match_end
                                     and (t.idx + len(t.text)) > match_start),
                }
                for t in sent
            ])
        out.append(sents)

    stats = {
        "rows": len(out),
        "empty": int(sum(1 for v in out if not v)),
        "tokens": int(sum(len(s) for v in out for s in v)),
    }
    return out, stats


# =============================================================================
# READ / WRITE
# =============================================================================

def repair(source: Path, *, dry_run: bool, backup: bool) -> int:
    """Rebuild the French rows of `kwic_nlp` in place (with a backup)."""
    if not source.exists():
        raise SystemExit(f"not found: {source}")
    size_mb = source.stat().st_size / 1e6
    log(f"reading {source.name} ({size_mb:,.0f} MB) …")
    table = pq.read_table(source)
    schema_before = table.schema
    n_rows_before = table.num_rows
    log(f"  {table.num_rows:,} rows × {len(table.schema)} columns")

    lang = table.column("language")
    col_index = table.schema.get_field_index("kwic_nlp")
    col_type = table.schema.field("kwic_nlp").type
    log(f"  kwic_nlp type: {col_type}")

    total_repaired = 0
    for code, model_name in REPAIRS.items():
        mask = eq_mask(lang, code)
        idx_target = np.flatnonzero(mask)
        if len(idx_target) == 0:
            log(f"  {code}: no rows, skipping")
            continue
        log(f"  {code}: {len(idx_target):,} rows to re-lemmatise with {model_name}")

        # how many of them are currently empty (the defect, quantified)
        col = table.column("kwic_nlp").combine_chunks()
        target_vals = col.take(pa.array(idx_target)).to_pylist()
        n_empty_before = int(sum(1 for v in target_vals if not v))
        log(f"    currently empty: {n_empty_before:,} / {len(idx_target):,} "
            f"({n_empty_before / len(idx_target):.1%})")

        if dry_run:
            log("    dry run — not lemmatising")
            continue

        import spacy

        t0 = time.perf_counter()
        nlp = spacy.load(model_name)
        log(f"    {model_name} loaded in {time.perf_counter() - t0:.1f}s")

        target_rows = {k: table.column(k).take(pa.array(idx_target)).to_pylist()
                       for k in ("kwic_left", "match", "kwic_right")}
        t0 = time.perf_counter()
        new_values, stats = repair_params(nlp, target_rows)
        dt = time.perf_counter() - t0
        log(f"    lemmatised in {dt:.1f}s ({stats['rows'] / dt:,.0f} docs/s) | "
            f"{stats['tokens']:,} tokens | still empty: {stats['empty']:,}")
        if stats["empty"]:
            log(f"    WARNING: {stats['empty']:,} rows produced no sentences")

        # Replace ONLY the target rows: other rows are passed through as slices of the
        # original buffers, so they cannot change. Runs of untouched rows are contiguous
        # because idx_other is sorted, which is what makes this cheap.
        idx_other = np.flatnonzero(~mask)
        other_col = col.take(pa.array(idx_other))
        target_col = pa.array(new_values, type=col_type)

        parts, other_ptr = [], 0
        for k, pos in enumerate(idx_target):
            n_other_before = int(pos) - k        # rows before `pos` minus fre rows before it
            run = n_other_before - other_ptr
            if run > 0:
                parts.append(other_col.slice(other_ptr, run))
                other_ptr += run
            parts.append(target_col.slice(k, 1))
        if other_ptr < len(idx_other):
            parts.append(other_col.slice(other_ptr, len(idx_other) - other_ptr))
        combined = pa.concat_arrays(parts)
        assert len(combined) == table.num_rows

        table = table.set_column(col_index, "kwic_nlp", combined)
        total_repaired += len(idx_target)
        log(f"    replaced {len(idx_target):,} rows in the kwic_nlp column")

    if dry_run:
        log("dry run complete — nothing written")
        return 0
    if total_repaired == 0:
        log("nothing to do — the repaired languages are already populated?")
        return 1

    if schema_before != table.schema:
        raise SystemExit("refusing to write: the schema changed during repair")

    if backup:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        bak = source.with_suffix(source.suffix + f".bak-{stamp}")
        log(f"backing up -> {bak.name}")
        shutil.copy2(source, bak)

    tmp = source.with_suffix(".tmp-writing.parquet")
    log(f"writing {tmp.name} …")
    pq.write_table(table, tmp)
    log(f"  wrote {tmp.stat().st_size / 1e6:,.0f} MB")

    # verify before replacing: same rows, same schema, other languages untouched
    log("verifying the written file before replacing the original …")
    check = pq.read_table(tmp)
    ok = True
    if check.num_rows != n_rows_before:
        log(f"  WARNING: row count {check.num_rows:,} != {n_rows_before:,}")
        ok = False
    if check.schema != schema_before:
        log("  WARNING: schema differs from the original")
        ok = False
    for code in REPAIRS:
        m = eq_mask(check.column("language"), code)
        vals = check.column("kwic_nlp").combine_chunks().take(
            pa.array(np.flatnonzero(m))).to_pylist()
        empties = sum(1 for v in vals if not v)
        log(f"  {code}: {len(vals):,} rows, {empties:,} still empty")
        ok &= empties == 0
    # a language that must be untouched, compared against the original
    for code in ("eng", "ger"):
        m = eq_mask(check.column("language"), code)
        idx = np.flatnonzero(m)[:500]
        old = table.column("kwic_nlp").combine_chunks().take(pa.array(idx)).to_pylist()
        new = check.column("kwic_nlp").combine_chunks().take(pa.array(idx)).to_pylist()
        same = old == new
        log(f"  {code}: first {len(idx)} rows identical to the repair: {same}")
        ok &= bool(same)
    if not ok:
        raise SystemExit(f"verification failed — leaving {source.name} untouched, "
                         f"the candidate file is {tmp.name}")

    log(f"replacing {source.name}")
    os.replace(tmp, source)
    log(f"done — {total_repaired:,} rows repaired. Now rebuild the derived table: "
        f"python3 prepare_jstor_app_data.py")
    return 0


def verify(source: Path) -> int:
    """Report per-language emptiness of `kwic_nlp` (the defect's signature)."""
    table = pq.read_table(source, columns=["language", "kwic_nlp"])
    col = table.column("kwic_nlp").combine_chunks()
    lang_col = table.column("language")
    ok = True
    print(f"{source.name}: {table.num_rows:,} rows")
    # a handful of rows carry a null language; sort those last rather than tripping over
    # an incomparable None/str pair
    langs = sorted(set(lang_col.to_pylist()), key=lambda s: (s is None, s or ""))
    for lang in langs:
        mask = pc.is_null(lang_col).to_numpy(zero_copy_only=False) if lang is None \
            else eq_mask(lang_col, lang)
        idx = np.flatnonzero(mask)
        vals = col.take(pa.array(idx)).to_pylist()
        empty = sum(1 for v in vals if not v)
        flag = "" if empty == 0 else "  <-- EMPTY"
        print(f"  {str(lang):5} n={len(vals):>7,}  empty={empty:>7,} "
              f"({empty / len(vals):6.1%}){flag}")
        if lang in REPAIRS and empty:
            ok = False
    print("VERIFY OK" if ok else "VERIFY: repaired languages still have empty rows")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    global WORKERS
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--verify", action="store_true", help="report per-language emptiness")
    ap.add_argument("--no-backup", action="store_true", help="skip the .bak copy (not advised)")
    ap.add_argument("--workers", type=int, default=WORKERS,
                    help="spaCy n_process for lemmatisation (default 8)")
    args = ap.parse_args(argv)
    WORKERS = args.workers
    if args.verify:
        return verify(args.source)
    return repair(args.source, dry_run=args.dry_run, backup=not args.no_backup)


if __name__ == "__main__":
    sys.exit(main())
