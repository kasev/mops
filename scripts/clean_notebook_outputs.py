#!/usr/bin/env python3
"""Shrink scripts/5_tokens-analysis.ipynb and fix its broken division cell.

Why this exists
---------------
The notebook had grown to ~45 MB. The culprit is *not* pandas: a plain repr of
the 83 x 101,435 `vocab_matrix_counts` frame is only ~900 characters. It is the
Data Wrangler notebook output renderer, which embeds every column of every
DataFrame in the .ipynb as

    application/vnd.microsoft.datawrangler.viewer.v0+json

Two cells accounted for 29.6 MB of that. The renderer is now disabled in
.vscode/settings.json, but the previously-embedded payloads stay in the file
until they are removed, which is what this script does.

It also applies two edits that were made in the editor but never reached disk
(see the note at the bottom):

  1. the row-wise division in the `vocab_matrix_freqs` cell, which raised
     `ValueError: Unable to coerce to Series, length must be 101435: given 83`
     because dividing a DataFrame by a bare list aligns the list against the
     *columns* rather than the rows;
  2. a pandas display-options cell so wide frames wrap instead of printing
     one unbounded table (`display.max_columns`, `display.max_rows`).

Usage
-----
    python3 scripts/clean_notebook_outputs.py            # dry run, reports only
    python3 scripts/clean_notebook_outputs.py --write    # rewrite in place
    python3 scripts/clean_notebook_outputs.py --write --no-backup

The script is idempotent: running it twice changes nothing the second time.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent / "5_tokens-analysis.ipynb"
BACKUP = NOTEBOOK.with_suffix(".ipynb.bak")

WRANGLER_MIME = "application/vnd.microsoft.datawrangler.viewer.v0+json"

# --- edit 1: the broken division ------------------------------------------
# Dividing a DataFrame by a plain list makes pandas align that list against the
# columns (101,435 of them) instead of the 83 rows. Passing an explicitly
# indexed Series with axis=0 makes the intent unambiguous.
BROKEN_DIV = (
    "vocab_matrix_freqs = vocab_matrix_counts / journals_concs_n"
    " # divide each row by each value from journals_concs_n"
)
FIXED_DIV = [
    "journals_concs_n_series = pd.Series(journals_concs_n,"
    " index=vocab_matrix_counts.index)\n",
    "vocab_matrix_freqs = vocab_matrix_counts.div(journals_concs_n_series,"
    " axis=0)  # row-wise: each journal by its own concordance count\n",
    "vocab_matrix_freqs.head(5)",
]

# --- edit 2: pandas display options ---------------------------------------
OPTIONS_ANCHOR = "import google_conf"  # last line of the first (imports) cell
OPTIONS_CELL = '''# Keep notebook outputs small. `vocab_matrix_*` below is 83 x 101,435, so any
# "show everything" option would freeze the editor and bloat the .ipynb.
# max_columns=20 wraps a wide frame into several stacked blocks instead of one
# unbounded table (pandas then prints a `...` gap plus a trailing
# "[83 rows x 101435 columns]" line).
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 60)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.show_dimensions", True)'''


def source_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_source(cell: dict, lines: list[str]) -> None:
    cell["source"] = lines


def strip_wrangler(outputs: list[dict]) -> tuple[list[dict], int]:
    """Drop the Data Wrangler MIME from every output; keep text/plain & text/html."""
    kept, removed = [], 0
    for out in outputs:
        data = out.get("data")
        if isinstance(data, dict) and WRANGLER_MIME in data:
            removed += len(json.dumps(data.pop(WRANGLER_MIME), separators=(",", ":")))
            if not data:  # nothing left worth rendering
                continue
        kept.append(out)
    return kept, removed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true",
                    help="rewrite the notebook (default is a dry run)")
    ap.add_argument("--no-backup", action="store_true",
                    help="skip writing the .ipynb.bak backup")
    args = ap.parse_args()

    before = NOTEBOOK.stat().st_size
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = nb["cells"]

    changes: list[str] = []

    # 1. strip Data Wrangler payloads
    bytes_removed = 0
    for i, cell in enumerate(cells, 1):
        outs = cell.get("outputs", [])
        if outs:
            cell["outputs"], n = strip_wrangler(outs)
            if n:
                bytes_removed += n
                first = source_text(cell).strip().splitlines()
                label = first[0][:50] if first else "(no source)"
                changes.append(f"  cell {i:>2}: stripped {n/1e6:6.2f} MB  <- {label}")

    # 2. fix the row-wise division
    div_hits = [i for i, c in enumerate(cells, 1) if BROKEN_DIV in source_text(c)]
    if len(div_hits) == 1:
        cell = cells[div_hits[0] - 1]
        set_source(cell, FIXED_DIV)
        cell["outputs"] = []          # the traceback is stale now
        cell["execution_count"] = None
        changes.append(f"  cell {div_hits[0]:>2}: fixed row-wise division"
                       f" (vocab_matrix_counts.div(..., axis=0))")
    elif not div_hits:
        changes.append("  division cell already fixed (or rewritten) - skipped")
    else:
        print(f"! {BROKEN_DIV!r} found in {len(div_hits)} cells; refusing to guess",
              file=sys.stderr)
        return 2

    # 3. add the pandas display-options cell next to the imports
    has_options = any("display.max_columns" in source_text(c) for c in cells)
    if has_options:
        changes.append("  pandas display options already present - skipped")
    else:
        anchor = next((i for i, c in enumerate(cells)
                       if c.get("cell_type") == "code"
                       and OPTIONS_ANCHOR in source_text(c)), None)
        if anchor is None:
            print(f"! could not find a cell containing {OPTIONS_ANCHOR!r}",
                  file=sys.stderr)
            return 2
        cells.insert(anchor + 1, {
            "cell_type": "code",
            "execution_count": None,
            "id": "pd-display-options",
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in OPTIONS_CELL.splitlines()[:-1]]
                      + [OPTIONS_CELL.splitlines()[-1]],
        })
        changes.append(f"  inserted pandas display-options cell after cell {anchor + 1}")

    print(f"{NOTEBOOK.name}: {before/1e6:.2f} MB")
    print("\n".join(changes) if changes else "  (nothing to do)")

    if not args.write:
        print(f"\ndry run - would remove ~{bytes_removed/1e6:.2f} MB "
              "of Data Wrangler payload; re-run with --write to apply")
        return 0

    if not args.no_backup and not BACKUP.exists():
        shutil.copy2(NOTEBOOK, BACKUP)
        print(f"\nbackup written: {BACKUP.name}")

    # nbformat: 1-space indent keeps the file compact (it is still large)
    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    after = NOTEBOOK.stat().st_size
    print(f"rewritten: {before/1e6:.2f} MB -> {after/1e6:.2f} MB "
          f"(-{(before-after)/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
