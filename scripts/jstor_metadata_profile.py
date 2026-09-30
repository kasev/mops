#!/usr/bin/env python3
"""Single-pass profile of the 13.7 GB unfiltered JSTOR metadata JSONL.

Why not pandas: this answers "what is in the full delivery?" -- it counts
distinct values of every categorical field and the element frequency of every
list-valued field. It reads the file once with regular expressions per line
(~4 min, single core, C-locale-ish) instead of the ~6 min the Parquet build
takes, so it is only worth running when the Parquet build has not finished yet
or when the raw line format itself is in question. Once
`metadata_parquet/` exists, every question below is faster via pyarrow.dataset.

Decisions:
* list-valued fields (`languages`, `discipline_names`, `collections`,
  `publishers`) are counted element-wise, because a JSON-array regex would treat
  ["eng"] and ["eng","fre"] as different keys.
* `creators` is skipped (it is a list of objects, not of scalars); its size is
  reported by the Parquet build's `n_creators` column instead.
* nothing is normalised: the counts are of the literal strings in the file, so a
  value like "2011-01-0" shows up under `published_date` prefixes rather than
  being repaired.
"""

from __future__ import annotations

import collections
import re
import sys
import time
from pathlib import Path

SOURCE = Path("/srv/data/jstor/jstor2026-04/jstor_metadata_2026-04-28.jsonl")

SCALAR = ["content_type", "c5_data_type", "c5_section_type", "content_subtype",
          "ccda_resource_type", "ccda_resource_subtype", "licensing_status",
          "contributed_content", "review_required"]
LIST = ["languages", "discipline_names", "collections", "publishers"]

PAT = {k: re.compile(r'"%s":"([^"]*)"' % k) for k in SCALAR}
LIST_PAT = {k: re.compile(r'"%s":(\[[^\]]*\])' % k) for k in LIST}
ITEM = re.compile(r'"item_id":"([^"]*)"')


def main() -> int:
    counts = {k: collections.Counter() for k in SCALAR + LIST}
    year_buckets = collections.Counter()
    ids = set()
    n = 0
    t0 = time.time()
    with SOURCE.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            n += 1
            for k, pat in PAT.items():
                m = pat.search(line)
                counts[k][m.group(1) if m else "<absent/missing>"] += 1
            for k, pat in LIST_PAT.items():
                m = pat.search(line)
                if not m:
                    counts[k]["<null>"] += 1
                    continue
                for el in re.findall(r'"([^"]*)"', m.group(1)):
                    counts[k][el] += 1
            m = re.search(r'"published_date":"(\d{4})', line)
            year_buckets[m.group(1) if m else "<no 4-digit year>"] += 1
            m = ITEM.search(line)
            if m:
                ids.add(m.group(1))
            if n % 2_000_000 == 0:
                print(f"  ...{n:,} lines, {time.time()-t0:.0f}s", flush=True)

    print(f"\nlines: {n:,}   unique item_id: {len(ids):,}   "
          f"({time.time()-t0:.0f}s)\n")
    for k in SCALAR:
        print(f"== {k} ==")
        for v, c in counts[k].most_common(15):
            print(f"  {c:>10,}  {v}")
        print()
    for k in LIST:
        print(f"== {k} (element counts) ==")
        for v, c in counts[k].most_common(15):
            print(f"  {c:>10,}  {v}")
        print()
    print("== published_date decade (derived) ==")
    for v, c in sorted(year_buckets.items()):
        print(f"  {c:>10,}  {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
