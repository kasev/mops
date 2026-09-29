#!/usr/bin/env python3
"""Fill in the `verdict` column of paul_eval.tsv, so precision per pattern can be published.

`paul_eval.tsv` (one row per sampled hit) has an empty `verdict` column, and that is the
open item in AGENTS.md. Verdicts are written to `paul_eval.labelled.tsv` — never to
`paul_eval.tsv` — and are merged on `row_id`, so rows can be labelled in any order.

Two ways to label, for two different clients:

    # 1. In a notebook over a remote VSCode server (no JavaScript, no stdin needed):
    #    write a review sheet, edit its `verdict` column in VSCode, save, merge back.
    python3 scripts/label_eval.py sheet --tier B -n 50 --width 130
    python3 scripts/label_eval.py apply tierB_review.tsv

    # 2. In a real terminal: one row at a time, saved after every verdict.
    python3 scripts/label_eval.py interactive --tier B -n 50

    python3 scripts/label_eval.py stats        # progress and running precision

Verdicts are `apostle` (a genuine reference to the apostle), `other`, `unclear`, or `skip`.
The sheet accepts the abbreviations a/o/u/s and 1/0/? as well.

The current `paul_eval.tsv` was drawn from the buggy 2026-09-04 run; regenerate it first
with `python3 scripts/paul_concordance.py --dump-eval 300` before labelling for real.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCAN = Path(__file__).resolve().parents[1] / "data" / "large_files" / "paul_scan"
EVAL_IN = SCAN / "paul_eval.tsv"
EVAL_OUT = SCAN / "paul_eval.labelled.tsv"

CANONICAL = ("apostle", "other", "unclear", "skip")
ALIASES = {
    "a": "apostle", "apostle": "apostle", "p": "apostle", "paul": "apostle",
    "1": "apostle", "yes": "apostle", "y": "apostle", "true": "apostle",
    "o": "other", "other": "other", "not": "other", "no": "other", "n": "other",
    "0": "other", "false": "other", "x": "other",
    "u": "unclear", "unclear": "unclear", "?": "unclear",
    "s": "skip", "skip": "skip",
}


def normalise(value) -> str | None:
    """A verdict cell to one of CANONICAL, or None if blank/unrecognised."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return ALIASES.get(str(value).strip().lower())


# ---------------------------------------------------------------- loading the sample

def load_sample(path: Path = EVAL_IN) -> pd.DataFrame:
    """paul_eval.tsv with a stable `row_id` and `verdict` merged in if we have it."""
    if not path.exists():
        sys.exit(f"{path} not found — regenerate it:\n"
                 f"  python3 scripts/paul_concordance.py --dump-eval 300")
    ev = pd.read_csv(path, sep="\t")
    ev["row_id"] = ev.index
    ev["verdict"] = ev["verdict"].astype("object") if "verdict" in ev else None
    if EVAL_OUT.exists():
        done = pd.read_csv(EVAL_OUT, sep="\t").drop_duplicates("row_id", keep="last")
        merged = dict(zip(done["row_id"], done["verdict"]))
        ev["verdict"] = [merged.get(r, v) for r, v in zip(ev["row_id"], ev["verdict"])]
    return ev


def kwic(row, width: int = 130) -> str:
    """The hit inside its sentence, one cell wide enough to judge from."""
    left = str(row.get("kwic_left", ""))[-width:]
    right = str(row.get("kwic_right", ""))[:width]
    sent = " ".join(str(row.get("sentence", "")).split())
    if len(sent) > 6 and "[" not in sent:
        return " ".join((left + "[" + str(row["match"]) + "]" + right).split())
    return " ".join((left + "[" + str(row["match"]) + "]" + right).split())


# ---------------------------------------------------------------- the review sheet

SHEET_COLS = ["row_id", "tier", "pattern", "year", "language", "flags", "context", "verdict"]


def make_sheet(ev: pd.DataFrame, out: Path, n: int = 0, seed: int = 0,
               width: int = 130, **filters) -> pd.DataFrame:
    """Write a TSV to label in the editor: one narrow row per hit, `verdict` left blank."""
    d = select(ev, **filters)
    if n:
        d = d.sample(min(n, len(d)), random_state=seed)
    sheet = d.copy()
    sheet["context"] = [kwic(r, width) for _, r in d.iterrows()]
    sheet["verdict"] = sheet["verdict"].where(sheet["verdict"].isin(CANONICAL), "")
    sheet[SHEET_COLS].to_csv(out, sep="\t", index=False)
    filled = (sheet["verdict"] != "").sum()
    print(f"wrote {len(sheet):,} rows -> {out}"
          f"{f'  ({filled} already labelled, kept)' if filled else ''}")
    print(f"open it in VSCode, fill `verdict` with a/o/u/s, save, then:\n"
          f"  python3 scripts/label_eval.py apply {out}")
    return sheet


def select(ev: pd.DataFrame, tier: str | None = None, pattern: str | None = None,
           unlabelled_only: bool = False) -> pd.DataFrame:
    d = ev
    if tier and tier != "all":
        d = d[d["tier"] == tier]
    if pattern and pattern != "all":
        d = d[d["pattern"] == pattern]
    if unlabelled_only:
        d = d[d["verdict"].isna() | ~d["verdict"].isin(CANONICAL)]
    return d


def apply_sheet(path: Path, ev: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge verdicts from a saved review sheet into paul_eval.labelled.tsv."""
    ev = ev if ev is not None else load_sample()
    sheet = pd.read_csv(path, sep="\t")
    if "verdict" not in sheet or "row_id" not in sheet:
        sys.exit(f"{path} needs both `row_id` and `verdict` columns")
    ev = ev.copy()
    ev["verdict"] = ev["verdict"].astype("object")
    n_new = n_bad = 0
    known = set(ev["row_id"])
    for rid, val in zip(sheet["row_id"], sheet["verdict"]):
        verdict = normalise(val)
        if verdict is None:
            if val not in ("", None) and pd.notna(val):
                n_bad += 1
            continue
        if rid not in known:
            n_bad += 1
            continue
        ev.loc[ev["row_id"] == rid, "verdict"] = verdict
        n_new += 1
    save(ev)
    print(f"applied {n_new} verdicts from {path.name}"
          + (f"  ({n_bad} unrecognised rows ignored)" if n_bad else ""))
    report(ev)
    return ev


def save(ev: pd.DataFrame) -> None:
    ev.loc[ev["verdict"].notna(), ["row_id", "pattern", "tier", "verdict"]].to_csv(
        EVAL_OUT, sep="\t", index=False)


# ---------------------------------------------------------------- terminal loop

KEYS = """a apostle · o other · u unclear · s skip · Enter next · <N> jump · b undo · q save+quit"""


def interactive(ev: pd.DataFrame, n: int = 0, **filters) -> pd.DataFrame:
    """One row at a time in a real terminal. Use `sheet` instead from a notebook."""
    ev = ev.copy()
    ev["verdict"] = ev["verdict"].astype("object")
    todo = select(ev, unlabelled_only=True, **filters)
    if not len(todo):
        print("nothing left to label in that subset")
        return ev
    order = list(todo["row_id"])
    pos, budget, history = 0, (n or len(order)), []
    print(f"{len(todo)} rows in this subset\n{KEYS}")
    while budget > 0 and pos < len(order):
        i = order[pos]
        r = ev[ev["row_id"] == i].iloc[0]
        print(f"\n[{pos + 1}/{len(order)}] {r['tier']} · {r['pattern']} · {r['year']} · {r['language']}")
        if r.get("flags"):
            print(f"  flags: {r['flags']}")
        print("  " + kwic(r, 200))
        try:
            key = input("  verdict: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if key in ("q", "quit"):
            break
        if key == "b" and history:
            j, prev = history.pop()
            ev.loc[ev["row_id"] == j, "verdict"] = prev
            save(ev)
            continue
        if key.isdigit():
            pos = min(max(int(key) - 1, 0), len(order) - 1)
            continue
        if not key:
            pos += 1
            continue
        verdict = normalise(key)
        if verdict is None:
            print("  a=apostle o=other u=unclear s=skip · Enter next · N jump · b undo · q quit")
            continue
        history.append((i, ev.loc[ev["row_id"] == i, "verdict"].iloc[0]))
        ev.loc[ev["row_id"] == i, "verdict"] = verdict
        save(ev)
        budget -= 1
        pos += 1
    save(ev)
    report(ev)
    return ev


# ---------------------------------------------------------------- reporting

def report(ev: pd.DataFrame) -> None:
    done = ev["verdict"].isin(CANONICAL)
    print(f"\n{done.sum()} of {len(ev)} rows labelled ({ev['verdict'].nunique()} distinct verdicts)")
    lab = ev[ev["verdict"].isin(["apostle", "other", "unclear"])]
    if not len(lab):
        print("nothing scored yet — precision needs apostle/other/unclear verdicts")
        return
    prec = (lab.assign(hit=lab["verdict"].eq("apostle").astype(int))
               .groupby(["tier", "pattern"])
               .agg(n=("hit", "size"), apostle=("hit", "sum"),
                    unclear=("verdict", lambda s: (s == "unclear").sum()))
               .assign(precision=lambda d: (100 * d.apostle / d.n).round(1),
                       **{"upper bound %": lambda d: (100 * (d.apostle + d.unclear) / d.n).round(1)}))
    print(prec.to_string())


# ---------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sheet", help="write a TSV to label in the editor")
    p.add_argument("-n", type=int, default=0, help="random subset of this size (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=130, help="chars of context each side")
    p.add_argument("--tier", default=None, choices=["A", "B", "C"])
    p.add_argument("--pattern", default=None)
    p.add_argument("--unlabelled-only", action="store_true")
    p.add_argument("-o", "--out", default=None, help="default: eval_sheet[_tier].tsv")

    p = sub.add_parser("apply", help="merge verdicts from a saved sheet")
    p.add_argument("path", type=Path)

    p = sub.add_parser("interactive", help="label one row at a time (real terminal)")
    p.add_argument("-n", type=int, default=0)
    p.add_argument("--tier", default=None, choices=["A", "B", "C"])
    p.add_argument("--pattern", default=None)

    sub.add_parser("stats", help="progress and precision so far")

    args = ap.parse_args()
    ev = load_sample()

    if args.cmd == "sheet":
        out = Path(args.out) if args.out else \
            SCAN / (f"eval_sheet{('_' + args.tier) if args.tier else ''}.tsv")
        make_sheet(ev, out, n=args.n, seed=args.seed, width=args.width,
                   tier=args.tier, pattern=args.pattern, unlabelled_only=args.unlabelled_only)
    elif args.cmd == "apply":
        apply_sheet(args.path, ev)
    elif args.cmd == "interactive":
        interactive(ev, n=args.n, tier=args.tier, pattern=args.pattern)
    else:
        report(ev)


if __name__ == "__main__":
    main()
