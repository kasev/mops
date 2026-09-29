"""Make paul-explore.ipynb usable on a remote VSCode server.  Idempotent.

Two things break there, and neither is a data bug:
  · ipywidgets needs the notebook's JavaScript comm layer (renders an empty box);
  · `input()` in a cell never shows its prompt (the cell only looks hung).

So §6 becomes plain function calls and §8 becomes file-based labelling backed by
scripts/label_eval.py, which is also a CLI. Cells are located by content, not index,
because an `nbconvert --inplace` run that outlives its shell rewrites this file.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

PATH = Path("/home/jupyter-vojta/notebooks/mops/scripts/paul-explore.ipynb")

# --------------------------------------------------------------------------- §6

SEC6_MD = """## 6 · Browsing the hits

No widget here: `ipywidgets` needs the notebook's JavaScript comm layer, which does not
work over a remote VSCode server (the widget renders as an empty box). The same filters
are keyword arguments instead, so a query is an ordinary cell you edit and rerun.

`q()` wraps `browse()` + `show_hits()`; every argument is optional.

| argument | meaning |
|---|---|
| `tier` | `"A"` unflagged reference, `"B"` demoted, `"C"` needs judgement |
| `pattern`, `journal`, `language` | exact value, e.g. `pattern="A4_tarsus"` |
| `year_from`, `year_to` | inclusive |
| `require_flag`, `exclude_flag` | one name from `ALL_FLAGS` |
| `unflagged_only` | rows no flag touched |
| `sentence=True` | whole sentences instead of KWIC |
| `n`, `seed` | sample size and seed — same `seed` returns the same rows |"""

QDEF = '''def q(sentence=False, width=52, to_csv=None, n=25, seed=0, **filters):
    """Filter the hits and print them. Filters: tier, pattern, journal, language,
    year_from, year_to, require_flag, exclude_flag, exclude_flags, unflagged_only.
    """
    d = browse(n=n, seed=seed, **filters)
    if to_csv:
        d.to_csv(to_csv, index=False)
        print(f"wrote {len(d):,} rows -> {to_csv}\\n")
    if sentence:
        for _, r in d.iterrows():
            print(f"[{r['tier']} · {r['pattern']} · {r['year']} · {r['journal']}]")
            print(str(r["sentence"])[:600])
            if r["flags"]: print(f"   flags: {r['flags']}")
            print()
    else:
        show_hits(d, width=width)
    return d'''

EXAMPLES = [
    ("### The filters, in use", 'q(tier="A", n=8)', None),
    ("Tier B is real references that a flag knocked down. This is where recall is lost, "
     "and it is the largest tier at 145,240 rows, so it is the one worth sampling.",
     'q(tier="B", n=15)', None),
    ("One construction, one era. `pattern` takes any of the 14 names in "
     "`sorted(hits['pattern'].unique())`.",
     'q(pattern="A4_tarsus", year_from=1950, year_to=1980, n=15)', None),
    ("The 11 flags in `ALL_FLAGS` are `capital_after`, `citation_apparatus`, "
     "`demoted_name_after`, `excluded_name`, `initial_after`, `initial_before`, "
     "`papacy_or_numeral`, `place_or_building`, `possibly_given_name`, "
     "`probably_person_name`, `running_head`. Each is also a boolean column `f_<name>`.",
     'q(require_flag="probably_person_name", n=12)', None),
    ("Rows no flag touched are the cleanest subset to check precision against by hand.",
     'q(unflagged_only=True, tier="A", n=15)', None),
    ("When a KWIC window is too short to judge, read the sentence. `width=` widens the "
     "window without going all the way to the whole sentence.",
     'q(tier="C", n=5, sentence=True)', None),
    ("`to_csv=` saves the filtered subset for any purpose — a table, a closer look, a "
     "hand-built sample. Labelling the precision sample is a separate, dedicated flow in "
     "§8 (`scripts/label_eval.py`), which writes its own review sheet and merges verdicts "
     "on `row_id`.",
     'q(tier="B", n=50, seed=7, to_csv="tierB_sample.csv")', None),
]

# --------------------------------------------------------------------------- §8

SEC8_MD = """## 8 · Labelling the precision sample

`paul_eval.tsv` (280 rows, one per sampled hit) has an empty `verdict` column, and
precision per pattern cannot be published until it is filled in — the open item in
`AGENTS.md`.

**No widget and no `input()` here.** Both need something a notebook cell over a remote
VSCode server does not give you: widgets need the JavaScript comm layer, and an `input()`
prompt never appears (the cell only looks hung). What does work is the editor itself —
write a review sheet, fill its `verdict` column, save, merge it back.

Verdicts: `a`postle (a genuine reference to the apostle) · `o`ther · `u`nclear · `s`kip.
`skip` is recorded but excluded from precision; `unclear` sets its upper bound.

The current sample was drawn from the buggy 2026-09-04 run — regenerate it first with
`python3 scripts/paul_concordance.py --dump-eval 300`. Everything below goes through
`scripts/label_eval.py`, which is also a CLI, so a real terminal can instead run
`python3 scripts/label_eval.py interactive --tier B -n 50`."""

IMPORT = '''# One source of truth for labelling: scripts/label_eval.py, also a CLI.
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("label_eval", ROOT / "scripts" / "label_eval.py")
label_eval = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(label_eval)

_eval = label_eval.load_sample()          # paul_eval.tsv + any verdicts already saved
_done = _eval["verdict"].isin(label_eval.CANONICAL).sum()
print(f"{_done} of {len(_eval)} labelled, {len(_eval) - _done} to go")'''

SHEET_MD = """### Write a review sheet

`--tier B` is the highest-value subset: 145,240 rows a flag demoted, so whatever recall
the flags cost is in there. `context` is the hit inside its sentence, wide enough to judge
from. `row_id` is the join key — leave it alone."""

SHEET = '''SHEET = SCAN / "eval_sheet_B.tsv"
label_eval.make_sheet(_eval, SHEET, n=50, seed=7, tier="B", width=130)'''

APPLY_MD = """Fill the `verdict` column in VSCode, save, then merge it back. Verdicts accumulate in
**`paul_eval.labelled.tsv`** and merge on `row_id`, so several sheets can be labelled in
any order; `paul_eval.tsv` itself is never modified."""

APPLY = "_eval = label_eval.apply_sheet(SHEET, _eval)"
PREC_MD = "Precision from whatever is labelled so far."
PREC = "label_eval.report(_eval)"

# ------------------------------------------------------------- small in-place fixes

FLAG_FIX = ('hits["flags"].str.contains(rf"(?:^|;){re.escape(f)}(?:$|;)", regex=True)',
            'hits["flags"].str.contains(rf"(?:^|;){re.escape(f)}(?:$|;)", regex=True, na=False)')

BROWSE_FIXES = [
    ('''def browse(tier=None''',
     '''def _flagcol(d, name):
    """Flag columns are booleans, but a parquet round-trip can leave NA in them."""
    return d[f"f_{name}"].fillna(False).astype(bool)


def browse(tier=None'''),
    ('''    if require_flag and require_flag != "all": d = d[d[f"f_{require_flag}"]]
    if exclude_flag and exclude_flag != "all": d = d[~d[f"f_{exclude_flag}"]]
    for f in exclude_flags:
        if f: d = d[~d[f"f_{f}"]]
    if unflagged_only: d = d[~d["flagged"]]''',
     '''    if require_flag and require_flag != "all": d = d[_flagcol(d, require_flag)]
    if exclude_flag and exclude_flag != "all": d = d[~_flagcol(d, exclude_flag)]
    for f in exclude_flags:
        if f: d = d[~_flagcol(d, f)]
    if unflagged_only: d = d[~d["flagged"].fillna(False)]'''),
]

ROOT_FIX = ('JSTOR_TEXTS = Path(',
            'ROOT = SCAN.parents[2]            # .../mops — holds scripts/\n\nJSTOR_TEXTS = Path(')

COMMENT_FIX = ("from the browser above", "from §6")


def md(text):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "execution_count": None,
            "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def src(cell):
    return "".join(cell["source"]) if cell["cell_type"] == "code" else "".join(cell["source"])


def find(cells, needle, kind="code"):
    hits = [i for i, c in enumerate(cells)
            if c["cell_type"] == kind and needle in src(c)]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one {kind} cell containing {needle!r}, found {hits}")
    return hits[0]


def replace_in(cell, old, new):
    s = src(cell)
    if old in s:
        cell["source"] = s.replace(old, new).splitlines(keepends=True)
        return True
    return False


def patch_section(cells, heading, stop_heading, new_cells, label):
    """Replace heading..just-before-stop_heading with new_cells, unless already done."""
    try:
        start = find(cells, heading, "markdown")
    except SystemExit:
        print(f"  §{label}: heading {heading!r} not found — skipping")
        return
    stop = find(cells, stop_heading, "markdown")
    if any("label_eval" in src(c) for c in cells[start:stop]) or \
       any("def q(sentence" in src(c) for c in cells[start:stop]):
        print(f"  §{label}: already patched")
        return
    cells[start:stop] = new_cells


def main():
    nb = json.loads(PATH.read_text())
    cells = nb["cells"]
    before = len(cells)

    # small fixes that apply wherever they live
    fixed = []
    for c in cells:
        if c["cell_type"] != "code":
            continue
        # the q() cell should only define; the first example runs it
        s = src(c)
        if s.startswith("def q(sentence") and s.rstrip().endswith('q(tier="A", n=8)'):
            c["source"] = s.rstrip()[: -len('q(tier="A", n=8)')].rstrip().splitlines(keepends=True)
            fixed.append("q() demo call")
        if replace_in(c, *FLAG_FIX):
            fixed.append("flags na=False")
        for old, new in BROWSE_FIXES:
            if replace_in(c, old, new):
                fixed.append("_flagcol")
        if replace_in(c, *ROOT_FIX):
            fixed.append("ROOT")
        if replace_in(c, *COMMENT_FIX):
            fixed.append("§7 comment")
    print("in-place fixes:", sorted(set(fixed)) or "none needed")

    # §6 keeps its existing helper cell (META, jstor_text, kwic, browse, show_hits)
    helper_src = src(cells[find(cells, "def browse(tier=None")])
    sec6 = [md(SEC6_MD), code(helper_src), code(QDEF)]
    for note, qsrc, _ in EXAMPLES:
        sec6 += [md(note), code(qsrc)]
    patch_section(cells, "## 6 ·", "## 7 ·", sec6, "6")

    patch_section(cells, "## 8 ·", "## 9 · ", [
        md(SEC8_MD), code(IMPORT), md(SHEET_MD), code(SHEET),
        md(APPLY_MD), code(APPLY), md(PREC_MD), code(PREC)], "8")

    for i, c in enumerate(cells):
        c.setdefault("id", f"c{i:03d}")
        if c["cell_type"] == "code":
            s = src(c)
            for bad in ("ipywidgets", "widgets.Output", "input(", "raw_right", "labelling_tierB"):
                if bad in s:
                    raise SystemExit(f"cell {i} still uses {bad!r}")

    nb["nbformat"], nb["nbformat_minor"] = 4, 5
    PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"cells {before} -> {len(cells)}; wrote {PATH.name}")


if __name__ == "__main__":
    main()
