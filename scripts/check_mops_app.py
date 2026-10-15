#!/usr/bin/env python3
"""Smoke-test the MOPS streamlit app without starting a server.

Uses streamlit's own `AppTest` harness to execute each page in a bare script
runner and report any exception, so a broken page is caught in CI/terminal
instead of in the browser. Also asserts the notebook-parity numbers that
`mops_app/README.md` documents — if a refactor changes the aggregation, this
fails loudly.

Usage
-----
    python3 check_mops_app.py            # pages + parity
    python3 check_mops_app.py --pages    # pages only (skips the slower parity check)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

#: Page module -> source the AppTest runner executes for it.
PAGES = {
    "entry point (About)": "import mops_jstor",
    "Temporal frequencies": "from mops_app.pages import temporal; temporal.render()",
    "Concordance browser": "from mops_app.pages import concordance; concordance.render()",
}

#: The notebook's recorded output for jewish/judaism/jew, bidecades, English only,
#: no discipline/confession filter, metric `per_20` (see mops_app/README.md).
EXPECTED_PER_20 = {
    "1900–1919": {"jewish": 0.309423, "judaism": 0.171737, "jew": 0.074025},
    "1920–1939": {"jewish": 0.310975, "judaism": 0.135599, "jew": 0.072320},
    "1940–1959": {"jewish": 0.203965, "judaism": 0.092022, "jew": 0.069253},
    "1960–1979": {"jewish": 0.275765, "judaism": 0.174253, "jew": 0.042885},
    "1980–1999": {"jewish": 0.412247, "judaism": 0.218663, "jew": 0.074194},
    "2000–2019": {"jewish": 0.417121, "judaism": 0.176689, "jew": 0.093377},
}

#: Same run's `subset_length`, i.e. concordances per bin.
EXPECTED_BIN_SIZES = {
    "1900–1919": 13509, "1920–1939": 11062, "1940–1959": 21082,
    "1960–1979": 36843, "1980–1999": 76556, "2000–2019": 107307,
}

TOLERANCE = 5e-7


def check_pages() -> bool:
    """Run every page through AppTest; return False if any raised."""
    from streamlit.testing.v1 import AppTest

    ok = True
    for label, source in PAGES.items():
        at = AppTest.from_string(source, default_timeout=600)
        at.run()
        if at.exception:
            ok = False
            print(f"FAIL  {label}")
            for exc in at.exception:
                print(f"        {exc.value}")
        else:
            print(f"ok    {label}")
    return ok


def check_parity() -> bool:
    """Reproduce the notebook's recorded temporal table."""
    from mops_app import data

    tokens = data.load_tokens()
    filtered = data.filter_tokens(tokens, year_range=(1900, 2019),
                                  languages=list(data.ENGLISH_CODES))
    result = data.words_over_time(filtered, ["jewish", "judaism", "jew"],
                                  bin_length=20, year_range=(1900, 2019),
                                  metric="per_20")

    ok = True
    for label, expected in EXPECTED_PER_20.items():
        for word, want in expected.items():
            got = float(result.matrix.loc[label, word])
            if abs(got - want) > TOLERANCE:
                ok = False
                print(f"FAIL  {label} {word}: got {got:.6f}, notebook has {want:.6f}")
    for label, want in EXPECTED_BIN_SIZES.items():
        got = int(result.bin_sizes.loc[label])
        if got != want:
            ok = False
            print(f"FAIL  {label} bin size: got {got}, notebook has {want}")
    print("ok    notebook parity (18 values + 6 bin sizes)" if ok else "FAILED parity")
    return ok


def check_edge_cases() -> bool:
    """Exercise paths the browser does not reach by default.

    Three regressions this guards against:

    1. **Empty bins.** A bin with zero hits makes the frequency denominator zero.
       The original implementation wrote `denominator.replace(0, pd.NA).astype(float)`,
       which turns the series object-dtype and raises
       `TypeError: float() argument must be ... not 'NAType'` — a crash only
       visible once a filter emptied the oldest bin.
    2. **The matplotlib renderer**, which the app defaults away from (plotly is
       the default), so a break there would go unnoticed.
    3. **All three concordance search modes**, only one of which the app default
       exercises.
    """
    import matplotlib

    matplotlib.use("Agg")

    from mops_app import data, plots

    tokens = data.load_tokens()
    ok = True

    # 1. every metric over a window whose early bins are empty by construction
    #    (all rows are post-2000, so 1900-1999 bins have no concordances).
    modern = tokens[tokens["year"] >= 2000]
    for metric in data.METRICS:
        res = data.words_over_time(modern, ["jewish", "judaism"], bin_length=20,
                                   year_range=(1900, 2019), metric=metric)
        if int((res.bin_sizes == 0).sum()) == 0:
            ok = False
            print(f"FAIL  {metric}: expected empty bins in this fixture — "
                  f"the guard is no longer testing anything")
        if res.matrix.shape != (6, 2) or res.matrix.isna().any().any():
            ok = False
            print(f"FAIL  {metric}: matrix is {res.matrix.shape} "
                  f"with {int(res.matrix.isna().sum().sum())} NaN")
    print("ok    empty bins across all metrics" if ok else "FAILED empty bins")

    # 2. matplotlib: several option combinations must draw without raising
    res = data.words_over_time(modern, ["jewish", "judaism", "jew"], bin_length=20,
                               year_range=(1900, 2019), metric="per_1000")
    try:
        for log_y in (False, True):
            _, ax = plots.plot_timebins(res, log_y=log_y, show_bin_sizes=True)
            ax.get_figure().canvas.draw()
        fig = plots.plotly_timebins(res, log_y=True)
        if len(fig.data) < 2:
            ok = False
            print("FAIL  plotly figure has no series")
        print("ok    matplotlib + plotly renderers" if ok else "FAILED renderers")
    except Exception as exc:  # noqa: BLE001 - report and keep checking
        ok = False
        print(f"FAIL  renderers raised: {exc!r}")

    # 3. concordance search, one query per mode
    hits = data.load_hits()
    filters = {"year_range": (1900, 2019), "languages": list(data.ENGLISH_CODES)}
    for mode, query in [("lemma", "justification"),
                        ("kwic", "apostle Paul"),
                        ("pattern", "A6_saint_paul")]:
        found = data.search_concordance(tokens, hits, query=query, mode=mode, filters=filters)
        if found.empty or not {"kwic_left", "match", "kwic_right"} <= set(found.columns):
            ok = False
            print(f"FAIL  search mode {mode} returned {len(found)} rows")
    print("ok    concordance search (lemma / kwic / pattern)" if ok else "FAILED search")
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", action="store_true", help="skip the parity check")
    ap.add_argument("--no-edge", action="store_true", help="skip the empty-bin/renderer checks")
    args = ap.parse_args(argv)

    ok = check_pages()
    if not args.no_edge:
        ok = check_edge_cases() and ok
    if not args.pages:
        ok = check_parity() and ok
    print("\nALL OK" if ok else "\nFAILURES — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
