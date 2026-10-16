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
    "Topics around Paul": "from mops_app.pages import topics; topics.render()",
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


def check_topics() -> bool:
    """Check the topics layer against the fitted artifacts.

    Skips cleanly when the model has not been fitted, so this file stays a smoke
    test for the app rather than a requirement to have run an 11-minute fit.

    Two invariants, chosen because they are the ones a refactor would break
    silently:

    * the page's own aggregation must reproduce the precomputed
      `topics_over_time.parquet` when no filters are applied — that file is the
      independent statement of the same quantity, produced inside the fit;
    * every share column must respect its denominator (share sums to 1 within a
      bin; share_modelled to 1 over the non-outlier topics), because a wrong
      denominator is invisible in a plot and fatal to a claim.
    """
    from mops_app import topics_data as td

    if not td.is_available(td.TOPIC_RUN):
        print("skip  topic model not fitted (run scripts/fit_paul_topics.py)")
        return True

    docs = td.load_docs()
    info = td.load_topics_info()
    report = td.load_report()
    ok = True

    if len(docs) != report["documents"]["fitted"]:
        ok = False
        print(f"FAIL  docs rows {len(docs):,} != report {report['documents']['fitted']:,}")

    # 1. recomputed vs precomputed, unfiltered
    pre = td.load_topics_over_time()
    mine = td.topic_trends(docs, bin_length=report["topics"]["bin_length"],
                           metric="share")
    mine = mine.reindex(pre["bin"].unique())
    for _, row in pre[pre["topic"] != td.OUTLIER_TOPIC].iterrows():
        got = float(mine.loc[row["bin"], row["topic"]]) if row["topic"] in mine.columns else 0.0
        if abs(got - float(row["share"])) > 1e-6:
            ok = False
            print(f"FAIL  {row['bin']} topic {row['topic']}: recomputed {got:.6f} "
                  f"vs precomputed {row['share']:.6f}")
            break
    print("ok    topics: recomputed trends match topics_over_time.parquet" if ok
          else "FAILED topics trends")

    # 2. denominators
    for metric in td.TOPIC_METRICS:
        m = td.topic_trends(docs, bin_length=report["topics"]["bin_length"], metric=metric)
        if metric == "count":
            continue
        # `share` sums to 1 over everything including the outlier topic;
        # `share_modelled` sums to 1 over the topics it returns (the outlier topic
        # is omitted there, because it has no share of the clustered documents).
        sums = m.sum(axis=1)
        if not (abs(sums - 1.0) < 1e-6).all():
            ok = False
            print(f"FAIL  metric {metric}: bin sums {list(sums.round(6))}")
    print("ok    topics: share and share_modelled sum to 1 per bin" if ok
          else "FAILED topics denominators")

    # 3. the page's labels must describe every topic the docs use
    unknown = set(docs["topic"].unique()) - set(info["Topic"])
    if unknown:
        ok = False
        print(f"FAIL  topics in docs.parquet missing from topics_info: {sorted(unknown)[:5]}")
    else:
        print("ok    topics: every document topic has a label row")

    # 4. outlier share must be the one the record quotes (32.9% is not a bug to smooth over).
    #    The report rounds to 4 decimals, so compare at that precision -- a tighter
    #    tolerance fails on values that print identically.
    got = float((docs["topic"] == td.OUTLIER_TOPIC).mean())
    want = float(report["topics"]["outlier_share"])
    if abs(got - want) > 5e-5:
        ok = False
        print(f"FAIL  outlier share {got:.6f} != report {want:.6f}")
    else:
        print(f"ok    topics: outlier share {got:.1%} matches the report")
    return ok


def check_topics_comparison() -> bool:
    """Guard the subset-comparison layer used by `6_paul-topics.ipynb`.

    Three invariants, all of which were violated by a real bug during development:

    1. **The outlier topic must not appear under `share_modelled`.** It was being
       divided by a total that *excludes* it, yielding 1.28 for the outlier row and
       column sums of 2.28 — a number that is not a share of anything, and one that
       a plot would have shown without any error. Caught by checking the sums.
    2. **A distribution's distance from itself is exactly 0.** An off-by-one in the
       topic alignment (e.g. comparing non-overlapping supports) shows up here and
       almost nowhere else.
    3. **Per-bin distances stay in range.** Jensen–Shannon is bounded by 1 with
       `base=2`; a value outside [0, 1] means the vectors were not normalised.
    """
    from mops_app import topics_data as td

    if not td.is_available(td.TOPIC_RUN):
        print("skip  topic comparison checks (model not fitted)")
        return True

    docs = td.load_docs()
    subsets = {"overall": docs,
               "tier A": td.filter_docs(docs, tiers=["A"]),
               "English": td.filter_docs(docs, languages=["eng", "en"])}
    ok = True

    # 1. the outlier topic has no share of the clustered documents
    for metric, want_outlier in [("share", True), ("share_modelled", False),
                                 ("count", True)]:
        shares = td.subset_shares(subsets, metric=metric)
        present = td.OUTLIER_TOPIC in shares.index
        if present != want_outlier:
            ok = False
            print(f"FAIL  subset_shares({metric}): outlier row present={present}, "
                  f"expected {want_outlier}")
        if metric != "count":
            sums = shares.sum(axis=0)
            if not (abs(sums - 1.0) < 1e-6).all():
                ok = False
                print(f"FAIL  subset_shares({metric}): column sums "
                      f"{list(sums.round(4))}")
    print("ok    topics: outlier topic excluded from share_modelled, shares sum to 1"
          if ok else "FAILED subset_shares invariants")

    # 2. distance from itself is 0
    same = td.subset_shares({"a": docs, "b": docs}, metric="share")
    for method in td.DISTANCES:
        d = float(td.distribution_distance(same, "a", method=method)["distance"].iloc[0])
        if abs(d) > 1e-12:
            ok = False
            print(f"FAIL  {method}: distance from an identical subset is {d!r}, not 0")

    # 3. per-bin distances in range
    trends = td.trends_by_subset(subsets, metric="share")
    counts = td.trends_by_subset(subsets, metric="count")
    per_bin = counts.groupby(["subset", "bin"])["value"].sum()
    dist_time = td.distance_over_time(trends, "overall", method="js",
                                      min_docs=100, docs_per_bin=per_bin)
    vals = dist_time["distance"].dropna()
    if vals.empty or not ((vals >= 0) & (vals <= 1)).all():
        ok = False
        print(f"FAIL  distance_over_time: values outside [0, 1] "
              f"(min {vals.min() if len(vals) else 'n/a'}, max {vals.max() if len(vals) else 'n/a'})")
    else:
        print(f"ok    topics: js distance in range over {len(vals)} bins, "
              f"identical subset = 0, masked by the min_docs floor")
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", action="store_true", help="skip the parity check")
    ap.add_argument("--no-edge", action="store_true", help="skip the empty-bin/renderer checks")
    ap.add_argument("--no-topics", action="store_true", help="skip the topic-model checks")
    args = ap.parse_args(argv)

    ok = check_pages()
    if not args.no_edge:
        ok = check_edge_cases() and ok
    if not args.pages:
        ok = check_parity() and ok
    if not args.no_topics:
        ok = check_topics() and ok
        ok = check_topics_comparison() and ok
    print("\nALL OK" if ok else "\nFAILURES — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
