r"""
paul_concordance.py — find every reference to Paul the Apostle in the JSTOR
2026-04 full-text corpus and export the context (KWIC + sentence).

Input   /srv/data/jstor/jstor2026-04/fulltexts_txts/<iid>.txt      (384,807 files)
        /srv/data/jstor/jstor2026-04/jstor_metadata_2026-04-28-filtered.csv
Output  /srv/data/jstor/jstor2026-04/paul_scan/
          hits.tsv.gz          one row per hit (see FIELDS)
          hits_by_doc.tsv.gz   per-article counts per tier
          concordance_A.txt    aligned KWIC listing, readable / usable in the paper
          scan_report.json     totals per tier / pattern / flag  = audit trail
          paul_exclusions.txt  exclusion data (edited by hand, not hard-coded)

WHY TIERS AND NOT A BOOLEAN
    "Paul" is radically ambiguous in this corpus: 565k Paul-ish tokens, but the
    unambiguous wording ("apostle Paul", "Paul of Tarsus", "Pauline epistles")
    occurs in only 1.6% of documents.  A single yes/no match would have to pick
    one arbitrary point on that trade-off.  Every hit therefore carries a tier,
    the pattern that fired, and context flags, so a count can be reported as
    "N (strict) … N+M (inclusive)" and audited pattern by pattern.

      A  explicit identification — apostle Paul · Paul the apostle · Paul of
         Tarsus · the apostle/pauline writings (EN, DE, FR, NL, LA) · St./Saint
         Paul (measured: overwhelmingly the apostle in this corpus; building and
         city senses are flagged `place_or_building`, never dropped)
      B  strong — adjectival Pauline/paulinisch/paulinien (generalised: it refers
         to the apostle unless it is a woman's given name), Paulus, "of Paul",
         "Paul's <theological noun>"
      C  bare "Paul" resolved by context — the follow-word heuristic
         (`Paul writes…`, `Paul s`, punctuation + lowercase) plus an
         apostle-lexicon vote; tokens whose neighbourhood says "another Paul"
         (Ricœur, Sartre, Kegan Paul, John Paul II, author names) are kept but
         flagged, not silently dropped.

MATCHING
    Matching happens on a normalised *length-preserving* copy of the text
    (casefold, NBSP/newlines -> space, HTML tags/entities -> equal-width spaces),
    so every offset maps 1:1 back onto the file as it sits on disk and the
    concordance quotes the original exactly.

    Cheap prefilter: 73% of documents contain no "aul" substring at all.

Usage
    python3 paul_concordance.py --limit 5000            # smoke test
    python3 paul_concordance.py                          # full scan (~3 min)
    python3 paul_concordance.py --tier A --sample 40     # eyeball tier A
    python3 paul_concordance.py --limit 20000 --dump-eval 300
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
import time
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# The shared JSTOR corpus lives outside the project and is read-only here.
# Everything Paul-specific (outputs, exclusions) lives inside the mops project.
JSTOR = Path("/srv/data/jstor/jstor2026-04")
TXT = JSTOR / "fulltexts_txts"
METADATA = JSTOR / "jstor_metadata_2026-04-28-filtered.csv"

MOPS = Path(__file__).resolve().parents[1]           # .../notebooks/mops
OUT = MOPS / "data" / "large_files" / "paul_scan"

#: Tunables that the notebook exposes as CONFIG.  The defaults reproduce the
#: 2026-09-04 run; every one of them can be overridden on the command line.
CTX_CHARS = 46          # left/right window used by the flag detectors (--ctx)
KWIC_CHARS = 42         # left/right window kept in the output (--kwic)
APPARATUS_MIN_HINTS = 3  # citation-regex agreement needed to flag apparatus
                         # (--apparatus-threshold)
TIER_C_CAPITALS = "keep"  # what to do with tier-C hits followed by a
                          # Capitalised word: keep | drop  (--tier-c-capitals)

# --- regex building blocks -------------------------------------------------
S = r"[\s\u00a0\u2010-\u2015]+"          # gap: whitespace, NBSP, dashes
G = r"[,\s;:]+"                          # gap that may carry punctuation
ADJ  = rf"(?:[A-Za-zÀ-ÿ]{{1,14}}{S}){{0,2}}"    # up to two intervening words
PAUL = r"(?:paul|paulus|paulo|pavel|pawel)"
# adjectival stem: a whitelist.  NOT `paulin\w*`, which also catches
# Paulinus (Church father), Paulina/Paulino (given names, towns), Pauling
# (surname), Paulinum (place).
PAULIN = (r"(?:pauline|paulines|paulinian|paulinians?|paulinism(?:e|er|er)?|"
          r"paulinismus|paulinista\w*|paulinien(?:ne)?s?|pauliniens?|"
          r"paulinisch(?:e|en|er|em|es)?|paulinische|paulinischer|paulinischem|"
          r"paulinisches|paulinischen|pauliniano\w*|pauliniano|pauliniana)")
WRIT = (r"(?:epistles?|letters?|literature|literatur\w*|litteratur\w*|corpus|writings?|"
        r"works?|homiliary|canon|correspondence|breviary|"
        r"ép[îi]tres?|lettres?|litt[ée]rature|briefe?|brieven?|brevi\w*|"
        r"epistula\w*|ep[íi]stolas?|epístol\w*|litteras?)")

PATTERNS: list[tuple[str, str, str]] = [
    # ---------------- tier A: explicitly identified ----------------------
    ("A1_apostle_pre", "A", rf"\bapostles?\b(?:{S}(?:of{S}the{S})?){ADJ}?{PAUL}\b"),
    ("A2_apostle_post", "A", rf"\b{PAUL}\b{G}(?:the{S})?{ADJ}?apostles?\b"),
    ("A3_apostle_foreign", "A",
     rf"\b(?:apôtre|apostel|apóstol\w*|apostolo|apostolus|apostulen|apostola)\b{G}"
     rf"(?:de{S}l'{S})?(?:the{S})?{ADJ}?{PAUL}\b"
     rf"|\b{PAUL}\b{G}(?:[l’]{S})?(?:apôtre|apostel|apóstol\w*|apostolo|apostolus)\b"),
    ("A4_tarsus", "A",
     rf"\b{PAUL}\b{G}(?:of|de|von|van|di|da|from){S}tars(?:us|e|se|o)\b"
     rf"|\b{PAUL}\b{G}(?:the{S})?tars(?:ian|ine|ites)\b"),
    ("A5_pauline_writings", "A",
     rf"\b{PAULIN}{S}(?:of{S}the{S})?{WRIT}\b"
     rf"|\b{WRIT}{S}(?:of|de|von|van|du|des|aux|di|da|ἐκ){S}{PAUL}\b"
     rf"|\b{PAUL}(?:['’]s)?{S}(?:the{S})?{WRIT}\b"
     rf"|\b(?:epistula\w*|litterae){S}Pauli\b"),
    ("A6_saint_paul", "A",
     rf"\b(?:st|ste|saint|sainte|sanctus|sankt|sint|sey|san)\b[.,]?{S}{ADJ}?{PAUL}\b"
     rf"|\bS\.{S}{PAUL}\b"),
    # ---------------- tier B: strong -------------------------------------
    ("B1_pauline_adj", "B", rf"\b{PAULIN}\b"),
    ("B2_paulus", "B", r"\bpaulus\b"),
    ("B3_of_paul", "B", rf"\b(?:of|de|von|van|du|des|di){S}{PAUL}\b"),
    ("B4_paul_genitive_abstract", "B",
     rf"\b{PAUL}['’]s{S}(?:theology|christology|soteriology|ecclesiology|ethics|thought|"
     rf"teaching|doctrine|tradition|mission|letter|epistle|gospel|conversion|apostleship|"
     rf"converts?|churches|communities|interpretation|reading|readers|rhetoric|law|spirit|"
     rf"self|figure|person|theolog\w*|concept|language|use|view|position)\b"),
    # ---------------- tier C: bare Paul, context decides ------------------
    # C1 is a *candidate* detector only: "bare Paul followed by a word".  The
    # lowercase-vs-capital decision CANNOT be made here, because every pattern is
    # matched against the lowercased normalised copy (`normalise` lowercases), so
    # by the time this regex runs, "Paul Lorenzen" and "Paul wrote" are byte-for-
    # byte identical.  The case test therefore happens in `scan()` against the
    # ORIGINAL text -- see C1_NEEDS_LOWERCASE_FOLLOW below.  A lookahead like
    # (?=[a-zà-ÿ]) here is a no-op and must not be mistaken for the real test.
    ("C1_paul_then_lowercase", "C", rf"\b{PAUL}\b{G}(?=[^\W\d_])"),
    ("C2_paul_possessive", "C", rf"\b{PAUL}['’]s\b|\b{PAUL}{S}s\b"),
    ("C3_paul_verb", "C",
     rf"\b{PAUL}\b{G}(?:the{S}|his{S}|and{S}|or{S})?(?:writes?|wrote|writing|says|said|"
     rf"speak\w*|spoke|tells?|told|insists?|argues?|claims|understands?|interprets?|"
     rf"uses?|used|cites?|quotes?|calls?|describes?|means|thinks?|believes?|proclaims?|"
     rf"preaches?|testifies?|reports?|warns?|promises?|prays?|suffers?|died|lives?|"
     rf"travels?|journeys?|arrives?|found\w*|establishes?|visits?|meets)\b"),
    ("C4_the_paul", "C",
     rf"\b(?:to|for|with|by|from|as|than|like|per|against|according|in|of|and|or|his|our|"
     rf"both|called)\b{S}{PAUL}\b"),
]
COMPILED = [(n, t, re.compile(p, re.I)) for n, t, p in PATTERNS]

# --- flags -----------------------------------------------------------------
# NOTE ON CASE.  `norm` is lowercased, `orig` keeps the original case.  Every
# rule here must therefore either (a) be applied to `orig`, or (b) carry re.I if
# it uses uppercase letters in its pattern.  F_NUMERAL_AFTER was missing re.I and
# was silently dead code, so `Paul VI` / `Paul II` were never flagged.
F_POPERY = re.compile(rf"(?:\bpope|\bpapst|\bantipope){S}(?:John{S})?$"
                      rf"|{S}our\s+Saint{S}$", re.I)
F_NUMERAL_AFTER = re.compile(
    rf"^{G}(?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV)\b(?![a-z])", re.I)
F_PLACE = re.compile(
    rf"^(?:['’]s)?{G}?(?:cathedral|church(?![a-z])|abbey|chapel|street|road|avenue|boulevard|hospital|"
    r"school|seminary|institute|college|university|press|publishers?|company|minnesota|"
    r"minneapolis|ontario|alberta|canada|,?\s*London\b|,?\s*Paris\b|,?\s*Minn)", re.I)
F_INITIAL_BEFORE = re.compile(r"[A-ZÀ-Ž]\.[\s]?$")
F_INITIAL_AFTER = re.compile(rf"^{G}[A-ZÀ-Ž]\.")

CITATION_HINTS = [
    re.compile(r"\((?:1[89]|20)\d\d[a-z]?\)"),          # (1991) / (1975a)
    re.compile(r"(?:1[89]|20)\d\d\s*[,.]"),
    re.compile(r"\b[A-Z][a-zà-ÿ]+,\s+[A-Z]\.(\s+[A-Z]\.)?"),   # Surname, A. B.
    re.compile(r"\b(?:pp?|vols?|eds?|trans|transl|no|nos)\.?\s+\d"),
    re.compile(r"\b(?:Press|Verlag|University|University Press|Brill|De Gruyter|OUP|"
               r"CUP|Mohr|Siebeck|Kegan Paul|Routledge|Herder|Peeters|Brepols)\b"),
    re.compile(r"\d{1,4}\s*[,:]\s*\d{1,4}\b"),          # 12:345 volume:page
]

SENT_ABBREV = {
    "st", "ste", "ss", "s", "no", "nos", "vol", "vols", "ed", "eds", "edn", "trans",
    "transl", "cf", "e.g", "i.e", "approx", "fig", "figs", "pp", "p", "ch", "vs", "rev",
    "jr", "sr", "dr", "prof", "mr", "mrs", "ms", "fr", "hon", "gen", "ex", "lev", "num",
    "deut", "jos", "jgs", "jdg", "ruth", "sam", "kg", "chr", "ezr", "neh", "est", "job",
    "ps", "pss", "prov", "eccl", "song", "isa", "jer", "lam", "ezk", "dan", "hos", "joel",
    "amos", "obad", "jon", "mic", "nah", "hab", "zep", "hag", "zech", "mal", "mt", "mk",
    "mr", "lk", "jn", "ac", "acts", "rom", "cor", "corinthians", "gal", "galatians",
    "eph", "phil", "col", "thess", "tim", "tl", "phlm", "heb", "jas", "pet", "jn", "jud",
    "rev", "apoc", "1", "2", "3", "4", "v", "vv", "l", "ll", "ibid", "op", "cit", "loc",
    "n", "nn", "sec", "para", "fl", "fr", "n.n", "s", "t", "d", "m", "j", "b", "c",
    "bce", "ce", "bc", "ad", "n.e", "b.g", "u.s", "u.k", "eng", "ger", "fr", "lat",
}


TAG_RE = re.compile(r"<[^<>]{0,60}>|&[a-zA-Z]{1,8};|&#[0-9]{1,5};")
NORMALISE_TRANS = str.maketrans({
    "\u00a0": " ", "\u2007": " ", "\u202f": " ", "\u00ad": " ", "\u200b": " ",
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u00ab": '"', "\u00bb": '"',
    "\t": " ", "\n": " ", "\r": " ", "\v": " ", "\f": " ", "\u2028": " ", "\u2029": " ",
})


def flatten(text: str) -> str:
    """Tags/entities blanked, all whitespace -> single space.  Length preserved."""
    text = TAG_RE.sub(lambda m: " " * (m.end() - m.start()), text)
    return text.translate(NORMALISE_TRANS)


def normalise(flat: str) -> str:
    """Lowercased `flat`; same length, so offsets stay valid on the original file."""
    low = flat.lower()
    if len(low) != len(flat):        # rare ligature/digraph expansions
        low = "".join(c if len(c.lower()) == 1 else " " for c in flat)
    return low


SENT_END = re.compile(r"[.!?\u2026]+[\s\u00a0]*")


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Abbreviation-aware sentence spans over the normalised text.

    Single newlines are already spaces (see `normalise`), so only real stops are
    considered; `Rom. 7,24`, `1 Cor.`, `St.`, `e.g.`, `1991.` and initials are
    kept inside the sentence.
    """
    spans: list[tuple[int, int]] = []
    n = len(text)
    low = text.lower()
    start = 0
    pos = 0
    while True:
        m = SENT_END.search(text, pos)
        if not m:
            break
        stop = m.end()                       # just past ". " and any quotes handled below
        j = stop
        while j < n and text[j] in "\"'”’)]} ":
            j += 1
        nxt = text[j] if j < n else ""
        pos = max(stop, pos + 1)
        if not (nxt.isupper() or nxt in "\"‘“("):
            continue
        before = text[max(0, m.start() - 26):m.start()]
        low_before = low[max(0, m.start() - 26):m.start()]
        tok = re.search(r"([a-zà-ÿ]+)$", low_before)
        if tok and tok.group(1) in SENT_ABBREV:
            continue
        if re.search(r"(?:vol|no|nos|pp|p|fig|ch|v|vv|l|ll|ed|eds|trans|ibid|sec|fr|"
                     r"saint|st|ste|s|fr|rev|prof|dr|mme|mlle|cap|par)\.$", before, re.I):
            continue
        if re.search(r"(?:^|[^a-zà-ÿ])(?:[a-z])\.$", low_before):   # single-letter initial
            continue
        if re.search(r"(?:1[0-9]|20)[0-9]{2}\.?$", before):        # year
            continue
        if text[m.start():m.start() + 2] in ("1.", "2.", "3.") and not text[:m.start()].count(" "):
            continue
        spans.append((start, j))
        start = j
    if n - start > 12:
        spans.append((start, n))
    return spans


META: dict[str, tuple[str, str, str]] = {}


def load_metadata() -> None:
    with METADATA.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            date = (row.get("published_date") or "")[:4]
            year = date if (date.isdigit() and len(date) == 4) else ""
            lang = (row.get("languages") or "").strip("[]'\" ").split("'")[0].strip("', ")
            META[row["item_id"]] = (year, row.get("is_part_of") or "", lang)


EXCLUSIONS: list[re.Pattern] = []


def load_exclusions(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            EXCLUSIONS.append(re.compile(rf"\b{re.escape(line)}", re.I))


def flags_for(norm: str, orig: str, start: int, end: int, pattern: str) -> list[str]:
    head = norm[max(0, start - CTX_CHARS):start]
    tail = norm[end:end + CTX_CHARS]
    o_head = orig[max(0, start - CTX_CHARS):start]
    o_tail = orig[end:end + CTX_CHARS]
    fl = []
    if F_POPERY.search(head) or F_NUMERAL_AFTER.match(norm[end:end + 10]):
        fl.append("papacy_or_numeral")
    if F_PLACE.match(tail):
        fl.append("place_or_building")
    if F_INITIAL_BEFORE.search(o_head):
        fl.append("initial_before")
    if F_INITIAL_AFTER.match(o_tail):
        fl.append("initial_after")
    if re.match(rf"^{G}(?:[A-ZÀ-Ž][a-zà-ÿ]{{2,}})", o_tail):
        fl.append("capital_after")
        if pattern.startswith(("B1", "A5")) and not re.match(rf"^{G}(?:of|the|and|,|du|des|de)\b",
                                                             o_tail, re.I):
            fl.append("possibly_given_name")
    # Bare "Paul" + Capital + no comma is the proper-name shape: "Paul Tillich",
    # "As Paul Harris argues", "Paul Walker".  A comma rules it out, because
    # "Paul, an old man" is ordinary punctuation while "Paul Tillich" is a name.
    # Only bare-Paul patterns can be fooled this way; tier A/B carry their own
    # disambiguating words.
    if pattern.startswith(("C1", "C2", "C4")):
        if re.match(rf"^{G}[A-ZÀ-Ž][a-zà-ÿ]{{2,}}(?!,)", o_tail):
            fl.append("probably_person_name")
    if any(rx.search(norm[max(0, start - 60):end + 60]) for rx in EXCLUSIONS):
        fl.append("excluded_name")
    line_start = orig.rfind("\n", 0, start) + 1
    line_end = orig.find("\n", end)
    line = orig[line_start:line_end if line_end > 0 else len(orig)].strip()
    letters = [c for c in line if c.isalpha()]
    if len(letters) > 6 and sum(c.isupper() for c in letters) / len(letters) > 0.85:
        fl.append("running_head")
    near = norm[max(0, start - 320):end + 320]
    if sum(bool(rx.search(near)) for rx in CITATION_HINTS) >= APPARATUS_MIN_HINTS:
        fl.append("citation_apparatus")
    return fl


#: A hit whose next non-gap character is a letter, and that letter is uppercase:
#: `Paul Sartre`, `Paul Ricoeur`.  This is the mechanism behind most tier-C
#: false positives, and it is measured in the notebook before being acted on.
CAP_AFTER_RE = re.compile(r"^[,\s;:.\u2026]*[^\W\d_]", re.UNICODE)
NEXT_LETTER_RE = re.compile(r"^[,\s;:.\u2026]*([^\W\d_])", re.UNICODE)


def _followed_by_lowercase(orig: str, end: int) -> bool:
    """True if the first letter after ``end`` in the ORIGINAL text is lowercase.

    This is the real body of the C1 heuristic.  It has to run on the original
    text, not on the normalised copy, because the normalised copy is lowercased
    and therefore cannot distinguish `Paul wrote` from `Paul Lorenzen`.
    """
    m = NEXT_LETTER_RE.match(orig[end:end + 12])
    return bool(m) and m.group(1).islower()


#: Tier-C patterns that additionally require a lowercase follow-word in the
#: original text.  Any hit from one of these patterns failing the test is dropped
#: and counted under ``dropped:not_lowercase`` in the run report.
C1_NEEDS_LOWERCASE_FOLLOW = {"C1_paul_then_lowercase"}

#: Flags that mark a row as a *likely* false positive.  They never remove a row
#: from the data, but `--fp-filter` keeps them out of the concordance_*.txt
#: listings so the text files stay readable.
#:
#: The flags only make sense for tier C, and the filter defaults to tier C only.
#: Rationale: `capital_after` fires on perfectly good tier A/B hits -- "Pauline
#: Letters", "Pauline Theology", "St Paul and Origen" -- because those patterns
#: already contain the words that disambiguate.  Bare "Paul" is the only tier
#: where "followed by a Capital" signals a proper name rather than ordinary
#: English.  Applied corpus-wide it removed 7,420 tier-A and 31,034 tier-B rows.
FP_FLAGS = ("papacy_or_numeral", "capital_after", "excluded_name",
            "citation_apparatus", "probably_person_name", "possibly_given_name",
            "initial_after", "place_or_building")
DEFAULT_FP_FILTER = ("papacy_or_numeral", "capital_after", "excluded_name",
                     "probably_person_name")
DEFAULT_FP_FILTER_TIERS = "C"


def kwic(orig: str, start: int, end: int, side: int | None = None):
    # `side` must NOT default to KWIC_CHARS directly: a default argument is
    # evaluated once, when this module is imported, so it would freeze the value
    # at 42 and ignore the `global KWIC_CHARS` assignment in main() -- in this
    # process and in every forked worker.  Read the global at call time instead.
    if side is None:
        side = KWIC_CHARS
    l = max(0, start - side)
    r = min(len(orig), end + side)
    left = orig[l:start]
    right = orig[end:r]
    if l > 0:
        left = left[left.find(" ") + 1:] if " " in left else left
    if r < len(orig):
        cut = right.rfind(" ")
        right = right[:cut] if cut > 0 else right
    return ("…" if l else "") + left.strip().replace("\n", "⏎"), \
           right.strip().replace("\n", "⏎") + ("…" if r < len(orig) else "")


FIELDS = ["item_id", "year", "bidecade", "journal", "language", "tier", "pattern",
          "flags", "char_start", "char_end", "match", "kwic_left", "kwic_right",
          "sentence", "doc_len"]


def scan(path: Path, tiers: frozenset[str]):
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None, None
    if "aul" not in text:
        return None, None
    flat = flatten(text)
    norm = normalise(flat)
    spans: list[tuple[int, int, str, str, str]] = []
    for name, tier, rx in COMPILED:
        if tier not in tiers:
            continue
        for m in rx.finditer(norm):
            spans.append((m.start(), m.end(), tier, name, norm[m.start():m.end()]))
    if not spans:
        return None, None
    # tier precedence: drop a hit that lies inside a strictly better tier's span
    rank = {"A": 0, "B": 1, "C": 2}
    spans.sort(key=lambda s: (rank[s[2]], -(s[1] - s[0])))
    kept: list[tuple[int, int, str, str, str]] = []
    for cand in spans:
        # spans are ordered best-tier-first, longest-first: the first hit that
        # overlaps a mention is the one that describes it best
        if any(k[0] < cand[1] and cand[0] < k[1] for k in kept):
            continue
        kept.append(cand)
    if not kept:
        return None, None
    sentences = split_sentences(flat)
    starts = [a for a, _ in sentences]
    year, journal, lang = META.get(path.stem, ("", "", ""))
    bidecade = ""
    if year.isdigit():
        bidecade = f"{int(year) // 20 * 20}-{int(year) // 20 * 20 + 19}"
    counts = Counter()          # accumulates tiers, patterns, flags, dropped:*
    rows = []
    for start, end, tier, name, match in sorted(kept):
        i = bisect_right(starts, start) - 1
        sent = ""
        if 0 <= i < len(sentences):
            a, b = sentences[i]
            sent = text[a:b].replace("\n", " ").strip()
        fl = flags_for(norm, flat, start, end, name)
        if "excluded_name" in fl and ({"capital_after", "initial_after"} & set(fl)) \
                and tier != "C":
            tier = {"A": "B", "B": "C"}[tier]
            fl.append("demoted_name_after")
        # C1's real test: the word after `Paul` must begin with a LOWERCASE letter
        # in the original text (see C1_NEEDS_LOWERCASE_FOLLOW).  `Paul Lorenzen`
        # and `Paul, A.` fail here; `Paul wrote` and `Paul à ce sujet` pass.
        if name in C1_NEEDS_LOWERCASE_FOLLOW and not _followed_by_lowercase(text, end):
            counts["dropped:not_lowercase_follow"] += 1
            continue
        # Tier C + Capitalised follow-word ("Paul Sartre", "Paul Ricoeur") is the
        # big measured false-positive class.  Default `keep` leaves the row in and
        # flagged; `drop` removes it and counts it under its own key so the report
        # shows exactly how much was removed.
        if TIER_C_CAPITALS == "drop" and tier == "C" and "capital_after" in fl:
            counts["dropped:tier_c_capital_after"] += 1
            continue
        left, right = kwic(text, start, end)
        rows.append((path.stem, year, bidecade, journal, lang, tier, name,
                     ";".join(fl), start, end, text[start:end].replace("\n", " "),
                     left, right, sent[:400], len(text)))
    for _, _, _, _, _, tier, name, flags, *_ in rows:
        counts[f"tier:{tier}"] += 1
        counts[f"pattern:{name}"] += 1
        for f in flags.split(";"):
            if f:
                counts[f"flag:{f}"] += 1
    return rows, counts


def _worker(args):
    path, tiers = args
    return scan(path, tiers)


def main() -> int:
    global CTX_CHARS, KWIC_CHARS, APPARATUS_MIN_HINTS, TIER_C_CAPITALS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tier", default="ABC", help="any subset of ABC (default ABC)")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="scan only N documents")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--sample", type=int, default=0, help="print N concordances and stop")
    ap.add_argument("--dump-eval", type=int, default=0,
                    help="write N stratified hits to paul_eval.tsv for labelling")
    ap.add_argument("--journal", default="", help="only this journal (is_part_of, substring)")
    ap.add_argument("--language", default="", help="only this language code, e.g. eng")
    ap.add_argument("--year-from", type=int, default=0)
    ap.add_argument("--year-to", type=int, default=9999)
    ap.add_argument("--no-hits", action="store_true", help="report only")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. --tag pilot")
    # --- tunables (defaults reproduce the 2026-09-04 run) -------------------
    ap.add_argument("--ctx", type=int, default=CTX_CHARS,
                    help=f"flag-detection window in chars (default {CTX_CHARS})")
    ap.add_argument("--kwic", type=int, default=KWIC_CHARS,
                    help=f"KWIC window in chars (default {KWIC_CHARS})")
    ap.add_argument("--apparatus-threshold", type=int, default=APPARATUS_MIN_HINTS,
                    help=f"citation-hint agreement needed to flag apparatus "
                         f"(default {APPARATUS_MIN_HINTS})")
    ap.add_argument("--tier-c-capitals", choices=("keep", "drop"), default=TIER_C_CAPITALS,
                    help="tier-C hits followed by a Capitalised word: keep (flag them) "
                         "or drop (default keep)")
    ap.add_argument("--fp-filter", default=",".join(DEFAULT_FP_FILTER),
                    help="comma-separated flags to EXCLUDE from the concordance_*.txt "
                         "listings only (never from hits.tsv.gz).  Use 'none' to write "
                         "every hit.  Default: " + ",".join(DEFAULT_FP_FILTER))
    ap.add_argument("--fp-filter-tiers", default=DEFAULT_FP_FILTER_TIERS,
                    help="which tiers --fp-filter applies to (default "
                         f"'{DEFAULT_FP_FILTER_TIERS}'; use 'ABC' for all, 'none' to "
                         "disable the filter entirely)")
    args = ap.parse_args()

    # Apply tunables module-wide before any worker is spawned.  On Linux the
    # ProcessPoolExecutor forks, so children inherit these values; the explicit
    # assignment here is also what makes the behaviour visible in one place.
    CTX_CHARS = args.ctx
    KWIC_CHARS = args.kwic
    APPARATUS_MIN_HINTS = args.apparatus_threshold
    TIER_C_CAPITALS = args.tier_c_capitals

    # Flags to keep out of the human-readable concordance text files.  This is a
    # presentation filter only: hits.tsv.gz always keeps every row.
    if args.fp_filter.strip().lower() in ("", "none"):
        fp_flags: tuple[str, ...] = ()
    else:
        fp_flags = tuple(f.strip() for f in args.fp_filter.split(",") if f.strip())
    fp_tiers = frozenset(t for t in args.fp_filter_tiers.upper() if t in "ABC")
    print(f"[init] concordance listings exclude "
          f"{', '.join(fp_flags) if fp_flags else '(nothing)'} "
          f"from tier(s) {''.join(sorted(fp_tiers)) or '(none)'}")

    tiers = frozenset(t for t in args.tier.upper() if t in "ABC")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dirs = [out / "paul_exclusions.txt", OUT / "paul_exclusions.txt",
            Path(__file__).with_name("paul_exclusions.txt")]
    for cand in dirs:
        if cand.exists():
            load_exclusions(cand)
            break
    print(f"[paul] metadata …", flush=True)
    load_metadata()
    print(f"[paul] {len(META):,} metadata rows, {len(EXCLUSIONS)} exclusion patterns", flush=True)

    files = sorted(TXT.glob("*.txt"))
    if args.limit:
        files = files[:args.limit]
    if args.journal or args.language or args.year_from or args.year_to:
        def keep(p: Path) -> bool:
            year, journal, lang = META.get(p.stem, ("", "", ""))
            if args.journal and args.journal.lower() not in journal.lower():
                return False
            if args.language and not lang.startswith(args.language):
                return False
            if year.isdigit() and not (args.year_from <= int(year) <= args.year_to):
                return False
            return True
        files = [p for p in files if keep(p)]
    print(f"[paul] scanning {len(files):,} documents, tiers={''.join(sorted(tiers))}", flush=True)

    tag = f"_{args.tag}" if args.tag else ""
    hits_path = out / f"hits{tag}.tsv.gz"
    writer = doc_writer = None
    fh = gz = None
    if not args.no_hits and not args.sample:
        fh = open(hits_path, "wb")
        gz = gzip.open(fh, "wt", encoding="utf-8", newline="")
        writer = csv.writer(gz, delimiter="\t", lineterminator="\n")
        writer.writerow(FIELDS)
    kwic_files: dict[str, object] = {}
    if gz:
        for t in sorted(tiers):
            kwic_files[t] = open(out / f"concordance_{t}{tag}.txt", "w", encoding="utf-8")
    doc_path = out / f"hits_by_doc{tag}.tsv.gz"
    dfh = gzip.open(doc_path, "wt", encoding="utf-8", newline="") if (gz and not args.sample) else None
    doc_writer = csv.writer(dfh, delimiter="\t") if dfh else None
    if doc_writer:
        doc_writer.writerow(["item_id", "year", "bidecade", "journal", "language",
                             "tier_a", "tier_b", "tier_c", "total"])

    totals = Counter()
    docs_with = Counter()
    samples: list[tuple] = []
    eval_rows: list[tuple] = []
    n_docs = n_hits = 0
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        for rows, counts in ex.map(_worker, ((p, tiers) for p in files), chunksize=32):
            # `counts` is merged *before* the empty check: a document whose every
            # hit was rejected by the C1 follow-word test contributes no rows, but
            # its `dropped:*` tallies still belong in the report.
            totals.update(counts)
            if not rows:
                continue
            n_docs += 1
            n_hits += len(rows)
            for r in rows:
                docs_with[(r[0], r[5])] = 1
            if writer:
                writer.writerows(rows)
                for r in rows:
                    f = kwic_files.get(r[5])
                    if not f:
                        continue
                    # Presentation filter: keep likely false positives out of the
                    # readable listing, but count how many that removes.  Scoped
                    # to fp_tiers because the flags mean different things per tier.
                    if fp_flags and r[5] in fp_tiers and any(x in r[7] for x in fp_flags):
                        totals[f"fp_filtered:{r[5]}"] += 1
                        continue
                    f.write(f"{r[11]:>{KWIC_CHARS}} ⟪{r[10]}⟫ {r[12]:<{KWIC_CHARS}}"
                            f"  {r[1]} {r[3][:32]}" + chr(10))

            if args.sample or args.dump_eval:
                for r in rows:
                    if len(samples) < args.sample and hash((r[0], r[8])) % 97 == 0:
                        samples.append(r)
                    if len(eval_rows) < args.dump_eval and hash((r[0], r[8])) % 331 == 0:
                        eval_rows.append(r)
            if doc_writer:
                per = Counter(r[5] for r in rows)
                doc_writer.writerow([rows[0][0], rows[0][1], rows[0][2], rows[0][3], rows[0][4],
                                     per.get("A", 0), per.get("B", 0), per.get("C", 0), len(rows)])
            if n_docs % 40000 == 0:
                el = time.time() - t0
                print(f"  {n_docs:,} docs with hits  {el:5.0f}s  {len(files)/max(el,1):,.0f} docs/s",
                      flush=True)
    if gz:
        gz.close()
        fh.close()
    if dfh:
        dfh.close()
    for f in kwic_files.values():
        f.close()
    el = time.time() - t0

    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "documents_scanned": len(files),
        "documents_with_hits": n_docs,
        "hits": n_hits,
        "tiers": "".join(sorted(tiers)),
        "seconds": round(el, 1),
        "totals": dict(sorted(totals.items())),
        "docs_with_hits_per_tier": {t: sum(1 for k in docs_with if k[1] == t) for t in sorted(tiers)},
    }
    (out / f"scan_report{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.sample:
        print("[paul] wrote " + ", ".join(f"concordance_{t}.txt" for t in sorted(tiers))
              + " + hits.tsv.gz + hits_by_doc.tsv.gz")

    print(f"\n[paul] {n_hits:,} hits in {n_docs:,} documents, {el:.0f}s")
    print(f"  {'key':34s}{'value':>12s}")
    for k, v in sorted(totals.items()):
        if k.startswith(("tier:", "pattern:", "doc_")):
            print(f"  {k:34s}{v:12,}")
    print("  ---- flags ----")
    for k, v in sorted(totals.items()):
        if k.startswith("flag:"):
            print(f"  {k:34s}{v:12,}")
    for t in sorted(tiers):
        print(f"  docs with tier {t}: {report['docs_with_hits_per_tier'][t]:,}")

    if args.dump_eval:
        p = out / f"paul_eval{tag}.tsv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(FIELDS + ["verdict"])
            for r in eval_rows:
                w.writerow(list(r) + [""])
        print(f"\n[paul] wrote {len(eval_rows)} rows for labelling -> {p}")
    if samples:
        print(f"\n=== {len(samples)} sample concordances (tier {args.tier}) ===")
        for r in samples[:args.sample]:
            print(f"  [{r[6]:22s}] {r[11]} ⟪{r[10]}⟫ {r[12]}   ({r[0][:8]},{r[1]},{r[3][:28]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
