# MOPS — 120 years of Pauline scholarship, mined

Computational study of the language of modern Pauline and religious scholarship: word-embedding
and vocabulary models trained on JSTOR/ITHAKA deliveries of journal articles, used to trace how
key terms — above all **"Paul"** — change meaning across the twentieth century
(`figures/paul_*.png`, `figures/journals_comparison.png`).

Two corpora live in this project:

| | study corpus (2022 – Apr 2025) | new delivery (since Sep 2026) |
| --- | --- | --- |
| size | 25,070 articles, 1900–2019 | 384,829 articles, 1829–2026 |
| form | ITHAKA **n-gram counts** dataset | JSTOR **full text + references** |
| location | `data/large_files_BACKUP/`, `data/backup/` | `/srv/data/jstor/jstor2026-04/` (outside git) |
| status | frozen, all artifacts intact | ingested and readable, **not yet modelled** |

> Status and next steps are tracked in [`AGENTS.md`](AGENTS.md) (`### DONE` / `### TO-DO`).

---

## Authors

* Vojtěch Kaše
* Nina Nikki

## License

CC-BY-SA 4.0, see attached LICENSE.md

---

## Getting started

### Environment

The machine already has a working environment at `mops_venv/` (Python 3.12.3, jupyter, pandas
2.2, numpy 1.26, gensim 4.3, spacy 3.8, umap-learn, plotly, constellate-client). To recreate it
from scratch:

```bash
which python3                     # interpreter to build the venv from
pip install virtualenv

VENVNAME=mops_venv
INTERPRETER=$(which python3)      # or pin one, e.g. $HOME/.local/lib/python3.12/bin/python3
virtualenv $VENVNAME --python=$INTERPRETER

$VENVNAME/bin/python -m pip install -r requirements.txt
$VENVNAME/bin/python -m ipykernel install --user --name=$VENVNAME
$VENVNAME/bin/python -m spacy download en_core_web_lg   # lemmatisation step needs it

mkdir -p data/large_files         # gitignored scratch space for derived artifacts
```

Always check that a notebook is attached to the `mops_venv` kernel.
`scripts/histwords/` is a vendored copy of the histwords library (SGNS matrices, alignment);
it is used from the notebooks directly and needs no separate install.
`scripts/backup1/mops-vectors_streamlit.py` needs `streamlit`, which the venv has but
`requirements.txt` does not list.

### Reading a single article (fast path)

The 2026 text delivery is materialised as one file per article, so a read is a file open
(~0.04 ms) rather than a ~105 s scan of the 6 GB gzip:

```python
from pathlib import Path

FULLTEXTS  = Path("/srv/data/jstor/jstor2026-04/fulltexts_txts")
REFERENCES = Path("/srv/data/jstor/jstor2026-04/references_txts")

text = (FULLTEXTS / f"{item_id}.txt").read_text(encoding="utf-8")
ref_path = REFERENCES / f"{item_id}.txt"
refs = ref_path.read_text(encoding="utf-8").splitlines() if ref_path.exists() else []
```

`item_id` is the JSTOR `item_id` / `iid` from the metadata CSV; a missing reference file means the
article has no reference list. `scripts/jstor-metadata-loader.ipynb` wraps this in
`jstor_text()` / `jstor_references()` and explains the layout.

---

## Data

### A. The 2026-04 delivery — `/srv/data/jstor/jstor2026-04/` (not in git)

| path | contents |
| --- | --- |
| `jstor_metadata_2026-04-28.jsonl` | full metadata delivery, 13.7 GB |
| `jstor_metadata_2026-04-28-filtered.csv` | the selection actually used: **384,829** rows, all `content_type=article`, 320 journals, Philosophy (202,554) ∪ Religion (200,667), 237,345 English-language, years 1829–2026 (348,317 inside the old 1900–2019 window) |
| `f18084f6-…-…jsonl.gz` | text delivery, 6.0 GB, **384,807** records: `iid`, `full_text` (list of OCR pages), `references` (list of strings) |
| `f18084f6-…_unvalidated.txt` | 22 `item_id`s that have metadata but no text |
| `fulltexts_txts/` | 384,807 `<item_id>.txt`, unpaged continuous text, 14.86 GiB (295 empty: whitespace-only OCR) |
| `references_txts/` | 279,485 `<item_id>.txt`, one bibliography entry per line, 1.23 GiB |
| `jstor_fulltext_parser.py` | the extractor — run it to rebuild, `--verify` to check, `--help` for flags; its module docstring documents the schema and every normalisation decision |
| `extract_logs/` | `extract_report.json` (run stats), `extract.log`, `extract_errors.log` (empty = clean run) |

Rebuild takes ~4.5 min single-threaded; re-running resumes (only missing files are written).

### B. The 2022–2025 study corpus (gitignored, on disk)

* `data/large_files_BACKUP/` — derived per-article n-grams (`unigrams_raw|filtered|lemmatized`,
  `trigrams_*`), cleaned lemma dictionaries, per-bidecade trigram texts
  (`bidecade_trigrams_1900-1919.txt` … `2000-2019.txt`), trained vectors
  (`vectors_<bidecade>_sample{1000000,2000000}.kv`) and the original ITHAKA delivery
  `a414f68a-…-jsonl.jsonl.gz`.
* `data/backup/` — pickles the notebooks load: co-occurrence matrices (`cooc_dict*.pickle`),
  frequency dictionaries (`freqs_2000.pickle`, `bidecades_freqs_complete.pickle`), vocabularies,
  cleaning dictionaries, metadata (`metadata_df.json`, `metadata_rich_oalex.json`), 3-D vector
  positions for the interactive app.
* `data/journals_religion_2025.csv` — article counts per journal in the Religion subset (185 journals).

### C. Annotations

Journal classification and other hand-curated labels live in a Google Sheet read through
`scripts/google_conf.py` (`google_conf.setup(sheet_url=…, service_account_path=…)`); the service
account key is deliberately outside the repository and notebooks look for it at
`../../../ServiceAccountsKey.json`.

---

## Repository layout

```
AGENTS.md                  running log: what is DONE, what is TO-DO
README.md                  this file
requirements.txt           python dependencies
figures/                   output plots of the 2022-2025 study
data/
  backup/                  pickles/json of the old pipeline (moved out of data/ in Sep 2026)
  large_files/             gitignored scratch space (currently empty)
  large_files_BACKUP/      gitignored per-article n-grams, vectors, old delivery
scripts/
  jstor-metadata-loader.ipynb   current entry point: 2026 metadata + text access
  google_conf.py                Google-Sheet / service-account helper
  histwords/                    vendored histwords library
  backup1/, backup2/            the numbered notebooks of the 2022-2025 pipeline
```

`scripts/backup1` is the earlier iteration (constellate parsing, LDA development, the streamlit
app); `scripts/backup2` holds the final 2025 pipeline, whose steps are:

1. `1_jstor-parsing.ipynb` — read the ITHAKA n-grams delivery, write per-article raw n-grams.
2. `2_lemmatization-or-stemming.ipynb`, `unigrams-cleaning.ipynb` — spaCy lemmatisation, POS and
   stopword filtering, collation/collocation cleaning.
3. `2_jstor-metadata-explorations.ipynb`, `2_oalex-metadata.ipynb` — metadata curation, OpenAlex
   enrichment, `bidecade` split.
4. `3_exploring-vocabularies.ipynb` — journal-level vocabulary/frequency comparison (UMAP).
5. `3_word2vec_training.ipynb`, `4_count-based-matrices.ipynb` — per-bidecade word2vec and
   count-based SGNS matrices (histwords).
6. `4_word2vec_analysis.ipynb`, `7_word2vec_analysis-continuation.ipynb` — semantic change of the
   target terms, nearest neighbours of "Paul" over time.
7. `backup1/mops-vectors_streamlit.py` — interactive 3-D vector exploration app.

---

## Known breakages after the September 2026 tidy-up

The old notebooks and the streamlit app were written before `data/*.pickle` moved to
`data/backup/` and `data/large_files/*` moved to `data/large_files_BACKUP/`, so **none of their
data paths resolve today** — 53 distinct paths across 21 notebooks, checked programmatically.
35 of them still exist, only under the new names; to re-run an old notebook, restore the old
layout with links instead of editing every cell:

```bash
ln -s backup/* data/                          # pickles, json, metadata
mkdir -p data/large_files && ln -s ../large_files_BACKUP/* data/large_files/
ln -s constellate_metadata_rich_df.json data/metadata_rich_df.json   # see below
```

The remaining 18 were never committed and are gone from this machine: the first-pass n-gram
dictionaries (`{uni,bi,tri}gramCount_dict.pickle`, their `_cleaned` variants,
`unigrams_merged_cleaned.pickle`, `unigrams_lemmata_dict_unfiltered.pickle`,
`data_trigrams*.pickle`, `keys2_filtered.pickle`, `jstor_df_v1.feather`, `article_docs/`), the LDA
model `lda_global_v1`, and `entries_v1.pkl`. Most belong to the earliest (`scripts/backup1`)
iteration; the only gaps that touch the final 2025 pipeline are
`unigrams_lemmata_dict_unfiltered.pickle` and the two `*gramCount_dict.pickle` used by
`2_jstor-metadata-explorations.ipynb`.

One gap is already solved: the missing `data/metadata_rich_df.json` is `data/constellate_metadata_rich_df.json`
(verified — same 25,000 `id_kase`, identical `doi`/`language`/`bidecade`;
`data/backup/metadata_rich_oalex.json` is that same frame plus five OpenAlex columns).

The `colab_*.ipynb` notebooks also link to `https://github.com/kasev/mops/raw/master/data/…pickle`,
which now 404s for the same reason.
