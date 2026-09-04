### DONE

**2026-09-04 — project picked up again; new (2026-04) ITHAKA/JSTOR delivery ingested**

- Located the new delivery on `/srv/data/jstor/jstor2026-04/` (outside git): metadata
  `jstor_metadata_2026-04-28.jsonl` (13.7 GB) + `...-filtered.csv` (**384,829** rows, 320
  journals, Philosophy ∪ Religion, 1829–2026, 237,345 English-language) and the text delivery
  `f18084f6-....jsonl.gz` (6.0 GB, **384,807** records). `iid` == metadata `item_id`.
- Wrote `/srv/data/jstor/jstor2026-04/jstor_fulltext_parser.py` (stdlib only, fully documented:
  schema, parsing steps, and why each normalisation choice was or was not applied) and ran it:
  `fulltexts_txts/` 384,807 files / 14.86 GiB + `references_txts/` 279,485 files / 1.23 GiB,
  0 failures, 275 s. `--verify` is OK; `extract_logs/extract_report.json` has the run stats.
  Reading one article is now a 0.04 ms file open instead of a ~105 s scan of the gzip.
- Documented the layout and added `jstor_text()` / `jstor_references()` helpers at the end of
  `scripts/jstor-metadata-loader.ipynb`; notes in `~/.claude/memory/jstor-2026-04.md`.
- Repo tidy-up (uncommitted): numbered notebooks live in `scripts/backup1` / `scripts/backup2`,
  old pickles in `data/backup/`, old n-gram artifacts in `data/large_files_BACKUP/` (both
  gitignored); `data/large_files/` is now empty. `data/journals_religion_2025.csv` = per-journal
  article counts for the Religion subset (185 journals).
- **State**: nothing has been re-run on the 2026 corpus yet. The 2022–2025 pipeline and all of its
  derived artifacts are frozen and still readable, but several scripts still point at the pre-move
  paths (see "Known breakages" in `README.md`).

**2022 – Apr 2025 — the original study (120 years of Pauline scholarship)**

- Input: ITHAKA n-grams dataset `a414f68a-...-jsonl.jsonl.gz` read through the `constellate`
  client, 25,070 articles (`id_kase` 0–25069), 1900–2019, bidecadal slices 1900-1919 … 2000-2019.
- Pipeline: parse per-article unigram/trigram counts → spaCy (`en_core_web_lg`) lemmatisation and
  POS/stopword/collation cleaning → journal-level vocabulary and frequency comparisons (UMAP,
  `figures/journals_comparison.png`) → count-based SGNS matrices (vendored `scripts/histwords`)
  and per-bidecade word2vec → semantic-change analysis of target terms, above all **"Paul"**
  (`figures/paul_*.png`, `4_word2vec_analysis.ipynb`); OpenAlex + Google-Sheet enrichment
  (`2_oalex-metadata.ipynb`, `scripts/google_conf.py`); interactive exploration app
  (`scripts/backup1/mops-vectors_streamlit.py`).
- Outputs: `data/backup/*.pickle` (co-occurrence and frequency dicts, vocabularies, vector
  positions), `data/large_files_BACKUP/` (raw/filtered/lemmatised n-grams, bidecade trigram
  texts, `.kv` vectors), `figures/`.


### TO-DO

- [ ] design a regex commamnd to match "Paul" when refering to Paul the Apostle, not Paul Robertson etc. Thus, "Paul'?\s\[lowercase...]" is acceptible in most cases, looking at theological literature. the same when interpunction follows
- [ ] rebuild the corpus pipeline on the 2026-04 delivery (384,807 full texts vs. the old 25,070
      n-gram-only articles) — decide first whether to re-lemmatise from text or to request the
      n-grams dataset again for the wider 2026 selection
- [ ] derive `bidecade` and an English-only filter for the 2026 metadata (the old pipeline had
      both; `published_date` and `languages` are present but unused so far)
- [ ] repair the pre-move data paths before re-running any old notebook (21 notebooks, 53 dead
      paths — `data/*.pickle` now live in `data/backup/`, `data/large_files/*` in
      `data/large_files_BACKUP/`). Symlink recipe and the list of 18 genuinely missing
      intermediates are in `README.md` → "Known breakages". Includes
      `mops-vectors_streamlit.py` and the two `colab_*.ipynb` notebooks.
