### DONE

**2026-10-15 — web app rebuilt from the notebook: `scripts/mops_jstor.py` (`/mops/`)**

- Ported the "Temporal lineplots" section of `scripts/5_tokens-analysis.ipynb` into a
  multi-page streamlit app, deployed through the existing systemd + nginx setup.
- Layout: `scripts/mops_jstor.py` (entry, `st.navigation`) + `scripts/mops_app/`
  (`data.py` loaders/filters/temporal aggregation — **no streamlit import**, so notebooks can
  reuse it; `cache.py` the `st.cache_data` wrappers; `plots.py` the matplotlib port of
  `plot_timebins` plus its plotly twin; `pages/temporal.py`, `pages/concordance.py`).
- New `scripts/prepare_jstor_app_data.py` flattens `hits_nlp.parquet` (whose nested `kwic_nlp`
  column dominates load time) into `data/large_files/paul_scan/tokens.parquet` — 426,917 rows,
  18 MB, built in 20 s; carries hit metadata + `filtered_lemmata`. `(item_id, char_start)` is
  asserted unique and joins 100% to `hits.parquet` for KWIC display.
- **Notebook parity verified**: `jewish`/`judaism`/`jew`, bidecades, English, no
  discipline/confession filter, metric `per_20` reproduces all 18 recorded values and all six
  `subset_length`s (13,509 / 11,062 / 21,082 / 36,843 / 76,556 / 107,307) exactly. Pinned in
  `scripts/check_mops_app.py` (also smoke-tests every page via streamlit's `AppTest`).
- Defects fixed while porting: `from_words_to_freqs` had no `return` and a syntax error;
  `hits_filtering` was defined with one parameter and called with four; the 1900–2020 window and
  the `×20` scale were hard-coded. Replaced by `data.filter_tokens` + `data.words_over_time`;
  counting is a pandas groupby on the exploded lemma column, so the app has no `nltk` runtime
  dependency (the notebook used `nltk.FreqDist`).
- Deployment: unit `/etc/systemd/system/mops.service` (updated: `--server.headless true`,
  `PYTHONUNBUFFERED=1`, `--server.baseUrlPath mops`; backup at `mops.service.bak-*`) enabled +
  started. nginx `/mops/` block changed to `proxy_pass http://127.0.0.1:8061/mops/;` with the
  `rewrite ^/mops(/.*)$ $1 break;` **removed** (backup `ccs-lab.bak-*`).
- **Why `baseUrlPath` is needed here but not for the six sibling apps** (they all strip the
  prefix and work): streamlit's *bootstrap* HTML uses relative asset URLs (`./static/…`), which
  the stripping rewrite serves fine; but anything the frontend resolves against `location.pathname`
  breaks once the URL has more than one segment. Reloading or deep-linking a sub-page made the app
  request `/mops/_temporal/_stcore/health` (404). With `baseUrlPath=mops` all of it is
  `/mops/_stcore/*` (200). Verified: `/mops/`, `/mops/_temporal`, `/mops/_concordance`,
  `/mops/_stcore/health` all 200; the frontend's 404 probes on the bare page path are harmless
  (it falls back). Test the **sub-path reload**, not just the index page.
- Also added an nginx `location /media/ { proxy_pass http://127.0.0.1:8061/mops/media/; }`:
  streamlit 1.44 builds download-button URLs as bare `/media/<id>`
  (`web/server/server.py: MemoryMediaFileStorage(MEDIA_ENDPOINT)`) while only *serving* them under
  the base path, so every "Download as CSV" would 404. Nothing else on the host serves `/media/`.
  (Verified working by the user on the real machine; the headless browser could not capture the
  download tab, which is a browser-harness limitation, not an app bug.)
- `streamlit` added to `requirements.txt`; `__pycache__/` added to `scripts/.gitignore`;
  app docs in `scripts/mops_app/README.md`.
- One production crash found *by* the deployment and fixed: `words_over_time` used
  `denominator.replace(0, pd.NA).astype(float)` for a bin with zero concordances. That `replace`
  makes the series object-dtype, so `astype(float)` raises `TypeError: float() argument must be a
  string or a real number, not 'NAType'` — it only fires when a filter empties a bin (narrow year
  window, or a discipline the old journals do not belong to). Now `data._non_zero()`
  (`astype(float).where(> 0)`), covered by `check_mops_app.py`'s edge-case checks.

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
- **Paul concordance built and run** (`/srv/data/jstor/jstor2026-04/paul_concordance.py`, outputs
  in `paul_scan/`). **Checkpoint: `paul_scan/RUN_STATE.md` — read it first after any interruption**
  (what is finished, row counts, verify command, ordered next steps). Tiered regexes
  (A explicit / B strong / C bare-`Paul`-by-context) over all 384,807 texts in **298 s** →
  **571,896 hits in 106,005 articles (27.5%)**; **tier A 76,163 hits in 26,263 articles**
  (`A1` apostle Paul 3,179 · `A2` Paul the apostle 1,245 · `A3` foreign apostle 2,885 ·
  `A4` Paul of Tarsus 458 · `A5` Pauline / letters-of-Paul writings 17,178 ·
  `A6` St./Saint/Sanctus Paul 52,207), tier B 145,240 (`Pauline` 39,957 · `Paulus` 49,617 ·
  `of Paul` 50,474), tier C 350,493 (bare `Paul` + lowercase 250,661 …). Each row carries
  `pattern`, `tier`, `flags` (`papacy_or_numeral`, `place_or_building`, `excluded_name`,
  `citation_apparatus`, `running_head`, …), `char_start/end` into the file, a KWIC window and the
  abbreviation-aware sentence. Files: `hits.tsv.gz` (71 MB), `hits_by_doc.tsv.gz`,
  `concordance_A|B|C.txt`, `scan_report.json`, `paul_exclusions.txt`, `paul_eval.tsv`, `probes/`.
  Sanity check: tier-A density ranks the NT journals first (Revue Biblique, CBQ, Neotestamentica,
  Novum Testamentum 37% of its articles, Biblica, JBL) while `Synthese` sits at 0.6%.
  Notebook entry point at the end of `scripts/jstor-metadata-loader.ipynb`: `paul_hits(tier=…,
  journal=…, language=…, exclude_flags=…)` and `paul_concordance(hits)`.
  Re-run: `python3 /srv/data/jstor/jstor2026-04/paul_concordance.py --workers 30`
  (`--tier ABC --journal … --language eng --year-from/-to --sample N --dump-eval N --tag pilot`).
- Repo tidy-up (uncommitted): numbered notebooks live in `scripts/backup1` / `scripts/backup2`,
  old pickles in `data/backup/`, old n-gram artifacts in `data/large_files_BACKUP/` (both
  gitignored); `data/large_files/` is now empty. `data/journals_religion_2025.csv` = per-journal
  article counts for the Religion subset (185 journals).
- **State**: the corpus models have not been re-run on the 2026 delivery yet — the only 2026-corpus
  work so far is the Paul scan above. The 2022–2025 pipeline and all of its derived artifacts are
  frozen and still readable, but several scripts still point at the pre-move paths (see "Known
  breakages" in `README.md`).

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

- [ ] **Paul concordance — label and publish the precision figures.** The scan itself is built
      and run (see DONE and `paul_scan/RUN_STATE.md`); what is left is scholarly validation,
      not code:
      - [ ] label `/srv/data/jstor/jstor2026-04/paul_scan/paul_eval.tsv` (280 rows, 20 per
            pattern, `verdict` column empty) → publish precision per pattern;
      - [ ] decide which tiers back each quantitative claim ("76,163 explicit references",
            "221k explicit + strong", "572k including bare Paul") and state it once, in the paper;
      - [ ] Greek `Παῦλος`/`Σαῦλος` still unmatched (polytonic-normalised pattern needed);
      - [ ] if precision on tier C must be higher than regex gives: train a small classifier on
            the labelled `paul_eval.tsv` (the row already carries pattern + flags + sentence);
      - [ ] extend `paul_exclusions.txt` from the `excluded_name` / `demoted_name_after` rows
            (43,662 / 4,507) — data file, no code change needed, re-run is 5 min.
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
