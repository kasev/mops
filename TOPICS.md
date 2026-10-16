# TOPICS.md — decision record for the topic model

Methodological record for the BERTopic model of the Paul concordance: **what was decided, on
what evidence, and what was rejected**. Written to be sufficient to reproduce the model without
reading the code, and to state plainly what a result from it may and may not claim.

* **Status:** design frozen; **production fit complete (2026-10-16)** — 50 topics over 379,859
  documents, 56.2% outliers (§4.1; the encoder was re-reviewed in D6.1 and the HDBSCAN parameters
  corrected in D8.2). Items still open are listed in
  [§8](#8-open-items-not-yet-decided) — do not cite a value for them.
* **Decided:** 2026-10-15. **Deciders:** Vojtěch Kaše (domain), implementation by the assistant.
* **Related:** `AGENTS.md` (status log), `README.md` (project), `scripts/mops_app/README.md`
  (the web app that serves the artifacts).

---

## 1. Scope

The topic model describes **what is said in the immediate verbal neighbourhood of a mention of
Paul**, across the JSTOR 2026-04 Philosophy ∪ Religion corpus, and how those recurring patterns
move over time. It is a description of *collocations around the name*, not of articles, of
authors, or of Paul as a historical figure.

## 2. Data

Source: `data/large_files/paul_scan/tokens.parquet`, built by
`scripts/prepare_jstor_app_data.py` from the Paul-scan NLP table
(`hits_nlp.parquet` ← `hits.parquet` ← `scripts/paul_concordance.py`; see `AGENTS.md`).

| quantity | value |
| --- | --- |
| rows in `tokens.parquet` (concordance hits) | 426,917 |
| after D2 (language) | 414,922 (97.19%) |
| after D3 (1900–2019) | 380,850 |
| **fitted after D4 (documents with ≥1 content lemma)** | **379,859** |
| distinct articles | 59,943 |
| distinct journals | 297 |
| years | 1900–2019 |
| lemmas per document | mean 6.9, median 7, max 20 |
| languages present | `eng` 257,675 · `fre` 57,840 · `ger` 56,306 · `en` 8,038 |

**D4 is nearly inert here** (991 documents dropped, 0.26%): once D13 was repaired almost every
in-scope hit has content words. That is itself informative — the earlier 15.45% drop rate was
almost entirely the French defect, not a property of the corpus.

*(For contrast: dropping the D3 year window entirely would give a fitted set of 420,873 documents.)*

## 3. Decisions

Each decision states the choice, the evidence, and the alternative that was rejected.

### D1 — Unit of analysis: one concordance line, not an article or a sentence
**Choice.** One document = one hit = the **±10-token window** around the match, lemmatised.
**Rationale.** A hit is the analytic unit of the whole project (the concordance, the tier system,
the app filters). Keeping it as the topic unit means every topic is traceable to an exact
`(item_id, char_start)` and therefore to a quotable KWIC line. It also matches the established
practice in this author's comparable work, where BERTopic runs on lemmatised sentences.
**Rejected.** *Per-article* (69,709 docs): would blur distinct contexts and lose the per-hit
traceability. *Per-sentence* (from `hits.parquet`'s `sentence`): longer, more natural documents,
but reintroduces text whose lemmatisation was never validated for this purpose, and overlaps the
window unit without adding information the window lacks.
**Consequence to state in any write-up.** The documents are **fixed-width windows, not
self-contained texts**, and windows of adjacent hits in one article overlap, so observations are
**not independent**. Cluster sizes describe recurring configurations in Paul's verbal
neighbourhood, not shares of discourse.

### D2 — Corpus scope: English + French + German (`eng`, `en`, `fre`, `ger`)
**Choice.** Fit on those four JSTOR codes: 280,919 + 8,045 + 66,059 + 59,899.
**Rationale.** These are the languages the models support and the languages in which this
scholarship is actually conducted; French and German journals are a substantive part of the
corpus (**125,958 docs, 30.4%** of the in-scope total) and excluding them would misrepresent
the field.
**Status.** Repaired under D13: all four codes are now populated.
**Rejected.** *English only*: simpler and the app's default for the temporal view, but discards
126k documents. *All languages*: Latin (3,949) is **not supported** by the chosen models —
including it would silently garble those documents. The remaining tail (ita, dut, por, spa, heb,
pol, afr; ≈8k) is off by default to keep the semantic space coherent; adding it is a parameter.
**Note.** `en` and `eng` are both English and **must always travel together** — this is a
recurring trap in this corpus.

### D3 — Year window: 1900–2019 (the study window), parameterised
**Choice.** Default the fit to 1900–2019 → 379,859 documents. The full range (1852–2026) is
available as a parameter and excludes nothing structurally.
**Rationale.** Every other figure in this project is stated over 1900–2019; fitting topics over a
different span would make topic trends non-comparable with the temporal page and the published
plots. The excluded tail (25,945 docs, 7.5%) is thin and unevenly distributed by journal.
**Rejected.** *Full range*: more data, but the pre-1900 material is sparse enough that bins would
be dominated by a few journals. *(This is the one decision here that is a convention of the study
rather than a measurement — it is easily revisited, and the value used at fit time will be
recorded in `topics_report.json`.)*

### D4 — Documents with no content lemma are dropped (forced, not chosen)
**Choice.** Drop hits whose `filtered_lemmata` is empty: within the D2+D3 scope,
379,859 of 380,850 remain (**991 dropped, 0.26%**).
**Rationale.** An empty document cannot contribute to c-TF-IDF; this is a technical necessity,
recorded here only so the loss is visible. It removes hits that *did* match Paul but whose
window contained no noun/verb/adjective/proper noun outside the match — largely citation-style
and running-head material.
**Interaction.** Before D13 was repaired the same rule dropped 58,831 documents (15.45%) and was
silently absorbing the French defect; the two are worth reading together, because a large "drop
empty" count is a symptom to investigate, not a housekeeping statistic.

### D5 — The text passed to BERTopic is the **lemmatised** window, not the natural text
**Choice.** `docs = " ".join(filtered_lemmata)`. Embeddings are computed from the natural
window separately (D6).
**Evidence.** Fitting with the natural KWIC text produced topic terms of pure function words —
`the_of_paul_in`, `de_la_paulus_der` — making topics uninterpretable:

| `docs=` passed to BERTopic | topics | outliers | top terms |
| --- | --- | --- | --- |
| natural KWIC text | 2 | 0% | `the_of_paul_in`, `de_la_paulus_der` |
| **lemmatised window** | 35 | 45% | `jewish_jews_judaism_jew`, `romans_roman_rome_rom`, `fact_law_argument_case` |

**Rationale.** c-TF-IDF yields the *labels*, and the project's vocabulary is defined over lemmas
(POS- and stopword-filtered, see `5_tokens-analysis.ipynb`). Passing lemmatised text also avoids
the English-only stopword lists that let French/German function words through.
*(Measured on 4,226 sampled en+fr+de docs, `min_topic_size=15`, multilingual MiniLM.)*

### D6 — Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
**Choice.** One multilingual model for all three languages, 384 dimensions.
**Evidence.** With the English-only `all-MiniLM-L6-v2` the clustering **split by language rather
than by theme** — topic 0 (n=2064, "the_of_paul_and") was English and topic 1 (n=936,
"de_la_paulus_der") was French/German. With the multilingual model, topics are language-mixed:
the largest topics are 72–92% `eng` with 8–28% `ger`, i.e. thematic.
**Rejected.** *English-only model*: demonstrably splits on language (above). *LaBSE*
(`sentence-transformers/LaBSE`, already in the local cache, 768-d): a defensible second opinion
and the intended robustness check, but not the primary model — it is ~5× slower and its topic
structure on this corpus has not been validated. *Per-language models*: no cross-language topics,
which is precisely the comparison of interest.
**Note.** Embeddings are computed from the **natural** window while labels come from lemmas —
an intentional asymmetry, because lower-cased lemma strings with no punctuation or case embed
poorly, whereas they are exactly what c-TF-IDF should see.

#### D6.1 — Encoder review: E5-large tested and **rejected** (2026-10-16)

**Context.** D6 was decided during the smoke tests, on a corpus in which French was empty (D13).
Since the model is the smallest member of its family (384-d), it was re-opened and a modern
state-of-the-art multilingual encoder was fitted for comparison:
`intfloat/multilingual-e5-large` (1024-d), the standard current choice of the E5 family.

The comparison is confound-free: **identical 100,000 documents** (same seed, same canonical
order), **identical HDBSCAN parameters** (`min_cluster_size=100`, `min_samples=1`,
`epsilon=0.1`), and the same cached pipeline, so only the encoder varies
(`encoder_comparison_fair.json`).

| | MiniLM-L12-v2 | E5-large |
| --- | --- | --- |
| topics | 67 | 47 |
| outlier share | 41.9% | **31.1%** |
| largest topic | 11.1% | 21.7% |
| median topic size | 252 | 197 |
| **language purity (size-weighted)** | **0.716** | **0.871** |
| **topics that are >90% one language** | **9 / 67** | **46 / 47** |

*Language purity* is the size-weighted share of the most frequent language inside a topic. The
corpus baseline is 0.62 (English), so a value far above that means the clustering is separating
languages. **E5-large fails exactly the test that D6 exists to enforce**: 46 of its 47 topics are
effectively monolingual, and its two largest are literally the German one (`paulus, gott,
theologie, apostel, …`, n=21,699) and the French one (`dieu, texte, épître, chrétien, …`,
n=13,347) — together 35% of the sample. The mechanism is visible in its own terms: its largest
topic is 21.7% of the corpus where MiniLM's is 11.1%.

**E5's advantage is real but not the one that matters here.** It does reduce the outlier share
(31.1% vs 41.9%), which is the metric §7 identifies as the binding constraint. That is a genuine
benefit and the reason this was worth testing rather than assuming. It is rejected because the
clusters it produces are not comparable across the three languages, which is the precondition for
every cross-language claim in this study.

**Independent confirmation, and why this was not obvious a priori.** On five hand-written
translation-equivalent triples (en/fr/de, in the register of the corpus) MiniLM scores a **higher
mean cross-lingual cosine than E5, in every configuration tested**:

| encoder | prefix | mean | worst pair |
| --- | --- | --- | --- |
| **MiniLM-L12-v2** | — | **0.936** | 0.846 |
| E5-large | `passage: ` | 0.933 | **0.903** |
| E5-large | none | 0.920 | 0.867 |
| E5-large | `query: ` | 0.917 | 0.860 |

Two things follow, and both mattered. First, **the prefix was not the cause**: E5's `passage: `
setting is its *best*, so the language clustering is intrinsic to the model, not a usage error —
which is why an intermediate version of this review, having measured that E5's *worst* pair was
better (0.903 vs 0.846), briefly recommended switching. The **mean** is what governs global
language mixing; the worst-case floor does not, and E5's better floor bought nothing.
Second, this is not surprising in hindsight: MiniLM is fine-tuned on **paraphrase/translation
pairs**, which explicitly aligns languages, whereas E5 is trained with contrastive objectives that
do not force that alignment. "Larger and newer" does not imply "more cross-lingual".

**Outcome.** The encoder in D6 is **retained, now on post-D13 evidence** rather than on the
smoke test that D6 itself relied on. `--model` makes the comparison reproducible: the E5 weights
are kept in the project cache (`data/large_files/hf_cache`) together with the run's own
`embeddings_…npy` / `reduced5_…npy`, so re-clustering the comparison costs ~6 s and re-fitting it
costs nothing. The model's redundant export formats (ONNX, OpenVINO, the duplicate PyTorch
`.bin`) were pruned, removing 7.3 GB while leaving it loadable — verified by loading and
embedding with it after the prune.

*(Measured cost, for the record: embedding 100,000 documents took E5 **775 s** versus ≈81 s for
MiniLM — roughly 10× slower for a model that is rejected on its merits.)*

### D7 — Embeddings precomputed on GPU and cached to disk
**Choice.** Embed once to `embeddings.npy`; clustering reads the cache.
**Rationale.** Embeddings depend only on (text, model), so re-tuning the clustering — which is
where the iteration happens (D8) — must not re-embed. Measured cost makes this cheap either way:
**≈2,400 docs/s on the RTX A4000 → 347,964 documents in ≈2.5 min** (the 12-layer multilingual
model ≈2× that of the 6-layer English one, so ≈5 min).

### D8 — Algorithm stack: UMAP → HDBSCAN → c-TF-IDF, **unsupervised**
**Choice.** BERTopic defaults with `UMAP(n_components=5, metric="cosine", n_neighbors=15,
min_dist=0.0, random_state=42)` → `HDBSCAN` → c-TF-IDF. No seed topics, no zero-shot labels.
**Rationale.** An unsupervised baseline is the honest first pass: it shows what structure the
corpus actually has before any domain expectations are imposed, and it is the configuration that
can be reported without arguing for the seed list. The author's guided and zero-shot recipes
(`seed_topic_list`, `zeroshot_topic_list`) remain available as a documented comparison.
**Rejected for now.** *Guided / zero-shot*: would fix the answer to the categories the study
already suspects; useful as a *check* on the baseline, not as the baseline itself.
**Measured cost.** UMAP 384→5d: 138 s at 200k (measured: **326 s at 379,859**). HDBSCAN:
29 s at 100k (measured: **23 s at 379,859**). Embedding: 308 s at 379,859. **Whole fit ≈ 11 min.**

#### D8.1 — HDBSCAN parameters, fixed by the pilot sweep (2026-10-16)

A grid of 17 configurations was fitted on a 100k sample, reusing one cached UMAP space
(`pilot_sweep_sample100k.json`; each configuration is HDBSCAN + c-TF-IDF only, ≈6 s), so the
differences below are attributable to the parameters rather than to a re-drawn sample or a
re-fitted projection. `min_cluster_size` is a **proportion** of the corpus, which is what makes
the sample transferable; the chosen value is quoted as a fraction as well as a count.

| min_cluster_size (of 100k) | min_samples | epsilon | method | topics | outliers | largest topic | median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 (0.10%) | **1** | **0.1** | eom | **67** | **41.9%** | 11,123 (11.1%) | 252 |
| 100 (0.10%) | 1 | 0.0 | eom | 89 | 51.5% | 6,130 | 232 |
| 100 (0.10%) | 5 | 0.1 | eom | 60 | 52.8% | 5,850 | 266 |
| 200 (0.20%) | 1 | 0.1 | eom | 39 | 43.0% | 11,316 (11.3%) | 636 |
| 200 (0.20%) | 5 | 0.1 | eom | 39 | 53.7% | 5,850 | 564 |
| 50 (0.05%) | 1 | 0.1 | eom | 119 | 43.3% | 8,615 | 115 |
| **400 (0.40%)** | 1 | 0.1 | eom | **2** | **0.5%** | **99,117 (99.1%)** | 49,772 |
| 200 (0.20%) | 1 | 0.1 | **leaf** | 41 | 47.1% | 11,316 | 568 |

**Findings on the sample.**

1. **`min_cluster_size` is a proportion of the corpus and must not be carried across sample
   sizes.** This is why `--min-topic-size-frac` exists and is the recommended way to set it.
2. **`min_samples=1` beats 5 at every size tested** (at 200/0.1: 43.0% vs 53.7% outliers; at
   100/0.1: 41.9% vs 52.8%) **at equal or higher topic count**. This is the expected direction:
   `min_samples` controls how conservative the density estimate is, and the note in the 2026-10-15
   run ("43% outliers at 1 versus 54% at 5") is reproduced here on a fixed space. **This finding
   survived full-scale re-testing.**
3. **`cluster_selection_epsilon=0.1` lowered the outlier share at a nearly constant topic count**
   (at 200/1: 51.1% → 43.0% for 47 → 39 topics), because it merges clusters whose separation is
   below that distance instead of demoting their boundary documents. `leaf` selection at the same
   size gives finer topics (41) but a *higher* outlier share (47.1%) and no additional distinct
   labels, so `eom` is kept. **This finding did *not* survive full-scale re-testing — see below.**

#### D8.2 — An error caught by sweeping at the production scale (2026-10-16)

The first full-scale fit used the parameters chosen above and produced a **degenerate model that
looked good on the headline metric**: 36 topics, a *low* 32.9% outlier share, and **one topic
holding 199,733 documents — 52.6% of the corpus**. Two independent checks confirmed it was not a
reporting bug (the shares summed to 1.0; BERTopic's own `get_topic_info()` gave the same count).

Sweeping the same parameters at full scale (`pilot_sweep.json`, 16 configurations on the cached
space, ~20 s each) identified the cause immediately:

| epsilon | min_cluster_size | topics | outliers | largest topic |
| --- | --- | --- | --- | --- |
| 0.0 | 200 (0.053%) | 151 | 57.5% | 12,981 (3.4%) |
| 0.0 | 380 (0.100%) | 93 | 58.3% | 14,594 (3.8%) |
| **0.0** | **760 (0.200%)** | **50** | **56.2%** | **21,190 (5.6%)** |
| 0.0 | 1520 (0.400%) | 30 | 56.0% | 21,085 (5.6%) |
| 0.01 / 0.02 / 0.03 | 380, 760 | *identical to 0.0* | *identical* | *identical* |
| 0.05 | 380 | 90 | 57.9% | 14,594 |
| **0.1** | 200 | 58 | 31.4% | **199,733 (52.6%)** |
| **0.1** | 380 | 36 | 32.9% | **199,733 (52.6%)** |
| **0.1** | 760 | 20 | 34.8% | **199,733 (52.6%)** |
| **0.1** | 1520 | 12 | 30.8% | **221,897 (58.4%)** |

**`cluster_selection_epsilon` is an absolute distance, our UMAP space shrinks as *n* grows, and so
the value tuned on the sample does not transfer.** At full scale, `epsilon=0.1` produces the same
~200k-document blob at *every* `min_cluster_size` — it collapses a large part of the space rather
than merging genuinely nearby clusters. Values up to 0.03 are inert; 0.05 barely differs; 0.1
collapses. The transition is sharp, which is why it was invisible on the sample.

**The trap, stated plainly.** The failure *improves* the metric this project had been using to
choose parameters: the outlier share drops from 56% to 33%. A parameter search optimising that
metric alone would have selected the collapsed model confidently. This is the reason the sweep now
reports `largest_topic` and `largest_topic_share` alongside the outlier share, and why
**a single near-global cluster with an implausibly low outlier share is treated as a failure
signature rather than a result.**

**Correction to finding 1.** The sample's 0.40% configuration appeared to collapse into 2 topics,
and that was attributed to `min_cluster_size`. At full scale, 0.40% with `epsilon=0.0` gives a
healthy 30 topics. The attribution was wrong: the sample condition also carried `epsilon=0.1`.
The genuine proportionality statement is the weaker one — `min_cluster_size` must scale with *n*
and should be set as a fraction — not that large values collapse it.

**Unresolved, and not overclaimed.** The earlier degenerate 25k pilot (2 topics, ~0% outliers,
`min_cluster_size=100` = 0.40%) has the **same signature** as the epsilon collapse, but that run
also used `min_samples=10` and predates this sweep, so it is *consistent with* the epsilon
explanation rather than demonstrated by it. It was not re-run to settle the point.

**Choice (revised).** `min_cluster_size = 0.200%` of the documents actually fitted (**760** at
379,859), `min_samples = 1`, **`cluster_selection_epsilon = 0.0`** (BERTopic's default, now
explicit), `cluster_selection_method = "eom"`. 0.20% is taken over 0.10% (50 vs 93 topics) because
the median topic then holds **~1,742** documents, so a 20-year bin still contains ~290 of them —
enough for the per-bin trends D12 asks for. At 0.10% the median is 848 and the trends would be
noisier. 0.40% gives only 30 topics, which is too coarse to describe the field's vocabulary.

**Consequence for D9, stated plainly.** With the parameters above, **56.2% of the corpus falls in
the outlier topic** — the 32.9% reported earlier was an artefact of the collapsed model. So
§7's second limitation is not merely still in force; it is **worse than the first fit suggested**,
and the sweep does not rescue any claim of the form "most of the corpus is about X". *(Recorded
honestly: this is a worse model on the outlier metric than the degenerate one it replaces, and it
is the one that can be reported.)*

### D9 — Outliers are retained and reported, never reassigned
**Choice.** HDBSCAN's `Topic -1` stays as an explicit outlier topic in every artifact and in the
app; `reduce_outliers` is **not** applied.
**Evidence.** The outlier share is high and grows with n: **52% (25k) → 54% (50k) → 58% (100k)**.
**Rationale.** A large unclustered remainder is a *finding* about the corpus — the vocabulary
around Paul is diffuse — and reassigning it would inflate topic sizes and hide that. Reported as
a first-class quantity; a reader may filter it out in the app.
*(This is a decision about reporting, not about clustering quality: if the outlier share proves
unacceptable, the response is to tune D8's parameters, not to post-assign.)*

### D10 — Citation apparatus is kept in the default fit
**Choice.** Hits flagged `citation_apparatus` stay in the default model; excluding them is a
parameter and a filter in the app.
**Rationale.** A large cluster of bibliography-like text (`et_ed_university_philosophy`) is a
**real population** in the corpus, not an artefact, and silently dropping it would misstate how
much of the Paul-adjacent text is apparatus. Keeping it lets the topic be examined and, if a
given analysis requires it, excluded explicitly. The same reasoning applies to the other scan
flags (`excluded_name`, `place_or_building`, …): built once, filtered downstream.

### D11 — Artifacts are precomputed; the app never fits a model
**Choice.** All outputs written under `data/large_files/paul_scan/topics/` (gitignored); the web
app reads them read-only and degrades gracefully if they are absent.
**Rationale.** Fitting belongs to a reproducible script, not to a web request; this also keeps the
app's dependencies independent of the modelling stack. HuggingFace model downloads are likewise
directed into the project (`HF_HOME=data/large_files/hf_cache`) rather than `~/.cache`.
**Artifacts.** `docs.parquet` (per-document topic + metadata + 2-D coords), `topics_info.parquet`,
`topics_over_time.parquet`, `topics_per_class.parquet`, `embeddings.npy`, `topic_model/`,
`topics_report.json`, `pip-freeze-before.txt`, `pip-freeze-after.txt`.

### D12 — Time series are normalised as shares, not counts
**Choice.** In the app, topic frequency over a time bin is shown as a **share of that bin's
documents**; raw counts remain available.
**Rationale.** The corpus grows by ~8× across the century; a rising count would mostly measure
corpus growth. This mirrors the normalisation discipline already applied on the temporal page.

### D13 — RESOLVED 2026-10-15: French documents carried no lemmas (upstream defect)

**Finding.** All **66,059** French hits had an empty `filtered_lemmata` — 100% of them — and were
dropped by D4, so French was **absent from the fitted scope**, contradicting D2.

**Root cause (confirmed).** `nlp_kwic()` in `scripts/4_concordance-tokens-explore.ipynb`
dispatched spaCy on these codes:

```python
if   row["language"] in ["en", "eng"]:  doc = nlp_eng(...)
elif row["language"] in ["fr", "fra"]:  doc = nlp_fra(...)   # <- never matched
elif row["language"] in ["de", "ger"]:  doc = nlp_ger(...)
else: doc = nlp_eng("")                                      # <- 'fre' landed here
```

The JSTOR corpus uses **ISO 639-2/B** codes, so French is `fre`, not `fr`/`fra`. It matched no
branch and fell to `else`, which parses an **empty string**: `kwic_nlp == []`, hence no lemmas.

**Broader than French.** Measured `kwic_nlp` emptiness: `eng` 0%, `en` 0%, `ger` 0%, and **100%
for every other code** — `fre` (66,059), `lat` (3,949), `ita` (2,632), `dut` (2,349), `por`
(1,311), `spa` (1,252), `heb` (269), `pol` (130), plus small tails and 10 nulls. So the defect was
systematic for any language not in the hard-coded list. This makes D2's exclusion of the tail
languages a technical necessity as well as a choice — they were not merely out of scope, they were
unusable.
**Blast radius beyond this model.** Any lemma-based analysis of those languages was impossible:
the app's Temporal page selected to `fre` showed zeros, as did the Concordance page's *lemma*
mode. The KWIC/text modes were unaffected (they read `hits.parquet`).

**Resolution — option A, French only (chosen by V.K.).** Repair French (the only affected
language inside D2); leave the rest, and never re-run the whole corpus.

* `scripts/repair_fre_lemmas.py` re-ran the *same* `nlp_kwic` logic on the **66,059 French rows
  only**, with the correct model (`fr_core_news_md`), reconstructing the identical token
  structure. It works in Arrow and replaces only the target rows by **slicing the original
  column**, so other languages cannot change.
* Measured: **33 s** at 16 processes (2,004 docs/s), 2,980,957 tokens, **0** rows left empty.
  Verified before replacing the original: row count and schema identical; `eng` and `ger` samples
  byte-identical to the pre-repair file. Original preserved as
  `hits_nlp.parquet.bak-20261015-233854`.
* `tokens.parquet` rebuilt. Empty-lemma rows fell from **18.49% to 3.06%**.

**Effect on this model.** The fitted set grew **322,019 → 379,859 documents** (+57,840 French, 93
journals), and D4's drop rate collapsed from 15.45% to **0.26%** — i.e. the earlier large "drop
empty" count was almost entirely this defect. D2 now holds as written.

**Follow-through.** The buggy code map in notebook 4 is corrected so a future full run is right,
and repair/verification cells are appended to that notebook. The remaining 18 empty-language codes
are documented above; repairing them would need their own spaCy models and is out of scope.

## 4. Fixed parameters (for reproduction)

| parameter | value |
| --- | --- |
| unit | ±10-token window around the match (lemmatised) |
| languages | `eng`, `en`, `fre`, `ger` (D13 repaired) |
| year window | 1900–2019 |
| drop | documents with zero content lemmas (991 docs, 0.26%) |
| **fitted documents** | **379,859** |
| BERTopic | 0.17.4, unsupervised, no seed/zero-shot topics |
| embedding model | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-d) — reaffirmed against E5-large in D6.1 |
| UMAP | `n_components=5`, `metric="cosine"`, `n_neighbors=15`, `min_dist=0.0`, `random_state=42` |
| HDBSCAN | `min_cluster_size=760` (**0.200%** of the fit set), `min_samples=1`, `cluster_selection_epsilon=0.0`, `cluster_selection_method="eom"` (D8.1, revised in D8.2) |
| c-TF-IDF | BERTopic default, `n_gram_range=(1,1)` |
| random seeds | `random_state=42` throughout |
| device | CUDA (RTX A4000, 16 GB) for embedding; CPU (32 threads) for UMAP/HDBSCAN |
| fit set | 379,859 documents (en + fr + de) |

### 4.1 The production fit (2026-10-16)

Run: `fit_paul_topics.py --min-topic-size-frac 0.002`. Artifacts under
`data/large_files/paul_scan/topics/` (`report.json` carries the resolved parameters).

| quantity | value |
| --- | --- |
| fitted documents | 379,859 |
| **topics** | **50** (+ the outlier topic) |
| **outlier share** | **56.2%** (213,364 documents) |
| `min_cluster_size` resolved | 760 (= 0.200% of the fit set) |
| largest topic | 21,190 (5.6%) |
| median topic | 1,742 |
| runtime | 936 s (embedding 313 s cached-reused, 5-d UMAP cached-reused, HDBSCAN 21 s, 2-d map 909 s) |

**This is the second full-scale fit.** The first used the sample-tuned `cluster_selection_epsilon=0.1`
and produced a degenerate model (one topic holding 52.6% of the corpus) with a *better-looking*
32.9% outlier share; D8.2 records the diagnosis and the correction. The numbers above are the
reported ones.

**A scale effect, in the direction opposite to what the pilots suggested.** At the same
proportion the outlier share is ~42% on the 100k sample and **56.2% on the full 379,859** — it
*rises* with *n*, because a larger corpus contains more genuinely idiosyncratic windows and HDBSCAN
refuses to place them. The earlier "52% → 54% → 58% growing with n" note in D9 was therefore
closer to the truth than the sample's 42%, and my own first write-up of this section (which claimed
the share *falls* with size and that a pilot is pessimistic) was wrong — it compared a pilot against
a collapsed fit. **Do not infer the outlier share of one scale from another.**

The model's topics are language-mixed, which is the property D6 exists to protect — e.g.
`jewish, jews, juif, judaism, jew, jude, jüdisch, torah, gentile, rabbinic` (n=14,594),
`resurrection, résurrection, auferstehung, body, resurrect, dead` (n=3,066), and
`spirit, body, πνεύμα, flesh, geist, corps, holy, soma` (n=2,678), the last of which incidentally
picks up Greek forms.

**Two stages dominate and were both cached for that reason.** Embedding is 313 s and the 2-d map
909 s; the clustering that D8 actually tunes is 21 s. `embeddings.npy`, `reduced5.npy` and
`xy.npy` are all cached, so a re-cluster costs neither embedding nor either projection — which is
what made a 16-configuration full-scale sweep take under 5 minutes.

## 5. Software environment

Full environment snapshots are kept alongside the artifacts:
`topics/pip-freeze-before.txt` (199 packages, the app/notebook baseline) and
`topics/pip-freeze-after.txt` (230 packages). The difference is **purely additive — 31 packages
added, none removed, no version changed**, and every one of the app's core dependencies is
byte-identical before and after (`numpy 1.26.4`, `pandas 2.2.3`, `umap-learn 0.5.7`,
`scikit-learn 1.6.1`, `scipy 1.13.1`, `numba 0.61.0`, `streamlit 1.44.1`, `click 8.1.8`,
`setuptools 75.8.2`). This is why the topic stack could be added to the app's own virtualenv
rather than a second one.

Versions specific to the model:

`bertopic 0.17.4` · `sentence-transformers 5.2.0` · `transformers 4.57.6` ·
`huggingface-hub 0.36.0` · `tokenizers 0.22.2` · `safetensors 0.7.0` · `hdbscan 0.8.44` ·
`torch 2.10.0+cu128`.

Installed with (order matters — torch first, from the CUDA-12.8 index, so no CUDA-13 stack
is pulled and the app's environment is untouched). This is also encoded in
`requirements-topics.txt`, which re-resolves with nothing to install:

```bash
pip install "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-topics.txt
```

## 6. How to reproduce

```bash
cd /home/jupyter-vojta/notebooks/mops/scripts
export HF_HOME=../data/large_files/hf_cache      # keep model downloads in the project
PY=../mops_venv/bin/python3

$PY fit_paul_topics.py --sample 25000           # pilot: tune min_cluster_size (see §8)
$PY fit_paul_topics.py                          # full fit, ~15-25 min
$PY fit_paul_topics.py --verify                 # artifact self-consistency
$PY check_mops_app.py                           # app + notebook-parity regression
```

All artifacts land in `data/large_files/paul_scan/topics/` (gitignored); nothing large is written
outside `data/large_files/`.

`6_paul-topics.ipynb` holds the interactive inspection (BERTopic's own visualisations, topic
naming, optional merging) and re-exports the artifacts the app reads.

## 7. Limitations — what must not be overclaimed

1. **Windows are not texts.** Documents are ±10-token windows with overlapping spans between
   adjacent hits, so observations are non-independent; cluster sizes are *not* shares of
   discourse.
2. **~0.26% of in-scope hits are unmodelled** (D4) and **56.2% of the fitted documents fall in
   `Topic -1`** (D9, measured in the production fit §4.1). **More than half the corpus is
   unclustered**, so no claim of the form "most of the corpus is about X" is supportable from this
   model, and any statement about "the topics" must be read as a statement about the 43.8% that
   HDBSCAN was willing to cluster. The share cannot be inferred across scales: it was 41.9% on a
   100k sample and 56.2% on the full set (D8.2).
3. **Latin and the smaller tails are excluded** (D2) — and, as D13 showed, they are currently
   unusable anyway, since their lemmas are empty. So the model describes the English-, French- and
   German-language literature of this field only.
4. **Topics are descriptive, not causal or authorial.** They describe recurring lexical-semantic
   configurations near the name, not arguments, intentions, or intellectual traditions.
5. **The model is validated by inspection, not by labels.** There is no gold-standard topic
   annotation for this corpus; quality claims rest on term readability and the checks in §6.
6. **Part of the D5/D6 pilot evidence predates the D13 repair** and was measured on ~4,226
   documents in which French was empty (i.e. effectively en+de). That caveat no longer applies to
   the encoder choice: D6.1 re-tested it on the post-repair corpus at 100k documents, and the
   cross-lingual measurements there (MiniLM 0.936 mean, and its language-mixed topics) are on the
   repaired data. The **D5** comparison (natural text vs lemmatised) has still **not** been
   re-measured post-repair; it is the one claim here resting on pre-repair evidence.

## 8. Open items (not yet decided)

* **`min_cluster_size` (and any `nr_topics` merge)** — **resolved 2026-10-16**: 0.200% of the
  fitted documents (760), with `min_samples=1` and `cluster_selection_epsilon=0.0`; the evidence,
  including the rejected configurations, is in
  [D8.2](#d82--an-error-caught-by-sweeping-at-the-production-scale-2026-10-16) and
  `pilot_sweep.json` (16 full-scale configurations). No `nr_topics` merge is applied.
  `cluster_selection_epsilon` in particular must be swept **at the scale you intend to fit**.
* **Whether the embedding model in D6 should be upgraded** — **resolved 2026-10-16, no**:
  `intfloat/multilingual-e5-large` was fitted on identical documents and parameters and rejected
  for clustering by language rather than theme (46/47 topics >90% one language) — the exact D6
  failure mode. The evidence, including the cross-lingual cosine table and the ruled-out prefix
  confound, is in [D6.1](#d61--encoder-review-e5-large-tested-and-rejected-2026-10-16).
  `--model` reopens the question for any future encoder at the cost of one embedding pass.
* **Topic names** — auto-labels from lemmas are terse; human-readable names will be curated in
  `6_paul-topics.ipynb` (or a `topic_names.json`) without changing the assignment.
* **Whether to also fit LaBSE as a robustness check** (D6) — cheap (~15 min) if wanted.
* **Whether to also fit with citation apparatus excluded** (D10) — cheap; would quantify how much
  of the structure is apparatus.
* **Whether French and German warrant a per-language comparison** of topic structure.
