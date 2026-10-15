# `mops_app` — the streamlit app behind `/mops/`

Companion web app to the 2022–2026 study of Pauline scholarship. It reads the
Paul concordance built by `scripts/paul_concordance.py` and lets you explore it
two ways: frequencies over time, and the quoted lines themselves.

## Layout

```
mops_jstor.py            entry point (streamlit run target, wired into mops.service)
prepare_jstor_app_data.py  one-off: hits_nlp.parquet -> tokens.parquet
mops_app/
  data.py                loaders, filters, temporal aggregation — NO streamlit
  cache.py               st.cache_data wrappers around data.py
  plots.py               matplotlib port of the notebook's plot_timebins + plotly twin
  pages/
    temporal.py          "Temporal frequencies" page
    concordance.py       "Concordance browser" page
```

`data.py` and `plots.py` never import streamlit, so a notebook can reuse the exact
functions the app renders (`from mops_app import data, plots`).

## Run locally

```bash
cd /home/jupyter-vojta/notebooks/mops/scripts
../mops_venv/bin/python -m streamlit run mops_jstor.py \
    --server.address localhost --server.port 8061 --server.baseUrlPath mops
```

Then open <http://localhost:8061/mops/>.

## Data

All three files live in `data/large_files/paul_scan/` (gitignored):

| file | built by | contents |
| --- | --- | --- |
| `hits.parquet` | `3_paul-hits-explore.ipynb` | flat scan output: metadata, char offsets, KWIC, sentence |
| `hits_nlp.parquet` | `4_concordance-tokens-explore.ipynb` | the above + `kwic_nlp` (spaCy tokens) + journal annotations |
| `tokens.parquet` | `prepare_jstor_app_data.py` | **what the app loads** — hit metadata + `filtered_lemmata` |

Rebuild the app table (≈20 s, 18 MB):

```bash
../mops_venv/bin/python prepare_jstor_app_data.py            # write
../mops_venv/bin/python prepare_jstor_app_data.py --verify   # audit an existing file
```

`tokens.parquet` and `hits.parquet` hold the same 426,917 hits, joined on
`(item_id, char_start)`; the build asserts that key is unique.

## Deployment

`mops.service` runs `scripts/mops_jstor.py` on port 8061 **with `--server.baseUrlPath mops`**,
and nginx proxies the prefix through unchanged:

```nginx
location /mops/ { proxy_pass http://127.0.0.1:8061/mops/; ... }   # no `rewrite ... break;`

# streamlit 1.44 builds download-button URLs as bare /media/<id> while only serving them
# under the base path -> this route is required for "Download as CSV" to work
location /media/ { proxy_pass http://127.0.0.1:8061/mops/media/; ... }
```

Two things to know if you touch this:

* **Do not re-add the rewrite.** The six sibling streamlit apps in `ccs-lab` strip the prefix and
  work, but that only covers their *bootstrap* assets (relative `./static/…`). With a two-segment
  path (`/mops/_temporal`), anything the frontend resolves against `location.pathname` breaks —
  reloading a sub-page requested `/mops/_temporal/_stcore/health` and 404'd.
* `--server.baseUrlPath` and the nginx mapping must be changed together; changing one alone gives
  404s on either the assets or the app routes.

Verify with a **sub-page reload**, not just the index:

```bash
for p in /mops/ /mops/_temporal /mops/_concordance /mops/_stcore/health; do
  printf '%-26s %s\n' "$p" "$(curl -sk -o /dev/null -w '%{http_code}' -H 'Host: ccs-lab.zcu.cz' https://127.0.0.1$p)"
done
```

## Faithfulness to the notebook

The temporal aggregation is a port of the "Temporal lineplots" section of
`5_tokens-analysis.ipynb`, and it is checked against that notebook's recorded
output: `jewish` / `judaism` / `jew` over bidecades, English, no
discipline/confession filter must reproduce

```
                jew    jewish   judaism
 1900–1919  0.074025  0.309423  0.171737
 1920–1939  0.072320  0.310975  0.135599
 1940–1959  0.069253  0.203965  0.092022
 1960–1979  0.042885  0.275765  0.174253
 1980–1999  0.074194  0.412247  0.218663
 2000–2019  0.093377  0.417121  0.176689
```

For that, set the metric to `per_20` (the notebook formula `count / concordances * 20`).

### Defects fixed while porting

The notebook's `from_words_to_freqs` and `hits_filtering` never ran as written:

* `from_words_to_freqs` had no `return` (it assigned `timebins_freq_data` locally) and
  contained a syntax error (`["biblical/early Christian]`, `confessions=é`);
* `hits_filtering` was defined as `hits_filtering(hits_subset)` but called with four
  arguments, and it read `disciplines`/`confessions` as free variables;
* the year window (`1900–2020`) and the `* 20` scale were hard-coded.

Both functions are replaced by `data.filter_tokens` + `data.words_over_time`, which take
the same inputs and produce identical numbers (verified above).

## Add a page

1. Create `mops_app/pages/<name>.py` exposing `render()`.
2. Register it in `mops_jstor.py`: `st.Page(<name>.render, title=..., icon=...)`.

Keep page modules to UI only — put anything testable in `data.py`.

## Deliberately not here yet

* word clouds per bidecade (`5_tokens-analysis.ipynb` has the code);
* journal projection / t-SNE of journal vocabularies;
* any re-run of the corpus pipeline.
