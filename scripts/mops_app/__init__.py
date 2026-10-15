"""MOPS streamlit app — shared code behind the `/mops/` web service.

The package is deliberately split so the analysis layer is importable without
streamlit:

* `data`   — loaders, filters and the temporal aggregation (pure pandas).
* `plots`  — figure builders (matplotlib and plotly).
* `pages`  — one module per streamlit page; this is the only streamlit code.

`scripts/mops_jstor.py` is the runnable entry point (see the systemd unit
`mops.service`) and holds the navigation between pages.
"""
