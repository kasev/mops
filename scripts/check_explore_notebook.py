"""Run every code cell of paul-explore.ipynb headless, in order, and report per cell.

    MPLBACKEND=Agg python3 scripts/check_explore_notebook.py

Run this after changing the notebook or scripts/label_eval.py: it is how §6/§8 are
verified without the user's notebook, which is the only place the remote-VSCode limits
(ipywidgets, cell input()) actually bite. Use Agg — under a GUI backend plt.show() blocks
forever outside a kernel.


Why not `jupyter nbconvert --execute`: the whole-notebook kernel run timed out (the scan
plus §9's file sampling is slow), which hides which cell is at fault. Running cells
ourselves keeps one namespace (so later sections see earlier definitions) and prints a
pass/fail line per cell.
"""
import io
import json
import os
import sys
import time

NB = "/home/jupyter-vojta/notebooks/mops/scripts/paul-explore.ipynb"
os.chdir("/home/jupyter-vojta/notebooks/mops")

cells = json.load(open(NB))["cells"]
# `display()` is injected by IPython, so it is legitimately undefined under plain exec.
def _display(*objs, **kw):
    for o in objs:
        print(o)


ns = {"__name__": "__main__", "__file__": NB, "display": _display, "get_ipython": lambda: None}
fails = []
for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if not src.strip() or all(l.lstrip().startswith("#") for l in src.splitlines()):
        print(f"[skip] cell {i:>2} (commented out)")
        continue
    head = next((l for l in src.splitlines() if l.strip() and not l.lstrip().startswith("#")), "")[:56]
    buf = io.StringIO()
    real = sys.stdout
    sys.stdout = buf
    t0 = time.time()
    try:
        exec(compile(src, f"<cell {i}>", "exec"), ns)
        status, note = "ok", ""
    except Exception as exc:
        status, note = "FAIL", f" {type(exc).__name__}: {str(exc)[:150]}"
        fails.append(i)
    finally:
        sys.stdout = real
    print(f"[{status}] cell {i:>2} {time.time() - t0:5.1f}s  {head}{note}")
    if status == "FAIL":
        print("      " + "\n      ".join(buf.getvalue().splitlines()[-8:]))

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} failed cells: {fails}"))
