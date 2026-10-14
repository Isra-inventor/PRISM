# PRISM roadmap

## Now
Keep building PRISM as a website (Step 0 wizard, merge, Tier 1 audit, reports). Python 3.7 stays
supported for now; revisiting the minimum version (3.9+) is a later decision.

## Later: PRISM as a Python library for analysts (notebooks)
The website will become a thin layer over a `prism` library that analysts can use directly:

```python
import prism
ds = prism.read("SomaExpr_common.csv")        # parse + facts, no AI
ds.propose(ai=False)                          # or ai="gemini"
ds.answer("time_unit", "weeks"); ds.confirm() # writes schema.json + output folder
study = prism.Study("NHP SIV"); study.add(ds)
run = study.audit(); study.report("report.html")
```

Planned steps:
1. Move the Step 0 engine (`backend/workflow.py` and friends) into the library (e.g. `prism/step0/`),
   so `prism` no longer imports `backend`.
2. A small public API (above): propose, answer questions in code, confirm; Study, merge, audit, report;
   notebook display of findings and reports.
3. The web server (`prism.web`, `prism serve`) uses only that public API.
4. Packaging: `pyproject.toml`; core = numpy + pydantic; extras `prism[ai]`, `prism[web]`, optional
   pandas; a PyPI name that is free (e.g. `prism-omics`); docs.

## Rule until then
New deterministic logic goes into `prism/` (importable, no web or HTTP types), and `backend/` stays a
web layer that calls it. This keeps the library step small.
