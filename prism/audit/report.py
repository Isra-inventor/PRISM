"""The audit report (v3 §7): the data bundle the report page renders, and a single self-contained
HTML file (CSS, renderer and data inlined; no network) for sharing.

The live page (frontend/audit.js) and the static export use the same renderer,
frontend/audit_report.js, so they always show the same thing.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..util import read_json, write_bytes, write_json
from . import engine, params as params_mod

FRONTEND = Path(__file__).resolve().parent.parent.parent / "frontend"


def bundle(st, run_id=None):
    shown = engine.show(st, run_id, full=True)
    man = shown["manifest"]
    runs = engine.list_runs(st)
    i = runs.index(man["run_id"])
    datasets = []
    for d in st.datasets:
        if d["dataset_id"] in man["scope"]["datasets"]:
            schema = st.dataset_dir(d["dataset_id"]) / "schema.json"
            fam = (read_json(schema).get("omics_family") or {}).get("value") if schema.exists() else None
            datasets.append({"dataset_id": d["dataset_id"], "name": d["name"], "origin": d["origin"],
                             "omics_family": fam, "schema_sha256": d.get("schema_sha256")})
    return {"session": {"session_id": st.sid, "name": st.data.get("name")}, "datasets": datasets,
            "manifest": man, "findings": shown["findings"], "runs": runs,
            "previous_run": runs[i - 1] if i > 0 else None,
            "params_meta": {"defaults": params_mod.defaults(), "heuristic": list(params_mod.HEURISTIC)}}


def _script_json(obj):
    """JSON safe inside a <script> element."""
    return json.dumps(obj, ensure_ascii=False).replace("<", "\\u003c")


def html(st, run_id=None):
    b = bundle(st, run_id)
    css = (FRONTEND / "audit_report.css").read_text(encoding="utf-8")
    js = (FRONTEND / "audit_report.js").read_text(encoding="utf-8")
    title = f"PRISM Tier 1 audit · {b['session']['name']} · {b['manifest']['run_id']}"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>
{css}
</style>
</head>
<body class="ar-standalone">
<main id="report"></main>
<script id="prism-report-data" type="application/json">{_script_json(b)}</script>
<script>
{js}
</script>
<script>
  window.PRISM_REPORT.render(document.getElementById("report"),
    JSON.parse(document.getElementById("prism-report-data").textContent), {{ live: false }});
</script>
</body>
</html>
"""


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write(st, run_id=None, fmt="html", out=None):
    """Write the report into the run folder (report.html / report.json) or to `out`."""
    b = bundle(st, run_id)
    rdir = engine.runs_dir(st) / b["manifest"]["run_id"]
    if fmt == "json":
        path = Path(out) if out else rdir / "report.json"
        write_json(path, b)
    else:
        path = Path(out) if out else rdir / "report.html"
        write_bytes(path, html(st, b["manifest"]["run_id"]).encode("utf-8"))
    return path
