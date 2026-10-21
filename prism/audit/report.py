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
            "params_meta": {"defaults": params_mod.defaults(), "heuristic": list(params_mod.HEURISTIC)},
            "v4": v4_block(st)}


def _trim_plan(plan):
    """The plan for the report: feature-id lists become counts (the full plan stays in plan.json)."""
    import copy
    p = copy.deepcopy(plan)
    for e in p.get("entries", []):
        for c in e.get("candidates", []):
            prm = c.get("params") or {}
            for k in ("ids", "batch", "run_order", "qc_samples"):
                v = prm.get(k)
                if isinstance(v, (list, dict)) and len(v) > 12:
                    prm[k] = f"<{len(v)} values>"
        f = e.get("filtering")
        if f:
            for r in f.get("rows", []):
                r["n_removed"] = len(r.pop("remove_ids", []))
    for h in p.get("hand_off", []):
        ic = h.get("imputation_candidates") or {}
        ic.pop("profile", None)
    return p


def v4_block(st):
    """Registry grid, plan, decisions, states and checkpoint comparisons for the v4 report sections."""
    from .. import registry
    from ..executor import state as state_mod
    out = {"grid": registry.grid(), "steps": [{k: s[k] for k in ("id", "name", "kind", "scope", "needs",
                                                                  "evidence_level", "implemented")}
                                               for s in registry.steps()]}
    p = st.dir / "plan" / "plan.json"
    if p.exists():
        out["plan"] = _trim_plan(read_json(p))
    try:
        from ..plan import engine as pe
        out["decisions"] = pe.decisions(st)
    except Exception:
        out["decisions"] = []
    try:
        states = state_mod.list_states(st)
    except Exception:
        states = []
    out["states"] = states
    comps = []
    for d in st.confirmed():
        did = d["dataset_id"]
        mine = [s for s in states if s["dataset"] == did]
        best = {}
        for s in mine:               # the latest state reached at each checkpoint
            best[s["checkpoint"]] = s
        root = state_mod.root_id(st, did)
        for cp in ("C1", "C2"):
            if cp in best:
                try:
                    c = engine.compare_states(st, root, best[cp]["state_id"])
                    comps.append({"dataset": did, "from": "C0", "to": cp, "state_from": root,
                                  "state_to": best[cp]["state_id"], "counts": c["state_diff"]["counts"],
                                  "blocks": c["state_diff"]["blocks"], "table": c.get("table"),
                                  "missing_runs": c.get("missing_runs")})
                except Exception as e:
                    comps.append({"dataset": did, "from": "C0", "to": cp, "error": str(e)[:200]})
    out["checkpoints"] = comps
    return out


def _script_json(obj):
    """JSON safe inside a <script> element."""
    return json.dumps(obj, ensure_ascii=False).replace("<", "\\u003c")


def _page(title, data, call):
    """One self-contained page: CSS, the shared renderer and the data inlined; nothing fetched.
    The theme follows the reader's choice (stored per browser) or the OS setting."""
    css = (FRONTEND / "audit_report.css").read_text(encoding="utf-8")
    js = (FRONTEND / "audit_report.js").read_text(encoding="utf-8")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>{_esc(title)}</title>
<script>try{{var t=localStorage.getItem("prism.theme");if(t)document.documentElement.dataset.theme=t}}catch(e){{}}</script>
<style>
{css}
</style>
</head>
<body class="ar-standalone">
<main id="report"></main>
<script id="prism-report-data" type="application/json">{_script_json(data)}</script>
<script>
{js}
</script>
<script>
  window.PRISM_REPORT.{call}(document.getElementById("report"),
    JSON.parse(document.getElementById("prism-report-data").textContent), {{ live: false }});
</script>
</body>
</html>
"""


def html(st, run_id=None):
    b = bundle(st, run_id)
    return _page(f"PRISM audit · {b['session']['name']} · {b['manifest']['run_id']}", b, "render")


# ---------------------------------------------------------------- the final session report


def _side(side, unit=None):
    if not side or side.get("source") in (None, "none"):
        return None
    if side.get("source") == "metadata_column":
        return f"column '{side.get('column')}'"
    d = side.get("derivation") or {}
    return "from the sample names" + (f" (split at the {d.get('occurrence')} '{d.get('delimiter')}')" if d else "")


def dataset_summary(st, d):
    """What the final report says about one dataset, from its confirmed schema.json."""
    p = st.dataset_dir(d["dataset_id"]) / "schema.json"
    out = {"dataset_id": d["dataset_id"], "name": d["name"], "status": d["status"].replace("_", " "),
           "origin": {"wizard": "Step 0 wizard", "schema_import": "imported schema", "output_folder": "output folder"}.get(d.get("origin"), d.get("origin"))}
    if not p.exists():
        return out
    s = read_json(p)
    assays = []
    for a in s.get("assays", []):
        fi = a.get("feature_identity") or {}
        assays.append({"label": a.get("assay_label"), "omics_type": a.get("omics_type"), "n_features": a.get("n_features"),
                       "n_samples": a.get("n_samples"), "n_blocks": len(a.get("value_blocks") or []),
                       "feature_id": ", ".join(fi.get("columns") or []) or fi.get("from")})
    des = s.get("design") or {}
    t = des.get("time") or {}
    unit = (t.get("unit") or {}).get("value") if isinstance(t.get("unit"), dict) else t.get("unit")
    hist = {}
    for k in ("normalized", "log_transformed", "imputed", "batch_corrected"):
        v = (s.get("processing_history") or {}).get(k)
        hist[k] = v.get("answer") if isinstance(v, dict) else v
    types = [a["omics_type"] for a in assays if a.get("omics_type") not in (None, "", "unknown")]
    out.update(source_file=s.get("source_file"),
               omics_family=", ".join(dict.fromkeys(types)) or (s.get("omics_family") or {}).get("value"),
               layout=(s.get("layout") or {}).get("value"), assays=assays,
               feature_id=next((a["feature_id"] for a in assays if a.get("feature_id")), None),
               n_features=sum(a.get("n_features") or 0 for a in assays),
               n_samples=max([a.get("n_samples") or 0 for a in assays] + [0]),
               n_metadata_columns=len([c for c in s.get("sample_metadata", []) if c.get("keep", True)]),
               design={"subject": _side(des.get("subject")), "time": _side(t), "time_unit": unit,
                       "n_subjects": (des.get("subject") or {}).get("n_subjects"), "label": des.get("label")},
               processing_history=hist)
    return out


def final_bundle(st, run_id=None):
    from ..session import merge
    from ..util import now_iso
    from .. import __version__
    merged = None
    if st.confirmed():
        try:
            merged = merge.report(st)
        except merge.MergeError:
            merged = None
    audit = bundle(st, run_id) if engine.list_runs(st) else None
    return {"session": {"session_id": st.sid, "name": st.data.get("name")}, "generated_at": now_iso(),
            "prism_version": __version__, "datasets": [dataset_summary(st, d) for d in st.datasets],
            "merge": merged, "audit": audit, "v4": audit["v4"] if audit else v4_block(st)}


def final_html(st, run_id=None):
    fb = final_bundle(st, run_id)
    return _page(f"PRISM report · {fb['session']['name']}", fb, "renderFinal")


def write_final(st, run_id=None, fmt="html", out=None):
    fb = final_bundle(st, run_id)
    path = Path(out) if out else st.dir / ("final_report.json" if fmt == "json" else "final_report.html")
    if fmt == "json":
        write_json(path, fb)
    else:
        write_bytes(path, _page(f"PRISM report · {fb['session']['name']}", fb, "renderFinal").encode("utf-8"))
    return path


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
