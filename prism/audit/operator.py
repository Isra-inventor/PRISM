"""The AI operator of the audit (v3 §8): thin. The model sees the schemas and the finding JSON
only (never matrix rows), answers, and calls tools from a fixed registry. The tools are
deterministic and go through the same engine as the CLI:

    run_audit(factors, datasets, params)  rerun()  show_finding(finding)
    set_override(...)  -> a proposal the user applies (patches are diffs the user applies)
    plot(view_spec)    -> a declarative view the page applies
    request_addition(spec) -> requests/NNN.md; nothing runs

Guards: a parameter in a tool call is used only if the user's message names it with its value;
every number in the reply must appear in the findings (or the message), else it is listed as
unverified; every tool call goes into the ledger (audit/ledger.jsonl, plus the run's own ledger).
"""

from __future__ import annotations

import json
import re

from ..ledger import Ledger
from ..util import read_json, write_bytes
from . import engine, overrides as ov

TOOLS = ("run_audit", "rerun", "show_finding", "set_override", "plot", "request_addition")
SCHEMA_KEYS = ("schema_version", "source_file", "layout", "omics_family", "assays", "feature_annotations",
               "sample_metadata", "design", "processing_history", "excluded_columns")
_NUM = re.compile(r"(?<![A-Za-z_\d.])-?\d[\d,]*(?:\.\d+)?(?:[eE]-?\d+)?")


def ledger(st):
    return Ledger(engine.runs_dir(st) / "ledger.jsonl")


def _schema_view(doc):
    out = {k: doc.get(k) for k in SCHEMA_KEYS if k in doc}
    for a in out.get("assays") or []:
        for b in a.get("value_blocks") or []:
            b.pop("columns", None)        # sample column names only; no values anywhere in a schema
    return out


def context(st, run_id=None):
    """What the model may see: schemas, the run's findings without plot data, parameters."""
    schemas = {}
    for d in st.confirmed():
        p = st.dataset_dir(d["dataset_id"]) / "schema.json"
        if p.exists():
            schemas[d["dataset_id"]] = _schema_view(read_json(p))
    from . import params as params_mod
    out = {"session": st.sid, "schemas": schemas, "audits": [{"id": a, "name": c} for a, c, _ in engine.REGISTRY],
           "param_defaults": params_mod.defaults(), "overrides": ov.load(st), "run_id": None, "findings": []}
    runs = engine.list_runs(st)
    if runs:
        run_id = run_id or runs[-1]
        shown = engine.show(st, run_id, full=True)
        out["run_id"] = run_id
        out["params"] = shown["manifest"]["params"]
        for i, f in zip(shown["manifest"]["findings"], shown["findings"]):
            out["findings"].append(dict({k: v for k, v in f.items() if k != "plot_data"}, file=i["file"]))
        out["declared_vs_observed"] = shown["manifest"].get("declared_vs_observed")
    return out


# ------------------------------------------------------------------ numbers in the reply


def _numbers(obj, out):
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        for m in _NUM.findall(obj):
            try:
                out.append(float(m.replace(",", "")))
            except ValueError:
                pass
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _numbers(k, out)
            _numbers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _numbers(v, out)
    return out


def unverified_numbers(reply, ctx, message):
    """Numbers in the reply that match no number of the findings, schemas or message at the
    precision written (percentages may match shares)."""
    known = _numbers([ctx.get("findings"), ctx.get("schemas"), ctx.get("declared_vs_observed"), message,
                      ctx.get("params")], [])
    bad = []
    for m in _NUM.finditer(reply or ""):
        tok = m.group(0)
        try:
            x = float(tok.replace(",", ""))
        except ValueError:
            continue
        dec = len(tok.split(".")[1]) if "." in tok and "e" not in tok.lower() else 0
        tol = 0.5 * 10 ** -dec + 1e-12
        pct = reply[m.end():m.end() + 1] == "%"
        cands = [x] + ([x / 100] if pct else [])
        ok = False
        for c in cands:
            t = tol if c == x else tol / 100
            if any(abs(k - c) <= t for k in known):
                ok = True
                break
        if not ok:
            bad.append(tok + ("%" if pct else ""))
    return bad


# ------------------------------------------------------------------ tools


class ToolRejected(Exception):
    pass


def _named(message, key, value):
    """The user's message names the parameter and its value."""
    if not re.search(rf"(?<![A-Za-z_]){re.escape(key)}(?![A-Za-z_])", message):
        return False
    vals = [value] if not isinstance(value, list) else value
    return all(re.search(rf"(?<![\d.]){re.escape(str(v))}(?![\d])", message) for v in vals)


def execute(st, call, message, run_id, who="ai"):
    tool = call.get("tool")
    try:
        args = json.loads(call.get("args_json") or "{}")
    except ValueError:
        args = None
    if tool not in TOOLS:
        raise ToolRejected(f"'{tool}' is not an audit tool. Anything else is a request_addition.")
    if not isinstance(args, dict):
        raise ToolRejected("args_json is not a JSON object.")
    if tool == "run_audit":
        params = args.get("params") or {}
        silent = [k for k, v in params.items() if not _named(message, k, v)]
        if silent:
            raise ToolRejected(f"Parameter(s) {', '.join(silent)} were not named with their value in your message: "
                               "the AI never changes a parameter silently.")
        try:
            man = engine.run(st, args.get("factors") or None, args.get("datasets") or None, params, who=who)
        except engine.AuditError as e:
            raise ToolRejected(str(e))
        return {"run_id": man["run_id"], "params_sha256": man["params_sha256"], "n_findings": len(man["findings"])}
    if tool == "rerun":
        runs = engine.list_runs(st)
        if not runs:
            raise ToolRejected("No run to re-run.")
        man = read_json(engine.runs_dir(st) / (run_id or runs[-1]) / "manifest.json")
        params = {k: v["value"] for k, v in (man["scope"].get("params_changed") or {}).items() if v["source"] == "run"}
        try:
            new = engine.run(st, man["scope"]["audits"], man["scope"]["datasets"], params, who=who)
        except engine.AuditError as e:
            raise ToolRejected(str(e))
        return {"run_id": new["run_id"], "rerun_of": man["run_id"], "params_sha256": new["params_sha256"]}
    if tool == "show_finding":
        name = args.get("finding") or ""
        try:
            f = engine.finding(st, run_id, name)
        except engine.AuditError as e:
            raise ToolRejected(str(e))
        return {"finding": name, "audit_id": f["audit_id"], "dataset": f["dataset"], "assay": f["assay"],
                "status": f["status"], "indicators": [i["text"] for i in f["indicators"]]}
    if tool == "set_override":
        entry = {k: args.get(k) for k in ("kind", "sample", "role", "column", "dataset", "key", "value") if args.get(k) is not None}
        entry["reason"] = args.get("reason") or "proposed by the AI operator"
        try:
            ov.validate(entry)
        except ov.OverrideError as e:
            raise ToolRejected(str(e))
        return {"proposal": entry, "applied": False,
                "note": "Not applied: apply it to use it in the next run (overrides are inputs, never edits to results)."}
    if tool == "plot":
        name = args.get("finding") or ""
        try:
            f = engine.finding(st, run_id, name)
        except engine.AuditError as e:
            raise ToolRejected(str(e))
        pca = (f.get("plot_data") or {}).get("pca")
        if not pca:
            raise ToolRejected(f"{name} has no PCA plot.")
        names = set((pca["samples"][0].get("values") or {}).keys()) if pca["samples"] else set()
        color = args.get("color_by")
        if color and color not in names:
            match = next((n for n in names if n.lower() == str(color).lower()), None)
            if match is None:
                raise ToolRejected(f"'{color}' is not a variable of this plot ({', '.join(sorted(names)) or 'none'}).")
            color = match
        k = len(pca["variance_explained"])
        pcs = args.get("pcs") or [1, 2]
        if len(pcs) != 2 or not all(isinstance(x, int) and 1 <= x <= k for x in pcs):
            raise ToolRejected(f"pcs must be two of 1..{k}.")
        return {"view": {"finding": name, "color_by": color, "pcs": pcs}}
    # request_addition
    rdir = st.dir / "requests"
    rdir.mkdir(parents=True, exist_ok=True)
    n = len(list(rdir.glob("*.md"))) + 1
    path = rdir / f"{n:03d}.md"
    body = [f"# Request {n:03d}: {args.get('title') or 'addition to the audit'}", "",
            "Written by the AI operator from a chat message. Nothing was run.", "", f"> {message}", ""]
    for key in ("inputs", "outputs", "algorithm", "tests", "citations"):
        body += [f"## {key.capitalize()}", "", str(args.get(key) or "(not given)"), ""]
    write_bytes(path, "\n".join(body).encode("utf-8"))
    return {"request": str(path.relative_to(st.dir))}


def chat(st, message, run_id=None, mock_fn=None):
    from backend import ai
    ctx = context(st, run_id)
    run_id = ctx["run_id"]
    led = ledger(st)
    resp, meta = ai.audit_chat({"message": message, "context": ctx}, st.sid,
                               log=lambda ev, payload: None, mock_fn=mock_fn)
    if resp is None:
        led.record("ai.chat", {"message": message, "run_id": run_id}, "ai", {"error": meta.get("error")})
        return {"reply": None, "error": meta.get("error") or "The AI is not available.", "tool_results": []}
    results = []
    for c in resp.tool_calls:
        call = {"tool": c.tool, "args_json": c.args_json}
        try:
            out = execute(st, call, message, run_id)
            res = {"tool": c.tool, "ok": True, "result": out}
        except ToolRejected as e:
            res = {"tool": c.tool, "ok": False, "error": str(e)}
        results.append(res)
        led.record(f"ai.{c.tool}", {"args_json": c.args_json, "run_id": run_id}, "ai", res)
        if res["ok"] and run_id:
            Ledger(engine.runs_dir(st) / run_id / "ledger.jsonl").record(f"ai.{c.tool}", {"args_json": c.args_json}, "ai", res)
    bad = unverified_numbers(resp.reply, ctx, message)
    led.record("ai.chat", {"message": message, "run_id": run_id}, "ai",
               {"reply": resp.reply, "unverified_numbers": bad})
    st.log("audit_ai_chat", {"run_id": run_id, "tools": [r["tool"] for r in results], "unverified_numbers": bad})
    return {"reply": resp.reply, "tool_results": results, "unverified_numbers": bad, "run_id": run_id,
            "model": meta.get("model"), "provider": meta.get("provider")}
