"""The Tier 1 audit engine (v3 §6): runs the audits over every confirmed dataset of a session.

    sessions/<sid>/audit/<run_id>/
      manifest.json                       version, git hash, parameters, seed, input hashes,
                                          overrides hash, timings, scope, findings index
      findings/<audit_id>__<D>_<unit>.json  one finding per audit per dataset and audit unit
      ledger.jsonl                        every tool call: tool, params, who, timestamp, output hash

Deterministic: the same inputs, parameters and seed give byte-identical finding files. Reads the
output folders, the merge decisions and overrides.json only; writes nothing outside audit/.
"""

from __future__ import annotations

import time

from .. import manifest as manifest_mod
from ..ledger import Ledger
from ..session import merge
from ..util import canonical_json, now_iso, read_json, sha256_bytes, sha256_file, write_json
from . import (a02_scale, a03_distribution, a04_missingness, a05_dimensionality, a07_batch, a08_repeated,
               a11_outliers, context, overrides as ov, params as params_mod)

# id, code, module (None: built in a later stage). Order of execution: A8 before A5 (ICC).
REGISTRY = [
    ("A1", "integrity", None),
    ("A2", "scale", a02_scale),
    ("A3", "distribution", a03_distribution),
    ("A4", "missingness", a04_missingness),
    ("A8", "repeated_measures", a08_repeated),
    ("A5", "dimensionality", a05_dimensionality),
    ("A7", "batch", a07_batch),
    ("A6", "noise_qc", None),
    ("A9", "source_heterogeneity", None),
    ("A11", "outliers", a11_outliers),
    ("A10", "multiomics_overlap", None),
]


class AuditError(Exception):
    pass


def available():
    return [(a, c) for a, c, m in REGISTRY if m is not None]


def resolve_factors(factors):
    if not factors:
        return [a for a, _, m in REGISTRY if m is not None]
    out = []
    for f in factors:
        key = str(f).strip().lower()
        hit = next(((a, m) for a, c, m in REGISTRY if key in (a.lower(), c)), None)
        if hit is None:
            raise AuditError(f"Unknown audit '{f}'. Known: " + ", ".join(f"{a} ({c})" for a, c, _ in REGISTRY))
        if hit[1] is None:
            raise AuditError(f"Audit {hit[0]} is not built yet.")
        if hit[0] not in out:
            out.append(hit[0])
    return [a for a, _, _ in REGISTRY if a in out]


def runs_dir(st):
    return st.dir / "audit"


def _inputs(st, ctxs):
    """sha256 of every file the audit reads."""
    out = {}
    for c in ctxs:
        folder = st.output_dir(c.dataset_id)
        for p in sorted(folder.iterdir()):
            if p.is_file():
                out[f"datasets/{c.dataset_id}/output/{p.name}"] = sha256_file(p)
    for name in ("session_sample_table.csv", "session_schema.json"):
        p = st.dir / name
        if p.exists():
            out[name] = sha256_file(p)
    return out


def run(st, factors=None, datasets=None, params=None, who="user"):
    t0 = time.time()
    audit_ids = resolve_factors(factors)
    items = ov.load(st)
    try:
        P, changed = params_mod.resolve(params or {}, items)
    except params_mod.ParamError as e:
        raise AuditError(str(e))
    try:
        doc = merge.write(st)
    except merge.MergeError as e:
        raise AuditError(str(e))
    if not st.confirmed():
        raise AuditError("No confirmed dataset: confirm a dataset first.")
    if not doc["ready_for_audit"]:
        raise AuditError("The session is not ready for audit: " + " ".join(doc["blocking"]))
    known = {d["dataset_id"] for d in st.confirmed()}
    for d in datasets or []:
        if d not in known:
            raise AuditError(f"Unknown or unconfirmed dataset '{d}'.")
    seq = len(list_runs(st, complete=False)) + 1
    while True:   # r0001_20261007T180324Z: sorts in run order
        run_id = f"r{seq:04d}_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        rdir = runs_dir(st) / run_id
        try:
            (rdir / "findings").mkdir(parents=True)
            break
        except FileExistsError:
            seq += 1
    ledger = Ledger(rdir / "ledger.jsonl")
    timings = {}
    t = time.time()
    merged, ctxs = context.build(st, items, P, datasets)
    timings["load"] = round(time.time() - t, 3)
    index = []
    results = {}
    mods = {a: m for a, _, m in REGISTRY}
    for c in ctxs:
        for u in c.units:
            for aid in audit_ids:
                t = time.time()
                f = mods[aid].run(u, c, P, results).to_json()
                name = f"{aid}__{c.dataset_id}_{u.unit_id}.json"
                write_json(rdir / "findings" / name, f)
                dt = round(time.time() - t, 3)
                timings[f"{aid}:{c.dataset_id}:{u.unit_id}"] = dt
                ledger.record(f"audit.{aid}", {"dataset": c.dataset_id, "unit": u.unit_id}, who, f)
                index.append({"file": name, "audit_id": aid, "audit": f["audit"], "dataset": c.dataset_id,
                              "assay": u.unit_id, "assay_label": u.label, "status": f["status"],
                              "indicators": [i["code"] for i in f["indicators"]], "needs": f["needs"]})
    timings["total"] = round(time.time() - t0, 3)
    scope = {"audits": audit_ids, "datasets": [c.dataset_id for c in ctxs],
             "units": [{"dataset": c.dataset_id, "unit": u.unit_id, "label": u.label, "n_samples": u.n,
                        "n_features": u.p, "excluded_by_override": u.excluded_by_override,
                        "non_study_samples": u.non_study_samples, "scale": u.scale} for c in ctxs for u in c.units],
             "overrides_applied": items, "params_changed": changed}
    man = manifest_mod.build(run_id, st.sid, P, _inputs(st, ctxs), ov.sha(st), timings, scope)
    man["who"] = who
    man["findings"] = index
    man["findings_sha256"] = sha256_bytes(canonical_json(
        [read_json(rdir / "findings" / i["file"]) for i in index]).encode("utf-8"))
    write_json(rdir / "manifest.json", man)
    ledger.record("audit.run", {"factors": audit_ids, "datasets": datasets, "params": params or {}}, who,
                  {"findings_sha256": man["findings_sha256"]})
    st.log("audit_run", {"run_id": run_id, "who": who, "audits": audit_ids, "params_sha256": man["params_sha256"]})
    return man


def list_runs(st, complete=True):
    """Run ids in run order (complete: only runs that wrote their manifest)."""
    d = runs_dir(st)
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and p.name.startswith("r")
                  and (not complete or (p / "manifest.json").exists()))


def _run_dir(st, run_id=None):
    runs = list_runs(st)
    if not runs:
        raise AuditError("No audit run yet.")
    run_id = run_id or runs[-1]
    if run_id not in runs:
        raise AuditError(f"Unknown run '{run_id}'.")
    return runs_dir(st) / run_id


def show(st, run_id=None, full=False):
    rdir = _run_dir(st, run_id)
    man = read_json(rdir / "manifest.json")
    out = {"run_id": man["run_id"], "created_at": man["created_at"], "params_sha256": man["params_sha256"],
           "seed": man["seed"], "scope": man["scope"], "timings_s": man["timings_s"], "findings": man["findings"]}
    if full:
        out["manifest"] = man
        out["findings"] = [read_json(rdir / "findings" / i["file"]) for i in man["findings"]]
    return out


def finding(st, run_id, name):
    rdir = _run_dir(st, run_id)
    p = rdir / "findings" / name
    if not p.exists() or p.parent != rdir / "findings":
        raise AuditError(f"Unknown finding '{name}'.")
    return read_json(p)


def _leaves(obj, path="", out=None):
    out = {} if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            _leaves(v, f"{path}.{k}" if path else k, out)
    elif isinstance(obj, list):
        if len(obj) <= 50:
            for i, v in enumerate(obj):
                _leaves(v, f"{path}[{i}]", out)
        else:
            out[path] = f"<list of {len(obj)}>"
    else:
        out[path] = obj
    return out


def compare(st, run_a, run_b):
    a, b = _run_dir(st, run_a), _run_dir(st, run_b)
    ma, mb = read_json(a / "manifest.json"), read_json(b / "manifest.json")
    params = {k: {"a": ma["params"].get(k), "b": mb["params"].get(k)}
              for k in sorted(set(ma["params"]) | set(mb["params"])) if ma["params"].get(k) != mb["params"].get(k)}
    inputs = {k: {"a": ma["inputs"].get(k), "b": mb["inputs"].get(k)}
              for k in sorted(set(ma["inputs"]) | set(mb["inputs"])) if ma["inputs"].get(k) != mb["inputs"].get(k)}
    fa = {i["file"]: i for i in ma["findings"]}
    fb = {i["file"]: i for i in mb["findings"]}
    diffs = []
    for name in sorted(set(fa) | set(fb)):
        if name not in fa or name not in fb:
            diffs.append({"finding": name, "only_in": "a" if name in fa else "b"})
            continue
        x, y = read_json(a / "findings" / name), read_json(b / "findings" / name)
        if x == y:
            continue
        lx, ly = _leaves(x.get("measures", {})), _leaves(y.get("measures", {}))
        changed = [{"path": k, "a": lx.get(k), "b": ly.get(k)} for k in sorted(set(lx) | set(ly)) if lx.get(k) != ly.get(k)]
        ia, ib = {i["code"] for i in x["indicators"]}, {i["code"] for i in y["indicators"]}
        diffs.append({"finding": name, "status": [x["status"], y["status"]] if x["status"] != y["status"] else None,
                      "indicators_added": sorted(ib - ia), "indicators_removed": sorted(ia - ib),
                      "measures_changed": changed[:200], "n_measures_changed": len(changed)})
    return {"a": ma["run_id"], "b": mb["run_id"], "params_changed": params, "inputs_changed": inputs,
            "overrides_changed": ma.get("overrides_sha256") != mb.get("overrides_sha256"),
            "findings_identical": ma["findings_sha256"] == mb["findings_sha256"], "findings": diffs,
            "compared_at": now_iso()}
