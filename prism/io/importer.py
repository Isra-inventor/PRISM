"""Schema import: upload data + an existing schema.json and skip the wizard (v3 §4).

    validate (JSON Schema 0.4) -> bind to the file (exact | template | seeded_wizard)
    -> rebuild the Step 0 draft -> recompute everything value-dependent and diff it against
    what the schema stores -> re-run the Step 0 structural checks -> import_report.json
    -> the user Accepts (finalize through the wizard's own code path), Opens it in the wizard,
       or Rejects (nothing is left behind).

Nothing from a schema is executed; its file / path fields are ignored and outputs regenerated.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

from backend import consistency, outputs, workflow as wf
from backend.parsing import InputError, column_labels, parse_bytes, sanitize_filename

from .. import store
from ..util import now_iso, schema_sha256, sha256_bytes, write_json
from .rebuild import rebuild
from .schema_model import SchemaRejected, validate

MAX_DIFFS = 300
# recomputed, never compared as 'differences': the import's own record and run-specific fields
SKIP_DIFF = {"$.ai", "$.log_ref", "$.import"}


class ImportRejected(Exception):
    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors or []


# ---------------------------------------------------------------- recompute vs stored

def _num_equal(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if a is None or b is None:
        return a is b
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b or abs(a - b) <= 1e-9 * max(abs(a), abs(b))


def deep_diff(stored, recomputed, path="$", out=None, skip=()):
    """[{path, stored, recomputed}]: counts exact, floats within a relative 1e-9."""
    out = [] if out is None else out
    if path in skip or len(out) >= MAX_DIFFS:
        return out
    if isinstance(stored, dict) and isinstance(recomputed, dict):
        for k in list(stored) + [k for k in recomputed if k not in stored]:
            deep_diff(stored.get(k), recomputed.get(k), f"{path}.{k}", out, skip)
    elif isinstance(stored, list) and isinstance(recomputed, list):
        if len(stored) != len(recomputed):
            out.append({"path": path, "stored": f"{len(stored)} item(s)", "recomputed": f"{len(recomputed)} item(s)"})
        for i, (a, b) in enumerate(zip(stored, recomputed)):
            deep_diff(a, b, f"{path}[{i}]", out, skip)
    elif isinstance(stored, (int, float)) and isinstance(recomputed, (int, float)) and not (
            isinstance(stored, bool) or isinstance(recomputed, bool)):
        if not _num_equal(stored, recomputed):
            out.append({"path": path, "stored": stored, "recomputed": recomputed})
    elif stored != recomputed:
        out.append({"path": path, "stored": _short(stored), "recomputed": _short(recomputed)})
    return out


def _short(x):
    s = json.dumps(x, ensure_ascii=False)
    return x if len(s) <= 200 else s[:200] + "…"


# ---------------------------------------------------------------- checks (the Step 0 structural checks)

def structural_checks(s):
    d = s.draft
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
    bad = []
    for gid, it in d["groups"].items():
        if it["role"] == "value":
            v = wf.validate_group(it, s.groups_by_id[gid], s.cols, wf.layout_of(d))
            if v["status"] == "contradicted":
                bad.append(s.groups_by_id[gid]["columns"][0])
    add("value columns are numeric", not bad, ", ".join(bad[:5]))
    fi = d["feature_identity"]
    if wf.layout_of(d) == "samples_in_columns" and fi["group_ids"]:
        v = wf.validate_feature_identity(fi, s.groups_by_id, s.cols, wf.layout_of(d))
        add("feature IDs are unique", not v["messages"], " ".join(v["messages"]))
    dups = [k for k, n in __import__("collections").Counter(wf.sample_ids(s, d)).items() if n > 1]
    add("sample IDs are unique", not dups, ", ".join(dups[:5]))
    rep = wf.consistency_report(s, d)
    add("no sample-ID collisions across blocks", not rep["sample_collisions"],
        "; ".join(c["message"] for c in rep["sample_collisions"]))
    probs = wf.accounting.ledger_problems(wf.accounting.column_ledger(s, d))
    add("every source column is accounted for", not probs, "; ".join(probs[:5]))
    return checks


# ---------------------------------------------------------------- binding

def _referenced(doc):
    main, meta = [], []
    for a in doc.get("assays", []):
        main += a.get("feature_identity", {}).get("columns", [])
        for b in a.get("value_blocks", []):
            main += b.get("columns", [])
    main += [x["column"] for x in doc.get("feature_annotations", []) if not x.get("derived_from")]
    for x in doc.get("sample_metadata", []):
        if x.get("file") == "metadata":
            meta.append(x["column"])
        elif x.get("source") in (None, "data_file"):
            main.append(x["column"])
    if (doc.get("sample_id") or {}).get("column"):
        main.append(doc["sample_id"]["column"])
    for x in doc.get("excluded_columns", []):
        (meta if x.get("file") == "metadata" else main).append(x["column"])
    return list(dict.fromkeys(main)), list(dict.fromkeys(meta))


def _file_entry(doc, role):
    return next((f for f in doc.get("files", []) if f.get("file_role") == role), None)


def bind(doc, table, meta_table):
    """-> (mode, info). Raises ImportRejected for an inconsistent schema or a wrong metadata file."""
    main_ref, meta_ref = _referenced(doc)
    fe = _file_entry(doc, "main") or {}
    src = fe.get("source_columns")
    src_labels = column_labels(src) if src else None
    if src_labels is not None:
        invented = [c for c in main_ref if c not in src_labels]
        if invented:
            raise ImportRejected("The schema names column(s) that are not in its own list of source columns: "
                                 + ", ".join(repr(c) for c in invented[:8]) + ". It was edited by hand or is corrupt.")
    sha = table["sha256"]
    labels = column_labels(table["header"])
    if sha == (fe.get("sha256") or doc.get("file_sha256")):
        mode = "exact"
    elif (src is not None and sorted(table["header"]) == sorted(src)) or (
            src is None and set(main_ref) <= set(labels) and len(labels) == len(set(main_ref))):
        mode = "template"
    else:
        mode = "seeded_wizard"
    info = {"main": {"mode": mode, "uploaded_sha256": sha, "schema_sha256": fe.get("sha256") or doc.get("file_sha256"),
                     "missing_in_file": [c for c in (src_labels or main_ref) if c not in labels],
                     "extra_in_file": [c for c in labels if c not in (src_labels or main_ref)]}}
    mfe = _file_entry(doc, "metadata")
    needs_meta = mfe is not None and any(x.get("file") == "metadata" and x.get("keep", True)
                                         for x in doc.get("sample_metadata", []))
    if meta_table is None:
        if needs_meta:
            raise ImportRejected(f"This schema keeps columns from the metadata file '{mfe.get('name')}': "
                                 "upload that file too.")
        info["metadata"] = None
    else:
        if mfe is None:
            raise ImportRejected("This schema lists no metadata file; leave the metadata upload empty.")
        msrc = mfe.get("source_columns")
        if meta_table["sha256"] == mfe.get("sha256"):
            mmode = "exact"
        elif msrc is not None and sorted(meta_table["header"]) == sorted(msrc):
            mmode = "template"
        else:
            raise ImportRejected(f"The metadata file does not match the schema's metadata file '{mfe.get('name')}' "
                                 "(different contents and different columns).")
        mlabels = column_labels(meta_table["header"])
        bad = [c for c in meta_ref if c not in mlabels]
        if bad:
            raise ImportRejected("The schema names metadata column(s) the file does not have: " + ", ".join(bad[:8]))
        info["metadata"] = {"mode": mmode, "uploaded_sha256": meta_table["sha256"], "schema_sha256": mfe.get("sha256")}
        if mmode == "template" and mode == "exact":
            mode = "template"   # a metadata file with other values changes value-dependent results
            info["main"]["note"] = "the data file is identical but the metadata values differ: template mode"
    return mode, info


# ---------------------------------------------------------------- import

def import_bytes(st, data_name, data_raw, schema_raw, meta_name=None, meta_raw=None):
    try:
        doc, warnings = validate(schema_raw)
    except SchemaRejected as e:
        raise ImportRejected(str(e), e.errors)
    try:
        table = parse_bytes(sanitize_filename(data_name), data_raw)
        meta_table = parse_bytes(sanitize_filename(meta_name), meta_raw) if meta_raw is not None else None
    except InputError as e:
        raise ImportRejected(str(e))
    mode, info = bind(doc, table, meta_table)
    sha = schema_sha256(doc)
    ds = st.add_dataset(doc.get("source_file") or data_name, "import",
                        "wizard_in_progress" if mode == "seeded_wizard" else "imported_awaiting_confirm")
    did = ds["dataset_id"]
    try:
        up = st.dataset_dir(did) / "upload"
        up.mkdir(parents=True, exist_ok=True)
        (up / sanitize_filename(data_name)).write_bytes(data_raw)
        (up / "schema.json").write_bytes(schema_raw)
        if meta_raw is not None:
            (up / sanitize_filename(meta_name)).write_bytes(meta_raw)
        s = wf.create_session(data_name, data_raw)
        if mode == "exact":
            s.filename = sanitize_filename(doc["source_file"])
        s.study = {"session_id": st.sid, "dataset_id": did}
        imported_at = now_iso()
        meta = {"table": meta_table, "name": sanitize_filename(meta_name)} if meta_table is not None else None
        rb = rebuild(s, doc, mode, meta, imported_at=imported_at, schema_sha=sha)
        s.save()
        s.log("schema_import", {"mode": mode, "schema_sha256": sha, "binding": info, "warnings": warnings + rb["warnings"]})
        report = {"dataset_id": did, "step0_session_id": s.sid, "mode": mode, "status": ds["status"],
                  "schema_sha256": sha, "schema_version": doc["schema_version"], "binding": info,
                  "warnings": warnings + rb["warnings"], "missing_in_file": rb["missing_in_file"],
                  "extra_in_file": rb["extra_in_file"], "imported_at": imported_at}
        report.update(recheck(s, doc, mode))
        report["summary"] = summary(s, doc)
        ds["step0_session_id"] = s.sid
        ds["schema_sha256"] = sha
        ds["import"] = {"schema_sha256": sha, "mode": mode, "imported_at": imported_at}
        ds["files"] = {"data": {"name": sanitize_filename(data_name), "sha256": table["sha256"]}}
        if meta_table is not None:
            ds["files"]["metadata"] = {"name": sanitize_filename(meta_name), "sha256": meta_table["sha256"]}
        st.save()
        write_json(st.dataset_dir(did) / "import_report.json", report)
        st.log("schema_import", {"dataset_id": did, "mode": mode, "schema_sha256": sha,
                                 "n_differences": len(report["differences"])})
        return report
    except Exception:
        st.remove_dataset(did)
        raise


def import_dataset(st, data_path, schema_path, metadata_path=None):
    dp, sp = Path(data_path), Path(schema_path)
    mp = Path(metadata_path) if metadata_path else None
    return import_bytes(st, dp.name, dp.read_bytes(), sp.read_bytes(), mp.name if mp else None,
                        mp.read_bytes() if mp else None)


def recheck(s, doc, mode):
    """Recompute everything value-dependent and compare with what the schema stores."""
    checks = structural_checks(s) if mode != "seeded_wizard" else []
    diffs, recomputed = [], None
    if mode != "seeded_wizard":
        try:
            recomputed, _, _ = outputs.build(s, check=False)
            checks.append({"check": "counts agree (assays, blocks, ledger)", "ok": True, "detail": ""})
        except outputs.OutputError as e:
            checks.append({"check": "counts agree (assays, blocks, ledger)", "ok": False, "detail": str(e)})
    if recomputed is not None:
        skip = set(SKIP_DIFF)
        if mode != "exact":
            skip |= {"$.processing_history", "$.questions", "$.source_file", "$.file_sha256"}
        diffs = deep_diff(doc, recomputed, skip=skip)
    open_q = [q["text"] for q in wf.questions.open_questions(s.draft)]
    return {"checks": checks, "differences": diffs, "n_differences": len(diffs), "open_questions": open_q,
            "edited_or_corrupt": mode == "exact" and bool(diffs)}


def summary(s, doc):
    d = s.draft
    ann = [x for x in doc.get("feature_annotations", []) if not x.get("is_feature_id")]
    return {"layout": doc["layout"]["value"],
            "assays": [{"assay_id": a["assay_id"], "assay_label": a["assay_label"], "n_blocks": len(a["value_blocks"]),
                        "n_value_columns": a["n_value_columns"]} for a in doc["assays"]],
            "annotations_kept": sum(1 for x in ann if x.get("keep", True)),
            "annotations_excluded": sum(1 for x in ann if not x.get("keep", True)),
            "excluded_columns": len(doc.get("excluded_columns", [])),
            "design": (doc.get("design") or {}).get("label"),
            "processing_history": {q: (v or {}).get("answer") for q, v in (d.get("processing_history") or {}).items()},
            "n_samples": len(d["samples"])}


def _step0(st, did):
    ds = st.dataset(did)
    try:
        return ds, wf.get_session(ds["step0_session_id"])
    except (KeyError, TypeError):
        raise store.SessionError("The dataset's Step 0 state is missing.")


def accept(st, did, who="user"):
    """The explicit human confirmation (schema_import_confirmed): regenerate the output folder
    through the wizard's finalize and publish it into the session."""
    ds, s = _step0(st, did)
    if ds["status"] != "imported_awaiting_confirm":
        raise store.SessionError(f"Dataset {did} is not awaiting an import confirmation ({ds['status']}).")
    open_q = wf.questions.open_questions(s.draft)
    if open_q:
        raise store.SessionError(f"Answer or dismiss the {len(open_q)} open question(s) of this schema first: "
                                 + "; ".join(q["text"] for q in open_q[:3]))
    with s.lock:
        s.draft["steps"]["review"] = "pending"
        s.log("schema_import_confirmed", {"by": who, "mode": s.draft["import"]["mode"]})
        try:
            outputs.finalize(s)
        except outputs.OutputError as e:
            raise store.SessionError(str(e))
    st2 = store.load(st.sid)
    st2.log("schema_import_confirmed", {"dataset_id": did, "by": who, "mode": s.draft["import"]["mode"]})
    st.data = st2.data
    return st2.dataset(did)


def reject(st, did):
    ds, s = _step0(st, did)
    if ds["status"] == "confirmed":
        raise store.SessionError("A confirmed dataset is never deleted.")
    shutil.rmtree(str(s.dir), ignore_errors=True)
    wf._sessions.pop(s.sid, None)
    st.remove_dataset(did)


def open_in_wizard(st, did):
    """Seed the wizard with the imported schema: every step is asked again."""
    ds, s = _step0(st, did)
    if ds["status"] == "confirmed":
        raise store.SessionError("This dataset is already confirmed.")
    with s.lock:
        for k in s.draft["steps"]:
            s.draft["steps"][k] = "pending"
        wf.after_edit(s)
        s.save()
    st.set_status(did, "wizard_in_progress")
    st.log("schema_import_opened_in_wizard", {"dataset_id": did})
    return s.sid


# ---------------------------------------------------------------- "explain differences" (v3 §10 could-have)

_STAT = {"median": "median value", "p99": "99th percentile", "p1": "1st percentile", "max": "largest value",
         "min": "smallest value", "mean": "mean value", "log10_span": "log10 range", "sample_sum_cv": "CV of the sample sums",
         "n_missing": "number of empty cells", "frac_missing": "share of empty cells", "n_zero": "number of zeros",
         "frac_zero": "share of zeros", "share_integer_valued": "share of integer values", "n_unique": "number of distinct values"}
_BLOCK = re.compile(r"^\$\.assays\[(\d+)\]\.value_blocks\[(\d+)\]\.profile\.(\w+)$")


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def explain(rep):
    """Plain sentences for an import report's binding and its recomputed-vs-stored differences.
    Deterministic (no AI in import): one template per kind of path."""
    out = []
    b = (rep.get("binding") or {}).get("main") or {}
    if rep.get("edited_or_corrupt"):
        out.append({"kind": "edited", "text": "The data file is byte-identical to the one the schema was made from, so "
                    "every difference below means the schema file itself was edited or is corrupt."})
    if b.get("missing_in_file"):
        out.append({"kind": "columns", "text": f"Column(s) named in the schema but absent from the file: "
                    f"{', '.join(b['missing_in_file'])}. They cannot be bound, so the wizard asks about them."})
    if b.get("extra_in_file"):
        out.append({"kind": "columns", "text": f"Column(s) in the file the schema does not know: "
                    f"{', '.join(b['extra_in_file'])}. They start unresolved."})
    n_val = 0
    for d in rep.get("differences") or []:
        p, a, c = d["path"], d.get("stored"), d.get("recomputed")
        m = _BLOCK.match(p)
        if p.endswith(".sha256") or p == "$.file_sha256":
            out.append({"kind": "file", "path": p, "text": "The file is not the one the schema was made from (its sha256 "
                        "differs), so everything computed from the values was recomputed."})
        elif re.match(r"^\$\.files\[\d+\]\.name$", p) or p == "$.source_file":
            out.append({"kind": "file", "path": p, "text": f"The file name differs: the schema was made from '{a}', "
                        f"this upload is '{c}'. Names are not used for binding; the columns are."})
        elif m:
            n_val += 1
            what = _STAT.get(m.group(3), m.group(3).replace("_", " "))
            change = ""
            if _num(a) and _num(c) and a:
                change = f" ({(c - a) / abs(a) * 100:+.3g}%)"
            out.append({"kind": "values", "path": p, "text": f"Assay {int(m.group(1)) + 1}, block {int(m.group(2)) + 1}: "
                        f"the {what} was {a}, it is now {c}{change}. This follows from the values, not the structure."})
        elif re.search(r"\.(n_features|n_samples|n_rows|n_columns|n_value_columns)$", p):
            out.append({"kind": "counts", "path": p, "text": f"{p.rsplit('.', 1)[1].replace('_', ' ')} changed from {a} to {c}: "
                        "the file has a different number of rows or columns than the schema describes."})
        elif "flag" in p or "feature_facts" in p:
            out.append({"kind": "annotations", "path": p, "text": f"An annotation count changed ({p}: {a} → {c}): "
                        "rows marked by an annotation value differ in the new file."})
        else:
            out.append({"kind": "other", "path": p, "text": f"{p}: stored {a}, recomputed {c}."})
    if n_val:
        out.insert(0, {"kind": "summary", "text": f"{n_val} of {len(rep.get('differences') or [])} difference(s) are value "
                       "statistics. In template mode these are expected: the schema is reused as a template and the "
                       "statistics are recomputed from the new values."})
    if not out:
        out.append({"kind": "none", "text": "No difference: the schema matches the file as stored."})
    return out
