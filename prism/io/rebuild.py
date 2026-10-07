"""schema.json -> a Step 0 draft (v3 §4.3), so the regular finalize can regenerate the outputs.

The schema is the contract: every role, block, label, exclusion, sample label, design source,
derivation rule and answer is taken from it, and every item keeps its stored provenance.
Nothing value-dependent is taken from it: profiles, counts, flag counts, join results and
design facts are recomputed from the uploaded file by the same code the wizard uses.

mode 'exact'          the uploaded file is the one the schema was made from (sha256)
mode 'template'       same column names, other values: processing history, the time-unit answer
                      and answered questions are not imported
mode 'seeded_wizard'  other columns: what matches carries over, the rest is unresolved and the
                      wizard opens with this as its starting proposal
"""

from __future__ import annotations

import copy
import re
from collections import OrderedDict

from backend import ai, derive, questions as qz, workflow as wf
from backend.parsing import cell
from backend.profiling import Columns, make_group
from backend.schema import HISTORY_QUESTIONS, SCHEMA_VERSION, UNRESOLVED
from backend.validation import validate_feature_identity, validate_group

_MARK = {"__imported__": True}


def forge(item, fields, prov):
    """Set source / proposed / set_by so that backend provenance() returns the stored value."""
    snap = {f: copy.deepcopy(item.get(f)) for f in fields}
    item["imported"] = True
    if prov == "computed":
        item["source"] = "computed"
    elif prov in ("ai_proposed_confirmed", "ai_proposed_corrected"):
        item["source"] = "ai"
        if prov == "ai_proposed_corrected":
            snap[fields[0]] = dict(_MARK)
            item["set_by"], item["set_corrected"] = "ai_patch", True
    else:
        item["source"] = "none"
    item["proposed"] = snap
    return item


def forge_design_side(des, side, prov):
    x = des[side]
    x["proposed"] = {k: x.get(k) for k in wf.DESIGN_FIELDS}
    x["proposed_derivation"] = des.get("derivation")
    x["source_of_proposal"] = {"computed": "computed", "ai_proposed_confirmed": "ai",
                               "ai_proposed_corrected": "ai"}.get(prov, "none")
    if prov == "ai_proposed_corrected":
        x["proposed"] = dict(x["proposed"], source=dict(_MARK))
        x["set_by"], x["set_corrected"] = "ai_patch", True
    return x


class Collector:
    """Columns -> groups from the schema's sections (group ids kept where the schema has them)."""

    def __init__(self, labels):
        self.idx = {lab: i for i, lab in enumerate(labels)}
        self.groups = OrderedDict()
        self.placed = {}
        self.missing = []

    def add(self, gid, col, role, item, prov):
        if col not in self.idx:
            self.missing.append(col)
            return
        if col in self.placed:
            return
        g = self.groups.setdefault(gid, {"cols": [], "role": role, "item": item, "prov": prov})
        g["cols"].append(col)
        self.placed[col] = gid


def rebuild(s, doc, mode, meta=None, imported_at=None, schema_sha=None):
    """Install groups and a draft on the backend session s from the schema doc. meta: the parsed
    metadata table (or None). Returns {"missing_in_file": [...], "extra_in_file": [...], "warnings": [...]}."""
    labels = s.cols.labels
    col = Collector(labels)
    warnings = []
    lay = doc["layout"]["value"]
    fid_cols = []
    for a in doc["assays"]:
        fid_cols += [c for c in a["feature_identity"].get("columns", []) if c not in fid_cols]
        for b in a["value_blocks"]:
            for c in b["columns"]:
                col.add(b["group_id"], c, "value", {"label": b.get("label") or "", "confidence": b.get("confidence"),
                                                    "assay_label": a["assay_label"], "keep": True}, b["provenance"])
    for x in doc.get("feature_annotations", []):
        if x.get("derived_from"):
            continue
        role = "feature_id" if x.get("is_feature_id") or x["column"] in fid_cols else "feature_annotation"
        item = {"label": x.get("label") or "", "keep": x.get("keep", True), "family": x.get("family"),
                "marks_rows_as_suspect": bool(x.get("marks_rows_as_suspect")),
                "flagged_values": list(x.get("flagged_values") or ([] if x.get("flagged_value") is None
                                                                   else [x["flagged_value"]]))}
        col.add(x.get("group_id") or f"fa:{x['column']}", x["column"], role, item, x["provenance"])
    for x in doc.get("sample_metadata", []):
        if (x.get("file") or "main") != "main" or x.get("source") not in (None, "data_file"):
            continue
        item = {k: x.get(k) for k in ("audit_kind", "label", "detail", "family", "keep", "confidence", "evidence")}
        item["label"] = item["label"] or ""
        col.add(x.get("group_id") or f"sm:{x['column']}", x["column"], "sample_metadata", item, x["provenance"])
    sid_col = (doc.get("sample_id") or {}).get("column")
    if sid_col:
        col.add(f"si:{sid_col}", sid_col, "sample_id", {"label": "sample identifier"}, "computed")
    excl = {}
    for x in doc.get("excluded_columns", []):
        if (x.get("file") or "main") != "main":
            continue
        excl[x["column"]] = {"reason": x.get("reason"), "by": x.get("by"), "at": x.get("at")}
        role = x.get("role") or "ignore"
        col.add(x.get("group_id") or f"ex:{x['column']}", x["column"], role,
                {"label": x.get("label") or "", "keep": role == "ignore" and True or False}, "user_set")
    extra = [c for c in labels if c not in col.placed]
    if mode == "seeded_wizard":
        for c in extra:
            col.add(f"un:{c}", c, UNRESOLVED, {"label": ""}, "user_set")
    elif extra:
        warnings.append(f"{len(extra)} column(s) of the file are in no section of the schema: "
                        + ", ".join(extra[:6]))
        for c in extra:
            col.add(f"un:{c}", c, UNRESOLVED, {"label": ""}, "user_set")

    # group ids: keep the schema's, number the rest after them
    used = {g for g in col.groups if re.fullmatch(r"g\d+", g)}
    nxt = max([int(g[1:]) for g in used] + [0]) + 1
    gid_of = {}
    for g in col.groups:
        if re.fullmatch(r"g\d+", g):
            gid_of[g] = g
        else:
            gid_of[g] = f"g{nxt}"
            nxt += 1
    order = sorted(col.groups, key=lambda g: min(col.idx[c] for c in col.groups[g]["cols"]))
    s.groups = [make_group(s.cols, gid_of[g], sorted(col.idx[c] for c in col.groups[g]["cols"]), "imported")
                for g in order]

    d = {"schema_version": SCHEMA_VERSION, "groups": OrderedDict(), "sample_rules": {}, "samples": {},
         "sample_rules_ai": [], "rejected": [], "chat": [], "patches": [], "settings": {}, "answers": {},
         "questions": [], "derived_feature_annotations": [],
         "processing_history": {q: {"answer": None, "note": "", "provenance": "unanswered", "answered_at": None}
                                for q, _ in HISTORY_QUESTIONS},
         "software_and_version": "", "history_notes": "", "metadata": None,
         "steps": {st: "pending" for st in wf.STEPS}, "signature_hint": doc.get("signature_hint"),
         "grouping": {"source": "import", "chunks": 0, "merges_applied": [], "splits_applied": [],
                      "failed_chunks": [], "consolidation_error": None}}
    aib = doc.get("ai") or {}
    d["ai"] = {"enabled": bool(aib.get("enabled", True)), "available": False, "unavailable_reason": None, "used": False,
               "provider": aib.get("provider"), "model": aib.get("model"), "models_used": aib.get("models_used", []),
               "prompt_version": aib.get("prompt_version"), "temperature": aib.get("temperature"), "error": None,
               "cached": False, "calls": 0}
    if mode == "exact":
        d["log_ref"] = doc.get("log_ref")
    d["import"] = {"mode": mode, "schema_sha256": schema_sha, "imported_at": imported_at}
    lay_item = {"value": lay, "confidence": 1.0, "evidence": "From the imported schema.",
                "validation": {"status": "ok", "messages": []}, "hint": wf.layout_hint_text(s.hints)}
    d["layout"] = forge(lay_item, wf.FACT_FIELDS, doc["layout"]["provenance"])
    d["assays"] = []
    for a in doc["assays"]:
        x = {"assay_label": a["assay_label"], "omics_type": a.get("omics_type") or "unknown",
             "omics_family": (a.get("omics_family") or {}).get("value") or "unknown",
             "source_software": a.get("source_software") or "unknown",
             "in_supported_scope": a.get("in_supported_scope") or "unsure", "scope_reason": a.get("scope_reason") or "",
             "confidence": 1.0, "evidence": "From the imported schema."}
        d["assays"].append(forge(x, wf.ASSAY_FIELDS, a["provenance"]))

    for g in order:
        rec = col.groups[g]
        gid = gid_of[g]
        it = wf._blank(dict(rec["item"], role=rec["role"]))
        ai.normalize_item(it)
        if rec["role"] == "feature_annotation":
            it["marks_rows_as_suspect"] = bool(rec["item"].get("marks_rows_as_suspect"))
            it["flagged_values"] = list(rec["item"].get("flagged_values") or [])
        if rec["role"] == "value" and not it.get("assay_label"):
            it["assay_label"] = d["assays"][0]["assay_label"]
        it["confidence"] = rec["item"].get("confidence") if rec["item"].get("confidence") is not None else 1.0
        it["evidence"] = rec["item"].get("evidence") or ""
        grp = s.groups_by_id[gid]
        it["hint"] = wf.group_hint(grp)
        it["flag_values"] = wf.flag_values(s, gid) if it.get("marks_rows_as_suspect") else None
        if it.get("flag_values") is not None:
            bad = [v for v in it["flagged_values"] if v not in it["flag_values"]]
            if bad:
                warnings.append(f"{grp['columns'][0]}: flagged value(s) {bad} do not occur in this file.")
                it["flagged_values"] = [v for v in it["flagged_values"] if v in it["flag_values"]]
        excluded = it["role"] == "ignore" or not it.get("keep", True)
        if excluded:
            first = grp["columns"][0]
            it["excluded"] = excl.get(first) or {"reason": "left out of the outputs", "by": "user", "at": None}
        if it["role"] != UNRESOLVED:
            it["validation"] = validate_group(it, grp, s.cols, lay)
        d["groups"][gid] = forge(it, wf.GROUP_FIELDS, rec["prov"])

    fids = [gid_of[col.placed[c]] for c in fid_cols if c in col.placed]
    fi0 = doc["assays"][0]["feature_identity"]
    fi = {"group_ids": list(dict.fromkeys(fids)), "composite": len(set(fids)) > 1, "confidence": 1.0,
          "evidence": "From the imported schema."}
    fi["validation"] = validate_feature_identity(fi, s.groups_by_id, s.cols, lay)
    d["feature_identity"] = forge(fi, wf.FI_FIELDS, fi0.get("provenance") or "computed")
    sg = gid_of[col.placed[sid_col]] if sid_col and sid_col in col.placed else None
    d["sample_id_group"] = forge({"value": sg}, wf.FACT_FIELDS, "computed")
    rules = {b["group_id"]: b.get("sample_id_rule") for a in doc["assays"] for b in a["value_blocks"]}
    for g in s.groups:
        gid = g["group_id"]
        r = rules.get(gid) if gid in rules else None
        if r is not None:
            d["sample_rules"][gid] = wf._snap({"strip_prefix": r.get("strip_prefix", ""), "strip_suffix":
                                               r.get("strip_suffix", ""), "add_prefix": r.get("add_prefix", ""),
                                               "source": "computed"}, ("strip_prefix", "strip_suffix"))
        elif g.get("sample_id_rule") is not None:
            d["sample_rules"][gid] = wf._snap(dict(g["sample_id_rule"], source="computed"), ("strip_prefix", "strip_suffix"))
    s.draft = d

    # samples: labels as stored
    ids = wf.sample_ids(s, d)
    stored = {x["sample"]: x for x in doc.get("samples", [])}
    for sid in ids:
        x = stored.get(sid)
        if x is None:
            continue
        d["samples"][sid] = forge({"label": x.get("label") or "", "is_study_sample": bool(x.get("is_study_sample", True)),
                                   "evidence": "From the imported schema.", "confidence": 1.0, "user_edited": True},
                                  wf.SAMPLE_FIELDS, x["provenance"])
    wf.refresh_samples(s, d)

    # processing history: only an exact import keeps your answers
    ph = doc.get("processing_history") or {}
    if mode == "exact":
        for q, _ in HISTORY_QUESTIONS:
            if isinstance(ph.get(q), dict):
                d["processing_history"][q] = dict(ph[q])
        d["software_and_version"] = ph.get("software_and_version") or ""
        d["history_notes"] = ph.get("notes") or ""
    else:
        for q, _ in HISTORY_QUESTIONS:
            if isinstance(ph.get(q), dict) and ph[q].get("ai_hint"):
                d["processing_history"][q]["ai_hint"] = ph[q]["ai_hint"]

    # metadata file
    if meta is not None:
        d["metadata"] = _metadata(s, d, doc, meta, mode, ids)
    elif lay == "samples_in_columns":
        d["metadata"] = {"skipped": True}

    # design
    des = doc.get("design")
    if des:
        nd = {"subject": {"source": des["subject"]["source"], "column": des["subject"].get("column"),
                          "file": des["subject"].get("file")},
              "time": {"source": des["time"]["source"], "column": des["time"].get("column"),
                       "file": des["time"].get("file"), "unit": {"value": None, "provenance": "unanswered"}},
              "derivation": des["subject"].get("derivation") or des["time"].get("derivation")}
        unit = des["time"].get("unit") or {}
        if mode == "exact" and unit.get("value"):
            nd["time"]["unit"] = dict(unit)
        for side in ("subject", "time"):
            forge_design_side(nd, side, des[side].get("provenance") or "user_set")
        d["design"] = nd
    else:
        d["design"] = wf._snap_design(wf.design.blank(), "none")

    # derived feature annotations (rules recomputed on this file)
    for x in doc.get("feature_annotations", []):
        if x.get("derived_from") != "feature_names":
            continue
        rule = x.get("rule") or {}
        part = rule.get("part") or "left"
        e = {"name": x["column"], "label": x.get("label") or x["column"], "part": part, "derived_from": "feature_names",
             "source": x.get("source"), "column": x.get("from_column"),
             "rule": {"delimiter": rule.get("delimiter"), "occurrence": rule.get("occurrence")},
             "display_labels": dict(x.get("display_labels") or {}), "keep": x.get("keep", True), "imported": True}
        e["set_by"] = "user" if x.get("provenance") == "user_set" else "ai_patch"
        e["set_corrected"] = x.get("provenance") == "ai_proposed_corrected"
        try:
            names = wf.feature_names(s, d, e["source"], e["column"])
            parts = (e["name"], "") if part == "left" else ("", e["name"])
            prev = derive.preview(names, e["rule"], parts)
            e.update(coverage=prev["coverage"], n_failures=prev["n_failures"],
                     n_distinct=prev["parts"][e["name"]]["n_distinct"], n_empty=prev["parts"][e["name"]]["n_empty"])
            if mode == "exact" and x.get("coverage") is not None:
                e["coverage"] = x["coverage"]
        except (wf.StepError, derive.RuleError, KeyError) as err:
            warnings.append(f"derived annotation '{e['name']}' cannot be recomputed: {err}")
            e.update(coverage=0.0, n_failures=0, n_distinct=0, n_empty=0)
        d["derived_feature_annotations"].append(e)

    # questions: an exact import keeps them (open ones stay open); a template import keeps only
    # the AI's open questions (code questions are recomputed, answers were about other values)
    for q in doc.get("questions", []):
        if mode != "exact" and (q["status"] != "open" or q["source"] == "code"):
            continue
        qq = {"question_id": q["question_id"], "source": q["source"], "kind": q.get("kind") or "imported",
              "type": q["type"], "text": q["text"], "applies_to": q.get("applies_to") or {},
              "step": q.get("step") or "review", "allow_free_text": True, "imported": True,
              "options": [{"option_id": o["option_id"], "label": o["label"], "edits": []} for o in q.get("options", [])],
              "key": q.get("key") or qz.key_of(q["type"], q.get("applies_to") or {"text": q["text"]})}
        d["questions"].append(qq)
        if q["status"] != "open":
            a = q.get("answer") or {}
            d["answers"][q["question_id"]] = {"status": q["status"], "key": qq["key"], "option_ids": [],
                                             "labels": a.get("labels") or [], "text": a.get("text"),
                                             "by": a.get("by") or "user", "at": a.get("at"), "edit_id": None}

    # steps: an imported schema was confirmed step by step; a seeded wizard starts over
    for st in wf.STEPS[:-1]:
        d["steps"][st] = "pending" if mode == "seeded_wizard" else (
            "confirmed" if wf.step_applicable(s, d, st) else "not_applicable")
    wf.after_edit(s)
    return {"missing_in_file": sorted(set(col.missing)), "extra_in_file": extra, "warnings": warnings}


def _metadata(s, d, doc, meta, mode, ids):
    t = meta["table"]
    cols = Columns(t)
    s.metadata_table = {"table": t, "cols": cols}
    fe = next((f for f in doc.get("files", []) if f.get("file_role") == "metadata"), {}) or {}
    join = fe.get("join") or {}
    key = join.get("key_column")
    if key not in cols.labels:
        key = cols.labels[0]
    stored = {x["column"]: x for x in doc.get("sample_metadata", []) if x.get("file") == "metadata"}
    excl = {x["column"]: x for x in doc.get("excluded_columns", []) if x.get("file") == "metadata"}
    columns = []
    for i, name in enumerate(cols.labels):
        dg = cols.digests[i]
        x = stored.get(name) or {}
        c = {"column": name, "index": i, "role": "sample_id" if name == key else "sample_metadata",
             "audit_kind": x.get("audit_kind"), "label": x.get("label") or "", "detail": x.get("detail"),
             "family": x.get("family"), "keep": True if name == key else x.get("keep", True),
             "confidence": x.get("confidence") or 0.0, "evidence": x.get("evidence") or "",
             "hint": wf.group_hint({"kind": "single_column", "type": dg["type"], "profile": dg, "n_columns": 1})}
        if name != key and not c["keep"]:
            e = excl.get(name) or {}
            c["excluded"] = {"reason": e.get("reason") or "left out of the outputs", "by": e.get("by") or "user",
                             "at": e.get("at")}
        forge(c, ("role", "audit_kind", "label", "keep"), x.get("provenance") or ("computed" if name == key else "user_set"))
        columns.append(c)
    valid = {tuple(p) for p in join.get("accepted_near_misses") or []}
    rep = wf.match_report(ids, [cell(r, columns[cols.labels.index(key)]["index"]).strip() for r in t["rows"]])
    near = {(n["data_id"], n["metadata_id"]) for n in rep["near_misses"]}
    return {"filename": fe.get("name") if mode == "exact" and fe.get("name") else meta["name"], "id_column": key,
            "join_key_source": "imported", "join_key_evidence": "From the imported schema.", "join_key_confidence": 1.0,
            "columns": columns, "accepted_near_misses": [list(p) for p in valid if p in near], "skipped": False,
            "sha256": t["sha256"], "n_rows": len(t["rows"]), "n_columns": len(t["header"]),
            "source_columns": list(t["header"]), "parse_report": t["parse_report"], "ai": {"used": False, "error": None},
            "join_facts_top": [], "sample_source": join.get("sample_source") or (
                "value column headers" if d["layout"]["value"] == "samples_in_columns" else "sample ID column"),
            "report": rep}
