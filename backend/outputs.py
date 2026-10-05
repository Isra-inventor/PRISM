"""Canonical outputs (spec 2.2 / 2.3): schema.json + features x samples tables.

Values are copied exactly from the source file. The only change is that
missing-value tokens are written as empty cells (the tokens seen are listed in
parse_report). No row or column of data is dropped; QC / blank / pool samples
and decoy / contaminant rows are kept and only labelled.
"""

from __future__ import annotations

import csv
import io
from collections import Counter, OrderedDict

from . import accounting
from .parsing import cell, is_missing
from .schema import SCHEMA_VERSION
from .profiling import apply_rule
from .workflow import (ASSAY_FIELDS, FACT_FIELDS, FI_FIELDS, GROUP_FIELDS, SAMPLE_FIELDS, layout_of,
                       long_duplicates, provenance, sample_ids, unresolved_items, value_blocks)


class OutputError(Exception):
    pass


def _v(x):
    return "" if is_missing(x) else x


def _csv(header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def _fact(item):
    return {"value": item["value"], "provenance": provenance(item, FACT_FIELDS)}


def _assays(s, d):
    """Kept value groups grouped by assay label, in the order of d['assays']. No block is ranked."""
    assays = OrderedDict((a["assay_label"], (a, [])) for a in d["assays"])
    for g in s.groups:
        it = d["groups"][g["group_id"]]
        if it["role"] != "value" or not it.get("keep", True):
            continue
        lab = it.get("assay_label") or d["assays"][0]["assay_label"]
        if lab not in assays:
            assays[lab] = ({"assay_label": lab, "omics_type": "unknown", "source_software": "unknown",
                            "in_supported_scope": "unsure", "scope_reason": "", "source": "user"}, [])
        assays[lab][1].append(g)
    return assays


def feature_key_indices(s, d):
    return [i for gid in d["feature_identity"]["group_ids"] for i in s.groups_by_id[gid]["indices"]]


def build(s):
    d = s.draft
    unresolved = [u for u in unresolved_items(s, d) if u["step"] != "review" or u["what"].startswith("Column accounting")]
    if unresolved:
        raise OutputError("Resolve these first: " + "; ".join(u["what"] for u in unresolved[:6]))
    lay = layout_of(d)
    if lay == "long":
        dup = long_duplicates(s, d)
        if dup:
            raise OutputError(dup["message"] + f" ({dup['n_pairs']} duplicated feature/sample pairs.)")
    rows = s.table["rows"]
    header = s.table["header"]
    flags = []
    artifacts = OrderedDict()

    assays = _assays(s, d)
    schema_assays = []
    feature_rows_all = []
    fk_idx = feature_key_indices(s, d)
    block_n = 0
    for a_n, (a_label, (assay, gs)) in enumerate(assays.items(), 1):
        aid = f"A{a_n}"
        blocks = []
        n_features = 0
        block_ids = {}
        for b_n, g in enumerate(gs, 1):
            it = d["groups"][g["group_id"]]
            bid = f"B{b_n}"
            prof = dict(g.get("profile") or {})
            for k in ("column", "position", "type", "values"):
                prof.pop(k, None)
            rule = d["sample_rules"].get(g["group_id"], {"strip_prefix": "", "strip_suffix": ""})
            sids = sample_ids(s, d, g["group_id"]) if lay == "samples_in_columns" else sample_ids(s, d)
            where = f"{aid}/{bid}"
            if len(set(sids)) != len(sids):
                dup = [k for k, n in Counter(sids).items() if n > 1]
                flags.append({"flag": "duplicate_sample_ids", "assay": aid, "block": bid,
                              "detail": f"{len(dup)} sample ID(s) occur more than once: {', '.join(dup[:5])}"})
            if lay == "samples_in_columns":
                keys = ["|".join(cell(r, i) for i in fk_idx) for r in rows]
                mat = [[k] + [_v(cell(r, i)) for i in g["indices"]] for k, r in zip(keys, rows)]
                feat_keys = keys
            elif lay == "samples_in_rows":
                feat_keys = [header[i] for i in g["indices"]]
                mat = [[header[i]] + [_v(cell(r, i)) for r in rows] for i in g["indices"]]
            else:
                vi = g["indices"][0]
                si = s.groups_by_id[d["sample_id_group"]["value"]]["indices"][0]
                pivot, feat_keys = OrderedDict(), []
                for r in rows:
                    fk = "|".join(cell(r, i).strip() for i in fk_idx)
                    if fk not in pivot:
                        pivot[fk] = {}
                        feat_keys.append(fk)
                    pivot[fk][cell(r, si).strip()] = _v(cell(r, vi))
                mat = [[fk] + [pivot[fk].get(sm, "") for sm in sids] for fk in feat_keys]
            dupk = [k for k, n in Counter(feat_keys).items() if n > 1]
            if dupk and b_n == 1:
                flags.append({"flag": "duplicate_feature_keys", "assay": aid,
                              "detail": f"{len(dupk)} feature key(s) occur more than once (kept as is): {', '.join(dupk[:5])}"})
            if any(not k for k in feat_keys) and b_n == 1:
                flags.append({"flag": "empty_feature_keys", "assay": aid,
                              "detail": f"{sum(1 for k in feat_keys if not k)} feature(s) have an empty key (kept)."})
            fname = f"value_matrix_{aid}_{bid}.csv"
            artifacts[fname] = _csv(["feature_key"] + sids, mat)
            # one set of feature rows per assay (blocks of one assay share the rows),
            # except samples-in-rows, where each block's columns are its own features
            if lay == "samples_in_rows" or b_n == 1:
                feature_rows_all.append((aid, g, feat_keys))
            n_features = max(n_features, len(feat_keys))
            block_ids[g["group_id"]] = sids
            blocks.append({
                "block_id": bid, "group_id": g["group_id"], "file": fname, "keep": True,
                "columns": g["columns"],
                "sample_id_rule": {"strip_prefix": rule.get("strip_prefix", ""), "strip_suffix": rule.get("strip_suffix", ""),
                                   "add_prefix": rule.get("add_prefix", "")} if lay == "samples_in_columns" else None,
                "label": it.get("label") or "", "confidence": it.get("confidence"),
                "provenance": provenance(it, GROUP_FIELDS),
                "profile": prof,
                "n_features": len(feat_keys), "n_samples": len(sids),
            })
        if not blocks:
            continue
        union, structure = _reconcile_samples(a_label, gs, block_ids, d, lay)
        fi = d["feature_identity"]
        schema_assays.append({
            "assay_id": aid,
            "assay_label": a_label,
            "omics_type": assay.get("omics_type") or "unknown",
            "source_software": assay.get("source_software") or "unknown",
            "in_supported_scope": assay.get("in_supported_scope") or "unsure",
            "scope_reason": assay.get("scope_reason") or "",
            "provenance": provenance(assay, ASSAY_FIELDS),
            "feature_identity": ({"columns": [c for gid in fi["group_ids"] for c in s.groups_by_id[gid]["columns"]],
                                  "composite": len(fk_idx) > 1, "provenance": provenance(fi, FI_FIELDS)}
                                 if lay != "samples_in_rows" else
                                 {"columns": [], "from": "column headers of the value block", "composite": False,
                                  "provenance": "computed"}),
            "value_blocks": blocks,
            "n_features": n_features,
            "n_samples": len(union),
            "n_value_columns": sum(g["n_columns"] for g in gs),
            "sample_structure": structure,
        })
        if assay.get("in_supported_scope") in ("no", "unsure"):
            flags.append({"flag": "outside_supported_scope", "assay": aid,
                          "detail": f"Recognized as {assay.get('omics_type') or 'unknown'} "
                                    f"(scope: {assay.get('in_supported_scope')}) - not yet supported by PRISM's audit. "
                                    + (assay.get("scope_reason") or "")})

    # feature metadata
    ann = [(gid, it) for gid, it in d["groups"].items()
           if it["role"] in ("feature_annotation", "feature_id") and it.get("keep", True)]
    ann_idx = [(s.groups_by_id[gid], it) for gid, it in ann]
    if lay == "samples_in_columns":
        cols = [c for g, _ in ann_idx for c in g["columns"]]
        idxs = [i for g, _ in ann_idx for i in g["indices"]]
        fm = []
        for aid, g, keys in feature_rows_all:
            fm.extend([[k, aid] + [_v(cell(r, i)) for i in idxs] for k, r in zip(keys, rows)])
        artifacts["feature_metadata.csv"] = _csv(["feature_key", "assay_id"] + cols, fm)
    elif lay == "samples_in_rows":
        fm = []
        for aid, g, keys in feature_rows_all:
            fm.extend([[k, aid, g["group_id"]] for k in keys])
        artifacts["feature_metadata.csv"] = _csv(["feature_key", "assay_id", "source_block"], fm)
    else:
        cols, idxs = [], []
        for g, _ in ann_idx:
            for c, i in zip(g["columns"], g["indices"]):
                per = {}
                const = True
                for r in rows:
                    fk = "|".join(cell(r, j).strip() for j in fk_idx)
                    v = cell(r, i)
                    if per.setdefault(fk, v) != v:
                        const = False
                        break
                if const:
                    cols.append(c)
                    idxs.append(i)
                else:
                    flags.append({"flag": "annotation_not_constant_per_feature", "column": c,
                                  "detail": "Differs between rows of the same feature; not written to feature_metadata.csv."})
        first = OrderedDict()
        for r in rows:
            fk = "|".join(cell(r, j).strip() for j in fk_idx)
            first.setdefault(fk, r)
        fm = []
        for aid, g, keys in feature_rows_all:
            fm.extend([[k, aid] + [_v(cell(first[k], i)) for i in idxs] for k in keys])
        artifacts["feature_metadata.csv"] = _csv(["feature_key", "assay_id"] + cols, fm)

    # sample metadata
    sids = sample_ids(s, d)
    st = d["samples"]
    smd_cols, smd_vals = [], {}
    meta_groups = [(s.groups_by_id[gid], it) for gid, it in d["groups"].items()
                   if it["role"] == "sample_metadata" and it.get("keep", True)]
    if lay == "samples_in_rows":
        for g, it in meta_groups:
            for c, i in zip(g["columns"], g["indices"]):
                smd_cols.append(c)
                for sid, r in zip(sids, rows):
                    smd_vals.setdefault(sid, {})[c] = _v(cell(r, i))
    elif lay == "long":
        si = s.groups_by_id[d["sample_id_group"]["value"]]["indices"][0]
        for g, it in meta_groups:
            for c, i in zip(g["columns"], g["indices"]):
                per, const = {}, True
                for r in rows:
                    k = cell(r, si).strip()
                    if per.setdefault(k, cell(r, i)) != cell(r, i):
                        const = False
                        break
                if const:
                    smd_cols.append(c)
                    for k, v in per.items():
                        smd_vals.setdefault(k, {})[c] = _v(v)
                else:
                    flags.append({"flag": "sample_metadata_not_constant_per_sample", "column": c,
                                  "detail": "Differs between rows of the same sample; not written to sample_metadata.csv."})
    meta = d.get("metadata")
    if meta and not meta.get("skipped") and s.metadata_table:
        t = s.metadata_table["table"]
        id_i = t["header"].index(meta["id_column"])
        alias = {m: dd for dd, m in meta.get("accepted_near_misses", [])}
        kept = [c for c in meta["columns"] if c["role"] != "sample_id" and c.get("keep", True)]
        for c in kept:
            smd_cols.append(c["column"])
        for r in t["rows"]:
            mid = cell(r, id_i).strip()
            sid = alias.get(mid, mid)
            if sid in sids:
                for c in kept:
                    smd_vals.setdefault(sid, {})[c["column"]] = _v(cell(r, c["index"]))
        rep = meta["report"]
        if rep["only_in_data"]:
            flags.append({"flag": "samples_without_metadata",
                          "detail": f"{len(rep['only_in_data'])} sample(s) have no row in the metadata file."})
    elif lay == "samples_in_columns" and not d.get("derived_sample_metadata"):
        flags.append({"flag": "no_sample_metadata",
                      "detail": "No sample metadata file was provided (step skipped)."})
    der = d.get("derived_sample_metadata")
    der_cols = [c for c in (der or {}).get("columns", []) if c.get("keep", True)] if lay == "samples_in_columns" else []
    if der_cols:   # sample information parsed from column names (v2.3 'one block' confirmation)
        raw_to_sid = {}
        for gid in value_blocks(s, d):
            rule = d["sample_rules"].get(gid) or {}
            for i in s.groups_by_id[gid]["indices"]:
                raw_to_sid[header[i]] = apply_rule(header[i], rule)
        for c in der_cols:
            smd_cols.append(c["name"])
        for raw, vals in der["values"].items():
            sid = raw_to_sid.get(raw)
            if sid in sids:
                for c in der_cols:
                    smd_vals.setdefault(sid, {})[c["name"]] = vals.get(c["name"], "")
    artifacts["sample_metadata.csv"] = _csv(
        ["sample_id", "sample_label", "is_study_sample"] + smd_cols,
        [[sid, st.get(sid, {}).get("label", ""), "true" if st.get(sid, {}).get("is_study_sample", True) else "false"]
         + [smd_vals.get(sid, {}).get(c, "") for c in smd_cols] for sid in sids])

    # parse-level integrity flags
    pr = s.table["parse_report"]
    if pr["ragged_rows"]["count"]:
        flags.append({"flag": "ragged_rows", "detail": f"{pr['ragged_rows']['count']} row(s) have a different number of fields."})
    if pr["duplicate_column_names"]:
        flags.append({"flag": "duplicate_column_names",
                      "detail": ", ".join(x["name"] for x in pr["duplicate_column_names"][:5])})
    if pr["decimal_comma_columns"]:
        flags.append({"flag": "decimal_comma_values",
                      "detail": "Values with a decimal comma were copied as-is: " + ", ".join(pr["decimal_comma_columns"][:5])})
    if d.get("clarifying_questions"):
        flags.append({"flag": "open_ai_questions", "detail": "; ".join(q["question"] for q in d["clarifying_questions"][:3])})

    schema = {
        "schema_version": SCHEMA_VERSION,
        "source_file": s.filename,
        "file_sha256": s.sha,
        "layout": _fact(d["layout"]),
        "assays": schema_assays,
        "feature_annotations": [_annotation(s, gid, it, c, i)
                                for gid, it in d["groups"].items() if it["role"] in ("feature_annotation", "feature_id")
                                for c, i in zip(s.groups_by_id[gid]["columns"], s.groups_by_id[gid]["indices"])],
        "sample_metadata": [
            {"column": c, "audit_kind": it.get("audit_kind"), "label": it.get("label") or "",
             **({"detail": it["detail"]} if it.get("detail") else {}),
             "keep": it.get("keep", True), "source": "data_file", "provenance": provenance(it, GROUP_FIELDS)}
            for gid, it in d["groups"].items() if it["role"] == "sample_metadata"
            for c in s.groups_by_id[gid]["columns"]] + (
            [{"column": c["column"], "audit_kind": c["audit_kind"], "label": c.get("label") or "",
              **({"detail": c["detail"]} if c.get("detail") else {}), "keep": c.get("keep", True),
              "source": "metadata_file",
              "provenance": ("computed" if all(c.get("proposed", {}).get(k) == c.get(k) for k in ("audit_kind", "label"))
                             else "user_set")}
             for c in (meta or {}).get("columns", []) if c["role"] != "sample_id"]
            if meta and not meta.get("skipped") else []) + [
            {"column": c["name"], "audit_kind": c["audit_kind"], "label": c.get("label") or "", "keep": c.get("keep", True),
             "source": "column_names", "provenance": "user_set"} for c in der_cols],
        "sample_id": ({"column": s.groups_by_id[d["sample_id_group"]["value"]]["columns"][0]}
                      if d["sample_id_group"]["value"] else {"from": "value column headers"}),
        "samples": [{"sample": sid, "label": v.get("label") or "", "is_study_sample": bool(v.get("is_study_sample", True)),
                     "provenance": provenance(v, SAMPLE_FIELDS)} for sid, v in st.items()],
        "excluded_columns": [{k: x[k] for k in ("column", "file", "reason", "by", "at")}
                             for x in accounting.excluded_columns(s, d)],
        "files": accounting.files_entries(s, d),
        "column_ledger": {f: {k: v for k, v in x.items() if k != "problems"}
                          for f, x in accounting.column_ledger(s, d).items()},
        "processing_history": dict(d["processing_history"], software_and_version=d.get("software_and_version", ""),
                                   notes=d.get("history_notes", "")),
        "parse_report": pr,
        "integrity_flags": flags,
        "ai": {"provider": d["ai"]["provider"], "model": d["ai"].get("model"),
               "models_used": d["ai"].get("models_used", []), "prompt_version": d["ai"]["prompt_version"],
               "temperature": d["ai"]["temperature"], "enabled": d["ai"]["enabled"]},
        "signature_hint": d.get("signature_hint"),
        "clarifying_questions": d.get("clarifying_questions", []),
        "log_ref": f"{s.sid}.jsonl",
    }
    return schema, artifacts, flags


def _reconcile_samples(a_label, gs, block_ids, d, lay):
    """Finalization invariant (v2.3 §3): the unique samples of an assay must match the
    entries in samples[] and be explainable from the raw column counts; otherwise stop."""
    union = list(dict.fromkeys(x for ids in block_ids.values() for x in ids))
    if lay != "samples_in_columns":
        missing = [x for x in union if x not in d["samples"]]
        if missing:
            raise OutputError(f"Sample counts do not reconcile for assay '{a_label}': {len(union)} sample IDs, "
                              f"but {len(missing)} of them have no entry in samples[] (e.g. '{missing[0]}').")
        return union, "rows"
    raw = sum(g["n_columns"] for g in gs)
    dup_within = {gid: len(ids) - len(set(ids)) for gid, ids in block_ids.items() if len(ids) != len(set(ids))}
    sets = [set(ids) for ids in block_ids.values()]
    if len(sets) == 1:
        structure, expected = "single", len(union)
    elif all(x == sets[0] for x in sets):
        structure, expected = "parallel", len(union) * len(sets)
    elif sum(len(x) for x in sets) == len(union):
        structure, expected = "disjoint", len(union)
    else:
        structure, expected = "collision", None
    in_samples = [x for x in union if x in d["samples"]]
    problems = []
    if dup_within:
        problems.append(f"{sum(dup_within.values())} sample ID(s) repeat inside one block")
    if expected is None:
        problems.append("the same sample ID appears in blocks that do not hold the same samples")
    elif raw != expected:
        problems.append(f"{raw} raw value columns cannot be explained by {len(union)} samples in {len(sets)} "
                        f"block(s) ({structure})")
    if len(in_samples) != len(union):
        problems.append(f"{len(union) - len(in_samples)} sample ID(s) have no entry in samples[]")
    if problems:
        raise OutputError(f"Sample counts do not reconcile for assay '{a_label}': {raw} raw value columns, "
                          f"{len(union)} unique sample IDs, {len(in_samples)} matching entries in samples[] — "
                          + "; ".join(problems) + ". Fix the sample IDs in step 5 (Samples) before finishing.")
    return union, structure


def _annotation(s, gid, it, column, i):
    out = {"column": column, "label": it.get("label") or "", "is_feature_id": it["role"] == "feature_id",
           "marks_rows_as_suspect": bool(it.get("marks_rows_as_suspect")), "keep": it.get("keep", True),
           "provenance": provenance(it, GROUP_FIELDS)}
    if it.get("marks_rows_as_suspect"):
        fv = it.get("flag_values")
        out["flag_values"] = fv
        out["flagged_value"] = it.get("flagged_value")
        if it.get("flagged_value") is not None:
            out["n_flagged"] = sum(1 for r in s.table["rows"]
                                   if ("" if is_missing(cell(r, i)) else cell(r, i).strip()) == it["flagged_value"])
    return out
