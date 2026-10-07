"""Accounting (v2.4 §9): nothing disappears silently.

Every column of every uploaded file must be in exactly one place: feature ID,
annotation, value block, sample ID, sample metadata, or excluded (with reason, who
and when). The ledger counts them per file; finalization is blocked when a column
is unaccounted for (or still unresolved).
"""

from __future__ import annotations

import hashlib

from .schema import UNRESOLVED

CATEGORIES = ("feature_id", "annotation", "value", "sample_id", "sample_metadata", "excluded")
_ROLE_CAT = {"feature_id": "feature_id", "feature_annotation": "annotation", "value": "value",
             "sample_id": "sample_id", "sample_metadata": "sample_metadata"}


def columns_sha256(header):
    return hashlib.sha256("\x1f".join(header).encode("utf-8")).hexdigest()


def _main_places(s, d):
    """column index -> list of (category, group_id); normally exactly one entry each."""
    places = {i: [] for i in range(len(s.table["header"]))}
    for g in s.groups:
        it = d["groups"].get(g["group_id"])
        for i in g["indices"]:
            if i not in places:
                continue
            if it is None:
                continue
            if it["role"] == "ignore" or not it.get("keep", True):
                cat = "excluded"
            elif it["role"] == UNRESOLVED:
                cat = "unresolved"
            else:
                cat = _ROLE_CAT.get(it["role"], "unaccounted")
            places[i].append((cat, g["group_id"]))
    return places


def column_ledger(s, d):
    """{file_role: {category: n, ..., total, unresolved, unaccounted, problems: [...]}}."""
    out = {}
    places = _main_places(s, d)
    led = {c: 0 for c in CATEGORIES}
    led.update(unresolved=0, unaccounted=0, total=len(places))
    problems = []
    for i, where in places.items():
        name = s.cols.labels[i]
        if len(where) == 1:
            led[where[0][0]] = led.get(where[0][0], 0) + 1
            if where[0][0] == "unaccounted":
                problems.append(f"main: '{name}' has a role that maps to no output")
        elif not where:
            led["unaccounted"] += 1
            problems.append(f"main: '{name}' is in no category")
        else:
            led["unaccounted"] += 1
            problems.append(f"main: '{name}' is in {len(where)} places ({', '.join(g for _, g in where)})")
    led["problems"] = problems
    out["main"] = led
    meta = d.get("metadata")
    if meta and not meta.get("skipped") and meta.get("columns"):
        ml = {c: 0 for c in CATEGORIES}
        ml.update(unresolved=0, unaccounted=0, total=meta.get("n_columns") or len(meta["columns"]), problems=[])
        seen = {}
        for c in meta["columns"]:
            seen[c["column"]] = seen.get(c["column"], 0) + 1
            if c["role"] == "sample_id":
                ml["sample_id"] += 1
            elif not c.get("keep", True) or c["role"] == "ignore":
                ml["excluded"] += 1
            elif c["role"] == "sample_metadata":
                ml["sample_metadata"] += 1
            else:
                ml["unaccounted"] += 1
                ml["problems"].append(f"metadata: '{c['column']}' has role '{c['role']}'")
        missing = ml["total"] - len(meta["columns"])
        if missing > 0:
            ml["unaccounted"] += missing
            ml["problems"].append(f"metadata: {missing} column(s) of the file are in no category")
        out["metadata"] = ml
    return out


def ledger_text(led):
    parts = [f"{led[c]} {c.replace('_', ' ')}" for c in CATEGORIES if led.get(c)]
    extra = [f"{led[k]} {k}" for k in ("unresolved", "unaccounted") if led.get(k)]
    return f"{led['total']} = " + " + ".join(parts + extra or ["0"])


def ledger_problems(ledger):
    out = []
    for f, led in ledger.items():
        out.extend(led["problems"])
        counted = sum(led.get(c, 0) for c in CATEGORIES) + led.get("unresolved", 0) + led.get("unaccounted", 0)
        if counted != led["total"] and not led["problems"]:
            out.append(f"{f}: {counted} columns counted but the file has {led['total']}")
    return out


def excluded_columns(s, d):
    """[{column, file, reason, by, at}] for every column left out of the outputs."""
    out = []
    for gid, it in d["groups"].items():
        if not (it["role"] == "ignore" or not it.get("keep", True)):
            continue
        rec = it.get("excluded") or {}
        reason = rec.get("reason") or ("proposed as 'ignore' and confirmed with its step" if it["role"] == "ignore"
                                       else "left out of the outputs")
        for c in s.groups_by_id[gid]["columns"]:
            out.append({"column": c, "file": "main", "reason": reason, "by": rec.get("by") or "user",
                        "at": rec.get("at"), "role": it["role"], "label": it.get("label") or "", "group_id": gid})
    meta = d.get("metadata")
    if meta and not meta.get("skipped"):
        for c in meta.get("columns", []):
            if c["role"] != "sample_id" and (not c.get("keep", True) or c["role"] == "ignore"):
                rec = c.get("excluded") or {}
                out.append({"column": c["column"], "file": "metadata", "reason": rec.get("reason") or "left out of the outputs",
                            "by": rec.get("by") or "user", "at": rec.get("at"), "role": c["role"],
                            "label": c.get("label") or ""})
    return out


def files_entries(s, d):
    out = [{"file_role": "main", "name": s.filename, "sha256": s.sha, "n_rows": len(s.table["rows"]),
            "n_columns": len(s.table["header"]), "source_columns": list(s.table["header"]),
            "source_columns_sha256": columns_sha256(s.table["header"]),
            "parse_report": s.table["parse_report"]}]
    meta = d.get("metadata")
    if meta and not meta.get("skipped") and meta.get("columns"):
        rep = meta.get("report") or {}
        header = [c["column"] for c in meta["columns"]]
        out.append({"file_role": "metadata", "name": meta["filename"], "sha256": meta.get("sha256"),
                    "n_rows": meta.get("n_rows"), "n_columns": meta.get("n_columns") or len(header),
                    "source_columns": meta.get("source_columns") or header,
                    "source_columns_sha256": columns_sha256(meta.get("source_columns") or header),
                    "parse_report": meta.get("parse_report") or {},
                    "join": {"key_column": meta.get("id_column"), "sample_source": meta.get("sample_source") or (
                        "value column headers" if d["layout"]["value"] == "samples_in_columns" else "sample ID column"),
                             "matched": rep.get("n_matched", 0), "only_in_data": rep.get("only_in_data", []),
                             "only_in_metadata": rep.get("only_in_metadata", []),
                             "near_miss_suggestions": rep.get("near_misses", []),
                             "accepted_near_misses": meta.get("accepted_near_misses", [])}})
    return out
