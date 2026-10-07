"""Compare two schema.json files (v2.4 §9): which columns appeared or disappeared, and every
change of role, audit kind, keep, label and provenance, plus any difference in files[].

Run from the PRISM folder:
    python scripts/diff_schemas.py old_schema.json new_schema.json
Exit code 0 when nothing differs, 1 otherwise. Works on older schemas (0.3.x) too.
"""

from __future__ import annotations

import json
import sys

FIELDS = ("place", "audit_kind", "keep", "label", "provenance")


def column_map(schema):
    """(file, column) -> {place, audit_kind, keep, label, provenance} from every part of a schema."""
    out = {}

    def put(file, col, **kw):
        out[(file or "main", col)] = {k: kw.get(k) for k in FIELDS}

    for a in schema.get("assays", []):
        for c in (a.get("feature_identity") or {}).get("columns", []):
            put("main", c, place="feature_id", keep=True, provenance=(a.get("feature_identity") or {}).get("provenance"))
        for b in a.get("value_blocks", []):
            for c in b.get("columns", []):
                put("main", c, place=f"value ({a.get('assay_label')})", keep=b.get("keep", True), label=b.get("label"),
                    provenance=b.get("provenance"))
    for x in schema.get("feature_annotations", []):
        if x.get("derived_from"):
            put("derived", x["column"], place="derived annotation", keep=x.get("keep"), label=x.get("label"),
                provenance=x.get("provenance"))
            continue
        put("main", x["column"], place="feature_id" if x.get("is_feature_id") else "annotation", keep=x.get("keep"),
            label=x.get("label"), provenance=x.get("provenance"))
    for x in schema.get("sample_metadata", []):
        file = x.get("file") or ("metadata" if x.get("source") == "metadata_file" else "main")
        put(file, x["column"], place="sample_metadata", audit_kind=x.get("audit_kind"), keep=x.get("keep"),
            label=x.get("label"), provenance=x.get("provenance"))
    sid = (schema.get("sample_id") or {}).get("column")
    if sid:
        put("main", sid, place="sample_id", keep=True)
    for f in schema.get("files", []):
        key = (f.get("join") or {}).get("key_column")
        if f.get("file_role") == "metadata" and key:
            put("metadata", key, place="sample_id (join key)", keep=True)
    for x in schema.get("excluded_columns", []):
        k = (x.get("file") or "main", x["column"])
        prev = out.get(k, {})
        out[k] = dict({f: prev.get(f) for f in FIELDS}, place="excluded", keep=False,
                      excluded_reason=x.get("reason"), excluded_by=x.get("by"))
    return out


def file_map(schema):
    out = {}
    for f in schema.get("files", []):
        out[f.get("file_role")] = f
    if not out and schema.get("source_file"):
        out["main"] = {"file_role": "main", "name": schema.get("source_file"), "sha256": schema.get("file_sha256")}
    return out


def diff(old, new):
    lines = []
    a, b = column_map(old), column_map(new)
    gone = sorted(set(a) - set(b))
    added = sorted(set(b) - set(a))
    if gone:
        lines.append(f"Columns that disappeared ({len(gone)}):")
        lines += [f"  - [{f}] {c}  (was {a[(f, c)]['place']})" for f, c in gone]
    if added:
        lines.append(f"Columns that appeared ({len(added)}):")
        lines += [f"  + [{f}] {c}  ({b[(f, c)]['place']})" for f, c in added]
    changed = []
    for k in sorted(set(a) & set(b)):
        ch = [(fld, a[k].get(fld), b[k].get(fld)) for fld in FIELDS if a[k].get(fld) != b[k].get(fld)]
        if ch:
            changed.append((k, ch))
    if changed:
        lines.append(f"Columns that changed ({len(changed)}):")
        for (f, c), ch in changed:
            lines.append(f"  * [{f}] {c}: " + "; ".join(f"{fld} {o!r} -> {n!r}" for fld, o, n in ch))
    fa, fb = file_map(old), file_map(new)
    for role in sorted(set(fa) | set(fb)):
        x, y = fa.get(role), fb.get(role)
        if x is None or y is None:
            lines.append(f"files[]: the {role} file is {'new' if x is None else 'gone'}: {(y or x).get('name')}")
            continue
        for k in sorted(set(x) | set(y)):
            if k == "parse_report":
                if x.get(k) != y.get(k):
                    lines.append(f"files[{role}]: parse_report differs")
                continue
            if x.get(k) != y.get(k):
                lines.append(f"files[{role}].{k}: {json.dumps(x.get(k))[:120]} -> {json.dumps(y.get(k))[:120]}")
    for k in ("schema_version", "layout"):
        if old.get(k) != new.get(k):
            lines.append(f"{k}: {json.dumps(old.get(k))} -> {json.dumps(new.get(k))}")
    return lines


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    old, new = (json.load(open(p, encoding="utf-8")) for p in argv[1:3])
    lines = diff(old, new)
    print("\n".join(lines) if lines else "No differences in columns, roles, audit kinds, keep, labels, provenance or files[].")
    return 1 if lines else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
