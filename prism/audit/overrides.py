"""overrides.json (v3 §6.6): inputs to the next audit run, never edits to results.

    {"overrides": [{"override_id", "kind", ..., "by", "at", "reason"}]}

kinds: sample_role {sample, role: qc|blank|pool|study, dataset?}; exclude_from_audit {sample, dataset?};
batch_variable / design_variable / source_variable {column}; param {key, value}.
"""

from __future__ import annotations

from ..util import now_iso, read_json, sha256_file, write_json

KINDS = ("sample_role", "exclude_from_audit", "batch_variable", "design_variable", "source_variable", "param")
ROLES = ("qc", "blank", "pool", "study")


class OverrideError(Exception):
    pass


def path(st):
    return st.dir / "overrides.json"


def load(st):
    p = path(st)
    return read_json(p).get("overrides", []) if p.exists() else []


def sha(st):
    p = path(st)
    return sha256_file(p) if p.exists() else None


def validate(o):
    k = o.get("kind")
    if k not in KINDS:
        raise OverrideError(f"Unknown override kind '{k}'. Kinds: {', '.join(KINDS)}.")
    if k in ("sample_role", "exclude_from_audit") and not o.get("sample"):
        raise OverrideError(f"'{k}' needs a sample.")
    if k == "sample_role" and o.get("role") not in ROLES:
        raise OverrideError(f"role must be one of {', '.join(ROLES)}.")
    if k in ("batch_variable", "design_variable", "source_variable") and not o.get("column"):
        raise OverrideError(f"'{k}' needs a column.")
    if k == "param":
        from .params import ParamError, resolve
        try:
            resolve(overrides=[o])
        except ParamError as e:
            raise OverrideError(str(e))


def add(st, entry, who="user"):
    entry = {k: v for k, v in dict(entry).items() if v is not None}
    validate(entry)
    items = load(st)
    n = max([int(x["override_id"][1:]) for x in items if str(x.get("override_id", "")).startswith("o")] + [0]) + 1
    entry.update(override_id=f"o{n}", by=who, at=now_iso())
    entry.setdefault("reason", "")
    items.append(entry)
    write_json(path(st), {"overrides": items})
    st.log("override_added", entry)
    return entry


def remove(st, override_id, who="user"):
    items = load(st)
    keep = [x for x in items if x.get("override_id") != override_id]
    if len(keep) == len(items):
        raise OverrideError(f"Unknown override '{override_id}'.")
    write_json(path(st), {"overrides": keep})
    st.log("override_removed", {"override_id": override_id, "by": who})
