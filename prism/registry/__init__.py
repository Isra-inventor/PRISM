"""The step registry and the evidence files (Build Request v4, sections 1 and 8.5).

    steps.yaml        every Global / Study-dependent x Audit / Action step, validated by steps.schema.json
    evidence.yaml     one entry per reference: citation, verified, what_was_verified, used_for
    parameters.yaml   the evidence level of every audit parameter

`check()` is `prism evidence check`: it fails when a step, rule, parameter, threshold or catalog
entry has no evidence level, or names an evidence key that evidence.yaml does not hold.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PRISM = HERE.parent
LEVELS = ("literature_independent", "expert_guidance", "convention", "heuristic", "unverified")
WHAT = ("read_in_full", "abstract_only", "partly_read", "not_read")


class RegistryError(Exception):
    pass


def read_yaml(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ------------------------------------------------------------------ a small JSON Schema validator
# The subset used by PRISM's schema files: type, enum, required, properties, additionalProperties,
# items, minItems, pattern. (jsonschema is not a dependency.)

_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def _is_type(v, t):
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    return isinstance(v, _TYPES[t])


def validate(obj, schema, path="$"):
    """-> list of error strings (empty when valid)."""
    import re
    errs = []
    t = schema.get("type")
    if t is not None:
        ts = t if isinstance(t, list) else [t]
        if not any(_is_type(obj, x) for x in ts):
            return [f"{path}: expected {t}, got {type(obj).__name__}"]
    if "enum" in schema and obj not in schema["enum"]:
        errs.append(f"{path}: {obj!r} is not one of {schema['enum']}")
    if "pattern" in schema and isinstance(obj, str) and not re.search(schema["pattern"], obj):
        errs.append(f"{path}: {obj!r} does not match {schema['pattern']}")
    if isinstance(obj, dict):
        for k in schema.get("required", []):
            if k not in obj:
                errs.append(f"{path}: missing '{k}'")
        props = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for k, v in obj.items():
            if k in props:
                errs += validate(v, props[k], f"{path}.{k}")
            elif extra is False:
                errs.append(f"{path}: unexpected '{k}'")
            elif isinstance(extra, dict):
                errs += validate(v, extra, f"{path}.{k}")
    if isinstance(obj, list):
        if len(obj) < schema.get("minItems", 0):
            errs.append(f"{path}: fewer than {schema['minItems']} items")
        if "items" in schema:
            for i, v in enumerate(obj):
                errs += validate(v, schema["items"], f"{path}[{i}]")
    return errs


# ------------------------------------------------------------------ loading

_cache = {}


def registry():
    if "steps" not in _cache:
        doc = read_yaml(HERE / "steps.yaml")
        schema = json.loads((HERE / "steps.schema.json").read_text(encoding="utf-8"))
        errs = validate(doc, schema)
        ids = [s["id"] for s in doc.get("steps", [])] if isinstance(doc, dict) else []
        if len(ids) != len(set(ids)):
            errs.append("steps.yaml: duplicate step ids")
        if errs:
            raise RegistryError("steps.yaml is invalid:\n  " + "\n  ".join(errs))
        _cache["steps"] = doc
    return _cache["steps"]


def steps():
    return registry()["steps"]


def step(step_id):
    for s in steps():
        if s["id"] == step_id:
            return s
    raise RegistryError(f"Unknown step '{step_id}'.")


def step_ids():
    return [s["id"] for s in steps()]


def evidence():
    if "evidence" not in _cache:
        _cache["evidence"] = read_yaml(HERE / "evidence.yaml")["references"]
    return _cache["evidence"]


def parameters():
    return read_yaml(HERE / "parameters.yaml")["parameters"]


def grid():
    """The 2x2 (kind x scope) of steps, for the report panel and Figure 1."""
    out = {}
    for s in steps():
        kind = "action" if s["kind"] in ("action", "constraint") else "audit"
        out.setdefault(f"{s['scope']}|{kind}", []).append(
            {k: s[k] for k in ("id", "name", "kind", "implemented", "evidence_level", "needs")})
    return out


# ------------------------------------------------------------------ prism evidence check

# Extra data files whose entries must carry an evidence level. Each: (path relative to prism/,
# the key holding the list or mapping of entries). Files that do not exist yet are skipped.
EVIDENCE_FILES = [
    ("rules/rules_v0.yaml", "rules"),
    ("audit/imputation_catalog.yaml", "families"),
    ("audit/platform_notes.yaml", "notes"),
    ("audit/audit_params_v4.yaml", "parameters"),
    ("phase2/methods_registry.yaml", "methods"),
]


def _entries(doc, key):
    items = doc.get(key) if isinstance(doc, dict) else None
    if isinstance(items, dict):
        return [(k, v) for k, v in items.items()]
    if isinstance(items, list):
        return [(e.get("id", i) if isinstance(e, dict) else i, e) for i, e in enumerate(items)]
    return []


def _check_entry(where, e, keys, problems):
    if not isinstance(e, dict):
        problems.append(f"{where}: not a mapping")
        return
    lvl = e.get("evidence_level")
    if lvl is None:
        problems.append(f"{where}: no evidence_level")
    elif lvl not in LEVELS:
        problems.append(f"{where}: evidence_level '{lvl}' is not one of {', '.join(LEVELS)}")
    for k in e.get("evidence") or []:
        if k not in keys:
            problems.append(f"{where}: evidence key '{k}' is not in evidence.yaml")


def check():
    """-> {"ok": bool, "problems": [...], "counted": {...}}."""
    problems, counted = [], {}
    try:
        reg = registry()
    except RegistryError as e:
        return {"ok": False, "problems": [str(e)], "counted": {}}
    refs = evidence()
    for k, r in refs.items():
        for f in ("citation", "verified", "what_was_verified", "used_for"):
            if f not in r:
                problems.append(f"evidence.yaml {k}: missing '{f}'")
        if r.get("what_was_verified") not in WHAT:
            problems.append(f"evidence.yaml {k}: what_was_verified must be one of {', '.join(WHAT)}")
        if r.get("verified") is True and r.get("what_was_verified") == "not_read":
            problems.append(f"evidence.yaml {k}: verified but not_read")
    keys = set(refs)
    for s in reg["steps"]:
        _check_entry(f"steps.yaml {s['id']}", s, keys, problems)
    for c in reg.get("candidates", []):
        _check_entry(f"steps.yaml candidate {c['id']}", c, keys, problems)
        if c["step"] not in step_ids():
            problems.append(f"steps.yaml candidate {c['id']}: unknown step '{c['step']}'")
    counted["steps"] = len(reg["steps"])
    from ..audit import params as params_mod
    pmeta = parameters()
    for name in params_mod.defaults():
        if name not in pmeta:
            problems.append(f"parameters.yaml: audit parameter '{name}' has no evidence level")
        else:
            _check_entry(f"parameters.yaml {name}", pmeta[name], keys, problems)
    counted["parameters"] = len(pmeta)
    for rel, key in EVIDENCE_FILES:
        p = PRISM / rel
        if not p.exists():
            continue
        doc = read_yaml(p)
        ents = _entries(doc, key)
        counted[rel] = len(ents)
        for eid, e in ents:
            _check_entry(f"{rel} {eid}", e, keys, problems)
            if isinstance(e, dict):
                for st in _step_refs(e):
                    if st not in step_ids():
                        problems.append(f"{rel} {eid}: unknown step '{st}'")
                for th in e.get("thresholds") or []:
                    _check_entry(f"{rel} {eid} threshold {th.get('name', '?')}", th, keys, problems)
    return {"ok": not problems, "problems": problems, "counted": counted}


def _step_refs(e):
    out = []
    for k in ("step", "steps"):
        v = e.get(k)
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, list):
            out += [x for x in v if isinstance(x, str)]
    return out
