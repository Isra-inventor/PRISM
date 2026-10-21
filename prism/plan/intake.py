"""The intake (Build Request v4, section 8.1): four answers that unlock the study-dependent steps.

    sessions/<sid>/intake.json  {"version": n, "intake": {...}, "history": [{version, intake, by, at}]}

An empty field is not an error: the dependent steps become `Insufficient metadata` with the missing
field named.
"""

from __future__ import annotations

from ..util import now_iso, read_json, write_json

FIELDS = {
    "contrast_variable": "The column of the unified sample table your question compares (or 'none' for an "
                         "exploratory study).",
    "protected_variables": "Variables whose signal must not be removed (default: the contrast).",
    "nuisance_variables": "Variables that may be corrected (default: the batch candidates).",
    "question_type": "global_untargeted or targeted_features.",
    "downstream_family": "differential_abundance, clustering_exploration, prediction, integration or unsure.",
}
QUESTION_TYPES = ("global_untargeted", "targeted_features")
DOWNSTREAM = ("differential_abundance", "clustering_exploration", "prediction", "integration", "unsure")


class IntakeError(Exception):
    pass


def path(st):
    return st.dir / "intake.json"


def load_doc(st):
    p = path(st)
    return read_json(p) if p.exists() else {"version": 0, "intake": {}, "history": []}


def get(st):
    return dict(load_doc(st)["intake"])


def columns(st):
    from ..session import merge
    merged = merge.compute(st)
    header, _ = merge.unified_table(merged)
    return [h for h in header if not h.startswith(("sample_id@", "in_")) and h not in ("unified_id", "present_in")]


def validate(st, intake):
    out = {}
    cols = columns(st)
    base = {c.split("@")[0] for c in cols}
    cv = intake.get("contrast_variable")
    if cv not in (None, ""):
        if cv != "none" and cv not in cols and cv not in base:
            raise IntakeError(f"contrast_variable '{cv}' is not a column of the unified sample table "
                              f"({', '.join(sorted(base))}) and is not 'none'.")
        out["contrast_variable"] = cv
    for k in ("protected_variables", "nuisance_variables"):
        v = intake.get(k)
        if v in (None, ""):
            continue
        if isinstance(v, str):
            v = [x.strip() for x in v.split(",") if x.strip()]
        bad = [x for x in v if x not in cols and x not in base]
        if bad:
            raise IntakeError(f"{k}: unknown column(s) {', '.join(bad)}.")
        out[k] = list(v)
    qt = intake.get("question_type")
    if qt not in (None, ""):
        if qt not in QUESTION_TYPES:
            raise IntakeError(f"question_type must be one of {', '.join(QUESTION_TYPES)}.")
        out["question_type"] = qt
    df = intake.get("downstream_family")
    if df not in (None, ""):
        if df not in DOWNSTREAM:
            raise IntakeError(f"downstream_family must be one of {', '.join(DOWNSTREAM)}.")
        out["downstream_family"] = df
    unknown = sorted(set(intake) - set(FIELDS))
    if unknown:
        raise IntakeError(f"Unknown intake field(s): {', '.join(unknown)}.")
    return out


def set_intake(st, intake, who="user"):
    clean = validate(st, intake or {})
    doc = load_doc(st)
    doc["version"] += 1
    doc["intake"] = clean
    doc["history"].append({"version": doc["version"], "intake": clean, "by": who, "at": now_iso()})
    write_json(path(st), doc)
    st.log("intake_set", {"version": doc["version"], "by": who, "intake": clean})
    return doc


def resolved(st, batch_candidates=()):
    """The intake with its defaults applied (protected = the contrast; nuisance = the batch candidates)."""
    i = get(st)
    cv = i.get("contrast_variable")
    if "protected_variables" not in i and cv and cv != "none":
        i["protected_variables"] = [cv]
        i["protected_default"] = True
    if "nuisance_variables" not in i and batch_candidates:
        i["nuisance_variables"] = list(batch_candidates)
        i["nuisance_default"] = True
    return i
