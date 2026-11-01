"""The intake (Build Request v4, section 8.1): four answers that unlock the study-dependent steps.

    contrast_variable     the sample variable your question compares (a column), or "none" (exploratory)
    protected_variables   variables whose signal must not be removed (default: the contrast)
    nuisance_variables    variables that may be corrected (default: the batch candidates)
    question_type         global_untargeted | targeted_features
    downstream_family     differential_abundance | clustering_exploration | prediction | integration | unsure

An empty field is not an error: the steps that need it become "Insufficient metadata" with the
missing field named. Every change is kept in the history with its time.
"""

from __future__ import annotations

import datetime as _dt

FIELDS = {
    "contrast_variable": "The sample variable your question compares (or 'none' for an exploratory study).",
    "protected_variables": "Variables whose signal must not be removed (default: the contrast).",
    "nuisance_variables": "Variables that may be corrected (default: the batch candidates).",
    "question_type": "global_untargeted or targeted_features.",
    "downstream_family": "differential_abundance, clustering_exploration, prediction, integration or unsure.",
}
QUESTION_TYPES = ("global_untargeted", "targeted_features")
DOWNSTREAM = ("differential_abundance", "clustering_exploration", "prediction", "integration", "unsure")


class IntakeError(Exception):
    pass


class Intake:
    def __init__(self):
        self.values: dict = {}
        self.history: list[dict] = []

    def set(self, columns: list[str], **fields) -> dict:
        out = dict(self.values)
        unknown = sorted(set(fields) - set(FIELDS))
        if unknown:
            raise IntakeError(f"Unknown intake field(s): {', '.join(unknown)}. Fields: {', '.join(FIELDS)}")
        cv = fields.get("contrast_variable")
        if cv not in (None, ""):
            if cv != "none" and cv not in columns:
                raise IntakeError(f"contrast_variable '{cv}' is not a sample variable ({', '.join(columns)}) "
                                  "and is not 'none'.")
            out["contrast_variable"] = cv
        for k in ("protected_variables", "nuisance_variables"):
            v = fields.get(k)
            if v in (None, ""):
                continue
            v = [x.strip() for x in v.split(",")] if isinstance(v, str) else list(v)
            bad = [x for x in v if x not in columns]
            if bad:
                raise IntakeError(f"{k}: unknown variable(s) {', '.join(bad)}.")
            out[k] = v
        if fields.get("question_type") not in (None, ""):
            if fields["question_type"] not in QUESTION_TYPES:
                raise IntakeError(f"question_type must be one of {', '.join(QUESTION_TYPES)}.")
            out["question_type"] = fields["question_type"]
        if fields.get("downstream_family") not in (None, ""):
            if fields["downstream_family"] not in DOWNSTREAM:
                raise IntakeError(f"downstream_family must be one of {', '.join(DOWNSTREAM)}.")
            out["downstream_family"] = fields["downstream_family"]
        self.values = out
        self.history.append({"version": len(self.history) + 1, "intake": dict(out), "by": "user",
                             "at": _dt.datetime.now().isoformat(timespec="seconds")})
        return out

    def resolved(self, batch_candidates=()) -> dict:
        """The intake with its defaults (protected = the contrast; nuisance = the batch candidates)."""
        i = dict(self.values)
        cv = i.get("contrast_variable")
        if "protected_variables" not in i and cv and cv != "none":
            i["protected_variables"], i["protected_default"] = [cv], True
        if "nuisance_variables" not in i and batch_candidates:
            i["nuisance_variables"], i["nuisance_default"] = list(batch_candidates), True
        return i
