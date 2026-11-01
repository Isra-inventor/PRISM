"""The plan engine (Build Request v4, sections 8.2 to 8.6). Ported from the v4 build: the conditions,
the metadata gate, the rule evaluation (decide / annotate / scope), the candidates, the joint node
D2 and the typed order are unchanged; the session layer is replaced by prism.Study.

Statuses: Recommended, Alternative, Do not automate, Insufficient metadata (and Not applicable,
for scope reasons only). Only Recommended and Alternative candidates can be approved and run.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import numpy as np
import yaml

from .. import registry
from .joint import resolve_d2

RULES_DIR = Path(__file__).resolve().parent.parent / "rules"
STATUSES = ("Recommended", "Alternative", "Do not automate", "Insufficient metadata", "Not applicable")
APPROVABLE = ("Recommended", "Alternative")
THRESHOLDS = (100, 90, 75, 50, 25)


class PlanError(Exception):
    pass


def rules():
    return yaml.safe_load((RULES_DIR / "rules_v0.yaml").read_text(encoding="utf-8"))


def order():
    return yaml.safe_load((RULES_DIR / "plan_order.yaml").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ conditions and the gate (ported)

def _cond(c, F):
    fact, op, val = c["fact"], c["op"], c.get("value")
    if op == "exists":
        present = F.get(fact) not in (None, "")
        return present == (True if val is None else bool(val))
    x = F.get(fact)
    if x is None or x == "indeterminate":
        return False                      # an indeterminate input never makes a condition hold
    if op == "eq":
        return x == val
    if op == "ne":
        return x != val
    if op == "in":
        return x in val
    if op == "not_in":
        return x not in val
    if op == "is_true":
        return x is True
    if op == "is_false":
        return x is False
    if op == "nonempty":
        return bool(x)
    try:
        if op == "gt":
            return x > val
        if op == "ge":
            return x >= val
        if op == "lt":
            return x < val
        if op == "le":
            return x <= val
    except TypeError:
        return False
    raise PlanError(f"Unknown condition op '{op}'.")


def holds(when, F):
    if not when:
        return True
    if "all" in when:
        return all(_cond(c, F) if "fact" in c else holds(c, F) for c in when["all"])
    if "any" in when:
        return any(_cond(c, F) if "fact" in c else holds(c, F) for c in when["any"])
    return _cond(when, F)


def facts_used(when):
    out = []
    for k in ("all", "any"):
        for c in (when or {}).get(k, []):
            out += [c["fact"]] if "fact" in c else facts_used(c)
    return out


# ------------------------------------------------------------------ the gate (8.2)


def need_met(need, F):
    i = lambda k: F.get(f"intake.{k}")  # noqa: E731
    if need == "data":
        return True
    if need == "technical_metadata":
        return bool(F.get("batch.present") or F.get("run_order.present"))
    if need == "qc_samples":
        return (F.get("qc.n") or 0) > 0
    if need == "run_order":
        return bool(F.get("run_order.present"))
    if need == "batch_variable":
        return bool(F.get("batch.present"))
    if need == "contrast_variable":
        return i("contrast_variable") not in (None, "")
    if need == "protected_variables":
        return bool(i("protected_variables")) or i("contrast_variable") == "none"
    if need == "downstream_family":
        return i("downstream_family") not in (None, "")
    if need == "subject":
        return bool(F.get("a8.subject_column"))
    if need == "confirmed_processing_state":
        return not F.get("processing_state.any_unconfirmed") and F.get("processing_state.log_transformed") in ("yes", "no")
    return False


def gate(F):
    out = []
    for s in registry.steps():
        unmet = [n for n in s["needs"] if not need_met(n, F)]
        out.append({"step": s["id"], "kind": s["kind"], "scope": s["scope"], "needs": s["needs"], "unmet": unmet,
                    "needs_satisfied": not unmet,
                    "text": None if not unmet else f"Insufficient metadata: needs {', '.join(unmet)}"})
    return out


# ------------------------------------------------------------------ parameter templates


class Resolver:
    """Fills $names in rule parameters and texts: $item.<k> (an expanded item), $batch_mapping,
    $run_order_mapping, $qc_samples (from the confirmed sample information), $<fact>."""

    def __init__(self, study, F, item=None):
        self.study, self.F, self.item = study, F, item

    def value(self, name):
        if name.startswith("$item."):
            return (self.item or {}).get(name[6:])
        if name == "$batch_mapping":
            col = self.F.get("batch.primary")
            return self.study.sample_mapping(col) if col else None
        if name == "$run_order_mapping":
            cols = self.F.get("batch.run_order_variables") or []
            return self.study.sample_mapping(cols[0]) if cols else None
        if name == "$qc_samples":
            return self.study.qc_samples()
        if name.startswith("$"):
            return self.F.get(name[1:])
        return name

    def params(self, p):
        return {k: self.value(v) if isinstance(v, str) and v.startswith("$") else copy.deepcopy(v)
                for k, v in (p or {}).items()}

    def text(self, t):
        def rep(m):
            v = self.value(m.group(0))
            if isinstance(v, float):
                return f"{v:.3g}"
            if isinstance(v, list):
                return ", ".join(f"{x:.3g}" if isinstance(x, float) else str(x) for x in v)
            return "n/a" if v is None else str(v)
        return re.sub(r"\$[a-z_][a-z0-9_.]*[a-z0-9_]", rep, t or "")


# ------------------------------------------------------------------ S3 table, S5 constraints


def completeness_table(study, F):
    """For thresholds 100 / 90 / 75 / 50 / 25 %: proteins observed in at least that share of study samples
    overall, within at least one contrast group, and per subject cluster. Reported, not chosen."""
    data = study.store.data(F["state_id"]) if study.store else None
    if data is None:
        return None
    b = data.block(F["block"])
    study_samples = set(study.schema.samples().query("is_study_sample")["sample"])
    cols = [j for j, s_ in enumerate(b.sample_ids) if s_ in study_samples]
    X = b.X[:, cols]
    samples = [b.sample_ids[j] for j in cols]
    keys = b.feature_keys
    contrast = F.get("intake.contrast_variable")
    groups = study.sample_mapping(contrast) if contrast and contrast != "none" else {}
    subj = study.sample_mapping("subject") if F.get("a8.subject_column") else {}
    ok = np.isfinite(X)
    levels = sorted({groups.get(s_) for s_ in samples if groups.get(s_)})
    rows = []
    for t in THRESHOLDS:
        overall = ok.mean(1) * 100 >= t - 1e-9
        within = np.zeros(len(keys), bool)
        for lv in levels:
            ix = [j for j, s_ in enumerate(samples) if groups.get(s_) == lv]
            within |= ok[:, ix].mean(1) * 100 >= t - 1e-9
        row = {"threshold": t, "n_kept_overall": int(overall.sum()),
               "n_kept_within_a_group": int(within.sum()) if levels else None}
        if subj:
            subs = sorted({subj.get(s_) for s_ in samples if subj.get(s_)})
            if subs:
                obs = np.stack([ok[:, [j for j, s_ in enumerate(samples) if subj.get(s_) == u]].any(1) for u in subs], 1)
                row["n_kept_by_subject_cluster"] = int((obs.mean(1) * 100 >= t - 1e-9).sum())
        keep = within if levels else overall
        row["remove_ids"] = [k for k, kk in zip(keys, keep) if not kk]
        rows.append(row)
    return {"rows": rows, "groups": levels, "contrast": contrast, "evidence_level": "convention",
            "n_proteins": len(keys), "note": "thresholds are convention; PRISM reports, you choose"}


def design_constraints(F_units, P):
    out = []
    for F in F_units:
        n, ne = F.get("a5.n"), F.get("a5.n_eff")
        e = {"block": F["block"], "subject_variable": F.get("a8.subject_column"),
             "within_subject_variables": F.get("a8.varies_within_subject", []),
             "constant_within_subject": F.get("a8.constant_within_subject", []),
             "nested": F.get("a8.nested", []), "nesting_in_batch": F.get("nested.text", []),
             "minimum_cell": F.get("a5.smallest_cell"), "n": n, "n_eff": ne,
             "rules": ["batch correction must not use a model that absorbs subject structure silently",
                       "filtering counts observed values per subject cluster when clusters exist"]}
        if n and ne and ne / n < P["neff_ratio_note"]:
            e["note"] = f"the effective sample size ({ne:.1f}) is far below n ({n})"
        out.append(e)
    return out


# ------------------------------------------------------------------ building the plan (ported)

def _entry(F, group, step, rule, status, text, R, item=None, suffix=None):
    eid = f"{F['assay']}:{group}" + (f":{suffix}" if suffix else "")
    reg = registry.step(step)
    return {"entry_id": eid, "dataset": F["dataset"], "assay": F["assay"], "group": group, "step": step,
            "step_name": reg["name"], "kind": reg["kind"], "scope": reg["scope"], "status": status,
            "candidates": [], "because": [R.text(text)] if text else [], "rule_ids": [rule["id"]] if rule else [],
            "evidence_levels": {rule["id"]: rule["evidence_level"]} if rule else {},
            "evidence": list(rule.get("evidence", [])) if rule else [],
            "requires": list(rule.get("requires", [])) if rule else [], "blocked_by": [],
            "preconditions": [], "state_id": F["state_id"]}


def _candidates(rule, R, entry):
    for c in (rule.get("then") or {}).get("candidates", []) or []:
        entry["candidates"].append({"op": c.get("op"), "params": R.params(c.get("params")), "label": R.text(c.get("label")),
                                    "status": c.get("status", entry["status"]), "joint": c.get("joint")})


def _preconditions(step, F):
    out = []
    for p in registry.step(step)["preconditions"]:
        ok = holds({"all": [p]}, F)
        out.append({"fact": p["fact"], "op": p["op"], "value": p.get("value"), "found": F.get(p["fact"]),
                    "met": ok, "text": p["text"]})
    return out


def unit_entries(st, F, RULES, ORDER, P):
    """st: the Study."""
    entries, skipped = [], []
    by_group = {}
    for r in RULES["rules"]:
        by_group.setdefault(r["decision_group"], []).append(r)
    for group in ORDER["order"]:
        if group.startswith("C"):
            entries.append({"entry_id": f"{F['assay']}:{group}", "checkpoint": group,
                            "dataset": F["dataset"], "assay": F["assay"],
                            "text": f"Checkpoint {group}: re-run the {group} audit battery on the new state."})
            continue
        step = ORDER.get("steps_of_groups", {}).get(group, group)
        reg = registry.step(step)
        rs = by_group.get(group, [])
        R = Resolver(st, F)
        scope = next((r for r in rs if r.get("effect") == "scope" and holds(r["when"], F)), None)
        if scope is not None:
            e = _entry(F, group, step, scope, scope["then"]["status"], scope["then"].get("text"), R)
            e["blocked_by"] = [scope["id"]]
            entries.append(e)
            continue
        unmet = [n for n in reg["needs"] if not need_met(n, F)]
        if unmet and reg["kind"] == "action":
            e = _entry(F, group, step, None, "Insufficient metadata", f"Insufficient metadata: needs {', '.join(unmet)}", R)
            e["blocked_by"] = [f"needs:{n}" for n in unmet]
            e["requires"] = unmet
            entries.append(e)
            continue
        decided = []
        for r in rs:
            if r.get("effect") != "decide" or not holds(r["when"], F):
                continue
            if r.get("expand"):
                for item in F.get(r["expand"]) or []:
                    Ri = Resolver(st, F, item)
                    e = _entry(F, group, step, r, r["then"]["status"], r["then"].get("text"), Ri, item,
                               suffix=item.get("class"))
                    _candidates(r, Ri, e)
                    decided.append(e)
            else:
                e = _entry(F, group, step, r, r["then"]["status"], r["then"].get("text"), R)
                _candidates(r, R, e)
                decided.append(e)
            break
        if not decided:
            skipped.append({"dataset": F["dataset"], "assay": F["assay"], "group": group, "step": step,
                            "reason": "no rule applies (nothing to do at this state)"})
            continue
        for e in decided:
            for r in rs:
                if r.get("effect") != "annotate" or not holds(r["when"], F):
                    continue
                t = r["then"]
                if t.get("text"):
                    e["because"].append(R.text(t["text"]))
                for op, stt in (t.get("candidate_status") or {}).items():
                    for c in e["candidates"]:
                        if c["op"] == op:
                            c["status"] = stt
                _candidates(r, R, e)
                e["rule_ids"].append(r["id"])
                e["evidence_levels"][r["id"]] = r["evidence_level"]
                e["evidence"] += [k for k in r.get("evidence", []) if k not in e["evidence"]]
            if step == "S2":
                for c in e["candidates"]:
                    if c.get("joint") == "D2" and c["op"]:
                        d2 = resolve_d2(bool(F.get("missing.any")), c["op"], F.get("missing.share_complete_features"))
                        c["d2"] = d2
                        if d2["status"] == "Do not automate":
                            c["status"] = "Do not automate"
                            e["because"].append(d2["text"].strip())
            if step == "S3":
                e["filtering"] = completeness_table(st, F)
                for row in (e["filtering"] or {}).get("rows", []):
                    if row["remove_ids"]:
                        e["candidates"].append({"op": "remove_features", "label": f"remove features below {row['threshold']}%",
                                                "params": {"ids": row["remove_ids"], "reason": f"S3 completeness below "
                                                                                              f"{row['threshold']}%"},
                                                "status": "Alternative"})
            if step == "S4":
                for s in F.get("outliers.flagged") or []:
                    e["candidates"].append({"op": "remove_samples", "label": f"remove sample {s}",
                                            "params": {"ids": [s], "reason": "A11 flag, approved per sample"},
                                            "status": "Do not automate"})
            e["preconditions"] = _preconditions(step, F)
            unmet_pre = [p for p in e["preconditions"] if not p["met"]]
            if unmet_pre and e["status"] == "Recommended":
                e["status"] = "Alternative"
                e["because"].append("A precondition is not met or not established: " +
                                    "; ".join(p["text"] for p in unmet_pre))
            entries.append(e)
    return entries, skipped


def notes(F, RULES):
    out = []
    for n in RULES.get("notes", []):
        if holds(n["when"], F):
            out.append({"id": n["id"], "dataset": F["dataset"], "assay": F["assay"], "text": n["text"],
                        "evidence_level": n["evidence_level"]})
    return out


