"""The plan engine and the approval flow (Build Request v4, sections 8.2 to 8.6).

    sessions/<sid>/plan/plan.json         the latest plan: entries in the typed order, the gate, notes, hand-off
    sessions/<sid>/plan/decisions.jsonl   every approval and rejection
    sessions/<sid>/design_constraints.json  S5: the design object S1 to S3 read

The engine evaluates rules_v0.yaml against the session facts (prism/plan/facts.py). Nothing runs
without approval: `approve` is the only path from a plan entry to the executor, and it re-runs the
checkpoint battery on the new state.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import numpy as np
import yaml

from .. import registry
from ..audit import engine as audit_engine, overrides as ov
from ..executor import ExecutorError, apply as exec_apply, replay as exec_replay
from ..executor import state as state_mod
from ..ledger import append_jsonl
from ..util import now_iso, read_json, write_json
from . import facts as facts_mod
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


def plan_dir(st):
    return st.dir / "plan"


# ------------------------------------------------------------------ conditions


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


def session_column(st, did, col):
    from ..session import merge
    merged = merge.compute(st)
    header, rows = merge.unified_table(merged, only=did)
    key = f"sample_id@{did}"
    for h in (f"{col}@{did}", col):
        if h in header:
            return {r[header.index(key)]: (r[header.index(h)] or "").strip() for r in rows}
    return {}


class Resolver:
    def __init__(self, st, F, item=None):
        self.st, self.F, self.item = st, F, item
        self._cache = {}

    def value(self, name):
        if name.startswith("$item."):
            return (self.item or {}).get(name[6:])
        if name == "$batch_mapping":
            col = self.F.get("batch.primary")
            return {k: v for k, v in session_column(self.st, self.F["dataset"], col).items() if v} if col else None
        if name == "$run_order_mapping":
            cols = self.F.get("batch.run_order_variables") or []
            if not cols:
                return None
            return {k: v for k, v in session_column(self.st, self.F["dataset"], cols[0]).items() if v}
        if name == "$qc_samples":
            return sorted(o["sample"] for o in ov.load(self.st) if o.get("kind") == "sample_role"
                          and o.get("role") in ("qc", "pool") and o.get("dataset") in (None, "", self.F["dataset"]))
        if name.startswith("$"):
            return self.F.get(name[1:])
        return name

    def params(self, p):
        out = {}
        for k, v in (p or {}).items():
            out[k] = self.value(v) if isinstance(v, str) and v.startswith("$") else copy.deepcopy(v)
        return out

    def text(self, t):
        def rep(m):
            v = self.value(m.group(0))
            if isinstance(v, float):
                return f"{v:.3g}"
            if isinstance(v, list):
                return ", ".join(f"{x:.3g}" if isinstance(x, float) else str(x) for x in v)
            return "n/a" if v is None else str(v)
        return re.sub(r"\$[a-z_][a-z0-9_.]*[a-z0-9_]", rep, t or "")


# ------------------------------------------------------------------ S3 table, S4 list, S5 constraints


def completeness_table(st, F):
    data, _ = state_mod.load_data(st, F["dataset"], F["state_id"])
    blocks = [b for b in data.blocks if b.assay_id == F["assay"] or F["assay"].startswith(b.assay_id + "_")]
    if not blocks:
        return None
    X = np.vstack([b.X for b in blocks])
    keys = [k for b in blocks for k in b.feature_keys]
    samples = blocks[0].sample_ids
    contrast = F.get("intake.contrast_variable")
    groups = session_column(st, F["dataset"], contrast) if contrast and contrast != "none" else {}
    subj = session_column(st, F["dataset"], F["a8.subject_column"]) if F.get("a8.subject_column") else {}
    ok = np.isfinite(X)
    rows = []
    levels = sorted({groups.get(s) for s in samples if groups.get(s)})
    for t in THRESHOLDS:
        overall = ok.mean(1) * 100 >= t - 1e-9
        within = np.zeros(len(keys), bool)
        for lv in levels:
            ix = [j for j, s in enumerate(samples) if groups.get(s) == lv]
            within |= ok[:, ix].mean(1) * 100 >= t - 1e-9
        row = {"threshold": t, "n_kept_overall": int(overall.sum()),
               "n_kept_within_a_group": int(within.sum()) if levels else None}
        if subj:
            subs = sorted({subj.get(s) for s in samples if subj.get(s)})
            obs = np.array([[ok[:, [j for j, s in enumerate(samples) if subj.get(s) == u]].any(1)] for u in subs])[:, 0, :].T \
                if subs else None
            row["n_kept_by_subject_cluster"] = int((obs.mean(1) * 100 >= t - 1e-9).sum()) if obs is not None else None
        keep = within if levels else overall
        row["remove_ids"] = [k for k, kk in zip(keys, keep) if not kk]
        rows.append(row)
    return {"rows": rows, "groups": levels, "contrast": contrast, "evidence_level": "convention",
            "n_features": len(keys), "note": "thresholds are convention; PRISM reports, the user chooses"}


def design_constraints(F_units, P):
    out = []
    for F in F_units:
        n, ne = F.get("a5.n"), F.get("a5.n_eff")
        e = {"dataset": F["dataset"], "assay": F["assay"], "subject_variable": F.get("a8.subject_column"),
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


# ------------------------------------------------------------------ building the plan


def _entry(F, group, step, rule, status, text, R, item=None, suffix=None):
    eid = f"{F['dataset']}:{F['assay']}:{group}" + (f":{suffix}" if suffix else "")
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
    entries, skipped = [], []
    by_group = {}
    for r in RULES["rules"]:
        by_group.setdefault(r["decision_group"], []).append(r)
    for group in ORDER["order"]:
        if group.startswith("C"):
            entries.append({"entry_id": f"{F['dataset']}:{F['assay']}:{group}", "checkpoint": group,
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


def build(st, who="user", run_audit=True):
    RULES, ORDER = rules(), order()
    col = facts_mod.collect(st, run_audit=run_audit, who="plan")
    P = col["params"]
    entries, skipped, gates, nts, hand = [], [], [], [], []
    for F in col["units"]:
        e, s = unit_entries(st, F, RULES, ORDER, P)
        entries += e
        skipped += s
        gates.append({"dataset": F["dataset"], "assay": F["assay"], "steps": gate(F)})
        nts += notes(F, RULES)
        hand.append({"dataset": F["dataset"], "assay": F["assay"], "missingness_pattern": F.get("missing.pattern_class"),
                     "imputation_candidates": F.get("missing.imputation_candidates"),
                     "text": "hand-off to the method-selection stage; nothing is imputed in this build"})
    insufficient = [e for e in entries if e.get("status") == "Insufficient metadata"]
    plan = {"session_id": st.sid, "created_at": now_iso(), "rules_version": RULES["version"],
            "order": ORDER["order"], "intake": col.get("intake"), "runs": col["runs"], "states": col["states"],
            "insufficient_metadata_first": [e["entry_id"] for e in insufficient],
            "entries": entries, "skipped": skipped, "gate": gates, "notes": nts, "hand_off": hand,
            "statuses": list(STATUSES)}
    dc = design_constraints(col["units"], P)
    write_json(st.dir / "design_constraints.json", {"constraints": dc, "created_at": now_iso()})
    plan["design_constraints"] = dc
    write_json(plan_dir(st) / "plan.json", plan)
    append_jsonl(st.dir / "ledger.jsonl", {"tool": "plan.build", "who": who, "at": now_iso(),
                                           "n_entries": len(entries), "runs": col["runs"]})
    return plan


def load(st):
    p = plan_dir(st) / "plan.json"
    if not p.exists():
        raise PlanError("No plan yet: run `prism plan` first.")
    return read_json(p)


def _find(plan, entry_id):
    e = next((x for x in plan["entries"] if x["entry_id"] == entry_id), None)
    if e is None:
        raise PlanError(f"Unknown plan entry '{entry_id}'.")
    if e.get("checkpoint"):
        raise PlanError("A checkpoint marker is not an action.")
    return e


def approve(st, entry_id, candidate=0, user="user", params=None):
    """Approve one candidate of a plan entry: run the op on the dataset's current state, then re-run
    the checkpoint battery on the new state. -> {"decision", "state", "run_id"}."""
    plan = load(st)
    e = _find(plan, entry_id)
    if e["status"] not in APPROVABLE:
        raise PlanError(f"Entry {entry_id} is '{e['status']}': it cannot be approved and run. "
                        + " ".join(e.get("because", [])))
    if not (0 <= candidate < len(e["candidates"])):
        raise PlanError(f"Entry {entry_id} has {len(e['candidates'])} candidate(s).")
    c = e["candidates"][candidate]
    if c.get("status") not in APPROVABLE:
        raise PlanError(f"Candidate '{c['label']}' is '{c.get('status')}': not approvable.")
    head = state_mod.head(st, e["dataset"])
    if head != e["state_id"]:
        raise PlanError("The dataset changed since this plan was built: rebuild the plan (`prism plan`).")
    approval = {"user": user, "at": now_iso(), "plan_entry": entry_id}
    decision = {"decision": "approved", "entry_id": entry_id, "candidate": candidate, "label": c["label"],
                "op": c["op"], "approval": approval}
    out = {"decision": decision, "state": None, "run_id": None}
    if c["op"]:
        p = copy.deepcopy(c["params"])
        p.update(params or {})
        assay = e["assay"]
        if assay and c["op"] not in ("remove_samples",):
            p.setdefault("assay", assay.split("_B")[0] if "_B" in assay else assay)
        try:
            m = exec_apply(st, e["dataset"], c["op"], p, approval, step=e["step"])
        except ExecutorError as ex:
            decision["decision"] = "approved_but_refused"
            decision["error"] = str(ex)
            _record(st, decision)
            raise PlanError(str(ex))
        out["state"] = {k: m[k] for k in ("state_id", "label", "checkpoint", "scale", "counts")}
        man = audit_engine.run(st, state=m["state_id"], who="plan")
        out["run_id"] = man["run_id"]
        decision["state_id"], decision["run_id"] = m["state_id"], man["run_id"]
    else:
        decision["decision"] = "approved_decision_only"
    _record(st, decision)
    return out


def reject(st, entry_id, user="user", reason=""):
    plan = load(st)
    _find(plan, entry_id)
    d = {"decision": "rejected", "entry_id": entry_id, "by": user, "at": now_iso(), "reason": reason}
    _record(st, d)
    return d


def _record(st, d):
    append_jsonl(plan_dir(st) / "decisions.jsonl", d)
    append_jsonl(st.dir / "ledger.jsonl", dict(d, tool="plan.decision", at=d.get("at") or now_iso()))
    st.log("plan_decision", d)


def decisions(st):
    p = plan_dir(st) / "decisions.jsonl"
    if not p.exists():
        return []
    import json
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def replay(st):
    """Re-apply the approved ops of every dataset from S0; each final state id must be reproduced."""
    out = {}
    for d in st.confirmed():
        out[d["dataset_id"]] = exec_replay(st, d["dataset_id"])
    return {"datasets": out, "identical": all(r["identical"] for r in out.values())}
