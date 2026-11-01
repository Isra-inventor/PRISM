"""prism.Study: the audit, the intake, the plan, approvals, data states and verification for one
confirmed schema. This is what the old website's "session" did after Step 0 (Build Request v4).

    st = prism.Study(s, overrides=f"{OUT}/soma/overrides.json", folder=f"{OUT}/soma/work")
    st.audit()                                   # C0: the full battery on S0, and S0 itself
    st.questions()                               # what only you can answer (processing state, floors)
    st.set_intake(contrast_variable="Group", downstream_family="differential_abundance")
    plan = st.build_plan()                       # every step with its status and candidates
    st.approve("A1B1:G5")                        # run the first candidate: a new state, then the checkpoint battery
    st.reject("A1B1:S1", reason="...")
    st.verify()                                  # C2: before / after with the same instrument
    st.replay()                                  # re-apply every approved op from S0: same state id
    st.save()                                    # plan.json, decisions.jsonl, intake.json, overrides.json, ...

Nothing changes the data without your approval, and every approval and rejection is recorded.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pandas as pd

from .. import executor
from ..audit import Audit
from ..audit import verification
from ..executor import state as state_mod
from . import engine
from . import facts as facts_mod
from .intake import Intake

C1_BATTERY = ("integrity", "scale", "distribution", "quality", "missingness", "missingness_profile",
              "batch_structure", "repeated_measures")
CHECKPOINT_OF_STEP = {"G3": "C1", "G5": "C1", "G6": "C1", "S1": "C1", "G7": "C1", "S2": "C2", "S3": "C2", "S4": "C2"}


def _now():
    return _dt.datetime.now().isoformat(timespec="seconds")


class Plan:
    def __init__(self, d: dict):
        self.d = d

    @property
    def entries(self) -> list[dict]:
        return self.d["entries"]

    def table(self) -> pd.DataFrame:
        rows = []
        for e in self.entries:
            if e.get("checkpoint"):
                rows.append({"entry": e["entry_id"], "step": e["checkpoint"], "status": "checkpoint",
                             "candidates": "", "because": e["text"]})
                continue
            rows.append({"entry": e["entry_id"], "step": f"{e['step']} {e['step_name']}", "status": e["status"],
                         "candidates": " | ".join(f"[{i}] {c['label']} ({c['status']})"
                                                  for i, c in enumerate(e["candidates"])),
                         "because": " ".join(e["because"])})
        return pd.DataFrame(rows)

    def entry(self, entry_id: str) -> dict:
        for e in self.entries:
            if e["entry_id"] == entry_id:
                return e
        raise engine.PlanError(f"Unknown plan entry '{entry_id}'. Entries: {[e['entry_id'] for e in self.entries]}")

    def _repr_html_(self) -> str:
        return (f"<p><b>Plan</b> (rules v{self.d['rules_version']}, state {self.d['state_id'][:12]}, checkpoint "
                f"{self.d['checkpoint']}). Only Recommended and Alternative candidates can be approved.</p>"
                + self.table().to_html(index=False))


class Study:

    def __init__(self, schema, overrides=None, folder=None, params: dict | None = None):
        self.schema = schema
        self.folder = Path(folder) if folder else None
        self.root_audit = Audit(schema, overrides=overrides, params=params)
        self.params = self.root_audit.params
        self.intake = Intake()
        self.store = None
        self.audits: dict[str, Audit] = {}
        self.g3: dict[str, dict] = {}
        self.answers: dict[str, str] = {}
        self.plan: Plan | None = None
        self.decisions: list[dict] = []

    # ---------------------------------------------------------------- audit and S0

    def audit(self) -> pd.DataFrame:
        """C0: the full battery on the uploaded data, then S0 on the confirmed scale."""
        summary = self.root_audit.run_all()
        self._make_root()
        return summary

    def _make_root(self):
        if self.store is not None and len(self.store.chain()) > 0:
            return                                    # ops already applied: S0 stays as it is
        root = state_mod.from_schema(self.schema, self.root_audit.effective_scale())
        self.store = state_mod.StateStore(root, self.folder)
        self.audits = {self.store.root_id: self.root_audit}
        self.g3 = {self.store.root_id: self.root_audit.results.get("g3") or self.root_audit.cleaning_candidates()}

    def _need_audit(self):
        if self.store is None:
            raise engine.PlanError("Run study.audit() first.")

    # ---------------------------------------------------------------- your answers

    def questions(self) -> pd.DataFrame:
        self._need_audit()
        qs = []
        for F in self._facts():
            qs += F["questions"]
        return pd.DataFrame(qs)

    def confirm_processing_state(self, item: str, value: str, assay: str | None = None) -> None:
        self.root_audit.confirm_processing_state(item, value, assay)
        self._make_root()                             # the scale of S0 follows the confirmation

    def answer_question(self, kind: str, value: str) -> None:
        if kind != "floor_censored" or value not in ("yes", "no"):
            raise ValueError("answer_question('floor_censored', 'yes' | 'no')")
        self.answers[kind] = value
        self.root_audit.overrides.append({"kind": kind, "value": value, "by": "user", "time": _now()})

    def answer(self, kind: str):
        return self.answers.get(kind)

    def variable_names(self) -> list[str]:
        return list(self.root_audit.variables())

    def set_intake(self, **fields) -> dict:
        return self.intake.set(self.variable_names(), **fields)

    def sample_mapping(self, column: str) -> dict:
        v = self.root_audit.variables().get(column)
        if v is None:
            return {}
        s = v[1].fillna("").astype(str).str.strip()
        return {k: x for k, x in s.items() if x}

    def qc_samples(self) -> list[str]:
        st = self.schema.samples()
        return sorted(st[st["sample_type"].isin(["qc", "pool"])]["sample"])

    # ---------------------------------------------------------------- states and checkpoints

    def checkpoint_of(self, meta: dict) -> str:
        steps = [o["step"] for o in meta.get("applied_ops", [])]
        if any(CHECKPOINT_OF_STEP.get(s_) == "C2" for s_ in steps):
            return "C2"
        return "C1" if steps else "C0"

    def _audit_state(self, sid: str) -> Audit:
        if sid in self.audits:
            return self.audits[sid]
        meta = self.store.meta(sid)
        a = Audit(self.schema, params=self.params, state=self.store.data(sid), state_id=sid,
                  checkpoint=self.checkpoint_of(meta), root=False)
        a.overrides = self.root_audit.overrides
        for name in C1_BATTERY:
            getattr(a, name)()
        self.audits[sid] = a
        self.g3[sid] = a.cleaning_candidates()
        return a

    def findings_at(self, sid: str | None = None) -> dict:
        """The findings that hold at a state: its own battery, then the nearest ancestor's for audits it
        did not run; A13 always from S0 (it describes the uploaded data)."""
        sid = sid or self.store.head
        chain, cur = [], sid
        while cur is not None:
            chain.append(cur)
            cur = self.store.meta(cur)["parent"]
        out = {}
        for s_ in reversed(chain):                # S0 first, the state itself last
            for k, f in self._audit_state(s_).findings.items():
                if k == "A13" and s_ != self.store.root_id:
                    continue
                out[k] = f
        return out

    def _facts(self, sid: str | None = None) -> list[dict]:
        sid = sid or self.store.head
        meta = self.store.meta(sid)
        findings = self.findings_at(sid)
        bnames = [v["name"] for bid in self.schema.block_map()
                  for v in ((findings.get("A7").measures.get(bid) or {}).get("variables") or {}).get("B", [])] \
            if "A7" in findings else []
        intake = self.intake.resolved(sorted(set(bnames)))
        return [facts_mod.block_facts(self, bid, findings, meta, self.checkpoint_of(meta), intake, self.params)
                for bid in self.schema.block_map()]

    # ---------------------------------------------------------------- the plan

    def build_plan(self) -> Plan:
        self._need_audit()
        RULES, ORDER = engine.rules(), engine.order()
        units = self._facts()
        entries, skipped, gates, notes, hand, questions = [], [], [], [], [], []
        for F in units:
            questions += F["questions"]
            e, sk = engine.unit_entries(self, F, RULES, ORDER, self.params)
            entries += e
            skipped += sk
            gates.append({"block": F["block"], "steps": engine.gate(F)})
            notes += engine.notes(F, RULES)
            hand.append({"block": F["block"], "missingness_pattern": F.get("missing.pattern_class"),
                         "imputation_candidates": F.get("missing.imputation_candidates"),
                         "text": "hand-off to the method-selection stage; nothing is imputed"})
        meta = self.store.meta()
        self.plan = Plan({"created_at": _now(), "rules_version": RULES["version"], "order": ORDER["order"],
                          "state_id": meta["state_id"], "checkpoint": self.checkpoint_of(meta),
                          "intake": self.intake.resolved(), "entries": entries, "skipped": skipped,
                          "insufficient_metadata_first": [e["entry_id"] for e in entries
                                                          if e.get("status") == "Insufficient metadata"],
                          "gate": gates, "notes": notes, "hand_off": hand, "questions": questions,
                          "design_constraints": engine.design_constraints(units, self.params),
                          "statuses": list(engine.STATUSES)})
        return self.plan

    def approve(self, entry_id: str, candidate: int = 0, user: str = "you", params: dict | None = None) -> dict:
        """Approve one candidate: run its op on the current state (a new state), then re-run the checkpoint
        battery on it. Rebuild the plan afterwards (study.build_plan())."""
        if self.plan is None:
            raise engine.PlanError("Build the plan first (study.build_plan()).")
        e = self.plan.entry(entry_id)
        if e.get("checkpoint"):
            raise engine.PlanError("A checkpoint marker is not an action.")
        if e["status"] not in engine.APPROVABLE:
            raise engine.PlanError(f"Entry {entry_id} is '{e['status']}': it cannot be approved and run. "
                                   + " ".join(e.get("because", [])))
        if not (0 <= candidate < len(e["candidates"])):
            raise engine.PlanError(f"Entry {entry_id} has {len(e['candidates'])} candidate(s).")
        c = e["candidates"][candidate]
        if c.get("status") not in engine.APPROVABLE:
            raise engine.PlanError(f"Candidate '{c['label']}' is '{c.get('status')}': not approvable.")
        if self.store.head != e["state_id"]:
            raise engine.PlanError("The data changed since this plan was built: rebuild it (study.build_plan()).")
        approval = {"user": user, "at": _now(), "plan_entry": entry_id}
        decision = {"decision": "approved", "entry_id": entry_id, "candidate": candidate, "label": c["label"],
                    "op": c["op"], "approval": approval}
        out = {"decision": decision, "state": None}
        if c["op"]:
            p = dict(c["params"])
            p.update(params or {})
            if c["op"] != "remove_samples":
                p.setdefault("assay", e["assay"])
            try:
                m = executor.apply(self.store, c["op"], p, approval, step=e["step"])
            except executor.ExecutorError as ex:
                decision.update(decision="approved_but_refused", error=str(ex))
                self.decisions.append(decision)
                raise engine.PlanError(str(ex))
            self._audit_state(m["state_id"])
            out["state"] = {"state_id": m["state_id"], "scale": m["scale"], "counts": m["counts"],
                            "checkpoint": self.checkpoint_of(m)}
            decision["state_id"] = m["state_id"]
        else:
            decision["decision"] = "approved_decision_only"
        self.decisions.append(decision)
        return out

    def reject(self, entry_id: str, reason: str = "", user: str = "you") -> None:
        if self.plan is None:
            raise engine.PlanError("Build the plan first.")
        self.plan.entry(entry_id)
        self.decisions.append({"decision": "rejected", "entry_id": entry_id, "reason": reason,
                               "approval": {"user": user, "at": _now(), "plan_entry": entry_id}})

    # ---------------------------------------------------------------- after the plan

    def states(self) -> pd.DataFrame:
        self._need_audit()
        t = self.store.list()
        t["checkpoint"] = [self.checkpoint_of(self.store.meta(m)) for m in self.store._meta]
        return t

    def compare(self, before: str | None = None, after: str | None = None) -> pd.DataFrame:
        """A before / after table of the verification measures for two states (default S0 and HEAD)."""
        a, b = self._audit_state(before or self.store.root_id), self._audit_state(after or self.store.head)
        rows = []
        for bid in self.schema.block_map():
            for r in verification.compare(verification.measures(a, bid), verification.measures(b, bid)):
                rows.append(dict(r, block=bid))
        return pd.DataFrame(rows)

    def verify(self):
        """C2: the same measures at S0, at the last C1 state and now. Checks the plan; never ranks methods."""
        head = self.store.meta()
        if self.checkpoint_of(head) != "C2":
            raise engine.PlanError("Verification runs at C2: approve a batch correction or a filtering step first.")
        before = {"S0": self._audit_state(self.store.root_id)}
        cur = head["parent"]
        while cur is not None:
            if self.checkpoint_of(self.store.meta(cur)) == "C1":
                before["C1"] = self._audit_state(cur)
                break
            cur = self.store.meta(cur)["parent"]
        f = verification.run(self._audit_state(self.store.head), before)
        self.audits[self.store.head].findings["V"] = f
        return f

    def replay(self) -> dict:
        """Re-apply every approved op from S0 in memory; identical is True when the final id is the same."""
        root = state_mod.from_schema(self.schema, self.store.meta(self.store.root_id)["scale"])
        return executor.replay(self.store, root)

    def save(self, folder=None) -> Path:
        folder = Path(folder or self.folder or ".")
        folder.mkdir(parents=True, exist_ok=True)
        from ..audit.bridge import clean
        if self.plan:
            (folder / "plan.json").write_text(json.dumps(clean(self.plan.d), indent=1, default=str))
            (folder / "design_constraints.json").write_text(json.dumps(clean(self.plan.d["design_constraints"]),
                                                                       indent=1, default=str))
        (folder / "decisions.jsonl").write_text("".join(json.dumps(clean(d), default=str) + "\n"
                                                        for d in self.decisions))
        (folder / "intake.json").write_text(json.dumps({"intake": self.intake.values, "history": self.intake.history},
                                                       indent=1))
        (folder / "overrides.json").write_text(json.dumps(self.root_audit.overrides, indent=1))
        return folder
