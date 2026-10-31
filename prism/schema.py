"""The schema: everything decided about a table, who decided it, and what is still open.

This is the package version of the old website's Step 0 wizard. prism.start(table) runs steps 2-4
(layout hint, digest, AI grouping) and returns a Schema; you then go through the steps below,
in order, confirming or correcting each proposal. s.todo() always says what is still open.

    step 5  groups        s.groups(), s.confirm(...), s.set(...), s.split(...), s.merge(...), s.set_layout(...)
    step 6  features      s.feature_id(), s.flags()                       (features.py)
    step 7  values        s.blocks(), s.samples()                         (values.py)
    step 8  design        s.add_sample_info(...), s.design()              (design.py)
    step 9  history       s.history()                                     (history.py)
    step 10 review, save  s.review(), s.save(folder); prism.load(...)     (outputs.py)

Provenance: every decision carries one of
    computed               worked out by code from facts (counts, matches), nothing to judge
    ai_proposed            proposed by the AI, not looked at yet (still open)
    ai_proposed_confirmed  proposed by the AI, confirmed by you
    ai_proposed_corrected  proposed by the AI, changed by you
    user_set               decided by you, without an AI proposal
and every action is appended to s.log with a time stamp.
"""

from __future__ import annotations

import datetime as _dt

import pandas as pd

from .design import DesignSteps
from .features import FeatureSteps
from .grouping import Group, Grouping, propose_groups
from .history import HistorySteps
from .layout import propose_layout
from .outputs import OutputSteps
from .table import Table
from .values import ValueSteps
from .vocabulary import UNRESOLVED, VOCABULARY

# Fields of a group that you can set with s.set(...). Closed fields are checked against VOCABULARY.
GROUP_FIELDS = ("role", "label", "assay_label", "family", "audit_kind", "marks_rows_as_suspect",
                "flagged_values", "flagged_prefixes", "keep", "zero_means")
ZERO_MEANS = ("not_detected", "zero")
PENDING = ("ai_proposed", "none")


def _was_ai(provenance: str) -> bool:
    return provenance.startswith("ai_proposed")


class Schema(FeatureSteps, ValueSteps, DesignSteps, HistorySteps, OutputSteps):

    def __init__(self, table: Table, grouping: Grouping, ai=None):
        self.table = table
        self.ai = ai                                   # used by the steps that ask the AI; never saved
        self.ai_models = [grouping.model] if grouping.model else []
        self.examples_sent = grouping.sent.get("settings", {}).get("example_values_sent", True)
        self.layout_decision = dict(grouping.layout, provenance="ai_proposed")
        self.group_map: dict[str, Group] = {g.id: g for g in grouping.groups}
        self.assays: list[dict] = [dict(a, provenance="ai_proposed") for a in grouping.assays]
        self.rejected = list(grouping.rejected)
        self.log: list[dict] = []
        self._note("groups", "AI proposal received", grouping.describe())
        self._init_features()
        self._init_values()
        self._init_design()
        self._init_history()

    # ---------------------------------------------------------------- bookkeeping

    def _note(self, step: str, action: str, detail="") -> None:
        self.log.append({"time": _dt.datetime.now().isoformat(timespec="seconds"), "step": step,
                         "action": action, "detail": detail})

    def group(self, gid: str) -> Group:
        if gid not in self.group_map:
            raise KeyError(f"No group '{gid}'. Groups: {', '.join(self.group_map)}")
        return self.group_map[gid]

    def group_of(self, column: str) -> Group:
        for g in self.group_map.values():
            if column in g.columns:
                return g
        raise KeyError(f"No column named '{column}'.")

    def groups_with(self, role: str) -> list[Group]:
        return [g for g in self._ordered() if g.role == role]

    def _ordered(self) -> list[Group]:
        pos = {c: i for i, c in enumerate(self.table.data.columns)}
        return sorted(self.group_map.values(), key=lambda g: min(pos[c] for c in g.columns))

    def _next_id(self) -> str:
        return f"g{max(int(i[1:]) for i in self.group_map) + 1}"

    @property
    def layout(self) -> str | None:
        return self.layout_decision.get("value")

    # ---------------------------------------------------------------- step 5: groups

    def groups(self) -> pd.DataFrame:
        """One row per group: what it is, and who decided."""
        rows = []
        for g in self._ordered():
            shown = ", ".join(g.columns[:3]) + (f" ... (+{len(g.columns) - 3})" if len(g.columns) > 3 else "")
            rows.append({"group": g.id, "n_columns": len(g.columns), "columns": shown, "role": g.role,
                         "audit_kind": g.audit_kind or "", "label": g.label, "assay": g.assay_label or "",
                         "confidence": g.confidence, "provenance": g.provenance})
        return pd.DataFrame(rows)

    def show(self, gid: str) -> dict:
        """Everything about one group: all its columns, the AI's evidence, the computed facts."""
        from .digest import column_facts
        g = self.group(gid)
        out = {k: v for k, v in vars(g).items()}
        out["facts"] = [column_facts(self.table, c) for c in g.columns[:5]]
        return out

    def confirm(self, *gids: str) -> None:
        """Confirm AI proposals as they are."""
        for gid in gids:
            g = self.group(gid)
            if g.role == UNRESOLVED:
                raise ValueError(f"{gid} is unresolved: give it a role with s.set('{gid}', role=...).")
            if g.provenance == "ai_proposed":
                g.provenance = "ai_proposed_confirmed"
        self._note("groups", "confirmed", list(gids))

    def confirm_all(self) -> None:
        """Confirm every AI-proposed group you have not changed. Unresolved groups stay open."""
        ids = [g.id for g in self._ordered() if g.provenance == "ai_proposed" and g.role != UNRESOLVED]
        if ids:
            self.confirm(*ids)
        self.confirm_layout()

    def _check_fields(self, fields: dict) -> None:
        for k, v in fields.items():
            if k not in GROUP_FIELDS:
                raise KeyError(f"'{k}' is not a group field. Fields: {', '.join(GROUP_FIELDS)}")
        if "role" in fields and fields["role"] not in VOCABULARY["column_role"]:
            raise ValueError(f"role must be one of {VOCABULARY['column_role']}")
        if fields.get("audit_kind") is not None and fields["audit_kind"] not in VOCABULARY["audit_kind"]:
            raise ValueError(f"audit_kind must be one of {VOCABULARY['audit_kind']}")
        if fields.get("zero_means") is not None and fields["zero_means"] not in ZERO_MEANS:
            raise ValueError(f"zero_means must be one of {ZERO_MEANS}")

    def _apply(self, g: Group, fields: dict) -> None:
        for k, v in fields.items():
            setattr(g, k, list(v) if k in ("flagged_values", "flagged_prefixes") else v)
        if g.role != "sample_metadata":
            g.audit_kind = None
        elif g.audit_kind is None:
            g.audit_kind = "other"
        if g.role != "feature_annotation":
            g.marks_rows_as_suspect = False
        if g.role not in ("feature_annotation", "feature_id"):
            g.flagged_values, g.flagged_prefixes = [], []
        g.provenance = "ai_proposed_corrected" if _was_ai(g.provenance) else "user_set"

    def set(self, gid: str, **fields) -> None:
        """Change a group, e.g. s.set("g4", role="feature_annotation", label="dissociation constant")."""
        self._check_fields(fields)
        g = self.group(gid)
        self._apply(g, fields)
        self._note("groups", f"set {gid}", fields)

    def split(self, gid: str, columns: list[str], **fields) -> str:
        """Move some columns of a group into a new group (give its role and label). Returns the new id."""
        self._check_fields(fields)
        g = self.group(gid)
        bad = [c for c in columns if c not in g.columns]
        if bad:
            raise KeyError(f"Not in {gid}: {bad}")
        if len(columns) == len(g.columns):
            raise ValueError("That would empty the group; use s.set to change it instead.")
        new = Group(id=self._next_id(), columns=[c for c in g.columns if c in columns], role=UNRESOLVED, label="",
                    origin="split", provenance="none")
        g.columns = [c for c in g.columns if c not in columns]
        if _was_ai(g.provenance):
            g.provenance = "ai_proposed_corrected"
        self.group_map[new.id] = new
        if fields:
            self._apply(new, fields)
        self._note("groups", f"split {gid} -> {new.id}", {"columns": columns, **fields})
        return new.id

    def merge(self, *gids: str, **fields) -> str:
        """Join groups into the first one (its description is kept unless you give fields)."""
        self._check_fields(fields)
        if len(gids) < 2:
            raise ValueError("Name at least two groups.")
        keep = self.group(gids[0])
        pos = {c: i for i, c in enumerate(self.table.data.columns)}
        for gid in gids[1:]:
            keep.columns += self.group(gid).columns
            del self.group_map[gid]
        keep.columns.sort(key=pos.get)
        self._apply(keep, fields)
        self._note("groups", f"merged {', '.join(gids)} -> {gids[0]}", fields)
        return gids[0]

    def set_layout(self, value: str) -> None:
        if value not in VOCABULARY["layout"]:
            raise ValueError(f"layout must be one of {VOCABULARY['layout']}")
        d = self.layout_decision
        d["provenance"] = "ai_proposed_corrected" if value != d.get("value") and _was_ai(d["provenance"]) else \
            ("ai_proposed_confirmed" if _was_ai(d["provenance"]) else "user_set")
        d["value"] = value
        self._note("groups", "layout set", value)

    def confirm_layout(self) -> None:
        if not self.layout:
            raise ValueError("No layout proposed: use s.set_layout(...).")
        if self.layout_decision["provenance"] == "ai_proposed":
            self.layout_decision["provenance"] = "ai_proposed_confirmed"
            self._note("groups", "layout confirmed", self.layout)

    def _todo_groups(self) -> list[dict]:
        out = []
        if self.layout_decision["provenance"] == "ai_proposed" or not self.layout:
            out.append({"step": "5 groups", "what": f"Confirm the layout ({self.layout}): s.confirm_layout() "
                                                    "or s.set_layout(...)"})
        for g in self._ordered():
            if g.role == UNRESOLVED:
                out.append({"step": "5 groups", "what": f"{g.id} ({', '.join(g.columns[:3])}) is unresolved: "
                                                        f"s.set('{g.id}', role=..., label=...)"})
            elif g.provenance == "ai_proposed":
                out.append({"step": "5 groups", "what": f"{g.id} ({g.label}) is not confirmed: s.confirm('{g.id}')"})
        return out

    # ---------------------------------------------------------------- what is open

    def todo(self) -> pd.DataFrame:
        """Everything still open, in step order. Empty means ready to save."""
        items = (self._todo_groups() + self._todo_features() + self._todo_values() + self._todo_design()
                 + self._todo_history())
        return pd.DataFrame(items, columns=["step", "what"])

    def __repr__(self) -> str:
        return f"<prism.Schema {self.table.path.name}: {len(self.group_map)} groups, {len(self.todo())} open item(s)>"

    def _repr_html_(self) -> str:
        todo = self.todo()
        head = (f"<p><b>Schema for {self.table.path.name}</b>: layout {self.layout}, {len(self.group_map)} groups. "
                f"{len(todo)} open item(s){':' if len(todo) else ' - ready to save.'}</p>")
        return head + (todo.head(15).to_html(index=False) if len(todo) else "")


def start(table: Table, ai=None, examples: bool = True) -> Schema:
    """Steps 2-4 in one call: layout hint, digest, AI grouping. Returns the Schema to confirm."""
    if ai is None:
        from .ai import connect
        ai = connect()
    grouping = propose_groups(table, ai=ai, layout=propose_layout(table), examples=examples)
    return Schema(table, grouping, ai=ai)
