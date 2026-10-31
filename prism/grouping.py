"""Step 4: the AI proposes how the columns group together; code checks the proposal.

Which columns belong together, and what each group is, is proposed by the AI (later: corrected
by you). Code never decides it. Code only checks the answer, as the old website did:

  - a template the AI names is replaced by its columns;
  - every column named must exist, and is claimed once (first claim wins);
  - every column the AI did not mention becomes its own 'unresolved' group, so no column is
    ever silently dropped;
  - a merge is applied only when every group id it names exists;
  - closed fields must hold allowed values (audit_kind only for sample information,
    marks_rows_as_suspect only for feature annotations);
  - everything rejected is kept in .rejected with the reason.

    g = prism.propose_groups(table)     # asks the AI (connects with your key if needed)
    g                                   # one row per group, with role, label and confidence
    g.columns()                         # one row per column
    g.rejected                          # what code refused from the answer, and why
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import pandas as pd

from .digest import digest as make_digest
from .digest import numeric_templates
from .layout import Layout
from .prompts import grouping_prompt, response_schema, system_prompt
from .table import Table
from .vocabulary import UNRESOLVED, VOCABULARY

MAX_DIGEST_CHARS = 300_000   # above this, the file needs to be sent in parts (not built yet)


@dataclass
class Group:
    id: str
    columns: list[str]
    role: str
    label: str
    assay_label: str | None = None
    family: str | None = None
    audit_kind: str | None = None
    marks_rows_as_suspect: bool = False
    confidence: float = 0.0
    evidence: str = ""
    origin: str = "ai_proposed"      # ai_proposed | merged_ai | unmentioned | split | sample_info
    # who decided: ai_proposed (not looked at yet) | ai_proposed_confirmed | ai_proposed_corrected |
    # user_set | none (nobody: a column the AI left out)
    provenance: str = "ai_proposed"
    flagged_values: list[str] = field(default_factory=list)    # feature annotations: values that flag a row
    flagged_prefixes: list[str] = field(default_factory=list)  # ... or text the value starts with
    keep: bool = True                 # value groups: written to the outputs
    zero_means: str | None = None     # value groups: "not_detected" or "zero" (asked when there are zeros)


@dataclass
class Grouping:
    table: Table
    layout: dict
    groups: list[Group]
    assays: list[dict]
    rejected: list[dict] = field(default_factory=list)
    sent: dict = field(default_factory=dict)          # the digest that was sent, exactly
    model: str | None = None

    def summary(self) -> pd.DataFrame:
        """One row per group."""
        rows = []
        for g in self.groups:
            shown = ", ".join(g.columns[:3]) + (f" ... (+{len(g.columns) - 3})" if len(g.columns) > 3 else "")
            rows.append({"group": g.id, "n_columns": len(g.columns), "columns": shown, "role": g.role,
                         "audit_kind": g.audit_kind or "", "label": g.label, "assay": g.assay_label or "",
                         "confidence": g.confidence, "origin": g.origin})
        return pd.DataFrame(rows)

    def columns(self) -> pd.DataFrame:
        """One row per column of the file, in file order, with its group and role."""
        where = {c: g for g in self.groups for c in g.columns}
        return pd.DataFrame([{"column": c, "group": where[c].id, "role": where[c].role, "label": where[c].label}
                             for c in self.table.data.columns])

    def out_of_scope(self) -> list[dict]:
        """Assays the AI marked as outside PRISM's current scope (proteomics), or unsure."""
        return [a for a in self.assays if a.get("in_supported_scope") != "yes"]

    def unresolved(self) -> list[Group]:
        return [g for g in self.groups if g.role == UNRESOLVED]

    def evidence(self) -> pd.DataFrame:
        """Why the AI proposed each group."""
        return pd.DataFrame([{"group": g.id, "label": g.label, "confidence": g.confidence, "evidence": g.evidence}
                             for g in self.groups])

    def to_dict(self) -> dict:
        return {"layout": self.layout, "groups": [asdict(g) for g in self.groups], "assays": self.assays,
                "rejected": self.rejected, "model": self.model}

    def describe(self) -> str:
        n_cols = sum(len(g.columns) for g in self.groups)
        return (f"{len(self.groups)} groups over {n_cols} columns; layout {self.layout.get('value')} "
                f"(confidence {self.layout.get('confidence')}); {len(self.unresolved())} unresolved; "
                f"{len(self.rejected)} item(s) rejected by code"
                + "".join(f"; NOT IN SCOPE ({a.get('in_supported_scope')}): {a.get('assay_label')}, "
                          f"{a.get('scope_reason') or 'no reason given'}" for a in self.out_of_scope()))

    def __repr__(self) -> str:
        return f"<prism.Grouping {self.describe()}>"

    def _repr_html_(self) -> str:
        assays = "".join(f"<li>{a['assay_label']}: {a.get('omics_type')} ({a.get('omics_family')})</li>"
                         for a in self.assays)
        return (f"<p><b>Proposed by the AI ({self.model})</b>: {self.describe()}</p><ul>{assays}</ul>"
                + self.summary().to_html(index=False))


# ---------------------------------------------------------------- checking the answer

def _clean(raw: dict, columns: list[str]) -> tuple[Group, list[dict]]:
    """A Group from one raw AI group, with closed fields checked. Returns the notes for .rejected."""
    notes, gid = [], raw.get("group_id")
    role = raw.get("role")
    if role not in VOCABULARY["column_role"]:
        notes.append({"group_id": gid, "reason": f"Role {role!r} is not an allowed role; set to unresolved."})
        role = UNRESOLVED
    audit_kind = raw.get("audit_kind")
    if role != "sample_metadata":
        audit_kind = None
    elif audit_kind not in VOCABULARY["audit_kind"]:
        notes.append({"group_id": gid, "reason": f"audit_kind {audit_kind!r} is not allowed; set to 'other'."})
        audit_kind = "other"
    suspect = bool(raw.get("marks_rows_as_suspect")) and role == "feature_annotation"
    can_flag = role in ("feature_annotation", "feature_id")
    flagged_values = [str(v) for v in raw.get("flagged_values") or []] if can_flag else []
    flagged_prefixes = [str(v) for v in raw.get("flagged_prefixes") or []] if can_flag else []
    try:
        conf = min(max(float(raw.get("confidence") or 0), 0.0), 1.0)
    except (TypeError, ValueError):
        conf = 0.0
    return Group(id=str(gid), columns=columns, role=role, label=raw.get("label") or "",
                 assay_label=raw.get("assay_label") if role == "value" else None, family=raw.get("family"),
                 audit_kind=audit_kind, marks_rows_as_suspect=suspect, confidence=conf,
                 evidence=raw.get("evidence") or "", flagged_values=flagged_values,
                 flagged_prefixes=flagged_prefixes), notes


def check_answer(table: Table, answer: dict) -> tuple[dict, list[Group], list[dict], list[dict]]:
    """Apply the AI's answer to the table's real columns. Returns layout, groups, assays, rejected."""
    all_columns = list(table.data.columns)
    position = {c: i for i, c in enumerate(all_columns)}
    templates = numeric_templates(table)
    rejected: list[dict] = []

    # 1. collect groups: templates -> columns, each column claimed once, unknown names rejected
    groups: dict[str, Group] = {}
    claimed: dict[str, str] = {}
    for raw in answer.get("groups") or []:
        gid = str(raw.get("group_id") or "")
        if not gid or gid in groups:
            rejected.append({"group_id": gid, "reason": "Missing or repeated group id; the group was rejected."})
            continue
        names = []
        for t in raw.get("templates") or []:
            if t in templates:
                names += templates[t]
            else:
                rejected.append({"group_id": gid, "template": t, "reason": "No such template; ignored."})
        names += raw.get("columns") or []
        cols = []
        for name in names:
            if name not in position:
                rejected.append({"group_id": gid, "column": name, "reason": "No such column; the claim was rejected."})
            elif name in claimed:
                if claimed[name] != gid:
                    rejected.append({"group_id": gid, "column": name,
                                     "reason": f"Already placed in group {claimed[name]}; the second claim was rejected."})
            else:
                claimed[name] = gid
                cols.append(name)
        if not cols:
            rejected.append({"group_id": gid, "reason": "The group names no valid column; it was rejected."})
            continue
        group, notes = _clean(raw, cols)
        groups[gid] = group
        rejected += notes

    # 2. merges, only when every group id exists; the largest group's description is kept
    for m in answer.get("propose_merge") or []:
        ids = list(dict.fromkeys(m.get("group_ids") or []))
        missing = [i for i in ids if i not in groups]
        if len(ids) < 2 or missing:
            rejected.append({"group_ids": ids, "reason": "Merge names groups that do not exist: " + ", ".join(missing)
                             if missing else "A merge needs at least two groups.", "merge_reason": m.get("reason")})
            continue
        keep = max(ids, key=lambda i: len(groups[i].columns))
        for i in ids:
            if i != keep:
                groups[keep].columns += groups.pop(i).columns
        groups[keep].origin = "merged_ai"
        groups[keep].evidence += f" [merged {', '.join(ids)}: {m.get('reason', '')}]"

    # 3. every column nobody mentioned: its own unresolved group
    for c in all_columns:
        if c not in claimed:
            groups[f"_u{position[c]}"] = Group(id="", columns=[c], role=UNRESOLVED, label="",
                                               evidence="Not placed in any group by the AI: choose what this column is.",
                                               origin="unmentioned", provenance="none")

    # 4. file order, renumbered g1, g2, ...; columns in file order inside each group
    ordered = sorted(groups.items(), key=lambda kv: min(position[c] for c in kv[1].columns))
    new_id = {}
    for n, (old, g) in enumerate(ordered, start=1):
        new_id[old] = g.id = f"g{n}"
        g.columns.sort(key=position.get)
    final = [g for _, g in ordered]

    # 5. layout and assays
    layout = dict(answer.get("layout") or {})
    if layout.get("value") not in VOCABULARY["layout"]:
        rejected.append({"layout": layout.get("value"), "reason": "Not an allowed layout; left undecided."})
        layout["value"] = None
    assays = []
    for a in answer.get("assays") or []:
        a = dict(a)
        if a.get("omics_family") not in VOCABULARY["omics_family"]:
            a["omics_family"] = "unknown"
        if a.get("in_supported_scope") not in VOCABULARY["in_supported_scope"]:
            a["in_supported_scope"] = "unsure"
        ids = a.get("feature_identity") or []
        a["feature_identity"] = [new_id[i] for i in ids if i in new_id]
        for i in ids:
            if i not in new_id:
                rejected.append({"assay": a.get("assay_label"), "group_id": i,
                                 "reason": "feature_identity names a group that does not exist; dropped."})
        assays.append(a)
    return layout, final, assays, rejected


# ---------------------------------------------------------------- asking

def propose_groups(table: Table, ai=None, layout: Layout | None = None, examples: bool = True) -> Grouping:
    """Ask the AI how the columns group, and check its answer. ai: leave it out to connect to Gemini."""
    if ai is None:
        from .ai import connect
        ai = connect()
    if layout is None:
        from .layout import propose_layout
        layout = propose_layout(table)
    d = make_digest(table, layout, examples)
    text = json.dumps(d, ensure_ascii=False)
    if len(text) > MAX_DIGEST_CHARS:
        raise ValueError(f"The digest has {len(text):,} characters (limit {MAX_DIGEST_CHARS:,}): this file needs to "
                         "be sent in parts, which is not built yet.")
    answer = ai.ask(system_prompt(), grouping_prompt(text), response_schema())
    lay, groups, assays, rejected = check_answer(table, answer)
    return Grouping(table=table, layout=lay, groups=groups, assays=assays, rejected=rejected, sent=d,
                    model=ai.calls[-1]["model"] if ai.calls else None)
