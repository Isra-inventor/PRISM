"""Step 11: the audit, on a confirmed schema. It measures; it never changes data and never
recommends (that is the plan, later). Steps follow the registry (prism/registry/steps.yaml):

    G1  integrity and structure          a.integrity()            audit A1
    G2  processing history and scale     a.scale()                audit A2
                                         a.processing_state()     audit A13
                                         a.questions(), a.confirm_processing_state(item, value, assay=...)

    a = prism.Audit(s)                      # s: a Schema with nothing open
    a = prism.Audit(s, overrides=path)      # with confirmations saved earlier (a.save_overrides(path))
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pandas as pd

from . import integrity, processing_state, scale
from .finding import Finding
from .params import defaults


class Audit:

    def __init__(self, schema, overrides=None, params: dict | None = None):
        open_items = schema.todo()
        if len(open_items):
            raise ValueError(f"The schema still has {len(open_items)} open item(s) (s.todo()): the audit needs every "
                             "decision made first.")
        self.schema = schema
        self.params = {**defaults(), **(params or {})}
        self.findings: dict[str, Finding] = {}
        self.results: dict = {}                    # facts one audit hands to another (e.g. the scale class)
        self.overrides: list[dict] = []            # your confirmations, kept apart from schema.json
        if overrides:
            self.overrides = json.loads(Path(overrides).read_text())

    # ---------------------------------------------------------------- G1, G2

    def integrity(self) -> Finding:
        return self._keep(integrity.run(self))

    def scale(self) -> Finding:
        return self._keep(scale.run(self))

    def processing_state(self) -> Finding:
        if "scale" not in self.results:
            self.scale()
        return self._keep(processing_state.run(self))

    def _keep(self, f: Finding) -> Finding:
        self.findings[f.check] = f
        return f

    # ---------------------------------------------------------------- confirmations

    def questions(self) -> pd.DataFrame:
        """What A13 asks you to confirm (a mismatch, or an answer G5 / S1 need)."""
        f = self.findings.get("A13") or self.processing_state()
        return pd.DataFrame(f.measures["questions"])

    def confirmed(self, assay: str, item: str) -> str | None:
        for o in reversed(self.overrides):
            if o["kind"] == "processing_state" and o["item"] == item and o["assay"] == assay:
                return o["value"]
        return None

    def confirm_processing_state(self, item: str, value: str, assay: str | None = None) -> None:
        if item not in processing_state.ITEMS:
            raise KeyError(f"'{item}' is not an item: {processing_state.ITEMS}")
        if value not in ("yes", "no", "unsure"):
            raise ValueError("value is 'yes', 'no' or 'unsure'.")
        assays = [assay] if assay else sorted({g.assay_label or "assay" for g in self.schema.block_map().values()})
        for a in assays:
            self.overrides.append({"kind": "processing_state", "assay": a, "item": item, "value": value,
                                   "time": _dt.datetime.now().isoformat(timespec="seconds"), "by": "user"})
        self.processing_state()

    def save_overrides(self, path) -> None:
        Path(path).write_text(json.dumps(self.overrides, indent=1))

    # ---------------------------------------------------------------- data and display

    def audit_matrix(self, block: str) -> pd.DataFrame:
        """The audit set: study samples, unflagged proteins."""
        s = self.schema
        X = s.matrix(block, numbers=True)
        flagged = set(s.feature_metadata().query("flagged")["feature"])
        study = set(s.samples().query("is_study_sample")["sample"])
        return X.loc[~X.index.isin(flagged), [c for c in X.columns if c in study]]

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([{"step": f.step, "audit": k, "title": f.title, "status": f.status}
                             for k, f in self.findings.items()])

    def __repr__(self) -> str:
        return "\n\n".join(repr(f) for f in self.findings.values()) or "<prism.Audit: no check run yet>"

    def _repr_html_(self) -> str:
        return "".join(f._repr_html_() for f in self.findings.values()) or "<p>No check run yet.</p>"
