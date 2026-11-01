"""Step 11: the audit. Checks on the data itself, one at a time, on a confirmed schema.

    a = prism.Audit(s)        # s: a Schema with nothing open (s.todo() empty), e.g. from prism.load
    a.integrity()             # 11.1  is the data sound enough to compute anything on?
    a                         # every finding so far

The audit never changes the data and never recommends a method: it describes what the data
show, with numbers, so that step 12 can propose preprocessing that fits. Every check says
which data it ran on and how.
"""

from __future__ import annotations

import pandas as pd

from . import integrity
from .finding import Finding


class Audit:

    def __init__(self, schema):
        open_items = schema.todo()
        if len(open_items):
            raise ValueError(f"The schema still has {len(open_items)} open item(s) (s.todo()): the audit needs every "
                             "decision made first.")
        self.schema = schema
        self.findings: dict[str, Finding] = {}

    def integrity(self) -> Finding:
        self.findings["11.1"] = integrity.run(self)
        return self.findings["11.1"]

    # the data later checks use: study samples, unflagged proteins
    def audit_matrix(self, block: str) -> pd.DataFrame:
        s = self.schema
        X = s.matrix(block, numbers=True)
        flagged = set(s.feature_metadata().query("flagged")["feature"])
        study = set(s.samples().query("is_study_sample")["sample"])
        return X.loc[~X.index.isin(flagged), [c for c in X.columns if c in study]]

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([{"check": k, "title": f.title, "status": f.status,
                              "n_sentences": len(f.sentences)} for k, f in self.findings.items()])

    def __repr__(self) -> str:
        return "\n\n".join(repr(f) for f in self.findings.values()) or "<prism.Audit: no check run yet>"

    def _repr_html_(self) -> str:
        return "".join(f._repr_html_() for f in self.findings.values()) or "<p>No check run yet.</p>"
