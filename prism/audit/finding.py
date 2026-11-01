"""What every audit check returns: a Finding.

A finding describes; it never decides and never recommends (recommendations are step 12).
Its sentences use careful wording ("consistent with", "a plausible sign of"), never
"diagnosed" or "confirmed": an audit sees numbers, not the lab.

    status        ok | note | problem | not_applicable | insufficient_data | insufficient_metadata
                  problem = something that would make later results wrong (fix it in the file
                  or the schema first); note = worth knowing, nothing is wrong by itself
    sentences     the findings in plain words, each with its numbers
    measures      every number computed, for the report and for later steps
    method        how it was computed, and on which data
    feeds         which later decisions use it (step 12)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

ORDER = {"ok": 0, "not_applicable": 0, "insufficient_data": 0, "insufficient_metadata": 0, "partial": 0,
         "note": 1, "problem": 2}


@dataclass
class Finding:
    check: str                     # audit id: A1, A2, A13 ...
    title: str
    step: str = ""                 # registry step: G1, G2 ... (prism/registry/steps.yaml)
    status: str = "ok"
    sentences: list[tuple[str, str]] = field(default_factory=list)    # (status, sentence)
    measures: dict = field(default_factory=dict)
    method: str = ""
    feeds: list[str] = field(default_factory=list)

    def say(self, status: str, sentence: str) -> None:
        """Add a sentence; the finding's status is the worst of its sentences."""
        self.sentences.append((status, sentence))
        neutral = ("not_applicable", "insufficient_data", "insufficient_metadata", "partial")
        if ORDER[status] > ORDER[self.status]:
            self.status = status
        elif all(st in neutral for st, _ in self.sentences):
            self.status = status            # nothing but "could not assess": say so, not "ok"

    # ---- the calls of the audits ported from the v4 build (bridge.py) ----

    @property
    def plot(self) -> dict:
        return self.measures.setdefault("plot_data", {})

    def indicate(self, code: str, evidence=None, **values) -> "Finding":
        """A templated sentence (bridge.TEMPLATES), kept with its evidence in measures['indicators']."""
        from .bridge import TEMPLATES, fmt
        text = TEMPLATES[code].format(**{k: fmt(v) for k, v in values.items()})
        self.measures.setdefault("indicators", []).append(
            {"code": code, "text": text, "evidence": evidence if evidence is not None else values})
        self.say("note", text)
        return self

    def set_status(self, status: str, reason: str | None = None, needs=None) -> "Finding":
        if needs:
            self.measures.setdefault("needs", [])
            self.measures["needs"] += [n for n in needs if n not in self.measures["needs"]]
        if reason:
            self.say(status if status in ORDER else "note", reason)
        return self

    def set_method(self, name: str, n_used=None, transform=None, **params) -> "Finding":
        self.method = name
        self.measures["method"] = {"name": name, "n_used": n_used, "transform": transform, "params": params}
        return self

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.sentences, columns=["status", "finding"])

    def __repr__(self) -> str:
        lines = [f"[{self.step} / {self.check}] {self.title}: {self.status.upper()}"]
        lines += [f"  {s:>7}  {t}" for s, t in self.sentences]
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        colour = {"ok": "#2e7d32", "note": "#b26a00", "problem": "#c62828", "not_applicable": "#666",
                  "insufficient_data": "#666", "insufficient_metadata": "#666", "partial": "#666"}
        items = "".join(f"<li><b style='color:{colour[s]}'>{s}</b> {t}</li>" for s, t in self.sentences)
        return (f"<p><b>{self.step} / {self.check} {self.title}</b>: <b style='color:{colour[self.status]}'>{self.status}</b></p>"
                f"<ul>{items}</ul><p style='color:#666'>Method: {self.method}</p>")
