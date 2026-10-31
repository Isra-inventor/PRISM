"""Step 9: what was done to the values before they reached PRISM (per assay).

Five questions, answered by you with "yes", "no" or "not_sure". Code adds hints from the numbers
(e.g. negative values usually mean a log scale); a hint raises the question, it never answers it.
Platform exports often include processing already (SomaScan ADAT files are usually normalized
unless they are the raw "unnormalized" export; Olink NPX is log2 and normalized).

    s.history()                                     # questions, answers, hints
    s.answer_history(normalized="yes", log_transformed="no", imputed="no",
                     batch_corrected="not_sure", features_or_samples_removed_before_upload="no")
    s.answer_history(assay="Olink Explore", log_transformed="yes")   # one assay of several
"""

from __future__ import annotations

import pandas as pd

from .vocabulary import HISTORY_QUESTIONS, VOCABULARY

QUESTIONS = dict(HISTORY_QUESTIONS)


class HistorySteps:

    def _init_history(self) -> None:
        self.history_answers: dict[str, dict[str, str]] = {}    # assay -> question -> answer (user_set)

    def _assay_labels(self) -> list[str]:
        return list(dict.fromkeys(g.assay_label or self._assay(g.assay_label)["assay_label"]
                                  for g in self.block_map().values()))

    def history_hints(self, assay: str) -> dict[str, str]:
        """Facts from the numbers that bear on the questions. Hints only."""
        hints = {}
        for bid, g in self.block_map().items():
            if (g.assay_label or "assay") != assay:
                continue
            try:
                x = self.matrix(bid, numbers=True)
            except ValueError:
                continue
            v = x.stack().dropna()
            if v.empty:
                continue
            if (v < 0).any():
                hints["log_transformed"] = f"{bid}: {int((v < 0).sum())} negative values: usually a log scale"
            elif v.max() < 50 and v.median() > 0:
                hints["log_transformed"] = f"{bid}: all values below 50 (median {v.median():.3g}): possibly a log scale"
            else:
                hints["log_transformed"] = f"{bid}: values up to {v.max():.3g}: looks linear (not log)"
            med = x.median()
            if len(med) >= 3 and med.mean() and abs(med.std() / med.mean()) < 0.02:
                hints["normalized"] = f"{bid}: sample medians within 2% of each other: possibly median-normalized"
            if not x.isna().any().any() and not (v == 0).any():
                hints["imputed"] = f"{bid}: no missing values and no zeros: missing values may have been filled in"
        return hints

    def history(self) -> pd.DataFrame:
        rows = []
        for a in self._assay_labels():
            hints = self.history_hints(a)
            for q, text in HISTORY_QUESTIONS:
                rows.append({"assay": a, "question": q, "text": text,
                             "answer": self.history_answers.get(a, {}).get(q, ""), "hint": hints.get(q, "")})
        return pd.DataFrame(rows)

    def answer_history(self, assay: str | None = None, **answers) -> None:
        labels = self._assay_labels()
        targets = [assay] if assay else labels
        for a in targets:
            if a not in labels:
                raise KeyError(f"No assay '{a}'. Assays: {labels}")
        for q, v in answers.items():
            if q not in QUESTIONS:
                raise KeyError(f"'{q}' is not a question: {list(QUESTIONS)}")
            if v not in VOCABULARY["yes_no_unsure"]:
                raise ValueError(f"Answer with one of {VOCABULARY['yes_no_unsure']}")
        for a in targets:
            self.history_answers.setdefault(a, {}).update(answers)
        self._note("history", f"answered for {targets}", answers)

    def _todo_history(self) -> list[dict]:
        out = []
        for a in self._assay_labels():
            missing = [q for q in QUESTIONS if q not in self.history_answers.get(a, {})]
            if missing:
                out.append({"step": "9 history", "what": f"{a}: answer {', '.join(missing)} "
                                                         "(s.history(), then s.answer_history(...))"})
        return out
