"""Step 6: which column(s) identify each protein, and which rows are flagged as suspect.

Protein ID. Proposed by the AI (its assays' feature_identity, else the feature_id groups).
Code checks the facts: every row has an ID, and no ID repeats. Several columns make a
composite ID, joined with "|". In a samples-in-rows table the proteins are the value column
names, so there is nothing to choose.

    s.feature_id()            # the proposal and the check (missing, repeated IDs)
    s.confirm_feature_id()    # or s.set_feature_id(["SeqId"])

Flagged rows. Some rows are not ordinary measured proteins: SomaScan controls (Type other than
"Protein"), other species, decoys ("REV__"), contaminants ("CON__", "Potential contaminant" = "+").
The AI proposes which column and which values (or starting text) mark them; you confirm.
Flagged rows are KEPT and labelled in the outputs (flagged, flag_reason); the audit decides later.

    s.flags()                                         # proposed flags with how many rows each marks
    s.values_of("Type")                               # the values of a column, with counts
    s.flag_rows("Type", values=["Hybridization Control Elution"])
    s.flag_rows("feature_id", prefixes=["CON__", "REV__"])   # "feature_id" means the protein ID
    s.unflag("Organism")
    s.confirm_flags()
"""

from __future__ import annotations

import pandas as pd

FEATURE_ID = "feature_id"   # in flag_rows: the protein ID itself


class FeatureSteps:

    def _init_features(self) -> None:
        ids = []
        for a in self.assays:
            for gid in a.get("feature_identity") or []:
                if gid in self.group_map and gid not in ids:
                    ids.append(gid)
        if not ids:
            ids = [g.id for g in self.groups_with("feature_id")]
        self.feature_id_decision = {"columns": [c for gid in ids for c in self.group(gid).columns],
                                    "provenance": "ai_proposed" if ids else "none"}
        # flags: column -> {"values", "prefixes", "provenance"}
        self.flag_decisions: dict[str, dict] = {}
        self.flags_confirmed = False
        for g in self.groups_with("feature_annotation") + self.groups_with("feature_id"):
            if g.marks_rows_as_suspect or g.flagged_values or g.flagged_prefixes:
                for c in g.columns:
                    self.flag_decisions[c] = {"values": list(g.flagged_values), "prefixes": list(g.flagged_prefixes),
                                              "provenance": "ai_proposed"}

    # ---------------------------------------------------------------- protein ID

    def feature_ids(self) -> pd.Series:
        """The protein ID of every row (samples_in_columns, long) or every value column (samples_in_rows)."""
        if self.layout == "samples_in_rows":
            return pd.Series([c for g in self.groups_with("value") for c in g.columns], name="feature")
        cols = self.feature_id_decision["columns"]
        if not cols:
            return pd.Series([str(i + 1) for i in range(self.table.shape[0])], name="feature")
        ids = self.table.data[cols[0]].str.strip()
        for c in cols[1:]:
            ids = ids + "|" + self.table.data[c].str.strip()
        return ids.rename("feature")

    def check_feature_id(self) -> dict:
        """Facts: rows without an ID, IDs that repeat (in a long table repeats are expected)."""
        ids = self.feature_ids()
        from .table import MISSING
        missing = int(ids.isin(MISSING).sum())
        counts = ids.value_counts()
        dup = counts[counts > 1]
        out = {"columns": self.feature_id_decision["columns"] or "(row numbers)", "n": len(ids),
               "n_distinct": int(ids.nunique()), "n_missing": missing}
        if self.layout != "long":
            out["n_repeated"] = int(len(dup))
            out["repeated_examples"] = dup.head(5).to_dict()
        return out

    def feature_id(self) -> dict:
        d = dict(self.feature_id_decision)
        d["check"] = self.check_feature_id()
        return d

    def set_feature_id(self, columns: list[str]) -> None:
        """The column(s) that identify each protein. Group ids (g1) or column names."""
        cols = []
        for c in columns:
            if c in self.group_map:
                cols += self.group(c).columns
            else:
                self.group_of(c)                  # raises if there is no such column
                cols.append(c)
        before = self.feature_id_decision
        prov = "ai_proposed_corrected" if before["provenance"].startswith("ai") and cols != before["columns"] else \
            ("ai_proposed_confirmed" if before["provenance"].startswith("ai") else "user_set")
        self.feature_id_decision = {"columns": cols, "provenance": prov}
        for c in cols:   # a column that identifies proteins has the role feature_id
            g = self.group_of(c)
            if g.role != "feature_id":
                self.set(g.id, role="feature_id")
        self._note("features", "feature ID set", cols)

    def confirm_feature_id(self) -> None:
        if self.feature_id_decision["provenance"] == "ai_proposed":
            self.feature_id_decision["provenance"] = "ai_proposed_confirmed"
            self._note("features", "feature ID confirmed", self.feature_id_decision["columns"])

    # ---------------------------------------------------------------- flagged rows

    def _column_values(self, column: str) -> pd.Series:
        if column == FEATURE_ID:
            return self.feature_ids()
        return self.table.data[column].str.strip()

    def values_of(self, column: str, top: int = 30) -> pd.Series:
        """The values of a column with their counts (to choose which ones flag a row)."""
        return self._column_values(column).value_counts().head(top)

    def flag_mask(self) -> tuple[pd.Series, pd.Series]:
        """Which rows (or proteins) are flagged, and why. Two Series aligned with feature_ids()."""
        ids = self.feature_ids()
        flagged = pd.Series(False, index=range(len(ids)))
        reason = pd.Series("", index=range(len(ids)))
        for col, f in self.flag_decisions.items():
            vals = self._column_values(col).reset_index(drop=True)
            if col != FEATURE_ID and self.layout == "samples_in_rows":
                continue          # a column of a samples-in-rows table describes samples, not proteins
            hit = vals.isin(f["values"])
            for p in f["prefixes"]:
                hit |= vals.str.startswith(p)
            before = reason[hit]
            reason[hit] = before.where(before == "", before + "; ") + col + "=" + vals[hit]
            flagged |= hit
        return flagged, reason

    def flags(self) -> pd.DataFrame:
        rows = []
        for col, f in self.flag_decisions.items():
            vals = self._column_values(col)
            hit = vals.isin(f["values"])
            for p in f["prefixes"]:
                hit |= vals.str.startswith(p)
            rows.append({"column": col, "values": f["values"], "prefixes": f["prefixes"],
                         "n_rows_flagged": int(hit.sum()), "provenance": f["provenance"]})
        return pd.DataFrame(rows, columns=["column", "values", "prefixes", "n_rows_flagged", "provenance"])

    def flag_rows(self, column: str, values: list[str] | None = None, prefixes: list[str] | None = None) -> None:
        if column != FEATURE_ID:
            self.group_of(column)                 # raises if there is no such column
        before = self.flag_decisions.get(column)
        prov = "ai_proposed_corrected" if before and before["provenance"].startswith("ai") else "user_set"
        self.flag_decisions[column] = {"values": list(values or []), "prefixes": list(prefixes or []),
                                       "provenance": prov}
        self._note("features", f"flag rows by {column}", {"values": values, "prefixes": prefixes})

    def unflag(self, column: str) -> None:
        self.flag_decisions.pop(column, None)
        self._note("features", f"no flag from {column}")

    def confirm_flags(self) -> None:
        for f in self.flag_decisions.values():
            if f["provenance"] == "ai_proposed":
                f["provenance"] = "ai_proposed_confirmed"
        self.flags_confirmed = True
        self._note("features", "flags confirmed", list(self.flag_decisions))

    def _todo_features(self) -> list[dict]:
        out = []
        if self.layout != "samples_in_rows":
            d = self.feature_id_decision
            if not d["columns"]:
                out.append({"step": "6 features", "what": "No protein ID column: s.set_feature_id([...])"})
            elif d["provenance"] == "ai_proposed":
                out.append({"step": "6 features", "what": f"Confirm the protein ID {d['columns']}: "
                                                          "s.confirm_feature_id() or s.set_feature_id([...])"})
            chk = self.check_feature_id()
            if chk["n_missing"] or chk.get("n_repeated"):
                out.append({"step": "6 features", "what": f"The protein ID has {chk['n_missing']} missing and "
                                                          f"{chk.get('n_repeated', 0)} repeated value(s): choose "
                                                          "another column or several (composite)"})
        for col, f in self.flag_decisions.items():
            if not f["values"] and not f["prefixes"]:
                out.append({"step": "6 features", "what": f"{col} marks rows as suspect, but which values? "
                                                          f"s.values_of('{col}') then s.flag_rows('{col}', values=[...])"
                                                          f" or s.unflag('{col}')"})
        if not self.flags_confirmed:
            out.append({"step": "6 features", "what": "Confirm the flagged rows (s.flags()): s.confirm_flags()"})
        return out
