"""Step 7: the measured values (blocks and assays) and the samples.

Blocks. Every kept value group is one block: one number per protein and sample, e.g. "RFU" or
"LFQ intensity". Blocks belong to an assay (SomaScan, Olink Explore, a MaxQuant run ...) and get
ids A1B1, A1B2, A2B1 ... Nothing ranks blocks: which one is analysed is decided later.

    s.blocks()                                 # one row per block, with facts (zeros, negatives, missing)
    s.set_block("A1B1", zero_means="not_detected")   # asked whenever a block holds zeros
    s.set_block("A1B2", keep=False)            # leave a block out of the outputs
    s.assays(); s.set_assay("SomaScan", source_software="SomaLogic ADAT export")
    s.matrix("A1B1")                           # proteins x samples, values exactly as in the file
    s.confirm_values()

Samples. Sample names come from the value columns (samples in columns), the sample ID column
(samples in rows) or the sample ID column of a long table. Which samples are QC, blank, pool,
calibrator or reference is proposed by the AI from the sample names, or set by you.

    s.samples()
    s.propose_sample_types()                   # sends the sample NAMES (not values) to the AI
    s.set_sample_type("QC_*", "qc")            # a pattern (* any text, ? one character) or a list of names
    s.confirm_samples()

Long tables (Olink): one row per protein-sample pair. The matrix is built by pivoting the value
column on the protein ID and the sample ID; a protein-sample pair that occurs twice is an error.
"""

from __future__ import annotations

import fnmatch
import json

import pandas as pd

from .layout import _numbers
from .table import MISSING
from .vocabulary import VOCABULARY

SEPARATORS = " _.-:|[]()"


def _common_prefix(names: list[str]) -> str:
    p = names[0]
    for n in names[1:]:
        while not n.startswith(p):
            p = p[:-1]
    # cut back to a separator so "S1", "S12" do not lose a digit
    while p and p[-1] not in SEPARATORS:
        p = p[:-1]
    return p


def _common_suffix(names: list[str]) -> str:
    return _common_prefix([n[::-1] for n in names])[::-1]


def strip_names(columns: list[str]) -> list[str]:
    """Sample names from the columns of one block: the text all columns share at the start and
    the end is removed ("LFQ intensity S1" -> "S1"), if that leaves distinct, non-empty names."""
    if len(columns) < 2:
        return list(columns)
    pre, suf = _common_prefix(columns), _common_suffix(columns)
    out = [c[len(pre):len(c) - len(suf)].strip(SEPARATORS + " ") for c in columns]
    return out if all(out) and len(set(out)) == len(out) else list(columns)


SAMPLE_TYPE_SCHEMA = {"type": "OBJECT", "properties": {
    "rules": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "pattern": {"type": "STRING"}, "sample_type": {"type": "STRING", "enum": VOCABULARY["sample_type"]},
        "evidence": {"type": "STRING"}}, "required": ["pattern", "sample_type", "evidence"]}},
    "column_values": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "value": {"type": "STRING"}, "sample_type": {"type": "STRING", "enum": VOCABULARY["sample_type"]}},
        "required": ["value", "sample_type"]}},
}, "required": ["rules"]}

SAMPLE_TYPE_PROMPT = """You label the samples of a proteomics table for PRISM. You get the sample
names (and, if the table has one, the values of a sample-type column with counts). Return:
- rules: glob patterns over sample names (* = any text, ? = one character) with a sample_type from
  {types}. Only for samples that are NOT study samples (QC, blank/buffer, pool, calibrator,
  reference/bridge); everything matched by no rule is a study sample. Be specific: a pattern must
  not match study samples. evidence: what in the names made you decide.
- column_values: for each value of the sample-type column, its sample_type (empty if no column).
Never guess: if names give no sign of non-study samples, return no rules."""


class ValueSteps:

    def _init_values(self) -> None:
        self.values_confirmed = False
        self.samples_confirmed = False
        self.sample_type_decisions: dict[str, dict] = {}     # sample -> {sample_type, provenance, rule}

    # ---------------------------------------------------------------- blocks and assays

    def _assay(self, label: str | None) -> dict:
        for a in self.assays:
            if a["assay_label"] == label:
                return a
        a = {"assay_label": label or "assay", "omics_type": "unknown", "omics_family": "unknown",
             "in_supported_scope": "unsure", "scope_reason": "", "feature_identity": [], "provenance": "none"}
        self.assays.append(a)
        return a

    def block_map(self) -> dict[str, "object"]:
        """Block id (A1B1 ...) -> its value group. Only kept groups; assays in the AI's order."""
        by_assay: dict[str, list] = {}
        for g in self.groups_with("value"):
            if g.keep:
                by_assay.setdefault(self._assay(g.assay_label)["assay_label"], []).append(g)
        out, n = {}, 0
        for a in self.assays:
            gs = by_assay.get(a["assay_label"])
            if gs:
                n += 1
                for m, g in enumerate(gs, start=1):
                    out[f"A{n}B{m}"] = g
        return out

    def _block(self, ref: str):
        bm = self.block_map()
        if ref in bm:
            return bm[ref]
        g = self.group(ref)
        if g.role != "value":
            raise ValueError(f"{ref} is not a value group.")
        return g

    def sample_names(self, block: str | None = None) -> list[str]:
        """The sample names of a block (or of the first block)."""
        if block is None and not self.block_map():
            raise ValueError("No value block yet: give the measurement columns role 'value' (step 5).")
        g = self._block(block) if block else next(iter(self.block_map().values()))
        if self.layout == "samples_in_columns":
            return strip_names(g.columns) if len(self.block_map()) > 1 else list(g.columns)
        sid = self.groups_with("sample_id")
        if self.layout == "samples_in_rows":
            if sid:
                return self.table.data[sid[0].columns[0]].str.strip().tolist()
            return [str(i + 1) for i in range(self.table.shape[0])]
        if not sid:
            raise ValueError("A long table needs a sample_id column: s.set(gid, role='sample_id').")
        return list(dict.fromkeys(self.table.data[sid[0].columns[0]].str.strip()))

    def matrix(self, block: str, numbers: bool = False) -> pd.DataFrame:
        """Proteins x samples. Values exactly as in the file (text), missing cells empty;
        numbers=True gives floats (missing = NaN). Nothing is filled, scaled or removed."""
        g = self._block(block)
        data = self.table.data
        if self.layout == "samples_in_columns":
            m = data[g.columns].copy()
            m.index, m.columns = self.feature_ids().tolist(), self.sample_names(block)
        elif self.layout == "samples_in_rows":
            m = data[g.columns].T.copy()
            m.index, m.columns = list(g.columns), self.sample_names(block)
        else:
            sid = self.groups_with("sample_id")[0].columns[0]
            long = pd.DataFrame({"feature": self.feature_ids().values, "sample": data[sid].str.strip().values,
                                 "value": data[g.columns[0]].values})
            dup = long.duplicated(["feature", "sample"], keep=False)
            if dup.any():
                raise ValueError(f"{int(dup.sum())} rows repeat a protein-sample pair, e.g. "
                                 f"{long[dup].head(3)[['feature', 'sample']].values.tolist()}: the table may hold "
                                 "several plates or panels; add the column that tells them apart to the protein ID.")
            m = long.pivot(index="feature", columns="sample", values="value")
            m = m.reindex(index=list(dict.fromkeys(long["feature"])), columns=self.sample_names())
        m = m.fillna("").apply(lambda col: col.str.strip()).mask(lambda x: x.isin(MISSING), "")
        if numbers:
            m = m.apply(_numbers)
        m.index.name, m.columns.name = "feature", "sample"
        return m

    def blocks(self) -> pd.DataFrame:
        rows = []
        for bid, g in self.block_map().items():
            try:
                x = self.matrix(bid, numbers=True)
            except ValueError as e:
                rows.append({"block": bid, "group": g.id, "assay": g.assay_label, "label": g.label, "problem": str(e)})
                continue
            v = x.stack().dropna() if hasattr(x, "stack") else x
            rows.append({"block": bid, "group": g.id, "assay": g.assay_label, "label": g.label,
                         "n_proteins": x.shape[0], "n_samples": x.shape[1],
                         "frac_missing": round(float(x.isna().mean().mean()), 4),
                         "n_zero": int((v == 0).sum()), "n_negative": int((v < 0).sum()),
                         "median": float(v.median()) if len(v) else None,
                         "zero_means": g.zero_means or "", "provenance": g.provenance})
        return pd.DataFrame(rows)

    def set_block(self, block: str, **fields) -> None:
        """keep, zero_means ("not_detected" | "zero"), label, assay_label."""
        self.set(self._block(block).id, **fields)

    def assays_table(self) -> pd.DataFrame:
        return pd.DataFrame([{k: a.get(k) for k in ("assay_label", "omics_type", "omics_family", "source_software",
                                                    "in_supported_scope", "scope_reason", "provenance")}
                             for a in self.assays])

    def set_assay(self, label: str, **fields) -> None:
        a = self._assay(label)
        allowed = ("assay_label", "omics_type", "omics_family", "source_software", "in_supported_scope", "scope_reason")
        for k, v in fields.items():
            if k not in allowed:
                raise KeyError(f"'{k}' is not an assay field: {allowed}")
            if k in ("omics_family", "in_supported_scope") and v not in VOCABULARY[k]:
                raise ValueError(f"{k} must be one of {VOCABULARY[k]}")
        if "assay_label" in fields:
            for g in self.groups_with("value"):
                if g.assay_label == label:
                    g.assay_label = fields["assay_label"]
        a.update(fields)
        a["provenance"] = "ai_proposed_corrected" if a["provenance"].startswith("ai") else "user_set"
        self._note("values", f"assay {label} set", fields)

    def confirm_values(self) -> None:
        for a in self.assays:
            if a["provenance"] == "ai_proposed":
                a["provenance"] = "ai_proposed_confirmed"
        self.values_confirmed = True
        self._note("values", "blocks and assays confirmed", list(self.block_map()))

    # ---------------------------------------------------------------- samples

    def all_samples(self) -> list[str]:
        names = []
        for bid in self.block_map():
            names += self.sample_names(bid)
        return list(dict.fromkeys(names))

    def samples(self) -> pd.DataFrame:
        rows = []
        for sname in self.all_samples():
            d = self.sample_type_decisions.get(sname, {"sample_type": "study", "provenance": "computed",
                                                       "rule": "matched no rule: study sample"})
            rows.append({"sample": sname, "sample_type": d["sample_type"],
                         "is_study_sample": d["sample_type"] == "study", "provenance": d["provenance"],
                         "rule": d.get("rule", "")})
        return pd.DataFrame(rows, columns=["sample", "sample_type", "is_study_sample", "provenance", "rule"])

    def _type_column(self) -> pd.Series | None:
        """Values of a sample-type column in the main table, one per sample (samples in rows / long)."""
        cols = [g for g in self.groups_with("sample_metadata") if g.audit_kind == "sample_type"]
        if not cols or self.layout == "samples_in_columns":
            return None
        sid = self.groups_with("sample_id")
        ids = self.sample_names() if self.layout == "samples_in_rows" else None
        col = self.table.data[cols[0].columns[0]].str.strip()
        if self.layout == "samples_in_rows":
            return pd.Series(col.values, index=ids)
        return pd.Series(col.values, index=self.table.data[sid[0].columns[0]].str.strip()).groupby(level=0).first()

    def propose_sample_types(self) -> pd.DataFrame:
        """Ask the AI which samples are not study samples. Sends the sample NAMES only."""
        if self.ai is None:
            from .ai import connect
            self.ai = connect()
        names = self.all_samples()
        print(f"[PRISM] sending {len(names)} sample names (no values) to the AI.")
        tc = self._type_column()
        payload = {"sample_names": names[:3000], "n_samples": len(names),
                   "sample_type_column_values": tc.value_counts().to_dict() if tc is not None else None}
        answer = self.ai.ask(SAMPLE_TYPE_PROMPT.format(types=VOCABULARY["sample_type"]),
                             json.dumps(payload, ensure_ascii=False), SAMPLE_TYPE_SCHEMA)
        self.ai_models.append(self.ai.calls[-1]["model"])
        self.sample_type_decisions = {}
        if tc is not None:
            vmap = {x["value"]: x["sample_type"] for x in answer.get("column_values") or []
                    if x.get("sample_type") in VOCABULARY["sample_type"]}
            for sname, v in tc.items():
                if v in vmap:
                    self.sample_type_decisions[sname] = {"sample_type": vmap[v], "provenance": "ai_proposed",
                                                         "rule": f"sample-type column = {v}"}
        for r in answer.get("rules") or []:
            if r.get("sample_type") not in VOCABULARY["sample_type"]:
                continue
            for sname in fnmatch.filter(names, r["pattern"]):
                self.sample_type_decisions.setdefault(sname, {"sample_type": r["sample_type"], "provenance": "ai_proposed",
                                                              "rule": f"{r['pattern']}: {r.get('evidence', '')}"})
        self._note("values", "AI proposed sample types", answer)
        return self.samples()

    def set_sample_type(self, pattern, sample_type: str) -> int:
        """Set the type of the samples matching a pattern (or a list of names). Returns how many."""
        if sample_type not in VOCABULARY["sample_type"]:
            raise ValueError(f"sample_type must be one of {VOCABULARY['sample_type']}")
        names = self.all_samples()
        hits = [n for n in names if n in pattern] if isinstance(pattern, (list, tuple, set)) \
            else fnmatch.filter(names, pattern)
        if not hits:
            raise ValueError(f"No sample matches {pattern!r}.")
        for n in hits:
            before = self.sample_type_decisions.get(n)
            prov = "ai_proposed_corrected" if before and before["provenance"].startswith("ai") else "user_set"
            self.sample_type_decisions[n] = {"sample_type": sample_type, "provenance": prov, "rule": f"set: {pattern}"}
        self._note("values", f"sample type {sample_type}", {"pattern": pattern, "n": len(hits)})
        return len(hits)

    def confirm_samples(self) -> None:
        for d in self.sample_type_decisions.values():
            if d["provenance"] == "ai_proposed":
                d["provenance"] = "ai_proposed_confirmed"
        self.samples_confirmed = True
        self._note("values", "samples confirmed")

    def _todo_values(self) -> list[dict]:
        out, step = [], "7 values"
        bm = self.block_map()
        if not bm:
            return [{"step": step, "what": "No value group: give the measurement columns role 'value' (step 5)"}]
        if self.layout == "long" and not self.groups_with("sample_id"):
            out.append({"step": step, "what": "A long table needs a sample_id column (step 5)"})
            return out
        sets = {}
        for bid, g in bm.items():
            if self.layout == "long" and len(g.columns) != 1:
                out.append({"step": step, "what": f"{bid}: in a long table each value group is one column"})
                continue
            try:
                x = self.matrix(bid, numbers=True)
            except ValueError as e:
                out.append({"step": step, "what": f"{bid}: {e}"})
                continue
            if g.zero_means is None and (x == 0).any().any():
                out.append({"step": step, "what": f"{bid} ({g.label}) holds zeros: do they mean 'not detected' or a "
                                                  f"real zero? s.set_block('{bid}', zero_means='not_detected' or 'zero')"})
            names = list(x.columns)
            if len(set(names)) != len(names):
                out.append({"step": step, "what": f"{bid}: some sample names repeat"})
            sets[bid] = set(names)
        if len({frozenset(v) for v in sets.values()}) > 1:
            out.append({"step": step, "what": "The blocks do not have the same samples: check s.sample_names(block)"})
        if not self.values_confirmed:
            out.append({"step": step, "what": "Confirm blocks and assays (s.blocks(), s.assays_table()): "
                                              "s.confirm_values()"})
        if not self.samples_confirmed:
            out.append({"step": step, "what": "Sample types: s.propose_sample_types() or s.set_sample_type(...), "
                                              "then s.confirm_samples()"})
        return out
