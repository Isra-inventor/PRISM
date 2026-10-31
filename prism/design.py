"""Step 8: sample information and study design.

Sample information can be in the main table (samples in rows, or a long table) or in a separate
file (common for SomaScan and MaxQuant exports, where the main table only has sample names).

    s.add_sample_info(f"{DATA}/samples.csv")    # the AI proposes the sample ID column and what each
                                                # column is (subject, time point, batch, group ...)
    s.sample_info()                             # the proposal, and how many sample names match
    s.set_info_column("Visit", audit_kind="timepoint")
    s.confirm_sample_info()
    s.sample_table()                            # one row per sample: type + all sample information

Design: where the subject (the individual a sample came from) and the time point come from:
a column (main table or sample-information file), the sample names (a split rule such as
"P01_T2" -> subject P01, time T2), or nowhere. Code then computes the facts: subjects, samples
per subject, time points, and a label such as "longitudinal, balanced (3 samples per subject)".
The time unit is always asked, never guessed.

    s.design()                                  # proposal + computed facts
    s.name_patterns()                           # ways the sample names split (facts, to choose a rule)
    s.set_design(subject="PatientID", time="Visit", time_unit="weeks")
    s.set_design(derive={"delimiter": "_", "occurrence": "last", "left": "subject", "right": "time"},
                 time_unit="visit_index")
    s.confirm_design()
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from .digest import digest as make_digest
from .table import MISSING, Table, read
from .vocabulary import VOCABULARY

DELIMITERS = ["_", "-", ".", " ", "|", ":"]

INFO_PROMPT = """You describe the columns of a SAMPLE INFORMATION table for PRISM (proteomics). Each
row should describe one sample. You get a digest (column names and computed facts, no rows).
Return one entry per column: is_sample_id (true for the ONE column that names the samples, i.e.
matches the sample names of the measurement table), audit_kind from {kinds}, a short label in your
own words, confidence and evidence quoting the digest. audit_kind guidance: subject_id for the
individual (repeats across repeated measures); timepoint for visit / day / week / time columns;
batch for plate / batch / run date; run_order for injection or run order; group for
arm / cohort / status / treatment / condition (a candidate only); covariate for age, sex, BMI and
other characteristics; sample_type for QC / sample / blank labels; other when nothing fits."""

INFO_SCHEMA = {"type": "OBJECT", "properties": {"columns": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
    "column": {"type": "STRING"}, "is_sample_id": {"type": "BOOLEAN"},
    "audit_kind": {"type": "STRING", "enum": VOCABULARY["audit_kind"]}, "label": {"type": "STRING"},
    "confidence": {"type": "NUMBER"}, "evidence": {"type": "STRING"}},
    "required": ["column", "is_sample_id", "audit_kind", "label", "confidence", "evidence"]}}},
    "required": ["columns"]}


def split_name(name: str, delimiter: str, occurrence: str) -> tuple[str, str] | None:
    """Split a sample name at the first or last delimiter; None if it has none."""
    i = name.find(delimiter) if occurrence == "first" else name.rfind(delimiter)
    if i <= 0 or i >= len(name) - len(delimiter):
        return None
    return name[:i], name[i + len(delimiter):]


def name_patterns(names: list[str], top: int = 8) -> pd.DataFrame:
    """For each delimiter and first / last occurrence: how many names it splits and how many distinct
    values each side has. A subject / time split repeats values on both sides. Facts, not a choice."""
    rows, seen = [], set()
    for dl in DELIMITERS:
        for occ in ("first", "last"):
            parts = [split_name(n, dl, occ) for n in names]
            ok = [p for p in parts if p]
            if not ok:
                continue
            key = (dl, len(ok), len({p[0] for p in ok}), len({p[1] for p in ok}))
            if key in seen:
                continue
            seen.add(key)
            rows.append({"delimiter": dl, "occurrence": occ, "n_split": len(ok), "n_names": len(names),
                         "n_distinct_left": key[2], "n_distinct_right": key[3],
                         "example": f"{ok[0][0]} | {ok[0][1]}"})
    rows.sort(key=lambda r: -r["n_split"])
    return pd.DataFrame(rows[:top])


def _time_sort(v):
    try:
        return (0, float(re.sub(r"[^\d.\-]", "", v) or "x"), v)
    except ValueError:
        return (1, 0.0, v)


def summarize(subject: pd.Series, time: pd.Series | None) -> dict:
    """Computed design facts from one subject (and time) value per study sample."""
    s = subject.dropna()
    s = s[s != ""]
    if s.empty:
        return {"label": "no subject information"}
    counts = s.value_counts()
    dist = Counter(counts.values)
    out = {"n_subjects": int(len(counts)), "n_samples_with_subject": int(len(s)),
           "samples_per_subject": {str(k): int(dist[k]) for k in sorted(dist)},
           "repeated_measures": bool((counts >= 2).any()), "balanced": len(dist) == 1}
    if time is not None:
        t = time.dropna()
        t = t[t != ""]
        out["time_values"] = sorted(set(t), key=_time_sort)[:60]
        pairs = pd.DataFrame({"s": subject, "t": time}).dropna()
        out["n_subject_time_repeats"] = int(pairs.duplicated().sum())   # same subject and time twice
    if not out["repeated_measures"]:
        out["label"] = "cross-sectional (one sample per subject)"
    else:
        kind = "longitudinal" if out.get("time_values") else "repeated measures"
        lo, hi = int(counts.min()), int(counts.max())
        out["label"] = f"{kind}, balanced ({lo} samples per subject)" if lo == hi else \
            f"{kind}, unbalanced ({lo} to {hi} samples per subject)"
    return out


class DesignSteps:

    def _init_design(self) -> None:
        self.info = None                    # the sample-information file and its decisions
        self.design_decision = None         # None until proposed / set
        self.design_confirmed = False

    # ---------------------------------------------------------------- sample information file

    def add_sample_info(self, source, ask_ai: bool = True) -> pd.DataFrame:
        """Add a sample-information file (path or prism.read table). The AI sees its digest only."""
        t = source if isinstance(source, Table) else read(source)
        cols = {c: {"audit_kind": "other", "label": "", "keep": True, "provenance": "none", "evidence": ""}
                for c in t.data.columns}
        id_col, id_prov = None, "none"
        if ask_ai:
            if self.ai is None:
                from .ai import connect
                self.ai = connect()
            d = make_digest(t, examples=self.examples_sent)
            answer = self.ai.ask(INFO_PROMPT.format(kinds=VOCABULARY["audit_kind"]),
                                 "Digest:\n" + json.dumps(d, ensure_ascii=False), INFO_SCHEMA)
            self.ai_models.append(self.ai.calls[-1]["model"])
            for x in answer.get("columns") or []:
                c = x.get("column")
                if c not in cols or cols[c]["provenance"] != "none":
                    self.rejected.append({"sample_info_column": c, "reason": "No such column, or named twice."})
                    continue
                kind = x.get("audit_kind") if x.get("audit_kind") in VOCABULARY["audit_kind"] else "other"
                cols[c] = {"audit_kind": kind, "label": x.get("label", ""), "keep": True, "provenance": "ai_proposed",
                           "evidence": x.get("evidence", ""), "confidence": x.get("confidence")}
                if x.get("is_sample_id") and id_col is None:
                    id_col, id_prov = c, "ai_proposed"
        if id_col is None:   # fall back: the column whose values match the most sample names (a computed fact)
            names = set(self.all_samples())
            best = max(t.data.columns, key=lambda c: len(names & set(t.data[c].str.strip())))
            if names & set(t.data[best].str.strip()):
                id_col, id_prov = best, "computed"
        self.info = {"table": t, "path": str(t.path), "id_column": id_col, "id_provenance": id_prov,
                     "columns": cols, "confirmed": False}
        self._note("design", "sample information added", {"file": t.path.name, "id_column": id_col})
        return self.sample_info()

    def info_match(self) -> dict:
        """How the sample names of the measurements match the sample-information file (facts)."""
        if not self.info or not self.info["id_column"]:
            return {}
        ids = self.info["table"].data[self.info["id_column"]].str.strip()
        names = self.all_samples()
        inn, have = set(names), set(ids)
        return {"n_samples": len(names), "n_matched": len(inn & have),
                "samples_without_info": sorted(inn - have)[:20], "n_samples_without_info": len(inn - have),
                "info_rows_without_sample": len(have - inn), "repeated_ids_in_info": int(ids.duplicated().sum())}

    def sample_info(self) -> pd.DataFrame:
        if not self.info:
            return pd.DataFrame()
        rows = [{"column": c, "is_sample_id": c == self.info["id_column"], **{k: v[k] for k in
                 ("audit_kind", "label", "keep", "provenance")}} for c, v in self.info["columns"].items()]
        print(self.info_match())
        return pd.DataFrame(rows)

    def set_info_column(self, column: str, **fields) -> None:
        """audit_kind, label, keep (False leaves the column out of sample_metadata.csv)."""
        c = self.info["columns"][column]
        if "audit_kind" in fields and fields["audit_kind"] not in VOCABULARY["audit_kind"]:
            raise ValueError(f"audit_kind must be one of {VOCABULARY['audit_kind']}")
        c.update({k: v for k, v in fields.items() if k in ("audit_kind", "label", "keep")})
        c["provenance"] = "ai_proposed_corrected" if c["provenance"].startswith("ai") else "user_set"
        self._note("design", f"sample info column {column}", fields)

    def set_info_id(self, column: str) -> None:
        if column not in self.info["columns"]:
            raise KeyError(column)
        prov = self.info["id_provenance"]
        self.info["id_provenance"] = ("ai_proposed_confirmed" if column == self.info["id_column"] else
                                      "ai_proposed_corrected") if prov.startswith("ai") else "user_set"
        self.info["id_column"] = column
        self._note("design", "sample info ID column", column)

    def confirm_sample_info(self) -> None:
        for c in self.info["columns"].values():
            if c["provenance"] == "ai_proposed":
                c["provenance"] = "ai_proposed_confirmed"
        if self.info["id_provenance"] == "ai_proposed":
            self.info["id_provenance"] = "ai_proposed_confirmed"
        self.info["confirmed"] = True
        self._note("design", "sample information confirmed")

    # ---------------------------------------------------------------- one table of sample facts

    def _main_sample_columns(self) -> pd.DataFrame:
        """Sample information columns of the main table, one row per sample."""
        groups = self.groups_with("sample_metadata")
        cols = [c for g in groups for c in g.columns]
        if not cols or self.layout == "samples_in_columns":
            return pd.DataFrame(index=self.all_samples())
        data = self.table.data
        if self.layout == "samples_in_rows":
            out = data[cols].copy()
            out.index = self.sample_names()
            return out[~out.index.duplicated()]
        sid = self.groups_with("sample_id")[0].columns[0]
        out = data[cols + [sid]].copy()
        out[sid] = out[sid].str.strip()
        return out.groupby(sid, sort=False).first()

    def sample_columns(self) -> dict[str, str]:
        """Every sample information column -> its audit_kind (main table and sample-information file)."""
        out = {c: g.audit_kind for g in self.groups_with("sample_metadata") for c in g.columns}
        if self.info:
            out.update({c: v["audit_kind"] for c, v in self.info["columns"].items()
                        if v["keep"] and c != self.info["id_column"]})
        return out

    def sample_table(self) -> pd.DataFrame:
        """One row per sample: type, the sample information, and subject / time from the design."""
        t = self.samples().set_index("sample")[["sample_type", "is_study_sample"]]
        t = t.join(self._main_sample_columns())
        t = self._join_info(t, kept_only=True)
        st = self._subject_time()
        if st is not None:
            t["subject"], t["time"] = st["subject"], st["time"]
        t.index.name = "sample"
        return t

    def _join_info(self, t: pd.DataFrame, kept_only: bool) -> pd.DataFrame:
        """t (indexed by sample) with the sample-information file's columns joined on its ID column.
        A column name the main table already has gets " (info)" added."""
        if not (self.info and self.info["id_column"]):
            return t
        idc = self.info["id_column"]
        cols = [c for c, v in self.info["columns"].items() if c != idc and (v["keep"] or not kept_only)]
        info = self.info["table"].data[[idc] + cols].copy()
        info[idc] = info[idc].str.strip()
        info = info.drop_duplicates(idc).set_index(idc)
        info = info.rename(columns={c: f"{c} (info)" for c in info.columns if c in t.columns})
        return t.join(info)

    # ---------------------------------------------------------------- design

    def name_patterns(self) -> pd.DataFrame:
        return name_patterns([s for s in self.samples().query("is_study_sample")["sample"]])

    def _proposed_design(self) -> dict:
        """From the confirmed audit kinds: the one subject_id column and the one timepoint column, if any."""
        cols = self.sample_columns()
        subj = [c for c, k in cols.items() if k == "subject_id"]
        time = [c for c, k in cols.items() if k == "timepoint"]
        return {"subject": {"source": "metadata_column", "column": subj[0]} if len(subj) == 1 else {"source": "none"},
                "time": {"source": "metadata_column", "column": time[0]} if len(time) == 1 else {"source": "none"},
                "derivation": None, "time_unit": None, "provenance": "computed"}

    def _subject_time(self) -> pd.DataFrame | None:
        d = self.design_decision or self._proposed_design()
        if d["subject"]["source"] == "none" and d["time"]["source"] == "none":
            return None
        names = self.all_samples()
        out = pd.DataFrame(index=names, columns=["subject", "time"], dtype=object)
        base = None
        for part in ("subject", "time"):
            src = d[part]
            if src["source"] == "metadata_column":
                if base is None:
                    base = self._join_info(pd.DataFrame(index=names).join(self._main_sample_columns()),
                                           kept_only=False)
                vals = base[src["column"]].str.strip()
                out[part] = vals.where(~vals.isin(MISSING))
            elif src["source"] == "derived_from_sample_names":
                rule = d["derivation"]
                for n in names:
                    p = split_name(n, rule["delimiter"], rule["occurrence"])
                    if p:
                        side = dict(zip((rule.get("left"), rule.get("right")), p))
                        out.loc[n, part] = side.get(part)
        return out

    def design(self) -> dict:
        """The design decision and the facts code computes from it (study samples only)."""
        d = dict(self.design_decision or self._proposed_design())
        st = self._subject_time()
        if st is not None:
            study = self.samples().query("is_study_sample")["sample"]
            st = st.loc[st.index.isin(study)]
            d["facts"] = summarize(st["subject"], st["time"] if d["time"]["source"] != "none" else None)
            d["facts"]["n_study_samples_without_subject"] = int(st["subject"].isna().sum())
        else:
            d["facts"] = {"label": "no subject information"}
        return d

    def set_design(self, subject: str | None = "keep", time: str | None = "keep", derive: dict | None = None,
                   time_unit: str | None = None) -> dict:
        """subject / time: a sample-information column name, "derived" (from derive=...), or None."""
        d = dict(self.design_decision or self._proposed_design())
        if derive is not None:
            if derive.get("delimiter") not in DELIMITERS or derive.get("occurrence") not in ("first", "last") or \
                    not {derive.get("left"), derive.get("right")} & {"subject", "time"}:
                raise ValueError("derive = {'delimiter': '_', 'occurrence': 'first'|'last', "
                                 "'left': 'subject'|'time'|None, 'right': 'subject'|'time'|None}")
            d["derivation"] = derive
            for side in ("left", "right"):
                if derive.get(side) in ("subject", "time"):
                    d[derive[side]] = {"source": "derived_from_sample_names"}
        cols = self.sample_columns()
        for part, value in (("subject", subject), ("time", time)):
            if value == "keep":
                continue
            if value is None:
                d[part] = {"source": "none"}
            elif value == "derived":
                if not d.get("derivation"):
                    raise ValueError("Give derive={...} to derive from sample names.")
                d[part] = {"source": "derived_from_sample_names"}
            elif value in cols:
                d[part] = {"source": "metadata_column", "column": value}
            else:
                raise KeyError(f"No sample information column '{value}'. Columns: {list(cols)}")
        if time_unit is not None:
            if time_unit not in VOCABULARY["time_unit"]:
                raise ValueError(f"time_unit must be one of {VOCABULARY['time_unit']}")
            d["time_unit"] = time_unit
        d["provenance"] = "user_set"
        self.design_decision = d
        self._note("design", "design set", {k: d[k] for k in ("subject", "time", "derivation", "time_unit")})
        return self.design()

    def confirm_design(self) -> None:
        d = self.design_decision or self._proposed_design()
        if d["time"]["source"] != "none" and not d.get("time_unit"):
            raise ValueError(f"Give the time unit: s.set_design(time_unit=...) with one of {VOCABULARY['time_unit']}")
        if self.design_decision is None:
            self.design_decision = d
        self.design_confirmed = True
        self._note("design", "design confirmed", self.design()["facts"].get("label"))

    def _todo_design(self) -> list[dict]:
        out, step = [], "8 design"
        if self.info:
            if not self.info["id_column"]:
                out.append({"step": step, "what": "Which column of the sample information names the samples? "
                                                  "s.set_info_id(...)"})
            elif self.info_match().get("n_matched", 0) == 0:
                out.append({"step": step, "what": "No sample name matches the sample information file: check "
                                                  "s.info_match() and the ID column"})
            if not self.info["confirmed"]:
                out.append({"step": step, "what": "Confirm the sample information (s.sample_info()): "
                                                  "s.confirm_sample_info()"})
        if not self.design_confirmed:
            try:
                label = self.design()["facts"].get("label")
            except (ValueError, KeyError) as e:
                label = f"not computable yet: {e}"
            out.append({"step": step, "what": f"Design ({label}): s.design(), then "
                                              "s.set_design(...) and/or s.confirm_design()"})
        return out
