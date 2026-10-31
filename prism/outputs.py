"""Step 10: review everything, then save schema.json and the output files.

    s.review()             # every decision with who made it, and what is still open
    s.save("/content/drive/MyDrive/PRISM_out/soma")    # refuses while anything is open
    s2 = prism.load("/content/drive/MyDrive/PRISM_out/soma/schema.json", table)   # no questions again

The output folder (as in the old website):
    schema.json              every decision, its provenance, the log of actions
    value_matrix_A1_B1.csv   one file per kept block: proteins x samples, values exactly as in the
                             file (missing cells empty). Nothing filled, scaled or removed.
    feature_metadata.csv     one row per protein: ID, annotations, flagged, flag_reason
    sample_metadata.csv      one row per sample: sample_type, is_study_sample, sample information,
                             subject, time
    import_manifest.json     source file (name, size, SHA-256 fingerprint), counts, AI models used
Flagged proteins and QC / blank samples are kept and labelled, never dropped.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .vocabulary import SCHEMA_VERSION


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class OutputSteps:

    # ---------------------------------------------------------------- review

    def review(self) -> None:
        """Print every decision with its provenance, then what is still open."""
        p = print
        p(f"FILE  {self.table.path.name}: {self.table.describe()}")
        p(f"\n5 LAYOUT  {self.layout}  [{self.layout_decision['provenance']}]")
        p("5 GROUPS")
        for g in self._ordered():
            extra = f", {g.audit_kind}" if g.audit_kind else ""
            p(f"   {g.id:>4}  {g.role}{extra}: {g.label}  ({len(g.columns)} col.)  [{g.provenance}]")
        if self.layout != "samples_in_rows":
            p(f"\n6 PROTEIN ID  {self.feature_id_decision['columns']}  [{self.feature_id_decision['provenance']}]")
        flagged, _ = self.flag_mask()
        p(f"6 FLAGS  {int(flagged.sum())} of {len(flagged)} proteins flagged")
        for col, f in self.flag_decisions.items():
            p(f"   {col}: values {f['values']} prefixes {f['prefixes']}  [{f['provenance']}]")
        p("\n7 ASSAYS AND BLOCKS")
        for a in self.assays:
            p(f"   {a['assay_label']}: {a.get('omics_type')} ({a.get('omics_family')}), in scope "
              f"{a.get('in_supported_scope')}  [{a['provenance']}]")
        for bid, g in self.block_map().items():
            p(f"   {bid}  {g.label}  zero_means={g.zero_means}  [{g.provenance}]")
        types = self.samples()["sample_type"].value_counts().to_dict()
        p(f"7 SAMPLES  {sum(types.values())}: {types}")
        d = self.design()
        p(f"\n8 DESIGN  {d['facts'].get('label')}; subject {d['subject']}, time {d['time']}, "
          f"unit {d.get('time_unit')}  [{d['provenance']}]")
        p("\n9 HISTORY")
        for a, ans in self.history_answers.items():
            p(f"   {a}: {ans}  [user_set]")
        todo = self.todo()
        p(f"\nOPEN ITEMS: {len(todo)}" + ("" if len(todo) else "  -> ready: s.save(folder)"))
        for _, r in todo.iterrows():
            p(f"   [{r['step']}] {r['what']}")

    # ---------------------------------------------------------------- the files

    def feature_metadata(self) -> pd.DataFrame:
        flagged, reason = self.flag_mask()
        ids = self.feature_ids()
        if self.layout == "samples_in_rows":
            block = {c: bid for bid, g in self.block_map().items() for c in g.columns}
            return pd.DataFrame({"feature": ids.values, "block": [block.get(i, "") for i in ids],
                                 "flagged": flagged.values, "flag_reason": reason.values})
        cols = [c for g in self.groups_with("feature_id") + self.groups_with("feature_annotation") for c in g.columns]
        out = self.table.data[cols].copy()
        out.insert(0, "feature", ids.values)
        out["flagged"], out["flag_reason"] = flagged.values, reason.values
        if self.layout == "long":     # one row per protein: first value; flagged if any of its rows is
            agg = {c: "first" for c in out.columns if c not in ("feature", "flagged", "flag_reason")}
            agg.update({"flagged": "any", "flag_reason": lambda r: "; ".join(sorted(set(x for x in r if x)))})
            out = out.groupby("feature", sort=False).agg(agg).reset_index()
        return out

    def to_dict(self) -> dict:
        t = self.table
        bm = self.block_map()
        return {
            "schema_version": SCHEMA_VERSION,
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "source": {"file_name": t.path.name, "sha256": sha256(t.path), "delimiter": t.delimiter,
                       "n_rows": t.shape[0], "n_columns": t.shape[1], "columns": list(t.data.columns)},
            "layout": self.layout_decision,
            "groups": [asdict(g) for g in self._ordered()],
            "assays": [dict(a, blocks=[{"block_id": bid, "group_id": g.id, "label": g.label,
                                        "zero_means": g.zero_means, "matrix_file": f"value_matrix_{bid.replace('B', '_B')}.csv"}
                                       for bid, g in bm.items() if (g.assay_label or "assay") == a["assay_label"]])
                       for a in self.assays],
            "feature_id": self.feature_id_decision,
            "flags": {"columns": self.flag_decisions, "confirmed": self.flags_confirmed},
            "values_confirmed": self.values_confirmed,
            "samples": {"types": self.sample_type_decisions, "confirmed": self.samples_confirmed},
            "sample_info": None if not self.info else {k: v for k, v in self.info.items() if k != "table"},
            "design": {"decision": self.design_decision, "confirmed": self.design_confirmed,
                       "facts": self.design().get("facts")},
            "history": self.history_answers,
            "ai": {"models_used": sorted(set(self.ai_models)), "example_values_sent": self.examples_sent,
                   "data_rows_sent": False, "rejected_from_ai_answers": self.rejected},
            "log": self.log,
        }

    def save(self, folder) -> Path:
        todo = self.todo()
        if len(todo):
            raise ValueError(f"{len(todo)} item(s) still open (s.todo()); first: {todo.iloc[0]['what']}")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        schema = self.to_dict()
        written = []
        for bid in self.block_map():
            name = f"value_matrix_{bid.replace('B', '_B')}.csv"
            self.matrix(bid).to_csv(folder / name)
            written.append(name)
        self.feature_metadata().to_csv(folder / "feature_metadata.csv", index=False)
        self.sample_table().to_csv(folder / "sample_metadata.csv")
        (folder / "schema.json").write_text(json.dumps(schema, indent=1, ensure_ascii=False, default=str))
        manifest = {"schema_version": SCHEMA_VERSION, "created": schema["created"], "source": {
            k: schema["source"][k] for k in ("file_name", "sha256", "n_rows", "n_columns")},
            "files": written + ["feature_metadata.csv", "sample_metadata.csv", "schema.json"],
            "n_proteins": {bid: int(self.matrix(bid).shape[0]) for bid in self.block_map()},
            "n_samples": len(self.all_samples()), "n_flagged_proteins": int(self.flag_mask()[0].sum()),
            "ai": schema["ai"]["models_used"]}
        (folder / "import_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
        self._note("review", "saved", str(folder))
        print(f"Saved to {folder}: " + ", ".join(manifest["files"] + ["import_manifest.json"]))
        return folder


def load(schema_path, table) -> "Schema":
    """Rebuild a saved schema on the same table, so nothing is asked again."""
    from .grouping import Group, Grouping
    from .schema import Schema
    from .table import read
    d = json.loads(Path(schema_path).read_text())
    if list(table.data.columns) != d["source"]["columns"]:
        raise ValueError("This table's columns differ from the saved schema's: it is a different file.")
    if sha256(table.path) != d["source"]["sha256"]:
        print("[PRISM] Note: the file's contents changed since the schema was saved (same columns).")
    groups = [Group(**g) for g in d["groups"]]
    s = Schema(table, Grouping(table=table, layout=d["layout"], groups=groups, assays=[]))
    s.layout_decision = d["layout"]
    s.assays = [{k: v for k, v in a.items() if k != "blocks"} for a in d["assays"]]
    s.feature_id_decision = d["feature_id"]
    s.flag_decisions, s.flags_confirmed = d["flags"]["columns"], d["flags"]["confirmed"]
    s.values_confirmed = d["values_confirmed"]
    s.sample_type_decisions, s.samples_confirmed = d["samples"]["types"], d["samples"]["confirmed"]
    if d["sample_info"]:
        info = dict(d["sample_info"])
        info["table"] = read(info["path"])
        s.info = info
    s.design_decision, s.design_confirmed = d["design"]["decision"], d["design"]["confirmed"]
    s.history_answers = d["history"]
    s.ai_models = list(d["ai"]["models_used"])
    s.examples_sent = d["ai"]["example_values_sent"]
    s.rejected = d["ai"]["rejected_from_ai_answers"]
    s.log = d["log"] + [{"time": _dt.datetime.now().isoformat(timespec="seconds"), "step": "review",
                         "action": "loaded", "detail": str(schema_path)}]
    return s
