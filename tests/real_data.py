"""Drive the Step 0 wizard on the real files in tests/data (git-ignored; tests skip without them).

The mock AI is steered the way the real model groups these files: SomaScan's 27 <animal>_<week>
columns are one value block with SeqId as the feature ID; the metabolomics file has samples in
rows ('ID') and one value block of <class>_<number> columns. The design is derived from the
sample names (<subject>_<time>, time in weeks). Every other proposal is confirmed as made.
"""

import csv
import json
import re
from pathlib import Path

from backend import mock_llm
from conftest import Flow

DATA = Path(__file__).parent / "data"
SOMA, METAB = "SomaExpr_common.csv", "MetaboExpr_common.csv"
_SAMPLE = re.compile(r"^[A-Z0-9]{3,4}_\d+$")


def have(*names):
    return all((DATA / n).exists() for n in (names or (SOMA, METAB)))


def _soma_respond(original):
    def respond(system, prompt):
        out = json.loads(original(system, prompt))
        if not prompt.startswith("Digest"):
            return json.dumps(out)
        digest = json.loads(prompt.split("\n", 1)[1])
        with open(DATA / SOMA, encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        names = [n for n in header if _SAMPLE.match(n)]
        tpls = {t["template"] for t in digest["templates"]
                if any(_SAMPLE.match(c) for c in t["columns"] if not c.startswith("... ("))}
        keep = [g for g in out["groups"] if not (set(g.get("templates") or []) & tpls)
                and not any(_SAMPLE.match(c) for c in g.get("columns") or [])]
        keep.append({"group_id": "rfu", "columns": names, "role": "value", "assay_label": "proteomics",
                     "label": "SomaScan RFU", "confidence": 0.9, "evidence": "27 numeric <animal>_<week> columns"})
        fid = None
        for g in keep:
            if g.get("columns") == ["SeqId"]:
                g["role"], fid = "feature_id", g["group_id"]
            elif g["role"] in ("unresolved", "value") and g["group_id"] != "rfu":
                g.update(role="feature_annotation", label=f"{(g.get('columns') or g.get('templates'))[0]} (per aptamer)",
                         marks_rows_as_suspect=False)
        out["groups"] = keep
        out["assays"] = [dict(out["assays"][0], assay_label="proteomics", omics_type="proteomics",
                              source_software="SomaScan", feature_identity={"group_ids": [fid]})]
        return json.dumps(out)
    return staticmethod(respond)


DESIGN = {"derivation": {"delimiter": "_", "occurrence": "last", "left": "subject", "right": "time"},
          "subject": {"source": "derived_from_sample_names"},
          "time": {"source": "derived_from_sample_names", "unit": "weeks"}}


def wizard(client, name):
    """-> the Flow after finalize (the output folder is sessions/<sid>/outputs)."""
    original = mock_llm.MockLLM.respond
    if name == SOMA:
        mock_llm.MockLLM.respond = _soma_respond(original)
    try:
        f = Flow(client, name, content=(DATA / name).read_bytes())
    finally:
        mock_llm.MockLLM.respond = original
    for q in [q for q in f.draft["questions"] if q["status"] == "open"]:
        if q["kind"] == "fragmentation":
            f.answer(q, "One measurement")
    for q in [q for q in f.draft["questions"] if q["status"] == "open" and q["text"].startswith("Which values of")]:
        r = client.post("/api/question/dismiss", json={"session_id": f.sid, "question_id": q["question_id"]})
        assert r.status_code == 200, r.text
        f.draft = r.json()["draft"]
    d = f.draft
    f.step("layout", {"layout": d["layout"]["value"],
                      "assays": [{k: a.get(k) for k in ("assay_label", "omics_type", "source_software",
                                                         "in_supported_scope", "scope_reason")} for a in d["assays"]]})
    d = f.draft
    dec = {"feature_identity": {"group_ids": d["feature_identity"]["group_ids"]}}
    if d["sample_id_group"]["value"]:
        dec["sample_id_group"] = d["sample_id_group"]["value"]
    f.step("feature_id", dec)
    for st in ("annotations", "values", "samples"):
        if f.draft["steps"][st] != "not_applicable":
            f.step(st, {})
    f.step("sample_info", {"metadata": {"skip": True}})
    f.step("design", {"design": DESIGN})
    f.step("history", {"processing_history": {q: {"answer": "not_sure"} for q in (
        "normalized", "log_transformed", "imputed", "batch_corrected", "features_or_samples_removed_before_upload")}})
    f.finalize()
    return f


def wizard_output(client, isolated, name):
    f = wizard(client, name)
    return isolated / "sessions" / f.sid / "outputs"
