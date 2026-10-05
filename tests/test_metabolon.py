"""v2.4 §22: the metabolomics run. One median-scaled export of 1,174 <class>_<number> columns that a
(mocked) AI splits wrongly: 3 assays, 9 blocks, a chunk-local label and 2 orphan columns. Code must
ask (fragmentation, orphans), never merge by itself, keep chunk labels out of the schema, and count
right (n_features = 1,174; ledger 1,174 values + 1 ID)."""

import csv
import io
import json

from backend import config, mock_llm
from conftest import fixture_bytes

L = "L_metabolon_like.csv"
HEADER = next(csv.reader(io.StringIO(fixture_bytes(L).decode())))
ORPHANS = ["Amino Acid_100010863", "Xenobiotics_100010955"]
CHUNK_LABEL = "Metabolite feature measurements chunk"


def wrong_split(system, prompt):
    payload = json.loads(prompt.split("\n", 1)[1])
    if prompt.startswith("Consolidation"):
        vals = [g["group_id"] for g in payload["groups"] if g["role"] == "value"]
        while len(vals) > 9:                                   # 9 final blocks: fold extra chunk groups in
            last = vals.pop()
            vals[-1] += "+" + last
        assays = ["Amino Acid", "Lipid", "Xenobiotics"]
        final = [{"group_id": f"b{k}", "members": v.split("+"), "role": "value", "assay_label": assays[k * 3 // 9],
                  "label": f"metabolite levels, set {k + 1}", "confidence": 0.7, "evidence": "x"}
                 for k, v in enumerate(vals)]
        final += [{"group_id": "id", "members": [g["group_id"] for g in payload["groups"] if g["role"] == "sample_id"],
                   "role": "sample_id", "label": "sample name", "confidence": 0.9, "evidence": "x"}]
        return json.dumps({"groups": final, "assays": [{"assay_label": a, "omics_type": "metabolomics",
                                                        "in_supported_scope": "yes", "confidence": 0.6,
                                                        "evidence": "class prefix", "feature_identity": {"group_ids": []}}
                                                       for a in assays]})
    tpls = {t["template"] for t in payload["templates"]}
    singles = [c["column"] for c in payload["columns"]]
    mine = [c for c in HEADER[1:] if c not in ORPHANS and (c in singles or any(
        t == mock_llm.name_template(c) for t in tpls))]
    blocks = [mine[k:k + 125] for k in range(0, len(mine), 125)]   # boundaries follow neither names nor profiles
    groups = [{"group_id": f"g{k}", "columns": b, "role": "value", "assay_label": "metabolites", "label": CHUNK_LABEL,
               "confidence": 0.6, "evidence": "numeric"} for k, b in enumerate(blocks)]
    if "ID" in singles:
        groups.append({"group_id": "id", "columns": ["ID"], "role": "sample_id", "label": "sample", "confidence": 0.9,
                       "evidence": "x"})
    return json.dumps({"layout": {"value": "samples_in_rows", "confidence": 0.9, "evidence": "27 rows"},
                       "assays": [{"assay_label": "metabolites", "omics_type": "metabolomics", "in_supported_scope": "yes",
                                   "confidence": 0.5, "evidence": "x"}], "groups": groups})


def answer(f, q, label_start):
    opt = next(o for o in q["options"] if o["label"].startswith(label_start))
    r = f.c.post("/api/question/answer", json={"session_id": f.sid, "question_id": q["question_id"],
                                                "option_ids": [opt["option_id"]]})
    assert r.status_code == 200, r.text
    f.upload, f.draft = r.json()["session"], r.json()["draft"]


def test_wrong_split_becomes_questions_and_one_measurement(flow, monkeypatch):
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(wrong_split))
    monkeypatch.setattr(config, "GROUPING_CHUNK_SIZE", 4)
    f = flow(L)
    gr = f.draft["grouping"]
    assert gr["n_templates"] == 11 and gr["chunks"] == 3                 # 10 class templates + ID, in chunks of 4
    assert [a["assay_label"] for a in f.draft["assays"]] == ["Amino Acid", "Lipid", "Xenobiotics"]
    blocks = [g for g in f.upload["groups"] if f.draft["groups"][g["group_id"]]["role"] == "value"]
    assert len(blocks) == 9
    assert not any(CHUNK_LABEL in (it["label"] or "") for it in f.draft["groups"].values())   # chunk outputs internal
    # code asks, never merges by itself
    frag = [q for q in f.draft["questions"] if q["kind"] == "fragmentation"]
    assert len(frag) == 1 and frag[0]["text"].startswith("9 blocks in 3 assays have the same samples")
    orphans = [q for q in f.draft["questions"] if q["kind"] == "orphan"]
    assert sorted(q["applies_to"]["columns"][0] for q in orphans) == ORPHANS
    for q in orphans:                                              # candidate blocks, not a manual role
        assert q["options"][0]["label"].startswith("Add to 'metabolite levels")
        assert [o["label"] for o in q["options"][-2:]] == ["Keep it as its own block", "Exclude it from the outputs"]
        answer(f, q, "Add to")
    frag = next(q for q in f.draft["questions"] if q["kind"] == "fragmentation" and q["status"] == "open")
    assert "1174 columns" in frag["options"][0]["label"]               # the click shows the count
    answer(f, frag, "One measurement")
    assert len(f.draft["assays"]) == 1
    blocks = [g for g in f.upload["groups"] if f.draft["groups"][g["group_id"]]["role"] == "value"]
    assert len(blocks) == 1 and blocks[0]["n_columns"] == 1174
    assert not [q for q in f.draft["questions"] if q["status"] == "open"]
    f.confirm_all_as_proposed()
    out = f.finalize()["schema"]
    (a,) = out["assays"]
    assert a["n_features"] == 1174 and a["n_value_columns"] == 1174 and a["n_samples"] == 27
    led = out["column_ledger"]["main"]
    assert (led["value"], led["sample_id"], led["total"]) == (1174, 1, 1175)
    text = json.dumps(out)
    assert CHUNK_LABEL not in text and "metabolites" not in [x["assay_label"] for x in out["assays"]]


def test_separate_measurements_is_remembered_and_counts_add_up(flow, monkeypatch):
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(wrong_split))
    monkeypatch.setattr(config, "GROUPING_CHUNK_SIZE", 4)
    f = flow(L)
    for q in [q for q in f.draft["questions"] if q["kind"] == "orphan"]:
        answer(f, q, "Exclude")
    frag = next(q for q in f.draft["questions"] if q["kind"] == "fragmentation")
    answer(f, frag, "Separate")
    f.confirm_all_as_proposed()
    out = f.finalize()["schema"]
    assert sum(a["n_value_columns"] for a in out["assays"]) == 1172
    for a in out["assays"]:                         # samples in rows: n_features = the blocks' feature columns
        assert a["n_features"] == sum(b["n_features"] for b in a["value_blocks"]) == a["n_value_columns"]
    led = out["column_ledger"]["main"]
    assert (led["value"], led["excluded"], led["sample_id"]) == (1172, 2, 1)
    assert next(q for q in out["questions"] if q["question_id"] == frag["question_id"])["status"] == "answered"


def test_inconsistent_counts_block_finalize(flow, monkeypatch):
    """The same feature name in two kept blocks: distinct features != block columns."""
    header = ["ID", "A_1", "A_2", "B_1", "A_1"]
    rows = [[f"S{r}", "1.1", "1.2", "1.3", "1.4"] for r in range(6)]
    buf = io.StringIO()
    csv.writer(buf).writerows([header] + rows)

    def two(system, prompt):
        return json.dumps({"layout": {"value": "samples_in_rows", "confidence": 1, "evidence": "x"},
                           "assays": [{"assay_label": "m", "omics_type": "x", "in_supported_scope": "yes",
                                       "confidence": 1, "evidence": "x"}],
                           "groups": [{"group_id": "i", "columns": ["ID"], "role": "sample_id", "label": "s",
                                       "confidence": 1, "evidence": "x"},
                                      {"group_id": "a", "columns": ["A_1 [column 2]", "A_2"], "role": "value",
                                       "assay_label": "m", "label": "a", "confidence": 1, "evidence": "x"},
                                      {"group_id": "b", "columns": ["B_1", "A_1 [column 5]"], "role": "value",
                                       "assay_label": "m", "label": "b", "confidence": 1, "evidence": "x"}]})
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(two))
    f = flow("dup.csv", content=buf.getvalue().encode())
    frag = next(q for q in f.draft["questions"] if q["kind"] == "fragmentation")
    answer(f, frag, "Separate")
    f.confirm_all_as_proposed()
    r = f.finalize(expect=422)
    assert "Counts do not agree" in r["detail"] and "3 distinct features but its blocks hold 4" in r["detail"]
