import prism


def soma(tmp_path):
    header = "SeqId,Target,Organism,Kd,CV,S01,S02,S03,QC_01"
    rows = [f"{1000 + i}-{i},Prot{i},{'Mouse' if i == 3 else 'Human'},1e-9,{5 + i},{100 + i},{200 + i},{300 + i},{150 + i}"
            for i in range(8)]
    p = tmp_path / "soma.csv"
    p.write_text("\n".join([header] + rows) + "\n")
    return prism.read(p)


def group(gid, role, label, columns=(), templates=(), **k):
    return {"group_id": gid, "role": role, "label": label, "columns": list(columns), "templates": list(templates),
            "confidence": 0.9, "evidence": "from the digest", **k}


GOOD = {
    "layout": {"value": "samples_in_columns", "confidence": 0.95, "evidence": "S# template"},
    "groups": [
        group("a", "feature_id", "SomaScan sequence ID", ["SeqId"]),
        group("b", "feature_annotation", "target protein name", ["Target"]),
        group("c", "feature_annotation", "organism of the target", ["Organism"], marks_rows_as_suspect=True),
        group("d", "feature_annotation", "dissociation constant", ["Kd"]),
        group("e", "feature_annotation", "assay CV", ["CV"]),
        group("f", "value", "RFU per sample", ["QC_01"], ["S#"], assay_label="SomaScan"),
    ],
    "assays": [{"assay_label": "SomaScan", "omics_type": "aptamer proteomics", "omics_family": "proteomics",
                "in_supported_scope": "yes", "feature_identity": ["a"], "confidence": 0.9, "evidence": "SeqId"}],
}


def test_good_answer(tmp_path):
    g = prism.propose_groups(soma(tmp_path), ai=prism.MockAI(GOOD))
    assert g.rejected == []
    assert [x.id for x in g.groups] == ["g1", "g2", "g3", "g4", "g5", "g6"]
    values = g.groups[-1]
    assert values.role == "value" and values.columns == ["S01", "S02", "S03", "QC_01"]   # template expanded
    assert g.assays[0]["feature_identity"] == ["g1"]                                       # ids renumbered
    assert g.groups[2].marks_rows_as_suspect
    assert len(g.columns()) == 9 and g.model == "mock"


def test_ai_sees_digest_not_rows(tmp_path):
    ai = prism.MockAI(GOOD)
    prism.propose_groups(soma(tmp_path), ai=ai)
    q = ai.questions[0]
    assert "Prot5" not in q["prompt"] and "S#" in q["prompt"]
    assert "unresolved" in q["system"]


def test_unknown_twice_and_unmentioned(tmp_path):
    answer = {"layout": {"value": "samples_in_columns", "confidence": 1, "evidence": ""}, "assays": [],
              "groups": [group("a", "feature_id", "id", ["SeqId", "Nonsense"]),
                         group("b", "value", "values", ["SeqId"], ["S#", "X#"]),
                         group("a", "value", "dup id", ["Kd"]),
                         group("c", "value", "nothing real", ["Ghost"])]}
    g = prism.propose_groups(soma(tmp_path), ai=prism.MockAI(answer))
    reasons = " | ".join(r["reason"] for r in g.rejected)
    assert "No such column" in reasons and "Already placed in group a" in reasons
    assert "No such template" in reasons and "repeated group id" in reasons and "names no valid column" in reasons
    # every column is still somewhere; the ones nobody claimed are unresolved groups of one
    assert sorted(g.columns()["column"]) == sorted(g.table.data.columns)
    unresolved = [x.columns[0] for x in g.unresolved()]
    assert unresolved == ["Target", "Organism", "Kd", "CV", "QC_01"]


def test_closed_fields_and_merge(tmp_path):
    answer = {"layout": {"value": "sideways", "confidence": 1, "evidence": ""}, "assays": [],
              "groups": [group("a", "feature_id", "id", ["SeqId"], audit_kind="batch"),
                         group("b", "made_up_role", "?", ["Target"]),
                         group("v1", "value", "samples", ["S01", "S02"]),
                         group("v2", "value", "more samples", ["S03"])],
              "propose_merge": [{"group_ids": ["v1", "v2"], "reason": "same RFU"},
                                {"group_ids": ["v1", "zzz"], "reason": "bad"}]}
    g = prism.propose_groups(soma(tmp_path), ai=prism.MockAI(answer))
    assert g.layout["value"] is None
    assert g.groups[0].audit_kind is None                     # only sample information has an audit_kind
    assert g.groups[1].role == "unresolved"
    merged = [x for x in g.groups if x.origin == "merged_ai"][0]
    assert merged.columns == ["S01", "S02", "S03"]
    assert any("zzz" in r["reason"] for r in g.rejected)


def test_scope_is_proteomics(tmp_path):
    ai = prism.MockAI(GOOD)
    g = prism.propose_groups(soma(tmp_path), ai=ai)
    assert "Proteomics only" in ai.questions[0]["system"] and g.out_of_scope() == []
    answer = dict(GOOD, assays=[{"assay_label": "Metabolon", "omics_type": "untargeted metabolomics",
                                 "omics_family": "metabolomics", "in_supported_scope": "no",
                                 "scope_reason": "not proteomics", "confidence": 0.9, "evidence": ""}])
    g = prism.propose_groups(soma(tmp_path), ai=prism.MockAI(answer))
    assert g.out_of_scope()[0]["assay_label"] == "Metabolon" and "NOT IN SCOPE" in g.describe()
