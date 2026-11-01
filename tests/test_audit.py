import numpy as np
import pytest

import prism


def finished(tmp_path, values, extra_header="", types=None, history=None, plates=None):
    """A confirmed SomaScan-like schema. values: rows of cell text for samples S1..S(n-1) + QC_1.
    plates: optional {sample: plate} written as a sample-information file (Plate = batch)."""
    types = types or ("Protein",) * len(values)
    n = len(values[0])
    header = "SeqId,Type," + ",".join(f"S{j}" for j in range(1, n)) + ",QC_1"
    rows = [f"{1000 + i}-1,{types[i]}," + ",".join(r) for i, r in enumerate(values)]
    p = tmp_path / "soma.csv"
    p.write_text("\n".join([header] + rows) + "\n")
    answer = {"layout": {"value": "samples_in_columns", "confidence": 1, "evidence": ""},
              "groups": [{"group_id": "a", "role": "feature_id", "label": "id", "columns": ["SeqId"],
                          "confidence": 1, "evidence": ""},
                         {"group_id": "b", "role": "feature_annotation", "label": "type", "columns": ["Type"],
                          "marks_rows_as_suspect": True, "flagged_values": ["Control"], "confidence": 1, "evidence": ""},
                         {"group_id": "c", "role": "value", "label": "RFU", "templates": ["S#"], "columns": ["S2", "QC_1"],
                          "assay_label": "SomaScan", "confidence": 1, "evidence": ""}],
              "assays": [{"assay_label": "SomaScan", "omics_type": "x", "omics_family": "proteomics",
                          "in_supported_scope": "yes", "feature_identity": ["a"], "confidence": 1, "evidence": ""}]}

    def ai(system, prompt, schema):
        if system.startswith("You label the samples"):
            return {"rules": [{"pattern": "QC_*", "sample_type": "qc", "evidence": ""}]}
        if system.startswith("You describe the columns of a SAMPLE"):
            return {"columns": [
                {"column": "Sample", "is_sample_id": True, "audit_kind": "other", "label": "id", "confidence": 1,
                 "evidence": ""},
                {"column": "Plate", "is_sample_id": False, "audit_kind": "batch", "label": "plate", "confidence": 1,
                 "evidence": ""}]}
        return answer
    s = prism.start(prism.read(p), ai=prism.MockAI(ai))
    s.confirm_all(); s.confirm_feature_id(); s.confirm_flags(); s.confirm_values()
    s.propose_sample_types(); s.confirm_samples()
    if plates:
        info = tmp_path / "info.csv"
        info.write_text("Sample,Plate\n" + "".join(f"{k},{v}\n" for k, v in plates.items()))
        s.add_sample_info(info)
        s.confirm_sample_info()
    s.confirm_design()
    for bid, g in s.block_map().items():
        if g.zero_means is None:
            s.set_block(bid, zero_means="not_detected")
    s.answer_history(**{**dict(normalized="no", log_transformed="no", imputed="no", batch_corrected="no",
                                features_or_samples_removed_before_upload="no"), **(history or {})})
    assert s.todo().empty, s.todo()
    return s


def good_values():
    rng = np.random.default_rng(1)
    return [[f"{v:.1f}" for v in rng.lognormal(7, 1, 6)] for _ in range(6)]


def test_audit_needs_a_finished_schema(tmp_path):
    s = finished(tmp_path, good_values())
    s.set("g1", role="feature_id", label="changed")   # still fine: nothing open
    a = prism.Audit(s)
    s.answer_history.__self__.history_answers.clear()
    with pytest.raises(ValueError, match="open item"):
        prism.Audit(s)
    assert a.schema is s


def test_clean_data(tmp_path):
    types = ("Protein",) * 5 + ("Control",)
    a = prism.Audit(finished(tmp_path, good_values(), types=types))
    f = a.integrity()
    assert f.status == "ok"
    assert "No integrity problem" in f.sentences[0][1]
    assert f.measures["audit_set"] == {"n_study_samples": 5, "n_proteins_unflagged": 5}
    X = a.audit_matrix("A1B1")
    assert X.shape == (5, 5) and "QC_1" not in X.columns and "1005-1" not in X.index


def test_problems_and_notes(tmp_path):
    v = good_values()
    v[0][1] = "<LOD"                 # not a number
    v[1] = ["5"] * 6                 # constant protein
    v[2][4] = v[2][3] = "1.0"        # a value that recurs ...
    v[3][4] = v[3][3] = "1.0"
    v[4] = list(v[5])                # identical proteins
    for r in v:                      # S5 = S4 exactly
        r[4] = r[3]
    f = prism.Audit(finished(tmp_path, v)).integrity()
    text = " | ".join(t for _, t in f.sentences)
    assert f.status == "problem"
    assert "'<LOD'" in text and "not numbers" in text
    assert "same value in every sample" in text
    assert "S4, S5 have exactly the same values" in text
    assert "group(s) of proteins have exactly the same values" in text
    assert "fills" in text and "placeholder" in text
