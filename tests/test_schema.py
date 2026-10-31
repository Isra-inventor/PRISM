"""Steps 5-10 end to end, with a stand-in AI that answers each kind of question."""
import json

import pandas as pd
import pytest

import prism


def grp(gid, role, label, columns=(), templates=(), **k):
    return {"group_id": gid, "role": role, "label": label, "columns": list(columns), "templates": list(templates),
            "confidence": 0.9, "evidence": "digest", **k}


def assay(label, ids, family="proteomics"):
    return {"assay_label": label, "omics_type": "proteomics", "omics_family": family, "in_supported_scope": "yes",
            "feature_identity": ids, "confidence": 0.9, "evidence": ""}


def mock(grouping, sample_rules=(), info_columns=()):
    def answer(system, prompt, schema):
        if system.startswith("You label the samples"):
            return {"rules": list(sample_rules), "column_values": []}
        if system.startswith("You describe the columns of a SAMPLE"):
            return {"columns": list(info_columns)}
        return grouping
    return prism.MockAI(answer)


# ---------------------------------------------------------------- SomaScan-like, samples in columns

def soma_file(tmp_path):
    header = "SeqId,Target,Type,Kd,P01_T1,P01_T2,P02_T1,P02_T2,QC_1"
    rows = []
    for i in range(6):
        typ = "Hybridization Control" if i == 5 else "Protein"
        rows.append(f"{1000 + i}-1,Prot{i},{typ},1e-9," + ",".join(str(100 * (i + 1) + j) for j in range(5)))
    p = tmp_path / "soma.csv"
    p.write_text("\n".join([header] + rows) + "\n")
    info = tmp_path / "info.csv"
    info.write_text("SampleId,Patient,Visit,Plate\nP01_T1,P01,1,A\nP01_T2,P01,2,A\nP02_T1,P02,1,B\nP02_T2,P02,2,B\n")
    return prism.read(p), info


SOMA = {"layout": {"value": "samples_in_columns", "confidence": 0.9, "evidence": ""},
        "groups": [grp("a", "feature_id", "aptamer ID", ["SeqId"]),
                   grp("b", "feature_annotation", "target", ["Target"]),
                   grp("c", "feature_annotation", "row type", ["Type"], marks_rows_as_suspect=True,
                       flagged_values=["Hybridization Control"]),
                   grp("d", "value", "Kd ??", ["Kd"], assay_label="SomaScan"),
                   grp("e", "value", "RFU", ["P01_T1", "P01_T2", "P02_T1", "P02_T2", "QC_1"], assay_label="SomaScan")],
        "assays": [assay("SomaScan", ["a"])]}
INFO = [{"column": "SampleId", "is_sample_id": True, "audit_kind": "other", "label": "id", "confidence": 1, "evidence": ""},
        {"column": "Patient", "is_sample_id": False, "audit_kind": "subject_id", "label": "patient", "confidence": 1, "evidence": ""},
        {"column": "Visit", "is_sample_id": False, "audit_kind": "timepoint", "label": "visit", "confidence": 1, "evidence": ""},
        {"column": "Plate", "is_sample_id": False, "audit_kind": "batch", "label": "plate", "confidence": 1, "evidence": ""}]


def test_somascan_whole_wizard(tmp_path):
    table, info = soma_file(tmp_path)
    s = prism.start(table, ai=mock(SOMA, [{"pattern": "QC_*", "sample_type": "qc", "evidence": "QC prefix"}], INFO))
    assert len(s.todo()) > 0

    # step 5: Kd is not a sample measurement -> correct it, confirm the rest
    s.set("g4", role="feature_annotation", label="dissociation constant")
    s.confirm_all()
    assert s.group("g4").provenance == "ai_proposed_corrected"
    assert s.group("g1").provenance == "ai_proposed_confirmed"
    assert not any(s.todo()["step"] == "5 groups")

    # step 6: ID and flags
    assert s.feature_id()["check"]["n_repeated"] == 0
    s.confirm_feature_id()
    assert s.flags().iloc[0]["n_rows_flagged"] == 1
    s.confirm_flags()

    # step 7: one block, sample types from the AI
    assert list(s.block_map()) == ["A1B1"]
    m = s.matrix("A1B1")
    assert m.shape == (6, 5) and m.loc["1000-1", "P01_T1"] == "100"
    s.confirm_values()
    smp = s.propose_sample_types().set_index("sample")
    assert smp.loc["QC_1", "sample_type"] == "qc" and smp.loc["P01_T1", "provenance"] == "computed"
    s.confirm_samples()

    # step 8: sample info file and design
    s.add_sample_info(info)
    assert s.info["id_column"] == "SampleId" and s.info_match()["n_matched"] == 4
    s.confirm_sample_info()
    d = s.design()
    assert d["subject"]["column"] == "Patient" and d["facts"]["label"].startswith("longitudinal, balanced (2")
    with pytest.raises(ValueError, match="time unit"):
        s.confirm_design()
    s.set_design(time_unit="visit_index")
    s.confirm_design()

    # step 9
    hist = s.history()
    assert "linear" in hist.set_index("question").loc["log_transformed", "hint"]
    with pytest.raises(ValueError):
        s.save(tmp_path / "out")                       # still open
    s.answer_history(normalized="yes", log_transformed="no", imputed="no", batch_corrected="no",
                     features_or_samples_removed_before_upload="not_sure")

    # step 10
    assert s.todo().empty
    s.review()
    out = s.save(tmp_path / "out")
    files = sorted(p.name for p in out.iterdir())
    assert files == ["feature_metadata.csv", "import_manifest.json", "sample_metadata.csv", "schema.json",
                     "value_matrix_A1_B1.csv"]
    fm = pd.read_csv(out / "feature_metadata.csv")
    assert fm["flagged"].sum() == 1 and "Kd" in fm.columns
    sm = pd.read_csv(out / "sample_metadata.csv", index_col="sample")
    assert sm.loc["P02_T2", "subject"] == "P02" and sm.loc["QC_1", "is_study_sample"] == False  # noqa: E712
    schema = json.loads((out / "schema.json").read_text())
    assert schema["ai"]["data_rows_sent"] is False and schema["schema_version"] == "0.4.1"

    # load: same decisions, nothing open
    s2 = prism.load(out / "schema.json", prism.read(table.path))
    assert s2.todo().empty and s2.group("g4").role == "feature_annotation"


# ---------------------------------------------------------------- MaxQuant-like: two blocks, prefixes, zeros

def test_maxquant_two_blocks(tmp_path):
    header = "Protein IDs,Gene names,Potential contaminant,Intensity A_1,Intensity B_2,LFQ intensity A_1,LFQ intensity B_2"
    rows = ["P1,G1,,10,20,11,21", "CON__P2,G2,+,0,30,0,31", "REV__P3,,,5,0,6,0"]
    p = tmp_path / "proteinGroups.txt"
    p.write_text("\n".join([r.replace(",", "\t") for r in [header] + rows]) + "\n")
    answer = {"layout": {"value": "samples_in_columns", "confidence": 0.9, "evidence": ""},
              "groups": [grp("a", "feature_id", "protein group", ["Protein IDs"], flagged_prefixes=["REV__", "CON__"]),
                         grp("b", "feature_annotation", "gene", ["Gene names"]),
                         grp("c", "feature_annotation", "contaminant flag", ["Potential contaminant"],
                             marks_rows_as_suspect=True, flagged_values=["+"]),
                         grp("d", "value", "raw intensity", ["Intensity A_1", "Intensity B_2"], assay_label="MQ"),
                         grp("e", "value", "LFQ", ["LFQ intensity A_1", "LFQ intensity B_2"], assay_label="MQ")],
              "assays": [assay("MQ", ["a"])]}
    s = prism.start(prism.read(p), ai=mock(answer))
    s.confirm_all()
    assert list(s.block_map()) == ["A1B1", "A1B2"]
    assert s.sample_names("A1B2") == ["A_1", "B_2"]                 # shared text removed
    flagged, reason = s.flag_mask()
    assert flagged.tolist() == [False, True, True] and "Protein IDs=CON__P2" in reason[1]
    assert any("holds zeros" in w for w in s.todo()["what"])
    s.set_block("A1B1", zero_means="not_detected")
    s.set_block("A1B2", zero_means="not_detected")
    assert not any("holds zeros" in w for w in s.todo()["what"])


# ---------------------------------------------------------------- Olink-like long table

def test_olink_long(tmp_path):
    rows = ["SampleID,OlinkID,Assay,NPX,Group"]
    for smp, grp_ in (("S1", "case"), ("S2", "ctrl"), ("S3", "case")):
        for k, a in enumerate(("IL6", "TNF")):
            rows.append(f"{smp},OID{k},{a},{-0.5 + k},{grp_}")
    p = tmp_path / "olink.csv"
    p.write_text("\n".join(rows) + "\n")
    answer = {"layout": {"value": "long", "confidence": 0.9, "evidence": ""},
              "groups": [grp("a", "sample_id", "sample", ["SampleID"]),
                         grp("b", "feature_id", "Olink ID", ["OlinkID"]),
                         grp("c", "feature_annotation", "assay name", ["Assay"]),
                         grp("d", "value", "NPX", ["NPX"], assay_label="Olink"),
                         grp("e", "sample_metadata", "group", ["Group"], audit_kind="group")],
              "assays": [assay("Olink", ["b"])]}
    s = prism.start(prism.read(p), ai=mock(answer))
    s.confirm_all()
    m = s.matrix("A1B1", numbers=True)
    assert m.shape == (2, 3) and m.loc["OID0", "S2"] == -0.5
    assert "negative" in s.history().set_index("question").loc["log_transformed", "hint"]
    st = s.sample_table()
    assert st.loc["S2", "Group"] == "ctrl"
    assert len(s.feature_metadata()) == 2


def test_long_duplicate_pairs_reported(tmp_path):
    p = tmp_path / "dup.csv"
    p.write_text("SampleID,OlinkID,NPX\nS1,O1,1\nS1,O1,2\nS2,O1,3\n")
    answer = {"layout": {"value": "long", "confidence": 0.9, "evidence": ""},
              "groups": [grp("a", "sample_id", "s", ["SampleID"]), grp("b", "feature_id", "f", ["OlinkID"]),
                         grp("c", "value", "NPX", ["NPX"], assay_label="Olink")],
              "assays": [assay("Olink", ["b"])]}
    s = prism.start(prism.read(p), ai=mock(answer))
    assert any("repeat a protein-sample pair" in w for w in s.todo()["what"])


# ---------------------------------------------------------------- samples in rows, derived design

def test_samples_in_rows_derived_design(tmp_path):
    header = "Sample,Age," + ",".join(f"P{i}" for i in range(5))
    rows = [f"{subj}_{t},{40 + n}," + ",".join(str(n + i) for i in range(5))
            for n, (subj, t) in enumerate([(s_, t_) for s_ in ("A", "B", "C") for t_ in ("w0", "w4")])]
    p = tmp_path / "rows.csv"
    p.write_text("\n".join([header] + rows) + "\n")
    answer = {"layout": {"value": "samples_in_rows", "confidence": 0.9, "evidence": ""},
              "groups": [grp("a", "sample_id", "sample", ["Sample"]),
                         grp("b", "sample_metadata", "age", ["Age"], audit_kind="covariate"),
                         grp("c", "value", "intensity", templates=["P#"], assay_label="X")],
              "assays": [assay("X", [])]}
    s = prism.start(prism.read(p), ai=mock(answer))
    s.confirm_all()
    assert s.matrix("A1B1").shape == (5, 6)
    pats = s.name_patterns()
    assert pats.iloc[0]["delimiter"] == "_"
    d = s.set_design(derive={"delimiter": "_", "occurrence": "last", "left": "subject", "right": "time"},
                     time_unit="weeks")
    assert d["facts"]["n_subjects"] == 3 and d["facts"]["time_values"] == ["w0", "w4"]
    assert s.sample_table().loc["B_w4", "subject"] == "B"
