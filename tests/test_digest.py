import json

import prism
from prism.digest import name_template


def write(tmp_path, text, name="t.csv"):
    p = tmp_path / name
    p.write_text(text)
    return p


def metabolomics(tmp_path):
    header = "PatientCode,Group," + ",".join(f"Amino Acid_{i}" for i in range(1, 41)) + ",Lipid_1,Lipid_2"
    rows = [f"PT{100 + r},{'case' if r % 2 else 'control'}," + ",".join(str(1000 + r * 7 + i) for i in range(40))
            + f",{r}.5,{r}.25" for r in range(10)]
    return prism.read(write(tmp_path, "\n".join([header] + rows) + "\n", "secret_patients.csv"))


def test_name_template():
    assert name_template("Amino Acid_35") == "Amino Acid_#"
    assert name_template("S01_rep2") == "S#_rep#"


def test_templates_merge_numeric_columns(tmp_path):
    d = prism.digest(metabolomics(tmp_path))
    t = {x["template"]: x for x in d["templates"]}
    assert t["Amino Acid_#"]["n_columns"] == 40
    assert t["Lipid_#"]["n_columns"] == 2
    assert [c["column"] for c in d["columns"]] == ["PatientCode", "Group"]


def test_no_rows_no_file_name_no_id_values(tmp_path):
    d = prism.digest(metabolomics(tmp_path))
    text = json.dumps(d)
    assert "secret_patients" not in text and d["file"]["extension"] == ".csv"
    assert "PT105" not in text                                  # identifier values are never sent
    code = next(c for c in d["columns"] if c["column"] == "PatientCode")
    assert "values" not in code and code["value_shapes"][0]["shape"] == "AA999"
    group = next(c for c in d["columns"] if c["column"] == "Group")
    assert {v["value"] for v in group["values"]} == {"case", "control"}
    assert d["settings"] == {"example_values_sent": True, "data_rows_sent": False}


def test_examples_off(tmp_path):
    d = prism.digest(metabolomics(tmp_path), examples=False)
    assert all("values" not in c for c in d["columns"])
    assert d["settings"]["example_values_sent"] is False


def test_numeric_facts_and_layout_hint(tmp_path):
    t = prism.read(write(tmp_path, "id,a\nx,1\ny,1000\nz,0\n"))
    d = prism.digest(t, prism.propose_layout(t))
    a = next(c for c in d["columns"] if c["column"] == "a")
    assert a["min"] == 0 and a["max"] == 1000 and a["log10_span"] == 3 and a["integer_valued"]
    assert d["file"]["layout_hint"]["row_id_column"] == "id"
