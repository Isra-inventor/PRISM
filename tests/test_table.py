"""Step 1 tests: reading a table and describing its columns."""

import pytest

import prism


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_reads_a_csv_and_describes_columns(tmp_path):
    p = write(tmp_path, "a.csv", "ID,Gene,S1,S2\n001,MARCH1,10.5,NA\n002,1E5,,3\n003,TP53,7,8\n")
    t = prism.read(p)
    assert t.shape == (3, 4) and t.delimiter == ","
    cols = {c.name: c for c in t.columns}
    assert cols["ID"].kind == "numeric" and t.data["ID"].tolist() == ["001", "002", "003"]   # kept as written
    assert cols["Gene"].kind == "text" and t.data["Gene"][1] == "1E5"                        # never a number
    assert cols["S1"].kind == "numeric" and cols["S1"].n_missing == 1
    assert cols["S2"].n_missing == 1 and cols["S2"].example == "3"


def test_detects_tab_and_semicolon(tmp_path):
    assert prism.read(write(tmp_path, "a.tsv", "id\tx\n1\t2\n")).delimiter == "\t"
    assert prism.read(write(tmp_path, "b.csv", "id;x\n1;2\n")).delimiter == ";"


def test_empty_column_and_summary(tmp_path):
    t = prism.read(write(tmp_path, "c.csv", "a,b\n1,\n2,NA\n"))
    assert t.counts() == {"numeric": 1, "text": 0, "empty": 1}
    s = t.summary()
    assert list(s.columns) == ["name", "kind", "n_missing", "n_unique", "example"]
    assert "2 rows x 2 columns" in repr(t) and "<table" in t._repr_html_()


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        prism.read("no_such_file.csv")
