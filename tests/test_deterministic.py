"""Parsing, digests, grouping, signatures and validation rules (no AI)."""

import csv
import io
import random

import pytest

from backend.format_detect import match_signatures, signature_prefill
from backend.parsing import InputError, parse_bytes, parse_number
from backend.profiling import profile_table
from backend.validation import validate_group
from conftest import fixture_bytes


def load(name):
    t = parse_bytes(name, fixture_bytes(name))
    cols, groups, hints = profile_table(t)
    return t, cols, groups, hints


def blocks(groups):
    return {(g["pattern"] or {}).get("text"): g for g in groups if g["kind"] == "numeric_block"}


def by_col(groups, name):
    return next(g for g in groups if name in g["columns"])


def make(header, rows, name="x.csv"):
    buf = io.StringIO()
    csv.writer(buf).writerows([header] + rows)
    t = parse_bytes(name, buf.getvalue().encode())
    return (t,) + profile_table(t)


# ---------------------------------------------------------------- parsing

def test_messy_file_is_reported_not_fixed():
    t = parse_bytes("G_messy_parsing.csv", fixture_bytes("G_messy_parsing.csv"))
    r = t["parse_report"]
    assert r["delimiter"] == "semicolon"
    assert r["duplicate_column_names"] == [{"name": "value", "positions": [2, 3]}]
    assert r["empty_or_unnamed_headers"] == [4]
    assert r["fully_empty_rows"] == 1
    assert "value [column 2]" in r["decimal_comma_columns"]
    assert r["whitespace_padded_columns"][0]["column"] == "note"
    assert {"(empty)", "NA", "N/A", "#N/A", "Filtered", "-"} <= set(r["missing_value_tokens"])
    assert t["header"] == ["id", "value", "value", "", "note"]          # header order kept, not renamed
    assert t["rows"][0][1] == "1,5" and t["rows"][1][4] == " padded "  # values unchanged


def test_zero_is_not_missing():
    assert parse_number("0") == 0.0
    assert parse_number("NA") is None and parse_number("") is None
    _, cols, groups, _ = make(["id", "v"], [["a", "0"], ["b", "NA"], ["c", "5"]])
    d = cols.digests[1]
    assert d["frac_na"] == pytest.approx(1 / 3) and d["frac_zero"] == 0.5


@pytest.mark.parametrize("name,content", [("run.raw", b"\x00\x01"), ("x.xlsx", b"PK\x03\x04"),
                                          ("s.mzML", b"<mzML/>"), ("one.csv", b"justonecolumn\n1\n2\n")])
def test_rejects_non_tables(name, content):
    with pytest.raises(InputError):
        parse_bytes(name, content)


def test_text_digest_hides_identifiers():
    _, cols, _, _ = make(["id", "grp"], [[f"P{i}", "a" if i % 2 else "b"] for i in range(50)])
    assert "values" not in cols.digests[0]                 # 50 unique ids: no examples
    assert {v["value"] for v in cols.digests[1]["values"]} == {"a", "b"}


# ---------------------------------------------------------------- grouping

def test_maxquant_families():
    _, cols, groups, _ = load("A_maxquant_proteinGroups.txt")
    b = blocks(groups)
    assert set(b) == {"Intensity ", "LFQ intensity ", "iBAQ ", "Peptides "}
    assert all(g["n_columns"] == 6 for g in b.values())
    assert b["LFQ intensity "]["sample_names"][0] == "S01"
    assert b["Peptides "]["profile"]["integer_valued"] is True
    assert by_col(groups, "Intensity")["kind"] == "single_column"   # the total column is not a sample


def test_qc_and_blank_stay_in_the_sample_block():
    _, _, groups, _ = load("C_mzmine_feature_table.csv")
    g = by_col(groups, "blank_01 Peak area")
    assert g["n_columns"] == 17 and "QC_01" in g["sample_names"]
    assert by_col(groups, "row ID")["kind"] == "single_column"     # 'row ' is not a measurement family


def test_samples_in_rows_blocks_and_covariates():
    _, _, groups, hints = load("D_samples_in_rows_multiomics.csv")
    nb = [g for g in groups if g["kind"] == "numeric_block"]
    assert sorted(g["n_columns"] for g in nb) == [21, 25]           # metabolites(+iron) and proteins
    for c in ("age", "CD4_count", "batch"):
        assert by_col(groups, c)["kind"] == "single_column"
    assert hints["rows_to_block_columns_ratio"] < 1


def test_somascan_technical_columns_not_merged():
    _, _, groups, _ = load("E_somascan_adat_like.csv")
    feat = next(g for g in groups if g["n_columns"] == 1500)
    assert feat["pattern"]["text"] == "seq."
    for c in ("PlateScale_Scalar", "HybControlNormScale", "NormScale_20", "SlideId"):
        assert c not in feat["columns"]


def test_deviant_split_and_fragpipe_families():
    r = random.Random(0)
    samples = ["Sample_01", "Sample_02", "Sample_03", "Sample_04"]
    fams = [" Intensity", " MaxLFQ Intensity", " Spectral Count"]
    header = ["Protein"] + [s + f for f in fams for s in samples]
    rows = [[f"P{k}"] + [r.randint(0, 30) if "Count" in c else round(10 ** r.uniform(5, 8), 1) for c in header[1:]]
            for k in range(30)]
    _, _, groups, _ = make(header, rows)
    assert {(g["pattern"] or {}).get("text") for g in groups if g["kind"] == "numeric_block"} == set(fams)

    header = ["id"] + [f"conc_{i}" for i in range(8)]
    rows = [[k] + [round(r.uniform(1, 9), 2) for _ in range(7)] + [round(r.uniform(1e5, 1e6), 1)] for k in range(30)]
    _, _, groups, _ = make(header, rows)
    assert by_col(groups, "conc_7")["origin"] == "split_from_family"


# ---------------------------------------------------------------- signatures

def test_signature_requires_all_columns():
    assert match_signatures(["Protein IDs", "Reverse"]) == ["maxquant_proteinGroups"]
    assert match_signatures(["Protein IDs"]) == []
    assert match_signatures(["protein ids", "reverse"]) == []


def test_maxquant_prefill():
    t, cols, groups, _ = load("A_maxquant_proteinGroups.txt")
    p = signature_prefill(t["header"], groups)
    g = p["groups"]
    assert g[by_col(groups, "Reverse")["group_id"]]["kind"] == "flag_decoy"
    assert g[by_col(groups, "Potential contaminant")["group_id"]]["kind"] == "flag_contaminant"
    assert g[by_col(groups, "LFQ intensity S01")["group_id"]]["block_role"] == "primary"
    assert g[by_col(groups, "Peptides S01")["group_id"]]["measurement_type"] == "count"
    assert p["layout"]["value"] == "samples_in_columns"


def test_mzmine_headers_do_not_match_v1_signature():
    t = parse_bytes("C_mzmine_feature_table.csv", fixture_bytes("C_mzmine_feature_table.csv"))
    assert match_signatures(t["header"]) == []


# ---------------------------------------------------------------- validation rules

def _item(**k):
    base = {"role": "value", "kind": None, "measurement_type": "intensity", "scale": "linear", "label": ""}
    base.update(k)
    return base


def test_count_on_non_integers_is_contradicted():
    t, cols, groups, _ = load("A_maxquant_proteinGroups.txt")
    g = blocks(groups)["LFQ intensity "]
    res = validate_group(_item(measurement_type="count"), dict(g, profile=dict(g["profile"], integer_valued=False)), cols)
    assert res["status"] == "contradicted"
    assert validate_group(_item(measurement_type="count"), blocks(groups)["Peptides "], cols)["status"] == "ok"


def test_scale_warnings():
    t, cols, groups, _ = load("B_diann_pg_matrix.tsv")
    g = next(x for x in groups if x["n_columns"] == 8)
    assert validate_group(_item(scale="log2"), g, cols)["status"] == "ok"
    assert validate_group(_item(scale="linear"), g, cols)["status"] == "warning"
    _, cols2, groups2, _ = load("A_maxquant_proteinGroups.txt")
    assert validate_group(_item(scale="log2"), blocks(groups2)["LFQ intensity "], cols2)["status"] == "warning"


def test_value_role_on_text_is_contradicted_and_id_checks():
    t, cols, groups, _ = load("D_samples_in_rows_multiomics.csv")
    assert validate_group({"role": "value"}, by_col(groups, "visit"), cols)["status"] == "contradicted"
    subj = by_col(groups, "subject_id")
    assert validate_group({"role": "sample_id"}, subj, cols)["status"] == "warning"   # repeats
    samp = by_col(groups, "sample_id")
    res = validate_group({"role": "sample_metadata", "kind": "subject_id"}, samp, cols)
    assert "sample ID" in res["messages"][0]
    visit = validate_group({"role": "sample_metadata", "kind": "timepoint", "detail": "date"}, by_col(groups, "visit"), cols)
    assert visit["status"] == "warning"
    proteins = next(g for g in groups if g["n_columns"] == 25)
    assert validate_group({"role": "value", "measurement_type": "proportion_or_relative_abundance"},
                          proteins, cols)["status"] == "contradicted"
