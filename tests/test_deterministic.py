"""Parsing, digests, grouping, signatures and validation rules (no AI)."""

import csv
import io
import random

import pytest

from backend.format_detect import match_signatures, signature_hint, signature_prefill
from backend.parsing import InputError, parse_bytes, parse_number
from backend.profiling import profile_table
from backend.validation import timepoint_detail, validate_group
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


def test_maxquant_prefill_is_manual_starting_point():
    t, cols, groups, _ = load("A_maxquant_proteinGroups.txt")
    p = signature_prefill(t["header"], groups)
    g = p["groups"]
    rev, con = g[by_col(groups, "Reverse")["group_id"]], g[by_col(groups, "Potential contaminant")["group_id"]]
    assert rev["role"] == con["role"] == "feature_annotation"
    assert rev["marks_rows_as_suspect"] is True and con["marks_rows_as_suspect"] is True
    lfq, pep = g[by_col(groups, "LFQ intensity S01")["group_id"]], g[by_col(groups, "Peptides S01")["group_id"]]
    assert lfq["role"] == pep["role"] == "value" and "block_role" not in lfq   # described, never ranked
    assert all(it["source"] == "computed" for it in g.values())
    assert p["layout"]["value"] == "samples_in_columns"
    assert p["assays"][0]["in_supported_scope"] == "yes"


def test_signature_hint_is_one_sentence():
    t, _, _, _ = load("A_maxquant_proteinGroups.txt")
    assert signature_hint(t["header"]) == "headers match the known MaxQuant proteinGroups.txt pattern"
    assert signature_hint(["a", "b"]) is None


def test_mzmine_headers_do_not_match_v1_signature():
    t = parse_bytes("C_mzmine_feature_table.csv", fixture_bytes("C_mzmine_feature_table.csv"))
    assert match_signatures(t["header"]) == []


# ---------------------------------------------------------------- validation rules (structure only)

def test_value_role_on_text_is_contradicted_and_id_checks():
    t, cols, groups, _ = load("D_samples_in_rows_multiomics.csv")
    assert validate_group({"role": "value"}, by_col(groups, "visit"), cols)["status"] == "contradicted"
    proteins = next(g for g in groups if g["n_columns"] == 25)
    assert validate_group({"role": "value"}, proteins, cols)["status"] == "ok"
    subj = by_col(groups, "subject_id")
    assert validate_group({"role": "sample_id"}, subj, cols)["status"] == "warning"   # repeats: warning only
    assert validate_group({"role": "sample_id"}, by_col(groups, "sample_id"), cols)["status"] == "ok"


def test_closed_fields_must_be_in_their_sets():
    t, cols, groups, _ = load("D_samples_in_rows_multiomics.csv")
    visit = by_col(groups, "visit")
    assert validate_group({"role": "sample_metadata", "audit_kind": "timepoint"}, visit, cols)["status"] == "ok"
    assert validate_group({"role": "sample_metadata", "audit_kind": "foo"}, visit, cols)["status"] == "contradicted"
    assert validate_group({"role": "descriptor"}, visit, cols)["status"] == "contradicted"
    # free-text labels are never checked
    assert validate_group({"role": "sample_metadata", "audit_kind": "timepoint", "label": "anything at all"},
                          visit, cols)["status"] == "ok"


def test_timepoint_detail_is_computed():
    t, cols, groups, _ = load("D_samples_in_rows_multiomics.csv")
    assert timepoint_detail(cols.digests[by_col(groups, "visit")["indices"][0]]).startswith("ordinal label")
    assert timepoint_detail(cols.digests[by_col(groups, "age")["indices"][0]]) == "numeric"


def test_out_of_scope_tables_group_into_one_block():
    """Sample names carrying a design code (Stool.D0.A, Stool.D7.A, ...) are one block,
    not one family per day; sparse count columns are not split as deviants."""
    _, _, groups, _ = load("H_16S_otu_table.tsv")
    nb = [g for g in groups if g["kind"] == "numeric_block"]
    assert len(nb) == 1 and nb[0]["n_columns"] == 12
    _, _, groups, _ = load("I_methylation_beta.csv")
    nb = [g for g in groups if g["kind"] == "numeric_block"]
    assert len(nb) == 1 and nb[0]["n_columns"] == 12
