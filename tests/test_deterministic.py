"""Parsing, digests, shared name parts, chunking, signatures and validation rules (no AI)."""

import csv
import io
import random

import pytest

from backend.format_detect import match_signatures, signature_hint, signature_prefill
from backend.parsing import InputError, parse_bytes, parse_number
from backend.profiling import chunk_columns, make_group, profile_table
from backend.validation import timepoint_detail, validate_group
from conftest import fixture_bytes


def load(name):
    t = parse_bytes(name, fixture_bytes(name))
    cols, affixes, hints = profile_table(t)
    return t, cols, affixes, hints


def col(cols, name, gid="g"):
    """A one-column group record (tests build groups by hand; nothing groups automatically)."""
    return make_group(cols, gid, [cols.labels.index(name)], "test")


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


# ---------------------------------------------------------------- facts, not grouping

def test_shared_name_parts_are_facts_without_thresholds():
    _, cols, aff, _ = load("A_maxquant_proteinGroups.txt")
    a = aff[cols.labels.index("LFQ intensity S01")]
    assert a["shared_prefix"] == [{"text": "LFQ intensity S0", "n_others": 5}]
    _, cols, aff, _ = load("C_mzmine_feature_table.csv")
    a = aff[cols.labels.index("QC_01 Peak area")]
    assert a["shared_suffix"][0] == {"text": "_01 Peak area", "n_others": 1}      # longest first ...
    assert {"text": " Peak area", "n_others": 16} in a["shared_suffix"]           # ... and every level
    # no minimum length or group size: a one-letter overlap is still reported
    _, cols, aff, _ = make(["Pt003_visit1", "004-w1", "P-7"], [["1", "2", "3"], ["4", "5", "6"]])
    assert aff[0]["shared_prefix"] == [{"text": "P", "n_others": 1}]
    assert aff[0]["shared_suffix"] == [{"text": "1", "n_others": 1}]
    assert aff[2]["shared_suffix"] == []


def test_no_grouping_happens_in_profiling():
    import backend.profiling as prof
    for gone in ("build_groups", "_best_family", "_split_deviants", "_profile_clusters"):
        assert not hasattr(prof, gone)


def test_chunks_cover_every_column_and_keep_literal_families_together():
    r = random.Random(3)
    fams = ["LFQ intensity ", "iBAQ ", "Peptides ", "seq."]
    labels = [f"{f}S{k:02d}" for f in fams for k in range(35)] + [f"{r.choice('ABC')}{r.randint(0, 999)}_x{k}" for k in range(40)]
    r.shuffle(labels)
    header = labels
    _, cols, aff, _ = make(header, [["1"] * len(header), ["2"] * len(header)])
    chunks = chunk_columns(range(len(header)), cols.labels, aff, 50)
    flat = [i for c in chunks for i in c]
    assert sorted(flat) == list(range(len(header))) and all(len(c) <= 50 for c in chunks)
    for f in fams:  # 35 columns of one family fit in a 50-column chunk: they stay together
        where = {k for k, c in enumerate(chunks) for i in c if cols.labels[i].startswith(f)}
        assert len(where) == 1, f
    assert chunk_columns(range(10), cols.labels, aff, 150) == [list(range(10))]   # small files: one call


def test_group_record_only_describes_a_given_grouping():
    _, cols, _, _ = load("A_maxquant_proteinGroups.txt")
    idx = [i for i, c in enumerate(cols.labels) if c.startswith("LFQ intensity ")]
    g = make_group(cols, "g1", idx, "ai_proposed")
    assert g["kind"] == "numeric_block" and g["pattern"] == {"side": "prefix", "text": "LFQ intensity S0"}
    assert g["sample_names"][0] == "S01"          # the common text is cut back to a separator
    mixed = make_group(cols, "g2", [cols.labels.index("Gene names"), cols.labels.index("Score")], "test")
    assert mixed["kind"] == "column_group" and mixed["type"] == "mixed"


def test_layout_hints_are_facts():
    _, _, _, hints = load("D_samples_in_rows_multiomics.csv")
    assert hints["rows_to_numeric_columns_ratio"] < 1
    _, _, _, hints = load("A_maxquant_proteinGroups.txt")
    assert hints["rows_to_numeric_columns_ratio"] >= 1


# ---------------------------------------------------------------- signatures

def test_signature_requires_all_columns():
    assert match_signatures(["Protein IDs", "Reverse"]) == ["maxquant_proteinGroups"]
    assert match_signatures(["Protein IDs"]) == []
    assert match_signatures(["protein ids", "reverse"]) == []


def test_maxquant_prefill_is_manual_starting_point():
    t, cols, _, _ = load("A_maxquant_proteinGroups.txt")
    p = signature_prefill(t["header"], cols)
    owner = {i: pg for pg in p["groups"].values() for i in pg["indices"]}
    assert sorted(owner) == list(range(len(t["header"])))                   # every column exactly once
    item = lambda c: owner[t["header"].index(c)]["item"]
    rev, con = item("Reverse"), item("Potential contaminant")
    assert rev["role"] == con["role"] == "feature_annotation"
    assert rev["marks_rows_as_suspect"] is True and con["marks_rows_as_suspect"] is True
    lfq, pep = owner[t["header"].index("LFQ intensity S01")], owner[t["header"].index("Peptides S01")]
    assert lfq["item"]["role"] == pep["item"]["role"] == "value" and "block_role" not in lfq["item"]
    assert len(lfq["indices"]) == 6 and owner[t["header"].index("Intensity")] is not owner[t["header"].index("Intensity S01")]
    assert all(pg["item"]["source"] in ("computed", "none") for pg in p["groups"].values())
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
    t, cols, _, _ = load("D_samples_in_rows_multiomics.csv")
    assert validate_group({"role": "value"}, col(cols, "visit"), cols)["status"] == "contradicted"
    proteins = make_group(cols, "p", [i for i, c in enumerate(cols.labels) if c[:1] in "PQO" and c[1:].isdigit()], "test")
    assert validate_group({"role": "value"}, proteins, cols)["status"] == "ok"
    assert validate_group({"role": "sample_id"}, col(cols, "subject_id"), cols)["status"] == "warning"   # repeats
    assert validate_group({"role": "sample_id"}, col(cols, "sample_id"), cols)["status"] == "ok"


def test_closed_fields_must_be_in_their_sets():
    t, cols, _, _ = load("D_samples_in_rows_multiomics.csv")
    visit = col(cols, "visit")
    assert validate_group({"role": "sample_metadata", "audit_kind": "timepoint"}, visit, cols)["status"] == "ok"
    assert validate_group({"role": "sample_metadata", "audit_kind": "foo"}, visit, cols)["status"] == "contradicted"
    assert validate_group({"role": "descriptor"}, visit, cols)["status"] == "contradicted"
    # free-text labels are never checked
    assert validate_group({"role": "sample_metadata", "audit_kind": "timepoint", "label": "anything at all"},
                          visit, cols)["status"] == "ok"


def test_timepoint_detail_is_computed():
    t, cols, _, _ = load("D_samples_in_rows_multiomics.csv")
    assert timepoint_detail(cols.digests[cols.labels.index("visit")]).startswith("ordinal label")
    assert timepoint_detail(cols.digests[cols.labels.index("age")]) == "numeric"
