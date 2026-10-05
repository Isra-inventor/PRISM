"""v2.2: grouping is an AI proposal. Code checks coverage and applies merges /
splits only when the ids and column names exist. Nothing here asserts which
columns 'should' form a family: the (mocked) AI decides that."""

import csv
import io
import json
import random

import pytest

from backend import config, mock_llm


_ORIGINAL = mock_llm.MockLLM.respond
FIXTURES = ["A_maxquant_proteinGroups.txt", "B_diann_pg_matrix.tsv", "C_mzmine_feature_table.csv",
            "D_samples_in_rows_multiomics.csv", "E_somascan_adat_like.csv", "F1_long_unique.csv",
            "H_16S_otu_table.tsv", "I_methylation_beta.csv", "J_subject_code_blocks.csv", "K_somascan_nhp.csv",
            "L_metabolon_like.csv"]


def owners(upload):
    out = {}
    for g in upload["groups"]:
        for c in g["columns"]:
            out.setdefault(c, []).append(g["group_id"])
    return out


def assert_full_coverage(f):
    own = owners(f.upload)
    assert sorted(own) == sorted(f.upload["labels"]), "every column is in a group"
    assert all(len(v) == 1 for v in own.values()), "no column is in two groups"
    assert set(f.draft["groups"]) == {g["group_id"] for g in f.upload["groups"]}


@pytest.mark.parametrize("name", FIXTURES)
def test_every_column_in_exactly_one_group(flow, name):
    assert_full_coverage(flow(name))


@pytest.mark.parametrize("ai", [True, False])
def test_incomplete_answer_leaves_unresolved_singletons(flow, monkeypatch, ai):
    def half(system, prompt):
        base = json.loads(_ORIGINAL(system, prompt))
        if prompt.startswith("Digest"):
            base["groups"] = base["groups"][::2]          # the AI forgets every other group
        return json.dumps(base)
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(half))
    f = flow("A_maxquant_proteinGroups.txt", ai=ai)
    assert_full_coverage(f)
    if ai:
        left = [g for g in f.upload["groups"] if g["origin"] == "unmentioned"]
        assert left and all(g["n_columns"] == 1 for g in left)
        assert all(f.draft["groups"][g["group_id"]]["role"] == "unresolved" for g in left)


def test_double_claims_and_unknown_columns_are_rejected(flow, monkeypatch):
    def greedy(system, prompt):
        base = json.loads(_ORIGINAL(system, prompt))
        if prompt.startswith("Digest"):
            base["groups"].append({"group_id": "dup", "columns": ["Reverse", "not a column"], "role": "feature_annotation",
                                   "label": "again", "confidence": 0.5, "evidence": "x"})
        return json.dumps(base)
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(greedy))
    f = flow("A_maxquant_proteinGroups.txt")
    assert_full_coverage(f)
    reasons = " ".join(r["reason"] for r in f.draft["rejected"])
    assert "already placed" in reasons and "does not exist" in reasons


def test_propose_merge_and_split_apply_only_with_valid_ids(flow, monkeypatch):
    def actions(system, prompt):
        base = json.loads(_ORIGINAL(system, prompt))
        if prompt.startswith("Digest"):
            ids = {(g.get("templates") or g["columns"])[0]: g["group_id"] for g in base["groups"]}
            base["propose_merge"] = [
                {"group_ids": [ids["Intensity S#"], ids["iBAQ S#"]], "reason": "valid ids"},
                {"group_ids": [ids["LFQ intensity S#"], "ghost"], "reason": "one id does not exist"}]
            for g in base["groups"]:
                if g.get("templates") == ["Peptides S#"]:
                    g["suggest_split"] = ["Peptides S06", "Reverse"]    # Reverse is not in this group
                    g["suggest_split_role"] = "ignore"
            base["groups"].append({"group_id": "x", "columns": [], "role": "value", "label": "",
                                   "confidence": 0, "evidence": "", "suggest_split": ["iBAQ S02"]})
            base["groups"].append({"group_id": "y", "templates": ["No such S#"], "role": "value", "label": "",
                                   "confidence": 0, "evidence": ""})
        return json.dumps(base)
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(actions))
    f = flow("A_maxquant_proteinGroups.txt")
    assert_full_coverage(f)
    own = owners(f.upload)
    assert own["Intensity S01"] == own["iBAQ S01"]                         # merged
    assert own["LFQ intensity S01"] != own["Intensity S01"]                  # merge with a ghost id: not applied
    assert f.upload["groups"][[g["group_id"] for g in f.upload["groups"]].index(own["Peptides S06"][0])]["columns"] == ["Peptides S06"]
    assert own["Reverse"] != own["Peptides S01"]
    rej = " ".join(r["reason"] for r in f.draft["rejected"])
    assert "merge names groups that do not exist: ghost" in rej and "not in this group" in rej
    assert "name template does not exist" in rej


def test_digest_lists_templates_first(flow):
    f = flow("A_maxquant_proteinGroups.txt")
    d = f.digests[0]
    tpl = {t["template"]: t for t in d["templates"]}
    assert set(tpl) >= {"Intensity S#", "LFQ intensity S#", "iBAQ S#", "Peptides S#"}
    assert tpl["LFQ intensity S#"]["n_columns"] == 6 and tpl["LFQ intensity S#"]["aggregate"]["median_of_column_medians"]
    singles = {c["column"] for c in d["columns"]}
    assert "LFQ intensity S01" not in singles and {"Protein IDs", "Peptides", "Intensity"} <= singles


def test_name_templates_compress_and_unique_names_fall_back_to_column_chunks(flow, monkeypatch):
    from backend.profiling import Columns, name_templates
    from backend.parsing import parse_bytes
    names = [f"Pt{k:03d}_visit{v}" for k in range(1, 40) for v in (1, 2)]
    tb = parse_bytes("x.csv", (",".join(names) + "\n" + ",".join("1" for _ in names) + "\n").encode())
    (one,) = name_templates(Columns(tb), range(len(names)))
    assert one["template"] == "Pt#_visit#" and one["n_columns"] == len(names)
    words = ["gene" + chr(97 + k // 26) + chr(97 + k % 26) for k in range(40)]   # unique, no digits
    rows = [["x" + str(r)] + [str(r * k + 1) for k in range(40)] for r in range(5)]
    buf = io.StringIO()
    csv.writer(buf).writerows([["id"] + words] + rows)
    monkeypatch.setattr(config, "GROUPING_CHUNK_SIZE", 10)
    f = flow("genes.csv", content=buf.getvalue().encode())
    gr = f.draft["grouping"]
    assert gr["chunk_mode"] == "columns" and gr["chunks"] == 5
    assert_full_coverage(f)


def test_user_merge_and_split(flow):
    f = flow("A_maxquant_proteinGroups.txt")
    a, b = (next(g["group_id"] for g in f.upload["groups"] if c in g["columns"]) for c in ("Intensity S01", "iBAQ S01"))
    assert f.c.post("/api/merge", json={"session_id": f.sid, "group_ids": [a, "ghost"]}).status_code == 422
    r = f.c.post("/api/merge", json={"session_id": f.sid, "group_ids": [a, b]})
    assert r.status_code == 200, r.text
    f.upload, f.draft = r.json()["session"], r.json()["draft"]
    assert_full_coverage(f)
    merged = next(g for g in f.upload["groups"] if "iBAQ S01" in g["columns"])
    assert merged["n_columns"] == 12 and merged["origin"] == "merged_user"
    assert f.draft["groups"][merged["group_id"]]["provenance"] == "user_set"
    r = f.c.post("/api/split", json={"session_id": f.sid, "group_id": merged["group_id"], "columns": ["iBAQ S01"]})
    assert r.status_code == 200, r.text
    f.upload, f.draft = r.json()["session"], r.json()["draft"]
    assert_full_coverage(f)
    alone = next(g for g in f.upload["groups"] if g["columns"] == ["iBAQ S01"])
    assert alone["origin"] == "split_user" and f.draft["groups"][alone["group_id"]]["role"] == "unresolved"


# ---------------------------------------------------------------- the wide, messy file

def _wide_messy(n_samples=2040, n_rows=30, seed=7):
    """Sample columns with no clean shared substring (Pt003_visit1, 004-w1, ...), plus annotations."""
    r = random.Random(seed)
    fmts = [lambda k, v: f"Pt{k:04d}_visit{v}", lambda k, v: f"{k:04d}-w{v}", lambda k, v: f"subj{k}.{v}b",
            lambda k, v: f"{chr(65 + k % 26)}{k}_{v}x", lambda k, v: f"sample {k} wk{v}"]
    samples = [fmts[k % len(fmts)](k, k % 3 + 1) for k in range(n_samples)]
    ann = ["feature", "gene", "mass", "rt", "flag"]
    header = ann[:3] + samples[:1000] + ann[3:] + samples[1000:]
    rows = []
    for i in range(n_rows):
        base = {"feature": f"F{i}", "gene": f"G{i % 7}", "mass": f"{r.uniform(100, 900):.3f}",
                "rt": f"{r.uniform(1, 20):.2f}", "flag": "+" if i % 9 == 0 else ""}
        rows.append([base[h] if h in base else f"{10 ** r.uniform(4, 7):.1f}" for h in header])
    buf = io.StringIO()
    csv.writer(buf).writerows([header] + rows)
    return buf.getvalue().encode(), set(samples), ann


def _chunk_answer(payload, samples, k, leave_out=("rt",)):
    """A chunk's draft: sample templates as one value group (with a chunk-local label)."""
    ents = mock_llm.digest_entries(payload)
    vals = [e["_template"] for e in ents if e.get("_template")]
    others = [e["column"] for e in ents if not e.get("_template") and e["column"] not in leave_out]
    groups = [{"group_id": "v", "templates": vals, "role": "value", "assay_label": f"assay of chunk {k}",
               "label": f"intensity (chunk {k})", "confidence": 0.8, "evidence": "numeric"}] if vals else []
    groups += [{"group_id": f"a{j}", "columns": [n], "role": "feature_id" if n == "feature" else "feature_annotation",
                "label": n, "confidence": 0.8, "evidence": "name"} for j, n in enumerate(others)]
    return {"layout": {"value": "samples_in_columns", "confidence": 0.9, "evidence": "x"},
            "assays": [{"assay_label": f"assay of chunk {k}", "omics_type": "unknown", "in_supported_scope": "unsure",
                        "confidence": 0.5, "evidence": "x"}], "groups": groups}


def _final_answer(payload):
    vals = [g["group_id"] for g in payload["groups"] if g["role"] == "value"]
    final = [{"group_id": "values", "members": vals, "role": "value", "assay_label": "proteins",
              "label": "intensity, apparently linear", "confidence": 0.8, "evidence": "one measurement"}] if vals else []
    final += [{"group_id": f"f{k}", "members": [g["group_id"]], "role": g["role"], "label": g["label"],
               "confidence": 0.8, "evidence": "x"} for k, g in enumerate(payload["groups"])
              if g["role"] not in ("value", "unresolved")]
    fid = [f["group_id"] for f in final if f["role"] == "feature_id"]
    return {"groups": final, "assays": [{"assay_label": "proteins", "omics_type": "proteomics", "in_supported_scope": "yes",
                                         "confidence": 0.7, "evidence": "x", "feature_identity": {"group_ids": fid}}]}


def test_wide_messy_file_templates_chunks_and_final_answer(flow, monkeypatch):
    content, samples, ann = _wide_messy()
    calls = {"digest": 0, "consolidation": 0, "max_units": 0}

    def ai(system, prompt):
        payload = json.loads(prompt.split("\n", 1)[1])
        if prompt.startswith("Consolidation"):
            calls["consolidation"] += 1
            assert {g["chunk"] for g in payload["groups"]} >= {1, 2} and payload["chunk_assays"]
            return json.dumps(_final_answer(payload))
        calls["digest"] += 1
        calls["max_units"] = max(calls["max_units"], len(payload["templates"]) + len(payload["columns"]))
        return json.dumps(_chunk_answer(payload, samples, payload.get("chunk", {}).get("index", 1)))
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(ai))
    monkeypatch.setattr(config, "GROUPING_CHUNK_SIZE", 10)
    f = flow("wide_messy.csv", content=content)
    n_cols = len(f.upload["labels"])
    assert n_cols > 2000
    gr = f.draft["grouping"]
    assert gr["chunk_mode"] == "templates" and gr["n_templates"] == 35 and gr["chunks"] == calls["digest"] >= 4
    assert calls["consolidation"] == 1 and calls["max_units"] <= 10       # chunks of templates, not of columns
    assert_full_coverage(f)
    blocks = [g for g in f.upload["groups"] if f.draft["groups"][g["group_id"]]["role"] == "value"]
    assert len(blocks) == 1 and blocks[0]["n_columns"] == len(samples)  # joined by the final answer
    assert [a["assay_label"] for a in f.draft["assays"]] == ["proteins"]  # assays from the final answer only
    labels = {it["label"] for it in f.draft["groups"].values()} | {a["assay_label"] for a in f.draft["assays"]}
    assert not any("chunk" in (x or "") for x in labels)                  # no chunk-local label survives
    (rt,) = [g for g in f.upload["groups"] if g["columns"] == ["rt"]]
    assert rt["origin"] == "unmentioned" and f.draft["groups"][rt["group_id"]]["role"] == "unresolved"
    q = next(q for q in f.draft["questions"] if q["kind"] == "orphan")    # a question, not a manual role
    assert q["applies_to"] == {"columns": ["rt"]} and q["options"][0]["label"].startswith("Add to 'intensity")
    # the same file again: every chunk call and the final answer come from the disk cache
    before = dict(calls)
    g2 = flow("wide_messy.csv", content=content)
    assert calls == before and g2.draft["ai"]["calls"] == gr["chunks"] + 1


def test_small_files_make_one_call(flow, monkeypatch):
    n = {"calls": 0}

    def count(system, prompt):
        n["calls"] += 1
        return _ORIGINAL(system, prompt)
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(count))
    f = flow("A_maxquant_proteinGroups.txt")
    assert n["calls"] == 1 and f.draft["grouping"]["chunks"] == 1


def test_failed_chunk_and_consolidation_can_be_retried(flow, monkeypatch):
    """Quota errors mid-way: the failed chunk's columns stay (unresolved), chunk drafts lose their
    labels, and both calls can be retried from the wizard."""
    content, samples, ann = _wide_messy(n_samples=400)
    state = {"fail": True, "digest_calls": 0}

    def flaky(system, prompt):
        payload = json.loads(prompt.split("\n", 1)[1])
        if prompt.startswith("Consolidation"):
            if state["fail"]:
                raise mock_llm_error()
            return json.dumps(_final_answer(payload))
        state["digest_calls"] += 1
        if state["fail"] and state["digest_calls"] == 2:
            raise mock_llm_error()
        return json.dumps(_chunk_answer(payload, samples, state["digest_calls"], leave_out=()))
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(flaky))
    monkeypatch.setattr(config, "GROUPING_CHUNK_SIZE", 10)
    f = flow("wide_small.csv", content=content)
    gr = f.draft["grouping"]
    assert gr["chunks"] >= 3 and gr["failed_chunks"] == [2] and gr["consolidation_error"]
    assert_full_coverage(f)
    assert [a["assay_label"] for a in f.draft["assays"]] == ["assay 1"]          # placeholder, not a chunk's assay
    assert not any("chunk" in (it["label"] or "") for it in f.draft["groups"].values())
    left = f.draft["ai_ungrouped"]
    assert left and all(f.draft["groups"][g]["role"] == "unresolved" for g in left)
    state["fail"] = False
    r = f.c.post("/api/reconsider", json={"session_id": f.sid, "group_ids": left, "user_hint": "retry"})
    assert r.status_code == 200, r.text
    r = f.c.post("/api/consolidate", json={"session_id": f.sid})
    assert r.status_code == 200, r.text
    f.upload, f.draft = r.json()["session"], r.json()["draft"]
    assert_full_coverage(f)
    blocks = [g for g in f.upload["groups"] if f.draft["groups"][g["group_id"]]["role"] == "value"]
    assert len(blocks) == 1 and blocks[0]["n_columns"] == len(samples)
    assert f.draft["groups"][blocks[0]["group_id"]]["label"] == "intensity, apparently linear"
    assert [a["assay_label"] for a in f.draft["assays"]] == ["proteins"]
    assert not f.draft["ai_ungrouped"] and f.draft["grouping"]["consolidation_error"] is None


def mock_llm_error():
    from backend.llm_providers import LLMError
    return LLMError("HTTP 429: quota exceeded (test)", 429)
