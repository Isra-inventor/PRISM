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
            "H_16S_otu_table.tsv", "I_methylation_beta.csv"]


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
            ids = {g["columns"][0]: g["group_id"] for g in base["groups"]}
            base["propose_merge"] = [
                {"group_ids": [ids["Intensity S01"], ids["iBAQ S01"]], "reason": "valid ids"},
                {"group_ids": [ids["LFQ intensity S01"], "ghost"], "reason": "one id does not exist"}]
            for g in base["groups"]:
                if g["columns"][0] == "Peptides S01":
                    g["suggest_split"] = ["Peptides S06", "Reverse"]    # Reverse is not in this group
                    g["suggest_split_role"] = "ignore"
            base["groups"].append({"group_id": "x", "columns": [], "role": "value", "label": "",
                                   "confidence": 0, "evidence": "", "suggest_split": ["iBAQ S02"]})
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


def test_user_merge_check_merge_and_split(flow):
    f = flow("A_maxquant_proteinGroups.txt")
    a, b = (next(g["group_id"] for g in f.upload["groups"] if c in g["columns"]) for c in ("Intensity S01", "iBAQ S01"))
    r = f.c.post("/api/merge-check", json={"session_id": f.sid, "group_ids": [a, b], "user_hint": "same thing"})
    assert r.status_code == 200 and r.json()["agrees"] is True                 # the AI's opinion only
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


def test_wide_messy_file_chunking_consolidation_and_coverage(flow, monkeypatch):
    content, samples, ann = _wide_messy()
    calls = {"digest": 0, "consolidation": 0, "max_cols": 0}

    def ai(system, prompt):
        payload = json.loads(prompt.split("\n", 1)[1])
        if prompt.startswith("Consolidation"):
            calls["consolidation"] += 1
            vals = [g["group_id"] for g in payload["groups"] if g["role"] == "value"]
            return json.dumps({"cross_chunk_merges": [{"group_ids": vals, "reason": "same measurement, split by chunking"}]})
        calls["digest"] += 1
        names = [c["column"] for c in payload["columns"]]
        calls["max_cols"] = max(calls["max_cols"], len(names))
        vals = [n for n in names if n in samples]
        others = [n for n in names if n not in samples]
        groups = [{"group_id": "v", "columns": vals[1:], "role": "value", "assay_label": "assay",
                   "label": "intensity", "confidence": 0.8, "evidence": "numeric, similar medians"}] if vals else []
        groups += [{"group_id": f"a{k}", "columns": [n], "role": "feature_id" if n == "feature" else "feature_annotation",
                    "label": n, "confidence": 0.8, "evidence": "name"} for k, n in enumerate(others)]
        # vals[0] is left out on purpose: it must come back as an unresolved singleton
        return json.dumps({"layout": {"value": "samples_in_columns", "confidence": 0.9, "evidence": "x"},
                           "assays": [{"assay_label": "assay", "omics_type": "unknown", "in_supported_scope": "unsure",
                                       "confidence": 0.5, "evidence": "x"}],
                           "groups": groups})
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(ai))
    f = flow("wide_messy.csv", content=content)
    n_cols = len(f.upload["labels"])
    assert n_cols > 2000
    n_chunks = f.draft["grouping"]["chunks"]
    assert n_chunks >= -(-n_cols // config.GROUPING_CHUNK_SIZE) and calls["digest"] == n_chunks
    assert calls["consolidation"] == 1 and calls["max_cols"] <= config.GROUPING_CHUNK_SIZE
    assert_full_coverage(f)
    blocks = [g for g in f.upload["groups"] if f.draft["groups"][g["group_id"]]["role"] == "value"]
    assert len(blocks) == 1 and blocks[0]["n_columns"] == len(samples) - n_chunks   # joined across chunks
    left = [g for g in f.upload["groups"] if g["origin"] == "unmentioned"]
    assert len(left) == n_chunks and all(f.draft["groups"][g["group_id"]]["role"] == "unresolved" for g in left)

    # the same file again: every chunk call and the consolidation come from the disk cache
    before = dict(calls)
    g2 = flow("wide_messy.csv", content=content)
    assert calls == before and g2.draft["ai"]["calls"] == n_chunks + 1


def test_small_files_make_one_call(flow, monkeypatch):
    n = {"calls": 0}

    def count(system, prompt):
        n["calls"] += 1
        return _ORIGINAL(system, prompt)
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(count))
    f = flow("A_maxquant_proteinGroups.txt")
    assert n["calls"] == 1 and f.draft["grouping"]["chunks"] == 1


def test_failed_chunk_and_consolidation_can_be_retried(flow, monkeypatch):
    """Quota errors mid-way: the failed chunk's columns stay (unresolved), the failure is
    recorded, and both calls can be retried from the wizard."""
    content, samples, ann = _wide_messy(n_samples=400)
    state = {"fail": True, "digest_calls": 0}

    def flaky(system, prompt):
        payload = json.loads(prompt.split("\n", 1)[1])
        if prompt.startswith("Consolidation"):
            if state["fail"]:
                raise mock_llm_error()
            vals = [g["group_id"] for g in payload["groups"] if g["role"] == "value"]
            return json.dumps({"cross_chunk_merges": [{"group_ids": vals, "reason": "one measurement"}]})
        state["digest_calls"] += 1
        if state["fail"] and state["digest_calls"] == 2:
            raise mock_llm_error()
        names = [c["column"] for c in payload["columns"]]
        vals = [n for n in names if n in samples]
        groups = [{"group_id": "v", "columns": vals, "role": "value", "assay_label": "a", "label": "intensity",
                   "confidence": 0.8, "evidence": "x"}] if vals else []
        groups += [{"group_id": f"o{k}", "columns": [n], "role": "feature_annotation", "label": n,
                    "confidence": 0.8, "evidence": "x"} for k, n in enumerate(n for n in names if n not in samples)]
        return json.dumps({"layout": {"value": "samples_in_columns", "confidence": 0.9, "evidence": "x"},
                           "assays": [], "groups": groups})
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(flaky))
    monkeypatch.setattr(config, "GROUPING_CHUNK_SIZE", 150)
    f = flow("wide_small.csv", content=content)
    gr = f.draft["grouping"]
    assert gr["chunks"] >= 3 and gr["failed_chunks"] == [2] and gr["consolidation_error"]
    assert_full_coverage(f)
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
    assert not f.draft["ai_ungrouped"] and f.draft["grouping"]["consolidation_error"] is None


def mock_llm_error():
    from backend.llm_providers import LLMError
    return LLMError("HTTP 429: quota exceeded (test)", 429)
