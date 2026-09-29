"""Literature retrieval (Europe PMC, faked) and grounded AI suggestions."""

import json
import threading
from http.server import HTTPServer

import pytest

import fake_europepmc
from backend import literature, mock_llm


def gid_of(f, col):
    return next(g["group_id"] for g in f.upload["groups"] if col in g["columns"])


def lit(f, queries=None, expect=200):
    r = f.c.post("/api/literature", json={"session_id": f.sid, "queries": queries or []})
    assert r.status_code == expect, r.text
    if expect == 200:
        f.draft = r.json()["draft"]
    return r.json()


def test_queries_come_from_the_description(flow):
    f = flow("A_maxquant_proteinGroups.txt", ai=False)
    qs = f.draft["literature_queries_suggested"]
    assert any('"LFQ intensity"' in q and '"iBAQ"' in q for q in qs)
    assert all("S01" not in q for q in qs)              # names and terms only, never data


def test_literature_suggestions_are_cited_and_verified(flow):
    f = flow("A_maxquant_proteinGroups.txt")
    f.confirm_all_as_proposed()
    lit(f)
    rec = f.draft["literature"]
    assert rec["status"] == "done" and rec["ai_used"]
    assert "90000001" in rec["papers"] and rec["passages"]
    assert any(p["section"] != "abstract" for p in rec["passages"])        # open-access full text was read
    lfq = rec["blocks"][gid_of(f, "LFQ intensity S01")]
    assert lfq["suggested_for_analysis"] == "yes" and lfq["supported"]
    by_id = {p["passage_id"]: p for p in rec["passages"]}
    for c in lfq["citations"]:                                              # every kept quote is really there
        assert literature._norm(c["quote"]) in literature._norm(by_id[c["passage_id"]]["text"])
    out = f.finalize()
    schema = out["schema"]
    assert schema["literature"]["papers"] and schema["literature"]["passages"]
    block = next(b for b in schema["assays"][0]["value_blocks"] if b["group_id"] == gid_of(f, "LFQ intensity S01"))
    assert block["literature"]["suggested_for_analysis"] == "yes"
    assert "block_role" not in block                                       # a suggestion, not a ranking
    assert any(x["topic"] for x in schema["literature"]["for_later_steps"])  # stored for later steps


def test_invented_quote_is_dropped_and_yes_downgraded(flow, monkeypatch):
    def liar(payload):
        return json.dumps({"summary": "s", "summary_citations": [], "for_later_steps": [], "blocks": [
            {"group_id": b["group_id"], "description": "d", "typical_use": "u", "suggested_for_analysis": "yes",
             "reason": "r", "citations": [{"passage_id": "P1", "quote": "this sentence appears in no paper at all"},
                                          {"passage_id": "P99", "quote": "a passage that was never retrieved here"}]}
            for b in payload["file"]["value_blocks"]]})
    monkeypatch.setattr(mock_llm.MockLLM, "literature", staticmethod(liar))
    f = flow("A_maxquant_proteinGroups.txt")
    lit(f)
    rec = f.draft["literature"]
    for b in rec["blocks"].values():
        assert b["suggested_for_analysis"] == "unsure" and not b["citations"] and "Downgraded" in b["reason"]
    reasons = {r["reason"] for r in rec["rejected"]}
    assert {"quote not found in the passage", "no such passage"} <= reasons


def test_ai_off_still_retrieves(flow):
    f = flow("C_mzmine_feature_table.csv", ai=False)
    f.step("layout", {"layout": "samples_in_columns",
                      "assays": [{"assay_label": "LC-MS", "omics_type": "metabolomics", "source_software": "MZmine"}]})
    f.step("values", {"items": [{"group_id": gid_of(f, "S01 Peak area"), "role": "value", "label": "peak area"}]})
    lit(f)
    rec = f.draft["literature"]
    assert rec["status"] == "retrieved_only" and not rec["ai_used"] and rec["passages"]
    assert "90000003" in rec["papers"]


def test_user_queries_and_no_results(flow):
    f = flow("A_maxquant_proteinGroups.txt")
    lit(f, ['"zebrafish" AND "otolith"'])
    assert f.draft["literature"]["status"] == "no_results"
    assert f.draft["literature"]["queries"] == ['"zebrafish" AND "otolith"']


def test_unreachable_service_is_a_clear_error(flow, monkeypatch):
    def down(url, kind):
        raise literature.LiteratureError("Europe PMC could not be reached (URLError: offline).")
    monkeypatch.setattr(literature, "_get", down)
    f = flow("A_maxquant_proteinGroups.txt")
    r = lit(f, expect=422)
    assert "could not be reached" in r["detail"] and "continue without it" in r["detail"]


def test_real_http_client_against_fake_server(monkeypatch, tmp_path):
    """Exercise the actual urllib client + cache against a local fake server."""
    srv = HTTPServer(("127.0.0.1", 0), fake_europepmc.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(literature, "_get", _REAL_GET)
        monkeypatch.setattr(literature, "EUROPEPMC_URL", f"http://127.0.0.1:{srv.server_port}")
        papers = literature.search('"LFQ intensity" AND MaxQuant')
        assert papers[0]["pmid"] == "90000001" and papers[0]["journal"] == "Journal of Test Proteomics"
        assert papers[0]["url"] == "https://europepmc.org/article/MED/90000001"
        paras = literature.full_text("PMC9000001")
        assert [s for s, _ in paras] == ["Materials and methods", "Materials and methods"]   # intro skipped
        assert "iBAQ" in paras[1][1]                                                       # inline tags flattened
        assert literature.full_text("PMC0000000") == []                                    # 404 -> nothing
        assert len(list(literature.CACHE_DIR.glob("*"))) == 2                               # cached (not the 404)
    finally:
        srv.shutdown()


_REAL_GET = literature._get
