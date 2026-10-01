"""Literature module (kept for a later Tier 2 feature; not part of Step 0)."""

import json
import threading
from http.server import HTTPServer

import pytest

import fake_europepmc
from backend import literature, mock_llm


def gid_of(f, col):
    return next(g["group_id"] for g in f.upload["groups"] if col in g["columns"])


def test_step0_has_no_literature(flow):
    """v2.3: literature is deferred to a later Tier 2 feature; Step 0 neither calls it nor outputs it."""
    f = flow("A_maxquant_proteinGroups.txt")
    assert f.c.post("/api/literature", json={"session_id": f.sid}).status_code in (404, 405)
    assert not any(k.startswith("literature") for k in f.draft)
    f.confirm_all_as_proposed()
    out = f.finalize()
    schema = out["schema"]
    text = json.dumps(schema)
    assert "literature" not in schema and "for_later_steps" not in text and "suggested_for_analysis" not in text
    assert all("literature" not in b for a in schema["assays"] for b in a["value_blocks"])
    assert "literature_queries" not in json.dumps(f.digests)


def test_quote_verification_is_kept_for_later():
    """The module stays (Tier 2): a quote must appear word for word in its passage."""
    passages = {"P1": {"passage_id": "P1", "paper_id": "1", "text": "LFQ intensity values were used for all comparisons."}}
    ok, bad = literature.verify([literature.Cite(passage_id="P1", quote="LFQ intensity values were used"),
                                 literature.Cite(passage_id="P1", quote="iBAQ values were never used here"),
                                 literature.Cite(passage_id="P9", quote="a passage that does not exist")], passages)
    assert [c["passage_id"] for c in ok] == ["P1"]
    assert {b["reason"] for b in bad} == {"quote not found in the passage", "no such passage"}


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
