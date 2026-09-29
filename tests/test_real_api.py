"""Manual acceptance test against the real AI (v2.1 §10). Skipped by default.

Run with a key in the environment (never commit it):
    PRISM_REAL_API=1 GEMINI_API_KEY=... python -m pytest -s tests/test_real_api.py

For tables outside PRISM's current scope (16S OTU counts, methylation beta
values) the AI must describe them sensibly, flag the scope as 'no' or
'unsure', and the wizard must still complete.
"""

import os
import re

import pytest

_ENV = {k: os.environ.get(k) for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "LLM_PROVIDER", "PRISM_LLM_MODEL",
                                       "ANTHROPIC_API_KEY", "OPENAI_API_KEY")}

pytestmark = pytest.mark.skipif(os.environ.get("PRISM_REAL_API") != "1",
                                reason="set PRISM_REAL_API=1 and an API key to run against the real AI")

CASES = {
    "H_16S_otu_table.tsv": r"16s|microbio|amplicon|otu|taxonom|metagenom",
    "I_methylation_beta.csv": r"methyl|epigen|beta",
}


@pytest.fixture
def real_ai(monkeypatch):
    for k, v in _ENV.items():
        if v:
            monkeypatch.setenv(k, v)
    if not _ENV["LLM_PROVIDER"]:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)


@pytest.mark.parametrize("name", sorted(CASES))
def test_out_of_scope_described_and_wizard_completes(flow, real_ai, name):
    f = flow(name)
    d = f.draft
    assert d["ai"]["used"] and not d["ai"]["error"], d["ai"]
    a = d["assays"][0]
    print(f"\n{name}: {a}")
    described = " ".join(str(a.get(k) or "") for k in ("assay_label", "omics_type", "scope_reason"))
    assert re.search(CASES[name], described, re.I), described
    assert a["in_supported_scope"] in ("no", "unsure")
    block = next(it for it in d["groups"].values() if it["role"] == "value")
    assert block["keep"] and block["label"]
    f.confirm_all_as_proposed()
    for st, it in f.draft["groups"].items():
        assert it["role"] != "unresolved", (st, it)
    out = f.finalize()
    assert any(x["flag"] == "outside_supported_scope" for x in out["integrity_flags"])
