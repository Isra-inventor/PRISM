"""v4 stage 2: the step registry, the evidence files and `prism evidence check`."""

import copy
import json

import pytest

from prism import registry, cli


def test_registry_validates_and_has_the_17_steps():
    ids = registry.step_ids()
    assert ids == ["G1", "G2", "G3", "G4", "G5", "G6", "G7"] + [f"S{i}" for i in range(1, 11)]
    assert registry.step("S5")["kind"] == "constraint"
    assert registry.step("S6")["implemented"] is False and registry.step("S7")["implemented"] is False
    for s in registry.steps():
        assert s["tolerates_missing"] in ("yes", "no", "native", "not_applicable")   # YAML 1.1 yes/no stay strings
    g = registry.grid()
    assert sum(len(v) for v in g.values()) == 17
    assert {x["id"] for x in g["global|action"]} == {"G3", "G5", "G6"}
    assert {x["id"] for x in g["study_dependent|audit"]} == {"S8", "S9", "S10"}


def test_evidence_check_passes_and_cli():
    res = registry.check()
    assert res["ok"], res["problems"]
    assert cli.main(["evidence", "check"]) == 0
    assert cli.main(["registry", "show"]) == 0


def test_every_evidence_key_exists():
    keys = set(registry.evidence())
    for s in registry.steps():
        assert set(s["evidence"]) <= keys, s["id"]


def test_schema_rejects_bad_steps():
    schema = json.loads((registry.HERE / "steps.schema.json").read_text())
    doc = copy.deepcopy(registry.registry())
    doc["steps"][0]["kind"] = "transform"
    del doc["steps"][1]["evidence_level"]
    errs = registry.validate(doc, schema)
    assert any("kind" in e for e in errs) and any("evidence_level" in e for e in errs)


def test_check_fails_on_missing_level_and_unknown_key(monkeypatch):
    doc = copy.deepcopy(registry.registry())
    doc["steps"][2]["evidence_level"] = "folklore"
    doc["steps"][3]["evidence"] = ["nobody2099"]
    monkeypatch.setitem(registry._cache, "steps", doc)
    res = registry.check()
    assert not res["ok"]
    assert any("folklore" in p for p in res["problems"]) and any("nobody2099" in p for p in res["problems"])


def test_check_fails_on_parameter_without_level(monkeypatch):
    from prism.audit import params
    d = params.defaults()
    monkeypatch.setattr(params, "defaults", lambda: dict(d, brand_new_cutoff=1))
    res = registry.check()
    assert any("brand_new_cutoff" in p for p in res["problems"])
