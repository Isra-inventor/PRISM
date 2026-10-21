"""Joint nodes (prism/rules/joint_nodes.yaml). D2 decides batch correction together with the
missingness handling (v4 5.9, 6.2)."""

from __future__ import annotations

from pathlib import Path

import yaml

TOLERATES_MISSING = {"batch_center": True, "drift_sample_loess": True, "batch_combat": False}


def nodes():
    p = Path(__file__).resolve().parent.parent / "rules" / "joint_nodes.yaml"
    return {n["id"]: n for n in yaml.safe_load(p.read_text(encoding="utf-8"))["nodes"]}


def resolve_d2(any_missing, method, complete_share=None):
    """-> {case, status, order, alternatives, text, evidence_level}."""
    node = nodes()["D2"]
    facts = {"any_missing": bool(any_missing), "method_tolerates_missing": TOLERATES_MISSING.get(method, False)}
    for c in node["cases"]:
        if all(facts.get(k) == v for k, v in c["when"].items()):
            out = dict(c["then"], case=c["id"], node="D2", method=method, evidence_level=node["evidence_level"],
                       evidence=node["evidence"])
            if complete_share is not None:
                out["share_complete_features"] = complete_share
            return out
    return {"case": None, "node": "D2", "status": "Insufficient metadata", "order": [], "method": method,
            "text": "no D2 case matches", "evidence_level": node["evidence_level"]}
