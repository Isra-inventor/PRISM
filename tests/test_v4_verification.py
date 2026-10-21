"""v4 stage 11: verification at C2 (descriptive re-measurement; never a ranking)."""

import ast
import inspect

import numpy as np
import pytest

from prism import ports
from prism.audit import engine, verification
from prism.plan import engine as pe, intake
from test_v4_plan import batch_session, entry


@pytest.mark.skipif(not ports.enabled(), reason="S2 ops need the proBatch port")
def test_c2_verification_before_after(isolated, tmp_path):
    st = batch_session(tmp_path, "verify")
    intake.set_intake(st, {"contrast_variable": "group"})
    for group in ("G5", "S1", "S2"):
        plan = pe.build(st)
        out = pe.approve(st, entry(plan, group)["entry_id"], 0)
    man = engine.show(st, out["run_id"])
    v = next(i for i in man["findings"] if i["audit_id"] == "V")
    f = engine.finding(st, out["run_id"], v["file"])
    assert f["checkpoint"] == "C2"
    M = f["measures"]
    assert set(M["comparisons"]) == {"S0_vs_C2", "C1_vs_C2"}
    t = {r["measure"]: r for r in M["comparisons"]["C1_vs_C2"]["table"]}
    assert t["permanova_r2[batch]"]["after"] < t["permanova_r2[batch]"]["before"]
    assert t["batch_effect_on_correlation[batch]"]["after"] < t["batch_effect_on_correlation[batch]"]["before"]
    assert M["negative_controls"]["no_batch_clustering"] is True
    assert M["comparisons"]["S0_vs_C2"]["nan_unchanged_except_removals"]
    assert "never used to rank or select" in M["use"]


def test_verification_exposes_no_ranking():
    """No function of the verification module returns a method name or a ranking."""
    src = inspect.getsource(verification)
    tree = ast.parse(src)
    names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    for n in names:
        assert not any(w in n.lower() for w in ("rank", "select", "best", "choose", "winner", "recommend")), n
    import sys
    str_types = (ast.Constant,) if sys.version_info >= (3, 8) else (ast.Str,)

    def sval(n):
        v = getattr(n, "value", None) if sys.version_info >= (3, 8) else getattr(n, "s", None)
        return v if isinstance(n, str_types) and isinstance(v, str) else None

    docs = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.FunctionDef))
            and n.body and isinstance(n.body[0], ast.Expr) and sval(n.body[0].value) is not None}
    strings = [sval(n) for n in ast.walk(tree) if sval(n) is not None and id(n) not in docs]
    assert strings
    idents = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)] + \
             [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
    from prism.executor import op_names
    assert not set(op_names()) & set(strings)                  # no method name is ever produced
    for word in ("ranking", "winner", "best_method", "selected_method", "rank"):
        assert not any(word in x.lower() for x in idents), word
        assert not any(x.lower().startswith(word) for x in strings), word
