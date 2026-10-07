"""v2.4 stage 8: scripts/diff_schemas.py and scripts/replay_digest.py."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_diff_schemas_reports_a_column_that_disappeared(flow, tmp_path):
    f = flow("C_mzmine_feature_table.csv")
    f.confirm_all_as_proposed()
    old = f.finalize()["schema"]
    new = json.loads(json.dumps(old))
    new["feature_annotations"] = [x for x in new["feature_annotations"] if x["column"] != "formula"]
    new["files"][0]["sha256"] = "0" * 64
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(old))
    b.write_text(json.dumps(new))
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "diff_schemas.py"), str(a), str(b)],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "- [main] formula" in r.stdout and "files[main].sha256" in r.stdout
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "diff_schemas.py"), str(a), str(a)],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "No differences" in r.stdout


def test_replay_digest_with_the_mock(flow, isolated):
    f = flow("C_mzmine_feature_table.csv")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "replay_digest.py"), f.sid, "--provider", "mock",
                        "--sessions-dir", str(isolated / "sessions"), "--logs-dir", str(isolated / "logs")],
                       capture_output=True, text=True, env={"PATH": "", "PRISM_CACHE_DIR": str(isolated / "c2")})
    assert r.returncode == 0, r.stderr
    out = json.loads((isolated / "logs" / f"{f.sid}_replay_mock-llm-1.json").read_text())
    assert out["columns"]["formula"]["role"] == "feature_annotation"
    assert "Differences (stored -> replay)" in r.stdout
