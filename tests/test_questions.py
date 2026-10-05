"""v2.4 stage 3: the question queue."""

import json

from backend import mock_llm
from test_consistency import per_code

J = "J_subject_code_blocks.csv"
_ORIGINAL = mock_llm.MockLLM.respond


def post(f, path, expect=200, **body):
    r = f.c.post(path, json={"session_id": f.sid, **body})
    assert r.status_code == expect, r.text
    if expect == 200:
        f.draft = r.json()["draft"]
        if r.json().get("session"):
            f.upload = r.json()["session"]
    return r.json()


def qs(f, **match):
    return [q for q in f.draft["questions"] if all(q.get(k) == v for k, v in match.items())]


def test_code_question_answer_applies_and_undo_reopens(flow, monkeypatch):
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(per_code))
    f = flow(J)
    (q,) = qs(f, kind="near_identical_blocks")
    assert q["source"] == "code" and q["status"] == "open" and q["step"] == "values" and q["type"] == "single"
    assert [o["label"].split(":")[0] for o in q["options"]] == ["One measurement", "Separate measurements", "Ask the AI"]
    assert any(u.get("question_id") == q["question_id"] for u in f.draft["unresolved"])      # blocks finishing
    n_values = sum(1 for it in f.draft["groups"].values() if it["role"] == "value")
    post(f, "/api/question/answer", question_id=q["question_id"], option_ids=["o1"])
    assert qs(f, question_id=q["question_id"])[0]["status"] == "answered"
    assert sum(1 for it in f.draft["groups"].values() if it["role"] == "value") == 1
    ch = f.draft["changes"][0]
    assert ch["actor"] == "question_option" and ch["summary"].startswith("Answered:")
    post(f, "/api/undo")
    assert qs(f, question_id=q["question_id"])[0]["status"] == "open"
    assert sum(1 for it in f.draft["groups"].values() if it["role"] == "value") == n_values
    post(f, "/api/question/answer", expect=422, question_id=q["question_id"], option_ids=["o1", "o2"])  # single: one


def test_dismissed_question_is_not_asked_again_after_reproposal(flow, monkeypatch):
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(per_code))
    f = flow(J)
    (q,) = qs(f, kind="near_identical_blocks")
    post(f, "/api/question/dismiss", question_id=q["question_id"], text="they are different arrays")
    assert qs(f, question_id=q["question_id"])[0]["status"] == "dismissed"
    r = f.c.post("/api/propose", json={"session_id": f.sid, "ai": True})
    f.draft = r.json()["draft"]
    assert [x["status"] for x in qs(f, kind="near_identical_blocks")] == ["dismissed"]
    open_kinds = {q["kind"] for q in f.draft["questions"] if q["status"] == "open"}
    assert open_kinds == {"sample_id_collision"}       # separate blocks: their stripped IDs now collide


def _with_question(question):
    def respond(system, prompt):
        out = json.loads(_ORIGINAL(system, prompt))
        if prompt.startswith("Digest"):
            out["clarifying_questions"] = [question]
        return json.dumps(out)
    return staticmethod(respond)


def test_ai_multi_question_flags_rows_and_finalize_waits(flow, monkeypatch):
    monkeypatch.setattr(mock_llm.MockLLM, "respond", _with_question({
        "type": "multi", "text": "Which values of 'adduct' mark rows to flag?", "applies_to": {"columns": ["adduct"]},
        "options": [{"label": f"{v}", "patches": [{"op": "set_flag_values", "target": {"selector": {"columns": ["adduct"]}},
                                                     "args": {"flagged_values": [v]}, "reason": "x"}]}
                    for v in ("[M+H]+", "[M+Na]+")] + [
            {"label": "broken", "patches": [{"op": "set_label", "target": {"selector": {"columns": ["nope"]}},
                                             "args": {"label": "x"}}]}]}))
    f = flow("C_mzmine_feature_table.csv")
    (q,) = qs(f, source="ai")
    assert q["type"] == "multi" and q["step"] == "annotations" and q["applies_to"] == {"columns": ["adduct"]}
    assert "does not exist" in q["options"][2]["problems"][0].replace("do(es)", "does")
    f.confirm_all_as_proposed()
    assert "Open question" in f.finalize(expect=422)["detail"]
    post(f, "/api/question/answer", expect=422, question_id=q["question_id"], option_ids=["o3"])
    post(f, "/api/question/answer", question_id=q["question_id"], option_ids=["o1", "o2"])
    g = next(gid for gid, it in f.draft["groups"].items() if it["label"] and "adduct" in str(it.get("label")).lower()
             or gid == next(x["group_id"] for x in f.upload["groups"] if x["columns"] == ["adduct"]))
    it = f.draft["groups"][g]
    assert it["marks_rows_as_suspect"] and it["flagged_values"] == ["[M+H]+", "[M+Na]+"]
    assert it["provenance"] == "ai_proposed_confirmed"
    f.step("annotations", {})
    out = f.finalize()
    assert out["schema"]["questions"][0]["status"] == "answered"
    assert out["schema"]["questions"][0]["answer"]["labels"] == ["[M+H]+", "[M+Na]+"]


def test_chat_can_ask_and_questions_dedupe(flow, monkeypatch):
    question = {"type": "confirm", "text": "Is 'formula' the molecular formula?", "applies_to": {"columns": ["formula"]},
                "options": [{"label": "Yes", "patches": [{"op": "set_label", "target": {"selector": {"columns": ["formula"]}},
                                                         "args": {"label": "molecular formula"}}]}, {"label": "No"}]}

    def respond(system, prompt):
        if prompt.startswith("Chat"):
            return json.dumps({"reply": "Let me ask.", "patches": [], "questions": [question]})
        return _ORIGINAL(system, prompt)
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(respond))
    f = flow("C_mzmine_feature_table.csv")
    msg = post(f, "/api/chat", message="what is formula?")["message"]
    (q,) = qs(f, source="ai")
    assert msg["question_ids"] == [q["question_id"]] and q["origin"] == "chat"
    post(f, "/api/chat", message="again?")                              # same (type, applies_to): not queued twice
    assert len(qs(f, source="ai")) == 1
    post(f, "/api/question/answer", question_id=q["question_id"], option_ids=["o1"])
    g = next(x["group_id"] for x in f.upload["groups"] if x["columns"] == ["formula"])
    assert f.draft["groups"][g]["label"] == "molecular formula"
    post(f, "/api/chat", message="and now?")                            # answered: never re-asked
    assert len(qs(f, source="ai")) == 1
