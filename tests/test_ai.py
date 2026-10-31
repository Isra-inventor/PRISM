import io
import json
import urllib.error

import pytest

import prism
from prism import ai as ai_module


def gemini_reply(payload, finish="STOP"):
    body = {"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}, "finishReason": finish}]}
    return io.BytesIO(json.dumps(body).encode())


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_error(code, message="nope"):
    body = json.dumps({"error": {"message": message}}).encode()
    return urllib.error.HTTPError("url", code, message, {}, io.BytesIO(body))


def test_mock_ai_records_questions():
    m = prism.MockAI({"ok": True})
    assert m.ask("sys", "prompt", {}) == {"ok": True}
    assert m.questions[0]["prompt"] == "prompt"


def test_no_key_explains_where_to_put_it(monkeypatch):
    for k in ai_module.KEY_NAMES:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(ai_module.getpass, "getpass", lambda prompt: "")
    with pytest.raises(prism.AIError, match="GEMINI_API_KEY"):
        prism.connect()


def test_asks_once_then_remembers(monkeypatch):
    for k in ai_module.KEY_NAMES:
        monkeypatch.setenv(k, "x")      # so pytest removes what connect() sets afterwards
        monkeypatch.delenv(k)
    asked = []
    monkeypatch.setattr(ai_module.getpass, "getpass", lambda prompt: asked.append(1) or "typed-key")
    assert prism.connect(verbose=False).key == "typed-key"
    assert prism.connect(verbose=False).key == "typed-key"
    assert len(asked) == 1


def test_key_from_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "abc")
    assert prism.connect(verbose=False).key == "abc"


def test_request_and_answer(monkeypatch):
    sent = []

    def fake_urlopen(req, timeout):
        sent.append(req)
        return FakeResponse(gemini_reply({"ok": True}).getvalue())

    monkeypatch.setattr(ai_module.urllib.request, "urlopen", fake_urlopen)
    g = prism.connect("k", verbose=False)
    assert g.check().startswith("Gemini is working")
    body = json.loads(sent[0].data)
    assert body["generationConfig"]["temperature"] == 0
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert sent[0].get_header("X-goog-api-key") == "k"
    assert g.calls[0]["model"] == ai_module.MODELS[0]


def test_falls_through_to_next_model(monkeypatch):
    def fake_urlopen(req, timeout):
        if ai_module.MODELS[0] in req.full_url:
            raise http_error(404, "model not found")
        return FakeResponse(gemini_reply({"ok": True}).getvalue())

    monkeypatch.setattr(ai_module.urllib.request, "urlopen", fake_urlopen)
    g = prism.connect("k", verbose=False)
    g.check()
    assert g.calls[-1]["model"] == ai_module.MODELS[1]


def test_bad_key_is_not_retried_on_other_models(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(req.full_url)
        raise http_error(400, "API key not valid")

    monkeypatch.setattr(ai_module.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(prism.AIError, match="API key not valid"):
        prism.connect("bad", verbose=False).check()
    assert len(calls) == 1


def test_cut_off_answer_is_an_error(monkeypatch):
    monkeypatch.setattr(ai_module.urllib.request, "urlopen",
                        lambda req, timeout: FakeResponse(gemini_reply({"ok": True}, "MAX_TOKENS").getvalue()))
    g = prism.connect("k", models=["m1"], verbose=False)
    with pytest.raises(prism.AIError, match="stopped early"):
        g.check()
