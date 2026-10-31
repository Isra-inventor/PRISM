"""Step 3a: the connection to the AI (Google Gemini), and a stand-in AI for tests.

What the AI is for in PRISM: it reads a short description of the columns (names and computed
facts, see digest.py) and proposes what the columns are. It never sees data rows and never
changes data. Everything it proposes is shown to you to confirm or correct, and the schema
records which answers came from the AI and which from you.

The key is a Gemini API key from Google AI Studio (https://aistudio.google.com/apikey).
In Colab, store it in Secrets as GEMINI_API_KEY; connect() finds it there.

    ai = prism.connect()        # finds the key, checks nothing yet
    ai.check()                  # one tiny call: is the key working, which model answered

Every call is temperature 0 (the same question gives the same answer as far as the model
allows) and asks for JSON in a fixed shape (a "response schema"), so code can check the answer.
If a model is unavailable or overloaded, the next model in MODELS is tried.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

MODELS = [                      # tried in this order; lite models answer in seconds
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.8-flash",
]
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
KEY_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
TIMEOUT_S = 120                 # a model slower than this is skipped for the next one
TEMPERATURE = 0
RETRY_STATUS = {429, 500, 502, 503, 504}         # worth one more try with the same model
NEXT_MODEL_STATUS = {404, 429, 500, 502, 503, 504}  # worth trying the next model
RETRY_DELAY_S = 2
MAX_RATE_LIMIT_WAIT_S = 35      # Google sometimes says "wait N seconds"; we wait if N is this short
MAX_RATE_LIMIT_WAITS = 3


class AIError(RuntimeError):
    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


def find_key(key: str | None = None) -> str | None:
    """The key given, else an environment variable, else Colab Secrets. None if there is none."""
    if key:
        return key
    for name in KEY_NAMES:
        if os.environ.get(name):
            return os.environ[name]
    try:
        from google.colab import userdata      # only exists inside Colab
        for name in KEY_NAMES:
            try:
                value = userdata.get(name)
            except Exception:                  # secret missing, or notebook access not granted
                value = None
            if value:
                return value
    except ImportError:
        pass
    return None


@dataclass
class Gemini:
    key: str
    models: list[str] = field(default_factory=lambda: list(MODELS))
    verbose: bool = True
    calls: list[dict] = field(default_factory=list)   # one entry per answered call: model, seconds

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(f"[PRISM AI] {msg}")

    def ask(self, system: str, prompt: str, schema: dict) -> dict:
        """Send one question; return the answer as a dict. Raises AIError if no model answers."""
        last = None
        for model in self.models:
            try:
                return self._ask_model(model, system, prompt, schema)
            except AIError as e:
                last = e
                if e.code not in NEXT_MODEL_STATUS:
                    raise
                self._say(f"{e} -> trying the next model")
        raise last or AIError("No model to call.")

    def check(self) -> str:
        """One tiny call to see that the key works. Returns which model answered."""
        schema = {"type": "OBJECT", "properties": {"ok": {"type": "BOOLEAN"}}, "required": ["ok"]}
        answer = self.ask("You check a connection.", 'Answer with {"ok": true}.', schema)
        if answer.get("ok") is not True:
            raise AIError(f"Unexpected answer: {answer}")
        return f"Gemini is working (model {self.calls[-1]['model']})."

    # ---------------------------------------------------------------- one model

    def _ask_model(self, model, system, prompt, schema) -> dict:
        rate_waits, retried = 0, False
        while True:
            self._say(f"calling {model} ...")
            start = time.time()
            try:
                text, finish = self._post(model, system, prompt, schema)
                break
            except urllib.error.HTTPError as e:
                code = e.code
                message, wait = _api_error(e)
                msg = f"Gemini error (HTTP {code}, model {model}): {message}"
                if code == 429 and wait is not None and wait <= MAX_RATE_LIMIT_WAIT_S \
                        and rate_waits < MAX_RATE_LIMIT_WAITS:
                    rate_waits += 1
                    self._say(f"rate limit; Google asks to wait {wait:.0f}s -> waiting")
                    time.sleep(wait + 1)
                    continue
            except (urllib.error.URLError, OSError) as e:
                reason = getattr(e, "reason", e)
                if isinstance(e, TimeoutError) or "timed out" in str(reason):
                    raise AIError(f"{model} did not answer within {TIMEOUT_S}s", 504)
                code, msg = None, f"Could not reach Gemini: {reason}"
            if retried or (code is not None and code not in RETRY_STATUS):
                raise AIError(msg, code)
            retried = True
            self._say(f"{msg} -> retrying in {RETRY_DELAY_S}s")
            time.sleep(RETRY_DELAY_S)

        seconds = round(time.time() - start, 1)
        if finish != "STOP":
            raise AIError(f"{model} stopped early (finish reason {finish}); the answer is incomplete.", 500)
        try:
            answer = json.loads(text)
        except ValueError:
            raise AIError(f"{model} did not answer with valid JSON.", 500)
        self.calls.append({"model": model, "seconds": seconds})
        self._say(f"{model} answered in {seconds}s")
        return answer

    def _post(self, model, system, prompt, schema) -> tuple[str, str]:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema,
                                 "temperature": TEMPERATURE, "maxOutputTokens": 32768},
        }
        req = urllib.request.Request(URL.format(model=model), data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": self.key})
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        candidates = data.get("candidates") or []
        if not candidates:     # Google refused to answer (blocked prompt and similar)
            return "", (data.get("promptFeedback") or {}).get("blockReason", "NO_ANSWER")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        return text, candidates[0].get("finishReason", "UNKNOWN")


def _api_error(err: urllib.error.HTTPError) -> tuple[str, float | None]:
    """Google's error message, and how long it asks us to wait (rate limits), if it says."""
    body = err.read().decode("utf-8", "replace")
    try:
        e = json.loads(body)["error"]
    except (ValueError, KeyError, TypeError):
        return body[:300] or str(err), None
    wait = None
    for d in e.get("details") or []:
        if str(d.get("@type", "")).endswith("RetryInfo"):
            try:
                wait = float(str(d.get("retryDelay", "")).rstrip("s"))
            except ValueError:
                pass
    return (e.get("message") or "").split("\n")[0], wait


class MockAI:
    """A stand-in AI for tests and offline demos: returns a fixed answer (a dict), or calls
    answer(system, prompt, schema) to build one. Records what it was asked in .questions."""

    def __init__(self, answer):
        self.answer = answer
        self.questions: list[dict] = []
        self.calls: list[dict] = []

    def ask(self, system: str, prompt: str, schema: dict) -> dict:
        self.questions.append({"system": system, "prompt": prompt, "schema": schema})
        self.calls.append({"model": "mock", "seconds": 0.0})
        return self.answer(system, prompt, schema) if callable(self.answer) else json.loads(json.dumps(self.answer))

    def check(self) -> str:
        return "Mock AI (no network)."


def connect(key: str | None = None, models: list[str] | None = None, verbose: bool = True) -> Gemini:
    """A Gemini connection. key: leave it out to use GEMINI_API_KEY from Colab Secrets or the environment."""
    found = find_key(key)
    if not found:
        raise AIError("No Gemini key found. Get one at https://aistudio.google.com/apikey, then in Colab: "
                      "key icon (Secrets) -> Add new secret -> name GEMINI_API_KEY -> switch on Notebook access.")
    return Gemini(key=found, models=list(models or MODELS), verbose=verbose)
