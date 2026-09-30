"""LLM transport: Gemini (default), Anthropic, OpenAI and a Mock provider.

LLM_PROVIDER = gemini | anthropic | openai | mock   (default: gemini if a Gemini
key is set, else anthropic / openai if their key is set, else none = manual).
Model names come from PRISM_LLM_MODEL (comma-separated list = fallback order).

Gemini uses plain HTTPS from the standard library (works on Python 3.7).
Anthropic / OpenAI use their SDKs if installed (optional dependencies).
Every call is temperature 0 and asks for JSON that matches a response schema.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

GEMINI_MODELS = [
    "gemini-3.5-flash-lite",      # lite models: no thinking, answer in seconds
    "gemini-flash-lite-latest",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.8-flash",
]
DEFAULT_MODELS = {
    "gemini": GEMINI_MODELS,
    "anthropic": ["claude-opus-5"],
    "openai": ["gpt-5"],
    "mock": ["mock-llm-1"],
}
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# a model slower than this is skipped for the next one; grouping answers list every
# column name, so a chunk of 150 columns needs more time than labelling did
REQUEST_TIMEOUT_S = int(os.environ.get("PRISM_AI_TIMEOUT_S") or 120)
RETRY_STATUS = {429, 500, 502, 503, 504}
RETRY_DELAYS_S = (2,)
MAX_RATE_LIMIT_WAITS = 3
MAX_RATE_LIMIT_WAIT_S = 35
NEXT_MODEL_STATUS = {404, 429, 500, 502, 503, 504}
TEMPERATURE = 0

_ctx = threading.local()   # per-thread progress reporter


def say(msg):
    """Status line to the server terminal and to the current progress reporter."""
    print(f"[PRISM AI] {msg}", file=sys.stderr, flush=True)
    report = getattr(_ctx, "report", None)
    if report is not None:
        try:
            report(msg)
        except Exception:
            pass


class LLMError(RuntimeError):
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


def gemini_key():
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def provider_name():
    p = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if p:
        return p
    if gemini_key():
        return "gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "none"


def available():
    """(bool, reason) whether the configured provider can be called."""
    p = provider_name()
    if p == "mock":
        return True, "Mock provider (offline demo: canned proposals)."
    if p == "gemini":
        return (True, "") if gemini_key() else (False, "GEMINI_API_KEY is not set (put it in .env).")
    if p == "anthropic":
        return (True, "") if os.environ.get("ANTHROPIC_API_KEY") else (False, "ANTHROPIC_API_KEY is not set.")
    if p == "openai":
        return (True, "") if os.environ.get("OPENAI_API_KEY") else (False, "OPENAI_API_KEY is not set.")
    if p == "none":
        return False, "No AI key configured (set GEMINI_API_KEY in .env, or LLM_PROVIDER=mock for a demo)."
    return False, f"Unknown LLM_PROVIDER '{p}'."


def model_list():
    raw = os.environ.get("PRISM_LLM_MODEL")
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    return list(DEFAULT_MODELS.get(provider_name(), ["unknown"]))


def to_json_schema(g):
    """Gemini (OpenAPI subset) schema -> standard JSON schema (for Anthropic/OpenAI)."""
    t = g.get("type", "").lower()
    out = {}
    if t == "object":
        props = {k: to_json_schema(v) for k, v in g.get("properties", {}).items()}
        out = {"type": "object", "properties": props, "required": list(props),
               "additionalProperties": False}
    elif t == "array":
        out = {"type": "array", "items": to_json_schema(g["items"])}
    else:
        out = {"type": t}
        if "enum" in g:
            out["enum"] = list(g["enum"]) + ([None] if g.get("nullable") else [])
    if g.get("nullable"):
        out["type"] = [out["type"], "null"]
    return out


# ---------------------------------------------------------------- providers

def complete_json(system, prompt, gemini_schema, mock_fn=None):
    """Returns (text, meta) where meta = {provider, model, finish_reason}.
    Raises LLMError when every model fails."""
    p = provider_name()
    if p == "mock":
        from .mock_llm import MockLLM
        text = (mock_fn or MockLLM.respond)(system, prompt)
        return text, {"provider": "mock", "model": model_list()[0], "finish_reason": "STOP"}
    if p == "gemini":
        return _gemini_with_fallback(system, prompt, gemini_schema)
    if p == "anthropic":
        return _anthropic(system, prompt, gemini_schema)
    if p == "openai":
        return _openai(system, prompt, gemini_schema)
    raise LLMError(available()[1])


def _gemini_with_fallback(system, prompt, schema):
    models = model_list()
    last = None
    for model in models:
        try:
            text, finish = _gemini_with_retry(model, system, prompt, schema)
            return text, {"provider": "gemini", "model": model, "finish_reason": finish}
        except LLMError as e:
            last = e
            if e.code not in NEXT_MODEL_STATUS or model == models[-1]:
                raise
            say(f"{e} -> switching to next model")
    raise last


def _api_error(err):
    body = err.read().decode("utf-8", "replace")
    try:
        e = json.loads(body)["error"]
    except (ValueError, KeyError, TypeError):
        return body[:300] or str(err), None
    delay = None
    for d in e.get("details") or []:
        if str(d.get("@type", "")).endswith("RetryInfo"):
            try:
                delay = float(str(d.get("retryDelay", "")).rstrip("s"))
            except ValueError:
                pass
    return e.get("message", body[:300]), delay


def _gemini_with_retry(model, system, prompt, schema):
    rate_waits, attempt = 0, 0
    while True:
        say(f"calling {model} ...")
        try:
            text, finish = _gemini_call(model, system, prompt, schema)
            say(f"{model} answered (finish_reason={finish})")
            return text, finish
        except urllib.error.HTTPError as e:
            code = e.code
            message, retry_after = _api_error(e)
            msg = f"Gemini API error (HTTP {code}, model {model}): {message.splitlines()[0] if message else ''}"
            if code == 429 and retry_after is not None and rate_waits < MAX_RATE_LIMIT_WAITS \
                    and retry_after <= MAX_RATE_LIMIT_WAIT_S:
                rate_waits += 1
                say(f"{model}: rate limit reached; Google asks to wait {retry_after:.0f}s -> waiting")
                time.sleep(retry_after + 1)
                continue
            retryable = code in RETRY_STATUS
        except (urllib.error.URLError, OSError) as e:
            reason = getattr(e, "reason", e)
            if isinstance(e, TimeoutError) or "timed out" in str(reason):
                raise LLMError(f"Gemini model {model} did not answer within {REQUEST_TIMEOUT_S}s", 504)
            code, retryable = None, True
            msg = f"Could not reach the Gemini API ({type(e).__name__}): {reason}"
        if not retryable or attempt >= len(RETRY_DELAYS_S):
            raise LLMError(msg, code)
        delay = RETRY_DELAYS_S[attempt]
        attempt += 1
        say(f"{msg} -> retrying in {delay}s")
        time.sleep(delay)


def _gemini_call(model, system, prompt, schema):
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema,
            "temperature": TEMPERATURE,
            "maxOutputTokens": 32768,
        },
    }
    req = urllib.request.Request(
        GEMINI_URL.format(model=model), data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": gemini_key()}, method="POST")
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    candidates = data.get("candidates") or []
    if not candidates:
        return "", (data.get("promptFeedback") or {}).get("blockReason", "NO_CANDIDATES")
    cand = candidates[0]
    parts = (cand.get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts if not p.get("thought")), cand.get("finishReason", "UNKNOWN")


def _anthropic(system, prompt, schema):
    try:
        import anthropic
    except ImportError:
        raise LLMError("LLM_PROVIDER=anthropic needs the 'anthropic' package: pip install anthropic")
    model = model_list()[0]
    client = anthropic.Anthropic()
    say(f"calling {model} ...")
    try:
        resp = client.messages.create(
            model=model, max_tokens=16000, system=system,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"output_config": {"format": {"type": "json_schema", "schema": to_json_schema(schema)}}},
        )
    except anthropic.APIStatusError as e:
        raise LLMError(f"Anthropic API error (HTTP {e.status_code}): {e}", e.status_code)
    except anthropic.APIConnectionError as e:
        raise LLMError(f"Could not reach the Anthropic API: {e}")
    text = "".join(b.text for b in resp.content if b.type == "text")
    finish = "STOP" if resp.stop_reason == "end_turn" else str(resp.stop_reason)
    return text, {"provider": "anthropic", "model": model, "finish_reason": finish}


def _openai(system, prompt, schema):
    try:
        import openai
    except ImportError:
        raise LLMError("LLM_PROVIDER=openai needs the 'openai' package: pip install openai")
    model = model_list()[0]
    client = openai.OpenAI()
    say(f"calling {model} ...")
    try:
        resp = client.chat.completions.create(
            model=model, temperature=TEMPERATURE,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "prism_step0", "schema": to_json_schema(schema)}},
        )
    except Exception as e:  # SDK error classes differ across versions
        raise LLMError(f"OpenAI API error: {e}", getattr(e, "status_code", None))
    choice = resp.choices[0]
    finish = "STOP" if choice.finish_reason == "stop" else str(choice.finish_reason)
    return choice.message.content or "", {"provider": "openai", "model": model, "finish_reason": finish}
