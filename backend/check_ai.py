"""Check that the AI provider works on this machine.

Run from the project folder:  python -m backend.check_ai
Reads the key from .env (or the environment), sends one tiny request to each
configured model and prints exactly what comes back.
"""

from __future__ import annotations

import ssl
import sys

from .envfile import DOTENV_REPORT, _load_dotenv

_load_dotenv()

from . import llm_providers as llm  # noqa: E402

SCHEMA = {"type": "OBJECT", "properties": {"answer": {"type": "STRING"}}, "required": ["answer"]}


def run():
    print(f"Python {sys.version.split()[0]}, {ssl.OPENSSL_VERSION}")
    ok, why = llm.available()
    print(f"Provider: {llm.provider_name()}")
    if not ok:
        print("FAIL:", why)
        for line in DOTENV_REPORT:
            print("  -", line)
        return 1
    models = llm.model_list()
    good = False
    for m in models:
        import os
        os.environ["PRISM_LLM_MODEL"] = m
        try:
            text, meta = llm.complete_json("Answer with JSON.", 'Reply {"answer": "ok"}.', SCHEMA)
            print(f"  OK    {m}: {text[:60]!r}")
            good = True
        except llm.LLMError as e:
            print(f"  FAIL  {m}: {e}")
    print("Result:", "AI works (at least one model answered)." if good else "No model answered; see above.")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(run())
