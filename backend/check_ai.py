"""Check that the Gemini AI fallback works on this machine.

Run from the project folder:  python -m backend.check_ai
It reads GEMINI_API_KEY from .env (or the environment), sends one tiny
request to each configured model and prints exactly what comes back.
"""

from __future__ import annotations

import logging
import ssl
import sys
import urllib.error

from . import main  # noqa: F401  (loads .env)
from . import llm_fallback as L


def run():
    logging.getLogger("prism").setLevel(logging.ERROR)
    print(f"Python {sys.version.split()[0]}, {ssl.OPENSSL_VERSION}")
    key = L.api_key()
    if not key:
        print("FAIL: GEMINI_API_KEY is not set. Put it in a file named .env in the project folder:")
        print("      GEMINI_API_KEY=your-key")
        return 1
    print(f"Key found ({key[:4]}...{key[-4:]}, {len(key)} characters)")
    prompt = 'Columns: {"column": "sample_code", "sample_values": ["S1", "S2"]}'
    ok = False
    for model in L.model_list():
        try:
            text, finish = L.call_gemini(model, key, L.SYSTEM_PROMPT, prompt)
            parsed = L.AIProposalBatch.model_validate_json(text)
            print(f"  OK    {model}: {parsed.columns[0].column} -> {parsed.columns[0].proposed_role}")
            ok = True
        except urllib.error.HTTPError as e:
            print(f"  FAIL  {model}: HTTP {e.code} - {L._api_message(e)}")
        except urllib.error.URLError as e:
            reason = getattr(e, "reason", e)
            print(f"  FAIL  {model}: could not connect - {reason}")
            if isinstance(reason, ssl.SSLError) or "CERTIFICATE" in str(reason):
                print("        SSL certificate problem. On macOS run 'Install Certificates.command' from your "
                      "Python folder; elsewhere try: pip install --upgrade certifi")
        except Exception as e:  # show anything else verbatim
            print(f"  FAIL  {model}: {type(e).__name__}: {e}")
    print("Result:", "AI fallback works (at least one model answered)." if ok
          else "No model answered. See the errors above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
