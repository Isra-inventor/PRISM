"""Run one long operation in its own process for the web server (backend/jobs.py).

    python -m prism.jobrun --root SESSIONS_DIR --out RESULT.json KIND SESSION_ID [JSON_ARGS]

KIND: audit | plan | approve. Progress lines go to stdout as "\\x1ePRISM-JOB progress <text>";
the result is written to RESULT.json. Exit codes: 0 done, 3 a refusal the user can act on
(message on stdout as "\\x1ePRISM-JOB refused <text>"), anything else a failure (traceback on
stderr). A separate process keeps the web server answering whatever the audit does: heavy C
loops cannot starve it, and a crash or an out-of-memory kill takes only this process down.
"""

from __future__ import annotations

import argparse
import json
import sys

MARK = "\x1ePRISM-JOB"


def emit(kind, text):
    sys.stdout.write(f"{MARK} {kind} {str(text).replace(chr(10), ' ')}\n")
    sys.stdout.flush()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("kind", choices=["audit", "plan", "approve"])
    ap.add_argument("session")
    ap.add_argument("args", nargs="?", default="{}")
    a = ap.parse_args(argv)
    args = json.loads(a.args)
    from . import store
    store.ROOT = a.root
    from .audit import engine, overrides
    from .executor import ExecutorError
    from .executor.state import StateError
    from .plan import engine as pe
    from .plan.intake import IntakeError
    refusals = (engine.AuditError, overrides.OverrideError, ExecutorError, StateError, pe.PlanError, IntakeError,
                store.SessionError)
    progress = lambda text: emit("progress", text)  # noqa: E731
    try:
        st = store.load(a.session)
        if a.kind == "audit":
            result = engine.run(st, args.get("factors"), args.get("datasets"), args.get("params") or {}, who="user",
                                state=args.get("state"), progress=progress)
        elif a.kind == "plan":
            from .audit.report import _trim_plan
            result = _trim_plan(pe.build(st, progress=progress))
        else:
            result = pe.approve(st, args["entry_id"], args.get("candidate") or 0, args.get("user") or "user",
                                progress=progress)
    except refusals as e:
        emit("refused", e)
        return 3
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
