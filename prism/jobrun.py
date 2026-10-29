"""Run one long operation in its own process for the web server (backend/jobs.py).

    python -m prism.jobrun --root SESSIONS_DIR --job-file JOB.json KIND SESSION_ID [JSON_ARGS]

KIND: audit | plan | approve. The process writes its status, progress and result to JOB.json
(atomically), so the web server can restart while it runs and the page still gets the result.
It runs in its own session (no shared process group with the server or the terminal).

SIGTERM: the first one is recorded with the process that sent it (pid and command) and the job
keeps going (audits finish in seconds to minutes); a second one stops the job and the status
names the sender. Exit codes: 0 done, 3 refused (the user can act on the message), 1 failed,
143 stopped by SIGTERM.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
import traceback


class Status:
    def __init__(self, path, **fields):
        self.path = path
        self.d = dict(fields)
        self.lock = threading.Lock()
        self._last = 0.0

    def write(self, force=True, **fields):
        with self.lock:
            self.d.update(fields)
            now = time.time()
            if not force and now - self._last < 0.4:
                return
            self._last = now
            tmp = f"{self.path}.tmp{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.d, f)
            os.replace(tmp, self.path)


def _cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", "replace").strip() or "?"
    except OSError:
        return "(process already gone)"


def _say(text):
    try:
        sys.stderr.write(f"[PRISM job] {text}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _watch_sigterm(status):
    """SIGTERM is blocked in every thread; this thread receives it with the sender's pid."""
    seen = []
    while True:
        info = signal.sigwaitinfo({signal.SIGTERM})
        sender = {"signal": 15, "from_pid": info.si_pid, "from_command": _cmdline(info.si_pid) if info.si_pid else
                  "(the kernel or an unknown process)", "at": time.time()}
        seen.append(sender)
        status.write(signals=seen)
        _say(f"received SIGTERM from pid {sender['from_pid']}: {sender['from_command']}")
        if len(seen) >= 2:
            status.write(status="error", finished=time.time(), progress="finished",
                         error=f"The job was stopped by SIGTERM sent twice by process {sender['from_pid']} "
                               f"({sender['from_command']}).")
            os._exit(143)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--job-file", required=True)
    ap.add_argument("--job-id", required=True)
    ap.add_argument("kind", choices=["audit", "plan", "approve"])
    ap.add_argument("session")
    ap.add_argument("args", nargs="?", default="{}")
    a = ap.parse_args(argv)
    status = Status(a.job_file, job_id=a.job_id, kind=a.kind, session=a.session, status="running",
                    progress="starting", started=time.time(), finished=None, result=None, error=None,
                    pid=os.getpid(), signals=[])
    status.write()
    if hasattr(signal, "pthread_sigmask") and hasattr(signal, "sigwaitinfo"):
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})   # before numpy starts its threads
        threading.Thread(target=_watch_sigterm, args=(status,), daemon=True).start()
    args = json.loads(a.args)
    try:
        from . import store
        store.ROOT = a.root
        from .audit import engine, overrides
        from .executor import ExecutorError
        from .executor.state import StateError
        from .plan import engine as pe
        from .plan.intake import IntakeError
        refusals = (engine.AuditError, overrides.OverrideError, ExecutorError, StateError, pe.PlanError,
                    IntakeError, store.SessionError)
        progress = lambda text: status.write(force=False, progress=str(text))  # noqa: E731
        try:
            st = store.load(a.session)
            if a.kind == "audit":
                result = engine.run(st, args.get("factors"), args.get("datasets"), args.get("params") or {},
                                    who="user", state=args.get("state"), progress=progress)
            elif a.kind == "plan":
                from .audit.report import _trim_plan
                result = _trim_plan(pe.build(st, progress=progress))
            else:
                result = pe.approve(st, args["entry_id"], args.get("candidate") or 0, args.get("user") or "user",
                                    progress=progress)
        except refusals as e:
            status.write(status="error", error=str(e), finished=time.time(), progress="finished")
            return 3
        status.write(status="done", result=result, finished=time.time(), progress="finished")
        return 0
    except Exception as e:
        _say(traceback.format_exc())
        status.write(status="error", finished=time.time(), progress="finished",
                     error=f"Internal error while running {a.kind}: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
