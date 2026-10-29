"""Background jobs for long requests (audit runs, plans, approvals).

A long audit used to run inside one HTTP request; behind a proxy (GitHub Codespaces, a reverse
proxy) that request could time out and the page received a non-JSON error. Now the request starts
a job and returns at once; the page polls GET /api/jobs/{id} and shows the progress.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid

_JOBS = {}
_LOCK = threading.Lock()
_KEEP = 200


class Job:
    def __init__(self, kind):
        self.id = "j" + uuid.uuid4().hex[:12]
        self.kind = kind
        self.status = "running"
        self.progress = "starting"
        self.result = None
        self.error = None
        self.started = time.time()
        self.finished = None

    def view(self):
        return {"job_id": self.id, "kind": self.kind, "status": self.status, "progress": self.progress,
                "elapsed_s": round((self.finished or time.time()) - self.started, 1), "result": self.result,
                "error": self.error}


def start(kind, fn, user_errors=()):
    """fn(progress) runs in a thread; progress(text) updates what the page shows."""
    job = Job(kind)

    def progress(text):
        job.progress = str(text)

    def body():
        try:
            job.result = fn(progress)
            job.status = "done"
        except user_errors as e:                      # a refusal the user can act on
            job.error = str(e)
            job.status = "error"
        except Exception as e:                       # a bug: say what failed, keep the traceback in the log
            job.error = f"Internal error while running {kind}: {type(e).__name__}: {e}"
            job.status = "error"
            print(f"[PRISM job {job.id}] {kind} failed:\n{traceback.format_exc()}", flush=True)
        finally:
            job.finished = time.time()
            job.progress = "finished"

    with _LOCK:
        _JOBS[job.id] = job
        if len(_JOBS) > _KEEP:
            for k in sorted(_JOBS, key=lambda k: _JOBS[k].started)[: len(_JOBS) - _KEEP]:
                if _JOBS[k].status != "running":
                    _JOBS.pop(k, None)
    threading.Thread(target=body, name=f"prism-{kind}-{job.id}", daemon=True).start()
    return job


def start_process(kind, session_id, args=None):
    """Run prism.jobrun in a child process; a thread follows its progress and collects the result."""
    import json
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path
    from prism import store
    job = Job(kind)
    project = Path(__file__).resolve().parent.parent
    fd, out = tempfile.mkstemp(prefix=f"prism-{kind}-", suffix=".json")
    os.close(fd)
    cmd = [sys.executable, "-m", "prism.jobrun", "--root", str(store.root()), "--out", out, kind, session_id,
           json.dumps(args or {})]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = str(project) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.Popen(cmd, cwd=str(project), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace")
    job.pid = proc.pid
    err_tail = []

    def read_err():
        for line in proc.stderr:
            err_tail.append(line.rstrip())
            del err_tail[:-40]
            print(f"[PRISM job {job.id}] {line.rstrip()}", flush=True)

    def follow():
        refused = None
        try:
            for line in proc.stdout:
                if line.startswith("\x1ePRISM-JOB progress "):
                    job.progress = line[len("\x1ePRISM-JOB progress "):].strip()
                elif line.startswith("\x1ePRISM-JOB refused "):
                    refused = line[len("\x1ePRISM-JOB refused "):].strip()
                else:
                    print(line.rstrip(), flush=True)
            code = proc.wait()
            te.join(timeout=5)
            if code == 0:
                with open(out, encoding="utf-8") as f:
                    job.result = json.load(f)
                job.status = "done"
            elif code == 3:
                job.error, job.status = refused or f"{kind} was refused.", "error"
            else:
                last = next((l for l in reversed(err_tail) if l.strip()), "")
                if code < 0 or code in (137, -9):
                    why = (f"the {kind} process was stopped by signal {-code if code < 0 else 9}"
                           + (" - most likely the machine ran out of memory" if code in (-9, 137) else ""))
                else:
                    why = f"the {kind} process failed (exit code {code})"
                job.error, job.status = f"{why}. {last}".strip(), "error"
        except Exception as e:
            job.error, job.status = f"Internal error while following the {kind} job: {e}", "error"
        finally:
            job.finished = time.time()
            job.progress = "finished"
            try:
                os.unlink(out)
            except OSError:
                pass

    te = threading.Thread(target=read_err, daemon=True)
    te.start()
    threading.Thread(target=follow, name=f"prism-{kind}-{job.id}", daemon=True).start()
    with _LOCK:
        _JOBS[job.id] = job
    return job


def get(job_id):
    return _JOBS.get(job_id)
