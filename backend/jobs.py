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


def get(job_id):
    return _JOBS.get(job_id)
