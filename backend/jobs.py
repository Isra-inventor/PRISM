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


def _jobs_dir():
    from pathlib import Path
    from prism import store
    d = Path(store.root()) / "_jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _alive(pid):
    import os
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    try:                                  # a zombie (finished, not yet reaped) is not alive
        with open(f"/proc/{int(pid)}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


def _read_file(job_id):
    import json
    p = _jobs_dir() / f"{job_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return None


def _file_view(d):
    import time as _t
    end = d.get("finished") or _t.time()
    v = {"job_id": d["job_id"], "kind": d.get("kind"), "status": d.get("status"), "progress": d.get("progress"),
         "elapsed_s": round(end - (d.get("started") or end), 1), "result": d.get("result"), "error": d.get("error")}
    sig = d.get("signals") or []
    if sig:
        v["signals"] = sig
        if v["status"] == "running":
            v["progress"] = f"{v['progress']} (ignored a stop request from: {sig[-1]['from_command'][:80]})"
    return v


class ProcJob:
    """A job run by prism.jobrun in its own process; its truth is the status file on disk."""

    def __init__(self, job_id, kind, proc):
        self.id, self.kind, self.proc, self.pid = job_id, kind, proc, proc.pid
        self.exit_note = None

    @property
    def status(self):
        return self.view()["status"]

    @property
    def progress(self):
        return self.view()["progress"]

    def view(self):
        d = _read_file(self.id) or {"job_id": self.id, "kind": self.kind, "status": "running", "progress": "starting"}
        if d.get("status") == "running" and self.exit_note:
            d = dict(d, status="error", error=self.exit_note, progress="finished")
        return _file_view(d)


def start_process(kind, session_id, args=None):
    """Run prism.jobrun in its own process and session. Its status file survives a server restart."""
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path
    from prism import store
    job_id = "j" + uuid.uuid4().hex[:12]
    d = _jobs_dir()
    project = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, "-m", "prism.jobrun", "--root", str(store.root()), "--job-file", str(d / f"{job_id}.json"),
           "--job-id", job_id, kind, session_id, json.dumps(args or {})]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = str(project) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    log = open(d / f"{job_id}.log", "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(project), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                            start_new_session=True)
    log.close()
    job = ProcJob(job_id, kind, proc)
    print(f"[PRISM] {kind} job {job_id} started (pid {proc.pid}); log: {d / (job_id + '.log')}", flush=True)

    def follow():
        code = proc.wait()
        st_ = _read_file(job_id) or {}
        if st_.get("status") == "running":            # ended without writing its outcome
            if code < 0:
                job.exit_note = (f"The {kind} process was stopped by signal {-code}"
                                 + (" (most likely the machine ran out of memory)" if code == -9 else "")
                                 + f". See {d / (job_id + '.log')}.")
            else:
                job.exit_note = f"The {kind} process ended with exit code {code}. See {d / (job_id + '.log')}."
        tail = ""
        try:
            tail = (d / f"{job_id}.log").read_text(encoding="utf-8", errors="replace").strip()[-2000:]
        except OSError:
            pass
        print(f"[PRISM] {kind} job {job_id} finished (exit {code}){chr(10) + tail if tail and code else ''}",
              flush=True)

    threading.Thread(target=follow, name=f"prism-{kind}-{job_id}", daemon=True).start()
    with _LOCK:
        _JOBS[job_id] = job
    return job


def get(job_id):
    """A job of this server, or one started before a restart (read from its status file)."""
    j = _JOBS.get(job_id)
    if j is not None:
        return j
    if not job_id or not all(c.isalnum() for c in job_id):
        return None
    d = _read_file(job_id)
    if d is None:
        return None
    if d.get("status") == "running" and not _alive(d.get("pid")):
        d = dict(d, status="error", progress="finished",
                 error="The job process is no longer running and did not record an outcome "
                       f"(see {_jobs_dir() / (job_id + '.log')}).")
    return _Snapshot(d)


class _Snapshot:
    def __init__(self, d):
        self.d = d
        self.status = d.get("status")

    def view(self):
        return _file_view(self.d)
