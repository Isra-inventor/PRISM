"""Re-send a session's stored digests through another model and compare (v2.4 §12).

Purpose: tell whether poor labels come from the design (digest, prompt) or from model capacity.

Run from the PRISM folder:
    python scripts/replay_digest.py <session_id> --provider gemini --model gemini-2.5-pro
    python scripts/replay_digest.py <session_id> --provider mock          # offline check of the script

The stored chunk digests (sessions/<id>/state.json) are sent unchanged; when the file was sent in
chunks, the final (consolidation) call is rebuilt from the replayed chunk answers. The replayed
proposal is saved as logs/<session_id>_replay_<model>.json and a diff against the stored
proposal (the 'proposal' event of the session log) is printed: role, audit_kind, label, grouping
and question differences. Nothing in the session changes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("session_id")
    p.add_argument("--provider", default=None, help="gemini | anthropic | openai | mock (default: as configured)")
    p.add_argument("--model", default=None, help="model id for that provider")
    p.add_argument("--sessions-dir", default=str(ROOT / "backend" / "sessions"))
    p.add_argument("--logs-dir", default=str(ROOT / "backend" / "logs"))
    return p.parse_args(argv)


def stored_proposal(logs_dir, sid, state):
    """column -> {group, role, label, audit_kind} from the first 'proposal' event (or the saved state)."""
    path = Path(logs_dir) / f"{sid}.jsonl"
    prop = None
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            if e.get("event") == "proposal":
                prop = e
                break
    cols, questions = {}, []
    if prop:
        groups = prop.get("groups") or {}
        structure = {g["group_id"]: g for g in state.get("groups", [])}
        for gid, it in groups.items():
            names = it.get("columns") or [None]
            if names == [None] and gid in structure:     # logs written before v2.4 stage 8
                names = None
            for c in names or []:
                cols[c] = {"group": gid, **{k: it.get(k) for k in ("role", "label", "audit_kind", "assay_label")}}
        questions = prop.get("questions") or []
    return cols, questions


def replay(sid, args):
    from backend import ai, workflow
    from backend.profiling import name_template
    workflow.SESSIONS_DIR = Path(args.sessions_dir)
    s = workflow.get_session(sid)
    state = json.loads((Path(args.sessions_dir) / sid / "state.json").read_text(encoding="utf-8"))
    digests = [x for x in (s.digests or []) if "columns" in x and "settings" in x and "file" in x]
    if not digests:
        raise SystemExit("This session has no stored grouping digests (was the AI used?).")
    labels = s.cols.labels
    idx = {lab: i for i, lab in enumerate(labels)}
    by_tpl = {}
    for i, lab in enumerate(labels):
        by_tpl.setdefault(name_template(lab), []).append(i)
    out = {"groups": {}, "rejected": [], "samples": [], "clarifying_questions": [], "assays": [],
           "splits_applied": [], "merges_applied": [], "alias": {}, "failed_chunks": []}
    n = len(digests)
    chunk_of = {}
    for k, dg in enumerate(digests, 1):
        ns = f"_chunk{k}" if n > 1 else ""
        tmap = {t["template"]: by_tpl.get(t["template"], []) for t in dg.get("templates", [])}
        chunk = sorted({idx[c["column"]] for c in dg["columns"] if c["column"] in idx} | {i for v in tmap.values() for i in v})
        resp, meta = ai._call(dg, s.sha, None, None, kind="digest")
        if resp is None:
            out["failed_chunks"].append(k)
            print(f"  chunk {k}/{n}: FAILED ({meta.get('error')})")
            continue
        before = set(out["groups"])
        ai.read_chunk(resp, chunk, idx, ns, out, tmap)
        for pid in set(out["groups"]) - before:
            chunk_of[pid] = k
        print(f"  chunk {k}/{n}: {len(set(out['groups']) - before)} group(s) from {meta.get('model')}")
    if n > 1 and len(out["failed_chunks"]) < n:
        payload = {"groups": [ai.group_summary(s.cols, pid, g, chunk_of.get(pid)) for pid, g in out["groups"].items()],
                   "chunk_assays": [{k: a.get(k) for k in ("assay_label", "omics_type")} for a in out["assays"]]}
        cresp, cmeta = ai._call(payload, s.sha, None, None, kind="consolidation")
        if cresp is None:
            print(f"  final call: FAILED ({cmeta.get('error')})")
            ai.drafts_only(out)
        else:
            ai.apply_consolidation(cresp, out)
    cols = {}
    for pid, g in out["groups"].items():
        it = g["item"]
        for i in g["indices"]:
            cols[labels[i]] = {"group": pid, **{k: it.get(k) for k in ("role", "label", "audit_kind", "assay_label")}}
    questions = [q.get("text") or q.get("question") for q in out["clarifying_questions"]]
    return s, state, cols, questions, out


def partition_diff(a, b):
    """Columns whose group-mates differ between the two proposals."""
    def mates(m):
        by = {}
        for c, x in m.items():
            by.setdefault(x["group"], set()).add(c)
        return {c: frozenset(by[x["group"]]) for c, x in m.items()}
    ma, mb = mates(a), mates(b)
    return sorted(c for c in set(ma) & set(mb) if ma[c] != mb[c])


def main(argv):
    args = parse_args(argv)
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["PRISM_LLM_MODEL"] = args.model
    sys.path.insert(0, str(ROOT))
    from backend.envfile import _load_dotenv
    _load_dotenv()
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    from backend import llm_providers as llm
    ok, why = llm.available()
    if not ok:
        raise SystemExit(f"The AI is not available: {why}")
    model = args.model or llm.model_list()[0]
    print(f"Replaying session {args.session_id} through {llm.provider_name()} / {model}")
    s, state, new, new_q, out = replay(args.session_id, args)
    old, old_q = stored_proposal(args.logs_dir, args.session_id, state)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
    dest = Path(args.logs_dir) / f"{args.session_id}_replay_{safe}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"session_id": args.session_id, "provider": llm.provider_name(), "model": model,
                                "columns": new, "questions": new_q, "assays": out["assays"],
                                "rejected": out["rejected"]}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {dest}")
    if not old:
        print("No stored proposal found in the session log; nothing to compare.")
        return 0
    print(f"\nDifferences (stored -> replay), {len(set(old) | set(new))} columns:")
    n = 0
    for c in sorted(set(old) | set(new)):
        a, b = old.get(c), new.get(c)
        if a is None or b is None:
            print(f"  {c}: only in the {'replay' if a is None else 'stored proposal'}")
            n += 1
            continue
        ch = [f"{k} {a.get(k)!r} -> {b.get(k)!r}" for k in ("role", "audit_kind", "label") if a.get(k) != b.get(k)]
        if ch:
            print(f"  {c}: " + "; ".join(ch))
            n += 1
    regroup = partition_diff(old, new)
    if regroup:
        print(f"  grouping differs for {len(regroup)} column(s), e.g. {', '.join(regroup[:6])}")
    gained, lost = sorted(set(new_q) - set(old_q)), sorted(set(old_q) - set(new_q))
    for q in gained:
        print(f"  question only in the replay: {q}")
    for q in lost:
        print(f"  question only in the stored proposal: {q}")
    if not (n or regroup or gained or lost):
        print("  none")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
