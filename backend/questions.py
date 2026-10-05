"""The question queue (v2.4 §5): what the AI or code cannot resolve becomes a queued
question with clickable options, never a silent default.

- Sources: the AI's proposal and chat replies (source 'ai'), and deterministic code
  (source 'code': near-identical blocks, sample-ID collisions, derivation coverage,
  ...). Code questions are recomputed after every edit: a new situation adds one, a
  situation that no longer holds withdraws its open question.
- type single | multi | confirm. Choosing options applies their patches through the
  edit layer (actor 'question_option'), as one undoable edit; undo reopens the
  question. 'Other' sends your text to the chat.
- Status open | answered | dismissed. Finishing is blocked while any is open.
- Dedupe by (type or kind, applies_to): an answered or dismissed question is never
  asked again, also after a re-proposal.

Question definitions live in draft['questions'] (a record, never rewound by undo);
answers live in draft['answers'] (tracked by the undo stack).
"""

from __future__ import annotations

import json
import uuid

from . import edits
from .errors import StepError

TYPES = ("single", "multi", "confirm")
STEP_IDS = ["layout", "feature_id", "annotations", "values", "samples", "sample_info", "design", "history", "review"]


def key_of(kind, applies_to):
    return f"{kind}|{json.dumps(applies_to or {}, sort_keys=True)}"


def answer_of(d, q):
    return (d.get("answers") or {}).get(q["question_id"])


def status(d, q):
    a = answer_of(d, q)
    return a["status"] if a else "open"


def closed_keys(d):
    return {a["key"] for a in (d.get("answers") or {}).values()}


def open_questions(d):
    return [q for q in d.get("questions", []) if status(d, q) == "open"]


def _step(x):
    if isinstance(x, int) or (isinstance(x, str) and x.isdigit()):
        k = int(x) - 1
        return STEP_IDS[k] if 0 <= k < len(STEP_IDS) else None
    return x if x in STEP_IDS else None


def add(s, d, q):
    """Queue a question unless one with the same key exists (open, answered or dismissed)."""
    if q["key"] in closed_keys(d) or any(x["key"] == q["key"] for x in d.get("questions", [])):
        return None
    q.setdefault("question_id", "q" + uuid.uuid4().hex[:7])
    q.setdefault("created_at", edits.now_iso())
    for k, o in enumerate(q["options"], 1):
        o.setdefault("option_id", f"o{k}")
    d.setdefault("questions", []).append(q)
    s.log("question_created", {k: q.get(k) for k in ("question_id", "source", "kind", "type", "text", "applies_to",
                                                     "step", "options")})
    return q


# ---------------------------------------------------------------- AI questions

def from_ai(s, d, raw, origin):
    """An AI question (proposal or chat) -> queued question. Option patches are checked
    (op in the closed set, selectors resolve); a broken option is shown as not possible."""
    from . import patches as P
    typ = raw.get("type") if raw.get("type") in TYPES else "single"
    applies = {k: v for k, v in (raw.get("applies_to") or {}).items() if v}
    if raw.get("group_id") and raw["group_id"] in d["groups"]:
        applies.setdefault("group_ids", [raw["group_id"]])
    labels = set(s.cols.labels) | {c["column"] for c in (d.get("metadata") or {}).get("columns", [])}
    if applies.get("columns"):
        applies["columns"] = [c for c in applies["columns"] if c in labels]
    options = []
    for o in raw.get("options") or []:
        opt = {"label": str(o.get("label") or "").strip()[:200] or "(no label)", "patches": [], "problems": []}
        n = 0
        for p in o.get("patches") or []:
            if p.get("op") not in P.AI_OPS:
                opt["problems"].append(f"'{p.get('op')}' is not an allowed operation.")
                continue
            if "processing_history" in json.dumps(p).lower():
                opt["problems"].append("Processing history is answered by you only.")
                continue
            if p["op"] in P.COLUMN_OPS | P.SAMPLE_OPS:
                try:
                    res = P.resolve(s, d, p["op"], p.get("target"))
                    n += len(res["main"]) + len(res["metadata"])
                except P.PatchError as e:
                    opt["problems"].append(str(e))
                    continue
            opt["patches"].append({k: p.get(k) for k in ("op", "target", "args", "reason", "consequences")})
        opt["n_columns"] = n
        options.append(opt)
    if typ == "confirm" and not options:
        options = [{"label": "Yes", "patches": [], "problems": []}, {"label": "No", "patches": [], "problems": []}]
    text = str(raw.get("text") or raw.get("question") or "").strip()[:600]
    if not text:
        return None
    q = {"source": "ai", "origin": origin, "kind": "ai", "type": typ, "text": text, "applies_to": applies,
         "step": _step(raw.get("step")) or _guess_step(s, d, applies), "options": options,
         "allow_free_text": raw.get("allow_free_text") is not False,
         "key": key_of(typ, applies) if applies else key_of(typ, {"text": text})}
    return add(s, d, q)


def _guess_step(s, d, applies):
    from . import workflow as wf
    gids = list(applies.get("group_ids") or [])
    for c in applies.get("columns") or []:
        g = next((g["group_id"] for g in s.groups if c in g["columns"]), None)
        if g:
            gids.append(g)
    for g in gids:
        if g in d["groups"]:
            return wf.step_for_group(s, d, g)
    return "review"


# ---------------------------------------------------------------- code questions

def refresh_code(s, d):
    """Recompute the code-generated questions after an edit (called by the edit layer)."""
    from . import workflow as wf
    cands = wf.code_question_candidates(s, d)
    keys = {c["key"] for c in cands}
    answered = closed_keys(d)
    qs = d.setdefault("questions", [])
    for q in list(qs):   # withdraw open code questions whose situation no longer holds
        if q["source"] == "code" and q["key"] not in keys and status(d, q) == "open":
            qs.remove(q)
            s.log("question_withdrawn", {"question_id": q["question_id"], "kind": q["kind"], "text": q["text"]})
    by_key = {q["key"]: q for q in qs}
    for c in cands:
        if c["key"] in answered:
            continue
        cur = by_key.get(c["key"])
        if cur is None:
            add(s, d, dict(c, source="code"))
        elif status(d, cur) == "open":     # facts may have moved: refresh text and options in place
            cur.update(text=c["text"], options=[dict(o, option_id=cur["options"][k]["option_id"]
                                                     if k < len(cur["options"]) else f"o{k + 1}")
                                                for k, o in enumerate(c["options"])],
                       evidence=c.get("evidence"))


# ---------------------------------------------------------------- answering

def _merge_flag_patches(patches):
    """multi answers: several set_flag_values options on one column become one list."""
    out, flags = [], {}
    for p in patches:
        if p["op"] == "set_flag_values":
            k = json.dumps(p.get("target"), sort_keys=True)
            if k in flags:
                flags[k]["args"]["flagged_values"] = list(dict.fromkeys(
                    flags[k]["args"]["flagged_values"] + list((p.get("args") or {}).get("flagged_values") or [])))
                continue
            p = dict(p, args=dict(p.get("args") or {}, flagged_values=list((p.get("args") or {}).get("flagged_values") or [])))
            flags[k] = p
        out.append(p)
    return out


def find(d, qid):
    q = next((x for x in d.get("questions", []) if x["question_id"] == qid), None)
    if q is None:
        raise StepError("Unknown question.")
    return q


def answer(s, qid, option_ids, text=None):
    """Your answer: apply the chosen options' patches / edits as one undoable edit."""
    from . import patches as P
    d = s.draft
    q = find(d, qid)
    if status(d, q) != "open":
        raise StepError("This question is already answered or dismissed.")
    by = {o["option_id"]: o for o in q["options"]}
    chosen = [by[o] for o in dict.fromkeys(option_ids or []) if o in by]
    if not chosen:
        raise StepError("Choose an option.")
    if q["type"] in ("single", "confirm") and len(chosen) != 1:
        raise StepError("Choose one option.")
    bad = [p for o in chosen for p in o.get("problems") or []]
    if bad:
        raise StepError("That option cannot be applied: " + " ".join(bad))
    labels = [o["label"] for o in chosen]
    eid = "e" + uuid.uuid4().hex[:9]
    with edits.recording(s, "answer_question", "question_option", f"Answered: {q['text'][:70]} → {'; '.join(labels)[:80]}",
                         {"question_id": qid, "options": [o["option_id"] for o in chosen]}) as tx:
        tx["edit_id"] = eid
        for o in chosen:
            for e in o.get("edits") or []:      # code questions: concrete ops written by code
                edits.apply_edit(s, e["op"], e.get("args") or {}, "question_option", internal=True,
                                 reason=f"Your answer to: {q['text'][:120]}")
        for p in _merge_flag_patches([p for o in chosen for p in o.get("patches") or []]):
            P._run(s, dict(p, reason=p.get("reason") or f"Your answer to: {q['text'][:120]}"), "question_option")
        s.draft.setdefault("answers", {})[qid] = {
            "status": "answered", "key": q["key"], "option_ids": [o["option_id"] for o in chosen], "labels": labels,
            "text": (text or "").strip()[:1000] or None, "by": "user", "at": edits.now_iso(), "edit_id": eid}
    s.log("question_answered", {"question_id": qid, "text": q["text"], "options": labels, "edit_id": eid})
    s.save()


def dismiss(s, qid, note=None):
    d = s.draft
    q = find(d, qid)
    if status(d, q) != "open":
        raise StepError("This question is already answered or dismissed.")
    with edits.recording(s, "dismiss_question", "user", f"Dismissed: {q['text'][:80]}", {"question_id": qid}):
        s.draft.setdefault("answers", {})[qid] = {"status": "dismissed", "key": q["key"], "option_ids": [],
                                                  "labels": [], "text": (note or "").strip()[:1000] or None,
                                                  "by": "user", "at": edits.now_iso()}
    s.log("question_dismissed", {"question_id": qid, "text": q["text"], "note": note})
    s.save()


def public(d):
    out = []
    for q in d.get("questions", []):
        a = answer_of(d, q)
        out.append(dict(q, status=a["status"] if a else "open", answer=a))
    return out
