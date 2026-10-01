"""Literature retrieval (RAG) from Europe PMC, and AI suggestions grounded in it.

NOT USED IN STEP 0 (v2.3): deferred to a future Tier 2 feature, not abandoned. Step 0
runs before the research-focus intake, so "should this block be analysed" cannot be
answered there. Kept for later: Europe PMC client, passage ranking and the
word-for-word quote verification.

Pipeline:
    queries  (the AI's search queries from the proposal + ones built from the
              confirmed description: software, omics type, block names)
    -> Europe PMC search (abstracts), open-access full text for the top papers
    -> passages (abstract windows; paragraphs from methods / results sections)
    -> BM25 ranking against the block names and queries, at most 3 per paper
    -> the AI describes each value block from the passages only: what it is, how
       similar studies used it, whether it is suggested for analysis
    -> every citation must quote its passage word for word; unverifiable
       citations are dropped, and a "suggested: yes" without one becomes "unsure".

What is sent to Europe PMC: the search queries (names and terms, never data).
What is sent to the AI: the confirmed description (labels + computed block
statistics) and the passages. The retrieved papers and passages are stored in
the draft and in schema.json so later steps (normalization, imputation...) can
re-rank and reuse them without searching again.

Settings: PRISM_LITERATURE=off disables it; PRISM_EUROPEPMC_URL points at a
mirror (the tests use a local fake).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ValidationError

from . import llm_providers as llm
from .profiling import FILE_EXTENSIONS
from .schema import PROMPT_VERSION

SUGGESTED = ["yes", "no", "unsure"]   # was VOCABULARY['suggested_for_analysis'] in Step 0

EUROPEPMC_URL = os.environ.get("PRISM_EUROPEPMC_URL", "https://www.ebi.ac.uk/europepmc/webservices/rest").rstrip("/")
CACHE_DIR = Path(os.environ.get("PRISM_LITERATURE_CACHE", Path(__file__).parent / "cache" / "literature"))
BRIEFING = (Path(__file__).parent / "briefing.md").read_text(encoding="utf-8")
TIMEOUT_S = 25
PAPERS_PER_QUERY = 8
FULL_TEXTS = 4
PASSAGES_TO_AI = 14
PER_PAPER = 3
MAX_QUERIES = 5
PASSAGE_CHARS = 700
_USEFUL_SECTIONS = re.compile(r"(?i)method|material|data (analysis|processing)|quantif|statist|bioinformat|"
                              r"processing|normali|result|experimental")


class LiteratureError(Exception):
    pass


def enabled():
    return os.environ.get("PRISM_LITERATURE", "on").strip().lower() not in ("0", "off", "false", "no")


def say(msg):
    print(f"[PRISM literature] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- HTTP (cached)

def _get(url, kind):
    """GET with an on-disk cache (literature does not change between runs)."""
    key = hashlib.sha1(url.encode("utf-8")).hexdigest()[:24]
    path = CACHE_DIR / f"{key}.{kind}"
    if path.exists():
        return path.read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": "PRISM-step0/0.3 (research tool; Europe PMC REST)",
                                               "Accept": "application/json, application/xml"})
    last = None
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                body = resp.read()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            return body
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last = f"HTTP {e.code} from Europe PMC"
        except Exception as e:  # network down, DNS, timeout
            last = f"{type(e).__name__}: {e}"
        time.sleep(1.5 * attempt)
    raise LiteratureError(f"Europe PMC could not be reached ({last}).")


def search(query, n=PAPERS_PER_QUERY):
    """Europe PMC search -> list of papers with abstracts."""
    q = f"({query}) AND HAS_ABSTRACT:y"
    url = (f"{EUROPEPMC_URL}/search?" +
           urllib.parse.urlencode({"query": q, "format": "json", "resultType": "core", "pageSize": n}))
    body = _get(url, "json")
    if not body:
        return []
    data = json.loads(body.decode("utf-8"))
    out = []
    for r in (data.get("resultList") or {}).get("result") or []:
        journal = ((r.get("journalInfo") or {}).get("journal") or {}).get("title") or r.get("journalTitle") or ""
        pid = r.get("pmid") or r.get("pmcid") or r.get("doi") or r.get("id")
        if not pid:
            continue
        out.append({
            "paper_id": str(pid), "title": _clean(r.get("title")), "authors": _clean(r.get("authorString"))[:160],
            "journal": _clean(journal), "year": str(r.get("pubYear") or ""), "pmid": r.get("pmid"),
            "pmcid": r.get("pmcid"), "doi": r.get("doi"),
            "open_access": (r.get("isOpenAccess") == "Y"), "abstract": _clean(r.get("abstractText")),
            "url": (f"https://europepmc.org/article/MED/{r['pmid']}" if r.get("pmid") else
                    f"https://europepmc.org/article/PMC/{r['pmcid']}" if r.get("pmcid") else
                    f"https://doi.org/{r['doi']}" if r.get("doi") else ""),
            "query": query,
        })
    return out


def full_text(pmcid):
    """Open-access JATS full text -> [(section title, paragraph)] from useful sections."""
    body = _get(f"{EUROPEPMC_URL}/{urllib.parse.quote(pmcid)}/fullTextXML", "xml")
    if not body:
        return []
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return []
    out = []
    for sec in root.iter("sec"):
        title_el = sec.find("title")
        title = _clean("".join(title_el.itertext())) if title_el is not None else ""
        if not _USEFUL_SECTIONS.search(title):
            continue
        for p in sec.findall("p"):
            text = _clean("".join(p.itertext()))
            if len(text) > 80:
                out.append((title, text))
    return out


def _clean(x):
    x = re.sub(r"<[^>]+>", " ", str(x or ""))
    return re.sub(r"\s+", " ", x).strip()


# ---------------------------------------------------------------- passages + ranking

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
_TOK = re.compile(r"[a-z0-9]+(?:[-/][a-z0-9]+)*")
_STOP = set("a an and are as at be by for from has have in is it its of on or that the this to was were with we our "
            "which these those using used use than can not no also into between both each other such their been "
            "after before all per".split())


def _windows(text, max_chars=PASSAGE_CHARS, step_sents=2):
    sents = [x for x in _SENT.split(text) if x]
    out, k = [], 0
    while k < len(sents):
        w, j = "", k
        while j < len(sents) and len(w) + len(sents[j]) < max_chars:
            w = (w + " " + sents[j]).strip()
            j += 1
        if not w:  # one very long sentence
            w, j = sents[k][:max_chars], k + 1
        out.append(w)
        if j >= len(sents):
            break
        k += max(1, min(step_sents, j - k))
    return out


def build_passages(papers, texts):
    passages = []
    for p in papers:
        for w in _windows(p["abstract"]):
            passages.append({"paper_id": p["paper_id"], "section": "abstract", "text": w})
        for sec, para in texts.get(p["paper_id"], []):
            for w in _windows(para):
                passages.append({"paper_id": p["paper_id"], "section": sec, "text": w})
    return passages


def tokens(text):
    return [t for t in _TOK.findall((text or "").lower()) if t not in _STOP and len(t) > 1]


def rank(passages, terms, phrases=(), k=PASSAGES_TO_AI, per_paper=PER_PAPER):
    """BM25 over passages; exact phrase hits (block names) add a bonus.
    At most `per_paper` passages from one paper, so one paper cannot dominate."""
    if not passages:
        return []
    docs = [tokens(p["text"]) for p in passages]
    avg = sum(len(d) for d in docs) / len(docs) or 1
    df = Counter(t for d in docs for t in set(d))
    n = len(docs)
    q = Counter(terms)
    scored = []
    for p, d in zip(passages, docs):
        tf = Counter(d)
        s = 0.0
        for t, qn in q.items():
            if t in tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += qn * idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(d) / avg))
        low = p["text"].lower()
        s += sum(2.0 for ph in phrases if ph and ph in low)
        if p["section"] != "abstract":
            s *= 1.15  # methods / results say what was actually done with the data
        scored.append((s, p))
    scored.sort(key=lambda x: -x[0])
    out, per = [], Counter()
    for s, p in scored:
        if s <= 0 or per[p["paper_id"]] >= per_paper:
            continue
        per[p["paper_id"]] += 1
        out.append(dict(p, score=round(s, 3)))
        if len(out) >= k:
            break
    for i, p in enumerate(out, 1):
        p["passage_id"] = f"P{i}"
    return out


# ---------------------------------------------------------------- queries

def _term(x):
    x = re.sub(r"\(unconfirmed\)|unconfirmed|unknown", "", str(x or ""), flags=re.I)
    x = re.sub(r"[\"()\[\]{}:*?]", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def block_name(label, pattern):
    """Short measurement name of a block: the text its column names share, cut back to
    the measurement part ('LFQ intensity S0' -> 'LFQ intensity'), else the label's
    first phrase."""
    if isinstance(pattern, dict) and pattern.get("text"):
        txt, side = pattern["text"], pattern.get("side", "prefix")
        if side == "prefix":
            txt = re.split(r"\d", txt, maxsplit=1)[0]
            if txt and txt[-1].isalnum() and re.search(r"[\s_.\-]", txt):
                txt = re.sub(r"[^\s_.\-]*$", "", txt)
        else:
            txt = re.split(r"\d", txt)[-1]
            if txt and txt[0].isalnum() and re.search(r"[\s_.\-]", txt):
                txt = re.sub(r"^[^\s_.\-]*", "", txt)
        txt = txt.strip(" _.-")
        if len(txt) >= 3 and "." + txt.lower() not in FILE_EXTENSIONS:
            return _term(txt)
    return _term(re.split(r"[,;(]| apparently ", label or "")[0])[:60]


def default_queries(desc):
    """Queries built from the confirmed description (used with the AI's own queries)."""
    out = []
    for a in desc["assays"]:
        sw, om = _term(a.get("source_software")), _term(a.get("omics_type"))
        names = [b["name"] for b in desc["value_blocks"] if b["assay_label"] == a["assay_label"] and b["name"]]
        names = list(dict.fromkeys(names))
        if len(names) > 1:
            out.append(" AND ".join(f'"{x}"' for x in names[:3]) + (f' AND "{sw}"' if sw else ""))
        for x in names[:2]:
            out.append(f'"{x}"' + (f' AND "{om}"' if om else "") + (f' AND "{sw}"' if sw and len(names) == 1 else ""))
        if sw and om:
            out.append(f'"{sw}" AND "{om}" AND (quantification OR "data analysis")')
    return list(dict.fromkeys(q for q in out if q.strip()))


# ---------------------------------------------------------------- AI synthesis

SYSTEM = BRIEFING + """

## This call: what does the literature say?
You get the confirmed description of an uploaded table (assays, and value blocks with their labels
and computed statistics) and numbered passages retrieved from papers in Europe PMC.
For every value block:
- description: what this measurement is, in plain words.
- typical_use: how studies with similar data used it (e.g. used as the quantitative matrix, used
  only for quality control, reported alongside another measurement).
- suggested_for_analysis: "yes" if the passages show that similar studies analysed this
  measurement, "no" if they show it is typically not analysed (e.g. only a count or QC metric),
  "unsure" otherwise. Several blocks may be "yes"; do not rank them.
- reason: one or two sentences.
- citations: passage ids with an exact quote copied word for word from that passage (6-30 words).
Rules:
- Use ONLY the passages. Do not present outside knowledge as what the literature says. When the
  passages do not cover a block, say so in reason and use "unsure".
- A citation whose quote is not found word for word in its passage is discarded by code; a "yes"
  without a valid citation becomes "unsure".
- for_later_steps: list passages that will matter for later preprocessing steps (normalization,
  missing values / imputation, log transformation, filtering, batch effects), with a short topic.
  Only point to them; do not give preprocessing advice now.
- summary: 2-3 sentences on what kind of data this is according to the passages, with citations.
"""


class Cite(BaseModel):
    passage_id: str
    quote: str = ""


class BlockLit(BaseModel):
    group_id: str
    description: str = ""
    typical_use: str = ""
    suggested_for_analysis: str = "unsure"
    reason: str = ""
    citations: List[Cite] = []


class LaterStep(BaseModel):
    topic: str
    passage_ids: List[str] = []


class LitResponse(BaseModel):
    summary: str = ""
    summary_citations: List[Cite] = []
    blocks: List[BlockLit] = []
    for_later_steps: List[LaterStep] = []


def response_schema():
    s = lambda **k: dict(type="STRING", **k)
    cite = {"type": "OBJECT", "properties": {"passage_id": s(), "quote": s()}, "required": ["passage_id", "quote"]}
    return {"type": "OBJECT", "properties": {
        "summary": s(),
        "summary_citations": {"type": "ARRAY", "items": cite},
        "blocks": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "group_id": s(), "description": s(), "typical_use": s(),
            "suggested_for_analysis": s(enum=SUGGESTED),
            "reason": s(), "citations": {"type": "ARRAY", "items": cite}},
            "required": ["group_id", "description", "typical_use", "suggested_for_analysis", "reason", "citations"]}},
        "for_later_steps": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "topic": s(), "passage_ids": {"type": "ARRAY", "items": s()}}, "required": ["topic", "passage_ids"]}},
    }, "required": ["summary", "summary_citations", "blocks", "for_later_steps"]}


def _norm(x):
    return " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (x or "").lower())).strip() + " "


def verify(cites, by_id):
    """Keep citations whose quote appears word for word in the cited passage."""
    ok, bad = [], []
    for c in cites:
        p = by_id.get(c.passage_id)
        q = _norm(c.quote)
        if p is None:
            bad.append({"passage_id": c.passage_id, "quote": c.quote, "reason": "no such passage"})
        elif len(q.split()) < 4:
            bad.append({"passage_id": c.passage_id, "quote": c.quote, "reason": "quote too short to verify"})
        elif q not in _norm(p["text"]):
            bad.append({"passage_id": c.passage_id, "quote": c.quote, "reason": "quote not found in the passage"})
        else:
            ok.append({"passage_id": c.passage_id, "quote": c.quote.strip(), "paper_id": p["paper_id"]})
    return ok, bad


def synthesize(desc, passages, log=None, mock_fn=None):
    """-> (result dict, error or None). Result blocks are keyed by group_id and verified."""
    payload = {"file": desc, "passages": [{"id": p["passage_id"], "paper": p["paper_id"], "section": p["section"],
                                           "text": p["text"]} for p in passages]}
    prompt = "Literature (JSON):\n" + json.dumps(payload, ensure_ascii=False)
    if log:
        log("literature_ai_request", {"prompt_version": PROMPT_VERSION, "n_passages": len(passages),
                                      "description": desc})
    raw, err = None, None
    for attempt in (1, 2):
        try:
            raw, m = llm.complete_json(SYSTEM, prompt, response_schema(), mock_fn)
            parsed = LitResponse.model_validate_json(raw)
            break
        except (ValueError, ValidationError) as e:
            err = f"invalid JSON from the model ({str(e).splitlines()[0][:160]})"
            parsed = None
        except llm.LLMError as e:
            return None, str(e)
        except Exception as e:
            traceback.print_exc()
            return None, f"Unexpected error: {type(e).__name__}: {e}"
    if log:
        log("literature_ai_response_raw", {"raw": raw, "error": None if parsed else err})
    if parsed is None:
        return None, err
    by_id = {p["passage_id"]: p for p in passages}
    valid_blocks = {b["group_id"] for b in desc["value_blocks"]}
    out = {"blocks": {}, "rejected": []}
    s_ok, s_bad = verify(parsed.summary_citations, by_id)
    out["summary"] = {"text": parsed.summary.strip(), "citations": s_ok, "supported": bool(s_ok)}
    out["rejected"] += [dict(b, where="summary") for b in s_bad]
    for b in parsed.blocks:
        if b.group_id not in valid_blocks or b.group_id in out["blocks"]:
            out["rejected"].append({"group_id": b.group_id, "reason": "not a kept value block"})
            continue
        ok, bad = verify(b.citations, by_id)
        out["rejected"] += [dict(x, where=b.group_id) for x in bad]
        sug = b.suggested_for_analysis if b.suggested_for_analysis in SUGGESTED else "unsure"
        reason = b.reason.strip()
        if sug == "yes" and not ok:
            sug = "unsure"
            reason = (reason + " " if reason else "") + "(Downgraded to unsure: no citation could be verified.)"
        out["blocks"][b.group_id] = {"description": b.description.strip(), "typical_use": b.typical_use.strip(),
                                     "suggested_for_analysis": sug, "reason": reason, "citations": ok,
                                     "supported": bool(ok)}
    out["for_later_steps"] = [{"topic": x.topic.strip()[:80], "passage_ids": [i for i in x.passage_ids if i in by_id]}
                              for x in parsed.for_later_steps if any(i in by_id for i in x.passage_ids)]
    return out, None


# ---------------------------------------------------------------- orchestration

def run(desc, queries, ai_on=True, log=None, on_progress=None, mock_fn=None):
    """Search, rank, synthesize. Returns the 'literature' record stored in the draft."""
    def prog(done, total, msg):
        say(msg)
        if on_progress:
            on_progress(done, total, msg)

    queries = [q for q in dict.fromkeys(q.strip() for q in queries if q and q.strip())][:MAX_QUERIES]
    rec = {"source": "Europe PMC", "endpoint": EUROPEPMC_URL, "queries": queries, "ran_at": time.strftime("%Y-%m-%d %H:%M"),
           "papers": {}, "passages": [], "summary": None, "blocks": {}, "for_later_steps": [], "rejected": [],
           "ai_used": False, "error": None, "status": "done"}
    if not queries:
        rec.update(status="error", error="No search query: describe the value blocks first.")
        return rec
    total = len(queries) + 3
    papers = {}
    for n, q in enumerate(queries):
        prog(n, total, f"Searching Europe PMC: {q}")
        for p in search(q):
            papers.setdefault(p["paper_id"], p)
    rec["papers"] = papers
    if log:
        log("literature_search", {"queries": queries, "n_papers": len(papers),
                                  "papers": [{k: p[k] for k in ("paper_id", "title", "year", "query")} for p in papers.values()]})
    if not papers:
        rec.update(status="no_results", error="Europe PMC found no papers for these queries. Edit the queries and search again.")
        return rec
    # rank with abstracts first, then fetch full text for the best open-access papers
    phrases = [b["name"].lower() for b in desc["value_blocks"] if b["name"]]
    terms = [t for q in queries for t in tokens(q)] + [t for ph in phrases for t in tokens(ph)] * 2 + \
        [t for b in desc["value_blocks"] for t in tokens(b["label"])]
    first = rank(build_passages(list(papers.values()), {}), terms, phrases, k=40, per_paper=40)
    order = list(dict.fromkeys(p["paper_id"] for p in first))
    oa = [pid for pid in order if papers[pid]["open_access"] and papers[pid]["pmcid"]][:FULL_TEXTS]
    texts = {}
    for k, pid in enumerate(oa):
        prog(len(queries), total, f"Reading open-access full text {k + 1}/{len(oa)}: {papers[pid]['title'][:70]}")
        try:
            texts[pid] = full_text(papers[pid]["pmcid"])
        except LiteratureError as e:
            say(f"full text skipped: {e}")
    passages = rank(build_passages(list(papers.values()), texts), terms, phrases)
    rec["passages"] = passages
    rec["full_text_papers"] = [pid for pid in texts if texts[pid]]
    if not ai_on:
        rec["status"] = "retrieved_only"
        prog(total, total, f"{len(passages)} relevant passages from {len(papers)} papers (AI off: not summarised)")
        return rec
    prog(len(queries) + 1, total, f"Asking the AI what {len(passages)} passages say about each block")
    res, err = synthesize(desc, passages, log, mock_fn)
    if err:
        rec.update(status="retrieved_only", error=f"The AI could not summarise the papers: {err}")
        return rec
    rec.update(res, ai_used=True)
    prog(total, total, "Literature suggestions checked against the passages")
    return rec
