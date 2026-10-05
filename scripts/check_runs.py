# Read-only diagnostics for v2.4 section 0 (reads backend/logs and backend/sessions, changes nothing).
# Run from the PRISM folder:  python scripts/check_runs.py [somascan_session_id] [metabolomics_session_id]
import glob, json, os, sys
SOMA, METAB = (sys.argv[1:3] + ["4412909f9e4a", "151b352ce3cd"][len(sys.argv[1:3]):])[:2]
META_COLS = ["nhp_id", "time_point", "TimePoint", "study_group", "SubjectID", "nhp_week_index"]

def events(sid):
    p = os.path.join("backend", "logs", sid + ".jsonl")
    if not os.path.exists(p):
        print(f"  !! {p} not found"); return []
    return [json.loads(l) for l in open(p, encoding="utf-8")]

def cols_of(digest):
    return [c.get("column") for c in digest.get("columns", [])] + \
           [c for g in digest.get("groups", []) for c in (g.get("columns") or g.get("first_columns", []) + g.get("last_columns", []))]

print(f"=== 0.1 SomaScan run {SOMA}: did metadata columns reach the AI?")
ev = events(SOMA)
reqs = [e for e in ev if e.get("event") == "ai_request"]
print(f"  {len(reqs)} ai_request(s)")
for k, r in enumerate(reqs):
    names = cols_of(r.get("digest", {}))
    print(f"  request {k + 1} (kind={r.get('kind', 'digest')}): {len(names)} columns; metadata names present: "
          f"{[m for m in META_COLS if m in names] or 'NONE'}")
for e in ev:
    if e.get("event") == "metadata_upload":
        p = e
        print(f"  metadata file {p.get('filename')!r}: id column chosen by code = {p.get('id_column')!r}, "
              f"matched {p.get('report', {}).get('n_matched')}")

print("\n=== 0.2 where did nhp_week_index go? (every saved session)")
for st in sorted(glob.glob(os.path.join("backend", "sessions", "*", "state.json")), key=os.path.getmtime):
    d = (json.load(open(st, encoding="utf-8")).get("draft") or {})
    meta = d.get("metadata") or {}
    if not meta.get("columns"):
        continue
    names = [c["column"] for c in meta["columns"]]
    nw = next((c for c in meta["columns"] if c["column"] == "nhp_week_index"), None)
    print(f"  {st.split(os.sep)[-2]} schema {d.get('schema_version')} file {meta.get('filename')!r}: {len(names)} columns, "
          f"id column = {meta.get('id_column')!r}, nhp_week_index = "
          f"{'ABSENT from the file' if nw is None else nw['role'] + ', keep=' + str(nw.get('keep'))}")

print(f"\n=== 0.3 / 0.4 metabolomics run {METAB}: chunks, consolidation, orphans")
ev = events(METAB)
for e in ev:
    p = e
    if e.get("event") == "ai_request":
        dg = p.get("digest", {})
        ch = dg.get("chunk")
        if p.get("kind", "digest") == "digest":
            print(f"  chunk {ch['index'] if ch else 1}/{ch['of'] if ch else 1}: {len(dg.get('columns', []))} columns")
        else:
            print(f"  {p.get('kind')} call with {len(dg.get('groups', []))} group summaries")
    elif e.get("event") == "ai_response_raw" and '"cross_chunk_merges"' in (p.get("raw") or ""):
        print("  consolidation answered:", json.loads(p["raw"]).get("cross_chunk_merges"))
    elif e.get("event") == "ai_response_raw" and p.get("error"):
        print("  AI call FAILED:", p["error"])
    elif e.get("event") == "grouping_result":
        print(f"  grouping_result: {p['chunks']} chunk(s), {p['n_groups']} groups, merges applied: {p['merges']}, "
              f"rejected: {len(p['rejected'])}")
    elif e.get("event") == "step_confirmed":
        for ch in p.get("changes") or []:
            if "role" in (ch.get("changes") or {}) and ch["changes"]["role"][0] == "unresolved":
                print(f"  step '{p['step']}': you set {ch['group_id']} from unresolved to {ch['changes']['role'][1]}")
print("\n  PRISM_GROUPING_CHUNK_SIZE in .env:", next((l.strip() for l in open(".env", encoding="utf-8-sig")
                                                    if "GROUPING_CHUNK_SIZE" in l), "not set (default 150)")
      if os.path.exists(".env") else "no .env (default 150)")
