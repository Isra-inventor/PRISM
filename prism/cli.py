"""Command line: python -m prism <command> (v3 §7). Everything here is deterministic.

    python -m prism session new [--name N]
    python -m prism session list
    python -m prism session show SESSION
    python -m prism session add --session SESSION --output STEP0_OUTPUT_FOLDER [--name N]
    python -m prism import --data X --schema S [--metadata M] --session SESSION
    python -m prism session merge-report --session SESSION
    python -m prism session merge-decide --session SESSION ITEM_ID DECISION [--unified-id U]
    python -m prism session merge-mapping --session SESSION MAPPING.csv
    python -m prism audit run --session SESSION [--factor A4] [--dataset D1] [--param key=value]
    python -m prism audit show --session SESSION [--run RUN]
    python -m prism audit report --session SESSION [--run RUN] --format html|json
    python -m prism audit override --session SESSION --kind sample_role --sample S --role qc [--reason R]
    python -m prism audit compare --session SESSION RUN_A RUN_B
    python -m prism report --session SESSION [--run RUN] [--format html|json] [--out FILE]
    python -m prism audit run --session SESSION --state STATE_ID      (the checkpoint battery of that state)
    python -m prism audit compare --session SESSION --state A --state B
    python -m prism state list|show|diff --session SESSION [STATE] [STATE_B]
    python -m prism intake --session SESSION [--contrast C] [--protected A,B] [--nuisance A,B]
                           [--question-type T] [--downstream F]
    python -m prism plan --session SESSION                       (build plan.json; runs the audit if needed)
    python -m prism plan approve ENTRY_ID --session SESSION [--candidate N] [--user U]
    python -m prism plan reject ENTRY_ID --session SESSION [--reason R]
    python -m prism plan replay --session SESSION
    python -m prism registry show [--json]
    python -m prism evidence check
"""

from __future__ import annotations

import argparse
import json
import sys


def _session_cmds(sub):
    p = sub.add_parser("session", help="sessions of one or more datasets")
    s2 = p.add_subparsers(dest="action", required=True)
    n = s2.add_parser("new")
    n.add_argument("--name", default="")
    s2.add_parser("list")
    sh = s2.add_parser("show")
    sh.add_argument("session")
    a = s2.add_parser("add", help="add a finalized Step 0 output folder as a confirmed dataset")
    a.add_argument("--session", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--name", default=None)
    m = s2.add_parser("merge-report")
    m.add_argument("--session", required=True)
    md = s2.add_parser("merge-decide", help="decide on a merge item: ms.. confirm|dismiss, mc.. take:D1|keep_both|drop, "
                                            "mq.. an option or dismiss")
    md.add_argument("--session", required=True)
    md.add_argument("item_id")
    md.add_argument("decision")
    md.add_argument("--unified-id", default=None)
    mm = s2.add_parser("merge-mapping", help="upload a mapping CSV (dataset_id, sample_id, unified_id)")
    mm.add_argument("--session", required=True)
    mm.add_argument("csv")


def _audit_cmds(sub):
    p = sub.add_parser("audit", help="the Tier 1 audit")
    s2 = p.add_subparsers(dest="action", required=True)
    r = s2.add_parser("run")
    r.add_argument("--session", required=True)
    r.add_argument("--factor", action="append", default=None, help="audit id or code (A4 / missingness); repeatable")
    r.add_argument("--dataset", action="append", default=None)
    r.add_argument("--param", action="append", default=[], help="key=value, recorded in the manifest")
    r.add_argument("--state", default=None, help="a state id (or prefix): runs that state's checkpoint battery")
    sh = s2.add_parser("show")
    sh.add_argument("--session", required=True)
    sh.add_argument("--run", default=None)
    rp = s2.add_parser("report")
    rp.add_argument("--session", required=True)
    rp.add_argument("--run", default=None)
    rp.add_argument("--format", choices=["html", "json"], default="html")
    rp.add_argument("--out", default=None)
    o = s2.add_parser("override", help="add an entry to overrides.json (input to the next run)")
    o.add_argument("--session", required=True)
    from .audit.overrides import KINDS
    o.add_argument("--kind", required=True, choices=list(KINDS))
    o.add_argument("--item", default=None, help="processing_state item")
    o.add_argument("--feature", default=None, help="control_feature")
    o.add_argument("--level", default=None, help="feature_level")
    o.add_argument("--assay", default=None)
    o.add_argument("--sample", default=None)
    o.add_argument("--role", default=None)
    o.add_argument("--column", default=None)
    o.add_argument("--dataset", default=None)
    o.add_argument("--key", default=None)
    o.add_argument("--value", default=None)
    o.add_argument("--reason", default="")
    c = s2.add_parser("compare")
    c.add_argument("--session", required=True)
    c.add_argument("run_a", nargs="?")
    c.add_argument("run_b", nargs="?")
    c.add_argument("--state", action="append", default=None, help="two states: a before/after table")


def _plan_cmds(sub):
    it = sub.add_parser("intake", help="the four intake answers (v4 section 8.1)")
    it.add_argument("--session", required=True)
    it.add_argument("--contrast", default=None)
    it.add_argument("--protected", default=None)
    it.add_argument("--nuisance", default=None)
    it.add_argument("--question-type", default=None)
    it.add_argument("--downstream", default=None)
    pl = sub.add_parser("plan", help="the plan engine (v4 section 8.6)")
    pl.add_argument("--session", required=False)
    s2 = pl.add_subparsers(dest="action", required=False)
    for name in ("approve", "reject"):
        a = s2.add_parser(name)
        a.add_argument("entry_id")
        a.add_argument("--session", required=True)
        a.add_argument("--candidate", type=int, default=0)
        a.add_argument("--user", default="user")
        a.add_argument("--reason", default="")
    r = s2.add_parser("replay")
    r.add_argument("--session", required=True)
    sh = s2.add_parser("show")
    sh.add_argument("--session", required=True)


def _state_cmds(sub):
    p = sub.add_parser("state", help="data states (v4 section 2)")
    s2 = p.add_subparsers(dest="action", required=True)
    ls = s2.add_parser("list")
    ls.add_argument("--session", required=True)
    ls.add_argument("--dataset", default=None)
    sh = s2.add_parser("show")
    sh.add_argument("--session", required=True)
    sh.add_argument("state")
    df = s2.add_parser("diff")
    df.add_argument("--session", required=True)
    df.add_argument("state_a")
    df.add_argument("state_b")


def build_parser():
    ap = argparse.ArgumentParser(prog="prism", description="PRISM sessions, schema import, merge and Tier 1 audit")
    ap.add_argument("--root", default=None, help="sessions folder (default: backend/sessions)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    _session_cmds(sub)
    i = sub.add_parser("import", help="import a dataset with an existing schema.json (skips the wizard)")
    i.add_argument("--data", required=True)
    i.add_argument("--schema", required=True)
    i.add_argument("--metadata", default=None)
    i.add_argument("--session", required=True)
    i.add_argument("--accept", action="store_true", help="accept the import if every check passes")
    _audit_cmds(sub)
    _state_cmds(sub)
    _plan_cmds(sub)
    fr = sub.add_parser("report", help="the final session report (datasets, merge, audit) as one file")
    fr.add_argument("--session", required=True)
    fr.add_argument("--run", default=None)
    fr.add_argument("--format", choices=["html", "json"], default="html")
    fr.add_argument("--out", default=None)
    rg = sub.add_parser("registry", help="the step registry (v4 section 1)")
    rg2 = rg.add_subparsers(dest="action", required=True)
    rs = rg2.add_parser("show")
    rs.add_argument("--json", action="store_true")
    ev = sub.add_parser("evidence", help="evidence levels and the bibliography (v4 section 8.5)")
    ev2 = ev.add_subparsers(dest="action", required=True)
    ev2.add_parser("check")
    return ap


def parse_params(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--param needs key=value, got '{it}'")
        k, v = it.split("=", 1)
        try:
            out[k.strip()] = json.loads(v)
        except ValueError:
            out[k.strip()] = v
    return out


def main(argv=None):
    args = build_parser().parse_args(argv)
    from . import store
    if args.root:
        store.ROOT = args.root
    try:
        return _dispatch(args)
    except store.SessionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        from .session.merge import MergeError
        from .audit.engine import AuditError
        from .audit.overrides import OverrideError
        from .executor import ExecutorError
        from .executor.state import StateError
        if isinstance(e, (MergeError, AuditError, OverrideError, ExecutorError, StateError)):
            print(f"error: {e}", file=sys.stderr)
            return 2
        raise


def _dispatch(args):
    from . import store
    if args.cmd == "registry":
        from . import registry
        if args.json:
            print(json.dumps(registry.registry(), indent=2))
            return 0
        for s in registry.steps():
            mark = "" if s["implemented"] else "  (not implemented)"
            print(f"{s['id']:<4} {s['kind']:<10} {s['scope']:<16} {s['name']}{mark}")
            print(f"     needs: {', '.join(s['needs'])}; evidence level: {s['evidence_level']}")
        return 0
    if args.cmd == "evidence":
        from . import registry
        res = registry.check()
        for k, v in res["counted"].items():
            print(f"  {k}: {v} entries")
        for p in res["problems"]:
            print(f"  problem: {p}", file=sys.stderr)
        print("evidence check: ok" if res["ok"] else f"evidence check: {len(res['problems'])} problem(s)")
        return 0 if res["ok"] else 1
    if args.cmd == "intake":
        from .plan import intake
        st = store.load(args.session)
        cur = intake.get(st)
        for k, v in (("contrast_variable", args.contrast), ("protected_variables", args.protected),
                     ("nuisance_variables", args.nuisance), ("question_type", args.question_type),
                     ("downstream_family", args.downstream)):
            if v is not None:
                cur[k] = v
        try:
            doc = intake.set_intake(st, cur)
        except intake.IntakeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(json.dumps(doc["intake"], indent=2))
        return 0
    if args.cmd == "plan":
        from .plan import engine as pe
        sid = args.session
        if not sid:
            raise SystemExit("--session is required")
        st = store.load(sid)
        try:
            if args.action in (None, "show"):
                plan = pe.build(st) if args.action is None else pe.load(st)
                for e in plan["entries"]:
                    if e.get("checkpoint"):
                        print(f"  -- {e['checkpoint']} --")
                        continue
                    print(f"  {e['entry_id']:<28} {e['status']:<22} {e['because'][0] if e['because'] else ''}")
                    for i, c in enumerate(e["candidates"]):
                        print(f"      [{i}] {c['status']:<20} {c['label']}")
                print(str(pe.plan_dir(st) / "plan.json"))
            elif args.action == "approve":
                out = pe.approve(st, args.entry_id, args.candidate, args.user)
                print(json.dumps(out, indent=2))
            elif args.action == "reject":
                print(json.dumps(pe.reject(st, args.entry_id, args.user, args.reason), indent=2))
            elif args.action == "replay":
                r = pe.replay(st)
                print(json.dumps(r, indent=2))
                return 0 if r["identical"] else 1
        except pe.PlanError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        return 0
    if args.cmd == "state":
        from .executor import state as state_mod
        st = store.load(args.session)
        try:
            if args.action == "list":
                for s in state_mod.list_states(st, args.dataset):
                    c = s["counts"]
                    print(f"{s['dataset']:<4} {s['label']:<4} {s['state_id'][:12]} {s['checkpoint']} {s['scale']:<8} "
                          f"{c['n_features']}x{c['n_samples']} nan={c['n_nan']} {s['op'] or 'root'}{'  HEAD' if s['head'] else ''}")
            elif args.action == "show":
                did, sid = state_mod.find(st, args.state)
                print(json.dumps(state_mod.meta(st, did, sid), indent=2))
            elif args.action == "diff":
                da, a = state_mod.find(st, args.state_a)
                db, b = state_mod.find(st, args.state_b)
                if da != db:
                    raise state_mod.StateError("The two states belong to different datasets.")
                print(json.dumps(state_mod.diff(st, da, a, b), indent=2))
        except state_mod.StateError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        return 0
    if args.cmd == "session":
        if args.action == "new":
            s = store.create(args.name)
            print(s.sid)
        elif args.action == "list":
            for d in store.list_sessions():
                print(f"{d['session_id']}  {d['name']}  {len(d['datasets'])} dataset(s)")
        elif args.action == "show":
            print(json.dumps(store.load(args.session).data, indent=2))
        elif args.action == "add":
            from pathlib import Path
            st = store.load(args.session)
            out = Path(args.output)
            ds = st.add_dataset(args.name or out.name, "output_folder", "imported_awaiting_confirm")
            store.publish_output(st, ds["dataset_id"], out, {"mode": "output_folder", "source": str(out)})
            print(ds["dataset_id"])
        elif args.action == "merge-report":
            from .session import merge
            print(json.dumps(merge.report(store.load(args.session)), indent=2))
        elif args.action == "merge-decide":
            from .session import merge
            doc = merge.decide(store.load(args.session), args.item_id, args.decision, "user", args.unified_id)
            print("ready for audit" if doc["ready_for_audit"] else "\n".join(doc["blocking"]))
        elif args.action == "merge-mapping":
            from pathlib import Path
            from .session import merge
            doc = merge.set_mapping_csv(store.load(args.session), Path(args.csv).read_bytes(), Path(args.csv).name)
            print(f"{len(doc['cross_dataset']['id_mapping']['entries'])} mapped sample(s)")
        return 0
    if args.cmd == "import":
        from .io import importer
        st = store.load(args.session)
        rep = importer.import_dataset(st, args.data, args.schema, args.metadata)
        print(json.dumps({k: rep[k] for k in ("dataset_id", "mode", "status", "checks", "differences")}, indent=2))
        if args.accept:
            importer.accept(st, rep["dataset_id"])
            print(f"accepted {rep['dataset_id']}")
        return 0
    if args.cmd == "report":
        from .audit import report
        print(report.write_final(store.load(args.session), args.run, args.format, args.out))
        return 0
    if args.cmd == "audit":
        from .audit import engine
        st = store.load(args.session)
        if args.action == "run":
            run = engine.run(st, factors=args.factor, datasets=args.dataset, params=parse_params(args.param), who="user",
                             state=args.state)
            print(run["run_id"])
            for i in run["findings"]:
                print(f"  {i['audit_id']:<4} {i['dataset']}/{i['assay']:<8} {i['status']:<22} {', '.join(i['indicators'])}")
        elif args.action == "show":
            print(json.dumps(engine.show(st, args.run), indent=2))
        elif args.action == "report":
            from .audit import report
            path = report.write(st, args.run, args.format, args.out)
            print(path)
        elif args.action == "override":
            from .audit import overrides
            val = parse_params([f"v={args.value}"])["v"] if args.value is not None else None
            if args.kind in ("processing_state", "zeros_are_missing", "floor_censored"):
                val = args.value
            e = overrides.add(st, {"kind": args.kind, "sample": args.sample, "role": args.role, "column": args.column,
                                   "dataset": args.dataset, "key": args.key, "value": val, "reason": args.reason,
                                   "item": args.item, "feature": args.feature, "level": args.level,
                                   "assay": args.assay})
            print(e["override_id"])
        elif args.action == "compare":
            if args.state:
                if len(args.state) != 2:
                    raise SystemExit("--state is given twice: --state BEFORE --state AFTER")
                print(json.dumps(engine.compare_states(st, args.state[0], args.state[1]), indent=2))
            else:
                print(json.dumps(engine.compare(st, args.run_a, args.run_b), indent=2))
        return 0
    return 2
