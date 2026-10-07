"""Command line: python -m prism <command> (v3 §7). Everything here is deterministic.

    python -m prism session new [--name N]
    python -m prism session list
    python -m prism session show SESSION
    python -m prism session add --session SESSION --output STEP0_OUTPUT_FOLDER [--name N]
    python -m prism import --data X --schema S [--metadata M] --session SESSION
    python -m prism session merge-report --session SESSION
    python -m prism audit run --session SESSION [--factor A4] [--dataset D1] [--param key=value]
    python -m prism audit show --session SESSION [--run RUN]
    python -m prism audit report --session SESSION [--run RUN] --format html|json
    python -m prism audit compare --session SESSION RUN_A RUN_B
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


def _audit_cmds(sub):
    p = sub.add_parser("audit", help="the Tier 1 audit")
    s2 = p.add_subparsers(dest="action", required=True)
    r = s2.add_parser("run")
    r.add_argument("--session", required=True)
    r.add_argument("--factor", action="append", default=None, help="audit id or code (A4 / missingness); repeatable")
    r.add_argument("--dataset", action="append", default=None)
    r.add_argument("--param", action="append", default=[], help="key=value, recorded in the manifest")
    sh = s2.add_parser("show")
    sh.add_argument("--session", required=True)
    sh.add_argument("--run", default=None)
    rp = s2.add_parser("report")
    rp.add_argument("--session", required=True)
    rp.add_argument("--run", default=None)
    rp.add_argument("--format", choices=["html", "json"], default="html")
    rp.add_argument("--out", default=None)
    c = s2.add_parser("compare")
    c.add_argument("--session", required=True)
    c.add_argument("run_a")
    c.add_argument("run_b")


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


def _dispatch(args):
    from . import store
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
    if args.cmd == "audit":
        from .audit import engine
        st = store.load(args.session)
        if args.action == "run":
            run = engine.run(st, factors=args.factor, datasets=args.dataset, params=parse_params(args.param), who="user")
            print(run["run_id"])
        elif args.action == "show":
            print(json.dumps(engine.show(st, args.run), indent=2))
        elif args.action == "report":
            from .audit import report
            path = report.write(st, args.run, args.format, args.out)
            print(path)
        elif args.action == "compare":
            print(json.dumps(engine.compare(st, args.run_a, args.run_b), indent=2))
        return 0
    return 2
