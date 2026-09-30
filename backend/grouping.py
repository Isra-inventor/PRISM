"""Applying grouping proposals: the AI's (or yours), never code's own judgment.

Which columns belong together is proposed by the AI (per chunk, then a
consolidation pass across chunks) or decided by you in the wizard. Code only:
  - checks that every column named exists and is claimed once (first claim wins),
  - turns every column nobody mentioned into its own 'unresolved' group, so a
    column is never silently dropped,
  - applies suggest_split / propose_merge / cross_chunk_merges when every group
    id they name exists, and rejects (and logs) them otherwise.
"""

from __future__ import annotations

from collections import OrderedDict

from .schema import UNRESOLVED


def unmentioned_item():
    return {"role": UNRESOLVED, "assay_label": None, "label": "", "audit_kind": None, "marks_rows_as_suspect": False,
            "confidence": 0.0, "evidence": "Not placed in any group by the AI: choose what this column is.",
            "source": "none", "validation": {"status": "ok", "messages": []}}


def collect(proposed, allowed, label_to_idx, ns, rejected):
    """proposed: [(ai_group_id, [column names], item)] for one chunk.
    Returns OrderedDict pid -> {"indices", "item", "origin"}; ns namespaces the ids."""
    out, claimed = OrderedDict(), {}
    for gid, names, item in proposed:
        pid = f"{gid}{ns}"
        if pid in out:
            rejected.append({"group_id": gid, "reason": "This group id was used twice; the second group was rejected."})
            continue
        idx = []
        for name in names or []:
            i = label_to_idx.get(name)
            if i is None or i not in allowed:
                rejected.append({"group_id": gid, "column": name,
                                 "reason": "This column does not exist in the digest; the claim was rejected."})
            elif i in claimed:
                rejected.append({"group_id": gid, "column": name,
                                 "reason": f"Column already placed in group {claimed[i]}; the second claim was rejected."})
            elif i not in idx:
                idx.append(i)
                claimed[i] = gid
        if not idx:
            rejected.append({"group_id": gid, "reason": "The group names no valid column; it was rejected."})
            continue
        out[pid] = {"indices": idx, "item": item, "origin": "ai_proposed"}
    for i in sorted(allowed):
        if i not in claimed:
            out[f"u{i}{ns}"] = {"indices": [i], "item": unmentioned_item(), "origin": "unmentioned"}
    return out


def apply_split(groups, pid, names, label_to_idx, make_item, rejected, where="ai"):
    """Pull the named columns out of group pid, one new group each."""
    g = groups.get(pid)
    if g is None:
        rejected.append({"group_id": pid, "reason": "suggest_split names a group that does not exist; not applied."})
        return []
    cols = [label_to_idx.get(n) for n in names or []]
    bad = [n for n, i in zip(names or [], cols) if i is None or i not in g["indices"]]
    if bad:
        rejected.append({"group_id": pid, "columns": bad, "reason": "suggest_split names columns that are not in this "
                         "group; those were ignored."})
    cols = [i for i in cols if i is not None and i in g["indices"]]
    if not cols:
        return []
    if len(cols) == len(g["indices"]):
        rejected.append({"group_id": pid, "reason": "suggest_split would empty the group; not applied."})
        return []
    g["indices"] = [i for i in g["indices"] if i not in cols]
    new = []
    for i in cols:
        npid = f"{pid}s{i}"
        groups[npid] = {"indices": [i], "item": make_item(i), "origin": f"split_{where}", "split_from": pid}
        new.append(npid)
    return new


def apply_merge(groups, pids, reason, rejected, where="ai"):
    """Merge the named groups into one; all ids must exist. The item of the group
    with the most columns is kept for the merged group."""
    pids = list(dict.fromkeys(pids or []))
    missing = [p for p in pids if p not in groups]
    if len(pids) < 2 or missing:
        rejected.append({"group_ids": pids, "reason": ("merge names groups that do not exist: " + ", ".join(missing))
                         if missing else "a merge needs at least two groups", "merge_reason": reason})
        return None
    keep = max(pids, key=lambda p: (len(groups[p]["indices"]), -pids.index(p)))
    for p in pids:
        if p != keep:
            groups[keep]["indices"] = groups[keep]["indices"] + groups.pop(p)["indices"]
    groups[keep]["origin"] = f"merged_{where}"
    groups[keep]["merged_from"] = pids
    groups[keep]["merge_reason"] = reason
    return keep


def finalize(groups, first_id=1, keep_ids=None):
    """pid -> final group id g1, g2, ... in file order. keep_ids maps a pid to an
    existing id to keep (e.g. an unchanged group on reconsider)."""
    order = sorted(groups, key=lambda p: min(groups[p]["indices"]))
    id_map, n = {}, first_id
    for p in order:
        if keep_ids and p in keep_ids:
            id_map[p] = keep_ids[p]
        else:
            id_map[p] = f"g{n}"
            n += 1
    return order, id_map
