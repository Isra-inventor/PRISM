"""Deterministic consistency checks on the proposed structure. Questions, never fixes.

1. Fragmentation (v2.4 §16): kept value blocks, across assays, with the same samples
   and near-identical value distributions may be one measurement split in pieces. Code
   only asks (a queued question); it never merges by itself.
2. Sample-ID collisions across blocks (v2.3 §3): within an assay, blocks either
   measure the same samples in parallel (identical ID sets, descriptive names,
   e.g. LFQ / iBAQ / raw intensity) or hold different samples (disjoint sets).
   Anything else, e.g. 'DEAB_0' and 'T623_0' both becoming '0', is a collision
   and must be resolved before finishing.

Tolerances are settings (config.py), not claims about the data.
"""

from __future__ import annotations

import math
import re

from . import config

_SEP = " _.-:|/\\"
_CODE = re.compile(r"[A-Za-z0-9]{1,8}")


def name_code(g):
    """The literal code a group's column names start with ('DEAB' for DEAB_0, DEAB_4, ...),
    or None. For a single column: its leading token before a separator."""
    pat = g.get("pattern")
    if g["n_columns"] > 1:
        if not pat or pat.get("side") != "prefix":
            return None
        text = pat["text"]
    else:
        m = re.match(r"^([A-Za-z0-9]+)[" + re.escape(_SEP) + r"]", g["columns"][0])
        text = m.group(1) if m else ""
    # the shared part ends where the varying part starts: cut back to the last separator
    # ('DEAB_1' over DEAB_1, DEAB_12 -> 'DEAB_'; 'LFQ intensity S0' -> 'LFQ intensity ')
    cut = max(text.rfind(c) for c in _SEP)
    if cut >= 0 and text[-1] not in _SEP:
        text = text[:cut + 1]
    core = text.strip(_SEP)
    return core or None


def identifier_like(code):
    """A short alphanumeric code (with a digit, or all capitals), not a measurement term."""
    return bool(code and _CODE.fullmatch(code) and (re.search(r"\d", code) or code.isupper()))


def _within(a, b, factor):
    """a and b within a factor of each other (both positive), or both not positive."""
    if a is None or b is None:
        return False
    if a <= 0 or b <= 0:
        return a <= 0 and b <= 0
    return max(a, b) / min(a, b) <= factor


def near_identical(p, q):
    """Near-identical value distributions (v2.4 §16): medians within a factor of
    FRAGMENT_MEDIAN_FACTOR, p1 and p99 each within FRAGMENT_TAIL_FACTOR."""
    return (_within(p.get("median"), q.get("median"), config.FRAGMENT_MEDIAN_FACTOR)
            and _within(p.get("p1"), q.get("p1"), config.FRAGMENT_TAIL_FACTOR)
            and _within(p.get("p99"), q.get("p99"), config.FRAGMENT_TAIL_FACTOR))


def _stat_line(g, label, assay):
    p = g.get("profile") or {}
    return (f"{label or g['columns'][0]} ({g['n_columns']} col, assay '{assay}'): median {p.get('median')}, "
            f"p1 {p.get('p1')}, p99 {p.get('p99')}, zeros {round(100 * (p.get('frac_zero') or 0))}%, "
            f"missing {round(100 * (p.get('frac_na') or 0))}%")


def fragmentation(groups_by_id, items, layout, block_samples):
    """Kept value blocks, compared ACROSS assays (v2.4 §16, replacing the v2.3 identifier-name
    condition): two blocks are one measurement candidate when their value distributions are
    near-identical and they can be pieces of one matrix: samples in rows, the blocks are different
    features of the same samples; samples in columns, the same feature rows for different sample
    columns (blocks whose sample IDs are identical sets are parallel measurements such as LFQ vs
    iBAQ and are not compared). Returns clusters of two or more blocks."""
    gids = [g for g, it in items.items() if it["role"] == "value" and it.get("keep", True)
            and groups_by_id.get(g, {}).get("type") == "numeric"]
    if layout not in ("samples_in_rows", "samples_in_columns") or len(gids) < 2:
        return []
    parent = {g: g for g in gids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    sets = {g: set(block_samples.get(g) or []) for g in gids}
    for i, a in enumerate(gids):
        for b in gids[i + 1:]:
            if layout == "samples_in_columns" and sets[a] and sets[a] == sets[b]:
                continue          # identical sample sets: parallel measurements (LFQ vs iBAQ), not pieces
            if near_identical(groups_by_id[a]["profile"], groups_by_id[b]["profile"]):
                parent[find(a)] = find(b)
    clusters = {}
    for g in gids:
        clusters.setdefault(find(g), []).append(g)
    out = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda g: min(groups_by_id[g]["indices"]))
        assays = list(dict.fromkeys(items[g].get("assay_label") for g in members))
        out.append({"group_ids": members, "assays": assays,
                    "n_columns": sum(groups_by_id[g]["n_columns"] for g in members),
                    "evidence": [_stat_line(groups_by_id[g], items[g].get("label"), items[g].get("assay_label"))
                                 for g in members]})
    return out


def sample_collisions(blocks_ids, codes):
    """blocks_ids: {gid: [sample ids]} for one assay's kept blocks (samples in columns).
    Returns (collisions [{group_ids, ids}], structure) where structure is
    'single' | 'parallel' | 'disjoint' | 'collision'."""
    gids = list(blocks_ids)
    if len(gids) < 2:
        return [], "single"
    sets = {g: set(blocks_ids[g]) for g in gids}
    first = sets[gids[0]]
    if all(sets[g] == first for g in gids) and not any(identifier_like(codes.get(g)) for g in gids):
        return [], "parallel"
    clashes = {}
    for i, a in enumerate(gids):
        for b in gids[i + 1:]:
            for x in sets[a] & sets[b]:
                clashes.setdefault(x, set()).update((a, b))
    if not clashes:
        return [], "disjoint"
    involved = sorted({g for v in clashes.values() for g in v}, key=gids.index)
    return [{"group_ids": involved, "ids": sorted(clashes)[:20], "n_ids": len(clashes)}], "collision"
