"""Deterministic consistency checks on the proposed structure (v2.3). Flags, never fixes.

1. Near-identical value blocks (v2.3 §2): two or more value blocks of one assay
   whose computed profiles are near-identical AND whose names look like
   identifiers (short codes such as 'DEAB', 'GA43', 'T623', not measurement
   terms such as 'LFQ intensity') are flagged: they may be one measurement
   split per subject. The user confirms (one block, subject / name suffix parsed
   into sample information) or dismisses. This runs in addition to the AI's own
   propose_merge.
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


def _close(a, b, tol):
    return a is not None and b is not None and abs(a - b) <= tol


def similar(p, q):
    """Near-identical computed profiles (median, log10 span, zeros, missing, whole numbers)."""
    ma, mb = p.get("median"), q.get("median")
    if not ma or not mb or ma <= 0 or mb <= 0:
        return False
    if abs(math.log10(ma / mb)) > config.CONSISTENCY_MEDIAN_DECADES:
        return False
    sa, sb = p.get("log10_span"), q.get("log10_span")
    if sa is not None and sb is not None and abs(sa - sb) > config.CONSISTENCY_SPAN_TOL:
        return False
    if not _close(p.get("frac_zero") or 0, q.get("frac_zero") or 0, config.CONSISTENCY_FRAC_TOL):
        return False
    if not _close(p.get("frac_na") or 0, q.get("frac_na") or 0, config.CONSISTENCY_FRAC_TOL):
        return False
    return bool(p.get("integer_valued")) == bool(q.get("integer_valued"))


def _stat_line(g):
    p = g.get("profile") or {}
    return (f"{', '.join(g['columns'][:2])}{' ...' if g['n_columns'] > 2 else ''} ({g['n_columns']} col): median "
            f"{p.get('median')}, span {p.get('log10_span')} decades, zeros {round(100 * (p.get('frac_zero') or 0))}%")


def block_flags(groups_by_id, items, dismissed=()):
    """Clusters of near-identical identifier-named value blocks within one assay."""
    by_assay = {}
    for gid, it in items.items():
        g = groups_by_id.get(gid)
        if g and it["role"] == "value" and it.get("keep", True) and g["type"] == "numeric":
            by_assay.setdefault(it.get("assay_label"), []).append(gid)
    flags = []
    for assay, gids in by_assay.items():
        cand = [g for g in gids if identifier_like(name_code(groups_by_id[g]))]
        parent = {g: g for g in cand}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i, a in enumerate(cand):
            for b in cand[i + 1:]:
                if similar(groups_by_id[a]["profile"], groups_by_id[b]["profile"]):
                    parent[find(a)] = find(b)
        clusters = {}
        for g in cand:
            clusters.setdefault(find(g), []).append(g)
        for members in clusters.values():
            if len(members) < 2:
                continue
            members = sorted(members, key=lambda g: min(groups_by_id[g]["indices"]))
            key = "|".join(sorted(members))
            if key in dismissed:
                continue
            flags.append({
                "kind": "near_identical_blocks", "key": key, "assay": assay, "group_ids": members,
                "codes": [name_code(groups_by_id[g]) for g in members],
                "n_columns": sum(groups_by_id[g]["n_columns"] for g in members),
                "message": (f"These {len(members)} blocks look statistically identical — are they really different "
                            "measurements, or the same measurement across different subjects/samples?"),
                "evidence": [_stat_line(groups_by_id[g]) for g in members]})
    return flags


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
