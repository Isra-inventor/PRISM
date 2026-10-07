"""audit_params.yaml: defaults, plus overrides (kind 'param') and CLI/API values (v3 §6.4)."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULTS_PATH = Path(__file__).resolve().parent / "audit_params.yaml"
HEURISTIC = ("modified_z_cutoff", "rsd_levels", "median_scaling_mad", "sentinel_share", "near_duplicate_r",
             "abundance_rho_indicator", "icc_indicator")


class ParamError(Exception):
    pass


def _scalar(v):
    v = v.strip()
    if v == "":
        return None
    try:
        return json.loads(v)
    except ValueError:
        return v.strip("'\"")


def read_yaml(text):
    """The flat subset used here: 'key: value' lines, # comments, JSON-style lists."""
    out = {}
    for n, line in enumerate(text.splitlines(), start=1):
        line = line.split(" #", 1)[0].rstrip() if not line.lstrip().startswith("#") else ""
        if not line.strip():
            continue
        if ":" not in line or line.startswith((" ", "\t")):
            raise ParamError(f"audit_params line {n}: expected 'key: value'.")
        k, v = line.split(":", 1)
        out[k.strip()] = _scalar(v)
    return out


def defaults():
    return read_yaml(DEFAULTS_PATH.read_text(encoding="utf-8"))


def resolve(cli=None, overrides=None):
    """-> (params, changed): defaults < overrides 'param' entries < CLI/API values."""
    base = defaults()
    out = dict(base)
    sources = {}
    for o in overrides or []:
        if o.get("kind") == "param":
            _set(out, o.get("key"), o.get("value"), base)
            sources[o["key"]] = "override"
    for k, v in (cli or {}).items():
        _set(out, k, v, base)
        sources[k] = "run"
    changed = {k: {"default": base[k], "value": out[k], "source": sources[k]} for k in sorted(sources) if out[k] != base[k]}
    return out, changed


def _set(out, k, v, base):
    if k not in base:
        raise ParamError(f"Unknown audit parameter '{k}'. Known: {', '.join(sorted(base))}.")
    if isinstance(base[k], bool) or not isinstance(base[k], (int, float, list)):
        out[k] = v
        return
    if isinstance(base[k], list):
        if not isinstance(v, list) or not all(isinstance(x, (int, float)) for x in v):
            raise ParamError(f"'{k}' must be a list of numbers.")
        out[k] = v
        return
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        raise ParamError(f"'{k}' must be a number.")
    out[k] = int(v) if isinstance(base[k], int) and float(v).is_integer() else v
