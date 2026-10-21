"""Run R oracles for the v4 port tests (section 11). Tests skip when Rscript or a package is
missing; tests that source the proBatch repository need PROBATCH_SRC (a checkout of
https://github.com/symbioticMe/proBatch) and skip without it."""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

PROBATCH_FILES = ("auxiliary.R", "utility_funcs.R", "transform_raw.R", "normalize.R", "fit_non_linear.R",
                  "correct_batch_effects.R", "CV_calculation.R")


def have_r(*packages):
    if not shutil.which("Rscript"):
        return False
    for p in packages:
        r = subprocess.run(["Rscript", "-e", f"suppressMessages(library({p}))"], capture_output=True)
        if r.returncode != 0:
            return False
    return True


def probatch_src():
    p = os.environ.get("PROBATCH_SRC")
    return Path(p) if p and (Path(p) / "R" / "fit_non_linear.R").exists() else None


def need(*packages, probatch=False):
    if not have_r(*packages):
        pytest.skip(f"R or package(s) {', '.join(packages) or '(base)'} not available")
    if probatch and probatch_src() is None:
        pytest.skip("PROBATCH_SRC (the proBatch repository) is not set")


def write_matrix(path, X, rows=None, cols=None):
    rows = rows or [f"f{i}" for i in range(X.shape[0])]
    cols = cols or [f"s{j}" for j in range(X.shape[1])]
    with open(path, "w") as f:
        f.write("id," + ",".join(cols) + "\n")
        for r, row in zip(rows, X):
            f.write(r + "," + ",".join("NA" if not np.isfinite(v) else repr(float(v)) for v in row) + "\n")


def read_matrix(path):
    lines = Path(path).read_text().strip().splitlines()
    vals = [[np.nan if x in ("NA", "") else float(x) for x in l.split(",")[1:]] for l in lines[1:]]
    return np.array(vals)


def run(script, tmp, packages=(), probatch=False):
    head = "suppressMessages({" + "".join(f"library({p});" for p in packages) + "})\n"
    if probatch:
        src = probatch_src()
        head += "suppressMessages({library(dplyr); library(tidyr); library(purrr); library(rlang); library(magrittr); library(reshape2); library(tibble)})\n"
        head += "".join(f"source('{src / 'R' / f}')\n" for f in PROBATCH_FILES)
    p = Path(tmp) / "oracle.R"
    p.write_text(head + "setwd('" + str(tmp) + "')\n" + script)
    r = subprocess.run(["Rscript", str(p)], capture_output=True, text=True)
    return r
