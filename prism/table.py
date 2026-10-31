"""Step 1: read a data table exactly as it is written, and describe it.

Nothing is guessed or changed here. Every cell is read as text, so an ID like "007" stays "007",
and a gene name like "1E5" or "MARCH1" is never turned into a number or a date. Deciding what
the columns *mean* (feature IDs, samples, annotations) is a later step.

For each column we only record facts:
    - kind:      "numeric" if every non-missing cell reads as a number, "text" otherwise,
                 "empty" if every cell is missing
    - n_missing: how many cells are missing (see MISSING below)
    - n_unique:  how many distinct values
    - example:   the first non-missing value
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Cells we treat as "missing". Listed explicitly so that the rule is visible and can be discussed.
MISSING = {"", "NA", "N/A", "NaN", "nan", "null", "NULL", "None"}

# Delimiters we look for, in this order when two are equally likely.
DELIMITERS = [",", "\t", ";"]


@dataclass
class Column:
    name: str
    kind: str          # "numeric" | "text" | "empty"
    n_missing: int
    n_unique: int
    example: str


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def detect_delimiter(path: Path) -> str:
    """Look at the first lines and pick the delimiter that splits them most consistently."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        sample = "".join(f.readline() for _ in range(20))
    try:
        return csv.Sniffer().sniff(sample, delimiters="".join(DELIMITERS)).delimiter
    except csv.Error:
        # Sniffer gave up: fall back to the delimiter that appears most in the first line.
        first = sample.splitlines()[0] if sample else ""
        return max(DELIMITERS, key=first.count)


def describe_column(name: str, values: pd.Series) -> Column:
    cells = values.str.strip()
    missing = cells.isin(MISSING)
    present = cells[~missing]
    if present.empty:
        kind = "empty"
    elif present.map(_is_number).all():
        kind = "numeric"
    else:
        kind = "text"
    return Column(name=name, kind=kind, n_missing=int(missing.sum()), n_unique=int(present.nunique()),
                  example=present.iloc[0] if not present.empty else "")


class Table:
    """A table read as text, plus a description of every column."""

    def __init__(self, path: Path, delimiter: str, data: pd.DataFrame):
        self.path = path
        self.delimiter = delimiter
        self.data = data                          # every cell as text, exactly as in the file
        self.columns = [describe_column(c, data[c]) for c in data.columns]

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    def summary(self) -> pd.DataFrame:
        """One row per column: name, kind, missing cells, distinct values, an example."""
        return pd.DataFrame([vars(c) for c in self.columns])

    def counts(self) -> dict:
        """How many columns of each kind."""
        out = {"numeric": 0, "text": 0, "empty": 0}
        for c in self.columns:
            out[c.kind] += 1
        return out

    def describe(self) -> str:
        """One sentence: size, delimiter and column kinds."""
        n_rows, n_cols = self.shape
        k = self.counts()
        names = {",": "comma", "\t": "tab", ";": "semicolon"}
        return (f"{n_rows} rows x {n_cols} columns, {names.get(self.delimiter, repr(self.delimiter))}-separated; "
                f"{k['numeric']} numeric, {k['text']} text, {k['empty']} empty columns")

    def __repr__(self) -> str:
        return f"<prism.Table {self.path.name}: {self.describe()}>"

    def _repr_html_(self) -> str:
        # What a notebook shows when the table is the last line of a cell.
        return f"<p><b>{self.path.name}</b>: {self.describe()}</p>" + self.summary().to_html(index=False)


def read(path, delimiter: str | None = None) -> Table:
    """Read a CSV / TSV file as text and describe its columns.

    delimiter: leave it out to detect it from the file, or give ",", "\\t" or ";".
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No file at {path}")
    delim = delimiter or detect_delimiter(path)
    data = pd.read_csv(path, sep=delim, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return Table(path, delim, data)
