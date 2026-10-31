"""Step 2: work out what the columns of a table are, and get the measurements as numbers.

A table can be laid out two ways:

    features_in_rows   one row per protein / metabolite; samples are columns     (e.g. SomaScan export)
    samples_in_rows    one row per sample; proteins / metabolites are columns    (e.g. many metabolomics exports)

propose_layout(table) makes a proposal, with a reason for every decision:

    orientation           which of the two layouts the table has
    id_column             the column that names each row (one value per row, none missing, none repeated)
    measurement_columns   the columns holding measured values
    annotation_columns    everything else (gene names, descriptions, quality columns, sample groups ...)

It is a proposal, not a decision: Layout.set(...) changes any part, and nothing downstream runs
on a layout you have not looked at. layout.matrix() then returns the measurements as numbers,
always features x samples, whatever the file's orientation.

How the measurement columns are found: every numeric column is proposed as a measurement.
An export often also carries numeric annotation columns (a CV, a detection limit, an affinity
constant); their type alone does not tell them apart from samples, so the reasons list them
and you move them with set(annotations=[...]). With samples in rows, whole-number columns with
only a few values (a 0/1 group code) are proposed as annotations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import pandas as pd

from .table import MISSING, Table

CODE_MAX_LEVELS = 5  # a whole-number column with this few distinct values reads as a code (e.g. 0/1 sex)


def _numbers(series: pd.Series) -> pd.Series:
    """Text cells to numbers; missing cells become NaN."""
    s = series.str.strip()
    return pd.to_numeric(s.where(~s.isin(MISSING)), errors="coerce")


def _pick_id_column(table: Table) -> tuple[str | None, str]:
    """The first text column with a value in every row and no repeated value. A numeric column
    qualifies only if it is the first column and holds whole numbers (a row number or numeric ID):
    any other numeric column with distinct values could just as well be a measurement."""
    n = table.shape[0]
    complete = [c for c in table.columns if c.n_missing == 0 and c.n_unique == n]
    text = [c for c in complete if c.kind == "text"]
    first = table.columns[0] if table.columns else None
    if text:
        c = text[0]
    elif first is not None and first in complete and first.kind == "numeric" and \
            (_numbers(table.data[first.name]) % 1 == 0).all():
        c = first
    else:
        return None, "No column names the rows (text, a value in every row, none repeated), so rows are numbered 1, 2, 3 ..."
    return c.name, f"'{c.name}' names the rows: it has a value in all {n} rows and none repeats."


@dataclass
class Layout:
    table: Table
    orientation: str
    id_column: str | None
    measurement_columns: list[str]
    annotation_columns: list[str]
    reasons: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- changing the proposal

    def set(self, orientation: str | None = None, id_column: str | None = None,
            measurements: list[str] | None = None, annotations: list[str] | None = None) -> "Layout":
        """A corrected layout. Give only what you want to change; the rest is kept.

        measurements / annotations move the named columns into that role.
        """
        all_names = list(self.table.data.columns)
        for n in (measurements or []) + (annotations or []) + ([id_column] if id_column else []):
            if n not in all_names:
                raise KeyError(f"No column named '{n}'.")
        if orientation and orientation not in ("features_in_rows", "samples_in_rows"):
            raise ValueError("orientation is 'features_in_rows' or 'samples_in_rows'.")
        meas = [c for c in self.measurement_columns if c not in (annotations or [])]
        meas += [c for c in (measurements or []) if c not in meas]
        idc = id_column if id_column is not None else self.id_column
        meas = [c for c in all_names if c in meas and c != idc]                   # keep file order
        ann = [c for c in all_names if c not in meas and c != idc]
        changed = [f"set by you: {k}" for k, v in (("orientation", orientation), ("id_column", id_column),
                                                     ("measurements", measurements), ("annotations", annotations)) if v]
        return replace(self, orientation=orientation or self.orientation, id_column=idc, measurement_columns=meas,
                       annotation_columns=ann, reasons=self.reasons + changed)

    # ---------------------------------------------------------------- results

    def _row_names(self) -> list[str]:
        if self.id_column:
            return self.table.data[self.id_column].str.strip().tolist()
        return [str(i + 1) for i in range(self.table.shape[0])]

    def matrix(self) -> pd.DataFrame:
        """The measurements as numbers, features x samples. Missing cells are NaN; nothing is filled."""
        values = pd.DataFrame({c: _numbers(self.table.data[c]) for c in self.measurement_columns})
        values.index = self._row_names()
        if self.orientation == "samples_in_rows":
            values = values.T                       # always features x samples
        values.index.name, values.columns.name = "feature", "sample"
        return values

    def annotations(self) -> pd.DataFrame:
        """The annotation columns, one row per feature (features_in_rows) or per sample (samples_in_rows)."""
        out = self.table.data[self.annotation_columns].copy()
        out.index = self._row_names()
        out.index.name = "feature" if self.orientation == "features_in_rows" else "sample"
        return out

    def describe(self) -> str:
        n_feat, n_samp = (len(self._row_names()), len(self.measurement_columns))
        if self.orientation == "samples_in_rows":
            n_feat, n_samp = n_samp, n_feat
        return (f"{self.orientation.replace('_', ' ')}: {n_feat} features x {n_samp} samples; "
                f"ID column {self.id_column!r}; {len(self.annotation_columns)} annotation column(s)")

    def roles(self) -> pd.DataFrame:
        """One row per column with its role."""
        role = {c: "measurement" for c in self.measurement_columns}
        role.update({c: "annotation" for c in self.annotation_columns})
        if self.id_column:
            role[self.id_column] = "id"
        return pd.DataFrame({"column": list(self.table.data.columns),
                             "role": [role.get(c, "?") for c in self.table.data.columns],
                             "kind": [c.kind for c in self.table.columns]})

    def __repr__(self) -> str:
        return f"<prism.Layout {self.describe()}>"

    def _repr_html_(self) -> str:
        items = "".join(f"<li>{r}</li>" for r in self.reasons)
        return f"<p><b>Proposed layout</b>: {self.describe()}</p><ul>{items}</ul>"


def propose_layout(table: Table) -> Layout:
    """Propose orientation, ID column, measurement and annotation columns, with reasons."""
    n_rows = table.shape[0]
    numeric = [c.name for c in table.columns if c.kind == "numeric"]
    reasons = []

    # 1. orientation: more numeric columns than rows means one row per sample
    if len(numeric) > 2 * n_rows:
        orientation = "samples_in_rows"
        reasons.append(f"Samples in rows: there are {len(numeric)} numeric columns but only {n_rows} rows; "
                       "a table with many more measured variables than rows usually has one row per sample.")
    else:
        orientation = "features_in_rows"
        reasons.append(f"Features in rows: {n_rows} rows against {len(numeric)} numeric columns; "
                       "one row per protein or metabolite is the usual export.")

    # 2. the ID column (a text column preferred: names, not running numbers)
    id_column, why = _pick_id_column(table)
    reasons.append(why)
    candidates = [n for n in numeric if n != id_column]

    # 3. measurement columns
    if orientation == "features_in_rows":
        measurements = candidates
        reasons.append(f"{len(measurements)} numeric column(s) are proposed as the samples. Numeric annotation "
                       "columns (a CV, a detection limit, a reference value ...) look the same by type: check the "
                       "list and move them with set(annotations=[...]).")
    else:
        codes = []
        for c in table.columns:
            if c.name in candidates and c.n_unique <= CODE_MAX_LEVELS:
                v = _numbers(table.data[c.name]).dropna()
                if len(v) and (v == v.round()).all():
                    codes.append(c.name)
        measurements = [n for n in candidates if n not in codes]
        reasons.append(f"{len(measurements)} numeric column(s) are the measured features.")
        if codes:
            reasons.append(f"{len(codes)} whole-number column(s) with at most {CODE_MAX_LEVELS} values look like codes "
                           f"(groups, sex ...), not measurements: {', '.join(codes[:8])}.")

    annotations = [c for c in table.data.columns if c not in measurements and c != id_column]
    text_ann = [c.name for c in table.columns if c.name in annotations and c.kind != "numeric"]
    if text_ann:
        reasons.append(f"{len(text_ann)} text or empty column(s) are annotations.")
    return Layout(table, orientation, id_column, measurements, annotations, reasons)
