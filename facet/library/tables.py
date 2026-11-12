"""Rows the library hands back: a list of dictionaries with a few helpers.

Every table of the library is a :class:`Table`. It *is* a list of rows, each
row a dictionary from column name to value, so everything that takes a list
of dictionaries takes it as it is (``pandas.DataFrame(table)``,
``csv.DictWriter``, a loop). The helpers are for the common next steps: one
column as an array, the rows that match, a CSV file, a data frame, aligned
text.
"""
from __future__ import annotations

import csv
import dataclasses
import enum
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


def _plain(value):
    """A value as plain Python, for a row: arrays as lists, enumerations by
    name, numpy scalars as numbers."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, enum.Enum):
        return value.name.lower()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_plain(v) for v in value)
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {_plain(k): _plain(v) for k, v in value.items()}
    return value


def _text(value, decimals: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if abs(value) < 1e-10:
            return "0"                      # rounding noise about zero
        if abs(value) < 10 ** -decimals or abs(value) >= 1e7:
            return f"{value:.3e}"
        return f"{value:.{decimals}f}".rstrip("0").rstrip(".") or "0"
    if isinstance(value, (list, tuple)):
        return " ".join(_text(v, decimals) for v in value)
    if isinstance(value, dict):
        return ", ".join(f"{k} {_text(v, decimals)}" for k, v in value.items())
    return str(value)


class Table(list):
    """Rows as dictionaries, in order, with the columns of the first row.

    ``title`` says what the rows are; ``notes`` holds the statements of how
    they were produced (thresholds, parameter set, what was left out).
    """

    def __init__(self, rows: Iterable[Mapping] = (), title: str = "",
                 notes: Sequence[str] = ()):
        super().__init__({k: _plain(v) for k, v in dict(r).items()} for r in rows)
        self.title = str(title)
        self.notes = [str(n) for n in notes]

    # -- what is in it --------------------------------------------------------
    @property
    def columns(self) -> list[str]:
        """Column names, in the order they first appear."""
        seen: dict[str, None] = {}
        for row in self:
            for key in row:
                seen.setdefault(key, None)
        return list(seen)

    def column(self, name: str) -> np.ndarray:
        """One column as an array (floats when every value is a number)."""
        if name not in self.columns:
            raise KeyError(f"no column {name!r}; the columns are "
                           f"{', '.join(self.columns)}")
        values = [row.get(name) for row in self]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool)
               or v is None for v in values):
            return np.array([np.nan if v is None else v for v in values], float)
        return np.array(values, dtype=object)

    def where(self, **equals) -> "Table":
        """The rows whose columns equal the values given
        (``table.where(element="O")``)."""
        return Table((r for r in self
                      if all(r.get(k) == v for k, v in equals.items())),
                     self.title, self.notes)

    def select(self, *names: str) -> "Table":
        """The same rows with only these columns, in this order."""
        missing = [n for n in names if n not in self.columns]
        if missing:
            raise KeyError(f"no column {', '.join(map(repr, missing))}; the "
                           f"columns are {', '.join(self.columns)}")
        return Table(({n: row.get(n) for n in names} for row in self),
                     self.title, self.notes)

    def sorted_by(self, name: str, reverse: bool = False) -> "Table":
        def key(row):
            v = row.get(name)
            missing = v is None or (isinstance(v, float) and math.isnan(v))
            return (missing, 0 if missing else v)
        return Table(sorted(self, key=key, reverse=reverse), self.title, self.notes)

    def head(self, n: int = 10) -> "Table":
        return Table(self[:n], self.title, self.notes)

    # -- out ------------------------------------------------------------------
    def to_frame(self):
        """A ``pandas.DataFrame`` of the rows (pandas has to be installed)."""
        try:
            import pandas
        except ImportError as error:            # pragma: no cover
            raise ImportError("to_frame() needs pandas: pip install pandas") \
                from error
        return pandas.DataFrame(list(self), columns=self.columns)

    def to_csv(self, path) -> Path:
        """Write the rows to a CSV file, the notes as leading comment lines."""
        path = Path(path)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            for line in ([self.title] if self.title else []) + self.notes:
                handle.write(f"# {line}\n")
            writer = csv.DictWriter(handle, fieldnames=self.columns)
            writer.writeheader()
            for row in self:
                writer.writerow({k: _csv(v) for k, v in row.items()})
        return path

    def text(self, decimals: int = 4, max_rows: int | None = None,
             notes: bool = True) -> str:
        """The rows as aligned text, the notes under them (``notes=False``
        leaves them out)."""
        if not self:
            return (self.title + "\n" if self.title else "") + "(no rows)"
        columns = self.columns
        shown = self if max_rows is None else self[:max_rows]
        cells = [[_text(row.get(c), decimals) for c in columns] for row in shown]
        widths = [max(len(c), *(len(line[i]) for line in cells))
                  for i, c in enumerate(columns)]
        lines = []
        if self.title:
            lines.append(self.title)
        lines.append("  ".join(c.ljust(w) for c, w in zip(columns, widths)).rstrip())
        lines.append("  ".join("-" * w for w in widths))
        lines += ["  ".join(v.ljust(w) for v, w in zip(line, widths)).rstrip()
                  for line in cells]
        if max_rows is not None and len(self) > max_rows:
            lines.append(f"... {len(self) - max_rows} more row(s)")
        if notes:
            lines += [f"note: {n}" for n in self.notes]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.text()

    def __repr__(self) -> str:
        what = f" {self.title!r}" if self.title else ""
        return f"<Table{what}: {len(self)} row(s), columns {self.columns}>"

    def _repr_html_(self) -> str:               # a notebook shows it as a table
        import html

        columns = self.columns
        head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(_text(row.get(c)))}</td>"
                             for c in columns) + "</tr>" for row in self[:200])
        more = (f"<p>... {len(self) - 200} more row(s)</p>"
                if len(self) > 200 else "")
        title = f"<p><b>{html.escape(self.title)}</b></p>" if self.title else ""
        notes = "".join(f"<p><i>note: {html.escape(n)}</i></p>" for n in self.notes)
        return (f"{title}<table><thead><tr>{head}</tr></thead>"
                f"<tbody>{body}</tbody></table>{more}{notes}")


def _csv(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return value


def row_of(obj, names: Mapping[str, str] | Sequence[str] | None = None) -> dict:
    """One object as a row.

    A mapping is copied; a dataclass gives its fields. ``names`` chooses the
    fields and, as a mapping ``field -> column name``, renames them (a
    column name states its unit). Properties are read as well as fields.
    """
    if isinstance(obj, Mapping):
        source = dict(obj)
        fields = list(source)
        read = source.get
    elif dataclasses.is_dataclass(obj):
        fields = [f.name for f in dataclasses.fields(obj)]

        def read(name):
            return getattr(obj, name)
    else:
        fields = [n for n in vars(obj) if not n.startswith("_")]

        def read(name):
            return getattr(obj, name)
    if names is None:
        chosen = {f: f for f in fields}
    elif isinstance(names, Mapping):
        chosen = dict(names)
    else:
        chosen = {n: n for n in names}
    return {column: _plain(read(field)) for field, column in chosen.items()}


def table_of(objects: Iterable, names=None, title: str = "",
             notes: Sequence[str] = ()) -> Table:
    """A :class:`Table` from dataclasses, mappings or objects with
    ``as_rows()``."""
    if hasattr(objects, "as_rows"):
        return Table(objects.as_rows(), title, notes)
    return Table((row_of(o, names) for o in objects), title, notes)
