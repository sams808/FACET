"""Writing an MD analysis run: one CSV file per descriptor, or one sheet each.

:func:`md_analysis.analyse` returns a :class:`~.md_analysis.ModelResult`:
for each analysis, a set of descriptors (md_stats containers or tables) and
the provenance of the analysis. A descriptor written without the frames,
the type map, the oxidation states, the bond-valence set and threshold, the
g(r) minima and the method parameters that produced it cannot be
reproduced, and FACET exists to make that point (``exporters.py``). So every
file written here carries that header:

* :func:`write_csv_files`: a directory with one CSV per descriptor, through
  ``exporters.write_csv``, each starting with the provenance of its analysis
  as ``#`` comment lines, then the descriptor's own notes; an ``index.csv``
  lists every file with its analysis and descriptor, and ``provenance.csv``
  holds the header of the run;
* :func:`write_workbook`: one XLSX workbook through ``exporters.write_xlsx``,
  its ``provenance`` sheet holding the run's header and every analysis's
  header in turn, an ``index`` sheet naming the analysis and the full
  descriptor name of each sheet (Excel cuts sheet names at 31 characters),
  a ``notes`` sheet with every descriptor's notes, and one sheet per
  descriptor. It needs openpyxl, which is not one of FACET's dependencies:
  without it the workbook is refused rather than written as the six-digit
  CSV files ``exporters.write_xlsx`` falls back to.

Column names carry their units: the md_stats containers name theirs
(``r_ang``, ``mean (1/Å^2)``, ``bin low (deg)``); the mean and spread of a
:class:`~.md_stats.Scalar` are renamed here to carry its unit, and those of
a :class:`~.md_stats.Distribution` its kind (fraction or count).

Numbers are written in full. ``exporters.write_csv`` writes a float with six
significant figures; here every float is handed to it as the shortest text
that reads back as the same float (``repr``), so a CSV read back gives the
numbers the analysis computed, bit for bit (``tests/test_md_export.py``
reads both formats back). The workbook stores floats as numbers, which
openpyxl writes with 16 significant digits (``'%.16g'``,
``openpyxl.compat.strings.safe_string``, read 2026-10-07), so a float read
back from the workbook can differ from the computed one in its 17th digit
(33.333333333333336 came back as 33.33333333333334); the CSV keeps every
digit. NaN is an empty cell in both, as ``exporters`` writes it; an infinite
value is written as the text ``inf`` / ``-inf`` (openpyxl would write it as
an empty cell).

TIMINGS
-------
Measured 2026-10-07 on this machine (Windows 11, i5-13420H, Python 3.11.9,
openpyxl 3.1.5), unpinned, on the glass and scattering results of two frames
of the 3 000-atom Na2O-3SiO2 model: 101 descriptors, 49 293 rows. The CSV
directory took 2.1 s. The workbook took 23.2 s, 13.1 s of it in 209 calls
of ``io.open``: openpyxl writes each sheet through a temporary file, and
opening one took about 63 ms here (Windows Defender scanning each new
file); the rest is openpyxl building and writing the cells. The results of
every analysis on 20 frames of each of the four 3 000-atom glasses of the
MD verification (208-373 descriptors, 66 691-132 987 rows) took 1.5-3.1 s
as CSV and 14-27 s as a workbook; writing them did not raise the process's
peak memory above what the analysis had reached.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from . import exporters
from .md_analysis import ANALYSES, ModelResult, Table
from .md_stats import Distribution, Histogram, Provenance, Scalar, Series, \
    _compact

__all__ = ["SHEET_NAME_MAX", "RESERVED_SHEETS", "descriptor_rows",
           "header_lines", "write_csv_files", "write_workbook", "export"]

# Excel's limit on a sheet name, and the sheets this module writes itself.
SHEET_NAME_MAX = 31
RESERVED_SHEETS = ("provenance", "index", "notes")
_SHEET_FORBIDDEN = re.compile(r"[\[\]:*?/\\]")
_FILE_UNSAFE = re.compile(r"[^A-Za-z0-9.+=()-]+")
_TAGS = {"glass": "glass", "scattering": "scat", "rings": "rings",
         "coordination-sequences": "cseq", "polyhedral-sharing": "share",
         "components": "comp", "warren-cowley": "wc", "bond-order": "bo",
         "tetrahedral-order": "qtet", "polyhedron-shape": "shape",
         "voronoi": "voro", "empty-spheres": "void", "free-volume": "fv",
         "nmr": "nmr", "exafs": "exafs", "bond-lifetimes": "life",
         "msd": "msd", "self-correlations": "self", "distinct-van-hove": "gd",
         "vacf": "vacf", "kinetic-temperature": "temp",
         "conductivity": "cond", "nmr-comparison": "nmrcmp",
         "scattering-comparison": "scatcmp", "feff": "feff"}


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

def descriptor_rows(container, *, per_frame: bool = False) -> list[dict]:
    """The rows of one descriptor, every column named with its unit.

    md_stats containers give their own rows (with one column per frame when
    ``per_frame``); a Scalar's mean and spread are renamed with its unit and
    a Distribution's with its kind, and each per-frame column ('frame 3',
    or 'block 0' for a block average) with the unit of its values
    ('frame 3 (Å)'); a :class:`~.md_analysis.Table` and any other result
    with ``as_rows()`` give theirs. The rows are made uniform: every row
    holds every column, in the order first seen.
    """
    if isinstance(container, Scalar):
        rows = container.as_rows(per_frame=per_frame)
        unit = container.unit or "1"
        rows = [_renamed(row, {"mean": f"mean ({unit})",
                               "std": f"std ({unit})"}) for row in rows]
    elif isinstance(container, Distribution):
        rows = container.as_rows(per_frame=per_frame)
        unit = container.kind
        rows = [_renamed(row, {"mean": f"mean ({unit})",
                               "std": f"std ({unit})"}) for row in rows]
    elif isinstance(container, (Histogram, Series)):
        rows = container.as_rows(per_frame=per_frame)
        unit = container.value_unit
    elif isinstance(container, Table):
        rows = container.as_rows()
    elif hasattr(container, "as_rows"):
        rows = list(container.as_rows())
    else:
        raise ValueError(f"no rows can be formed from "
                         f"{type(container).__name__}")
    if per_frame and isinstance(container, (Scalar, Distribution, Histogram,
                                            Series)):
        label = re.compile(rf"{re.escape(container.row_kind)} -?\d+")
        rows = [{(f"{key} ({unit})" if label.fullmatch(str(key)) else key):
                 value for key, value in row.items()} for row in rows]
    return _uniform(rows)


def _renamed(row: Mapping, names: Mapping[str, str]) -> dict:
    return {names.get(key, key): value for key, value in row.items()}


def _uniform(rows: Iterable[Mapping]) -> list[dict]:
    """Every row with every column (the ordered union of their keys), the
    missing ones None. ``exporters.write_csv`` takes its columns from the
    first row and leaves out any other key."""
    rows = [dict(r) for r in rows]
    columns: dict[str, None] = {}
    for row in rows:
        for key in row:
            columns.setdefault(str(key), None)
    return [{key: row.get(key) for key in columns} for row in rows]


def _notes_of(container) -> tuple[str, ...]:
    notes = getattr(container, "notes", ()) or ()
    return tuple(str(n) for n in notes)


def _kind(container) -> str:
    for cls in (Distribution, Histogram, Series, Scalar, Table):
        if isinstance(container, cls):
            return cls.__name__
    return type(container).__name__


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


# ---------------------------------------------------------------------------
# the header
# ---------------------------------------------------------------------------

def _minima_lines(result: ModelResult) -> list[str]:
    lines = []
    for (a, b), found in sorted(result.minima.items()):
        if a > b:
            continue
        if found.found:
            lines.append(f"g(r) first minimum {a}-{b}: {found.r_ang!r} Å "
                         f"({found.source}; {found.method})")
        else:
            lines.append(f"g(r) first minimum {a}-{b}: none ({found.reason})")
    return lines


def _search_lines(result: ModelResult) -> list[str]:
    counts = sorted({v for v in result.searches.values()})
    if not counts:
        return ["bulk pair searches: none (no per-frame analysis needed "
                "neighbour pairs)"]
    return ["bulk pair searches per frame (pass 1, pass 2), shared by every "
            "per-frame analysis: " + "; ".join(f"{a}, {b}" for a, b in counts)]


def header_lines(result: ModelResult, analysis: str | None = None,
                 descriptor: str | None = None, container=None
                 ) -> list[str]:
    """The provenance header of one descriptor (or of the run).

    The analysis's own provenance (the file, the frames it used and those
    it left out with the reason, the type map and its source, the oxidation
    states, the bond-valence set, v_bond and v_list, the cutoffs and their
    sources, the method parameters, the FACET version), then the run's pair
    searches and g(r) first minima, the model notes of the run (a
    composition that is not neutral, named formers absent from the model),
    then the analysis's and the descriptor's notes. A descriptor that
    covers fewer frames than its analysis (the distance-definition
    descriptors after a cancel in pass 2) says which.

    Each note appears once: one already in the provenance is not repeated
    as an analysis or descriptor note. On the 20-frame Na2O-3SiO2 export
    the three lists overlapped so much that a median header of 75 lines
    held 16 repeated notes, and one file 93 of its 164 (measured
    2026-10-07).
    """
    lines = []
    if analysis is not None:
        lines.append(f"analysis: {analysis}")
    if descriptor is not None:
        lines.append(f"descriptor: {descriptor}")
    output = result.outputs.get(analysis) if analysis else None
    provenance: Provenance = (output.provenance if output is not None
                              and output.provenance is not None
                              else result.provenance)
    seen: set[str] = set()

    def note(prefix: str, text) -> None:
        text = _one_line(text)
        if text not in seen:
            seen.add(text)
            lines.append(f"{prefix}: {text}")

    for row in provenance.as_rows():
        if row["field"] == "note":
            note("note", row["value"])
        else:
            lines.append(f"{row['field']}: {row['value']}")
    frames = _descriptor_frames(container)
    if frames is not None and set(frames) != set(provenance.frames_used):
        lines.append(f"descriptor frames used: {len(frames)}: "
                     f"{_compact(frames)} (of the analysis's frames above)")
    lines.extend(_search_lines(result))
    lines.extend(_minima_lines(result))
    for text in getattr(result, "model_notes", ()):
        note("model note", text)
    if output is not None:
        for text in output.notes:
            note("analysis note", text)
    if container is not None:
        for text in _notes_of(container):
            note("descriptor note", text)
    return [_one_line(line) for line in lines]


def _descriptor_frames(container) -> tuple[int, ...] | None:
    """The frames a descriptor covers, when it says (an md_stats container
    over frames, or a Table that names them); None otherwise."""
    if container is None:
        return None
    if isinstance(container, Table):
        return container.frames
    if isinstance(container, (Distribution, Histogram, Series, Scalar)) \
            and container.row_kind == "frame":
        return tuple(int(k) for k in container.frames)
    return None


# ---------------------------------------------------------------------------
# values
# ---------------------------------------------------------------------------

def _csv_value(value):
    """Full-precision text for exporters.write_csv (which would round a
    float to six significant figures)."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return ""
        if math.isinf(number):
            return "inf" if number > 0 else "-inf"
        return repr(number)
    if isinstance(value, (tuple, list)):
        return "; ".join(str(_csv_value(v)) for v in value)
    return value


def _xlsx_value(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isinf(number):
            return "inf" if number > 0 else "-inf"
        return number
    if isinstance(value, (tuple, list)):
        return "; ".join(str(_csv_value(v)) for v in value)
    return value


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------

def _file_name(analysis: str, descriptor: str, taken: set[str]) -> str:
    slug = _FILE_UNSAFE.sub("_", descriptor).strip("_") or "table"
    base = f"{analysis}__{slug}"[:120]
    name = f"{base}.csv"
    n = 2
    while name.casefold() in taken:          # Windows names ignore case
        name = f"{base}_{n}.csv"
        n += 1
    taken.add(name.casefold())
    return name


def _sheet_name(analysis: str, descriptor: str, taken: set[str]) -> str:
    tag = _TAGS.get(analysis, analysis[:6])
    text = _SHEET_FORBIDDEN.sub("-", f"{tag} {descriptor}").strip("'")
    name = text[:SHEET_NAME_MAX].rstrip()
    n = 2
    while name.casefold() in taken:          # Excel names ignore case
        suffix = f" ~{n}"
        name = text[:SHEET_NAME_MAX - len(suffix)].rstrip() + suffix
        n += 1
    taken.add(name.casefold())
    return name


def _entries(result: ModelResult, analyses: Sequence[str] | None):
    chosen = [a for a in ANALYSES if a in result.outputs
              and (analyses is None or a in analyses)]
    for analysis in chosen:
        for descriptor, container in result.outputs[analysis].tables.items():
            yield analysis, descriptor, container


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------

def write_csv_files(result: ModelResult, directory, *,
                    per_frame: bool = False,
                    analyses: Sequence[str] | None = None) -> list[Path]:
    """One CSV per descriptor in ``directory`` (created), each with its
    header, plus ``index.csv`` and ``provenance.csv``. Returns the paths,
    the index first. ``analyses`` limits the export to those analyses."""
    if not isinstance(result, ModelResult):
        raise ValueError("a ModelResult (md_analysis.analyse) is needed")
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    taken = {name.casefold() for name in ("index.csv", "provenance.csv")}
    index, written = [], []
    for analysis, descriptor, container in _entries(result, analyses):
        rows = descriptor_rows(container, per_frame=per_frame)
        name = _file_name(analysis, descriptor, taken)
        path = folder / name
        lines = header_lines(result, analysis, descriptor, container)
        if not rows:
            rows = [{"note": "no rows"}]
        exporters.write_csv([{k: _csv_value(v) for k, v in row.items()}
                             for row in rows], path, provenance=lines)
        written.append(path)
        index.append({"file": name, "analysis": analysis,
                      "descriptor": descriptor, "kind": _kind(container),
                      "rows": len(rows),
                      "notes": len(_notes_of(container))})
    for analysis, output in result.outputs.items():
        if analyses is not None and analysis not in analyses:
            continue
        if output.error:
            index.append({"file": None, "analysis": analysis,
                          "descriptor": None, "kind": None, "rows": 0,
                          "notes": 0, "not computed": output.error})
    index_rows = _uniform(index) if index else [{"file": None}]
    run_lines = header_lines(result)
    index_path = exporters.write_csv(
        [{k: _csv_value(v) for k, v in row.items()} for row in index_rows],
        folder / "index.csv", provenance=run_lines)
    provenance_path = exporters.write_csv(
        result.provenance.as_rows(), folder / "provenance.csv",
        provenance=[_one_line(line) for line in _search_lines(result)
                    + _minima_lines(result)])
    return [index_path, provenance_path] + written


def write_workbook(result: ModelResult, path, *, per_frame: bool = False,
                   analyses: Sequence[str] | None = None) -> Path:
    """One XLSX workbook: provenance, index, notes, then one sheet per
    descriptor (``exporters.write_xlsx``).

    Without openpyxl this raises ValueError before anything is written.
    ``exporters.write_xlsx`` would instead write one CSV per sheet beside
    ``path``, through ``exporters.write_csv``, which rounds every float to
    six significant figures: a run asked for ``result.xlsx`` would leave
    no such file and 74 rounded CSV files in its place (measured with
    openpyxl hidden, 2026-10-07). :func:`write_csv_files` writes the same
    results in full without openpyxl.
    """
    if not isinstance(result, ModelResult):
        raise ValueError("a ModelResult (md_analysis.analyse) is needed")
    try:
        import openpyxl  # noqa: F401  (exporters.write_xlsx writes with it)
    except ImportError:
        raise ValueError(
            f"{path}: an .xlsx workbook needs openpyxl, which this Python "
            "does not have (pip install openpyxl); a directory path writes "
            "the same results as CSV files, in full") from None
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    taken = {name.casefold() for name in RESERVED_SHEETS}
    sheets: dict[str, list[dict]] = {}
    index, notes = [], []
    for analysis, descriptor, container in _entries(result, analyses):
        rows = descriptor_rows(container, per_frame=per_frame)
        sheet = _sheet_name(analysis, descriptor, taken)
        sheets[sheet] = [{k: _xlsx_value(v) for k, v in row.items()}
                         for row in rows] or [{"note": "no rows"}]
        index.append({"sheet": sheet, "analysis": analysis,
                      "descriptor": descriptor, "kind": _kind(container),
                      "rows": len(rows),
                      "notes": len(_notes_of(container))})
        for note in _notes_of(container):
            notes.append({"sheet": sheet, "analysis": analysis,
                          "descriptor": descriptor,
                          "note": _one_line(note)})
    for analysis, output in result.outputs.items():
        if analyses is not None and analysis not in analyses:
            continue
        if output.error:
            index.append({"sheet": None, "analysis": analysis,
                          "descriptor": None, "kind": None, "rows": 0,
                          "notes": 0, "not computed": output.error})
        for note in output.notes:
            notes.append({"sheet": None, "analysis": analysis,
                          "descriptor": None, "note": _one_line(note)})
    lines = header_lines(result)
    for analysis, output in result.outputs.items():
        if analyses is not None and analysis not in analyses:
            continue
        if output.provenance is None:
            continue
        lines.append("")
        lines.append(f"== analysis: {analysis} ==")
        lines.extend(_one_line(line) for line in output.provenance.as_lines())
    # a note repeated for one sheet or analysis (each element's fit repeats
    # the notes of the tracks) is listed once
    notes = list({tuple(row.items()): row for row in notes}.values())
    ordered = {"index": _uniform(index) or [{"sheet": None}],
               "notes": notes or [{"note": None}]}
    ordered.update(sheets)
    return exporters.write_xlsx(ordered, target, provenance=lines)


def export(result: ModelResult, path, *, per_frame: bool = False,
           analyses: Sequence[str] | None = None) -> list[Path]:
    """``path`` ending in .xlsx: :func:`write_workbook`; otherwise a
    directory of CSV files (:func:`write_csv_files`)."""
    target = Path(path)
    if target.suffix.lower() == ".xlsx":
        return [write_workbook(result, target, per_frame=per_frame,
                               analyses=analyses)]
    if target.suffix.lower() == ".csv":
        raise ValueError(f"{target}: one CSV per descriptor is written, so "
                         "the output needs a directory (or an .xlsx file)")
    return write_csv_files(result, target, per_frame=per_frame,
                           analyses=analyses)
