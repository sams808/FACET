"""Reading published bond-valence parameter sets.

FACET ships two sets. The built-in table is Brese and O'Keeffe's 1991 Table 2
(the values the tutorials and the verification corpus are stated against)
with the O'Keeffe-Brese electronegativity estimator behind it. Beside it sits
I. D. Brown's accumulated table, the IUCr ``bvparm2020.cif`` distribution
(``facet/data/bvparm2020.cif``, :func:`bundled_compilation`): 1 931 rows
drawn from the literature -- Brown & Altermatt 1985, Brese & O'Keeffe 1991,
Gagné & Hawthorne 2015, Adams, Palenik and the rest -- each row naming its
source. The file is distributed under its author's notice (free for
non-profit use, no fee, the notice kept, changes recorded) and is carried
unchanged. The softBV set has terms of its own and is not shipped; it is
read from a file like any other.

Any other set is read from a file: you obtain it from its publisher, point
FACET at it, and every parameter used afterwards carries that file's name as
its provenance. Three formats are understood:

* the IUCr **bvparm.cif** distribution, a CIF with a ``_valence_param_`` loop;
* **softBV**-style and other plain tables, whitespace- or comma-separated;
* FACET's own JSON, for a set you have edited or assembled yourself.

Every reader records where each value came from. A bond-valence sum is only as
trustworthy as its R0, and a number whose source has been forgotten is not a
measurement.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from . import bv, elements

BUNDLED_NAME = "IUCr bvparm 2020 (Brown)"
BUNDLED_KEY = "bvparm2020"


def bundled_path() -> Path:
    """Where the bundled IUCr table is, frozen or from source (the same
    layout rule as :func:`facet.core.examples.folder`)."""
    import sys
    meipass = getattr(sys, "_MEIPASS", None)
    base = Path(meipass) / "facet" / "data" if meipass \
        else Path(__file__).resolve().parent.parent / "data"
    return base / "bvparm2020.cif"


_bundled_cache: dict = {}


def bundled_compilation(allow_estimated: bool = True) -> bv.ParameterSet:
    """The IUCr accumulated table as a parameter set, read once. Where a pair
    has several rows the first is taken, which is the file's own order of
    reliability; pairs the table lacks fall to the estimator when
    ``allow_estimated`` is True, and are labelled as estimated."""
    key = bool(allow_estimated)
    if key not in _bundled_cache:
        params, _report = load(bundled_path(), name=BUNDLED_NAME,
                               allow_estimated=allow_estimated)
        _bundled_cache[key] = params
    return _bundled_cache[key]


def named_set(name: str | None) -> bv.ParameterSet:
    """A parameter set by its short name: 'built-in' (or None) for Brese &
    O'Keeffe 1991, 'bvparm2020' for the bundled IUCr table, else a path to
    read."""
    if not name or name == "built-in":
        return bv.DEFAULT
    if name == BUNDLED_KEY:
        return bundled_compilation()
    params, _report = load(name)
    return params


FILE_FILTER = (
    "Bond-valence parameters (*.cif *.txt *.dat *.csv *.tsv *.json);;"
    "IUCr bvparm (*.cif);;"
    "Plain table (*.txt *.dat *.csv *.tsv);;"
    "FACET parameter set (*.json);;"
    "All files (*)"
)

# The IUCr file marks the values it recommends. Where a pair appears more than
# once -- different authors, different data -- the recommended row is taken and
# the others are counted and reported rather than silently discarded.
_RECOMMENDED = {"1", "y", "yes", "true"}


@dataclass
class LoadReport:
    """What a parameter file turned out to contain."""

    name: str
    path: str
    format: str
    pairs: int = 0
    skipped: int = 0
    duplicates: int = 0
    with_own_b: int = 0
    notes: tuple = ()

    def describe(self) -> str:
        parts = [f"{self.pairs} pairs from {self.format}"]
        if self.with_own_b:
            parts.append(f"{self.with_own_b} with their own b")
        if self.duplicates:
            parts.append(f"{self.duplicates} pairs had alternatives; "
                         "the recommended row was used")
        if self.skipped:
            parts.append(f"{self.skipped} rows could not be read")
        return "; ".join(parts)


def _parse_charge(text) -> int | None:
    """A charge written any of the ways a table writes one: 3, +3, 3+, -2."""
    if text is None:
        return None
    raw = str(text).strip().strip("'\"")
    if not raw or raw in {".", "?"}:
        return None
    match = re.fullmatch(r"([+-]?)(\d+)([+-]?)", raw)
    if not match:
        return None
    lead, digits, trail = match.groups()
    sign = -1 if "-" in (lead, trail) else 1
    return sign * int(digits)


def _clean_element(text) -> str | None:
    if text is None:
        return None
    raw = str(text).strip().strip("'\"")
    match = re.match(r"^([A-Za-z]{1,2})", raw)
    if not match:
        return None
    symbol = elements.normalise(match.group(1))
    return symbol or None


def _number(text):
    if text is None:
        return None
    raw = str(text).strip().strip("'\"")
    if not raw or raw in {".", "?"}:
        return None
    # a value written with an esd in brackets: 1.973(4)
    raw = re.sub(r"\(\d+\)$", "", raw)
    try:
        return float(raw)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# the IUCr bvparm.cif distribution
# ---------------------------------------------------------------------------

def read_bvparm_cif(path) -> tuple[dict, dict, dict, LoadReport]:
    """The IUCr bond-valence parameter CIF.

    The loop is ``_valence_param_atom_1`` / ``_atom_1_valence`` /
    ``_atom_2`` / ``_atom_2_valence`` / ``_Ro`` / ``_B``, with a
    ``_valence_param_ref_id`` naming the source of each row. A pair may appear
    several times from different authors; where the file marks one row as
    recommended that row wins, and otherwise the first is taken and the rest
    counted.
    """
    import gemmi

    path = Path(path)
    document = gemmi.cif.read(str(path))
    fitted: dict[tuple[str, int, str], float] = {}
    fitted_b: dict[tuple[str, int, str], float] = {}
    anion_ox: dict[str, int] = {}
    skipped = duplicates = 0
    recommended: set[tuple[str, int, str]] = set()

    tags = ["_valence_param_atom_1", "_valence_param_atom_1_valence",
            "_valence_param_atom_2", "_valence_param_atom_2_valence",
            "_valence_param_Ro", "_valence_param_B"]

    found_loop = False
    references: dict[str, str] = {}
    pair_sources: dict[tuple[str, int, str], str] = {}
    for block in document:
        table = block.find(tags)
        if not len(table):
            continue
        found_loop = True
        # the file's own list of sources, one letter code per paper
        try:
            ref_loop = block.find_loop("_valence_ref_id").get_loop()
            if ref_loop is not None:
                text_tag = next((t for t in ref_loop.tags
                                 if "reference" in t.lower()), None)
                if text_tag is not None:
                    ids = block.find_values("_valence_ref_id")
                    texts = block.find_values(text_tag)
                    for i in range(min(len(ids), len(texts))):
                        references[str(ids[i]).strip()] = \
                            gemmi.cif.as_string(texts[i]).strip()
        except Exception:                            # noqa: BLE001
            references = {}
        # the optional columns, looked up by row index so a missing one is fine
        details = block.find_values("_valence_param_details")
        ref = block.find_values("_valence_param_ref_id")
        for index, row in enumerate(table):
            cation = _clean_element(row[0])
            cation_ox = _parse_charge(row[1])
            anion = _clean_element(row[2])
            anion_charge = _parse_charge(row[3])
            r0 = _number(row[4])
            b = _number(row[5])
            if not (cation and anion and cation_ox and r0 and r0 > 0):
                skipped += 1
                continue
            key = (cation, int(cation_ox), anion)

            note = ""
            try:
                if details is not None and index < len(details):
                    note = str(details[index]).strip().strip("'\"").lower()
            except Exception:
                note = ""
            is_recommended = any(word in note for word in
                                 ("recommend", "preferred", "best"))

            if key in fitted:
                duplicates += 1
                if not is_recommended or key in recommended:
                    continue
            if is_recommended:
                recommended.add(key)

            fitted[key] = r0
            try:
                code = str(ref[index]).strip() if ref is not None \
                    and index < len(ref) else ""
            except Exception:                        # noqa: BLE001
                code = ""
            if code and code not in ("?", "."):
                pair_sources[key] = references.get(code, f"ref. {code}")
            if b is not None and b <= 0:
                raise ValueError(
                    f"{path.name}: {cation}{cation_ox:+d}-{anion} is given "
                    f"b = {b:g}. The valence is exp((R0 - d)/b), so a b of zero "
                    "or less has no meaning; substituting the default silently "
                    "would change every valence in the file.")
            if b:
                fitted_b[key] = b
            if anion_charge is not None:
                anion_ox[anion] = int(anion_charge)

    if not found_loop:
        raise ValueError(
            f"{path.name}: no _valence_param_ loop found. FACET expects the "
            "IUCr bvparm distribution, or a plain table of "
            "cation, charge, anion, charge, R0, b.")
    if not fitted:
        raise ValueError(f"{path.name}: the loop held no usable rows")

    report = LoadReport(
        name=path.stem, path=str(path), format="IUCr bvparm CIF",
        pairs=len(fitted), skipped=skipped, duplicates=duplicates,
        with_own_b=len(fitted_b))
    report.pair_sources = pair_sources
    return fitted, fitted_b, anion_ox, report


# ---------------------------------------------------------------------------
# plain tables
# ---------------------------------------------------------------------------

def read_table(path) -> tuple[dict, dict, dict, LoadReport]:
    """A whitespace- or comma-separated table of parameters.

    Expected columns, in this order: cation, cation charge, anion, anion charge,
    R0, and optionally b. Comment lines start with ``#``, ``!``, ``;`` or ``*``;
    a header line is recognised and skipped. This is the shape softBV and most
    supplementary tables come in.

    The charge columns may be written ``3``, ``+3`` or ``3+``, and the element
    columns may carry the charge themselves (``Bi3+``), in which case a separate
    charge column is not needed -- checked by trying both layouts per row rather
    than guessing once for the file, because real tables are not consistent.
    """
    path = Path(path)
    fitted: dict[tuple[str, int, str], float] = {}
    fitted_b: dict[tuple[str, int, str], float] = {}
    anion_ox: dict[str, int] = {}
    skipped = duplicates = 0

    for line in path.read_text(encoding="utf-8",
                               errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] in "#!;*/":
            continue
        fields = [f for f in re.split(r"[,\t;]|\s+", stripped) if f]
        if len(fields) < 3:
            continue

        parsed = _parse_table_row(fields)
        if parsed is None:
            # a header line, or a row that cannot be read
            if any(character.isdigit() for character in stripped):
                skipped += 1
            continue
        cation, cation_ox, anion, anion_charge, r0, b = parsed
        key = (cation, cation_ox, anion)
        if key in fitted:
            duplicates += 1
            continue
        if b is not None and b <= 0:
            raise ValueError(
                f"{path.name}: {cation}{cation_ox:+d}-{anion} is given "
                f"b = {b:g}. The valence is exp((R0 - d)/b), so a b of zero or "
                "less has no meaning; substituting the default silently would "
                "change every valence in the file.")
        fitted[key] = r0
        if b:
            fitted_b[key] = b
        if anion_charge is not None:
            anion_ox[anion] = anion_charge

    if not fitted:
        raise ValueError(
            f"{path.name}: no parameter rows found. FACET expects columns of "
            "cation, charge, anion, charge, R0 and optionally b.")

    report = LoadReport(
        name=path.stem, path=str(path), format="plain table",
        pairs=len(fitted), skipped=skipped, duplicates=duplicates,
        with_own_b=len(fitted_b))
    return fitted, fitted_b, anion_ox, report


def _parse_table_row(fields):
    """One row, trying the layouts a real table uses. None if unreadable."""
    # cation, cation_ox, anion, anion_ox, R0 [, b]
    if len(fields) >= 5:
        cation = _clean_element(fields[0])
        cation_ox = _parse_charge(fields[1])
        anion = _clean_element(fields[2])
        anion_charge = _parse_charge(fields[3])
        r0 = _number(fields[4])
        b = _number(fields[5]) if len(fields) >= 6 else None
        if cation and anion and cation_ox and r0 and 0.3 < r0 < 4.0:
            return cation, int(cation_ox), anion, anion_charge, r0, b

    # cation_with_charge, anion_with_charge, R0 [, b]
    if len(fields) >= 3:
        cation = _clean_element(fields[0])
        cation_ox = _parse_charge(re.sub(r"^[A-Za-z]{1,2}", "", fields[0]))
        anion = _clean_element(fields[1])
        anion_charge = _parse_charge(re.sub(r"^[A-Za-z]{1,2}", "", fields[1]))
        r0 = _number(fields[2])
        b = _number(fields[3]) if len(fields) >= 4 else None
        if cation and anion and cation_ox and r0 and 0.3 < r0 < 4.0:
            return cation, int(cation_ox), anion, anion_charge, r0, b
    return None


# ---------------------------------------------------------------------------
# FACET's own JSON
# ---------------------------------------------------------------------------

def read_json(path) -> tuple[dict, dict, dict, LoadReport]:
    """A set written by :func:`write_json`, or edited by hand."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        # a JSONDecodeError is a ValueError, but its message says nothing about
        # which file, and a caller catching (ValueError, OSError) deserves a
        # sentence it can show someone
        raise ValueError(
            f"{path.name}: this is not valid JSON ({error.msg} at line "
            f"{error.lineno})") from error
    fitted: dict[tuple[str, int, str], float] = {}
    fitted_b: dict[tuple[str, int, str], float] = {}
    anion_ox = {str(k): int(v) for k, v in (data.get("anion_ox") or {}).items()}
    skipped = 0

    for row in data.get("parameters", []):
        cation = _clean_element(row.get("cation"))
        anion = _clean_element(row.get("anion"))
        cation_ox = _parse_charge(row.get("cation_ox"))
        r0 = _number(row.get("r0"))
        if not (cation and anion and cation_ox and r0):
            skipped += 1
            continue
        key = (cation, int(cation_ox), anion)
        fitted[key] = r0
        b = _number(row.get("b"))
        if b:
            fitted_b[key] = b
    if not fitted:
        raise ValueError(f"{path.name}: no usable parameters in the file")

    report = LoadReport(
        name=str(data.get("name") or path.stem), path=str(path),
        format="FACET JSON", pairs=len(fitted), skipped=skipped,
        with_own_b=len(fitted_b),
        notes=tuple(data.get("notes") or ()))
    return fitted, fitted_b, anion_ox, report


def write_json(path, params: bv.ParameterSet) -> Path:
    """Write a parameter set out, so an edited one can be shared as data.

    Data, not a redistribution of somebody's compilation: what comes out is what
    the user put in, plus whatever they changed, with its provenance attached.
    """
    path = Path(path)
    rows = []
    for key in params.pairs():
        cation, ox, anion = key
        param = params.get(cation, ox, anion)
        if param is None:
            continue
        rows.append({"cation": cation, "cation_ox": ox, "anion": anion,
                     "r0": param.r0, "b": param.b,
                     "source": param.source, "fitted": param.fitted})
    path.write_text(json.dumps({
        "name": params.name,
        "source": params.source,
        "b": params.b,
        "notes": list(params.notes),
        "anion_ox": params._anion_ox,
        "parameters": rows,
    }, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

def load(path, name: str | None = None,
         allow_estimated: bool = True) -> tuple[bv.ParameterSet, LoadReport]:
    """Read any supported parameter file into a :class:`~facet.core.bv.ParameterSet`.

    The format is chosen by looking at the file, not only at its extension: a
    bvparm distribution is sometimes saved as .txt, and a plain table is
    sometimes saved as .cif.
    """
    path = Path(path)
    text = ""
    try:
        # the IUCr header runs to some 170 lines before its loop
        text = path.read_text(encoding="utf-8", errors="replace")[:60000]
    except OSError as error:
        raise ValueError(f"{path.name}: {error}") from error

    if path.suffix.lower() == ".json" or text.lstrip().startswith("{"):
        fitted, fitted_b, anion_ox, report = read_json(path)
    elif "_valence_param" in text:
        fitted, fitted_b, anion_ox, report = read_bvparm_cif(path)
    elif path.suffix.lower() in {".cif", ".mcif"}:
        # a CIF without the loop is not a parameter file; say which it is
        raise ValueError(
            f"{path.name}: this is a CIF but has no _valence_param_ loop, so "
            "it is a structure rather than a parameter set.")
    else:
        fitted, fitted_b, anion_ox, report = read_table(path)

    notes = [f"parameters loaded from {path.name}", report.describe()]
    notes.extend(report.notes)
    params = bv.ParameterSet(
        name=name or report.name,
        fitted=fitted,
        source=f"{report.name} ({report.format}, {path.name})",
        b=bv.B_DEFAULT,
        allow_estimated=allow_estimated,
        fitted_b=fitted_b,
        anion_ox=anion_ox,
        notes=notes,
        pair_sources=getattr(report, "pair_sources", None))
    return params, report


def compare(first: bv.ParameterSet, second: bv.ParameterSet,
            pairs=None) -> list[tuple]:
    """Where two parameter sets disagree, and by how much.

    Returns ``(cation, ox, anion, r0_first, r0_second, delta_r0,
    valence_ratio)`` for each pair both sets cover. The valence ratio is
    ``exp(delta/b)``: the factor by which every valence for that pair changes
    between the two sets, which is the number that matters -- a 0.02 A difference
    in R0 is a 5 percent difference in every bond valence it produces.
    """
    import math

    keys = list(pairs) if pairs is not None else sorted(
        set(first.pairs()) & set(second.pairs()))
    out = []
    for cation, ox, anion in keys:
        a = first.get(cation, ox, anion)
        b = second.get(cation, ox, anion)
        if a is None or b is None:
            continue
        delta = b.r0 - a.r0
        ratio = math.exp(delta / a.b) if a.b else float("nan")
        out.append((cation, ox, anion, a.r0, b.r0, delta, ratio))
    out.sort(key=lambda row: -abs(row[5]))
    return out
