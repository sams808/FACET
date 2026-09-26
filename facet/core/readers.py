"""Reading structures that are not CIFs.

FACET writes POSCAR, XYZ and `.vesta` already; these are the matching readers,
plus SHELX and the formats gemmi covers. Each returns the same
:class:`~facet.core.structure.Structure` the CIF reader does, so nothing
downstream needs to know where a structure came from.

Every reader here produces a **P1 structure with explicit coordinates**. None of
these formats carries symmetry in a form worth re-expanding: POSCAR and XYZ have
none at all, a `.vesta` file records the operators it already applied, and a
SHELX file gives generators in its own notation. Reading them as P1 and letting
spglib say what the symmetry actually is avoids inventing an expansion the file
did not intend.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from . import elements
from .structure import Atom, Cell, Site, Structure


class UnsupportedFormat(ValueError):
    pass


# ---------------------------------------------------------------------------
# shared
# ---------------------------------------------------------------------------

def cell_from_vectors(vectors: np.ndarray) -> Cell:
    """A Cell from three lattice vectors given as rows."""
    a, b, c = (np.asarray(v, float) for v in vectors)
    la, lb, lc = (float(np.linalg.norm(v)) for v in (a, b, c))

    def angle(u, v, nu, nv):
        if nu < 1e-12 or nv < 1e-12:
            return 90.0
        return math.degrees(math.acos(
            float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))))

    return Cell(la, lb, lc,
                angle(b, c, lb, lc), angle(c, a, lc, la), angle(a, b, la, lb),
                np.column_stack([a, b, c]))


def cell_from_parameters(a, b, c, alpha, beta, gamma) -> Cell:
    """The standard orthogonalisation, a along x and b in the xy plane."""
    al, be, ga = (math.radians(float(x)) for x in (alpha, beta, gamma))
    ca, cb, cg = math.cos(al), math.cos(be), math.cos(ga)
    sg = math.sin(ga)
    if abs(sg) < 1e-12:
        raise ValueError("degenerate cell: gamma is 0 or 180 degrees")
    inner = 1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg
    if inner <= 0:
        raise ValueError("cell angles are geometrically impossible")
    volume = a * b * c * math.sqrt(inner)
    orth = np.array([
        [a, b * cg, c * cb],
        [0.0, b * sg, c * (ca - cb * cg) / sg],
        [0.0, 0.0, volume / (a * b * sg)],
    ])
    return Cell(float(a), float(b), float(c),
                float(alpha), float(beta), float(gamma), orth)


def _assemble(name: str, cell: Cell, rows, source: str | None = None,
              formula: str | None = None, spacegroup: str | None = None,
              note_p1: bool = True) -> Structure:
    """Build a structure from rows of (label, element, fractional, occupancy).

    A fifth element, a site index, may be present: it says which
    crystallographic site an expanded atom came from, so that results are
    reported per site rather than per atom. Without it every row becomes its
    own site, which is right for a format that carries no symmetry.
    """
    expanded = len(rows) > 0 and len(rows[0]) >= 5

    sites, atoms = [], []
    if expanded:
        first: dict[int, int] = {}
        for label, element, frac, occ, origin in rows:
            element = elements.normalise(element)
            frac = np.asarray(frac, float) % 1.0
            if origin not in first:
                first[origin] = len(sites)
                sites.append(Site(label, element, frac, float(occ)))
            atoms.append(Atom(element=element, frac=frac,
                              cart=cell.to_cartesian(frac),
                              site_index=first[origin], label=label,
                              occupancy=float(occ)))
        counts: dict[int, int] = {}
        for atom in atoms:
            counts[atom.site_index] = counts.get(atom.site_index, 0) + 1
        for index, site in enumerate(sites):
            site.multiplicity = counts.get(index, 1)
    else:
        for index, (label, element, frac, occ) in enumerate(rows):
            element = elements.normalise(element)
            frac = np.asarray(frac, float)
            sites.append(Site(label, element, frac, float(occ), multiplicity=1))
            atoms.append(Atom(element=element, frac=frac % 1.0,
                              cart=cell.to_cartesian(frac % 1.0),
                              site_index=index, label=label,
                              occupancy=float(occ)))

    # An expanded structure has no declared symmetry of its own -- claiming P1
    # would make the spglib cross-check report a disagreement that is not one.
    if spacegroup:
        declared, number = spacegroup, None
    elif expanded:
        declared, number = None, None
    else:
        declared, number = "P 1", 1
    structure = Structure(name=name, cell=cell, sites=sites, atoms=atoms,
                          spacegroup_hm=declared, spacegroup_number=number,
                          source_path=source, formula=formula)
    for site in structure.sites:
        site.ox = elements.COMMON_OX.get(site.element)
        site.ox_source = "common"
    if note_p1:
        structure.notes.append(
            "read with explicit coordinates; the symmetry reported is what "
            "spglib finds in them, not what the file declared")
    _annotate(structure)
    return structure


def _annotate(structure: Structure) -> None:
    """Let spglib say what the symmetry is, as for a CIF.

    A failure here is not fatal -- the structure is complete without it -- but it
    is not invisible either. Swallowing it silently would mean a missing spglib
    removed the symmetry cross-check from every non-CIF file with nothing to show
    that it had, which is the kind of quiet downgrade this program is supposed to
    make impossible.
    """
    try:
        from .cif import _annotate_symmetry

        _annotate_symmetry(structure)
    except Exception as error:
        structure.notes.append(
            f"the symmetry could not be determined ({type(error).__name__}: "
            f"{error}), so no space group is reported for this file and the "
            "Wyckoff letters are absent")


# ---------------------------------------------------------------------------
# symmetry operators, in the "-x, 1/2+y, 1/2-z" notation SHELX and CIF share
# ---------------------------------------------------------------------------

_TERM = re.compile(r"([+-]?)\s*(\d*\.?\d*)\s*(?:/\s*(\d+))?\s*\*?\s*([xyz]?)",
                   re.IGNORECASE)


def parse_symop(text: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Parse one symmetry operation into a rotation matrix and a translation.

    Accepts the forms both SHELX and CIF use: ``-X, 0.5+Y, 0.5-Z``,
    ``-x, y+1/2, -z+1/2``, ``1/2-x, -y, 1/2+z``.
    """
    parts = [p.strip() for p in text.replace(";", ",").split(",")]
    if len(parts) != 3:
        return None
    rotation = np.zeros((3, 3))
    translation = np.zeros(3)
    axes = {"x": 0, "y": 1, "z": 2}

    for row, expression in enumerate(parts):
        if not expression:
            return None
        cursor = 0
        text_lower = expression.lower().replace(" ", "")
        while cursor < len(text_lower):
            match = _TERM.match(text_lower, cursor)
            if not match or match.end() == cursor:
                cursor += 1
                continue
            cursor = match.end()
            sign, number, denominator, axis = match.groups()
            if not number and not axis:
                continue
            factor = -1.0 if sign == "-" else 1.0
            if number:
                value = float(number)
                if denominator:
                    value /= float(denominator)
            else:
                value = 1.0
            if axis:
                rotation[row, axes[axis]] += factor * value
            else:
                translation[row] += factor * value
    return rotation, translation


# SHELX LATT: the centring translations, and whether a centre of symmetry is
# implied. A positive LATT number means centrosymmetric.
_LATT_CENTRING = {
    1: [],                                                   # P
    2: [(0.5, 0.5, 0.5)],                                    # I
    3: [(2 / 3, 1 / 3, 1 / 3), (1 / 3, 2 / 3, 2 / 3)],       # R (hexagonal)
    4: [(0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)],  # F
    5: [(0.0, 0.5, 0.5)],                                    # A
    6: [(0.5, 0.0, 0.5)],                                    # B
    7: [(0.5, 0.5, 0.0)],                                    # C
}


def expand_symmetry(rows, operators, latt: int | None = None,
                    tolerance: float = 1e-4):
    """Apply symmetry operators and centring to an asymmetric unit.

    Returns the expanded rows, each carrying the index of the site it came
    from, so a coordination analysis still reports per crystallographic site
    rather than per atom.

    Duplicates are removed by distance in fractional space, which is what
    collapses an atom sitting on a special position.
    """
    ops = [(np.eye(3), np.zeros(3))] + list(operators)

    centring = [(0.0, 0.0, 0.0)]
    inversion = False
    if latt is not None:
        centring = centring + [tuple(t) for t in
                               _LATT_CENTRING.get(abs(int(latt)), [])]
        inversion = int(latt) > 0

    out = []
    for site_index, (label, element, frac, occ) in enumerate(rows):
        seen: list[np.ndarray] = []
        for rotation, translation in ops:
            base = rotation @ np.asarray(frac, float) + translation
            for centre in centring:
                for invert in ((False, True) if inversion else (False,)):
                    position = (-base if invert else base) + np.array(centre)
                    position = position % 1.0
                    if any(_same(position, other, tolerance) for other in seen):
                        continue
                    seen.append(position)
                    out.append((label, element, position, occ, site_index))
    return out


def _same(a: np.ndarray, b: np.ndarray, tolerance: float) -> bool:
    """Equality modulo one lattice translation."""
    delta = np.abs(a - b) % 1.0
    delta = np.minimum(delta, 1.0 - delta)
    return bool(np.all(delta < tolerance))


# ---------------------------------------------------------------------------
# VASP
# ---------------------------------------------------------------------------

def read_poscar(path: str | Path) -> Structure:
    """VASP POSCAR or CONTCAR.

    Handles both the VASP 4 layout, where element symbols are absent and are
    taken from the comment line, and VASP 5, where they are on their own line.
    Selective dynamics is skipped; its flags say nothing about the structure.
    """
    path = Path(path)
    lines = [ln.rstrip("\n") for ln in
             path.read_text(encoding="utf-8", errors="replace").splitlines()]
    if len(lines) < 8:
        raise UnsupportedFormat(f"{path.name}: too short to be a POSCAR")

    comment = lines[0].strip()
    scale = float(lines[1].split()[0])
    vectors = np.array([[float(x) for x in lines[i].split()[:3]]
                        for i in (2, 3, 4)], float)
    if scale < 0:
        # a negative scale is a target volume, not a factor
        current = abs(float(np.linalg.det(vectors)))
        scale = (abs(scale) / current) ** (1.0 / 3.0) if current else 1.0
    vectors = vectors * scale

    tokens = lines[5].split()
    if tokens and not tokens[0].replace("-", "").isdigit():
        symbols = tokens
        counts = [int(x) for x in lines[6].split()]
        cursor = 7
    else:                               # VASP 4: symbols live in the comment
        counts = [int(x) for x in tokens]
        symbols = comment.split()[:len(counts)] or ["X"] * len(counts)
        cursor = 6

    if lines[cursor].strip().lower().startswith("s"):
        cursor += 1                     # selective dynamics
    mode = lines[cursor].strip().lower()
    direct = not mode.startswith("c")   # 'cartesian' or 'direct'/'fractional'
    cursor += 1

    cell = cell_from_vectors(vectors)
    inverse = np.linalg.inv(cell.orth)
    rows, n = [], 0
    for symbol, count in zip(symbols, counts):
        for _ in range(count):
            if cursor >= len(lines):
                break
            values = [float(x) for x in lines[cursor].split()[:3]]
            cursor += 1
            n += 1
            position = np.array(values, float)
            frac = position if direct else inverse @ (position * scale)
            rows.append((f"{elements.normalise(symbol)}{n}", symbol, frac, 1.0))
    return _assemble(comment or path.stem, cell, rows, str(path))


# ---------------------------------------------------------------------------
# XYZ
# ---------------------------------------------------------------------------

_LATTICE = re.compile(r'Lattice\s*=\s*"([^"]+)"', re.IGNORECASE)


def read_xyz(path: str | Path, cell_padding: float = 6.0) -> Structure:
    """XYZ, preferring the extended form.

    ``Lattice="..."`` in the comment line is the convention ASE and OVITO use
    and is what FACET writes. Without it there is no cell, and one is invented
    large enough that the molecule does not see its own images — which is
    stated in a note, because an invented cell is not a crystallographic fact.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise UnsupportedFormat(f"{path.name}: too short to be an XYZ file")

    count = int(lines[0].split()[0])
    comment = lines[1]

    positions, symbols = [], []
    for line in lines[2:2 + count]:
        parts = line.split()
        if len(parts) < 4:
            continue
        symbols.append(parts[0])
        positions.append([float(x) for x in parts[1:4]])
    positions = np.array(positions, float)
    if positions.size == 0:
        raise UnsupportedFormat(f"{path.name}: no coordinates")

    invented = False
    match = _LATTICE.search(comment)
    if match:
        values = [float(x) for x in match.group(1).split()]
        if len(values) != 9:
            raise UnsupportedFormat(f"{path.name}: malformed Lattice=")
        vectors = np.array(values, float).reshape(3, 3)
        cell = cell_from_vectors(vectors)
    else:
        span = positions.max(axis=0) - positions.min(axis=0)
        box = np.maximum(span + cell_padding, cell_padding)
        cell = cell_from_vectors(np.diag(box))
        positions = positions - positions.min(axis=0) + cell_padding / 2.0
        invented = True

    inverse = np.linalg.inv(cell.orth)
    rows = [(f"{elements.normalise(s)}{i + 1}", s, inverse @ p, 1.0)
            for i, (s, p) in enumerate(zip(symbols, positions))]
    structure = _assemble(comment.strip() or path.stem, cell, rows, str(path))
    if invented:
        structure.notes.append(
            f"no Lattice= in the comment line, so a {box[0]:.1f} x {box[1]:.1f} "
            f"x {box[2]:.1f} Å box was invented to hold the coordinates; it is "
            "not from the file")
    return structure


# ---------------------------------------------------------------------------
# VESTA
# ---------------------------------------------------------------------------

def read_vesta(path: str | Path) -> Structure:
    """A VESTA `.vesta` file.

    Only CELLP and STRUC are read — the cell and the atoms. The rest of the
    file is VESTA's presentation state, which FACET has its own equivalents for
    and which would be misleading to half-import.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    section, cell, rows, title = None, None, [], path.stem
    pending_cell: list[float] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        head = line.split()[0]
        if head.isupper() and len(head) >= 4 and head.isalpha():
            section = head
            continue

        if section == "TITLE" and title == path.stem:
            title = line
        elif section == "CELLP" and cell is None:
            values = _floats(line)
            if len(values) >= 6 and not pending_cell:
                pending_cell = values[:6]
                cell = cell_from_parameters(*pending_cell)
        elif section == "STRUC":
            parts = line.split()
            # index element label occupancy x y z ...
            # VESTA closes the section with a row of zeros, which has the same
            # shape as an atom line and must not be read as one
            if len(parts) >= 7 and parts[0].isdigit():
                if all(t.strip("0.-") == "" for t in parts):
                    section = None
                    continue
                try:
                    occ = float(parts[3])
                    frac = [float(parts[4]), float(parts[5]), float(parts[6])]
                except ValueError:
                    continue
                if not elements.normalise(parts[1]):
                    continue
                rows.append((parts[2], parts[1], np.array(frac), occ))

    if cell is None:
        raise UnsupportedFormat(f"{path.name}: no CELLP section")
    if not rows:
        raise UnsupportedFormat(f"{path.name}: no atoms in the STRUC section")
    return _assemble(title, cell, rows, str(path))


def _floats(line: str) -> list[float]:
    out = []
    for token in line.split():
        try:
            out.append(float(token))
        except ValueError:
            return out
    return out


# ---------------------------------------------------------------------------
# SHELX
# ---------------------------------------------------------------------------

_SHELX_COMMANDS = {
    "TITL", "CELL", "ZERR", "LATT", "SYMM", "SFAC", "UNIT", "TEMP", "SIZE",
    "OMIT", "ESEL", "ACTA", "FMAP", "PLAN", "WGHT", "FVAR", "L.S.", "BOND",
    "CONF", "HKLF", "END", "REM", "MORE", "BLOC", "DAMP", "SHEL", "MERG",
    "EQIV", "AFIX", "PART", "RESI", "ANIS", "HFIX", "DFIX", "SADI", "SIMU",
    "DELU", "ISOR", "EADP", "FREE", "SWAT", "EXTI", "BASF", "TWIN", "SPEC",
}


def read_shelx(path: str | Path) -> Structure:
    """A SHELX `.res` or `.ins`.

    The atom list is what is wanted; the instruction set is not. Atom lines are
    identified by having a label, a scattering-factor index that indexes into
    SFAC, and three coordinates — which is how SHELX itself distinguishes them
    from commands.

    Free variables are not resolved. An occupancy written as ``10.5`` or
    ``21.0`` encodes a free-variable reference in SHELX's own notation; the
    integer part is stripped and the site is noted, rather than guessed at.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    # SHELX continues a line with a trailing '='
    joined, buffer = [], ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.endswith("="):
            buffer += line[:-1] + " "
            continue
        joined.append(buffer + line)
        buffer = ""
    if buffer:
        joined.append(buffer)

    title, cell, sfac = path.stem, None, []
    rows, free_variable_sites = [], []
    operators, latt = [], None
    for line in joined:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        keyword = parts[0].upper()

        if keyword == "TITL":
            title = " ".join(parts[1:]) or title
        elif keyword == "CELL" and len(parts) >= 8:
            # CELL wavelength a b c alpha beta gamma
            cell = cell_from_parameters(*[float(x) for x in parts[2:8]])
        elif keyword == "SFAC":
            for token in parts[1:]:
                if token.replace(".", "").replace("-", "").isdigit():
                    break         # the numeric form of SFAC, not symbols
                sfac.append(elements.normalise(token))
        elif keyword == "LATT" and len(parts) >= 2:
            try:
                latt = int(float(parts[1]))
            except ValueError:
                latt = None
        elif keyword == "SYMM":
            parsed = parse_symop(" ".join(parts[1:]))
            if parsed is not None:
                operators.append(parsed)
        elif keyword == "END":
            break
        elif keyword in _SHELX_COMMANDS:
            continue
        elif len(parts) >= 5 and cell is not None and sfac:
            # label sfac_index x y z [occupancy] [U...]
            try:
                index = int(float(parts[1]))
                frac = [float(parts[2]), float(parts[3]), float(parts[4])]
            except ValueError:
                continue
            if not 1 <= index <= len(sfac):
                continue
            occupancy = 1.0
            if len(parts) >= 6:
                try:
                    raw_occ = float(parts[5])
                except ValueError:
                    raw_occ = 1.0
                if abs(raw_occ) > 1.5:
                    # SHELX encodes occupancy as 10*m + f, where m is a free
                    # variable index and f the value. 11.0 is the ordinary way
                    # to write "fixed at full occupancy" and is not a refined
                    # parameter, so only m > 1 is worth remarking on.
                    magnitude = abs(raw_occ)
                    m = int(magnitude // 10)
                    occupancy = magnitude - 10 * m
                    if occupancy <= 0.0:
                        occupancy = 1.0
                    if m > 1:
                        free_variable_sites.append(parts[0])
                else:
                    occupancy = abs(raw_occ)
            rows.append((parts[0], sfac[index - 1], np.array(frac), occupancy))

    if cell is None:
        raise UnsupportedFormat(f"{path.name}: no CELL instruction")
    if not rows:
        raise UnsupportedFormat(f"{path.name}: no atom lines found")

    # SHELX gives the asymmetric unit. Reading it without applying LATT and
    # SYMM would report a coordination number computed from a fraction of the
    # structure -- silently, and wrongly.
    asymmetric = len(rows)
    expanded = expand_symmetry(rows, operators, latt)
    structure = _assemble(title, cell, expanded, str(path), note_p1=False)
    structure.notes.append(
        f"SHELX: {asymmetric} sites in the asymmetric unit expanded to "
        f"{len(expanded)} atoms by {len(operators)} SYMM operation(s)"
        + (f" and LATT {latt}" if latt is not None else "")
        + (" including a centre of symmetry"
           if latt is not None and latt > 0 else ""))
    if free_variable_sites:
        structure.notes.append(
            "occupancy on " + ", ".join(free_variable_sites[:6])
            + " is a free-variable reference; the fractional part was taken "
              "and the free variables were not resolved")
    return structure


# ---------------------------------------------------------------------------
# PDB and mmCIF, through gemmi
# ---------------------------------------------------------------------------

def read_pdb(path: str | Path) -> Structure:
    """PDB or mmCIF, via gemmi.

    A macromolecular file usually has no meaningful small-molecule cell — many
    carry the placeholder P1 1 Å cube — so the cell is taken when it is real
    and invented around the coordinates when it is not.
    """
    import gemmi

    path = Path(path)
    doc = gemmi.read_structure(str(path))
    doc.setup_entities()
    if not len(doc):
        raise UnsupportedFormat(f"{path.name}: no models")

    positions, symbols, labels = [], [], []
    for chain in doc[0]:
        for residue in chain:
            for atom in residue:
                positions.append([atom.pos.x, atom.pos.y, atom.pos.z])
                symbols.append(atom.element.name)
                labels.append(f"{atom.name}_{residue.seqid.num}")
    if not positions:
        raise UnsupportedFormat(f"{path.name}: no atoms")
    positions = np.array(positions, float)

    invented = False
    g = doc.cell
    if g.a > 1.5 and g.b > 1.5 and g.c > 1.5:
        cell = cell_from_parameters(g.a, g.b, g.c, g.alpha, g.beta, g.gamma)
    else:
        span = positions.max(axis=0) - positions.min(axis=0)
        box = np.maximum(span + 8.0, 8.0)
        cell = cell_from_vectors(np.diag(box))
        positions = positions - positions.min(axis=0) + 4.0
        invented = True

    inverse = np.linalg.inv(cell.orth)
    rows = [(label, symbol, inverse @ p, 1.0)
            for label, symbol, p in zip(labels, symbols, positions)]
    structure = _assemble(doc.name or path.stem, cell, rows, str(path))
    if invented:
        structure.notes.append(
            "the file carries no usable unit cell, so a box was invented "
            "around the coordinates; it is not from the file")
    return structure


# ---------------------------------------------------------------------------
# CrystalMaker
# ---------------------------------------------------------------------------

def read_crystalmaker(path: str | Path) -> Structure:
    """CrystalMaker's own formats.

    `.cmtx` is CrystalMaker's **text** format and is readable: it is keyword
    driven, with CELL and ATOM records. `.cmdf` is a binary format that is not
    openly documented; rather than guess at its layout and risk returning a
    plausible but wrong structure, it is refused with a message saying how to
    get the data out — CrystalMaker exports CIF, and a CIF is unambiguous.
    """
    path = Path(path)
    if path.suffix.lower() == ".cmdf":
        raise UnsupportedFormat(
            f"{path.name}: CrystalMaker's .cmdf is an undocumented binary "
            "format. Rather than guess at its layout, FACET does not read it. "
            "In CrystalMaker use File ▸ Export ▸ CIF, or save as .cmtx, and "
            "open that instead.")

    text = path.read_text(encoding="utf-8", errors="replace")
    cell, rows, title = None, [], path.stem
    operators, spacegroup = [], None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("!"):
            continue
        parts = line.split()
        keyword = parts[0].upper()
        if keyword == "CELL" and len(parts) >= 7:
            cell = cell_from_parameters(*[float(x) for x in parts[1:7]])
        elif keyword in ("TITL", "TITLE"):
            title = " ".join(parts[1:]) or title
        elif keyword in ("SGRP", "SPGP", "SPCGRP") and len(parts) >= 2:
            spacegroup = " ".join(parts[1:])
        elif keyword == "SYMM" and len(parts) >= 2:
            parsed = parse_symop(" ".join(parts[1:]))
            if parsed is not None:
                operators.append(parsed)
        elif keyword == "ATOM" and len(parts) >= 6 and cell is not None:
            # ATOM element label x y z [occupancy]
            try:
                frac = [float(parts[3]), float(parts[4]), float(parts[5])]
            except ValueError:
                continue
            occupancy = float(parts[6]) if len(parts) >= 7 else 1.0
            rows.append((parts[2], parts[1], np.array(frac), occupancy))

    if cell is None or not rows:
        raise UnsupportedFormat(
            f"{path.name}: no CELL and ATOM records found. If this is a binary "
            ".cmdf saved with a .cmtx extension, export a CIF instead.")
    if operators:
        asymmetric = len(rows)
        rows = expand_symmetry(rows, operators)
        structure = _assemble(title, cell, rows, str(path),
                              spacegroup=spacegroup, note_p1=False)
        structure.notes.append(
            f"{asymmetric} sites expanded to {len(rows)} atoms by "
            f"{len(operators)} SYMM operation(s)")
        return structure
    structure = _assemble(title, cell, rows, str(path), spacegroup=spacegroup)
    if spacegroup:
        structure.notes.append(
            f"the file declares space group {spacegroup} but lists no SYMM "
            "operations, so the coordinates were taken as given")
    return structure


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

READERS = {
    ".cif": None,                    # handled by facet.core.cif
    ".mcif": None,
    ".vasp": read_poscar,
    ".poscar": read_poscar,
    ".contcar": read_poscar,
    ".xyz": read_xyz,
    ".extxyz": read_xyz,
    ".vesta": read_vesta,
    ".res": read_shelx,
    ".ins": read_shelx,
    ".pdb": read_pdb,
    ".ent": read_pdb,
    ".cmtx": read_crystalmaker,
    ".cmdf": read_crystalmaker,
}

FILE_FILTER = (
    "All structures (*.cif *.mcif POSCAR* CONTCAR* *.vasp *.xyz *.extxyz "
    "*.vesta *.res *.ins *.pdb *.ent *.cmtx);;"
    "CIF (*.cif *.mcif);;"
    "VASP (POSCAR* CONTCAR* *.vasp);;"
    "XYZ (*.xyz *.extxyz);;"
    "VESTA (*.vesta);;"
    "SHELX (*.res *.ins);;"
    "PDB / mmCIF (*.pdb *.ent);;"
    "CrystalMaker text (*.cmtx);;"
    "All files (*)"
)


def read(path: str | Path) -> Structure:
    """Read any supported structure file, choosing by extension then content.

    POSCAR and CONTCAR have no extension, so the stem is checked too. A file
    whose extension says nothing is sniffed rather than refused.
    """
    from . import cif as cif_module

    path = Path(path)
    suffix = path.suffix.lower()
    stem = path.name.upper()

    if suffix in (".cif", ".mcif"):
        return cif_module.read(path)
    if stem.startswith(("POSCAR", "CONTCAR")):
        return read_poscar(path)

    reader = READERS.get(suffix)
    if reader is not None:
        return reader(path)
    return _sniff(path)


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _sniff(path: Path) -> Structure:
    """Identify a file by looking at it."""
    from . import cif as cif_module

    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:4000]
    except OSError as exc:
        raise UnsupportedFormat(f"{path.name}: {exc}") from exc

    if "#VESTA_FORMAT_VERSION" in head:
        return read_vesta(path)
    if re.search(r"^data_", head, re.MULTILINE):
        return cif_module.read(path)
    if re.search(r"^CELL\s", head, re.MULTILINE) and \
            re.search(r"^SFAC\s", head, re.MULTILINE):
        return read_shelx(path)
    if head.lstrip().startswith(("ATOM  ", "HETATM", "HEADER", "CRYST1")):
        return read_pdb(path)

    lines = [ln for ln in head.splitlines() if ln.strip()]
    # An XYZ starts with a bare atom count and its third line is
    # "element x y z". Testing only the first line confuses it with a POSCAR
    # whose title happens to be numeric -- a COD number, for instance.
    if len(lines) >= 3:
        first = lines[0].split()
        third = lines[2].split()
        if (len(first) == 1 and first[0].isdigit() and len(third) >= 4
                and elements.normalise(third[0])
                and all(_is_float(t) for t in third[1:4])):
            return read_xyz(path)
    # A POSCAR's second line is a single scale factor and lines 3-5 are three
    # numbers each.
    if len(lines) > 7:
        try:
            float(lines[1].split()[0])
            for i in (2, 3, 4):
                if len([x for x in lines[i].split() if _is_float(x)]) < 3:
                    raise ValueError
            return read_poscar(path)
        except (ValueError, IndexError):
            pass
    raise UnsupportedFormat(
        f"{path.name}: the format was not recognised. FACET reads CIF, POSCAR, "
        "XYZ, .vesta, SHELX .res/.ins, PDB/mmCIF and CrystalMaker .cmtx.")
