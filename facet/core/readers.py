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


class MDModelFile(UnsupportedFormat):
    """An MD model or trajectory: opened with ``md_readers.read_trajectory``.

    :func:`read` raises it for LAMMPS data and dump files, multi-frame XYZ,
    VASP XDATCAR, DL_POLY CONFIG / HISTORY, PDB files holding more than one
    structure and the formats of ``md_formats_base.FORMAT_MODULES`` (by
    content, and by name where the name is not a crystal one), whatever the
    file's extension where its content says so, so that a 10 000-atom frame never
    becomes a per-site crystal Structure here: that path runs spglib on every
    file and the crystal window resolves oxidation states from the geometry,
    while an MD model's states are the force field's inputs. A subclass of
    UnsupportedFormat, so every caller that already handles an unreadable file
    handles this one. ``path`` and ``file_format`` (one of
    ``md_readers.format_names()``, or ``md_readers.PDB_TRAJECTORY`` for a
    PDB of several structures no format module claims) say what was
    recognised.
    """

    def __init__(self, message: str, *, path: str = "",
                 file_format: str = "") -> None:
        super().__init__(message)
        self.path = path
        self.file_format = file_format


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

    try:
        count = int(lines[0].split()[0])
    except (ValueError, IndexError):
        from . import md_readers

        looks = md_readers.describe_unread(path, for_readers_read=True)
        raise UnsupportedFormat(
            f"{path.name}: line 1 ({lines[0].strip()[:40]!r}) is not the atom "
            "count an XYZ file starts with" + (f"; {looks}" if looks else "")
            + ". " + _READS) from None
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
    unnamed = [s for s in symbols if not elements.normalise(s)]
    if unnamed:
        raise UnsupportedFormat(
            f"{path.name}: {len(unnamed)} of its {len(symbols)} atom rows name "
            f"the atom {unnamed[0]!r}, which is not an element symbol, and the "
            "crystal reader names atoms by their symbols. A file that numbers "
            "them (LAMMPS's 'dump xyz' writes each atom's type number when no "
            "element is set) reads with FACET's MD reader and a type map: "
            "facet.core.md_readers.read_trajectory(path, type_map={<number>: "
            "'<element>', ...}), with box_from= when the file states no box")

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

def read_pdb(path: str | Path, *, pdb_layout: bool = False) -> Structure:
    """PDB or mmCIF, via gemmi.

    A macromolecular file usually has no meaningful small-molecule cell — many
    carry the placeholder P1 1 Å cube — so the cell is taken when it is real
    and invented around the coordinates when it is not. gemmi chooses the
    layout by the extension; ``pdb_layout`` reads the file as PDB whatever
    its name (gemmi raised RuntimeError 'Unknown format' for a PDB saved as
    .txt).
    """
    import gemmi

    path = Path(path)
    if pdb_layout:
        doc = gemmi.read_structure(str(path), format=gemmi.CoorFormat.Pdb)
    else:
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

    An MD model raises :class:`MDModelFile` instead (see :func:`_md_format`),
    so that no reader here keeps one frame of a trajectory silently: a
    multi-frame ``.xyz`` / ``.extxyz`` with ``Lattice=``, an XDATCAR saved as
    ``.vasp`` / ``.poscar`` / POSCAR, a LAMMPS dump or data file saved under
    a crystal extension (``.cif`` included when no ``data_`` line opens a CIF
    block in its first 4 kB), a PDB holding more than one structure, a binary
    format a format module reads (by its content, or by its extension when
    the content shows nothing else), a file whose name a format module
    claims, and any file with no crystal extension that
    ``md_readers.sniff_md`` recognises. A multi-frame XYZ without
    ``Lattice=`` (LAMMPS ``dump xyz``, ASE's xyz, CP2K's pos.xyz), and one
    frame of LAMMPS's ``dump xyz`` or CP2K's XMOL trajectory, are refused
    with UnsupportedFormat saying how the MD reader opens them (box_from=):
    the crystal reader would invent a box around a periodic model. A CIF
    holding several images of one model (ASE writes a trajectory so) is
    refused, saying how to convert it; a CIF holding several structures is
    read as its first block with a cell, and a note says so. A missing file
    raises FileNotFoundError. A single-frame XYZ and every crystal file take
    the path they always took; when a crystal reader stops on a file whose
    content a format module's sniff recognises (a LAMMPS YAML dump saved as
    .vasp), MDModelFile is raised instead, and otherwise its IndexError,
    RuntimeError (gemmi) or a ValueError that does not name the file becomes
    UnsupportedFormat naming the file (with what it looks like, when that is
    known and is not what its name already says).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path}: no such file")
    suffix = path.suffix.lower()
    stem = path.name.upper()

    if suffix in (".cif", ".mcif"):
        return _read_cif(path, suffix, stem)

    found = _md_format(path, suffix, stem)
    if found is not None:
        raise _md_model_file(path, *found)

    try:
        if stem.startswith(("POSCAR", "CONTCAR")):
            return read_poscar(path)
        reader = READERS.get(suffix)
        if reader is not None:
            return reader(path)
        return _sniff(path)
    except MDModelFile:
        raise
    except (ValueError, IndexError, RuntimeError) as error:
        routed = _module_format_after_failure(path)
        if routed is not None:
            raise _md_model_file(path, *routed) from error
        if isinstance(error, UnsupportedFormat) or (
                isinstance(error, ValueError) and path.name in str(error)):
            raise                   # the reader's own words, naming the file
        looks = _looks_after_failure(path, suffix, stem)
        raise UnsupportedFormat(
            f"{path.name}: the structure reader stopped on it "
            f"({type(error).__name__}: {error})"
            + (f"; {looks}" if looks else "") + f". {_READS}") from error


# The names under which readers.read gives each single-structure kind of
# md_readers.describe_unread to its reader.
_KIND_NAMES = (("a CIF", (".cif", ".mcif"), ()),
               ("a PDB file holding one structure", (".pdb", ".ent"), ()),
               ("a VASP POSCAR", (".vasp", ".poscar"), ("POSCAR", "CONTCAR")))


def _looks_after_failure(path: Path, suffix: str, stem: str) -> str:
    """What the file looks like, for a crystal reader that stopped on it
    (gemmi on a CIF saved as .pdb), or '' -- also when the advice would be
    to save it under the name it already has."""
    from . import md_readers

    looks = md_readers.describe_unread(path, for_readers_read=True)
    for kind, suffixes, stems in _KIND_NAMES:
        if f"it looks like {kind}" in looks and (
                suffix in suffixes or stem.startswith(stems or ("\0",))):
            return ""
    return looks


# Format-module formats that a crystal reader may also read: a lone POSCAR
# stays the crystal reader's (md_formats_text reads it as one frame only when
# read_trajectory is asked to), so its refusal stays the crystal reader's.
_CRYSTAL_SHAPED_FORMATS = ("vasp-poscar",)


def _module_format_after_failure(path: Path) -> tuple[str, str] | None:
    """(format, why) for a file under a crystal name that the crystal reader
    could not read and whose content a format module's sniff recognises (a
    LAMMPS YAML dump saved as .vasp, a CASTEP .md saved as .pdb or .res);
    None otherwise. Consulted only after the crystal reader stopped, so a
    file the crystal reader reads (a POSCAR, a one-model PDB) is never taken
    from it: _md_format does not consult the format modules for a crystal
    name. Integration re-run of the recognition corpus, 2026-10-07: such
    files gave 'no atoms', 'no CELL instruction', or a bare ValueError
    ("could not convert string to float: 'creator:'") naming no file."""
    from . import md_readers

    md_format = md_readers.sniff_md(path)
    if md_format is None or md_format in md_readers.MD_FORMATS \
            or md_format in _CRYSTAL_SHAPED_FORMATS:
        return None
    return md_format, (f"is an MD model ({md_format}) under a crystal file "
                       "name, which the crystal reader does not read")


# What the crystal readers and the MD reader read, for refusals.
_CRYSTAL_TEXT = ("CIF, POSCAR, XYZ, .vesta, SHELX .res/.ins, PDB/mmCIF and "
                 "CrystalMaker .cmtx")
_READS = (f"FACET reads {_CRYSTAL_TEXT} as crystal structures, and MD models "
          "with facet.core.md_readers.read_trajectory.")


def _read_cif(path: Path, suffix: str, stem: str) -> Structure:
    """A .cif / .mcif file: a binary MD format or MD text under the name is
    routed (MD text only when no ``data_`` line opens a CIF block in the
    first 4 kB, so a CIF costs one 4 kB read), NUL bytes are refused in
    words, a CIF of several images of one model is refused, and a CIF of
    several structures is read as its first block with a cell, noted."""
    from . import cif as cif_module
    from . import md_readers

    if _holds_nul(path):
        found = _binary_md_format(path)
        if found is not None:
            raise _md_model_file(path, *found)
        _refuse_binary(path, md_readers)
    block_in_head = _cif_block_in_head(path)
    if not block_in_head:
        found = _md_format(path, suffix, stem)
        if found is not None:
            raise _md_model_file(path, *found)
    blocks = _cif_blocks(path)
    if len(blocks) > 1 and blocks[0][1] \
            and len({signature for _, signature in blocks}) == 1:
        names = ", ".join(name for name, _ in blocks[:4]) \
            + (", ..." if len(blocks) > 4 else "")
        raise UnsupportedFormat(
            f"{path.name} holds {len(blocks)} data blocks with a cell ({names}) "
            "that list the same atoms in the same order: images of one model, "
            "as ASE writes a trajectory to CIF. The crystal reader takes one "
            "structure, so it would keep the first image only. ASE writes the "
            "images as extended XYZ (ase.io.write('out.extxyz', "
            "ase.io.read(path, index=':'))), which "
            "facet.core.md_readers.read_trajectory reads as a trajectory; to "
            "open one image here, save it as a CIF of its own.")
    try:
        structure = cif_module.read(path)
    except (ValueError, RuntimeError) as error:
        if block_in_head and isinstance(error, ValueError):
            raise                   # the CIF reader's own words
        if block_in_head:
            # gemmi's parser (a duplicate tag, a syntax error)
            raise UnsupportedFormat(
                f"{path.name}: gemmi's CIF parser stopped on it ({error})") \
                from error
        routed = _module_format_after_failure(path)
        if routed is not None:
            raise _md_model_file(path, *routed) from error
        looks = md_readers.describe_unread(path, for_readers_read=True)
        raise UnsupportedFormat(
            f"{path.name}: not read as a CIF ({error})"
            + (f"; {looks}" if looks else "")
            + f". {_READS} A file in another format opens once converted to "
            "one of these (OVITO and ASE write extended XYZ and CIF).") \
            from error
    if len(blocks) > 1:
        names = ", ".join(name for name, _ in blocks[:6]) \
            + (", ..." if len(blocks) > 6 else "")
        structure.notes.append(
            f"the file holds {len(blocks)} data blocks with a cell ({names}); "
            "the first of them was read, and the others are not")
    return structure


def _cif_block_in_head(path: Path, size: int = 4096) -> bool:
    """Whether a line starting with data_ (any case, as CIF allows) lies in
    the first ``size`` bytes."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(size)
    except OSError:
        return False
    return re.search(rb"(?im)^data_", head) is not None


def _cif_blocks(path: Path) -> list[tuple[str, tuple[str, ...]]]:
    """(name, the atom-site labels in order) of every data block that has a
    cell, when the file holds more than one data block; [] otherwise (one
    block, or text gemmi does not parse, which cif.read then reports)."""
    try:
        with open(path, "rb") as handle:
            starts = sum(1 for line in handle
                         if line[:5].lower() == b"data_")
    except OSError:
        return []
    if starts < 2:
        return []
    import gemmi

    try:
        doc = gemmi.cif.read(str(path))
    except (ValueError, RuntimeError):
        return []
    out = []
    for block in doc:
        if not block.find_value("_cell_length_a"):
            continue
        labels = tuple(block.find_values("_atom_site_label"))
        out.append((f"data_{block.name}", labels))
    return out


def _pdb_first(path: Path) -> bool:
    """Whether the first non-blank line of the file is a PDB record that a
    PDB file starts with (wwPDB format 3.3)."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(4096)
    except OSError:
        return False
    first = next((x for x in head.splitlines() if x.strip()), b"")
    return first[:6].rstrip().decode("ascii", "replace") in (
        "HEADER", "REMARK", "CRYST1", "MODEL", "ATOM", "HETATM", "TITLE",
        "COMPND")


# The MD formats that are never a crystal file, whatever the file's name.
_NOT_CRYSTAL = ("lammps-data", "lammps-dump", "vasp-xdatcar", "dlpoly-config",
                "dlpoly-history")


def _md_model_file(path: Path, md_format: str, why: str) -> MDModelFile:
    return MDModelFile(
        f"{path.name} {why}. The crystal window does not open it; FACET's "
        "MD reader does (facet.core.md_readers.read_trajectory), reading "
        "its frames as arrays and keeping the model's oxidation states, "
        "where the crystal reader would build one site per atom, search "
        "its symmetry and resolve oxidation states from the geometry.",
        path=str(path), file_format=md_format)


def _holds_nul(path: Path, size: int = 4096) -> bool:
    try:
        with open(path, "rb") as handle:
            return b"\x00" in handle.read(size)
    except OSError:
        return False


def _binary_md_format(path: Path) -> tuple[str, str] | None:
    """(format, why) for a binary format a format module reads, by the
    file's first bytes or, when they show nothing else, its extension;
    None otherwise."""
    from . import md_readers

    found = md_readers.binary_md_match(path)
    if found is None:
        return None
    spec, by_content = found
    if by_content:
        return spec.name, (f"is a binary MD file: {spec.description} "
                           f"({spec.name})")
    return spec.name, (f"has the name of {spec.description} ({spec.name}), a "
                       "binary format FACET's MD reader reads (its first "
                       "bytes do not show another format)")


def _md_format(path: Path, suffix: str, stem: str = "") -> tuple[str, str] | None:
    """(the MD format :func:`read` refuses this file as, why), or None.

    In this order:

    * a binary format a format module reads, by its first bytes, or by its
      extension when the file is not empty and its first bytes show no
      crystal structure (``md_readers.binary_md_match``): a binary file
      never reaches a crystal reader, and a CIF renamed .dcd reaches it;
    * a crystal name (an extension of :data:`READERS`, or a POSCAR / CONTCAR
      stem): the built-in MD formats are sniffed from the first 4 kB (a VASP
      or extended XYZ header with long lines is followed past it), and a
      LAMMPS dump or data file, an XDATCAR or a DL_POLY file is routed
      whatever its name (an XDATCAR saved as .vasp would otherwise give its
      first configuration as a POSCAR, a dump saved as .xyz a ValueError);
      an XYZ-shaped file is routed when a second frame follows the first,
      or when it is gzip-compressed with ``Lattice=``, and refused when its
      comment line is LAMMPS's 'dump xyz' or CP2K's XMOL one (a periodic
      model that states no box); a file holding more than one PDB structure
      (MODEL records, or ATOM records after an END record) is routed; a
      file holding binary bytes that nothing recognised is refused, because
      a text reader would fail on it with an error that is not
      UnsupportedFormat. A format module's sniff is not consulted for a
      crystal name: a single-model PDB or a POSCAR is a crystal file;
    * any other name: ``md_readers.sniff_md`` (the built-in formats, then
      the format modules), then a format module's extension or stem
      (``md_readers.named_md_format``) for a file that is not empty and
      shows no crystal structure, so a LAMMPS input script named ``.lmp``
      and a junk file still reach ``_sniff`` and its message.

    Raises UnsupportedFormat for an XYZ without ``Lattice=`` that the MD
    reader opens only with ``box_from=``, and for a multi-frame XYZ whose
    lines end with CR alone.
    """
    from . import md_readers

    binary = _binary_md_format(path)
    if binary is not None:
        return binary
    crystal = suffix in READERS or stem.startswith(("POSCAR", "CONTCAR"))
    if crystal:
        if suffix == ".cmdf":
            return None
        md_format = md_readers.sniff_md(path, registered=False)
        if md_format in _NOT_CRYSTAL:
            return md_format, (f"is an MD model ({md_format}) under a crystal "
                               "file name, which the crystal reader would "
                               "read as one structure or not at all")
        xyz_name = suffix in (".xyz", ".extxyz")
        if xyz_name or md_format == "extxyz":
            if md_readers.is_multiframe_xyz(path):
                return "extxyz", _multiframe_xyz_reason(path, md_readers)
            _refuse_boxless_frame(path, md_readers)
            if xyz_name:
                gzipped = _gzipped_xyz(path, md_readers)
                if gzipped is not None:
                    return gzipped
        if (suffix in (".pdb", ".ent") or _pdb_first(path)) \
                and md_readers.pdb_model_count(path) >= 2:
            return md_readers.pdb_trajectory_format(path), (
                "holds more than one structure (MODEL records, or ATOM records "
                "after an END record): a trajectory, of which the crystal "
                "reader would keep only the first")
        _refuse_binary(path, md_readers)
        return None
    md_format = md_readers.sniff_md(path)
    if md_format == "extxyz":
        if md_readers.is_multiframe_xyz(path):
            return "extxyz", _multiframe_xyz_reason(path, md_readers)
        _refuse_boxless_frame(path, md_readers)
        return _gzipped_xyz(path, md_readers)
    if md_format is not None:
        return md_format, f"is an MD model ({md_format})"
    named = md_readers.named_md_format(path)
    if named is not None and md_readers._name_may_route(path):
        return named.name, (f"has the name of {named.description} "
                            f"({named.name}), a format FACET's MD reader reads")
    return None


def _refuse_binary(path: Path, md_readers) -> None:
    """UnsupportedFormat for a file with a crystal extension whose first
    bytes hold a NUL byte, as binary files and UTF-16 text do (PowerShell
    5.1's Out-File writes UTF-16) and no text format FACET reads does; a
    text reader would fail on it with an error that is not
    UnsupportedFormat. Says what the file looks like when that is known."""
    try:
        with open(path, "rb") as handle:
            raw = handle.read(md_readers.SNIFF_BYTES)
    except OSError:
        return
    if b"\x00" not in raw or raw[:2] == b"\x1f\x8b":
        return
    looks = md_readers.describe_unread(path, for_readers_read=True)
    utf16 = raw[:2] in (b"\xff\xfe", b"\xfe\xff")
    raise UnsupportedFormat(
        f"{path.name} holds NUL bytes, as "
        + ("UTF-16 text does (it starts with a UTF-16 byte-order mark): save "
           "it as UTF-8 or ASCII text" if utf16 else
           "binary data and UTF-16 text do, which no text structure format "
           "holds")
        + (f" ({looks})" if looks and not utf16 else "")
        + f". FACET reads {_CRYSTAL_TEXT} as crystal structures, and "
        f"{md_readers.readable_formats()} as MD models "
        "(facet.core.md_readers.read_trajectory).")


def _gzipped_xyz(path: Path, md_readers) -> tuple[str, str] | None:
    """A one-frame XYZ with Lattice= compressed with gzip, whatever its name:
    the crystal reader reads plain text only (read_xyz never read such a
    file), and the MD reader opens it."""
    try:
        with open(path, "rb") as handle:
            gzipped = handle.read(2) == b"\x1f\x8b"
    except OSError:
        return None
    if gzipped and md_readers.xyz_has_lattice(path):
        return "extxyz", ("is a gzip-compressed extended XYZ model, and the "
                          "crystal reader reads plain text only")
    return None


_BOX_FROM = ("FACET's MD reader (facet.core.md_readers.read_trajectory) reads "
             "periodic models only: it reads this file given a box, "
             "box_from=<another MD file of the same run, such as its LAMMPS "
             "data file or CP2K PROJECT-1.cell file> or the box as (3, 3) rows "
             "in Å, which holds every frame in that one box (a constant "
             "volume)")


def _refuse_boxless_frame(path: Path, md_readers) -> None:
    """UnsupportedFormat for one frame of LAMMPS's 'dump xyz' or CP2K's XMOL
    trajectory: a periodic model that states no box, which the crystal
    reader would place in an invented box (surfaces in a periodic glass),
    and whose LAMMPS type numbers it would not read as elements."""
    if md_readers.xyz_has_lattice(path):
        return
    writer = md_readers.plain_xyz_writer(path)
    if writer is None:
        return
    raise UnsupportedFormat(
        f"{path.name} is a frame of {writer} (its comment line is the one that "
        "program writes), which states no periodic box (no Lattice=). The "
        "crystal reader would place its atoms in an invented box, which puts "
        f"surfaces into a periodic model. {_BOX_FROM}; LAMMPS type numbers in "
        "place of element names take type_map={<number>: '<element>', ...}. "
        "A LAMMPS run writes the box with 'dump custom' or 'dump extxyz'.")


def _multiframe_xyz_reason(path: Path, md_readers) -> str:
    """Why a multi-frame XYZ goes to the MD reader; UnsupportedFormat when it
    has no Lattice=, because then neither reader opens it, and when its
    lines end with CR alone, which the MD reader refuses."""
    if md_readers.cr_only_line_endings(path):
        raise UnsupportedFormat(
            f"{path.name} holds more than one XYZ frame, and its lines end with "
            "CR alone (classic Mac OS line endings). The crystal reader takes "
            "one structure per file (it would keep only the first frame), and "
            "FACET's MD reader (facet.core.md_readers.read_trajectory) reads "
            "LF and CRLF line endings: convert the file's line endings, and "
            "the MD reader opens it.")
    if not md_readers.xyz_has_lattice(path):
        raise UnsupportedFormat(
            f"{path.name} holds more than one XYZ frame and no periodic box (no "
            "Lattice= on its first comment line). The crystal reader takes one "
            f"structure per file, and {_BOX_FROM}. To open one frame here, "
            "save it as an .xyz file of its own; a LAMMPS run writes the box "
            "with 'dump custom' or 'dump extxyz' (its 'dump xyz' writes none).")
    return ("holds more than one XYZ frame, and the crystal reader takes one "
            "structure per file (it would keep only the first frame)")


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _sniff(path: Path) -> Structure:
    """Identify a file by looking at it."""
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:4000]
    except OSError as exc:
        raise UnsupportedFormat(f"{path.name}: {exc}") from exc

    if "#VESTA_FORMAT_VERSION" in head:
        return read_vesta(path)
    if re.search(r"^data_", head, re.MULTILINE):
        # as a .cif: images of one trajectory are refused, several
        # structures noted
        return _read_cif(path, path.suffix.lower(), path.name.upper())
    if re.search(r"^CELL\s", head, re.MULTILINE) and \
            re.search(r"^SFAC\s", head, re.MULTILINE):
        return read_shelx(path)
    if head.lstrip().startswith(("ATOM  ", "HETATM", "HEADER", "CRYST1")) or (
            _pdb_first(path)
            and re.search(r"^(ATOM  |HETATM)", head, re.MULTILINE)):
        # gemmi chooses PDB or mmCIF by the extension; this name has none
        # it knows
        return read_pdb(path, pdb_layout=True)

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
    from . import md_readers

    looks = md_readers.describe_unread(path, for_readers_read=True)
    raise UnsupportedFormat(
        f"{path.name}: the format was not recognised"
        + (f" ({looks})" if looks else "")
        + f". FACET reads {_CRYSTAL_TEXT} as crystal structures, and MD models "
        f"and trajectories ({md_readers.readable_formats()}) with "
        "facet.core.md_readers.read_trajectory. A file in another format "
        "opens once converted to one of these (OVITO and ASE write extended "
        "XYZ and CIF).")
