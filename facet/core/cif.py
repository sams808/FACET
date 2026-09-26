"""CIF ingestion.

gemmi parses (about 0.8 ms for a small-molecule CIF, roughly sixty times faster
than pymatgen's reader) and expands symmetry; spglib independently determines
the space group, Wyckoff letters and site symmetries from the coordinates.

Where the two disagree, FACET says so rather than picking a winner. A CIF whose
stated Hermann-Mauguin symbol contradicts its own symmetry operator list is a
real and not-rare defect, and silently resolving it is how a wrong coordination
number gets published.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import disorder as disorder_mod
from . import elements
from .structure import Atom, Cell, Site, Structure

_ESD_RE = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?:\((\d+)\))?\s*$")


def parse_value(text) -> tuple[float | None, float | None]:
    """Split a CIF numeric such as ``0.7771(4)`` into value and esd.

    The esd digits apply to the last digits of the mantissa, so ``0.7771(4)``
    is 0.7771 +- 0.0004 and ``12.34(12)`` is 12.34 +- 0.12. Returns
    ``(None, None)`` for the CIF nulls ``?`` and ``.``.
    """
    if text is None:
        return None, None
    s = str(text).strip().strip("'\"")
    if s in ("", "?", "."):
        return None, None
    m = _ESD_RE.match(s)
    if not m:
        try:
            return float(s), None
        except ValueError:
            return None, None
    value = float(m.group(1))
    if m.group(2) is None:
        return value, None
    digits = m.group(1)
    if "." in digits:
        decimals = len(digits.split(".")[1])
        esd = int(m.group(2)) * 10.0 ** (-decimals)
    else:
        esd = float(m.group(2))
    return value, esd


def read(path: str | Path, block: str | None = None) -> Structure:
    """Read one CIF into a :class:`Structure`.

    `block` selects a data block by name in a multi-block file; the first block
    carrying a unit cell is used otherwise.
    """
    import gemmi

    path = Path(path)
    doc = gemmi.cif.read(str(path))
    blk = _choose_block(doc, block)
    st = gemmi.make_small_structure_from_block(blk)

    if st.cell is None or st.cell.a <= 0:
        raise ValueError(f"{path.name}: no usable unit cell in block {blk.name!r}")

    cell = Cell(st.cell.a, st.cell.b, st.cell.c,
                st.cell.alpha, st.cell.beta, st.cell.gamma,
                np.array(st.cell.orth.mat.tolist(), float))

    sites = _read_sites(blk, st)
    if not sites:
        raise ValueError(f"{path.name}: block {blk.name!r} has no atom sites "
                         "with coordinates")

    atoms, symmetry_note = _expand(st, cell, sites, blk)

    out = Structure(
        name=_text(blk, "_chemical_name_mineral")
             or _text(blk, "_chemical_name_systematic")
             or blk.name or path.stem,
        cell=cell,
        sites=sites,
        atoms=atoms,
        spacegroup_hm=_spacegroup_symbol(blk, st),
        source_path=str(path),
        formula=_text(blk, "_chemical_formula_sum"),
        database_code=_text(blk, "_cod_database_code")
                      or _text(blk, "_database_code_ICSD"),
        reference=_text(blk, "_publ_author_name"),
        year=_int(blk, "_journal_year"),
    )
    out.notes.append(f"symmetry expansion used {symmetry_note}")
    _annotate_symmetry(out)
    return out


def read_all_blocks(path: str | Path) -> list[Structure]:
    """Every usable block of a multi-block CIF. Unusable blocks are skipped and
    the reason recorded on the structures that did parse."""
    import gemmi

    doc = gemmi.cif.read(str(path))
    out, skipped = [], []
    for blk in doc:
        try:
            out.append(read(path, block=blk.name))
        except Exception as exc:
            skipped.append(f"block {blk.name!r}: {exc}")
    for s in out:
        s.notes.extend(skipped)
    if not out:
        raise ValueError(f"{Path(path).name}: no usable block "
                         f"({'; '.join(skipped) if skipped else 'file empty'})")
    return out


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

def _choose_block(doc, want: str | None):
    """The block to read, or a reason there is none.

    An empty file, or one with no data block, used to reach gemmi's
    ``sole_block()`` and come back as ``IndexError: invalid vector subscript``.
    Dropping an empty file on the window is an ordinary thing to do by accident,
    and a bare subscript error from three layers down is not an answer.
    """
    blocks = list(doc)
    if want is not None:
        for blk in blocks:
            if blk.name == want:
                return blk
        raise KeyError(
            f"no block named {want!r}; this file has "
            + (", ".join(repr(b.name) for b in blocks) if blocks
               else "no data blocks at all"))
    if not blocks:
        raise ValueError(
            "the file contains no CIF data block. An empty file, or a text "
            "file that is not a CIF, looks like this.")
    for blk in blocks:
        if blk.find_value("_cell_length_a"):
            return blk
    return blocks[0]


def _text(blk, tag: str) -> str | None:
    import gemmi

    v = blk.find_value(tag)
    if v is None:
        return None
    s = gemmi.cif.as_string(v).strip()
    return s or None


def _int(blk, tag: str) -> int | None:
    v, _ = parse_value(_text(blk, tag))
    return int(v) if v is not None else None


def _spacegroup_number_of(symbol: str | None) -> int | None:
    """International Tables number for a Hermann-Mauguin symbol, or None.

    gemmi understands the several spellings CIFs use, including the underscore
    form and non-standard settings.
    """
    if not symbol:
        return None
    import gemmi

    for candidate in (symbol, symbol.replace("_", ""), symbol.replace(" ", "")):
        try:
            sg = gemmi.find_spacegroup_by_name(candidate)
        except Exception:
            sg = None
        if sg is not None:
            return int(sg.number)
    return None


def _spacegroup_symbol(blk, st) -> str | None:
    for tag in ("_space_group_name_H-M_alt", "_symmetry_space_group_name_H-M",
                "_space_group_name_H-M"):
        s = _text(blk, tag)
        if s:
            return s
    return st.spacegroup_hm or None


def _read_sites(blk, st) -> list[Site]:
    """Sites of the asymmetric unit, with coordinate esds preserved.

    gemmi's SmallStructure drops the esds, so the raw loop is read alongside it
    to recover them -- they are needed later to say whether a reported
    coordination number is resolvable at all.
    """
    esd_by_label: dict[str, np.ndarray] = {}
    try:
        loop = blk.find(["_atom_site_label", "_atom_site_fract_x",
                         "_atom_site_fract_y", "_atom_site_fract_z"])
        for row in loop:
            e = [parse_value(row[i])[1] for i in (1, 2, 3)]
            if any(x is not None for x in e):
                esd_by_label[row[0].strip("'\"")] = np.array(
                    [x if x is not None else 0.0 for x in e], float)
    except Exception:
        pass

    # Disorder tags. Read from the raw loop rather than from gemmi, because the
    # two columns are optional and gemmi's SmallStructure exposes only the group.
    # Without them, two alternative configurations are drawn on top of each other
    # and counted as one impossibly crowded site.
    disorder_by_label: dict[str, tuple[str, str]] = {}
    for assembly_tag, group_tag in (
            ("_atom_site_disorder_assembly", "_atom_site_disorder_group"),
            ("_atom_site_disorder_assembly", "_atom_site_disorder_group_")):
        try:
            loop = blk.find(["_atom_site_label", assembly_tag, group_tag])
            for row in loop:
                label = row[0].strip("'\"")
                disorder_by_label[label] = (
                    disorder_mod.normalise_assembly(row[1]),
                    disorder_mod.normalise_group(row[2]))
            if disorder_by_label:
                break
        except Exception:
            continue
    if not disorder_by_label:
        # a file may give the group without naming an assembly
        try:
            loop = blk.find(["_atom_site_label",
                             "_atom_site_disorder_group"])
            for row in loop:
                label = row[0].strip("'\"")
                disorder_by_label[label] = (
                    "", disorder_mod.normalise_group(row[1]))
        except Exception:
            pass

    sites: list[Site] = []
    for s in st.sites:
        frac = np.array([s.fract.x, s.fract.y, s.fract.z], float)
        if not np.all(np.isfinite(frac)):
            continue
        sym = elements.normalise(s.type_symbol or s.label)
        site = Site(
            label=s.label,
            element=sym,
            frac=frac,
            occupancy=float(s.occ) if s.occ else 1.0,
            u_iso=float(s.u_iso) if getattr(s, "u_iso", None) else None,
            frac_esd=esd_by_label.get(s.label),
        )
        assembly, group = disorder_by_label.get(s.label, ("", ""))
        site.disorder_assembly = assembly
        site.disorder_group = group
        site.ox, site.ox_source = _oxidation_from_symbol(s.type_symbol, sym)
        sites.append(site)
    return sites


# A charge on a CIF type symbol is a suffix on the bare element: 'Bi3+',
# 'Bi+3', 'O2-', 'O-'. It is ANCHORED, and that matters. A loose search finds
# '1+' inside the site label 'Bi1+3' -- which means "site Bi1, charge 3+" --
# and reads the charge as +1. That silently selects the wrong bond-valence
# parameter and, worse, marks the site as having a stated charge so that the
# self-consistency check never runs on it.
_CHARGE_TRAILING = re.compile(r"^[A-Za-z]{1,2}(\d*)([+-])$")   # Bi3+, Bi+
_CHARGE_LEADING = re.compile(r"^[A-Za-z]{1,2}([+-])(\d+)$")    # Bi+3


def _oxidation_from_symbol(type_symbol, element) -> tuple[int | None, str]:
    """Read a charge out of a CIF type symbol such as ``Bi3+`` or ``O2-``.

    Returns ``(charge, "cif")`` only for an unambiguous, fully anchored symbol.
    Anything else falls through, so that a site whose charge is not actually
    stated is resolved by bond valence rather than by a misreading.
    """
    if type_symbol:
        text = str(type_symbol).strip().strip("'\"")
        m = _CHARGE_TRAILING.match(text)
        if m:
            mag = int(m.group(1)) if m.group(1) else 1
            return (mag if m.group(2) == "+" else -mag), "cif"
        m = _CHARGE_LEADING.match(text)
        if m:
            mag = int(m.group(2))
            return (mag if m.group(1) == "+" else -mag), "cif"
    common = elements.COMMON_OX.get(element)
    return (common, "common") if common is not None else (None, "unset")


def _already_there(frac, existing, cell: Cell, tolerance: float = 0.05) -> bool:
    """Is this fractional position already in the list, allowing for wrapping?

    ``tolerance`` is in angstrom. 0.05 A is far below any real bond and far
    above any rounding in a CIF, and it is the same threshold the bond builder
    uses to recognise a repeated image.
    """
    if not existing:
        return False
    delta = np.asarray(existing, float) - np.asarray(frac, float)
    delta -= np.round(delta)                      # minimum image
    cart = delta @ cell.orth.T
    return bool(np.any(np.linalg.norm(cart, axis=1) < tolerance))


def symmetry_operations(st, blk):
    """The operations to expand with, and a sentence saying where they came from.

    The file's own ``_symmetry_equiv_pos_as_xyz`` loop takes precedence, because
    that is what the file means whatever its space-group symbol happens to say.
    Failing that, the operations of the named group; failing that, P1.
    """
    from .readers import parse_symop

    for tag in ("_symmetry_equiv_pos_as_xyz",
                "_space_group_symop_operation_xyz"):
        try:
            values = list(blk.find_values(tag))
        except Exception:
            continue
        operations = []
        for raw in values:
            parsed = parse_symop(str(raw).strip().strip("'\""))
            if parsed is not None:
                operations.append(parsed)
        if operations:
            # Any length, including a single 'x, y, z'. That is not a degenerate
            # case to be second-guessed: a file listing one operation is saying
            # its coordinates are the whole cell, which is what FACET itself
            # writes and what every P1 file means. Requiring more than one
            # operation here made such a file fall through to its declared space
            # group and be expanded a second time, quadrupling its atoms.
            note = f"{len(operations)} operations from the file"
            declared = _declared_operation_count(st, blk)
            if declared and declared > len(operations):
                note += (f"; the declared space group has {declared}, so the "
                         "file's own list is shorter than its symbol implies")
            return operations, note

    # No operation loop, so the named group has to supply them. Looked up by
    # number first and by symbol second: a number is unambiguous, while a symbol
    # may be a non-standard setting, misspelled, or spaced in a way no lookup
    # recognises.
    import gemmi

    group = None
    number = _int(blk, "_symmetry_Int_Tables_number") or \
        _int(blk, "_space_group_IT_number")
    if number:
        group = gemmi.find_spacegroup_by_number(int(number))
    if group is None:
        symbol = (_text(blk, "_symmetry_space_group_name_H-M")
                  or _text(blk, "_space_group_name_H-M_alt")
                  or getattr(st, "spacegroup_hm", None))
        for candidate in _symbol_variants(symbol):
            group = gemmi.find_spacegroup_by_name(candidate)
            if group is not None:
                break
    if group is not None:
        operations = []
        for operation in group.operations():
            parsed = parse_symop(operation.triplet())
            if parsed is not None:
                operations.append(parsed)
        if operations:
            return operations, f"{len(operations)} operations of {group.hm}"

    return [(np.eye(3), np.zeros(3))], "no symmetry given, treated as P1"


def _declared_operation_count(st, blk) -> int | None:
    """How many operations the file's space-group symbol implies, if any.

    Used only to notice that a file's operation loop is shorter than its symbol,
    which is worth saying out loud: one of the two is wrong, and FACET follows
    the loop.
    """
    import gemmi

    number = _int(blk, "_symmetry_Int_Tables_number") or \
        _int(blk, "_space_group_IT_number")
    group = gemmi.find_spacegroup_by_number(int(number)) if number else None
    if group is None:
        symbol = (_text(blk, "_symmetry_space_group_name_H-M")
                  or _text(blk, "_space_group_name_H-M_alt"))
        for candidate in _symbol_variants(symbol):
            group = gemmi.find_spacegroup_by_name(candidate)
            if group is not None:
                break
    return len(group.operations()) if group is not None else None


def _symbol_variants(symbol):
    """The spellings of a space-group symbol worth trying, in order.

    CIF files write the same group as ``Fm-3m``, ``F m -3 m``, ``F M -3 M`` and
    occasionally ``Fm3m``. A lookup that only accepts one of those silently
    falls back to P1, and a structure expanded as P1 is not wrong in any way a
    reader would notice -- it simply has a quarter or a twelfth of its atoms.
    """
    if not symbol:
        return []
    text = str(symbol).strip().strip("'\"")
    if not text:
        return []
    out = [text, text.replace(" ", ""), " ".join(text.split())]
    # a trailing setting such as ':1' or ':H'
    if ":" in text:
        out.append(text.split(":")[0].strip())
    seen = []
    for candidate in out:
        if candidate and candidate not in seen:
            seen.append(candidate)
    return seen


def _expand(st, cell: Cell, sites: list[Site], blk=None) -> list[Atom]:
    """Apply the symmetry operations to fill the unit cell.

    FACET applies them itself rather than taking gemmi's
    ``get_all_unit_cell_sites()``, and the reason is a structure in this very
    collection. gemmi's small-structure expansion merges positions that are
    close in *fractional* coordinates; in a small cell that discards real atoms.
    For the ICDD entry for Bi0.92Si0.08O1.54 -- a = 5.542 A, Fm-3m, oxygen on the
    32-fold (0.266, 0.266, 0.266) -- it returns 8 oxygen atoms where the file,
    the Wyckoff letter and the 192 operations all say 32, because the F-centred
    images land 0.032 in fractional coordinates away. That is 0.18 A, which is a
    real separation. Three quarters of the oxygen, and of the anion charge, went
    missing.

    De-duplicating by distance in angstrom instead cannot make that mistake at
    any cell size: 0.05 A is below every real interatomic distance whatever the
    cell, while 0.03 in fractional coordinates is 0.18 A in this cell and 0.6 A
    in a 20 A one.
    """
    atoms: list[Atom] = []
    accepted: dict[int, list[np.ndarray]] = {}

    if blk is not None:
        operations, provenance = symmetry_operations(st, blk)
    else:
        operations, provenance = [(np.eye(3), np.zeros(3))], "no symmetry"

    for index, site in enumerate(sites):
        for rotation, translation in operations:
            frac = (rotation @ site.frac + translation) % 1.0
            # De-duplicate positions that coincide after wrapping, which is what
            # happens for an atom on a special position. Compared by
            # minimum-image distance in angstrom, not by a rounded key: a
            # coordinate of 0.99999 and one of 0.0 are the same atom but round to
            # different keys, and the pair would survive as two atoms a hair
            # apart -- inflating the multiplicity, the structure factor and the
            # neighbour list.
            if _already_there(frac, accepted.setdefault(index, []), cell):
                continue
            accepted[index].append(frac)
            atoms.append(Atom(
                element=site.element,
                frac=frac,
                cart=cell.to_cartesian(frac),
                site_index=index,
                label=site.label,
                occupancy=float(site.occupancy),
            ))

    # multiplicity follows from how many positions each site generated
    counts: dict[int, int] = {}
    for a in atoms:
        counts[a.site_index] = counts.get(a.site_index, 0) + 1
    for i, site in enumerate(sites):
        site.multiplicity = counts.get(i, 0)
    return atoms, provenance


def _annotate_symmetry(struct: Structure) -> None:
    """Cross-check the space group with spglib and attach Wyckoff information.

    Disagreement between the stated symbol and what the coordinates actually
    show is recorded as a note, never silently corrected.
    """
    try:
        import spglib
    except Exception:
        struct.notes.append("spglib unavailable: no symmetry cross-check")
        return

    lattice = struct.cell.orth.T
    positions = np.array([a.frac for a in struct.atoms], float)
    if positions.size == 0:
        return
    numbers = [elements.info(a.element).z or 0 for a in struct.atoms]

    try:
        ds = spglib.get_symmetry_dataset((lattice, positions, numbers),
                                         symprec=1e-3)
    except Exception as exc:
        struct.notes.append(f"spglib failed: {exc}")
        return
    if ds is None:
        struct.notes.append("spglib could not determine the symmetry")
        return

    number = int(getattr(ds, "number", 0) or 0)
    found = str(getattr(ds, "international", "") or "")
    struct.spacegroup_number = number or None

    # Compare by International Tables number, not by string. 'P 1 21/c 1',
    # 'P2_1/c' and 'P 21/c' are the same group written three ways, and a symbol
    # comparison would raise an alarm on almost every well-formed CIF.
    # spglib is given element identities only, so a partially occupied or
    # split site looks like a full one to it and can legitimately break a
    # centring. Say so rather than implying the file is wrong.
    groups = {site.disorder_group for site in struct.sites
              if site.disorder_group}
    if len(groups) > 1:
        struct.notes.append(
            f"the file declares {len(groups)} disorder groups "
            f"({', '.join(sorted(groups))}). They are alternatives: only one "
            "exists in any unit cell, and drawing them together puts atoms "
            "within a fraction of an angstrom of each other. Choose a "
            "configuration under Disorder.")
    disordered = any(a.occupancy < 0.999 for a in struct.atoms)
    because = (" (site occupancies are not used by the symmetry search; this "
               "structure has partially occupied sites)") if disordered else ""

    stated_number = _spacegroup_number_of(struct.spacegroup_hm)
    if stated_number and number and stated_number != number:
        struct.notes.append(
            f"stated space group {struct.spacegroup_hm!r} (No. {stated_number}) "
            f"but the coordinates give {found!r} (No. {number}) at symprec "
            f"1e-3{because}")
    elif struct.spacegroup_hm and stated_number is None and number:
        struct.notes.append(
            f"space group symbol {struct.spacegroup_hm!r} not recognised; "
            f"the coordinates give {found!r} (No. {number})")

    # spglib returns numpy arrays here; `or []` would evaluate their truth
    def _listify(attr):
        v = getattr(ds, attr, None)
        return [] if v is None else list(v)

    wyckoffs = _listify("wyckoffs")
    equiv = _listify("equivalent_atoms")
    symbols = _listify("site_symmetry_symbols")
    for ai, atom in enumerate(struct.atoms):
        if ai >= len(wyckoffs):
            break
        site = struct.sites[atom.site_index]
        if site.wyckoff is None:
            site.wyckoff = wyckoffs[ai]
            if ai < len(symbols):
                site.site_symmetry = symbols[ai].strip()

    # spglib finding fewer distinct sites than the CIF declares means the CIF
    # lists symmetry-equivalent atoms separately -- worth knowing before the
    # site table is read as a list of independent environments.
    if equiv:
        n_distinct = len(set(equiv))
        if n_distinct < struct.n_sites:
            struct.notes.append(
                f"the CIF declares {struct.n_sites} sites but spglib finds "
                f"{n_distinct} symmetry-distinct positions")
