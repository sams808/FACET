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

    atoms = _expand(st, cell, sites)

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
    if want is not None:
        for blk in doc:
            if blk.name == want:
                return blk
        raise KeyError(f"no block named {want!r}")
    for blk in doc:
        if blk.find_value("_cell_length_a"):
            return blk
    return doc.sole_block()


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


def _expand(st, cell: Cell, sites: list[Site]) -> list[Atom]:
    """Apply the symmetry operations to fill the unit cell.

    gemmi does the expansion; the work here is mapping each produced position
    back to the Site it came from, which gemmi reports only by label.
    """
    by_label = {s.label: i for i, s in enumerate(sites)}
    atoms: list[Atom] = []
    seen: set[tuple[int, int, int, int]] = set()

    for s in st.get_all_unit_cell_sites():
        idx = by_label.get(s.label)
        if idx is None:
            continue
        frac = np.array([s.fract.x, s.fract.y, s.fract.z], float) % 1.0
        # de-duplicate positions that coincide after wrapping, which happens for
        # atoms on special positions
        key = (idx, *(int(round(x * 1e4)) for x in frac))
        if key in seen:
            continue
        seen.add(key)
        atoms.append(Atom(
            element=sites[idx].element,
            frac=frac,
            cart=cell.to_cartesian(frac),
            site_index=idx,
            label=s.label,
            occupancy=float(s.occ) if s.occ else 1.0,
        ))

    # multiplicity follows from how many positions each site generated
    counts: dict[int, int] = {}
    for a in atoms:
        counts[a.site_index] = counts.get(a.site_index, 0) + 1
    for i, site in enumerate(sites):
        site.multiplicity = counts.get(i, 0)
    return atoms


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
    disordered = any(a.occupancy < 0.999 for a in struct.atoms)
    because = (" -- but site occupancies are not passed to the symmetry search, "
               "and this structure has partially occupied sites, which alone "
               "can lower the apparent symmetry") if disordered else ""

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
