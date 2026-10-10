"""Getting results and structures out of FACET.

Phase 4 of the roadmap. Until now an analysis could only leave as a PNG, which
made FACET a dead end; these are the formats the rest of the work actually
uses.

Every table export carries the parameters that produced it -- the parameter
set and its citation, both thresholds, and whether any R0 was estimated. A
table of coordination numbers without the threshold that produced them is not
reproducible, and this application exists to make that point.

Writers are pure text where possible, so they need no optional dependency.
XLSX is the exception and degrades to CSV when openpyxl is absent.
"""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from . import elements
from .structure import Structure
from .version_info import provenance_lines


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], path: str | Path,
              provenance: list[str] | None = None) -> Path:
    """Write rows to CSV, with the provenance as leading comment lines.

    Comment lines start with '#', which every spreadsheet and every csv reader
    with a `comment` option skips, and which a person reading the file sees
    first.
    """
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        for line in (provenance or provenance_lines()):
            handle.write(f"# {line}\n")
        writer = csv.DictWriter(handle, fieldnames=fields,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in fields})
    return path


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return f"{value:.6g}"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    return value


def write_xlsx(sheets: dict[str, list[dict]], path: str | Path,
               provenance: list[str] | None = None) -> Path:
    """Write several tables to one workbook.

    Falls back to a CSV per sheet if openpyxl is not installed, rather than
    failing: the data matters more than the container.
    """
    path = Path(path)
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
    except ImportError:
        written = []
        for name, rows in sheets.items():
            target = path.with_name(f"{path.stem}_{name}.csv")
            write_csv(rows, target, provenance)
            written.append(target)
        return written[0] if written else path

    book = Workbook()
    book.remove(book.active)

    notes = book.create_sheet("provenance")
    notes["A1"] = "How this file was produced"
    notes["A1"].font = Font(bold=True)
    for i, line in enumerate(provenance or provenance_lines(), start=3):
        notes.cell(row=i, column=1, value=line)
    notes.column_dimensions["A"].width = 110

    for name, rows in sheets.items():
        sheet = book.create_sheet(name[:31])
        if not rows:
            continue
        fields = list(rows[0].keys())
        for col, field in enumerate(fields, start=1):
            cell = sheet.cell(row=1, column=col, value=field)
            cell.font = Font(bold=True)
        for r, row in enumerate(rows, start=2):
            for col, field in enumerate(fields, start=1):
                value = row.get(field)
                if isinstance(value, float) and np.isnan(value):
                    value = None
                elif isinstance(value, (np.floating, np.integer)):
                    value = value.item()
                sheet.cell(row=r, column=col, value=value)
        sheet.freeze_panes = "A2"
    book.save(path)
    return path


# ---------------------------------------------------------------------------
# structures out
# ---------------------------------------------------------------------------

def write_cif(structure: Structure, path: str | Path,
              results=None, provenance: list[str] | None = None) -> Path:
    """Write a CIF, optionally with the analysis as comments.

    The structure is written as the asymmetric unit plus the symmetry, not as
    the expanded cell, so the file stays equivalent to what was read. Any
    analysis goes in comments: a coordination number is not a CIF data item,
    and inventing a tag for it would make a file other software misreads.
    """
    path = Path(path)
    cell = structure.cell
    out = io.StringIO()

    out.write(f"# Written by FACET on {datetime.now():%Y-%m-%d %H:%M}\n")
    for line in (provenance or provenance_lines()):
        out.write(f"# {line}\n")
    out.write("\n")

    name = (structure.name or "structure").replace(" ", "_")
    out.write(f"data_{name}\n")
    if structure.formula:
        out.write(f"_chemical_formula_sum   '{structure.formula}'\n")
    out.write(f"_cell_length_a          {cell.a:.6f}\n")
    out.write(f"_cell_length_b          {cell.b:.6f}\n")
    out.write(f"_cell_length_c          {cell.c:.6f}\n")
    out.write(f"_cell_angle_alpha       {cell.alpha:.4f}\n")
    out.write(f"_cell_angle_beta        {cell.beta:.4f}\n")
    out.write(f"_cell_angle_gamma       {cell.gamma:.4f}\n")
    out.write(f"_cell_volume            {cell.volume:.4f}\n")
    # The coordinates below are the expanded cell, so the file IS P1 and must say
    # so. Declaring the original space group beside a P1 operation list makes a
    # file that contradicts itself: a reader that believes the symbol expands the
    # already-expanded coordinates again and ends up with four times the atoms.
    # The original symbol is kept as a comment, where it informs without
    # instructing.
    if structure.spacegroup_hm:
        out.write(f"# the source structure was {structure.spacegroup_hm}"
                  + (f" (No. {structure.spacegroup_number})"
                     if structure.spacegroup_number else "")
                  + "; the coordinates below are its full cell\n")
    out.write("_space_group_name_H-M_alt  'P 1'\n")
    out.write("_space_group_IT_number  1\n")
    out.write("\n")

    # Expanded coordinates with P1 symmetry. Writing the asymmetric unit
    # without the exact operator list it was expanded with risks a file that
    # regenerates a different structure; P1 always round-trips.
    out.write("loop_\n"
              "  _symmetry_equiv_pos_as_xyz\n"
              "  'x, y, z'\n\n")
    out.write("loop_\n"
              "  _atom_site_label\n"
              "  _atom_site_type_symbol\n"
              "  _atom_site_fract_x\n"
              "  _atom_site_fract_y\n"
              "  _atom_site_fract_z\n"
              "  _atom_site_occupancy\n")
    for i, atom in enumerate(structure.atoms, start=1):
        out.write(f"  {atom.label}_{i:<4d} {atom.element:<4s} "
                  f"{atom.frac[0]:10.6f} {atom.frac[1]:10.6f} "
                  f"{atom.frac[2]:10.6f} {atom.occupancy:8.4f}\n")

    if results:
        out.write("\n# Coordination analysis\n")
        for r in results:
            out.write(f"#   {r.summary()}\n")
            for note in r.notes:
                out.write(f"#     note: {note}\n")

    path.write_text(out.getvalue(), encoding="utf-8")
    return path


def write_poscar(structure: Structure, path: str | Path,
                 direct: bool = True) -> Path:
    """VASP POSCAR. Atoms are grouped by element, as the format requires."""
    path = Path(path)
    orth = structure.cell.orth
    order: dict[str, list] = {}
    for atom in structure.atoms:
        order.setdefault(atom.element, []).append(atom)

    out = io.StringIO()
    out.write(f"{structure.name or 'structure'} (written by FACET)\n")
    out.write("1.0\n")
    for i in range(3):
        v = orth[:, i]
        out.write(f"  {v[0]:18.12f} {v[1]:18.12f} {v[2]:18.12f}\n")
    out.write("  " + "  ".join(order) + "\n")
    out.write("  " + "  ".join(str(len(v)) for v in order.values()) + "\n")
    out.write("Direct\n" if direct else "Cartesian\n")
    for atoms in order.values():
        for atom in atoms:
            p = atom.frac if direct else atom.cart
            out.write(f"  {p[0]:18.12f} {p[1]:18.12f} {p[2]:18.12f}\n")
    path.write_text(out.getvalue(), encoding="utf-8")
    return path


def write_xyz(structure: Structure, path: str | Path,
              cell_range: tuple[int, int, int] = (1, 1, 1)) -> Path:
    """Extended XYZ, with the lattice in the comment line.

    The `Lattice=` convention is what ASE, OVITO and most other readers look
    for, so the cell survives even though XYZ has no place for it.
    """
    path = Path(path)
    orth = structure.cell.orth
    nx, ny, nz = (max(1, int(n)) for n in cell_range)

    atoms = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                shift = orth @ np.array([i, j, k], float)
                for atom in structure.atoms:
                    atoms.append((atom.element, atom.cart + shift))

    lattice = " ".join(f"{orth[r, c] * (nx, ny, nz)[c]:.8f}"
                       for c in range(3) for r in range(3))
    out = io.StringIO()
    out.write(f"{len(atoms)}\n")
    out.write(f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 '
              f'name="{structure.name or "structure"}"\n')
    for element, p in atoms:
        out.write(f"{element:<4s} {p[0]:14.8f} {p[1]:14.8f} {p[2]:14.8f}\n")
    path.write_text(out.getvalue(), encoding="utf-8")
    return path


def write_vesta(structure: Structure, path: str | Path,
                theme=None, v_bond: float | None = None) -> Path:
    """A VESTA `.vesta` file, so a collaborator can open the same view there.

    Written in P1 with explicit coordinates, which is what makes it open
    identically rather than depending on VESTA's own symmetry expansion. Bond
    search rules are emitted per cation-anion pair at the distance that
    corresponds to the current bond-valence threshold -- VESTA has no notion of
    a valence cutoff, so the equivalent distance is the honest translation.
    """
    from . import bv as bv_mod

    path = Path(path)
    cell = structure.cell
    out = io.StringIO()
    out.write("#VESTA_FORMAT_VERSION 3.5.0\n\n\nCRYSTAL\n\n")
    out.write(f"TITLE\n{structure.name or 'structure'}\n\n")
    out.write("GROUP\n1 1 P 1\n\n")
    out.write("SYMOP\n")
    out.write(" 0.000000  0.000000  0.000000  1  0  0   0  1  0   0  0  1   1\n")
    out.write(" -1.0 -1.0 -1.0  0 0 0  0 0 0  0 0 0\n\n")
    out.write("TRANM 0\n")
    out.write(" 0.000000  0.000000  0.000000  1  0  0   0  1  0   0  0  1\n")
    out.write("LTRANSL\n -1\n 0.000000  0.000000  0.000000  "
              "0.000000  0.000000  0.000000\n")
    out.write("LORIENT\n -1   0   0   0   0\n"
              " 1.000000  0.000000  0.000000  1.000000  0.000000  0.000000\n"
              " 0.000000  0.000000  1.000000  0.000000  0.000000  1.000000\n")
    out.write("LMATRIX\n"
              " 1.000000  0.000000  0.000000  0.000000\n"
              " 0.000000  1.000000  0.000000  0.000000\n"
              " 0.000000  0.000000  1.000000  0.000000\n"
              " 0.000000  0.000000  0.000000  1.000000\n"
              " 0.000000  0.000000  0.000000\n")
    out.write("CELLP\n")
    out.write(f" {cell.a:.6f} {cell.b:.6f} {cell.c:.6f} "
              f"{cell.alpha:.6f} {cell.beta:.6f} {cell.gamma:.6f}\n")
    out.write("  0.000000   0.000000   0.000000   "
              "0.000000   0.000000   0.000000\n")

    out.write("STRUC\n")
    for i, atom in enumerate(structure.atoms, start=1):
        out.write(f"  {i} {atom.element} {atom.label}{i} {atom.occupancy:.4f} "
                  f"{atom.frac[0]:.6f} {atom.frac[1]:.6f} {atom.frac[2]:.6f} "
                  f"1a       1\n")
        out.write("                            0.000000   0.000000   "
                  "0.000000  0.00\n")
    out.write("  0 0 0 0 0 0 0\n")

    out.write("THERI 0\n")
    for i, atom in enumerate(structure.atoms, start=1):
        out.write(f"  {i} {atom.label}{i}  0.000000\n")
    out.write("  0 0 0\n")

    out.write("SHAPE\n  0       0       0       0   0.000000  0   192   192   192   192\n")
    out.write("BOUND\n       0        1         0        1         0        1\n"
              "  0   0   0   0  0\n")

    # bond rules, translated from the valence threshold to a distance
    params = bv_mod.DEFAULT
    v = v_bond if v_bond is not None else bv_mod.V_BOND_DEFAULT
    out.write("SBOND\n")
    n = 0
    seen = set()
    for ci in structure.cation_sites:
        for ai in structure.anion_sites:
            cat = structure.sites[ci]
            an = structure.sites[ai]
            if (cat.element, an.element) in seen:
                continue
            seen.add((cat.element, an.element))
            p = params.get(cat.element, cat.ox, an.element)
            if p is None:
                continue
            n += 1
            out.write(f"  {n} {cat.element:<3s} {an.element:<3s}    0.00000    "
                      f"{p.distance_for(v):.5f}  0  1  1  0  1  0.250  2.000 "
                      f"127 127 127\n")
    out.write("  0 0 0 0\n")

    out.write("SITET\n")
    for i, atom in enumerate(structure.atoms, start=1):
        info = elements.info(atom.element)
        colour = (theme.element_color(atom.element) if theme is not None
                  else info.color)
        rgb = tuple(int(round(255 * c)) for c in colour)
        out.write(f"  {i} {atom.label}{i}  {info.display_radius:.4f} "
                  f"{rgb[0]} {rgb[1]} {rgb[2]} "
                  f"{rgb[0]} {rgb[1]} {rgb[2]} 204 0\n")
    out.write("  0 0 0 0 0 0\n")

    out.write("ATOMT\n")
    for i, element in enumerate(structure.elements_present, start=1):
        info = elements.info(element)
        colour = (theme.element_color(element) if theme is not None
                  else info.color)
        rgb = tuple(int(round(255 * c)) for c in colour)
        out.write(f"  {i} {element:<3s} {info.display_radius:.4f} "
                  f"{rgb[0]} {rgb[1]} {rgb[2]} "
                  f"{rgb[0]} {rgb[1]} {rgb[2]} 204\n")
    out.write("  0 0 0 0 0 0\n")

    out.write("SCENE\n"
              " 1.000000  0.000000  0.000000  0.000000\n"
              " 0.000000  1.000000  0.000000  0.000000\n"
              " 0.000000  0.000000  1.000000  0.000000\n"
              " 0.000000  0.000000  0.000000  1.000000\n"
              "  0.000   0.000\n  0.000\n  1.000\n")
    path.write_text(out.getvalue(), encoding="utf-8")
    return path


def write_feff(structure: Structure, site_index: int, path: str | Path,
               rmax: float = 6.0, edge: str = "L3",
               r_path: float | None = None,
               paths: bool = True) -> Path:
    """A FEFF input for one absorbing site.

    Distances come from the same neighbour search the coordination analysis
    uses, so the input and the reported shells cannot disagree.

    Three details that decide whether FEFF will run the file at all, each
    learned by running FEFF8L on it.

    *The absorber is the only atom with potential index 0.* An atom of the same
    element that is not the absorber needs a potential of its own -- so a
    bismuth cluster about a bismuth absorber gets ``0 83 Bi`` and ``2 83 Bi``.
    Giving both ipot 0 is refused.

    *``PRINT 0 0 0 0 0 3`` writes the path files.* With ``PRINT 1 0 0 0 0 0``
    FEFF writes ``chi.dat`` and an empty ``files.dat`` and no ``feffNNNN.dat``,
    which is exactly what a fit in Artemis or Larch needs.

    *The cluster is larger than RPATH.* An atom at the edge of the cluster has
    the worst potential in it, so the paths asked for stop short of the
    boundary. ``r_path`` defaults to ``rmax - 1 A``.

    FEFF has no notion of partial occupancy. Where a site is partly occupied the
    atom is written in full -- that is the only thing FEFF can read -- with its
    occupancy as a comment on the line and a statement of the fact in the
    header, rather than passing over it silently.
    """
    from .neighbors import NeighborFinder

    path = Path(path)
    site = structure.sites[site_index]
    finder = NeighborFinder(structure, rmax=rmax)
    contacts = finder.contacts_for_site(site_index)
    if r_path is None:
        r_path = max(rmax - 1.0, 2.0)

    # ipot 0 is the absorber alone. Every element gets a potential, and the
    # absorber's element gets a second one for the atoms of it that are not the
    # absorber -- without which FEFF refuses the file.
    potential: dict[str, int] = {}
    next_ipot = 1
    for element in contacts.elements:
        if element not in potential:
            potential[element] = next_ipot
            next_ipot += 1

    fractional = [(structure.sites[int(contacts.neighbor_site[i])].label,
                   float(contacts.occupancy[i]))
                  for i in range(len(contacts))
                  if float(contacts.occupancy[i]) < 0.999]

    out = io.StringIO()
    out.write(f"* FEFF input written by FACET for {site.label} "
              f"in {structure.name or 'structure'}\n")
    for line in provenance_lines():
        out.write(f"* {line}\n")
    out.write(f"* Cluster radius {rmax:.2f} A, RPATH {r_path:.2f} A. The "
              "cluster is larger than RPATH on purpose: an atom at its edge\n"
              "* has the worst potential in it.\n")
    if fractional:
        out.write("* PARTIAL OCCUPANCY: FEFF has no way to represent it, so "
                  "every atom below is written in full.\n")
        seen = {}
        for label, occupancy in fractional:
            seen[label] = occupancy
        for label, occupancy in sorted(seen.items()):
            out.write(f"*   {label} is {occupancy:.4f} occupied in the file "
                      "and appears here as a whole atom.\n")
    out.write(f"\nTITLE {structure.name or 'structure'} {site.label}\n\n")
    out.write(f"EDGE      {edge}\nS02       1.0\n\n")
    out.write("CONTROL   1 1 1 1 1 1\n")
    # 0 0 0 0 0 3 writes files.dat and feffNNNN.dat, which is what a fit needs
    out.write(f"PRINT     {'0 0 0 0 0 3' if paths else '1 0 0 0 0 0'}\n\n")
    out.write(f"RPATH     {r_path:.2f}\nEXCHANGE  0 0 0\n\n")

    # FEFF reads both blocks below positionally, so nothing may follow the
    # fields it expects -- a trailing word is read as the next number and the
    # file is rejected. Every annotation therefore goes in a comment line above.
    out.write(f"* ipot 0 is the absorber, {site.label}. ")
    same = [e for e in potential if e == site.element]
    if same:
        out.write(f"ipot {potential[site.element]} is the other {site.element} "
                  "atoms,\n*   which need a potential of their own because "
                  "ipot 0 is the absorber alone.\n")
    else:
        out.write("The others are the scatterers.\n")
    out.write("POTENTIALS\n*    ipot   Z  element\n")
    z_absorber = elements.info(site.element).z or 0
    out.write(f"     {0:<5d} {z_absorber:<3d} {site.element}\n")
    for element, ipot in potential.items():
        z = elements.info(element).z or 0
        out.write(f"     {ipot:<5d} {z:<3d} {element}\n")

    out.write("\nATOMS\n*    x          y          z      ipot  label"
              "  distance\n")
    out.write(f"  {0.0:10.5f} {0.0:10.5f} {0.0:10.5f}   0     "
              f"{site.element:<4s} 0.00000\n")
    order = np.argsort(contacts.distance)
    for i in order:
        v = contacts.vector[i]
        element = contacts.elements[i]
        out.write(f"  {v[0]:10.5f} {v[1]:10.5f} {v[2]:10.5f}   "
                  f"{potential[element]}     {element:<4s} "
                  f"{contacts.distance[i]:.5f}\n")
    out.write("END\n")
    path.write_text(out.getvalue(), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------

def save_session(project, path: str | Path, theme=None,
                 camera=None, labels=None, presentation=None) -> Path:
    """Everything needed to reopen the same view.

    File paths rather than structures: a session is a pointer to the user's
    files, not a copy of them, so it stays small and does not go stale in a
    different way from the data.
    """
    path = Path(path)
    data = {
        "facet_session": 1,
        "written": datetime.now().isoformat(timespec="seconds"),
        "v_bond": project.v_bond,
        "v_list": project.v_list,
        "overlay": project.overlay,
        "overlay_spacing": project.overlay_spacing,
        "active": project.active,
        "entries": [_entry_record(e) for e in project.entries],
    }
    if presentation is not None:
        # planes and the slab: presentation state that belongs to the view
        # rather than to any one structure
        data["presentation"] = presentation
    if theme is not None:
        data["theme"] = theme.to_dict()
    if camera is not None:
        data["camera"] = {
            "target": list(map(float, camera.target)),
            "distance": float(camera.distance),
            "orientation": list(map(float, camera.orientation)),
            "fov": float(camera.fov),
            "orthographic": bool(camera.orthographic),
        }
    if labels is not None:
        data["labels"] = {
            "atom": labels.atom.value, "bond": labels.bond.value,
            "atom_scope": labels.atom_scope.value,
            "bond_scope": labels.bond_scope.value,
            "show_axes": labels.show_axes,
            "font_points": labels.font_points,
        }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _entry_record(entry) -> dict:
    """One project entry as the session file holds it. A crystal: its path,
    visibility, selected site, overrides and disorder choice. An MD model
    (``kind`` 'model'): its source, the read options it was read with and
    the request of its setup, never a result (a result is re-measured from
    the file when the request is run again)."""
    if getattr(entry, "kind", "crystal") == "model":
        return {"kind": "model", "path": entry.path,
                "read_options": _plain(entry.read_options or {}),
                "request": _plain(entry.request_spec or {}),
                "visible": entry.visible}
    return {"path": entry.path, "visible": entry.visible,
            "selected_site": entry.selected_site,
            "overrides": entry.overrides.to_dict(),
            "disorder": (entry.disorder.to_dict() if entry.disorder is not None
                         else None)}


def _plain(value):
    """``value`` with every array a list and every mapping key text, so
    json writes it; a type map keyed by integers is written as text keys
    and read back by the MD reader as either."""
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, frozenset):
        return sorted(value)
    return value


def load_session(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
