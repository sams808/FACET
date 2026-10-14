"""One atom of an MD frame as a crystallographic site.

The crystal workspace analyses a site from its contact list: coordination
number, bond-valence sum, phi, the polyhedron's shape, the staircase of CN
against threshold. An atom of a frame has the same contact list in the
valence table the Model workspace already holds (``bulk.ValenceTable``:
every counter-ion contact above the listing threshold, with its distance,
vector, valence and parameter). This module turns one row of that table
into a ``SiteResult``, so the Site tab, the cutoff explorer and the
reclassification on a drag all work for an atom exactly as for a site.
"""
from __future__ import annotations

import numpy as np

from . import bv
from .coordination import (
    ContactRow,
    SiteResult,
    _fill_geometry,
    _fill_valence_quantities,
    _note_empty_coordination,
    plateaus,
)
from .md_model import Frame


def site_result_for_atom(frame: Frame, table, row: int, *,
                         v_bond: float = bv.V_BOND_DEFAULT,
                         v_list: float | None = None,
                         k: int | None = None) -> SiteResult:
    """The ``SiteResult`` of frame row ``row`` from the valence table.

    The contacts are the table's listed contacts of that row (counter-ions
    above ``v_list``), each with the parameter the table used for the pair,
    so the distance axis of the explorer and the R0 line of the Site tab
    read the same parameter the analysis did.
    """
    n = len(frame.elements)
    if not 0 <= row < n:
        raise ValueError(f"atom row {row} is not in a frame of {n} atoms")
    v_list = float(getattr(table, "v_list_vu", 0.0) if v_list is None else v_list)
    elements = np.asarray(frame.elements)
    ids = np.asarray(frame.atom_id)
    ox = int(np.asarray(table.ox)[row])
    centre_is_anion = bool(np.asarray(table.is_anion)[row])
    params = getattr(table, "params", None)

    rows: list[ContactRow] = []
    used_estimate = False
    missing = set()
    nbr = np.asarray(table.nbr[row])
    for j in range(len(nbr)):
        other = int(nbr[j])
        if other < 0:
            continue
        d = float(table.d_ang[row, j])
        v = float(table.v_vu[row, j])
        other_el = str(elements[other])
        other_ox = int(np.asarray(table.ox)[other])
        if centre_is_anion:
            key = (other_el, other_ox, str(elements[row]))
        else:
            key = (str(elements[row]), ox, other_el)
        param = None
        if params is not None:
            param = params.param.get(key)
        has_valence = v > 0 or param is not None
        if param is not None and not param.fitted:
            used_estimate = True
        if not has_valence:
            missing.add(f"{key[0]}{key[1]:+d}-{key[2]}")
        rows.append(ContactRow(
            label=f"{other_el}{int(ids[other])}", element=other_el,
            site_index=other, atom_index=other, distance=d,
            vector=np.asarray(table.vec_ang[row, j], float),
            image=tuple(int(x) for x in np.asarray(table.image[row, j])),
            occupancy=1.0, valence=(v if has_valence else None),
            param=param))
    rows.sort(key=lambda c: c.distance)

    result = SiteResult(
        site_index=row,
        label=f"{elements[row]}{int(ids[row])}",
        element=str(elements[row]),
        ox=ox,
        ox_source="model oxidation states",
        wyckoff=None,
        site_symmetry=None,
        multiplicity=None,
        contacts=[c for c in rows if not c.has_valence or c.valence > v_list],
        v_bond=float(v_bond),
        v_list=v_list,
        uses_estimated_params=used_estimate,
    )
    if k is not None:
        result.notes.append(f"frame {k}, atom row {row}, id {int(ids[row])}")
    if missing:
        result.notes.append(
            "no bond-valence parameter for " + ", ".join(sorted(missing))
            + "; those contacts are listed without a valence")
    if used_estimate:
        result.notes.append(
            "some parameters are estimated from the O'Keeffe-Brese expression "
            "rather than fitted")
    _fill_valence_quantities(result)
    _fill_geometry(result)
    _note_empty_coordination(result)
    result.plateaus = plateaus(result)
    return result
