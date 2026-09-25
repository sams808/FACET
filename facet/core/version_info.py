"""Provenance lines stamped into every export.

A table of coordination numbers without the threshold that produced them is
not reproducible, and this application exists to make that point. Every file
FACET writes therefore carries the parameter set, both thresholds, and whether
any parameter was estimated rather than fitted.
"""
from __future__ import annotations

from datetime import datetime


def provenance_lines(params=None, v_bond: float | None = None,
                     v_list: float | None = None,
                     estimated: bool | None = None) -> list[str]:
    from ..version import NAME, __version__
    from . import bv as bv_mod

    params = params or bv_mod.DEFAULT
    v_bond = bv_mod.V_BOND_DEFAULT if v_bond is None else v_bond
    v_list = bv_mod.V_LIST_DEFAULT if v_list is None else v_list

    lines = [
        f"{NAME} {__version__} - {datetime.now():%Y-%m-%d %H:%M}",
        f"Bond-valence parameters: {params.name}",
        f"  {params.source}",
        f"  b = {params.b} A; v = exp((R0 - d)/b)",
        f"Bond threshold:      {v_bond:g} v.u.",
        f"Tabulation threshold:{v_list:g} v.u.",
        "Coordination numbers are reported at the bond threshold above. "
        "They change with it; the plateau columns give the range over which "
        "each one holds.",
    ]
    if estimated:
        lines.append(
            "Some R0 values are ESTIMATED from the O'Keeffe-Brese "
            "electronegativity expression rather than fitted.")
    return lines
