"""The shells of one site: its contacts in the order a plot includes them,
with the running count and the running bond-valence sum after each.

The coordination number against threshold is a staircase; the bond-valence
sum over the same contacts is the integral that staircase is the derivative
of. Both are re-sums of the contacts the analysis already found: no neighbour
search, no new parameter. A contact adds its own occupancy-weighted valence,
and the curve shows how much of the sum each shell carries.
"""
from __future__ import annotations

from dataclasses import dataclass

from .coordination import SiteResult

AXIS_DISTANCE = "distance"
AXIS_VALENCE = "valence"


@dataclass(frozen=True)
class Shell:
    """One contact in the order the plot includes it, with the totals after it."""

    index: int            # into result.contacts
    label: str
    element: str
    distance: float
    valence: float
    occupancy: float
    cn_after: int         # contacts counted once this one is included
    sum_after: float      # bond-valence sum, occupancy-weighted, once included


def shell_curve(result: SiteResult | None, axis: str = AXIS_DISTANCE) -> list[Shell]:
    """The contacts of a site in the order the plot includes them, with the
    running count and sum after each one.

    Against distance the contacts are included from the nearest outwards.
    Against valence they are included from the strongest down, which is the
    same order only when every contact uses the same parameter; with two
    anions (oxide and fluoride, say) the two orders differ, and each axis
    shows its own.

    Contacts with no bond-valence parameter are left out: they have no
    valence to add, and the staircase leaves them out too.
    """
    if result is None:
        return []
    rows = [(i, c) for i, c in enumerate(result.contacts) if c.has_valence]
    if axis == AXIS_VALENCE:
        rows.sort(key=lambda t: -t[1].valence)
    else:
        rows.sort(key=lambda t: t[1].distance)
    out: list[Shell] = []
    total = 0.0
    for k, (i, c) in enumerate(rows, start=1):
        total += float(c.valence) * float(c.occupancy)
        out.append(Shell(index=i, label=c.label, element=c.element,
                         distance=float(c.distance), valence=float(c.valence),
                         occupancy=float(c.occupancy), cn_after=k,
                         sum_after=total))
    return out


def at_threshold(result: SiteResult | None,
                 v_bond: float) -> tuple[int, float, Shell | None]:
    """(CN, bond-valence sum, the next contact outside) at a threshold.

    Bonded means valence above the threshold, as the analysis counts it. The
    next contact is the strongest one left outside, with what it would add.
    """
    if result is None:
        return 0, float("nan"), None
    shells = shell_curve(result, AXIS_VALENCE)
    inside = [s for s in shells if s.valence > v_bond]
    outside = [s for s in shells if s.valence <= v_bond]
    cn = len(inside)
    total = inside[-1].sum_after if inside else 0.0
    return cn, total, (outside[0] if outside else None)
