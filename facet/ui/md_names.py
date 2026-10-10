"""Readable names for MD descriptors, and the question groups they sit in.

The engine names its descriptors tersely ('CN Si (BV)', 'Qn(mSi) Si
(distance)', 'S_NN(Q)', 'R_C(n), king rings'). :func:`display_name` turns
such an id into a label a reader meets in a textbook ('Coordination of Si —
bond-valence cut'); an id no rule covers is returned unchanged, so nothing
is ever hidden behind a translation. The raw id stays beside the label in
the Results tree (muted) and in its tooltip.

:data:`QUESTION_GROUPS` places every analysis under the question it
answers, the same grouping the Setup page uses
(:data:`facet.ui.md_setup.ANALYSES_BY_QUESTION`); :func:`group_of` and
:func:`analysis_title` read it. Nothing here computes or judges a value:
these are names only.
"""
from __future__ import annotations

import re

from .md_setup import ANALYSES_BY_QUESTION

__all__ = ["display_name", "group_of", "analysis_title", "group_titles",
           "ion_text", "kind_badge", "criterion_text", "QUESTION_GROUPS",
           "OTHER_GROUP"]

OTHER_GROUP = "Other"

# (group title, (analysis name, analysis title) ...), the Setup page's order.
QUESTION_GROUPS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = \
    ANALYSES_BY_QUESTION

_TITLES = {name: title for _, members in QUESTION_GROUPS
           for name, title in members}
_GROUPS = {name: group for group, members in QUESTION_GROUPS
           for name, _ in members}


def group_titles() -> list[str]:
    """Every group title, in the Setup page's order, then 'Other'."""
    return [group for group, _ in QUESTION_GROUPS] + [OTHER_GROUP]


def group_of(analysis: str) -> str:
    """The question group of an analysis ('Other' when not placed)."""
    return _GROUPS.get(str(analysis), OTHER_GROUP)


def analysis_title(analysis: str) -> str:
    """The Setup page's title of an analysis (the name itself when it has
    none)."""
    return _TITLES.get(str(analysis), str(analysis))


# ---------------------------------------------------------------------------
# descriptor ids -> readable labels
# ---------------------------------------------------------------------------

# an element symbol, or the '*' a family key puts where one stood
_EL = r"(?:\*|[A-Z][a-z]?)"

_SUPERSCRIPTS = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")


def ion_text(label: str) -> str:
    """'Na1+' -> 'Na⁺', 'O2-' -> 'O²⁻'; text without a charge is kept."""
    match = re.fullmatch(r"([A-Z][a-z]?)(\d*)([+-])", str(label))
    if match is None:
        return str(label)
    symbol, number, sign = match.groups()
    digits = "" if number in ("", "1") else number
    return symbol + (digits + sign).translate(_SUPERSCRIPTS)


def _el(symbol: str) -> str:
    return "each element" if symbol == "*" else symbol


def _pair(a: str, b: str) -> str:
    if a == "*" or b == "*":
        return "each pair"
    return f"{a}–{b}"


_CUTS = (
    (" (BV, distance)", " — bond-valence vs distance"),
    (" (BV)", " — bond-valence cut"),
    (" (distance)", " — distance cut"),
)

_CRITERIA = {"king": "King's criterion", "guttman": "Guttman's criterion",
             "primitive": "primitive rings"}

_RING_STATS = {
    "R_C": "Rings per node R_C(n)",
    "R_N": "Ring finds per node R_N(n)",
    "P_N": "Nodes with a ring of n members P_N(n)",
    "P_max": "Largest-ring fraction P_max(n)",
    "P_min": "Smallest-ring fraction P_min(n)",
}

_BT = {"NN": "number–number", "NC": "number–concentration",
       "CC": "concentration–concentration"}

_RADIATION = ("neutron", "x-ray", "X-ray", "electron")


def criterion_text(text: str) -> str:
    """A ring criterion as the label names it ('king' -> \"King's
    criterion\"); other text is kept."""
    return _CRITERIA.get(str(text).strip().lower(), str(text))


_criterion = criterion_text


def _rules() -> tuple[tuple[re.Pattern, object], ...]:
    E = _EL
    return (
        # --- glass -------------------------------------------------------
        (re.compile(rf"^g ({E})-({E})$"),
         lambda m: f"Pair distribution g(r), {_pair(m[1], m[2])}"),
        (re.compile(rf"^N ({E}) around ({E})$"),
         lambda m: f"Running coordination N(r): {_el(m[1])} around "
                   f"{_el(m[2])}"),
        (re.compile(r"^g\(r\) first minima$"),
         "First minima of the partial g(r) — the distance cutoffs"),
        (re.compile(r"^distance cutoffs$"), "Distance cutoffs used"),
        (re.compile(rf"^mean CN ({E})$"),
         lambda m: f"Mean coordination of {_el(m[1])}"),
        (re.compile(rf"^CN ({E}) BV-distance differences$"),
         lambda m: f"Atoms of {_el(m[1])} whose coordination differs "
                   "between the two cuts"),
        (re.compile(rf"^CN ({E})$"),
         lambda m: f"Coordination of {_el(m[1])}"),
        (re.compile(rf"^({E}) speciation$"),
         lambda m: f"{_el(m[1])} speciation: free, NBO, BO, tricluster"),
        (re.compile(rf"^Qn\(m({E})\) ({E})$"),
         lambda m: f"Qⁿ bridges of {_el(m[2])} to {_el(m[1])}"),
        (re.compile(rf"^Qn ({E})$"),
         lambda m: f"Qⁿ distribution of {_el(m[1])}"),
        (re.compile(r"^connectivity all formers$"),
         "Mean bridges per former polyhedron"),
        (re.compile(rf"^connectivity ({E})$"),
         lambda m: f"Mean bridges per {_el(m[1])} polyhedron"),
        (re.compile(rf"^({E}) linkages$"),
         lambda m: f"Cation pairs linked through {_el(m[1])}"),
        (re.compile(rf"^({E}) environments$"),
         lambda m: f"Cation environments of {_el(m[1])}"),
        (re.compile(rf"^angle ({E})-({E})-({E})$"),
         lambda m: "Bond angle " + "–".join(_el(g) for g in m.groups())
         if "*" not in m.groups() else "Bond angles"),
        (re.compile(rf"^bond length ({E})-({E})$"),
         lambda m: f"Bond length {_pair(m[1], m[2])}"),
        (re.compile(rf"^phi ({E})$"),
         lambda m: f"Threshold-stability index φ of {_el(m[1])}"),
        (re.compile(rf"^plateau width ({E})$"),
         lambda m: f"Valence plateau width of {_el(m[1])}"),
        (re.compile(r"^density$"), "Mass density"),
        (re.compile(r"^number density$"), "Number density"),
        (re.compile(r"^composition$"), "Composition by element"),
        (re.compile(r"^composition summary$"),
         "Composition totals and net charge"),
        (re.compile(r"^oxide mol %$"), "Composition as oxides (mol %)"),
        # --- rings, components, network -----------------------------------
        (re.compile(r"^(R_C|R_N|P_N|P_max|P_min)\(n\), (\w+) rings$"),
         lambda m: f"{_RING_STATS[m[1]]} — {_criterion(m[2])}"),
        (re.compile(r"^ring sizes \(nodes\), (\w+)$"),
         lambda m: f"Ring sizes in nodes — {_criterion(m[1])}"),
        (re.compile(r"^ring sizes \(T atoms: ([^)]*)\), (\w+)$"),
         lambda m: f"Ring sizes in T atoms ({m[1]}) — {_criterion(m[2])}"),
        (re.compile(r"^distinct rings by size \(nodes\), (\w+)$"),
         lambda m: f"Distinct rings by size — {_criterion(m[1])}"),
        (re.compile(r"^largest piece, fraction of nodes$"),
         "Largest connected piece, fraction of nodes"),
        (re.compile(r"^number of components$"), "Number of connected pieces"),
        (re.compile(r"^number of interpenetrating pieces$"),
         "Interpenetrating pieces"),
        (re.compile(r"^nodes by component dimensionality$"),
         "Nodes by piece dimensionality (0–3)"),
        (re.compile(r"^nodes by component size$"), "Nodes by piece size"),
        (re.compile(rf"^coordination sequence, ({E})$"),
         lambda m: f"Coordination sequence of {_el(m[1])}"),
        (re.compile(rf"^polyhedral sharing ({E})-({E})$"),
         lambda m: f"Corner, edge and face sharing, {_pair(m[1], m[2])}"),
        (re.compile(rf"^Warren-Cowley alpha ({E})-({E})$"),
         lambda m: f"Warren–Cowley order α, {_pair(m[1], m[2])}"),
        # --- scattering ----------------------------------------------------
        (re.compile(r"^S_(NN|NC|CC)\(Q\)$"),
         lambda m: f"Bhatia–Thornton {_BT[m[1]]} S(Q)"),
        (re.compile(rf"^S\(Q\) ({E})-({E}) \(Faber-Ziman\)$"),
         lambda m: f"Faber–Ziman partial S(Q), {_pair(m[1], m[2])}"),
        (re.compile(r"^S\(Q\) (.+)$"),
         lambda m: f"Total structure factor S(Q) — {m[1]}"),
        (re.compile(r"^F\(Q\) = Q\[S\(Q\)-1\] (.+)$"),
         lambda m: f"Reduced structure factor F(Q) — {m[1]}"),
        (re.compile(r"^F_K\(Q\) (.+)$"),
         lambda m: f"Keen's structure factor F_K(Q) — {m[1]}"),
        (re.compile(r"^G\(r\) from S\(Q\) (.+)$"),
         lambda m: f"Pair distribution G(r), transformed from S(Q) — {m[1]}"),
        (re.compile(r"^G\(r\) (.+)$"),
         lambda m: f"Pair distribution function G(r) — {m[1]}"),
        (re.compile(r"^G_K\(r\) (.+)$"),
         lambda m: f"Keen's pair distribution G_K(r) — {m[1]}"),
        (re.compile(r"^T\(r\) (.+)$"),
         lambda m: f"Total correlation function T(r) — {m[1]}"),
        (re.compile(r"^D\(r\) (.+)$"),
         lambda m: f"Differential correlation function D(r) — {m[1]}"),
        (re.compile(r"^FSDP (position|height|FWHM) (.+)$"),
         lambda m: f"First sharp diffraction peak {m[1]} — {m[2]}"),
        # --- dynamics ------------------------------------------------------
        (re.compile(r"^MSD of the centre of mass$"),
         "Mean-square displacement of the centre of mass"),
        (re.compile(rf"^MSD ({E})$"),
         lambda m: f"Mean-square displacement of {_el(m[1])}"),
        (re.compile(r"^charge-displacement MSD$"),
         "Collective charge-displacement MSD"),
        (re.compile(rf"^Green-Kubo D ({E})$"),
         lambda m: f"Green–Kubo diffusion coefficient of {_el(m[1])}"),
        (re.compile(rf"^D ({E})$"),
         lambda m: f"Diffusion coefficient D of {_el(m[1])}"),
        (re.compile(rf"^normalised VACF ({E})$"),
         lambda m: f"Velocity autocorrelation of {_el(m[1])}, normalised"),
        (re.compile(rf"^VACF ({E})$"),
         lambda m: f"Velocity autocorrelation of {_el(m[1])}"),
        (re.compile(r"^VDOS (.+)$"),
         lambda m: f"Vibrational density of states, {m[1]}"),
        (re.compile(rf"^alpha2 ({E})$"),
         lambda m: f"Non-Gaussian parameter α₂(t) of {_el(m[1])}"),
        (re.compile(rf"^<dr\^2> ({E})$"),
         lambda m: f"Displacement moment ⟨δr²⟩ of {_el(m[1])}"),
        (re.compile(rf"^<dr\^4> ({E})$"),
         lambda m: f"Displacement moment ⟨δr⁴⟩ of {_el(m[1])}"),
        (re.compile(rf"^4 pi r\^2 G_s\(r, t = (.+)\) ({E})$"),
         lambda m: f"Self van Hove 4πr²G_s(r) at t = {m[1]} — {_el(m[2])}"),
        (re.compile(rf"^G_s\(r, t = (.+)\) ({E})$"),
         lambda m: f"Self van Hove function G_s(r) at t = {m[1]} — "
                   f"{_el(m[2])}"),
        (re.compile(rf"^F_s\(q = (.+), t\) ({E})$"),
         lambda m: f"Self intermediate scattering F_s(t) at q = {m[1]} — "
                   f"{_el(m[2])}"),
        (re.compile(rf"^tau ({E})-({E})$"),
         lambda m: f"Bond lifetime τ, {_pair(m[1], m[2])}"),
        (re.compile(r"^Haven ratio$"), "Haven ratio"),
        (re.compile(r"^Nernst-Einstein conductivity$"),
         "Nernst–Einstein conductivity"),
        (re.compile(r"^collective conductivity$"),
         "Collective (Green–Kubo) conductivity"),
        (re.compile(r"^kinetic temperature$"), "Kinetic temperature"),
        # --- order, voids ---------------------------------------------------
        (re.compile(r"^Voronoi CN \((.+)\)$"),
         lambda m: f"Voronoi coordination number ({m[1]})"),
        (re.compile(r"^Voronoi cell volume$"), "Voronoi cell volume"),
        (re.compile(r"^Voronoi index (.+)$"),
         lambda m: f"Voronoi index {m[1]}"),
        (re.compile(r"^empty-sphere radius$"), "Empty-sphere radius"),
        (re.compile(r"^q_tet (.+)$"),
         lambda m: f"Tetrahedral order parameter q_tet, {m[1]}"),
        # --- channels, modifier density, void regions ----------------------
        (re.compile(r"^accessible volume fraction of (\S+) at Delta (.+)$"),
         lambda m: f"Volume fraction open to {ion_text(m[1])} at "
                   f"Δ = {m[2]}"),
        (re.compile(r"^accessible volume fraction of (\S+)$"),
         lambda m: f"Volume fraction open to {ion_text(m[1])} vs the "
                   "mismatch Δ"),
        (re.compile(r"^accessible regions of (\S+) at Delta (.+)$"),
         lambda m: f"Regions open to {ion_text(m[1])} at Δ = {m[2]}"),
        (re.compile(r"^accessible regions at Delta (.+)$"),
         lambda m: f"The regions open at Δ = {m[1]}, listed"),
        (re.compile(r"^percolation threshold of (\S+), any axis$"),
         lambda m: f"Percolation threshold of {ion_text(m[1])}, any axis"),
        (re.compile(r"^percolation threshold of (\S+) along (\w)$"),
         lambda m: f"Percolation threshold of {ion_text(m[1])} along "
                   f"{m[2]}"),
        (re.compile(r"^lowest mismatch of (\S+) on the grid$"),
         lambda m: f"Lowest mismatch of {ion_text(m[1])} on the grid"),
        (re.compile(rf"^modifier count per anion \(({E}) around ({E})\)$"),
         lambda m: f"Modifier count per anion: {_el(m[1])} around "
                   f"{_el(m[2])}"),
        (re.compile(r"^modifier-rich anion fraction \(k_rich (\d+)\)$"),
         lambda m: f"Modifier-rich anions (≥ {m[1]} modifiers), fraction"),
        (re.compile(r"^modifier-rich fraction among (\w+) anions$"),
         lambda m: f"Modifier-rich fraction among {m[1]} anions"),
        (re.compile(r"^modifier-rich anions by speciation$"),
         "Modifier-rich anions by speciation"),
        (re.compile(r"^number of modifier clusters$"),
         "Number of modifier clusters"),
        (re.compile(r"^largest modifier cluster, (.+)$"),
         lambda m: f"Largest modifier cluster, {m[1]}"),
        (re.compile(r"^fraction of modifier atoms in a cluster$"),
         "Modifier atoms sitting in a cluster, fraction"),
        (re.compile(r"^modifier clusters span (\w) \(1 yes, 0 no\)$"),
         lambda m: f"Modifier clusters span {m[1]} (1 yes, 0 no)"),
        (re.compile(r"^modifier clusters$"), "The modifier clusters, listed"),
        (re.compile(r"^void fraction \(union of the void spheres\)$"),
         "Void fraction (union of the void spheres)"),
        (re.compile(r"^number of void regions$"), "Number of void regions"),
        (re.compile(r"^largest void region, union volume$"),
         "Largest void region, union volume"),
        (re.compile(r"^void regions spanning an axis$"),
         "Void regions spanning an axis"),
        (re.compile(r"^void region union volume$"),
         "Void region volumes (union)"),
        (re.compile(r"^void region elongation$"), "Void region elongation"),
        (re.compile(r"^void regions$"), "The void regions, listed"),
        (re.compile(r"^free volume$"), "Free volume"),
        # --- spectroscopy ---------------------------------------------------
        (re.compile(r"^predicted delta (\S+)$"),
         lambda m: f"Predicted chemical shift δ, {m[1]}"),
        (re.compile(r"^mean predicted delta (\S+)$"),
         lambda m: f"Mean predicted chemical shift δ, {m[1]}"),
        (re.compile(r"^(\S+) spectrum$"),
         lambda m: f"Predicted {m[1]} NMR spectrum"),
    )


_RULES = _rules()


def display_name(name: str) -> str:
    """The readable label of a descriptor id; the id itself when no rule
    names it (nothing is hidden behind a translation)."""
    text = str(name)
    suffix = ""
    for cut, readable in _CUTS:
        if text.endswith(cut):
            text, suffix = text[: -len(cut)], readable
            break
    for pattern, out in _RULES:
        match = pattern.match(text)
        if match is not None:
            label = out(match) if callable(out) else out
            return label + suffix
    return str(name)


def kind_badge(container) -> str:
    """A one-word tag of what a container holds: 'curve' (a function of an
    axis), 'distribution' (categories or a histogram), 'number' (one value
    per frame) or 'table' (rows)."""
    from ..core.md_stats import Distribution, Histogram, Scalar, Series

    if isinstance(container, Series):
        return "curve"
    if isinstance(container, (Distribution, Histogram)):
        return "distribution"
    if isinstance(container, Scalar):
        return "number"
    return "table"
