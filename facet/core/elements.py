"""Per-element reference data.

Deliberately dependency-light: this module is imported at startup, so it must
not pull in pymatgen (1.7 s) or anything else heavy. gemmi supplies atomic
number, weight and radii; everything here is data gemmi does not carry.

Nothing in FACET is specialised to a particular element. Where an element-family
default is wanted -- a lone-pair cation wants different display defaults from a
d0 transition metal -- it is expressed through :func:`family`, never by naming
an element in the analysis code.
"""
from __future__ import annotations

from dataclasses import dataclass

# --- Pauling electronegativity ----------------------------------------------
# Used to decide which species in a pair is the anion. Values are the usual
# Pauling scale; elements for which Pauling gives no value are omitted and are
# treated as cations by :func:`is_anion_like`.
ELECTRONEGATIVITY: dict[str, float] = {
    "H": 2.20, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04,
    "O": 3.44, "F": 3.98, "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90,
    "P": 2.19, "S": 2.58, "Cl": 3.16, "K": 0.82, "Ca": 1.00, "Sc": 1.36,
    "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55, "Fe": 1.83, "Co": 1.88,
    "Ni": 1.91, "Cu": 1.90, "Zn": 1.65, "Ga": 1.81, "Ge": 2.01, "As": 2.18,
    "Se": 2.55, "Br": 2.96, "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33,
    "Nb": 1.60, "Mo": 2.16, "Tc": 1.90, "Ru": 2.20, "Rh": 2.28, "Pd": 2.20,
    "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Sn": 1.96, "Sb": 2.05, "Te": 2.10,
    "I": 2.66, "Xe": 2.60, "Cs": 0.79, "Ba": 0.89, "La": 1.10, "Ce": 1.12,
    "Pr": 1.13, "Nd": 1.14, "Pm": 1.13, "Sm": 1.17, "Eu": 1.20, "Gd": 1.20,
    "Tb": 1.10, "Dy": 1.22, "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Yb": 1.10,
    "Lu": 1.27, "Hf": 1.30, "Ta": 1.50, "W": 2.36, "Re": 1.90, "Os": 2.20,
    "Ir": 2.20, "Pt": 2.28, "Au": 2.54, "Hg": 2.00, "Tl": 1.62, "Pb": 2.33,
    "Bi": 2.02, "Po": 2.00, "At": 2.20, "Fr": 0.70, "Ra": 0.90, "Ac": 1.10,
    "Th": 1.30, "Pa": 1.50, "U": 1.38, "Np": 1.36, "Pu": 1.28, "Am": 1.13,
}

# Species that act as the anion in essentially every inorganic structure FACET
# will see. Used only as a tie-break; the electronegativity rule is primary.
TYPICAL_ANIONS = ("O", "F", "Cl", "Br", "I", "S", "Se", "Te", "N", "H", "C", "P")

# --- oxidation states --------------------------------------------------------
# The single most common formal charge, used when a CIF states none and charge
# balance is under-determined. A guess made here is always reported as a guess.
COMMON_OX: dict[str, int] = {
    "H": 1, "Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "Ag": 1, "Fr": 1,
    "Be": 2, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2, "Ra": 2, "Zn": 2, "Cd": 2,
    "Hg": 2, "Cu": 2, "Ni": 2, "Co": 2, "Mn": 2, "Pd": 2, "Pt": 2, "Pb": 2,
    "Sn": 4, "Ge": 4, "Si": 4, "Ti": 4, "Zr": 4, "Hf": 4, "Th": 4, "C": 4,
    "Ce": 4, "Al": 3, "Ga": 3, "In": 3, "Tl": 3, "Sc": 3, "Y": 3, "B": 3,
    "Fe": 3, "Cr": 3, "Rh": 3, "Ir": 3, "Ru": 3, "Bi": 3, "Sb": 3, "As": 5,
    "La": 3, "Pr": 3, "Nd": 3, "Pm": 3, "Sm": 3, "Eu": 3, "Gd": 3, "Tb": 3,
    "Dy": 3, "Ho": 3, "Er": 3, "Tm": 3, "Yb": 3, "Lu": 3, "Ac": 3, "Am": 3,
    "P": 5, "V": 5, "Nb": 5, "Ta": 5, "Mo": 6, "W": 6, "U": 6, "Re": 7,
    "Tc": 7, "Os": 4, "Au": 3, "Pa": 5, "Np": 5, "Pu": 4, "Po": 4, "Xe": 0,
    "O": -2, "S": -2, "Se": -2, "Te": -2, "F": -1, "Cl": -1, "Br": -1,
    "I": -1, "At": -1, "N": -3,
}

# Oxidation states worth offering in the interface when the user overrides one.
ALT_OX: dict[str, tuple[int, ...]] = {
    "Bi": (3, 5), "Pb": (2, 4), "Sn": (2, 4), "Tl": (1, 3), "Sb": (3, 5),
    "As": (3, 5), "Ce": (3, 4), "Eu": (2, 3), "Yb": (2, 3), "Cu": (1, 2, 3),
    "Fe": (2, 3), "Mn": (2, 3, 4, 7), "Cr": (3, 6), "V": (3, 4, 5),
    "Ti": (3, 4), "Co": (2, 3), "Ni": (2, 3), "U": (4, 5, 6), "Au": (1, 3),
    "Ag": (1, 2), "Hg": (1, 2), "S": (-2, 4, 6), "N": (-3, 3, 5),
    "In": (1, 3), "Ga": (1, 3), "Mo": (4, 5, 6), "W": (4, 5, 6), "Re": (4, 6, 7),
}

# --- display ----------------------------------------------------------------
# A muted palette. Saturated CPK colours fight with ambient occlusion and depth
# cueing; these are desaturated enough that the shading stays readable.
COLOR: dict[str, tuple[float, float, float]] = {
    "H": (0.90, 0.90, 0.90), "Li": (0.80, 0.50, 1.00), "Be": (0.76, 1.00, 0.00),
    "B": (0.55, 0.85, 0.66), "C": (0.35, 0.35, 0.38), "N": (0.19, 0.31, 0.97),
    "O": (0.94, 0.30, 0.24), "F": (0.55, 0.88, 0.80), "Na": (0.67, 0.80, 0.36),
    "Mg": (0.54, 1.00, 0.00), "Al": (0.74, 0.62, 0.55), "Si": (0.42, 0.55, 0.78),
    "P": (0.95, 0.60, 0.22), "S": (0.95, 0.88, 0.19), "Cl": (0.35, 0.83, 0.28),
    "K": (0.56, 0.25, 0.83), "Ca": (0.24, 1.00, 0.00), "Sc": (0.90, 0.90, 0.90),
    "Ti": (0.62, 0.66, 0.70), "V": (0.85, 0.42, 0.42), "Cr": (0.54, 0.60, 0.78),
    "Mn": (0.61, 0.48, 0.78), "Fe": (0.87, 0.46, 0.22), "Co": (0.44, 0.56, 0.78),
    "Ni": (0.31, 0.72, 0.42), "Cu": (0.78, 0.50, 0.20), "Zn": (0.49, 0.50, 0.69),
    "Ga": (0.76, 0.56, 0.56), "Ge": (0.40, 0.56, 0.56), "As": (0.74, 0.50, 0.89),
    "Se": (1.00, 0.63, 0.00), "Br": (0.65, 0.16, 0.16), "Rb": (0.44, 0.18, 0.69),
    "Sr": (0.00, 1.00, 0.00), "Y": (0.58, 1.00, 1.00), "Zr": (0.58, 0.88, 0.88),
    "Nb": (0.45, 0.76, 0.79), "Mo": (0.36, 0.66, 0.66), "Tc": (0.23, 0.62, 0.62),
    "Ru": (0.14, 0.56, 0.56), "Rh": (0.04, 0.49, 0.55), "Pd": (0.00, 0.41, 0.52),
    "Ag": (0.75, 0.75, 0.75), "Cd": (1.00, 0.85, 0.56), "In": (0.65, 0.46, 0.45),
    "Sn": (0.40, 0.50, 0.50), "Sb": (0.62, 0.39, 0.71), "Te": (0.83, 0.48, 0.00),
    "I": (0.58, 0.00, 0.58), "Cs": (0.34, 0.09, 0.56), "Ba": (0.34, 0.76, 0.55),
    "La": (0.44, 0.83, 1.00), "Ce": (1.00, 1.00, 0.78), "Pr": (0.85, 1.00, 0.78),
    "Nd": (0.78, 1.00, 0.78), "Sm": (0.62, 1.00, 0.78), "Eu": (0.38, 1.00, 0.78),
    "Gd": (0.27, 1.00, 0.78), "Tb": (0.19, 1.00, 0.78), "Dy": (0.12, 1.00, 0.78),
    "Ho": (0.00, 1.00, 0.61), "Er": (0.00, 0.90, 0.46), "Tm": (0.00, 0.83, 0.32),
    "Yb": (0.00, 0.75, 0.22), "Lu": (0.00, 0.67, 0.14), "Hf": (0.30, 0.76, 1.00),
    "Ta": (0.30, 0.65, 1.00), "W": (0.30, 0.52, 0.66), "Re": (0.15, 0.49, 0.67),
    "Os": (0.15, 0.40, 0.59), "Ir": (0.09, 0.33, 0.53), "Pt": (0.82, 0.82, 0.88),
    "Au": (1.00, 0.82, 0.14), "Hg": (0.72, 0.72, 0.82), "Tl": (0.65, 0.33, 0.30),
    "Pb": (0.34, 0.35, 0.38), "Bi": (0.62, 0.31, 0.71), "Po": (0.67, 0.36, 0.00),
    "At": (0.46, 0.31, 0.27), "Ra": (0.00, 0.49, 0.00), "Ac": (0.44, 0.67, 0.98),
    "Th": (0.00, 0.73, 1.00), "Pa": (0.00, 0.63, 1.00), "U": (0.00, 0.56, 1.00),
    "Np": (0.00, 0.50, 1.00), "Pu": (0.00, 0.42, 1.00), "Am": (0.33, 0.36, 0.95),
}
_FALLBACK_COLOR = (0.70, 0.70, 0.72)

# Display radii in angstrom. Not physical radii -- a ball-and-stick model reads
# best when the cation is clearly larger than the anion it is drawn against,
# which is the opposite of the ionic-radius ordering. Roughly 0.4x the covalent
# radius, hand-adjusted for the common cases.
_DISPLAY_RADIUS: dict[str, float] = {
    "H": 0.20, "Li": 0.42, "Be": 0.30, "B": 0.28, "C": 0.30, "N": 0.30,
    "O": 0.32, "F": 0.30, "Na": 0.52, "Mg": 0.46, "Al": 0.36, "Si": 0.38,
    "P": 0.36, "S": 0.40, "Cl": 0.40, "K": 0.62, "Ca": 0.56, "Sc": 0.50,
    "Ti": 0.42, "V": 0.40, "Cr": 0.40, "Mn": 0.42, "Fe": 0.40, "Co": 0.40,
    "Ni": 0.40, "Cu": 0.42, "Zn": 0.44, "Ga": 0.44, "Ge": 0.44, "As": 0.44,
    "Se": 0.46, "Br": 0.46, "Rb": 0.68, "Sr": 0.62, "Y": 0.56, "Zr": 0.50,
    "Nb": 0.48, "Mo": 0.44, "Ag": 0.52, "Cd": 0.54, "In": 0.54, "Sn": 0.54,
    "Sb": 0.54, "Te": 0.54, "I": 0.54, "Cs": 0.76, "Ba": 0.66, "La": 0.62,
    "Ce": 0.60, "Nd": 0.60, "Sm": 0.58, "Eu": 0.58, "Gd": 0.58, "Dy": 0.56,
    "Er": 0.56, "Yb": 0.56, "Lu": 0.54, "Hf": 0.50, "Ta": 0.48, "W": 0.44,
    "Re": 0.44, "Pt": 0.46, "Au": 0.48, "Hg": 0.52, "Tl": 0.60, "Pb": 0.62,
    "Bi": 0.62, "Th": 0.62, "U": 0.58,
}
_FALLBACK_RADIUS = 0.45


@dataclass(frozen=True)
class ElementInfo:
    symbol: str
    z: int
    electronegativity: float | None
    common_ox: int | None
    color: tuple[float, float, float]
    display_radius: float
    covalent_radius: float | None
    vdw_radius: float | None
    weight: float | None


_cache: dict[str, ElementInfo] = {}


def info(symbol: str) -> ElementInfo:
    """Reference data for one element symbol. Cached; safe to call per atom."""
    sym = normalise(symbol)
    hit = _cache.get(sym)
    if hit is not None:
        return hit
    z = cov = vdw = wt = None
    try:
        import gemmi

        el = gemmi.Element(sym)
        if el.atomic_number:
            z = el.atomic_number
            cov = el.covalent_r or None
            vdw = el.vdw_r or None
            wt = el.weight or None
    except Exception:
        pass
    out = ElementInfo(
        symbol=sym,
        z=z or 0,
        electronegativity=ELECTRONEGATIVITY.get(sym),
        common_ox=COMMON_OX.get(sym),
        color=COLOR.get(sym, _FALLBACK_COLOR),
        display_radius=_DISPLAY_RADIUS.get(sym, _FALLBACK_RADIUS),
        covalent_radius=cov,
        vdw_radius=vdw,
        weight=wt,
    )
    _cache[sym] = out
    return out


def normalise(label: str) -> str:
    """Reduce a CIF type symbol or site label to a bare element symbol.

    CIFs write the same element as ``Na``, ``Na1``, ``Na+``, ``Na+1``, ``NA``
    and ``Na1+`` depending on the depositor and the database. Everything after
    the leading alphabetic run is dropped, and the case is normalised.
    """
    s = str(label).strip()
    if not s:
        return ""
    alpha = []
    for ch in s:
        if ch.isalpha():
            alpha.append(ch)
        else:
            break
    if not alpha:
        return ""
    sym = "".join(alpha)
    # 'NA' -> 'Na'; 'c' -> 'C'
    sym = sym[0].upper() + sym[1:].lower()
    if sym in ELECTRONEGATIVITY or sym in COMMON_OX or sym in COLOR:
        return sym
    # two-letter guess failed; try the first letter alone ('Om' from a label)
    one = sym[0]
    if one in ELECTRONEGATIVITY or one in COMMON_OX:
        return one
    return sym


def is_anion_like(symbol: str) -> bool:
    """Whether this element is more electronegative than a typical cation.

    A threshold on the Pauling scale, not a list. Elements above 2.0 act as the
    anion against essentially every metal, which is the distinction that matters
    when deciding which contacts of a site are bonds.
    """
    x = ELECTRONEGATIVITY.get(normalise(symbol))
    return x is not None and x >= 2.0


def more_electronegative(a: str, b: str) -> str | None:
    """Which of two elements is the anion in a pair, or None if undecidable."""
    xa = ELECTRONEGATIVITY.get(normalise(a))
    xb = ELECTRONEGATIVITY.get(normalise(b))
    if xa is None or xb is None or xa == xb:
        return None
    return normalise(a) if xa > xb else normalise(b)


def family(symbol: str, ox: int | None = None) -> str:
    """Coarse chemical family, used only to choose interface defaults.

    Never used by the analysis. Its purpose is that selecting a lone-pair cation
    should bring up displays suited to a one-sided environment without the user
    having to know to ask for them.
    """
    sym = normalise(symbol)
    if is_anion_like(sym) and (ox is None or ox < 0):
        return "anion"
    if ox is None:
        ox = COMMON_OX.get(sym)
    # ns2 cations: the stereochemically active lone pair
    if (sym, ox) in {("Bi", 3), ("Pb", 2), ("Sn", 2), ("Sb", 3), ("As", 3),
                     ("Tl", 1), ("Ge", 2), ("Se", 4), ("Te", 4), ("I", 5)}:
        return "lone-pair"
    if sym in ("Li", "Na", "K", "Rb", "Cs", "Fr"):
        return "alkali"
    if sym in ("Be", "Mg", "Ca", "Sr", "Ba", "Ra"):
        return "alkaline-earth"
    if sym in ("La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
               "Ho", "Er", "Tm", "Yb", "Lu", "Y", "Sc"):
        return "rare-earth"
    if sym in ("Th", "Pa", "U", "Np", "Pu", "Am"):
        return "actinide"
    if sym in ("B", "Al", "Si", "P", "Ga", "Ge", "As", "In", "Sn", "Sb"):
        return "main-group"
    if (sym, ox) in {("Ti", 4), ("Zr", 4), ("Hf", 4), ("V", 5), ("Nb", 5),
                     ("Ta", 5), ("Mo", 6), ("W", 6), ("Cr", 6)}:
        return "d0-transition-metal"
    if info(sym).z and 21 <= info(sym).z <= 80:
        return "transition-metal"
    return "other"
