"""Colour, and how a structure is coloured.

Two separable things live here.

**Palettes** are per-element colours. FACET ships several because people arrive
with different habits -- a VESTA user and a Jmol user expect different oxygens --
and because a printed figure often wants neither. Any element can be overridden,
and the result saved as a theme file and shared with a group so that a set of
figures matches.

**Colour modes** decide what an atom's colour *means*. Colouring by element is
the default and the only one most software offers. The others exist because this
application is about a quantity that varies from site to site: colouring by
bond-valence sum, by coordination number, or by phi puts the analysis onto the
structure itself, so a stereoactive site is visible in the picture rather than
only in the table beside it.

Nothing here imports Qt. A theme is data, and the interface reads it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

import numpy as np

from . import elements

RGB = tuple[float, float, float]

# What a renderer draws with when it has been handed no theme at all: the
# placeholder before a structure is loaded, a PainterRenderer built directly,
# an export called without a theme. These repeat the defaults of ``Theme``
# below as plain numbers, because the modules that need them draw before a
# theme object exists; ``test_theme_explorer`` asserts the two agree, so the
# app cannot end up half dark and half light after the default changes.
FALLBACK_BACKGROUND: RGB = (1.00, 1.00, 1.00)
FALLBACK_CELL_COLOR: RGB = (0.16, 0.17, 0.20)
FALLBACK_LABEL_COLOR: RGB = (0.08, 0.09, 0.11)
FALLBACK_SELECTION_COLOR: RGB = (0.93, 0.35, 0.05)
FALLBACK_FOG: float = 0.18


class ColorMode(Enum):
    """What an atom's colour encodes."""

    ELEMENT = "element"
    SITE = "site"                    # one colour per crystallographic site
    BVS = "bond-valence sum"
    COORDINATION = "coordination number"
    PHI = "stereoactivity (phi)"
    VALENCE_DISCREPANCY = "valence discrepancy"
    UNIFORM = "uniform"


class BondColorMode(Enum):
    HALF_ATOM = "half by atom"       # each half takes its atom's colour
    UNIFORM = "uniform"
    BY_VALENCE = "by bond valence"   # a ramp from weak to strong


# ---------------------------------------------------------------------------
# palettes
# ---------------------------------------------------------------------------

def _facet_muted() -> dict[str, RGB]:
    """Muted, so shading stays readable on top of it.

    Fully saturated CPK colours fight with ambient occlusion and depth cueing --
    a pure red oxygen has nowhere to go darker. This palette was the shipped
    default while the default background was dark; on white the saturated one
    below reads better, so that is now the default.
    """
    return dict(elements.COLOR)


def _vesta_like() -> dict[str, RGB]:
    """The colours a VESTA user expects, on a white ground.

    Saturated: red oxygen, yellow sodium, blue silicon, violet bismuth. Matched
    by eye to what VESTA draws rather than copied from its tables, so treat it
    as a familiar convention and not as a claim of byte equality with another
    program. Every element not listed keeps its muted colour.
    """
    out = dict(elements.COLOR)
    out.update({
        "H": (1.00, 0.80, 0.80), "He": (0.99, 0.91, 0.81),
        "Li": (0.53, 0.88, 0.45), "Be": (0.37, 0.84, 0.48),
        "B": (0.12, 0.64, 0.06), "C": (0.50, 0.29, 0.16),
        "N": (0.69, 0.73, 0.90), "O": (1.00, 0.05, 0.02),
        "F": (0.70, 0.85, 0.45), "Ne": (0.70, 0.89, 0.96),
        "Na": (0.98, 0.86, 0.24), "Mg": (0.98, 0.48, 0.08),
        "Al": (0.51, 0.70, 0.84), "Si": (0.11, 0.23, 0.98),
        "P": (0.75, 0.61, 0.76), "S": (1.00, 0.98, 0.00),
        "Cl": (0.19, 0.99, 0.01), "K": (0.63, 0.13, 0.96),
        "Ca": (0.35, 0.59, 0.74), "Ti": (0.47, 0.79, 1.00),
        "V": (0.90, 0.44, 0.20), "Cr": (0.00, 0.00, 0.62),
        "Mn": (0.66, 0.00, 0.69), "Fe": (0.71, 0.44, 0.00),
        "Co": (0.00, 0.00, 0.68), "Ni": (0.72, 0.72, 0.72),
        "Cu": (0.13, 0.28, 0.86), "Zn": (0.56, 0.56, 0.51),
        "Ga": (0.62, 0.47, 0.35), "Ge": (0.40, 0.51, 0.51),
        "As": (0.46, 0.31, 0.62), "Se": (0.60, 0.73, 0.00),
        "Br": (0.49, 0.19, 0.01), "Sr": (0.00, 1.00, 0.16),
        "Y": (0.58, 1.00, 0.00), "Zr": (0.00, 1.00, 0.00),
        "Nb": (0.30, 0.70, 0.46), "Mo": (0.33, 0.71, 0.71),
        "Ag": (0.75, 0.75, 0.75), "Cd": (1.00, 0.85, 0.56),
        "In": (0.65, 0.46, 0.45), "Sn": (0.60, 0.56, 0.56),
        "Sb": (0.56, 0.56, 0.82), "Te": (0.83, 0.72, 0.02),
        "I": (0.58, 0.00, 0.58), "Ba": (0.00, 0.79, 0.00),
        "La": (0.35, 0.77, 0.29), "Ce": (1.00, 1.00, 0.78),
        "W": (0.15, 0.40, 0.57), "Pt": (0.82, 0.82, 0.88),
        "Au": (1.00, 0.82, 0.14), "Hg": (0.72, 0.72, 0.82),
        "Tl": (0.65, 0.33, 0.30), "Pb": (0.34, 0.35, 0.38),
        "Bi": (0.62, 0.31, 0.71), "Th": (0.00, 0.73, 0.00),
        "U": (0.00, 0.56, 0.00),
    })
    return out


def _jmol_like() -> dict[str, RGB]:
    """Closer to the saturated CPK convention most chemists learned."""
    out = dict(elements.COLOR)
    out.update({
        "H": (1.00, 1.00, 1.00), "C": (0.56, 0.56, 0.56),
        "N": (0.19, 0.31, 0.97), "O": (1.00, 0.05, 0.05),
        "F": (0.56, 0.88, 0.31), "S": (1.00, 1.00, 0.19),
        "Cl": (0.12, 0.94, 0.12), "P": (1.00, 0.50, 0.00),
        "Na": (0.67, 0.36, 0.95), "Mg": (0.54, 1.00, 0.00),
        "Si": (0.94, 0.78, 0.63), "Ca": (0.24, 1.00, 0.00),
        "Fe": (0.88, 0.40, 0.20), "Bi": (0.62, 0.31, 0.71),
    })
    return out


def _greyscale() -> dict[str, RGB]:
    """For a printed figure that must survive being photocopied.

    Lightness tracks atomic number, so heavier is darker and the ordering
    survives conversion to grey.
    """
    out: dict[str, RGB] = {}
    for sym in elements.COLOR:
        z = elements.info(sym).z or 1
        t = 1.0 - min(z, 92) / 110.0
        out[sym] = (0.18 + 0.72 * t,) * 3
    return out


def _high_contrast() -> dict[str, RGB]:
    """Distinguishable with the commonest colour-vision deficiencies.

    Avoids relying on red against green, which is the pairing that fails most
    often and is exactly what CPK uses for oxygen against chlorine.
    """
    out = dict(elements.COLOR)
    out.update({
        "O": (0.85, 0.37, 0.01), "Cl": (0.00, 0.62, 0.45),
        "N": (0.00, 0.45, 0.70), "S": (0.94, 0.89, 0.26),
        "F": (0.34, 0.71, 0.91), "C": (0.35, 0.35, 0.38),
        "Bi": (0.80, 0.47, 0.65), "Na": (0.00, 0.62, 0.45),
        "Si": (0.34, 0.71, 0.91), "P": (0.90, 0.62, 0.00),
    })
    return out


PALETTES: dict[str, callable] = {
    "VESTA-like": _vesta_like,
    "FACET (muted)": _facet_muted,
    "Jmol / CPK": _jmol_like,
    "Greyscale": _greyscale,
    "High contrast": _high_contrast,
}

# A qualitative sequence for colouring by site rather than by element. Chosen to
# stay distinguishable in both light and dark surroundings.
SITE_CYCLE: tuple[RGB, ...] = (
    (0.34, 0.71, 0.91), (0.90, 0.62, 0.00), (0.00, 0.62, 0.45),
    (0.80, 0.47, 0.65), (0.94, 0.89, 0.26), (0.55, 0.45, 0.85),
    (0.85, 0.37, 0.01), (0.45, 0.75, 0.55), (0.70, 0.30, 0.30),
    (0.40, 0.55, 0.78),
)


# ---------------------------------------------------------------------------
# continuous ramps
# ---------------------------------------------------------------------------
# Perceptually ordered and safe in greyscale: lightness increases monotonically,
# so the ramp still reads when printed in black and white.

_VIRIDIS = np.array([
    (0.267, 0.005, 0.329), (0.283, 0.141, 0.458), (0.254, 0.265, 0.530),
    (0.207, 0.372, 0.553), (0.164, 0.471, 0.558), (0.128, 0.567, 0.551),
    (0.135, 0.659, 0.518), (0.267, 0.749, 0.441), (0.478, 0.821, 0.318),
    (0.741, 0.873, 0.150), (0.993, 0.906, 0.144),
])

# Diverging, for a signed quantity such as a valence discrepancy where zero is
# the meaningful midpoint and the sign matters.
_DIVERGING = np.array([
    (0.230, 0.299, 0.754), (0.516, 0.596, 0.876), (0.790, 0.825, 0.927),
    (0.940, 0.940, 0.940), (0.945, 0.800, 0.740), (0.876, 0.565, 0.470),
    (0.706, 0.016, 0.150),
])


def ramp(t, diverging: bool = False) -> RGB:
    """Sample a ramp at ``t`` in 0..1. NaN gives a neutral grey."""
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return (0.60, 0.62, 0.66)
    table = _DIVERGING if diverging else _VIRIDIS
    t = float(np.clip(t, 0.0, 1.0)) * (len(table) - 1)
    i = int(np.floor(t))
    j = min(i + 1, len(table) - 1)
    f = t - i
    c = table[i] * (1 - f) + table[j] * f
    return (float(c[0]), float(c[1]), float(c[2]))


# ---------------------------------------------------------------------------
# the theme
# ---------------------------------------------------------------------------

@dataclass
class Theme:
    """Everything the viewer needs to know about colour and sizing.

    The defaults are the shipped theme, and they are deliberately a white
    ground: that is what the crystallography a user is comparing against looks
    like, on paper and in VESTA. Every preset below states its own values in
    full rather than inheriting these, so that changing the default cannot
    silently redefine another preset.
    """

    name: str = "VESTA-like (white)"
    palette_name: str = "VESTA-like"
    overrides: dict[str, RGB] = field(default_factory=dict)

    color_mode: ColorMode = ColorMode.ELEMENT
    bond_color_mode: BondColorMode = BondColorMode.HALF_ATOM
    uniform_atom_color: RGB = (0.62, 0.64, 0.68)
    uniform_bond_color: RGB = (0.45, 0.47, 0.51)

    background: RGB = (1.00, 1.00, 1.00)
    cell_color: RGB = (0.16, 0.17, 0.20)
    label_color: RGB = (0.08, 0.09, 0.11)
    selection_color: RGB = (0.93, 0.35, 0.05)
    polyhedron_color: RGB = (0.28, 0.58, 0.88)
    polyhedron_alpha: float = 0.42
    subthreshold_color: RGB = (0.66, 0.68, 0.72)

    atom_scale: float = 1.0
    bond_scale: float = 1.0
    fog_amount: float = 0.18
    ambient_occlusion: bool = True
    outlines: bool = True
    specular: float = 1.0

    # range of the continuous colour modes; None means autoscale per structure
    scale_min: float | None = None
    scale_max: float | None = None

    # -- palette -----------------------------------------------------------
    def palette(self) -> dict[str, RGB]:
        base = PALETTES.get(self.palette_name, _vesta_like)()
        base.update(self.overrides)
        return base

    def element_color(self, symbol: str) -> RGB:
        sym = elements.normalise(symbol)
        if sym in self.overrides:
            return self.overrides[sym]
        return PALETTES.get(self.palette_name, _vesta_like)().get(
            sym, (0.70, 0.70, 0.72))

    def set_element_color(self, symbol: str, rgb: RGB) -> None:
        self.overrides[elements.normalise(symbol)] = tuple(float(c) for c in rgb)

    def clear_element_color(self, symbol: str) -> None:
        self.overrides.pop(elements.normalise(symbol), None)

    def reset_overrides(self) -> None:
        self.overrides.clear()

    @property
    def is_light_background(self) -> bool:
        r, g, b = self.background
        return (0.2126 * r + 0.7152 * g + 0.0722 * b) > 0.5

    def contrasting_ink(self) -> RGB:
        """Text and cell colour that stays legible on this background.

        A theme switched to a white background for printing must not keep the
        pale grey unit cell that was designed for a dark one.
        """
        return (0.10, 0.11, 0.13) if self.is_light_background else self.label_color

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        out = {
            "name": self.name,
            "palette": self.palette_name,
            "overrides": {k: list(v) for k, v in self.overrides.items()},
            "color_mode": self.color_mode.value,
            "bond_color_mode": self.bond_color_mode.value,
        }
        for key in ("uniform_atom_color", "uniform_bond_color", "background",
                    "cell_color", "label_color", "selection_color",
                    "polyhedron_color", "subthreshold_color"):
            out[key] = list(getattr(self, key))
        for key in ("polyhedron_alpha", "atom_scale", "bond_scale",
                    "fog_amount", "specular", "scale_min", "scale_max"):
            out[key] = getattr(self, key)
        for key in ("ambient_occlusion", "outlines"):
            out[key] = bool(getattr(self, key))
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "Theme":
        t = cls()
        t.name = data.get("name", t.name)
        t.palette_name = data.get("palette", t.palette_name)
        t.overrides = {k: tuple(v) for k, v in (data.get("overrides") or {}).items()}
        try:
            t.color_mode = ColorMode(data.get("color_mode", t.color_mode.value))
        except ValueError:
            pass
        try:
            t.bond_color_mode = BondColorMode(
                data.get("bond_color_mode", t.bond_color_mode.value))
        except ValueError:
            pass
        for key in ("uniform_atom_color", "uniform_bond_color", "background",
                    "cell_color", "label_color", "selection_color",
                    "polyhedron_color", "subthreshold_color"):
            if key in data and data[key] is not None:
                setattr(t, key, tuple(float(c) for c in data[key]))
        for key in ("polyhedron_alpha", "atom_scale", "bond_scale",
                    "fog_amount", "specular", "scale_min", "scale_max"):
            if key in data and data[key] is not None:
                setattr(t, key, float(data[key]))
        for key in ("ambient_occlusion", "outlines"):
            if key in data:
                setattr(t, key, bool(data[key]))
        return t

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Theme":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def copy(self) -> "Theme":
        return replace(self, overrides=dict(self.overrides))


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------

def vesta() -> Theme:
    """The shipped default: white ground, saturated element colours.

    Spelled out rather than returning a bare ``Theme()`` so that the preset and
    the dataclass defaults are two statements of the same thing, and a future
    change to one is visible as a difference from the other.
    """
    return Theme(
        name="VESTA-like (white)",
        palette_name="VESTA-like",
        background=(1.00, 1.00, 1.00),
        cell_color=(0.16, 0.17, 0.20),
        label_color=(0.08, 0.09, 0.11),
        selection_color=(0.93, 0.35, 0.05),
        polyhedron_color=(0.28, 0.58, 0.88),
        polyhedron_alpha=0.42,
        subthreshold_color=(0.66, 0.68, 0.72),
        uniform_atom_color=(0.62, 0.64, 0.68),
        uniform_bond_color=(0.45, 0.47, 0.51),
        fog_amount=0.18,
    )


def dark() -> Theme:
    """The dark ground FACET shipped with before white became the default.

    Every field is stated, because the alternative -- letting it fall back to
    the dataclass defaults -- is what would make this preset change whenever
    the default theme does.
    """
    return Theme(
        name="Dark",
        palette_name="FACET (muted)",
        background=(0.086, 0.094, 0.110),
        cell_color=(0.43, 0.46, 0.53),
        label_color=(0.92, 0.93, 0.96),
        selection_color=(1.00, 0.84, 0.36),
        polyhedron_color=(0.30, 0.74, 0.96),
        polyhedron_alpha=0.38,
        subthreshold_color=(0.42, 0.44, 0.48),
        uniform_atom_color=(0.72, 0.74, 0.78),
        uniform_bond_color=(0.70, 0.72, 0.76),
        fog_amount=0.65,
    )


def slate() -> Theme:
    """A mid-grey ground: neither glare nor the loss of pale atoms on black."""
    return Theme(
        name="Slate",
        palette_name="FACET (muted)",
        background=(0.29, 0.31, 0.35),
        cell_color=(0.80, 0.82, 0.86),
        label_color=(0.96, 0.97, 0.99),
        selection_color=(1.00, 0.84, 0.36),
        polyhedron_color=(0.36, 0.78, 0.98),
        polyhedron_alpha=0.40,
        subthreshold_color=(0.52, 0.54, 0.58),
        uniform_atom_color=(0.72, 0.74, 0.78),
        uniform_bond_color=(0.70, 0.72, 0.76),
        fog_amount=0.45,
    )


def light() -> Theme:
    """For a figure going into a paper, which is printed on white."""
    return Theme(
        name="Light (for print)",
        palette_name="FACET (muted)",
        background=(0.99, 0.99, 0.99),
        cell_color=(0.35, 0.37, 0.42),
        label_color=(0.10, 0.11, 0.13),
        selection_color=(0.93, 0.35, 0.05),
        polyhedron_color=(0.30, 0.62, 0.90),
        polyhedron_alpha=0.40,
        subthreshold_color=(0.62, 0.64, 0.68),
        uniform_atom_color=(0.62, 0.64, 0.68),
        uniform_bond_color=(0.45, 0.47, 0.51),
        fog_amount=0.30,
        outlines=True,
    )


def publication() -> Theme:
    """White ground, greyscale palette, no depth cueing.

    Depth cueing is off deliberately: it reads as haze in print and makes a
    figure look poorly reproduced rather than three-dimensional.
    """
    return Theme(
        name="Publication (greyscale)",
        palette_name="Greyscale",
        background=(1.0, 1.0, 1.0),
        cell_color=(0.25, 0.25, 0.25),
        label_color=(0.0, 0.0, 0.0),
        selection_color=(0.00, 0.00, 0.00),
        polyhedron_color=(0.55, 0.60, 0.66),
        polyhedron_alpha=0.45,
        subthreshold_color=(0.70, 0.70, 0.70),
        uniform_atom_color=(0.60, 0.60, 0.60),
        uniform_bond_color=(0.35, 0.35, 0.35),
        fog_amount=0.0,
        ambient_occlusion=True,
        outlines=True,
    )


def accessible() -> Theme:
    """Distinguishable with the commonest colour-vision deficiencies."""
    return Theme(
        name="High contrast",
        palette_name="High contrast",
        background=(1.00, 1.00, 1.00),
        cell_color=(0.05, 0.05, 0.07),
        label_color=(0.00, 0.00, 0.00),
        selection_color=(0.00, 0.35, 0.80),
        polyhedron_color=(0.34, 0.71, 0.91),
        polyhedron_alpha=0.40,
        subthreshold_color=(0.58, 0.60, 0.64),
        uniform_atom_color=(0.55, 0.57, 0.60),
        uniform_bond_color=(0.25, 0.27, 0.30),
        fog_amount=0.0,
    )


# Order matters: the first entry is what the preset list and the View menu show
# first, and it is the theme the application starts in.
PRESETS: dict[str, callable] = {
    "VESTA-like (white)": vesta,
    "Light (for print)": light,
    "Publication (greyscale)": publication,
    "High contrast": accessible,
    "Slate": slate,
    "Dark": dark,
}


# ---------------------------------------------------------------------------
# applying a colour mode
# ---------------------------------------------------------------------------

def site_values(results, mode: ColorMode) -> dict[int, float]:
    """The scalar each analysed site contributes to a continuous colour mode."""
    out: dict[int, float] = {}
    for r in results:
        if mode is ColorMode.BVS:
            out[r.site_index] = r.bvs
        elif mode is ColorMode.COORDINATION:
            out[r.site_index] = float(r.cn_valence)
        elif mode is ColorMode.PHI:
            out[r.site_index] = r.phi
        elif mode is ColorMode.VALENCE_DISCREPANCY:
            out[r.site_index] = (r.valence_discrepancy
                                 if r.valence_discrepancy is not None
                                 else float("nan"))
    return out


def scale_range(values: dict[int, float], theme: Theme,
                mode: ColorMode) -> tuple[float, float]:
    """Autoscale a continuous mode, unless the theme pins the range.

    A pinned range is what makes two figures comparable; autoscaling is what
    makes one figure readable. Both are wanted, at different moments.
    """
    if theme.scale_min is not None and theme.scale_max is not None:
        return theme.scale_min, theme.scale_max
    finite = [v for v in values.values() if v is not None and not np.isnan(v)]
    if not finite:
        return 0.0, 1.0
    lo, hi = float(min(finite)), float(max(finite))
    if mode is ColorMode.PHI:
        return 0.0, max(hi, 0.05)
    if mode is ColorMode.VALENCE_DISCREPANCY:
        m = max(abs(lo), abs(hi), 0.05)
        return -m, m                       # symmetric, so zero sits mid-ramp
    if hi - lo < 1e-9:
        return lo - 0.5, hi + 0.5
    return lo, hi


def color_for_value(value: float, lo: float, hi: float,
                    diverging: bool = False) -> RGB:
    if value is None or np.isnan(value):
        return (0.60, 0.62, 0.66)
    if hi - lo < 1e-12:
        return ramp(0.5, diverging)
    return ramp((value - lo) / (hi - lo), diverging)
