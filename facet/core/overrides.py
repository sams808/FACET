"""Per-site and per-atom presentation overrides.

The theme decides how an element looks everywhere. This decides how one site, or
one single atom, looks differently -- which is what VESTA and CrystalMaker both
give you, and what you need the moment a structure has two crystallographically
distinct bismuth sites and you want to tell them apart in a figure.

Three levels, each overriding the one before:

1. the element, from the theme
2. the **site**, keyed by its label: every atom generated from that
   symmetry-distinct position
3. the **atom**, keyed by its index in the drawn scene: one position

Only the fields actually set are applied, so overriding a colour leaves the
radius following the theme, and clearing an override restores whatever it was
covering rather than a default. That is why every field is optional and ``None``
means "not overridden" rather than "no colour".

Overrides belong to a structure, not to the theme: a label like "Bi1" means
something in one file and something else in another, and a palette is meant to
be reusable across files.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

RGB = tuple[float, float, float]


@dataclass
class Style:
    """What may be overridden. ``None`` means "leave it alone"."""

    color: RGB | None = None
    radius_scale: float | None = None        # multiplies the drawn radius
    visible: bool | None = None
    label: str | None = None                 # shown instead of the site label
    note: str = ""                           # why, for the user's own record

    @property
    def is_empty(self) -> bool:
        return (self.color is None and self.radius_scale is None
                and self.visible is None and self.label is None)

    def over(self, other: "Style") -> "Style":
        """This style on top of ``other``: a field set here wins."""
        return Style(
            color=self.color if self.color is not None else other.color,
            radius_scale=(self.radius_scale if self.radius_scale is not None
                          else other.radius_scale),
            visible=self.visible if self.visible is not None else other.visible,
            label=self.label if self.label is not None else other.label,
            note=self.note or other.note)

    def describe(self) -> str:
        parts = []
        if self.color is not None:
            parts.append("colour %.2f/%.2f/%.2f" % self.color)
        if self.radius_scale is not None:
            parts.append(f"radius x{self.radius_scale:.2f}")
        if self.visible is not None:
            parts.append("shown" if self.visible else "hidden")
        if self.label is not None:
            parts.append(f"labelled {self.label!r}")
        return ", ".join(parts) if parts else "nothing overridden"


@dataclass
class StyleOverrides:
    """Every override for one structure.

    Sites are keyed by label because that is what a CIF gives and what the user
    reads in the site list. Atoms are keyed by their index in the built scene,
    which is stable for a given structure and cell range but not across a change
    of either -- so an atom override is a deliberate, local thing, and
    :meth:`prune` drops any that no longer address an atom.
    """

    by_site: dict[str, Style] = field(default_factory=dict)
    by_atom: dict[int, Style] = field(default_factory=dict)
    by_element: dict[str, Style] = field(default_factory=dict)

    # -- querying ----------------------------------------------------------
    def for_atom(self, atom_index: int, site_label: str,
                 element: str = "") -> Style:
        """The combined style for one atom: atom over site over element.

        The element is normalised on the way in, matching the setter. Scene atom
        elements are already normalised, so this only matters for a caller
        passing a raw symbol -- but a lookup that does not normalise where the
        setter does is the kind of asymmetry that works until it does not.
        """
        from . import elements

        style = Style()
        if atom_index in self.by_atom:
            style = self.by_atom[atom_index]
        if site_label in self.by_site:
            style = style.over(self.by_site[site_label])
        if element:
            symbol = elements.normalise(element)
            if symbol in self.by_element:
                style = style.over(self.by_element[symbol])
        return style

    @property
    def is_empty(self) -> bool:
        return not (self.by_site or self.by_atom or self.by_element)

    def count(self) -> int:
        return len(self.by_site) + len(self.by_atom) + len(self.by_element)

    # -- editing -----------------------------------------------------------
    def set_site(self, label: str, **fields) -> None:
        self.by_site[label] = replace(self.by_site.get(label, Style()),
                                      **fields)
        if self.by_site[label].is_empty and not self.by_site[label].note:
            del self.by_site[label]

    def set_atom(self, index: int, **fields) -> None:
        index = int(index)
        self.by_atom[index] = replace(self.by_atom.get(index, Style()),
                                      **fields)
        if self.by_atom[index].is_empty and not self.by_atom[index].note:
            del self.by_atom[index]

    def set_element(self, symbol: str, **fields) -> None:
        from . import elements

        symbol = elements.normalise(symbol)
        self.by_element[symbol] = replace(
            self.by_element.get(symbol, Style()), **fields)
        if (self.by_element[symbol].is_empty
                and not self.by_element[symbol].note):
            del self.by_element[symbol]

    def clear_site(self, label: str) -> None:
        self.by_site.pop(label, None)

    def clear_atom(self, index: int) -> None:
        self.by_atom.pop(int(index), None)

    def clear_element(self, symbol: str) -> None:
        from . import elements

        self.by_element.pop(elements.normalise(symbol), None)

    def clear(self) -> None:
        self.by_site.clear()
        self.by_atom.clear()
        self.by_element.clear()

    def prune(self, n_atoms: int, site_labels) -> None:
        """Drop overrides that no longer address anything.

        An atom override is keyed by scene index, which changes with the cell
        range; a site override is keyed by label, which changes with the file.
        Keeping a stale override would apply it to whatever atom happened to
        land on that index, which is worse than losing it.
        """
        labels = set(site_labels)
        self.by_site = {k: v for k, v in self.by_site.items() if k in labels}
        self.by_atom = {k: v for k, v in self.by_atom.items()
                        if 0 <= k < n_atoms}

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        def encode(style: Style) -> dict:
            out = {}
            if style.color is not None:
                out["color"] = list(style.color)
            if style.radius_scale is not None:
                out["radius_scale"] = style.radius_scale
            if style.visible is not None:
                out["visible"] = style.visible
            if style.label is not None:
                out["label"] = style.label
            if style.note:
                out["note"] = style.note
            return out

        return {
            "by_site": {k: encode(v) for k, v in self.by_site.items()},
            "by_atom": {str(k): encode(v) for k, v in self.by_atom.items()},
            "by_element": {k: encode(v) for k, v in self.by_element.items()},
        }

    @classmethod
    def from_dict(cls, data) -> "StyleOverrides":
        def decode(raw) -> Style:
            colour = raw.get("color")
            return Style(
                color=tuple(float(c) for c in colour) if colour else None,
                radius_scale=(float(raw["radius_scale"])
                              if "radius_scale" in raw else None),
                visible=(bool(raw["visible"]) if "visible" in raw else None),
                label=raw.get("label"),
                note=raw.get("note", ""))

        data = data or {}
        out = cls()
        for key, raw in (data.get("by_site") or {}).items():
            out.by_site[str(key)] = decode(raw)
        for key, raw in (data.get("by_atom") or {}).items():
            try:
                out.by_atom[int(key)] = decode(raw)
            except (TypeError, ValueError):
                continue
        for key, raw in (data.get("by_element") or {}).items():
            out.by_element[str(key)] = decode(raw)
        return out


def apply_to_scene(scene, overrides: StyleOverrides) -> int:
    """Apply overrides to a built scene. Returns how many atoms changed.

    Colour and radius are written in place. Hidden atoms are not removed: their
    radius is set to zero and their bonds are dropped, because removing them
    would renumber every atom and the override keys are atom indices. A zero
    radius draws nothing, and the arrays stay in step with the keys that
    address them.
    """
    import numpy as np

    if overrides is None or overrides.is_empty or scene.n_atoms == 0:
        return 0

    hidden = []
    touched = 0
    for index in range(scene.n_atoms):
        label = scene.atom_label[index] if index < len(scene.atom_label) else ""
        element = (scene.atom_element[index]
                   if index < len(scene.atom_element) else "")
        style = overrides.for_atom(index, label, element)
        if style.is_empty:
            continue
        touched += 1
        if style.color is not None:
            scene.atom_color[index] = np.asarray(style.color, np.float32)
        if style.radius_scale is not None:
            scene.atom_radius[index] = (scene.atom_radius[index]
                                        * float(style.radius_scale))
        if style.visible is False:
            hidden.append(index)

    if hidden:
        scene.atom_radius[hidden] = 0.0
        if scene.n_bonds:
            gone = np.zeros(scene.n_atoms, bool)
            gone[hidden] = True
            # A bond is dropped if either end is hidden. Judged on the named
            # atoms here rather than the drawn endpoints, because hiding is
            # about an atom, and an image is hidden with the atom it copies.
            pairs = scene.bond_atoms
            keep = ~(gone[pairs[:, 0]] | gone[pairs[:, 1]])
            for name in ("bond_a", "bond_b", "bond_radius", "bond_color_a",
                         "bond_color_b", "bond_valence", "bond_distance",
                         "bond_atoms", "_bond_base_a", "_bond_base_b"):
                array = getattr(scene, name)
                if len(array) == len(keep):
                    setattr(scene, name, array[keep])
    return touched
