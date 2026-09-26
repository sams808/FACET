"""The structure model.

Two levels, kept distinct throughout FACET:

* a :class:`Site` is a symmetry-distinct position in the asymmetric unit. It is
  what a result is reported for -- "the Bi1 site", one row of a table.
* an :class:`Atom` is one concrete position in space, produced by applying a
  symmetry operation and possibly a lattice translation. It is what gets drawn
  and what neighbour searching operates on.

Confusing the two is the usual source of quietly wrong coordination numbers, so
every Atom carries the index of the Site it came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import elements


@dataclass
class Site:
    """A symmetry-distinct crystallographic site."""

    label: str
    element: str
    frac: np.ndarray                     # fractional coordinates, shape (3,)
    occupancy: float = 1.0
    multiplicity: int | None = None
    wyckoff: str | None = None
    site_symmetry: str | None = None
    u_iso: float | None = None
    u_iso_esd: float | None = None
    frac_esd: np.ndarray | None = None   # esds on the coordinates, if given

    # Disorder, as the file declares it: which region of the cell has
    # alternatives, and which alternative this site belongs to. Empty means the
    # site is present in every configuration.
    disorder_assembly: str = ""
    disorder_group: str = ""

    # Oxidation state, and -- always -- where it came from.
    ox: int | None = None
    ox_source: str = "unset"             # cif | charge-balance | common | user

    def __post_init__(self):
        self.frac = np.asarray(self.frac, float).reshape(3)
        self.element = elements.normalise(self.element)

    @property
    def is_anion(self) -> bool:
        if self.ox is not None:
            return self.ox < 0
        return elements.is_anion_like(self.element)

    @property
    def display(self) -> str:
        if self.ox is None:
            return self.label
        return f"{self.label} ({self.element}{abs(self.ox)}{'+' if self.ox > 0 else '-'})"


@dataclass
class Atom:
    """One atom at one position, expanded from a Site."""

    element: str
    frac: np.ndarray
    cart: np.ndarray
    site_index: int
    label: str
    occupancy: float = 1.0
    image: tuple[int, int, int] = (0, 0, 0)

    @property
    def in_home_cell(self) -> bool:
        return self.image == (0, 0, 0)


@dataclass
class Cell:
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    orth: np.ndarray = field(default_factory=lambda: np.eye(3))

    @property
    def lengths(self) -> tuple[float, float, float]:
        return (self.a, self.b, self.c)

    @property
    def angles(self) -> tuple[float, float, float]:
        return (self.alpha, self.beta, self.gamma)

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.orth)))

    def to_cartesian(self, frac) -> np.ndarray:
        f = np.asarray(frac, float)
        return (self.orth @ f.T).T if f.ndim > 1 else self.orth @ f

    def to_fractional(self, cart) -> np.ndarray:
        inv = np.linalg.inv(self.orth)
        c = np.asarray(cart, float)
        return (inv @ c.T).T if c.ndim > 1 else inv @ c

    def __repr__(self) -> str:
        return (f"Cell({self.a:.4f}, {self.b:.4f}, {self.c:.4f}, "
                f"{self.alpha:.3f}, {self.beta:.3f}, {self.gamma:.3f})")


@dataclass
class Structure:
    """A parsed crystal structure, with its provenance and its health record."""

    name: str
    cell: Cell
    sites: list[Site]
    atoms: list[Atom]                       # the home cell, symmetry-expanded
    spacegroup_hm: str | None = None
    spacegroup_number: int | None = None
    source_path: str | None = None
    formula: str | None = None
    database_code: str | None = None
    reference: str | None = None
    year: int | None = None

    # Filled by facet.core.quality; a structure always knows how sound it is.
    issues: list = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # -- convenience ------------------------------------------------------
    @property
    def n_sites(self) -> int:
        return len(self.sites)

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def elements_present(self) -> list[str]:
        seen: dict[str, None] = {}
        for s in self.sites:
            seen.setdefault(s.element, None)
        return list(seen)

    @property
    def cation_sites(self) -> list[int]:
        return [i for i, s in enumerate(self.sites) if not s.is_anion]

    @property
    def anion_sites(self) -> list[int]:
        return [i for i, s in enumerate(self.sites) if s.is_anion]

    def site_by_label(self, label: str) -> int | None:
        for i, s in enumerate(self.sites):
            if s.label == label:
                return i
        return None

    def atoms_of_site(self, site_index: int) -> list[int]:
        return [i for i, a in enumerate(self.atoms)
                if a.site_index == site_index and a.in_home_cell]

    def cart_array(self) -> np.ndarray:
        return np.array([a.cart for a in self.atoms], float) if self.atoms \
            else np.zeros((0, 3))

    # -- chemistry --------------------------------------------------------
    def net_charge(self) -> float | None:
        """Sum of (occupancy x multiplicity x oxidation state) over the cell.

        None if any site has no assigned oxidation state. A structure that does
        not balance is not necessarily wrong -- it is usually a missing hydrogen
        or a disordered site -- but it is always worth saying so.
        """
        total = 0.0
        for s in self.sites:
            if s.ox is None:
                return None
            mult = s.multiplicity if s.multiplicity else 1
            total += s.ox * s.occupancy * mult
        return total

    def composition(self) -> dict[str, float]:
        """Element counts in the cell, weighted by occupancy and multiplicity."""
        out: dict[str, float] = {}
        for s in self.sites:
            mult = s.multiplicity if s.multiplicity else 1
            out[s.element] = out.get(s.element, 0.0) + s.occupancy * mult
        return out

    def __repr__(self) -> str:
        return (f"<Structure {self.name!r} {self.spacegroup_hm} "
                f"{self.n_sites} sites, {self.n_atoms} atoms>")
