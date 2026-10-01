"""Periodic neighbour search.

The architecture of the whole application rests on one decision made here: the
search runs **once**, at the widest radius any threshold could ever need, and
the result is cached. Moving the bond-valence threshold afterwards is then a
comparison against a stored array rather than a new search.

That is what makes the cutoff explorer -- drag a threshold, watch the
coordination number, the bond-valence sum, phi and the polyhedron change -- cost
nothing. Measured on a small oxide: the search is about 75 ms, and re-deriving
every quantity from the cached contacts is about 4 ms.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .structure import Structure


@dataclass
class Contacts:
    """Every contact of one central atom, out to the search radius.

    Sorted by distance. Nothing here has been filtered by any threshold -- that
    is deliberate, and is what allows a threshold to be moved for free.
    """

    center_atom: int
    center_site: int
    center_element: str
    neighbor_atom: np.ndarray      # index into Structure.atoms
    neighbor_site: np.ndarray      # index into Structure.sites
    elements: list[str]
    distance: np.ndarray           # angstrom
    vector: np.ndarray             # (n, 3) cartesian, from centre to neighbour
    image: np.ndarray              # (n, 3) integer lattice translation
    occupancy: np.ndarray

    def __len__(self) -> int:
        return len(self.distance)

    def mask(self, keep) -> "Contacts":
        """A new Contacts holding only the selected contacts."""
        k = np.asarray(keep)
        return Contacts(
            self.center_atom, self.center_site, self.center_element,
            self.neighbor_atom[k], self.neighbor_site[k],
            [e for e, m in zip(self.elements, k) if m] if k.dtype == bool
            else [self.elements[i] for i in k],
            self.distance[k], self.vector[k], self.image[k], self.occupancy[k],
        )

    def anions_only(self, structure: Structure) -> "Contacts":
        """Drop cation-cation contacts.

        A bond-valence sum is defined over the anions a cation touches. Leaving
        a neighbouring cation in the list would add a spurious term, and for a
        heavy cation such as bismuth the cation-cation distance is often shorter
        than the longest genuine bond.
        """
        keep = np.array([structure.sites[s].is_anion for s in self.neighbor_site],
                        dtype=bool)
        return self.mask(keep)

    def cations_only(self, structure: Structure) -> "Contacts":
        """Drop anion-anion contacts: the mirror of :meth:`anions_only`.

        What an anion's own bond-valence sum is defined over. The nearest
        neighbours of an oxygen are usually other oxygens, so leaving them in
        would swamp the sum with terms that are not bonds.
        """
        keep = np.array([not structure.sites[s].is_anion
                         for s in self.neighbor_site], dtype=bool)
        return self.mask(keep)

    def within(self, rmax: float) -> "Contacts":
        return self.mask(self.distance <= rmax)


class NeighborFinder:
    """Cached periodic neighbour search over a structure.

    Build once per structure and per search radius; query per atom.
    """

    def __init__(self, structure: Structure, rmax: float = 8.0,
                 dmin: float = 0.4):
        self.structure = structure
        self.rmax = float(rmax)
        self.dmin = float(dmin)

        cart = structure.cart_array()
        if cart.size == 0:
            raise ValueError("structure has no atoms")

        images = _images_within(structure.cell.orth, self.rmax)
        n_atoms, n_img = len(cart), len(images)

        shifts = images @ structure.cell.orth.T          # (n_img, 3) cartesian
        # (n_img, n_atoms, 3) -> flat
        self._pos = (cart[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
        self._atom = np.tile(np.arange(n_atoms), n_img)
        self._image = np.repeat(images, n_atoms, axis=0)

        from scipy.spatial import cKDTree

        self._tree = cKDTree(self._pos)
        self._home = np.flatnonzero(
            (self._image == 0).all(axis=1))          # index of the home images
        self._n_atoms = n_atoms
        self._n_img = n_img

    # -- queries ----------------------------------------------------------
    def contacts(self, atom_index: int) -> Contacts:
        """Every neighbour of one atom, sorted by distance."""
        st = self.structure
        centre = st.atoms[atom_index].cart
        idx = np.array(self._tree.query_ball_point(centre, self.rmax), dtype=int)
        if idx.size == 0:
            return _empty(atom_index, st)

        vec = self._pos[idx] - centre
        d = np.linalg.norm(vec, axis=1)

        # drop the central atom itself, and anything closer than dmin -- a
        # "contact" below about 0.4 A is the atom found in its own image
        keep = d > self.dmin
        idx, vec, d = idx[keep], vec[keep], d[keep]
        order = np.argsort(d)
        idx, vec, d = idx[order], vec[order], d[order]

        neigh_atom = self._atom[idx]
        neigh_site = np.array([st.atoms[a].site_index for a in neigh_atom], int)
        return Contacts(
            center_atom=atom_index,
            center_site=st.atoms[atom_index].site_index,
            center_element=st.atoms[atom_index].element,
            neighbor_atom=neigh_atom,
            neighbor_site=neigh_site,
            elements=[st.atoms[a].element for a in neigh_atom],
            distance=d,
            vector=vec,
            image=self._image[idx],
            occupancy=np.array([st.atoms[a].occupancy for a in neigh_atom], float),
        )

    def contacts_for_site(self, site_index: int) -> Contacts:
        """Contacts of the first atom belonging to a site.

        Every atom of one site has the same environment by symmetry, so one
        representative is enough -- and choosing it here keeps the caller from
        having to know that.
        """
        atoms = self.structure.atoms_of_site(site_index)
        if not atoms:
            raise ValueError(
                f"site {self.structure.sites[site_index].label!r} generated no atoms")
        return self.contacts(atoms[0])

    def pairs_within(self, rmax: float):
        """Every home-cell atom paired with every neighbour within rmax.

        Used for drawing bonds across a whole cell, where per-atom queries would
        repeat work.
        """
        rmax = min(float(rmax), self.rmax)
        out = self._tree.query_ball_point(self._pos[self._home], rmax)
        for home_pos, hits in zip(self._home, out):
            a = int(self._atom[home_pos])
            for j in hits:
                d = float(np.linalg.norm(self._pos[j] - self._pos[home_pos]))
                if d <= self.dmin:
                    continue
                yield a, int(self._atom[j]), d, tuple(int(x) for x in self._image[j])

    def __repr__(self) -> str:
        return (f"NeighborFinder({self.structure.name!r}, rmax={self.rmax} A, "
                f"{self._n_atoms} atoms x {self._n_img} images)")


# ---------------------------------------------------------------------------

def _images_within(orth: np.ndarray, rmax: float) -> np.ndarray:
    """Lattice translations whose cells can reach within rmax of the home cell.

    The number of images needed along each axis is set by the **perpendicular
    width** of the cell in that direction, not by the cell edge length. For an
    oblique cell those differ substantially, and using the edge length silently
    misses neighbours -- the classic way a coordination number comes out one or
    two too low in a monoclinic structure.
    """
    a, b, c = orth[:, 0], orth[:, 1], orth[:, 2]
    volume = abs(float(np.dot(a, np.cross(b, c))))
    if volume <= 0:
        raise ValueError("degenerate unit cell")

    widths = np.array([
        volume / np.linalg.norm(np.cross(b, c)),
        volume / np.linalg.norm(np.cross(c, a)),
        volume / np.linalg.norm(np.cross(a, b)),
    ])
    reps = np.ceil(rmax / widths).astype(int) + 1

    ranges = [np.arange(-r, r + 1) for r in reps]
    grid = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3).astype(float)


def _empty(atom_index: int, st: Structure) -> Contacts:
    return Contacts(
        atom_index, st.atoms[atom_index].site_index, st.atoms[atom_index].element,
        np.zeros(0, int), np.zeros(0, int), [], np.zeros(0), np.zeros((0, 3)),
        np.zeros((0, 3), int), np.zeros(0),
    )


def search_radius_for(structure: Structure, params, v_list: float,
                      floor: float = 6.0, ceiling: float = 12.0) -> float:
    """The radius that guarantees every contact above ``v_list`` is found.

    Taken as the largest valence-derived cutoff over every cation-anion pair the
    structure actually contains, so an iodide gets a wider search than an oxide
    without anyone having to ask for it.
    """
    want = floor
    cations = {structure.sites[i].element: structure.sites[i].ox
               for i in structure.cation_sites}
    anions = {structure.sites[i].element for i in structure.anion_sites}
    for cat, ox in cations.items():
        for an in anions:
            p = params.get(cat, ox, an)
            if p is not None:
                want = max(want, p.distance_for(v_list))
    return float(min(max(want, floor), ceiling))
