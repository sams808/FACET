"""Five independent checks on every geometric quantity FACET computes.

The bond-construction bug — propagating a representative atom's contact vectors
to its symmetry copies by translation — passed every test in the suite while
being wrong for nearly every structure. It was caught by looking at a picture.
That is not a method, so this file is the method.

Five layers, each able to catch what the others cannot:

1. **Analytic.** Shapes whose answers are known in closed form: a cubic cell, a
   regular octahedron, a regular tetrahedron. Any disagreement is a bug, with
   no tolerance argument available.
2. **Independent recomputation.** The same quantities computed a second time by
   brute force, sharing no code with the engine — explicit loops over lattice
   translations rather than a KD-tree, explicit dot products rather than the
   polyhedron routines.
3. **Invariance.** Physical results cannot depend on arbitrary choices. Shift
   the origin, rotate the whole crystal, permute the axes, expand to a
   supercell, or pick a different symmetry copy as the representative — every
   distance, angle, coordination number and bond-valence sum must be unchanged.
4. **External cross-check.** gemmi and spglib compute cell geometry and
   symmetry by entirely separate code paths.
5. **Real data.** Every structure in the reference collection, checked for
   physical impossibilities rather than for agreement with anything.
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pytest

from facet.core import bv, cif, coordination, elements, polyhedra, utilities
from facet.core.neighbors import NeighborFinder, _images_within
from facet.core.structure import Atom, Cell, Site, Structure

CIFS = Path(r"C:\Users\samso\Desktop\WSU_work\XRD\cif\Bi\cifs")
SAMPLE = CIFS / "1526458_Bi2O3.cif"


# ---------------------------------------------------------------------------
# helpers: build a structure by hand, with no CIF involved
# ---------------------------------------------------------------------------

def cell_from_parameters(a, b, c, alpha, beta, gamma) -> Cell:
    """The standard crystallographic orthogonalisation.

    Written out here rather than imported so that layer 2 does not lean on the
    same matrix the engine uses.
    """
    al, be, ga = (math.radians(x) for x in (alpha, beta, gamma))
    ca, cb, cg = math.cos(al), math.cos(be), math.cos(ga)
    sg = math.sin(ga)
    volume = a * b * c * math.sqrt(
        1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg)
    orth = np.array([
        [a, b * cg, c * cb],
        [0.0, b * sg, c * (ca - cb * cg) / sg],
        [0.0, 0.0, volume / (a * b * sg)],
    ])
    return Cell(a, b, c, alpha, beta, gamma, orth)


def make_structure(cell: Cell, atoms: list[tuple[str, tuple[float, float, float]]],
                   name: str = "test") -> Structure:
    sites, expanded = [], []
    for i, (element, frac) in enumerate(atoms):
        frac = np.asarray(frac, float)
        sites.append(Site(f"{element}{i + 1}", element, frac, multiplicity=1))
        expanded.append(Atom(element=elements.normalise(element), frac=frac,
                             cart=cell.to_cartesian(frac), site_index=i,
                             label=f"{element}{i + 1}"))
    s = Structure(name=name, cell=cell, sites=sites, atoms=expanded)
    for site in s.sites:
        site.ox = elements.COMMON_OX.get(site.element)
        site.ox_source = "common"
    return s


def rocksalt(a: float = 4.2) -> Structure:
    """NaCl. Each cation sits at the centre of a regular octahedron of anions
    at exactly a/2, and the next-nearest neighbours are at a/sqrt(2)."""
    cell = cell_from_parameters(a, a, a, 90, 90, 90)
    atoms = []
    for frac in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
        atoms.append(("Na", frac))
    for frac in [(0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5), (0.5, 0.5, 0.5)]:
        atoms.append(("Cl", frac))
    return make_structure(cell, atoms, "rocksalt")


# ===========================================================================
# LAYER 1 — analytic
# ===========================================================================

class TestLayer1Analytic:
    """Answers known in closed form. No tolerance argument is available."""

    def test_cubic_orthogonalisation_is_the_identity_times_a(self):
        cell = cell_from_parameters(5.0, 5.0, 5.0, 90, 90, 90)
        assert np.allclose(cell.orth, np.eye(3) * 5.0, atol=1e-12)

    def test_cell_volume_matches_the_closed_form(self):
        a, b, c, al, be, ga = 5.8444, 8.1574, 7.5032, 90.0, 112.97, 90.0
        cell = cell_from_parameters(a, b, c, al, be, ga)
        ca, cb, cg = (math.cos(math.radians(x)) for x in (al, be, ga))
        want = a * b * c * math.sqrt(
            1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg)
        assert cell.volume == pytest.approx(want, rel=1e-12)

    def test_fractional_and_cartesian_are_exact_inverses(self):
        cell = cell_from_parameters(4.1, 7.3, 9.9, 71.0, 103.0, 118.0)
        rng = np.random.default_rng(0)
        for frac in rng.random((200, 3)):
            back = cell.to_fractional(cell.to_cartesian(frac))
            assert np.allclose(back, frac, atol=1e-12)

    def test_rocksalt_first_shell_is_exactly_a_over_two(self):
        a = 4.2
        s = rocksalt(a)
        finder = NeighborFinder(s, rmax=5.0)
        contacts = finder.contacts_for_site(0).anions_only(s)
        first = np.sort(contacts.distance)[:6]
        assert np.allclose(first, a / 2.0, atol=1e-12)
        # and the seventh contact is much further out
        assert contacts.distance[6] > a / 2.0 * 1.3

    def test_rocksalt_octahedron_has_exactly_the_right_angles(self):
        s = rocksalt()
        results = coordination.analyse_structure(s)
        site = results[0]
        assert site.cn_valence == 6
        angles = sorted(round(r.angle, 9) for r in utilities.bond_angles(site))
        assert angles.count(90.0) == 12
        assert angles.count(180.0) == 3

    def test_a_regular_octahedron_has_zero_angle_variance(self):
        v = np.array([[2.0, 0, 0], [-2.0, 0, 0], [0, 2.0, 0],
                      [0, -2.0, 0], [0, 0, 2.0], [0, 0, -2.0]])
        out = polyhedra.shape(v)
        assert out["angle_variance"] == pytest.approx(0.0, abs=1e-18)
        assert out["spread"] == pytest.approx(0.0, abs=1e-15)
        assert out["eccentricity"] == pytest.approx(0.0, abs=1e-15)

    def test_octahedron_hull_volume_matches_the_closed_form(self):
        r = 2.0
        v = np.array([[r, 0, 0], [-r, 0, 0], [0, r, 0],
                      [0, -r, 0], [0, 0, r], [0, 0, -r]])
        # a regular octahedron of circumradius r has volume 4/3 r^3
        assert polyhedra.shape(v)["volume"] == pytest.approx(
            4.0 / 3.0 * r ** 3, rel=1e-9)

    def test_quadratic_elongation_of_a_regular_octahedron_is_one(self):
        r = 2.0
        v = np.array([[r, 0, 0], [-r, 0, 0], [0, r, 0],
                      [0, -r, 0], [0, 0, r], [0, 0, -r]])
        assert polyhedra.shape(v)["quadratic_elongation"] == pytest.approx(
            1.0, rel=1e-9)

    @pytest.mark.parametrize("vectors,expected", [
        # regular octahedron
        ([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
         54.735610317),
        # regular tetrahedron
        ([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], 70.528779366),
        # trigonal planar
        ([[1, 0, 0], [-0.5, math.sqrt(3) / 2, 0], [-0.5, -math.sqrt(3) / 2, 0]],
         90.0),
    ])
    def test_void_cone_matches_the_analytic_half_angles(self, vectors, expected):
        """The largest ligand-free cone of a regular polyhedron is exact."""
        angle, _ = polyhedra.void_cone(np.array(vectors, float))
        assert angle == pytest.approx(expected, abs=0.15)

    def test_void_cone_of_a_single_bond_is_a_hemisphere_and_more(self):
        angle, axis = polyhedra.void_cone(np.array([[0.0, 0.0, 1.0]]))
        assert angle == pytest.approx(180.0, abs=0.5)
        assert np.dot(axis, [0, 0, 1]) < -0.99      # points away from the bond

    def test_effective_cn_of_a_regular_polyhedron_is_its_vertex_count(self):
        for n, d in ((4, 1.9), (6, 2.1), (8, 2.4)):
            ecn, dav = polyhedra.effective_cn(np.full(n, d))
            assert ecn == pytest.approx(n, rel=1e-9)
            assert dav == pytest.approx(d, rel=1e-12)

    def test_phi_is_zero_for_any_centrosymmetric_set(self):
        rng = np.random.default_rng(3)
        half = rng.normal(size=(5, 3))
        vectors = np.vstack([half, -half])
        _, phi, _ = bv.phi_index(vectors, np.ones(10))
        assert phi == pytest.approx(0.0, abs=1e-15)

    def test_valence_fraction_of_a_regular_octahedron_is_exactly_one_half(self):
        """The identity f3 = 0.5 for a regular octahedron, which is what makes
        f3 - 0.5 an absolute measure of shell splitting."""
        assert polyhedra.valence_fraction_in_shortest(np.full(6, 0.5), 3) == \
            pytest.approx(0.5, rel=1e-15)

    def test_cubic_d_spacings_match_the_closed_form_exactly(self):
        a = 5.0
        s = make_structure(cell_from_parameters(a, a, a, 90, 90, 90),
                           [("Na", (0, 0, 0))])
        for h, k, l in itertools.product(range(4), repeat=3):
            if (h, k, l) == (0, 0, 0):
                continue
            want = a / math.sqrt(h * h + k * k + l * l)
            assert utilities.d_spacing(s, h, k, l) == pytest.approx(want, rel=1e-12)

    def test_hexagonal_d_spacings_match_the_closed_form(self):
        a, c = 3.0, 5.0
        s = make_structure(cell_from_parameters(a, a, c, 90, 90, 120),
                           [("Na", (0, 0, 0))])
        for h, k, l in [(1, 0, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (2, 1, 3)]:
            inv2 = 4.0 / 3.0 * (h * h + h * k + k * k) / (a * a) + l * l / (c * c)
            assert utilities.d_spacing(s, h, k, l) == pytest.approx(
                1.0 / math.sqrt(inv2), rel=1e-11)

    def test_reciprocal_of_a_cubic_cell_is_one_over_a(self):
        s = make_structure(cell_from_parameters(4.0, 4.0, 4.0, 90, 90, 90),
                           [("Na", (0, 0, 0))])
        rec = utilities.reciprocal_cell(s)
        for key in ("a*", "b*", "c*"):
            assert rec[key] == pytest.approx(0.25, rel=1e-12)

    def test_reciprocal_lattice_is_dual_to_the_direct_one(self):
        """a_i* . a_j = delta_ij, the defining property."""
        cell = cell_from_parameters(4.1, 7.3, 9.9, 71.0, 103.0, 118.0)
        orth = cell.orth
        a, b, c = orth[:, 0], orth[:, 1], orth[:, 2]
        volume = float(np.dot(a, np.cross(b, c)))
        star = [np.cross(b, c) / volume, np.cross(c, a) / volume,
                np.cross(a, b) / volume]
        for i in range(3):
            for j in range(3):
                want = 1.0 if i == j else 0.0
                assert float(np.dot(star[i], orth[:, j])) == pytest.approx(
                    want, abs=1e-12)


# ===========================================================================
# LAYER 2 — independent recomputation, sharing no code with the engine
# ===========================================================================

def brute_force_contacts(structure: Structure, atom_index: int,
                         rmax: float) -> list[tuple[int, float, tuple]]:
    """Every neighbour within rmax, by explicit loops over lattice images.

    No KD-tree, no cached image list, no engine helpers. Deliberately the
    slowest possible implementation, because slow and obvious is what an
    independent check needs to be.
    """
    orth = structure.cell.orth
    centre = structure.atoms[atom_index].cart

    # enough images in every direction that nothing within rmax can be missed
    reach = int(math.ceil(rmax / min(structure.cell.a, structure.cell.b,
                                     structure.cell.c))) + 2
    out = []
    for i in range(-reach, reach + 1):
        for j in range(-reach, reach + 1):
            for k in range(-reach, reach + 1):
                shift = orth @ np.array([i, j, k], float)
                for n, atom in enumerate(structure.atoms):
                    d = float(np.linalg.norm(atom.cart + shift - centre))
                    if 0.4 < d <= rmax:
                        out.append((n, d, (i, j, k)))
    out.sort(key=lambda t: t[1])
    return out


class TestLayer2Independent:
    """A second implementation, written to be obvious rather than fast."""

    @pytest.mark.parametrize("params", [
        (4.2, 4.2, 4.2, 90, 90, 90),              # cubic
        (5.8444, 8.1574, 7.5032, 90, 112.97, 90),  # monoclinic, real
        (4.1, 7.3, 9.9, 71.0, 103.0, 118.0),      # triclinic, oblique
        (3.0, 3.0, 20.0, 90, 90, 120),            # hexagonal, very anisotropic
    ])
    def test_neighbour_distances_match_a_brute_force_search(self, params):
        cell = cell_from_parameters(*params)
        rng = np.random.default_rng(7)
        atoms = [("Na", tuple(rng.random(3))) for _ in range(3)]
        atoms += [("Cl", tuple(rng.random(3))) for _ in range(3)]
        s = make_structure(cell, atoms)

        rmax = 5.5
        finder = NeighborFinder(s, rmax=rmax)
        for index in range(len(s.atoms)):
            mine = np.sort(finder.contacts(index).distance)
            theirs = np.sort([d for _, d, _ in
                              brute_force_contacts(s, index, rmax)])
            assert len(mine) == len(theirs), (
                f"atom {index}: engine found {len(mine)}, brute force "
                f"{len(theirs)}")
            assert np.allclose(mine, theirs, atol=1e-9)

    def test_image_enumeration_never_misses_a_neighbour(self):
        """The perpendicular-width calculation is the subtle part: using the
        cell edge length instead silently misses neighbours in an oblique
        cell."""
        cell = cell_from_parameters(4.0, 4.0, 4.0, 30.0, 30.0, 30.0)
        s = make_structure(cell, [("Na", (0.0, 0.0, 0.0)),
                                  ("Cl", (0.5, 0.5, 0.5))])
        rmax = 8.0
        finder = NeighborFinder(s, rmax=rmax)
        mine = np.sort(finder.contacts(0).distance)
        theirs = np.sort([d for _, d, _ in brute_force_contacts(s, 0, rmax)])
        assert len(mine) == len(theirs)
        assert np.allclose(mine, theirs, atol=1e-9)

    def test_perpendicular_widths_bound_the_image_range(self):
        """Every lattice translation that could reach within rmax must be in
        the enumerated set."""
        for params in [(4.0, 4.0, 4.0, 90, 90, 90),
                       (4.0, 9.0, 4.0, 90, 140.0, 90),
                       (3.0, 3.0, 3.0, 40.0, 50.0, 60.0)]:
            cell = cell_from_parameters(*params)
            rmax = 7.0
            images = {tuple(int(x) for x in row)
                      for row in _images_within(cell.orth, rmax)}
            for i, j, k in itertools.product(range(-4, 5), repeat=3):
                shift = cell.orth @ np.array([i, j, k], float)
                if np.linalg.norm(shift) <= rmax:
                    assert (i, j, k) in images, (
                        f"{params}: translation {(i, j, k)} reaches "
                        f"{np.linalg.norm(shift):.2f} A but was not enumerated")

    def test_bond_angles_match_explicit_dot_products(self, ):
        s = rocksalt()
        results = coordination.analyse_structure(s)
        site = results[0]
        rows = utilities.bond_angles(site)
        bonded = site.bonds
        expected = []
        for i in range(len(bonded)):
            for j in range(i + 1, len(bonded)):
                u = np.asarray(bonded[i].vector, float)
                v = np.asarray(bonded[j].vector, float)
                cos = float(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
                expected.append(math.degrees(math.acos(min(1.0, max(-1.0, cos)))))
        assert np.allclose(sorted(r.angle for r in rows), sorted(expected),
                           atol=1e-9)

    def test_bond_valence_sum_matches_an_explicit_exponential_sum(self):
        s = rocksalt()
        results = coordination.analyse_structure(s)
        site = results[0]
        param = next(c.param for c in site.contacts if c.param)
        manual = sum(math.exp((param.r0 - c.distance) / param.b)
                     for c in site.contacts
                     if c.valence is not None and c.valence > site.v_bond)
        assert site.bvs == pytest.approx(manual, rel=1e-12)

    def test_phi_matches_an_explicit_vector_sum(self):
        s = cif.read(SAMPLE) if SAMPLE.exists() else rocksalt()
        results = coordination.analyse_structure(s)
        for site in results:
            bonded = site.bonds
            if not bonded:
                continue
            total = np.zeros(3)
            magnitude = 0.0
            for c in bonded:
                v = np.asarray(c.vector, float)
                total += c.valence * c.occupancy * v / np.linalg.norm(v)
                magnitude += c.valence * c.occupancy
            assert site.phi == pytest.approx(
                float(np.linalg.norm(total)) / magnitude, rel=1e-10)

    def test_polyhedron_volume_matches_a_tetrahedral_decomposition(self):
        """Hull volume, recomputed by summing signed tetrahedra from the
        centroid — a different algorithm to the same number."""
        from scipy.spatial import ConvexHull

        rng = np.random.default_rng(11)
        for _ in range(6):
            pts = rng.normal(size=(8, 3)) * 2.0
            hull = ConvexHull(pts)
            centroid = pts.mean(axis=0)
            total = 0.0
            for simplex in hull.simplices:
                a, b, c = pts[simplex] - centroid
                total += abs(float(np.dot(a, np.cross(b, c)))) / 6.0
            assert polyhedra.shape(pts)["volume"] == pytest.approx(
                total, rel=1e-9)


# ===========================================================================
# LAYER 3 — invariance: no result may depend on an arbitrary choice
# ===========================================================================

def _fingerprint(results, v_bond: float) -> list[tuple]:
    """A comparable summary of an analysis, independent of site ordering."""
    out = []
    for r in results:
        out.append((
            r.element,
            r.cn_at(v_bond),
            round(r.bvs, 9),
            round(r.phi, 9) if r.phi == r.phi else None,
            round(r.shape.get("d_mean") or 0.0, 9),
            round(r.shape.get("spread") or 0.0, 9),
            tuple(round(c.distance, 9) for c in r.bonds),
        ))
    return sorted(out)


class TestLayer3Invariance:
    """A physical result cannot depend on how the crystal was written down."""

    @pytest.fixture(scope="class")
    def base(self):
        if SAMPLE.exists():
            return cif.read(SAMPLE)
        return rocksalt()

    def test_shifting_the_origin_changes_nothing(self, base):
        shifted = Structure(
            name=base.name, cell=base.cell,
            sites=[Site(s.label, s.element, (s.frac + 0.137) % 1.0,
                        s.occupancy, s.multiplicity) for s in base.sites],
            atoms=[Atom(a.element, (a.frac + 0.137) % 1.0,
                        base.cell.to_cartesian((a.frac + 0.137) % 1.0),
                        a.site_index, a.label, a.occupancy)
                   for a in base.atoms])
        for s in shifted.sites:
            s.ox = elements.COMMON_OX.get(s.element)
            s.ox_source = "common"
        assert _fingerprint(coordination.analyse_structure(shifted), 0.075) == \
            _fingerprint(coordination.analyse_structure(base), 0.075)

    def test_rotating_the_whole_crystal_changes_nothing(self, base):
        """Distances and angles are rotation invariant; a rigid rotation of the
        lattice vectors must leave every reported number alone."""
        theta = 0.7
        rot = np.array([[math.cos(theta), -math.sin(theta), 0.0],
                        [math.sin(theta), math.cos(theta), 0.0],
                        [0.0, 0.0, 1.0]])
        cell = Cell(base.cell.a, base.cell.b, base.cell.c, base.cell.alpha,
                    base.cell.beta, base.cell.gamma, rot @ base.cell.orth)
        rotated = Structure(
            name=base.name, cell=cell,
            sites=[Site(s.label, s.element, s.frac, s.occupancy,
                        s.multiplicity) for s in base.sites],
            atoms=[Atom(a.element, a.frac, cell.to_cartesian(a.frac),
                        a.site_index, a.label, a.occupancy)
                   for a in base.atoms])
        for s in rotated.sites:
            s.ox = elements.COMMON_OX.get(s.element)
            s.ox_source = "common"
        assert _fingerprint(coordination.analyse_structure(rotated), 0.075) == \
            _fingerprint(coordination.analyse_structure(base), 0.075)

    def test_a_supercell_reports_the_same_environments(self, base):
        """Expanding to 2x1x1 duplicates every site; the set of distinct
        environments must be unchanged."""
        cell = base.cell
        big_orth = cell.orth.copy()
        big_orth[:, 0] *= 2
        big = Cell(cell.a * 2, cell.b, cell.c, cell.alpha, cell.beta,
                   cell.gamma, big_orth)

        sites, atoms = [], []
        for shift in (0.0, 0.5):
            for i, a in enumerate(base.atoms):
                frac = np.array([a.frac[0] / 2.0 + shift, a.frac[1], a.frac[2]])
                index = len(sites)
                sites.append(Site(f"{a.label}_{shift}", a.element, frac,
                                  a.occupancy, 1))
                atoms.append(Atom(a.element, frac, big.to_cartesian(frac),
                                  index, f"{a.label}_{shift}", a.occupancy))
        super_cell = Structure(name="super", cell=big, sites=sites, atoms=atoms)
        for s in super_cell.sites:
            s.ox = elements.COMMON_OX.get(s.element)
            s.ox_source = "common"

        got = {f[:6] for f in _fingerprint(
            coordination.analyse_structure(super_cell), 0.075)}
        want = {f[:6] for f in _fingerprint(
            coordination.analyse_structure(base), 0.075)}
        assert got == want

    def test_every_symmetry_copy_of_a_site_has_the_same_environment(self, base):
        """The bug that started this file. Each atom of one site must report
        the same distances, whichever copy is asked."""
        finder = NeighborFinder(base, rmax=6.0)
        for site_index in range(base.n_sites):
            atoms = base.atoms_of_site(site_index)
            if len(atoms) < 2:
                continue
            reference = None
            for atom_index in atoms:
                d = np.sort(finder.contacts(atom_index).distance)[:12]
                if reference is None:
                    reference = d
                else:
                    assert np.allclose(d, reference, atol=1e-9), (
                        f"site {base.sites[site_index].label}: copies disagree")

    def test_permuting_the_cell_axes_changes_nothing(self):
        """Relabelling a, b and c is a change of description, not of crystal."""
        rng = np.random.default_rng(5)
        atoms = [("Na", tuple(rng.random(3))) for _ in range(2)]
        atoms += [("Cl", tuple(rng.random(3))) for _ in range(2)]
        a, b, c = 4.3, 6.1, 7.7
        base = make_structure(cell_from_parameters(a, b, c, 90, 90, 90), atoms)

        permuted_atoms = [(e, (f[2], f[0], f[1])) for e, f in atoms]
        permuted = make_structure(
            cell_from_parameters(c, a, b, 90, 90, 90), permuted_atoms)

        assert _fingerprint(coordination.analyse_structure(permuted), 0.075) == \
            _fingerprint(coordination.analyse_structure(base), 0.075)

    def test_the_search_radius_does_not_change_the_bonded_set(self, base):
        """Widening the search must add distant contacts, never alter the
        close ones."""
        near = coordination.analyse_structure(base, v_list=0.02)
        far = coordination.analyse_structure(base, v_list=0.002)
        for a, b in zip(near, far):
            assert a.cn_valence == b.cn_valence
            assert a.bvs == pytest.approx(b.bvs, rel=1e-12)
            assert [round(c.distance, 9) for c in a.bonds] == \
                   [round(c.distance, 9) for c in b.bonds]

    def test_scaling_the_cell_scales_every_distance(self):
        """Doubling the lattice must double every distance exactly."""
        atoms = [("Na", (0.0, 0.0, 0.0)), ("Cl", (0.5, 0.5, 0.5))]
        small = make_structure(cell_from_parameters(4.0, 4.0, 4.0, 90, 90, 90),
                               atoms)
        big = make_structure(cell_from_parameters(8.0, 8.0, 8.0, 90, 90, 90),
                             atoms)
        d_small = np.sort(NeighborFinder(small, 7.0).contacts(0).distance)[:6]
        d_big = np.sort(NeighborFinder(big, 14.0).contacts(0).distance)[:6]
        assert np.allclose(d_big, 2.0 * d_small, atol=1e-9)

    def test_bond_keys_are_symmetric(self):
        """The same contact seen from either end must give one key."""
        from facet.gl.scene import _bond_key

        rng = np.random.default_rng(2)
        for _ in range(200):
            i, j = (int(x) for x in rng.integers(0, 40, 2))
            image = tuple(int(x) for x in rng.integers(-2, 3, 3))
            forward = _bond_key(i, j, image)
            backward = _bond_key(j, i, tuple(-x for x in image))
            assert forward == backward


# ===========================================================================
# LAYER 4 — cross-check against gemmi and spglib
# ===========================================================================

class TestLayer4External:

    @pytest.mark.skipif(not SAMPLE.exists(), reason="sample not present")
    def test_cell_volume_agrees_with_gemmi(self):
        import gemmi

        s = cif.read(SAMPLE)
        g = gemmi.UnitCell(s.cell.a, s.cell.b, s.cell.c,
                           s.cell.alpha, s.cell.beta, s.cell.gamma)
        assert s.cell.volume == pytest.approx(g.volume, rel=1e-9)

    @pytest.mark.skipif(not SAMPLE.exists(), reason="sample not present")
    def test_cartesian_coordinates_agree_with_gemmi(self):
        import gemmi

        s = cif.read(SAMPLE)
        g = gemmi.UnitCell(s.cell.a, s.cell.b, s.cell.c,
                           s.cell.alpha, s.cell.beta, s.cell.gamma)
        for atom in s.atoms[:20]:
            p = g.orthogonalize(gemmi.Fractional(*atom.frac))
            assert np.allclose(atom.cart, [p.x, p.y, p.z], atol=1e-9)

    @pytest.mark.skipif(not SAMPLE.exists(), reason="sample not present")
    def test_distances_agree_with_gemmi_nearest_image(self):
        import gemmi

        s = cif.read(SAMPLE)
        g = gemmi.UnitCell(s.cell.a, s.cell.b, s.cell.c,
                           s.cell.alpha, s.cell.beta, s.cell.gamma)
        finder = NeighborFinder(s, rmax=6.0)
        contacts = finder.contacts(0)
        centre = gemmi.Fractional(*s.atoms[0].frac)
        for n in range(min(10, len(contacts))):
            other = gemmi.Fractional(*s.atoms[contacts.neighbor_atom[n]].frac)
            nearest = g.find_nearest_pbc_image(centre, other, 0)
            # the engine's contact is at least as close as the nearest image
            assert contacts.distance[n] >= nearest.dist() - 1e-6

    @pytest.mark.skipif(not SAMPLE.exists(), reason="sample not present")
    def test_spglib_agrees_on_the_number_of_distinct_sites(self):
        import spglib

        s = cif.read(SAMPLE)
        lattice = s.cell.orth.T
        positions = np.array([a.frac for a in s.atoms])
        numbers = [elements.info(a.element).z for a in s.atoms]
        dataset = spglib.get_symmetry_dataset((lattice, positions, numbers),
                                              symprec=1e-3)
        assert dataset is not None
        equivalent = list(dataset.equivalent_atoms)
        assert len(set(equivalent)) == s.n_sites

    def test_metric_tensor_reproduces_the_cell_parameters(self):
        """G_ii = a_i^2 and G_ij = a_i a_j cos(angle) — the definition."""
        a, b, c, al, be, ga = 4.1, 7.3, 9.9, 71.0, 103.0, 118.0
        s = make_structure(cell_from_parameters(a, b, c, al, be, ga),
                           [("Na", (0, 0, 0))])
        g = utilities.metric_tensor(s)
        assert math.sqrt(g[0, 0]) == pytest.approx(a, rel=1e-12)
        assert math.sqrt(g[1, 1]) == pytest.approx(b, rel=1e-12)
        assert math.sqrt(g[2, 2]) == pytest.approx(c, rel=1e-12)
        assert math.degrees(math.acos(g[1, 2] / (b * c))) == pytest.approx(
            al, rel=1e-10)
        assert math.degrees(math.acos(g[0, 2] / (a * c))) == pytest.approx(
            be, rel=1e-10)
        assert math.degrees(math.acos(g[0, 1] / (a * b))) == pytest.approx(
            ga, rel=1e-10)


# ===========================================================================
# LAYER 5 — every real structure, checked for impossibilities
# ===========================================================================

@pytest.mark.skipif(not CIFS.exists(), reason="reference collection not present")
class TestLayer5RealData:

    @pytest.fixture(scope="class")
    def analysed(self):
        out = []
        for path in sorted(CIFS.glob("*.cif")):
            try:
                s = cif.read(path)
                out.append((path.name, s, coordination.analyse_structure(s)))
            except Exception:
                continue
        return out

    def test_the_collection_actually_loaded(self, analysed):
        assert len(analysed) >= 30

    def test_impossibly_short_contacts_are_detected_not_hidden(self, analysed):
        """Below about 1.2 Å nothing is an interatomic distance.

        Such contacts do occur in real collections — one diffraction-card
        export here gives P–O = 1.101 Å against a real P–O of about 1.53 Å.
        FACET still computes with the coordinates it was given; what it must
        not do is stay silent, so every such contact has to be reported.
        """
        from facet.core import quality

        for name, s, results in analysed:
            short = [c for r in results for c in r.bonds
                     if c.distance < quality.MIN_PHYSICAL_BOND]
            if not short:
                continue
            report = quality.check(s, results)
            codes = {f.code for f in report.of_level(quality.Level.IMPOSSIBLE)}
            assert "short-contact" in codes, (
                f"{name}: a contact of {min(c.distance for c in short):.3f} Å "
                f"went unreported")

    def test_a_sound_structure_raises_nothing(self, analysed):
        """The check has to be quiet on good data, or it is noise."""
        from facet.core import quality

        by_name = {name: (s, r) for name, s, r in analysed}
        if "1526458_Bi2O3.cif" not in by_name:
            pytest.skip("reference structure not present")
        s, results = by_name["1526458_Bi2O3.cif"]
        report = quality.check(s, results)
        assert not report.of_level(quality.Level.IMPOSSIBLE)
        assert not report.of_level(quality.Level.CHECK)

    def test_every_bond_distance_matches_its_vector_length(self, analysed):
        """The scalar and the vector must agree, or the geometry and the
        numbers have drifted apart."""
        for name, s, results in analysed:
            for r in results:
                for c in r.contacts:
                    assert float(np.linalg.norm(c.vector)) == pytest.approx(
                        c.distance, abs=1e-9), f"{name}:{r.label}-{c.label}"

    def test_every_bond_valence_matches_its_distance(self, analysed):
        for name, s, results in analysed:
            for r in results:
                for c in r.contacts:
                    if c.param is None or c.valence is None:
                        continue
                    want = math.exp((c.param.r0 - c.distance) / c.param.b)
                    assert c.valence == pytest.approx(want, rel=1e-12)

    def test_the_bond_valence_sum_equals_the_sum_of_its_bonds(self, analysed):
        for name, s, results in analysed:
            for r in results:
                if not r.bonds:
                    continue
                manual = sum(c.valence * c.occupancy for c in r.bonds)
                assert r.bvs == pytest.approx(manual, rel=1e-12), \
                    f"{name}:{r.label}"

    def test_phi_stays_within_its_bounds(self, analysed):
        for name, s, results in analysed:
            for r in results:
                if r.phi != r.phi:
                    continue
                assert -1e-9 <= r.phi <= 1.0 + 1e-9, f"{name}:{r.label} {r.phi}"

    def test_contacts_are_sorted_and_consistent_with_the_threshold(self, analysed):
        for name, s, results in analysed:
            for r in results:
                distances = [c.distance for c in r.contacts]
                assert distances == sorted(distances), f"{name}:{r.label}"
                for c in r.bonds:
                    assert c.valence > r.v_bond

    def test_coordination_numbers_are_physically_plausible(self, analysed):
        """Counted by occupancy, not by contact.

        A disordered average structure has many partially occupied neighbours
        at one distance: delta-Bi2O3 gives 24 oxygen contacts each a quarter
        occupied, which is six oxygens in any one cell. The raw count is 24 and
        is not wrong; the number that means something is the occupancy-weighted
        one, and that is what has to be physical.
        """
        for name, s, results in analysed:
            for r in results:
                assert 0 <= r.cn_valence <= 40, f"{name}:{r.label} raw {r.cn_valence}"
                if r.cn_occupancy == r.cn_occupancy:
                    assert 0 <= r.cn_occupancy <= 16, (
                        f"{name}:{r.label} {r.cn_occupancy:.2f} by occupancy")

    def test_occupancy_weighted_cn_equals_the_raw_count_when_ordered(self,
                                                                     analysed):
        for name, s, results in analysed:
            for r in results:
                if r.bonds and not r.is_disordered:
                    assert r.cn_occupancy == pytest.approx(
                        float(r.cn_valence), abs=1e-9), f"{name}:{r.label}"

    def test_occupancy_weighting_brings_disordered_sites_into_range(self,
                                                                     analysed):
        """A disordered average structure reports many partial neighbours.

        The raw count can be 24; the occupancy-weighted one has to be a
        coordination number a cation could actually have. The exact value
        depends on the disorder model — the fluorite delta-Bi2O3 of Aidhy
        gives 6.00 where Battle's different model gives 5.36 — so the
        assertion is on the range, not on one number.
        """
        seen = 0
        for name, s, results in analysed:
            for r in results:
                if not (r.bonds and r.is_disordered):
                    continue
                seen += 1
                assert 1.0 <= r.cn_occupancy <= 12.0, (
                    f"{name}:{r.label} {r.cn_occupancy:.2f} by occupancy "
                    f"({r.cn_valence} raw)")
                assert r.cn_occupancy <= r.cn_valence + 1e-9
        assert seen, "no disordered sites in the collection to check"

    def test_the_fluorite_model_resolves_to_exactly_six(self, analysed):
        """delta-Bi2O3 as a fluorite: eight tetrahedral sites a quarter
        occupied around each cation is six oxygens in any one cell."""
        target = next((t for t in analysed if "Aidhy" in t[0]), None)
        if target is None:
            pytest.skip("the fluorite model is not in this collection")
        name, s, results = target
        checked = 0
        for r in results:
            if r.element == "Bi" and r.is_disordered:
                assert r.cn_occupancy == pytest.approx(6.0, abs=0.01), (
                    f"{name}:{r.label} {r.cn_occupancy}")
                checked += 1
        assert checked

    def test_polyhedron_volume_is_positive_where_it_exists(self, analysed):
        for name, s, results in analysed:
            for r in results:
                volume = r.shape.get("volume")
                if volume is not None:
                    assert volume > 0, f"{name}:{r.label}"

    def test_void_cone_is_within_range(self, analysed):
        for name, s, results in analysed:
            for r in results:
                if r.void_angle != r.void_angle:
                    continue
                assert 0.0 <= r.void_angle <= 180.001, f"{name}:{r.label}"

    def test_drawn_bonds_match_the_neighbour_search(self, analysed):
        """The check that would have caught the propagation bug: every bond in
        the drawn scene must be a real contact between the two atoms it joins,
        at the distance it claims."""
        from facet.gl.scene import build_scene

        for name, s, results in analysed[:20]:
            scene = build_scene(s, results)
            for i in range(scene.n_bonds):
                a = scene.bond_a[i]
                b = scene.bond_b[i]
                drawn = float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
                assert drawn == pytest.approx(
                    float(scene.bond_distance[i]), abs=1e-4), (
                    f"{name}: bond {i} is drawn {drawn:.4f} A long but "
                    f"reports {scene.bond_distance[i]:.4f} A")

    def test_every_drawn_bond_ends_on_a_drawn_atom(self, analysed):
        from scipy.spatial import cKDTree

        from facet.gl.scene import build_scene

        for name, s, results in analysed[:20]:
            scene = build_scene(s, results)
            if not scene.n_bonds:
                continue
            tree = cKDTree(scene.atom_position)
            for endpoint in (scene.bond_a, scene.bond_b):
                d, _ = tree.query(endpoint)
                assert float(d.max()) < 1e-3, (
                    f"{name}: a bond ends {d.max():.4f} A from any atom")

    def test_no_two_drawn_atoms_coincide_unless_the_file_says_so(self, analysed):
        """Overlapping atoms are legitimate when the CIF has a split or shared
        site. They are a bug when the CIF does not."""
        from scipy.spatial import cKDTree

        from facet.gl.scene import build_scene

        for name, s, results in analysed:
            in_file = cKDTree(np.array([a.cart for a in s.atoms])).query_pairs(0.35)
            scene = build_scene(s, results)
            drawn = cKDTree(scene.atom_position).query_pairs(0.35)
            assert len(drawn) <= len(in_file) * 12, (
                f"{name}: {len(drawn)} overlapping drawn pairs against "
                f"{len(in_file)} in the file")
            if not in_file:
                assert not drawn, f"{name}: overlaps the file does not have"
