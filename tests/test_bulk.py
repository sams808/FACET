"""The bulk engine on its own terms: thresholds, search, parameters, edge cases.

``tests/test_bulk_equivalence.py`` shows the engine reproduces the crystal path
on real crystals. This file pins what that test cannot reach:

* **Moving the threshold is free and exact.** ``at_threshold(table, v)`` must
  equal a fresh ``analyse_frame(..., v_bond_vu=v)`` on every field, because the
  MD panel's slider depends on it, and the crystal path does not do this (its
  results cache keeps BVS and phi from the analysis-time threshold, Step 0
  §4 item 17). The plateau rule is checked against ``coordination.plateaus``
  itself at thresholds equal to the contact valences, where the code and its
  docstring disagree.
* **phi and BVS are bv.phi_index and bv.bvs.** Applied to the same rows of the
  table, the reference functions give the same numbers.
* **The two searches are one.** ``'images'`` and ``'boxsize'`` return the same
  pairs wherever the periodic tree is allowed, both equal ``NeighborFinder``'s
  contacts, and the periodic tree is refused where it would miss images.
* **Nothing is dropped silently.** A pair with no parameter is counted, not
  given zero; a 0.3 Å pair is counted; an element that ``elements.normalise``
  would rename gets no parameter; an atom with no oxidation state is refused.
* **A physical law.** Every listed cation-anion contact is seen from both ends
  with one valence, so the cation bond-valence sums and the anion ones add up
  to the same total.
* **The constants are NeighborFinder's.** D_MIN_ANG and the radius floor and
  ceiling are compared with the defaults in the crystal code's signatures.

The synthetic frames are built in the tests; their distances are test inputs,
not reference values for any material.
"""
from __future__ import annotations

import ast
import dataclasses
import functools
import inspect
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from facet.core import bulk, bv, coordination, md_model, readers
from facet.core.neighbors import NeighborFinder, search_radius_for
from facet.core.structure import Atom, Site, Structure

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data" / "crystals"
EXAMPLES = ("quartz_SiO2_cod9013321.cif",
            "bismuth_phosphate_BiPO4_cod9008088.cif",
            "cryolite_Na3AlF6_cod9004097.cif",
            "eulytite_Bi4SiO4_3_cod9012894.cif",
            "senarmontite_Sb2O3_cod9009747.cif",
            "valentinite_Sb2O3_cod9007587.cif")
ATOL = 1e-10

RESULT_FIELDS = ("cn", "cn_listed", "bvs_vu", "bvs_listed_vu", "bvv_vu",
                 "bvv_listed_vu", "bvv_vector_vu", "phi", "phi_listed",
                 "plateau_decades", "valence_discrepancy_vu")


@functools.lru_cache(maxsize=None)
def _example(name: str):
    """(structure, frame, ox) for a 2x2x2 supercell of a bundled example,
    with the oxidation states the crystal path resolves."""
    structure = readers.read(DATA / name)
    coordination.analyse_structure(structure, cations_only=False)
    frame, parent = md_model.supercell_frame(structure, (2, 2, 2))
    ox = np.array([structure.sites[structure.atoms[p].site_index].ox
                   for p in parent], dtype=np.int64)
    return structure, frame, ox


def _crystal(frame, states: dict, params=None, v_bond=bv.V_BOND_DEFAULT,
             v_list=bv.V_LIST_DEFAULT):
    """analyse_site on every atom of the frame, through frame_to_structure."""
    params = params or bv.DEFAULT
    ox_model = md_model.model_oxidation(frame.species, states)
    structure = md_model.frame_to_structure(frame, ox_model)
    r = bulk.search_radius_ang(frame.elements, ox_model.per_atom(frame.elements),
                               params, v_list)
    finder = NeighborFinder(structure, rmax=r)
    return [coordination.analyse_site(structure, finder.contacts(k), params,
                                      v_bond, v_list)
            for k in range(frame.n_atoms)]


def _close(got, want, name=""):
    """|got - want| <= ATOL; NaN equal only to NaN, infinity only to the same
    infinity (inf - inf is NaN, so infinities are compared, not subtracted)."""
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    assert np.array_equal(np.isnan(got), np.isnan(want)), name
    assert np.array_equal(np.isinf(got), np.isinf(want)), name
    assert np.array_equal(got[np.isinf(got)], want[np.isinf(want)]), name
    ok = np.isfinite(got)
    assert np.max(np.abs(got[ok] - want[ok]), initial=0.0) <= ATOL, name


def test_the_comparison_helper_treats_matching_infinities_as_equal():
    """A plateau is infinite when its lower edge is not positive; two equal
    infinities have to compare equal, and a finite value never equals one."""
    _close(np.array([np.inf, 1.0, np.nan]), np.array([np.inf, 1.0, np.nan]))
    for want in (np.array([1.0]), np.array([-np.inf])):
        with pytest.raises(AssertionError):
            _close(np.array([np.inf]), want)


# bulk.AtomResults field -> coordination.SiteResult field, every one that the
# AtomResults docstring says it holds
CRYSTAL_FIELDS = (("cn", "cn_valence"), ("cn_listed", "cn_listed"),
                  ("bvs_vu", "bvs"), ("bvs_listed_vu", "bvs_listed"),
                  ("bvv_vu", "bvv"), ("bvv_listed_vu", "bvv_listed"),
                  ("phi", "phi"), ("phi_listed", "phi_listed"),
                  ("plateau_decades", "plateau_decades"),
                  ("valence_discrepancy_vu", "valence_discrepancy"))


def _same_as_crystal(results, crystal, v_bond=bv.V_BOND_DEFAULT):
    """Every AtomResults field against the SiteResult of the same atom; a
    valence_discrepancy of None (no bond) is NaN here."""
    for bulk_name, site_name in CRYSTAL_FIELDS:
        want = [getattr(c, site_name) for c in crystal]
        want = [np.nan if w is None else w for w in want]
        _close(getattr(results, bulk_name), want, f"{bulk_name} {v_bond}")
    _close(results.bvv_vector_vu, np.array([c.bvv_vector for c in crystal]),
           f"bvv_vector_vu {v_bond}")


def _jittered(symbols, frac, edge_ang, seed, jitter_ang=0.05, origin=None):
    """A cubic frame from fractional positions, each moved at random so no two
    contacts tie."""
    rng = np.random.default_rng(seed)
    cart = np.asarray(frac, float) * edge_ang \
        + rng.uniform(-jitter_ang, jitter_ang, (len(symbols), 3))
    return md_model.frame_from_arrays(symbols, cart, box_ang=np.eye(3) * edge_ang,
                                      origin_ang=origin)


# ---------------------------------------------------------------------------
# moving the threshold
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["cryolite_Na3AlF6_cod9004097.cif",
                                  "bismuth_phosphate_BiPO4_cod9008088.cif"])
def test_v_bond_never_reaches_the_table(name):
    """A determinism pin, not a check of the numbers: analyse_frame builds the
    table from v_list alone and calls at_threshold on it, so moving the
    threshold on a kept table has to equal a fresh analysis bit for bit. A
    failure means v_bond leaked into the table or the search is not
    deterministic. Whether the numbers at a moved threshold are right is
    checked against the crystal path in
    test_a_moved_threshold_equals_the_crystal_path. Five thresholds, one below
    v_list (where every listed contact is a bond and no plateau contains the
    threshold) and one above most valences."""
    _, frame, ox = _example(name)
    table, _ = bulk.analyse_frame(frame, ox)
    for v_bond in (0.01, 0.03, bv.V_BOND_DEFAULT, 0.2, 0.6):
        moved = bulk.at_threshold(table, v_bond)
        _, fresh = bulk.analyse_frame(frame, ox, v_bond_vu=v_bond)
        assert moved.v_bond_vu == fresh.v_bond_vu == v_bond
        for field in RESULT_FIELDS:
            assert np.array_equal(getattr(moved, field), getattr(fresh, field),
                                  equal_nan=True), (field, v_bond)
    below = bulk.at_threshold(table, 0.01)
    assert np.array_equal(below.cn, table.n_listed)
    assert np.isnan(below.plateau_decades).all()


def test_a_moved_threshold_equals_the_crystal_path():
    """The table is built once, at v_list; at_threshold then moves v_bond to
    random values between 0.005 and 1.2 v.u., to v_list itself and below it.
    At each, every field of every atom of a sheared random frame (a glass in
    miniature, nothing on a symmetry position) equals analyse_site run from
    scratch at that threshold through NeighborFinder: an independent search,
    independent valence calls and the crystal code's own sums and plateaus.

    The thresholds are kept at least 1e-9 v.u. from every contact valence:
    the two paths form a distance from different roundings, so their
    valences can differ in the last bits, and a threshold set exactly on one
    can then count that contact on one path and not the other."""
    rng = np.random.default_rng(20261006)
    box = np.array([[11.0, 0.0, 0.0], [3.3, 11.0, 0.0], [2.2, 2.75, 11.0]])
    symbols = rng.choice(np.array(["Si", "O", "O", "Na", "Al"]), 110)
    frame = md_model.frame_from_arrays(symbols, frac=rng.random((110, 3)),
                                       box_ang=box,
                                       origin_ang=np.array([2.5, -1.0, 7.0]))
    ox = md_model.model_oxidation(frame.species).per_atom(frame.elements)
    table, _ = bulk.analyse_frame(frame, ox)
    listed = table.v_vu[np.arange(table.v_vu.shape[1])[None, :]
                        < table.n_listed[:, None]]
    thresholds = [float(v) for v in rng.uniform(0.005, 1.2, 6)]
    thresholds += [table.v_list_vu, table.v_list_vu / 2, 0.0]
    for v_bond in thresholds:
        assert np.abs(listed - v_bond).min() > 1e-9, v_bond
        crystal = _crystal(frame, {}, v_bond=v_bond)
        _same_as_crystal(bulk.at_threshold(table, v_bond), crystal, v_bond)


def _site_result(valences, v_bond, v_list):
    """A SiteResult holding exactly these valences, its plateaus built by the
    crystal code itself."""
    rows = [coordination.ContactRow(
        label=f"O{k}", element="O", site_index=0, atom_index=0,
        distance=1.0 + k, vector=np.array([1.0, 0, 0]), image=(0, 0, 0),
        occupancy=1.0, valence=float(v), param=None)
        for k, v in enumerate(valences)]
    result = coordination.SiteResult(
        site_index=0, label="X", element="X", ox=1, ox_source="user",
        wyckoff=None, site_symmetry=None, multiplicity=1, contacts=rows,
        v_bond=v_bond, v_list=v_list)
    result.plateaus = coordination.plateaus(result)
    return result


def test_the_plateau_rule_is_the_crystal_codes_on_every_edge():
    """At v_bond equal to a contact's valence the crystal code reports the
    plateau above it (Plateau.contains is v_low < v <= v_high), although its
    docstring says NaN on a step edge (coordination.py:172-173). The engine
    follows the code: checked at every valence of several rows, at the
    midpoints, at v_list and below it, and with tied valences."""
    _, frame, ox = _example("cryolite_Na3AlF6_cod9004097.cif")
    table, _ = bulk.analyse_frame(frame, ox)
    rows = [0, 3, 17, 40, 77, 130, 159]
    candidates = set()
    for r in rows:
        values = table.v_vu[r, :table.n_listed[r]]
        candidates.update(values.tolist())
        candidates.update(((values[1:] + values[:-1]) / 2).tolist())
    candidates.update([table.v_list_vu, table.v_list_vu / 2, 5.0])
    for v_bond in sorted(candidates):
        results = bulk.at_threshold(table, v_bond)
        for r in rows:
            values = table.v_vu[r, :table.n_listed[r]]
            site = _site_result(values, v_bond, table.v_list_vu)
            assert results.cn[r] == site.cn_at(v_bond)
            want, got = site.plateau_decades, results.plateau_decades[r]
            assert (np.isnan(want) and np.isnan(got)) or \
                abs(want - got) <= ATOL, (r, v_bond, want, got)

    # tied valences: the degenerate plateau between them does not exist
    tied = np.array([0.5, 0.3, 0.3, 0.1])
    for v_bond in (0.5, 0.4, 0.3, 0.2, 0.1, 0.05):
        site = _site_result(tied, v_bond, 0.02)
        k = int((tied >= v_bond).sum())
        low = tied[k] if k < 4 else 0.02
        assert site.plateau_decades == pytest.approx(np.log10(tied[k - 1] / low),
                                                     abs=1e-12)


def test_phi_and_bvs_are_bv_phi_index_and_bv_bvs_on_the_same_rows():
    """The reference functions, applied to each row's bonded and listed
    contacts, give the engine's phi, |BVV|, vector and sums."""
    _, frame, ox = _example("bismuth_phosphate_BiPO4_cod9008088.cif")
    table, results = bulk.analyse_frame(frame, ox)
    for r in range(frame.n_atoms):
        for count, phi, bvs, mag in (
                (results.cn[r], results.phi[r], results.bvs_vu[r],
                 results.bvv_vu[r]),
                (results.cn_listed[r], results.phi_listed[r],
                 results.bvs_listed_vu[r], results.bvv_listed_vu[r])):
            vectors = table.vec_ang[r, :count]
            valences = table.v_vu[r, :count]
            want_mag, want_phi, want_vec = bv.phi_index(vectors, valences)
            assert abs(bv.bvs(valences) - bvs) <= ATOL
            assert abs(want_phi - phi) <= ATOL
            assert abs(want_mag - mag) <= ATOL
        want = bv.phi_index(table.vec_ang[r, :results.cn[r]],
                            table.v_vu[r, :results.cn[r]])[2]
        assert np.abs(want - results.bvv_vector_vu[r]).max() <= ATOL


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------

def _sorted_pairs(table: bulk.PairTable):
    order = np.lexsort((table.image[:, 2], table.image[:, 1], table.image[:, 0],
                        table.j, table.i))
    return (table.i[order], table.j[order], table.image[order],
            table.d_ang[order], table.vec_ang[order])


def _random_cubic(n, edge_ang, seed, origin=None):
    rng = np.random.default_rng(seed)
    symbols = rng.choice(np.array(["Si", "O", "O", "Na"]), n)
    return md_model.frame_from_arrays(symbols, rng.random((n, 3)) * edge_ang,
                                      box_ang=np.diag([edge_ang, edge_ang * 1.1,
                                                       edge_ang * 0.95]),
                                      origin_ang=origin)


@pytest.mark.parametrize("frame_kind", ["eulytite", "senarmontite", "random"])
def test_images_and_boxsize_give_the_same_pairs(frame_kind):
    """Same pair set, same images, bit-identical distances and vectors, on
    two orthogonal crystals and on a random box with an origin away from 0."""
    if frame_kind == "random":
        frame = _random_cubic(700, 15.0, 3, origin=np.array([-7.3, 2.1, 40.0]))
    else:
        frame = _example(next(n for n in EXAMPLES if n.startswith(frame_kind)))[1]
    images = bulk.find_pairs(frame, 6.0, method="images", block_atoms=97)
    periodic = bulk.find_pairs(frame, 6.0, method="boxsize", block_atoms=131)
    assert images.method == "images" and periodic.method == "boxsize"
    assert bulk.find_pairs(frame, 6.0).method == "boxsize"          # auto
    assert len(images) == len(periodic) > 0
    for a, b in zip(_sorted_pairs(images), _sorted_pairs(periodic)):
        assert np.array_equal(a, b)


def test_atoms_on_the_box_faces_give_the_same_pairs_on_both_paths():
    """Frame.frac lies in [0, 1) exactly, and the periodic cKDTree refuses a
    coordinate equal to L. Atoms at fraction 0 and at the largest double
    below 1, on every axis, are where the two searches could part: the
    periodic tree sees them across one face, the image tree in the next cell.

    Under round-to-nearest f * L stays below L for every f < 1 (the exact
    product is at least half an ulp of L below L), so the guard in iter_pairs
    that maps L to 0 is not expected to fire; this pins that too, on a power
    of two, one ulp above one, and two ordinary lengths."""
    f = np.nextafter(1.0, 0.0)
    # rows 0-1, 2-3 and 4-5 face each other across the x, y and z faces, 0.1
    # of the box apart; rows 6 and 7 sit on opposite corners
    frac = np.array([[f, 0.5, 0.5], [0.1, 0.5, 0.5], [0.5, 0.0, 0.2],
                     [0.5, 0.9, 0.2], [0.7, 0.3, f], [0.7, 0.3, 0.1],
                     [f, f, f], [0.1, 0.1, 0.1]])
    symbols = ["Si", "O", "Na", "O", "Si", "O", "Na", "O"]
    for edge in (16.0, float(np.nextafter(16.0, 32.0)), 13.0, 17.3):
        assert f * edge < edge
        frame = md_model.frame_from_arrays(symbols, frac=frac,
                                           box_ang=np.eye(3) * edge)
        assert frame.frac.max() == f and frame.frac.min() == 0.0
        images = bulk.find_pairs(frame, 6.0, method="images")
        periodic = bulk.find_pairs(frame, 6.0, method="boxsize")
        assert len(images) == len(periodic) > 0
        assert images.n_below_d_min == periodic.n_below_d_min == 0
        for a, b in zip(_sorted_pairs(images), _sorted_pairs(periodic)):
            assert np.array_equal(a, b), edge
        # each facing pair is found through the face, 0.1 of the box apart
        for a, b, axis, step in ((0, 1, 0, 1), (2, 3, 1, -1), (4, 5, 2, 1)):
            sel = (images.i == a) & (images.j == b)
            nearest = int(np.argmin(images.d_ang[sel]))
            assert images.d_ang[sel][nearest] == pytest.approx(0.1 * edge,
                                                               abs=1e-9)
            assert images.image[sel][nearest][axis] == step


def test_boxsize_is_refused_where_it_would_miss_images():
    """A tilt of 1e-3 Å, a box exactly 2 r wide, and a negative length
    on the diagonal (a left-handed box) each send 'auto' to the image path and
    make 'boxsize' raise, naming the reason."""
    tilted = md_model.frame_from_arrays(
        ["Si", "O"], [[0.0, 0, 0], [1.6, 0, 0]],
        box_ang=np.array([[20.0, 0, 0], [1e-3, 20.0, 0], [0, 0, 20.0]]))
    small = md_model.frame_from_arrays(["Si", "O"], [[0.0, 0, 0], [1.6, 0, 0]],
                                       box_ang=np.eye(3) * 12.0)
    left = md_model.frame_from_arrays(["Si", "O"], [[0.0, 0, 0], [1.6, 0, 0]],
                                      box_ang=np.diag([20.0, 20.0, -20.0]))
    for frame, reason in ((tilted, "off-diagonal"),
                          (small, "half the smallest box length"),
                          (left, "not positive")):
        with pytest.raises(ValueError, match=reason):
            next(bulk.iter_pairs(frame, 6.0, method="boxsize"))
        assert bulk.find_pairs(frame, 6.0).method == "images"
    roomy = md_model.frame_from_arrays(["Si", "O"], [[0.0, 0, 0], [1.6, 0, 0]],
                                       box_ang=np.eye(3) * 12.1)
    assert bulk.find_pairs(roomy, 6.0).method == "boxsize"


def test_a_neighbour_seen_through_two_images_is_found_twice():
    """L = 9 Å and r = 6 Å: atom 1 at 4 Å along x is also at 5 Å through the
    next cell. The minimum image would give only one of them."""
    frame = md_model.frame_from_arrays(
        ["Si", "O"], [[0.0, 0, 0], [4.0, 0, 0]], box_ang=np.eye(3) * 9.0)
    pairs = bulk.find_pairs(frame, 6.0)
    sel = (pairs.i == 0) & (pairs.j == 1) & (pairs.image[:, 1:] == 0).all(1)
    found = sorted(zip(pairs.d_ang[sel].round(12), pairs.image[sel][:, 0]))
    assert found == [(4.0, 0), (5.0, -1)]


def _every_translation_pairs(frame, r_ang):
    """Reference pair set: every atom against every translated copy that
    neighbors._images_within lists (NeighborFinder's unfiltered set), with
    distances from origin-free positions. Plain loops over translations; no
    tree, no filtering."""
    from facet.core.neighbors import _images_within

    home = frame.frac @ frame.box_ang
    out = set()
    for n in _images_within(frame.box_ang.T, r_ang):
        sep = (home[None, :, :] + n @ frame.box_ang) - home[:, None, :]
        d = np.linalg.norm(sep, axis=2)
        i, j = np.nonzero((d > bulk.D_MIN_ANG) & (d <= r_ang))
        out |= {(int(a), int(b), *map(int, n)) for a, b in zip(i, j)}
    return out


@pytest.mark.parametrize("kind", ["tilted", "thin", "left-handed", "sheared"])
def test_the_image_search_keeps_every_image_within_reach(kind):
    """The image search leaves out the translated copies farther than r from
    every point of the box. On cells where a neighbour has several images
    within r (a 4.6 Å-thin cell, a strongly tilted one, a left-handed one)
    and on a LAMMPS-style sheared box, the pair set equals the one from every
    copy NeighborFinder would build, and the tree holds fewer points than
    NeighborFinder's 125 N or more."""
    rng = np.random.default_rng(42)
    box = {"tilted": np.array([[9.0, 0, 0], [6.5, 8.0, 0], [-5.0, 4.5, 9.5]]),
           "thin": np.array([[12.0, 0, 0], [0.0, 13.0, 0], [1.5, -2.0, 4.6]]),
           "left-handed": np.array([[10.0, 0, 0], [2.0, -11.0, 0],
                                    [1.0, 3.0, 9.5]]),
           "sheared": np.array([[15.0, 0, 0], [4.5, 15.0, 0],
                                [3.0, 3.75, 15.0]])}[kind]
    frame = md_model.frame_from_arrays(
        rng.choice(np.array(["Si", "O", "Na"]), 60), frac=rng.random((60, 3)),
        box_ang=box, origin_ang=np.array([-3.0, 8.0, 1.5]))
    pairs = bulk.find_pairs(frame, 7.5, method="images")
    got = {(int(a), int(b), *map(int, n))
           for a, b, n in zip(pairs.i, pairs.j, pairs.image)}
    assert len(got) == len(pairs) > 0
    assert got == _every_translation_pairs(frame, 7.5)
    atom, _, _ = bulk._image_points(frame, 7.5)
    assert atom.size < 125 * frame.n_atoms


def test_the_search_reads_the_fractions_not_cart_ang():
    """A Frame built directly may hold a cart_ang up to 1e-9 Å from
    origin + frac @ box (md_model._CART_CONSISTENCY_ANG), as large as
    QUERY_PAD_ANG. When the image path measured cart_ang and the periodic
    tree measured the fractions, a pair at 6 Å + 1.5e-9 by the fractions and
    6 Å - 4e-10 by cart_ang was found by one method and not the other. Both
    now measure the fractions, so they agree, and the distance is the one the
    fractions give."""
    edge, r = 20.0, 6.0
    frac = np.array([[0.30, 0.5, 0.5], [0.30 + (r + 1.5e-9) / edge, 0.5, 0.5],
                     [0.30, 0.5 + 2.0 / edge, 0.5]])
    cart = frac * edge
    cart[0, 0] += 0.95e-9
    cart[1, 0] -= 0.95e-9
    frame = md_model.Frame(elements=np.array(["Si", "O", "O"]), frac=frac,
                           cart_ang=cart, box_ang=np.eye(3) * edge,
                           origin_ang=np.zeros(3), atom_id=np.arange(3))
    images = bulk.find_pairs(frame, r, method="images")
    periodic = bulk.find_pairs(frame, r, method="boxsize")
    assert len(images) == len(periodic) > 0
    for a, b in zip(_sorted_pairs(images), _sorted_pairs(periodic)):
        assert np.array_equal(a, b)
    assert not ((images.i == 0) & (images.j == 1)).any()
    sel = (images.i == 0) & (images.j == 2)
    assert images.d_ang[sel].tolist() == [
        float(np.linalg.norm((frac[2] - frac[0]) * edge))]
    assert images.d_ang[sel][0] == pytest.approx(2.0, abs=1e-12)


def test_the_origin_does_not_move_a_bit():
    """cart_ang carries origin_ang, so a distance formed from it carries the
    origin's rounding: at an origin of 1e7 Å per-atom sums moved by 8e-8 v.u.
    Distances are formed from the fractions, so the same fractions in the
    same box give bit-identical pairs and results at any origin."""
    _, frame, ox = _example("quartz_SiO2_cod9013321.cif")
    sheared = md_model.frame_from_arrays(
        frame.elements, frac=frame.frac,
        box_ang=frame.box_ang + np.array([[0, 0, 0], [1.1, 0, 0], [0.7, 0.4, 0]]))
    for base in (frame, sheared):
        reference = bulk.find_pairs(base, 6.0)
        _, results = bulk.analyse_frame(base, ox)
        for origin in (1e3, 1e6, 1e7):
            moved = md_model.frame_from_arrays(
                base.elements, frac=base.frac, box_ang=base.box_ang,
                origin_ang=np.array([origin, -origin, 0.5 * origin]))
            assert np.array_equal(moved.frac, base.frac)
            pairs = bulk.find_pairs(moved, 6.0)
            for a, b in zip(_sorted_pairs(pairs), _sorted_pairs(reference)):
                assert np.array_equal(a, b), origin
            _, got = bulk.analyse_frame(moved, ox)
            for field in RESULT_FIELDS:
                assert np.array_equal(getattr(got, field),
                                      getattr(results, field),
                                      equal_nan=True), (field, origin)


def test_a_box_too_thin_for_the_image_indices_is_refused():
    """Image indices are stored as int16. A box 1.5e-4 Å wide with r = 6 Å
    needs translations up to 40 002 boxes away; they wrapped round to other
    images and find_pairs returned distances up to 1.1 Å off, with no
    message. Such a box is refused, naming its width."""
    needle = md_model.frame_from_arrays(["Si"], frac=[[0.1, 0.1, 0.1]],
                                        box_ang=np.diag([10.0, 10.0, 1.5e-4]))
    with pytest.raises(ValueError, match=r"0\.00015 Å wide across its c faces"):
        bulk.find_pairs(needle, 6.0)


def test_a_radius_far_beyond_the_box_is_refused_with_the_estimate():
    """r = 10 000 Å in an 11.5 Å box would put 1.6e11 image points in the
    tree; it ended in a MemoryError deep inside numpy. It is refused before
    anything is built, with the estimate and IMAGE_POINTS_LIMIT."""
    rng = np.random.default_rng(3)
    frame = md_model.frame_from_arrays(
        rng.choice(np.array(["Si", "O"]), 30), rng.random((30, 3)) * 11.5,
        box_ang=np.eye(3) * 11.5)
    with pytest.raises(ValueError, match="IMAGE_POINTS_LIMIT"):
        next(bulk.iter_pairs(frame, 1e4))


def test_a_radius_far_beyond_the_box_is_refused_before_the_blocks_are_cut(
        monkeypatch):
    """The block map estimates pairs as density x 4/3 pi r^3, and used to list
    one cut per PAIRS_PER_BLOCK estimated pairs before the image count was
    checked: at r = 1e5 Å in this 30-atom box that list held 5e9 entries and
    the process grew past 17 GB instead of raising. The image count is now
    checked first, when iter_pairs is called, so neither the blocks nor the
    image points are ever reached; the int16 refusal names a radius that
    large as well as a box that thin."""
    def never(*args, **kwargs):
        raise AssertionError("reached before the image-count refusal")

    monkeypatch.setattr(bulk, "_block_bounds", never)
    monkeypatch.setattr(bulk, "_image_points", never)
    rng = np.random.default_rng(3)
    frame = md_model.frame_from_arrays(
        rng.choice(np.array(["Si", "O"]), 30), rng.random((30, 3)) * 11.5,
        box_ang=np.eye(3) * 11.5)
    ox = np.where(frame.elements == "Si", 4, -2)
    for r_ang, match in ((3e4, "IMAGE_POINTS_LIMIT"), (1e5, "IMAGE_POINTS_LIMIT"),
                         (1e6, "a radius that large"),
                         (1e300, "a radius that large")):
        with pytest.raises(ValueError, match=match):
            bulk.iter_pairs(frame, r_ang)                # no next(): at the call
        with pytest.raises(ValueError, match=match):
            bulk.analyse_frame(frame, ox, r_search_ang=r_ang)


def _block_bounds_from_targets(frame, r_ang):
    """The block cuts as listed before 2026-10-06, one target per multiple of
    PAIRS_PER_BLOCK; kept here as the reference the one-pass numbering has to
    reproduce."""
    import math

    bins = np.clip(np.floor(frame.perpendicular_widths_ang / r_ang), 1,
                   bulk.DENSITY_CELLS_MAX).astype(np.int64)
    cell = np.minimum((frame.frac * bins).astype(np.int64), bins - 1)
    flat = np.ravel_multi_index(tuple(cell.T), tuple(bins))
    count = np.bincount(flat, minlength=int(bins.prod()))
    estimate = count[flat] * (float(bins.prod()) / frame.volume_ang3) \
        * 4.0 / 3.0 * math.pi * r_ang ** 3
    running = np.cumsum(np.maximum(estimate, 1.0))
    targets = bulk.PAIRS_PER_BLOCK * np.arange(
        1, int(running[-1] // bulk.PAIRS_PER_BLOCK) + 1)
    cuts = np.searchsorted(running, targets, side="right")
    edges = np.unique(np.concatenate(([0], cuts, [frame.n_atoms])))
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def test_the_block_cuts_take_one_pass_over_the_atoms(monkeypatch):
    """Each atom's block is numbered from its own running estimate, so the
    work is one pass over the atoms whatever the radius: at r = 1e9 Å the
    30 atoms come back as at most 30 contiguous blocks, at once. At ordinary
    radii the cuts are those of the target list they replaced (also checked,
    on 24 benchmark configurations of 9 261 and 29 791 atoms, when the change
    was made)."""
    rng = np.random.default_rng(3)
    small = md_model.frame_from_arrays(
        rng.choice(np.array(["Si", "O"]), 30), rng.random((30, 3)) * 11.5,
        box_ang=np.eye(3) * 11.5)
    bounds = bulk._block_bounds(small, 1e9, None)
    assert bounds[0][0] == 0 and bounds[-1][1] == small.n_atoms
    assert all(a < b for a, b in bounds)
    assert all(b == c for (_, b), (c, _) in zip(bounds, bounds[1:]))
    monkeypatch.setattr(bulk, "PAIRS_PER_BLOCK", 2_000)
    frame = _random_cubic(900, 21.0, 8)
    for r_ang in (3.0, 6.0, 9.5):
        bounds = bulk._block_bounds(frame, r_ang, None)
        assert len(bounds) > 3
        assert bounds == _block_bounds_from_targets(frame, r_ang), r_ang


def test_every_argument_is_refused_where_iter_pairs_is_called():
    """iter_pairs used to be a generator throughout, so an argument out of
    range was only refused when a consumer asked for the first block, inside
    valence_table or distance_cn. The arguments are now checked at the call;
    the search itself still waits for the first block."""
    tilted = md_model.frame_from_arrays(
        ["Si", "O"], [[0.0, 0, 0], [1.6, 0, 0]],
        box_ang=np.array([[20.0, 0, 0], [1.0, 20.0, 0], [0, 0, 20.0]]))
    for kwargs, match in (({"r_ang": -1.0}, "r_ang"),
                          ({"r_ang": 6.0, "method": "fast"}, "method"),
                          ({"r_ang": 6.0, "block_atoms": 0}, "block_atoms"),
                          ({"r_ang": 6.0, "method": "boxsize"}, "off-diagonal"),
                          ({"r_ang": 6.0, "d_min_ang": 7.0}, "d_min_ang")):
        with pytest.raises(ValueError, match=match):
            bulk.iter_pairs(tilted, **kwargs)
    with pytest.raises(ValueError, match="Frame"):
        bulk.iter_pairs("frame", 6.0)
    blocks = bulk.iter_pairs(tilted, 6.0)
    assert len(next(blocks)) == 2


def test_pairs_of_another_frame_are_refused():
    """The next frame of a trajectory has the same atoms; its pair table used
    to be accepted for the frame before, attaching one frame's distances to
    the other's atoms. The table carries a digest of the fractions and box it
    was searched on. The same positions with other elements or another
    origin have the same pairs and are accepted."""
    rng = np.random.default_rng(1)
    symbols = rng.choice(np.array(["Si", "O", "O", "Na"]), 120)
    one = md_model.frame_from_arrays(symbols, rng.random((120, 3)) * 13.0,
                                     box_ang=np.eye(3) * 13.0)
    two = md_model.frame_from_arrays(symbols, rng.random((120, 3)) * 13.0,
                                     box_ang=np.eye(3) * 13.0)
    ox = md_model.model_oxidation(one.species).per_atom(one.elements)
    pairs = bulk.find_pairs(one, 6.0)
    with pytest.raises(ValueError, match="another frame"):
        bulk.valence_table(two, pairs, ox)
    with pytest.raises(ValueError, match="another frame"):
        bulk.valence_table(two, bulk.iter_pairs(one, 6.0), ox)
    with pytest.raises(ValueError, match="another frame"):
        bulk.distance_cn(pairs, two, {("Si", "O"): 2.0})
    shifted = md_model.frame_from_arrays(symbols, frac=one.frac,
                                         box_ang=one.box_ang,
                                         origin_ang=np.array([5.0, 0, -2.0]))
    bulk.valence_table(shifted, pairs, ox)
    by_frame = bulk.distance_cn(pairs, one, {("Si", "O"): 2.0})
    by_symbols = bulk.distance_cn(pairs, one.elements, {("Si", "O"): 2.0})
    assert np.array_equal(by_frame[("Si", "O")], by_symbols[("Si", "O")])


def test_blocks_follow_the_local_density_of_a_cluster(monkeypatch):
    """Blocks were sized from the average number density. A cluster in a box
    five times its size holds its atoms at 125 times the average, and one
    block then took every atom, 4.3 times PAIRS_PER_BLOCK pairs on a
    29 791-atom cluster. Sized from a density map, each block stays within
    twice PAIRS_PER_BLOCK; here PAIRS_PER_BLOCK is lowered so a 1 728-atom
    cluster shows it. The table does not change."""
    monkeypatch.setattr(bulk, "PAIRS_PER_BLOCK", 20_000)
    rng = np.random.default_rng(3)
    grid = np.stack(np.meshgrid(*[np.arange(12)] * 3, indexing="ij"),
                    -1).reshape(-1, 3)
    cart = grid * 2.3 + rng.uniform(-0.2, 0.2, grid.shape) + 60.0
    symbols = rng.choice(np.array(["Si", "O", "O", "Na"]), len(grid))
    frame = md_model.frame_from_arrays(symbols, cart,
                                       box_ang=np.eye(3) * 12 * 2.3 * 5)
    blocks = list(bulk.iter_pairs(frame, 6.0))
    total = sum(len(b) for b in blocks)
    assert total > 4 * bulk.PAIRS_PER_BLOCK
    assert max(len(b) for b in blocks) <= 2 * bulk.PAIRS_PER_BLOCK
    ox = md_model.model_oxidation(frame.species).per_atom(frame.elements)
    many = bulk.valence_table(frame, blocks, ox)
    one = bulk.valence_table(frame, bulk.find_pairs(frame, 6.0,
                                                    block_atoms=frame.n_atoms),
                             ox)
    for field in ("nbr", "image", "v_vu", "cum_v_vu", "n_unparameterised"):
        assert np.array_equal(getattr(many, field), getattr(one, field)), field


@pytest.mark.parametrize("name", ["quartz_SiO2_cod9013321.cif",
                                  "eulytite_Bi4SiO4_3_cod9012894.cif"])
def test_the_pairs_are_neighborfinders_contacts(name):
    """Every atom's pairs equal NeighborFinder.contacts on the same frame
    (through frame_to_structure): same neighbours, same images, distances to
    1e-12 Å. One crystal on the image path, one on the periodic tree."""
    _, frame, _ = _example(name)
    structure = md_model.frame_to_structure(
        frame, md_model.model_oxidation(frame.species))
    finder = NeighborFinder(structure, rmax=6.0)
    pairs = bulk.find_pairs(frame, 6.0)
    for k in range(frame.n_atoms):
        contacts = finder.contacts(k)
        sel = pairs.i == k
        mine = sorted(zip(pairs.j[sel].tolist(),
                          map(tuple, pairs.image[sel].tolist()),
                          pairs.d_ang[sel].tolist()))
        theirs = sorted(zip(contacts.neighbor_atom.tolist(),
                            map(tuple, contacts.image.astype(int).tolist()),
                            contacts.distance.tolist()))
        assert [m[:2] for m in mine] == [t[:2] for t in theirs], k
        assert max((abs(m[2] - t[2]) for m, t in zip(mine, theirs)),
                   default=0.0) <= 1e-12


def test_blocks_give_the_table_of_one_block():
    """The table does not depend on how the centres were split into blocks,
    nor on the search method."""
    _, frame, ox = _example("eulytite_Bi4SiO4_3_cod9012894.cif")
    one = bulk.valence_table(frame, bulk.find_pairs(frame, 6.0), ox)
    for kw in ({"block_atoms": 7}, {"block_atoms": 250, "method": "images"}):
        many = bulk.valence_table(frame, bulk.iter_pairs(frame, 6.0, **kw), ox)
        for field in ("nbr", "image", "d_ang", "vec_ang", "v_vu", "n_listed",
                      "cum_v_vu", "cum_bvv_vu", "n_unparameterised",
                      "uses_estimated"):
            assert np.array_equal(getattr(one, field), getattr(many, field),
                                  equal_nan=True), (field, kw)
    blocks = list(bulk.iter_pairs(frame, 6.0, block_atoms=100))
    with pytest.raises(ValueError, match="cover"):
        bulk.valence_table(frame, blocks[1:], ox)
    with pytest.raises(ValueError, match="cover"):
        bulk.valence_table(frame, blocks + blocks[:1], ox)


def test_a_wider_search_gives_the_same_table_at_the_bond_valence_radius():
    """A search run to 8 Å (as a g(r) pass would) and limited to 6 Å by
    r_search_ang gives the 6 Å table, flags included."""
    _, frame, ox = _example("valentinite_Sb2O3_cod9007587.cif")
    narrow = bulk.valence_table(frame, bulk.find_pairs(frame, 6.0), ox)
    wide = bulk.valence_table(frame, bulk.iter_pairs(frame, 8.0), ox,
                              r_search_ang=6.0)
    for field in ("nbr", "v_vu", "cum_v_vu", "n_unparameterised"):
        assert np.array_equal(getattr(narrow, field), getattr(wide, field),
                              equal_nan=True), field
    assert wide.r_search_ang == 6.0
    with pytest.raises(ValueError, match="exceeds"):
        bulk.valence_table(frame, bulk.find_pairs(frame, 6.0), ox,
                           r_search_ang=7.0)


def test_a_radius_below_the_valence_reach_is_noted():
    """A table built from a search narrower than the distance at which a
    parameter's valence falls to v_list leaves listed contacts out; that is
    stated, with both distances. At the default radius nothing is said."""
    _, frame, ox = _example("quartz_SiO2_cod9013321.cif")
    reach = bv.DEFAULT.get("Si", 4, "O").distance_for(bv.V_LIST_DEFAULT)
    narrow = bulk.valence_table(frame, bulk.find_pairs(frame, 2.5), ox)
    note = next(n for n in narrow.notes if n.startswith("contacts were taken"))
    assert "2.5 Å" in note and f"{reach:.6g} Å" in note and "Si4+-O" in note
    table, _ = bulk.analyse_frame(frame, ox)
    assert table.r_search_ang >= reach
    assert not any(n.startswith("contacts were taken") for n in table.notes)
    assert "search_radius_ang gives a radius that holds them" in note


def test_a_reach_beyond_the_radius_ceiling_is_noted_as_such():
    """At v_list 1e-14 v.u. Na+-O keeps a valence above v_list out to
    13.7 Å, beyond search_radius_ang's 12 Å ceiling. The note used to say
    search_radius_ang gives a radius that holds those contacts, though it
    had just returned the 12 Å that does not; it now names the ceiling."""
    frame = md_model.frame_from_arrays(
        ["Si", "O", "Na", "O"], [[1.0, 1, 1], [2.6, 1, 1], [6, 6, 6], [7.5, 6, 6]],
        box_ang=np.eye(3) * 30.0)
    na_reach = bv.DEFAULT.get("Na", 1, "O").distance_for(1e-14)
    assert na_reach > bulk.RADIUS_CEILING_ANG
    table, _ = bulk.analyse_frame(frame, [4, -2, 1, -2], v_list_vu=1e-14)
    assert table.r_search_ang == bulk.RADIUS_CEILING_ANG
    note = next(n for n in table.notes if n.startswith("contacts were taken"))
    assert "ceiling of search_radius_ang" in note
    assert "search_radius_ang gives a radius" not in note


def test_vectors_can_be_left_out_for_a_distance_only_pass():
    _, frame, ox = _example("quartz_SiO2_cod9013321.cif")
    none = bulk.find_pairs(frame, 6.0, vectors_within_ang=0.0)
    assert none.vec_ang is None
    with pytest.raises(ValueError, match="no vectors"):
        bulk.valence_table(frame, none, ox)
    part = bulk.find_pairs(frame, 6.0, vectors_within_ang=3.0)
    assert np.isnan(part.vec_ang[part.d_ang > 3.0]).all()
    assert np.isfinite(part.vec_ang[part.d_ang <= 3.0]).all()
    with pytest.raises(ValueError, match="only within"):
        bulk.valence_table(frame, part, ox)


# ---------------------------------------------------------------------------
# parameters, and what is counted rather than dropped
# ---------------------------------------------------------------------------

class _CountingSet(bv.ParameterSet):
    """bv.DEFAULT's lookups, counted."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = []

    def get(self, cation, cation_ox, anion):
        self.calls.append((cation, cation_ox, anion))
        return super().get(cation, cation_ox, anion)


def test_parameters_are_looked_up_once_per_combination():
    """One get per (cation, state, anion) present, in analyse_site's order,
    however many contacts there are."""
    _, frame, ox = _example("eulytite_Bi4SiO4_3_cod9012894.cif")
    counting = _CountingSet()
    table = bulk.valence_table(frame, bulk.find_pairs(frame, 6.0), ox, counting)
    assert sorted(counting.calls) == [("Bi", 3, "O"), ("Si", 4, "O")]
    assert table.params.category == {("Bi", 3, "O"): "fitted",
                                     ("Si", 4, "O"): "fitted"}


def _unparameterised_frame():
    """Tc with six O, beside a SiO4 group, in a 12 Å box; jittered.

    Technetium because no table FACET holds has a Tc-O parameter: Brese &
    O'Keeffe's Table 2 has no Tc row, and the estimator gives none either.
    (Titanium was used before, but its Table 2 rows were found misfiled in
    bv.py, which would make this test depend on that defect.)"""
    frac = [[0.25, 0.25, 0.25],
            [0.41, 0.25, 0.25], [0.09, 0.25, 0.25], [0.25, 0.41, 0.25],
            [0.25, 0.09, 0.25], [0.25, 0.25, 0.41], [0.25, 0.25, 0.09],
            [0.75, 0.75, 0.75], [0.88, 0.75, 0.75], [0.62, 0.75, 0.75],
            [0.75, 0.88, 0.75], [0.75, 0.62, 0.75]]
    symbols = ["Tc"] + ["O"] * 6 + ["Si"] + ["O"] * 4
    return _jittered(symbols, frac, 12.0, seed=5)


def test_a_pair_with_no_parameter_is_counted_not_given_zero():
    """Tc(IV)-O has no parameter in Table 2, and with the estimator off none
    is estimated. Its contacts get no valence, enter no sum, and are counted
    per atom and per pair, as analyse_site lists them."""
    frame = _unparameterised_frame()
    states = {"Tc": 4}
    ox = md_model.model_oxidation(frame.species, states).per_atom(frame.elements)
    strict = bv.ParameterSet(allow_estimated=False)
    assert strict.get("Tc", 4, "O") is None
    table, results = bulk.analyse_frame(frame, ox, strict)
    tc = int(np.flatnonzero(frame.elements == "Tc")[0])
    assert table.n_listed[tc] == 0 and np.isnan(results.bvs_vu[tc])
    assert table.params.category[("Tc", 4, "O")] == "none"
    crystal = _crystal(frame, states, strict)
    without = [sum(1 for c in r.contacts if not c.has_valence) for r in crystal]
    assert table.n_unparameterised.tolist() == without
    assert table.n_unparameterised[tc] >= 6
    assert table.missing_pairs == {"Tc4+-O": int(table.n_unparameterised[tc])}
    assert any("no bond-valence parameter for Tc4+-O" in n for n in table.notes)
    assert any("Tc-O" in n for n in crystal[tc].notes)
    _same_as_crystal(results, crystal)
    # the oxygens around Tc have their Si contacts only, never a zero term
    o_near_tc = table.nbr[tc][table.nbr[tc] >= 0]
    assert o_near_tc.size == 0
    assert (table.v_vu[table.v_vu > 0] > table.v_list_vu).all()


class _OneEstimated(bv.ParameterSet):
    """bv.DEFAULT, except Si(IV)-O is returned as an estimated parameter.

    A test double: the R0 is Table 2's own value, only the fitted flag is
    changed, so no parameter value is invented.
    """

    def get(self, cation, cation_ox, anion):
        p = super().get(cation, cation_ox, anion)
        if p is not None and (p.cation, p.cation_ox, p.anion) == ("Si", 4, "O"):
            return dataclasses.replace(p, fitted=False, source="test double")
        return p


def test_estimated_parameters_are_flagged_per_atom_and_per_pair():
    _, frame, ox = _example("quartz_SiO2_cod9013321.cif")
    params = _OneEstimated()
    table, results = bulk.analyse_frame(frame, ox, params)
    assert table.uses_estimated.all()
    assert table.params.category[("Si", 4, "O")] == "estimated"
    n_si = int((frame.elements == "Si").sum())
    within = bulk.find_pairs(frame, table.r_search_ang)
    from_si = (frame.elements[within.i] == "Si") & (frame.elements[within.j] == "O")
    assert table.estimated_pairs == {"Si4+-O": int(from_si.sum())}
    assert n_si and any("estimated" in n and "test double" in n
                        for n in table.notes)
    crystal = _crystal(frame, {}, params)
    assert all(r.uses_estimated_params for r in crystal)
    _same_as_crystal(results, crystal)


def test_an_oxyfluoride_matches_analyse_site():
    """Al bonded to both O and F, Na beside them: two anions, so the anion
    split and the per-pair parameters are both exercised."""
    frac = [[0.5, 0.5, 0.5],
            [0.69, 0.5, 0.5], [0.31, 0.5, 0.5], [0.5, 0.69, 0.5],
            [0.5, 0.31, 0.5], [0.5, 0.5, 0.685], [0.5, 0.5, 0.315],
            [0.1, 0.1, 0.1], [0.85, 0.15, 0.3], [0.2, 0.8, 0.85]]
    symbols = ["Al", "O", "O", "O", "O", "F", "F", "Na", "Na", "Na"]
    frame = _jittered(symbols, frac, 9.5, seed=11, origin=np.array([3.0, -2.0, 1.0]))
    ox = md_model.model_oxidation(frame.species).per_atom(frame.elements)
    for v_bond in (0.04, bv.V_BOND_DEFAULT, 0.15):
        table, results = bulk.analyse_frame(frame, ox, v_bond_vu=v_bond)
        _same_as_crystal(results, _crystal(frame, {}, v_bond=v_bond), v_bond)
    al = int(np.flatnonzero(frame.elements == "Al")[0])
    bonded = table.nbr[al, :results.cn[al]]
    assert set(frame.elements[bonded].tolist()) == {"O", "F"}
    assert set(table.params.category.values()) == {"fitted"}


def test_a_pair_closer_than_d_min_is_counted_and_left_out():
    """Two atoms 0.3 Å apart: two ordered pairs below the 0.4 Å NeighborFinder
    dmin, counted in n_below_d_min and in a note, and absent from both atoms'
    contacts, as in the crystal path."""
    frame = md_model.frame_from_arrays(
        ["Si", "O", "O", "Na"],
        [[2.0, 2.0, 2.0], [2.3, 2.0, 2.0], [3.7, 2.1, 2.0], [6.0, 6.5, 7.0]],
        box_ang=np.eye(3) * 13.0)
    pairs = bulk.find_pairs(frame, 6.0)
    assert pairs.n_below_d_min == 2
    assert not ((pairs.i == 0) & (pairs.j == 1)).any()
    ox = md_model.model_oxidation(frame.species).per_atom(frame.elements)
    table, results = bulk.analyse_frame(frame, ox)
    assert table.n_below_d_min == 2
    note = next(n for n in table.notes if "0.4" in n)
    assert note.startswith("2 ordered pair(s)") and "0.3000 Å" in note
    _same_as_crystal(results, _crystal(frame, {}))


def test_an_atom_without_an_oxidation_state_is_refused():
    frame = md_model.frame_from_arrays(["Si", "O", "O"], np.eye(3) * 2.0,
                                       box_ang=np.eye(3) * 10.0)
    for ox in (np.array([4, None, -2], dtype=object), [4.0, np.nan, -2.0]):
        with pytest.raises(ValueError, match="no oxidation state"):
            bulk.analyse_frame(frame, ox)
    with pytest.raises(ValueError, match="one oxidation state per atom"):
        bulk.analyse_frame(frame, [4, -2])
    with pytest.raises(ValueError, match="booleans"):
        bulk.analyse_frame(frame, [True, False, False])
    with pytest.raises(ValueError, match="whole numbers"):
        bulk.analyse_frame(frame, [4.5, -2, -2])


def test_a_state_beyond_the_int64_cast_is_refused_not_flipped():
    """A float state of 1e20 passed the whole-number check and cast to
    -2**63 with only a RuntimeWarning, so the cation became an anion. States
    beyond OX_ABS_LIMIT are refused, from floats, Python ints and int64."""
    frame = md_model.frame_from_arrays(["Si", "O"], [[1.0, 1, 1], [2.6, 1, 1]],
                                       box_ang=np.eye(3) * 13.0)
    for ox in ([1e20, -2.0], [4, -1e300], np.array([2 ** 62, -2]),
               np.array([2 ** 70, -2], dtype=object)):
        with pytest.raises(ValueError, match="OX_ABS_LIMIT"):
            bulk.analyse_frame(frame, ox)
    table, _ = bulk.analyse_frame(frame, [bulk.OX_ABS_LIMIT, -2])
    assert table.ox.tolist() == [bulk.OX_ABS_LIMIT, -2]
    assert table.is_anion.tolist() == [False, True]


def test_an_element_normalise_would_rename_gets_no_parameter():
    """ParameterSet.get normalises its arguments, so 'Kr' would be looked up
    as potassium. The engine gives such a pair no parameter and says why.
    Kr at state 1, so the trap is Table 2's fitted K(I)-O; at another state
    it would reach the estimator, which needs pymatgen, an optional extra."""
    frame = md_model.frame_from_arrays(
        ["Kr", "O", "O", "Si"], [[1.0, 1, 1], [3.0, 1, 1], [1.0, 3, 1],
                                 [5.0, 5, 5]], box_ang=np.eye(3) * 12.0)
    trap = bv.DEFAULT.get("Kr", 1, "O")
    assert trap is not None and trap.fitted
    table, _ = bulk.analyse_frame(frame, [1, -2, -2, 4])
    kr = 0
    assert table.params.category[("Kr", 1, "O")] == "none"
    assert table.n_listed[kr] == 0 and table.n_unparameterised[kr] >= 2
    assert any("Kr (elements.normalise reads it as K)" in n for n in table.notes)


def test_a_state_of_zero_is_on_the_cation_side_and_noted():
    frame = md_model.frame_from_arrays(["Ar", "O"], [[1.0, 1, 1], [4.0, 1, 1]],
                                       box_ang=np.eye(3) * 12.0)
    table, _ = bulk.analyse_frame(frame, [0, -2])
    assert table.is_anion.tolist() == [False, True]
    assert any("state 0 count on the cation side" in n for n in table.notes)


# ---------------------------------------------------------------------------
# the search radius and the constants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", EXAMPLES)
def test_the_search_radius_is_search_radius_fors(name):
    structure, frame, ox = _example(name)
    for floor in (bulk.RADIUS_FLOOR_ANG, 0.0):
        assert bulk.search_radius_ang(frame.elements, ox, bv.DEFAULT,
                                      bv.V_LIST_DEFAULT, floor_ang=floor) == \
            search_radius_for(structure, bv.DEFAULT, bv.V_LIST_DEFAULT,
                              floor=floor)


def test_two_states_of_one_element_widen_the_radius_never_narrow_it():
    """search_radius_for keeps one state per element (the last cation site's);
    the engine takes every species, so Fe(II) and Fe(III) give Fe(III)'s
    longer cutoff whichever site comes last."""
    cell = readers.cell_from_vectors(np.eye(3) * 10.0)
    sites = [Site("Fe1", "Fe", np.zeros(3), ox=3, ox_source="user"),
             Site("Fe2", "Fe", np.full(3, 0.5), ox=2, ox_source="user"),
             Site("O1", "O", np.array([0.2, 0.0, 0.0]), ox=-2, ox_source="user")]
    atoms = [Atom(s.element, s.frac, cell.to_cartesian(s.frac), k, s.label)
             for k, s in enumerate(sites)]
    structure = Structure("FeO test", cell, sites, atoms)
    crystal = search_radius_for(structure, bv.DEFAULT, bv.V_LIST_DEFAULT, floor=0.0)
    mine = bulk.search_radius_ang(np.array(["Fe", "Fe", "O"]), [3, 2, -2],
                                  bv.DEFAULT, bv.V_LIST_DEFAULT, floor_ang=0.0)
    fe3 = bv.DEFAULT.get("Fe", 3, "O").distance_for(bv.V_LIST_DEFAULT)
    fe2 = bv.DEFAULT.get("Fe", 2, "O").distance_for(bv.V_LIST_DEFAULT)
    assert fe3 > fe2
    assert crystal == fe2 and mine == fe3


def test_the_constants_are_neighborfinders_defaults():
    finder = inspect.signature(NeighborFinder.__init__).parameters
    assert bulk.D_MIN_ANG == finder["dmin"].default
    radius = inspect.signature(search_radius_for).parameters
    assert bulk.RADIUS_FLOOR_ANG == radius["floor"].default
    assert bulk.RADIUS_CEILING_ANG == radius["ceiling"].default
    mine = inspect.signature(bulk.search_radius_ang).parameters
    assert mine["floor_ang"].default == radius["floor"].default
    assert mine["ceiling_ang"].default == radius["ceiling"].default
    assert inspect.signature(bulk.iter_pairs).parameters["d_min_ang"].default \
        == finder["dmin"].default
    for function in (bulk.valence_table, bulk.analyse_frame):
        assert inspect.signature(function).parameters["v_list_vu"].default \
            == bv.V_LIST_DEFAULT
    assert inspect.signature(bulk.at_threshold).parameters["v_bond_vu"].default \
        == bv.V_BOND_DEFAULT


# ---------------------------------------------------------------------------
# a physical law, and the bonds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", EXAMPLES)
def test_cation_and_anion_valence_sums_balance(name):
    """Each listed cation-anion contact is in one cation's sum and one anion's
    sum with the same valence, so the two totals agree, and so do the contact
    counts, exactly; at v_bond too, and the bonds are those counts.

    The law is checked bond by bond first: the (cation, anion, image,
    valence) entries read from the cation rows and from the anion rows (image
    negated) are the same multiset, valences compared bit for bit. Comparing
    only the two totals needs a tolerance, and at MD size the order of a
    float sum alone moves a 10 000 v.u. total by more than 1e-10."""
    _, frame, ox = _example(name)
    table, results = bulk.analyse_frame(frame, ox)
    cation, anion = ~table.is_anion, table.is_anion
    valid = np.arange(table.v_vu.shape[1])[None, :] < table.n_listed[:, None]
    for v_bond in (table.v_list_vu, bv.V_BOND_DEFAULT):
        take = valid & (table.v_vu > v_bond)
        rows, cols = np.nonzero(take & cation[:, None])
        from_cations = sorted(zip(
            rows.tolist(), table.nbr[rows, cols].tolist(),
            map(tuple, table.image[rows, cols].tolist()),
            table.v_vu[rows, cols].tolist()))
        rows, cols = np.nonzero(take & anion[:, None])
        from_anions = sorted(zip(
            table.nbr[rows, cols].tolist(), rows.tolist(),
            map(tuple, (-table.image[rows, cols].astype(int)).tolist()),
            table.v_vu[rows, cols].tolist()))
        assert from_cations == from_anions and len(from_cations) > 0, v_bond
    for sums, counts in ((results.bvs_listed_vu, results.cn_listed),
                         (results.bvs_vu, results.cn)):
        assert abs(np.nansum(sums[cation]) - np.nansum(sums[anion])) <= ATOL
        assert counts[cation].sum() == counts[anion].sum() > 0
    bonds = bulk.bonds_at(table)
    assert len(bonds) == results.cn[anion].sum()
    assert np.allclose(np.linalg.norm(bonds.vec_ang, axis=1), bonds.d_ang,
                       rtol=0, atol=1e-12)
    assert (~table.is_anion[bonds.cation]).all() and table.is_anion[bonds.anion].all()
    per_anion = np.bincount(bonds.anion, minlength=frame.n_atoms)
    assert np.array_equal(per_anion[anion], results.cn[anion])


def test_the_distance_cut_cn_counts_neighbours_within_the_cutoff():
    """Si-O and O-Si within 2.0 Å in quartz: 4 and 2, as NeighborFinder's
    contacts give; rows of other elements hold -1; a cutoff beyond the search
    radius is refused."""
    _, frame, _ = _example("quartz_SiO2_cod9013321.cif")
    pairs = bulk.find_pairs(frame, 6.0)
    cut = {("Si", "O"): 2.0, ("O", "Si"): 2.0}
    counts = bulk.distance_cn(pairs, frame.elements, cut)
    si, o = frame.elements == "Si", frame.elements == "O"
    assert (counts[("Si", "O")][si] == 4).all()
    assert (counts[("Si", "O")][o] == -1).all()
    assert (counts[("O", "Si")][o] == 2).all()
    blocks = bulk.distance_cn(bulk.iter_pairs(frame, 6.0, block_atoms=10),
                              frame.elements, cut)
    for key in cut:
        assert np.array_equal(blocks[key], counts[key])
    structure = md_model.frame_to_structure(
        frame, md_model.model_oxidation(frame.species))
    finder = NeighborFinder(structure, rmax=6.0)
    for k in range(frame.n_atoms):
        c = finder.contacts(k)
        other = "O" if frame.elements[k] == "Si" else "Si"
        want = sum(1 for e, d in zip(c.elements, c.distance)
                   if e == other and d <= 2.0)
        assert counts[(str(frame.elements[k]), other)][k] == want
    with pytest.raises(ValueError, match="beyond"):
        bulk.distance_cn(pairs, frame.elements, {("Si", "O"): 7.0})
    with pytest.raises(ValueError, match="above 0"):
        bulk.distance_cn(pairs, frame.elements, {("Si", "O"): -1.0})


def test_the_distance_cut_cn_refuses_blocks_that_miss_or_repeat_atoms():
    """A generator from iter_pairs yields its blocks once. Passed to
    distance_cn after valence_table had drained it, it gave CN 0 for every
    atom with no message; blocks given twice doubled every count, and a
    missing block left its atoms at 0. Each is refused, as valence_table
    refuses them."""
    _, frame, ox = _example("quartz_SiO2_cod9013321.cif")
    cut = {("Si", "O"): 2.0}
    pairs = bulk.iter_pairs(frame, 6.0)
    bulk.valence_table(frame, pairs, ox)
    with pytest.raises(ValueError, match="no pair table"):
        bulk.distance_cn(pairs, frame.elements, cut)
    blocks = list(bulk.iter_pairs(frame, 6.0, block_atoms=10))
    with pytest.raises(ValueError, match="more than once"):
        bulk.distance_cn(blocks + blocks[:1], frame.elements, cut)
    with pytest.raises(ValueError, match="not at all"):
        bulk.distance_cn(blocks[1:], frame.elements, cut)
    with pytest.raises(ValueError, match="no pair table"):
        bulk.distance_cn([], frame.elements, cut)
    whole = bulk.distance_cn(blocks, frame, cut)
    assert (whole[("Si", "O")][frame.elements == "Si"] == 4).all()


# ---------------------------------------------------------------------------
# refusals, no verdicts, no Qt
# ---------------------------------------------------------------------------

def test_bad_arguments_are_refused_with_a_message():
    _, frame, ox = _example("quartz_SiO2_cod9013321.cif")
    table, _ = bulk.analyse_frame(frame, ox)
    for call, match in (
            (lambda: next(bulk.iter_pairs(frame, -1.0)), "r_ang"),
            (lambda: next(bulk.iter_pairs(frame, float("nan"))), "r_ang"),
            (lambda: next(bulk.iter_pairs(frame, 0.3)), "d_min_ang"),
            (lambda: next(bulk.iter_pairs(frame, 6.0, method="fast")), "method"),
            (lambda: next(bulk.iter_pairs(frame, 6.0, block_atoms=0)), "block"),
            (lambda: next(bulk.iter_pairs("frame", 6.0)), "Frame"),
            (lambda: bulk.at_threshold(table, float("inf")), "finite"),
            (lambda: bulk.at_threshold("table"), "ValenceTable"),
            (lambda: bulk.valence_table(frame, [], ox), "no pair table"),
            (lambda: bulk.search_radius_ang(frame.elements, ox, bv.DEFAULT, 0.0),
             "v_list_vu")):
        with pytest.raises(ValueError, match=match):
            call()


VERDICT = re.compile(
    r"\b(good|bad|poor|excellent|acceptable|unacceptable|correct|incorrect|"
    r"wrong|reliable|unreliable|trustworthy|untrustworthy|unusable|should|"
    r"proves|confirms)\b", re.IGNORECASE)


def _every_note_and_message() -> list[str]:
    out: list[str] = []
    frame = _unparameterised_frame()
    ox = md_model.model_oxidation(frame.species, {"Tc": 4}).per_atom(frame.elements)
    table, results = bulk.analyse_frame(frame, ox, bv.ParameterSet(
        allow_estimated=False))
    out += list(table.notes) + list(table.params.notes) + list(results.notes)
    out += list(bulk.analyse_frame(*_example("quartz_SiO2_cod9013321.cif")[1:],
                                   _OneEstimated())[0].notes)
    close = md_model.frame_from_arrays(["Si", "O", "Kr", "Ar"],
                                       [[2.0, 2, 2], [2.3, 2, 2], [6.0, 6, 6],
                                        [8.0, 2, 2]], box_ang=np.eye(3) * 13.0)
    out += list(bulk.analyse_frame(close, [4, -2, 2, 0])[0].notes)
    out += list(bulk.analyse_frame(close, [4, 2, 2, 0])[0].notes)  # no anion
    _, quartz, qox = _example("quartz_SiO2_cod9013321.cif")
    out += list(bulk.valence_table(quartz, bulk.find_pairs(quartz, 2.5),
                                   qox).notes)
    out += list(bulk.analyse_frame(quartz, qox, v_list_vu=1e-14)[0].notes)
    tilted = md_model.frame_from_arrays(["Si", "O"], [[0.0, 0, 0], [1.6, 0, 0]],
                                        box_ang=np.array([[20.0, 0, 0],
                                                          [1.0, 20.0, 0],
                                                          [0, 0, 20.0]]))
    needle = md_model.frame_from_arrays(["Si"], frac=[[0.1, 0.1, 0.1]],
                                        box_ang=np.diag([10.0, 10.0, 1.5e-4]))
    other = md_model.frame_from_arrays(quartz.elements, frac=quartz.frac[::-1],
                                       box_ang=quartz.box_ang)
    drained = bulk.iter_pairs(quartz, 6.0)
    list(drained)
    for call in (lambda: next(bulk.iter_pairs(tilted, 6.0, method="boxsize")),
                 lambda: next(bulk.iter_pairs(quartz, 6.0, method="boxsize")),
                 lambda: bulk.analyse_frame(quartz, [None] * quartz.n_atoms),
                 lambda: bulk.analyse_frame(quartz, qox * 1e19),
                 lambda: bulk.valence_table(quartz, list(bulk.iter_pairs(
                     quartz, 6.0, block_atoms=10))[1:], qox),
                 lambda: bulk.valence_table(other, bulk.find_pairs(quartz, 6.0),
                                            qox),
                 lambda: bulk.distance_cn(drained, quartz.elements,
                                          {("Si", "O"): 2.0}),
                 lambda: bulk.distance_cn(bulk.find_pairs(quartz, 6.0),
                                          quartz.elements, {("Si", "O"): 9.0}),
                 lambda: bulk.find_pairs(needle, 6.0),
                 lambda: bulk.find_pairs(tilted, 1e4),
                 lambda: bulk.iter_pairs(tilted, 1e6)):
        try:
            call()
        except ValueError as error:
            out.append(str(error))
        else:
            raise AssertionError("expected a refusal")
    return out


def test_no_note_or_message_carries_a_verdict():
    notes = _every_note_and_message()
    assert len(notes) >= 10
    assert [n for n in notes if VERDICT.search(n)] == []


def test_no_string_in_the_module_carries_a_verdict():
    """Every string literal in bulk.py, docstrings included."""
    source = (ROOT / "facet" / "core" / "bulk.py").read_text(encoding="utf-8")
    strings = [node.value for node in ast.walk(ast.parse(source))
               if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert len(strings) > 50
    assert [s for s in strings if VERDICT.search(s)] == []


def test_the_engine_imports_without_qt():
    code = ("import sys, facet.core.bulk;"
            "print('QT' if any(m.startswith('PySide6') for m in sys.modules) "
            "else 'CLEAN')")
    result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout


def test_the_engine_never_reaches_the_per_site_crystal_path():
    """The reason the module exists: no coordination, oxidation or
    Structure.atoms_of_site on the MD path (Step 0 measured them at 17-28 s
    per 9 261-atom frame)."""
    source = (ROOT / "facet" / "core" / "bulk.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {alias.name for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) for alias in node.names}
    imported |= {node.module for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and node.module}
    assert not imported & {"coordination", "oxidation", "structure"}
    assert "atoms_of_site" not in source.replace(
        "Structure.atoms_of_site", "")
