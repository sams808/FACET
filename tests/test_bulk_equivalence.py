"""The crystal-as-glass equality: the bulk engine is the crystal path, unrolled.

The bulk engine (``facet.core.bulk``) is a second implementation of what
``coordination.analyse_site`` computes, written for arrays instead of objects.
The check that it is right is the one ``VERIFICATION.md`` asks of any second
implementation: give both the same input and require the same numbers. A
crystal is read, its oxidation states resolved by the crystal path, its
expanded cell tiled 2 x 2 x 2 into a frame with ``md_model.supercell_frame``,
and the frame analysed by ``bulk`` with the same per-atom states. Every
supercell atom then has to reproduce, to 1e-10 with no relative slack and NaN
equal only to NaN, what ``analyse_site`` gives for the home-cell atom it was
copied from (``NeighborFinder(st, rmax).contacts(k)``), field by field: CN,
CN at v_list, the two bond-valence sums, |BVV| and the BVV vector over each
set, the two phi values, the valence discrepancy and the plateau width, at
the default v_bond and at one threshold either side.

CN is exact only where no contact valence lies within a rounding of v_bond.
The two paths form a contact distance from different roundings (here
(f_j - f_i + n) . (a, b, c), in NeighborFinder (image position) - centre), so
a valence can differ in its last bits between them: measured on 34-72 % of
the listed contacts of the home-cell atoms of the eleven files, by up to 51
units in the last place (1.1e-14 v.u.). A threshold set exactly on such a
valence counts the contact on one path and not the other. Every threshold
used here is therefore checked to sit more than 1e-9 v.u. from every listed
valence, so a later change of THRESHOLDS cannot land on an edge unnoticed;
the thresholds below sit at least 7.0e-4 v.u. from every listed valence of
the eleven files.

WHY THE PARENT ATOM AND NOT THE PARENT SITE
-------------------------------------------
The specification asked for equality with "the atom's site". That cannot pass
on quartz or BiPO4 with any engine: those CIFs round their special positions
(quartz Si z = 0.66667; BiPO4 Bi and P at z = 0.83333 and 0.33333), and
``cif._expand`` keeps the first copy of each position without symmetrising, so
symmetry copies of one site already differ inside the crystal. The agreed
reference (Step 0, question 1, default kept) is the parent atom, to 1e-10, with
the coordination number also required to equal the parent site's exactly. The
residual against the site is computed here and recorded with
``record_property``; it is not asserted. Measured (largest over the supercell
atoms, at the default thresholds, against ``analyse_structure``'s site):

=========================  ==========  ==========  =============  ==========
file                       BVS (v.u.)  phi         plateau (dec)  |BVV| (v.u.)
=========================  ==========  ==========  =============  ==========
quartz                     1.1e-04     7.7e-06     3.0e-05        4.3e-05
BiPO4                      5.9e-05     3.5e-05     4.4e-05        8.7e-05
cryolite                   2.7e-15     1.1e-15     1.8e-15        9.4e-16
eulytite                   4.4e-15     8.9e-16     2.2e-15        4.7e-15
senarmontite               1.3e-15     1.7e-16     0              7.8e-16
valentinite                4.9e-15     6.1e-16     1.8e-15        2.7e-15
1526458 alpha-Bi2O3        4.7e-15     7.2e-16     1.9e-15        2.4e-15
1526788 NaBiO3             1.1e-14     1.8e-15     2.2e-15        9.2e-15
2104291 BiB3O6             2.2e-15     6.7e-16     2.2e-15        1.3e-15
1004091 Na3Bi(PO4)2        2.5e-14     7.3e-15     5.6e-15        1.7e-14
1541071 NaSbOSiO4          9.8e-15     1.3e-15     4.4e-15        6.7e-15
=========================  ==========  ==========  =============  ==========

Against the parent atom the largest difference over all eleven files, all
three thresholds and every field was 2.5e-14 (BVS, 1004091 Na3Bi(PO4)2),
with every CN equal.

WHAT ELSE IS PINNED
-------------------
* **Invariance** (``VERIFICATION.md`` §3, and the MD prompt's wrapped against
  unwrapped input): translating every atom by a random vector (re-wrapped
  into the box), permuting the atom order (un-permuted after), a rigid
  rotation of box and atoms (the BVV vector rotating with them), a
  reflection, a cyclic relabelling of the box axes, positions given
  unwrapped up to three boxes away, and a box corner moved to -L/2 change
  nothing beyond 1e-10 (measured at most 7.4e-14, and exactly 0 for the
  corner, since distances never read the origin). On an orthogonal crystal
  the reflection leaves a diagonal box with a negative length and the
  relabelling one whose diagonal is zero; the periodic tree refuses both,
  so the image path is exercised on boxes it would otherwise not see.
* **Supercell size and shape**: 3 x 3 x 3 and 2 x 3 x 1 give the per-atom
  values of 2 x 2 x 2. A 2 x 2 x 2 supercell of a small cell already needs
  images beyond the minimum one (cryolite's is 10.8 Å wide against a 6 Å
  search), so this catches an image rule that only works for large boxes.

The six bundled examples are byte-for-byte copies of ``facet/data/examples``
in ``tests/data/crystals`` (public-domain COD entries; see ATTRIBUTION.md
there), pinned by their SHA-256, so these tests never skip and never read
``facet/data``. The five reference structures come from
the reference collection through ``conftest.sample_cif`` and skip one by one
when it is absent; between them they add Bi(V), three-and-four-coordinated B,
an 896-atom supercell and an orthogonal box the periodic tree is used on.
"""
from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from conftest import sample_cif
from facet.core import bulk, bv, coordination, md_model, readers
from facet.core.neighbors import NeighborFinder, search_radius_for

DATA = Path(__file__).resolve().parent / "data" / "crystals"

# Named, not globbed, so a missing copy fails instead of shrinking the test.
EXAMPLES = ("quartz_SiO2_cod9013321.cif",
            "bismuth_phosphate_BiPO4_cod9008088.cif",
            "cryolite_Na3AlF6_cod9004097.cif",
            "eulytite_Bi4SiO4_3_cod9012894.cif",
            "senarmontite_Sb2O3_cod9009747.cif",
            "valentinite_Sb2O3_cod9007587.cif")

REFERENCES = (("1526458", "1526458_Bi2O3.cif"),
              ("1526788", "1526788_Na(BiO3).cif"),
              ("2104291", "2104291_BiB3O6.cif"),
              ("1004091", "1004091_BiNa3O8P2.cif"),
              ("1541071", "1541071_NaSbOSiO4_Pagnoux1992.cif"))

# The default bond threshold and one value either side of it.
THRESHOLDS = (0.05, bv.V_BOND_DEFAULT, 0.10)
ATOL = 1e-10
# How far every threshold has to sit from every listed valence (see the
# module docstring): a test choice, many orders above the last-bit
# differences between the two paths' valences.
EDGE_MARGIN_VU = 1e-9

# bulk.AtomResults field -> coordination.SiteResult field; every scalar field
# AtomResults documents as a SiteResult field (bvv_vector is compared apart)
FIELDS = (("cn", "cn_valence"), ("cn_listed", "cn_listed"),
          ("bvs_vu", "bvs"), ("bvs_listed_vu", "bvs_listed"),
          ("bvv_vu", "bvv"), ("bvv_listed_vu", "bvv_listed"),
          ("phi", "phi"), ("phi_listed", "phi_listed"),
          ("plateau_decades", "plateau_decades"),
          ("valence_discrepancy_vu", "valence_discrepancy"))


def _cases():
    out = [pytest.param(DATA / name, id=name.split("_")[0]) for name in EXAMPLES]
    for code, hint in REFERENCES:
        path = Path(sample_cif(code, hint))
        out.append(pytest.param(
            path, id=code, marks=pytest.mark.skipif(
                not path.exists(), reason=f"reference structure {hint} absent")))
    return out


CASES = _cases()


@dataclass
class Case:
    structure: object
    sites: dict                  # site index -> SiteResult (analyse_structure)
    per_atom: dict               # v_bond -> [SiteResult per home-cell atom]
    frame: md_model.Frame
    parent: np.ndarray
    ox: np.ndarray
    table: bulk.ValenceTable


def _parent_ox(structure, parent: np.ndarray) -> np.ndarray:
    """Each supercell row's state: its parent atom's site, as resolved."""
    return np.array([structure.sites[structure.atoms[p].site_index].ox
                     for p in parent], dtype=np.int64)


@functools.lru_cache(maxsize=None)
def _case(path: Path) -> Case:
    structure = readers.read(path)
    # resolves the oxidation states in place, as the crystal window does
    sites = {r.site_index: r for r in coordination.analyse_structure(
        structure, cations_only=False)}
    rmax = search_radius_for(structure, bv.DEFAULT, bv.V_LIST_DEFAULT)
    finder = NeighborFinder(structure, rmax=rmax)
    per_atom = {v: [coordination.analyse_site(structure, finder.contacts(k),
                                              bv.DEFAULT, v, bv.V_LIST_DEFAULT)
                    for k in range(structure.n_atoms)]
                for v in THRESHOLDS}
    frame, parent = md_model.supercell_frame(structure, (2, 2, 2))
    ox = _parent_ox(structure, parent)
    table, _ = bulk.analyse_frame(frame, ox, bv.DEFAULT)
    return Case(structure, sites, per_atom, frame, parent, ox, table)


def _assert_close(name: str, got, want) -> None:
    """|got - want| <= ATOL everywhere, NaN equal only to NaN."""
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    nan_got, nan_want = np.isnan(got), np.isnan(want)
    mismatch = np.flatnonzero(nan_got != nan_want)
    assert mismatch.size == 0, (
        f"{name}: NaN in one result only at rows {mismatch[:10].tolist()}")
    both = ~nan_got
    inf_got, inf_want = np.isinf(got), np.isinf(want)
    assert np.array_equal(inf_got, inf_want), f"{name}: infinite in one only"
    finite = both & ~inf_got
    worst = float(np.max(np.abs(got[finite] - want[finite]), initial=0.0))
    assert worst <= ATOL, f"{name}: largest difference {worst:.3g}"
    assert np.array_equal(got[inf_got], want[inf_want])


def _same_results(a: bulk.AtomResults, b: bulk.AtomResults, rows=None) -> None:
    """Every AtomResults field of a equals b (b indexed by rows when given)."""
    for name in ("cn", "cn_listed", "bvs_vu", "bvs_listed_vu", "bvv_vu",
                 "bvv_listed_vu", "phi", "phi_listed", "plateau_decades",
                 "valence_discrepancy_vu"):
        want = getattr(b, name)
        _assert_close(name, getattr(a, name), want if rows is None
                      else want[rows])


# ---------------------------------------------------------------------------
# the equality
# ---------------------------------------------------------------------------

def _listed_valences(table: bulk.ValenceTable) -> np.ndarray:
    valid = np.arange(table.v_vu.shape[1])[None, :] < table.n_listed[:, None]
    return table.v_vu[valid]


@pytest.mark.parametrize("path", CASES)
@pytest.mark.parametrize("v_bond", THRESHOLDS)
def test_every_supercell_atom_equals_its_parent_atom(path, v_bond):
    """CN, CN_listed, both sums, |BVV| and the BVV vector over each set, both
    phi values, the valence discrepancy and the plateau of every row equal
    analyse_site on the home-cell atom it was copied from. The threshold is
    first checked to sit off every step edge (module docstring)."""
    case = _case(path)
    gap = float(np.abs(_listed_valences(case.table) - v_bond).min())
    assert gap > EDGE_MARGIN_VU, (
        f"v_bond {v_bond} lies {gap:.3g} v.u. from a contact valence; the "
        "two paths' last-bit differences can then change CN")
    results = bulk.at_threshold(case.table, v_bond)
    crystal = case.per_atom[v_bond]
    for bulk_name, site_name in FIELDS:
        want = [getattr(crystal[p], site_name) for p in case.parent]
        want = np.array([np.nan if w is None else w for w in want],
                        dtype=np.float64)
        _assert_close(f"{path.name} {bulk_name} at {v_bond}",
                      getattr(results, bulk_name), want)
    _assert_close(f"{path.name} bvv_vector_vu at {v_bond}",
                  results.bvv_vector_vu,
                  np.array([crystal[p].bvv_vector for p in case.parent]))


@pytest.mark.parametrize("path", CASES)
def test_cn_equals_the_parent_site_and_the_site_residual_is_recorded(
        path, record_property):
    """CN equals the parent SITE's exactly at every threshold. The BVS / phi /
    plateau / |BVV| residual against the site is recorded, not asserted (see
    the module docstring for why it is not 1e-10 on quartz and BiPO4)."""
    case = _case(path)
    st = case.structure
    site_of_row = np.array([st.atoms[p].site_index for p in case.parent])
    for v_bond in THRESHOLDS:
        results = bulk.at_threshold(case.table, v_bond)
        want = np.array([case.sites[s].cn_at(v_bond) for s in site_of_row])
        assert np.array_equal(results.cn, want), v_bond

    results = bulk.at_threshold(case.table, bv.V_BOND_DEFAULT)
    residual = {}
    for bulk_name, site_name in (("bvs_vu", "bvs"), ("phi", "phi"),
                                 ("plateau_decades", "plateau_decades"),
                                 ("bvv_vu", "bvv")):
        got = getattr(results, bulk_name)
        want = np.array([getattr(case.sites[s], site_name)
                         for s in site_of_row])
        both = np.isfinite(got) & np.isfinite(want)
        residual[site_name] = float(np.max(np.abs(got[both] - want[both]),
                                           initial=0.0))
        assert np.array_equal(np.isnan(got), np.isnan(want))
    record_property("site_residual", residual)


# ---------------------------------------------------------------------------
# invariance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", CASES)
def test_a_random_translation_changes_nothing(path):
    """Every atom moved by one random vector and wrapped back into the box:
    other atoms cross the box faces, other images are used, the same numbers
    come out."""
    case = _case(path)
    frame = case.frame
    shift = np.random.default_rng(20261006).uniform(-30.0, 30.0, 3)
    moved = md_model.frame_from_arrays(
        frame.elements, frame.cart_ang + shift, box_ang=frame.box_ang,
        origin_ang=frame.origin_ang, atom_id=frame.atom_id)
    assert np.abs(moved.frac - frame.frac).max() > 0.1      # atoms did wrap
    _, moved_results = bulk.analyse_frame(moved, case.ox, bv.DEFAULT)
    _same_results(moved_results, bulk.at_threshold(case.table))


@pytest.mark.parametrize("path", CASES)
def test_a_random_permutation_changes_nothing(path):
    """The atoms listed in a random order: row k of the permuted frame is row
    perm[k] of the original, and carries its numbers."""
    case = _case(path)
    frame = case.frame
    perm = np.random.default_rng(7).permutation(frame.n_atoms)
    shuffled = md_model.frame_from_arrays(
        frame.elements[perm], frac=frame.frac[perm], box_ang=frame.box_ang,
        origin_ang=frame.origin_ang)
    _, shuffled_results = bulk.analyse_frame(shuffled, case.ox[perm],
                                             bv.DEFAULT)
    _same_results(shuffled_results, bulk.at_threshold(case.table), rows=perm)


@pytest.mark.parametrize("path", CASES)
def test_a_3x3x3_supercell_gives_the_2x2x2_values(path):
    """Row r of the 3 x 3 x 3 frame comes from parent atom r % n, which is row
    r % n of the 2 x 2 x 2 frame (supercell_frame's row order)."""
    case = _case(path)
    st = case.structure
    frame3, parent3 = md_model.supercell_frame(st, (3, 3, 3))
    _, results3 = bulk.analyse_frame(frame3, _parent_ox(st, parent3),
                                     bv.DEFAULT)
    assert np.array_equal(case.parent[:st.n_atoms], np.arange(st.n_atoms))
    _same_results(results3, bulk.at_threshold(case.table), rows=parent3)


BUNDLED = [pytest.param(DATA / name, id=name.split("_")[0]) for name in EXAMPLES]


def _rotation(seed: int) -> np.ndarray:
    """A random proper rotation (QR of a Gaussian matrix, det +1)."""
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def _transformed(frame, kind: str):
    """(frame, M): the frame under one change of description, and the matrix
    a Cartesian vector is multiplied by on the right (row vectors)."""
    box, cart, origin = frame.box_ang, frame.cart_ang, frame.origin_ang
    if kind in ("rotation", "reflection"):
        m = _rotation(11).T if kind == "rotation" else np.diag([1.0, 1.0, -1.0])
        return md_model.frame_from_arrays(
            frame.elements, cart @ m, box_ang=box @ m, origin_ang=origin @ m,
            atom_id=frame.atom_id), m
    if kind == "relabelled axes":
        return md_model.frame_from_arrays(
            frame.elements, cart, box_ang=box[[1, 2, 0]], origin_ang=origin,
            atom_id=frame.atom_id), np.eye(3)
    if kind == "unwrapped":
        k = np.random.default_rng(5).integers(-3, 4, (frame.n_atoms, 3))
        moved = md_model.frame_from_arrays(
            frame.elements, cart + k @ box, box_ang=box, origin_ang=origin,
            atom_id=frame.atom_id)
        assert np.abs(k).max() == 3
        return moved, np.eye(3)
    if kind == "origin at -L/2":
        return md_model.frame_from_arrays(
            frame.elements, frac=frame.frac, box_ang=box,
            origin_ang=-0.5 * box.sum(axis=0), atom_id=frame.atom_id), np.eye(3)
    raise ValueError(kind)


@pytest.mark.parametrize("path", BUNDLED)
@pytest.mark.parametrize("kind", ["rotation", "reflection", "relabelled axes",
                                  "unwrapped", "origin at -L/2"])
def test_a_change_of_description_changes_nothing(path, kind):
    """The same model described another way: rotated rigidly, reflected,
    with its box axes relabelled a, b, c -> b, c, a, given as unwrapped
    positions up to three boxes out, or with its corner at -L/2. Every scalar
    is unchanged to 1e-10 and the BVV vector turns with the atoms. Measured
    when this test was written, over the six crystals: rotation 4.3e-14,
    reflection 2.6e-14, relabelling 2.6e-14, unwrapped 7.4e-14, corner 0."""
    case = _case(path)
    moved, m = _transformed(case.frame, kind)
    _, results = bulk.analyse_frame(moved, case.ox, bv.DEFAULT)
    reference = bulk.at_threshold(case.table)
    _same_results(results, reference)
    _assert_close(f"{kind} bvv_vector_vu", results.bvv_vector_vu,
                  reference.bvv_vector_vu @ m)


def test_a_reflection_of_an_orthogonal_box_takes_the_image_path():
    """diag(1, 1, -1) leaves eulytite's cubic box diagonal with a negative
    length; the periodic tree accepts such a box and measures distances in
    another one, so it is refused and the image path gives the numbers the
    previous test compares."""
    case = _case(DATA / "eulytite_Bi4SiO4_3_cod9012894.cif")
    assert case.frame.box_is_diagonal
    moved, _ = _transformed(case.frame, "reflection")
    assert moved.box_is_diagonal and (np.diag(moved.box_ang) < 0).any()
    assert bulk.find_pairs(moved, 6.0).method == "images"
    assert bulk.find_pairs(case.frame, 6.0).method == "boxsize"


@pytest.mark.parametrize("path", BUNDLED)
def test_a_2x3x1_supercell_gives_the_2x2x2_values(path):
    """A supercell of another shape: each row carries its parent atom's
    numbers, which are row parent of the 2 x 2 x 2 frame."""
    case = _case(path)
    st = case.structure
    frame, parent = md_model.supercell_frame(st, (2, 3, 1))
    _, results = bulk.analyse_frame(frame, _parent_ox(st, parent), bv.DEFAULT)
    _same_results(results, bulk.at_threshold(case.table), rows=parent)


# SHA-256 of each copy with its line endings as LF. When the copies were made
# (2026-10-06) every file was compared with facet/data/examples byte for byte
# and found identical; these digests pin that content without the tests
# reading facet/data, which is another session's uncommitted work. Line
# endings are normalised first because git's core.autocrlf may write CRLF on
# a Windows checkout; the CIF reader parses either.
COPY_SHA256 = {
    "ATTRIBUTION.md":
        "62849b1dcd8cca2084a63b251577a367ef2ac367b3983bd47f99e89f18aabdb5",
    "bismuth_phosphate_BiPO4_cod9008088.cif":
        "15be1c944a118926e4297ba6760f3516455135df9bc33a6bb4d686b18b193d56",
    "cryolite_Na3AlF6_cod9004097.cif":
        "e24d255874bc8cd13e2358e17e9b7f5cbbe55d73e72081d1e12086345e0d0d2d",
    "eulytite_Bi4SiO4_3_cod9012894.cif":
        "fec4177b207bb2cbaa20f349bf045956464608e9b6c8880917be7af2b966bd83",
    "quartz_SiO2_cod9013321.cif":
        "d7fbac5b43970cf0a0cda4c9bfc6564221af1b5d1efbb397b32e151e2bc267b8",
    "senarmontite_Sb2O3_cod9009747.cif":
        "964f8b38654f979a08f4bbfca8636046c7d6b6d8db4f9ddaedcdcf8bb72f00a8",
    "valentinite_Sb2O3_cod9007587.cif":
        "86d693982b8be0274d14aa0ccfd75860e52dd232125533812fe9459b50899649",
}


def test_the_six_examples_are_unchanged_copies_with_their_attribution():
    """The copies these tests read hold the content of the bundled examples
    (COD entries redistributed unchanged under CC0), and the attribution file
    travels with them."""
    assert set(COPY_SHA256) == set(EXAMPLES) | {"ATTRIBUTION.md"}
    for name, digest in COPY_SHA256.items():
        content = (DATA / name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(content).hexdigest() == digest, name


def test_the_oxidation_states_are_the_resolved_ones():
    """The supercell carries the states the crystal path resolved (here
    Bi(V) in NaBiO3 and Bi(III) elsewhere), not the elements' usual ones;
    otherwise the equality would compare different parameters."""
    case = _case(DATA / "bismuth_phosphate_BiPO4_cod9008088.cif")
    states = {(str(e), int(o)) for e, o in zip(case.frame.elements, case.ox)}
    assert states == {("Bi", 3), ("P", 5), ("O", -2)}
    sources = {s.ox_source for s in case.structure.sites if s.ox > 0}
    assert sources == {"bond valence"}
    path = Path(sample_cif("1526788", "1526788_Na(BiO3).cif"))
    if not path.exists():
        return
    case = _case(path)
    assert ("Bi", 5) in {(str(e), int(o)) for e, o in
                         zip(case.frame.elements, case.ox)}
