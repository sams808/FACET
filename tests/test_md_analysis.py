"""The MD analysis driver (md_analysis.analyse) against the modules it runs.

What these tests pin:

* every analysis gives, through the driver, the numbers its own module gives
  when called directly on the same frames: integer descriptors (CN, Q^n,
  rings, Voronoi indices, bonds) exactly, the others to 1e-12. The driver
  hands each module a restricted copy of one wider pair search, so this is
  the check that a copy equals the module's own search;
* one ``bulk`` pair search per frame per pass, counted on the search
  function itself (``bulk._search``, which every caller of
  ``bulk.iter_pairs`` reaches whatever name it imported it under), and one
  pass when every cutoff is given;
* every missing input is named at once, before any frame is read, and no
  physical input (formers, timestep, temperature, charges, correlation) is
  filled in;
* a cancelled run returns what was finished, with the frames left out and
  why; a run cancelled before its first frame raises;
* a frame an analysis refuses is left out of that analysis only, with the
  reason, and the others keep it;
* progress never goes back and its total never grows;
* the driver imports no Qt.

The model is a 3 x 3 x 3 quartz supercell (243 atoms, the bundled CIF
copied under tests/data/crystals) with each atom moved by a seeded
Gaussian of 0.03 Å per axis in each of three frames, so the frames differ
and their pair searches do too. Its smallest perpendicular width is
12.77 Å, so the scattering grid (to half of it) widens the shared search
beyond the 6.0 Å the bond valences need.
"""
from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from facet.core import (bulk, bv, glass, md_analysis as ma, md_dynamics,
                        md_model, md_network, md_order, md_scattering,
                        md_spectroscopy, md_stats, readers)

ROOT = Path(__file__).resolve().parent.parent
QUARTZ = Path(__file__).resolve().parent / "data" / "crystals" / \
    "quartz_SiO2_cod9013321.cif"
FRAMES = [0, 1, 2]
# a stand-in correlation for the NMR tests: its numbers are test values,
# not a published correlation (none ships with FACET)
CORRELATION = md_spectroscopy.Correlation(
    "29Si", -10.0, (md_spectroscopy.Term("T-O-T angle", -0.5,
                                         angle_function="deg"),),
    reference="test values")


def _frames(n=3, seed=5, scale=1.0):
    structure = readers.read(QUARTZ)
    base, _ = md_model.supercell_frame(structure, (3, 3, 3))
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        cart = base.cart_ang * scale + rng.normal(0.0, 0.03, base.cart_ang.shape)
        out.append(md_model.frame_from_arrays(
            base.elements, cart, box_ang=base.box_ang * scale,
            timestep=1000 * k, unwrapped_cart_ang=cart,
            vel_ang_per_ps=rng.normal(0.0, 5.0, cart.shape)))
    return out


class _Counting(md_model.MemoryTrajectory):
    """A MemoryTrajectory that counts the frames it loads, so a test can
    pin that a request is refused before any frame is analysed."""

    def __init__(self, frames, **kw):
        super().__init__(frames, **kw)
        self.loads = 0

    def _load(self, k):
        self.loads += 1
        return super()._load(k)


@pytest.fixture(scope="module")
def trajectory():
    return md_model.MemoryTrajectory(_frames())


@pytest.fixture(scope="module")
def ox(trajectory):
    return md_model.model_oxidation(trajectory.frame(0).species)


def _request(**kw):
    base = dict(
        analyses=("glass", "scattering", "rings", "coordination-sequences",
                  "polyhedral-sharing", "components", "warren-cowley",
                  "bond-order", "tetrahedral-order", "polyhedron-shape",
                  "voronoi", "nmr", "exafs", "bond-lifetimes", "msd", "vacf",
                  "kinetic-temperature"),
        formers=frozenset({"Si"}), timestep_fs=1.0,
        scattering=ma.ScatteringOptions(r_window="Lorch",
                                        radiations=("neutron", "X-ray")),
        network=ma.NetworkOptions(ring_criterion="primitive",
                                  ring_max_size=12, n_shells=5),
        nmr=ma.NmrOptions(correlation=CORRELATION,
                          delta_ppm=(-130.0, -50.0, 0.5), fwhm_ppm=2.0,
                          lineshape="gaussian"),
        exafs=ma.ExafsOptions(absorber="Si"),
        dynamics=ma.DynamicsOptions(fit_t_min_ps=0.5, fit_t_max_ps=2.0))
    base.update(kw)
    return ma.AnalysisRequest(**base)


@pytest.fixture(scope="module")
def driven(trajectory):
    return ma.analyse(trajectory, _request())


@pytest.fixture(scope="module")
def tables(trajectory, ox):
    """Each frame's valence table and bonds from the module's own search."""
    out = {}
    for k in FRAMES:
        frame = trajectory.frame(k)
        table, results = bulk.analyse_frame(frame, ox.per_atom(frame.elements))
        out[k] = (frame, table, results, bulk.bonds_at(table,
                                                       bv.V_BOND_DEFAULT))
    return out


def _same(a, b, *, exact=False, where=""):
    """Two md_stats containers hold the same frames and values."""
    assert type(a) is type(b), where
    np.testing.assert_array_equal(a.frames, b.frames, err_msg=where)
    if isinstance(a, md_stats.Series):
        np.testing.assert_array_equal(a.axis, b.axis, err_msg=where)
    if isinstance(a, md_stats.Distribution):
        assert a.keys == b.keys, where
    if exact:
        np.testing.assert_array_equal(a.per_frame, b.per_frame, err_msg=where)
    else:
        np.testing.assert_allclose(a.per_frame, b.per_frame, rtol=1e-12,
                                   atol=1e-12, err_msg=where)


# ---------------------------------------------------------------------------
# the driver gives each module's own numbers
# ---------------------------------------------------------------------------

def test_glass_through_the_driver_equals_glass_called_directly(
        driven, trajectory, ox):
    """Every glass container, the minima and the cutoffs: the counts
    exactly, the g(r) (deposited block by block, in another block order)
    to 1e-12."""
    g = ma.GlassOptions()
    direct = glass.analyse_trajectory(
        trajectory, ox, formers={"Si"}, rdf_r_max_ang=6.0,
        minimum=g.minimum(), bins=g.bins(), frames=FRAMES)
    mine = driven.outputs["glass"].raw
    assert mine.provenance.frames_used == direct.provenance.frames_used
    for section in ("bv", "distance", "comparison"):
        ours, theirs = getattr(mine, section), getattr(direct, section)
        assert set(ours) == set(theirs), section
        for name in theirs:
            exact = isinstance(theirs[name], (md_stats.Distribution,
                                              md_stats.Histogram))
            _same(ours[name], theirs[name], exact=exact, where=name)
    for pair, series in direct.rdf.items():
        _same(mine.rdf[pair], series, where=str(pair))
        _same(mine.running_cn[pair], direct.running_cn[pair], exact=True,
              where=str(pair))
    assert {p: m.r_ang for p, m in mine.minima.items()} == \
        {p: m.r_ang for p, m in direct.minima.items()}
    assert mine.cutoffs_ang == direct.cutoffs_ang
    assert any("pair_search" in n for n in mine.notes)


def test_scattering_through_the_driver_equals_its_own_search(driven,
                                                              trajectory):
    """md_scattering with partials from the shared search (d_min 0.4 Å)
    against its own search (d_min 0): no pair of this model is that close,
    so every curve agrees to 1e-12."""
    out = driven.outputs["scattering"]
    method = out.provenance.method_parameters
    o = ma.ScatteringOptions(r_window="Lorch", radiations=("neutron", "X-ray"))
    q = np.arange(o.q_step_inv_ang, o.q_grid_max_inv_ang
                  + 0.5 * o.q_step_inv_ang, o.q_step_inv_ang)
    direct = md_scattering.analyse_trajectory(
        trajectory, r_max_ang=method["r_max_ang"], dr_ang=o.dr_ang,
        q_inv_ang=q, r_window="Lorch", radiations=("neutron", "X-ray"),
        frames=FRAMES)
    mine = out.raw
    for pair, series in direct.partial_g.items():
        _same(mine.partial_g[pair], series, where=str(pair))
        _same(mine.partial_s[pair], direct.partial_s[pair], where=str(pair))
    for radiation, total in direct.totals.items():
        for attr in ("s", "f_reduced", "pdf_g", "keen_d"):
            _same(getattr(mine.totals[radiation], attr), getattr(total, attr),
                  where=f"{radiation} {attr}")
    # the shared search reached further than the scattering grid needs
    reach = method["r_max_ang"] + o.dr_ang
    assert driven.provenance.method_parameters[
        "pass 1 search radius (Å)"] >= reach


def test_the_network_analyses_equal_the_modules_on_each_frame(driven, tables):
    """Rings, coordination sequences, components and polyhedral sharing
    from the module functions on the module's own search: identical."""
    rings, cseq, comps, share = [], [], [], []
    for k in FRAMES:
        frame, table, _, _ = tables[k]
        graph = md_network.bridged_graph(frame, table, bv.V_BOND_DEFAULT,
                                         formers={"Si"}, anions={"O"})
        rings.append(md_network.ring_statistics(graph, "primitive", 12,
                                                t_elements={"Si"}))
        cseq.append(md_network.coordination_sequences(graph, 5,
                                                      centres={"Si"}))
        comps.append(md_network.components(graph))
        sharing = md_network.graph_from_bonds(frame, table, bv.V_BOND_DEFAULT,
                                              elements={"Si", "O"})
        share.append(md_network.polyhedral_connectivity(
            sharing, centres={"Si"}, ligands={"O"}))
    for name, direct in (
            ("rings", md_network.average_rings(rings, frames=FRAMES)),
            ("coordination-sequences",
             md_network.average_coordination_sequences(cseq, frames=FRAMES)),
            ("components", md_network.average_components(comps,
                                                         frames=FRAMES)),
            ("polyhedral-sharing",
             md_network.average_polyhedral_connectivity(share,
                                                        frames=FRAMES))):
        mine = driven.outputs[name].tables
        assert len(mine) == len(direct), name
        for value in direct.values():
            _same(mine[value.name], value, exact=True, where=value.name)


def test_the_order_parameters_equal_the_modules_on_the_same_bonds(driven,
                                                                  tables):
    """Steinhardt q_l, q_tet and the polyhedron shape on the bonds of the
    module's own search: the same values bit for bit, so the same
    histograms."""
    rows = {"q_6": [], "q_tet": [], "baur": []}
    for k in FRAMES:
        frame, _, _, bonds = tables[k]
        nb = md_order.neighbours_from_bonds(frame, bonds)
        rows["q_6"].append((nb.elements, md_order.steinhardt(nb).values("q",
                                                                         6)))
        rows["q_tet"].append((nb.elements,
                              md_order.tetrahedral_order(nb).q_tet))
        rows["baur"].append((nb.elements, md_order.polyhedron_shape(nb).baur))
    names = {"q_6": ("bond-order", "q_6"), "q_tet": ("tetrahedral-order",
                                                      "q_tet"),
             "baur": ("polyhedron-shape", "Baur distortion index")}
    for key, (analysis, label) in names.items():
        mine = driven.outputs[analysis].tables[f"{label} (Si)"]
        edges = mine.edges
        direct = md_order.element_histograms(rows[key], edges, name=label,
                                             unit=mine.unit, frames=FRAMES)
        np.testing.assert_array_equal(mine.counts, direct["Si"].counts)
        np.testing.assert_array_equal(mine.n_nonfinite,
                                      direct["Si"].n_nonfinite)


def test_voronoi_through_the_driver_equals_the_module(driven, trajectory):
    cells = [md_order.voronoi_cells(trajectory.frame(k)) for k in FRAMES]
    direct = md_order.voronoi_index_distribution(cells, frames=FRAMES)
    _same(driven.outputs["voronoi"].tables[direct.name], direct, exact=True)


def test_exafs_through_the_driver_equals_md_exafs(driven, trajectory):
    """The absorber g(r), the automatic first-shell limits and the shell
    cumulants, against md_exafs on its own two searches per frame."""
    direct = md_spectroscopy.md_exafs(
        trajectory, "Si", r_max_ang=6.0, dr_ang=0.01,
        minimum=ma.GlassOptions().minimum(), frames=FRAMES)
    mine = driven.outputs["exafs"].raw
    assert direct.searches_per_frame == mine.searches_per_frame == 2
    for element, series in direct.g.items():
        _same(mine.g[element], series, where=element)
        _same(mine.n_cum[element], direct.n_cum[element], exact=True)
    assert {e: x.r_ang for e, x in mine.limits.items()} == \
        {e: x.r_ang for e, x in direct.limits.items()}
    assert len(mine.shells) == len(direct.shells) > 0
    for ours, theirs in zip(mine.shells, direct.shells, strict=True):
        assert (ours.neighbour, ours.r_lo_ang, ours.r_hi_ang) == \
            (theirs.neighbour, theirs.r_lo_ang, theirs.r_hi_ang)
        for label, scalar in theirs.per_frame.items():
            _same(ours.per_frame[label], scalar, where=label)
        assert ours.n_pairs_total == theirs.n_pairs_total


def test_nmr_through_the_driver_equals_md_nmr(driven, trajectory, ox):
    grid = np.arange(-130.0, -50.0 + 0.25, 0.5)
    direct = md_spectroscopy.md_nmr(
        trajectory, ox, CORRELATION, formers={"Si"}, delta_ppm=grid,
        fwhm_ppm=2.0, lineshape="gaussian", frames=FRAMES)
    mine = driven.outputs["nmr"].raw
    _same(mine.spectrum, direct.spectrum)
    _same(mine.counts, direct.counts, exact=True)
    _same(mine.mean_shift, direct.mean_shift)


def test_bond_lifetimes_equal_collect_bonds(driven, trajectory, ox):
    timeline = md_dynamics.collect_bonds(trajectory, ox, frames=FRAMES,
                                         timestep_fs=1.0)
    direct = md_dynamics.bond_lifetimes(timeline)
    mine = driven.outputs["bond-lifetimes"].raw[1]
    for pair, series in direct.continuous.items():
        _same(mine.continuous[pair], series, exact=True)
        _same(mine.intermittent[pair], direct.intermittent[pair], exact=True)


def test_dynamics_through_the_driver_equals_the_module(driven, trajectory):
    tracks = md_dynamics.collect_tracks(trajectory, frames=FRAMES,
                                        timestep_fs=1.0)
    msd = md_dynamics.msd(tracks)
    for element, series in msd.msd.items():
        _same(driven.outputs["msd"].raw[0].msd[element], series, exact=True)
    vacf = md_dynamics.vacf(tracks)
    for element, series in vacf.c.items():
        _same(driven.outputs["vacf"].raw[0].c[element], series, exact=True)


# ---------------------------------------------------------------------------
# one pair search per frame
# ---------------------------------------------------------------------------

def _count_searches(monkeypatch):
    """Record the frame of every call of bulk._search, the search behind
    every bulk.iter_pairs, however its caller imported it."""
    calls = []
    real = bulk._search

    def counting(frame, *args, **kw):
        calls.append(bulk._frame_digest(frame))
        return real(frame, *args, **kw)
    monkeypatch.setattr(bulk, "_search", counting)
    return calls


def _per_frame(calls, trajectory):
    digests = {bulk._frame_digest(trajectory.frame(k)): k for k in FRAMES}
    counts = {k: 0 for k in FRAMES}
    for digest in calls:
        counts[digests[digest]] += 1
    return counts


def test_one_search_per_frame_per_pass(monkeypatch, trajectory):
    """Every per-frame analysis at once, with automatic cutoffs: pass 1 and
    pass 2 each search every frame once (glass, EXAFS, Warren-Cowley and the
    order parameters on distance neighbours all read pass 2), and the count
    the result reports is the count made."""
    calls = _count_searches(monkeypatch)
    request = _request(order=ma.OrderOptions(neighbours="distance"),
                       analyses=("glass", "scattering", "rings",
                                 "warren-cowley", "bond-order", "nmr",
                                 "exafs", "polyhedral-sharing"))
    result = ma.analyse(trajectory, request)
    assert _per_frame(calls, trajectory) == {k: 2 for k in FRAMES}
    assert result.searches == {k: (1, 1) for k in FRAMES}
    assert not result.failed


def test_one_search_per_frame_when_every_cutoff_is_given(monkeypatch,
                                                         trajectory):
    """The same analyses with the cutoffs and shell limits given: one
    pass, one search per frame."""
    calls = _count_searches(monkeypatch)
    request = _request(
        analyses=("glass", "scattering", "rings", "warren-cowley",
                  "bond-order", "exafs"),
        glass=ma.GlassOptions(cutoffs_ang={("Si", "O"): 2.2}),
        network=ma.NetworkOptions(ring_criterion="king", ring_max_size=12,
                                  wc_cutoffs_ang={("Si", "O"): 2.2,
                                                  ("O", "O"): 2.9,
                                                  ("Si", "Si"): 3.4}),
        order=ma.OrderOptions(neighbours="distance",
                              distance_cutoffs_ang={("Si", "O"): 2.2}),
        exafs=ma.ExafsOptions(absorber="Si",
                              first_shell_limits_ang={"O": 2.2, "Si": 3.4}))
    result = ma.analyse(trajectory, request)
    assert _per_frame(calls, trajectory) == {k: 1 for k in FRAMES}
    assert result.searches == {k: (1, 0) for k in FRAMES}
    assert not result.failed
    assert result.outputs["exafs"].raw.searches_per_frame == 1


def test_analyses_that_read_no_pairs_search_nothing(monkeypatch, trajectory):
    calls = _count_searches(monkeypatch)
    result = ma.analyse(trajectory, ma.AnalysisRequest(
        analyses=("voronoi", "msd"), timestep_fs=1.0))
    assert calls == []
    assert result.searches == {}
    assert result.outputs["voronoi"].ok and result.outputs["msd"].ok


def test_a_copy_of_a_wider_search_is_the_narrower_search(trajectory):
    """_restrict on a 7 Å search gives, block for block, the pairs of a
    3 Å search: the same (i, j, image) set with bit-identical distances and
    vectors."""
    frame = trajectory.frame(1)
    wide = [ma._restrict(b, 3.0, True) for b in bulk.iter_pairs(frame, 7.0)]
    narrow = list(bulk.iter_pairs(frame, 3.0))

    def keyed(blocks):
        out = {}
        for b in blocks:
            for n in range(len(b)):
                out[(int(b.i[n]), int(b.j[n]),
                     tuple(int(x) for x in b.image[n]))] = (
                    float(b.d_ang[n]), tuple(b.vec_ang[n]))
        return out
    assert keyed(wide) == keyed(narrow)
    table_wide = bulk.valence_table(frame, wide, md_model.model_oxidation(
        frame.species).per_atom(frame.elements), r_search_ang=3.0)
    assert table_wide.n_listed.sum() > 0


# ---------------------------------------------------------------------------
# inputs: none invented, all named
# ---------------------------------------------------------------------------

def test_every_missing_input_is_named_at_once():
    request = ma.AnalysisRequest(analyses=(
        "glass", "rings", "conductivity", "free-volume", "nmr", "exafs",
        "scattering", "self-correlations"))
    names = {m.name for m in ma.missing_inputs(request)}
    assert {"formers", "network.ring_criterion", "network.ring_max_size",
            "temperature_k", "charges_e", "dynamics.fit_t_min_ps/fit_t_max_ps",
            "voids.radii", "voids.probe_radius_ang", "voids.grid_spacing_ang",
            "nmr.correlation", "nmr.delta_ppm", "nmr.fwhm_ppm",
            "nmr.lineshape", "exafs.absorber", "scattering.r_window",
            "scattering.radiations", "dynamics.lag_t_ps"} <= names
    with pytest.raises(ma.RequestError) as error:
        ma.analyse(md_model.MemoryTrajectory(_frames(1)), request)
    for name in ("formers", "temperature_k", "nmr.correlation"):
        assert name in str(error.value)


def test_no_physical_input_has_a_default():
    request = ma.AnalysisRequest(analyses=())
    assert request.formers is None
    assert request.timestep_fs is None and request.frame_interval_ps is None
    assert request.temperature_k is None and request.charges_e is None
    assert request.nmr.correlation is None
    assert request.voids.radii is None and request.voids.probe_radius_ang is None
    assert request.exafs.absorber is None
    assert request.scattering.r_window is None
    assert request.network.ring_criterion is None
    assert request.dynamics.fit_t_min_ps is None
    assert ma.missing_inputs(dataclasses.replace(request,
                                                 analyses=("glass",)))


def test_a_correlation_without_its_reference_is_refused():
    correlation = dataclasses.replace(CORRELATION, reference=None)
    request = _request(nmr=ma.NmrOptions(
        correlation=correlation, delta_ppm=(-130.0, -50.0, 0.5),
        fwhm_ppm=2.0, lineshape="gaussian"), analyses=("nmr",))
    missing = ma.missing_inputs(request)
    assert any("reference" in m.why for m in missing)


def test_formers_stated_as_none_run_glass_without_former_descriptors(
        trajectory):
    result = ma.analyse(trajectory, ma.AnalysisRequest(
        analyses=("glass",), formers=ma.NO_FORMERS))
    tables = result.outputs["glass"].tables
    assert "CN Si (BV)" in tables
    assert not any(name.startswith(("Qn", "O speciation")) for name in tables)
    assert any("formers not given" in n
               for n in result.outputs["glass"].raw.notes)


def test_the_model_is_checked_before_any_frame_is_analysed(trajectory):
    with pytest.raises(ma.RequestError, match="anions of this model"):
        ma.analyse(trajectory, ma.AnalysisRequest(analyses=("glass",),
                                                  formers={"O"}))
    with pytest.raises(ma.RequestError, match="holds no Al"):
        ma.analyse(trajectory, ma.AnalysisRequest(
            analyses=("exafs",), exafs=ma.ExafsOptions(absorber="Al")))
    with pytest.raises(ma.RequestError, match="no charge for O"):
        ma.analyse(trajectory, ma.AnalysisRequest(
            analyses=("conductivity",), temperature_k=300.0,
            charges_e={"Si": 2.4}, timestep_fs=1.0,
            dynamics=ma.DynamicsOptions(fit_t_min_ps=0.5, fit_t_max_ps=2.0)))


def test_available_analyses_lists_what_each_left_out_lacks():
    request = ma.AnalysisRequest(analyses=(), formers={"Si"})
    runnable, left = ma.available_analyses(request)
    assert "glass" in runnable and "voronoi" in runnable
    assert "warren-cowley" in runnable          # reads the glass minima
    assert "network.ring_criterion" in left["rings"]
    assert "voids.radii" in left["empty-spheres"]
    assert "nmr.correlation" in left["nmr"]
    alone = ma.available_analyses(ma.AnalysisRequest(analyses=()))
    assert "glass" not in alone[0] and "formers" in alone[1]["glass"]
    assert "warren-cowley" not in alone[0]


def test_v_bond_below_v_list_is_refused_naming_both():
    """A contact below v_list is never tabulated, so a CN at a v_bond below
    it is the CN at v_list under another label (the Na2O-3SiO2 glass gave
    CN Na 5.59 'at 0.01 v.u.' where the count at 0.01 v.u. is 6.65)."""
    request = ma.AnalysisRequest(analyses=("glass",), formers={"Si"},
                                 v_bond_vu=0.01)
    problems = [m for m in ma.missing_inputs(request) if m.name == "v_bond_vu"]
    assert len(problems) == 1
    assert "0.01" in problems[0].why and "0.02" in problems[0].why
    counting = _Counting(_frames(1))
    with pytest.raises(ma.RequestError, match="v_list"):
        ma.analyse(counting, request)
    assert counting.loads == 0
    # at or below v_bond, v_list is accepted
    assert not ma.invalid_inputs(dataclasses.replace(request,
                                                     v_list_vu=0.01))


def test_every_impossible_value_is_named_before_any_frame():
    """Zero bin widths and steps (a ZeroDivisionError, some only after
    every frame), names outside each module's list, reversed windows, both
    time axes, a missing measured file: all named in one RequestError, and
    no frame is read."""
    request = _request(
        analyses=("scattering", "bond-order", "voronoi", "empty-spheres",
                  "free-volume", "exafs", "nmr", "rings",
                  "coordination-sequences", "distinct-van-hove", "msd"),
        frame_interval_ps=0.5,
        scattering=ma.ScatteringOptions(
            r_window="Lorch", radiations=("neutron", "gamma"), dr_ang=0.0,
            q_step_inv_ang=0.0, fsdp_window_inv_ang=(1.0, 2.0),
            fsdp_baseline="none",
            measured=(ma.MeasuredCurve("no_such_file.dat", "neutron",
                                       "S(Q)"),)),
        order=ma.OrderOptions(q_bin=0.0, cell_volume_bin_ang3=0.0,
                              degrees=(0,)),
        voids=ma.VoidOptions(radii="vdw", probe_radius_ang=-1.0,
                             grid_spacing_ang=0.0, sphere_bin_ang=0.0),
        exafs=ma.ExafsOptions(absorber="Si", dr_ang=0.0, weighting="bogus"),
        nmr=ma.NmrOptions(correlation=CORRELATION,
                          delta_ppm=(-130.0, -50.0, 0.0), fwhm_ppm=-2.0,
                          lineshape="gaussian"),
        network=ma.NetworkOptions(ring_criterion="bogus", ring_max_size=0,
                                  n_shells=-1),
        glass=ma.GlassOptions(minimum_rule="bogus", angle_bin_deg=-1.0,
                              cutoffs_ang={("Si", "O"): -1.0}),
        dynamics=ma.DynamicsOptions(
            fit_t_min_ps=2.0, fit_t_max_ps=1.0, lag_t_ps=(1.0,),
            van_hove_edges_r_ang=(0.0, 8.0, 0.0),
            distinct_pairs=(("Si", "Si"),), distinct_n_origins=1))
    named = {m.name for m in ma.missing_inputs(request)}
    for name in ("scattering.radiations", "scattering.dr_ang",
                 "scattering.q_step_inv_ang", "scattering.fsdp_baseline",
                 "scattering.measured[0].path", "order.q_bin",
                 "order.cell_volume_bin_ang3", "order.degrees",
                 "voids.probe_radius_ang", "voids.grid_spacing_ang",
                 "voids.sphere_bin_ang", "exafs.dr_ang", "exafs.weighting",
                 "nmr.delta_ppm step", "nmr.fwhm_ppm",
                 "network.ring_criterion", "network.ring_max_size",
                 "network.n_shells", "glass.minimum_*", "glass bin widths",
                 "glass.cutoffs_ang Si-O", "timestep_fs",
                 "dynamics.fit_t_min_ps/fit_t_max_ps",
                 "dynamics.van_hove_edges_r_ang step"):
        assert name in named, name
    counting = _Counting(_frames(1))
    with pytest.raises(ma.RequestError) as error:
        ma.analyse(counting, request)
    assert counting.loads == 0
    assert "scattering.dr_ang" in str(error.value) and \
        "nmr.delta_ppm step" in str(error.value)
    # the defaults themselves are all valid
    assert ma.invalid_inputs(ma.AnalysisRequest(analyses=())) == ()


def test_a_frame_selection_the_model_cannot_give_is_a_request_error():
    """Frames beyond the trajectory, an empty slice, a step of 0: refused as
    inputs (RequestError, exit 2 on the command line), not as a failed
    analysis."""
    for frames in (slice(50, 60), slice(3, 3), (25,)):
        with pytest.raises(ma.RequestError, match="frame"):
            ma.analyse(md_model.MemoryTrajectory(_frames(2)),
                       ma.AnalysisRequest(analyses=("voronoi",),
                                          frames=frames))
    assert any(m.name == "frames" for m in ma.invalid_inputs(
        ma.AnalysisRequest(analyses=("voronoi",), frames=slice(0, 4, 0))))


def test_unreadable_frames_are_a_read_failure_not_a_request_error():
    class Broken(md_model.MemoryTrajectory):
        def _load(self, k):
            raise md_model.FrameError(f"frame {k} is truncated")

    with pytest.raises(ma.FramesUnreadable, match="truncated"):
        ma.analyse(Broken(_frames(2)), ma.AnalysisRequest(
            analyses=("voronoi",)))


def test_formers_none_of_which_are_in_the_model_are_refused():
    """--formers Ge on a silicate used to give 100 % free O with exit 0;
    formers of which only some are present are a model note."""
    with pytest.raises(ma.RequestError, match="none of the formers"):
        ma.analyse(md_model.MemoryTrajectory(_frames(1)),
                   ma.AnalysisRequest(analyses=("glass",), formers={"Ge"}))
    result = ma.analyse(md_model.MemoryTrajectory(_frames(1)),
                        ma.AnalysisRequest(analyses=("glass",),
                                           formers={"Si", "B"}))
    assert any("B named but not in the model" in n
               for n in result.model_notes)
    assert any("B named but not in the model" in n for n in result.notes)


def test_an_option_naming_an_absent_element_is_refused():
    with pytest.raises(ma.RequestError) as error:
        ma.analyse(md_model.MemoryTrajectory(_frames(2)), _request(
            analyses=("distinct-van-hove", "coordination-sequences"),
            network=ma.NetworkOptions(n_shells=3, cseq_centres=("Al",)),
            dynamics=ma.DynamicsOptions(lag_t_ps=(1.0,),
                                        van_hove_edges_r_ang=(0.0, 6.0, 0.1),
                                        distinct_pairs=(("Na", "Na"),),
                                        distinct_n_origins=1)))
    text = str(error.value)
    assert "dynamics.distinct_pairs: the model holds no Na" in text
    assert "network.cseq_centres: the model holds no Al" in text


def test_missing_bv_pairs_and_the_net_charge_reach_every_bond_analysis(
        tmp_path):
    """Si2+ with the estimator off: no Si-O parameter, and a charge of
    -162 e on the quartz supercell. Both used to reach the export only
    through glass; now a bond-order run states them too."""
    from facet.core import md_export

    params = bv.ParameterSet(allow_estimated=False)
    result = ma.analyse(md_model.MemoryTrajectory(_frames(2)),
                        ma.AnalysisRequest(analyses=("bond-order",),
                                           ox_overrides={"Si": 2},
                                           params=params))
    provenance = result.outputs["bond-order"].provenance
    assert provenance.missing_pairs and \
        sum(provenance.missing_pairs.values()) > 0
    assert provenance.method_parameters[
        "bond-valence estimated parameters (allow_estimated)"] is False
    assert result.provenance.missing_pairs == provenance.missing_pairs
    assert any("not neutral" in n and "-162" in n
               for n in result.model_notes)
    md_export.write_csv_files(result, tmp_path)
    header = (tmp_path / "bond-order__q_6_(Si).csv").read_text(
        encoding="utf-8")
    assert "pair without a bond-valence parameter: Si2+-O" in header
    assert "model note: the model is not neutral" in header


def test_a_scattering_r_max_beyond_half_the_box_is_refused_up_front(
        monkeypatch):
    """r_max_ang = 100 Å set the shared search radius before anything
    refused it (11 GB on a 3 000-atom glass); now it is refused before any
    search."""
    calls = _count_searches(monkeypatch)
    with pytest.raises(ma.RequestError, match="half the smallest"):
        ma.analyse(md_model.MemoryTrajectory(_frames(2)), _request(
            analyses=("scattering",),
            scattering=ma.ScatteringOptions(r_window="none",
                                            radiations=("neutron",),
                                            r_max_ang=100.0)))
    assert calls == []


def test_the_scattering_average_does_not_depend_on_the_frame_order():
    """Boxes that change between frames (NPT): the default grid is the one
    the smallest box holds, every frame is used, and the averages are the
    same in either order. Taking the first frame's grid refused the smaller
    frames in one order only."""
    base = _frames(4)
    frames = [md_model.frame_from_arrays(
        f.elements, frac=f.frac, box_ang=f.box_ang * s, timestep=k)
        for k, (f, s) in enumerate(zip(base, (1.0, 0.98, 1.01, 0.99),
                                       strict=True))]
    request = _request(analyses=("scattering",),
                       scattering=ma.ScatteringOptions(
                           r_window="Lorch", radiations=("neutron",),
                           q_grid_max_inv_ang=10.0))
    forward = ma.analyse(md_model.MemoryTrajectory(frames), request)
    backward = ma.analyse(md_model.MemoryTrajectory(frames[::-1]), request)
    smallest = ma._ScatteringConsumer.largest_r_max(frames[1],
                                                    glass.RDF_DR_ANG)
    for result in (forward, backward):
        out = result.outputs["scattering"]
        assert out.ok and out.provenance.frames_used == (0, 1, 2, 3)
        assert out.raw.partial_g[("O", "Si")].axis[-1] == pytest.approx(
            smallest, abs=1e-9)
        assert any("does not depend on the order" in n for n in out.notes)
    s_forward = forward.outputs["scattering"].raw.totals["neutron"].s
    s_backward = backward.outputs["scattering"].raw.totals["neutron"].s
    np.testing.assert_allclose(s_forward.mean, s_backward.mean, rtol=0,
                               atol=1e-12)


def test_cut_partials_equal_the_partials_of_the_shorter_grid():
    """The cut that makes the scattering average order-free is exact: a
    frame's partials cut to a shorter grid equal frame_partials run to that
    grid, value for value."""
    frame = _frames(1)[0]
    dr = glass.RDF_DR_ANG
    long_ = md_scattering.frame_partials(frame, r_max_ang=6.0, dr_ang=dr)
    short = md_scattering.frame_partials(frame, r_max_ang=4.5, dr_ang=dr)
    cut = ma._trimmed_partials(long_, short.r_ang.size)
    np.testing.assert_array_equal(cut.r_ang, short.r_ang)
    for field in ("pair_hist", "g", "n_cum"):
        for pair, values in getattr(short, field).items():
            np.testing.assert_array_equal(getattr(cut, field)[pair], values,
                                          err_msg=f"{field} {pair}")


def test_g_from_sq_is_computed_with_the_default_q_min_and_its_reason_kept(
        tmp_path):
    """termination_q_min_inv_ang defaulted to 0, below the first Q of the
    grid, so G(r) from S(Q) was never computed and the comparison blamed a
    missing Qmax. Now it starts at the grid's first Q; when the module
    refuses it for another reason, the comparison gives that reason."""
    measured = tmp_path / "gr.dat"
    r_ang = np.arange(1.0, 6.0, 0.05)
    measured.write_text("\n".join(f"{r} {np.sin(r)}" for r in r_ang),
                        encoding="utf-8")
    options = dict(r_window="Lorch", radiations=("neutron",),
                   termination_q_max_inv_ang=18.0, q_window="Lorch",
                   measured=(ma.MeasuredCurve(str(measured), "neutron",
                                              "G(r) from S(Q)"),))
    result = ma.analyse(md_model.MemoryTrajectory(_frames(1)), _request(
        analyses=("scattering", "scattering-comparison"),
        scattering=ma.ScatteringOptions(**options)))
    total = result.outputs["scattering"].raw.totals["neutron"]
    assert total.pdf_g_from_sq is not None
    method = result.outputs["scattering"].provenance.method_parameters
    assert method["termination_q_min_inv_ang from"].startswith("the first Q")
    comparison = result.outputs["scattering-comparison"]
    assert comparison.ok, comparison.error
    table = next(t for name, t in comparison.tables.items()
                 if not name.endswith("(R_chi)"))
    unit = total.pdf_g_from_sq.value_unit
    assert f"measured ({unit})" in table.rows[0]
    assert f"measured - model ({unit})" in table.rows[0]
    # a Q step too coarse for the r grid: the module's own reason
    coarse = ma.ScatteringOptions(**options, q_step_inv_ang=0.6)
    refused = ma.analyse(md_model.MemoryTrajectory(_frames(1)), _request(
        analyses=("scattering", "scattering-comparison"),
        scattering=coarse))
    error = refused.outputs["scattering-comparison"].error
    assert "pi / dQ" in error and "termination Qmax" not in error


def test_analyses_of_time_without_a_time_axis_or_velocities_are_left_out():
    """A dump without TIME and without velocities: the default selection
    leaves out the analyses of time (they used to run and fail at the end),
    naming what each lacks; given explicitly they are refused before any
    frame."""
    bare = [md_model.frame_from_arrays(f.elements, f.cart_ang,
                                       box_ang=f.box_ang, timestep=1000 * k)
            for k, f in enumerate(_frames(3))]
    trajectory = md_model.MemoryTrajectory(bare)
    request = ma.AnalysisRequest(analyses=(), formers={"Si"})
    runnable, left = ma.available_analyses(request, trajectory)
    for name in ("msd", "vacf", "kinetic-temperature", "bond-lifetimes"):
        assert name not in runnable
        assert "timestep_fs or frame_interval_ps" in left[name], name
    assert "glass" in runnable
    timed = dataclasses.replace(request, timestep_fs=1.0)
    runnable, left = ma.available_analyses(timed, trajectory)
    assert "msd" in runnable and "bond-lifetimes" in runnable
    assert left["kinetic-temperature"] == ["velocities in the file"]
    assert left["vacf"] == ["velocities in the file"]
    counting = _Counting(bare)
    with pytest.raises(ma.RequestError, match="timestep_fs or "
                                              "frame_interval_ps"):
        ma.analyse(counting, ma.AnalysisRequest(analyses=("voronoi", "msd")))
    assert counting.loads == 1


def test_a_glass_radius_beyond_the_box_is_cut_and_the_radius_searched_stated(
        monkeypatch):
    """glass cuts its g(r) grid at half the box (glass.py), so a large
    glass.rdf_r_max_ang never widens the shared search past it; the radius
    the provenance states is the one searched (it used to state the 6 Å of
    the plan while glass searched to its grid, 6.39 Å here)."""
    radii = []
    real = bulk.iter_pairs

    def recording(frame, r_ang, *args, **kw):
        radii.append(float(r_ang))
        return real(frame, r_ang, *args, **kw)
    monkeypatch.setattr(bulk, "iter_pairs", recording)
    frames = _frames(2)
    result = ma.analyse(md_model.MemoryTrajectory(frames), ma.AnalysisRequest(
        analyses=("glass",), formers={"Si"},
        glass=ma.GlassOptions(rdf_r_max_ang=100.0)))
    half = float(frames[0].perpendicular_widths_ang.min()) / 2.0
    stated = result.provenance.method_parameters["pass 1 search radius (Å)"]
    assert max(radii) <= half + glass.RDF_DR_ANG
    assert stated == max(radii[:1] + radii[1:2]) > 6.0
    assert any("grid stops" in n for n in result.outputs["glass"].notes)


def test_a_cancel_before_anything_in_a_dynamics_run_raises(trajectory):
    with pytest.raises(ma.AnalysisCancelled):
        ma.analyse(trajectory, ma.AnalysisRequest(analyses=("msd",),
                                                  timestep_fs=1.0),
                   cancelled=lambda: True)


# ---------------------------------------------------------------------------
# cancel, refusals, progress
# ---------------------------------------------------------------------------

def test_a_cancel_mid_run_gives_the_finished_frames_with_notes(trajectory):
    state = {"done": 0}

    def progress(done, total, stage):
        state["done"] = done

    # two pass-1 frames done: PROGRESS_UNITS_PASS1 units each
    result = ma.analyse(trajectory, _request(
        analyses=("glass", "rings", "voronoi", "warren-cowley")),
        progress=progress,
        cancelled=lambda: state["done"] >= 2 * ma.PROGRESS_UNITS_PASS1)
    assert result.cancelled
    assert result.provenance.frames_used == (0, 1)
    assert 2 in result.provenance.frames_skipped
    assert any("cancelled" in n for n in result.notes)
    glass_out = result.outputs["glass"]
    assert glass_out.ok and glass_out.provenance.frames_used == (0, 1)
    assert result.outputs["rings"].provenance.frames_used == (0, 1)
    assert result.outputs["voronoi"].ok
    # Warren-Cowley reads pass 2 only, which the cancel stopped before: its
    # reason says so (it used to read 'no frame was analysed', no reason)
    assert "the run was cancelled" in result.outputs["warren-cowley"].error


def test_a_cancel_before_the_first_frame_raises(trajectory):
    with pytest.raises(ma.AnalysisCancelled):
        ma.analyse(trajectory, ma.AnalysisRequest(analyses=("voronoi",)),
                   cancelled=lambda: True)


def test_a_frame_one_analysis_refuses_stays_in_the_others():
    """A box that shrinks (NPT) in the last frame below the scattering grid
    the user asked for (the largest the first frame holds): scattering
    leaves that frame out with the reason, glass keeps it."""
    frames = _frames(2) + _frames(1, seed=9, scale=0.96)
    frames = [md_model.frame_from_arrays(f.elements, f.cart_ang,
                                         box_ang=f.box_ang, timestep=k)
              for k, f in enumerate(frames)]
    trajectory = md_model.MemoryTrajectory(frames)
    r_max = ma._ScatteringConsumer.largest_r_max(frames[0], glass.RDF_DR_ANG)
    result = ma.analyse(trajectory, _request(
        analyses=("glass", "scattering"),
        scattering=ma.ScatteringOptions(r_window="Lorch",
                                        radiations=("neutron",),
                                        r_max_ang=r_max)))
    scattering = result.outputs["scattering"].provenance
    assert scattering.frames_used == (0, 1)
    assert "half the smallest perpendicular width" in \
        scattering.frames_skipped[2]
    assert result.outputs["glass"].provenance.frames_used == (0, 1, 2)


def test_progress_never_goes_back_and_its_total_never_grows(trajectory):
    """And it weighs a pass-1 frame above a pass-2 frame, naming the pass:
    counting both alike, a 20-frame run with every analysis showed 38 % at
    160 s of 181 s."""
    events = []
    ma.analyse(trajectory, _request(analyses=("glass", "rings", "exafs",
                                              "msd")),
               progress=lambda d, t, s: events.append((d, t, s)))
    done = [d for d, _, _ in events]
    totals = [t for _, t, _ in events]
    assert done == sorted(done)
    assert totals == sorted(totals, reverse=True)
    assert events[-1][0] == events[-1][1]
    end_of_pass1 = [d / t for d, t, s in events
                    if s.startswith("pass 1: frame 3 of 3")]
    assert end_of_pass1 and end_of_pass1[0] > 0.8
    assert any(s.startswith("pass 2: frame 1 of 3") for _, _, s in events)
    assert ma.PROGRESS_UNITS_PASS1 > ma.PROGRESS_UNITS_PASS2


def test_an_analysis_with_no_result_reports_why_and_the_others_go_on(
        trajectory):
    """An MSD fit window beyond the lags the three frames hold, which only
    the MSD itself can find: msd reports why it produced nothing, the
    per-frame analyses are kept."""
    result = ma.analyse(trajectory, ma.AnalysisRequest(
        analyses=("voronoi", "msd"), timestep_fs=1.0,
        dynamics=ma.DynamicsOptions(fit_t_min_ps=1.0, fit_t_max_ps=50.0)))
    assert result.outputs["voronoi"].ok
    assert not result.outputs["msd"].ok and result.outputs["msd"].error
    assert "msd" in result.failed


def test_one_frame_for_an_analysis_of_time_is_refused_before_any_frame():
    """One frame holds no time axis: the request is refused with the
    reason, before the per-frame analyses run (it used to end the MSD only
    after them)."""
    one = _Counting(_frames(1))
    with pytest.raises(ma.RequestError, match="at least two"):
        ma.analyse(one, ma.AnalysisRequest(analyses=("voronoi", "msd"),
                                           timestep_fs=1.0))
    assert one.loads == 1                     # the first frame, for checks


def test_the_run_provenance_carries_the_header_fields(driven):
    lines = "\n".join(driven.provenance.as_lines())
    for text in ("source file", "frames used", "type map source",
                 "oxidation states", "bond-valence parameters",
                 "bond threshold v_bond", "FACET", "distance cutoff",
                 "glass.minimum_rule = valley",
                 "bulk pair searches per frame, pass 1 = 1",
                 "g(r) first minimum O-Si"):
        assert text in lines, text


def test_an_extreme_value_does_not_stretch_a_histogram_and_is_counted():
    """A polyhedron of nearly zero volume gives a quadratic elongation near
    830 among values near 1 (measured on an aluminosilicate glass): the
    edges stay within HISTOGRAM_BINS_MAX bins, a note says so, and the
    values beyond them are counted in n_above / n_below."""
    rng = np.random.default_rng(2)
    rows = [1.0 + rng.random(500) * 0.05 for _ in range(3)]
    rows[1] = np.append(rows[1], [830.0, -40.0])
    edges, notes = ma._edges_over(rows, 0.001)
    assert len(edges) - 1 <= ma.HISTOGRAM_BINS_MAX + 1
    assert notes and "830.0" in notes[0] and "n_above" in notes[0]
    histogram = md_stats.Histogram.from_samples(rows, edges, name="x",
                                                unit="1")
    assert int(histogram.n_above.sum()) >= 1 and \
        int(histogram.n_below.sum()) >= 1
    assert int(histogram.counts.sum() + histogram.n_above.sum()
               + histogram.n_below.sum()) == sum(r.size for r in rows)
    plain, none = ma._edges_over([np.array([0.2, 0.31])], 0.1)
    assert none == () and plain[0] <= 0.2 and plain[-1] >= 0.31


VERDICT_WORDS = ("good", "bad", "poor", "excellent", "acceptable", "correct",
                 "incorrect", "wrong", "reliable", "unreliable", "trustworthy",
                 "untrustworthy", "unusable", "should", "proves", "confirms")


def test_no_verdict_word_reaches_the_user(driven):
    """Every text a run, a refusal and the command line produce: the
    provenance, the notes, the errors, the missing-input reasons, the help
    and the template."""
    import re

    from facet.md import cli

    texts = list(driven.provenance.as_lines()) + list(driven.notes)
    for out in driven.outputs.values():
        texts += list(out.notes) + [out.error or ""]
        if out.provenance is not None:
            texts += out.provenance.as_lines()
    texts += [m.describe() for m in ma.missing_inputs(
        ma.AnalysisRequest(analyses=ma.ANALYSES))]
    texts += [cli.build_parser().format_help(), cli.analyses_text(),
              cli.template_text()]
    pattern = re.compile(r"\b(" + "|".join(VERDICT_WORDS) + r")\b",
                         re.IGNORECASE)
    found = sorted({m.group(0) for text in texts
                    for m in pattern.finditer(text)})
    assert not found, found


def test_the_driver_imports_no_qt():
    code = ("import sys; import facet.core.md_analysis, "
            "facet.core.md_export, facet.md.cli; "
            "print('QT' if [m for m in sys.modules if "
            "m.startswith('PySide6')] else 'CLEAN')")
    result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout
