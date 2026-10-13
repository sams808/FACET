"""Moving the bond threshold reclassifies every number that depends on it.

The contacts of a site are found once, down to the listing threshold. The
bond threshold decides which of them are bonds, and so the coordination
number, the bond-valence sum over the bonded set, phi and the polyhedron
shape. Those used to be computed once, at the threshold in force when the
file was opened, and only the analysis panel re-summed the contacts when the
cut moved: the labels on the atoms, the site list, the tables and the report
kept the first numbers. Now the project hands every consumer the results at
the threshold in force, re-summed from the cached contacts.

The explorer's right axis draws the same contacts shell by shell: the
running sum beside the count, with the pin at the opening threshold.
"""
from __future__ import annotations

import math

import pytest

from conftest import dispose, sample_cif

from facet.core import bv, cif, coordination, project as project_mod
from facet.gl import labels as labels_mod
from facet.core import shells as shell_plot

SAMPLE = sample_cif("1526458", "1526458_Bi2O3.cif")


@pytest.fixture(scope="module")
def structure():
    return cif.read(str(SAMPLE))


@pytest.fixture(scope="module")
def base(structure):
    return coordination.analyse_structure(structure, bv.DEFAULT, v_bond=0.075,
                                          v_list=0.02, cations_only=True)


# -- the core: a reclassified result equals a fresh analysis at that cut ------

def _same(a, b) -> bool:
    def eq(x, y):
        if isinstance(x, float) and isinstance(y, float):
            return (math.isnan(x) and math.isnan(y)) or abs(x - y) < 1e-12
        return x == y
    return all(eq(getattr(a, n), getattr(b, n)) for n in (
        "cn_valence", "cn_ecoN", "d_ecoN", "cn_gap", "gap_ratio",
        "cn_occupancy", "bvs", "bvv", "phi", "f3", "bvs_uncertainty",
        "void_angle", "valence_discrepancy")) and a.shape.keys() == b.shape.keys()


@pytest.mark.parametrize("v", [0.03, 0.12, 0.2, 0.5])
def test_reclassifying_equals_analysing_at_that_threshold(structure, base, v):
    direct = coordination.analyse_structure(structure, bv.DEFAULT, v_bond=v,
                                            v_list=0.02, cations_only=True)
    for fresh, moved in zip(direct, (coordination.reclassify(r, v) for r in base)):
        assert moved.v_bond == v
        assert _same(fresh, moved), (fresh.label, v)


def test_what_the_threshold_does_not_decide_is_shared(base):
    moved = coordination.reclassify(base[0], 0.2)
    assert moved.contacts is base[0].contacts
    assert moved.plateaus is base[0].plateaus
    assert moved.bvs_listed == base[0].bvs_listed
    assert moved.phi_listed == base[0].phi_listed
    assert moved.notes == base[0].notes


def test_the_original_threshold_returns_the_original_object(base):
    assert coordination.reclassify(base[0], 0.075) is base[0]
    back = coordination.reclassify(coordination.reclassify(base[0], 0.2), 0.075)
    assert _same(back, base[0])


def test_a_threshold_above_every_contact_leaves_an_empty_coordination(base):
    moved = coordination.reclassify(base[0], 5.0)
    assert moved.cn_valence == 0
    assert math.isnan(moved.bvs) and math.isnan(moved.phi)
    assert moved.shape == {}


# -- the project hands out the results at the threshold in force --------------

def test_project_results_follow_the_threshold(structure):
    p = project_mod.Project()
    entry = p.add(structure, path=str(SAMPLE)) if hasattr(p, "add") else None
    if entry is None:
        entry = project_mod.Entry(source=structure, path=str(SAMPLE))
        p.entries.append(entry)
    first = p.results_for(entry)
    assert entry.first_v_bond == p.v_bond
    assert first[0].v_bond == p.v_bond

    p.set_bond_threshold(0.2)
    moved = p.results_for(entry)
    assert moved[0].v_bond == 0.2
    assert moved[0].cn_valence == first[0].cn_at(0.2)
    # the pin does not move with the cut
    assert entry.first_v_bond == first[0].v_bond
    # and the same list comes back while the threshold stays
    assert p.results_for(entry) is moved
    p.set_bond_threshold(first[0].v_bond)
    assert p.results_for(entry)[0] is first[0]


def test_labels_read_the_threshold_in_force(base):
    r = base[0]
    moved = coordination.reclassify(r, 0.03)
    assert moved.cn_valence != r.cn_valence, "pick a threshold that changes CN"
    text_at = labels_mod.atom_text(labels_mod.AtomLabel.COORDINATION,
                                   element=r.element, label=r.label, index=1,
                                   occupancy=1.0, site=None, result=moved,
                                   frac=None)
    assert text_at == f"CN {moved.cn_valence}"
    bvs_at = labels_mod.atom_text(labels_mod.AtomLabel.BVS, element=r.element,
                                  label=r.label, index=1, occupancy=1.0,
                                  site=None, result=moved, frac=None)
    assert bvs_at == f"{moved.bvs:.2f} v.u."


# -- the shell curve: the staircase and its integral ---------------------------

def test_shell_curve_counts_and_sums_the_contacts_in_order(base):
    r = base[0]
    by_d = shell_plot.shell_curve(r, shell_plot.AXIS_DISTANCE)
    by_v = shell_plot.shell_curve(r, shell_plot.AXIS_VALENCE)
    n = sum(1 for c in r.contacts if c.has_valence)
    assert [s.cn_after for s in by_d] == list(range(1, n + 1))
    assert [s.distance for s in by_d] == sorted(s.distance for s in by_d)
    assert [s.valence for s in by_v] == sorted((s.valence for s in by_v),
                                               reverse=True)
    # the integral ends at the sum to the listing cut, on both axes
    assert abs(by_d[-1].sum_after - r.bvs_listed) < 1e-12
    assert abs(by_v[-1].sum_after - r.bvs_listed) < 1e-12
    # and each step adds exactly that contact's occupancy-weighted valence
    for prev, s in zip([None] + by_d[:-1], by_d):
        before = 0.0 if prev is None else prev.sum_after
        assert abs(s.sum_after - before - s.valence * s.occupancy) < 1e-12


@pytest.mark.parametrize("v", [0.02, 0.075, 0.2])
def test_the_readout_at_a_threshold_is_the_analysis_at_that_threshold(base, v):
    r = coordination.reclassify(base[0], v)
    cn, total, nxt = shell_plot.at_threshold(r, v)
    assert cn == r.cn_valence
    if cn:
        assert abs(total - r.bvs) < 1e-12
    if nxt is not None:
        assert nxt.valence <= v
        assert all(c.valence <= nxt.valence for c in r.contacts
                   if c.has_valence and c.valence <= v)


def test_an_empty_site_has_no_curve():
    assert shell_plot.shell_curve(None) == []
    assert shell_plot.at_threshold(None, 0.1)[0] == 0


# -- the widgets ---------------------------------------------------------------

def test_the_explorer_marks_the_opening_threshold(qapp, base):
    from facet.ui.cutoff_explorer import CutoffExplorer
    w = CutoffExplorer()
    w.set_result(base[0])
    w.set_pin(0.075)
    marks = w._reference_marks()
    assert any(v == 0.075 and label.startswith("opened at")
               for v, label, _ in marks)
    # the pin stays on the axis wherever the data are
    w.set_pin(0.9)
    lo, hi = w._v_range()
    assert lo < 0.9 < hi
    w.set_pin(None)
    assert not any(label.startswith("opened at")
                   for _, label, _ in w._reference_marks())
    dispose(w)


def test_the_explorer_draws_the_contacts_where_the_numbers_put_them(qapp, base):
    """The tick for each contact sits at its own valence or distance, in
    order, and each staircase has one step per contact."""
    from facet.ui.cutoff_explorer import CutoffExplorer
    w = CutoffExplorer()
    w.resize(900, 230)
    w.set_result(base[0])
    w.set_pin(0.075)
    w.set_threshold(0.075)
    w.set_axis(shell_plot.AXIS_VALENCE)
    w._layout()
    by_v = shell_plot.shell_curve(base[0], shell_plot.AXIS_VALENCE)
    xv = [w._shell_x(s) for s in by_v]
    assert xv == sorted(xv, reverse=True)          # strongest at the right
    path = w._step_path(by_v, lambda s: 0.0 if s is None else float(s.cn_after))
    assert path.elementCount() == 2 + 2 * len(by_v)
    w.set_axis(shell_plot.AXIS_DISTANCE)
    w._layout()
    by_d = shell_plot.shell_curve(base[0], shell_plot.AXIS_DISTANCE)
    xd = [w._shell_x(s) for s in by_d]
    assert xd == sorted(xd)                         # nearest at the left
    assert w._plot.left() <= xd[0] and xd[-1] <= w._plot.right()
    cn, total, _ = shell_plot.at_threshold(base[0], 0.075)
    assert f"CN {cn}" in w.readout_text() and f"{total:.3f}" in w.readout_text()
    for axis in (shell_plot.AXIS_DISTANCE, shell_plot.AXIS_VALENCE):
        w.set_axis(axis)
        w._layout()
        assert abs(w._x_to_v(w._v_to_x(0.12)) - 0.12) < 1e-3, axis
    assert w.grab().toImage().width() > 0
    dispose(w)


def test_the_window_refreshes_every_consumer_on_a_drag(qapp):
    """Labels, the site list, both bottom plots and the analysis panel all
    read the threshold the cursor was dragged to."""
    from facet.ui import preview
    win = preview.PreviewWindow()
    try:
        win.load_many([str(SAMPLE)])
        qapp.processEvents()
        entry = win.project.current
        results = win.project.results_for(entry)
        entry.selected_site = results[0].site_index
        win._rebuild()
        qapp.processEvents()
        assert win.explorer.pin_v == entry.first_v_bond == 0.075

        win.explorer.set_threshold(0.03, emit=True)
        qapp.processEvents()
        r = win._current_result()
        assert r.v_bond == 0.03
        assert win.site_list.item(0).text().endswith(f"CN {r.cn_valence}")
        assert win.view.results[0] is r
        assert win.explorer.v_bond == 0.03
        assert f"CN {r.cn_valence}" in win.explorer.readout_text()

        # and back, on the distance axis
        win.explorer.set_axis(shell_plot.AXIS_DISTANCE)
        win.explorer.set_threshold(0.2, emit=True)
        qapp.processEvents()
        assert win.project.v_bond == 0.2
        assert win.explorer.v_bond == 0.2
        assert win._current_result().cn_valence == results[0].cn_at(0.2)
    finally:
        win.close()
        dispose(win)
