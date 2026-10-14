"""One atom's environment in a frame, and the four things Wyatt asked for:
a right-click menu on the frame (centre, isolate, export), an 'equal to'
rule, a double-click on a result that fills the Highlight tab, and Qn read
as integer counts beside the fractions.

The shell walk is checked on a hand-built graph; the unwrapping on a bond
that crosses the box; the XYZ on both. The controller parts run on the
Pedone NS3 model when it is on this machine and are skipped otherwise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import dispose

from facet.core import md_environment, md_model

NS3 = Path(r"C:\Users\samso\AppData\Local\Temp\claude\C--Users-samso"
           r"\342b7e11-8171-41a0-8eee-4126254c9cc5\scratchpad"
           r"\facet_md_descriptors\models\ns3_pedone\nvt300K_last.data")
NS3_TYPES = {1: "Si", 2: "O", 3: "Na"}


class _Bonds:
    def __init__(self, pairs):
        self.cation = np.array([a for a, _ in pairs], int)
        self.anion = np.array([b for _, b in pairs], int)


# -- the shells -----------------------------------------------------------------

def test_shells_walk_outwards_over_undirected_bonds():
    # 0-1, 0-2 (first shell of 0); 1-3, 2-4 (second); 4-5 (third); 6 alone
    bonds = _Bonds([(0, 1), (0, 2), (3, 1), (4, 2), (4, 5)])
    keep, shell = md_environment.neighbour_shells(bonds, 7, 0, shells=1)
    assert keep.tolist() == [True, True, True, False, False, False, False]
    assert shell.tolist() == [0, 1, 1, -1, -1, -1, -1]
    keep, shell = md_environment.neighbour_shells(bonds, 7, 0, shells=2)
    assert shell.tolist() == [0, 1, 1, 2, 2, -1, -1]
    keep, shell = md_environment.neighbour_shells(bonds, 7, 0, shells=0)
    assert keep.sum() == 1 and shell[0] == 0
    # from an atom in the middle, the walk goes both ways
    _, shell = md_environment.neighbour_shells(bonds, 7, 2, shells=1)
    assert shell.tolist() == [1, -1, 0, -1, 1, -1, -1]


def test_a_row_outside_the_frame_is_refused():
    with pytest.raises(ValueError):
        md_environment.neighbour_shells(_Bonds([(0, 1)]), 2, 5)


# -- the unwrapping and the XYZ ----------------------------------------------------

@pytest.fixture
def frame():
    # a Si at the box edge with an O just across it: the bond is short by
    # the minimum image and a box length long as stored
    box = np.diag([10.0, 10.0, 10.0])
    cart = np.array([[9.8, 5.0, 5.0],     # Si, row 0
                     [0.4, 5.0, 5.0],     # O across the edge, row 1
                     [9.8, 6.6, 5.0],     # O beside, row 2
                     [2.0, 5.0, 5.0],     # Na bonded to the first O, row 3
                     [5.0, 1.0, 1.0]])    # far away, row 4
    return md_model.frame_from_arrays(np.array(["Si", "O", "O", "Na", "Ca"]),
                                      cart, box_ang=box)


def test_unwrapping_brings_the_image_next_to_the_centre(frame):
    rows = np.array([0, 1, 2, 3])
    cart = md_environment.unwrapped_about(frame, rows, 0)
    d = np.linalg.norm(cart - cart[0], axis=1)
    assert d[1] == pytest.approx(0.6, abs=1e-6)       # not 9.4
    assert d[2] == pytest.approx(1.6, abs=1e-6)
    assert d[3] == pytest.approx(2.2, abs=1e-6)       # 9.8 -> 12.0


def test_the_xyz_lists_the_centre_first_then_by_distance(frame):
    bonds = _Bonds([(0, 1), (0, 2), (3, 1)])
    keep, shell = md_environment.neighbour_shells(bonds, 5, 0, shells=2)
    text = md_environment.environment_xyz(frame, keep, 0, shell=shell,
                                          comment="a test\nwith a newline")
    lines = text.splitlines()
    assert lines[0] == "4"
    assert "\n" not in lines[1] and "newline" in lines[1]
    atoms = [ln.split() for ln in lines[2:]]
    assert [a[0] for a in atoms] == ["Si", "O", "O", "Na"]
    assert "row=0" in lines[2] and "shell=0" in lines[2]
    assert "shell=2" in lines[5]
    d = [float(ln.rsplit("d=", 1)[1]) for ln in lines[2:]]
    assert d == sorted(d) and d[0] == 0.0
    # the atom that is not in the environment is not written
    assert "Ca" not in [a[0] for a in atoms]


def test_write_environment_xyz_writes_the_same_text(frame, tmp_path):
    bonds = _Bonds([(0, 1)])
    keep, shell = md_environment.neighbour_shells(bonds, 5, 0, shells=1)
    out = md_environment.write_environment_xyz(tmp_path / "env.xyz", frame,
                                               keep, 0, shell=shell)
    assert out.read_text(encoding="utf-8") == md_environment.environment_xyz(
        frame, keep, 0, shell=shell)


# -- the equals rule -----------------------------------------------------------------

def test_an_equals_rule_picks_exactly_that_value():
    from facet.ui.md_highlight import Rule, _keep_of_rule

    class FD:
        elements = np.array(["Si", "Si", "Si", "O"])

        def descriptor(self, name):
            return np.array([3.0, 4.0, 4.0, np.nan])

    keep = _keep_of_rule(FD(), Rule(kind="show-only", element="Si",
                                    descriptor="Qn", sense="=", value=4))
    assert keep.tolist() == [False, True, True, False]
    keep = _keep_of_rule(FD(), Rule(kind="show-only", element="any",
                                    descriptor="Qn", sense="=", value=3))
    assert keep.tolist() == [True, False, False, False]


# -- a result into rules ------------------------------------------------------------

def test_a_qn_distribution_maps_to_an_equals_rule_on_its_mode():
    from facet.core.md_stats import Distribution
    from facet.ui.md_views import rules_for_result
    d = Distribution(name="Qn Si (BV)", keys=(2, 3, 4), kind="fraction",
                     per_frame=np.array([[0.1, 0.3, 0.6], [0.2, 0.2, 0.6]]),
                     n_items=np.array([400, 400]))
    rules, words = rules_for_result("Qn Si (BV)", d)
    assert rules == [{"element": "Si", "descriptor": "Qn", "sense": "=",
                      "value": 4.0}]
    assert "Q4" in words
    assert rules_for_result("mean CN Si (BV)", 3.9) == ([], "")


def test_qn_counts_are_the_fractions_times_the_items():
    from facet.core.md_stats import Distribution
    from facet.ui.md_results_overview import _counts
    d = Distribution(name="Qn Si (BV)", keys=(3, 4), kind="fraction",
                     per_frame=np.array([[0.25, 0.75], [0.35, 0.65]]),
                     n_items=np.array([400, 400]))
    counts = dict(_counts(d, lambda k: f"Q{k}"))
    assert counts["Q3"] == pytest.approx(120.0)
    assert counts["Q4"] == pytest.approx(280.0)


# -- the controller -----------------------------------------------------------------

@pytest.mark.skipif(not NS3.exists(), reason="the NS3 model is not on this machine")
def test_the_frame_menu_isolates_exports_and_restores(qapp, tmp_path):
    from facet.ui import preview
    import time
    win = preview.PreviewWindow()
    try:
        win.load_many([str(NS3)])
        qapp.processEvents()
        c = win.project.entries[0].controller

        def wait(cond, s=90):
            end = time.monotonic() + s
            while time.monotonic() < end and not cond():
                qapp.processEvents()
                time.sleep(0.05)
            return cond()

        if c.summary is None:
            c.reread({"type_map": NS3_TYPES})
            assert wait(lambda: c.summary is not None)
        c.show_tab("3D view")
        assert wait(lambda: c.frame_view is not None
                    and c.frame_view.table is not None)
        assert wait(lambda: c.view.scene is not None and c.view.scene.n_atoms > 0)
        scene = c.view.scene
        n_all = scene.n_atoms
        atom = int(np.flatnonzero(np.asarray(scene.atom_element) == "Si")[0])
        row = int(scene.atom_site[atom])

        menu = c.build_context_menu(atom)
        texts = [a.text() for a in menu.actions() if not a.isSeparator()]
        assert any(t.startswith("Rotate about Si") for t in texts)
        assert any(t.startswith("Isolate Si") for t in texts)
        assert any(t.startswith("Export the environment") for t in texts)

        c.isolate(row, 1)
        assert wait(lambda: c.view.scene is not None
                    and c.view.scene.n_atoms < n_all)
        small = c.view.scene
        keep, shell = c.environment(row, 1)
        drawn_rows = set(int(s) for s in small.atom_site)
        assert drawn_rows == set(np.flatnonzero(keep).tolist())
        assert row in drawn_rows

        out = c.export_environment(row, tmp_path / "env.xyz")
        lines = out.read_text(encoding="utf-8").splitlines()
        keep2, _ = c.environment(row, 2)
        assert int(lines[0]) == int(keep2.sum())
        assert lines[2].startswith("Si")

        c.isolate(None)
        assert wait(lambda: c.view.scene is not None
                    and c.view.scene.n_atoms == n_all)

        c.highlight_from_result(
            [{"element": "Si", "descriptor": "Qn", "sense": "=", "value": 4}],
            "Si atoms with Q4")
        qapp.processEvents()
        rules = c.highlight.rules()
        assert len(rules) == 1 and rules[0].sense == "=" \
            and rules[0].value == 4 and rules[0].enabled
    finally:
        win.close()
        dispose(win)


def test_the_rest_choice_reads_both_ways(qapp):
    """Dimmed or left out, as a two-way control; the hidden checkbox the
    older callers read says the same thing."""
    from facet.ui.md_highlight import HighlightPage
    page = HighlightPage()
    try:
        assert page.dim_rest() and page.rest.currentIndex() == 0
        page.rest.setCurrentIndex(1)
        assert not page.dim_rest()
        page.dim.setChecked(True)
        assert page.rest.currentIndex() == 0 and page.dim_rest()
    finally:
        dispose(page)
