"""The environments batch: a per-atom environment descriptor, grouping
within a tolerance, a representative per group, group highlight, scalars
per group, populations over frames, group populations registered for
Compare, the per-atom table, and the atom lookup in the Elements dock.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from conftest import dispose

from facet.core import bulk, bv, md_environments as env_mod, md_model, md_readers, prefs
from facet.ui.md_highlight import FrameData

NS3 = Path(r"C:\Users\samso\AppData\Local\Temp\claude\C--Users-samso"
           r"\342b7e11-8171-41a0-8eee-4126254c9cc5\scratchpad"
           r"\facet_md_descriptors\models\ns3_pedone\nvt300K_last.data")
NS3_TYPES = {1: "Si", 2: "O", 3: "Na"}
needs_ns3 = pytest.mark.skipif(not NS3.exists(),
                               reason="the NS3 model is not on this machine")
GLASS = Path(__file__).resolve().parent / "data" / "md" / "glass.extxyz"


class _Bonds:
    def __init__(self, pairs, vectors):
        self.cation = np.array([a for a, _ in pairs], int)
        self.anion = np.array([b for _, b in pairs], int)
        self.vec_ang = np.array(vectors, float)


def _tetrahedron(scale=1.6):
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    return v / np.linalg.norm(v, axis=1)[:, None] * scale


def test_environments_describe_shells_and_angles():
    # two identical SiO4 tetrahedra sharing nothing, and one distorted
    box = np.diag([30.0, 30.0, 30.0])
    tet = _tetrahedron()
    cart = [[5, 5, 5]] + list(np.array([5, 5, 5]) + tet) \
        + [[15, 15, 15]] + list(np.array([15, 15, 15]) + tet) \
        + [[25, 5, 5]] + list(np.array([25, 5, 5]) + tet * np.array([1.0, 1.0, 1.0, 1.25])[:, None])
    elements = np.array((["Si"] + ["O"] * 4) * 3)
    frame = md_model.frame_from_arrays(elements, np.array(cart, float), box_ang=box)
    pairs, vectors = [], []
    for centre in (0, 5, 10):
        for k in range(1, 5):
            pairs.append((centre, centre + k))
            vectors.append(np.asarray(cart[centre + k]) - np.asarray(cart[centre]))
    bonds = _Bonds(pairs, vectors)
    envs = env_mod.environments(frame, bonds, element="Si")
    assert [e.cn for e in envs] == [4, 4, 4]
    assert envs[0].first_shell == (("O", 4),)
    assert envs[0].angles_deg.mean() == pytest.approx(109.47, abs=0.05)
    assert np.allclose(envs[0].distances_ang, 1.6)
    groups = env_mod.group_environments(envs, tol_distance_ang=0.05,
                                        tol_angle_deg=3.0)
    assert [g.n for g in groups] == [2, 1]
    assert groups[0].label == "Si environment 1" and groups[1].label == "Si environment 2"
    assert groups[0].representative in (0, 5)
    assert groups[1].members == [10]
    # a wide tolerance puts them together
    wide = env_mod.group_environments(envs, tol_distance_ang=0.5, tol_angle_deg=30.0)
    assert len(wide) == 1 and wide[0].n == 3
    env_mod.group_scalars(groups, {"bvs": np.arange(15, dtype=float)})
    assert groups[0].scalars["bvs"] == pytest.approx(2.5)
    rows = env_mod.group_rows(groups)
    assert rows[0]["n"] == 2 and rows[0]["first_shell"] == "O4"
    atoms = env_mod.per_atom_rows(frame, envs, groups, {"bvs": np.arange(15, dtype=float)})
    assert atoms[2]["group"] == "Si environment 2" and atoms[0]["bvs"] == 0.0


@needs_ns3
def test_ns3_silicon_groups_and_the_populations(qapp):
    frame = md_readers.read_trajectory(str(NS3), type_map=NS3_TYPES).frame(0)
    ox = md_model.model_oxidation(frame.species).per_atom(frame.elements)
    table, _ = bulk.analyse_frame(frame, ox, bv.DEFAULT, v_bond_vu=0.075,
                                  v_list_vu=0.02)
    fd = FrameData.from_frame(frame, table, k=0, v_bond=0.075,
                              formers=("Si",), ox_atom=ox)
    envs = env_mod.environments(frame, fd.bonds, element="Si",
                                formers=("Si",), qn=fd.qn)
    assert len(envs) == int((frame.elements == "Si").sum())
    groups = env_mod.group_environments(envs, tol_distance_ang=0.05,
                                        tol_angle_deg=8.0,
                                        same_second_shell=False)
    assert sum(g.n for g in groups) == len(envs)
    assert all(g.representative in g.members for g in groups)
    assert groups[0].label.startswith("Si Q")
    pops = env_mod.populations(groups)
    assert sum(pops.values()) == len(envs)


@needs_ns3
def test_the_tab_groups_highlights_exports_and_looks_up(qapp, tmp_path):
    prefs.set("tour_offered", True)
    from facet.ui import preview
    win = preview.PreviewWindow()
    try:
        win.load_many([str(NS3)])
        qapp.processEvents()
        c = win.project.entries[0].controller

        def wait(cond, s=120):
            end = time.monotonic() + s
            while time.monotonic() < end and not cond():
                qapp.processEvents()
                time.sleep(0.05)
            return cond()

        if c.summary is None:
            c.reread({"type_map": NS3_TYPES})
            assert wait(lambda: c.summary is not None)
        c.show_tab("3D view")
        assert wait(lambda: c.view.scene is not None and c.view.scene.n_atoms > 0)
        c.page.panel.formers.set_ticked(["Si"]) if hasattr(c.page.panel.formers, "set_ticked") else None
        tab = c.environments
        assert "Environments" in [c.tabs.tabText(i) for i in range(c.tabs.count())]
        tab.species.setCurrentIndex(tab.species.findText("Si"))
        tab.second.setChecked(False)
        c.group_environments()
        assert wait(lambda: bool(tab.groups), 120)
        assert tab.table.rowCount() == len(tab.groups)
        assert "bvs" in tab.groups[0].scalars
        # a click highlights the group: a rows rule on the Highlight tab
        n_all = c.view.scene.n_atoms
        tab.table.setCurrentCell(0, 0)
        tab._on_click(tab.table.item(0, 0))
        rules = c.highlight.rules()
        assert any(r.kind == "rows" and r.enabled for r in rules)
        assert wait(lambda: c.highlight.result is not None
                    and c.highlight.result.n_matching == tab.groups[0].n, 60)
        # the populations are registered for Compare once a result exists
        if c.result is not None:
            assert "environments" in c.result.outputs
            assert any(k.startswith("environments")
                       for k in c.result.outputs["environments"].tables)
        # exports
        rows = tab.group_rows()
        assert rows and rows[0]["group"] == tab.groups[0].label
        atoms = tab.atom_rows(c.frame_view.frame)
        assert len(atoms) == len(tab.envs) and "group" in atoms[0]
        out = c.export_environment(tab.groups[0].representative, tmp_path / "rep.xyz")
        assert out.exists()
        # the lookup by id
        ids = np.asarray(c.frame_view.frame.atom_id)
        c.go_to_atom(int(ids[5]))
        assert c._atom_row == 5
        c.elements.lookup.setText("not a number")
        c.elements._on_lookup()
        assert "not an atom id" in c.elements.note.text()
    finally:
        win.close()
        dispose(win)


def test_populations_over_two_frames_of_the_small_glass(qapp):
    prefs.set("tour_offered", True)
    from facet.ui import preview
    win = preview.PreviewWindow()
    try:
        win.load_many([str(GLASS)])
        qapp.processEvents()
        c = win.project.entries[0].controller

        def wait(cond, s=60):
            end = time.monotonic() + s
            while time.monotonic() < end and not cond():
                qapp.processEvents()
                time.sleep(0.05)
            return cond()

        assert wait(lambda: c.summary is not None)
        c.show_tab("3D view")
        assert wait(lambda: c.view.scene is not None and c.view.scene.n_atoms > 0)
        tab = c.environments
        tab.species.setCurrentIndex(tab.species.findText("O"))
        c.group_environments()
        assert wait(lambda: tab.groups is not None and tab.summary.text() != "No grouping yet.")
        if tab.groups:
            c.environments_over_frames()
            assert wait(lambda: getattr(c, "environments_series", None) is not None)
            assert len(c.environments_series) == 2
    finally:
        win.close()
        dispose(win)
