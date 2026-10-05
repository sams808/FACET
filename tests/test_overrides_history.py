"""Per-site and per-atom overrides, and the undo history.

Two things worth testing carefully here.

The override precedence -- atom over site over element -- with only the fields
actually set taking effect. A partial override that silently blanked the fields
it did not mention would be the obvious bug, and it would look like a theme
problem rather than an override problem.

And undo, which stores snapshots rather than inverse operations. The property to
check is the round trip: change something, undo, and the state must be
*identical* to before, not merely similar. Redo must land back on the change.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from conftest import sample_cif

from facet.core import history as history_mod
from facet.core import overrides as O

SAMPLE = sample_cif("1526458", "1526458_Bi2O3.cif")


@pytest.fixture(scope="module")
def structure():
    if not SAMPLE.is_file():
        pytest.skip("the sample structure is not present")
    from facet.core import cif

    return cif.read(SAMPLE)


# ===========================================================================
# Style: only what is set applies
# ===========================================================================

def test_an_empty_style_overrides_nothing():
    assert O.Style().is_empty
    assert O.Style().describe() == "nothing overridden"


def test_a_style_with_one_field_is_not_empty():
    assert not O.Style(color=(1.0, 0.0, 0.0)).is_empty
    assert not O.Style(radius_scale=1.5).is_empty
    assert not O.Style(visible=False).is_empty
    assert not O.Style(label="X").is_empty


def test_over_takes_the_set_fields_and_keeps_the_rest():
    """The precedence rule, stated directly.

    A field set on top wins; a field left unset falls through. A style that
    blanked the fields it did not mention would make every partial override
    destructive.
    """
    under = O.Style(color=(0.1, 0.2, 0.3), radius_scale=2.0, visible=True,
                    label="under")
    over = O.Style(color=(0.9, 0.9, 0.9))
    combined = over.over(under)
    assert combined.color == (0.9, 0.9, 0.9)          # from over
    assert combined.radius_scale == 2.0               # fell through
    assert combined.visible is True
    assert combined.label == "under"


def test_over_with_an_empty_style_changes_nothing():
    under = O.Style(color=(0.1, 0.2, 0.3), radius_scale=2.0)
    assert O.Style().over(under) == under


def test_visible_false_survives_the_fall_through():
    """False is a value, not an absence. `or` would lose it."""
    combined = O.Style(visible=False).over(O.Style(visible=True))
    assert combined.visible is False


def test_a_zero_radius_scale_survives_the_fall_through():
    combined = O.Style(radius_scale=0.0).over(O.Style(radius_scale=3.0))
    assert combined.radius_scale == 0.0


def test_describe_names_every_set_field():
    text = O.Style(color=(1.0, 0.5, 0.0), radius_scale=1.5, visible=False,
                   label="Bi(A)").describe()
    for piece in ("colour", "radius", "hidden", "Bi(A)"):
        assert piece in text


# ===========================================================================
# StyleOverrides: the three levels
# ===========================================================================

def test_atom_beats_site_beats_element():
    overrides = O.StyleOverrides()
    overrides.set_element("Bi", color=(0.0, 0.0, 1.0), radius_scale=3.0)
    overrides.set_site("Bi1", color=(0.0, 1.0, 0.0))
    overrides.set_atom(7, color=(1.0, 0.0, 0.0))

    # the atom's own colour wins, and the element's radius still falls through
    style = overrides.for_atom(7, "Bi1", "Bi")
    assert style.color == (1.0, 0.0, 0.0)
    assert style.radius_scale == 3.0

    # a different atom of the same site takes the site's colour
    assert overrides.for_atom(8, "Bi1", "Bi").color == (0.0, 1.0, 0.0)

    # a different site of the same element takes the element's
    assert overrides.for_atom(9, "Bi2", "Bi").color == (0.0, 0.0, 1.0)

    # an atom of another element entirely is untouched
    assert overrides.for_atom(10, "O1", "O").is_empty


def test_setting_a_field_keeps_the_others():
    overrides = O.StyleOverrides()
    overrides.set_site("Bi1", color=(1.0, 0.0, 0.0))
    overrides.set_site("Bi1", radius_scale=2.0)
    style = overrides.by_site["Bi1"]
    assert style.color == (1.0, 0.0, 0.0)
    assert style.radius_scale == 2.0


def test_setting_a_field_back_to_none_removes_an_emptied_override():
    overrides = O.StyleOverrides()
    overrides.set_site("Bi1", color=(1.0, 0.0, 0.0))
    assert "Bi1" in overrides.by_site
    overrides.set_site("Bi1", color=None)
    assert "Bi1" not in overrides.by_site, "an empty override was kept"
    assert overrides.is_empty


def test_element_keys_are_normalised():
    overrides = O.StyleOverrides()
    overrides.set_element("bi", color=(1.0, 0.0, 0.0))
    assert "Bi" in overrides.by_element
    assert overrides.for_atom(0, "X", "BI").color == (1.0, 0.0, 0.0)
    overrides.clear_element("BI")
    assert overrides.is_empty


def test_clearing_one_level_leaves_the_others():
    overrides = O.StyleOverrides()
    overrides.set_element("Bi", color=(0.0, 0.0, 1.0))
    overrides.set_site("Bi1", color=(0.0, 1.0, 0.0))
    overrides.set_atom(3, color=(1.0, 0.0, 0.0))
    overrides.clear_site("Bi1")
    assert overrides.for_atom(3, "Bi1", "Bi").color == (1.0, 0.0, 0.0)
    assert overrides.for_atom(4, "Bi1", "Bi").color == (0.0, 0.0, 1.0)


def test_count_and_is_empty_agree():
    overrides = O.StyleOverrides()
    assert overrides.is_empty and overrides.count() == 0
    overrides.set_site("Bi1", visible=False)
    overrides.set_atom(2, visible=False)
    overrides.set_element("O", visible=False)
    assert not overrides.is_empty and overrides.count() == 3
    overrides.clear()
    assert overrides.is_empty and overrides.count() == 0


def test_prune_drops_overrides_that_address_nothing():
    """A stale override applied to whatever landed on that index would be worse
    than losing it."""
    overrides = O.StyleOverrides()
    overrides.set_atom(3, visible=False)
    overrides.set_atom(500, visible=False)
    overrides.set_site("Bi1", visible=False)
    overrides.set_site("Gone1", visible=False)
    overrides.set_element("Bi", visible=False)

    overrides.prune(n_atoms=20, site_labels=["Bi1", "O1"])
    assert set(overrides.by_atom) == {3}
    assert set(overrides.by_site) == {"Bi1"}
    # elements are not pruned: a symbol is a symbol
    assert set(overrides.by_element) == {"Bi"}


# ===========================================================================
# persistence
# ===========================================================================

def test_overrides_round_trip_through_a_dictionary():
    overrides = O.StyleOverrides()
    overrides.set_element("Bi", color=(0.1, 0.2, 0.3))
    overrides.set_site("Bi1", radius_scale=1.75, note="the lone-pair site")
    overrides.set_atom(12, visible=False, label="Bi(A)")

    restored = O.StyleOverrides.from_dict(overrides.to_dict())
    assert restored.by_element["Bi"].color == pytest.approx((0.1, 0.2, 0.3))
    assert restored.by_site["Bi1"].radius_scale == pytest.approx(1.75)
    assert restored.by_site["Bi1"].note == "the lone-pair site"
    assert restored.by_atom[12].visible is False
    assert restored.by_atom[12].label == "Bi(A)"


def test_a_round_trip_survives_json():
    overrides = O.StyleOverrides()
    overrides.set_atom(4, color=(1.0, 0.0, 0.0), radius_scale=0.5)
    text = json.dumps(overrides.to_dict())
    restored = O.StyleOverrides.from_dict(json.loads(text))
    # keys come back as strings from JSON and must become integers again
    assert 4 in restored.by_atom
    assert restored.by_atom[4].radius_scale == pytest.approx(0.5)


def test_loading_rubbish_does_not_raise():
    for data in (None, {}, {"by_atom": {"not a number": {}}},
                 {"by_site": {"Bi1": {"color": None}}}):
        restored = O.StyleOverrides.from_dict(data)
        assert isinstance(restored, O.StyleOverrides)


def test_an_unset_field_is_not_written():
    overrides = O.StyleOverrides()
    overrides.set_site("Bi1", color=(1.0, 0.0, 0.0))
    encoded = overrides.to_dict()["by_site"]["Bi1"]
    assert set(encoded) == {"color"}


# ===========================================================================
# applying to a scene
# ===========================================================================

def test_a_colour_override_recolours_only_its_target(structure):
    from facet.gl.scene import build_scene

    plain = build_scene(structure)
    site_label = structure.sites[0].label

    overrides = O.StyleOverrides()
    overrides.set_site(site_label, color=(1.0, 0.0, 0.0))
    styled = build_scene(structure, overrides=overrides)

    for index in range(styled.n_atoms):
        if styled.atom_label[index] == site_label:
            assert np.allclose(styled.atom_color[index], (1.0, 0.0, 0.0))
        else:
            assert np.allclose(styled.atom_color[index],
                               plain.atom_color[index])


def test_a_radius_override_scales_only_its_target(structure):
    from facet.gl.scene import build_scene

    plain = build_scene(structure)
    site_label = structure.sites[0].label

    overrides = O.StyleOverrides()
    overrides.set_site(site_label, radius_scale=2.0)
    styled = build_scene(structure, overrides=overrides)

    for index in range(styled.n_atoms):
        factor = 2.0 if styled.atom_label[index] == site_label else 1.0
        assert styled.atom_radius[index] == pytest.approx(
            plain.atom_radius[index] * factor, rel=1e-6)


def test_hiding_a_site_removes_its_bonds_but_not_the_other_atoms(structure):
    from facet.gl.scene import build_scene

    plain = build_scene(structure)
    site_label = structure.sites[0].label

    overrides = O.StyleOverrides()
    overrides.set_site(site_label, visible=False)
    styled = build_scene(structure, overrides=overrides)

    # the atom count is unchanged: hiding sets the radius to zero rather than
    # removing atoms, because the override keys are atom indices
    assert styled.n_atoms == plain.n_atoms
    hidden = [i for i in range(styled.n_atoms)
              if styled.atom_label[i] == site_label]
    assert hidden
    for index in hidden:
        assert styled.atom_radius[index] == 0.0
    for index in range(styled.n_atoms):
        if index not in hidden:
            assert styled.atom_radius[index] > 0.0

    # and no bond reaches a hidden atom
    assert styled.n_bonds < plain.n_bonds
    gone = set(hidden)
    for a, b in styled.bond_atoms:
        assert a not in gone and b not in gone


def test_hiding_leaves_every_bond_array_in_step(structure):
    from facet.gl.scene import build_scene

    overrides = O.StyleOverrides()
    overrides.set_site(structure.sites[0].label, visible=False)
    scene = build_scene(structure, overrides=overrides)
    n = scene.n_bonds
    for name in ("bond_a", "bond_b", "bond_radius", "bond_color_a",
                 "bond_color_b", "bond_valence", "bond_distance",
                 "bond_atoms", "_bond_base_a", "_bond_base_b"):
        assert len(getattr(scene, name)) == n, f"{name} is out of step"


def test_an_atom_override_applies_to_exactly_one_atom(structure):
    from facet.gl.scene import build_scene

    overrides = O.StyleOverrides()
    overrides.set_atom(3, color=(1.0, 0.0, 1.0))
    styled = build_scene(structure, overrides=overrides)
    plain = build_scene(structure)

    assert np.allclose(styled.atom_color[3], (1.0, 0.0, 1.0))
    for index in range(styled.n_atoms):
        if index != 3:
            assert np.allclose(styled.atom_color[index],
                               plain.atom_color[index])


def test_no_overrides_leaves_the_scene_exactly_as_it_was(structure):
    from facet.gl.scene import build_scene

    plain = build_scene(structure)
    same = build_scene(structure, overrides=O.StyleOverrides())
    assert np.array_equal(plain.atom_color, same.atom_color)
    assert np.array_equal(plain.atom_radius, same.atom_radius)
    assert plain.n_bonds == same.n_bonds


def test_apply_to_scene_reports_how_many_atoms_it_touched(structure):
    from facet.gl.scene import build_scene

    scene = build_scene(structure)
    overrides = O.StyleOverrides()
    overrides.set_site(structure.sites[0].label, color=(1.0, 0.0, 0.0))
    touched = O.apply_to_scene(scene, overrides)
    expected = sum(1 for label in scene.atom_label
                   if label == structure.sites[0].label)
    assert touched == expected


def test_applying_to_an_empty_scene_is_harmless():
    from facet.gl.scene import Scene

    overrides = O.StyleOverrides()
    overrides.set_site("Bi1", visible=False)
    assert O.apply_to_scene(Scene(), overrides) == 0
    assert O.apply_to_scene(Scene(), None) == 0


# ===========================================================================
# the undo history
# ===========================================================================

def test_a_new_history_can_neither_undo_nor_redo():
    history = history_mod.History()
    assert not history.can_undo and not history.can_redo
    assert history.undo() is None and history.redo() is None


def test_reset_gives_one_state_with_nothing_to_undo():
    history = history_mod.History()
    history.reset({"a": 1}, "opened")
    assert len(history) == 1
    assert not history.can_undo and not history.can_redo
    assert history.current() == {"a": 1}


def test_undo_returns_the_previous_state_exactly():
    history = history_mod.History()
    history.reset({"n": 0})
    history.push("first", {"n": 1})
    history.push("second", {"n": 2})

    assert history.can_undo and not history.can_redo
    assert history.undo_description == "second"
    assert history.undo() == {"n": 1}
    assert history.undo() == {"n": 0}
    assert not history.can_undo
    assert history.undo() is None


def test_redo_retraces_the_steps():
    history = history_mod.History()
    history.reset({"n": 0})
    history.push("first", {"n": 1})
    history.push("second", {"n": 2})
    history.undo()
    history.undo()
    assert history.can_redo
    assert history.redo() == {"n": 1}
    assert history.redo() == {"n": 2}
    assert not history.can_redo
    assert history.redo() is None


def test_acting_after_an_undo_discards_the_redo_branch():
    """What every editor does: once you undo and act, the branch you left is
    gone."""
    history = history_mod.History()
    history.reset({"n": 0})
    history.push("first", {"n": 1})
    history.push("second", {"n": 2})
    history.undo()
    assert history.can_redo
    history.push("different", {"n": 99})
    assert not history.can_redo
    assert history.current() == {"n": 99}
    assert history.undo() == {"n": 1}


def test_a_snapshot_is_copied_not_aliased():
    """The point of snapshots: mutating the live state must not rewrite history.

    Storing a reference instead of a copy is the way a snapshot history quietly
    stops working, and it looks exactly like an undo that does nothing.
    """
    history = history_mod.History()
    state = {"planes": [{"h": 1}]}
    history.reset(state)
    state["planes"][0]["h"] = 99
    state["planes"].append({"h": 5})

    restored = history.current()
    assert restored == {"planes": [{"h": 1}]}

    # and what undo hands back must itself be a copy
    history.push("changed", {"planes": []})
    first = history.undo()
    first["planes"].append({"h": 123})
    assert history.current() == {"planes": [{"h": 1}]}


def test_the_history_is_bounded_and_keeps_the_newest():
    history = history_mod.History(limit=5)
    history.reset({"n": 0})
    for i in range(1, 20):
        history.push(f"step {i}", {"n": i})
    assert len(history) == 5
    assert history.current() == {"n": 19}
    # five states means four undos
    seen = []
    while history.can_undo:
        seen.append(history.undo())
    assert seen == [{"n": 18}, {"n": 17}, {"n": 16}, {"n": 15}]


def test_descriptions_follow_the_cursor():
    history = history_mod.History()
    history.reset({"n": 0}, "opened")
    history.push("added a plane", {"n": 1})
    history.push("hid a site", {"n": 2})
    assert history.undo_description == "hid a site"
    history.undo()
    assert history.undo_description == "added a plane"
    assert history.redo_description == "hid a site"


def test_describe_marks_where_the_cursor_is():
    history = history_mod.History()
    history.reset({"n": 0}, "opened")
    history.push("one", {"n": 1})
    history.undo()
    lines = history.describe()
    assert lines[0].startswith(">")
    assert not lines[1].startswith(">")
