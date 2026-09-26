"""Disorder groups.

The point of this feature is a specific, damaging failure: two alternative
configurations drawn together sit within a fraction of an angstrom of each other,
so they read as a cluster of impossibly close atoms and a coordination analysis
counts neighbours the real structure never has. That is one of the ways a
published coordination number stops meaning anything, which is what this program
is about -- so the tests measure it rather than assume it.

The chosen configuration is a real Structure with fewer sites, not a flag. That
is what these tests mostly check: that everything downstream sees a structure
which could exist, with no disorder special case of its own.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from facet.core import cif
from facet.core import disorder as D

DISORDERED = """data_disordered
_cell_length_a 8.0000
_cell_length_b 8.0000
_cell_length_c 8.0000
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_disorder_assembly
_atom_site_disorder_group
Bi1  Bi 0.0000 0.0000 0.0000 1.00  .  .
O1   O  0.2500 0.0000 0.0000 1.00  .  .
O2   O  0.0000 0.2500 0.0000 1.00  .  .
Na1  Na 0.5100 0.5000 0.5000 0.60  A  1
Na2  Na 0.5400 0.5000 0.5000 0.40  A  2
K1   K  0.0000 0.5000 0.2500 0.70  B  1
K2   K  0.0000 0.5000 0.3000 0.30  B  2
"""

ORDERED = """data_ordered
_cell_length_a 8.0000
_cell_length_b 8.0000
_cell_length_c 8.0000
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Bi1  Bi 0.0000 0.0000 0.0000 1.00
O1   O  0.2500 0.0000 0.0000 1.00
"""

# A file that fills the column in but declares only one alternative. Not
# disorder: a single group is no choice at all.
ONE_GROUP = DISORDERED.replace("Na2  Na 0.5400 0.5000 0.5000 0.40  A  2\n", "")\
    .replace("K2   K  0.0000 0.5000 0.3000 0.30  B  2\n", "")


def _write(text):
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "test.cif")
    Path(path).write_text(text, encoding="utf-8")
    return cif.read(path)


@pytest.fixture
def structure():
    return _write(DISORDERED)


# ===========================================================================
# reading the tags
# ===========================================================================

def test_the_tags_are_read_off_the_sites(structure):
    tags = {site.label: (site.disorder_assembly, site.disorder_group)
            for site in structure.sites}
    assert tags["Bi1"] == ("", "")
    assert tags["O1"] == ("", "")
    assert tags["Na1"] == ("A", "1")
    assert tags["Na2"] == ("A", "2")
    assert tags["K1"] == ("B", "1")
    assert tags["K2"] == ("B", "2")


def test_a_file_with_no_tags_reads_as_ordered():
    structure = _write(ORDERED)
    for site in structure.sites:
        assert site.disorder_assembly == ""
        assert site.disorder_group == ""
    assert not D.find(structure).present


def test_the_cif_nulls_mean_no_group():
    for value in (".", "?", "", "0", None, " . ", "'.'"):
        assert D.normalise_group(value) == ""
    assert D.normalise_group("1") == "1"
    assert D.normalise_group(" 2 ") == "2"
    for value in (".", "?", "", None):
        assert D.normalise_assembly(value) == ""
    # 0 is an ordered site's group but a legitimate assembly name
    assert D.normalise_assembly("0") == "0"


def test_loading_a_disordered_file_says_so(structure):
    joined = " ".join(structure.notes).lower()
    assert "disorder" in joined
    assert "alternatives" in joined


# ===========================================================================
# what the assemblies are
# ===========================================================================

def test_the_assemblies_and_their_groups_are_found(structure):
    disorder = D.find(structure)
    assert set(disorder.assemblies) == {"A", "B"}
    assert disorder.assemblies["A"].groups == ["1", "2"]
    assert disorder.assemblies["B"].groups == ["1", "2"]
    assert disorder.present
    assert disorder.n_alternatives == 4


def test_one_group_is_not_disorder():
    """A file that filled in the column is not a file with alternatives."""
    structure = _write(ONE_GROUP)
    disorder = D.find(structure)
    assert disorder.assemblies          # the tags were read
    assert not disorder.present         # but there is no choice to make
    assert disorder.n_alternatives == 0


def test_an_assembly_describes_its_groups(structure):
    disorder = D.find(structure)
    text = disorder.assemblies["A"].describe()
    assert "A" in text and "1" in text and "2" in text


def test_the_description_says_what_is_being_shown(structure):
    disorder = D.find(structure)
    disorder.show_everything()
    assert any("every group" in line for line in disorder.describe())
    disorder.choose("A", "2")
    assert any("showing 2" in line for line in disorder.describe())


# ===========================================================================
# the overlap, measured
# ===========================================================================

def test_the_alternatives_really_do_overlap(structure):
    """The measurement that justifies the whole feature.

    Na1 and Na2 are 0.03 in fractional x apart in an 8 A cell: 0.24 A. No real
    pair of sodium atoms is 0.24 A apart, which is exactly why they are
    alternatives and not two sites.
    """
    disorder = D.find(structure)
    pairs = D.close_pairs_between_groups(structure, disorder)
    assert pairs, "the overlapping alternatives were not found"

    by_labels = {(a, b): d for a, _, b, _, d in pairs}
    assert by_labels[("Na1", "Na2")] == pytest.approx(0.24, abs=1e-6)
    assert by_labels[("K1", "K2")] == pytest.approx(0.40, abs=1e-6)
    # sorted closest first
    assert [row[4] for row in pairs] == sorted(row[4] for row in pairs)


def test_only_cross_group_pairs_within_one_assembly_are_reported(structure):
    """A and B are different regions; their atoms are not alternatives."""
    disorder = D.find(structure)
    pairs = D.close_pairs_between_groups(structure, disorder)
    for label_a, group_a, label_b, group_b, _ in pairs:
        assert group_a != group_b
        site_a = next(s for s in structure.sites if s.label == label_a)
        site_b = next(s for s in structure.sites if s.label == label_b)
        assert site_a.disorder_assembly == site_b.disorder_assembly


def test_an_ordered_structure_reports_no_overlaps():
    structure = _write(ORDERED)
    assert D.close_pairs_between_groups(structure, D.find(structure)) == []


# ===========================================================================
# choosing a configuration
# ===========================================================================

def test_showing_everything_keeps_every_site(structure):
    disorder = D.find(structure)
    disorder.show_everything()
    assert disorder.is_showing_everything()
    assert len(D.shown_sites(structure, disorder)) == len(structure.sites)
    assert D.configuration(structure, disorder) is structure


def test_choosing_the_first_group_drops_the_others(structure):
    disorder = D.find(structure)
    disorder.choose_first()
    shown = D.shown_sites(structure, disorder)
    labels = [structure.sites[i].label for i in shown]
    assert labels == ["Bi1", "O1", "O2", "Na1", "K1"]
    assert not disorder.is_showing_everything()


def test_an_untagged_site_is_in_every_configuration(structure):
    disorder = D.find(structure)
    for choice in (("1", "1"), ("1", "2"), ("2", "1"), ("2", "2")):
        disorder.choose("A", choice[0])
        disorder.choose("B", choice[1])
        labels = [structure.sites[i].label
                  for i in D.shown_sites(structure, disorder)]
        for always in ("Bi1", "O1", "O2"):
            assert always in labels


def test_every_alternative_can_be_chosen(structure):
    disorder = D.find(structure)
    seen = set()
    for a in ("1", "2"):
        for b in ("1", "2"):
            disorder.choose("A", a)
            disorder.choose("B", b)
            labels = tuple(structure.sites[i].label
                           for i in D.shown_sites(structure, disorder))
            seen.add(labels)
    assert len(seen) == 4 == disorder.n_alternatives


def test_the_configuration_is_a_real_structure(structure):
    """Not a flag: a Structure with fewer sites, consistent throughout."""
    disorder = D.find(structure)
    disorder.choose_first()
    one = D.configuration(structure, disorder)

    assert one is not structure
    assert len(one.sites) == 5
    assert [s.label for s in one.sites] == ["Bi1", "O1", "O2", "Na1", "K1"]
    # every atom points at a site that exists, and at the right one
    for atom in one.atoms:
        assert 0 <= atom.site_index < len(one.sites)
        assert one.sites[atom.site_index].element == atom.element
    # the cell is untouched
    assert one.cell.a == structure.cell.a
    # and the original is not modified
    assert len(structure.sites) == 7


def test_the_configuration_says_what_it_left_out(structure):
    disorder = D.find(structure)
    disorder.choose_first()
    one = D.configuration(structure, disorder)
    note = " ".join(one.notes).lower()
    assert "one disorder configuration" in note
    assert "not been renormalised" in note
    assert "2 of the file" in note


def test_occupancies_are_not_renormalised(structure):
    """Scaling a chosen group to full occupancy would invent a structure.

    It would also silently change every bond valence, since valence is weighted
    by occupancy -- an unannounced change to the numbers the program exists to
    report.
    """
    disorder = D.find(structure)
    disorder.choose("A", "1")
    one = D.configuration(structure, disorder)
    sodium = next(s for s in one.sites if s.label == "Na1")
    assert sodium.occupancy == pytest.approx(0.60)


def test_the_configuration_removes_the_overlap(structure):
    """The whole point: after choosing, no two atoms are absurdly close."""
    disorder = D.find(structure)
    disorder.choose_first()
    one = D.configuration(structure, disorder)

    frac = np.array([a.frac for a in one.atoms])
    delta = frac[:, None, :] - frac[None, :, :]
    delta -= np.round(delta)
    cart = np.einsum("ij,mnj->mni", one.cell.orth, delta)
    distance = np.linalg.norm(cart, axis=-1)
    np.fill_diagonal(distance, 9e9)
    assert float(distance.min()) > 1.2, "an overlapping pair survived"

    # and before choosing, it did not hold
    everything = D.find(structure)
    everything.show_everything()
    frac = np.array([a.frac for a in structure.atoms])
    delta = frac[:, None, :] - frac[None, :, :]
    delta -= np.round(delta)
    cart = np.einsum("ij,mnj->mni", structure.cell.orth, delta)
    distance = np.linalg.norm(cart, axis=-1)
    np.fill_diagonal(distance, 9e9)
    assert float(distance.min()) < 0.5


def test_the_selection_round_trips(structure):
    disorder = D.find(structure)
    disorder.choose("A", "2")
    disorder.choose("B", "1")
    restored = D.find(structure)
    restored.apply_dict(disorder.to_dict())
    assert restored.selected == disorder.selected


def test_applying_an_unknown_assembly_is_ignored(structure):
    disorder = D.find(structure)
    disorder.apply_dict({"selected": {"Z": "9"}})
    assert "Z" not in disorder.selected


# ===========================================================================
# the effect on the analysis
# ===========================================================================

def test_the_coordination_number_changes_with_the_configuration(structure):
    """The consequence that matters: a different alternative, a different CN.

    If the alternatives were drawn together, the bismuth would be counted as
    coordinated to atoms from both -- neighbours the real structure never has.
    """
    from facet.core import bv, coordination

    def bismuth(struct):
        results = coordination.analyse_structure(struct, bv.DEFAULT)
        return next(r for r in results if r.label.startswith("Bi"))

    disorder = D.find(structure)
    disorder.show_everything()
    both = bismuth(D.configuration(structure, disorder))

    disorder.choose_first()
    one = bismuth(D.configuration(structure, disorder))

    # fewer contacts once the alternatives are separated
    assert len(one.contacts) <= len(both.contacts)
    assert one.bvs <= both.bvs + 1e-9


def test_the_entry_hands_out_one_configuration_by_default(structure):
    """A program about coordination numbers must not default to a structure
    whose coordination numbers are those of no real configuration."""
    from facet.core.project import Entry

    entry = Entry(source=structure)
    assert entry.disorder.present
    assert not entry.disorder.is_showing_everything()
    assert len(entry.structure.sites) < len(entry.source.sites)
    assert entry.source is structure


def test_an_ordered_entry_hands_back_the_file_itself():
    from facet.core.project import Entry

    structure = _write(ORDERED)
    entry = Entry(source=structure)
    assert not entry.disorder.present
    assert entry.structure is structure


def test_choosing_through_the_entry_invalidates_the_analysis(structure):
    from facet.core import bv
    from facet.core.project import Entry

    entry = Entry(source=structure)
    first = entry.results(bv.DEFAULT, 0.075, 0.02)
    labels_first = {r.label for r in first}

    entry.choose_disorder("A", "2")
    second = entry.results(bv.DEFAULT, 0.075, 0.02)
    assert second is not first
    assert {r.label for r in second} != labels_first


def test_the_configuration_is_cached_until_it_changes(structure):
    from facet.core.project import Entry

    entry = Entry(source=structure)
    first = entry.structure
    assert entry.structure is first, "the configuration was rebuilt"
    entry.choose_disorder("A", "2")
    assert entry.structure is not first


def test_a_disordered_structure_still_gives_a_diffraction_pattern(structure):
    """Two alternatives both contributing would be a different pattern."""
    from facet.core import diffraction
    from facet.core.project import Entry

    entry = Entry(source=structure)
    one = diffraction.powder_pattern(entry.structure, two_theta_max=60.0)
    assert one.reflections

    entry.disorder.show_everything()
    entry.invalidate()
    everything = diffraction.powder_pattern(entry.structure, two_theta_max=60.0)
    assert {r.hkl for r in one.reflections} == {
        r.hkl for r in everything.reflections}
    # the same lines, different intensities: both groups scatter
    intensities_one = {r.hkl: r.intensity for r in one.reflections}
    intensities_all = {r.hkl: r.intensity for r in everything.reflections}
    assert intensities_one != intensities_all


# ===========================================================================
# the panel
# ===========================================================================

@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_the_panel_lists_a_chooser_per_assembly(qapp, structure):
    from facet.core.project import Entry
    from facet.ui.disorder_panel import DisorderPanel

    panel = DisorderPanel()
    panel.set_entry(Entry(source=structure))
    assert set(panel._choosers) == {"A", "B"}
    for chooser in panel._choosers.values():
        # "all at once" plus one per group
        assert chooser.count() == 3


def test_the_panel_reports_the_overlap(qapp, structure):
    from facet.core.project import Entry
    from facet.ui.disorder_panel import DisorderPanel

    panel = DisorderPanel()
    panel.set_entry(Entry(source=structure))
    assert panel.overlaps.rowCount() == 2
    assert "0.2400" in panel.overlap_note.text()


def test_choosing_in_the_panel_changes_the_entry(qapp, structure):
    from facet.core.project import Entry
    from facet.ui.disorder_panel import DisorderPanel

    entry = Entry(source=structure)
    panel = DisorderPanel()
    panel.set_entry(entry)

    seen = []
    panel.changed.connect(seen.append)
    chooser = panel._choosers["A"]
    chooser.setCurrentIndex(chooser.findData("2"))

    assert entry.disorder.selected["A"] == "2"
    assert [s.label for s in entry.structure.sites].count("Na2") == 1
    assert seen and "2" in seen[0]


def test_the_panel_can_show_everything_and_go_back(qapp, structure):
    from facet.core.project import Entry
    from facet.ui.disorder_panel import DisorderPanel

    entry = Entry(source=structure)
    panel = DisorderPanel()
    panel.set_entry(entry)

    panel._show_all()
    assert entry.disorder.is_showing_everything()
    assert len(entry.structure.sites) == len(structure.sites)

    panel._choose_first()
    assert not entry.disorder.is_showing_everything()
    assert len(entry.structure.sites) < len(structure.sites)


def test_the_panel_says_nothing_when_a_file_is_ordered(qapp):
    from facet.core.project import Entry
    from facet.ui.disorder_panel import DisorderPanel

    panel = DisorderPanel()
    panel.set_entry(Entry(source=_write(ORDERED)))
    assert panel._choosers == {}
    assert "no disorder" in panel.hint.text().lower()
    assert not panel.first_button.isEnabled()


def test_the_panel_never_says_which_alternative_is_right(qapp, structure):
    import re

    from facet.core.project import Entry
    from facet.ui.disorder_panel import DisorderPanel

    panel = DisorderPanel()
    panel.set_entry(Entry(source=structure))
    text = " ".join([panel.hint.text(), panel.overlap_note.text(),
                     panel.summary.text()]).lower()
    forbidden = {"correct", "incorrect", "wrong", "better", "worse", "should",
                 "likely", "probably", "clearly", "obviously", "best"}
    assert not (set(re.findall(r"[a-z]+", text)) & forbidden)


def test_the_panel_survives_having_no_entry(qapp):
    from facet.ui.disorder_panel import DisorderPanel

    panel = DisorderPanel()
    panel.set_entry(None)
    assert panel._choosers == {}
    assert panel.overlaps.rowCount() == 0
