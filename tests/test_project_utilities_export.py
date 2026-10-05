"""The project model, the crystallographic utilities, and the exporters."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from conftest import sample_cif

from facet.core import bv, cif, coordination, exporters, project as P, utilities

CIFS = Path(r"C:\Users\samso\Desktop\WSU_work\XRD\cif\Bi\cifs")
SAMPLE = sample_cif("1526458", "1526458_Bi2O3.cif")
SECOND = sample_cif("7023719", "7023719_BiPO4.cif")

pytestmark = pytest.mark.skipif(not SAMPLE.exists(),
                                reason="sample structures not present")


@pytest.fixture(scope="module")
def loaded():
    s = cif.read(SAMPLE)
    return s, coordination.analyse_structure(s)


@pytest.fixture(scope="module")
def two_structures():
    proj = P.Project()
    proj.add_files([SAMPLE, SECOND])
    return proj


# ---------------------------------------------------------------------------
# project
# ---------------------------------------------------------------------------

class TestProject:

    def test_files_load_and_the_first_becomes_active(self, two_structures):
        assert len(two_structures) == 2
        assert two_structures.active == 0
        assert two_structures.current is not None

    def test_a_bad_file_is_reported_rather_than_raised(self, tmp_path):
        """A folder of downloaded CIFs reliably contains a few that will not
        parse; the rest must still open."""
        broken = tmp_path / "broken.cif"
        broken.write_text("not a cif at all", encoding="utf-8")
        proj = P.Project()
        added, failed = proj.add_files([SAMPLE, broken, SECOND])
        assert len(added) == 2
        assert len(failed) == 1
        assert "broken.cif" in failed[0][0]

    def test_results_are_cached_between_calls(self, two_structures):
        entry = two_structures.entries[0]
        first = two_structures.results_for(entry)
        assert two_structures.results_for(entry) is first

    def test_moving_the_bond_threshold_does_not_re_analyse(self, two_structures):
        """The whole architecture rests on this: contacts are found down to the
        tabulation threshold, and the bond threshold only classifies them."""
        entry = two_structures.entries[0]
        first = two_structures.results_for(entry)
        two_structures.set_bond_threshold(0.2)
        assert two_structures.results_for(entry) is first
        two_structures.set_bond_threshold(bv.V_BOND_DEFAULT)

    def test_moving_the_tabulation_threshold_does_re_analyse(self, two_structures):
        entry = two_structures.entries[0]
        first = two_structures.results_for(entry)
        two_structures.set_list_threshold(0.01)
        assert two_structures.results_for(entry) is not first
        two_structures.set_list_threshold(bv.V_LIST_DEFAULT)

    def test_only_the_active_entry_is_visible_outside_overlay(self, two_structures):
        two_structures.set_overlay(False)
        two_structures.set_active(1)
        visible = two_structures.visible_entries
        assert len(visible) == 1 and visible[0] is two_structures.entries[1]
        two_structures.set_active(0)

    def test_overlay_shows_every_visible_entry(self, two_structures):
        two_structures.set_overlay(True)
        assert len(two_structures.visible_entries) == 2
        two_structures.toggle_visible(1, False)
        assert len(two_structures.visible_entries) == 1
        two_structures.toggle_visible(1, True)
        two_structures.set_overlay(False)

    def test_spacing_lays_structures_out_and_centres_them(self, two_structures):
        two_structures.set_overlay(True, spacing=6.0)
        offsets = [e.offset[0] for e in two_structures.entries]
        assert offsets[0] != offsets[1]
        assert abs(sum(offsets)) < abs(offsets[0]) + abs(offsets[1])
        two_structures.set_overlay(False, spacing=0.0)
        assert all(np.allclose(e.offset, 0) for e in two_structures.entries)

    def test_removing_an_entry_keeps_the_selection_valid(self):
        proj = P.Project()
        proj.add_files([SAMPLE, SECOND])
        proj.set_active(1)
        proj.remove(1)
        assert proj.active == 0 and proj.current is not None
        proj.remove(0)
        assert proj.active is None and proj.current is None

    def test_the_site_table_spans_every_structure(self, two_structures):
        rows = two_structures.site_table()
        assert {r["structure"] for r in rows} == {SAMPLE.name, SECOND.name}
        assert all("v_bond" in r and "plateau_decades" in r for r in rows)

    def test_the_contact_table_records_the_parameter_used(self, two_structures):
        rows = two_structures.contact_table()
        assert rows
        row = rows[0]
        for key in ("r0", "b", "parameter_fitted", "parameter_source"):
            assert key in row


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

class TestUtilities:

    def test_reciprocal_cell_inverts_the_direct_one(self, loaded):
        s, _ = loaded
        rec = utilities.reciprocal_cell(s)
        assert rec["volume*"] == pytest.approx(1.0 / s.cell.volume, rel=1e-9)

    def test_reciprocal_of_a_cubic_cell_is_one_over_a(self):
        from facet.core.structure import Cell, Site, Structure

        cell = Cell(4.0, 4.0, 4.0, 90, 90, 90, np.eye(3) * 4.0)
        s = Structure(name="cubic", cell=cell,
                      sites=[Site("A", "Na", [0, 0, 0])], atoms=[])
        rec = utilities.reciprocal_cell(s)
        assert rec["a*"] == pytest.approx(0.25)
        assert rec["alpha*"] == pytest.approx(90.0, abs=1e-9)

    def test_d_spacing_of_a_cubic_cell_matches_the_closed_form(self):
        from facet.core.structure import Cell, Site, Structure

        a = 5.0
        cell = Cell(a, a, a, 90, 90, 90, np.eye(3) * a)
        s = Structure(name="cubic", cell=cell,
                      sites=[Site("A", "Na", [0, 0, 0])], atoms=[])
        for h, k, l in ((1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 0)):
            want = a / math.sqrt(h * h + k * k + l * l)
            assert utilities.d_spacing(s, h, k, l) == pytest.approx(want, rel=1e-12)

    def test_two_theta_satisfies_bragg(self, loaded):
        s, _ = loaded
        wavelength = 1.5406
        d = utilities.d_spacing(s, 1, 1, 1)
        tt = utilities.two_theta(s, 1, 1, 1, wavelength)
        assert 2 * d * math.sin(math.radians(tt / 2)) == pytest.approx(
            wavelength, rel=1e-9)

    def test_reflection_list_is_sorted_and_bounded(self, loaded):
        s, _ = loaded
        rows = utilities.reflection_list(s, two_theta_max=60.0, max_index=4)
        assert rows
        assert all(r["two_theta"] <= 60.0 for r in rows)
        assert rows == sorted(rows, key=lambda r: -r["d"])

    def test_density_of_bismite_is_near_the_published_value(self, loaded):
        """alpha-Bi2O3 is about 9.4 g/cm3."""
        s, _ = loaded
        assert utilities.density(s)["density"] == pytest.approx(9.4, abs=0.2)

    def test_density_accounts_for_occupancy(self):
        from facet.core.structure import Cell, Site, Structure

        cell = Cell(5.0, 5.0, 5.0, 90, 90, 90, np.eye(3) * 5.0)
        full = Structure(name="f", cell=cell, atoms=[],
                         sites=[Site("A", "Bi", [0, 0, 0], occupancy=1.0,
                                     multiplicity=1)])
        half = Structure(name="h", cell=cell, atoms=[],
                         sites=[Site("A", "Bi", [0, 0, 0], occupancy=0.5,
                                     multiplicity=1)])
        assert utilities.density(half)["density"] == pytest.approx(
            utilities.density(full)["density"] / 2.0, rel=1e-9)

    def test_bond_angles_cover_every_pair(self, loaded):
        s, res = loaded
        site = next(r for r in res if r.element == "Bi")
        n = len(site.bonds)
        rows = utilities.bond_angles(site)
        assert len(rows) == n * (n - 1) // 2
        assert all(0.0 <= r.angle <= 180.0 for r in rows)
        assert rows == sorted(rows, key=lambda r: r.angle)

    def test_bond_angles_of_a_regular_octahedron_are_90_and_180(self):
        class Contact:
            def __init__(self, v, label):
                self.vector = np.array(v, float)
                self.label = label
                self.valence = 0.5

        class Result:
            label = "M"
            bonds = [Contact(v, str(i)) for i, v in enumerate(
                [(2, 0, 0), (-2, 0, 0), (0, 2, 0),
                 (0, -2, 0), (0, 0, 2), (0, 0, -2)])]

        angles = sorted(round(r.angle, 6) for r in utilities.bond_angles(Result()))
        assert angles.count(90.0) == 12
        assert angles.count(180.0) == 3

    def test_radial_shells_group_equal_distances(self, loaded):
        s, res = loaded
        site = next(r for r in res if r.element == "Bi")
        shells = utilities.radial_shells(s, site.site_index, rmax=4.0)
        assert shells
        assert shells == sorted(shells, key=lambda x: x["distance"])
        assert all(x["count"] >= 1 for x in shells)

    def test_global_instability_index_is_zero_for_perfect_agreement(self):
        class R:
            def __init__(self, d):
                self.valence_discrepancy = d
                self.label = "X"
        assert utilities.global_instability_index(
            [R(0.0), R(0.0)])["gii"] == pytest.approx(0.0)

    def test_global_instability_index_names_its_worst_site(self, loaded):
        s, res = loaded
        out = utilities.global_instability_index(res)
        assert out["n_sites"] == 2
        assert out["worst_site"] in {r.label for r in res}

    def test_valence_map_point_recovers_a_real_site(self, loaded):
        """A probe cation placed exactly on a bismuth site should sum to
        roughly that site's own bond-valence sum."""
        s, res = loaded
        site = next(r for r in res if r.element == "Bi")
        atom = s.atoms[s.atoms_of_site(site.site_index)[0]]
        got = utilities.valence_map_point(s, atom.cart, "Bi", 3)
        assert got == pytest.approx(site.bvs_listed, rel=0.05)

    def test_polyhedral_connectivity_classifies_by_shared_ligands(self, loaded):
        s, res = loaded
        rows = utilities.polyhedral_connectivity(s, res, 0.075)
        for row in rows:
            expected = {1: "corner", 2: "edge"}.get(row["shared_ligands"], "face")
            assert row["sharing"] == expected

    def test_composition_summary_balances_a_neutral_structure(self, loaded):
        s, _ = loaded
        out = utilities.composition_summary(s)
        assert out["counts"] == {"O": 12.0, "Bi": 8.0}
        assert out["balanced"]
        assert sum(out["atomic_percent"].values()) == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# exporters
# ---------------------------------------------------------------------------

class TestExporters:

    def test_csv_carries_the_provenance_as_comments(self, two_structures, tmp_path):
        path = exporters.write_csv(two_structures.site_table(),
                                   tmp_path / "sites.csv")
        text = path.read_text(encoding="utf-8")
        assert text.startswith("#")
        assert "Bond threshold" in text
        assert "FACET" in text

    def test_csv_is_readable_past_its_comments(self, two_structures, tmp_path):
        path = exporters.write_csv(two_structures.site_table(),
                                   tmp_path / "sites.csv")
        with open(path, encoding="utf-8") as handle:
            rows = list(csv.DictReader(
                line for line in handle if not line.startswith("#")))
        assert len(rows) == len(two_structures.site_table())
        assert "cn" in rows[0] and "phi" in rows[0]

    def test_an_empty_table_writes_an_empty_file(self, tmp_path):
        path = exporters.write_csv([], tmp_path / "none.csv")
        assert path.read_text(encoding="utf-8") == ""

    def test_the_written_cif_reads_back_with_the_same_cell_and_atoms(
            self, loaded, tmp_path):
        s, res = loaded
        path = exporters.write_cif(s, tmp_path / "rt.cif", res)
        back = cif.read(path)
        assert back.n_atoms == s.n_atoms
        for a, b in ((s.cell.a, back.cell.a), (s.cell.b, back.cell.b),
                     (s.cell.c, back.cell.c), (s.cell.beta, back.cell.beta)):
            assert a == pytest.approx(b, abs=1e-5)

    def test_the_written_cif_reproduces_the_analysis(self, loaded, tmp_path):
        """The round trip that matters: the numbers must survive it."""
        s, res = loaded
        path = exporters.write_cif(s, tmp_path / "rt.cif", res)
        again = coordination.analyse_structure(cif.read(path))
        before = sorted((r.cn_valence, round(r.bvs, 3))
                        for r in res if r.element == "Bi")
        after = sorted({(r.cn_valence, round(r.bvs, 3))
                        for r in again if r.element == "Bi"})
        assert before == after

    def test_poscar_has_the_lattice_and_the_right_atom_count(self, loaded,
                                                             tmp_path):
        s, _ = loaded
        path = exporters.write_poscar(s, tmp_path / "POSCAR")
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[1].strip() == "1.0"
        counts = [int(x) for x in lines[6].split()]
        assert sum(counts) == s.n_atoms
        assert lines[7].strip().lower().startswith("direct")

    def test_xyz_declares_the_lattice_and_scales_with_the_cell_range(
            self, loaded, tmp_path):
        s, _ = loaded
        one = exporters.write_xyz(s, tmp_path / "a.xyz")
        two = exporters.write_xyz(s, tmp_path / "b.xyz", cell_range=(2, 1, 1))
        assert int(one.read_text(encoding="utf-8").splitlines()[0]) == s.n_atoms
        assert int(two.read_text(encoding="utf-8").splitlines()[0]) == 2 * s.n_atoms
        assert 'Lattice="' in one.read_text(encoding="utf-8")

    def test_the_vesta_file_has_the_sections_vesta_requires(self, loaded,
                                                            tmp_path):
        s, _ = loaded
        path = exporters.write_vesta(s, tmp_path / "out.vesta")
        text = path.read_text(encoding="utf-8")
        assert text.startswith("#VESTA_FORMAT_VERSION")
        for section in ("CELLP", "STRUC", "SBOND", "ATOMT", "SITET"):
            assert f"\n{section}" in text or text.startswith(section)

    def test_the_vesta_bond_rule_matches_the_valence_threshold(self, loaded,
                                                               tmp_path):
        """VESTA has no valence cutoff, so the equivalent distance is written.
        For Bi-O at 0.075 v.u. that is 3.05 A."""
        s, _ = loaded
        path = exporters.write_vesta(s, tmp_path / "out.vesta", v_bond=0.075)
        line = next(ln for ln in path.read_text(encoding="utf-8").splitlines()
                    if " Bi " in ln and " O " in ln)
        assert any(abs(float(tok) - 3.048) < 0.02
                   for tok in line.split() if _isfloat(tok))

    def test_feff_input_puts_the_absorber_first_at_the_origin(self, loaded,
                                                             tmp_path):
        s, res = loaded
        site = next(r for r in res if r.element == "Bi")
        path = exporters.write_feff(s, site.site_index, tmp_path / "feff.inp")
        text = path.read_text(encoding="utf-8")
        assert "POTENTIALS" in text and "ATOMS" in text
        body = text.split("ATOMS")[1].splitlines()
        first = next(ln for ln in body if ln.strip()
                     and not ln.strip().startswith("*"))
        values = first.split()
        assert [float(v) for v in values[:3]] == [0.0, 0.0, 0.0]
        assert values[3] == "0"           # absorber is potential 0

    def test_feff_distances_match_the_coordination_analysis(self, loaded,
                                                            tmp_path):
        s, res = loaded
        site = next(r for r in res if r.element == "Bi")
        path = exporters.write_feff(s, site.site_index, tmp_path / "feff.inp")
        written = []
        for line in path.read_text(encoding="utf-8").split("ATOMS")[1].splitlines():
            parts = line.split()
            if len(parts) >= 6 and _isfloat(parts[-1]):
                written.append(float(parts[-1]))
        for contact in site.bonds:
            assert any(abs(contact.distance - w) < 1e-4 for w in written), \
                f"{contact.distance} not in the FEFF cluster"

    def test_xlsx_or_its_csv_fallback_is_produced(self, two_structures, tmp_path):
        out = exporters.write_xlsx({"sites": two_structures.site_table(),
                                    "contacts": two_structures.contact_table()},
                                   tmp_path / "book.xlsx")
        assert Path(out).exists() and Path(out).stat().st_size > 0

    def test_a_session_round_trips(self, two_structures, tmp_path):
        from facet.core import theme as T
        from facet.gl.camera import Camera

        camera = Camera()
        camera.frame([0, 0, 0], 5.0)
        path = exporters.save_session(two_structures, tmp_path / "s.json",
                                      theme=T.publication(), camera=camera)
        data = exporters.load_session(path)
        assert data["facet_session"] == 1
        assert len(data["entries"]) == 2
        assert data["v_bond"] == two_structures.v_bond
        assert data["theme"]["palette"] == "Greyscale"
        assert data["camera"]["distance"] == pytest.approx(camera.distance)

    def test_a_session_stores_paths_not_copies(self, two_structures, tmp_path):
        path = exporters.save_session(two_structures, tmp_path / "s.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert all(e["path"].endswith(".cif") for e in data["entries"])
        assert path.stat().st_size < 4000


def _isfloat(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False
