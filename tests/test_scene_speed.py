"""The scene's bonds, built from one sparse tree query (1.0.5), equal the
contacts the per-pair enumeration they replaced gives: every cation-anion
contact above the listing threshold, once, with its valence and distance.
And the panels that cost a pair search out to r_max wait for their tab.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from facet.core import bv, examples, readers
from facet.core.neighbors import NeighborFinder, search_radius_for
from facet.gl.scene import build_scene


def _reference(structure, params, v_list):
    """The contacts as the old loop found them: (lo, hi, image) -> (d, v)."""
    rmax = search_radius_for(structure, params, v_list)
    finder = NeighborFinder(structure, rmax=rmax)
    out = {}
    for i, j, d, image in finder.pairs_within(rmax):
        si = structure.atoms[i].site_index
        sj = structure.atoms[j].site_index
        ai, aj = structure.sites[si].is_anion, structure.sites[sj].is_anion
        if ai == aj:
            continue
        key = min((i, j, tuple(image)), (j, i, tuple(-x for x in image)))
        if key in out:
            continue
        cation = structure.atoms[j if ai else i]
        anion = structure.atoms[i if ai else j]
        site = structure.sites[cation.site_index]
        p = params.get(site.element, site.ox, anion.element)
        if p is None:
            continue
        v = float(p.valence(d))
        if v <= v_list:
            continue
        out[key] = (d, v)
    return out


@pytest.mark.parametrize("example", examples.available(), ids=lambda e: e.path.name)
def test_the_vectorised_bonds_equal_the_per_pair_enumeration(example):
    structure = readers.read(example.path)
    params = bv.DEFAULT
    ref = _reference(structure, params, bv.V_LIST_DEFAULT)
    scene = build_scene(structure, None, v_list=bv.V_LIST_DEFAULT)
    orth = structure.cell.orth
    cart = structure.cart_array()
    got = {}
    for k in range(scene.n_bonds):
        i, j = (int(x) for x in scene.bond_atoms[k])
        # the image from the drawn far end
        shift = np.asarray(scene.bond_b[k], float) - cart[j]
        image = tuple(int(round(x)) for x in np.linalg.solve(orth, shift))
        key = min((i, j, image), (j, i, tuple(-x for x in image)))
        assert key not in got, "a contact drawn twice"
        got[key] = (float(scene.bond_distance[k]), float(scene.bond_valence[k]))
    assert set(got) == set(ref)
    for key, (d, v) in ref.items():
        assert got[key][0] == pytest.approx(d, abs=1e-5)
        assert got[key][1] == pytest.approx(v, rel=1e-5)
    # the images that close a bond: each drawn once
    pos = np.asarray(scene.atom_position, float)
    from scipy.spatial import cKDTree
    assert not len(cKDTree(pos).query_pairs(0.05))


def test_the_estimator_needs_no_pymatgen():
    import sys
    from facet.core import bv as bv_mod
    bv_mod._okb_cache = None
    before = "pymatgen.analysis.bond_valence" in sys.modules
    r0 = bv_mod.estimate_r0("Si", "O")
    if not before:
        assert "pymatgen.analysis.bond_valence" not in sys.modules
    from facet.core.okb_table import OKB_PARAMS
    assert len(OKB_PARAMS) == 75 and OKB_PARAMS["O"]["r"] > 0
    # the bundled table is pymatgen's, value for value, where pymatgen is here
    pm = pytest.importorskip("pymatgen.analysis.bond_valence")
    assert set(OKB_PARAMS) == {str(k) for k in pm.BV_PARAMS}
    for k, v in pm.BV_PARAMS.items():
        assert OKB_PARAMS[str(k)]["r"] == pytest.approx(float(v["r"]))
        assert OKB_PARAMS[str(k)]["c"] == pytest.approx(float(v["c"]))
    a, b = pm.BV_PARAMS[pm.Element("Si")], pm.BV_PARAMS[pm.Element("O")]
    import math
    expected = (a["r"] + b["r"] - a["r"] * b["r"] * (math.sqrt(a["c"]) - math.sqrt(b["c"])) ** 2
                / (a["c"] * a["r"] + b["c"] * b["r"]))
    assert r0 == pytest.approx(expected, abs=1e-12)
