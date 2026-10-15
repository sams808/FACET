"""The self-check: the numbers FACET states for its bundled examples,
recomputed on this machine and laid beside the reference file.

``facet/data/examples/REFERENCE.json`` holds, for every site of every
bundled example at the default thresholds, the coordination number, the
bond-valence sum, phi, the plateau and the bond distances the tutorials
quote. :func:`check_examples` recomputes each of them here and reports the
difference against the stated tolerance, so a user can see on their own
machine that the engine reproduces the numbers the documentation states.
:func:`check_sums` adds an internal consistency check that needs no
reference: each site's bond-valence sum equals the sum of the valences of
the contacts counted as bonds, weighted by occupancy, to the precision the
tests use.

No Qt. ``py -3.11 -m facet.check`` is the command-line face.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from . import bv, cif, coordination, examples

REFERENCE = Path(__file__).resolve().parent.parent / "data" / "examples" / "REFERENCE.json"

# The precision each quantity is checked to. These are the tolerances of
# the permanent tests (VERIFICATION.md, "Stated tolerances"): an integer is
# exact; a value printed to three decimals must round the same way, hence
# half a unit in the last place plus the floating-point slack.
TOLERANCES = {
    "cn": 0.0,
    "bvs_vu": 5e-4,
    "phi": 5e-4,
    "plateau_decades": 5e-3,
    "plateau_from_vu": 5e-5,
    "plateau_to_vu": 5e-5,
    "bond_distances_ang": 5e-4,
}
SUM_TOLERANCE = 1e-12


@dataclass
class CheckRow:
    example: str
    site: str
    quantity: str
    reference: object
    computed: object
    difference: float
    tolerance: float
    agrees: bool

    def as_dict(self) -> dict:
        return asdict(self)


def load_reference(path: Path | None = None) -> dict:
    return json.loads(Path(path or REFERENCE).read_text(encoding="utf-8"))


def _analyse(path: Path, reference: dict) -> list:
    structure = cif.read(str(path))
    return coordination.analyse_structure(
        structure, bv.DEFAULT,
        v_bond=float(reference.get("v_bond_vu", bv.V_BOND_DEFAULT)),
        v_list=float(reference.get("v_list_vu", bv.V_LIST_DEFAULT)),
        cations_only=True)


def _computed(result, v_bond: float) -> dict:
    plateau = next((p for p in result.plateaus if p.contains(v_bond)), None)
    bonded = sorted(c.distance for c in result.contacts
                    if c.has_valence and c.valence > v_bond)
    return {
        "cn": int(result.cn_valence),
        "bvs_vu": float(result.bvs),
        "phi": float(result.phi),
        "plateau_decades": float(plateau.width_decades) if plateau else None,
        "plateau_from_vu": float(plateau.v_low) if plateau else None,
        "plateau_to_vu": float(plateau.v_high) if plateau else None,
        "bond_distances_ang": [float(d) for d in bonded],
    }


def _difference(reference, computed) -> float:
    if reference is None and computed is None:
        return 0.0
    if reference is None or computed is None:
        return math.inf
    if isinstance(reference, list):
        if len(reference) != len(computed):
            return math.inf
        if not reference:
            return 0.0
        return float(np.max(np.abs(np.asarray(reference, float)
                                   - np.asarray(computed, float))))
    return abs(float(reference) - float(computed))


def check_examples(reference: dict | None = None,
                   folder: Path | None = None) -> list[CheckRow]:
    """Every stated number of every bundled example, recomputed. An example
    whose file is missing gives one row saying so (agrees False)."""
    reference = reference or load_reference()
    v_bond = float(reference.get("v_bond_vu", bv.V_BOND_DEFAULT))
    where = Path(folder) if folder else examples.folder()
    rows: list[CheckRow] = []
    for name, record in reference.get("examples", {}).items():
        path = where / name
        if not path.is_file():
            rows.append(CheckRow(name, "", "file", "present", "missing",
                                 math.inf, 0.0, False))
            continue
        results = {r.label: r for r in _analyse(path, reference)}
        for label, stated in record.get("sites", {}).items():
            result = results.get(label)
            if result is None:
                rows.append(CheckRow(name, label, "site", "present", "missing",
                                     math.inf, 0.0, False))
                continue
            got = _computed(result, v_bond)
            for quantity, tol in TOLERANCES.items():
                if quantity not in stated:
                    continue
                ref, comp = stated[quantity], got[quantity]
                d = _difference(ref, comp)
                shown = ([round(x, 4) for x in comp] if isinstance(comp, list)
                         else (round(comp, 6) if isinstance(comp, float) else comp))
                rows.append(CheckRow(name, label, quantity, ref, shown, d, tol,
                                     bool(d <= tol)))
    return rows


def check_sums(reference: dict | None = None,
               folder: Path | None = None) -> list[CheckRow]:
    """Internal consistency on every example: the bond-valence sum of each
    site equals the occupancy-weighted sum of the valences of the contacts
    above the bond threshold, to SUM_TOLERANCE; and the plateau the site
    reports contains the threshold."""
    reference = reference or load_reference()
    v_bond = float(reference.get("v_bond_vu", bv.V_BOND_DEFAULT))
    where = Path(folder) if folder else examples.folder()
    rows: list[CheckRow] = []
    for name in reference.get("examples", {}):
        path = where / name
        if not path.is_file():
            continue
        for r in _analyse(path, reference):
            total = sum(c.valence * c.occupancy for c in r.contacts
                        if c.has_valence and c.valence > v_bond)
            d = abs(total - float(r.bvs))
            rows.append(CheckRow(name, r.label, "sum of bond valences",
                                 round(float(r.bvs), 9), round(total, 9), d,
                                 SUM_TOLERANCE, d <= SUM_TOLERANCE))
            plateau = next((p for p in r.plateaus if p.contains(v_bond)), None)
            inside = plateau is not None and plateau.v_low <= v_bond <= plateau.v_high
            rows.append(CheckRow(name, r.label, "plateau holds v_bond",
                                 "yes", "yes" if inside else "no",
                                 0.0 if inside else math.inf, 0.0, inside))
    return rows


def summary(rows: Sequence[CheckRow]) -> str:
    agreed = sum(1 for r in rows if r.agrees)
    return f"{agreed} of {len(rows)} checks within tolerance"


def format_rows(rows: Sequence[CheckRow], only_failures: bool = False) -> str:
    """A plain-text table of the rows."""
    lines = []
    head = f"{'example':<40} {'site':<6} {'quantity':<22} {'reference':<28} " \
           f"{'computed':<28} {'|diff|':>10} {'tol':>8}  agrees"
    lines.append(head)
    lines.append("-" * len(head))
    for r in rows:
        if only_failures and r.agrees:
            continue
        diff = "inf" if math.isinf(r.difference) else f"{r.difference:.2e}"
        lines.append(f"{r.example:<40.40} {r.site:<6.6} {r.quantity:<22.22} "
                     f"{str(r.reference):<28.28} {str(r.computed):<28.28} "
                     f"{diff:>10} {r.tolerance:>8.0e}  {'yes' if r.agrees else 'NO'}")
    return "\n".join(lines)
