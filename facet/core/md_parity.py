"""A parity harness: another code's per-atom numbers for one MD frame laid
beside FACET's, descriptor by descriptor.

The other code writes a table (CSV or whitespace-separated, one header
line) with one row per atom: a column naming the atom (its 0- or 1-based
index in the frame) and one column per quantity. :func:`compare` reads
FACET's valence table for the same frame, pairs the rows by atom and
reports, for each quantity mapped, how many atoms were compared, the
largest and mean difference, and the fraction of atoms within a stated
tolerance. It states; it does not judge which code is right.

Quantities FACET can supply: ``CN`` (contacts above the bond threshold),
``CN_listed`` (above the listing threshold), ``BVS`` (bond-valence sum,
v.u.), ``BVV`` (length of the bond-valence vector), ``phi``, and
``plateau`` (the plateau width in decades). Everything else in the other
code's table is left as it is.

No Qt. ``py -3.11 -m facet.check --parity TABLE TRAJECTORY`` is the
command-line face.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from . import bulk, bv, md_model, md_readers

QUANTITIES = {
    "CN": lambda r: r.cn,
    "CN_listed": lambda r: r.cn_listed,
    "BVS": lambda r: r.bvs_vu,
    "BVV": lambda r: r.bvv_vu,
    "phi": lambda r: r.phi,
    "plateau": lambda r: r.plateau_decades,
}


def read_table(path: str | Path) -> tuple[list[str], list[list[str]]]:
    """Header and rows of a CSV / TSV / whitespace table; '#' lines skipped."""
    text = Path(path).read_text(encoding="utf-8-sig")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"{path}: no table rows")
    sample = lines[0]
    if "," in sample:
        rows = list(csv.reader(lines))
    elif "\t" in sample:
        rows = [ln.split("\t") for ln in lines]
    else:
        rows = [ln.split() for ln in lines]
    header = [h.strip() for h in rows[0]]
    body = [[c.strip() for c in r] for r in rows[1:] if any(c.strip() for c in r)]
    return header, body


def parse_mapping(text: str) -> dict[str, str]:
    """'cn=CN,sum=BVS' -> {'cn': 'CN', 'sum': 'BVS'} (their column -> FACET
    quantity). A bare name maps a column to the quantity of the same name."""
    out: dict[str, str] = {}
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            col, quantity = (s.strip() for s in part.split("=", 1))
        else:
            col = quantity = part
        if quantity not in QUANTITIES:
            raise ValueError(f"unknown quantity {quantity!r}; one of "
                             + ", ".join(QUANTITIES))
        out[col] = quantity
    return out


@dataclass
class ParityRow:
    column: str
    quantity: str
    n_compared: int
    n_theirs_missing: int
    max_abs_difference: float
    mean_difference: float          # theirs minus FACET
    fraction_within: float
    tolerance: float
    unit: str

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def facet_values(trajectory, k: int, v_bond: float = bv.V_BOND_DEFAULT,
                 v_list: float = bv.V_LIST_DEFAULT, params=None,
                 ox_overrides: Mapping[str, int] | None = None):
    """FACET's per-atom results for frame ``k``."""
    frame = trajectory.frame(int(k))
    ox = md_model.model_oxidation(frame.species, ox_overrides).per_atom(frame.elements)
    table, _ = bulk.analyse_frame(frame, ox, params or bv.DEFAULT,
                                  v_bond_vu=float(v_bond), v_list_vu=float(v_list))
    return bulk.at_threshold(table, float(v_bond)), frame


def compare(theirs: tuple[list[str], list[list[str]]], trajectory, k: int,
            mapping: Mapping[str, str], *, atom_column: str = "atom",
            one_based: bool = False, tolerance: float = 1e-3,
            v_bond: float = bv.V_BOND_DEFAULT, v_list: float = bv.V_LIST_DEFAULT,
            params=None, ox_overrides=None) -> list[ParityRow]:
    """Their table against FACET's frame ``k``, column by column."""
    header, body = theirs
    if atom_column not in header:
        raise ValueError(f"no column {atom_column!r} in the table; columns: "
                         + ", ".join(header))
    results, frame = facet_values(trajectory, k, v_bond, v_list, params, ox_overrides)
    n = int(frame.n_atoms)
    i_atom = header.index(atom_column)
    units = {"CN": "1", "CN_listed": "1", "BVS": "v.u.", "BVV": "v.u.",
             "phi": "1", "plateau": "decades"}
    out: list[ParityRow] = []
    for column, quantity in mapping.items():
        if column not in header:
            raise ValueError(f"no column {column!r} in the table")
        j = header.index(column)
        ours = np.asarray(QUANTITIES[quantity](results), float)
        theirs_v = np.full(n, np.nan)
        for row in body:
            try:
                index = int(float(row[i_atom])) - (1 if one_based else 0)
                value = float(row[j])
            except (ValueError, IndexError):
                continue
            if 0 <= index < n:
                theirs_v[index] = value
        both = np.isfinite(theirs_v) & np.isfinite(ours)
        d = theirs_v[both] - ours[both]
        out.append(ParityRow(
            column, quantity, int(both.sum()), int((~np.isfinite(theirs_v)).sum()),
            float(np.max(np.abs(d))) if d.size else math.nan,
            float(np.mean(d)) if d.size else math.nan,
            float(np.mean(np.abs(d) <= tolerance)) if d.size else math.nan,
            float(tolerance), units[quantity]))
    return out


def compare_files(table_path: str | Path, trajectory_path: str | Path, k: int,
                  mapping: str | Mapping[str, str], **kwargs) -> list[ParityRow]:
    theirs = read_table(table_path)
    trajectory = md_readers.read_trajectory(str(trajectory_path))
    if isinstance(mapping, str):
        mapping = parse_mapping(mapping)
    return compare(theirs, trajectory, k, mapping, **kwargs)


def format_rows(rows: Sequence[ParityRow]) -> str:
    head = (f"{'column':<16} {'FACET':<10} {'compared':>8} {'missing':>8} "
            f"{'max |diff|':>11} {'mean diff':>10} {'within tol':>10}  unit")
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r.column:<16.16} {r.quantity:<10} {r.n_compared:>8} "
                     f"{r.n_theirs_missing:>8} {r.max_abs_difference:>11.4g} "
                     f"{r.mean_difference:>+10.4g} {r.fraction_within:>10.1%}  {r.unit}")
    return "\n".join(lines)
