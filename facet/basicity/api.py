"""The optical-basicity calculations of FACET as plain Python calls.

Nothing here opens a window or imports Qt. Every function returns an object
that carries the numbers, the scale and threshold that produced them and the
notes the calculation left, and can write itself as text, CSV or a mapping.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

from ..core import basicity as B
from ..core import basicity_composition as C
from ..core import basicity_field as F
from ..core import bv

DEFAULT_SCALE = B.DEFAULT_SCALE
SCALE_KEYS = B.SCALE_KEYS
PROBE_IONS = tuple(B.PROBE_IONS)
CRYSTAL_SUFFIXES = (".cif", ".mcif", ".vesta", ".res", ".ins", ".pdb",
                    ".cmtx", ".cmdf", ".vasp", ".poscar")

oxide_polarisability = B.oxide_polarisability
basicity_from_polarisability = B.basicity_from_polarisability
probe_band = B.probe_band
probe_wavelength = B.probe_wavelength
basicity_from_band = B.basicity_from_band
CompositionError = C.CompositionError


# --- small helpers ------------------------------------------------------------------

def _num(value, decimals: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "-" if not math.isfinite(value) else f"{value:.{decimals}f}"
    return str(value)


def text_table(headers: Sequence[str], rows: Sequence[Sequence]) -> str:
    """Rows as aligned columns of plain text."""
    cells = [[_num(v) for v in row] for row in rows]
    widths = [max([len(h)] + [len(r[k]) for r in cells])
              for k, h in enumerate(headers)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths)),
             "  ".join("-" * w for w in widths)]
    lines += ["  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip()
              for row in cells]
    return "\n".join(lines)


def _write_csv(path, headers, rows) -> Path:
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(["" if isinstance(v, float) and not math.isfinite(v)
                             else v for v in row])
    return path


def _plain(value):
    """A value a JSON writer accepts."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _derived(value: float) -> dict:
    """What a basicity implies through the published relations."""
    out = {"alpha(O2-) / Å3": B.oxide_polarisability(value)}
    for ion in B.PROBE_IONS:
        out[f"{ion} band / cm-1"] = B.probe_band(value, ion)
    return out


# --- scales and single oxides -------------------------------------------------------

def scales() -> list[dict]:
    """The scales of basicity moderating parameters: key, name, source and
    the species each lists itself."""
    return [{"key": s.key, "name": s.name, "source": s.source,
             "listed": s.measured_species} for s in B.SCALES.values()]


def moderator(element: str, ox: int | None = None, scale=DEFAULT_SCALE) -> dict:
    """gamma of one cation on a scale, with where it came from."""
    m = B.scale_named(scale).moderator(str(element), ox)
    if m is None:
        raise ValueError(f"no basicity moderating parameter for {element} on "
                         f"the {B.scale_named(scale).key} scale")
    return {"species": m.species, "gamma": m.gamma, "Lambda oxide": m.basicity,
            "Lambda fluoride": m.basicity / B.ANION_FACTORS["F"],
            "from": m.origin, "coordination": m.coordination,
            "note": m.note, "source": m.source}


def oxide(element: str, ox: int | None = None, scale=DEFAULT_SCALE,
          anion: str = "O") -> float:
    """Lambda of the single oxide (``anion='F'``: fluoride) of a cation."""
    return B.oxide_basicity(str(element), ox, scale, anion)


# --- a written composition ------------------------------------------------------------

@dataclass
class CompositionResult:
    """Lambda of a composition written as on a batch sheet."""

    result: C.CompositionBasicity

    @property
    def value(self) -> float:
        return self.result.value

    @property
    def scale(self) -> str:
        return self.result.scale

    @property
    def notes(self) -> list[str]:
        return list(self.result.notes)

    @property
    def polarisability(self) -> float:
        return B.oxide_polarisability(self.value)

    def band(self, ion: str = "Pb2+") -> float:
        return B.probe_band(self.value, ion)

    HEADERS = ("component", "amount", "mole fraction", "X (anion charge)",
               "Lambda")

    def rows(self) -> list[list]:
        return [[c.formula, c.amount, c.fraction, c.share, c.basicity]
                for c in self.result.components]

    def to_dict(self) -> dict:
        r = self.result
        return _plain({
            "composition": r.text, "unit": r.unit, "scale": r.scale,
            "Lambda": r.value, **_derived(r.value),
            "components": [dict(zip(self.HEADERS, row)) for row in self.rows()],
            "moderators": [{"species": m.species, "gamma": m.gamma,
                            "from": m.origin} for m in r.moderators],
            "notes": r.notes})

    def to_csv(self, path) -> Path:
        return _write_csv(path, self.HEADERS, self.rows())

    def table(self) -> str:
        r = self.result
        lines = [f"{r.text}   ({r.unit} amounts, {r.scale} scale)",
                 f"Lambda = {_num(r.value, 4)}", ""]
        lines.append(text_table(self.HEADERS, self.rows()))
        lines.append("")
        lines.append(text_table(
            ("species", "gamma", "Lambda oxide", "gamma from"),
            [[m.species, m.gamma, m.basicity, m.origin] for m in r.moderators]))
        lines.append("")
        lines.append(_derived_text(r.value, all(
            ox > 0 or e == "O" for c in r.components for (e, ox) in c.species)))
        lines += [f"note: {n}" for n in r.notes]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.table()


def _derived_text(value: float, oxide_only: bool = True) -> str:
    if not math.isfinite(value):
        return ""
    bands = ", ".join(f"{ion} {B.probe_band(value, ion):.0f} cm-1 "
                      f"({B.probe_wavelength(value, ion):.0f} nm)"
                      for ion in B.PROBE_IONS)
    alpha = (f"alpha(O2-) = {_num(B.oxide_polarisability(value))} Å3 "
             "(Duffy 1989, Lambda = 1.67 (1 - 1/alpha))\n") if oxide_only else ""
    return (alpha + f"probe-ion 1S0-3P1 band at this Lambda: {bands} "
            "(Duffy and Ingram 1976)")


def composition(text, scale=DEFAULT_SCALE, *, unit: str = "mol") -> CompositionResult:
    """Lambda of ``'30Na2O-70SiO2'``, ``'Na2O 25, CaO 10, SiO2 65'``,
    ``'20CaF2-80SiO2'`` or a mapping of formula to amount; ``unit='wt'``
    for weights."""
    return CompositionResult(C.composition_basicity(text, scale, unit=unit))


def series(start, end, scale=DEFAULT_SCALE, *, steps: int = 21,
           unit: str = "mol") -> tuple[np.ndarray, np.ndarray]:
    """Lambda along the join of two compositions: (x, Lambda), x the mole
    fraction of ``end``."""
    x, values = C.mixing_series(start, end, scale, steps=steps, unit=unit)
    return np.array(x), np.array(values)


# --- the field a probe senses ---------------------------------------------------------

@dataclass
class FieldResult:
    """lambda as a probe cation would sense it, on a grid over the cell."""

    field: F.BasicityField
    symbols: np.ndarray                  # the atoms of the cell
    frac: np.ndarray                     # their fractional coordinates
    source: str = ""

    @property
    def values(self) -> np.ndarray:
        return np.asarray(self.field.grid.values)

    @property
    def range(self) -> tuple[float, float]:
        return self.field.range

    @property
    def notes(self) -> list[str]:
        return list(self.field.grid.notes)

    @property
    def vectors(self) -> np.ndarray:
        """The cell's three vectors as rows, angstrom."""
        return np.asarray(self.field.grid.cell.to_cartesian(np.eye(3)), float)

    def smoothed(self, sigma_ang: float) -> "FieldResult":
        """The same field averaged over a Gaussian of this width."""
        return FieldResult(F.smoothed(self.field, float(sigma_ang)),
                           self.symbols, self.frac, self.source)

    def slice(self, axis="c", at: float = 0.5, thickness_ang: float = 4.0) -> dict:
        """The plane of the grid nearest fraction ``at`` along a cell axis
        (``'a'``, ``'b'``, ``'c'`` or 0, 1, 2), with the atoms within
        ``thickness_ang`` of it. The mapping holds ``values`` (n_u + 1,
        n_v + 1: the grid's plane with its first row and column repeated at
        the far edge, the field being periodic), the corner coordinates
        ``x`` and ``y`` of those cells in angstrom in the plane, the
        ``outline`` of the cell, and the atoms' ``atom_x``, ``atom_y``,
        ``atom_symbols`` and ``atom_distance``."""
        k = {"a": 0, "b": 1, "c": 2, "x": 0, "y": 1, "z": 2}.get(
            str(axis).lower(), axis)
        if k not in (0, 1, 2):
            raise ValueError(f"axis {axis!r}: a, b or c")
        values = self.values
        n = values.shape[k]
        index = int(round(float(at) * n)) % n
        plane = np.take(values, index, axis=k)
        iu, iv = [i for i in (0, 1, 2) if i != k]
        vectors = self.vectors
        u, v = vectors[iu], vectors[iv]
        e1 = u / np.linalg.norm(u)
        rest = v - (v @ e1) * e1
        e2 = rest / np.linalg.norm(rest)
        nu, nv = plane.shape
        plane = np.pad(plane, ((0, 1), (0, 1)), mode="wrap")
        fu = (np.arange(nu + 2) - 0.5) / nu
        fv = (np.arange(nv + 2) - 0.5) / nv
        gu, gv = np.meshgrid(fu, fv, indexing="ij")
        x = gu * (u @ e1) + gv * (v @ e1)
        y = gv * (v @ e2)
        normal = np.cross(u, v)
        width = abs(float(vectors[k] @ normal)) / float(np.linalg.norm(normal))
        frac = np.asarray(self.frac, float).reshape(-1, 3)
        level = index / n
        off = ((frac[:, k] - level + 0.5) % 1.0) - 0.5
        distance = off * width
        near = np.abs(distance) <= float(thickness_ang)
        au, av = frac[near, iu] % 1.0, frac[near, iv] % 1.0
        corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], float)
        outline = np.column_stack([
            corners[:, 0] * (u @ e1) + corners[:, 1] * (v @ e1),
            corners[:, 1] * (v @ e2)])
        return {"values": plane, "x": x, "y": y, "axis": "abc"[k],
                "outline": outline,
                "at": level, "axes": ("abc"[iu], "abc"[iv]),
                "atom_x": au * (u @ e1) + av * (v @ e1),
                "atom_y": av * (v @ e2),
                "atom_symbols": np.asarray(self.symbols)[near],
                "atom_distance": distance[near],
                "label": self.field.label, "range": self.range}

    def to_cube(self, path, *, fill: float = 0.0) -> Path:
        """Write the field as a Gaussian cube file (VESTA, VMD, OVITO)."""
        cart = np.asarray(self.frac, float).reshape(-1, 3) @ self.vectors
        F.write_cube(path, self.field.grid, self.symbols, cart, fill=fill,
                     title=f"{self.field.label} {self.source}")
        return Path(path)

    def to_dict(self) -> dict:
        lo, hi = self.range
        return _plain({"probe": self.field.probe, "probe ox": self.field.probe_ox,
                       "scale": self.field.scale, "shape": self.values.shape,
                       "low": lo, "high": hi,
                       "fraction defined": self.field.fraction_defined,
                       "notes": self.notes})


def _probe_ox(probe: str, ox, known: Mapping[str, int]) -> int:
    if ox is not None:
        return int(ox)
    if probe in known and known[probe] > 0:
        return int(known[probe])
    from ..core import elements

    usual = elements.COMMON_OX.get(probe)
    if usual is None or usual <= 0:
        raise ValueError(f"no usual oxidation state for a {probe} probe; "
                         "give ox")
    return int(usual)


# --- a crystal structure ----------------------------------------------------------------

@dataclass
class StructureResult:
    """Every anion site's lambda, every cation site's surroundings and the
    composition's Lambda for one structure file."""

    structure: object
    results: list
    basicity: B.StructureBasicity
    formers: tuple[str, ...] = ()
    source: str = ""

    @property
    def scale(self) -> str:
        return self.basicity.scale

    @property
    def v_bond(self) -> float:
        return self.basicity.v_bond

    @property
    def value(self) -> float:
        """The mean lambda of the anions, from the bond valences."""
        return self.basicity.mean

    @property
    def bulk(self) -> float:
        """Lambda from the composition alone."""
        return self.basicity.bulk.value

    @property
    def difference(self) -> float:
        return self.basicity.difference

    @property
    def notes(self) -> list[str]:
        return list(self.basicity.bulk.notes) + list(self.basicity.notes)

    ANION_HEADERS = ("site", "element", "atoms", "CN", "valence sum",
                     "lambda", "lambda listed", "bonded to", "notes")
    CATION_HEADERS = ("site", "element", "CN", "lambda of its anions",
                      "host only", "anions")

    def _atoms(self, index: int) -> float:
        site = self.structure.sites[index]
        return float(site.occupancy) * (site.multiplicity or 1)

    def anions(self) -> list[list]:
        by_index = {r.site_index: r for r in self.results}
        rows = []
        for index, m in self.basicity.anions.items():
            counts: dict[str, int] = {}
            for s in m.shares:
                if s.bonded:
                    counts[s.element] = counts.get(s.element, 0) + 1
            rows.append([m.label, m.element, self._atoms(index),
                         by_index[index].cn_valence if index in by_index else None,
                         m.sum_bonded, m.value, m.value_listed,
                         B.linkage_pattern(m.element, counts, self.formers),
                         "; ".join(m.notes)])
        return rows

    def cations(self) -> list[list]:
        by_index = {r.site_index: r for r in self.results}
        return [[e.label, e.element,
                 by_index[index].cn_valence if index in by_index else None,
                 e.value, e.value_host,
                 "; ".join(f"{label} {value:.3f}" for label, _, value, _ in e.anions)]
                for index, e in self.basicity.environments.items()]

    def site(self, label: str):
        """The MicroBasicity (an anion) or EnvironmentBasicity (a cation)
        of the site with this label, shares and notes included."""
        for group in (self.basicity.anions, self.basicity.environments):
            for found in group.values():
                if found.label == label:
                    return found
        raise KeyError(f"no site labelled {label!r}")

    def linkages(self, formers: Sequence[str] | None = None) -> list[B.Linkage]:
        return B.structure_linkages(
            self.structure, self.basicity,
            self.formers if formers is None else tuple(formers))

    def values(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(lambda, weight in atoms per cell, element) of the anion sites
        that have a lambda."""
        rows = [(m.value, self._atoms(i), m.element)
                for i, m in self.basicity.anions.items()
                if m.defined and math.isfinite(m.value)]
        if not rows:
            return np.zeros(0), np.zeros(0), np.zeros(0, dtype=str)
        lam, w, el = zip(*rows)
        return np.array(lam), np.array(w), np.array(el)

    def field(self, probe: str, ox: int | None = None, *,
              resolution: float = 0.25, floor: float = F.VALENCE_FLOOR_DEFAULT,
              smooth: float = 0.0) -> FieldResult:
        """lambda as a ``probe`` cation would sense it over the cell."""
        known = {s.element: s.ox for s in self.structure.sites if s.ox is not None}
        found = F.basicity_grid(self.structure, self.results, self.scale,
                                probe=str(probe),
                                probe_ox=_probe_ox(str(probe), ox, known),
                                resolution=float(resolution),
                                floor_vu=float(floor), v_bond=self.v_bond)
        atoms = [a for a in self.structure.atoms]
        out = FieldResult(found, np.array([a.element for a in atoms]),
                          np.array([a.frac for a in atoms], float).reshape(-1, 3),
                          self.source)
        return out.smoothed(smooth) if smooth and smooth > 0 else out

    def to_dict(self) -> dict:
        return _plain({
            "source": self.source, "scale": self.scale, "v_bond": self.v_bond,
            "Lambda composition": self.bulk, "mean lambda": self.value,
            "mean lambda listed": self.basicity.mean_listed,
            "difference": self.difference, **_derived(self.bulk),
            "anions": [dict(zip(self.ANION_HEADERS, r)) for r in self.anions()],
            "cations": [dict(zip(self.CATION_HEADERS, r)) for r in self.cations()],
            "linkages": [vars(link) for link in self.linkages()],
            "notes": self.notes})

    def to_csv(self, path) -> Path:
        """The anion sites, one row each."""
        return _write_csv(path, self.ANION_HEADERS, self.anions())

    def table(self) -> str:
        lines = [f"{self.source}   ({self.scale} scale, bond threshold "
                 f"{self.v_bond:g} v.u.)",
                 f"Lambda from the composition       {_num(self.bulk, 4)}",
                 f"mean lambda of the anions (bonds) {_num(self.value, 4)}"
                 f"   ({_num(self.basicity.mean_listed, 4)} to the listing cut)",
                 "difference, bonds - composition   "
                 + ("-" if not math.isfinite(self.difference)
                    else f"{self.difference:+.4f}"), "",
                 text_table(self.ANION_HEADERS, self.anions()), "",
                 text_table(self.CATION_HEADERS, self.cations()), "",
                 linkage_table(self.linkages()), "",
                 _derived_text(self.bulk, all(
                     m.element == "O" for m in self.basicity.anions.values()))]
        lines += [f"note: {n}" for n in self.notes]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.table()


LINKAGE_HEADERS = ("anion", "bonded to", "atoms", "mean lambda", "sd", "low",
                   "high")


def _count(atoms: float):
    return int(round(atoms)) if abs(atoms - round(atoms)) < 1e-9 else atoms


def linkage_rows(links: Sequence[B.Linkage]) -> list[list]:
    return [[k.anion, k.pattern, _count(k.atoms), k.mean, k.sd, k.low, k.high]
            for k in links]


def linkage_table(links: Sequence[B.Linkage], top: int | None = 15) -> str:
    """The linkages as text, the most populated first; ``top`` rows (None:
    all of them), with the rest counted in a last line."""
    links = sorted(links, key=lambda k: -k.atoms)
    shown = links if top is None else links[:int(top)]
    text = text_table(LINKAGE_HEADERS, linkage_rows(shown))
    rest = links[len(shown):]
    if rest:
        text += (f"\n... and {len(rest)} more group(s) holding "
                 f"{_count(sum(k.atoms for k in rest))} atoms "
                 "(linkages() gives all of them)")
    return text


def structure(source, scale=DEFAULT_SCALE, *,
              v_bond: float = bv.V_BOND_DEFAULT,
              v_list: float = bv.V_LIST_DEFAULT,
              formers: Sequence[str] = ()) -> StructureResult:
    """The optical basicity of a crystal structure: a file FACET reads (CIF,
    VESTA, POSCAR, SHELX, PDB, CrystalMaker) or a ``Structure``."""
    from ..core import coordination, readers

    if isinstance(source, (str, Path)):
        loaded, name = readers.read(str(source)), Path(source).name
    else:
        loaded, name = source, getattr(source, "name", "") or "structure"
    results = coordination.analyse_structure(
        loaded, bv.DEFAULT, v_bond=float(v_bond), v_list=float(v_list),
        cations_only=False)
    if not results:
        raise ValueError(f"{name}: no site to analyse")
    found = B.structure_basicity(loaded, results, scale, v_bond=float(v_bond))
    return StructureResult(loaded, results, found,
                           tuple(str(f) for f in formers), name)


# --- a frame of an MD model ---------------------------------------------------------------

@dataclass
class FrameResult:
    """lambda of every anion of one frame of an MD model."""

    frame: object
    valence_table: object
    ox_atom: np.ndarray
    basicity: B.FrameBasicity
    formers: tuple[str, ...] = ()
    source: str = ""
    index: int = 0
    notes_read: tuple[str, ...] = field(default_factory=tuple)

    @property
    def scale(self) -> str:
        return self.basicity.scale

    @property
    def v_bond(self) -> float:
        return self.basicity.v_bond

    @property
    def values(self) -> np.ndarray:
        """lambda per atom, NaN for a cation."""
        return self.basicity.value

    @property
    def environment(self) -> np.ndarray:
        """Per cation, the mean lambda of the anions it is bonded to."""
        return self.basicity.environment

    @property
    def elements(self) -> np.ndarray:
        return np.asarray(self.frame.elements).astype(str)

    @property
    def positions(self) -> np.ndarray:
        """Cartesian positions in the box, angstrom."""
        from ..core import bulk

        return bulk._lattice(np.asarray(self.frame.frac, float),
                             np.asarray(self.frame.box_ang, np.float64))

    @property
    def classes(self) -> np.ndarray:
        """Bonded formers per anion (-1 for a cation or with none named)."""
        return self.basicity.classes

    @property
    def bulk(self) -> float:
        return self.basicity.bulk.value

    @property
    def value(self) -> float:
        """The mean lambda over the anions that have one."""
        v = self.values
        return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")

    @property
    def notes(self) -> list[str]:
        return list(self.basicity.bulk.notes) + list(self.basicity.notes)

    def mean(self, element: str | None = None) -> float:
        v = self.values
        mask = np.isfinite(v)
        if element is not None:
            mask &= self.elements == str(element)
        return float(v[mask].mean()) if mask.any() else float("nan")

    CLASS_HEADERS = ("anion", "class", "atoms", "mean lambda", "sd", "low",
                     "high")

    def by_class(self) -> list[list]:
        """Mean lambda of the free, non-bridging, bridging and tricluster
        anions (needs ``formers``)."""
        rows = []
        v, cls, el = self.values, self.classes, self.elements
        for symbol in sorted(set(el[np.isfinite(v)].tolist())):
            for count, name in B.CLASS_NAMES.items():
                chosen = (cls >= 3) if count == 3 else (cls == count)
                mask = chosen & (el == symbol) & np.isfinite(v)
                if mask.any():
                    x = v[mask]
                    rows.append([symbol, name, int(mask.sum()), float(x.mean()),
                                 float(x.std()), float(x.min()), float(x.max())])
        return rows

    def linkages(self, formers: Sequence[str] | None = None) -> list[B.Linkage]:
        return B.frame_linkages(
            self.valence_table, self.basicity,
            self.formers if formers is None else tuple(formers))

    def correlation(self, r_max: float = 10.0, bin: float = 0.25
                    ) -> F.SpatialCorrelation:
        """How lambda is correlated between anions a distance r apart."""
        return F.spatial_correlation(self.frame, self.values,
                                     r_max_ang=float(r_max), bin_ang=float(bin))

    def band(self, element: str, ion: str | None = None) -> np.ndarray:
        """The 1S0-3P1 band (cm-1) each atom of a probe element (Tl, Pb,
        Bi) would show in the basicity of its own anions; NaN elsewhere."""
        element = str(element)
        if ion is None:
            ion = next((k for k in B.PROBE_IONS if k.startswith(element)), None)
        if ion is None:
            raise ValueError(f"{element} is not a probe ion; the probe ions "
                             "are " + ", ".join(B.PROBE_IONS))
        free, shift = B.PROBE_IONS[ion]
        return np.where(self.elements == element,
                        free - shift * self.environment, np.nan)

    def field(self, probe: str, ox: int | None = None, *, grid: float = 1.0,
              r_cut: float = 6.0, floor: float = F.VALENCE_FLOOR_DEFAULT,
              smooth: float = 0.0) -> FieldResult:
        """lambda as a ``probe`` cation would sense it over the box."""
        known = {str(e): int(o) for e, o in zip(self.elements, self.ox_atom)}
        found = F.basicity_landscape(
            self.frame, self.ox_atom, self.values, probe=str(probe),
            probe_ox=_probe_ox(str(probe), ox, known),
            grid_spacing_ang=float(grid), r_cut_ang=float(r_cut),
            floor_vu=float(floor), scale=self.scale)
        out = FieldResult(found, self.elements,
                          np.asarray(self.frame.frac, float), self.source)
        return out.smoothed(smooth) if smooth and smooth > 0 else out

    ATOM_HEADERS = ("atom", "element", "x", "y", "z", "lambda",
                    "lambda listed", "bonded formers", "lambda of its anions")

    def atoms(self) -> list[list]:
        pos = self.positions
        b = self.basicity
        return [[k + 1, str(e), float(p[0]), float(p[1]), float(p[2]),
                 float(b.value[k]), float(b.value_listed[k]),
                 int(b.classes[k]) if b.classes[k] >= 0 else None,
                 float(b.environment[k])]
                for k, (e, p) in enumerate(zip(self.elements, pos))]

    def to_csv(self, path) -> Path:
        """Every atom, one row each."""
        return _write_csv(path, self.ATOM_HEADERS, self.atoms())

    def to_dict(self) -> dict:
        return _plain({
            "source": self.source, "frame": self.index, "scale": self.scale,
            "v_bond": self.v_bond, "atoms": int(self.frame.n_atoms),
            "Lambda composition": self.bulk, "mean lambda": self.value,
            **_derived(self.bulk),
            "by element": {e: self.mean(e) for e in
                           sorted(set(self.elements[np.isfinite(self.values)]))},
            "by class": [dict(zip(self.CLASS_HEADERS, r)) for r in self.by_class()],
            "linkages": [vars(link) for link in self.linkages()],
            "notes": self.notes})

    def table(self) -> str:
        lines = [f"{self.source}, frame {self.index}: {int(self.frame.n_atoms)} "
                 f"atoms   ({self.scale} scale, bond threshold "
                 f"{self.v_bond:g} v.u.)",
                 f"Lambda from the composition  {_num(self.bulk, 4)}",
                 f"mean lambda of the anions    {_num(self.value, 4)}", ""]
        classes = self.by_class()
        if classes:
            lines += [text_table(self.CLASS_HEADERS, classes), ""]
        only_oxide = set(self.elements[np.asarray(self.ox_atom) < 0]) <= {"O"}
        lines += [linkage_table(self.linkages()), "",
                  _derived_text(self.bulk, only_oxide)]
        lines += [f"note: {n}" for n in self.notes]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.table()


def _frame_result(fr, scale, v_bond, v_list, formers, ox, source, index,
                  notes=()) -> FrameResult:
    from ..core import bulk, md_model

    states = md_model.model_oxidation(fr.species, dict(ox or {}))
    ox_atom = states.per_atom(fr.elements)
    table, _ = bulk.analyse_frame(fr, ox_atom, bv.DEFAULT,
                                  v_bond_vu=float(v_bond), v_list_vu=float(v_list))
    formers = tuple(str(f) for f in (formers or ()))
    found = B.frame_basicity(table, scale, v_bond=float(v_bond),
                             formers=formers or None)
    return FrameResult(fr, table, np.asarray(ox_atom), found, formers, source,
                       int(index), tuple(notes))


def frame(source, scale=DEFAULT_SCALE, *, index: int = -1,
          v_bond: float = bv.V_BOND_DEFAULT, v_list: float = bv.V_LIST_DEFAULT,
          formers: Sequence[str] | None = None,
          ox: Mapping[str, int] | None = None, **read_options) -> FrameResult:
    """The optical basicity of one frame of an MD model: a file FACET reads
    (LAMMPS data or dump, extended XYZ, DL_POLY, VASP, ...) or a ``Frame``.

    ``index`` counts the frames from 0, negative from the end. ``formers``
    names the network formers (for the free / NBO / BO split), ``ox`` the
    oxidation states that differ from the usual ones, and any other keyword
    goes to the reader (``type_map={1: 'Si', 2: 'O'}``)."""
    if not isinstance(source, (str, Path, list, tuple)):
        return _frame_result(source, scale, v_bond, v_list, formers, ox,
                             "frame", 0)
    from ..core import md_readers

    with md_readers.read_trajectory(source, **read_options) as trajectory:
        n = int(trajectory.n_frames)
        k = int(index) % n if -n <= int(index) < n else None
        if k is None:
            raise IndexError(f"frame {index} of a model with {n} frame(s)")
        fr = trajectory.frame(k)
    name = Path(source).name if isinstance(source, (str, Path)) else \
        Path(source[0]).name
    return _frame_result(fr, scale, v_bond, v_list, formers, ox, name, k)


def frames(source, scale=DEFAULT_SCALE, *, start: int = 0,
           stop: int | None = None, step: int = 1,
           v_bond: float = bv.V_BOND_DEFAULT, v_list: float = bv.V_LIST_DEFAULT,
           formers: Sequence[str] | None = None,
           ox: Mapping[str, int] | None = None, **read_options
           ) -> Iterator[FrameResult]:
    """One :class:`FrameResult` per frame of a trajectory, in order."""
    from ..core import md_readers

    name = Path(source).name if isinstance(source, (str, Path)) else "model"
    with md_readers.read_trajectory(source, **read_options) as trajectory:
        n = int(trajectory.n_frames)
        for k in range(*slice(start, stop, step).indices(n)):
            yield _frame_result(trajectory.frame(k), scale, v_bond, v_list,
                                formers, ox, name, k)


def is_md_file(path) -> bool:
    """Whether FACET reads this file as an MD model rather than a crystal
    structure: not a crystal format by its name, and an MD format by its
    content."""
    p = Path(path)
    if p.suffix.lower() in CRYSTAL_SUFFIXES or p.name.upper() in ("POSCAR",
                                                                 "CONTCAR"):
        return False
    from ..core import md_readers

    try:
        return md_readers.sniff_md(p) is not None
    except Exception:
        return False


def load(source, scale=DEFAULT_SCALE, **options):
    """:func:`structure` or :func:`frame`, by what the file is."""
    if is_md_file(source):
        return frame(source, scale, **options)
    allowed = {k: v for k, v in options.items()
               if k in ("v_bond", "v_list", "formers")}
    return structure(source, scale, **allowed)
