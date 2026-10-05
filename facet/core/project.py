"""A project: several structures held together.

FACET began as a one-structure viewer. Most real questions are comparative --
four polymorphs of the same composition, a dozen phosphates, the same phase
refined by two groups -- and answering them means having the structures open at
once, under one threshold, and being able to overlay or step between them.

A :class:`Project` owns the structures and the settings that apply across all
of them. Per-structure state (visibility, offset, which site is selected) lives
on the :class:`Entry` beside each one.

Analysis is cached per entry and invalidated when the parameters that produced
it change. Moving the *bond* threshold does not invalidate anything: contacts
are found down to the tabulation threshold and the bond threshold only decides
where the line falls among them, which is the whole reason it is free to drag.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import bv, cif, coordination, readers
from . import disorder as disorder_mod
from . import overrides as overrides_mod
from .structure import Structure


@dataclass
class Entry:
    """One structure in a project, with its own presentation state."""

    # The structure exactly as the file gave it, alternatives included. Read it
    # to see what the file says; use `structure` to work with it.
    source: Structure
    path: str | None = None
    visible: bool = True
    selected_site: int | None = None
    # offset applied when several structures are shown at once, in angstrom
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    color_key: int = 0          # index into a qualitative cycle, for overlay mode
    # Per-site and per-atom presentation overrides. They live with the entry
    # rather than with the theme because a label like "Bi1" means something in
    # one file and something else in another, while a palette is meant to be
    # reusable across files.
    overrides: overrides_mod.StyleOverrides = field(
        default_factory=lambda: overrides_mod.StyleOverrides())

    # Which disorder alternative is in use. Found from the file at construction.
    disorder: disorder_mod.Disorder | None = field(default=None, repr=False)

    _results: list | None = field(default=None, repr=False)
    # (by site index, reason it is empty) for the a priori valences
    _network: tuple | None = field(default=None, repr=False)
    _network_key: tuple | None = field(default=None, repr=False)
    _results_key: tuple | None = field(default=None, repr=False)
    _configuration: Structure | None = field(default=None, repr=False)
    _configuration_key: tuple | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.disorder is None:
            self.disorder = disorder_mod.find(self.source)
            # A file that describes alternatives is shown as one of them. Drawing
            # them all at once puts atoms a fraction of an angstrom apart and
            # makes every coordination number in the structure wrong, which is
            # not a reasonable default for a program about coordination numbers.
            # "Show everything" stays one click away.
            if self.disorder.present:
                self.disorder.choose_first()

    @property
    def structure(self) -> Structure:
        """The structure to draw and analyse: one disorder configuration.

        A real Structure with fewer sites rather than a flag, so the neighbour
        search, the coordination numbers, the bond valences and the diffraction
        pattern all work on something that could exist, with no special case for
        disorder anywhere downstream.
        """
        if self.disorder is None or not self.disorder.present:
            return self.source
        key = tuple(sorted(self.disorder.selected.items()))
        if self._configuration is None or self._configuration_key != key:
            self._configuration = disorder_mod.configuration(self.source,
                                                             self.disorder)
            self._configuration_key = key
        return self._configuration

    def choose_disorder(self, assembly: str, group: str) -> None:
        """Pick an alternative. Invalidates the analysis, which must be redone."""
        if self.disorder is None:
            return
        self.disorder.choose(assembly, group)
        self._configuration = None
        self._configuration_key = None
        self.invalidate()

    @property
    def name(self) -> str:
        if self.path:
            return Path(self.path).name
        return self.structure.name or "structure"

    @property
    def label(self) -> str:
        sg = self.structure.spacegroup_hm or "?"
        return f"{self.name}  ·  {sg}"

    def results(self, params: bv.ParameterSet, v_bond: float,
                v_list: float, anions: bool = False) -> list:
        """Analysis for this entry, cached.

        The cache key deliberately excludes `v_bond`: contacts are found down
        to `v_list`, and the bond threshold only classifies what was already
        found. Including it would re-run the neighbour search on every drag.
        It does include `anions`, because that changes which sites are in the
        list rather than how they are classified.
        """
        key = (id(params), round(v_list, 6), bool(anions))
        if self._results is None or self._results_key != key:
            self._results = coordination.analyse_structure(
                self.structure, params, v_bond=v_bond, v_list=v_list,
                cations_only=not anions)
            self._results_key = key
        return self._results

    def network(self, params: bv.ParameterSet, v_bond: float,
                v_list: float):
        """A priori bond valences and the two indices, cached.

        ``(by site index, reason)``: the mapping is empty and the reason set
        when the bond topology does not close, which is a property of the file
        rather than a failure here.

        Needs its own analysis because the topology has to be counted from the
        anion end as well, and the ordinary results may be cations only. Lazy:
        a session that never opens the a priori tab never runs it.
        """
        key = (id(params), round(v_bond, 6), round(v_list, 6))
        if self._network is None or self._network_key != key:
            from . import coordination, network as network_mod

            try:
                full = coordination.analyse_structure(
                    self.structure, params, v_bond=v_bond, v_list=v_list,
                    cations_only=False)
                rows = network_mod.analyse(self.structure, full, v_bond)
                self._network = ({r.site_index: r for r in rows}, "")
            except ValueError as error:
                self._network = ({}, str(error))
            except Exception as error:           # a reader or parameter fault
                self._network = ({}, f"{type(error).__name__}: {error}")
            self._network_key = key
        return self._network

    def invalidate(self) -> None:
        self._results = None
        self._results_key = None
        self._network = None
        self._network_key = None
        self._configuration = None
        self._configuration_key = None


class Project:
    """The open set of structures, and the settings shared across them."""

    def __init__(self):
        self.entries: list[Entry] = []
        self.active: int | None = None
        # Whether the anions are analysed as sites in their own right as well
        # as being the ligands of the cations. Off by default: the cations are
        # what a coordination analysis is usually about, and in a phosphate the
        # anions outnumber them three to one.
        self.include_anions = False

        self.params: bv.ParameterSet = bv.DEFAULT
        self.v_bond: float = bv.V_BOND_DEFAULT
        self.v_list: float = bv.V_LIST_DEFAULT
        # an optional hard distance ceiling applied on top of the valence
        # threshold, for comparing against a paper that used one
        self.distance_cutoff: float | None = None

        self.overlay: bool = False          # draw every visible entry together
        self.overlay_spacing: float = 0.0   # 0 superimposes; >0 lays them out

    # -- contents ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    @property
    def current(self) -> Entry | None:
        if self.active is None or not (0 <= self.active < len(self.entries)):
            return None
        return self.entries[self.active]

    @property
    def visible_entries(self) -> list[Entry]:
        if self.overlay:
            return [e for e in self.entries if e.visible]
        current = self.current
        return [current] if current is not None else []

    def add(self, structure: Structure, path: str | None = None) -> Entry:
        entry = Entry(source=structure, path=path,
                      color_key=len(self.entries))
        self.entries.append(entry)
        if self.active is None:
            self.active = 0
        self._relayout()
        return entry

    def add_file(self, path: str | Path) -> Entry:
        """Load any format FACET reads, chosen by extension then by content."""
        return self.add(readers.read(path), str(path))

    def add_files(self, paths) -> tuple[list[Entry], list[tuple[str, str]]]:
        """Load many files. Returns the entries added and the ones that failed.

        A folder of downloaded CIFs reliably contains a few that will not
        parse. Loading stops for those files only; the rest still open, and the
        failures are handed back with their reasons rather than raised.
        """
        added, failed = [], []
        for p in paths:
            try:
                added.append(self.add_file(p))
            except Exception as exc:
                failed.append((str(p), str(exc)))
        return added, failed

    def remove(self, index: int) -> None:
        if not (0 <= index < len(self.entries)):
            return
        self.entries.pop(index)
        if not self.entries:
            self.active = None
        elif self.active is not None:
            self.active = min(self.active, len(self.entries) - 1)
        self._relayout()

    def clear(self) -> None:
        self.entries.clear()
        self.active = None

    def set_active(self, index: int) -> None:
        if 0 <= index < len(self.entries):
            self.active = index

    def index_of(self, entry: Entry) -> int | None:
        for i, e in enumerate(self.entries):
            if e is entry:
                return i
        return None

    # -- shared settings ---------------------------------------------------
    def set_parameters(self, params: bv.ParameterSet) -> None:
        self.params = params
        self.invalidate_all()

    def set_list_threshold(self, v_list: float) -> None:
        """The tabulation threshold. Changing it *does* re-run the search."""
        self.v_list = float(v_list)
        self.invalidate_all()

    def set_bond_threshold(self, v_bond: float) -> None:
        """The bond threshold. Free: nothing is re-analysed."""
        self.v_bond = float(v_bond)

    def invalidate_all(self) -> None:
        for e in self.entries:
            e.invalidate()

    def network_for(self, entry: Entry):
        """A priori valences and indices for one entry, by site index."""
        return entry.network(self.params, self.v_bond, self.v_list)

    def results_for(self, entry: Entry) -> list:
        return entry.results(self.params, self.v_bond, self.v_list,
                             self.include_anions)

    def set_include_anions(self, on: bool) -> None:
        """Analyse the anions as sites too, not only as ligands."""
        on = bool(on)
        if on == self.include_anions:
            return
        self.include_anions = on
        self.invalidate_all()

    # -- layout ------------------------------------------------------------
    def set_overlay(self, on: bool, spacing: float | None = None) -> None:
        self.overlay = bool(on)
        if spacing is not None:
            self.overlay_spacing = float(spacing)
        self._relayout()

    def _relayout(self) -> None:
        """Place the entries for overlay mode.

        With zero spacing every structure is drawn about a common origin, which
        is what is wanted for comparing two refinements of the same phase.
        With spacing they are laid out along x, which is what is wanted for a
        row of different compounds.
        """
        if self.overlay_spacing <= 0:
            for e in self.entries:
                e.offset = np.zeros(3)
            return
        visible = [e for e in self.entries if e.visible]
        widths = [2.0 * _radius(e.structure) for e in visible]
        x = 0.0
        for e, w in zip(visible, widths):
            e.offset = np.array([x, 0.0, 0.0])
            x += w + self.overlay_spacing
        span = x - self.overlay_spacing if visible else 0.0
        for e in visible:
            e.offset = e.offset - np.array([span * 0.5, 0.0, 0.0])
        for e in self.entries:
            if not e.visible:
                e.offset = np.zeros(3)

    def toggle_visible(self, index: int, visible: bool | None = None) -> None:
        if not (0 <= index < len(self.entries)):
            return
        e = self.entries[index]
        e.visible = (not e.visible) if visible is None else bool(visible)
        self._relayout()

    def show_all(self, visible: bool = True) -> None:
        for e in self.entries:
            e.visible = bool(visible)
        self._relayout()

    # -- summary -----------------------------------------------------------
    def site_table(self) -> list[dict]:
        """Every analysed site of every entry, as flat rows.

        The basis of the comparison view and of every export: one row per site,
        carrying which structure it came from.
        """
        rows: list[dict] = []
        for e in self.entries:
            for r in self.results_for(e):
                plateau = r.current_plateau
                rows.append({
                    "structure": e.name,
                    "path": e.path or "",
                    "formula": e.structure.formula or "",
                    "spacegroup": e.structure.spacegroup_hm or "",
                    "site": r.label,
                    "element": r.element,
                    "oxidation_state": r.ox,
                    "oxidation_source": r.ox_source,
                    "wyckoff": r.wyckoff or "",
                    "site_symmetry": r.site_symmetry or "",
                    "multiplicity": r.multiplicity,
                    "cn": r.cn_at(self.v_bond),
                    "cn_ecoN": r.cn_ecoN,
                    "cn_gap_split": r.cn_gap,
                    "cn_listed": r.cn_listed,
                    "bvs": r.bvs,
                    "phi": r.phi,
                    "void_angle": r.void_angle,
                    "d_min": r.shape.get("d_min"),
                    "d_max": r.shape.get("d_max"),
                    "d_mean": r.shape.get("d_mean"),
                    "spread": r.shape.get("spread"),
                    "polyhedron_volume": r.shape.get("volume"),
                    "angle_variance": r.shape.get("angle_variance"),
                    "quadratic_elongation": r.shape.get("quadratic_elongation"),
                    "plateau_decades": plateau.width_decades if plateau else None,
                    "plateau_angstrom": plateau.width_angstrom if plateau else None,
                    "v_bond": self.v_bond,
                    "v_list": self.v_list,
                    "estimated_parameters": r.uses_estimated_params,
                })
        return rows

    def contact_table(self) -> list[dict]:
        """Every contact of every analysed site, as flat rows."""
        rows: list[dict] = []
        for e in self.entries:
            for r in self.results_for(e):
                for c in r.contacts:
                    rows.append({
                        "structure": e.name,
                        "site": r.label,
                        "element": r.element,
                        "neighbour": c.label,
                        "neighbour_element": c.element,
                        "distance": c.distance,
                        "valence": c.valence,
                        "occupancy": c.occupancy,
                        "is_bond": (c.valence is not None
                                    and c.valence > self.v_bond),
                        "r0": c.param.r0 if c.param else None,
                        "b": c.param.b if c.param else None,
                        "parameter_fitted": c.param.fitted if c.param else None,
                        "parameter_source": c.param.source if c.param else "",
                    })
        return rows


def _radius(structure: Structure) -> float:
    pts = structure.cart_array()
    if pts.size == 0:
        return 1.0
    centre = pts.mean(axis=0)
    return float(np.linalg.norm(pts - centre, axis=1).max()) or 1.0
