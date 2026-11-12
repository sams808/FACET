"""A crystal structure, and everything FACET computes from one.

    import facet

    c = facet.read("albite.cif")            # or facet.example("quartz")
    print(c.sites())                        # CN, valence sum, phi, plateau
    bi = c.site("Bi1")
    bi.cn, bi.bvs, bi.phi
    print(bi.contacts())                    # every contact with its valence
    v, cn = bi.cn_function()                # CN against the threshold
    c.diffraction().reflections             # the powder pattern
    c.pdf(r_max=10).G                       # the pair distribution function
    c.basicity().value                      # optical basicity

Each method is the calculation behind one part of the FACET window, called
with the same defaults and returning the same numbers. The objects the
engine itself returns (``SiteResult``, ``Pattern``, ``PDF``, ``Grid`` ...)
are handed back as they are, with their notes; lists of rows come back as a
:class:`~facet.library.tables.Table`. Nothing here imports the window
toolkit, except :meth:`Crystal.render`, which draws a picture.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..core import bv
from .tables import Table, table_of

V_BOND = bv.V_BOND_DEFAULT
V_LIST = bv.V_LIST_DEFAULT


# --- reading -----------------------------------------------------------------------

def parameters(name=None):
    """A set of bond-valence parameters.

    ``None`` or ``'built-in'``: Brese and O'Keeffe 1991. ``'bvparm2020'``:
    the IUCr compilation bundled with FACET. A path: a parameter file (IUCr
    bvparm CIF, a plain table, or a set saved by FACET). A ``ParameterSet``
    is returned as it is.
    """
    if isinstance(name, bv.ParameterSet):
        return name
    from ..core import bv_files

    return bv_files.named_set(None if name is None else str(name))


def read(path, block: str | None = None, **thresholds) -> "Crystal":
    """Read a crystal structure: CIF, VESTA, POSCAR/CONTCAR, SHELX (.res,
    .ins), PDB, CrystalMaker text, XYZ. ``block`` names the data block of a
    CIF holding several. ``v_bond``, ``v_list`` and ``params`` set the
    thresholds and the bond-valence parameters of the analysis."""
    from ..core import cif, readers

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no file {path}")
    try:
        loaded = cif.read(path, block) if block else readers.read(path)
    except readers.MDModelFile as error:
        raise ValueError(f"{path.name} is an MD model, not a crystal "
                         f"structure: open it with facet.model() ({error})") \
            from None
    return Crystal(loaded, **thresholds)


def read_all(path, **thresholds) -> list["Crystal"]:
    """Every data block of a CIF that holds a structure."""
    from ..core import cif

    return [Crystal(s, **thresholds) for s in cif.read_all_blocks(Path(path))]


def examples() -> dict[str, Path]:
    """The files bundled with FACET, by short name: six crystal structures
    and a practice glass model (``'glass'``, with its short trajectory
    ``'glass-trajectory'``)."""
    from ..core import examples as bundled

    out = {e.key: e.path for e in bundled.available()}
    for key, name in MODEL_EXAMPLES.items():
        found = bundled.folder() / name
        if found.is_file():
            out[key] = found
    return out


MODEL_EXAMPLES = {"glass": "practice_glass.data",
                  "glass-trajectory": "practice_glass.lammpstrj.gz"}
#: the type map of the practice glass: LAMMPS types to elements
GLASS_TYPE_MAP = {1: "O", 2: "Si", 3: "Al", 4: "Na", 5: "Ca"}


def example(name: str = "quartz", **thresholds):
    """A bundled file, opened: a :class:`Crystal`, or a ``Model`` for
    ``'glass'`` and ``'glass-trajectory'``."""
    found = examples()
    if name not in found:
        raise KeyError(f"no example {name!r}; the examples are "
                       f"{', '.join(found)}")
    if name in MODEL_EXAMPLES:
        from .model import model

        options = {} if found[name].name.endswith((".lammpstrj", ".lammpstrj.gz")) \
            else {"type_map": GLASS_TYPE_MAP}
        return model(found[name], **options)
    return read(found[name], **thresholds)


# --- one site ----------------------------------------------------------------------

class SiteView:
    """One site of a crystal (or one atom of a model frame) with its
    analysis: the ``SiteResult`` of the engine and the tables drawn from it.

    ``result`` is the engine's object and carries everything; the attributes
    here are its most used fields under short names.
    """

    def __init__(self, result, owner=None):
        self.result = result
        self._owner = owner

    # -- the numbers ------------------------------------------------------------
    label = property(lambda self: self.result.label)
    element = property(lambda self: self.result.element)
    ox = property(lambda self: self.result.ox)
    v_bond = property(lambda self: self.result.v_bond)
    cn = property(lambda self: self.result.cn_valence,
                  doc="coordination number at the bond threshold")
    cn_listed = property(lambda self: self.result.cn_listed)
    cn_occupancy = property(lambda self: self.result.cn_occupancy)
    econ = property(lambda self: self.result.cn_ecoN,
                    doc="Hoppe's effective coordination number")
    bvs = property(lambda self: self.result.bvs, doc="bond-valence sum, v.u.")
    bvs_uncertainty = property(lambda self: self.result.bvs_uncertainty)
    bvv = property(lambda self: self.result.bvv,
                   doc="length of the bond-valence vector sum, v.u.")
    phi = property(lambda self: self.result.phi,
                   doc="bvv / bvs, the stereoactivity index")
    plateau_decades = property(lambda self: self.result.plateau_decades)
    plateau_angstrom = property(lambda self: self.result.plateau_angstrom)
    void_angle = property(lambda self: self.result.void_angle,
                          doc="half-angle of the largest ligand-free cone, degrees")
    notes = property(lambda self: list(self.result.notes))

    @property
    def plateau(self):
        """The plateau the bond threshold sits on (None on a step edge)."""
        return self.result.current_plateau

    def cn_at(self, v_bond: float) -> int:
        """The coordination number at another threshold."""
        return self.result.cn_at(float(v_bond))

    def at(self, v_bond: float) -> "SiteView":
        """The same site analysed at another bond threshold."""
        from ..core import coordination

        return SiteView(coordination.reclassify(self.result, float(v_bond)),
                        self._owner)

    def cn_function(self, n: int = 400, v_min: float | None = None,
                    v_max: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        """(thresholds, CN): the coordination number as a function of the
        bond threshold, sampled for a plot."""
        from ..core import coordination

        return coordination.staircase(self.result, n=n, v_min=v_min, v_max=v_max)

    # -- tables -----------------------------------------------------------------
    def contacts(self) -> Table:
        """Every contact above the listing threshold, nearest first."""
        r = self.result
        rows = []
        for c in sorted(r.contacts, key=lambda c: c.distance):
            rows.append({
                "neighbour": c.label, "element": c.element,
                "distance / Å": c.distance,
                "valence / v.u.": c.valence,
                "bonded": bool(c.has_valence and c.valence > r.v_bond),
                "occupancy": c.occupancy,
                "R0 / Å": None if c.param is None else c.param.r0,
                "b / Å": None if c.param is None else c.param.b,
                "parameters": None if c.param is None else
                ("fitted" if c.param.fitted else "estimated"),
                "source": None if c.param is None else c.param.source,
            })
        return Table(rows, f"contacts of {r.label}",
                     [f"bonded: valence above {r.v_bond:g} v.u.; listed: "
                      f"above {r.v_list:g} v.u."])

    def plateaus(self) -> Table:
        """Every coordination number the site produces and the range of
        threshold over which it holds."""
        return Table(
            ({"CN": p.cn, "from / v.u.": p.v_low, "to / v.u.": p.v_high,
              "width / decades": p.width_decades,
              "width / Å": p.width_angstrom,
              "holds the bond threshold": p.contains(self.result.v_bond)}
             for p in self.result.plateaus), f"plateaus of {self.label}")

    def angles(self) -> Table:
        """Every ligand-centre-ligand angle between bonded contacts."""
        from ..core import utilities

        return table_of(utilities.bond_angles(self.result),
                        {"a": "ligand 1", "b": "ligand 2", "angle": "angle / deg",
                         "d_a": "d1 / Å", "d_b": "d2 / Å"},
                        f"angles at {self.label}")

    def shells(self, axis: str = "distance") -> Table:
        """The contacts in the order a growing cutoff includes them, with the
        coordination number and the valence sum after each (``axis``:
        'distance' or 'valence')."""
        from ..core import shells

        return table_of(shells.shell_curve(self.result, axis),
                        {"label": "neighbour", "element": "element",
                         "distance": "distance / Å", "valence": "valence / v.u.",
                         "occupancy": "occupancy", "cn_after": "CN after",
                         "sum_after": "valence sum after / v.u."},
                        f"shells of {self.label} by {axis}")

    def cn_methods(self, pymatgen: bool = False,
                   hard_cutoff: float | None = None) -> Table:
        """The coordination number by every definition FACET applies
        (bond-valence threshold, ECoN, CHARDI, gap, Voronoi, percent of
        shortest, ...), each with its parameters. ``pymatgen=True`` adds
        pymatgen's when it is installed."""
        from ..core import cn_methods

        structure = getattr(self._owner, "structure", None)
        found = cn_methods.all_methods(self.result, structure,
                                       include_pymatgen=pymatgen,
                                       hard_cutoff_distance=hard_cutoff)
        table = Table(({"method": m.method, "CN": m.cn, "integer": m.integer,
                        "parameters": m.parameters, "available": m.available,
                        "note": m.note} for m in found),
                      f"coordination number of {self.label} by method")
        spread = cn_methods.spread(found)
        table.notes.append("spread over the methods: " + ", ".join(
            f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}"
            for k, v in spread.items()))
        return table

    def shape(self) -> dict:
        """Distance statistics, hull volume and distortion indices of the
        coordination polyhedron."""
        return dict(self.result.shape)

    def lone_pair(self) -> dict:
        """What the bond-valence vector and the empty cone say of a
        stereoactive lone pair: phi, the vector and the largest ligand-free
        cone."""
        r = self.result
        return {"phi": r.phi, "phi listed": r.phi_listed,
                "bond-valence vector / v.u.": np.asarray(r.bvv_vector).tolist(),
                "void half-angle / deg": r.void_angle,
                "void axis": np.asarray(r.void_axis).tolist()}

    def radial_shells(self, r_max: float = 6.0, tolerance: float = 0.05) -> Table:
        """Neighbours grouped into shells by distance, out to ``r_max``."""
        from ..core import utilities

        crystal = self._need_crystal("radial_shells")
        return Table(utilities.radial_shells(crystal.structure,
                                             self.result.site_index,
                                             rmax=float(r_max),
                                             tolerance=float(tolerance)),
                     f"radial shells about {self.label}")

    def exafs_shells(self, r_max: float = 6.0, tolerance: float = 0.05,
                     temperature: float = 300.0,
                     einstein_temperature: float = 500.0):
        """The shells about this site as an EXAFS fit wants them (N, R,
        sigma squared): the engine's ``ShellTable``; ``.shells`` holds the
        rows, :func:`facet.table` turns them into a table."""
        from ..core import exafs

        crystal = self._need_crystal("exafs_shells")
        return exafs.shell_table(crystal.structure, self.result.site_index,
                                 r_max=float(r_max), tolerance=float(tolerance),
                                 temperature=float(temperature),
                                 einstein_temperature=float(einstein_temperature))

    def write_feff(self, path, r_max: float = 6.0, edge: str = "L3") -> Path:
        """A FEFF input with this site as the absorber."""
        from ..core import exporters

        crystal = self._need_crystal("write_feff")
        return exporters.write_feff(crystal.structure, self.result.site_index,
                                    Path(path), rmax=float(r_max), edge=edge)

    def text(self, fmt: str = "txt") -> str:
        """The site written out: plain text ('txt'), Markdown ('md') or
        LaTeX ('tex')."""
        from ..core import site_text

        source = getattr(self._owner, "name", "") or ""
        return site_text.site_text(self.result, self.result.v_bond, fmt=fmt,
                                   source=source)

    def _need_crystal(self, what: str):
        if not isinstance(self._owner, Crystal):
            raise ValueError(f"{what}() needs a site of a crystal structure")
        return self._owner

    def row(self) -> dict:
        """The site as one row of :meth:`Crystal.sites`."""
        r = self.result
        p = r.current_plateau
        distances = [c.distance for c in r.bonds]
        return {
            "site": r.label, "element": r.element, "ox": r.ox,
            "Wyckoff": r.wyckoff, "multiplicity": r.multiplicity,
            "CN": r.cn_valence, "CN listed": r.cn_listed,
            "ECoN": r.cn_ecoN,
            "BVS / v.u.": r.bvs, "BVS listed / v.u.": r.bvs_listed,
            "BVS - ox / v.u.": r.valence_discrepancy,
            "phi": r.phi, "void half-angle / deg": r.void_angle,
            "plateau / decades": r.plateau_decades,
            "plateau from / v.u.": None if p is None else p.v_low,
            "plateau to / v.u.": None if p is None else p.v_high,
            "shortest bond / Å": min(distances) if distances else None,
            "longest bond / Å": max(distances) if distances else None,
            "estimated parameters": bool(r.uses_estimated_params),
        }

    def __repr__(self) -> str:
        return f"<Site {self.result.summary()}>"

    def __str__(self) -> str:
        return self.result.summary()


# --- a plane -----------------------------------------------------------------------

class PlaneView:
    """One family of (hkl) lattice planes of a crystal."""

    def __init__(self, crystal: "Crystal", h: int, k: int, l: int,
                 offset: float = 0.0):
        from ..core import planes

        self._crystal = crystal
        self.plane = planes.LatticePlane(int(h), int(k), int(l), float(offset))
        if not self.plane.is_valid:
            raise ValueError("(0 0 0) is not a plane")

    @property
    def spacing(self) -> float:
        """The interplanar spacing d, in angstrom."""
        from ..core import planes

        return planes.spacing(self._crystal.structure, self.plane)

    def atoms(self, tolerance: float = 0.3) -> Table:
        """The atoms within ``tolerance`` angstrom of a plane of the family."""
        from ..core import planes

        s = self._crystal.structure
        rows = []
        for index in planes.atoms_on_plane(s, self.plane, float(tolerance)):
            a = s.atoms[index]
            rows.append({"atom": index, "site": a.label, "element": a.element,
                         "x": float(a.frac[0]), "y": float(a.frac[1]),
                         "z": float(a.frac[2])})
        return Table(rows, f"atoms on ({self.plane.hkl}) within {tolerance:g} Å")

    def occupancy(self, tolerance: float = 0.3) -> dict[str, float]:
        """Element counts on the planes, weighted by occupancy."""
        from ..core import planes

        return planes.plane_occupancy(self._crystal.structure, self.plane,
                                      float(tolerance))

    def angle_to(self, h: int, k: int, l: int) -> float:
        """The angle between this plane's normal and another's, in degrees."""
        from ..core import planes

        p = self.plane
        return planes.interplanar_angle(self._crystal.structure,
                                        (p.h, p.k, p.l), (int(h), int(k), int(l)))

    def __repr__(self) -> str:
        return f"<Plane ({self.plane.hkl}) d = {self.spacing:.4f} Å>"


# --- the crystal -------------------------------------------------------------------

class Crystal:
    """A crystal structure with FACET's analysis of it.

    ``v_bond`` is the bond threshold in valence units (a contact whose bond
    valence is above it counts as a bond), ``v_list`` the listing threshold
    (contacts above it are kept in the tables), ``params`` the bond-valence
    parameter set (see :func:`parameters`). Changing one of them through
    :meth:`with_thresholds` gives a new ``Crystal``; the structure is shared.
    """

    def __init__(self, structure, v_bond: float = V_BOND, v_list: float = V_LIST,
                 params=None):
        self.structure = structure
        self.v_bond = float(v_bond)
        self.v_list = float(v_list)
        self.params = parameters(params)
        self._results: dict[bool, list] = {}

    # -- identity -----------------------------------------------------------------
    name = property(lambda self: self.structure.name)
    formula = property(lambda self: self.structure.formula)
    spacegroup = property(lambda self: self.structure.spacegroup_hm)
    cell = property(lambda self: self.structure.cell)
    path = property(lambda self: self.structure.source_path)
    notes = property(lambda self: list(self.structure.notes))
    n_sites = property(lambda self: self.structure.n_sites)
    n_atoms = property(lambda self: self.structure.n_atoms)

    @property
    def labels(self) -> list[str]:
        """The label of every site, in file order."""
        return [s.label for s in self.structure.sites]

    def with_thresholds(self, v_bond: float | None = None,
                        v_list: float | None = None, params=None) -> "Crystal":
        """The same structure analysed with other thresholds or parameters."""
        return Crystal(self.structure,
                       self.v_bond if v_bond is None else v_bond,
                       self.v_list if v_list is None else v_list,
                       self.params if params is None else params)

    def __repr__(self) -> str:
        return (f"<Crystal {self.name!r} {self.spacegroup or '?'}: "
                f"{self.n_sites} sites, {self.n_atoms} atoms in the cell; "
                f"v_bond {self.v_bond:g} v.u.>")

    # -- the analysis ---------------------------------------------------------------
    def results(self, anions: bool = False) -> list:
        """The engine's ``SiteResult`` of every cation site (``anions=True``:
        of every site). Computed once and kept."""
        from ..core import coordination

        key = bool(anions)
        if key not in self._results:
            self._results[key] = coordination.analyse_structure(
                self.structure, self.params, v_bond=self.v_bond,
                v_list=self.v_list, cations_only=not key)
        return self._results[key]

    def site(self, label) -> SiteView:
        """One site by its label (or by its index in the file)."""
        if isinstance(label, (int, np.integer)):
            index = int(label)
        else:
            index = self.structure.site_by_label(str(label))
            if index is None:
                raise KeyError(f"no site labelled {label!r}; the sites are "
                               f"{', '.join(self.labels)}")
        for r in self.results(anions=True):
            if r.site_index == index:
                return SiteView(r, self)
        raise KeyError(f"site {label!r} has no analysis")

    def __getitem__(self, label) -> SiteView:
        return self.site(label)

    def sites(self, anions: bool = False) -> Table:
        """One row per cation site (``anions=True``: per site): coordination
        number, valence sum, phi, the plateau the threshold sits on."""
        return Table((SiteView(r, self).row() for r in self.results(anions)),
                     f"sites of {self.name}", self._provenance())

    def _provenance(self) -> list[str]:
        return [f"bond threshold {self.v_bond:g} v.u., listing threshold "
                f"{self.v_list:g} v.u., parameters: {self.params.name}"]

    # -- composition, cell, file ------------------------------------------------------
    def composition(self) -> dict:
        """Cell contents, formula weight and the charge balance."""
        from ..core import utilities

        self.results()                      # the oxidation states are assigned
        return utilities.composition_summary(self.structure)

    def density(self) -> dict:
        """Cell mass, volume and density."""
        from ..core import utilities

        return utilities.density(self.structure)

    def reciprocal_cell(self) -> dict:
        from ..core import utilities

        return utilities.reciprocal_cell(self.structure)

    def check(self) -> Table:
        """What FACET's reading of the file found worth stating: occupancies,
        missing fields, short contacts, valence sums far from the charge.
        One row per finding, each with its level."""
        from ..core import quality

        report = quality.check(self.structure, self.results(), self.v_bond)
        return Table(({"level": f.level.name.lower(), "code": f.code,
                       "where": f.where, "message": f.message,
                       "value": f.value, "reference": f.reference}
                      for f in report.findings), f"file check of {self.name}")

    def oxidation_states(self) -> Table:
        """The oxidation state given to every cation site and where it came
        from (the file, or the state whose valence sum fits)."""
        from ..core import oxidation

        found = oxidation.resolve(self.structure, self.params,
                                  v_bond=self.v_bond, v_list=self.v_list)
        return Table(({"site": v.label, "element": v.element, "ox": v.ox,
                       "source": v.source, "margin / v.u.": v.margin,
                       "candidates": v.candidates,
                       "close second": v.ambiguous} for v in found),
                     f"oxidation states of {self.name}")

    def disorder(self):
        """The disorder assemblies the sites declare (the engine's
        ``Disorder``): ``.present``, ``.describe()``, ``.choose(assembly,
        group)``."""
        from ..core import disorder

        return disorder.find(self.structure)

    def configuration(self, choices: Mapping[str, str] | None = None) -> "Crystal":
        """One configuration of a disordered structure: the first group of
        every assembly, or the groups named in ``choices`` (assembly ->
        group)."""
        from ..core import disorder

        found = disorder.find(self.structure)
        found.choose_first()
        for assembly, group in (choices or {}).items():
            found.choose(str(assembly), str(group))
        return Crystal(disorder.configuration(self.structure, found),
                       self.v_bond, self.v_list, self.params)

    # -- reflections --------------------------------------------------------------------
    def d_spacing(self, h: int, k: int, l: int) -> float:
        from ..core import utilities

        return utilities.d_spacing(self.structure, int(h), int(k), int(l))

    def two_theta(self, h: int, k: int, l: int, wavelength: float = 1.5406) -> float:
        from ..core import utilities

        return utilities.two_theta(self.structure, int(h), int(k), int(l),
                                   float(wavelength))

    def reflections(self, wavelength: float = 1.5406, two_theta_max: float = 90.0,
                    max_index: int = 8) -> Table:
        """Every reflection position up to a 2-theta limit, with its d
        spacing (positions only; :meth:`diffraction` gives intensities)."""
        from ..core import utilities

        return Table(utilities.reflection_list(self.structure, float(wavelength),
                                               float(two_theta_max), int(max_index)),
                     f"reflections of {self.name}",
                     [f"wavelength {wavelength:g} Å"])

    def diffraction(self, wavelength: float = 1.540598, radiation: str = "X-ray",
                    two_theta_min: float = 0.0, two_theta_max: float = 90.0,
                    b_iso: float = 0.5, min_intensity: float = 0.01,
                    monochromator_two_theta: float | None = None):
        """The calculated powder pattern (the engine's ``Pattern``):
        ``.reflections`` with h, k, l, d, 2-theta, relative intensity and
        multiplicity; ``.profile()`` broadens them into a curve.
        ``radiation``: 'X-ray' or 'neutron'."""
        from ..core import diffraction

        return diffraction.powder_pattern(
            self.structure, wavelength=float(wavelength),
            two_theta_max=float(two_theta_max), two_theta_min=float(two_theta_min),
            radiation=radiation, b_iso=float(b_iso),
            min_intensity=float(min_intensity),
            monochromator_two_theta=monochromator_two_theta)

    def structure_factor(self, h: int, k: int, l: int, radiation: str = "X-ray",
                         b_iso: float = 0.5) -> complex:
        from ..core import diffraction

        return diffraction.structure_factor(self.structure, int(h), int(k), int(l),
                                            radiation=radiation, b_iso=float(b_iso))

    def pdf(self, r_max: float = 20.0, dr: float = 0.01, radiation: str = "X-ray",
            **options):
        """The pair distribution function (the engine's ``PDF``): ``.r``,
        ``.G``, ``.g``, ``.R`` and every assumption in ``.notes``. Options:
        ``u_iso``, ``delta1``, ``delta2``, ``q_min``, ``q_max``, ``q_damp``,
        ``q_broad``, ``window``."""
        from ..core import pdf

        return pdf.pair_distribution(self.structure, r_max=float(r_max),
                                     dr=float(dr), radiation=radiation, **options)

    # -- bond valence over the structure ----------------------------------------------
    def cutoffs(self) -> Table:
        """The distance each cation-anion pair is cut at by the two
        thresholds, with the parameters used."""
        from ..core import bv_report

        found = bv_report.cutoff_table(self.structure, self.params,
                                       self.v_bond, self.v_list,
                                       results=self.results())
        return Table(({"pair": p.label, "R0 / Å": p.r0, "b / Å": p.b,
                       "parameters": "fitted" if p.fitted else "estimated",
                       "source": p.source,
                       "bond cutoff / Å": p.d_bond, "listing cutoff / Å": p.d_list,
                       "shortest / Å": p.shortest, "bonds": p.n_bonds,
                       "listed": p.n_listed} for p in found.rows),
                     f"cutoffs of {self.name}",
                     self._provenance() + list(found.notes))

    def stability(self, low: float = 0.005, high: float = 0.30, steps: int = 120):
        """The coordination number of every site against the threshold (the
        engine's ``ThresholdScan``): ``.thresholds``, ``.per_site[label]``,
        ``.stable_range(label, at)``, ``.widths(at)``."""
        from ..core import bv_report

        return bv_report.threshold_scan(self.results(), float(low), float(high),
                                        int(steps))

    def anion_sums(self) -> Table:
        """The bond-valence sum of every anion site, from the cations around
        it."""
        from ..core import bv_report

        found = bv_report.anion_sums(self.structure, self.params,
                                     self.v_bond, self.v_list)
        return Table(({"site": a.label, "element": a.element,
                       "expected / v.u.": a.expected, "BVS / v.u.": a.bvs,
                       "contacts": a.n_contacts, "BVS - expected / v.u.": a.discrepancy,
                       "BVS listed / v.u.": a.bvs_listed, "listed": a.n_listed}
                      for a in found), f"anion valence sums of {self.name}",
                     self._provenance())

    def charge_balance(self) -> dict:
        """Whether the cation and the anion valence sums account for each
        other, as numbers."""
        from ..core import bv_report

        return bv_report.charge_balance(self.structure, self.results(),
                                        self.params, self.v_bond, self.v_list)

    def gii(self) -> dict:
        """The global instability index over the cation sites, with the site
        contributing most."""
        from ..core import utilities

        return utilities.global_instability_index(self.results())

    def bond_strain(self) -> float:
        """The bond strain index, in valence units."""
        from ..core import utilities

        return utilities.bond_strain_index(self.results())

    def a_priori(self) -> Table:
        """A priori bond valences of the bond network and the two indices
        that compare them with the observed ones, per cation polyhedron."""
        from ..core import network

        found = network.analyse(self.structure, self.results(anions=True),
                                self.v_bond)
        return Table(({"site": s.label, "element": s.element, "ox": s.ox,
                       "CN": s.coordination,
                       "a priori / v.u.": s.a_priori, "observed / v.u.": s.observed,
                       "delta topol / v.u.": s.delta_topol,
                       "delta cryst / v.u.": s.delta_cryst} for s in found),
                     f"a priori valences of {self.name}", self._provenance())

    def connectivity(self) -> Table:
        """How the coordination polyhedra join: corner, edge or face."""
        from ..core import utilities

        return Table(utilities.polyhedral_connectivity(
            self.structure, self.results(), self.v_bond),
            f"polyhedral connectivity of {self.name}", self._provenance())

    # -- geometry -----------------------------------------------------------------------
    def plane(self, h: int, k: int, l: int, offset: float = 0.0) -> PlaneView:
        """A family of lattice planes: spacing, the atoms on it, what it
        holds, its angle to another."""
        return PlaneView(self, h, k, l, offset)

    def torsion(self, a: int, b: int, c: int, d: int) -> float:
        """The dihedral angle through four atoms of the cell (indices into
        ``structure.atoms``), in degrees."""
        from ..core import utilities

        return utilities.torsions(self.structure, int(a), int(b), int(c), int(d))

    def atoms(self) -> Table:
        """Every atom of the cell: site, element, fractional and Cartesian
        coordinates."""
        rows = []
        for index, a in enumerate(self.structure.atoms):
            if not a.in_home_cell:
                continue
            rows.append({"atom": index, "site": a.label, "element": a.element,
                         "x": float(a.frac[0]), "y": float(a.frac[1]),
                         "z": float(a.frac[2]),
                         "X / Å": float(a.cart[0]), "Y / Å": float(a.cart[1]),
                         "Z / Å": float(a.cart[2]), "occupancy": a.occupancy})
        return Table(rows, f"atoms of {self.name}")

    # -- maps ---------------------------------------------------------------------------
    def valence_map(self, element: str, ox: int | None = None,
                    resolution: float = 0.25, r_max: float = 6.0):
        """The bond-valence sum a probe cation would have at every point of
        the cell (the engine's ``Grid``: ``.values``, ``.statistics()``,
        ``.at_fractional(xyz)``)."""
        from ..core import volume

        return volume.bond_valence_grid(self.structure, str(element),
                                        self._probe_ox(element, ox),
                                        resolution=float(resolution),
                                        params=self.params, rmax=float(r_max))

    def energy_map(self, element: str, ox: int | None = None,
                   resolution: float = 0.25, temperature: float = 300.0):
        """The valence map turned into an energy-like landscape for an ion's
        pathways."""
        from ..core import volume

        state = self._probe_ox(element, ox)
        grid = self.valence_map(element, state, resolution)
        return volume.bond_valence_energy(grid, state, float(temperature))

    def _probe_ox(self, element: str, ox: int | None) -> int:
        if ox is not None:
            return int(ox)
        for s in self.structure.sites:
            if s.element == element and s.ox:
                return int(s.ox)
        from ..core import elements

        usual = elements.COMMON_OX.get(str(element))
        if not usual:
            raise ValueError(f"no usual oxidation state for {element}; give ox")
        return int(usual)

    # -- optical basicity -----------------------------------------------------------------
    def basicity(self, scale: str = "pauling", formers: Sequence[str] = ()):
        """Optical basicity: Lambda of the composition and lambda of every
        anion site (``facet.basicity.StructureResult``)."""
        from .. import basicity as fb

        return fb.structure(self.structure, scale, v_bond=self.v_bond,
                            v_list=self.v_list, formers=formers)

    # -- writing ---------------------------------------------------------------------------
    def write(self, path, **options) -> Path:
        """Write the structure, the format from the file name: ``.cif`` (the
        analysis as comments), ``.vasp`` or a name starting with POSCAR,
        ``.xyz`` (extended XYZ; ``cell_range=(2, 2, 2)`` for a supercell),
        ``.vesta``."""
        from ..core import exporters

        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".cif":
            return exporters.write_cif(self.structure, path,
                                       results=self.results(),
                                       provenance=self._provenance(), **options)
        if suffix in (".vasp", ".poscar") or path.name.upper().startswith(
                ("POSCAR", "CONTCAR")):
            return exporters.write_poscar(self.structure, path, **options)
        if suffix in (".xyz", ".extxyz"):
            return exporters.write_xyz(self.structure, path, **options)
        if suffix == ".vesta":
            return exporters.write_vesta(self.structure, path,
                                         v_bond=self.v_bond, **options)
        raise ValueError(f"{path.name}: the formats written are .cif, .vasp "
                         "(POSCAR), .xyz and .vesta")

    def render(self, path, **options) -> Path:
        """A picture of the structure (PNG, SVG or PDF); see
        :func:`facet.library.render.render`."""
        from .render import render

        return render(self, path, **options)


# --- several structures ------------------------------------------------------------------

def report_on(paths: Iterable, out=None, fmt: str | None = None, *,
           v_bond: float = V_BOND, v_list: float = V_LIST, params=None,
           anions: bool = False, v_bond_other: float | None = None,
           basicity_scale: str | None = "pauling"):
    """The bond-valence report over a set of structure files or folders.

    Returns the engine's ``Report``: ``.tables()`` maps a table name
    ('sites', 'contacts', 'plateaus', 'cutoffs', 'balance', 'basicity') to
    its rows. With ``out`` the report is also written (``fmt``: 'xlsx',
    'csv', 'html', 'md', 'tex', 'json'; taken from the file name when not
    given)."""
    from ..core import report as engine

    found = engine.analyse_files([str(p) for p in paths], parameters(params),
                                 v_bond=float(v_bond), v_list=float(v_list),
                                 anions=bool(anions), v_bond_other=v_bond_other,
                                 basicity_scale=basicity_scale)
    if out is not None:
        out = Path(out)
        chosen = fmt or (out.suffix.lstrip(".").lower() or "xlsx")
        engine.write_report(found, out, chosen)
    return found


def compare(structures: Iterable, **thresholds) -> Table:
    """The cation sites of several structures in one table, one row per
    site, the structure's name in the first column. Takes paths or
    ``Crystal`` objects."""
    rows = []
    for item in structures:
        crystal = item if isinstance(item, Crystal) else read(item, **thresholds)
        for row in crystal.sites():
            rows.append({"structure": crystal.name, **row})
    return Table(rows, "sites of several structures")


def bond_valence(distance, r0: float, b: float = bv.B_DEFAULT):
    """v = exp((R0 - d) / b), for a distance or an array of them."""
    return bv.valence(distance, float(r0), float(b))


def cutoff_distance(r0: float, v: float = V_BOND, b: float = bv.B_DEFAULT) -> float:
    """The distance at which the bond valence falls to ``v``."""
    return bv.cutoff_for_valence(float(r0), float(v), float(b))


def read_volume(path):
    """A volumetric file (CHGCAR, Gaussian cube, XSF): ``(grid, structure
    or None)``."""
    from ..core import volume

    return volume.read_volume(Path(path))


def selfcheck() -> Table:
    """The numbers FACET's manual states for the bundled examples,
    recomputed on this machine, each against its tolerance."""
    from ..core import selfcheck as engine

    reference = engine.load_reference()
    rows = engine.check_examples(reference) + engine.check_sums(reference)
    return Table((r.as_dict() for r in rows), engine.summary(rows))


__all__ = ["Crystal", "SiteView", "PlaneView", "read", "read_all", "example",
           "examples", "parameters", "report_on", "compare", "bond_valence",
           "cutoff_distance", "read_volume", "selfcheck", "V_BOND", "V_LIST"]
