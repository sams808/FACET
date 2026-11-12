"""A molecular-dynamics model (one frame or a trajectory), and everything
FACET measures on one.

    import facet

    m = facet.model("glass.data", type_map={1: "O", 2: "Si", 3: "Na"})
    print(m)                                    # what was read and assumed
    r = m.analyse("glass", "rings", formers=["Si"],
                  network={"ring_criterion": "king", "ring_max_size": 12})
    print(r.descriptors())                      # every table that was measured
    print(r.table("glass", "Qn"))               # one of them, as rows
    r.export("glass.xlsx")                      # all of them, with provenance

    m.atoms()                                   # CN, valence sum, phi per atom
    m.site(17)                                  # one atom read as a site
    m.basicity(formers=["Si"]).value            # optical basicity

``analyse`` runs the same driver as the Model workspace of the window and as
``python -m facet.md analyse``: a request names the analyses and their
inputs, every missing input is named before a frame is read, and each
result carries its provenance.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..core import bv
from .tables import Table

V_BOND = bv.V_BOND_DEFAULT
V_LIST = bv.V_LIST_DEFAULT


def analyses() -> Table:
    """Every analysis a model can be given: its name, when it runs (per
    frame, over the whole trajectory, after the others), what it measures
    and the inputs it needs."""
    from ..core import md_analysis as ma

    empty = ma.AnalysisRequest(analyses=())
    rows = []
    for name in ma.ANALYSES:
        trial = dataclasses.replace(empty, analyses=(name,)
                                    + ma.dependencies(empty, name))
        needs = sorted({m.name for m in ma.missing_inputs(trial)
                        if name in m.analysis.split("/")})
        if name in ma.TRAJECTORY_ANALYSES or name == "bond-lifetimes":
            needs.append("a time axis (timestep_fs or frame_interval_ps) "
                         "and two frames or more")
        if name in ("kinetic-temperature", "vacf"):
            needs.append("velocities")
        rows.append({
            "analysis": name,
            "runs": ("per frame" if name in ma.PER_FRAME_ANALYSES else
                     "whole trajectory" if name in ma.TRAJECTORY_ANALYSES
                     else "after the others"),
            "measures": ma.ANALYSIS_SUMMARIES[name],
            "needs": ", ".join(needs),
            "reads the results of": ", ".join(ma.dependencies(empty, name)),
        })
    return Table(rows, "analyses of an MD model")


def presets() -> Table:
    """The questions the Model workspace offers as presets: the analyses,
    the candidate formers and the method inputs each one sets."""
    from ..core import md_presets

    return Table(({"preset": p.name, "question": p.description,
                   "analyses": ", ".join(p.analyses),
                   "formers": ", ".join(p.formers),
                   "options": dict(p.options), "needs": ", ".join(p.needs)}
                  for p in md_presets.PRESETS if not p.custom),
                 "presets")


def options() -> str:
    """Every input and option of the analyses, as a commented text: the
    top-level inputs, then each option group with its fields, their types
    and the defaults that are method choices."""
    from ..md import cli

    return cli.template_text()


def request(*names: str, **options):
    """The engine's ``AnalysisRequest`` for these analyses and inputs,
    checked as the command line checks a request file.

    Top-level inputs: ``formers`` (a list of elements, or ``"none"``),
    ``frames`` (``"0:100:5"``, a slice, or a list of indices), ``ox``
    (oxidation states that differ from the usual ones), ``v_bond``,
    ``v_list``, ``timestep_fs`` or ``frame_interval_ps``, ``temperature_k``,
    ``charges_e``, ``workers``. Option groups, each a mapping: ``glass``,
    ``scattering``, ``network``, ``order``, ``voids``, ``channels``, ``nmr``,
    ``exafs``, ``feff``, ``dynamics``, ``basicity``.
    """
    from ..md import cli

    spec: dict = {}
    for key, value in options.items():
        key = cli._TOP_ALIASES.get(key, key)
        if isinstance(value, slice):
            value = ":".join("" if v is None else str(v)
                             for v in (value.start, value.stop, value.step))
        spec[key] = value
    listed = [n for name in names for n in
              ([name] if isinstance(name, str) else list(name))]
    if listed:
        spec["analyses"] = listed
    return cli.request_from_mapping(spec)


def _number(value):
    """A value written as text for a file, back as the number it is: the
    rows of a descriptor are those of the exports, where a key such as the
    n of Qn or a count is text."""
    if isinstance(value, str):
        text = value.strip()
        if text.lstrip("-").isdigit():
            return int(text)
        try:
            return float(text) if any(c in text for c in ".eE") and \
                text.lower() not in ("e", "inf", "-inf", "nan") else value
        except ValueError:
            return value
    return value


class Results:
    """What one :meth:`Model.analyse` run measured.

    ``result`` is the engine's ``ModelResult``. A descriptor is one table of
    one analysis (``"glass"``, ``"Qn"``); its container (a distribution, a
    histogram, a series, a scalar, or plain rows) keeps the per-frame values
    and the mean and spread over the frames.
    """

    def __init__(self, result):
        self.result = result

    @property
    def analyses(self) -> list[str]:
        """The analyses that ran, in the order they ran."""
        return list(self.result.outputs)

    @property
    def failed(self) -> dict[str, str]:
        """The analyses that produced nothing, each with the reason."""
        return dict(self.result.failed)

    @property
    def notes(self) -> list[str]:
        """What the run states about the model and about itself."""
        return list(self.result.model_notes) + list(self.result.notes)

    @property
    def provenance(self) -> list[str]:
        """The run's header: file, frames, type map, oxidation states,
        thresholds, formers, cutoffs and every method parameter."""
        return list(self.result.provenance.as_lines())

    @property
    def minima(self) -> Table:
        """The first minimum of each partial g(r), the distance cutoffs of
        the glass analysis."""
        rows = []
        for pair, found in dict(self.result.minima).items():
            row = {}
            if dataclasses.is_dataclass(found):
                row.update({f.name: getattr(found, f.name)
                            for f in dataclasses.fields(found)
                            if not isinstance(getattr(found, f.name), np.ndarray)})
            else:
                row["value"] = found
            row["pair"] = "-".join(str(e) for e in pair)                 if isinstance(pair, (tuple, list)) else str(pair)
            rows.append({"pair": row.pop("pair"), **row})
        return Table(rows, "first minima of the partial g(r)")

    def descriptors(self, analysis: str | None = None) -> Table:
        """Every descriptor measured: analysis, name, kind of container and
        number of rows."""
        from ..core import md_export

        rows = []
        for name, key, container in self.result.descriptors():
            if analysis is not None and name != analysis:
                continue
            try:
                n = len(md_export.descriptor_rows(container))
            except Exception:
                n = None
            rows.append({"analysis": name, "descriptor": key,
                         "kind": type(container).__name__.lower(), "rows": n})
        return Table(rows, "descriptors measured")

    def find(self, text: str) -> Table:
        """The descriptors whose name holds ``text`` (case ignored)."""
        wanted = str(text).lower()
        return Table((r for r in self.descriptors()
                      if wanted in r["descriptor"].lower()),
                     f"descriptors matching {text!r}")

    def get(self, analysis: str, descriptor: str):
        """One descriptor's container, as the engine made it."""
        out = self._output(analysis)
        if descriptor not in out.tables:
            raise KeyError(f"{analysis} has no descriptor {descriptor!r}; it "
                           f"has {', '.join(out.tables)}")
        return out.tables[descriptor]

    def __getitem__(self, analysis: str) -> dict:
        return dict(self._output(analysis).tables)

    def table(self, analysis: str, descriptor: str,
              per_frame: bool = False) -> Table:
        """One descriptor as rows, every column named with its unit: the
        mean and the spread over the frames, and with ``per_frame=True`` one
        more column per frame."""
        from ..core import md_export

        container = self.get(analysis, descriptor)
        notes = list(getattr(container, "notes", ()) or ())
        rows = md_export.descriptor_rows(container, per_frame=per_frame)
        return Table(({k: _number(v) for k, v in row.items()} for row in rows),
                     f"{analysis}: {descriptor}", notes)

    def raw(self, analysis: str):
        """The analysis module's own result object (``GlassResult``,
        ``ScatteringResult`` ...), for what the tables do not hold."""
        return self._output(analysis).raw

    def _output(self, analysis: str):
        outputs = self.result.outputs
        if analysis not in outputs:
            raise KeyError(f"{analysis!r} did not run; the analyses run are "
                           f"{', '.join(outputs)}")
        out = outputs[analysis]
        if out.error is not None:
            raise ValueError(f"{analysis} produced nothing: {out.error}")
        return out

    def export(self, path, per_frame: bool = False,
               analyses: Sequence[str] | None = None) -> list[Path]:
        """Write every descriptor with its provenance: one workbook when
        ``path`` ends in .xlsx, otherwise one CSV per descriptor in the
        directory ``path``."""
        from ..core import md_export

        return md_export.export(self.result, Path(path), per_frame=per_frame,
                                analyses=analyses)

    def __repr__(self) -> str:
        n = sum(1 for _ in self.result.descriptors())
        failed = f", {len(self.failed)} produced nothing" if self.failed else ""
        return (f"<Results of {', '.join(self.analyses)}: {n} descriptor(s)"
                f"{failed}>")


class Model:
    """An MD model: a file FACET reads (LAMMPS data and dumps, extended XYZ,
    DL_POLY, VASP XDATCAR and POSCAR series, GROMACS, CP2K, ...), with its
    frames loaded on demand.

    ``type_map`` gives the element of each LAMMPS type or label
    (``{1: "O", 2: "Si"}``) when the file does not say; any other keyword
    goes to the reader. Used as a context manager, the file is closed at the
    end of the block.
    """

    def __init__(self, source, **read_options):
        from ..core import md_readers

        if isinstance(source, (str, Path)):
            if not Path(source).exists():
                raise FileNotFoundError(f"no file {source}")
        self.read_options = dict(read_options)
        self.trajectory = md_readers.read_trajectory(
            str(source) if isinstance(source, (str, Path)) else source,
            **read_options)
        self.name = Path(self.trajectory.source_path).name

    # -- what was read ---------------------------------------------------------------
    n_frames = property(lambda self: int(self.trajectory.n_frames))
    n_atoms = property(lambda self: int(self.trajectory.n_atoms))
    path = property(lambda self: self.trajectory.source_path)

    def describe(self) -> list[str]:
        """What was read and what was assumed, one statement per line."""
        return list(self.trajectory.describe())

    def frame(self, k: int = -1):
        """One frame (the engine's ``Frame``): ``.elements``, ``.frac``,
        ``.cart_ang``, ``.box_ang``, ``.composition``, ``.time_ps``.
        ``k`` counts from 0, negative from the end."""
        return self.trajectory.frame(self._index(k))

    def _index(self, k: int) -> int:
        n = self.n_frames
        if not -n <= int(k) < n:
            raise IndexError(f"frame {k} of a model with {n} frame(s)")
        return int(k) % n

    @property
    def elements(self) -> list[str]:
        """The elements present, in the order of the first frame."""
        return list(self.frame(0).species)

    def composition(self, k: int = -1) -> dict[str, int]:
        """Atom counts per element."""
        return dict(self.frame(k).composition)

    def density(self, k: int = -1) -> float:
        """The mass density of a frame, g/cm3."""
        from ..core import glass

        return glass.density_g_per_cm3(self.frame(k))

    def oxidation_states(self, ox: Mapping[str, int] | None = None) -> dict[str, int]:
        """The oxidation state each element takes in the analyses (the usual
        one unless ``ox`` says otherwise)."""
        from ..core import md_model

        return dict(md_model.model_oxidation(self.frame(0).species,
                                             dict(ox or {})).ox)

    def __repr__(self) -> str:
        return (f"<Model {self.name!r}: {self.n_atoms} atoms, "
                f"{self.n_frames} frame(s)>")

    def __str__(self) -> str:
        return "\n".join(self.describe())

    def close(self) -> None:
        close = getattr(self.trajectory, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- the analyses -------------------------------------------------------------------
    def request(self, *names: str, **options):
        """The request :meth:`analyse` would run (see :func:`request`)."""
        if "type_map" in self.read_options and "type_map" not in options:
            options = {**options, "type_map": self.read_options["type_map"]}
        return request(*names, **options)

    def missing(self, *names: str, **options) -> list[str]:
        """The inputs these analyses still lack, one statement each (an
        empty list: the request can run)."""
        from ..core import md_analysis as ma

        return [m.describe() for m in ma.missing_inputs(
            self.request(*names, **options))]

    def available(self, **options) -> tuple[list[str], dict[str, list[str]]]:
        """(the analyses that have every input they need on this model, the
        others with what each lacks)."""
        from ..core import md_analysis as ma

        runnable, left = ma.available_analyses(self.request(**options),
                                               self.trajectory)
        return list(runnable), {k: list(v) for k, v in left.items()}

    def analyse(self, *names: str, progress=None, **options) -> Results:
        """Run analyses on the model and return their :class:`Results`.

        ``names``: analyses from :func:`analyses`; with none, every analysis
        that has the inputs it needs is run. ``options``: the inputs (see
        :func:`request`). ``progress``: ``True`` prints the stage as it
        goes; a callable ``(done, total, stage)`` receives it.
        """
        from ..core import md_analysis as ma

        req = self.request(*names, **options)
        if not req.analyses:
            runnable, left = ma.available_analyses(req, self.trajectory)
            if not runnable:
                raise ValueError("no analysis has every input it needs: "
                                 + "; ".join(f"{n} needs {', '.join(v)}"
                                             for n, v in left.items()))
            req = dataclasses.replace(req, analyses=runnable)
        if progress is True:
            def progress(done, total, stage):          # noqa: F811
                print(f"  {done}/{total}  {stage}", flush=True)
        return Results(ma.analyse(self.trajectory, req,
                                  progress=progress or None))

    def preset(self, name: str) -> dict:
        """What a preset of the Model workspace sets on this model, as the
        keywords of :meth:`analyse`: ``m.analyse(*p.pop("analyses"), **p)``.
        The notes say what the model left out."""
        from ..core import md_presets

        first = self.frame(0)
        summary = _Summary(first.species, self.n_frames,
                           getattr(self.trajectory, "times_ps", None) is not None,
                           first.vel_ang_per_ps is not None)
        applied = md_presets.apply_preset(md_presets.preset_named(name), summary)
        out: dict = {"analyses": list(applied.analyses),
                     "formers": sorted(applied.formers) or "none",
                     "notes": list(applied.notes)}
        for key, value in applied.options.items():
            group, _, option = key.partition(".")
            out.setdefault(group, {})[option] = value
        return out

    # -- one frame, atom by atom ------------------------------------------------------------
    def _frame_analysis(self, k: int, v_bond: float, v_list: float,
                        ox: Mapping[str, int] | None, params=None):
        from ..core import bulk, md_model
        from .crystal import parameters

        frame = self.frame(k)
        states = md_model.model_oxidation(frame.species, dict(ox or {}))
        ox_atom = states.per_atom(frame.elements)
        table, atoms = bulk.analyse_frame(frame, ox_atom, parameters(params),
                                          v_bond_vu=float(v_bond),
                                          v_list_vu=float(v_list))
        return frame, np.asarray(ox_atom), table, atoms

    def atoms(self, k: int = -1, v_bond: float = V_BOND, v_list: float = V_LIST,
              ox: Mapping[str, int] | None = None, params=None) -> Table:
        """Every atom of a frame with what FACET measures on a site: the
        coordination number, the bond-valence sum, phi and the width of the
        plateau the threshold sits on. Atoms are numbered from 1."""
        frame, ox_atom, _, a = self._frame_analysis(k, v_bond, v_list, ox, params)
        cart = np.asarray(frame.cart_ang, float)
        rows = []
        for i, element in enumerate(np.asarray(frame.elements).astype(str)):
            rows.append({
                "atom": i + 1, "element": element, "ox": int(ox_atom[i]),
                "x / Å": float(cart[i, 0]), "y / Å": float(cart[i, 1]),
                "z / Å": float(cart[i, 2]),
                "CN": int(a.cn[i]), "CN listed": int(a.cn_listed[i]),
                "BVS / v.u.": float(a.bvs_vu[i]),
                "BVS listed / v.u.": float(a.bvs_listed_vu[i]),
                "BVS - |ox| / v.u.": float(a.valence_discrepancy_vu[i]),
                "phi": float(a.phi[i]),
                "plateau / decades": float(a.plateau_decades[i]),
            })
        return Table(rows, f"atoms of {self.name}, frame {self._index(k)}",
                     [f"bond threshold {v_bond:g} v.u., listing threshold "
                      f"{v_list:g} v.u."] + list(a.notes))

    def site(self, atom: int, k: int = -1, v_bond: float = V_BOND,
             v_list: float = V_LIST, ox: Mapping[str, int] | None = None):
        """One atom of a frame read as a site: its contacts, plateaus, CN
        function and angles, as for a site of a crystal. ``atom`` is the
        number in :meth:`atoms` (from 1)."""
        from ..core import md_site
        from .crystal import SiteView

        frame, _, table, _ = self._frame_analysis(k, v_bond, v_list, ox)
        row = int(atom) - 1
        if not 0 <= row < frame.n_atoms:
            raise IndexError(f"atom {atom} of a frame with {frame.n_atoms} atoms")
        return SiteView(md_site.site_result_for_atom(
            frame, table, row, v_bond=float(v_bond), v_list=float(v_list),
            k=self._index(k)), self)

    def environments(self, element: str, k: int = -1, formers: Sequence[str] = (),
                     v_bond: float = V_BOND, tol_distance: float = 0.05,
                     tol_angle: float = 5.0) -> tuple[Table, Table]:
        """The environments of every atom of an element in a frame, grouped
        where their signatures and geometries agree: (one row per group,
        one row per atom)."""
        from ..core import bulk, md_environments as me

        frame, _, table, _ = self._frame_analysis(k, v_bond, V_LIST, None)
        bonds = bulk.bonds_at(table, float(v_bond))
        envs = me.environments(frame, bonds, element=str(element),
                               formers=tuple(formers))
        groups = me.group_environments(envs, tol_distance_ang=float(tol_distance),
                                       tol_angle_deg=float(tol_angle))
        return (Table(me.group_rows(groups), f"environments of {element}"),
                Table(me.per_atom_rows(frame, envs, groups),
                      f"environment of each {element} atom"))

    def write_environment(self, atom: int, path, shells: int = 1, k: int = -1,
                          v_bond: float = V_BOND) -> Path:
        """Write one atom with the atoms within ``shells`` bonds of it,
        unwrapped about it: ``.xyz``, or extended XYZ for ``.extxyz``."""
        from ..core import bulk, md_environment as env

        frame, _, table, _ = self._frame_analysis(k, v_bond, V_LIST, None)
        bonds = bulk.bonds_at(table, float(v_bond))
        centre = int(atom) - 1
        keep, shell = env.neighbour_shells(bonds, frame.n_atoms, centre,
                                           int(shells))
        path = Path(path)
        if path.suffix.lower() == ".extxyz":
            return env.write_environment_extxyz(path, frame, keep, centre)
        return env.write_environment_xyz(path, frame, keep, centre, shell=shell)

    def basicity(self, k: int = -1, scale: str = "pauling",
                 formers: Sequence[str] | None = None,
                 ox: Mapping[str, int] | None = None, v_bond: float = V_BOND):
        """Optical basicity of a frame: lambda of every anion, the classes,
        the linkages, C(r), the field (``facet.basicity.FrameResult``)."""
        from .. import basicity as fb

        result = fb.frame(self.frame(k), scale, v_bond=float(v_bond),
                          formers=formers, ox=ox)
        result.source, result.index = self.name, self._index(k)
        return result

    def density_map(self, elements: Sequence[str], k: int = -1,
                    grid: float = 1.0, sigma: float = 1.5):
        """The smoothed number density of some elements over the box (the
        engine's ``HotMap``): where the modifiers gather."""
        from ..core import md_hotmap

        return md_hotmap.modifier_density_grid(
            self.frame(k), tuple(str(e) for e in elements),
            grid_spacing_ang=float(grid), sigma_ang=float(sigma))

    def parity(self, table_path, k: int = 0, mapping="CN,BVS",
               atom_column: str = "atom", one_based: bool = False,
               tolerance: float = 1e-3, v_bond: float = V_BOND) -> Table:
        """Another code's per-atom table for one frame beside FACET's values
        for the same frame, quantity by quantity. ``mapping`` names their
        columns: ``"CN,BVS"`` when they carry the quantities' names, or
        ``"coord=CN,sum=BVS"`` (their column = FACET's quantity; the
        quantities are CN, CN_listed, BVS, BVV, phi, plateau)."""
        from ..core import md_parity

        theirs = md_parity.read_table(Path(table_path))
        if isinstance(mapping, str):
            mapping = md_parity.parse_mapping(mapping)
        rows = md_parity.compare(theirs, self.trajectory, self._index(k), mapping,
                                 atom_column=atom_column, one_based=one_based,
                                 tolerance=float(tolerance), v_bond=float(v_bond))
        return Table((dataclasses.asdict(r) for r in rows),
                     f"{Path(table_path).name} against frame {self._index(k)}")


@dataclasses.dataclass(frozen=True)
class _Summary:
    """What a preset reads of a model."""

    species: tuple
    n_frames: int
    has_times: bool
    has_velocities: bool


def model(source, **read_options) -> Model:
    """Open an MD model. ``type_map={1: "Si", 2: "O"}`` names the elements
    of LAMMPS types; other keywords go to the reader."""
    return Model(source, **read_options)


def load(source, **options):
    """A :class:`Model` when the file is an MD model, a ``Crystal``
    otherwise."""
    from ..basicity import is_md_file
    from .crystal import read

    if is_md_file(source):
        return model(source, **options)
    return read(source, **options)


__all__ = ["Model", "Results", "model", "load", "analyses", "presets", "request",
           "options"]
