"""Presets of the Model workspace's Setup page: a question, as a request.

A preset is what a user who opens a glass model and asks one question
would tick by hand: the analyses that answer it, the network formers among
the cations present, and the method inputs those analyses require and have
no default for (the ring criterion and largest ring, the void radii, the
free-volume probe and grid, the scattering window and radiations). Nothing
physical is filled in: no timestep, no temperature, no charges, no measured
curve, no absorber. What a preset needs and cannot give is listed in its
``needs`` and said back as a note when it is applied.

A preset is an explicit user action, so :func:`apply_preset` may fill the
formers; it never names an element the model does not hold. "Custom"
changes nothing. Everything here is Qt-free and reads only
:mod:`facet.core.md_analysis` and :mod:`facet.core.elements`.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from . import elements, md_analysis as ma

__all__ = [
    "Preset", "PresetApplication", "PRESETS", "FAMILY_FORMERS", "HALIDES",
    "NEEDS", "preset_named", "apply_preset", "option_readers",
]

# The cations a glass family's network is built from; "the formers of the
# family detected" is this set intersected with the cations present.
FAMILY_FORMERS: tuple[str, ...] = ("Si", "Al", "B", "P", "Ge")
HALIDES: tuple[str, ...] = ("F", "Cl", "Br", "I")

# What a preset can need beyond the model's atoms and the method inputs it
# fills: each is a model property or a measured input only the user has.
NEEDS = {
    "time axis": "a time axis (timestep_fs or frame_interval_ps, or frame "
                 "times in the file)",
    "velocities": "velocities in the file (the VACF and the kinetic "
                  "temperature read them)",
    "measured curve": "a measured S(Q) or G(r) file (scattering.measured)",
    "absorber": "the absorber element (exafs.absorber)",
    "correlation": "a published shift correlation (nmr.correlation)",
    "measured fractions": "measured Qn, N4 or CN fractions "
                          "(nmr.measured_fractions)",
    "charges and temperature": "the ionic charges and the temperature "
                               "(charges_e, temperature_k) of the "
                               "conductivity",
}

# Which analyses read each method input a preset may fill ('group.field'),
# so an input is only set when one of its readers is ticked.
_OPTION_READERS = {
    "network.ring_criterion": ("rings",),
    "network.ring_max_size": ("rings",),
    "network.n_shells": ("coordination-sequences",),
    "voids.radii": ("empty-spheres", "free-volume"),
    "voids.probe_radius_ang": ("free-volume",),
    "voids.grid_spacing_ang": ("free-volume",),
    "scattering.r_window": ("scattering", "scattering-comparison"),
    "scattering.radiations": ("scattering", "scattering-comparison"),
}

# The method inputs, as the text a request file or the Setup field takes.
# Each is required by its reader (no default in md_analysis) and is a method
# choice, not a physical input: King's criterion to 12 graph nodes, FACET's
# van der Waals radii, a 0.5 Å probe on a 0.5 Å grid, the Lorch window for
# both radiations, six coordination shells.
_RINGS = {"network.ring_criterion": "king", "network.ring_max_size": "12"}
_VOIDS = {"voids.radii": "vdw"}
_FREE = {"voids.probe_radius_ang": "0.5", "voids.grid_spacing_ang": "0.5"}
_SCATTER = {"scattering.r_window": "Lorch",
            "scattering.radiations": "neutron, X-ray"}
_CSEQ = {"network.n_shells": "6"}


@dataclass(frozen=True)
class Preset:
    """One question, as the analyses, formers and method inputs it takes.

    ``formers`` are the candidate former elements (the model's cations
    among them are ticked); ``family`` True takes :data:`FAMILY_FORMERS`
    instead. ``options`` maps 'group.field' to the text the field takes.
    ``needs`` names what the preset needs that it cannot fill
    (:data:`NEEDS`). ``halide`` True notes the halide present (or absent).
    """

    name: str
    description: str
    analyses: tuple[str, ...] = ()
    formers: tuple[str, ...] = ()
    options: Mapping[str, str] = field(default_factory=dict)
    needs: tuple[str, ...] = ()
    family: bool = False
    halide: bool = False

    @property
    def custom(self) -> bool:
        return self.name == "Custom"


@dataclass(frozen=True)
class PresetApplication:
    """What applying a preset to one model sets: the analyses to tick, the
    formers to tick (a frozenset; empty leaves the picker as it is), the
    method inputs to fill, and the notes to show. ``changes`` is False for
    Custom, which leaves everything as it is."""

    analyses: tuple[str, ...]
    formers: frozenset[str]
    options: dict[str, str]
    notes: tuple[str, ...]
    changes: bool = True


def _merged(*parts: Mapping[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in parts:
        out.update(part)
    return out


_STRUCTURE = ("glass", "rings", "components", "polyhedral-sharing")

PRESETS: tuple[Preset, ...] = (
    Preset("Silicate",
           "Coordination, speciation and Qn, rings, connectivity, polyhedral "
           "sharing, tetrahedral order and empty spheres over Si.",
           _STRUCTURE + ("tetrahedral-order", "empty-spheres"),
           ("Si",), _merged(_RINGS, _VOIDS)),
    Preset("Aluminosilicate",
           "The silicate question over Si and Al, with the Steinhardt bond "
           "order and the polyhedron shape (Al CN and its distortion).",
           _STRUCTURE + ("bond-order", "polyhedron-shape", "empty-spheres"),
           ("Si", "Al"), _merged(_RINGS, _VOIDS)),
    Preset("Borosilicate",
           "The silicate question over Si and B, with the polyhedron shape "
           "(N4, the BO3 and BO4 units).",
           _STRUCTURE + ("polyhedron-shape", "empty-spheres"),
           ("Si", "B"), _merged(_RINGS, _VOIDS)),
    Preset("Phosphate",
           "Speciation and Qn over P, connectivity, polyhedral sharing and "
           "the polyhedron shape; no rings (phosphate chains).",
           ("glass", "components", "polyhedral-sharing", "polyhedron-shape"),
           ("P",)),
    Preset("Oxyfluoride",
           "Speciation over the formers present with the anion environments "
           "and the Warren-Cowley order of the halide around them.",
           ("glass", "components", "polyhedral-sharing", "warren-cowley"),
           ("Si", "Al", "P", "B"), halide=True),
    Preset("Ion conduction / channels",
           "Where a mobile ion can move: the voids (Voronoi cells, empty "
           "spheres, free volume, void regions), the channels by charge "
           "(bond-valence landscape of the probe ion) and by modifier "
           "density, the mean-square displacement, the bond lifetimes and "
           "the conductivity, over the formers of the family present. The "
           "probe ion, its Delta thresholds, the modifier set, k_rich and "
           "the void probe radius are the user's (no default).",
           ("glass", "voronoi", "empty-spheres", "free-volume", "channels",
            "modifier-density", "void-regions", "msd", "bond-lifetimes",
            "conductivity"),
           options=_merged(_VOIDS, _FREE), family=True,
           needs=("time axis", "charges and temperature")),
    Preset("Scattering vs experiment",
           "Total scattering (neutron and X-ray, Lorch window) and its "
           "comparison with a measured curve, with the glass descriptors.",
           ("glass", "scattering", "scattering-comparison"),
           options=_SCATTER, family=True, needs=("measured curve",)),
    Preset("Dynamics",
           "Mean-square displacement, self correlations, bond lifetimes, the "
           "VACF and the kinetic temperature.",
           ("msd", "self-correlations", "bond-lifetimes", "vacf",
            "kinetic-temperature"),
           family=True, needs=("time axis", "velocities")),
    Preset("Everything",
           "Every analysis FACET runs, with every method input filled; what "
           "needs a measured input or a time axis says so.",
           tuple(ma.ANALYSES),
           options=_merged(_RINGS, _VOIDS, _FREE, _SCATTER, _CSEQ),
           family=True,
           needs=("time axis", "velocities", "measured curve", "absorber",
                  "correlation", "measured fractions",
                  "charges and temperature")),
    Preset("Custom", "Leaves every tick and field as it is."),
)


def preset_named(name: str) -> Preset:
    for preset in PRESETS:
        if preset.name == name:
            return preset
    raise KeyError(f"no preset named {name!r}; the presets are "
                   + ", ".join(p.name for p in PRESETS))


def option_readers(key: str) -> tuple[str, ...]:
    """The analyses that read the method input ``key`` ('group.field')."""
    return _OPTION_READERS.get(key, ())


def _cations(summary, states: Mapping[str, int] | None = None) -> list[str]:
    """The cations of the model: the species whose oxidation state (the
    given states, else elements.COMMON_OX) is positive."""
    states = dict(states or {})
    out = []
    for symbol in summary.species:
        state = states.get(symbol, elements.COMMON_OX.get(symbol))
        if state is not None and state > 0:
            out.append(str(symbol))
    return out


def apply_preset(preset: Preset, summary,
                 states: Mapping[str, int] | None = None
                 ) -> PresetApplication:
    """What ``preset`` sets on the model ``summary`` describes
    (:class:`facet.ui.md_jobs.ModelSummary`, or anything with ``species``,
    ``n_frames``, ``has_times`` and ``has_velocities``).

    Restricted to what the model allows: an analysis of time is left out on
    a one-frame model, and one that reads the file's velocities when the
    file holds none; both are noted. A time axis the file lacks is noted,
    not filled. The formers are the preset's candidates among the cations
    present (``states`` overrides elements.COMMON_OX for who is a cation);
    none present is noted and leaves the picker as it is. A 'channels'
    analysis is ticked by the ion-conduction preset when md_analysis offers
    one, and noted otherwise.
    """
    if preset.custom:
        return PresetApplication((), frozenset(), {},
                                 ("Custom: nothing is changed.",), False)
    notes: list[str] = []
    present = tuple(str(s) for s in summary.species)
    cations = _cations(summary, states)
    candidates = FAMILY_FORMERS if preset.family else preset.formers
    formers = frozenset(s for s in candidates if s in cations)
    if candidates and not formers:
        notes.append(f"No former of this preset ({', '.join(candidates)}) "
                     "is among the cations present "
                     f"({', '.join(cations) or 'none'}); the formers are "
                     "left as they are.")
    if preset.halide:
        halides = [h for h in HALIDES if h in present]
        notes.append(f"Halide present: {', '.join(halides)}." if halides
                     else "No halide (F, Cl, Br, I) is among the species; "
                          "the anion environments are those of the oxide.")

    analyses = [a for a in preset.analyses if a in ma.ANALYSES]
    n_frames = int(getattr(summary, "n_frames", 1))
    has_times = bool(getattr(summary, "has_times", False))
    has_velocities = bool(getattr(summary, "has_velocities", False))
    timed = [a for a in analyses if a in ma._TIMED]
    if timed and n_frames < 2:
        notes.append("One frame: " + ", ".join(timed) + " (analyses of "
                     "time) are left out.")
        analyses = [a for a in analyses if a not in ma._TIMED]
    elif timed and not has_times:
        notes.append("The file states no frame times: " + ", ".join(timed)
                     + " need the MD timestep (fs) or the time between "
                       "frames (ps).")
    needs_velocities = [a for a in analyses
                        if a in ("vacf", "kinetic-temperature")]
    if needs_velocities and not has_velocities:
        notes.append("The file holds no velocities: "
                     + ", ".join(needs_velocities) + " are left out "
                     "(dynamics.velocities 'finite difference' gives the "
                     "VACF from the positions).")
        analyses = [a for a in analyses if a not in needs_velocities]
    if preset.name.startswith("Ion conduction"):
        if "channels" in ma.ANALYSES:
            if "channels" not in analyses:
                analyses.append("channels")
        else:
            notes.append("No 'channels' analysis is in this build's "
                         "md_analysis; the Highlight page draws the "
                         "channels of the frame shown.")
    for need in preset.needs:
        if need == "time axis" and (has_times or n_frames < 2):
            continue
        if need == "velocities" and has_velocities:
            continue
        if need == "charges and temperature" and \
                "conductivity" not in analyses:
            continue
        notes.append(f"Needs {NEEDS.get(need, need)}.")

    chosen = set(analyses)
    options = {key: text for key, text in preset.options.items()
               if set(_OPTION_READERS.get(key, ())) & chosen}
    ordered = tuple(a for a in ma.ANALYSES if a in chosen)
    return PresetApplication(ordered, formers, options, tuple(notes))
