"""The Model window's setup: what to read, and what to measure.

Two panels, both plain widgets the Model window puts in its own pages (a
modal dialog would block the window, and a test, until it closes):

* :class:`ReadOptionsPanel`, shown when ``md_readers.read_trajectory``
  refuses a file for want of an input only the user has: a type map (with
  the reader's own evidence per type: its label, element column or mass,
  and the atoms of each type in frame 0), masses from a LAMMPS data file, a
  topology (DCD, XTC, AMBER NetCDF), a box (a plain XYZ), units or column
  names. It shows the reader's message in full, asks for the options the
  message names, and keeps every other option the format takes one click
  away. No element is filled in for the user.
* :class:`SetupPanel`, shown once the model is read: the load summary, the
  oxidation states (elements.COMMON_OX shown with its source, each one
  editable), the network-former picker (the cations present, none ticked),
  the frames, the thresholds and an analysis checklist grouped by the module
  that measures each analysis, with every option group of
  ``md_analysis.AnalysisRequest`` beside the analyses that read it. An
  analysis whose inputs are missing is disabled, with what it needs written
  beside it (``md_analysis.missing_inputs``, plus what the model shows:
  the frame count, the time axis, velocities, charges).

Values typed into the option fields are read by the command line's own
converter (``facet.md.cli.convert``), so a field takes exactly the text a
request file or ``--set group.option=value`` takes, and the field's tool tip
names it as ``group.option``.
"""
from __future__ import annotations

import dataclasses
import functools
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import chrome
from .md_jobs import READ_OPTION_TEXT, ModelSummary, ReadProblem

__all__ = [
    "ReadOptionsPanel", "SetupPanel", "OxidationTable", "FormerPicker",
    "FrameRange", "AnalysisGroup", "OptionField", "MODULE_GROUPS",
    "OPTION_GROUPS", "module_of", "unit_of", "label_of",
]

# Text colours of the inline reasons. On a dark window, Okabe-Ito orange
# (230, 159, 0; Okabe and Ito, Color Universal Design, 2008). On a light
# window, Okabe-Ito vermillion (213, 94, 0) darkened to (176, 74, 0) for text
# contrast on white: about 5.5:1 against 3.9:1 for the published value, below
# the 4.5:1 body text needs. The word 'needs' carries the meaning; the colour
# only repeats it.
_NEEDS_ON_LIGHT = "#b04a00"
_NEEDS_ON_DARK = "#e69f00"


def _needs_colour(theme=None) -> str:
    return _NEEDS_ON_LIGHT if chrome.ui_colors(theme).is_light \
        else _NEEDS_ON_DARK


def _hint(text: str = "", *, wrap: bool = True) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(wrap)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    chrome.mark_hint(label)
    return label


def _escape(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


# ---------------------------------------------------------------------------
# names, units and groups (Qt-free)
# ---------------------------------------------------------------------------

# The modules the driver runs, in its order, and their titles. Each
# analysis's module is the one its ANALYSIS_SUMMARIES text names.
MODULE_GROUPS = (
    ("glass", "Glass descriptors"),
    ("md_scattering", "Total scattering"),
    ("md_network", "Network: rings, connectivity, chemical order"),
    ("md_order", "Local order, Voronoi cells and voids"),
    ("md_spectroscopy", "Spectroscopy: NMR, EXAFS, FEFF"),
    ("md_dynamics", "Dynamics"),
)

# The option groups of AnalysisRequest each module box holds, and the
# top-level request fields shown with them.
OPTION_GROUPS = {
    "glass": ("glass",),
    "md_scattering": ("scattering",),
    "md_network": ("network",),
    "md_order": ("order", "voids"),
    "md_spectroscopy": ("nmr", "exafs", "feff"),
    "md_dynamics": ("dynamics",),
}
TOP_FIELDS = {
    "glass": ("bridging_anions",),
    "md_dynamics": ("timestep_fs", "frame_interval_ps", "temperature_k",
                    "charges_e"),
}

# The analyses that read each option group of AnalysisRequest (the driver's
# consumers: EXAFS reads the glass minimum rule and g(r) radius, FEFF the
# EXAFS absorber), and the top-level fields only the dynamics read.
GROUP_READERS = {
    "glass": ("glass", "exafs", "feff"),
    "scattering": ("scattering", "scattering-comparison"),
    "network": ("rings", "coordination-sequences", "polyhedral-sharing",
                "components", "warren-cowley"),
    "order": ("bond-order", "tetrahedral-order", "polyhedron-shape",
              "voronoi"),
    "voids": ("empty-spheres", "free-volume"),
    "nmr": ("nmr", "nmr-comparison"),
    "exafs": ("exafs", "feff"),
    "feff": ("feff",),
    "dynamics": ("msd", "self-correlations", "distinct-van-hove", "vacf",
                 "kinetic-temperature", "conductivity", "bond-lifetimes"),
}
DYNAMICS_TOP_FIELDS = ("timestep_fs", "frame_interval_ps", "temperature_k",
                       "charges_e")


def run_request(base, names: Sequence[str]):
    """``base`` with ``names`` as its analyses, as the setup sends it.

    The engine refuses a value no analysis can take wherever it stands
    (``md_analysis.invalid_inputs``), so a value typed for an analysis that
    is not ticked used to block every run. Such a value is left out of the
    request sent: the option group it sits in goes back to its defaults
    when no analysis of the run reads that group (:data:`GROUP_READERS`),
    and a dynamics field (timestep, temperature, charges) to unset when no
    dynamics analysis runs. A value an analysis of the run reads is never
    touched, and stays refused with its reason.
    """
    from ..core import md_analysis as ma

    request = dataclasses.replace(base, analyses=tuple(names))
    chosen = set(request.analyses)
    groups: set[str] = set()
    tops: set[str] = set()
    for m in ma.invalid_inputs(request):
        readers = set(m.analysis.split("/"))
        if m.analysis == "request" or readers & chosen:
            continue
        for group, field_name in field_keys(m.name):
            word = field_name.split()[0] if field_name.split() else ""
            if group is None and word in GROUP_READERS:
                group = word
            if group in GROUP_READERS:
                if not set(GROUP_READERS[group]) & chosen:
                    groups.add(group)
            elif group is None and word in DYNAMICS_TOP_FIELDS and \
                    not set(GROUP_READERS["dynamics"]) & chosen:
                tops.add(word)
    if not groups and not tops:
        return request
    changes = {g: group_class(g)() for g in groups}
    changes.update({f: None for f in tops})
    return dataclasses.replace(request, **changes)


def module_of(name: str) -> str:
    """The module that measures analysis ``name``, read from the last
    parenthesis of its summary ('... (md_network)')."""
    from ..core import md_analysis as ma

    text = ma.ANALYSIS_SUMMARIES.get(name, "")
    if text.endswith(")") and "(" in text:
        tag = text[text.rfind("(") + 1:-1].strip()
        if tag in dict(MODULE_GROUPS):
            return tag
    return "glass"


def group_class(group: str):
    from ..core import md_analysis as ma

    return {"glass": ma.GlassOptions, "scattering": ma.ScatteringOptions,
            "network": ma.NetworkOptions, "order": ma.OrderOptions,
            "voids": ma.VoidOptions, "nmr": ma.NmrOptions,
            "exafs": ma.ExafsOptions, "feff": ma.FeffOptions,
            "dynamics": ma.DynamicsOptions}[group]


_UNIT_SUFFIXES = (("_inv_ang", "Å⁻¹"), ("_ang3", "Å³"), ("_ang2", "Å²"),
                  ("_ang", "Å"), ("_deg2", "deg²"), ("_deg", "deg"),
                  ("_fs", "fs"), ("_ps", "ps"), ("_fm", "fm"),
                  ("_ppm", "ppm"), ("_vu", "v.u."), ("_k", "K"),
                  ("_e", "e"))

_LABELS = {
    "rdf_r_max_ang": "g(r) r max", "rdf_dr_ang": "g(r) step",
    "minimum_rule": "first-minimum rule",
    "minimum_smooth_sigma_ang": "first-minimum smoothing sigma",
    "minimum_flat_rule": "first-minimum flat-floor rule",
    "minimum_margin_std_errors": "first-minimum margin (standard errors)",
    "cutoffs_ang": "cation-anion cutoffs", "oxide_basis": "oxide basis",
    "r_window": "r-window M(r)", "radiations": "radiations",
    "q_step_inv_ang": "Q step", "q_grid_max_inv_ang": "Q grid max",
    "termination_q_max_inv_ang": "termination Q max",
    "termination_q_min_inv_ang": "termination Q min",
    "q_window": "Q window", "fsdp_window_inv_ang": "FSDP window",
    "fsdp_baseline": "FSDP baseline",
    "neutron_lengths_fm": "neutron scattering lengths",
    "lengths_source": "source of the lengths",
    "measured": "measured curves", "graph": "graph",
    "graph_elements": "graph elements",
    "distance_cutoffs_ang": "distance cutoffs",
    "ring_criterion": "ring criterion",
    # what a node is on the bridging-anion graph is in the field's
    # placeholder (_NO_DEFAULT_NOTES): as part of the label it set the
    # label column of the whole network box, its fields 165 px wide
    "ring_max_size": "largest ring (graph nodes)",
    "ring_t_elements": "ring T elements", "n_shells": "coordination shells",
    "cseq_centres": "sequence centres", "sharing_centres": "sharing centres",
    "sharing_ligands": "sharing ligands",
    "wc_elements": "Warren-Cowley elements",
    "wc_cutoffs_ang": "Warren-Cowley cutoffs",
    "neighbours": "neighbours", "degrees": "Steinhardt degrees l",
    "q_tet_selection": "q_tet neighbours", "radii": "atom radii",
    "radii_source": "source of the radii",
    "probe_radius_ang": "probe radius", "grid_spacing_ang": "grid spacing",
    "sphere_bin_ang": "empty-sphere bin", "correlation": "correlation file",
    "delta_ppm": "shift axis (first, last, step)", "fwhm_ppm": "line width",
    "lineshape": "line shape", "measured_fractions": "measured fractions",
    "absorber": "absorber", "first_shell_limits_ang": "first-shell limits",
    "shells": "shells (element:r_lo:r_hi)", "weighting": "weighting",
    "out_dir": "output folder", "n_clusters": "clusters",
    "seed": "random seed", "cluster_radius_ang": "cluster radius",
    "edge": "edge", "r_path_ang": "path r max", "overwrite": "overwrite",
    "unwrap": "unwrap", "step_limit_fraction": "step limit (box widths)",
    "elements": "elements", "n_blocks": "blocks",
    "remove_com_drift": "remove the centre-of-mass drift",
    "max_lag_t_ps": "largest lag", "fit_t_min_ps": "fit window from",
    "fit_t_max_ps": "fit window to", "lag_t_ps": "lag times",
    "van_hove_edges_r_ang": "van Hove r edges (first, last, step)",
    "isf_q_inv_ang": "ISF Q values", "distinct_pairs": "distinct pairs",
    "distinct_n_origins": "time origins per lag",
    "velocities": "velocities", "vdos_window": "VDOS window",
    "green_kubo_t_max_ps": "Green-Kubo t max",
    "lifetime_max_lag_t_ps": "bond lifetime largest lag",
    "gap_tolerance_frames": "gap tolerance (frames)",
    "residence_method": "residence-time method",
    "residence_t_min_ps": "residence fit from",
    "residence_t_max_ps": "residence fit to",
    "bridging_anions": "bridging anions", "timestep_fs": "MD timestep",
    "frame_interval_ps": "time between frames",
    "temperature_k": "temperature", "charges_e": "ionic charges",
    "v_bond_vu": "v_bond", "v_list_vu": "v_list",
}

# What a blank field means when the engine's value is None and None means
# an automatic or absent value rather than a missing input.
_BLANK_MEANS = {
    ("glass", "rdf_r_max_ang"): "the bond-valence search radius",
    ("glass", "minimum_smooth_sigma_ang"): "no smoothing",
    ("glass", "cutoffs_ang"): "the g(r) first minima; e.g. Si-O=2.1",
    ("glass", "oxide_basis"): "none; e.g. Si=SiO2, Na=Na2O",
    ("scattering", "r_max_ang"): "the largest grid the box holds",
    ("scattering", "termination_q_max_inv_ang"): "no termination",
    ("scattering", "termination_q_min_inv_ang"): "the first Q of the grid "
                                                 "(Q step)",
    ("scattering", "q_window"): "none",
    ("scattering", "fsdp_window_inv_ang"): "no FSDP; e.g. 1.0, 2.0",
    ("scattering", "fsdp_baseline"): "none",
    ("scattering", "neutron_lengths_fm"): "FACET's table; e.g. Si=4.1491",
    ("scattering", "lengths_source"): "FACET's table",
    ("network", "graph_elements"): "the formers and bridging anions",
    ("network", "distance_cutoffs_ang"): "the glass first minima",
    ("network", "ring_t_elements"): "the formers",
    ("network", "cseq_centres"): "the formers",
    ("network", "sharing_centres"): "the formers",
    ("network", "sharing_ligands"): "the model's anions",
    ("network", "wc_elements"): "every element",
    ("network", "wc_cutoffs_ang"): "the glass first minima",
    ("order", "distance_cutoffs_ang"): "the glass cation-anion cutoffs",
    ("voids", "radii_source"): "needed with a radius map",
    ("exafs", "r_max_ang"): "the glass g(r) radius",
    ("exafs", "neighbours"): "every element",
    ("exafs", "first_shell_limits_ang"): "first minimum of the absorber g(r)",
    ("feff", "r_path_ang"): "not given",
    ("dynamics", "elements"): "every element",
    ("dynamics", "max_lag_t_ps"): "md_dynamics's value",
    ("dynamics", "isf_q_inv_ang"): "not computed",
    ("dynamics", "green_kubo_t_max_ps"): "not computed",
    ("dynamics", "lifetime_max_lag_t_ps"): "md_dynamics's value",
    ("dynamics", "residence_method"): "no residence times",
    ("dynamics", "residence_t_min_ps"): "not given",
    ("dynamics", "residence_t_max_ps"): "not given",
    (None, "bridging_anions"): "every anion of the model",
    (None, "frame_interval_ps"): "not given",
}

# What a field with no default counts, said where its value is typed.
_NO_DEFAULT_NOTES = {
    ("network", "ring_max_size"): "T atoms on the bridging-anion graph",
}

# Method choices with a default that are shown beside the inputs rather than
# folded away, because the default changes the numbers and not only their
# resolution: the centre-of-mass drift enters every MSD and every D fitted
# from it (87-91 % of each element's MSD at 19 ps in a 300 K NVT run of the
# SHIK Na-borosilicate, measured by the new-user check of 2026-10-07).
_SHOWN_CHOICES = frozenset({("dynamics", "remove_com_drift")})

# What an analysis leaves out while an option that has no default is blank:
# (what is left out, the fields that add it). Shown beside the analysis
# when it is enabled, so a run with the required inputs alone says what it
# will not give.
OPTIONAL_OUTPUTS = {
    "msd": (("diffusion coefficients",
             ("dynamics.fit_t_min_ps", "dynamics.fit_t_max_ps")),),
    "vacf": (("the Green-Kubo diffusion coefficient",
              ("dynamics.green_kubo_t_max_ps",)),),
    "self-correlations": (
        ("the self van Hove function", ("dynamics.van_hove_edges_r_ang",)),
        ("the self intermediate scattering function",
         ("dynamics.isf_q_inv_ang",))),
    "bond-lifetimes": (("residence times", ("dynamics.residence_method",)),),
    "scattering": (
        ("the terminated G(r)", ("scattering.termination_q_max_inv_ang",)),
        ("the FSDP", ("scattering.fsdp_window_inv_ang",
                      "scattering.fsdp_baseline"))),
}

# Fields whose values form a fixed list; read at run time from the module
# that defines them.
def _choices(group, name):
    from ..core import (glass, md_analysis as ma, md_dynamics, md_network,
                        md_order, md_scattering, md_spectroscopy)

    table = {
        ("glass", "minimum_rule"): glass.MINIMUM_RULES,
        ("glass", "minimum_flat_rule"): glass.FLAT_RULES,
        ("scattering", "r_window"): md_scattering.R_WINDOWS,
        ("scattering", "q_window"): md_scattering.Q_WINDOWS,
        ("scattering", "fsdp_baseline"): md_scattering.FSDP_BASELINES,
        ("network", "graph"): ma.GRAPH_KINDS,
        ("network", "ring_criterion"): md_network.CRITERIA,
        ("order", "neighbours"): ma.NEIGHBOUR_KINDS,
        ("order", "q_tet_selection"): md_order.QTET_SELECTIONS,
        ("nmr", "lineshape"): md_spectroscopy.LINESHAPES,
        ("exafs", "weighting"): md_spectroscopy.WEIGHTINGS,
        ("dynamics", "unwrap"): md_dynamics.UNWRAP_METHODS,
        ("dynamics", "velocities"): md_dynamics.VELOCITY_SOURCES,
        ("dynamics", "vdos_window"): md_dynamics.WINDOWS,
        ("dynamics", "residence_method"): md_dynamics.RESIDENCE_METHODS,
    }
    return table.get((group, name))


# Fields that take one of a few words or typed text (a map).
def _editable_choices(group, name):
    from ..core import md_analysis as ma

    table = {
        ("voids", "radii"): ma.RADII_NAMES,
        ("feff", "edge"): ("K", "L1", "L2", "L3"),
        (None, "charges_e"): ma.CHARGE_SOURCES,
    }
    return table.get((group, name))


def unit_of(name: str) -> str:
    for suffix, unit in _UNIT_SUFFIXES:
        if name.endswith(suffix):
            return unit
    return ""


def label_of(name: str) -> str:
    text = _LABELS.get(name)
    if text is None:
        text = name
        for suffix, _ in _UNIT_SUFFIXES:
            if text.endswith(suffix):
                text = text[:-len(suffix)]
                break
        text = text.replace("_", " ")
    unit = unit_of(name)
    return f"{text} ({unit})" if unit else text


def user_text(text: str) -> str:
    """An engine text as the window shows it: a developer's note in
    parentheses ('(TODO: ...)') is left out, keeping the fact it states
    that none ships with FACET."""
    def replace(match) -> str:
        return "; none ships with FACET" if "none ships" in match.group(0) \
            else ""

    return re.sub(r"\s*\(TODO:[^()]*\)", replace, str(text)).strip()


@dataclasses.dataclass(frozen=True)
class Reason:
    """One thing an analysis needs, as the setup shows it: ``text`` in
    plain words, and ``link``, the part of it that names a field, which a
    click shows (``where``: 'group.name', 'formers' or 'frames')."""

    text: str
    where: str | None = None
    link: str = ""

    def __str__(self) -> str:
        return self.text


def _reason_html(reason) -> str:
    if isinstance(reason, Reason) and reason.where and reason.link \
            and reason.link in reason.text:
        before, _, after = reason.text.partition(reason.link)
        return (_escape(before) + f"<a href='field:{reason.where}'>"
                + _escape(reason.link) + "</a>" + _escape(after))
    return _escape(str(reason))


def _field_reason(name: str, why: str, *, prefix: str = "") -> Reason:
    """A reason naming request field ``name`` (as the engine names it):
    'label (request name): why', the label linked to the field."""
    if name in ("formers",):
        # the engine's text says what the formers are for and what else
        # answers the need; 'network formers' in it links to the picker
        text = prefix + user_text(why)
        link = "network formers" if "network formers" in text else ""
        return Reason(text, "formers" if link else None, link)
    label = field_label(name)
    where = None
    keys = field_keys(name)
    if keys:
        group, first = keys[0]
        where = f"{group}.{first}" if group else first
    return Reason(f"{prefix}{label} ({name}): {user_text(why)}", where, label)


def field_label(name: str) -> str:
    """The label the setup shows for a request field named as the engine
    names it: 'scattering.r_window', 'temperature_k', or two fields of one
    group joined by '/' ('dynamics.fit_t_min_ps/fit_t_max_ps')."""
    group, _, rest = str(name).rpartition(".")
    parts = [p.strip() for p in (rest if group else name).split("/")]
    return " / ".join(label_of(p) for p in parts if p)


def field_keys(name: str) -> list[tuple]:
    """(group, field) of each request field ``name`` names (see
    :func:`field_label`); group None for a top-level field."""
    group, _, rest = str(name).rpartition(".")
    if not group:
        return [(None, p.strip()) for p in str(name).split("/")
                if p.strip()]
    return [(group, p.strip()) for p in rest.split("/") if p.strip()]


# Each choice of a fixed list, said in a line: what it means and, where the
# module measured it, what it costs. The texts restate the modules'
# docstrings (md_network, md_scattering, md_dynamics, md_order).
_CHOICE_TIPS = {
    ("network", "ring_criterion"): {
        "king": "King: for every node and every pair of its neighbours, the "
                "shortest closed paths through both bonds. md_network "
                "measured 1.6 s on a 3000-node Na2O-3SiO2 graph, rings to "
                "24 nodes.",
        "guttman": "Guttman: for every bond, the shortest closed paths "
                   "through it. 0.81 s on the same graph.",
        "primitive": "Primitive: rings that no shortcut splits into two "
                     "smaller rings. 250-428 s on the same graph; the cost "
                     "grows steeply with the largest ring.",
    },
    ("scattering", "r_window"): {
        "none": "none: M(r) = 1, the bare sine transform cut at r_max; its "
                "termination ripples stay in S(Q).",
        "Lorch": "Lorch: M(r) = sin(pi r / r_max) / (pi r / r_max); damps "
                 "the termination ripples and broadens the peaks.",
    },
    ("scattering", "q_window"): {
        "boxcar": "boxcar: S(Q) cut at the termination Q max, as pdf.py "
                  "applies it.",
        "Lorch": "Lorch: S(Q) damped towards the termination Q max, as "
                 "pdf.py applies it.",
    },
    ("scattering", "fsdp_baseline"): {
        "zero": "zero: the half height is S_peak / 2.",
        "minima": "minima: the half height is taken above the line through "
                  "the lowest point on each side of the peak within the "
                  "window.",
    },
    ("voids", "radii"): {
        "vdw": "vdw: FACET's van der Waals radii (md_order.vdw_radii_ang).",
        "zero": "zero: radii of 0, the circumspheres of the Delaunay "
                "tetrahedra (empty spheres only).",
    },
    ("dynamics", "unwrap"): {
        "auto": "auto: the file's unwrapped coordinates when the first "
                "frame carries them, else continuity.",
        "file": "file: the file's unwrapped coordinates.",
        "minimum image": "minimum image: unwrapped by continuity, each step "
                         "between two frames taken as its shortest image.",
    },
    ("dynamics", "velocities"): {
        "file": "file: the velocities the file holds.",
        "finite difference": "finite difference: velocities derived from "
                             "the positions of consecutive frames.",
    },
    ("order", "q_tet_selection"): {
        "exactly four": "exactly four: q_tet of the atoms with four "
                        "neighbours.",
        "four nearest": "four nearest: q_tet over the four nearest entries "
                        "of the neighbour list, for atoms with four or "
                        "more.",
    },
}


@functools.lru_cache(maxsize=1)
def _engine_whys() -> dict[str, str]:
    """Per request field ('group.name' or 'name'), the engine's own words
    for what it is, from ``missing_inputs`` on requests that give
    nothing."""
    from ..core import md_analysis as ma

    out: dict[str, str] = {}
    trials = [ma.AnalysisRequest(analyses=tuple(ma.ANALYSES)),
              ma.AnalysisRequest(analyses=("rings",),
                                 network=ma.NetworkOptions(graph="distance"))]
    for request in trials:
        try:
            lacking = ma.missing_inputs(request)
        except Exception:               # noqa: BLE001 - tool tips are an aid
            continue
        for m in lacking:
            if m.name in ("formers", "analyses"):
                continue
            for group, name in field_keys(m.name):
                key = f"{group}.{name}" if group else name
                out.setdefault(key, user_text(m.why))
    return out


def _doc_sentence(group: str | None, name: str) -> str:
    """The sentence of the options class's docstring that names ``name``."""
    from ..core import md_analysis as ma

    cls = ma.AnalysisRequest if group is None else group_class(group)
    doc = " ".join((cls.__doc__ or "").split())
    for sentence in re.split(r"(?<=[.;])\s+(?=[A-Z`'(])", doc):
        if f"``{name}``" in sentence:
            return sentence.replace("``", "").strip()
    return ""


def field_meaning(group: str | None, name: str) -> str:
    """What a field is, in the engine's words (``missing_inputs``), or the
    sentence of its options class that names it."""
    key = f"{group}.{name}" if group else name
    try:
        return _engine_whys().get(key) or _doc_sentence(group, name)
    except Exception:                  # noqa: BLE001 - tool tips are an aid
        return ""


def _shown(value) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (tuple, list)):
        return ", ".join(_shown(v) for v in value)
    return str(value)


def _convert(annotation: str, text, where: str):
    """A field's text, typed as the command line types it."""
    try:
        from ..md import cli
    except ImportError as error:
        raise ValueError(f"{where}: the option reader (facet.md.cli) is not "
                         f"present in this build ({error})") from None
    try:
        return cli.convert(annotation, text, where)
    except ValueError as error:          # cli.UsageError is a ValueError
        raise ValueError(str(error)) from None


# ---------------------------------------------------------------------------
# one option field
# ---------------------------------------------------------------------------

class OptionField(QWidget):
    """One field of an option group (or a top-level request field).

    ``value()`` returns the typed value, or :data:`OptionField.UNSET` when
    blank (the engine's default then applies), and raises ValueError naming
    ``group.name`` when the text does not read.
    """

    changed = Signal()
    UNSET = object()

    def __init__(self, group: str | None, name: str, annotation: str,
                 default, *, species: Sequence[str] = (), parent=None):
        super().__init__(parent)
        self.group, self.name = group, name
        self.annotation = str(annotation)
        self.default = default
        self.where = f"{group}.{name}" if group else name
        self.kind = "text"
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        choices = _choices(group, name)
        editable = _editable_choices(group, name)
        blank = _BLANK_MEANS.get((group, name))
        if default is None or default == ():
            placeholder = f"blank: {blank}" if blank else "no default"
            counts = _NO_DEFAULT_NOTES.get((group, name))
            if counts and not blank:
                placeholder = f"no default; {counts}"
        else:
            placeholder = f"default {_shown(default)}"
        self.placeholder = placeholder
        self.widget: QWidget
        if self.annotation == "bool":
            self.kind = "bool"
            box = QCheckBox()
            box.setChecked(bool(default))
            box.toggled.connect(self.changed)
            self.widget = box
        elif (group, name) == ("scattering", "radiations"):
            from ..core import md_scattering

            self.kind = "radiations"
            self.boxes = {}
            holder = QWidget()
            line = QHBoxLayout(holder)
            line.setContentsMargins(0, 0, 0, 0)
            for radiation in md_scattering.RADIATIONS:
                box = QCheckBox(radiation)
                box.toggled.connect(self.changed)
                line.addWidget(box)
                self.boxes[radiation] = box
            line.addStretch(1)
            self.widget = holder
        elif (group, name) == ("exafs", "absorber"):
            self.kind = "choice"
            combo = QComboBox()
            combo.addItem("(not set)", None)
            for symbol in species:
                combo.addItem(symbol, symbol)
            combo.currentIndexChanged.connect(self.changed)
            self.widget = combo
        elif choices is not None:
            self.kind = "choice"
            combo = QComboBox()
            if default is None:
                combo.addItem("(not set)", None)
            for value in choices:
                combo.addItem(str(value), value)
            if default is not None:
                at = combo.findData(default)
                combo.setCurrentIndex(max(0, at))
            combo.currentIndexChanged.connect(self.changed)
            self.widget = combo
        elif editable is not None:
            self.kind = "editable"
            combo = QComboBox()
            combo.setEditable(True)
            combo.addItem("")
            for value in editable:
                combo.addItem(str(value))
            combo.setCurrentIndex(0)
            combo.lineEdit().setPlaceholderText(placeholder)
            combo.editTextChanged.connect(self.changed)
            self.widget = combo
        elif (group, name) in (("nmr", "correlation"), ("feff", "out_dir"),
                               (None, "params")):
            self.kind = "dir" if name == "out_dir" else "file"
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            edit.textChanged.connect(self.changed)
            browse = QPushButton("Browse…")
            browse.clicked.connect(self._browse)
            row.addWidget(edit, 1)
            row.addWidget(browse)
            self.widget = edit
            self.browse = browse
            return
        else:
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            edit.textChanged.connect(self.changed)
            self.widget = edit
        tips = _CHOICE_TIPS.get((group, name))
        if tips and isinstance(self.widget, QComboBox):
            # each choice says what it means (and what it costs) on hover
            combo = self.widget
            for i in range(combo.count()):
                value = combo.itemData(i)
                tip = tips.get(str(value if value is not None
                                   else combo.itemText(i)))
                if tip:
                    combo.setItemData(i, tip, Qt.ToolTipRole)
        row.addWidget(self.widget, 1)

    @property
    def typed(self) -> bool:
        """True for a field whose text is typed (lists, maps, numbers), as
        against a check box, a list to choose from or a file."""
        return self.kind in ("text", "editable")

    # -- reading ------------------------------------------------------------
    def text(self) -> str:
        if self.kind == "bool":
            return "true" if self.widget.isChecked() else "false"
        if self.kind == "radiations":
            return ", ".join(r for r, b in self.boxes.items() if b.isChecked())
        if self.kind == "choice":
            data = self.widget.currentData()
            return "" if data is None else str(data)
        if self.kind == "editable":
            return self.widget.currentText().strip()
        return self.widget.text().strip()

    def is_blank(self) -> bool:
        if self.kind == "bool":
            return bool(self.widget.isChecked()) == bool(self.default)
        if self.kind == "choice":
            data = self.widget.currentData()
            return data is None or data == self.default
        return self.text() == ""

    def value(self):
        if self.kind == "bool":
            return bool(self.widget.isChecked())
        if self.is_blank():
            return self.UNSET
        text = self.text()
        if self.kind == "radiations":
            return tuple(r for r, b in self.boxes.items() if b.isChecked())
        if self.kind == "choice":
            return self.widget.currentData()
        return _convert(self.annotation, text, self.where)

    # -- writing (tests and scripts) -----------------------------------------
    def set_text(self, text: str) -> None:
        """Type ``text`` into the field (a choice by its value)."""
        if self.kind == "bool":
            self.widget.setChecked(str(text).strip().lower() in
                                   ("true", "yes", "1", "on"))
        elif self.kind == "radiations":
            wanted = {t.strip() for t in str(text).split(",") if t.strip()}
            for radiation, box in self.boxes.items():
                box.setChecked(radiation in wanted)
        elif self.kind == "choice":
            at = self.widget.findData(text if text != "" else None)
            if at < 0:
                at = self.widget.findText(str(text))
            if at < 0:
                raise ValueError(f"{self.where}: {text!r} is not offered")
            self.widget.setCurrentIndex(at)
        elif self.kind == "editable":
            self.widget.setEditText(str(text))
        else:
            self.widget.setText(str(text))

    @Slot()
    def _browse(self) -> None:
        if self.kind == "dir":
            path = QFileDialog.getExistingDirectory(self, label_of(self.name))
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, label_of(self.name), "",
                "Request tables (*.json *.toml);;All files (*)"
                if self.name == "correlation" else "All files (*)")
        if path:
            self.widget.setText(path)


def _load_table_file(path: str, where: str):
    """A JSON or TOML file's content (TOML by its suffix), as the command
    line reads a request table; ValueError naming the file otherwise."""
    target = Path(str(path))
    try:
        text = target.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise ValueError(f"{where}: {target} could not be read "
                         f"({error.strerror or error})") from None
    try:
        if target.suffix.lower() == ".toml":
            import tomllib

            return tomllib.loads(text)
        return json.loads(text)
    except ValueError as error:
        kind = "TOML" if target.suffix.lower() == ".toml" else "JSON"
        raise ValueError(f"{where}: {target} is not valid {kind}: "
                         f"{error}") from None


def fractions_from_files(paths: Sequence[str],
                         where: str = "nmr.measured_fractions") -> tuple:
    """The measured sets of ``paths``, each file read as the command line's
    ``--nmr-fractions`` reads it: one set, a list of sets, or a table whose
    ``measured_fractions`` key holds them (a TOML file's
    ``[[measured_fractions]]`` tables). One file means the same in both
    front ends."""
    try:
        from ..md import cli
    except ImportError as error:
        raise ValueError(f"{where}: the option reader (facet.md.cli) is not "
                         f"present in this build ({error})") from None
    out = []
    for path in paths:
        loaded = _load_table_file(path, where)
        if isinstance(loaded, Mapping) and "measured_fractions" in loaded:
            loaded = loaded["measured_fractions"]
        items = list(loaded) if isinstance(loaded, (list, tuple)) \
            else [loaded]
        if not items:
            raise ValueError(f"{where}: {Path(str(path)).name} holds no "
                             "measured set")
        for item in items:
            try:
                out.append(cli.fractions_from(
                    item, f"{where} ({Path(str(path)).name})"))
            except ValueError as error:
                raise ValueError(str(error)) from None
    return tuple(out)


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{json.dumps(str(k))} = {_toml_value(v)}"
                               for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return json.dumps(str(value), ensure_ascii=False)


def request_toml(spec: Mapping, *, source: str = "") -> str:
    """A request file (TOML) holding ``spec`` (:meth:`SetupPanel.
    request_spec`), headed by the command that repeats the run with it."""
    lines = ["# FACET MD analysis request, written by the Model window."]
    if source:
        lines.append(f"# py -3.11 -m facet.md analyse {json.dumps(source)} "
                     "--request <this file> --out results.xlsx")
    lines.append("")
    tables = [(k, v) for k, v in spec.items() if isinstance(v, Mapping)
              and k not in ("ox", "type_map", "read_options")]
    for key, value in spec.items():
        if (key, value) in tables:
            continue
        lines.append(f"{key} = {_toml_value(value)}")
    for key, value in tables:
        lines.append("")
        lines.append(f"[{key}]")
        for name, item in value.items():
            lines.append(f"{name} = {_toml_value(item)}")
    return "\n".join(lines) + "\n"


class _PathList(QWidget):
    """A list of files (measured NMR fractions), added and removed."""

    changed = Signal()

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.list = QListWidget()
        self.list.setMaximumHeight(70)
        add = QPushButton("Add…")
        remove = QPushButton("Remove")
        add.clicked.connect(self._add)
        remove.clicked.connect(self._remove)
        column = QVBoxLayout()
        column.addWidget(add)
        column.addWidget(remove)
        column.addStretch(1)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.list, 1)
        row.addLayout(column)

    def paths(self) -> list[str]:
        return [self.list.item(i).text() for i in range(self.list.count())]

    def add_path(self, path: str) -> None:
        self.list.addItem(str(path))
        self.changed.emit()

    @Slot()
    def _add(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, self.title, "", "Request tables (*.json *.toml);;"
            "All files (*)")
        for path in paths:
            self.add_path(path)

    @Slot()
    def _remove(self) -> None:
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))
        self.changed.emit()


class _CurveTable(QWidget):
    """Measured curves for scattering-comparison: file, radiation, function,
    axis range and scale (blank scale: one factor fitted)."""

    changed = Signal()
    # short headers, each said in full in its tool tip: the long ones put
    # three of the five columns out of a 340 px table
    COLUMNS = ("file", "radiation", "function", "range", "scale")
    COLUMN_TIPS = ("The measured curve's file.",
                   "The radiation, one of md_scattering's.",
                   "The function the file holds; it fixes the axis (Q in "
                   "Å⁻¹ or r in Å).",
                   "range (axis unit): two numbers, first and last, that "
                   "limit the points compared; blank: no limit.",
                   "scale (blank: fitted): the scale factor; blank, one "
                   "factor is fitted by least squares.")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        for column, tip in enumerate(self.COLUMN_TIPS):
            self.table.horizontalHeaderItem(column).setToolTip(tip)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, len(self.COLUMNS)):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        # a long path keeps its file name in view
        self.table.setTextElideMode(Qt.ElideMiddle)
        self.table.setMaximumHeight(110)
        self.table.itemChanged.connect(self.changed)
        add = QPushButton("Add measured curve…")
        remove = QPushButton("Remove")
        add.clicked.connect(self._add)
        remove.clicked.connect(self._remove)
        buttons = QHBoxLayout()
        buttons.addWidget(add)
        buttons.addWidget(remove)
        buttons.addStretch(1)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self.table)
        column.addLayout(buttons)

    def add_curve(self, path: str, radiation: str | None = None,
                  function: str | None = None) -> None:
        from ..core import md_analysis as ma, md_scattering

        row = self.table.rowCount()
        self.table.insertRow(row)
        item = QTableWidgetItem(str(path))
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        item.setToolTip(str(path))
        self.table.setItem(row, 0, item)
        radiations = QComboBox()
        radiations.addItem("(not set)", None)
        for r in md_scattering.RADIATIONS:
            radiations.addItem(r, r)
        if radiation is not None:
            radiations.setCurrentIndex(max(0, radiations.findData(radiation)))
        functions = QComboBox()
        functions.addItem("(not set)", None)
        for f in ma.CURVE_FUNCTIONS:
            functions.addItem(f, f)
        if function is not None:
            functions.setCurrentIndex(max(0, functions.findData(function)))
        radiations.currentIndexChanged.connect(self.changed)
        functions.currentIndexChanged.connect(self.changed)
        self.table.setCellWidget(row, 1, radiations)
        self.table.setCellWidget(row, 2, functions)
        self.table.setItem(row, 3, QTableWidgetItem(""))
        self.table.setItem(row, 4, QTableWidgetItem(""))
        self.changed.emit()

    def curves(self) -> tuple:
        """The MeasuredCurve of each row; ValueError naming what a row
        lacks."""
        from ..core import md_analysis as ma

        out = []
        for row in range(self.table.rowCount()):
            path = self.table.item(row, 0).text()
            radiation = self.table.cellWidget(row, 1).currentData()
            function = self.table.cellWidget(row, 2).currentData()
            if radiation is None or function is None:
                raise ValueError(f"scattering.measured: {Path(path).name} "
                                 "needs its radiation and its function")
            span = (self.table.item(row, 3).text() or "").strip()
            scale = (self.table.item(row, 4).text() or "").strip()
            axis_range = None if not span else _convert(
                "tuple[float, float] | None", span, "scattering.measured "
                "range")
            factor = None if not scale else _convert(
                "float | None", scale, "scattering.measured scale")
            out.append(ma.MeasuredCurve(path, radiation, function,
                                        axis_range=axis_range, scale=factor))
        return tuple(out)

    @Slot()
    def _add(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Measured curve", "",
                                              "All files (*)")
        if path:
            self.add_curve(path)

    @Slot()
    def _remove(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()},
                      reverse=True)
        for row in rows:
            self.table.removeRow(row)
        self.changed.emit()


# ---------------------------------------------------------------------------
# reading: what the reader asks for
# ---------------------------------------------------------------------------

# The read options of md_readers.read_trajectory as the read panel labels
# them; each field's tool tip gives the option's own name.
READ_OPTION_LABELS = {
    "type_map": "type map (text)",
    "masses_from": "LAMMPS data file of the run",
    "topology": "topology file",
    "box_from": "box (a file, or three vectors)",
    "units": "LAMMPS unit style",
    "columns": "column names",
    "atom_style": "LAMMPS atom style",
    "mass_tol_amu": "mass tolerance (amu)",
    "timestep_fs": "MD timestep (fs)",
}


def _element_items() -> list[tuple[str, str]]:
    from ..core import md_readers

    return [(f"{symbol}   {weight:g} amu", symbol)
            for symbol, weight in md_readers.element_weights_amu()]


def _symbol_from(text: str) -> str | None:
    from ..core.md_model import validate_symbol

    token = str(text).strip().split()[0] if str(text).strip() else ""
    if not token:
        return None
    return validate_symbol(token)


class ReadOptionsPanel(QWidget):
    """The reader's refusal, and a field for each read option it asks for.

    ``readRequested(dict)`` carries the read options to try (those already
    given, with what was typed here); ``openOtherRequested()`` asks the
    window for another file.
    """

    readRequested = Signal(object)
    openOtherRequested = Signal()

    TYPE_COLUMNS = ("type", "atoms in frame 0", "what the file says",
                    "element")

    def __init__(self, parent=None, *, theme=None):
        super().__init__(parent)
        self.theme = theme
        self.problem: ReadProblem | None = None
        self.asked: tuple = ()
        self.combos: dict = {}
        self.fields: dict[str, QLineEdit] = {}
        self.companions: tuple = ()
        self.companion_buttons: dict[str, QPushButton] = {}
        outer = QVBoxLayout(self)
        self.title = QLabel()
        self.title.setWordWrap(True)
        font = self.title.font()
        font.setPointSizeF(font.pointSizeF() + 2)
        font.setBold(True)
        self.title.setFont(font)
        # what to do, in the window's words; the reader's own message (which
        # names read options as code does) is one click away under it, and
        # shown open when no input would answer it
        self.lead = QLabel()
        self.lead.setWordWrap(True)
        self.lead.setTextFormat(Qt.PlainText)
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setTextFormat(Qt.PlainText)
        self.message.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.message.setFrameShape(QFrame.StyledPanel)
        self.message.setMargin(8)
        self.message_fold = chrome.disclosure("What the reader reports",
                                              self.message)
        outer.addWidget(self.title)
        outer.addWidget(self.lead)
        outer.addWidget(self.message_fold)

        self.type_box = QGroupBox("Type map: the element of each type")
        type_column = QVBoxLayout(self.type_box)
        self.type_hint = _hint(
            "No element is filled in: each type's element is an input of the "
            "model. The middle columns are what the file itself holds.")
        self.type_table = QTableWidget(0, len(self.TYPE_COLUMNS))
        self.type_table.setHorizontalHeaderLabels(self.TYPE_COLUMNS)
        self.type_table.verticalHeader().setVisible(False)
        header = self.type_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        self.type_table.setColumnWidth(3, 210)
        self.type_table.setWordWrap(True)
        self.type_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.counts_note = _hint()
        type_column.addWidget(self.type_hint)
        type_column.addWidget(self.type_table)
        type_column.addWidget(self.counts_note)
        outer.addWidget(self.type_box)

        self.asked_box = QGroupBox("What the reader asks for")
        self.asked_form = QFormLayout(self.asked_box)
        outer.addWidget(self.asked_box)

        self.more_inner = QWidget()
        self.more_form = QFormLayout(self.more_inner)
        self.more = chrome.disclosure("Every read option this format takes",
                                      self.more_inner)
        outer.addWidget(self.more)

        buttons = QHBoxLayout()
        # mnemonics that no menu-bar title of the Model window takes (File,
        # Run, Help), so Alt+letter does what the underline shows
        self.read_button = QPushButton("Read a&gain")
        self.read_button.clicked.connect(self._on_read)
        self.other_button = QPushButton("Open &another MD file…")
        self.other_button.clicked.connect(self.openOtherRequested)
        buttons.addWidget(self.read_button)
        buttons.addWidget(self.other_button)
        buttons.addStretch(1)
        outer.addLayout(buttons)
        self.reason = _hint()
        outer.addWidget(self.reason)
        outer.addStretch(1)

    # -- filling ---------------------------------------------------------------
    def set_problem(self, problem: ReadProblem) -> None:
        self.problem = problem
        name = _escape(_label_of_source(problem.source))
        if problem.can_supply:
            self.title.setText(f"{name} needs more information to be read")
        else:
            self.title.setText(f"{name} was not read")
        self.message.setText(problem.message)
        lead = self._lead_text(problem)
        self.lead.setText(lead)
        self.lead.setVisible(bool(lead))
        # the reader's message open when nothing here would answer it
        self.message_fold.button.setChecked(not lead)
        self.combos = {}
        self.fields = {}
        self.companion_buttons = {}
        self._fill_types(problem)
        _clear_form(self.asked_form)
        _clear_form(self.more_form)
        # The type map is asked for here when the table cannot hold it (its
        # keys unknown or cut short), except beside a topology: there the
        # map applies to the topology's types, and the reader asks for it
        # again once the topology is read if that file does not name them.
        asked = [o for o in problem.needs
                 if o != "type_map" or (
                     "topology" not in problem.needs and
                     (not problem.type_keys or not problem.keys_complete))]
        for option in asked:
            self._add_option(self.asked_form, option, problem, asked=True)
        self.asked_box.setVisible(bool(asked))
        self.asked = tuple(asked)
        for option in sorted(problem.options_taken):
            if option in self.fields:
                continue
            self._add_option(self.more_form, option, problem, asked=False)
        refused = problem.error_type == "UnsupportedFormat"
        self.more.setVisible(bool(problem.options_taken) and not refused)
        # a format the reader refuses outright reads no differently with any
        # option: no Read again there (it re-read to the same message)
        self.read_button.setVisible(not refused and (
            problem.can_supply or bool(problem.options_taken)))
        self._update()

    @staticmethod
    def _lead_text(problem: ReadProblem) -> str:
        """What the reader asks for, as the window asks it (the reader's
        own message names read options as code does)."""
        needs = set(problem.needs)
        if not needs:
            return ""
        parts = []
        table = "type_map" in needs and bool(problem.type_keys) \
            and problem.keys_complete and "topology" not in needs
        if "topology" in needs:
            parts.append("Give the topology: a file FACET reads that holds "
                         "the same atoms in the same order (the LAMMPS data "
                         "file the run read, a dump, an extended XYZ).")
        if "type_map" in needs and "topology" not in needs:
            how = ("Choose the element of each type in the table"
                   if table else "Type the element of each type "
                   "(type=element, e.g. 1=Si, 2=O)")
            if "masses_from" in needs:
                how += (", or give the LAMMPS data file of the run, whose "
                        "Masses section names them")
            parts.append(how + ".")
        elif "masses_from" in needs:
            parts.append("Give the LAMMPS data file of the run, whose Masses "
                         "section names the elements of the types.")
        if "box_from" in needs:
            parts.append("Give the periodic box: another file of the same "
                         "run (its LAMMPS data file, a CP2K .cell file), or "
                         "the three box vectors in Å.")
        if "units" in needs:
            parts.append("State the unit style the file is written in.")
        if "columns" in needs:
            parts.append("Give the per-atom column names, in the file's "
                         "order.")
        if "atom_style" in needs:
            parts.append("State the LAMMPS atom style of the data file.")
        parts.append("The reader's own message is folded below.")
        return " ".join(parts)

    def set_companions(self, paths: Sequence[str]) -> None:
        """Files opened together with this one (a LAMMPS data file dropped
        with its dump): each is offered beside the file fields it can
        answer, with a button; none is filled in."""
        self.companions = tuple(str(p) for p in paths)
        if self.problem is not None:
            self.set_problem(self.problem)

    def _fill_types(self, problem: ReadProblem) -> None:
        table = self.type_table
        table.setRowCount(0)
        show = "type_map" in problem.needs and bool(problem.type_keys) \
            and problem.keys_complete
        self.type_box.setVisible(show)
        if not show:
            return
        items = _element_items()
        for key in problem.type_keys:
            row = table.rowCount()
            table.insertRow(row)
            label = f"type {key}" if isinstance(key, int) else f"label {key!r}"
            for column, text in enumerate((
                    label, "–" if problem.counts.get(key) is None
                    else str(problem.counts[key]),
                    problem.evidence.get(key) or "–")):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if column == 1:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                table.setItem(row, column, item)
            combo = QComboBox()
            combo.setEditable(True)
            combo.setInsertPolicy(QComboBox.NoInsert)
            combo.addItem("", None)
            for text, symbol in items:
                combo.addItem(text, symbol)
            combo.setCurrentIndex(0)
            combo.lineEdit().setPlaceholderText("element")
            combo.setToolTip(f"The element of {label}; every element is "
                             "listed with its standard atomic weight")
            combo.editTextChanged.connect(self._update)
            # show the symbol, not the end of 'Si   28.0855 amu'
            combo.currentIndexChanged.connect(
                lambda _i, c=combo: c.lineEdit().setCursorPosition(0))
            table.setCellWidget(row, 3, combo)
            self.combos[key] = combo
        table.resizeRowsToContents()
        height = table.horizontalHeader().height() + sum(
            table.rowHeight(r) for r in range(table.rowCount())) + 6
        table.setMinimumHeight(min(height, 320))
        table.setMaximumHeight(min(height, 320))
        self.counts_note.setText(problem.counts_note)
        self.counts_note.setVisible(bool(problem.counts_note))

    def _add_option(self, form: QFormLayout, option: str,
                    problem: ReadProblem, *, asked: bool) -> None:
        edit = QLineEdit()
        given = problem.read_options.get(option)
        if given is not None and option != "type_map":
            edit.setText(_option_text(given))
        suggestion = problem.suggested.get(option)
        holder = QWidget()
        stack = QVBoxLayout(holder)
        stack.setContentsMargins(0, 0, 0, 0)
        stack.setSpacing(2)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        stack.addLayout(row)
        row.addWidget(edit, 1)
        if option in ("masses_from", "topology", "box_from"):
            browse = QPushButton("Browse…")
            browse.clicked.connect(lambda _=False, e=edit: self._browse(e))
            row.addWidget(browse)
        if option == "type_map":
            edit.setPlaceholderText("type=element, ...  e.g. 1=Si, 2=O")
        elif option == "box_from":
            edit.setPlaceholderText("a file, or ax ay az; bx by bz; cx cy cz "
                                    "(Å)")
        elif suggestion:
            edit.setPlaceholderText(f"the reader names '{suggestion}'")
        text = READ_OPTION_TEXT.get(option, option)
        edit.setToolTip(f"{text}\n\nRead option {option} (the command "
                        f"line's --{option.replace('_', '-')} where it has "
                        "one)")
        edit.textChanged.connect(self._update)
        label = QLabel(READ_OPTION_LABELS.get(option, option))
        label.setToolTip(f"{text}\n\nRead option {option}")
        if asked:
            stack.addWidget(_hint(text))
        for path in self._companions_for(option):
            use = QPushButton(f"Use {Path(path).name} (opened with it)")
            use.setToolTip(f"Put {path} in this field; the file is read "
                           "only when Read again is pressed")
            use.clicked.connect(lambda _=False, e=edit, p=path: e.setText(p))
            stack.addWidget(use, 0, Qt.AlignLeft)
            self.companion_buttons.setdefault(option, use)
        form.addRow(label, holder)
        self.fields[option] = edit

    def _companions_for(self, option: str) -> list[str]:
        """The files opened with this one that can answer ``option``: a
        LAMMPS data file for its masses, its topology or its box, a CP2K
        cell file for a box."""
        if option not in ("masses_from", "topology", "box_from"):
            return []
        out = []
        for path in self.companions:
            name = Path(path).name.lower()
            base = name[:-3] if name.endswith(".gz") else name
            suffix = Path(base).suffix
            if suffix in (".data", ".lmp") or base.startswith("data.") or (
                    option == "box_from" and suffix == ".cell"):
                out.append(path)
        return out

    def _browse(self, edit: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose a file", "",
                                              "All files (*)")
        if path:
            edit.setText(path)

    # -- reading back ----------------------------------------------------------
    def type_map(self) -> dict:
        """The elements chosen in the table (only the rows set)."""
        out = {}
        for key, combo in self.combos.items():
            text = combo.currentText()
            data = combo.currentData()
            if data is not None and combo.itemText(combo.currentIndex()) \
                    == text:
                out[key] = data
                continue
            symbol = _symbol_from(text)
            if symbol is not None:
                out[key] = symbol
        return out

    def set_element(self, key, symbol: str) -> None:
        """Choose ``symbol`` for ``key`` (tests and scripts)."""
        combo = self.combos[key]
        at = combo.findData(symbol)
        if at >= 0:
            combo.setCurrentIndex(at)
        else:
            combo.setEditText(symbol)

    def set_option(self, option: str, text: str) -> None:
        self.fields[option].setText(str(text))

    def _new_text(self, option: str) -> str:
        """The text of a field when it says something the last read did not
        have: a value carried over from that read (shown so it can be
        changed) is not a new input."""
        text = self.fields[option].text().strip()
        given = self.problem.read_options.get(option) \
            if self.problem is not None else None
        if text and given is not None and option != "type_map":
            try:
                same = text == _option_text(given).strip()
            except Exception:           # noqa: BLE001 - compared as text
                same = False
            if same:
                return ""
        return text

    def missing(self) -> list[str]:
        """Why Read again is not offered yet (empty when it is)."""
        out = []
        new = {o: self._new_text(o) for o in self.fields}
        if self.combos:
            unread = []
            unset = []
            for key, combo in self.combos.items():
                text = combo.currentText().strip()
                if not text:
                    unset.append(key)
                    continue
                try:
                    _symbol_from(text)
                except ValueError:
                    unread.append(f"{key}: {text!r} is not an element "
                                  "symbol")
            supplied_elsewhere = any(new[o] for o in new
                                     if o in ("masses_from", "topology"))
            out.extend(unread)
            if unset and len(unset) < len(self.combos):
                out.append("no element yet for " + ", ".join(
                    (f"type {k}" if isinstance(k, int) else repr(k))
                    for k in unset))
            elif unset and not supplied_elsewhere and not any(new.values()):
                out.append("an element for each type, or another read "
                           "option")
        elif self.problem is not None and self.problem.can_supply and \
                not any(new.values()):
            out.append("a value for " + " or ".join(
                READ_OPTION_LABELS.get(o, o)
                for o in (self.asked or tuple(self.fields))))
        for option, edit in self.fields.items():
            text = edit.text().strip()
            if text:
                try:
                    _parse_read_option(option, text)
                except ValueError as error:
                    out.append(f"{READ_OPTION_LABELS.get(option, option)}: "
                               f"{error}")
        return out

    def read_options(self) -> dict:
        """The options to read with: those already given, then the ones
        typed here (a type map merged over the earlier one)."""
        if self.problem is None:
            return {}
        options = dict(self.problem.read_options)
        type_map = dict(options.get("type_map") or {})
        for option, edit in self.fields.items():
            text = edit.text().strip()
            if not text:
                continue
            value = _parse_read_option(option, text)
            if option == "type_map":
                type_map.update(value)
            else:
                options[option] = value
        type_map.update(self.type_map())
        if type_map:
            options["type_map"] = type_map
        return options

    def _update(self, *_args) -> None:
        missing = self.missing()
        self.read_button.setEnabled(not missing)
        self.reason.setText("" if not missing else
                            "Read again needs: " + "; ".join(missing))

    @Slot()
    def _on_read(self) -> None:
        if self.missing():
            return
        self.readRequested.emit(self.read_options())


def _label_of_source(source) -> str:
    from .md_jobs import source_label

    return source_label(source)


def _option_text(value) -> str:
    if isinstance(value, Mapping):
        return ", ".join(f"{k}={v}" for k, v in value.items())
    if isinstance(value, np.ndarray):
        return "; ".join(" ".join(f"{x:g}" for x in row) for row in value)
    return str(value)


def _parse_read_option(option: str, text: str):
    """A read option typed as text, as read_trajectory takes it."""
    from ..core.md_model import validate_symbol

    if option == "type_map":
        out = {}
        for item in text.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"{item!r} needs the form type=element")
            key, symbol = (p.strip() for p in item.split("=", 1))
            key = int(key) if key.lstrip("-").isdigit() else key.strip("'\"")
            out[key] = validate_symbol(symbol)
        return out
    if option == "box_from":
        rows = [r for r in text.replace(",", " ").split(";") if r.strip()]
        if len(rows) == 3:
            try:
                box = np.array([[float(x) for x in r.split()] for r in rows])
            except ValueError:
                raise ValueError("three rows of three numbers (Å), or a "
                                 "file") from None
            if box.shape != (3, 3):
                raise ValueError("three rows of three numbers (Å)")
            return box
        if not Path(text).exists():
            raise ValueError(f"no file {text}")
        return text
    if option in ("masses_from", "topology"):
        if not Path(text).exists():
            raise ValueError(f"no file {text}")
        return text
    if option in ("mass_tol_amu", "timestep_fs"):
        try:
            value = float(text)
        except ValueError:
            raise ValueError(f"{text!r} is not a number") from None
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{text!r}: a positive number is needed")
        return value
    return text


def _clear_form(form: QFormLayout) -> None:
    while form.rowCount():
        form.removeRow(0)


# ---------------------------------------------------------------------------
# setup: oxidation states, formers, frames
# ---------------------------------------------------------------------------

class OxidationTable(QTableWidget):
    """Element, atoms in frame 0, oxidation state, source.

    The states start at elements.COMMON_OX ('common'); a typed state other
    than the common one is the user's. An element with no common state
    starts blank and blocks the run until one is typed.
    """

    changed = Signal()
    COLUMNS = ("element", "atoms (frame 0)", "oxidation state", "source")

    def __init__(self, composition: Mapping[str, int], parent=None):
        super().__init__(0, len(self.COLUMNS), parent)
        from ..core import elements

        self.setHorizontalHeaderLabels(self.COLUMNS)
        self.verticalHeader().setVisible(False)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        header = self.horizontalHeader()
        for column in range(len(self.COLUMNS)):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        # a state is two or three characters: the column as wide as its
        # title, not as a line edit's default, so the source stays whole in
        # a 450 px setup ('common ...' at the Model window's first size)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        self.setColumnWidth(2, self.fontMetrics().horizontalAdvance(
            self.COLUMNS[2]) + 24)
        header.setStretchLastSection(True)
        self.common = {s: elements.COMMON_OX.get(s) for s in composition}
        self.edits: dict[str, QLineEdit] = {}
        for symbol, count in composition.items():
            row = self.rowCount()
            self.insertRow(row)
            for column, text in ((0, symbol), (1, str(count))):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if column == 1:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.setItem(row, column, item)
            edit = QLineEdit("" if self.common[symbol] is None
                             else f"{self.common[symbol]:+d}")
            edit.setPlaceholderText("no common state")
            edit.setToolTip(f"The formal oxidation state of {symbol}, a "
                            "model input; never resolved from the geometry")
            edit.textChanged.connect(self._on_edit)
            self.setCellWidget(row, 2, edit)
            self.setItem(row, 3, QTableWidgetItem(""))
            self.item(row, 3).setFlags(Qt.ItemIsEnabled)
            self.edits[symbol] = edit
        self._sources()
        self.resizeRowsToContents()
        height = header.height() + sum(self.rowHeight(r)
                                       for r in range(self.rowCount())) + 6
        self.setMinimumHeight(height)
        self.setMaximumHeight(height)

    def _sources(self) -> None:
        for row, (symbol, edit) in enumerate(self.edits.items()):
            state = _int_or_none(edit.text())
            if edit.text().strip() == "":
                text = "not given"
            elif state is None:
                text = "not a whole number"
            elif state == self.common[symbol]:
                # where the state comes from is in the tool tip: in the
                # text it was cut to 'common ...' in a 450 px setup
                text = "common"
            else:
                text = "user"
            self.item(row, 3).setText(text)
            self.item(row, 3).setToolTip(
                "common: the common state of elements.COMMON_OX"
                if text == "common" else text)

    def _on_edit(self, *_args) -> None:
        self._sources()
        self.changed.emit()

    def states(self) -> dict[str, int | None]:
        return {s: _int_or_none(e.text()) for s, e in self.edits.items()}

    def overrides(self) -> dict[str, int]:
        """The states that differ from elements.COMMON_OX (the request's
        ox_overrides)."""
        return {s: v for s, v in self.states().items()
                if v is not None and v != self.common[s]}

    def problems(self) -> list[str]:
        out = []
        for symbol, edit in self.edits.items():
            if _int_or_none(edit.text()) is None:
                out.append(f"no oxidation state for {symbol}"
                           if not edit.text().strip() else
                           f"the oxidation state of {symbol} is "
                           f"{edit.text()!r}, not a whole number")
        return out

    def set_state(self, symbol: str, state: int) -> None:
        self.edits[symbol].setText(f"{int(state):+d}")


def _int_or_none(text: str) -> int | None:
    text = str(text).strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if not value.is_integer():
        return None
    return int(value)


class FormerPicker(QWidget):
    """The cations present, as check boxes, none ticked; and a box stating
    that the model has no former to name.

    ``formers()`` is the request's value: a frozenset of the ticked
    elements, ``md_analysis.NO_FORMERS`` when the box is ticked, or None.
    """

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.list = QListWidget()
        self.list.setFlow(QListWidget.LeftToRight)
        self.list.setWrapping(True)
        self.list.setMaximumHeight(64)
        self.list.itemChanged.connect(self._on_item)
        self.none_box = QCheckBox("No former to name (the former-dependent "
                                  "descriptors are left out)")
        self.none_box.toggled.connect(self._on_none)
        self.note = _hint()
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self.list)
        column.addWidget(self.none_box)
        column.addWidget(self.note)
        self._cations: tuple = ()

    def set_cations(self, cations: Sequence[str]) -> None:
        kept = self.ticked()
        self.list.blockSignals(True)
        self.list.clear()
        for symbol in cations:
            item = QListWidgetItem(symbol)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if symbol in kept
                               else Qt.Unchecked)
            self.list.addItem(item)
        self.list.blockSignals(False)
        self._cations = tuple(cations)
        self._note()
        self.changed.emit()

    def cations(self) -> tuple:
        return self._cations

    def ticked(self) -> frozenset:
        return frozenset(self.list.item(i).text()
                         for i in range(self.list.count())
                         if self.list.item(i).checkState() == Qt.Checked)

    def formers(self):
        from ..core import md_analysis as ma

        if self.none_box.isChecked():
            return ma.NO_FORMERS
        ticked = self.ticked()
        return ticked if ticked else None

    def set_formers(self, symbols) -> None:
        wanted = set(symbols)
        for i in range(self.list.count()):
            item = self.list.item(i)
            item.setCheckState(Qt.Checked if item.text() in wanted
                               else Qt.Unchecked)

    def _note(self) -> None:
        value = self.formers()
        if value is None:
            text = ("No former chosen: the analyses that count over the "
                    "formers stay disabled until one is ticked. Stating that "
                    "there is none (the 'No former to name' box above) runs "
                    "the glass analysis without its former-dependent "
                    "descriptors.")
        elif isinstance(value, str):
            text = ("Stated: no former. Glass runs without its "
                    "former-dependent descriptors, with a note; the network "
                    "analyses still need named formers or their graph given "
                    "explicitly.")
        else:
            text = "Formers: " + ", ".join(sorted(value))
        self.note.setText(text)

    @Slot(QListWidgetItem)
    def _on_item(self, _item) -> None:
        if self.ticked() and self.none_box.isChecked():
            self.none_box.blockSignals(True)
            self.none_box.setChecked(False)
            self.none_box.blockSignals(False)
        self._note()
        self.changed.emit()

    @Slot(bool)
    def _on_none(self, on: bool) -> None:
        if on:
            self.list.blockSignals(True)
            for i in range(self.list.count()):
                self.list.item(i).setCheckState(Qt.Unchecked)
            self.list.blockSignals(False)
        self._note()
        self.changed.emit()


class FrameRange(QWidget):
    """First, last and stride over the readable frames, with the frames,
    timesteps and times they select."""

    changed = Signal()

    def __init__(self, summary: ModelSummary, parent=None):
        super().__init__(parent)
        self.summary = summary
        n = summary.n_frames
        self.first = QSpinBox()
        self.first.setRange(0, max(0, n - 1))
        self.last = QSpinBox()
        self.last.setRange(0, max(0, n - 1))
        self.last.setValue(n - 1)
        self.stride = QSpinBox()
        self.stride.setRange(1, max(1, n))
        for box, name in ((self.first, "first"), (self.last, "last"),
                          (self.stride, "every")):
            chrome.name_inside(box, name)
            box.valueChanged.connect(self._on_change)
            box.setEnabled(n > 1)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.first)
        row.addWidget(self.last)
        row.addWidget(self.stride)
        self.note = _hint()
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.addLayout(row)
        column.addWidget(self.note)
        self._describe()

    def frames(self) -> slice:
        return slice(self.first.value(), self.last.value() + 1,
                     self.stride.value())

    def indices(self) -> range:
        return range(self.summary.n_frames)[self.frames()]

    def count(self) -> int:
        return len(self.indices())

    def set_range(self, first: int, last: int, stride: int = 1) -> None:
        self.first.setValue(first)
        self.last.setValue(last)
        self.stride.setValue(stride)

    def _describe(self) -> None:
        from ..core.md_model import NO_TIMESTEP

        chosen = list(self.indices())
        n = self.summary.n_frames
        if not chosen:
            self.note.setText(f"No frame chosen (the file holds {n}).")
            return
        shown = ", ".join(str(k) for k in chosen[:6]) + (
            f", … {chosen[-1]}" if len(chosen) > 6 else "")
        parts = [f"{len(chosen)} of {n} frame(s): {shown}"]
        steps = self.summary.timesteps[chosen]
        known = steps[steps != NO_TIMESTEP]
        if known.size:
            parts.append(f"timesteps {int(known.min())} to {int(known.max())}")
        else:
            parts.append("no timesteps in the file")
        if self.summary.times_ps is not None:
            times = self.summary.times_ps[chosen]
            times = times[np.isfinite(times)]
            if times.size:
                parts.append(f"times {times.min():.6g} to {times.max():.6g} "
                             "ps")
        else:
            parts.append("no frame times in the file")
        self.note.setText("; ".join(parts))

    def _on_change(self, *_args) -> None:
        if self.first.value() > self.last.value():
            self.last.blockSignals(True)
            self.last.setValue(self.first.value())
            self.last.blockSignals(False)
        self._describe()
        self.changed.emit()


# ---------------------------------------------------------------------------
# setup: the analyses and their options
# ---------------------------------------------------------------------------

class AnalysisGroup(QGroupBox):
    """The analyses one module measures, each with its reason line, then
    the option groups they read: the inputs with no default first, the
    method choices (defaults stated in every export) folded away."""

    changed = Signal()
    # a reason's link: the field it names ('group.name', 'formers', ...)
    fieldRequested = Signal(str)

    def __init__(self, module: str, title: str, names: Sequence[str],
                 summary: ModelSummary, parent=None, *, theme=None):
        super().__init__(f"{title}  ({module})", parent)
        from ..core import md_analysis as ma

        self.module = module
        self.theme = theme
        self.boxes: dict[str, QCheckBox] = {}
        self.reasons: dict[str, QLabel] = {}
        self.fields: dict[tuple, OptionField] = {}
        self.labels: dict[tuple, QLabel] = {}
        self.extra: dict[str, QWidget] = {}
        self.wanted: set[str] = set()
        self.methods_fold = None
        column = QVBoxLayout(self)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        grid.setHorizontalSpacing(10)
        for row, name in enumerate(names):
            box = QCheckBox(name)
            box.setToolTip(ma.ANALYSIS_SUMMARIES.get(name, ""))
            box.toggled.connect(lambda on, n=name: self._on_box(n, on))
            reason = QLabel()
            reason.setWordWrap(True)
            reason.setTextFormat(Qt.RichText)
            reason.setTextInteractionFlags(Qt.TextSelectableByMouse
                                           | Qt.LinksAccessibleByMouse)
            reason.linkActivated.connect(self._on_link)
            chrome.mark_hint(reason)
            grid.addWidget(box, row, 0, Qt.AlignTop)
            grid.addWidget(reason, row, 1)
            self.boxes[name] = box
            self.reasons[name] = reason
        column.addLayout(grid)

        inputs = QFormLayout()
        inputs.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        methods_inner = QWidget()
        methods = QFormLayout(methods_inner)
        methods.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        species = tuple(summary.species)
        for name in TOP_FIELDS.get(module, ()):
            self._add_top(inputs, name)
        for group in OPTION_GROUPS.get(module, ()):
            cls = group_class(group)
            default = cls()
            for f in dataclasses.fields(cls):
                value = getattr(default, f.name)
                if (group, f.name) == ("scattering", "measured"):
                    widget = _CurveTable()
                    widget.changed.connect(self.changed)
                    self.extra["scattering.measured"] = widget
                    inputs.addRow(self._keep_label(group, f.name), widget)
                    continue
                if (group, f.name) == ("nmr", "measured_fractions"):
                    widget = _PathList("Measured fractions (JSON or TOML)")
                    widget.changed.connect(self.changed)
                    widget.setToolTip(
                        "JSON or TOML files, read as the command line's "
                        "--nmr-fractions reads them: one measured set, a "
                        "list of sets, or a TOML file's "
                        "[[measured_fractions]] tables. Each set holds "
                        "model_descriptor (a glass descriptor such as 'Qn "
                        "Si'), descriptor, values, source, and optionally "
                        "uncertainties, complete and sum_tol")
                    self.extra["nmr.measured_fractions"] = widget
                    inputs.addRow(self._keep_label(group, f.name), widget)
                    continue
                field_ = OptionField(group, f.name, f.type, value,
                                     species=species)
                field_.changed.connect(self.changed)
                field_.setToolTip(self._tip(group, f.name, field_))
                self.fields[(group, f.name)] = field_
                target = inputs if (value is None or value == ()
                                    or (group, f.name) in _SHOWN_CHOICES) \
                    else methods
                target.addRow(self._keep_label(group, f.name), field_)
        column.addLayout(inputs)
        if methods.rowCount():
            fold = chrome.disclosure(
                "Method choices (defaults, stated in every export)",
                methods_inner)
            column.addWidget(fold)
            self.methods_fold = fold
        self.errors = QLabel()
        self.errors.setWordWrap(True)
        self.errors.setTextInteractionFlags(Qt.TextSelectableByMouse)
        column.addWidget(self.errors)

    def _keep_label(self, group, name) -> QLabel:
        label = self._label(group, name)
        self.labels[(group, name)] = label
        return label

    def _add_top(self, form: QFormLayout, name: str) -> None:
        from ..core import md_analysis as ma

        f = {x.name: x for x in dataclasses.fields(ma.AnalysisRequest)}[name]
        field_ = OptionField(None, name, f.type, None)
        field_.changed.connect(self.changed)
        field_.setToolTip(self._tip(None, name, field_))
        self.fields[(None, name)] = field_
        form.addRow(self._keep_label(None, name), field_)

    def widget_for(self, group, name) -> QWidget | None:
        """The field (or the table, the list) of ``group.name`` here."""
        field_ = self.fields.get((group, name))
        if field_ is not None:
            return field_
        return self.extra.get(f"{group}.{name}")

    def reveal(self, group, name) -> QWidget | None:
        """Unfold the method choices when ``group.name`` sits there, and
        return its widget (None when this box does not hold it)."""
        widget = self.widget_for(group, name)
        if widget is None:
            return None
        fold = self.methods_fold
        if fold is not None and fold.inner.isAncestorOf(widget) and \
                not fold.button.isChecked():
            fold.button.setChecked(True)
        return widget

    @Slot(str)
    def _on_link(self, link: str) -> None:
        if link.startswith("field:"):
            self.fieldRequested.emit(link[len("field:"):])

    @staticmethod
    def _label(group, name) -> QLabel:
        label = QLabel(label_of(name))
        meaning = field_meaning(group, name)
        where = f"{group}.{name}" if group else name
        label.setToolTip(f"{meaning}\n\nRequest name: {where}" if meaning
                         else where)
        return label

    @staticmethod
    def _tip(group, name, field_: OptionField) -> str:
        """What the field is (the engine's words), its request name, what a
        blank means, and how to type it when it is typed."""
        where = f"{group}.{name}" if group else name
        meaning = field_meaning(group, name)
        lines = [f"{label_of(name)}: {meaning}" if meaning
                 else label_of(name),
                 f"Request name: {where} (in a request file, and with "
                 f"--set on the command line); {field_.placeholder}."]
        if field_.typed:
            lines.append("Typed as the command line takes it: lists with "
                         "commas, maps as key=value, pairs as Si-O.")
        return "\n".join(lines)

    def _on_box(self, name: str, on: bool) -> None:
        if self.boxes[name].isEnabled():
            if on:
                self.wanted.add(name)
            else:
                self.wanted.discard(name)
        self.changed.emit()

    def checked(self) -> list[str]:
        return [n for n, b in self.boxes.items()
                if b.isChecked() and b.isEnabled()]

    def set_reasons(self, reasons: Mapping[str, list],
                    adds: Mapping[str, list[str]],
                    optional: Mapping[str, list] | None = None) -> None:
        """Each analysis's line: what it needs (each field it names is a
        link to that field), or, once it can run, what it runs with and
        what it leaves out while an optional field is blank."""
        colour = _needs_colour(self.theme)
        optional = optional or {}
        for name, box in self.boxes.items():
            lacking = reasons.get(name) or []
            label = self.reasons[name]
            box.blockSignals(True)
            if lacking:
                box.setEnabled(False)
                box.setChecked(False)
                label.setText(f"<span style='color:{colour}'><b>needs</b> "
                              + "; ".join(_reason_html(r) for r in lacking)
                              + "</span>")
                label.setToolTip("; ".join(str(r) for r in lacking))
            else:
                box.setEnabled(True)
                box.setChecked(name in self.wanted)
                parts = []
                extra = adds.get(name)
                if extra:
                    parts.append(_escape("runs with " + ", ".join(extra)
                                         + " (it reads their results)"))
                left_out = optional.get(name) or []
                if left_out:
                    parts.append("left out while blank: " + "; ".join(
                        _reason_html(r) for r in left_out))
                label.setText(". ".join(parts))
                label.setToolTip("; ".join(str(r) for r in left_out))
            box.blockSignals(False)

    def set_errors(self, errors: Sequence[str]) -> None:
        colour = _needs_colour(self.theme)
        self.errors.setText("" if not errors else
                            f"<span style='color:{colour}'><b>does not "
                            "read:</b> " + "; ".join(_escape(e)
                                                    for e in errors)
                            + "</span>")
        self.errors.setVisible(bool(errors))

    def options(self, group: str):
        """(the group's options object, errors) from the fields."""
        cls = group_class(group)
        values, errors = {}, []
        for (g, name), field_ in self.fields.items():
            if g != group:
                continue
            try:
                value = field_.value()
            except ValueError as error:
                errors.append(str(error))
                continue
            if value is not OptionField.UNSET:
                values[name] = value
        if group == "scattering":
            try:
                curves = self.extra["scattering.measured"].curves()
            except ValueError as error:
                errors.append(str(error))
            else:
                if curves:
                    values["measured"] = curves
        if group == "nmr":
            paths = self.extra["nmr.measured_fractions"].paths()
            if paths:
                try:
                    values["measured_fractions"] = fractions_from_files(
                        paths)
                except ValueError as error:
                    errors.append(str(error))
        try:
            options = cls(**values)
        except (TypeError, ValueError) as error:
            errors.append(f"{group}: {error}")
            options = cls()
        return options, errors

    def top_values(self) -> tuple[dict, list[str]]:
        values, errors = {}, []
        for (g, name), field_ in self.fields.items():
            if g is not None:
                continue
            try:
                value = field_.value()
            except ValueError as error:
                errors.append(str(error))
                continue
            if value is not OptionField.UNSET:
                values[name] = value
        return values, errors


# ---------------------------------------------------------------------------
# the whole setup
# ---------------------------------------------------------------------------

class SetupPanel(QWidget):
    """Everything a run takes, with the Run button.

    ``request()`` is the AnalysisRequest the panel holds (None while it
    cannot run, :meth:`problems` saying why); ``runRequested()`` is emitted
    by the Run button; ``changed()`` after any edit, once the checks ran.
    """

    changed = Signal()
    runRequested = Signal()
    # the read options to read the file again with (the Read options box)
    rereadRequested = Signal(object)

    # A choice: how long the panel waits after an edit before it rebuilds
    # the request and the reasons, so typing a number does not check every
    # analysis once per key.
    REFRESH_MS = 120

    def __init__(self, summary: ModelSummary, trajectory=None, parent=None,
                 *, theme=None, type_map=None):
        super().__init__(parent)
        from ..core import bv, md_analysis as ma

        self.summary = summary
        self.trajectory = trajectory
        self.theme = theme
        self.type_map = dict(type_map or {})
        self._request = None
        self._problems: list[str] = []
        self._reasons: dict[str, list] = {}
        self._refused: dict[tuple, str] = {}
        self._running = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(self.REFRESH_MS)
        self._timer.timeout.connect(self.refresh)

        column = QVBoxLayout(self)
        heading = QLabel(f"<b>{_escape(summary.label)}</b>: "
                         f"{summary.n_atoms} atoms × {summary.n_frames} "
                         f"frame(s), {_escape(summary.file_format)}")
        heading.setWordWrap(True)
        column.addWidget(heading)
        self.summary_text = QPlainTextEdit("\n".join(summary.describe_lines))
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(150)
        column.addWidget(chrome.disclosure("Load summary (what the reader "
                                           "read and assumed)",
                                           self.summary_text))
        self.read_box = self._build_read_box(summary)
        if self.read_box is not None:
            column.addWidget(self.read_box)

        ox_box = QGroupBox("Oxidation states (model inputs)")
        ox_column = QVBoxLayout(ox_box)
        self.oxidation = OxidationTable(summary.composition)
        self.oxidation.changed.connect(self._on_ox)
        self.ox_notes = _hint()
        ox_column.addWidget(self.oxidation)
        ox_column.addWidget(self.ox_notes)
        column.addWidget(ox_box)

        formers_box = QGroupBox("Network formers")
        formers_column = QVBoxLayout(formers_box)
        self.formers = FormerPicker()
        self.formers.changed.connect(self.schedule)
        formers_column.addWidget(self.formers)
        column.addWidget(formers_box)

        frames_box = QGroupBox("Frames")
        frames_column = QVBoxLayout(frames_box)
        self.frame_range = FrameRange(summary)
        self.frame_range.changed.connect(self.schedule)
        frames_column.addWidget(self.frame_range)
        column.addWidget(frames_box)

        bv_box = QGroupBox("Bond valence")
        bv_form = QFormLayout(bv_box)
        self.v_bond = QDoubleSpinBox()
        self.v_bond.setDecimals(4)
        self.v_bond.setRange(0.0, 2.0)
        self.v_bond.setSingleStep(0.005)
        self.v_bond.setValue(bv.V_BOND_DEFAULT)
        self.v_bond.setSuffix(" v.u.")
        self.v_bond.setToolTip("v_bond_vu: a contact above it is a bond (CN, "
                               f"Qn, BO); FACET's value {bv.V_BOND_DEFAULT}")
        self.v_list = QDoubleSpinBox()
        self.v_list.setDecimals(4)
        self.v_list.setRange(0.0, 2.0)
        self.v_list.setSingleStep(0.005)
        self.v_list.setValue(bv.V_LIST_DEFAULT)
        self.v_list.setSuffix(" v.u.")
        self.v_list.setToolTip("v_list_vu: contacts above it are listed and "
                               "set the search radius; FACET's value "
                               f"{bv.V_LIST_DEFAULT}")
        self.v_bond.valueChanged.connect(self.schedule)
        self.v_list.valueChanged.connect(self.schedule)
        self.params_field = OptionField(None, "params", "bv.ParameterSet | "
                                        "None", None)
        self.params_field.widget.setPlaceholderText("blank: FACET's default "
                                                    "parameter set")
        self.params_field.changed.connect(self.schedule)
        bv_form.addRow(label_of("v_bond_vu"), self.v_bond)
        bv_form.addRow(label_of("v_list_vu"), self.v_list)
        bv_form.addRow("parameter file", self.params_field)
        column.addWidget(bv_box)

        self.groups: dict[str, AnalysisGroup] = {}
        by_module: dict[str, list[str]] = {m: [] for m, _ in MODULE_GROUPS}
        for name in ma.ANALYSES:
            by_module.setdefault(module_of(name), []).append(name)
        heading = QLabel("<b>Analyses</b> (grouped by the module that "
                         "measures them; tick to run)")
        column.addWidget(heading)
        for module, title in MODULE_GROUPS:
            group = AnalysisGroup(module, title, by_module[module], summary,
                                  theme=theme)
            group.changed.connect(self.schedule)
            group.fieldRequested.connect(self.focus_field)
            self.groups[module] = group
            column.addWidget(group)

        # The Run button and what the run still needs. A window puts this
        # footer under its scroll area (detach_footer), so both stay in view
        # whatever part of this long panel is scrolled to.
        self.footer = QWidget()
        foot = QVBoxLayout(self.footer)
        foot.setContentsMargins(0, 4, 0, 0)
        run_row = QHBoxLayout()
        # 'y': a letter no menu-bar title of the Model window takes
        self.run_button = QPushButton("Run anal&yses")
        self.run_button.setToolTip("Run the ticked analyses (Ctrl+R).")
        self.run_button.clicked.connect(self._on_run)
        run_row.addWidget(self.run_button)
        run_row.addStretch(1)
        foot.addLayout(run_row)
        self.run_reason = _hint()
        foot.addWidget(self.run_reason)
        column.addWidget(self.footer)
        column.addStretch(1)
        chrome.fit_every_combo(self)
        self._on_ox()
        self.refresh()

    # -- building --------------------------------------------------------------
    def detach_footer(self) -> QWidget:
        """Take the Run button and its line out of this panel, for the
        window to put under the panel's scroll area."""
        layout = self.layout()
        layout.removeWidget(self.footer)
        self.footer.setParent(None)
        return self.footer

    def _build_read_box(self, summary: ModelSummary):
        """How the file was read, and the read options to read it again
        with: units (a LAMMPS file read without a unit style is read as
        metal), the type map, and the format's other options. None for a
        format that takes none."""
        from .md_jobs import read_options_taken

        taken = sorted(read_options_taken(summary.file_format))
        if not taken:
            return None
        given = dict(summary.read_options)
        inner = QWidget()
        form = QFormLayout(inner)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.read_fields: dict[str, QLineEdit] = {}
        units_note = str(getattr(summary, "units_note", "") or "")
        for option in taken:
            edit = QLineEdit()
            value = given.get(option)
            if value is not None:
                edit.setText(_option_text(value))
            if option == "units":
                edit.setPlaceholderText("a LAMMPS unit style such as metal "
                                        "or real (blank: as read)")
            elif option == "type_map":
                edit.setPlaceholderText("type=element, ... (blank: as read)")
            elif option == "box_from":
                edit.setPlaceholderText("a file, or ax ay az; bx by bz; cx "
                                        "cy cz (Å)")
            else:
                edit.setPlaceholderText("blank: as read")
            text = READ_OPTION_TEXT.get(option, option)
            edit.setToolTip(f"{text}\n\nRead option {option}")
            edit.textChanged.connect(self._update_reread)
            label = QLabel(READ_OPTION_LABELS.get(option, option))
            label.setToolTip(f"{text}\n\nRead option {option}")
            form.addRow(label, edit)
            self.read_fields[option] = edit
        self.reread_button = QPushButton("Read the file again with these "
                                         "options")
        self.reread_button.setToolTip(
            "Reads the model file again; the setup below is built anew for "
            "the model read.")
        self.reread_button.clicked.connect(self._on_reread)
        self.reread_reason = _hint()
        form.addRow(self.reread_button)
        form.addRow(self.reread_reason)
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        # the unit assumption in view: it decides every time and velocity
        # (a 'units real' dump read as metal gives times 1000 times too
        # large and velocities 1000 times too small)
        self.units_note = _hint(f"Units: {units_note}" if units_note else "")
        self.units_note.setVisible(bool(units_note))
        column.addWidget(self.units_note)
        column.addWidget(chrome.disclosure(
            "Read options (how the file was read; read it again with "
            "others)", inner))
        self._update_reread()
        return holder

    def reread_options(self) -> dict:
        """The read options the Read options box holds (ValueError naming a
        field that does not read)."""
        options = {k: v for k, v in dict(self.summary.read_options).items()}
        for option, edit in getattr(self, "read_fields", {}).items():
            text = edit.text().strip()
            if not text:
                options.pop(option, None)
                continue
            try:
                options[option] = _parse_read_option(option, text)
            except ValueError as error:
                raise ValueError(f"{READ_OPTION_LABELS.get(option, option)}"
                                 f": {error}") from None
        return options

    def _update_reread(self, *_args) -> None:
        if not hasattr(self, "reread_button"):
            return
        try:
            self.reread_options()
        except ValueError as error:
            self.reread_button.setEnabled(False)
            self.reread_reason.setText(str(error))
        else:
            self.reread_button.setEnabled(not self._running)
            self.reread_reason.setText("")

    @Slot()
    def _on_reread(self) -> None:
        try:
            options = self.reread_options()
        except ValueError as error:
            self.reread_reason.setText(str(error))
            return
        self.rereadRequested.emit(options)

    # -- lookups ---------------------------------------------------------------
    def field(self, group: str | None, name: str) -> OptionField:
        """The field of ``group.name`` (``group`` None: a top-level one)."""
        if (group, name) == (None, "params"):
            return self.params_field
        for module in self.groups.values():
            if (group, name) in module.fields:
                return module.fields[(group, name)]
        raise KeyError(f"{group}.{name}")

    def all_fields(self) -> dict:
        out = {(None, "params"): self.params_field}
        for module in self.groups.values():
            out.update(module.fields)
        return out

    def extra(self, key: str):
        for module in self.groups.values():
            if key in module.extra:
                return module.extra[key]
        raise KeyError(key)

    def box(self, name: str) -> QCheckBox:
        return self.groups[module_of(name)].boxes[name]

    def reason_text(self, name: str) -> str:
        """The line beside analysis ``name``, as plain text (the label holds
        it with its links)."""
        from PySide6.QtGui import QTextDocumentFragment

        html_text = self.groups[module_of(name)].reasons[name].text()
        return QTextDocumentFragment.fromHtml(html_text).toPlainText()

    def set_analyses(self, names: Sequence[str]) -> None:
        """Tick these analyses (and untick the others); one that is disabled
        stays wanted and is ticked once its inputs are given."""
        wanted = set(names)
        for module in self.groups.values():
            module.wanted = {n for n in module.boxes if n in wanted}
            for name, box in module.boxes.items():
                if box.isEnabled():
                    box.blockSignals(True)
                    box.setChecked(name in wanted)
                    box.blockSignals(False)
        self.refresh()

    def reasons(self) -> dict[str, list[str]]:
        """Per analysis, what it needs, as text (empty when it can run)."""
        return {k: [str(r) for r in v] for k, v in self._reasons.items()}

    # -- state ---------------------------------------------------------------
    def schedule(self, *_args) -> None:
        self._timer.start()

    def _on_ox(self, *_args) -> None:
        from ..core import md_model

        states = self.oxidation.states()
        cations = [s for s, v in states.items() if v is not None and v > 0]
        self.formers.set_cations(cations)
        if self.oxidation.problems():
            self.ox_notes.setText("; ".join(self.oxidation.problems()))
        else:
            try:
                ox = md_model.model_oxidation(self.summary.species,
                                              self.oxidation.overrides())
            except ValueError as error:
                self.ox_notes.setText(str(error))
            else:
                self.ox_notes.setText("\n".join(ox.notes))
        self.schedule()

    def set_running(self, running: bool) -> None:
        self._running = bool(running)
        self._update_run()
        self._update_reread()

    def request(self):
        """The AnalysisRequest to run (None while :meth:`problems` names
        something)."""
        if self._timer.isActive():
            self._timer.stop()
            self.refresh()
        return self._request

    def problems(self) -> list[str]:
        if self._timer.isActive():
            self._timer.stop()
            self.refresh()
        return list(self._problems)

    @Slot()
    def refresh(self) -> None:
        """Rebuild the request from the fields; check every analysis."""
        from ..core import md_analysis as ma

        problems: list[str] = list(self.oxidation.problems())
        top: dict = {}
        group_errors: dict[str, list[str]] = {}
        group_options: dict[str, object] = {}
        for module in self.groups.values():
            errors: list[str] = []
            values, top_errors = module.top_values()
            top.update(values)
            errors.extend(top_errors)
            for group in OPTION_GROUPS.get(module.module, ()):
                options, errs = module.options(group)
                group_options[group] = options
                errors.extend(errs)
            group_errors[module.module] = errors
        try:
            params = self.params_field.value()
        except ValueError as error:
            problems.append(str(error))
            params = OptionField.UNSET
        frames = self.frame_range.frames()
        if self.frame_range.count() == 0:
            problems.append("no frame chosen")
        base = None
        try:
            base = ma.AnalysisRequest(
                analyses=(), formers=self.formers.formers(), frames=frames,
                v_bond_vu=float(self.v_bond.value()),
                v_list_vu=float(self.v_list.value()),
                params=None if params is OptionField.UNSET else params,
                ox_overrides=self.oxidation.overrides(),
                type_map=self.type_map or None,
                read_options={k: v for k, v in
                              self.summary.read_options.items()
                              if k != "type_map"},
                **top, **group_options)
        except (TypeError, ValueError) as error:
            problems.append(str(error))
        reasons, adds = self._check(base, group_errors)
        self._reasons = reasons
        optional = self._optional(base)
        for module in self.groups.values():
            module.set_reasons(reasons, adds, optional)
            module.set_errors(group_errors.get(module.module, []))
        chosen = [n for m in self.groups.values() for n in m.checked()]
        if not chosen:
            problems.append("no analysis ticked")
        request = None
        if base is not None and chosen and not problems:
            names = list(chosen)
            for name in chosen:
                for dep in ma.dependencies(base, name):
                    if dep not in names:
                        names.append(dep)
            try:
                request = run_request(base, names)
            except (TypeError, ValueError) as error:
                problems.append(str(error))
            else:
                lacking = ma.missing_inputs(request)
                if lacking:
                    problems.extend(user_text(m.describe()) for m in lacking)
                    request = None
        self._request = request
        self._problems = problems
        self._update_run()
        self.changed.emit()

    def _check(self, base, group_errors) -> tuple[dict, dict]:
        """What each analysis needs under the current fields (a list of
        :class:`Reason`), and the analyses each one adds to the run (those
        it reads)."""
        from ..core import md_analysis as ma

        reasons: dict[str, list] = {}
        adds: dict[str, list[str]] = {}
        if base is None:
            for name in ma.ANALYSES:
                reasons[name] = [Reason("the settings above to read (see the "
                                        "message under the Run button)")]
            return reasons, adds
        model = self._model_reasons(base)
        named_none = base.formers == ma.NO_FORMERS
        for name in ma.ANALYSES:
            out: list = []
            module = module_of(name)
            if group_errors.get(module):
                out.append(Reason("its options to read (below)"))
            needed = ma.dependencies(base, name)
            trial = dataclasses.replace(base, analyses=(name,) + needed)
            for m in ma.missing_inputs(trial):
                names = m.analysis.split("/")
                dep = None if name in names else next(
                    (d for d in needed if d in names), None)
                if name not in names and dep is None:
                    continue
                if m.name == "analyses":
                    continue
                # a need that belongs to an analysis this one reads is said
                # as that analysis's
                prefix = (f"{dep} (whose results it reads) needs "
                          if dep is not None else "")
                if m.name == "formers" and not named_none and \
                        (dep or name) == "glass":
                    # glass runs with the formers or with the statement that
                    # there are none: both are ticked above
                    text = (prefix + "network formers (tick them above, or "
                            "tick 'No former to name')")
                    reason = Reason(text, "formers", "network formers")
                else:
                    # every other former need is the engine's own text,
                    # which names what else answers it (a graph's elements
                    # given explicitly); 'No former to name' does not
                    reason = _field_reason(m.name, m.why, prefix=prefix)
                if all(r.text != reason.text for r in out):
                    out.append(reason)
            out.extend(model.get(name, []))
            reasons[name] = out
            if needed and not out:
                adds[name] = list(needed)
        return reasons, adds

    def _optional(self, base) -> dict[str, list]:
        """Per analysis, what it leaves out while an optional field is blank
        (:data:`OPTIONAL_OUTPUTS`), and what the model itself leaves out
        (Bhatia-Thornton is defined for two elements)."""
        out: dict[str, list] = {}
        if base is None:
            return out
        for name, entries in OPTIONAL_OUTPUTS.items():
            for what, fields in entries:
                blank = []
                for spec in fields:
                    group, _, field_name = spec.partition(".")
                    try:
                        widget = self.field(group, field_name)
                    except KeyError:
                        continue
                    if widget.is_blank():
                        blank.append(spec)
                if blank and len(blank) == len(fields):
                    label = " and ".join(field_label(f) for f in fields)
                    out.setdefault(name, []).append(Reason(
                        f"{what} (give {label})", fields[0], label))
        species = tuple(self.summary.species)
        if len(species) != 2:
            out.setdefault("scattering", []).append(Reason(
                f"Bhatia-Thornton (defined for two elements; the model has "
                f"{len(species)})"))
        return out

    def _model_reasons(self, request) -> dict[str, list]:
        """What the model itself shows a request cannot do: the frame
        count, the time axis, velocities, unwrapped positions, charges, and
        a refusal of the tracks the dynamics read (:meth:`note_refusal`)."""
        from ..core import md_analysis as ma, md_dynamics

        out: dict[str, list] = {}
        chosen = list(self.frame_range.indices())
        timed = list(ma.TRAJECTORY_ANALYSES) + ["bond-lifetimes"]
        if len(chosen) < 2:
            for name in timed:
                out.setdefault(name, []).append(Reason(
                    f"two or more chosen frames ({len(chosen)} chosen; the "
                    f"file holds {self.summary.n_frames})", "frames",
                    "chosen frames"))
            return out
        if self.trajectory is not None:
            try:
                md_dynamics.time_axis_ps(
                    self.trajectory, chosen, timestep_fs=request.timestep_fs,
                    frame_interval_ps=request.frame_interval_ps)
            except ValueError as error:
                text = str(error)
                if text.startswith("no time axis: "):
                    text = text[len("no time axis: "):]
                link = label_of("timestep_fs")
                for name in timed:
                    out.setdefault(name, []).append(Reason(
                        f"a time axis ({link}, or "
                        f"{label_of('frame_interval_ps')}): {text}",
                        "timestep_fs", link))
        dyn = request.dynamics
        if not self.summary.has_velocities:
            # the engine's rule (md_analysis._trajectory_problems): the
            # kinetic temperature always reads the file's velocities, the
            # VACF unless they are taken from the positions
            out.setdefault("kinetic-temperature", []).append(Reason(
                "velocities in the file: it holds none, and the kinetic "
                "temperature reads them"))
            if dyn.velocities == "file":
                link = label_of("velocities")
                out.setdefault("vacf", []).append(Reason(
                    f"velocities in the file: it holds none; {link} "
                    "'finite difference' takes them from the positions",
                    "dynamics.velocities", link))
        if dyn.unwrap == "file" and not self.summary.has_unwrapped:
            link = label_of("unwrap")
            for name in ma.TRAJECTORY_ANALYSES:
                out.setdefault(name, []).append(Reason(
                    f"unwrapped positions: the file holds none; {link} "
                    "'minimum image' or 'auto' unwraps by continuity",
                    "dynamics.unwrap", link))
        if request.charges_e == "file" and not self.summary.file_charges:
            link = label_of("charges_e")
            out.setdefault("conductivity", []).append(Reason(
                f"{link} 'file': the file states no charge per element",
                "charges_e", link))
        refused = self._refused.get(self._tracks_key(request))
        if refused:
            link = label_of("step_limit_fraction")
            for name in ma.TRAJECTORY_ANALYSES:
                out.setdefault(name, []).append(Reason(
                    f"tracks that can be collected: {refused} (fewer frames "
                    f"between those chosen, or the {link}, answer it)",
                    "dynamics.step_limit_fraction", link))
        return out

    def _tracks_key(self, request) -> tuple:
        """What the dynamics' tracks depend on: the frames, the time axis
        and the unwrap options."""
        dyn = request.dynamics
        return (tuple(self.frame_range.indices()), request.timestep_fs,
                request.frame_interval_ps, dyn.unwrap,
                dyn.step_limit_fraction,
                None if dyn.elements is None else tuple(dyn.elements))

    def note_refusal(self, request, reason: str) -> None:
        """The tracks of ``request``'s frames could not be collected (the
        check before a run): the dynamics analyses say so, and stay
        disabled with it until the frames or the unwrap options change."""
        self._refused[self._tracks_key(request)] = str(reason)
        self.refresh()

    def request_spec(self) -> dict:
        """The setup as a request file holds it, for ``py -3.11 -m facet.md
        analyse FILE --request X.toml`` to repeat the run: the analyses
        ticked (with those they read), the formers, frames, thresholds,
        oxidation states, type map and read options, and every field typed
        or chosen, as the text the command line takes. Fields left blank
        are left out, so the engine's defaults apply on both sides."""
        from ..core import md_analysis as ma

        request = self.request()
        spec: dict = {}
        if request is not None:
            spec["analyses"] = list(request.analyses)
        else:
            spec["analyses"] = [n for m in self.groups.values()
                                for n in m.checked()]
        formers = self.formers.formers()
        if formers == ma.NO_FORMERS:
            spec["formers"] = ma.NO_FORMERS
        elif formers:
            spec["formers"] = sorted(formers)
        chosen = self.frame_range.frames()
        spec["frames"] = f"{chosen.start}:{chosen.stop}:{chosen.step}"
        spec["v_bond_vu"] = float(self.v_bond.value())
        spec["v_list_vu"] = float(self.v_list.value())
        overrides = self.oxidation.overrides()
        if overrides:
            spec["ox"] = dict(overrides)
        if self.type_map:
            spec["type_map"] = {str(k): str(v)
                                for k, v in self.type_map.items()}
        read = {k: v for k, v in self.summary.read_options.items()
                if k != "type_map" and v is not None}
        if read:
            spec["read_options"] = {
                k: (np.asarray(v).tolist() if isinstance(v, np.ndarray)
                    else v) for k, v in read.items()}
        for (group, name), field_ in self.all_fields().items():
            if field_.is_blank():
                continue
            text = field_.text()
            if group is None:
                spec[name] = text
            else:
                spec.setdefault(group, {})[name] = text
        for module in self.groups.values():
            curves = module.extra.get("scattering.measured")
            if curves is not None:
                try:
                    measured = curves.curves()
                except ValueError:
                    measured = ()
                if measured:
                    spec.setdefault("scattering", {})["measured"] = [
                        {k: v for k, v in (
                            ("path", c.path), ("radiation", c.radiation),
                            ("function", c.function),
                            ("axis_range", None if c.axis_range is None
                             else list(c.axis_range)),
                            ("scale", c.scale)) if v is not None}
                        for c in measured]
            sets = module.extra.get("nmr.measured_fractions")
            if sets is not None and sets.paths():
                tables = []
                for path in sets.paths():
                    loaded = _load_table_file(path, "nmr.measured_fractions")
                    if isinstance(loaded, Mapping) and \
                            "measured_fractions" in loaded:
                        loaded = loaded["measured_fractions"]
                    tables.extend(loaded if isinstance(loaded, (list, tuple))
                                  else [loaded])
                spec.setdefault("nmr", {})["measured_fractions"] = tables
        return spec

    def _update_run(self) -> None:
        ok = self._request is not None and not self._running
        self.run_button.setEnabled(ok)
        if self._running:
            text = "A run is in progress."
        elif self._request is None:
            text = "Run needs: " + "; ".join(dict.fromkeys(self._problems))
        else:
            text = ("Runs: " + ", ".join(self._request.analyses) + f" on "
                    f"{self.frame_range.count()} frame(s).")
        self.run_reason.setText(text)
        self.run_reason.setToolTip(text)

    def focus_field(self, where: str) -> bool:
        """Show and focus the field a reason names ('group.name', a
        top-level name, 'formers', 'frames'): unfold it if folded, scroll
        it into view. False when no field of that name is here."""
        widget = None
        if where == "formers":
            widget = self.formers
        elif where == "frames":
            widget = self.frame_range
        elif where == "params":
            widget = self.params_field
        else:
            group, _, name = where.rpartition(".")
            group = group or None
            for module in self.groups.values():
                widget = module.reveal(group, name)
                if widget is not None:
                    break
        if widget is None:
            return False
        # the scroll area this panel sits in, if any
        parent = self.parentWidget()
        while parent is not None and not hasattr(parent,
                                                 "ensureWidgetVisible"):
            parent = parent.parentWidget()
        if parent is not None:
            parent.ensureWidgetVisible(widget, 40, 80)
        target = getattr(widget, "widget", widget)
        if isinstance(target, QWidget):
            target.setFocus(Qt.OtherFocusReason)
        return True

    @Slot()
    def _on_run(self) -> None:
        if self.request() is not None and not self._running:
            self.runRequested.emit()
