"""``py -3.11 -m facet.md``: FACET's MD analyses from the command line.

For a student with a LAMMPS dump on a workstation or a cluster node: no
window, no Qt (``tests/test_md_cli.py`` checks that importing this module
loads no PySide6). Four commands:

* ``describe FILE``: what the reader finds (atoms, frames, box,
  composition, type map, units, skipped frames) and the oxidation states
  the analyses would take;
* ``analyses``: every analysis, what it measures and the inputs it needs;
* ``template``: a request file (TOML) holding every option, with the
  method defaults and the inputs that have none;
* ``analyse FILE``: runs the analyses (``md_analysis.analyse``, one pair
  search per frame) and writes the results (``md_export``): an ``.xlsx``
  workbook, or a directory of CSV files, each with its provenance header.

An input that has no default (the network formers, the timestep, the
temperature, the charges, an NMR correlation...) is never filled in: a
request that lacks one stops before any frame is read, naming every missing
input. So does a value no analysis can take (a bin width of 0, a ring
criterion outside md_network's list, a measured file that does not exist,
v_bond below v_list, both a timestep and a frame interval): every one is
named at once (``md_analysis.invalid_inputs``), with exit code 2. Without
``--only``, every analysis whose inputs are given runs, and the others are
listed with what they lack; the analyses of time are left out when the file
gives no time axis (and neither --timestep-fs nor --frame-interval-ps is
given), and the VACF and the kinetic temperature when it holds no
velocities.

A field given twice, once under its own name and once under the shorter
one the command line uses (``ox_overrides`` and ``ox``), is refused rather
than one silently replacing the other; ``--ox`` adds to the request file's
oxidation states. Only the word ``none`` states that the model has no
network formers; a blank value leaves them not given.

Options are given in three layers, each overriding the one before: a
request file (``--request``, TOML or JSON, laid out as ``template``
prints), the flags, and ``--set group.option=value``. A list of numbers is
comma-separated (``1,2.5``) or, when it is a range, ``first:last:step``
(``0.05:1.0:0.05``, the Delta list of the channels).

EXIT CODES
----------
0 every analysis asked for was computed; 1 some analysis produced nothing
(its reason is printed; the others are written); 2 the command line or the
request lacks an input, gives a value no analysis can take or holds a
contradiction, against itself or against the model (nothing was run); 3
the model file could not be read; 4 the run was cancelled (Ctrl-C) and the
analyses finished so far were written; 5 an output could not be written.
For 5, every other ``--out`` target is still written (CSV directories
first), and when none can be, the results go to a CSV directory beside the
first target, which the message names.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import signal
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

EXIT_OK = 0
EXIT_PARTIAL = 1
EXIT_USAGE = 2
EXIT_READ = 3
EXIT_CANCELLED = 4
EXIT_WRITE = 5

# The option groups of a request, in the order the template prints them.
GROUPS = ("glass", "scattering", "network", "order", "voids", "channels",
          "nmr", "exafs", "feff", "dynamics")
# Top-level request fields a request file or --set may give, by name, with
# the friendlier names the command line uses.
_TOP_ALIASES = {"ox": "ox_overrides", "type_map": "type_map",
                "v_bond": "v_bond_vu", "v_list": "v_list_vu"}
_NONE_WORDS = ("none", "null", "")


class UsageError(ValueError):
    """A command line or request that cannot run (exit code 2)."""


# ---------------------------------------------------------------------------
# value parsing: one converter per annotation of the option classes
# ---------------------------------------------------------------------------

def _split(text: str, separators: str = ",") -> list[str]:
    parts = [text]
    for sep in separators:
        parts = [p for chunk in parts for p in chunk.split(sep)]
    return [p.strip() for p in parts if p.strip()]


def _float(value, where: str) -> float:
    if isinstance(value, bool):
        raise UsageError(f"{where}: {value!r} is not a number")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise UsageError(f"{where}: {value!r} is not a number") from None
    if not math.isfinite(out):
        raise UsageError(f"{where}: {value!r} is not a finite number")
    return out


def _int(value, where: str) -> int:
    if isinstance(value, bool):
        raise UsageError(f"{where}: {value!r} is not a whole number")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise UsageError(f"{where}: {value!r} is not a whole number") from None
    if not number.is_integer():
        raise UsageError(f"{where}: {value!r} is not a whole number")
    return int(number)


def _bool(value, where: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "yes", "1", "on"):
        return True
    if text in ("false", "no", "0", "off"):
        return False
    raise UsageError(f"{where}: {value!r} is not true or false")


def _list(value) -> list:
    if isinstance(value, (list, tuple)):
        return list(value)
    return _split(str(value), ",")


def _pair(value, where: str) -> tuple[str, str]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return str(value[0]).strip(), str(value[1]).strip()
    parts = _split(str(value), "-")
    if len(parts) != 2:
        raise UsageError(f"{where}: {value!r} is not an element pair such as "
                         "Si-O")
    return parts[0], parts[1]


def _mapping(value, where: str) -> dict[str, str]:
    """'A=1,B=2' or a table, as text keys and raw values."""
    if isinstance(value, Mapping):
        return {str(k).strip(): v for k, v in value.items()}
    out = {}
    for item in _split(str(value), ","):
        if "=" not in item:
            raise UsageError(f"{where}: {item!r} needs the form key=value")
        key, raw = item.split("=", 1)
        out[key.strip()] = raw.strip()
    return out


def _frames(value, where: str):
    """'0:100:5' -> slice; '3' or '1,3,5' or a list -> indices."""
    if isinstance(value, (list, tuple)):
        return tuple(_int(v, where) for v in value)
    text = str(value).strip()
    if ":" in text:
        parts = text.split(":")
        if len(parts) > 3:
            raise UsageError(f"{where}: {value!r} is not start:stop:step")
        numbers = [None if p.strip() == "" else _int(p, where) for p in parts]
        return slice(*numbers)
    return tuple(_int(v, where) for v in _split(text, ","))


def _type_map(value, where: str) -> dict:
    out = {}
    for key, element in _mapping(value, where).items():
        out[int(key) if key.lstrip("-").isdigit() else key] = str(element)
    return out


def _shells(value, where: str) -> tuple:
    items = value if isinstance(value, (list, tuple)) else [
        _split(item, ":") for item in _split(str(value), ";")]
    out = []
    for item in items:
        if isinstance(item, str):
            item = _split(item, ":")
        if len(item) != 3:
            raise UsageError(f"{where}: each shell needs element:r_lo:r_hi "
                             f"(got {item!r})")
        out.append((str(item[0]), _float(item[1], where),
                    _float(item[2], where)))
    return tuple(out)


def _float_range(text: str, where: str) -> tuple:
    """'first:last:step' -> first, first + step, ..., up to last (within
    half a step), as a list of numbers is written when it is a range (the
    Delta list of the channels, say); rounded to 10 decimals."""
    parts = text.split(":")
    if len(parts) != 3:
        raise UsageError(f"{where}: {text!r} is not first:last:step")
    first, last, step = (_float(p, where) for p in parts)
    if step <= 0.0 or not last > first:
        raise UsageError(f"{where}: {text!r} needs a step above 0 and last "
                         "above first")
    n = int(math.floor((last - first) / step + 1e-9)) + 1
    return tuple(round(first + k * step, 10) for k in range(n))


def _floats(value, where: str, count: int | None = None) -> tuple:
    if isinstance(value, str) and ":" in value and "," not in value:
        out = _float_range(value.strip(), where)
    else:
        out = tuple(_float(v, where) for v in _list(value))
    if count is not None and len(out) != count:
        raise UsageError(f"{where}: {count} numbers are needed, not "
                         f"{len(out)} ({value!r})")
    return out


def convert(annotation: str, value, where: str):
    """A value from a request file or the command line, typed for a request
    field whose annotation (as written in md_analysis) is ``annotation``."""
    from facet.core import md_analysis as ma

    optional = annotation.endswith(" | None")
    base = annotation[:-len(" | None")] if optional else annotation
    if base == "str":
        # a text option keeps its text: 'none' is a value of r_window,
        # weighting and vdos_window; an option is unset by leaving it out
        if value is None:
            if optional:
                return None
            raise UsageError(f"{where}: a value is needed")
        return str(value).strip()
    if base == "frozenset[str] | str" and isinstance(value, str) \
            and value.strip().lower() == ma.NO_FORMERS:
        # only the word itself states that the model has no formers; a blank
        # or 'null' leaves the formers not given (and refused where needed)
        return ma.NO_FORMERS
    if value is None or (isinstance(value, str)
                         and value.strip().lower() in _NONE_WORDS):
        if base == "frozenset[str] | str":
            return None
        if optional:
            return None
        if base.startswith("tuple"):
            return ()
        raise UsageError(f"{where}: a value is needed")
    if base == "float":
        return _float(value, where)
    if base == "int":
        return _int(value, where)
    if base == "bool":
        return _bool(value, where)
    if base == "str":
        return str(value).strip()
    if base == "tuple[str, ...]":
        return tuple(str(v).strip() for v in _list(value))
    if base == "tuple[float, ...]":
        return _floats(value, where)
    if base == "tuple[int, ...]":
        return tuple(_int(v, where) for v in _list(value))
    if base == "tuple[float, float]":
        return _floats(value, where, 2)
    if base == "tuple[float, float, float]":
        return _floats(value, where, 3)
    if base == "Mapping[tuple[str, str], float]":
        return {_pair(k, where): _float(v, where)
                for k, v in _mapping(value, where).items()}
    if base == "Mapping[str, float]":
        return {k: _float(v, where) for k, v in _mapping(value, where).items()}
    if base == "Mapping[str, int]":
        return {k: _int(v, where) for k, v in _mapping(value, where).items()}
    if base == "Mapping[str, str]":
        return {k: str(v) for k, v in _mapping(value, where).items()}
    if base == "Mapping[str, object]":
        return dict(_mapping(value, where))
    if base == "Mapping[int | str, str]":
        return _type_map(value, where)
    if base == "tuple[tuple[str, str], ...]":
        return tuple(_pair(v, where) for v in _list(value))
    if base == "tuple[tuple[str, float, float], ...]":
        return _shells(value, where)
    if base == "frozenset[str] | str":
        return frozenset(str(v).strip() for v in _list(value))
    if base == "slice | tuple[int, ...]":
        return _frames(value, where)
    if base in ("str | Mapping[str, float]", "Mapping[str, float] | str"):
        if isinstance(value, str) and "=" not in value:
            return value.strip()
        return {k: _float(v, where) for k, v in _mapping(value, where).items()}
    if base == "md_spectroscopy.Correlation":
        return correlation_from(value, where)
    if base == "tuple[MeasuredCurve, ...]":
        return tuple(measured_curve_from(v, where) for v in _list_of(value))
    if base == "tuple[FractionMeasurement, ...]":
        return tuple(fractions_from(v, where) for v in _list_of(value))
    if base == "bv.ParameterSet":
        return _parameter_set(value, where)
    raise UsageError(f"{where}: values of this option are given in a request "
                     f"file ({annotation})")


def _list_of(value) -> list:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


# ---------------------------------------------------------------------------
# the structured inputs: an NMR correlation, measured fractions and curves
# ---------------------------------------------------------------------------

def _load_mapping(path, where: str):
    target = Path(str(path))
    try:
        text = target.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise UsageError(f"{where}: {target} could not be read "
                         f"({error.strerror or error})") from None
    try:
        if target.suffix.lower() == ".toml":
            import tomllib
            return tomllib.loads(text)
        return json.loads(text)
    except (ValueError, json.JSONDecodeError) as error:
        raise UsageError(f"{where}: {target} is not valid "
                         f"{'TOML' if target.suffix.lower() == '.toml' else 'JSON'}"
                         f": {error}") from None


def correlation_from(value, where: str = "nmr.correlation"):
    """An ``md_spectroscopy.Correlation`` from a table, or from the path of a
    JSON / TOML file holding one: nucleus, intercept_ppm, reference,
    shift_reference, anion, terms (each: descriptor, coefficient_ppm, power,
    valid_range, angle_function, partner, name, unit)."""
    from facet.core import md_spectroscopy

    spec = value if isinstance(value, Mapping) else _load_mapping(value, where)
    if not isinstance(spec, Mapping):
        raise UsageError(f"{where}: a correlation table is needed")
    known = {"nucleus", "intercept_ppm", "terms", "reference",
             "shift_reference", "anion", "notes"}
    unknown = sorted(set(spec) - known)
    if unknown:
        raise UsageError(f"{where}: unknown key(s) {', '.join(unknown)}; a "
                         f"correlation holds {', '.join(sorted(known))}")
    terms = []
    for n, item in enumerate(spec.get("terms") or ()):
        if not isinstance(item, Mapping):
            raise UsageError(f"{where}: term {n} needs a table")
        fields = dict(item)
        if fields.get("valid_range") is not None:
            fields["valid_range"] = tuple(fields["valid_range"])
        try:
            terms.append(md_spectroscopy.Term(**fields))
        except TypeError as error:
            raise UsageError(f"{where}: term {n}: {error}") from None
    try:
        return md_spectroscopy.Correlation(
            nucleus=spec.get("nucleus"),
            intercept_ppm=spec.get("intercept_ppm"), terms=tuple(terms),
            reference=spec.get("reference"),
            shift_reference=spec.get("shift_reference"),
            anion=spec.get("anion", "O"), notes=tuple(spec.get("notes", ())))
    except ValueError as error:
        raise UsageError(f"{where}: {error}") from None


def fractions_from(value, where: str = "nmr.measured_fractions"):
    """A ``md_analysis.FractionMeasurement``: model_descriptor, descriptor,
    values, source, uncertainties, complete, sum_tol."""
    from facet.core import md_analysis as ma, md_spectroscopy

    spec = value if isinstance(value, Mapping) else _load_mapping(value, where)
    if not isinstance(spec, Mapping) or "model_descriptor" not in spec:
        raise UsageError(f"{where}: each measured set needs model_descriptor "
                         "(a glass descriptor such as 'Qn Si'), descriptor, "
                         "values and source")
    fields = {k: v for k, v in spec.items() if k != "model_descriptor"}
    try:
        measured = md_spectroscopy.MeasuredFractions(**fields)
    except (TypeError, ValueError) as error:
        raise UsageError(f"{where}: {error}") from None
    return ma.FractionMeasurement(str(spec["model_descriptor"]), measured)


def measured_curve_from(value, where: str = "scattering.measured"):
    """A ``md_analysis.MeasuredCurve`` from a table (path, radiation,
    function, axis_range, scale) or from 'RADIATION:FUNCTION:PATH'."""
    from facet.core import md_analysis as ma

    if isinstance(value, Mapping):
        fields = dict(value)
        if fields.get("axis_range") is not None:
            fields["axis_range"] = _floats(fields["axis_range"], where, 2)
        try:
            return ma.MeasuredCurve(**fields)
        except TypeError as error:
            raise UsageError(f"{where}: {error}") from None
    parts = str(value).split(":", 2)
    if len(parts) != 3:
        raise UsageError(f"{where}: {value!r} needs RADIATION:FUNCTION:PATH, "
                         "e.g. neutron:S(Q):sq.dat")
    radiation, function, path = (p.strip() for p in parts)
    if function not in ma.CURVE_FUNCTIONS:
        raise UsageError(f"{where}: function {function!r} is not one of "
                         f"{', '.join(ma.CURVE_FUNCTIONS)}")
    return ma.MeasuredCurve(path, radiation, function)


def _parameter_set(value, where: str):
    from facet.core import bv_files

    try:
        params, _ = bv_files.load(str(value))
    except ValueError as error:
        raise UsageError(f"{where}: {error}") from None
    return params


# ---------------------------------------------------------------------------
# a request from a mapping (request file, flags, --set)
# ---------------------------------------------------------------------------

def _group_class(name: str):
    from facet.core import md_analysis as ma

    return {"glass": ma.GlassOptions, "scattering": ma.ScatteringOptions,
            "network": ma.NetworkOptions, "order": ma.OrderOptions,
            "voids": ma.VoidOptions, "channels": ma.ChannelOptions,
            "nmr": ma.NmrOptions,
            "exafs": ma.ExafsOptions, "feff": ma.FeffOptions,
            "dynamics": ma.DynamicsOptions}[name]


def _fields(cls) -> dict[str, dataclasses.Field]:
    return {f.name: f for f in dataclasses.fields(cls)}


def request_from_mapping(spec: Mapping):
    """An ``md_analysis.AnalysisRequest`` from nested key/values: the
    top-level fields (``analyses``, ``formers``, ``frames``, ``ox``,
    ``timestep_fs`` ...) and one table per option group. Every value is
    typed from the field's annotation; an unknown key is refused, naming
    the known ones."""
    from facet.core import md_analysis as ma

    top_fields = _fields(ma.AnalysisRequest)
    top: dict = {}
    groups: dict[str, dict] = {}
    for key, value in _canonical(spec, "the request").items():
        if key in GROUPS:
            if not isinstance(value, Mapping):
                raise UsageError(f"[{key}] needs a table of options")
            cls = _group_class(key)
            known = _fields(cls)
            options = {}
            for option, raw in value.items():
                if option not in known:
                    raise UsageError(
                        f"unknown option {key}.{option}; the {key} options "
                        f"are {', '.join(known)}")
                options[option] = convert(known[option].type, raw,
                                          f"{key}.{option}")
            groups[key] = options
        elif key in top_fields:
            top[key] = convert(top_fields[key].type, value, key)
        else:
            raise UsageError(
                f"unknown request key {key!r}; the request keys are "
                f"{', '.join(sorted(set(top_fields) - set(GROUPS)))}, and "
                f"the option groups {', '.join(GROUPS)}")
    if "analyses" not in top:
        top["analyses"] = ()
    for name, options in groups.items():
        try:
            top[name] = _group_class(name)(**options)
        except (TypeError, ValueError) as error:
            raise UsageError(f"[{name}]: {error}") from None
    try:
        return ma.AnalysisRequest(**top)
    except ValueError as error:
        raise UsageError(str(error)) from None


def _canonical(spec: Mapping, where: str) -> dict:
    """``spec`` with each top-level alias ('ox', 'v_bond', ...) under the
    field's own name. A mapping that gives a field under both spellings is
    refused: the later one used to replace the earlier one silently, so a
    request file's ox_overrides were lost when --ox was given."""
    out: dict = {}
    spelled: dict[str, str] = {}
    for key, value in spec.items():
        name = _TOP_ALIASES.get(key, key)
        if name in out:
            raise UsageError(f"{where} gives {name} twice, as {spelled[name]!r} "
                             f"and {key!r}; give one")
        out[name] = value
        spelled[name] = key
    return out


def _merge(base: dict, extra: Mapping) -> dict:
    out = dict(base)
    for key, value in extra.items():
        if key in GROUPS and isinstance(value, Mapping):
            merged = dict(out.get(key) or {})
            merged.update(value)
            out[key] = merged
        else:
            out[key] = value
    return out


def _set_assignment(text: str) -> tuple[list[str], str]:
    if "=" not in text:
        raise UsageError(f"--set {text!r} needs the form group.option=value "
                         "(or option=value for a top-level field)")
    key, value = text.split("=", 1)
    return [p.strip() for p in key.split(".")], value.strip()


# ---------------------------------------------------------------------------
# texts: the analyses, the template
# ---------------------------------------------------------------------------

def analyses_text() -> str:
    """Every analysis, what it measures and the inputs it needs."""
    from facet.core import md_analysis as ma

    lines = []
    empty = ma.AnalysisRequest(analyses=())
    for name in ma.ANALYSES:
        trial = dataclasses.replace(empty, analyses=(name,)
                                    + ma.dependencies(empty, name))
        needs = sorted({m.name for m in ma.missing_inputs(trial)
                        if name in m.analysis.split("/")})
        if name in ma.TRAJECTORY_ANALYSES or name == "bond-lifetimes":
            needs.append("timestep_fs or frame_interval_ps (unless the file "
                         "states its frame times), two frames or more")
        if name == "kinetic-temperature":
            needs.append("velocities in the file")
        if name == "vacf":
            needs.append("velocities in the file (or dynamics.velocities = "
                         "'finite difference')")
        group = ("per frame" if name in ma.PER_FRAME_ANALYSES else
                 "whole trajectory" if name in ma.TRAJECTORY_ANALYSES
                 else "after the others")
        lines.append(f"  {name} ({group})")
        lines.append(f"      {ma.ANALYSIS_SUMMARIES[name]}")
        deps = ma.dependencies(empty, name)
        if needs:
            lines.append(f"      needs: {', '.join(needs)}")
        if deps:
            lines.append(f"      reads the results of: {', '.join(deps)}")
    return "\n".join(lines)


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (tuple, list, frozenset)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, Mapping):
        return "{" + ", ".join(
            f"{json.dumps('-'.join(k) if isinstance(k, tuple) else str(k))} "
            f"= {_toml_value(v)}" for k, v in value.items()) + "}"
    return json.dumps(str(value))


# Options whose absence is itself a stated method choice (md_analysis's
# option classes), so the template does not list them as inputs with no
# default.
_LEFT_OUT_MEANS = {
    ("glass", "rdf_r_max_ang"): "the bond-valence search radius",
    ("glass", "minimum_smooth_sigma_ang"): "the g(r) as measured",
    ("scattering", "r_max_ang"): "the largest grid the boxes hold",
    ("scattering", "termination_q_min_inv_ang"): "the first Q of the grid",
    ("exafs", "r_max_ang"): "the glass g(r) radius",
    ("channels", "repulsion_r_excl_ang"): "no repulsion exclusion",
    ("channels", "repulsion_elements"): "every cation element but the "
                                        "probe's (with repulsion_r_excl_ang)",
    ("channels", "modifier_cutoffs"): "the glass analysis's cutoffs when "
                                      "glass runs, else the first minimum "
                                      "of the first frame's own g(r)",
}


def template_text() -> str:
    """A TOML request file with every option: defaults set, the inputs with
    no default commented out with what they need."""
    from facet.core import md_analysis as ma

    lines = [
        "# FACET MD analysis request (py -3.11 -m facet.md analyse FILE "
        "--request this.toml)",
        "# Lines starting with # are not read. An option left commented out "
        "has no default:",
        "# an analysis that needs it is refused, naming it. The values given "
        "below are",
        "# method defaults (bin widths, grids, the first-minimum rule), "
        "stated in every export.",
        "",
        "# analyses = [\"glass\", \"rings\"]   # omit to run every analysis "
        "whose inputs are given",
        f"# formers = [\"Si\"]   # the network formers; \"{ma.NO_FORMERS}\" "
        "states that none are named",
        "# frames = \"0:100:5\"   # start:stop:step over the readable "
        "frames, or a list",
        "# ox = {Na = 1}   # oxidation states other than elements.COMMON_OX",
        "# type_map = {\"1\" = \"Si\", \"2\" = \"O\"}   # LAMMPS type -> "
        "element",
        "# read_options = {units = \"metal\"}   # md_readers.read_trajectory "
        "options",
        f"v_bond_vu = {ma.AnalysisRequest(analyses=()).v_bond_vu!r}",
        f"v_list_vu = {ma.AnalysisRequest(analyses=()).v_list_vu!r}",
        "# bridging_anions = [\"O\"]   # default: every anion of the model",
        "# timestep_fs = 1.0   # the MD timestep (time axis of the dynamics)",
        "# frame_interval_ps = 1.0   # or the time between frames",
        "# temperature_k = 300.0   # the run's temperature (conductivity)",
        "# charges_e = {Na = 0.6, O = -1.2, Si = 2.4}   # or \"oxidation "
        "states\" or \"file\"",
    ]
    for group in GROUPS:
        cls = _group_class(group)
        lines.append("")
        lines.append(f"[{group}]")
        default = cls()
        for f in dataclasses.fields(cls):
            value = getattr(default, f.name)
            if f.name == "correlation":
                lines.append("# correlation = \"correlation.json\"   # a "
                             "file: nucleus, intercept_ppm, reference, "
                             "terms")
                continue
            if f.name in ("measured", "measured_fractions"):
                lines.append(f"# {f.name} = []   # tables (see "
                             "md_analysis); or the --measured / "
                             "--nmr-fractions flags")
                continue
            if value is None or value == () or value == {}:
                meaning = _LEFT_OUT_MEANS.get((group, f.name))
                lines.append(f"# {f.name} =    # {f.type}"
                             + (f"; left out: {meaning}" if meaning else ""))
            else:
                lines.append(f"{f.name} = {_toml_value(value)}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# progress and Ctrl-C
# ---------------------------------------------------------------------------

class _Progress:
    """Progress lines on stderr, at most one every half second."""

    def __init__(self, quiet: bool):
        self.quiet = quiet
        self.last = 0.0
        self.start = time.perf_counter()

    def __call__(self, done: int, total: int, stage: str) -> None:
        if self.quiet:
            return
        now = time.perf_counter()
        if now - self.last < 0.5 and done < total:
            return
        self.last = now
        share = 100.0 * done / total if total else 100.0
        print(f"[{share:5.1f} %  {now - self.start:7.1f} s] {stage}",
              file=sys.stderr, flush=True)


class _CancelOnInterrupt:
    """The first Ctrl-C asks the run to stop after the step under way; the
    second stops at once."""

    def __init__(self):
        self.requested = False
        self.previous = None

    def __enter__(self):
        try:
            self.previous = signal.signal(signal.SIGINT, self._handler)
        except ValueError:          # not the main thread
            self.previous = None
        return self

    def __exit__(self, *exc):
        if self.previous is not None:
            signal.signal(signal.SIGINT, self.previous)
        return False

    def _handler(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        print("\ncancelling after the step under way (Ctrl-C again stops at "
              "once); the analyses finished so far will be written",
              file=sys.stderr, flush=True)

    def __call__(self) -> bool:
        return self.requested


# ---------------------------------------------------------------------------
# the parser
# ---------------------------------------------------------------------------

_EPILOG = """\
examples:
  py -3.11 -m facet.md describe dump.lammpstrj --type-map 1=Si,2=O,3=Na
  py -3.11 -m facet.md analyse dump.lammpstrj --type-map 1=Si,2=O,3=Na \\
      --formers Si --frames 0:100:5 --out results.xlsx
  py -3.11 -m facet.md analyse dump.lammpstrj --type-map 1=Si,2=O,3=Na \\
      --formers Si --only glass,rings --set network.ring_criterion=primitive \\
      --set network.ring_max_size=12 --out results/
  py -3.11 -m facet.md analyse dump.lammpstrj --type-map 1=Si,2=O,3=Na \\
      --only channels,modifier-density,void-regions --set channels.probe=Na \\
      --set channels.deltas_vu=0.05:1.0:0.05 --set channels.delta_vu=0.3 \\
      --set channels.modifiers=Na --set channels.k_rich=3 \\
      --set channels.void_probe_radius_ang=0.5 --set voids.radii=vdw --out r/
  py -3.11 -m facet.md template > request.toml   (then --request request.toml)

exit codes: 0 every analysis asked for was computed; 1 some analysis
produced nothing (the others are written); 2 an input is missing, out of
range or contradictory (nothing was run); 3 the model file could not be
read; 4 cancelled with Ctrl-C (the finished analyses are written); 5 an
output could not be written (the other outputs are written).

analyses:
"""


def _add_read_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("reading the model")
    group.add_argument("--type-map", metavar="MAP",
                       help="LAMMPS type (or label) -> element, e.g. "
                            "1=Si,2=O,3=Na; no element is guessed")
    group.add_argument("--units", help="LAMMPS unit style when the file "
                                       "does not say (metal, real)")
    group.add_argument("--atom-style", help="LAMMPS data file atom style")
    group.add_argument("--masses-from", metavar="DATA",
                       help="a LAMMPS data file whose Masses name the types")
    group.add_argument("--read-option", action="append", default=[],
                       metavar="KEY=VALUE",
                       help="any other md_readers.read_trajectory option")


def build_parser() -> argparse.ArgumentParser:
    from facet.core import md_analysis as ma

    parser = argparse.ArgumentParser(
        prog="py -3.11 -m facet.md",
        description="FACET's MD analyses: read a model or trajectory, "
                    "measure it, write the results with their provenance. "
                    "Nothing runs or edits MD.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG + analyses_text())
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    describe = commands.add_parser(
        "describe", help="what the reader finds in a model file",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    describe.add_argument("file")
    _add_read_options(describe)
    describe.add_argument("--ox", action="append", default=[],
                          metavar="EL=STATE",
                          help="oxidation states other than "
                               "elements.COMMON_OX, e.g. Na=1")

    commands.add_parser("analyses", help="every analysis and its inputs")
    template = commands.add_parser("template",
                                   help="a request file with every option")
    template.add_argument("--out", metavar="FILE",
                          help="write it to FILE instead of the screen")

    analyse = commands.add_parser(
        "analyse", help="run the analyses and write the results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="analyses:\n" + analyses_text())
    analyse.add_argument("file")
    _add_read_options(analyse)
    what = analyse.add_argument_group("what to measure")
    what.add_argument("--only", metavar="NAMES",
                      help="comma-separated analyses (default: every one "
                           "whose inputs are given): " + ", ".join(ma.ANALYSES))
    what.add_argument("--request", metavar="FILE",
                      help="a request file (TOML or JSON; see 'template')")
    what.add_argument("--formers", metavar="ELEMENTS",
                      help=f"network formers, e.g. Si,B; '{ma.NO_FORMERS}' "
                           "states that none are named (no default)")
    what.add_argument("--frames", metavar="START:STOP:STEP",
                      help="readable frames, e.g. 0:100:5 or 3 or 1,4,9")
    what.add_argument("--ox", action="append", default=[], metavar="EL=STATE",
                      help="oxidation states other than elements.COMMON_OX")
    what.add_argument("--v-bond", type=float, metavar="V_VU",
                      help="bond threshold in v.u. (default bv.V_BOND_DEFAULT)")
    what.add_argument("--v-list", type=float, metavar="V_VU",
                      help="tabulation threshold in v.u. (default "
                           "bv.V_LIST_DEFAULT)")
    what.add_argument("--bv-params", metavar="FILE",
                      help="a bond-valence parameter file (bv_files.load)")
    what.add_argument("--no-estimated", action="store_true",
                      help="no O'Keeffe-Brese estimated parameters: a pair "
                           "without a fitted one has no valence")
    physical = analyse.add_argument_group("physical inputs (no defaults)")
    physical.add_argument("--timestep-fs", type=float,
                          help="the MD timestep in fs (time axis)")
    physical.add_argument("--frame-interval-ps", type=float,
                          help="or the time between frames in ps")
    physical.add_argument("--temperature-k", type=float,
                          help="the run's temperature in K (conductivity)")
    physical.add_argument("--charges", metavar="CHARGES",
                          help="ionic charges: Na=0.6,O=-1.2,Si=2.4, or "
                               "'oxidation states', or 'file'")
    physical.add_argument("--absorber", help="EXAFS absorbing element")
    physical.add_argument("--nmr-correlation", metavar="FILE",
                          help="the NMR correlation (JSON or TOML) with its "
                               "reference")
    physical.add_argument("--nmr-fractions", metavar="FILE",
                          help="measured fractions to compare (JSON or TOML "
                               "list)")
    physical.add_argument("--measured", action="append", default=[],
                          metavar="RADIATION:FUNCTION:FILE",
                          help="a measured curve, e.g. neutron:S(Q):sq.dat")
    physical.add_argument("--set", action="append", default=[],
                          metavar="GROUP.OPTION=VALUE",
                          help="any request option (see 'template')")
    out = analyse.add_argument_group("output")
    out.add_argument("--out", action="append", default=[], metavar="PATH",
                     help="an .xlsx workbook or a directory of CSV files; "
                          "may be given twice")
    out.add_argument("--per-frame", action="store_true",
                     help="also write every frame's value beside the mean")
    out.add_argument("--quiet", action="store_true",
                     help="no progress lines")
    return parser


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def _read_options(args) -> dict:
    options = {}
    if args.type_map:
        options["type_map"] = _type_map(args.type_map, "--type-map")
    if args.units:
        options["units"] = args.units
    if args.atom_style:
        options["atom_style"] = args.atom_style
    if args.masses_from:
        options["masses_from"] = args.masses_from
    for item in args.read_option:
        key, value = _mapping(item, "--read-option").popitem()
        options[key] = value
    return options


def _ox(items: Sequence[str]) -> dict[str, int]:
    out = {}
    for item in items:
        for key, value in _mapping(item, "--ox").items():
            out[key] = _int(value, f"--ox {key}")
    return out


def _read(path: str, options: Mapping, out) -> object:
    from facet.core import md_analysis as ma

    try:
        return ma.read_model(path, **options)
    except FileNotFoundError:
        raise _ReadFailure(f"{path}: no such file") from None
    except (OSError, ValueError) as error:
        text = str(error)
        if "type_map" in text:
            text += ("\n(on the command line: --type-map 1=Si,2=O,... or "
                     "--masses-from the LAMMPS data file)")
        raise _ReadFailure(text) from None


class _ReadFailure(Exception):
    pass


def cmd_describe(args, out=None) -> int:
    from facet.core import md_model

    out = out or sys.stdout
    trajectory = _read(args.file, _read_options(args), out)
    for line in trajectory.describe():
        print(line, file=out)
    try:
        frame = trajectory.frame(0)
        ox = md_model.model_oxidation(frame.species, _ox(args.ox))
    except (ValueError, md_model.FrameError) as error:
        print(f"oxidation states: {error}", file=out)
        return EXIT_OK
    print(f"oxidation states (the analyses take these unless --ox says "
          f"otherwise): {ox.describe()}", file=out)
    for note in ox.notes[1:]:
        print(f"note: {note}", file=out)
    return EXIT_OK


def cmd_template(args, out=None) -> int:
    out = out or sys.stdout
    text = template_text()
    if args.out:
        try:
            Path(args.out).write_text(text, encoding="utf-8")
        except OSError as error:
            print(f"{args.out}: could not be written ({error})",
                  file=sys.stderr)
            return EXIT_WRITE
        print(f"written: {Path(args.out).resolve()}", file=out)
    else:
        out.write(text)
    return EXIT_OK


def _spec_from_args(args) -> dict:
    """The request as nested key/values: the file, then the flags, then
    --set, each over the one before."""
    spec: dict = {}
    if args.request:
        loaded = _load_mapping(args.request, "--request")
        if not isinstance(loaded, Mapping):
            raise UsageError("--request: the file needs a table at its top")
        spec = _merge(spec, _canonical(loaded, f"--request {args.request}"))
    flags: dict = {}
    if args.only:
        flags["analyses"] = _split(args.only, ",")
    if args.formers:
        flags["formers"] = args.formers
    if args.frames:
        flags["frames"] = args.frames
    if args.ox:
        # --ox adds to (and per element overrides) the file's states
        flags["ox_overrides"] = {
            **_mapping(spec.get("ox_overrides") or {}, "ox_overrides"),
            **_ox(args.ox)}
    if args.v_bond is not None:
        flags["v_bond_vu"] = args.v_bond
    if args.v_list is not None:
        flags["v_list_vu"] = args.v_list
    for name in ("timestep_fs", "frame_interval_ps", "temperature_k"):
        if getattr(args, name) is not None:
            flags[name] = getattr(args, name)
    if args.charges:
        flags["charges_e"] = args.charges
    if args.absorber:
        flags["exafs"] = {"absorber": args.absorber}
    if args.nmr_correlation:
        flags.setdefault("nmr", {})["correlation"] = args.nmr_correlation
    if args.nmr_fractions:
        loaded = _load_mapping(args.nmr_fractions, "--nmr-fractions")
        if isinstance(loaded, Mapping) and "measured_fractions" in loaded:
            loaded = loaded["measured_fractions"]     # a TOML file's tables
        flags.setdefault("nmr", {})["measured_fractions"] = _list_of(loaded)
    if args.measured:
        flags.setdefault("scattering", {})["measured"] = list(args.measured)
    spec = _merge(spec, flags)
    for item in args.set:
        path, value = _set_assignment(item)
        if len(path) == 1:
            spec[_TOP_ALIASES.get(path[0], path[0])] = value
        elif len(path) == 2 and path[0] in GROUPS:
            group = dict(spec.get(path[0]) or {})
            group[path[1]] = value
            spec[path[0]] = group
        else:
            raise UsageError(f"--set {item!r}: the key is option or "
                             f"group.option, the groups {', '.join(GROUPS)}")
    return spec


def cmd_analyse(args, out=None) -> int:
    from facet.core import bv, md_analysis as ma

    out = out or sys.stdout
    if not args.out:
        raise UsageError("--out is needed: an .xlsx workbook or a directory "
                         "for CSV files")
    for target in args.out:
        if Path(target).suffix.lower() == ".csv":
            raise UsageError(f"--out {target}: one CSV per descriptor is "
                             "written, so give a directory (or an .xlsx "
                             "file)")
    spec = _spec_from_args(args)
    request = request_from_mapping(spec)
    params = None
    if args.bv_params:
        params = _parameter_set(args.bv_params, "--bv-params")
        if args.no_estimated:
            params.allow_estimated = False
    elif args.no_estimated:
        params = bv.ParameterSet(allow_estimated=False)
    if params is not None:
        request = dataclasses.replace(request, params=params)
    read_options = {**dict(request.read_options), **_read_options(args)}
    if request.type_map is not None and "type_map" not in read_options:
        read_options["type_map"] = dict(request.type_map)

    explicit = bool(request.analyses)
    left: dict[str, list[str]] = {}
    # every input that is missing or that no analysis can take, named at
    # once before the model is read
    problems = (ma.missing_inputs(request) if explicit
                else ma.invalid_inputs(request))
    if problems:
        raise UsageError(("the analyses asked for lack inputs:\n" if explicit
                          else "the request cannot run:\n") + "\n".join(
            "  " + m.describe() for m in problems))

    trajectory = _read(args.file, read_options, out)
    try:
        if not explicit:
            # the selection reads the file: an analysis of time on a file
            # without a time axis or velocities is left out, not run to a
            # failure at the end
            runnable, left = ma.available_analyses(request, trajectory)
            if not runnable:
                raise UsageError("no analysis has every input it needs:\n"
                                 + "\n".join(f"  {n}: needs {', '.join(v)}"
                                             for n, v in left.items()))
            request = dataclasses.replace(request, analyses=runnable)
        if not args.quiet:
            for line in trajectory.describe()[:8]:
                print(line, file=sys.stderr)
            print("analyses: " + ", ".join(request.analyses), file=sys.stderr)
            for name, needs in left.items():
                print(f"left out: {name} (needs {', '.join(needs)})",
                      file=sys.stderr)
        progress = _Progress(args.quiet)
        with _CancelOnInterrupt() as cancel:
            try:
                result = ma.analyse(trajectory, request, progress=progress,
                                    cancelled=cancel)
            except ma.AnalysisCancelled:
                print("cancelled before any analysis was computed; nothing "
                      "was written", file=sys.stderr)
                return EXIT_CANCELLED
    except ma.FramesUnreadable as error:
        raise _ReadFailure(str(error)) from None
    except ValueError as error:
        # a RequestError names every input at once; any other ValueError
        # leaving analyse comes before any result exists (nothing written)
        raise UsageError(str(error)) from None
    written, failed = _write_outputs(result, args, out)
    _summary(result, written, out)
    if failed:
        return EXIT_WRITE
    if result.cancelled:
        return EXIT_CANCELLED
    return EXIT_PARTIAL if result.failed else EXIT_OK


def _write_outputs(result, args, out) -> tuple[list[Path], list[str]]:
    """Every --out target, CSV directories first (the cheaper write, and
    one that needs nothing beyond the standard library), each failure
    reported and the others still written. When none could be written, the
    results go to a CSV directory beside the first target that failed, so a
    long run is not lost to one locked or unwritable file (a workbook open
    in Excel on Windows refuses to be replaced)."""
    from facet.core import md_export

    ordered = sorted(args.out, key=lambda t: Path(t).suffix.lower() == ".xlsx")
    written: list[Path] = []
    failed: list[str] = []
    for target in ordered:
        try:
            written.extend(md_export.export(result, target,
                                            per_frame=args.per_frame))
        except (OSError, ValueError) as error:
            print(f"{target}: could not be written ({error})",
                  file=sys.stderr)
            failed.append(target)
    if failed and not written:
        first = Path(failed[0])
        base = first.parent / f"{first.stem or first.name}_csv"
        fallback, n = base, 2
        while fallback.exists():
            fallback = base.with_name(f"{base.name}_{n}")
            n += 1
        try:
            written.extend(md_export.write_csv_files(
                result, fallback, per_frame=args.per_frame))
            print(f"written instead: {fallback} (CSV files, since no --out "
                  "target could be written)", file=sys.stderr)
        except (OSError, ValueError) as error:
            print(f"{fallback}: could not be written either ({error}); "
                  "nothing was written", file=sys.stderr)
    return written, failed


def _summary(result, written: Sequence[Path], out) -> None:
    print(f"frames used: {len(result.provenance.frames_used)}"
          + (" (cancelled)" if result.cancelled else ""), file=out)
    for note in getattr(result, "model_notes", ()):
        print(f"model note: {note}", file=out)
    counts = sorted(set(result.searches.values()))
    if counts:
        print("bulk pair searches per frame (pass 1, pass 2): "
              + "; ".join(f"{a}, {b}" for a, b in counts), file=out)
    for name, output in result.outputs.items():
        state = (f"{len(output.tables)} descriptors" if output.ok
                 else f"not computed: {output.error}")
        print(f"  {name:24s} {output.seconds:8.2f} s  {state}", file=out)
    print(f"total {result.timings_s.get('total', 0.0):.2f} s", file=out)
    if written:
        folders = sorted({str(Path(p).parent) for p in written})
        print(f"written: {len(written)} file(s) in " + ", ".join(folders),
              file=out)


def _tolerant_streams() -> None:
    """Text the console cannot encode (a Windows code page and a Greek
    letter in a note, say) is escaped rather than ending the run."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass


def main(argv: Sequence[str] | None = None) -> int:
    _tolerant_streams()
    parser = build_parser()
    try:
        args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    except SystemExit as stop:
        return int(stop.code or 0)
    if args.command is None:
        parser.print_help()
        return EXIT_USAGE
    try:
        if args.command == "describe":
            return cmd_describe(args)
        if args.command == "analyses":
            print(analyses_text())
            return EXIT_OK
        if args.command == "template":
            return cmd_template(args)
        return cmd_analyse(args)
    except UsageError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    except _ReadFailure as error:
        print(f"error: the model could not be read: {error}", file=sys.stderr)
        return EXIT_READ
    except ValueError as error:
        # before any result exists: an input the checks did not name
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
        return EXIT_CANCELLED
