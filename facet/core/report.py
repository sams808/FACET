"""Reports on a set of CIFs: bond valences, bond-valence sums, distances and
cutoffs for every site of every file, written in the format asked for.

The batch counterpart of the window. Each file is read and analysed exactly
as the window analyses it (``coordination.analyse_structure`` at one bond
threshold and one listing threshold, with one parameter set), and the same
numbers are laid out as flat tables:

* **sites**: one row per site -- coordination number, the bond-valence sum
  over the bonded set and to the listing cut, phi, the polyhedron's distance
  statistics, the plateau the threshold sits on;
* **contacts**: one row per contact -- distance, valence, occupancy, whether
  it counts as a bond, the parameter it was read with;
* **plateaus**: every coordination number each site can produce and over
  what range of threshold (the staircase, as a table);
* **cutoffs**: the distance each cation-anion pair's thresholds correspond
  to, from the pair's own parameter (``bv_report.cutoff_table``);
* **balance**: the cation and anion totals of each structure against each
  other (``bv_report.charge_balance``).

Six formats write the same tables: an ``.xlsx`` workbook, a set of ``.csv``
files, ``.json``, an ``.html`` page, Markdown and plain text. Every one
carries the provenance -- thresholds, parameter set, version, time, and the
files -- because a table of coordination numbers without the threshold that
produced them is not a result.

No Qt here: the window's dialog and ``py -3.11 -m facet.report`` both call
this module.
"""
from __future__ import annotations

import datetime as _dt
import html as _html
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from . import bv, bv_report, cif, coordination, exporters
from ..version import __version__

FORMATS = ("xlsx", "csv", "json", "html", "md", "txt")
FORMAT_NAMES = {
    "xlsx": "Excel workbook (.xlsx)",
    "csv": "CSV files, one per table",
    "json": "JSON (.json)",
    "html": "Web page (.html)",
    "md": "Markdown (.md)",
    "txt": "Plain text (.txt)",
}
TABLES = ("sites", "contacts", "plateaus", "cutoffs", "balance")
# with a second threshold: the sites whose coordination number changes
# between the two, first, and every site at the second threshold
TABLES_TWO = ("cn_changes", "sites_at_second")


@dataclass
class StructureReport:
    """One file: what was read and what was found, or why it could not be."""

    name: str
    path: str
    formula: str = ""
    spacegroup: str = ""
    n_atoms: int = 0
    notes: list[str] = field(default_factory=list)
    results: list = field(default_factory=list)
    cutoffs: bv_report.CutoffTable | None = None
    balance: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class Report:
    v_bond: float
    v_list: float
    parameter_set: str
    anions: bool
    generated: str
    structures: list[StructureReport] = field(default_factory=list)
    # a second bond threshold to compare against, or None
    v_bond_other: float | None = None

    # -- provenance ----------------------------------------------------------
    def provenance(self) -> list[str]:
        lines = [
            f"FACET {__version__} report, {self.generated}",
            f"bond threshold v_bond = {self.v_bond:.4f} v.u. "
            f"(a contact above it counts as a bond)",
            f"listing threshold v_list = {self.v_list:.4f} v.u. "
            f"(contacts above it are listed and summed)",
            f"bond-valence parameters: {self.parameter_set}",
            "sites analysed: " + ("cations and anions" if self.anions
                                  else "cations only"),
            f"files: {len(self.structures)} "
            f"({sum(1 for s in self.structures if s.error)} not read)",
        ]
        if self.v_bond_other is not None:
            lines.insert(2, f"compared against a second bond threshold "
                            f"{self.v_bond_other:.4f} v.u.")
        for s in self.structures:
            lines.append(f"  {s.path}" + (f"  -- {s.error}" if s.error else ""))
        return lines

    # -- the tables ----------------------------------------------------------
    def site_rows(self) -> list[dict]:
        rows = []
        for s in self.structures:
            for r in s.results:
                plateau = r.current_plateau
                rows.append({
                    "structure": s.name,
                    "formula": s.formula,
                    "spacegroup": s.spacegroup,
                    "site": r.label,
                    "element": r.element,
                    "oxidation_state": r.ox,
                    "oxidation_source": r.ox_source,
                    "wyckoff": r.wyckoff or "",
                    "multiplicity": r.multiplicity,
                    "cn": r.cn_valence,
                    "cn_occupancy": _num(r.cn_occupancy),
                    "cn_ecoN": _num(r.cn_ecoN),
                    "cn_gap_split": r.cn_gap,
                    "cn_listed": r.cn_listed,
                    "bvs": _num(r.bvs),
                    "bvs_uncertainty": _num(r.bvs_uncertainty),
                    "bvs_listed": _num(r.bvs_listed),
                    "valence_discrepancy": _num(r.valence_discrepancy),
                    "phi": _num(r.phi),
                    "phi_listed": _num(r.phi_listed),
                    "bvv": _num(r.bvv),
                    "d_min": _num(r.shape.get("d_min")),
                    "d_max": _num(r.shape.get("d_max")),
                    "d_mean": _num(r.shape.get("d_mean")),
                    "spread": _num(r.shape.get("spread")),
                    "polyhedron_volume": _num(r.shape.get("volume")),
                    "plateau_cn": plateau.cn if plateau else None,
                    "plateau_v_low": _num(plateau.v_low) if plateau else None,
                    "plateau_v_high": _num(plateau.v_high) if plateau else None,
                    "plateau_decades": _num(plateau.width_decades) if plateau else None,
                    "plateau_angstrom": _num(plateau.width_angstrom) if plateau else None,
                    "nearest_cation": _num(r.nearest_cation),
                    "estimated_parameters": r.uses_estimated_params,
                    "notes": "; ".join(r.notes),
                })
        return rows

    def contact_rows(self) -> list[dict]:
        rows = []
        for s in self.structures:
            for r in s.results:
                running = 0.0
                for c in sorted(r.contacts, key=lambda c: c.distance):
                    if c.has_valence:
                        running += c.valence * c.occupancy
                    rows.append({
                        "structure": s.name,
                        "site": r.label,
                        "element": r.element,
                        "neighbour": c.label,
                        "neighbour_element": c.element,
                        "distance": _num(c.distance),
                        "valence": _num(c.valence),
                        "occupancy": _num(c.occupancy),
                        "running_sum": _num(running) if c.has_valence else None,
                        "is_bond": bool(c.has_valence and c.valence > self.v_bond),
                        "r0": _num(c.param.r0) if c.param else None,
                        "b": _num(c.param.b) if c.param else None,
                        "parameter_fitted": c.param.fitted if c.param else None,
                        "parameter_source": c.param.source if c.param else "",
                    })
        return rows

    def plateau_rows(self) -> list[dict]:
        rows = []
        for s in self.structures:
            for r in s.results:
                for q in r.plateaus:
                    rows.append({
                        "structure": s.name,
                        "site": r.label,
                        "element": r.element,
                        "cn": q.cn,
                        "v_low": _num(q.v_low),
                        "v_high": _num(q.v_high),
                        "width_decades": _num(q.width_decades),
                        "d_high": _num(q.d_high),
                        "d_low": _num(q.d_low),
                        "width_angstrom": _num(q.width_angstrom),
                        "holds_v_bond": q.contains(self.v_bond),
                    })
        return rows

    def cutoff_rows(self) -> list[dict]:
        rows = []
        for s in self.structures:
            if s.cutoffs is None:
                continue
            for p in s.cutoffs.rows:
                rows.append({
                    "structure": s.name,
                    "pair": p.label,
                    "cation": p.cation,
                    "cation_ox": p.cation_ox,
                    "anion": p.anion,
                    "r0": _num(p.r0),
                    "b": _num(p.b),
                    "fitted": p.fitted,
                    "source": p.source,
                    "d_bond": _num(p.d_bond),
                    "d_list": _num(p.d_list),
                    "window": _num(p.window),
                    "shortest": _num(p.shortest),
                    "n_bonds": p.n_bonds,
                    "n_listed": p.n_listed,
                })
        return rows

    def balance_rows(self) -> list[dict]:
        rows = []
        for s in self.structures:
            row = {"structure": s.name, "formula": s.formula,
                   "spacegroup": s.spacegroup, "n_atoms": s.n_atoms,
                   "read": s.error is None, "problem": s.error or ""}
            for k, v in s.balance.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    row[k] = _num(v) if isinstance(v, float) else v
            rows.append(row)
        return rows

    def tables(self) -> dict[str, list[dict]]:
        out = {}
        if self.v_bond_other is not None:
            changes, second = self.second_threshold_rows()
            out["cn_changes"] = changes
            out["sites_at_second"] = second
        out.update({"sites": self.site_rows(), "contacts": self.contact_rows(),
                    "plateaus": self.plateau_rows(), "cutoffs": self.cutoff_rows(),
                    "balance": self.balance_rows()})
        return out

    def second_threshold_rows(self) -> tuple[list[dict], list[dict]]:
        """Every site reclassified at the second threshold, and first the
        sites whose coordination number differs between the two."""
        from .coordination import reclassify
        other = float(self.v_bond_other)
        changes, second = [], []
        for s in self.structures:
            for r in s.results:
                moved = reclassify(r, other)
                row = {
                    "structure": s.name, "site": r.label, "element": r.element,
                    "cn_at_first": r.cn_valence, "cn_at_second": moved.cn_valence,
                    "bvs_at_first": _num(r.bvs), "bvs_at_second": _num(moved.bvs),
                    "phi_at_first": _num(r.phi), "phi_at_second": _num(moved.phi),
                    "v_bond_first": self.v_bond, "v_bond_second": other,
                }
                second.append(row)
                if moved.cn_valence != r.cn_valence:
                    changes.append(dict(row, cn_change=moved.cn_valence - r.cn_valence))
        return changes, second


def _num(value):
    """A float that JSON and a spreadsheet can carry: None for NaN."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    return None if math.isnan(f) or math.isinf(f) else f


# -- analysing ---------------------------------------------------------------

def collect_cifs(paths: Iterable[str | Path]) -> list[Path]:
    """Files to report on: each path as given, or every .cif in a folder."""
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out.extend(sorted(q for q in p.iterdir()
                              if q.suffix.lower() in (".cif", ".mcif")))
        else:
            out.append(p)
    seen, unique = set(), []
    for p in out:
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def analyse_files(paths: Iterable[str | Path],
                  params: bv.ParameterSet | None = None,
                  v_bond: float = bv.V_BOND_DEFAULT,
                  v_list: float = bv.V_LIST_DEFAULT,
                  anions: bool = False,
                  progress: Callable[[int, int, str], bool] | None = None,
                  v_bond_other: float | None = None,
                  ) -> Report:
    """Read and analyse every file. A file that cannot be read is kept in the
    report with its reason, so the table says which files are missing and
    why rather than silently having fewer rows.

    ``progress(done, total, name)`` is called before each file; returning
    False stops the run, and the report then holds what was done.
    """
    params = params or bv.DEFAULT
    if v_bond < v_list:
        raise ValueError(f"v_bond ({v_bond}) is below v_list ({v_list}): a "
                         "bond must be at least a listed contact")
    files = collect_cifs(paths)
    if v_bond_other is not None and v_bond_other < v_list:
        raise ValueError(f"the second threshold ({v_bond_other}) is below "
                         f"v_list ({v_list})")
    report = Report(v_bond=float(v_bond), v_list=float(v_list),
                    parameter_set=getattr(params, "name", None) or "built-in",
                    anions=bool(anions),
                    generated=_dt.datetime.now().isoformat(timespec="seconds"),
                    v_bond_other=None if v_bond_other is None else float(v_bond_other))
    for i, path in enumerate(files):
        if progress is not None and progress(i, len(files), path.name) is False:
            break
        report.structures.append(_one(path, params, v_bond, v_list, anions))
    return report


def _one(path: Path, params, v_bond: float, v_list: float,
         anions: bool) -> StructureReport:
    item = StructureReport(name=path.name, path=str(path))
    try:
        structure = cif.read(str(path))
    except Exception as error:                        # noqa: BLE001
        item.error = f"{type(error).__name__}: {error}"
        return item
    item.formula = structure.formula or ""
    item.spacegroup = structure.spacegroup_hm or ""
    item.n_atoms = len(structure.atoms)
    item.notes = list(structure.notes or [])
    try:
        item.results = coordination.analyse_structure(
            structure, params, v_bond=v_bond, v_list=v_list,
            cations_only=not anions)
        item.cutoffs = bv_report.cutoff_table(structure, params, v_bond=v_bond,
                                              v_list=v_list, results=item.results)
        item.balance = bv_report.charge_balance(structure, item.results, params,
                                                v_bond=v_bond, v_list=v_list)
    except Exception as error:                        # noqa: BLE001
        item.error = f"analysis failed: {type(error).__name__}: {error}"
    return item


# -- writing -----------------------------------------------------------------

def write_report(report: Report, path: str | Path, fmt: str,
                 tables_wanted: Sequence[str] | None = None) -> list[Path]:
    """Write the report in one format; returns the files written.
    ``tables_wanted`` limits the tables to those named (a template's
    choice); None writes every one."""
    fmt = fmt.lower().lstrip(".")
    if fmt == "markdown":
        fmt = "md"
    if fmt == "text":
        fmt = "txt"
    if fmt not in FORMATS:
        raise ValueError(f"no report format named {fmt!r}; one of "
                         + ", ".join(FORMATS))
    path = Path(path)
    if path.suffix.lower() != f".{fmt}" and fmt != "csv":
        path = path.with_suffix(f".{fmt}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tables = report.tables()
    if tables_wanted:
        wanted = {str(t) for t in tables_wanted}
        tables = {k: v for k, v in tables.items() if k in wanted} or tables
    provenance = report.provenance()
    sidecar = _provenance_json(report, path)
    if fmt == "xlsx":
        out = exporters.write_xlsx(tables, path, provenance=provenance)
        return ([out] if isinstance(out, Path) else list(out)) + [sidecar]
    if fmt == "csv":
        stem = path.with_suffix("") if path.suffix.lower() == ".csv" else path
        written = []
        for name, rows in tables.items():
            written.append(exporters.write_csv(
                rows, Path(f"{stem}_{name}.csv"), provenance=provenance))
        return written + [sidecar]
    if fmt == "json":
        payload = {"provenance": provenance, "v_bond": report.v_bond,
                   "v_list": report.v_list, "parameter_set": report.parameter_set,
                   "anions": report.anions, "generated": report.generated,
                   "tables": tables}
        path.write_text(json.dumps(payload, indent=1, default=str),
                        encoding="utf-8")
        return [path, sidecar]
    if fmt == "html":
        path.write_text(_as_html(report, tables, provenance), encoding="utf-8")
        return [path, sidecar]
    if fmt == "md":
        path.write_text(_as_markdown(report, tables, provenance), encoding="utf-8")
        return [path, sidecar]
    path.write_text(_as_text(report, tables, provenance), encoding="utf-8")
    return [path, sidecar]


def _provenance_json(report: Report, beside: Path) -> Path:
    """The provenance as a machine-readable file beside the export, so a
    pipeline can check what it reads without parsing a header."""
    target = Path(str(beside.with_suffix("")) + ".provenance.json")
    payload = {
        "application": "FACET", "version": __version__,
        "generated": report.generated, "v_bond_vu": report.v_bond,
        "v_list_vu": report.v_list, "v_bond_other_vu": report.v_bond_other,
        "parameter_set": report.parameter_set, "anions": report.anions,
        "files": [{"name": s.name, "path": s.path, "formula": s.formula,
                   "spacegroup": s.spacegroup, "n_atoms": s.n_atoms,
                   "error": s.error} for s in report.structures],
        "lines": report.provenance(),
    }
    target.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    return target


# -- templates ----------------------------------------------------------------------

def template_fields(fmt: str, v_bond: float, v_list: float, anions: bool,
                    v_bond_other: float | None = None,
                    tables: Sequence[str] | None = None) -> dict:
    """What a saved report template holds."""
    return {"format": fmt, "v_bond": float(v_bond), "v_list": float(v_list),
            "anions": bool(anions),
            "v_bond_other": None if v_bond_other is None else float(v_bond_other),
            "tables": list(tables) if tables else None}


def save_template(name: str, fields: dict) -> None:
    from . import prefs
    templates = prefs.get("report_templates", {}) or {}
    templates[str(name)] = dict(fields)
    prefs.set("report_templates", templates)


def load_template(name: str) -> dict | None:
    from . import prefs
    templates = prefs.get("report_templates", {}) or {}
    found = templates.get(str(name))
    return dict(found) if isinstance(found, dict) else None


def template_names() -> list[str]:
    from . import prefs
    return sorted((prefs.get("report_templates", {}) or {}).keys())


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _as_html(report: Report, tables: dict, provenance: list[str]) -> str:
    parts = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
             f"<title>FACET report {_html.escape(report.generated)}</title>",
             "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;"
             "color:#1c1f24}table{border-collapse:collapse;margin:8px 0 24px;"
             "font-size:12px}th,td{border:1px solid #d0d4dc;padding:3px 7px;"
             "text-align:right}th{background:#eef1f6}td:first-child,"
             "th:first-child{text-align:left}pre{background:#f4f6fa;"
             "padding:10px;font-size:12px}h2{margin-top:28px}</style></head><body>",
             "<h1>FACET report</h1>",
             "<pre>" + _html.escape("\n".join(provenance)) + "</pre>"]
    for name in tables:
        rows = tables[name]
        parts.append(f"<h2>{name}</h2>")
        if not rows:
            parts.append("<p>nothing to report</p>")
            continue
        keys = list(rows[0].keys())
        parts.append("<table><thead><tr>" + "".join(
            f"<th>{_html.escape(k)}</th>" for k in keys) + "</tr></thead><tbody>")
        for row in rows:
            parts.append("<tr>" + "".join(
                f"<td>{_html.escape(_fmt(row.get(k)))}</td>" for k in keys) + "</tr>")
        parts.append("</tbody></table>")
    parts.append("</body></html>")
    return "\n".join(parts)


def _as_markdown(report: Report, tables: dict, provenance: list[str]) -> str:
    out = ["# FACET report", ""]
    out += [f"    {line}" for line in provenance]
    out.append("")
    for name in tables:
        rows = tables[name]
        out.append(f"## {name}")
        out.append("")
        if not rows:
            out.append("nothing to report")
            out.append("")
            continue
        keys = list(rows[0].keys())
        out.append("| " + " | ".join(keys) + " |")
        out.append("|" + "|".join("---" for _ in keys) + "|")
        for row in rows:
            out.append("| " + " | ".join(
                _fmt(row.get(k)).replace("|", "\\|") for k in keys) + " |")
        out.append("")
    return "\n".join(out)


def _as_text(report: Report, tables: dict, provenance: list[str]) -> str:
    out = ["FACET report", "=" * 12, ""]
    out += provenance
    out.append("")
    for name in tables:
        rows = tables[name]
        out.append(name)
        out.append("-" * len(name))
        if not rows:
            out.append("nothing to report")
            out.append("")
            continue
        keys = list(rows[0].keys())
        cells = [[_fmt(row.get(k)) for k in keys] for row in rows]
        widths = [max(len(k), *(len(c[i]) for c in cells))
                  for i, k in enumerate(keys)]
        out.append("  ".join(k.ljust(w) for k, w in zip(keys, widths)))
        for c in cells:
            out.append("  ".join(v.rjust(w) for v, w in zip(c, widths)))
        out.append("")
    # the per-structure cutoff tables as the window prints them, for reading
    for s in report.structures:
        if s.cutoffs is not None:
            out.append(s.name)
            out.append(s.cutoffs.as_text())
            out.append("")
    return "\n".join(out)
