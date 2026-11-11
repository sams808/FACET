"""Optical basicity from the command line.

    python -m facet.basicity scales
    python -m facet.basicity oxide Si
    python -m facet.basicity composition "30Na2O-70SiO2" --scale all
    python -m facet.basicity series Na2O SiO2 --steps 11
    python -m facet.basicity band 0.60 --ion Pb2+
    python -m facet.basicity structure albite.cif --formers Si,Al --csv sites.csv
    python -m facet.basicity md glass.data --formers Si,B --correlation
    python -m facet.basicity field glass.data --probe Ca --cube lambda.cube
    python -m facet.basicity plot histogram glass.data --formers Si,B -o hist.png
    python -m facet.basicity plot slice glass.data --probe Ca --at 0.5
    python -m facet.basicity view glass.data

A figure is written to the file ``-o`` names, or opened in a window without
it. Exit code 0 on success, 2 for something that could not be read or
computed, with the reason on the error stream.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import api


def _list(text) -> list[str]:
    return [t for t in str(text or "").replace(";", ",").split(",") if t.strip()]


def _mapping(text, cast=str) -> dict:
    out = {}
    for item in _list(text):
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"{item!r}: expected name=value")
        out[key.strip()] = cast(value.strip())
    return out


def _type_map(text) -> dict:
    out = {}
    for key, value in _mapping(text).items():
        out[int(key) if key.isdigit() else key] = value
    return out


def _add_scale(parser, many: bool = False) -> None:
    choices = list(api.SCALE_KEYS) + (["all"] if many else [])
    parser.add_argument("--scale", default=api.DEFAULT_SCALE, choices=choices,
                        help="the scale of basicity moderating parameters "
                             f"(default {api.DEFAULT_SCALE})")


def _add_file(parser) -> None:
    parser.add_argument("file", help="a crystal structure or an MD model")
    _add_scale(parser)
    parser.add_argument("--v-bond", type=float, default=api.bv.V_BOND_DEFAULT,
                        help="bond threshold in v.u. (default %(default)s)")
    parser.add_argument("--formers", default="",
                        help="network formers, comma separated (Si,Al): the "
                             "free / NBO / BO split and the linkage names")
    parser.add_argument("--as", dest="kind", choices=("auto", "crystal", "md"),
                        default="auto", help="how to read the file")
    parser.add_argument("--frame", type=int, default=-1,
                        help="MD: the frame, from 0, negative from the end "
                             "(default the last)")
    parser.add_argument("--ox", default="",
                        help="MD: oxidation states that differ from the "
                             "usual ones (Fe=2,Sn=2)")
    parser.add_argument("--type-map", default="",
                        help="MD: LAMMPS types or labels to elements "
                             "(1=Si,2=O,3=Na)")


def _add_probe(parser) -> None:
    parser.add_argument("--probe", required=True,
                        help="the probe cation (Ca, Na, Pb, ...)")
    parser.add_argument("--probe-ox", type=int, default=None,
                        help="its oxidation state (default the model's or "
                             "the usual one)")
    parser.add_argument("--grid", type=float, default=None,
                        help="grid spacing in Å (default 0.25 for a crystal, "
                             "1.0 for an MD model)")
    parser.add_argument("--smooth", type=float, default=0.0,
                        help="average over a Gaussian of this width in Å")
    parser.add_argument("--floor", type=float, default=0.2,
                        help="the probe's valence sum below which the field "
                             "is undefined (default %(default)s v.u.)")


def _add_output(parser) -> None:
    parser.add_argument("--json", action="store_true",
                        help="print a JSON mapping instead of tables")
    parser.add_argument("--csv", default=None, metavar="FILE",
                        help="also write the rows to a CSV file")


def _load(args):
    formers = _list(args.formers)
    md = (args.kind == "md") or (args.kind == "auto" and api.is_md_file(args.file))
    if not md:
        return api.structure(args.file, args.scale, v_bond=args.v_bond,
                             formers=formers)
    options = {}
    if args.type_map:
        options["type_map"] = _type_map(args.type_map)
    return api.frame(args.file, args.scale, index=args.frame,
                     v_bond=args.v_bond, formers=formers or None,
                     ox=_mapping(args.ox, int) or None, **options)


def _field(result, args):
    if isinstance(result, api.StructureResult):
        return result.field(args.probe, args.probe_ox,
                            resolution=args.grid or 0.25, floor=args.floor,
                            smooth=args.smooth)
    return result.field(args.probe, args.probe_ox, grid=args.grid or 1.0,
                        floor=args.floor, smooth=args.smooth)


def _emit(result, args) -> None:
    if getattr(args, "csv", None):
        print(f"wrote {result.to_csv(args.csv)}", file=sys.stderr)
    if getattr(args, "json", False):
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.table())


# --- the commands ----------------------------------------------------------------------

def _cmd_scales(args) -> None:
    rows = api.scales()
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    for row in rows:
        print(f"{row['key']}: {row['name']}\n  {row['source']}")
        if row["listed"]:
            print("  lists " + ", ".join(row["listed"]))
        print()
    print("A cation a scale does not list takes gamma = 1.36 (chi - 0.26), "
          "Pauling's 1960 chi.")


def _cmd_oxide(args) -> None:
    keys = api.SCALE_KEYS if args.scale == "all" else (args.scale,)
    rows = []
    for key in keys:
        m = api.moderator(args.element, args.ox, key)
        rows.append([key, m["species"], m["gamma"], m["Lambda oxide"],
                     m["Lambda fluoride"], m["from"]])
    headers = ("scale", "species", "gamma", "Lambda oxide", "Lambda fluoride",
               "gamma from")
    if args.json:
        print(json.dumps([dict(zip(headers, r)) for r in rows], indent=2))
    else:
        print(api.text_table(headers, rows))


def _cmd_composition(args) -> None:
    unit = "wt" if args.wt else "mol"
    if args.scale == "all":
        found = [api.composition(args.composition, key, unit=unit)
                 for key in api.SCALE_KEYS]
        if args.json:
            print(json.dumps([f.to_dict() for f in found], indent=2))
            return
        print(f"{found[0].result.text}   ({unit} amounts)")
        print(api.text_table(
            ("scale", "Lambda", "alpha(O2-) / Å3", "Pb2+ band / cm-1"),
            [[f.scale, f.value, f.polarisability, f.band("Pb2+")]
             for f in found]))
        return
    _emit(api.composition(args.composition, args.scale, unit=unit), args)


def _cmd_series(args) -> None:
    keys = api.SCALE_KEYS if args.scale == "all" else (args.scale,)
    columns = [api.series(args.start, args.end, key, steps=args.steps)
               for key in keys]
    headers = (f"x({args.end})",) + tuple(keys)
    rows = [[float(columns[0][0][k])] + [float(c[1][k]) for c in columns]
            for k in range(len(columns[0][0]))]
    if args.csv:
        api._write_csv(args.csv, headers, rows)
        print(f"wrote {args.csv}", file=sys.stderr)
    if args.json:
        print(json.dumps([dict(zip(headers, r)) for r in rows], indent=2))
    else:
        print(api.text_table(headers, rows))


def _cmd_band(args) -> None:
    ions = api.PROBE_IONS if args.ion == "all" else (args.ion,)
    if args.from_band:
        rows = [[ion, args.value, api.basicity_from_band(args.value, ion)]
                for ion in ions]
        print(api.text_table(("probe ion", "band / cm-1", "Lambda"), rows))
        return
    rows = [[ion, args.value, api.probe_band(args.value, ion),
             api.probe_wavelength(args.value, ion)] for ion in ions]
    print(api.text_table(("probe ion", "Lambda", "band / cm-1", "nm"), rows))
    print(f"alpha(O2-) = {api.oxide_polarisability(args.value):.3f} Å3")


def _cmd_structure(args) -> None:
    _emit(_load(args), args)


def _cmd_md(args) -> None:
    args.kind = "md"
    result = _load(args)
    _emit(result, args)
    if args.correlation and not args.json:
        found = result.correlation(args.r_max, args.bin)
        print()
        print(api.text_table(
            ("r / Å", "C(r)", "pairs"),
            [[float(r), float(c), int(n)] for r, c, n in
             zip(found.r_ang, found.value, found.pairs) if n > 0]))
        for note in found.notes:
            print(f"note: {note}")


def _cmd_field(args) -> None:
    field = _field(_load(args), args)
    if args.cube:
        print(f"wrote {field.to_cube(args.cube)}", file=sys.stderr)
    if args.json:
        print(json.dumps(field.to_dict(), indent=2))
        return
    lo, hi = field.range
    shape = " x ".join(str(n) for n in field.values.shape)
    print(f"{field.field.label}: {shape} points, lambda {lo:.3f} to {hi:.3f}, "
          f"{100 * field.field.fraction_defined:.1f} % of the cell defined")
    for note in field.notes:
        print(f"note: {note}")


def _cmd_plot(args) -> None:
    from . import plot

    show = not args.output
    kind = args.kind_of_plot
    if kind == "series":
        if len(args.inputs) != 2:
            raise ValueError("plot series takes two compositions")
        plot.series(args.inputs[0], args.inputs[1],
                    None if args.scale == "all" else [args.scale],
                    save=args.output, show=show)
        return
    if len(args.inputs) != 1:
        raise ValueError(f"plot {kind} takes one file")
    args.file = args.inputs[0]
    if args.scale == "all":
        args.scale = api.DEFAULT_SCALE
    result = _load(args)
    if kind == "sites":
        if not isinstance(result, api.StructureResult):
            raise ValueError("plot sites is for a crystal structure; use "
                             "plot histogram for an MD model")
        plot.sites(result, save=args.output, show=show)
    elif kind == "histogram":
        plot.histogram(result, args.by, bins=args.bins, save=args.output,
                       show=show)
    elif kind == "linkages":
        plot.linkages(result, save=args.output, show=show)
    elif kind == "slice":
        if not args.probe:
            raise ValueError("plot slice needs --probe")
        plot.slice(_field(result, args), args.axis, args.at,
                   thickness=args.thickness, save=args.output, show=show)
    elif kind == "correlation":
        if isinstance(result, api.StructureResult):
            raise ValueError("plot correlation is for an MD model")
        plot.correlation(result, r_max=args.r_max, bin=args.bin,
                         save=args.output, show=show)
    elif kind == "band":
        if isinstance(result, api.StructureResult):
            raise ValueError("plot band is for an MD model")
        if not args.probe:
            raise ValueError("plot band needs --probe (Tl, Pb or Bi)")
        plot.band(result, args.probe, save=args.output, show=show)
    if args.output:
        print(f"wrote {args.output}", file=sys.stderr)


def _cmd_view(args) -> int:
    from ..ui.app import main as run

    return run(["facet", args.file, "--color-by-basicity"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="facet-basicity",
        description="Optical basicity (Duffy and Ingram): of a composition, "
                    "of every anion of a crystal structure or an MD model, "
                    "and as a field a probe cation senses.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scales", help="the scales of gamma and their sources")
    p.add_argument("--json", action="store_true")
    p.set_defaults(run=_cmd_scales)

    p = sub.add_parser("oxide", help="gamma and Lambda of one cation's oxide")
    p.add_argument("element")
    p.add_argument("--ox", type=int, default=None, help="oxidation state")
    _add_scale(p, many=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(run=_cmd_oxide, scale="all")

    p = sub.add_parser("composition",
                       help="Lambda of a written composition (30Na2O-70SiO2)")
    p.add_argument("composition")
    p.add_argument("--wt", action="store_true", help="the amounts are weights")
    _add_scale(p, many=True)
    _add_output(p)
    p.set_defaults(run=_cmd_composition)

    p = sub.add_parser("series", help="Lambda along the join of two compositions")
    p.add_argument("start")
    p.add_argument("end")
    p.add_argument("--steps", type=int, default=11)
    _add_scale(p, many=True)
    _add_output(p)
    p.set_defaults(run=_cmd_series)

    p = sub.add_parser("band", help="a probe ion's band at a Lambda, the "
                                    "oxide-ion polarisability, or the reverse")
    p.add_argument("value", type=float, help="Lambda, or with --from-band a "
                                             "band frequency in cm-1")
    p.add_argument("--ion", default="all", choices=api.PROBE_IONS + ("all",))
    p.add_argument("--from-band", action="store_true",
                   help="the value is a measured band; give Lambda")
    p.set_defaults(run=_cmd_band)

    p = sub.add_parser("structure", help="every anion site of a crystal "
                                         "structure (or of an MD frame)")
    _add_file(p)
    _add_output(p)
    p.set_defaults(run=_cmd_structure)

    p = sub.add_parser("md", help="every anion of one frame of an MD model")
    _add_file(p)
    _add_output(p)
    p.add_argument("--correlation", action="store_true",
                   help="also print C(r), the correlation of lambda with "
                        "distance")
    p.add_argument("--r-max", type=float, default=10.0)
    p.add_argument("--bin", type=float, default=0.25)
    p.set_defaults(run=_cmd_md)

    p = sub.add_parser("field", help="lambda as a probe cation senses it, "
                                     "on a grid; --cube writes it for VESTA")
    _add_file(p)
    _add_probe(p)
    p.add_argument("--cube", default=None, metavar="FILE",
                   help="write a Gaussian cube file")
    p.add_argument("--json", action="store_true")
    p.set_defaults(run=_cmd_field)

    p = sub.add_parser("plot", help="a quick figure: series, sites, "
                                    "histogram, linkages, slice, correlation, "
                                    "band")
    p.add_argument("kind_of_plot", choices=("series", "sites", "histogram",
                                            "linkages", "slice", "correlation",
                                            "band"))
    p.add_argument("inputs", nargs="+",
                   help="a file, or for series two compositions")
    p.add_argument("-o", "--output", default=None, metavar="FILE",
                   help="write the figure (png, pdf, svg); without it a "
                        "window opens")
    p.add_argument("--scale", default="all",
                   choices=list(api.SCALE_KEYS) + ["all"])
    p.add_argument("--v-bond", type=float, default=api.bv.V_BOND_DEFAULT)
    p.add_argument("--formers", default="")
    p.add_argument("--as", dest="kind", choices=("auto", "crystal", "md"),
                   default="auto")
    p.add_argument("--frame", type=int, default=-1)
    p.add_argument("--ox", default="")
    p.add_argument("--type-map", default="")
    p.add_argument("--by", default="class",
                   choices=("class", "element", "linkage"),
                   help="histogram: what to split the anions by")
    p.add_argument("--bins", type=int, default=40)
    p.add_argument("--probe", default=None)
    p.add_argument("--probe-ox", type=int, default=None)
    p.add_argument("--grid", type=float, default=None)
    p.add_argument("--smooth", type=float, default=0.0)
    p.add_argument("--floor", type=float, default=0.2)
    p.add_argument("--axis", default="c", choices=("a", "b", "c"))
    p.add_argument("--at", type=float, default=0.5,
                   help="slice: the fraction along the axis")
    p.add_argument("--thickness", type=float, default=1.5,
                   help="slice: atoms within this many Å are drawn")
    p.add_argument("--r-max", type=float, default=10.0)
    p.add_argument("--bin", type=float, default=0.25)
    p.set_defaults(run=_cmd_plot)

    p = sub.add_parser("view", help="open the file in FACET with the atoms "
                                    "coloured by microscopic basicity")
    p.add_argument("file")
    p.set_defaults(run=_cmd_view)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):         # a console that lacks a
        try:                                        # character prints '?'
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    try:
        return int(args.run(args) or 0)
    except BrokenPipeError:                         # output piped to head
        return 0
    except (ValueError, KeyError, IndexError, OSError, ImportError) as error:
        print(f"facet-basicity: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
