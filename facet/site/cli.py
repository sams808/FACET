"""``py -3.11 -m facet.site``: one site's analysis, printed.

    py -3.11 -m facet.site structure.cif
    py -3.11 -m facet.site structure.cif --site Bi1 --v-bond 0.10
    py -3.11 -m facet.site structure.cif --all --format md

What the Site tab shows, as text, Markdown or LaTeX: the coordination
number, the bond-valence sum, ECoN, phi, the plateau, every contact with
its distance, valence and running sum, and the staircase as rows. No Qt is
imported. Exit code 2 for a site the file does not have.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from facet.core import bv, coordination, readers
from facet.core.site_text import site_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="py -3.11 -m facet.site",
        description="One site's coordination analysis, as the Site tab "
                    "shows it, printed as text, Markdown or LaTeX.")
    parser.add_argument("file", help="a CIF or any structure file FACET reads")
    parser.add_argument("--site", metavar="LABEL",
                        help="the site's label (default: the first cation)")
    parser.add_argument("--all", action="store_true",
                        help="every analysed site, one after another")
    parser.add_argument("--anions", action="store_true",
                        help="analyse the anion sites as well")
    parser.add_argument("--v-bond", type=float, default=bv.V_BOND_DEFAULT,
                        help=f"bond threshold in v.u. (default {bv.V_BOND_DEFAULT})")
    parser.add_argument("--v-list", type=float, default=bv.V_LIST_DEFAULT,
                        help=f"listing threshold in v.u. (default {bv.V_LIST_DEFAULT})")
    parser.add_argument("--format", default="txt", choices=("txt", "md", "tex"),
                        help="plain text, Markdown tables or LaTeX tables")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # the text carries Σ, φ and °; a Windows console in cp1252 would refuse
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
    args = build_parser().parse_args(argv)
    path = Path(args.file)
    if not path.exists():
        print(f"not found: {path}", file=sys.stderr)
        return 2
    if args.v_bond < args.v_list:
        print("v_bond is below v_list", file=sys.stderr)
        return 2
    structure = readers.read(str(path))
    results = coordination.analyse_structure(
        structure, bv.DEFAULT, v_bond=args.v_bond, v_list=args.v_list,
        cations_only=not args.anions)
    if not results:
        print("no site to analyse", file=sys.stderr)
        return 2
    if args.all:
        chosen = results
    elif args.site:
        chosen = [r for r in results if r.label == args.site]
        if not chosen:
            print(f"no site labelled {args.site!r}; the file has: "
                  + ", ".join(r.label for r in results), file=sys.stderr)
            return 2
    else:
        chosen = results[:1]
    pieces = [site_text(r, args.v_bond, fmt=args.format,
                        source=f"{path.name}, FACET") for r in chosen]
    sep = "\n\n" if args.format == "txt" else "\n\n---\n\n" if args.format == "md" else "\n\n"
    print(sep.join(pieces))
    return 0
