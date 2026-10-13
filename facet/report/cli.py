"""``py -3.11 -m facet.report``: reports on CIFs from the command line.

    py -3.11 -m facet.report FILE_OR_FOLDER... -o report.xlsx
    py -3.11 -m facet.report structures/ -o out/report --format csv
    py -3.11 -m facet.report a.cif b.cif --format html --v-bond 0.10

Reads every file (and every .cif in a folder given), analyses each site at
the thresholds asked for, and writes the sites, contacts, plateaus, cutoffs
and charge-balance tables in the format chosen (``facet.core.report``). No
Qt is imported. Exit code 0 when every file was read, 1 when some were not
(the report still lists them with the reason), 2 for a bad argument.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from facet.core import bv
from facet.core.report import FORMATS, FORMAT_NAMES, analyse_files, write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="py -3.11 -m facet.report",
        description="Bond valences, bond-valence sums, distances and cutoffs "
                    "for every site of every CIF, as a report.")
    parser.add_argument("paths", nargs="+", metavar="FILE_OR_FOLDER",
                        help="CIF files, or folders of them")
    parser.add_argument("-o", "--out", default="facet_report",
                        help="where to write (extension follows the format; "
                             "for csv, the stem of one file per table)")
    parser.add_argument("--format", default="xlsx", choices=FORMATS,
                        help="; ".join(f"{k}: {v}" for k, v in FORMAT_NAMES.items()))
    parser.add_argument("--v-bond", type=float, default=bv.V_BOND_DEFAULT,
                        help="bond threshold in valence units "
                             f"(default {bv.V_BOND_DEFAULT})")
    parser.add_argument("--v-list", type=float, default=bv.V_LIST_DEFAULT,
                        help="listing threshold in valence units "
                             f"(default {bv.V_LIST_DEFAULT})")
    parser.add_argument("--anions", action="store_true",
                        help="analyse the anion sites as well as the cations")
    parser.add_argument("--quiet", action="store_true",
                        help="no progress on stderr")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.v_bond < args.v_list:
        print(f"v_bond ({args.v_bond}) is below v_list ({args.v_list}): a bond "
              "must be at least a listed contact", file=sys.stderr)
        return 2
    missing = [p for p in args.paths if not Path(p).exists()]
    if missing:
        print("not found: " + ", ".join(missing), file=sys.stderr)
        return 2

    def progress(done: int, total: int, name: str) -> bool:
        if not args.quiet:
            print(f"[{done + 1}/{total}] {name}", file=sys.stderr, flush=True)
        return True

    report = analyse_files(args.paths, v_bond=args.v_bond, v_list=args.v_list,
                           anions=args.anions, progress=progress)
    if not report.structures:
        print("no CIF files among the paths given", file=sys.stderr)
        return 2
    written = write_report(report, args.out, args.format)
    for path in written:
        print(path)
    failed = [s for s in report.structures if s.error]
    for s in failed:
        print(f"not read: {s.path}: {s.error}", file=sys.stderr)
    return 1 if failed else 0
