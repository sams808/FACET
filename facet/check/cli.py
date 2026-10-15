"""``py -3.11 -m facet.check``: the self-check and the parity harness.

    py -3.11 -m facet.check
    py -3.11 -m facet.check --failures --json out.json
    py -3.11 -m facet.check --parity theirs.csv run.extxyz --frame 0 \\
        --map cn=CN,sum=BVS --atom-column id --one-based --tol 1e-3

Without ``--parity``: every number FACET's documentation states for the
bundled examples is recomputed on this machine and compared with the
reference file within the stated tolerance, and each site's bond-valence
sum is checked against the sum of its counted contacts. Exit code 0 when
every check is within tolerance, 1 otherwise, 2 for a bad argument.

With ``--parity``: another code's per-atom table for one frame is laid
beside FACET's values for the same frame (``facet.core.md_parity``). The
exit code is 0 whenever the comparison ran; what the numbers mean is left
to the reader.

No Qt is imported.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from facet.core import bv
from facet.version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="py -3.11 -m facet.check",
        description="Recompute the numbers FACET states for its bundled "
                    "examples, or lay another code's per-atom table beside "
                    "FACET's for one MD frame.")
    parser.add_argument("--failures", action="store_true",
                        help="list only the checks outside tolerance")
    parser.add_argument("--json", metavar="PATH",
                        help="write every row as JSON as well")
    parser.add_argument("--reference", metavar="PATH",
                        help="another reference file (default: the bundled one)")
    parser.add_argument("--parity", nargs=2, metavar=("TABLE", "TRAJECTORY"),
                        help="another code's per-atom table and the trajectory "
                             "it describes")
    parser.add_argument("--frame", type=int, default=0,
                        help="the frame their table describes (default 0)")
    parser.add_argument("--map", default="CN,BVS", metavar="MAPPING",
                        help="their column = FACET quantity, comma separated "
                             "(default 'CN,BVS': columns named like the "
                             "quantities); quantities: CN, CN_listed, BVS, "
                             "BVV, phi, plateau")
    parser.add_argument("--atom-column", default="atom",
                        help="the column holding the atom index (default 'atom')")
    parser.add_argument("--one-based", action="store_true",
                        help="their atom indices start at 1")
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="the difference counted as agreement (default 1e-3)")
    parser.add_argument("--v-bond", type=float, default=bv.V_BOND_DEFAULT)
    parser.add_argument("--v-list", type=float, default=bv.V_LIST_DEFAULT)
    return parser


def _write_json(path: str, rows) -> None:
    Path(path).write_text(json.dumps([r.as_dict() for r in rows], indent=1,
                                     default=str), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    if args.parity:
        from facet.core import md_parity
        table, trajectory = args.parity
        for p in (table, trajectory):
            if not Path(p).exists():
                print(f"not found: {p}", file=sys.stderr)
                return 2
        try:
            rows = md_parity.compare_files(
                table, trajectory, args.frame, args.map,
                atom_column=args.atom_column, one_based=args.one_based,
                tolerance=args.tol, v_bond=args.v_bond, v_list=args.v_list)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2
        print(f"FACET {__version__} parity: {Path(table).name} against frame "
              f"{args.frame} of {Path(trajectory).name} at v_bond {args.v_bond:g}, "
              f"v_list {args.v_list:g} v.u.; 'mean diff' is theirs minus FACET")
        print(md_parity.format_rows(rows))
        if args.json:
            _write_json(args.json, rows)
        return 0

    from facet.core import selfcheck
    reference = selfcheck.load_reference(args.reference) if args.reference \
        else selfcheck.load_reference()
    rows = selfcheck.check_examples(reference) + selfcheck.check_sums(reference)
    print(f"FACET {__version__} self-check: the bundled examples at v_bond "
          f"{reference.get('v_bond_vu')} / v_list {reference.get('v_list_vu')} "
          f"v.u., {reference.get('parameter_set')}")
    print(selfcheck.format_rows(rows, only_failures=args.failures))
    print(selfcheck.summary(rows))
    if args.json:
        _write_json(args.json, rows)
    return 0 if all(r.agrees for r in rows) else 1
