"""Entry point: ``py -3.11 -m facet.report FILE_OR_FOLDER... -o report.xlsx``."""
import sys

from facet.report.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
