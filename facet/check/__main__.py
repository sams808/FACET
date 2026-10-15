"""Entry point: ``py -3.11 -m facet.check [--parity TABLE TRAJECTORY]``."""
import sys

from facet.check.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
