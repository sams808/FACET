"""Entry point: ``py -3.11 -m facet.md {describe,analyses,template,analyse}``."""
import sys

from facet.md.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
