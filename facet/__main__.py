"""Entry point: ``py -3.11 -m facet [structure.cif]``."""
import sys


def main() -> int:
    from facet.ui.app import main as run

    return run(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
