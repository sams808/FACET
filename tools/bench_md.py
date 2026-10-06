"""Time FACET's crystal path on boxes the size of an MD model.

    py -3.11 tools/bench_md.py                       # 1000, 3375, 9261 atoms
    py -3.11 tools/bench_md.py --sides 27 --repeats 1
    py -3.11 tools/bench_md.py --profile             # cProfile, largest size
    py -3.11 tools/bench_md.py --json results.json

The analysis was built for crystals, where a handful of crystallographic sites
stand for every atom in the cell. A glass model from molecular dynamics has no
symmetry, so every atom is its own site, and work that is per site -- cheap for a
crystal -- becomes per atom. Before an MD path is designed this measures where
the time goes on the machine at hand, so the design answers a measured cost
rather than an assumed one.

**The boxes are synthetic and only the timing means anything; the chemistry is
nonsense.** Atoms sit on a cubic grid with 2.3 Å spacing, each displaced by a
uniform random jitter, with the element of each drawn from Si, O, O, Na at
p = 0.25, 0.30, 0.30, 0.15 (so O at 0.60) from a fixed seed. The spacing and the
probabilities are those of the benchmark this reproduces. **The jitter amplitude
is a choice made here** -- that benchmark does not state one -- and defaults to
+/-0.1 Å per Cartesian component (``--jitter-ang``); it is there so the grid is
not exactly symmetric and the symmetry search sees what it would see in a real
model. Grid sides 10, 15 and 21 give 1000, 3375 and 9261 atoms.

Each box is written as extended XYZ, with ``Lattice=`` and
``Properties=species:S:1:pos:R:3`` in the comment line, into a temporary
directory that is deleted afterwards -- never into the repository. It is then
read with :func:`facet.core.readers.read_xyz` and analysed with
:func:`facet.core.coordination.analyse_structure` at its default arguments. That
is what opening such a file in the application does: the read runs the spglib
symmetry search, and the analysis resolves oxidation states (a neighbour search
of its own) before the per-site loop.

Sizes up to ``--once-above`` atoms are repeated ``--repeats`` times and the
median reported; larger sizes run once, because one run there takes tens of
seconds and the run-to-run noise is a small part of it. Every repeat reads the
file afresh, because the analysis writes the resolved oxidation states into the
structure it is given. One untimed pass on a 27-atom box goes first, so the
one-off cost of the first file opened in a process (imports, the bond-valence
table) is not charged to the smallest size; ``--no-warmup`` keeps it in.

``--profile`` replaces the timing table with a cProfile of the read and of the
analysis of the largest size, and prints the cumulative time in the two
functions the benchmark singled out: ``cif._annotate_symmetry`` (spglib, reached
from ``readers._annotate``), and ``Structure.atoms_of_site``, a scan over every
atom made once per site and so O(N^2) when every atom is a site. The profiler
adds a cost to every Python call, so profiled seconds run higher than the
unprofiled table; compare the shares, not the absolute times.

**Pin it on a hybrid CPU.** Left to itself, Windows ran a fixed pure-Python
loop on the performance cores for its first few seconds and then moved it to the
efficiency cores, where the median repeat took 0.50 s against 0.27 s. Unpinned,
the 1000-atom analysis came out at 1.9, 4.3 and 5.7 s in three consecutive
repeats. ``start /affinity FF /wait /b py -3.11 tools/bench_md.py`` (from cmd)
keeps it on logical CPUs 0-7, which on the machine below are the P-cores
(efficiency class 1 in ``GetSystemCpuSetInformation``); check the mask on any
other machine.

Measured 2026-10-06 on Windows 11, Intel i5-13420H (4 P-cores x 2 threads, 4
E-cores), Python 3.11.9, numpy 2.4.6, scipy 1.15.1, spglib 2.6.0, pinned as
above, with other sessions keeping the machine 15-55 % busy::

    n_atoms   read_xyz (s)   analyse_structure (s)   runs
      1 000       0.13              2.27               3
      3 375       1.12              7.28               3
      9 261       7.12             26.21               1
     19 683      47.32             71.09               1

Log-log slopes over the four sizes: read 1.95, analysis 1.16; between the two
largest, 2.51 and 1.32. With ``--profile`` at 9261 atoms,
``cif._annotate_symmetry`` was 6.85 s of a 7.18 s read (95 %) and
``Structure.atoms_of_site`` 12.2 s of a 41.9 s analysis (29 %, 7550 calls: once
per cation site in ``oxidation.resolve`` and once more in the analysis loop).
That second share is the profiler's: without it the same 7550 calls took 3.7 s
timed on their own, and 5.5-8.7 s in place, measured as the drop in the analysis
when a scratch process swapped them for an O(1) lookup.
"""
from __future__ import annotations

import argparse
import cProfile
import datetime as _dt
import gc
import json
import os
import platform
import pstats
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

SPACING_ANG = 2.3
SPECIES = ("Si", "O", "O", "Na")
PROBABILITIES = (0.25, 0.30, 0.30, 0.15)
DEFAULT_SIDES = (10, 15, 21)
DEFAULT_JITTER_ANG = 0.1
DEFAULT_SEED = 2026
WARMUP_SIDE = 3

# (label, file the function is defined in, function name)
WATCHED = (
    ("cif._annotate_symmetry", ("facet", "core", "cif.py"), "_annotate_symmetry"),
    ("spglib.get_symmetry_dataset", ("spglib", "spglib.py"), "get_symmetry_dataset"),
    ("Structure.atoms_of_site", ("facet", "core", "structure.py"), "atoms_of_site"),
    ("oxidation.resolve", ("facet", "core", "oxidation.py"), "resolve"),
    ("NeighborFinder.__init__", ("facet", "core", "neighbors.py"), "__init__"),
    ("NeighborFinder.contacts", ("facet", "core", "neighbors.py"), "contacts"),
    ("coordination.analyse_site", ("facet", "core", "coordination.py"),
     "analyse_site"),
)


def _facet():
    """Import the two FACET modules timed here, from this checkout."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from facet.core import coordination, readers

    return readers, coordination


# ---------------------------------------------------------------------------
# the boxes
# ---------------------------------------------------------------------------

def make_box(side: int, spacing_ang: float = SPACING_ANG,
             jitter_ang: float = DEFAULT_JITTER_ANG,
             seed: int = DEFAULT_SEED) -> tuple[np.ndarray, np.ndarray, float]:
    """A jittered cubic grid of ``side**3`` atoms in a cubic periodic box.

    Returns ``(species, cart_ang, box_ang)``. Each size draws from its own
    generator seeded with ``seed``, so a box does not depend on which other
    sizes were run before it.
    """
    rng = np.random.default_rng(seed)
    n_atoms = side ** 3
    grid = np.indices((side, side, side)).reshape(3, -1).T.astype(float)
    box_ang = side * spacing_ang
    jitter = rng.uniform(-jitter_ang, jitter_ang, size=(n_atoms, 3))
    cart_ang = ((grid + 0.5) * spacing_ang + jitter) % box_ang
    species = rng.choice(np.array(SPECIES), size=n_atoms, p=PROBABILITIES)
    return species, cart_ang, box_ang


def write_extxyz(path: Path, species: np.ndarray, cart_ang: np.ndarray,
                 box_ang: float) -> None:
    """Extended XYZ: one frame, an orthogonal cell given as three rows."""
    lattice = f"{box_ang:.6f} 0 0 0 {box_ang:.6f} 0 0 0 {box_ang:.6f}"
    lines = [str(len(species)),
             f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="T T T"']
    lines += [f"{s} {x:.6f} {y:.6f} {z:.6f}"
              for s, (x, y, z) in zip(species, cart_ang)]
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------

def time_once(path: Path) -> tuple[float, float, object, list]:
    """Read then analyse one file; return both wall times and the results."""
    readers, coordination = _facet()
    gc.collect()
    t0 = time.perf_counter()
    structure = readers.read_xyz(path)
    t1 = time.perf_counter()
    results = coordination.analyse_structure(structure)
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1, structure, results


def describe(structure, results) -> dict:
    """What the read and the analysis made of a box, for the record.

    Computed after the clock has stopped. The oxidation-state sources say what
    ``oxidation.resolve`` did with chemistry no real oxide has.
    """
    cation_sites = structure.cation_sites
    sources = Counter(structure.sites[i].ox_source for i in cation_sites)
    states = Counter(f"{structure.sites[i].element}{structure.sites[i].ox:+d}"
                     if structure.sites[i].ox is not None
                     else f"{structure.sites[i].element}?"
                     for i in cation_sites)
    composition = Counter(a.element for a in structure.atoms)
    return {
        "n_atoms": structure.n_atoms,
        "n_sites": structure.n_sites,
        "n_cation_sites": len(cation_sites),
        "n_site_results": len(results),
        "composition": dict(composition),
        "spacegroup_number": structure.spacegroup_number,
        "spacegroup_hm": structure.spacegroup_hm,
        "structure_name_chars": len(structure.name or ""),
        "ox_sources": dict(sources),
        "ox_states": dict(states),
        "net_charge": structure.net_charge(),
        "n_notes": len(structure.notes),
        "notes_head": [n[:240] for n in structure.notes[:6]],
    }


def warm_up(workdir: Path, jitter_ang: float, seed: int) -> dict:
    """One untimed pass on a tiny box before anything is measured.

    The first read and the first analysis in a process pay one-off costs --
    importing spglib and scipy.spatial, loading the bond-valence table -- that
    belong to opening the application, not to the size of the model. Without
    this they land on the first repeat of the smallest size.
    """
    species, cart_ang, box_ang = make_box(WARMUP_SIDE, jitter_ang=jitter_ang,
                                          seed=seed)
    path = workdir / "warmup.xyz"
    write_extxyz(path, species, cart_ang, box_ang)
    t_read, t_analyse, _structure, _results = time_once(path)
    return {"n_atoms": len(species), "read_xyz_s": t_read,
            "analyse_structure_s": t_analyse}


def run_timing(sides, repeats: int, once_above: int, jitter_ang: float,
               seed: int, workdir: Path) -> list[dict]:
    rows = []
    for side in sides:
        species, cart_ang, box_ang = make_box(side, jitter_ang=jitter_ang,
                                              seed=seed)
        path = workdir / f"box_{side}.xyz"
        write_extxyz(path, species, cart_ang, box_ang)
        n_atoms = len(species)
        n_runs = repeats if n_atoms <= once_above else 1
        read_s, analyse_s = [], []
        info = None
        for _ in range(n_runs):
            t_read, t_analyse, structure, results = time_once(path)
            read_s.append(t_read)
            analyse_s.append(t_analyse)
            if info is None:
                info = describe(structure, results)
            del structure, results
        rows.append({
            "side": side, "n_atoms": n_atoms, "box_ang": box_ang,
            "file_bytes": path.stat().st_size,
            "read_xyz_s": read_s, "analyse_structure_s": analyse_s,
            "read_xyz_median_s": statistics.median(read_s),
            "analyse_structure_median_s": statistics.median(analyse_s),
            "box": info,
        })
        print(f"  {n_atoms:>6} atoms: read "
              + ", ".join(f"{t:.2f}" for t in read_s) + " s; analyse "
              + ", ".join(f"{t:.2f}" for t in analyse_s) + " s", flush=True)
    return rows


def exponents(rows: list[dict], key: str) -> dict:
    """Log-log slopes of median time against atom count."""
    n = np.array([r["n_atoms"] for r in rows], float)
    t = np.array([r[key] for r in rows], float)
    out = {"pairwise": [
        {"from": int(n[i]), "to": int(n[i + 1]),
         "exponent": float(np.log(t[i + 1] / t[i]) / np.log(n[i + 1] / n[i]))}
        for i in range(len(n) - 1)]}
    if len(n) >= 2:
        out["fit"] = float(np.polyfit(np.log(n), np.log(t), 1)[0])
    return out


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------

def _matches(filename: str, parts: tuple[str, ...]) -> bool:
    tail = os.path.normcase(os.path.join(*parts))
    return os.path.normcase(filename).endswith(tail)


def _watched(stats: pstats.Stats) -> dict:
    out = {}
    for label, parts, name in WATCHED:
        calls, cumulative = 0, 0.0
        for (filename, _line, func), (_cc, nc, _tt, ct, _callers) in \
                stats.stats.items():
            if func == name and _matches(filename, parts):
                calls += nc
                cumulative += ct
        out[label] = {"calls": calls, "cumulative_s": cumulative}
    return out


def _top(stats: pstats.Stats, n: int, column: int) -> list[dict]:
    """The n entries with the largest tottime (column 2) or cumtime (3)."""
    ranked = sorted(stats.stats.items(), key=lambda kv: kv[1][column],
                    reverse=True)[:n]
    out = []
    for (filename, line, func), (_cc, nc, tt, ct, _callers) in ranked:
        where = (f"{Path(filename).name}:{line}" if filename not in ("~", "")
                 else "")
        out.append({"function": f"{where}({func})" if where else func,
                    "calls": nc, "tottime_s": tt, "cumtime_s": ct})
    return out


def run_profile(side: int, jitter_ang: float, seed: int, workdir: Path,
                top: int) -> dict:
    readers, coordination = _facet()
    species, cart_ang, box_ang = make_box(side, jitter_ang=jitter_ang, seed=seed)
    path = workdir / f"box_{side}.xyz"
    write_extxyz(path, species, cart_ang, box_ang)

    gc.collect()
    read_prof = cProfile.Profile()
    read_prof.enable()
    structure = readers.read_xyz(path)
    read_prof.disable()

    analyse_prof = cProfile.Profile()
    analyse_prof.enable()
    results = coordination.analyse_structure(structure)
    analyse_prof.disable()

    read_stats, analyse_stats = pstats.Stats(read_prof), pstats.Stats(analyse_prof)
    return {
        "side": side, "n_atoms": len(species),
        "read_total_s": read_stats.total_tt,
        "analyse_total_s": analyse_stats.total_tt,
        "read_watched": _watched(read_stats),
        "analyse_watched": _watched(analyse_stats),
        "read_top_cumulative": _top(read_stats, top, 3),
        "analyse_top_cumulative": _top(analyse_stats, top, 3),
        "analyse_top_tottime": _top(analyse_stats, top, 2),
        "box": describe(structure, results),
    }


# ---------------------------------------------------------------------------
# environment and output
# ---------------------------------------------------------------------------

def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _git_head() -> str | None:
    try:
        done = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None


def environment() -> dict:
    import scipy

    try:
        import spglib

        spglib_version = getattr(spglib, "__version__", "unknown")
    except ImportError:
        spglib_version = None
    _facet()
    from facet.version import __version__ as facet_version

    return {
        "when_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": _cpu_name(),
        "logical_cpus": os.cpu_count(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "spglib": spglib_version,
        "facet": facet_version,
        "git_head": _git_head(),
    }


def _thousands(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def print_environment(env: dict) -> None:
    print(f"FACET {env['facet']} ({env['git_head']}), Python {env['python']}, "
          f"numpy {env['numpy']}, scipy {env['scipy']}, spglib {env['spglib']}")
    print(f"{env['cpu']}, {env['logical_cpus']} logical CPUs, {env['platform']}, "
          f"{env['when_utc']}")


def print_table(rows: list[dict]) -> None:
    print()
    print("| n_atoms | `readers.read_xyz` (s) | "
          "`coordination.analyse_structure` (s) | runs |")
    print("|---|---|---|---|")
    for r in rows:
        print(f"| {_thousands(r['n_atoms'])} | {r['read_xyz_median_s']:.2f} | "
              f"{r['analyse_structure_median_s']:.2f} | {len(r['read_xyz_s'])} |")
    if len(rows) >= 2:
        for key, label in (("read_xyz_median_s", "read_xyz"),
                           ("analyse_structure_median_s", "analyse_structure")):
            e = exponents(rows, key)
            pairs = ", ".join(f"{p['from']}->{p['to']}: {p['exponent']:.2f}"
                              for p in e["pairwise"])
            print(f"log-log exponent, {label}: fit {e['fit']:.2f} ({pairs})")
    print()
    for r in rows:
        b = r["box"]
        print(f"{_thousands(r['n_atoms'])} atoms: {b['composition']}; "
              f"{b['n_site_results']} cation sites analysed; space group "
              f"No. {b['spacegroup_number']}; oxidation sources {b['ox_sources']}; "
              f"states {b['ox_states']}; net charge {b['net_charge']}; "
              f"{b['n_notes']} notes")


def print_profile(p: dict) -> None:
    print()
    print(f"cProfile, {_thousands(p['n_atoms'])} atoms: read "
          f"{p['read_total_s']:.2f} s, analysis {p['analyse_total_s']:.2f} s "
          "(profiled)")
    print("| function | calls | cumulative (s) | share |")
    print("|---|---|---|---|")
    for phase, total in (("read", p["read_total_s"]),
                         ("analyse", p["analyse_total_s"])):
        for label, w in p[f"{phase}_watched"].items():
            if w["calls"]:
                print(f"| {label} ({phase}) | {w['calls']} | "
                      f"{w['cumulative_s']:.2f} | "
                      f"{100 * w['cumulative_s'] / total:.0f} % of {phase} |")
    for key, title in (("read_top_cumulative", "read, by cumulative time"),
                       ("analyse_top_cumulative", "analysis, by cumulative time"),
                       ("analyse_top_tottime", "analysis, by own time")):
        print(f"\ntop of the {title}:")
        for t in p[key]:
            print(f"  {t['cumtime_s']:8.2f} cum {t['tottime_s']:8.2f} own "
                  f"{t['calls']:>10}  {t['function']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Time FACET's read and analysis on MD-sized synthetic boxes.")
    parser.add_argument("--sides", type=int, nargs="+", default=list(DEFAULT_SIDES),
                        help="grid sides; n_atoms = side**3 (default 10 15 21)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="runs per size up to --once-above atoms (default 3)")
    parser.add_argument("--once-above", type=int, default=5000,
                        help="sizes with more atoms run once (default 5000)")
    parser.add_argument("--jitter-ang", type=float, default=DEFAULT_JITTER_ANG,
                        help="uniform jitter amplitude per component, Å "
                             "(default 0.1, a choice; see the docstring)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--profile", action="store_true",
                        help="cProfile the largest size instead of timing")
    parser.add_argument("--top", type=int, default=15,
                        help="functions listed per profile table (default 15)")
    parser.add_argument("--no-warmup", action="store_true",
                        help="time the first size cold, imports included")
    parser.add_argument("--json", type=Path, default=None,
                        help="also write every number to this JSON file")
    args = parser.parse_args(argv)
    if args.repeats < 1 or any(s < 1 for s in args.sides):
        parser.error("--repeats and every side must be at least 1")

    env = environment()
    print_environment(env)
    record: dict = {"environment": env, "spacing_ang": SPACING_ANG,
                    "jitter_ang": args.jitter_ang, "seed": args.seed,
                    "species": SPECIES, "probabilities": PROBABILITIES}

    with tempfile.TemporaryDirectory(prefix="facet_bench_md_") as tmp:
        workdir = Path(tmp)
        if not args.no_warmup:
            record["warmup"] = warm_up(workdir, args.jitter_ang, args.seed)
            print(f"warm-up ({WARMUP_SIDE ** 3} atoms, not in the table): read "
                  f"{record['warmup']['read_xyz_s']:.2f} s, analyse "
                  f"{record['warmup']['analyse_structure_s']:.2f} s")
        if args.profile:
            record["profile"] = run_profile(max(args.sides), args.jitter_ang,
                                            args.seed, workdir, args.top)
            print_profile(record["profile"])
        else:
            rows = run_timing(args.sides, args.repeats, args.once_above,
                              args.jitter_ang, args.seed, workdir)
            record["timing"] = rows
            print_table(rows)

    if args.json is not None:
        args.json.write_text(json.dumps(record, indent=1, default=str),
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
