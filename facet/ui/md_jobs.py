"""The Model window's work off the GUI thread: opening an MD file, running
the analyses, loading a frame for the 3D view, writing the results.

THE THREADING RULE THIS MODULE KEEPS
------------------------------------
A PySide6 signal connected to a plain function or a lambda runs that
function on the thread that *emits* the signal. A worker that reports
progress through such a connection therefore runs the receiving code on the
worker thread, and a widget touched or built there hangs white or takes the
process down when the thread ends (the memory note on worker callbacks,
measured in an earlier FACET session). So:

* the work itself is a plain callable ``work(progress, cancelled)`` holding
  no widget; it runs in :class:`_Worker.run`, which lives on a ``QThread``;
* the worker's signals go only to ``@Slot`` methods of :class:`Job`, a
  ``QObject`` living on the GUI thread, with ``Qt.QueuedConnection`` stated,
  never to a lambda; the job re-emits them on the GUI thread, where the
  window's own slots receive them;
* cancelling sets a ``threading.Event`` that the engine's ``cancelled()``
  hook reads; nothing on the worker waits for the GUI thread, so a GUI
  thread that waits for the worker (a window closing) cannot deadlock;
* a ``QThread`` destroyed while it runs aborts the process, so a job whose
  work does not stop within the time a closing window waits for it is kept
  alive here (:data:`_KEEPER`) until its thread ends, and the application
  waits for those threads before it quits.

The Qt-free half of the module (:class:`ModelSummary`, :class:`ReadProblem`,
:func:`open_model`, :func:`diagnose`) is what the open job runs: it reads the
file, and when the reader refuses it, sorts the refusal into the read
options the reader asks for, so the window can ask for exactly those.
"""
from __future__ import annotations

import atexit
import re
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from PySide6.QtCore import QCoreApplication, QObject, QThread, Qt, Signal, Slot

__all__ = [
    "ModelSummary", "Opened", "ReadProblem", "JobFailure", "FrameView",
    "READ_OPTION_TEXT", "source_label", "summarise", "read_options_taken",
    "diagnose", "open_model", "load_frame_view",
    "Job", "AnalysisJob", "running_jobs",
]

# How many bytes a file may hold for the window to read it a second time,
# with stand-in elements, to count the atoms of each type in frame 0 when the
# reader asks for a type map (diagnose). A choice: re-reading a file indexes
# it again, which costs as long as the first read; above this size the
# counts are left out and the table says why.
COUNT_PROBE_MAX_BYTES = 200 * 2 ** 20

# Elements that stand in for the types of a file while their atoms are
# counted (never shown, never kept): the superheavy elements first, which no
# glass model holds, then the actinides beyond the lanthanide-free end.
_STAND_INS = ("Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh",
              "Fl", "Mc", "Lv", "Ts", "Og", "Lr", "No", "Md", "Fm", "Es",
              "Cf", "Bk", "Cm", "Am")

# What each read option of md_readers.read_trajectory supplies, in the words
# the window shows beside it.
READ_OPTION_TEXT = {
    "type_map": "the element of each atom type or label in the file",
    "masses_from": "a LAMMPS data file whose Masses section names the "
                   "elements of the types",
    "topology": "a file FACET reads that holds the same atoms in the same "
                "order (the LAMMPS data file the run read, a dump, an "
                "extended XYZ)",
    "box_from": "another MD file of the same run whose first-frame box is "
                "used (a LAMMPS data file, a CP2K PROJECT-1.cell file), or "
                "the three box vectors in Å",
    "units": "the LAMMPS unit style or length unit the file is written in",
    "columns": "the per-atom column names, in the file's order",
    "atom_style": "the LAMMPS atom style of the data file",
    "mass_tol_amu": "how far a type's mass may sit from an element's "
                    "standard atomic weight (amu)",
    "timestep_fs": "the time between steps in fs, for a file that gives "
                   "steps and no times",
}


# ---------------------------------------------------------------------------
# what the window holds of a model (Qt-free)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class ModelSummary:
    """What the setup panel shows and checks, read once on the open job.

    ``describe_lines`` is ``Trajectory.describe()`` (it loads frame 0);
    ``composition`` and ``species`` are frame 0's; ``min_width_ang`` is the
    smallest perpendicular width of frame 0's box.
    """

    source: object                  # what was opened: a path or a list
    label: str
    source_path: str
    file_format: str
    n_atoms: int
    n_frames: int
    timesteps: np.ndarray
    times_ps: np.ndarray | None
    composition: dict
    species: tuple
    has_unwrapped: bool
    has_velocities: bool
    file_charges: dict | None
    box_varies: bool | None
    min_width_ang: float
    type_map: dict
    type_map_source: str
    describe_lines: tuple
    skipped: dict
    read_options: dict
    # the reader's statement of the units it read the file in (a LAMMPS
    # file without a unit style is read as metal, and says so)
    units_note: str = ""

    @property
    def has_times(self) -> bool:
        return self.times_ps is not None and bool(
            np.isfinite(self.times_ps).any())


def source_label(source) -> str:
    """A file name, or the first name and a count for a list."""
    if isinstance(source, (list, tuple)):
        names = [Path(str(p)).name for p in source]
        if not names:
            return "(no file)"
        if len(names) == 1:
            return names[0]
        return f"{names[0]} and {len(names) - 1} more file(s)"
    return Path(str(source)).name


def summarise(trajectory, source, read_options: Mapping) -> ModelSummary:
    """A :class:`ModelSummary` of an opened trajectory (loads frame 0)."""
    lines = tuple(trajectory.describe())
    first = trajectory.frame(0)
    return ModelSummary(
        source=source, label=source_label(source),
        source_path=str(trajectory.source_path),
        file_format=str(trajectory.file_format),
        n_atoms=int(trajectory.n_atoms), n_frames=int(trajectory.n_frames),
        timesteps=np.asarray(trajectory.timesteps).copy(),
        times_ps=(None if trajectory.times_ps is None
                  else np.asarray(trajectory.times_ps).copy()),
        composition=dict(first.composition), species=tuple(first.species),
        has_unwrapped=first.unwrapped_cart_ang is not None,
        has_velocities=first.vel_ang_per_ps is not None,
        file_charges=(None if trajectory.charges_e is None
                      else dict(trajectory.charges_e)),
        box_varies=trajectory.box_varies,
        min_width_ang=float(np.min(first.perpendicular_widths_ang)),
        type_map=dict(trajectory.type_map),
        type_map_source=str(trajectory.type_map_source),
        describe_lines=lines, skipped=dict(trajectory.skipped),
        read_options=dict(read_options),
        units_note=str(getattr(trajectory, "units_note", "") or ""))


@dataclass(frozen=True, eq=False)
class Opened:
    """A model read: the trajectory and its summary."""

    trajectory: object
    summary: ModelSummary


@dataclass(frozen=True, eq=False)
class ReadProblem:
    """The reader's refusal, sorted into what the window can ask for.

    ``message`` is the reader's own text, shown in full. ``needs`` are the
    read options the message asks for among those the format takes
    (``options_taken``); ``type_keys`` the types or labels it names as
    lacking an element, ``keys_complete`` False when the reader shortened
    the list. ``evidence`` holds, per key, the reader's own words about it
    (its label, element column, mass); ``counts`` the atoms of each key in
    frame 0, or None per key when they could not be counted, with
    ``counts_note`` saying why. ``suggested`` holds option values the
    message names (units='nano'); they are shown, never filled in.
    """

    source: object
    read_options: dict
    message: str
    error_type: str
    file_format: str | None
    options_taken: frozenset
    needs: tuple
    type_keys: tuple = ()
    keys_complete: bool = False
    evidence: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    counts_note: str = ""
    suggested: dict = field(default_factory=dict)

    @property
    def can_supply(self) -> bool:
        """True when some read option would answer the refusal."""
        return bool(self.needs)


def read_options_taken(file_format: str | None) -> frozenset:
    """The read options ``md_readers.read_trajectory`` passes to a format's
    reader (an empty set for an unknown format)."""
    from ..core import md_readers

    if not file_format:
        return frozenset()
    builtin = getattr(md_readers, "_OPTIONS", {})
    if file_format in builtin:
        return frozenset(builtin[file_format])
    for spec in md_readers.format_specs():
        if spec.name == file_format:
            return frozenset(spec.options)
    return frozenset()


def _first_path(source) -> Path | None:
    if isinstance(source, (list, tuple)):
        return Path(str(source[0])) if source else None
    text = str(source)
    if any(c in text for c in "*?["):
        return None
    path = Path(text)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file())
        return files[0] if files else None
    return path


def _key(token: str):
    token = token.strip()
    if token.startswith("'") and token.endswith("'"):
        return token[1:-1]
    return int(token)


def _type_keys(message: str) -> tuple[tuple, bool]:
    """The keys of the reader's first example map (its own list of what
    lacks an element; a second map in the same message is the same types
    keyed another way), and complete False when it was cut ('...')."""
    maps = re.findall(r"type_map=\{([^{}]*)\}", message)
    keys: list = []
    complete = False
    if maps:
        inner = maps[0]
        complete = "..." not in inner
        for token in re.findall(r"(-?\d+|'[^']*')\s*:\s*'<element>'", inner):
            key = _key(token)
            if key not in keys:
                keys.append(key)
    head = message.split("type_map=")[0]
    if not keys or all(isinstance(k, int) for k in keys):
        listed = [int(t) for t in re.findall(r"\btype (\d+) \(", head)]
        if not keys and listed:
            complete = True
        keys.extend(k for k in listed if k not in keys)
    ints = sorted(k for k in keys if isinstance(k, int))
    labels = sorted(k for k in keys if isinstance(k, str))
    return tuple(ints) + tuple(labels), complete and bool(keys)


def _clause_around(text: str, at: int) -> str:
    """The clause of ``text`` holding position ``at``: from the '; ' or '. '
    before it to the ';' or '. ' after it, outside parentheses."""
    start = max(text.rfind("; ", 0, at), text.rfind(". ", 0, at))
    start = 0 if start < 0 else start + 2
    depth, end = 0, len(text)
    for i in range(at, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                end = i
                break
            depth -= 1
        elif depth == 0 and (ch == ";" or (ch == "." and text[i + 1:i + 2]
                                           in (" ", ""))):
            end = i
            break
    return text[start:end].strip(" ,")


def _evidence(message: str, keys: Sequence) -> dict:
    """Per key, the reader's own words about it: the parenthesis after
    'type N', or the clause that names the label."""
    head = message.split("type_map=")[0]
    out = {}
    for key in keys:
        text = ""
        if isinstance(key, int):
            match = re.search(rf"\btype {key} \(", head)
            if match:
                depth, start = 0, match.end() - 1
                for i in range(start, len(head)):
                    if head[i] == "(":
                        depth += 1
                    elif head[i] == ")":
                        depth -= 1
                        if depth == 0:
                            text = head[start + 1:i]
                            break
        if not text:
            at = head.find(f"'{key}'")
            if at >= 0:
                tail = _clause_around(head[at:], 0)
                if re.match(r"'[^']*'\s*(\(|:|at\b)", tail):
                    # the label with its own words: its mass, its reading
                    text = tail
                else:
                    # a list of labels: the clause that introduces it
                    text = _clause_around(head, at)
                    text = re.sub(r"^[^\s:]+: ", "", text)
        out[key] = text
    return out


def _asks_for(message: str, option: str) -> bool:
    return re.search(rf"\b{re.escape(option)}\s*=", message) is not None


def _suggested(message: str) -> dict:
    out = {}
    for option in ("units", "atom_style", "columns"):
        match = re.search(rf"\b{option}\s*=\s*'([^'<>]+)'", message)
        if match:
            out[option] = match.group(1)
    return out


def _count_types(source, read_options: Mapping, keys: Sequence
                 ) -> tuple[dict, str]:
    """Atoms per key in frame 0, by reading the file once more with each key
    mapped to a stand-in element. Returns (counts, why not counted)."""
    from ..core import md_readers

    empty = {k: None for k in keys}
    if not keys:
        return {}, ""
    if len(keys) > len(_STAND_INS):
        return empty, (f"{len(keys)} types: more than the {len(_STAND_INS)} "
                       "that can be counted")
    first = _first_path(source)
    try:
        size = sum(Path(str(p)).stat().st_size for p in source) \
            if isinstance(source, (list, tuple)) else \
            (first.stat().st_size if first is not None else 0)
    except OSError:
        size = 0
    if size > COUNT_PROBE_MAX_BYTES:
        return empty, (f"not counted: the file holds {size / 2**20:.0f} MB, "
                       "and counting reads it a second time")
    stand_in = {k: s for k, s in zip(keys, _STAND_INS)}
    options = dict(read_options)
    given = dict(options.get("type_map") or {})
    given.update(stand_in)
    options["type_map"] = given
    trajectory = None
    try:
        trajectory = md_readers.read_trajectory(source, **options)
        composition = trajectory.frame(0).composition
    except Exception as error:      # noqa: BLE001 - the counts are an aid
        return empty, f"not counted: {error}"
    finally:
        close = getattr(trajectory, "close", None)
        if close is not None:
            close()
    return {k: int(composition.get(s, 0)) for k, s in stand_in.items()}, ""


def diagnose(source, read_options: Mapping, error: BaseException
             ) -> ReadProblem:
    """Sort a refusal of ``md_readers.read_trajectory`` into the read
    options its message asks for (and the format takes)."""
    from ..core import md_readers

    message = str(error)
    first = _first_path(source)
    file_format = None
    if first is not None:
        try:
            file_format = md_readers.sniff_md(first)
        except Exception:           # noqa: BLE001 - sniff never raises
            file_format = None
    taken = read_options_taken(file_format)
    needs = [o for o in ("type_map", "masses_from", "topology", "box_from",
                         "units", "columns", "atom_style")
             if o in taken and _asks_for(message, o)]
    if "type_map" in taken and "type_map" not in needs and \
            re.search(r"\btype map\b", message):
        needs.insert(0, "type_map")
    keys, complete = _type_keys(message) if "type_map" in needs else ((),
                                                                     False)
    if "topology" in needs and not complete:
        # the map then applies to the topology's types, asked for once the
        # topology is read; a map by atom number is no use to type in
        keys, complete = (), False
    evidence = _evidence(message, keys)
    counts, note = ({}, "")
    if keys and complete:
        counts, note = _count_types(source, read_options, keys)
    return ReadProblem(
        source=source, read_options=dict(read_options), message=message,
        error_type=type(error).__name__, file_format=file_format,
        options_taken=taken, needs=tuple(needs), type_keys=tuple(keys),
        keys_complete=bool(complete), evidence=evidence, counts=counts,
        counts_note=note, suggested=_suggested(message))


def open_model(source, read_options: Mapping | None = None
               ) -> Opened | ReadProblem:
    """Read ``source`` (a path, a pattern, a directory or a list) with the
    read options; a refusal is returned as a :class:`ReadProblem`."""
    from ..core import md_readers

    # an option is kept unless it is None or an empty map; a value is never
    # compared with {} (a box typed as three rows is a (3, 3) array, whose
    # comparison has no single truth value)
    options = {k: v for k, v in dict(read_options or {}).items()
               if v is not None and not (isinstance(v, Mapping) and not v)}
    try:
        trajectory = md_readers.read_trajectory(source, **options)
    except FileNotFoundError as error:
        return ReadProblem(source=source, read_options=options,
                           message=str(error), error_type="FileNotFoundError",
                           file_format=None, options_taken=frozenset(),
                           needs=())
    except (ValueError, TypeError, OSError) as error:
        return diagnose(source, options, error)
    try:
        return Opened(trajectory, summarise(trajectory, source, options))
    except (ValueError, OSError) as error:
        close = getattr(trajectory, "close", None)
        if close is not None:
            close()
        return diagnose(source, options, error)


@dataclass(frozen=True, eq=False)
class FrameView:
    """One frame for the 3D view and the threshold panel: the frame, its
    states, its valence table at the thresholds (None when only the atoms
    are drawn), the scene built from it (None when facet.ui.md_scene is not
    present or refused), and notes."""

    k: int
    frame: object
    ox_atom: np.ndarray
    params: object
    v_bond_vu: float
    v_list_vu: float
    table: object
    scene: object
    n_bonds: int | None
    note: str
    seconds: float

    @property
    def bonds(self):
        """The valence table the scene was drawn from (None: atoms only)."""
        return self.table


def _no_bonds(v_bond_vu: float):
    from ..core import bulk

    empty = np.zeros(0, np.int32)
    return bulk.Bonds(cation=empty, anion=empty.copy(),
                      image=np.zeros((0, 3), np.int16),
                      vec_ang=np.zeros((0, 3)), d_ang=np.zeros(0),
                      v_vu=np.zeros(0), v_bond_vu=float(v_bond_vu))


def load_frame_view(trajectory, k: int, ox_overrides: Mapping, params,
                    v_bond_vu: float, v_list_vu: float, *, with_bonds: bool,
                    theme=None, atoms_only_note: str = "") -> FrameView:
    """Frame k, its valence table (bulk engine: one pair search, no
    NeighborFinder) and the scene ``md_scene.build_md_scene`` builds from
    it. A Scene is plain arrays, so it is built here, off the GUI thread;
    the GUI thread only hands it to the view."""
    from ..core import bulk, bv, md_model

    clock = time.perf_counter()
    frame = trajectory.frame(int(k))
    params = params or bv.DEFAULT
    ox = md_model.model_oxidation(frame.species, ox_overrides)
    ox_atom = ox.per_atom(frame.elements)
    table = None
    n_bonds = None
    notes = []
    if with_bonds:
        table, _ = bulk.analyse_frame(frame, ox_atom, params,
                                      v_bond_vu=v_bond_vu,
                                      v_list_vu=v_list_vu)
        n_bonds = len(bulk.bonds_at(table, v_bond_vu))
        notes.append(f"{frame.n_atoms} atoms, {n_bonds} cation-anion bonds "
                     f"at v_bond {v_bond_vu:g} v.u. (bulk engine); contacts "
                     f"above v_list {v_list_vu:g} v.u. drawn thin")
    else:
        notes.append(atoms_only_note or f"{frame.n_atoms} atoms, drawn "
                                        "without bonds")
    scene = None
    try:
        from .md_scene import build_md_scene
    except ImportError:
        notes.append("the 3D view needs facet.ui.md_scene, which is not "
                     "present in this build")
    else:
        try:
            scene = build_md_scene(
                frame, table if table is not None else _no_bonds(v_bond_vu),
                theme=theme, ox=ox, params=params, v_bond=v_bond_vu,
                v_list=v_list_vu)
        except ValueError as error:
            notes.append(f"the scene was not built: {error}")
    return FrameView(int(k), frame, ox_atom, params, float(v_bond_vu),
                     float(v_list_vu), table, scene, n_bonds,
                     "; ".join(notes), time.perf_counter() - clock)


@dataclass(frozen=True)
class JobFailure:
    """What a job that produced nothing reports.

    ``kind``: 'cancelled' (cancelled before anything was finished),
    'request' (the request lacks inputs; ``missing`` lists them, one
    sentence each) or 'error' (any other refusal; ``detail`` holds the
    traceback of an unexpected exception).
    """

    kind: str
    message: str
    missing: tuple = ()
    detail: str = ""


def failure_from(error: BaseException) -> JobFailure:
    from ..core import md_analysis

    if isinstance(error, md_analysis.AnalysisCancelled):
        return JobFailure("cancelled", str(error) or "cancelled")
    if isinstance(error, md_analysis.RequestError):
        return JobFailure("request", str(error),
                          tuple(m.describe() for m in error.missing))
    if isinstance(error, (ValueError, OSError)):
        return JobFailure("error", str(error))
    return JobFailure("error", f"{type(error).__name__}: {error}",
                      detail="".join(traceback.format_exception(error)))


# ---------------------------------------------------------------------------
# the worker and the job (Qt)
# ---------------------------------------------------------------------------

def delete_worker(worker) -> None:
    """Delete a worker QObject now, on the calling (GUI) thread, once the
    thread it ran on has ended (the caller has waited for it). Anything
    else (None, an object Qt deleted already) is left alone.

    A worker made in Python and deleted on its own thread (a deleteLater
    connected to the thread's finished signal) deadlocks: PySide's
    destructor takes the GIL inside QObject's destructor, under Qt's
    signal-slot locks, while the GUI thread holds the GIL and may be waiting
    for one of those locks in a connect (measured with py-spy on a hung
    tests/test_md_workspace.py run, 2026-10-07)."""
    if worker is None or not isinstance(worker, QObject):
        return
    try:
        import shiboken6

        if shiboken6.isValid(worker):
            shiboken6.delete(worker)
    except (ImportError, RuntimeError):
        pass


class _Worker(QObject):
    """Runs ``work(progress, cancelled)`` on its thread; reports by signal."""

    progressed = Signal(int, int, str)
    succeeded = Signal(object)
    failed = Signal(object)

    # The least time between two progress signals, unless the stage text
    # changes or the work ends. A choice that keeps a search that reports
    # thousands of times a second from flooding the GUI thread's queue.
    THROTTLE_S = 0.05

    def __init__(self, work: Callable, cancel: threading.Event):
        super().__init__()
        self._work = work
        self._cancel = cancel
        self._last = 0.0
        self._stage = None

    def _progress(self, done, total, stage="") -> None:
        now = time.perf_counter()
        stage = str(stage)
        if stage == self._stage and now - self._last < self.THROTTLE_S \
                and done < total:
            return
        self._last, self._stage = now, stage
        self.progressed.emit(int(done), int(total), stage)

    @Slot()
    def run(self) -> None:
        work, self._work = self._work, None
        try:
            result = work(self._progress, self._cancel.is_set)
        except Exception as error:        # noqa: BLE001 - reported, not raised
            self.failed.emit(failure_from(error))
        except BaseException as error:    # noqa: BLE001 - the thread must end
            self.failed.emit(JobFailure("error", f"{type(error).__name__}: "
                                                 f"{error}"))
        else:
            self.succeeded.emit(result)
        finally:
            # the work's closure holds what it read (a trajectory, a frame):
            # released here, whatever happens to this object afterwards
            work = result = None          # noqa: F841
            QThread.currentThread().quit()


class Job(QObject):
    """One piece of work on a thread of its own; lives on the GUI thread.

    ``progressed(done, total, stage)``, ``succeeded(result)`` and
    ``failed(JobFailure)`` are emitted on the GUI thread; ``ended()`` once
    the thread has stopped, after either of the two.
    """

    progressed = Signal(int, int, str)
    succeeded = Signal(object)
    failed = Signal(object)
    ended = Signal()

    def __init__(self, work: Callable, *, name: str = "job", parent=None):
        super().__init__(parent)
        self.name = str(name)
        self._work = work
        self._cancel = threading.Event()
        self._thread: QThread | None = None
        self._worker: _Worker | None = None
        self._started = False
        self._delivered = False
        self._done = False
        self.started_at = 0.0
        self.outcome = None

    # -- control (GUI thread) ------------------------------------------------
    def start(self) -> None:
        if self._started:
            raise RuntimeError(f"the {self.name} job was started already")
        self._started = True
        self.started_at = time.perf_counter()
        thread = QThread()
        thread.setObjectName(f"FACET {self.name}")
        worker = _Worker(self._work, self._cancel)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progressed.connect(self._on_progress, Qt.QueuedConnection)
        worker.succeeded.connect(self._on_succeeded, Qt.QueuedConnection)
        worker.failed.connect(self._on_failed, Qt.QueuedConnection)
        # The worker is deleted on the GUI thread once its thread has ended
        # (_on_thread_finished, delete_worker), never on the worker thread:
        # deleted there (thread.finished -> deleteLater), PySide's destructor
        # of a worker made in Python asks for the GIL while Qt holds the
        # signal-slot locks, and a GUI thread holding the GIL and connecting
        # a signal under one of those locks waits for ever. That hung the
        # Model window in show_frame's connect (py-spy, 2026-10-07).
        thread.finished.connect(self._on_thread_finished, Qt.QueuedConnection)
        self._thread, self._worker = thread, worker
        _register(self)
        thread.start()

    def cancel(self) -> None:
        """Ask the work to stop; the engine reads it before each frame."""
        self._cancel.set()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.isRunning()

    @property
    def done(self) -> bool:
        """True once the thread has ended."""
        return self._done

    @property
    def pending(self) -> bool:
        """True from start until the result (or failure) is delivered; the
        thread may run a moment longer, which is what :attr:`done` tells."""
        return self._started and not self._delivered and not self._done

    def wait(self, ms: int) -> bool:
        """Block the calling (GUI) thread until the work ends or ``ms`` pass;
        True when the thread is no longer running."""
        thread = self._thread
        if thread is None:
            return True
        return bool(thread.wait(int(max(0, ms))))

    def elapsed_s(self) -> float:
        return time.perf_counter() - self.started_at if self._started else 0.0

    # -- relays (GUI thread) -------------------------------------------------
    @Slot(int, int, str)
    def _on_progress(self, done: int, total: int, stage: str) -> None:
        self.progressed.emit(done, total, stage)

    @Slot(object)
    def _on_succeeded(self, result) -> None:
        self.outcome = result
        self._delivered = True
        self.succeeded.emit(result)

    @Slot(object)
    def _on_failed(self, failure) -> None:
        self.outcome = failure
        self._delivered = True
        self.failed.emit(failure)

    @Slot()
    def _on_thread_finished(self) -> None:
        thread = self._thread
        if thread is not None:
            # finished() is emitted just before the thread's function returns;
            # wait for it to return before the QThread may be destroyed
            thread.wait()
        self._done = True
        if thread is not None:
            thread.deleteLater()
        # the thread has ended: the worker (and what its work held) is
        # deleted here, on the GUI thread (see start)
        worker, self._worker = self._worker, None
        delete_worker(worker)
        self._thread = None
        self._work = None
        _unregister(self)
        self.ended.emit()


class AnalysisJob(Job):
    """``md_analysis.analyse(trajectory, request)`` on a worker thread, with
    progress (done, total, the engine's stage text: 'frame k of n ...', an
    analysis name) and cancel through the engine's ``cancelled()`` hook.

    ``succeeded`` carries the :class:`~facet.core.md_analysis.ModelResult`
    (a cancelled run's partial result has ``cancelled`` True and says so in
    its notes); ``failed`` a :class:`JobFailure`.
    """

    def __init__(self, trajectory, request, parent=None):
        def work(progress, cancelled):
            from ..core import md_analysis

            check_tracks_first(trajectory, request, progress)
            return md_analysis.analyse(trajectory, request, progress=progress,
                                       cancelled=cancelled)

        super().__init__(work, name="analysis", parent=parent)
        self.request = request


# The name a refusal of the dynamics' tracks carries in a JobFailure's
# missing list (check_tracks_first), which the window reads.
TRACKS_INPUT = "dynamics tracks"


def check_tracks_first(trajectory, request, progress=None) -> None:
    """Collect the tracks the dynamics analyses read before anything else
    runs, and refuse the run at once when they cannot be collected.

    The engine collects them after every per-frame analysis, so a refusal
    (an atom that moves further than ``dynamics.step_limit_fraction`` of a
    box width between two chosen frames, a time axis that is not even)
    used to come at the end of a run of several minutes, measured at 395 s
    on a 100-frame hot melt read every 10th frame. Done only when the run
    also holds analyses that are not dynamics: a run of the dynamics alone
    reaches the same refusal first anyway, and this reads the positions of
    the chosen frames once more. Raises ``md_analysis.RequestError`` naming
    :data:`TRACKS_INPUT`; a frame that cannot be read is left to the engine,
    which leaves it out and says so.
    """
    from ..core import md_analysis as ma, md_dynamics
    from ..core.md_model import FrameError

    timed = [a for a in request.analyses if a in ma.TRAJECTORY_ANALYSES]
    if not timed or len(timed) == len(request.analyses):
        return
    chosen, every = request.frames, range(trajectory.n_frames)
    try:
        frames = (list(every if chosen is None else every[chosen])
                  if chosen is None or isinstance(chosen, slice)
                  else sorted(int(k) for k in chosen))
    except (TypeError, ValueError):
        return                       # the engine states the refusal itself
    if len(frames) < 2:
        return
    if progress is not None:
        progress(0, 1, "checking the tracks the dynamics analyses read")
    dyn = request.dynamics
    try:
        tracks = md_dynamics.collect_tracks(
            trajectory, frames=frames, timestep_fs=request.timestep_fs,
            frame_interval_ps=request.frame_interval_ps,
            elements=dyn.elements, unwrap=dyn.unwrap,
            step_limit_fraction=dyn.step_limit_fraction)
    except FrameError:
        return
    except ValueError as error:
        raise ma.RequestError([ma.MissingInput(
            "/".join(timed), TRACKS_INPUT,
            f"the tracks could not be collected: {error}")]) from None
    del tracks


# ---------------------------------------------------------------------------
# jobs that outlive their window
# ---------------------------------------------------------------------------

class _Keeper(QObject):
    """Holds every running job until its thread ends, so that no QThread is
    destroyed while it runs; waits for them when the application quits."""

    # How long the application waits, at quit, for the work of a closed
    # window to notice its cancel (the engine looks once per frame).
    QUIT_WAIT_MS = 60_000

    def __init__(self):
        super().__init__()
        self.jobs: set = set()
        self._hooked = False
        # a script that ends without app.exec() never emits aboutToQuit; a
        # QThread still running when the interpreter tears it down aborts
        # the process (exit code 127, measured 2026-10-07), so the jobs are
        # also waited for at interpreter exit
        atexit.register(_wait_at_exit, self)

    def hook(self) -> None:
        if self._hooked:
            return
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.wait_all)
            self._hooked = True

    @Slot()
    def wait_all(self) -> None:
        for job in list(self.jobs):
            job.cancel()
        for job in list(self.jobs):
            job.wait(self.QUIT_WAIT_MS)


def _wait_at_exit(keeper: _Keeper) -> None:
    try:
        keeper.wait_all()
    except RuntimeError:            # Qt already tore the objects down
        pass


_KEEPER: _Keeper | None = None


def _keeper() -> _Keeper:
    global _KEEPER
    if _KEEPER is None:
        _KEEPER = _Keeper()
    _KEEPER.hook()
    return _KEEPER


def _register(job: Job) -> None:
    _keeper().jobs.add(job)


def _unregister(job: Job) -> None:
    if _KEEPER is not None:
        _KEEPER.jobs.discard(job)


def running_jobs() -> tuple:
    """Every job whose thread has not ended yet."""
    return tuple(j for j in (_KEEPER.jobs if _KEEPER else ())
                 if not j.done)
