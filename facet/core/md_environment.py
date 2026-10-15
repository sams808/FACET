"""One atom's environment in an MD frame: its neighbour shells, and that
cluster written out as XYZ for a calculation elsewhere.

The bonds are the ones the bond threshold counts (``bulk.bonds_at``), so the
first shell is the coordination the analysis reports and the second shell is
the coordination of those neighbours -- a Si with its four O and the Si, Na
or Ca each O touches. The cluster is unwrapped about the centre atom by the
minimum image, so a bond across the box edge comes out as one short bond
rather than two atoms a box apart, which is what a DFT or NMR input needs.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from .md_model import Frame


def neighbour_shells(bonds, n_atoms: int, centre: int, shells: int = 1
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Which atoms lie within ``shells`` bonds of ``centre``.

    Returns ``(keep, shell)``: ``keep`` (n_atoms,) bool, ``shell`` (n_atoms,)
    int with 0 at the centre, 1 for its bonded neighbours, 2 for theirs ...
    and -1 outside. Bonds are taken undirected; ``bonds`` is a ``bulk.Bonds``
    (``cation``, ``anion`` rows) or any pair of row arrays.
    """
    if hasattr(bonds, "cation"):
        a, b = np.asarray(bonds.cation, int), np.asarray(bonds.anion, int)
    else:
        a, b = (np.asarray(x, int) for x in bonds)
    if not 0 <= centre < n_atoms:
        raise ValueError(f"atom row {centre} is not in a frame of {n_atoms}")
    shells = max(int(shells), 0)
    order = np.argsort(a, kind="stable")
    a_sorted, b_sorted = a[order], b[order]
    order_b = np.argsort(b, kind="stable")
    a_by_b, b_sorted2 = a[order_b], b[order_b]

    def neighbours(row: int) -> np.ndarray:
        lo, hi = np.searchsorted(a_sorted, [row, row + 1])
        from_a = b_sorted[lo:hi]
        lo, hi = np.searchsorted(b_sorted2, [row, row + 1])
        from_b = a_by_b[lo:hi]
        return np.concatenate([from_a, from_b])

    shell = np.full(n_atoms, -1, np.int32)
    shell[centre] = 0
    queue = deque([centre])
    while queue:
        row = int(queue.popleft())
        if shell[row] >= shells:
            continue
        for nb in neighbours(row):
            nb = int(nb)
            if shell[nb] < 0:
                shell[nb] = shell[row] + 1
                queue.append(nb)
    return shell >= 0, shell


def slab_mask(frame: Frame, axis: int, centre_fraction: float,
              thickness_ang: float) -> np.ndarray:
    """Which atoms lie within a slab normal to box axis ``axis``, centred at
    ``centre_fraction`` of the box along it, ``thickness_ang`` thick, with
    the periodic wrap taken into account."""
    frac = np.asarray(frame.frac, float)[:, axis]
    length = float(np.linalg.norm(np.asarray(frame.box_ang, float)[axis]))
    if length <= 0:
        return np.ones(len(frac), bool)
    half = 0.5 * float(thickness_ang) / length
    d = frac - float(centre_fraction)
    d -= np.round(d)
    return np.abs(d) <= half + 1e-12


def unwrapped_about(frame: Frame, rows: np.ndarray, centre: int) -> np.ndarray:
    """Cartesian positions (Å) of ``rows``, each the minimum image relative
    to the centre atom, so the cluster is one connected piece."""
    frac = np.asarray(frame.frac, float)
    box = np.asarray(frame.box_ang, float)
    origin = np.asarray(getattr(frame, "origin_ang", np.zeros(3)), float)
    d = frac[rows] - frac[centre][None, :]
    d -= np.round(d)
    return (frac[centre][None, :] + d) @ box + origin[None, :]


def environment_xyz(frame: Frame, keep: np.ndarray, centre: int, *,
                    shell: np.ndarray | None = None,
                    comment: str = "") -> str:
    """The cluster as XYZ text: the centre first, then by distance from it.

    The comment line carries what was asked, and each atom line ends with
    its frame row, its atom id, its shell and its distance to the centre,
    so the cluster can be traced back to the model it was cut from.
    """
    rows = np.flatnonzero(np.asarray(keep, bool))
    if centre not in rows:
        rows = np.concatenate([[centre], rows])
    cart = unwrapped_about(frame, rows, centre)
    d = np.linalg.norm(cart - cart[list(rows).index(centre)][None, :], axis=1)
    order = np.argsort(d, kind="stable")
    rows, cart, d = rows[order], cart[order], d[order]
    symbols = np.asarray(frame.elements)
    ids = np.asarray(frame.atom_id)
    lines = [str(len(rows)), comment.replace("\n", " ").strip()]
    for row, p, dist in zip(rows, cart, d):
        sh = "" if shell is None else f" shell={int(shell[row])}"
        lines.append(f"{symbols[row]:<2s} {p[0]:14.6f} {p[1]:14.6f} "
                     f"{p[2]:14.6f}   # row={int(row)} id={int(ids[row])}"
                     f"{sh} d={dist:.4f}")
    return "\n".join(lines) + "\n"


def environment_extxyz(frame: Frame, keep: np.ndarray, centre: int, *,
                       comment: str = "") -> str:
    """The cluster as extended XYZ: a Lattice= of the frame's box, the
    columns species:S:1:pos:R:3, the frame's velocities when it carries
    them, and the frame row of each atom -- the form a restart or ASE
    reads."""
    rows = np.flatnonzero(np.asarray(keep, bool))
    if centre not in rows:
        rows = np.concatenate([[centre], rows])
    cart = unwrapped_about(frame, rows, centre)
    box = np.asarray(frame.box_ang, float)
    lattice = " ".join(f"{x:.6f}" for x in box.reshape(-1))
    vel = getattr(frame, "vel_ang_per_ps", None)
    props = "species:S:1:pos:R:3:row:I:1" + (":vel:R:3" if vel is not None else "")
    note = comment.replace("\n", " ").replace('"', "'")
    head = (f'Lattice="{lattice}" Properties={props} pbc="T T T" '
            f'comment="{note}"')
    symbols = np.asarray(frame.elements)
    lines = [str(len(rows)), head]
    for row, p in zip(rows, cart):
        line = f"{symbols[row]:<2s} {p[0]:14.6f} {p[1]:14.6f} {p[2]:14.6f} {int(row)}"
        if vel is not None:
            v = np.asarray(vel, float)[row]
            line += f" {v[0]:12.6f} {v[1]:12.6f} {v[2]:12.6f}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def write_environment_extxyz(path, frame: Frame, keep: np.ndarray, centre: int,
                             *, comment: str = "") -> Path:
    path = Path(path)
    path.write_text(environment_extxyz(frame, keep, centre, comment=comment),
                    encoding="utf-8")
    return path


def cluster_structure(frame: Frame, keep: np.ndarray, centre: int, *,
                      name: str = "cluster", ox=None):
    """The cluster as a P1 Structure in a box of its own, the centre atom
    first (site 0), unwrapped about it: what a FEFF input or a CIF of the
    environment is written from."""
    from .readers import _assemble, cell_from_vectors
    rows = np.flatnonzero(np.asarray(keep, bool))
    if centre not in rows:
        rows = np.concatenate([[centre], rows])
    cart = unwrapped_about(frame, rows, centre)
    centre_pos = cart[list(rows).index(centre)]
    order = np.argsort(np.linalg.norm(cart - centre_pos[None, :], axis=1),
                       kind="stable")
    rows, cart = rows[order], cart[order]
    span = cart.max(axis=0) - cart.min(axis=0)
    box = np.maximum(span + 8.0, 8.0)
    cell = cell_from_vectors(np.diag(box))
    shifted = cart - cart.min(axis=0) + 4.0
    frac = shifted / box[None, :]
    symbols = np.asarray(frame.elements)
    ids = np.asarray(frame.atom_id)
    table = []
    for row, f in zip(rows, frac):
        table.append((f"{symbols[row]}{int(ids[row])}", str(symbols[row]),
                      tuple(float(x) for x in f), 1.0))
    structure = _assemble(name, cell, table, None, note_p1=True)
    structure.notes.append(
        f"a cluster of {len(rows)} atoms cut from an MD frame about row "
        f"{int(centre)}, unwrapped by the minimum image, in a box 8 Å wider "
        "than it; the box is not the model's")
    if ox is not None:
        try:
            for site in structure.sites:
                site.ox = int(np.asarray(ox)[rows[[s.label for s in structure.sites].index(site.label)]])
        except Exception:                            # noqa: BLE001
            pass
    return structure


def write_environment_xyz(path, frame: Frame, keep: np.ndarray, centre: int,
                          *, shell: np.ndarray | None = None,
                          comment: str = "") -> Path:
    path = Path(path)
    path.write_text(environment_xyz(frame, keep, centre, shell=shell,
                                    comment=comment), encoding="utf-8")
    return path
