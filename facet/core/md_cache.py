"""A cache of valence tables on disk, so a frame opened again opens at once.

The pair search of a frame is what a second session pays for again. Its
result -- the ``bulk.ValenceTable`` of that frame at one listing threshold
with one parameter set -- is written beside the preferences as an ``.npz``
keyed by the file's path, size and modification time, the frame index, the
parameter set's name and the listing threshold. A file that changed on disk
misses the key and is searched again; ``clear`` empties the cache.

The table's parameter table (``ParamTable``, Python objects) is rebuilt from
the parameter set on load rather than stored, so the cache holds arrays
only and a published parameter file read later cannot be confused with the
built-in set.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from . import prefs

ARRAY_FIELDS = ("elements", "ox", "is_anion", "nbr", "image", "d_ang",
                "vec_ang", "v_vu", "n_listed", "cum_v_vu", "cum_bvv_vu",
                "n_unparameterised", "uses_estimated")
MAX_FILES = 400


def folder() -> Path:
    return prefs.folder() / "frame_cache"


def key_for(path, k: int, params_name: str, v_list_vu: float) -> str | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    text = f"{os.path.normcase(str(path))}|{stat.st_size}|{int(stat.st_mtime)}" \
           f"|{int(k)}|{params_name}|{float(v_list_vu):.6f}"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def save(path, k: int, table, v_list_vu: float) -> Path | None:
    """Write ``table`` (a ``bulk.ValenceTable``) for frame ``k`` of ``path``."""
    params_name = getattr(getattr(table, "params", None), "params_name", "") or ""
    key = key_for(path, k, params_name, v_list_vu)
    if key is None:
        return None
    try:
        folder().mkdir(parents=True, exist_ok=True)
        target = folder() / f"{key}.npz"
        arrays = {name: np.asarray(getattr(table, name)) for name in ARRAY_FIELDS}
        missing = dict(getattr(table, "missing_pairs", {}) or {})
        estimated = dict(getattr(table, "estimated_pairs", {}) or {})
        arrays["missing_keys"] = np.array(list(missing.keys()), dtype="U32")
        arrays["missing_values"] = np.array(list(missing.values()), dtype=np.int64)
        arrays["estimated_keys"] = np.array(list(estimated.keys()), dtype="U32")
        arrays["estimated_values"] = np.array(list(estimated.values()), dtype=np.int64)
        arrays["v_list_vu"] = np.array([float(getattr(table, "v_list_vu", v_list_vu))])
        arrays["r_search_ang"] = np.array([float(getattr(table, "r_search_ang", 0.0))])
        arrays["n_below_d_min"] = np.array([int(getattr(table, "n_below_d_min", 0))])
        arrays["method"] = np.array([str(getattr(table, "method", ""))], dtype="U32")
        arrays["notes"] = np.array(list(getattr(table, "notes", ()) or ()), dtype="U256")
        np.savez_compressed(target, **arrays)
        _prune()
        return target
    except (OSError, ValueError, TypeError):
        return None


def load(path, k: int, params, v_list_vu: float):
    """The cached table for frame ``k`` of ``path`` under ``params``, or
    None. ``params`` is the ``bv.ParameterSet`` the table was made with;
    its ``ParamTable`` is rebuilt from the cached elements and states."""
    from . import bulk

    key = key_for(path, k, getattr(params, "name", "") or "", v_list_vu)
    if key is None:
        return None
    target = folder() / f"{key}.npz"
    if not target.exists():
        return None
    try:
        with np.load(target, allow_pickle=False) as data:
            arrays = {name: data[name] for name in ARRAY_FIELDS}
            missing = dict(zip(data["missing_keys"].tolist(),
                               data["missing_values"].tolist()))
            estimated = dict(zip(data["estimated_keys"].tolist(),
                                 data["estimated_values"].tolist()))
            extra = {name: data[name] for name in
                     ("v_list_vu", "r_search_ang", "n_below_d_min", "method", "notes")}
        param_table = bulk.parameter_table(arrays["elements"], arrays["ox"], params)
        return bulk.ValenceTable(
            elements=arrays["elements"], ox=arrays["ox"],
            is_anion=arrays["is_anion"], nbr=arrays["nbr"], image=arrays["image"],
            d_ang=arrays["d_ang"], vec_ang=arrays["vec_ang"], v_vu=arrays["v_vu"],
            n_listed=arrays["n_listed"], cum_v_vu=arrays["cum_v_vu"],
            cum_bvv_vu=arrays["cum_bvv_vu"],
            n_unparameterised=arrays["n_unparameterised"],
            uses_estimated=arrays["uses_estimated"],
            missing_pairs=missing, estimated_pairs=estimated,
            params=param_table, v_list_vu=float(extra["v_list_vu"][0]),
            r_search_ang=float(extra["r_search_ang"][0]),
            n_below_d_min=int(extra["n_below_d_min"][0]),
            method=str(extra["method"][0]),
            notes=tuple(str(n) for n in extra["notes"].tolist()))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def clear() -> int:
    """Remove every cached table; returns how many."""
    n = 0
    try:
        for file in folder().glob("*.npz"):
            file.unlink()
            n += 1
    except OSError:
        pass
    return n


def _prune() -> None:
    """Keep the cache to MAX_FILES, the oldest going first."""
    try:
        files = sorted(folder().glob("*.npz"), key=lambda f: f.stat().st_mtime)
        for file in files[:-MAX_FILES]:
            file.unlink()
    except OSError:
        pass


def size_bytes() -> int:
    try:
        return sum(f.stat().st_size for f in folder().glob("*.npz"))
    except OSError:
        return 0
