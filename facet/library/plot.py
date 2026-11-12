"""Quick figures of what the library computes, with matplotlib.

    facet.plot.cn_function(c.site("Bi1"), save="cn.png")
    facet.plot.contacts(c.site("Bi1"))
    facet.plot.stability(c)
    facet.plot.pattern(c.diffraction(), save="pattern.pdf")
    facet.plot.pdf(c.pdf(r_max=10))
    facet.plot.grid(c.valence_map("Na"), "c", 0.5)
    facet.plot.descriptor(results, "glass", "Qn Si (BV)")

Each function draws one figure and returns it; ``save`` writes it (png, pdf
or svg by the suffix) and ``show`` opens a window. They are for a first look:
every number drawn is also in the object passed in, for a figure of one's
own. The figures of the optical basicity are in ``facet.basicity.plot``.
matplotlib is imported here and nowhere else in the library.
"""
from __future__ import annotations

import math

import numpy as np

from ..basicity.plot import (GRID, HUES, INK, MUTED, OTHER, _axes, _figure,
                             _finish)

# --- one site ----------------------------------------------------------------------


def cn_function(site, *, v_min: float = 0.01, v_max: float = 0.5,
                save=None, show: bool = False):
    """The coordination number of a site against the bond threshold: the
    staircase, with the threshold in force and the plateau it sits on."""
    v, cn = site.cn_function(n=600, v_min=v_min, v_max=v_max)
    fig = _figure((6.2, 3.8), show)
    ax = _axes(fig)
    p = site.plateau
    if p is not None:
        ax.axvspan(max(p.v_low, v_min), min(p.v_high, v_max), color=HUES[0],
                   alpha=0.10, linewidth=0)
    ax.step(v, cn, where="post", color=HUES[0], linewidth=2)
    ax.axvline(site.v_bond, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate(f"threshold {site.v_bond:g} v.u.\nCN {site.cn}",
                (site.v_bond, site.cn), xytext=(6, 8), textcoords="offset points",
                color=MUTED, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(v_min, v_max)
    ax.set_ylim(bottom=0)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_xlabel("bond threshold / v.u.")
    ax.set_ylabel("coordination number")
    ax.set_title(f"{site.label}: coordination number against the threshold")
    return _finish(fig, save, show)


def contacts(site, *, save=None, show: bool = False):
    """Every contact of a site: its bond valence against its distance, with
    the bond and the listing thresholds."""
    rows = site.contacts()
    d = rows.column("distance / Å")
    v = rows.column("valence / v.u.")
    bonded = np.array([bool(r["bonded"]) for r in rows])
    elements = [r["element"] for r in rows]
    fig = _figure((6.2, 3.8), show)
    ax = _axes(fig)
    hue = {e: HUES[i] if i < len(HUES) else OTHER
           for i, e in enumerate(dict.fromkeys(elements))}
    for element in hue:
        mask = np.array([e == element for e in elements])
        ax.vlines(d[mask], 0, v[mask], color=hue[element], linewidth=1.2,
                  alpha=0.6)
        ax.scatter(d[mask & bonded], v[mask & bonded], s=38, color=hue[element],
                   edgecolor="white", linewidth=1.2, zorder=3, label=element)
        ax.scatter(d[mask & ~bonded], v[mask & ~bonded], s=38, facecolor="white",
                   edgecolor=hue[element], linewidth=1.2, zorder=3)
    r = site.result
    for level, text in ((r.v_bond, "bond threshold"), (r.v_list, "listing threshold")):
        ax.axhline(level, color=MUTED, linewidth=0.8, linestyle="--")
        ax.annotate(text, (1.0, level), xycoords=("axes fraction", "data"),
                    xytext=(-4, 3), textcoords="offset points", ha="right",
                    color=MUTED, fontsize=8)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("distance / Å")
    ax.set_ylabel("bond valence / v.u.")
    ax.set_title(f"contacts of {site.label} (filled: counted as bonds)")
    if len(hue) > 1:
        ax.legend(frameon=False, title="neighbour")
    return _finish(fig, save, show)


# --- a structure -------------------------------------------------------------------

def stability(crystal, *, low: float = 0.005, high: float = 0.30,
              save=None, show: bool = False):
    """The coordination number of every cation site against the threshold."""
    scan = crystal.stability(low, high)
    fig = _figure((6.4, 4.0), show)
    ax = _axes(fig)
    labels = list(scan.per_site)
    for i, label in enumerate(labels):
        colour = HUES[i] if i < len(HUES) else OTHER
        counts = np.asarray(scan.per_site[label], float)
        ax.step(scan.thresholds, counts, where="mid", color=colour, linewidth=2,
                label=label)
        if len(labels) <= 4:
            ax.annotate(label, (scan.thresholds[0], counts[0]), xytext=(-4, 0),
                        textcoords="offset points", ha="right", va="center",
                        color=MUTED, fontsize=8)
    ax.axvline(crystal.v_bond, color=MUTED, linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("bond threshold / v.u.")
    ax.set_ylabel("coordination number")
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_title(f"{crystal.name}: coordination number of every site")
    if len(labels) > 1:
        ax.legend(frameon=False, title="site", ncols=2 if len(labels) > 6 else 1)
    return _finish(fig, save, show)


def pattern(found, *, two_theta_min: float | None = None,
            two_theta_max: float | None = None, profile: bool = True,
            labels: int = 8, save=None, show: bool = False):
    """A calculated powder pattern: the lines, the broadened profile and the
    indices of the strongest lines."""
    lines = found.reflections
    if not lines:
        raise ValueError("the pattern holds no reflection")
    x = np.array([r.two_theta for r in lines])
    y = np.array([r.intensity for r in lines])
    lo = float(x.min()) - 2.0 if two_theta_min is None else float(two_theta_min)
    hi = float(x.max()) + 2.0 if two_theta_max is None else float(two_theta_max)
    fig = _figure((7.0, 3.8), show)
    ax = _axes(fig)
    if profile:
        px, py = found.profile(two_theta_min=max(lo, 0.5), two_theta_max=hi)
        ax.plot(px, py, color=HUES[0], linewidth=1.5, label="profile")
    ax.vlines(x, 0, y, color=HUES[1], linewidth=1.2, label="reflections")
    strongest = sorted(lines, key=lambda r: -r.intensity)[:max(0, int(labels))]
    for r in strongest:
        ax.annotate(f"{r.h} {r.k} {r.l}", (r.two_theta, r.intensity),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    color=MUTED, fontsize=7)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 112)
    ax.set_xlabel("2θ / degrees")
    ax.set_ylabel("relative intensity")
    ax.set_title(f"{found.structure_name}: {found.radiation}, "
                 f"λ = {found.wavelength:g} Å")
    ax.legend(frameon=False)
    return _finish(fig, save, show)


def pdf(found, which: str = "G", *, save=None, show: bool = False):
    """A pair distribution function: ``which`` is 'G' (reduced), 'g' or 'R'
    (radial distribution)."""
    if which not in ("G", "g", "R"):
        raise ValueError("which: 'G', 'g' or 'R'")
    fig = _figure((6.6, 3.8), show)
    ax = _axes(fig)
    ax.plot(found.r, getattr(found, which), color=HUES[0], linewidth=1.6)
    if which == "G":
        ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlim(float(found.r[0]), float(found.r[-1]))
    ax.set_xlabel("r / Å")
    ax.set_ylabel({"G": "G(r) / Å⁻²", "g": "g(r)", "R": "R(r) / Å⁻¹"}[which])
    ax.set_title(f"{found.structure_name}: {which}(r), {found.radiation}")
    return _finish(fig, save, show)


def grid(found, axis: str = "c", at: float = 0.5, *, levels=None,
         save=None, show: bool = False):
    """One plane of a field over the cell (a valence map, an energy map, a
    density read from a file): the grid's plane nearest fraction ``at``
    along a cell axis, drawn in fractional coordinates of the other two."""
    k = {"a": 0, "b": 1, "c": 2}.get(str(axis).lower())
    if k is None:
        raise ValueError("axis: 'a', 'b' or 'c'")
    values = np.asarray(found.values, float)
    n = values.shape[k]
    index = int(round(float(at) * n)) % n
    plane = np.take(values, index, axis=k)
    names = [x for i, x in enumerate("abc") if i != k]
    fig = _figure((5.4, 4.4), show)
    ax = _axes(fig, grid_axis="")
    finite = plane[np.isfinite(plane)]
    lo, hi = (np.percentile(finite, [2, 98]) if finite.size else (0.0, 1.0))
    if levels is not None:
        lo, hi = levels
    mesh = ax.pcolormesh(np.linspace(0, 1, plane.shape[0] + 1),
                         np.linspace(0, 1, plane.shape[1] + 1), plane.T,
                         cmap="viridis", vmin=lo, vmax=hi, rasterized=True)
    bar = fig.colorbar(mesh, ax=ax, shrink=0.85)
    bar.set_label(f"{found.name}" + (f" / {found.units}" if found.units else ""))
    bar.outline.set_visible(False)
    ax.set_aspect("equal")
    ax.set_xlabel(f"fraction along {names[0]}")
    ax.set_ylabel(f"fraction along {names[1]}")
    ax.set_title(f"plane at {index / n:.3f} along {axis}")
    return _finish(fig, save, show)


# --- a descriptor of an MD model -----------------------------------------------------

def descriptor(results, analysis: str | None = None, name: str | None = None, *,
               save=None, show: bool = False):
    """One descriptor of an MD analysis: a distribution as bars, a histogram
    as steps, a series as a line, each with the spread over the frames.

    ``results`` is what ``Model.analyse`` returned, with the analysis and
    the descriptor named; or a container taken from it."""
    container = results.get(analysis, name) if analysis is not None else results
    kind = type(container).__name__
    title = getattr(container, "name", name or "")
    fig = _figure((6.4, 3.8), show)
    ax = _axes(fig)
    mean = np.asarray(getattr(container, "mean", []), float)
    std = np.asarray(getattr(container, "std", np.full(mean.shape, np.nan)), float)
    if kind == "Distribution":
        keys = [str(key) for key in container.keys]
        ax.bar(range(len(keys)), mean, width=0.62, color=HUES[0])
        if np.isfinite(std).any():
            ax.errorbar(range(len(keys)), mean, yerr=np.nan_to_num(std), fmt="none",
                        ecolor=INK, elinewidth=1, capsize=3)
        ax.set_xticks(range(len(keys)), keys,
                      rotation=45 if max(map(len, keys), default=0) > 5 else 0,
                      ha="right" if max(map(len, keys), default=0) > 5 else "center")
        ax.set_ylabel(str(container.kind))
        for x, value in enumerate(mean):
            if len(keys) <= 12 and math.isfinite(value):
                ax.annotate(f"{value:.3g}", (x, value), xytext=(0, 3),
                            textcoords="offset points", ha="center",
                            color=MUTED, fontsize=8)
    elif kind == "Histogram":
        edges = np.asarray(container.edges, float)
        ax.stairs(mean, edges, color=HUES[0], linewidth=1.8, fill=True,
                  alpha=0.15)
        ax.stairs(mean, edges, color=HUES[0], linewidth=1.8)
        ax.set_xlabel(f"{title} / {container.unit}" if container.unit else title)
        ax.set_ylabel("density" if container.density else "count")
    elif kind == "Series":
        x = np.asarray(container.axis, float)
        ax.plot(x, mean, color=HUES[0], linewidth=1.6)
        if np.isfinite(std).any():
            ax.fill_between(x, mean - std, mean + std, color=HUES[0], alpha=0.18,
                            linewidth=0)
        ax.set_xlabel(f"{container.axis_name} / {container.axis_unit}"
                      if container.axis_unit else str(container.axis_name))
        ax.set_ylabel(f"{title} / {container.value_unit}"
                      if container.value_unit else title)
    else:
        raise ValueError(f"{title or kind}: a {kind.lower()} is a table of "
                         "rows, not a curve; read it with results.table()")
    ax.set_title(title)
    return _finish(fig, save, show)


__all__ = ["cn_function", "contacts", "stability", "pattern", "pdf", "grid",
           "descriptor", "GRID"]
