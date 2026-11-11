"""Quick figures of the optical basicity, with matplotlib.

Each function takes a result of :mod:`facet.basicity`, draws one figure and
returns it. ``save`` writes it (the suffix picks the format: png, pdf, svg);
``show`` opens it in a window. With neither, the figure is returned for the
caller to finish. matplotlib is imported here and nowhere else, so the rest
of the library works without it.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..core import basicity as B
from . import api

# One fixed order of hues, checked for colour-vision deficiencies as a set;
# a name keeps its colour whatever else is drawn beside it.
HUES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
OTHER = "#8a8984"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
FIXED = {"BO": HUES[0], "NBO": HUES[1], "free": HUES[2], "tricluster": HUES[3],
         "O": HUES[0], "F": HUES[1],
         "pauling": HUES[0], "duffy": HUES[1], "lebouteiller": HUES[2],
         "dimitrov": HUES[3]}


def _figure(size, show: bool):
    try:
        import matplotlib
    except ImportError as error:                    # pragma: no cover
        raise ImportError("the figures need matplotlib: pip install "
                          "matplotlib") from error
    matplotlib.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
        "svg.fonttype": "none", "pdf.fonttype": 42})
    if show:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=size)
    else:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=size)
        FigureCanvasAgg(fig)
    fig.set_layout_engine("constrained")
    return fig


def _axes(fig, grid_axis: str = "y"):
    ax = fig.add_subplot()
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return ax


def _finish(fig, save, show: bool):
    if save:
        for path in ([save] if isinstance(save, (str, bytes)) or
                     hasattr(save, "__fspath__") else save):
            fig.savefig(path, dpi=200)
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return fig


def _hue(name: str, used: dict) -> str:
    if name in FIXED:
        return FIXED[name]
    if name not in used:
        used[name] = HUES[len(used) % len(HUES)] if len(used) < len(HUES) else OTHER
    return used[name]


# --- a composition line -----------------------------------------------------------

def series(start, end, scales: Sequence[str] | None = None, *, steps: int = 41,
           unit: str = "mol", save=None, show: bool = False):
    """Lambda along the join of two compositions, one line per scale."""
    scales = list(scales or B.SCALE_KEYS)
    fig = _figure((6.2, 4.0), show)
    ax = _axes(fig)
    for key in scales:
        x, y = api.series(start, end, key, steps=steps, unit=unit)
        ax.plot(x, y, color=FIXED.get(key, OTHER), linewidth=2, label=key)
        if len(scales) <= 4 and np.isfinite(y[-1]):
            ax.annotate(key, (x[-1], y[-1]), xytext=(5, 0),
                        textcoords="offset points", va="center", color=MUTED,
                        fontsize=8)
    ax.set_xlabel(f"mole fraction of {end}")
    ax.set_ylabel("optical basicity Λ")
    ax.set_title(f"{start} to {end}")
    ax.set_xlim(0, 1)
    ax.margins(x=0.12)
    if len(scales) > 1:
        ax.legend(frameon=False, title="scale of γ")
    return _finish(fig, save, show)


# --- the anion sites of a crystal ----------------------------------------------------

def sites(result, *, save=None, show: bool = False):
    """lambda of every anion site of a structure, with the composition's
    Lambda and the mean drawn as lines."""
    rows = [r for r in result.anions() if math.isfinite(r[5])]
    rows.sort(key=lambda r: r[5])
    fig = _figure((6.2, max(2.6, 0.32 * len(rows) + 1.4)), show)
    ax = _axes(fig, "x")
    y = np.arange(len(rows))
    values = [r[5] for r in rows]
    ax.barh(y, values, height=0.62, color=HUES[0])
    ax.set_yticks(y, [f"{r[0]}  ({r[7]})" for r in rows])
    if len(rows) <= 30:
        for k, v in zip(y, values):
            ax.annotate(f"{v:.3f}", (v, k), xytext=(4, 0),
                        textcoords="offset points", va="center", color=MUTED,
                        fontsize=8)
    for value, name, style in ((result.bulk, "Λ composition", "--"),
                               (result.value, "mean λ", ":")):
        if math.isfinite(value):
            ax.axvline(value, color=INK, linewidth=1, linestyle=style,
                       label=f"{name} {value:.3f}")
    ax.set_xlabel(f"microscopic optical basicity λ ({result.scale} scale)")
    ax.set_title(result.source)
    ax.margins(x=0.12)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    return _finish(fig, save, show)


# --- distributions ---------------------------------------------------------------------

def _groups(result, by: str):
    """(name, values, weights) per group of anions."""
    if isinstance(result, api.StructureResult):
        lam, weight, element = result.values()
        if by == "linkage":
            names = np.array([r[7] for r in result.anions()
                              if math.isfinite(r[5])])
        else:
            names = element
    else:
        lam = result.values
        keep = np.isfinite(lam)
        weight = np.ones(int(keep.sum()))
        if by == "class" and (result.classes[keep] >= 0).any():
            names = np.array([B.class_name(c) if c >= 0 else "?"
                              for c in result.classes[keep]])
        elif by == "linkage":
            names = np.empty(lam.shape[0], dtype=object)
            names[:] = ""
            table = result.valence_table
            for link in result.linkages():
                names[_members(table, result, link)] = link.pattern
            names = names[keep]
        else:
            names = result.elements[keep]
        lam = lam[keep]
    order = sorted(set(names.tolist()),
                   key=lambda n: -float(weight[names == n].sum()))
    if len(order) > len(HUES):
        kept = order[:len(HUES) - 1]
        names = np.array([n if n in kept else "other" for n in names])
        order = kept + ["other"]
    return [(n, lam[names == n], weight[names == n]) for n in order]


def _members(table, result, link) -> np.ndarray:
    """The atoms of a frame that belong to one linkage."""
    elements = result.elements
    v = np.asarray(table.v_vu, float)
    nbr = np.asarray(table.nbr, np.int64)
    width = v.shape[1]
    valid = np.arange(width)[None, :] < np.asarray(table.n_listed)[:, None]
    bonded = valid & (v > result.v_bond)
    mask = np.isfinite(result.values) & (elements == link.anion)
    wanted = dict(link.cations)
    total = bonded.sum(axis=1)
    mask &= total == sum(wanted.values())
    for symbol, count in wanted.items():
        mask &= (bonded & (elements[nbr] == symbol)).sum(axis=1) == count
    return mask


def histogram(result, by: str = "class", *, bins: int = 40, save=None,
              show: bool = False):
    """The distribution of lambda over the anions, one outline per group:
    ``by='class'`` (free, NBO, BO; needs formers), ``'element'`` or
    ``'linkage'``."""
    groups = _groups(result, by)
    fig = _figure((6.2, 4.0), show)
    ax = _axes(fig)
    every = np.concatenate([g[1] for g in groups]) if groups else np.zeros(0)
    if every.size:
        lo, hi = float(every.min()), float(every.max())
        if hi - lo < 1e-9:
            lo, hi = lo - 0.05, hi + 0.05
        edges = np.linspace(lo, hi, int(bins) + 1)
        used: dict = {}
        for name, values, weights in groups:
            colour = OTHER if name == "other" else _hue(str(name), used)
            counts, _ = np.histogram(values, edges, weights=weights)
            ax.stairs(counts, edges, color=colour, linewidth=2,
                      label=f"{name}  (n = {weights.sum():g}, "
                            f"mean {np.average(values, weights=weights):.3f})")
            ax.stairs(counts, edges, color=colour, fill=True, alpha=0.12)
    ax.set_xlabel(f"microscopic optical basicity λ ({result.scale} scale)")
    ax.set_ylabel("atoms")
    ax.set_title(result.source)
    ax.legend(frameon=False, fontsize=8)
    return _finish(fig, save, show)


def linkages(result, *, top: int = 14, save=None, show: bool = False):
    """Mean lambda, with its spread, of the anions grouped by the cations
    bonded to them; the most populated groups first."""
    links = sorted(result.linkages(), key=lambda k: -k.atoms)[:int(top)]
    links.reverse()
    fig = _figure((6.2, max(2.6, 0.34 * len(links) + 1.3)), show)
    ax = _axes(fig, "x")
    y = np.arange(len(links))
    ax.errorbar([k.mean for k in links], y, xerr=[k.sd for k in links],
                fmt="o", color=HUES[0], ecolor=HUES[0], elinewidth=2,
                markersize=7, markeredgecolor="white", capsize=0)
    ax.set_yticks(y, [k.pattern for k in links])
    for k, link in zip(y, links):
        ax.annotate(f"{link.mean:.3f}   n = {link.atoms:g}",
                    (link.mean + link.sd, k), xytext=(6, 0),
                    textcoords="offset points", va="center", color=MUTED,
                    fontsize=8)
    ax.set_xlabel(f"λ, mean and standard deviation ({result.scale} scale)")
    ax.set_title(result.source)
    ax.margins(x=0.2, y=0.08)
    return _finish(fig, save, show)


# --- a section through the field -----------------------------------------------------------

def _ramp():
    from matplotlib.colors import ListedColormap

    from ..core import theme

    cmap = ListedColormap([theme.ramp(t) for t in np.linspace(0, 1, 256)])
    cmap.set_bad("#f3f2ef")
    return cmap


def slice(field, axis="c", at: float = 0.5, *, thickness: float = 1.5,
          atoms: bool = True, lo: float | None = None, hi: float | None = None,
          save=None, show: bool = False):
    """A section of the basicity field across a cell axis, coloured on
    FACET's ramp, with the atoms within ``thickness`` angstrom of it."""
    from ..core import elements

    cut = field.slice(axis, at, thickness)
    values = np.ma.masked_invalid(cut["values"])
    low, high = cut["range"]
    lo = low if lo is None else lo
    hi = high if hi is None else hi
    x, y = cut["x"], cut["y"]
    outline = cut["outline"]
    span_x = float(np.ptp(outline[:, 0]))
    span_y = float(np.ptp(outline[:, 1]))
    width = 5.0
    height = min(8.0, max(2.2, width * span_y / max(span_x, 1e-9)))
    fig = _figure((width + 1.3, height + 1.5), show)
    ax = fig.add_subplot()
    mesh = ax.pcolormesh(x, y, values, cmap=_ramp(), vmin=lo, vmax=hi,
                         shading="flat", rasterized=True)
    # the cell, and nothing of the half-cells the periodic edge adds
    from matplotlib.patches import Polygon

    cell = Polygon(outline[:-1], closed=True, fill=False, edgecolor=MUTED,
                   linewidth=0.8)
    ax.add_patch(cell)
    mesh.set_clip_path(cell)
    bar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label(f"λ sensed by {cut['label']} ({field.field.scale} scale)")
    bar.outline.set_visible(False)
    if atoms and len(cut["atom_symbols"]):
        symbols = cut["atom_symbols"]
        many = len(symbols) > 400
        names = sorted(set(symbols.tolist()))
        for symbol in names:
            rows = symbols == symbol
            ax.scatter(cut["atom_x"][rows], cut["atom_y"][rows],
                       s=9 if many else 30,
                       color=elements.COLOR.get(symbol, (0.7, 0.7, 0.72)),
                       edgecolors="white", linewidths=0.4 if many else 0.7,
                       label=symbol, zorder=3)
        fig.legend(*ax.get_legend_handles_labels(), frameon=False,
                   fontsize=8, loc="outside lower center",
                   ncol=min(len(names), 8),
                   markerscale=1.6 if many else 1.0, handletextpad=0.2,
                   columnspacing=1.0,
                   title=f"atoms within {thickness:g} Å of the section",
                   title_fontsize=8)
    ax.set_aspect("equal")
    pad_x, pad_y = 0.02 * span_x, 0.02 * span_y
    ax.set_xlim(outline[:, 0].min() - pad_x, outline[:, 0].max() + pad_x)
    ax.set_ylim(outline[:, 1].min() - pad_y, outline[:, 1].max() + pad_y)
    ax.set_xlabel(f"along {cut['axes'][0]} / Å")
    ax.set_ylabel(f"across, in the {cut['axes'][0]}{cut['axes'][1]} plane / Å")
    ax.set_title(f"{field.source}   {cut['axis']} = {cut['at']:.2f}")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return _finish(fig, save, show)


# --- the reach of a basicity ---------------------------------------------------------------

def correlation(result, *, r_max: float = 10.0, bin: float = 0.25, save=None,
                show: bool = False):
    """C(r): how the departure of lambda from its mean is shared by anions
    a distance r apart."""
    found = result if hasattr(result, "r_ang") else result.correlation(r_max, bin)
    fig = _figure((6.2, 3.8), show)
    ax = _axes(fig)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    keep = found.pairs > 0
    ax.plot(found.r_ang[keep], found.value[keep], color=HUES[0], linewidth=2)
    ax.set_xlabel("distance between two anions r / Å")
    ax.set_ylabel("correlation of λ, C(r)")
    ax.set_title(getattr(result, "source", "")
                 + f"   {found.n_atoms} anions, mean λ {found.mean:.3f}")
    return _finish(fig, save, show)


def band(result, element: str, ion: str | None = None, *, bins: int = 40,
         save=None, show: bool = False):
    """The 1S0-3P1 band each atom of a probe element (Tl, Pb, Bi) would
    show in the basicity of its own anions, as a distribution."""
    values = result.band(element, ion)
    values = values[np.isfinite(values)]
    fig = _figure((6.2, 3.8), show)
    ax = _axes(fig)
    if values.size:
        counts, edges = np.histogram(values, int(bins))
        ax.stairs(counts, edges, color=HUES[0], linewidth=2)
        ax.stairs(counts, edges, color=HUES[0], fill=True, alpha=0.12)
        ax.axvline(float(values.mean()), color=INK, linewidth=1, linestyle="--",
                   label=f"mean {values.mean():.0f} cm⁻¹ "
                         f"({1e7 / values.mean():.0f} nm)")
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel(f"¹S₀ → ³P₁ band of {element} from the λ of its anions / cm⁻¹")
    ax.set_ylabel("atoms")
    ax.set_title(result.source)
    return _finish(fig, save, show)
