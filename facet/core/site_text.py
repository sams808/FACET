"""The Site tab as text: plain, Markdown or LaTeX.

The same numbers the window shows for one site (``SiteResult``), laid out
for a terminal, a README or a manuscript. No Qt, no HTML: the window's own
page is ``facet.ui.site_html``; this is the one for everywhere else.
"""
from __future__ import annotations

import math

from .coordination import SiteResult


def _rows(r: SiteResult, v_bond: float) -> list[tuple[str, str]]:
    bonded = [c for c in r.contacts if c.has_valence and c.valence > v_bond]
    bvs = sum(c.valence * c.occupancy for c in bonded)
    unc = "" if r.bvs_uncertainty != r.bvs_uncertainty else f" ± {r.bvs_uncertainty:.2f}"
    disc = "" if r.ox is None else f" ({bvs - abs(r.ox):+.2f} against {r.ox:+d})"
    plateau = r.current_plateau
    rows = [
        ("Coordination number", str(len(bonded))),
        ("Bond-valence sum", f"{bvs:.2f}{unc} v.u.{disc}"),
        ("Sum to the listing cut", _num(r.bvs_listed, 3, " v.u.")),
        ("ECoN (Hoppe)", _num(r.cn_ecoN, 2)),
        ("Maximum-gap split", str(r.cn_gap)),
        ("phi (stereoactivity)", _num(r.phi, 3)
         + ("" if r.phi_listed != r.phi_listed else f" ({r.phi_listed:.3f} to the listing cut)")),
        ("|Σ v_i u_i| (bond-valence vector)", _num(r.bvv, 3, " v.u.")),
        ("Void cone half-angle", _num(r.void_angle, 1, "°")),
        ("Mean bond length", _num(r.shape.get("d_mean"), 4, " Å")),
        ("Spread within the polyhedron", _num(r.shape.get("spread"), 4, " Å")),
        ("Plateau at this threshold",
         f"{plateau.width_decades:.2f} decades · {plateau.width_angstrom:.3f} Å"
         if plateau else "on a step edge"),
        ("Bond threshold", f"{v_bond:.4f} v.u."),
    ]
    return rows


def _num(value, decimals: int, unit: str = "") -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(v):
        return "—"
    return f"{v:.{decimals}f}{unit}"


def _contacts(r: SiteResult, v_bond: float) -> list[tuple[str, ...]]:
    out = []
    running = 0.0
    for c in sorted((c for c in r.contacts if c.has_valence),
                    key=lambda c: c.distance):
        running += c.valence * c.occupancy
        out.append((c.label, f"{c.distance:.4f}", f"{c.valence:.4f}",
                    f"{running:.3f}", "bond" if c.valence > v_bond else "",
                    "" if (c.param and c.param.fitted) else "est."))
    return out


def _plateaus(r: SiteResult, v_bond: float) -> list[tuple[str, ...]]:
    return [(str(q.cn), f"{q.v_low:.4f}–{q.v_high:.4f}", f"{q.width_decades:.2f}",
             f"{q.width_angstrom:.3f}", "●" if q.contains(v_bond) else "")
            for q in sorted(r.plateaus, key=lambda q: q.cn)]


def _headline(r: SiteResult, source: str) -> str:
    ox = "" if r.ox is None else f"{r.ox:+d}"
    bits = [f"{r.element}{ox}"]
    if r.wyckoff:
        bits.append(f"Wyckoff {r.wyckoff}")
    if r.multiplicity:
        bits.append(f"multiplicity {r.multiplicity}")
    bits.append(f"oxidation state from {r.ox_source}")
    if source:
        bits.append(source)
    return " · ".join(bits)


CONTACT_HEAD = ("atom", "d / Å", "v / v.u.", "Σ so far", "", "")
PLATEAU_HEAD = ("CN", "threshold range / v.u.", "decades", "gap / Å", "")


def site_text(r: SiteResult, v_bond: float, *, fmt: str = "txt",
              source: str = "") -> str:
    """The site as text (``fmt`` 'txt'), Markdown ('md') or LaTeX ('tex')."""
    fmt = fmt.lower()
    rows = _rows(r, v_bond)
    contacts = _contacts(r, v_bond)
    plateaus = _plateaus(r, v_bond)
    head = _headline(r, source)
    if fmt == "md":
        out = [f"## {r.label}", "", head, "", "| | |", "|---|---:|"]
        out += [f"| {k} | {v} |" for k, v in rows]
        out += ["", "**Contacts**", "", "| " + " | ".join(CONTACT_HEAD) + " |",
                "|---|---:|---:|---:|---|---|"]
        out += ["| " + " | ".join(c) + " |" for c in contacts]
        out += ["", "**Plateaus**", "", "| " + " | ".join(PLATEAU_HEAD) + " |",
                "|---:|---|---:|---:|---|"]
        out += ["| " + " | ".join(p) + " |" for p in plateaus]
        if r.notes:
            out += ["", *[f"_{n}_" for n in r.notes]]
        return "\n".join(out)
    if fmt == "tex":
        def esc(text: str) -> str:
            return (text.replace("\\", r"\textbackslash{}").replace("&", r"\&")
                    .replace("%", r"\%").replace("_", r"\_").replace("°", r"$^\circ$")
                    .replace("Σ", r"$\Sigma$").replace("±", r"$\pm$")
                    .replace("·", r"$\cdot$").replace("–", "--").replace("●", r"$\bullet$")
                    .replace("—", "--").replace("phi", r"$\varphi$"))
        out = [f"% {esc(r.label)}: {esc(head)}",
               r"\begin{tabular}{lr}", r"\hline"]
        out += [f"{esc(k)} & {esc(v)} \\\\" for k, v in rows]
        out += [r"\hline", r"\end{tabular}", "",
                r"\begin{tabular}{lrrrll}", r"\hline",
                " & ".join(esc(h) for h in CONTACT_HEAD) + r" \\", r"\hline"]
        out += [" & ".join(esc(x) for x in c) + r" \\" for c in contacts]
        out += [r"\hline", r"\end{tabular}", "",
                r"\begin{tabular}{rlrrl}", r"\hline",
                " & ".join(esc(h) for h in PLATEAU_HEAD) + r" \\", r"\hline"]
        out += [" & ".join(esc(x) for x in p) + r" \\" for p in plateaus]
        out += [r"\hline", r"\end{tabular}"]
        if r.notes:
            out += ["", *[f"% {esc(n)}" for n in r.notes]]
        return "\n".join(out)
    width = max(len(k) for k, _ in rows)
    out = [r.label, head, ""]
    out += [f"  {k.ljust(width)}  {v}" for k, v in rows]
    out += ["", "  Contacts"]
    cols = list(zip(*([CONTACT_HEAD] + contacts))) if contacts else []
    widths = [max(len(x) for x in col) for col in cols] if cols else []
    if contacts:
        out.append("  " + "  ".join(h.ljust(w) for h, w in zip(CONTACT_HEAD, widths)))
        for c in contacts:
            out.append("  " + "  ".join(x.ljust(w) for x, w in zip(c, widths)))
    out += ["", "  Plateaus"]
    cols = list(zip(*([PLATEAU_HEAD] + plateaus))) if plateaus else []
    widths = [max(len(x) for x in col) for col in cols] if cols else []
    if plateaus:
        out.append("  " + "  ".join(h.ljust(w) for h, w in zip(PLATEAU_HEAD, widths)))
        for p in plateaus:
            out.append("  " + "  ".join(x.ljust(w) for x, w in zip(p, widths)))
    if r.notes:
        out += ["", *[f"  {n}" for n in r.notes]]
    return "\n".join(out)
