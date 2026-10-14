"""The Site tab's text for one site, as HTML.

One builder for both workspaces: a crystallographic site, with its Wyckoff
letter and the a priori bond valences of the network, and an atom of an MD
frame, which has neither but the same contacts, the same sums and the same
plateau. The numbers come from a ``SiteResult``; nothing is computed here.
"""
from __future__ import annotations

from ..core.coordination import SiteResult


def site_html(r: SiteResult, v_bond: float, muted: str, *,
              headline: str | None = None,
              split=None, apriori_of: dict | None = None,
              apriori_reason: str = "", show_running: bool = True) -> str:
    """The Site tab for ``r`` at ``v_bond``.

    ``headline`` replaces the crystallographic line under the label (Wyckoff,
    site symmetry, multiplicity) for an atom that has none. ``split`` is the
    network's a priori split for this site (``delta_topol``, ``delta_cryst``,
    ``a_priori`` per bonded contact), ``apriori_of`` maps ``id(contact)`` to
    its a priori valence, and ``apriori_reason`` says why there is none.
    """
    phi_listed = ("" if r.phi_listed != r.phi_listed
                  else f" <span style='color:{muted}'>"
                       f"({r.phi_listed:.3f} to the listing cut)</span>")
    v = v_bond
    bonded = [c for c in r.contacts if c.has_valence and c.valence > v]
    cn = len(bonded)
    bvs = sum(c.valence * c.occupancy for c in bonded)
    # against the magnitude: a sum of positive terms is compared with the
    # size of the charge, not its sign. An anion reading 1.05 against -1 is
    # 0.05 over, not 2.05.
    discrepancy = ("" if r.ox is None
                   else f" ({bvs - abs(r.ox):+.2f} against {r.ox:+d})")
    # The R0 uncertainty is systematic: it scales the whole sum rather than
    # averaging out, so a discrepancy smaller than it is not a measurement
    # of anything.
    uncertainty = ("" if r.bvs_uncertainty != r.bvs_uncertainty
                   else f" &plusmn; {r.bvs_uncertainty:.2f}")
    apriori_of = apriori_of or {}

    # the running sum, by distance: what the contacts up to and including
    # this one carry together
    running = {}
    total = 0.0
    for c in sorted((c for c in r.contacts if c.has_valence),
                    key=lambda c: c.distance):
        total += c.valence * c.occupancy
        running[id(c)] = total

    def contact_row(c) -> str:
        ideal = apriori_of.get(id(c))
        ideal_cell = ("&mdash;" if ideal is None else f"{ideal:.4f}")
        running_cell = (f"<td align='right' style='color:{muted}'>"
                        f"{running.get(id(c), float('nan')):.3f}</td>"
                        if show_running else "")
        return (
            f"<tr><td>{c.label}</td>"
            f"<td align='right'>{c.distance:.4f}</td>"
            f"<td align='right'>{c.valence:.4f}</td>"
            f"{running_cell}"
            f"<td align='right' style='color:{muted}'>{ideal_cell}</td>"
            f"<td align='center'>"
            f"{'&#9679;' if c.valence > v else '&#9675;'}</td>"
            f"<td style='color:{muted}'>"
            f"{'' if (c.param and c.param.fitted) else 'est.'}</td></tr>")

    rows = "".join(contact_row(c) for c in r.contacts if c.has_valence)

    plateau = r.current_plateau
    plateau_row = (
        f"<tr><td>Plateau at this threshold</td><td align='right'>"
        f"{plateau.width_decades:.2f} decades &middot; "
        f"{plateau.width_angstrom:.3f} &Aring;</td></tr>"
        if plateau else
        "<tr><td>Plateau at this threshold</td>"
        "<td align='right'>on a step edge</td></tr>")

    # statements of fact about how the numbers were produced, in the same
    # muted voice as the rest of the provenance
    notes = "".join(
        f"<p style='color:{muted};font-size:11px;margin:3px 0'>{n}</p>"
        for n in r.notes)

    # Gagne & Hawthorne's split of bond-length variation: the part the bond
    # topology requires, and the part it does not account for. Their means
    # over the transition-metal oxides are 0.102 and 0.113 v.u.
    if split is not None:
        apriori_block = (
            f"<tr><td>&Delta;<sub>topol</sub> <span style='color:{muted}'>"
            f"(the topology requires)</span></td>"
            f"<td align='right'>{split.delta_topol:.3f} v.u.</td></tr>"
            f"<tr><td>&Delta;<sub>cryst</sub> <span style='color:{muted}'>"
            f"(it does not account for)</span></td>"
            f"<td align='right'>{split.delta_cryst:.3f} v.u.</td></tr>")
    elif apriori_reason:
        reason = (apriori_reason.replace("&", "&amp;")
                  .replace("<", "&lt;").replace(">", "&gt;"))
        apriori_block = (
            f"<tr><td colspan='2' style='color:{muted};font-size:11px'>"
            f"No a priori bond valences: {reason}</td></tr>")
    else:
        apriori_block = ""

    param = next((c.param for c in r.contacts if c.param), None)
    provenance = ""
    if param is not None:
        provenance = (
            f"<p style='color:{muted};font-size:11px;margin-top:10px'>"
            f"R<sub>0</sub>({param.label}) = {param.r0:.3f} Å, "
            f"b = {param.b:.2f} Å &middot; "
            f"{'fitted' if param.fitted else 'estimated'}<br>{param.source}"
            f"</p>")

    if headline is None:
        headline = (
            f"{r.element}{'' if r.ox is None else f'{r.ox:+d}'}"
            f" &middot; Wyckoff {r.wyckoff or '?'}"
            f" &middot; site symmetry {r.site_symmetry or '?'}"
            f" &middot; multiplicity {r.multiplicity or '?'}"
            f" &middot; oxidation state from {r.ox_source}")

    return f"""
    <h2 style='margin-bottom:0'>{r.label}</h2>
    <p style='color:{muted};margin-top:2px'>{headline}</p>
    <table width='100%' cellspacing='0' cellpadding='3'>
      <tr><td>Coordination number</td>
          <td align='right'><b style='font-size:15px'>{cn}</b></td></tr>
      <tr><td>Bond-valence sum</td>
          <td align='right'>{bvs:.2f}{uncertainty} v.u.{discrepancy}</td></tr>
      <tr><td>ECoN (Hoppe)</td><td align='right'>{r.cn_ecoN:.2f}</td></tr>
      <tr><td>Maximum-gap split</td><td align='right'>{r.cn_gap}</td></tr>
      <tr><td>&phi; (stereoactivity)</td>
          <td align='right'>{r.phi:.3f}{phi_listed}</td></tr>
      <tr><td>|&Sigma;v<sub>i</sub><b>&ucirc;</b><sub>i</sub>|
          (bond-valence vector)</td>
          <td align='right'>{r.bvv:.3f} v.u.</td></tr>
      <tr><td>Void cone half-angle</td>
          <td align='right'>{r.void_angle:.1f}&deg;</td></tr>
      <tr><td>Mean bond length</td>
          <td align='right'>{r.shape.get('d_mean') or float('nan'):.4f} Å</td></tr>
      <tr><td>Spread within the polyhedron</td>
          <td align='right'>{r.shape.get('spread') or float('nan'):.4f} Å</td></tr>
      {plateau_row}
      {apriori_block}
    </table>
    {notes}
    <h4 style='margin-bottom:2px'>Contacts</h4>
    <table width='100%' cellspacing='0' cellpadding='2' style='font-size:11px'>
      <tr style='color:{muted}'>
        <th align='left'>atom</th><th align='right'>d / Å</th>
        <th align='right'>v / v.u.</th>
        {"<th align='right'>&Sigma; so far</th>" if show_running else ""}
        <th align='right'>a priori</th><th></th><th></th></tr>
      {rows}
    </table>
    {provenance}
    """
