# FACET

Coordination analysis from crystal structure files, for people who do not write
code. Windows desktop application; no Python installation required.

## What it is for

A coordination number is not measured. It is produced by choosing a cutoff, and
for a cation with a stereochemically active lone pair there is no gap in the
contact distribution where a cutoff naturally falls. Published coordination
numbers for such sites therefore differ by two or three units between papers
describing the same structure, and the difference is a reporting artefact.

FACET sets cutoffs by **partial bond valence** rather than by distance, so every
anion is cut at the same minimum bond strength instead of at a distance chosen
for oxygen. It then reports not one coordination number but the whole function:
what coordination number each threshold produces, and how wide a range of
threshold each one survives.

A coordination number that holds across a wide range of threshold is a property
of the structure. One that occupies a narrow step is a property of whoever chose
the cutoff. FACET shows which it is.

Nothing in the design is specific to any element.

## The model

Bond valence is `v = exp((R0 - d) / b)`. A contact counts as a bond above a
threshold in valence units rather than within a distance:

| | threshold | Bi–O | Bi–I |
|---|---|---|---|
| bond | 0.075 v.u. | 3.05 Å | 3.72 Å |
| tabulated | 0.020 v.u. | 3.54 Å | 4.21 Å |

The Bi–O figures are the conventional literature cutoffs, recovered rather than
assumed. The Bi–I figures are what the same physical criterion gives for a much
larger anion, and no oxygen-derived convention would have produced them.

**The plateau identity.** Sorting a site's contacts by valence, the coordination
number is `k` for any threshold between the valence of the k-th and (k+1)-th
contact, and the width of that interval in log-valence is

```
ln(v_k / v_k+1) = (d_k+1 - d_k) / b
```

so a plateau width *is* a distance gap, divided by b. The two statements are the
same statement.

**Stereoactivity.** FACET reports `phi = |sum v_i u_i| / sum v_i`, the
bond-valence vector sum normalised by the bond-valence sum. It is dimensionless,
bounded 0 to 1, and exactly invariant to an error in R0 — shifting R0 by delta
scales every `v_i` by `exp(delta/b)`, which cancels. The unnormalised vector sum
has no such property, so only `phi` can be compared across anions or across
parameter sets.

## Parameter provenance

Every parameter says where it came from, and the distinction survives into every
exported report.

* **Fitted** — refined against experimental structures, quoted with its
  citation.
* **Estimated** — computed from the O'Keeffe & Brese electronegativity
  expression. Covers 75 elements including every lanthanide, and reproduces
  fitted values to about 0.03–0.08 Å, against the 0.02 Å a fitted value itself
  carries. Always labelled as estimated.

Oxidation states are resolved by bond-valence self-consistency where the file
does not state them, which is what separates the sites of a mixed-valence
structure. The route used is recorded per site.

## Status

Engine complete and under test. Interface in progress; see `PLAN.md`.

The engine is a generalisation of the validated code behind a 2026 survey of
105 bismuth sites across 74 structures, which was itself checked by an
independent recomputation sharing no code. That survey's results are kept as a
regression fixture: FACET reproduces its bond distances, coordination numbers,
bond-valence sums and `phi` across 92 sites, with one deliberate and documented
departure.

## Running the tests

```
py -3.11 -m pytest tests/ -q
```

Tests that depend on the author's structure library skip cleanly when it is
absent.

## Licence

MIT for FACET's own source. The built application bundles Qt under the LGPL,
which constrains how a binary may be redistributed — **read
`THIRD_PARTY_NOTICES.md` before passing a build on.** In short: build as
`onedir`, ship the `licenses/` directory, and Qt stays replaceable.
