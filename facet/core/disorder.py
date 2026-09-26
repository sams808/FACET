"""Disorder groups.

A CIF may describe a structure that is not one structure. Where two
configurations share a region of the cell, the file lists the sites of both and
tags them with ``_atom_site_disorder_assembly`` -- which region -- and
``_atom_site_disorder_group`` -- which alternative within it. Only one group of
each assembly exists in any one unit cell.

Drawing them all at once is wrong in a specific and damaging way: the
alternatives sit within a fraction of an angstrom of each other, so they look
like a cluster of impossibly close atoms, and a coordination analysis counts
neighbours the real structure never has. That is one of the ways a published
coordination number stops meaning anything -- which is the thing this program was
written about -- so FACET reads the tags, reports what it found, and lets you
choose a configuration.

What it does not do is choose for you, or average the alternatives into one
"representative" site. Whether a disorder model is the right description of a
material is a question about the material.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .structure import Structure

# The group label a CIF uses for "belongs to no alternative": present in every
# configuration. A dot or a question mark is the CIF null, and 0 is the
# convention for an ordered site in a file that tags the others.
ORDERED = {".", "?", "", "0", None}


def normalise_group(value) -> str:
    """A disorder group label as a string, with the CIF nulls collapsed."""
    if value is None:
        return ""
    text = str(value).strip().strip("'\"")
    return "" if text in ORDERED else text


def normalise_assembly(value) -> str:
    """An assembly label. An untagged site belongs to no assembly."""
    if value is None:
        return ""
    text = str(value).strip().strip("'\"")
    return "" if text in {".", "?", ""} else text


@dataclass
class Assembly:
    """One region of the cell with alternative configurations."""

    name: str
    groups: list[str] = field(default_factory=list)
    sites: dict[str, list[int]] = field(default_factory=dict)  # group -> sites

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    def describe(self) -> str:
        parts = [f"{group}: {len(self.sites.get(group, []))} site"
                 f"{'s' if len(self.sites.get(group, [])) != 1 else ''}"
                 for group in self.groups]
        label = self.name or "unnamed"
        return f"assembly {label} — " + ", ".join(parts)


@dataclass
class Disorder:
    """Every disorder assembly in a structure, and which group is chosen.

    ``selected`` maps an assembly name to the group currently shown. A group of
    ``""`` means "show them all", which is what a file's own authors drew and is
    the only way to see that the alternatives overlap at all -- so it stays
    available, it is simply not the default once a choice exists.
    """

    assemblies: dict[str, Assembly] = field(default_factory=dict)
    selected: dict[str, str] = field(default_factory=dict)

    @property
    def present(self) -> bool:
        """Does this structure actually describe alternatives?

        An assembly with one group is not disorder -- it is a file that filled in
        the column. Only a genuine choice counts.
        """
        return any(a.n_groups > 1 for a in self.assemblies.values())

    @property
    def n_alternatives(self) -> int:
        total = 1
        for assembly in self.assemblies.values():
            if assembly.n_groups > 1:
                total *= assembly.n_groups
        return total if self.present else 0

    def choose(self, assembly: str, group: str) -> None:
        self.selected[assembly] = group

    def choose_first(self) -> None:
        """Take the first group of every assembly: one configuration, not all."""
        for name, assembly in self.assemblies.items():
            if assembly.n_groups > 1:
                self.selected[name] = assembly.groups[0]

    def show_everything(self) -> None:
        for name in self.assemblies:
            self.selected[name] = ""

    def is_showing_everything(self) -> bool:
        return all(not self.selected.get(name, "")
                   for name in self.assemblies)

    def describe(self) -> list[str]:
        out = []
        for name, assembly in sorted(self.assemblies.items()):
            if assembly.n_groups <= 1:
                continue
            chosen = self.selected.get(name, "")
            out.append(assembly.describe()
                       + (f"; showing {chosen}" if chosen
                          else "; showing every group at once"))
        return out

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {"selected": dict(self.selected)}

    def apply_dict(self, data) -> None:
        for name, group in (data or {}).get("selected", {}).items():
            if name in self.assemblies:
                self.selected[name] = str(group)


def find(structure: Structure) -> Disorder:
    """Collect the disorder assemblies a structure's sites declare."""
    disorder = Disorder()
    for index, site in enumerate(structure.sites):
        group = normalise_group(getattr(site, "disorder_group", None))
        if not group:
            continue
        name = normalise_assembly(getattr(site, "disorder_assembly", None))
        assembly = disorder.assemblies.get(name)
        if assembly is None:
            assembly = Assembly(name)
            disorder.assemblies[name] = assembly
        if group not in assembly.groups:
            assembly.groups.append(group)
            assembly.sites[group] = []
        assembly.sites[group].append(index)
    for assembly in disorder.assemblies.values():
        assembly.groups.sort()
    return disorder


def site_is_shown(structure: Structure, site_index: int,
                  disorder: Disorder) -> bool:
    """Is this site part of the chosen configuration?

    A site with no disorder group is always shown: it belongs to every
    alternative. A tagged site is shown when its group is the selected one, or
    when nothing is selected for its assembly.
    """
    if site_index < 0 or site_index >= len(structure.sites):
        return False
    site = structure.sites[site_index]
    group = normalise_group(getattr(site, "disorder_group", None))
    if not group:
        return True
    name = normalise_assembly(getattr(site, "disorder_assembly", None))
    chosen = disorder.selected.get(name, "")
    return not chosen or chosen == group


def shown_sites(structure: Structure, disorder: Disorder) -> list[int]:
    return [i for i in range(len(structure.sites))
            if site_is_shown(structure, i, disorder)]


def configuration(structure: Structure, disorder: Disorder) -> Structure:
    """A copy of the structure holding one configuration only.

    A real structure with fewer sites, not a flag on the original: everything
    downstream -- the neighbour search, the coordination numbers, the bond-valence
    sums, the diffraction pattern -- then works on a structure that could exist,
    with no special case anywhere for disorder. That is the whole reason to do it
    this way rather than filter at each step and hope none was missed.

    Occupancies are left exactly as the file gave them. Renormalising a chosen
    group to full occupancy would be inventing a structure the file does not
    describe, and it would silently change every bond valence.
    """
    import copy

    if not disorder.present or disorder.is_showing_everything():
        return structure

    keep = shown_sites(structure, disorder)
    if len(keep) == len(structure.sites):
        return structure

    keep_set = set(keep)
    remap = {old: new for new, old in enumerate(keep)}

    out = copy.copy(structure)
    out.sites = [structure.sites[i] for i in keep]
    out.atoms = []
    for atom in structure.atoms:
        if atom.site_index not in keep_set:
            continue
        moved = copy.copy(atom)
        moved.site_index = remap[atom.site_index]
        out.atoms.append(moved)
    out.notes = list(structure.notes)
    chosen = ", ".join(f"{name or 'unnamed'}={group}"
                       for name, group in sorted(disorder.selected.items())
                       if group)
    out.notes.append(
        f"one disorder configuration is shown ({chosen}); "
        f"{len(structure.sites) - len(keep)} of the file's "
        f"{len(structure.sites)} sites belong to the other alternatives and "
        "are not included. Occupancies are as the file gave them and have not "
        "been renormalised.")
    return out


def close_pairs_between_groups(structure: Structure, disorder: Disorder,
                               tolerance: float = 1.2) -> list[tuple]:
    """Pairs of atoms from different groups that lie implausibly close.

    This is what makes a disorder model recognisable, and what makes drawing
    every group at once misleading: the alternatives overlap. Reported as a
    measurement -- the pair and the distance -- with no claim about whether the
    model is right.
    """
    import numpy as np

    if not disorder.present or not structure.atoms:
        return []

    groups = []
    for atom in structure.atoms:
        site = structure.sites[atom.site_index]
        groups.append((normalise_assembly(getattr(site, "disorder_assembly",
                                                  None)),
                       normalise_group(getattr(site, "disorder_group", None))))

    cart = np.array([a.cart for a in structure.atoms], float)
    frac = np.array([a.frac for a in structure.atoms], float)
    out = []
    for i in range(len(cart)):
        assembly_i, group_i = groups[i]
        if not group_i:
            continue
        for j in range(i + 1, len(cart)):
            assembly_j, group_j = groups[j]
            if not group_j or assembly_i != assembly_j or group_i == group_j:
                continue
            delta = frac[j] - frac[i]
            delta -= np.round(delta)
            distance = float(np.linalg.norm(structure.cell.orth @ delta))
            if distance <= tolerance:
                out.append((structure.atoms[i].label, group_i,
                            structure.atoms[j].label, group_j, distance))
    out.sort(key=lambda row: row[4])
    return out
