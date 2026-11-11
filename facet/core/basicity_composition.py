"""Optical basicity of a composition written as it is on a batch sheet.

``30Na2O-70SiO2``, ``Na2O 25, CaO 10, SiO2 65``, ``20CaF2-80SiO2``,
``Na2Si2O5``: the amounts are moles (or weights, converted with the standard
atomic weights) of oxide and fluoride components, and the value is Duffy and
Ingram's sum over the components,

    Lambda = sum_c X_c Lambda_c,

X_c the share of the anion charge component c carries -- the equivalent
fraction of their eq. (4) for an oxide glass, and Duffy's eq. (4) of 1989
(J. Non-Cryst. Solids 109, 35) for a fluoride or an oxide/fluoride glass,
where each cation is counted with the anion it is batched with. No
structure is read: this is the number a composition alone states.

A component's cation takes the oxidation state its formula states (FeO is
Fe2+, Fe2O3 is Fe3+). A component with several cations (Na2Si2O5,
NaAlSi3O8) takes the usual state of each, and is refused when those do not
balance its anions: write it as its oxides.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from itertools import product

from . import basicity as B
from . import elements

_NUMBER = r"(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_TOKEN = re.compile(rf"\s*({_NUMBER})\s*%?|\s*([A-Z][A-Za-z0-9().\[\]]*)")
_SEPARATORS = re.compile(r"[-–—·+,;:=|/*]")
_ELEMENT = re.compile(rf"([A-Z][a-z]?)({_NUMBER})?")


class CompositionError(ValueError):
    """A composition that cannot be read, with what was expected."""


def parse_formula(text: str) -> dict[str, float]:
    """Atoms by element of one formula, parentheses and decimals allowed:
    ``Ca3(PO4)2`` gives Ca 3, P 2, O 8."""
    source = str(text).strip().replace("[", "(").replace("]", ")")
    if not source:
        raise CompositionError("an empty formula")
    stack: list[dict[str, float]] = [{}]
    at = 0
    while at < len(source):
        ch = source[at]
        if ch == "(":
            stack.append({})
            at += 1
        elif ch == ")":
            if len(stack) < 2:
                raise CompositionError(f"{text!r}: a ')' with no '(' before it")
            at += 1
            m = re.match(_NUMBER, source[at:])
            times = float(m.group(0)) if m else 1.0
            at += len(m.group(0)) if m else 0
            inner = stack.pop()
            for element, n in inner.items():
                stack[-1][element] = stack[-1].get(element, 0.0) + n * times
        else:
            m = _ELEMENT.match(source, at)
            if not m:
                raise CompositionError(
                    f"{text!r}: cannot read {source[at:]!r} as an element")
            element = m.group(1)
            if element not in B.PAULING_1960 and element not in elements.COMMON_OX:
                raise CompositionError(f"{text!r}: {element!r} is not an "
                                       "element symbol")
            n = float(m.group(2)) if m.group(2) else 1.0
            stack[-1][element] = stack[-1].get(element, 0.0) + n
            at = m.end()
    if len(stack) != 1:
        raise CompositionError(f"{text!r}: a '(' is not closed")
    return {e: n for e, n in stack[0].items() if n > 0}


def formula_weight(atoms: dict[str, float]) -> float:
    """Grams per mole of a formula, standard atomic weights (gemmi's)."""
    import gemmi

    return sum(float(gemmi.Element(e).weight) * n for e, n in atoms.items())


def _states(formula: str, atoms: dict[str, float]) -> dict[tuple[str, int], float]:
    """Counts by (element, oxidation state) of one component."""
    anions = {e: n for e, n in atoms.items() if e in B.ANION_FACTORS}
    cations = {e: n for e, n in atoms.items() if e not in B.ANION_FACTORS}
    if not anions:
        raise CompositionError(
            f"{formula}: no oxide or fluoride ion; optical basicity is "
            "defined for oxides and, through Duffy's 1989 scale, fluorides")
    if not cations:
        raise CompositionError(f"{formula}: no cation")
    charge = sum(B.ANION_CHARGES[e] * n for e, n in anions.items())
    out = {(e, -B.ANION_CHARGES[e]): n for e, n in anions.items()}
    if len(cations) == 1:
        (element, n), = cations.items()
        ox = charge / n
        if abs(ox - round(ox)) > 1e-6 or not 1 <= round(ox) <= 8:
            raise CompositionError(
                f"{formula}: {element} would carry {ox:g}+, not a whole "
                "oxidation state; write a mixed-valence oxide as its parts "
                "(Fe3O4 as FeO and Fe2O3)")
        out[(element, int(round(ox)))] = n
        return out
    names = sorted(cations)
    choices = []
    for element in names:
        usual = elements.COMMON_OX.get(element)
        states = [usual] if usual is not None and usual > 0 else []
        states += [s for s in elements.ALT_OX.get(element, ()) if s > 0
                   and s not in states]
        if not states:
            raise CompositionError(f"{formula}: no usual oxidation state for "
                                   f"{element}; write it as its oxides")
        choices.append(states)
    for combo in product(*choices):               # the usual states first
        if abs(sum(s * cations[e] for e, s in zip(names, combo)) - charge) < 1e-6:
            for element, state in zip(names, combo):
                out[(element, int(state))] = cations[element]
            return out
    raise CompositionError(
        f"{formula}: the usual oxidation states of "
        + ", ".join(names) + " do not balance its anions; write it as its "
        "oxides")


@dataclass
class Component:
    formula: str
    amount: float                        # as written
    moles: float                         # after any weight conversion
    weight: float                        # g/mol
    species: dict[tuple[str, int], float]
    fraction: float = float("nan")       # mole fraction
    share: float = float("nan")          # X_c, of the anion charge
    basicity: float = float("nan")       # Lambda_c on the scale
    bulk: B.BulkBasicity | None = None

    @property
    def anion_charge(self) -> float:
        return sum(-ox * n for (_, ox), n in self.species.items() if ox < 0)


@dataclass
class CompositionBasicity:
    """Lambda of a written composition, component by component."""

    text: str
    unit: str
    scale: str
    value: float = float("nan")
    components: list[Component] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def species(self) -> dict[tuple[str, int], float]:
        """Counts by (element, oxidation state) over the whole composition,
        per mole of components."""
        out: dict[tuple[str, int], float] = {}
        for c in self.components:
            for key, n in c.species.items():
                out[key] = out.get(key, 0.0) + n * c.fraction
        return out

    @property
    def moderators(self) -> list[B.Moderator]:
        seen: dict[tuple[str, int | None], B.Moderator] = {}
        for c in self.components:
            for row in (c.bulk.rows if c.bulk else ()):
                if row.moderator is not None:
                    seen.setdefault((row.element, row.ox), row.moderator)
        return list(seen.values())


def split_components(text: str) -> list[tuple[str, float]]:
    """(formula, amount) pairs of a written composition. The amount comes
    before its formula (``30Na2O-70SiO2``) or after it (``Na2O: 30``). A
    formula with no amount counts one, as in ``Na2O-2SiO2`` or
    ``CaO Al2O3 2SiO2``; that reading needs the amounts in front."""
    source = _SEPARATORS.sub(" ", str(text))
    tokens: list[tuple[str, object]] = []
    at = 0
    while at < len(source):
        if source[at].isspace():
            at += 1
            continue
        m = _TOKEN.match(source, at)
        if not m or m.end() == at:
            raise CompositionError(
                f"cannot read {source[at:at + 12]!r} in {text!r}; a "
                "composition is amounts and formulas, as in 30Na2O-70SiO2")
        if m.group(1) is not None:
            tokens.append(("n", float(m.group(1))))
        else:
            tokens.append(("f", m.group(2)))
        at = m.end()
    formulas = [t for t in tokens if t[0] == "f"]
    numbers = [t for t in tokens if t[0] == "n"]
    if not formulas:
        raise CompositionError(f"no formula in {text!r}")
    if not numbers:
        return [(str(f), 1.0) for _, f in formulas]
    if len(numbers) != len(formulas):
        # amounts in front, one where none is written: Na2O-2SiO2
        out, pending = [], None
        for kind, value in tokens:
            if kind == "n":
                if pending is not None:
                    raise CompositionError(
                        f"{text!r}: two amounts in a row ({pending:g}, "
                        f"{value:g}); each amount goes in front of its formula")
                pending = float(value)
            else:
                out.append((str(value), 1.0 if pending is None else pending))
                pending = None
        if pending is not None:
            raise CompositionError(
                f"{text!r}: the amount {pending:g} has no formula after it")
        return out
    first = tokens[0][0]
    out = []
    for k in range(0, len(tokens), 2):
        a, b = tokens[k], tokens[k + 1]
        if a[0] == b[0] or a[0] != first:
            raise CompositionError(
                f"{text!r}: amounts and formulas do not alternate")
        number, formula = (a[1], b[1]) if first == "n" else (b[1], a[1])
        out.append((str(formula), float(number)))
    return out


def composition_basicity(text, scale=B.DEFAULT_SCALE, *,
                         unit: str = "mol") -> CompositionBasicity:
    """Lambda of a written composition (see the module text). ``text`` is
    the composition, or a mapping or sequence of (formula, amount);
    ``unit`` is ``'mol'`` or ``'wt'``."""
    scale = B.scale_named(scale)
    if unit not in ("mol", "wt"):
        raise CompositionError(f"unit {unit!r}: 'mol' or 'wt'")
    if isinstance(text, str):
        pairs, shown = split_components(text), text
    else:
        items = text.items() if hasattr(text, "items") else text
        pairs = [(str(f), float(a)) for f, a in items]
        shown = " ".join(f"{a:g}{f}" for f, a in pairs)
    out = CompositionBasicity(str(shown), unit, scale.key)
    for formula, amount in pairs:
        if not (math.isfinite(amount) and amount >= 0):
            raise CompositionError(f"{formula}: an amount of {amount!r}")
        atoms = parse_formula(formula)
        weight = formula_weight(atoms)
        moles = amount / weight if unit == "wt" else amount
        out.components.append(Component(formula, amount, moles, weight,
                                        _states(formula, atoms)))
    total = sum(c.moles for c in out.components)
    if not total > 0:
        raise CompositionError(f"{shown!r}: the amounts sum to nothing")
    charge = sum(c.moles * c.anion_charge for c in out.components)
    for c in out.components:
        c.fraction = c.moles / total
        c.share = c.moles * c.anion_charge / charge
        c.bulk = B.bulk_basicity(c.species, scale)
        c.basicity = c.bulk.value
        for species in c.bulk.missing:
            if species not in out.missing:
                out.missing.append(species)
    if out.missing:
        out.notes.append("no basicity moderating parameter for "
                         + ", ".join(out.missing))
        return out
    out.value = sum(c.share * c.basicity for c in out.components if c.share > 0)
    derived = sorted({m.species for m in out.moderators if not m.measured})
    if derived and scale.key != B.PAULING.key:
        out.notes.append("gamma from the electronegativity relation, not "
                         f"from the {scale.key} table, for " + ", ".join(derived))
    anions = {e for c in out.components for (e, ox) in c.species if ox < 0}
    if len(anions) > 1:
        out.notes.append("oxide and fluoride components: each cation is "
                         "counted with the anion it is written with (Duffy "
                         "1989, eq. 4)")
    if unit == "wt":
        out.notes.append("weights converted to moles with the standard "
                         "atomic weights")
    return out


def mixing_series(start, end, scale=B.DEFAULT_SCALE, *, steps: int = 21,
                  unit: str = "mol") -> tuple[list[float], list[float]]:
    """Lambda along the join of two compositions: (x, Lambda) with x the
    mole fraction of ``end``'s components in the mixture, 0 to 1."""
    a = composition_basicity(start, scale, unit=unit)
    b = composition_basicity(end, scale, unit=unit)
    steps = max(2, int(steps))
    xs, values = [], []
    for k in range(steps):
        x = k / (steps - 1)
        mixed: dict[str, float] = {}
        for comp, weight in ((a, 1.0 - x), (b, x)):
            for c in comp.components:
                mixed[c.formula] = mixed.get(c.formula, 0.0) + weight * c.fraction
        xs.append(x)
        values.append(composition_basicity(
            [(f, n) for f, n in mixed.items() if n > 0], scale).value)
    return xs, values
