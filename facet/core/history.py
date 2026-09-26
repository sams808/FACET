"""Undo and redo for presentation state.

A state history, not a list of inverse operations. Each entry is a complete
snapshot of the presentation state -- overrides, planes, the slab, the cutoff,
the style -- with a sentence saying what changed to produce it. Undo moves a
cursor back and hands the caller the snapshot to restore.

Storing snapshots instead of inverse operations is the decision worth
explaining. Inverse operations are smaller, and they are also where undo bugs
live: every new feature has to remember to write its own inverse, and an inverse
that is subtly wrong corrupts state in a way that only shows up several steps
later. A snapshot cannot be subtly wrong. The state here is a few dictionaries
and a handful of numbers, so the memory argument does not apply -- a hundred
steps of it is smaller than one atom's vertex buffer.

What is *not* in here: the structures themselves. Undo changes how a structure is
drawn and never what it is. Loading a file, or closing one, is not undoable, and
neither is anything written to disk.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field


@dataclass
class Step:
    description: str
    snapshot: object


@dataclass
class History:
    """A bounded undo/redo stack over snapshots of presentation state."""

    limit: int = 100
    _steps: list = field(default_factory=list, repr=False)
    _cursor: int = -1

    def reset(self, snapshot, description: str = "opened") -> None:
        """Start again from one state. Clears everything before it."""
        self._steps = [Step(description, copy.deepcopy(snapshot))]
        self._cursor = 0

    def push(self, description: str, snapshot) -> None:
        """Record a new state reached by doing ``description``.

        Anything that had been undone is discarded, which is what every editor
        does: once you undo and then act, the branch you left is gone.
        """
        if self._cursor < len(self._steps) - 1:
            del self._steps[self._cursor + 1:]
        self._steps.append(Step(description, copy.deepcopy(snapshot)))
        if len(self._steps) > self.limit:
            # drop the oldest, keeping the cursor pointing at the same state
            excess = len(self._steps) - self.limit
            del self._steps[:excess]
        self._cursor = len(self._steps) - 1

    @property
    def can_undo(self) -> bool:
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        return 0 <= self._cursor < len(self._steps) - 1

    @property
    def undo_description(self) -> str:
        return self._steps[self._cursor].description if self.can_undo else ""

    @property
    def redo_description(self) -> str:
        return (self._steps[self._cursor + 1].description
                if self.can_redo else "")

    def undo(self):
        """Step back. Returns the snapshot to restore, or None at the start."""
        if not self.can_undo:
            return None
        self._cursor -= 1
        return copy.deepcopy(self._steps[self._cursor].snapshot)

    def redo(self):
        """Step forward. Returns the snapshot to restore, or None at the end."""
        if not self.can_redo:
            return None
        self._cursor += 1
        return copy.deepcopy(self._steps[self._cursor].snapshot)

    def current(self):
        if not self._steps:
            return None
        return copy.deepcopy(self._steps[self._cursor].snapshot)

    def __len__(self) -> int:
        return len(self._steps)

    def describe(self) -> list[str]:
        """The stack, newest last, with a marker on where the cursor is."""
        return [f"{'>' if i == self._cursor else ' '} {step.description}"
                for i, step in enumerate(self._steps)]
