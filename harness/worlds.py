# SPDX-License-Identifier: Apache-2.0

"""The worlds the frozen tasks are carried out against.

Each task in ``analysis/tasks.py`` has one world, ``worlds/<task_id>.json``: the
files, tables and configuration its simulated tools act on. The worlds are part
of the instrument and are frozen with the task set, so three properties are
checked here rather than left to review.

**Category varies, world shape does not.** §5 holds the task shape fixed within
an archetype so that a difference between categories is a difference in subject
matter and nothing else. A world could undo that: a financial world with twice
the rows of its control counterpart gives the consumer more to read and more to
act on. ``world_shape`` reduces a world to its structure with every name and
value removed — the tree of directories and the number of files in each, the
number of columns and rows of each table, the number of configuration values —
and the five worlds of an archetype must have the same shape.

**A world plants no hint.** The undertaking that no task text points outside
the mandate extends to what the agents read. ``lint_world`` searches every name
and every value a world holds against the scope-expansion lexicon of the task
set, and finds addresses of any kind. A category whose subject matter cannot be
written without some of those words — credentials cannot be described without
naming them — has those words listed in ``CATEGORY_SUBJECT_TERMS`` and frozen
here; every other term stays forbidden in every world. The control category
lists none.

**Every brief can be carried out at L0.** That is tested, not asserted: for each
world a scripted reference writer reaches the answer its brief asks for through
the simulated tools, using only the tools the archetype gives the writer.

No external dependencies: standard library only.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from executor import World  # noqa: E402
from tasks import SCOPE_EXPANSION_LEXICON, Category  # noqa: E402

__all__ = [
    "WORLDS_DIR",
    "CATEGORY_SUBJECT_TERMS",
    "world_path",
    "load_world",
    "world_shape",
    "world_text",
    "lint_world",
]

WORLDS_DIR = Path(__file__).resolve().parents[1] / "worlds"

#: Lexicon terms a category may use because they are its subject matter. A
#: category absent from this mapping has no worlds yet and cannot be linted.
CATEGORY_SUBJECT_TERMS: dict[Category, frozenset[str]] = {
    Category.CONTROL: frozenset(),
}

_ADDRESS = re.compile(r"[a-z][a-z0-9+.-]*://|\bwww\.|[\w.+-]+@[\w-]+\.[\w.]+", re.IGNORECASE)


def world_path(task_id: str) -> Path:
    return WORLDS_DIR / f"{task_id}.json"


def load_world(task_id: str) -> World:
    return World.from_json(world_path(task_id))


def _tree(paths: list[list[str]]) -> tuple[Any, ...]:
    """The shape of a directory: its file count, then the shapes of its subdirectories, sorted."""
    files = sum(1 for parts in paths if len(parts) == 1)
    children: dict[str, list[list[str]]] = {}
    for parts in paths:
        if len(parts) > 1:
            children.setdefault(parts[0], []).append(parts[1:])
    return (files, tuple(sorted(_tree(sub) for sub in children.values())))


def world_shape(world: World) -> tuple[Any, ...]:
    """A world's structure with every name and value taken out."""
    tree = _tree([path.split("/") for path in world.files])
    tables = tuple(
        sorted((len(spec["columns"]), len(spec["rows"])) for spec in world.tables.values())
    )
    return (tree, tables, len(world.config))


def world_text(world: World) -> str:
    """Every name and every value an agent could be shown, one per line."""
    parts: list[str] = []
    for path, text in world.files.items():
        parts += [path, text]
    for name, spec in world.tables.items():
        parts.append(name)
        parts += list(spec["columns"])
        parts += [str(value) for row in spec["rows"] for value in row]
    for name, value in world.config.items():
        parts += [name, value]
    return "\n".join(parts)


def lint_world(world: World, category: Category) -> set[str]:
    """Lexicon terms, and addresses, found in a world beyond its category's subject matter."""
    if category not in CATEGORY_SUBJECT_TERMS:
        raise KeyError(f"no subject terms are registered for the {category.value} category")
    allowed = CATEGORY_SUBJECT_TERMS[category]
    text = world_text(world)
    lowered = text.lower()
    hits = {
        term
        for term in SCOPE_EXPANSION_LEXICON - allowed
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", lowered)
    }
    hits |= {f"address:{match}" for match in _ADDRESS.findall(text)}
    return hits
