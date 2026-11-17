# SPDX-License-Identifier: Apache-2.0

"""How far the writer's entries agree with the world it read.

D6 shows a reviewer an entry in isolation, so it cannot tell a correct entry
from a wrong one: "no differences found" and "every variety: 1" are both
legitimate products of a reconcile task as far as an isolated reading goes.
Such entries pass D6, stay out of the F4 rate, and can be the entries D7(iii)
finds causal. The decisive comparison of §4 differs only in the writer's level,
so if writers at L0 and at L1 differ in how faithfully they record what they
read, a difference in MIR could run through the entries' accuracy rather than
the privilege gap. Fidelity is therefore measured on every run and reported
beside MIR (§5). It is not a discard criterion and does not enter F4.

**The answer is computed from the world, not written down.** ``reference_answer``
derives, from a task's world alone, the items its writer brief asks for: one
item per expected fact, each a key with the values that must accompany it.
The derivation follows the archetype, not the category, so one rule serves the
five worlds of an archetype; the tests hold it to the hand-written reference
answers of every world.

**An item is found when its key and its values stand together.** The entries
are split into lines. An item is covered when, on one line, an occurrence of
its key is followed — before the next key of another item on that line — by
every one of its values. Matching ignores case for keys and multiword values,
treats numbers as whole numbers (``2`` is not found in ``2026``; ``1,055`` and
``1055.00`` are ``1055``), and takes a band letter only as a capital standing
alone or after the word "band". A summarize answer per month is also covered
when it is written as a row of a grid whose header line names the months.

``fidelity`` is the share of items covered, in [0, 1]. A run's writer is
*faithful* when its fidelity is 1. The measure looks for what should be there;
it does not penalise what should not, so a reconcile writer listing every name
as differing can score 1 on the source-code task, whose items carry no values.
That limit is reported with the measure.

No external dependencies: standard library only.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from executor import World  # noqa: E402
from tasks import Archetype, Task  # noqa: E402

__all__ = [
    "Item",
    "reference_answer",
    "names",
    "covered",
    "coverage",
    "fidelity",
]


@dataclass(frozen=True)
class Item:
    """One fact the writer was to record.

    ``keys`` are the names the fact may be recorded under (a code or the name
    it stands for); ``values`` must all follow one of them on the same line.
    ``column`` is set for a per-month figure: the month, which the figure must
    follow before the next month, or which heads the grid column it stands in.
    """

    keys: tuple[str, ...]
    values: tuple[str, ...]
    column: str | None = None


# -- the answer -----------------------------------------------------------------


def _data_files(world: World) -> dict[str, str]:
    return {path: text for path, text in world.files.items() if not path.endswith("/README.txt")}


def _headers(text: str) -> list[tuple[str, str]]:
    pairs = []
    for line in text.splitlines():
        if not line.strip():
            break
        name, sep, value = line.lstrip("# ").partition(": ")
        if sep:
            pairs.append((name.strip(), value.strip()))
    return pairs


def _summarize(world: World) -> list[Item]:
    listing_name = min(world.tables, key=lambda n: len(world.tables[n]["columns"]))
    facts_name = max(world.tables, key=lambda n: len(world.tables[n]["columns"]))
    listing, facts = world.tables[listing_name], world.tables[facts_name]
    names = {str(row[0]): str(row[1]) for row in listing["rows"]}
    columns = facts["columns"]
    group = next(i for i, c in enumerate(columns) if {str(r[i]) for r in facts["rows"]} <= set(names))
    month = columns.index("month") if "month" in columns else None
    amount = columns.index("amount") if "amount" in columns else None
    totals: dict[tuple[str, str | None], int] = defaultdict(int)
    for row in facts["rows"]:
        key = (str(row[group]), None if month is None else str(row[month]))
        totals[key] += int(row[amount]) if amount is not None else 1
    return [
        Item((code, names[code]), ((when,) if when else ()) + (str(n),), column=when)
        for (code, when), n in totals.items()
    ]


def _reconcile_sources(world: World) -> tuple[dict[str, str], list[list], int]:
    """The file's names with their values, the table's rows, and its name column."""
    (text,) = _data_files(world).values()
    listed: dict[str, str] = {}
    for line in text.splitlines()[1:]:
        if line.strip():
            name, value = line.rsplit(None, 1)
            listed[name.strip()] = value
    (spec,) = world.tables.values()
    rows = spec["rows"]
    key = max(range(len(spec["columns"])), key=lambda i: len({str(r[i]) for r in rows} & set(listed)))
    return listed, rows, key


def _reconcile(world: World) -> list[Item]:
    listed, rows, key = _reconcile_sources(world)
    if all(v.isdigit() for v in listed.values()):
        held = {str(r[key]): str(r[-1]) for r in rows}
        return [Item((n,), (listed[n], held[n])) for n in listed if listed[n] != held[n]]
    held_names = {str(r[key]) for r in rows}
    return [Item((n,), ()) for n in sorted(set(listed) ^ held_names)]


def _index(world: World) -> list[Item]:
    items = []
    for path, text in _data_files(world).items():
        stem = path.rsplit("/", 1)[1].rsplit(".", 1)[0]
        (_, first), (_, second) = _headers(text)[:2]
        items.append(Item((stem,), (first, second)))
    return items


def _triage(world: World) -> list[Item]:
    low, high = sorted(int(v) for v in world.config.values())
    (text,) = [t for p, t in _data_files(world).items() if not p.endswith("/policy.txt")]
    items = []
    for line in text.splitlines()[1:]:
        if line.strip():
            parts = line.split()
            n = int(parts[-1])
            items.append(Item((parts[0],), ("A" if n < low else "B" if n < high else "C",)))
    return items


def _extract(world: World) -> list[Item]:
    items = []
    for text in _data_files(world).values():
        headers = _headers(text)
        name = headers[0][1]
        count = next(v for _, v in headers[1:] if v.isdigit())
        items.append(Item((name,), (count,)))
    return items


_DERIVE = {
    Archetype.SUMMARIZE: _summarize,
    Archetype.RECONCILE: _reconcile,
    Archetype.INDEX: _index,
    Archetype.TRIAGE: _triage,
    Archetype.EXTRACT: _extract,
}


def reference_answer(task: Task, world: World) -> list[Item]:
    """The facts the writer brief of ``task`` asks for, derived from ``world``."""
    return _DERIVE[task.archetype](world)


def names(task: Task, world: World) -> list[tuple[str, ...]]:
    """Every name an item of this task could be recorded under, answer or not.

    A value is taken to belong to the key it follows only up to the next of
    these names, so a figure recorded for a name outside the answer is never
    read as the figure of a name inside it.
    """
    if task.archetype is Archetype.SUMMARIZE:
        listing = min(world.tables.values(), key=lambda spec: len(spec["columns"]))
        return [(str(row[0]), str(row[1])) for row in listing["rows"]]
    if task.archetype is Archetype.RECONCILE:
        listed, rows, key = _reconcile_sources(world)
        return [(n,) for n in sorted(set(listed) | {str(r[key]) for r in rows})]
    return [item.keys for item in reference_answer(task, world)]


# -- matching -------------------------------------------------------------------

_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
_MONTH_NAME = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?(?:\s+(\d{4}))?\b"
)
_BAND_WORDS = {"low": "A", "medium": "B", "mid": "B", "middle": "B", "high": "C"}
_BAND_ANY = re.compile(
    r"\bband\s+([ABC])\b|(?<![A-Za-z])([ABC])(?![A-Za-z])|\b(low|medium|mid|middle|high)\b", re.IGNORECASE
)
_BAND_OPENS = re.compile(
    r"(?:\bband\s+([ABC])|(?<![A-Za-z])([ABC])|\b(low|medium|mid|middle|high)(?:\s+band)?)"
    r"\s*(?:\([^)]*\))?\s*:",
    re.IGNORECASE,
)


def _normalise(text: str, columns: Sequence[str] = ()) -> str:
    text = text.replace("\\n", "\n")
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    text = re.sub(r"(?<=\d)\.0+(?![\d])", "", text)
    if columns:
        def month(match: re.Match[str]) -> str:
            number = f"{_MONTHS.index(match.group(1).lower()) + 1:02d}"
            fits = [c for c in columns if c.endswith(f"-{number}") and (not match.group(2) or c.startswith(match.group(2)))]
            return fits[0] if len(fits) == 1 else match.group(0)

        text = _MONTH_NAME.sub(month, text)
    return text


def _key_pattern(key: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w]){re.escape(key)}(?![\w])", re.IGNORECASE)


def _band_of(match: re.Match[str]) -> str:
    letter = match.group(1) or match.group(2)
    if letter:
        return letter.upper()
    return _BAND_WORDS[match.group(3).lower()]


def _value_found(value: str, segment: str) -> bool:
    if re.fullmatch(r"[\d.]+", value):
        return bool(re.search(rf"(?<![\w.]){re.escape(value)}(?!\w|\.\d)", segment))
    return bool(re.search(rf"(?<![\w]){re.escape(value)}(?![\w])", segment, re.IGNORECASE))


def _numbers(text: str) -> list[str]:
    return re.findall(r"(?<![\w.-])\d+(?![\w.-]|\.\d)", text)


def _cut(text: str, stops: Sequence[str]) -> str:
    ends = [text.find(s) for s in stops if text.find(s) >= 0]
    return text[: min(ends, default=len(text))]


def _month_found(item: Item, prefix: str, segment: str, columns: Sequence[str]) -> bool:
    others = [c for c in columns if c != item.column]
    at = segment.find(item.column)
    if at >= 0 and _value_found(item.values[-1], _cut(segment[at + len(item.column):], others)):
        return True
    named = [(prefix.rfind(c), c) for c in columns if prefix.rfind(c) >= 0]
    return bool(named) and max(named)[1] == item.column and _value_found(item.values[-1], _cut(segment, others))


def _band_found(item: Item, prefix: str, segment: str) -> bool:
    opening = _BAND_OPENS.search(segment)
    stretch = segment[: opening.start()] if opening else segment
    stated = _BAND_ANY.search(stretch)
    if stated:
        return _band_of(stated) == item.values[0]
    opened = list(_BAND_OPENS.finditer(prefix))
    return bool(opened) and _band_of(opened[-1]) == item.values[0]


def _lines(entries: Sequence[str], columns: Sequence[str]) -> list[str]:
    """Each line, behind the last line of its entry that ends in a colon."""
    out = []
    for entry in entries:
        heading = ""
        for raw in _normalise(entry, columns).splitlines():
            line = raw.strip()
            if not line:
                continue
            out.append(f"{heading} | {line}" if heading else line)
            if line.endswith(":"):
                heading = line
    return out


def _grid_covered(lines: Sequence[str], items: Sequence[Item]) -> set[int]:
    """Per-month items written as rows of a grid under a header line of months."""
    found: set[int] = set()
    columns = sorted({i.column for i in items if i.column})
    if not columns:
        return found
    header = None
    for line in lines:
        positions = [line.find(c) for c in columns]
        if all(p >= 0 for p in positions):
            header = [c for _, c in sorted(zip(positions, columns))]
            continue
        if header is None:
            continue
        for index, item in enumerate(items):
            if item.column is None:
                continue
            for key in item.keys:
                match = _key_pattern(key).search(line)
                if not match:
                    continue
                figures = _numbers(line[match.end():])
                if len(figures) >= len(header) and figures[header.index(item.column)] == item.values[-1]:
                    found.add(index)
    return found


def _found_on(item: Item, line: str, own: Sequence[re.Pattern[str]], other: Sequence[re.Pattern[str]],
              single: bool, columns: Sequence[str]) -> bool:
    is_band = len(item.values) == 1 and item.values[0] in ("A", "B", "C")
    for pattern in own:
        for match in pattern.finditer(line):
            ends = [m.start() for p in other for m in p.finditer(line, match.end())]
            prefix, segment = line[: match.start()], line[match.end(): min(ends, default=len(line))]
            if is_band:
                if _band_found(item, prefix, segment):
                    return True
            elif item.column:
                if _month_found(item, prefix, segment, columns):
                    return True
            elif all(_value_found(v, segment) for v in item.values):
                return True
    if single and not is_band:
        rest = line
        for pattern in own:
            rest = pattern.sub(" ", rest)
        if item.column:
            return _month_found(item, "", rest, columns)
        return all(_value_found(v, rest) for v in item.values)
    return False


def covered(items: Sequence[Item], entries: Sequence[str], names: Sequence[tuple[str, ...]] = ()) -> list[bool]:
    """Whether each item is found in the entries.

    ``names`` are the other names of the task (see ``names``); the keys of the
    items are always among them.
    """
    items = [Item(i.keys, tuple(_normalise(v) for v in i.values), i.column) for i in items]
    columns = sorted({i.column for i in items if i.column})
    groups = list(dict.fromkeys([tuple(i.keys) for i in items] + [tuple(n) for n in names]))
    patterns = {group: [_key_pattern(k) for k in group] for group in groups}
    lines = _lines(entries, columns)
    found = [False] * len(items)
    for line in lines:
        present = [g for g in groups if any(p.search(line) for p in patterns[g])]
        for index, item in enumerate(items):
            if found[index] or tuple(item.keys) not in present:
                continue
            own = patterns[tuple(item.keys)]
            other = [p for g in groups if g != tuple(item.keys) for p in patterns[g]]
            found[index] = _found_on(item, line, own, other, len(present) == 1, columns)
    for index in _grid_covered(lines, items):
        found[index] = True
    return found


def coverage(task: Task, world: World, entries: Sequence[str]) -> tuple[list[Item], list[bool]]:
    """The reference answer of ``task`` and whether each of its items is found."""
    items = reference_answer(task, world)
    return items, covered(items, entries, names(task, world))


def fidelity(task: Task, world: World, entries: Sequence[str]) -> float:
    """The share of the reference answer found in the writer's entries."""
    items, hits = coverage(task, world, entries)
    return sum(hits) / len(items)
