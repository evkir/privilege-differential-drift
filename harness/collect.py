# SPDX-License-Identifier: Apache-2.0

"""Collection runner: every cell of the design to 135 completed runs, then ablation.

Carries out the collection registered in PREREGISTRATION.md §5. A cell is a
model family, a data category and a gap condition: 3 × 5 × 3 = 45 cells.

**Replicates of a cell rotate over its tasks.** The replicates of a cell are
executed in index order, and replicate ``r`` is run on the ``r mod 5``-th task
of the category in task-identifier order. The first 135 that complete are the
cell's runs; a cell stops at 270 attempts and is then reported as underfilled
if it holds fewer than 135 (§5, *Run outcomes*). The seed of a run is the one
``derive_seed`` gives for its family, task, levels and replicate.

**Order.** Families run in blocks, in the order of ``COLLECTION_FAMILIES``.
Within a block collection goes in rounds: in round ``k`` every cell still open
executes its replicate ``k``, in an order drawn from ``ORDER_SEED``, the family
and ``k``. The order is drawn over all fifteen cells of the family and the
closed ones are then left out, so it does not depend on when a cell closed,
and time on the machine is spread over conditions and categories instead of
running with them.

**Resumption.** A run is finished when its record ``<run_id>.collect.json``
exists; it is written last, and atomically. The state of every cell is rebuilt
from the stored records each time collection starts, so an interrupted
collection resumes at the run it would have executed next and executes the
same sequence it would have executed without the interruption. Run, journal
or temporary files without a record are moved to ``orphans/`` and their run is
executed again. A stored record with settings other than those of collection,
or out of the order above, stops collection.

**A backend failure is not an attempt.** A run whose backend still fails after
three further attempts stops collection with nothing stored for it; it is
executed again when collection resumes (§5, *Run outcomes*).

**Ablation follows each block.** When every cell of a family is closed, every
stored run of the family that ended in a status ``harness/ablation.py``
ablates and whose consumer made a candidate invocation is ablated into
``ablation/<run_id>/``. A run is ablated when ``<run_id>.ablation.json``
exists there. A re-run lost to the backend is stored as §5 registers it, and
the pass then stops, so that a failing server does not leave every later run
unresolved.

What is not done here: MIR, review under D6 and writer fidelity are computed
from the stored records afterwards. The progress line shows statuses only.

No external dependencies: standard library only.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from ablation import (  # noqa: E402
    ABLATED_STATUSES,
    _load,
    ablate,
    candidates_of,
    store,
)
from ablation import BACKEND_FAILED as ABLATION_BACKEND_FAILED  # noqa: E402
from allowlists import GAP_CONDITIONS  # noqa: E402
from budget import ATTEMPT_CAP  # noqa: E402
from budget import FAMILIES as BUDGET_FAMILIES  # noqa: E402
from executor import (  # noqa: E402
    ChatClient,
    OllamaClient,
    RunConfig,
    RunResult,
    Status,
    World,
    derive_seed,
    execute_run,
    run_id_for,
    write_run,
)
from pilot import FAMILIES, _attempt  # noqa: E402
from power import N_PER_CELL  # noqa: E402
from tasks import TASKS  # noqa: E402
from worlds import world_path  # noqa: E402

__all__ = [
    "COLLECTION_FAMILIES",
    "COLLECTION_SETTINGS",
    "CONDITIONS",
    "TARGET",
    "CAP",
    "ORDER_SEED",
    "RETRIES",
    "MARKER",
    "CollectionError",
    "Cell",
    "CellState",
    "cells_of",
    "mode_of",
    "config_for",
    "check_offset",
    "round_order",
    "next_batch",
    "read_markers",
    "recover",
    "load_states",
    "run_one",
    "family_eta",
    "collect_family",
    "collect",
    "ablation_targets",
    "ablation_pass",
    "make_ablate_one",
    "summarise",
    "write_summary",
    "render_status",
    "main",
]

#: The families of collection, in the order their blocks run, each in the
#: reasoning mode the pilot selected (§5, *Pilot result*).
COLLECTION_FAMILIES: tuple[tuple[str, bool | None], ...] = (
    ("granite4.1", None),
    ("qwen3", False),
    ("gemma4", False),
)

#: Context size, reply-length limit and step limit fixed by the pilot; the
#: tool-result limit is the pilot's.
COLLECTION_SETTINGS: dict[str, int] = {
    "num_ctx": 8192,
    "num_predict": 8192,
    "max_steps": 28,
    "tool_result_limit": 4000,
}

#: The three gap conditions, writer level and consumer level.
CONDITIONS: tuple[tuple[int, int], ...] = GAP_CONDITIONS

#: Completed runs a cell collects, and the attempts at which it stops.
TARGET = N_PER_CELL
CAP = ATTEMPT_CAP

#: Seed of the order of the cells within a round. Fixed before collection.
ORDER_SEED = 20261125

#: Further attempts at a run whose backend failed, as in the pilot.
RETRIES = 3

#: The record that marks a run as finished.
MARKER = ".collect.json"

ORPHANS = "orphans"
ABLATION_DIR = "ablation"
LOG_NAME = "collection.log"
SUMMARY_NAME = "collection.summary.json"
LOCK_NAME = ".collect.lock"

#: Replicate offsets other than 0 start here: above the pilot (900–901), the
#: brief check (902) and the writer check (903), and above every index of
#: collection (0–269). An offset is for a smoke run outside the data.
MIN_OFFSET = 1000

WRITER_COLLAPSE = frozenset({Status.WRITER_NO_STATE, Status.WRITER_STEP_LIMIT, Status.WRITER_TRUNCATED})
CONSUMER_COLLAPSE = frozenset({Status.CONSUMER_STEP_LIMIT, Status.CONSUMER_TRUNCATED})
ENDINGS = frozenset({Status.COMPLETED}) | WRITER_COLLAPSE | CONSUMER_COLLAPSE


class CollectionError(RuntimeError):
    """Collection cannot go on: a fault of the setup or of what is stored, not an outcome of a run."""


# ---------------------------------------------------------------------------
# The design
# ---------------------------------------------------------------------------


def _category_tasks() -> dict[str, tuple[str, ...]]:
    tasks: dict[str, list[str]] = {}
    for task in TASKS:
        tasks.setdefault(task.category.value, []).append(task.task_id)
    return {category: tuple(sorted(ids)) for category, ids in sorted(tasks.items())}


_CATEGORY_TASKS = _category_tasks()


@dataclass(frozen=True)
class Cell:
    family: str
    category: str
    writer_level: int
    consumer_level: int

    @property
    def key(self) -> str:
        return f"{self.family}.{self.category}.{self.writer_level}{self.consumer_level}"


def cells_of(family: str) -> list[Cell]:
    """The fifteen cells of a family, in a fixed canonical order."""
    if family not in dict(COLLECTION_FAMILIES):
        raise CollectionError(f"{family!r} is not a family of collection")
    return [
        Cell(family, category, writer_level, consumer_level)
        for category in _CATEGORY_TASKS
        for writer_level, consumer_level in CONDITIONS
    ]


def mode_of(family: str) -> bool | None:
    return dict(COLLECTION_FAMILIES)[family]


def config_for(cell: Cell, index: int, *, offset: int = 0) -> RunConfig:
    """The configuration of replicate ``index`` of a cell."""
    tasks = _CATEGORY_TASKS[cell.category]
    task_id = tasks[index % len(tasks)]
    replicate = offset + index
    return RunConfig(
        model=FAMILIES[cell.family].model,
        model_family=cell.family,
        task_id=task_id,
        writer_level=cell.writer_level,
        consumer_level=cell.consumer_level,
        replicate=replicate,
        seed=derive_seed(cell.family, task_id, cell.writer_level, cell.consumer_level, replicate),
        temperature=None,
        think=mode_of(cell.family),
        **COLLECTION_SETTINGS,
    )


def check_offset(offset: int) -> None:
    if offset != 0 and offset < MIN_OFFSET:
        raise CollectionError(
            f"a replicate offset other than 0 must be at least {MIN_OFFSET}, "
            "above every index of the pilot, the two checks and collection"
        )


# ---------------------------------------------------------------------------
# Cells and their order
# ---------------------------------------------------------------------------


@dataclass
class CellState:
    cell: Cell
    completed: int = 0
    attempts: int = 0
    statuses: list[str] = field(default_factory=list)

    @property
    def next_index(self) -> int:
        return self.attempts

    @property
    def done(self) -> bool:
        return self.completed >= TARGET or self.attempts >= CAP

    @property
    def underfilled(self) -> bool:
        return self.attempts >= CAP and self.completed < TARGET

    def add(self, status: str) -> None:
        self.statuses.append(status)
        self.attempts += 1
        if status == Status.COMPLETED:
            self.completed += 1


def round_order(family: str, round_index: int, cells: Sequence[Cell]) -> list[Cell]:
    """The cells in the order round ``round_index`` executes them."""
    material = f"pdd-order|{ORDER_SEED}|{family}|{round_index}"
    rng = random.Random(int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big"))
    ordered = sorted(cells, key=lambda cell: cell.key)
    rng.shuffle(ordered)
    return ordered


def next_batch(states: Sequence[CellState], family: str) -> list[Cell]:
    """The cells left to execute in the current round, in order; empty when the family is closed.

    Every open cell is at the same index except within a round that was
    interrupted, so the current round is the lowest index of an open cell.
    """
    open_states = [s for s in states if not s.done]
    if not open_states:
        return []
    current = min(s.next_index for s in open_states)
    due = {s.cell for s in open_states if s.next_index == current}
    return [cell for cell in round_order(family, current, [s.cell for s in states]) if cell in due]


# ---------------------------------------------------------------------------
# What is stored
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, data: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_markers(out: Path) -> dict[str, dict[str, Any]]:
    """Every finished run in ``out``, by run identifier."""
    out = Path(out)
    markers: dict[str, dict[str, Any]] = {}
    if not out.is_dir():
        return markers
    for path in sorted(out.glob(f"*{MARKER}")):
        run_id = path.name[: -len(MARKER)]
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("run_id") != run_id:
            raise CollectionError(f"{path} records the run {record.get('run_id')!r}")
        markers[run_id] = record
    return markers


def recover(out: Path, markers: Mapping[str, Any], *, stamp: str | None = None) -> list[Path]:
    """Move files of runs that were never finished to ``orphans/<stamp>/``."""
    out = Path(out)
    strays = sorted(out.glob("*.tmp"))
    for suffix in (".run.json", ".journal.json"):
        strays += [p for p in sorted(out.glob(f"*{suffix}")) if p.name[: -len(suffix)] not in markers]
    if not strays:
        return []
    target = out / ORPHANS / (stamp or _stamp())
    target.mkdir(parents=True, exist_ok=True)
    moved = []
    for path in strays:
        destination = target / path.name
        if destination.exists():
            raise CollectionError(f"{destination} already exists")
        shutil.move(str(path), str(destination))
        moved.append(destination)
    return moved


def _verify(record: Mapping[str, Any], config: RunConfig, offset: int) -> None:
    expected = {
        "model": config.model,
        "think": config.think,
        "seed": config.seed,
        "settings": COLLECTION_SETTINGS,
        "replicate_offset": offset,
    }
    for name, value in expected.items():
        if record.get(name) != value:
            raise CollectionError(
                f"{record['run_id']}: stored {name} {record.get(name)!r}, collection has {value!r}"
            )


def load_states(markers: Mapping[str, Mapping[str, Any]], offset: int = 0) -> dict[str, list[CellState]]:
    """The state of every cell, rebuilt from the stored records.

    A record that does not continue its cell in index order, or that lies
    beyond the point at which its cell closed, stops collection.
    """
    consumed: set[str] = set()
    states: dict[str, list[CellState]] = {}
    for family, _ in COLLECTION_FAMILIES:
        family_states = []
        for cell in cells_of(family):
            state = CellState(cell)
            while not state.done:
                config = config_for(cell, state.next_index, offset=offset)
                record = markers.get(run_id_for(config))
                if record is None:
                    break
                _verify(record, config, offset)
                state.add(record["status"])
                consumed.add(record["run_id"])
            family_states.append(state)
        states[family] = family_states
    stray = sorted(set(markers) - consumed)
    if stray:
        raise CollectionError(
            f"{len(stray)} stored runs are not in the order of collection, the first {stray[0]}"
        )
    return states


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------


def run_one(
    config: RunConfig,
    world: World,
    out: Path,
    *,
    cell: Cell,
    index: int,
    offset: int,
    execute: Callable[[RunConfig, World], RunResult],
    write: Callable[[RunResult, Path], Any] = write_run,
    retries: int = RETRIES,
    timer: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Execute one run and store it; its record is written last."""
    out = Path(out)
    run_id = run_id_for(config)
    for suffix in (".run.json", ".journal.json", MARKER):
        path = out / f"{run_id}{suffix}"
        if path.exists():
            raise CollectionError(f"{path} already exists; a stored run is not executed again")
    result, elapsed, failures = _attempt(lambda: execute(config, world), timer, retries)
    if result.status == Status.MODEL_ERROR:
        raise CollectionError(
            f"{run_id}: the backend failed {len(failures)} times (last: {failures[-1] or 'no message'}); "
            "nothing is stored, and the run is executed again when collection resumes"
        )
    if result.status not in ENDINGS:
        raise CollectionError(f"{run_id}: {result.status!r} is not an ending of a collection run")
    record = {
        "run_id": run_id,
        "cell": cell.key,
        "family": config.model_family,
        "model": config.model,
        "think": config.think,
        "category": cell.category,
        "task_id": config.task_id,
        "writer_level": config.writer_level,
        "consumer_level": config.consumer_level,
        "index": index,
        "replicate": config.replicate,
        "replicate_offset": offset,
        "seed": config.seed,
        "settings": COLLECTION_SETTINGS,
        "status": result.status,
        "entries": len(result.entries),
        "candidates": sorted({c.tool for c in candidates_of(result.journal)}),
        "backend_failures": failures,
        "elapsed_seconds": elapsed,
        "finished_at": _utc_now(),
    }
    write(result, out)
    _write_json(out / f"{run_id}{MARKER}", record)
    return record


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

_BUDGET = {figures.label: figures for figures in BUDGET_FAMILIES}


def _fmt_duration(seconds: float) -> str:
    hours, minutes = divmod(int(round(seconds / 60)), 60)
    return f"{hours}h{minutes:02d}m"


def family_eta(states: Sequence[CellState], elapsed: Sequence[float], model: str) -> float:
    """Seconds left for a family: the observed median time per attempt and completion share,
    or the budget's figures until enough attempts have been made."""
    figures = _BUDGET.get(model)
    if len(elapsed) >= 10 or (elapsed and figures is None):
        per_attempt = statistics.median(elapsed)
    elif figures is not None:
        per_attempt = figures.run_seconds
    else:
        per_attempt = 0.0
    attempts = sum(s.attempts for s in states)
    completed = sum(s.completed for s in states)
    if attempts >= 20:
        share = completed / attempts
    elif figures is not None:
        share = 1 - figures.collapse_share
    else:
        share = 1.0
    share = max(share, 0.01)
    remaining = 0
    for state in states:
        if not state.done:
            remaining += min(CAP - state.attempts, math.ceil((TARGET - state.completed) / share))
    return remaining * per_attempt


def _progress_line(
    family: str, state: CellState, record: Mapping[str, Any], states: Sequence[CellState],
    elapsed: Sequence[float], rest_seconds: float,
) -> str:
    completed = sum(s.completed for s in states)
    eta = family_eta(states, elapsed, FAMILIES[family].model)
    line = (
        f"[{family} {completed}/{TARGET * len(states)}] {record['run_id']} {record['status']} "
        f"{record['elapsed_seconds']:.0f} s | cell {state.completed}/{state.attempts} | "
        f"ETA {_fmt_duration(eta)}"
    )
    if rest_seconds:
        line += f" (all {_fmt_duration(eta + rest_seconds)})"
    if record["backend_failures"]:
        line += f" | backend retries {len(record['backend_failures'])}"
    return line


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def _load_world(task_id: str) -> World:
    return World.from_json(world_path(task_id))


def collect_family(
    family: str,
    out: Path,
    *,
    execute: Callable[[RunConfig, World], RunResult],
    load: Callable[[str], World] = _load_world,
    emit: Callable[[str], None] = print,
    write: Callable[[RunResult, Path], Any] = write_run,
    retries: int = RETRIES,
    timer: Callable[[], float] = time.monotonic,
    offset: int = 0,
    max_runs: int | None = None,
    rest_seconds: float = 0.0,
) -> tuple[int, bool]:
    """Collect one family's block; returns the runs executed and whether the block is closed."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    markers = read_markers(out)
    for path in recover(out, markers):
        emit(f"orphan moved: {path}")
    states = load_states(markers, offset)[family]
    by_cell = {state.cell: state for state in states}
    elapsed = [r["elapsed_seconds"] for r in markers.values() if r["family"] == family]
    emit(
        f"== {family} ({FAMILIES[family].model}, think={mode_of(family)}): "
        f"{sum(s.completed for s in states)}/{TARGET * len(states)} completed, "
        f"{sum(s.attempts for s in states)} attempts, "
        f"ETA {_fmt_duration(family_eta(states, elapsed, FAMILIES[family].model))}"
    )
    worlds: dict[str, World] = {}
    executed = 0
    while True:
        batch = next_batch(states, family)
        if not batch:
            return executed, True
        for cell in batch:
            if max_runs is not None and executed >= max_runs:
                return executed, False
            state = by_cell[cell]
            config = config_for(cell, state.next_index, offset=offset)
            if config.task_id not in worlds:
                worlds[config.task_id] = load(config.task_id)
            record = run_one(
                config, worlds[config.task_id], out, cell=cell, index=state.next_index,
                offset=offset, execute=execute, write=write, retries=retries, timer=timer,
            )
            state.add(record["status"])
            elapsed.append(record["elapsed_seconds"])
            executed += 1
            emit(_progress_line(family, state, record, states, elapsed, rest_seconds))


def collect(
    out: Path,
    families: Sequence[str],
    *,
    execute: Callable[[RunConfig, World], RunResult],
    ablate_one: Callable[[Path, Path], Mapping[str, Any]],
    load: Callable[[str], World] = _load_world,
    emit: Callable[[str], None] = print,
    ablate: bool = True,
    write: Callable[[RunResult, Path], Any] = write_run,
    retries: int = RETRIES,
    timer: Callable[[], float] = time.monotonic,
    offset: int = 0,
    max_runs: int | None = None,
) -> int:
    """Collect ``families`` block by block, ablating each block once it is closed."""
    check_offset(offset)
    order = [label for label, _ in COLLECTION_FAMILIES if label in families]
    unknown = sorted(set(families) - set(order))
    if unknown:
        raise CollectionError(f"not families of collection: {unknown}")
    out = Path(out)
    executed = 0
    for position, family in enumerate(order):
        states = load_states(read_markers(out), offset)
        rest = sum(
            family_eta(states[later], [], FAMILIES[later].model) for later in order[position + 1:]
        )
        left = None if max_runs is None else max_runs - executed
        count, closed = collect_family(
            family, out, execute=execute, load=load, emit=emit, write=write, retries=retries,
            timer=timer, offset=offset, max_runs=left, rest_seconds=rest,
        )
        executed += count
        write_summary(out, offset)
        if not closed:
            emit(f"stopped after {executed} runs; run the same command to resume")
            return executed
        if ablate:
            ablation_pass(family, out, ablate_one=ablate_one, emit=emit)
    return executed


# ---------------------------------------------------------------------------
# Ablation pass
# ---------------------------------------------------------------------------


def ablation_targets(markers: Mapping[str, Mapping[str, Any]], family: str) -> list[str]:
    """The stored runs of a family that are ablated: an ablated status and a candidate."""
    return sorted(
        run_id for run_id, record in markers.items()
        if record["family"] == family and record["status"] in ABLATED_STATUSES and record["candidates"]
    )


def ablation_pass(
    family: str,
    out: Path,
    *,
    ablate_one: Callable[[Path, Path], Mapping[str, Any]],
    emit: Callable[[str], None] = print,
    stamp: str | None = None,
) -> int:
    """Ablate every target of a family not yet ablated; returns how many were ablated."""
    out = Path(out)
    stamp = stamp or _stamp()
    targets = ablation_targets(read_markers(out), family)
    done = 0
    for position, run_id in enumerate(targets, 1):
        directory = out / ABLATION_DIR / run_id
        if (directory / f"{run_id}.ablation.json").exists():
            continue
        if directory.exists():
            destination = out / ORPHANS / stamp / f"ablation-{run_id}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(directory), str(destination))
            emit(f"orphan moved: {destination}")
        summary = ablate_one(out / f"{run_id}.run.json", directory)
        done += 1
        verdicts = ", ".join(f"{tool}: {verdict}" for tool, verdict in sorted(summary.get("verdicts", {}).items()))
        emit(
            f"[ablation {family} {position}/{len(targets)}] {run_id} "
            f"{summary.get('entries')} entries, {summary.get('reruns')} re-runs"
            + (f" | {verdicts}" if verdicts else "")
        )
        if summary.get("backend_failed"):
            raise CollectionError(
                f"{run_id}: a re-run still failed at the backend; the ablation is stored as §5 "
                "registers it, and the pass stops"
            )
    return done


def make_ablate_one(
    client: ChatClient,
    load: Callable[[str], World] = _load_world,
    *,
    timer: Callable[[], float] = time.monotonic,
    clock: Callable[[], str] | None = None,
) -> Callable[[Path, Path], dict[str, Any]]:
    """Ablate one stored run with ``harness/ablation.py`` and store it in a directory of its own."""

    def ablate_one(record_path: Path, directory: Path) -> dict[str, Any]:
        config, run_id, status, entries, journal = _load(record_path)
        ablation, reruns = ablate(
            config, load(config.task_id), client, run_id=run_id, status=status,
            entries=entries, journal=journal, timer=timer, clock=clock,
        )
        store(ablation, reruns, directory)
        return {
            "run_id": run_id,
            "entries": len(entries),
            "reruns": len(ablation.reruns),
            "backend_failed": any(r["status"] == ABLATION_BACKEND_FAILED for r in ablation.reruns),
            "verdicts": {
                c.tool: c.unresolved or ("confirmed" if c.confirmed else "not_caused")
                for c in ablation.candidates
            },
        }

    return ablate_one


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarise(states: Mapping[str, Sequence[CellState]]) -> dict[str, Any]:
    families: dict[str, Any] = {}
    for family, family_states in states.items():
        cells = {}
        for state in family_states:
            counts = Counter(state.statuses)
            cells[state.cell.key] = {
                "completed": state.completed,
                "attempts": state.attempts,
                "closed": state.done,
                "underfilled": state.underfilled,
                "writer_collapsed": sum(counts[s] for s in WRITER_COLLAPSE),
                "consumer_collapsed": sum(counts[s] for s in CONSUMER_COLLAPSE),
            }
        families[family] = {
            "model": FAMILIES[family].model,
            "think": mode_of(family),
            "closed_cells": sum(c["closed"] for c in cells.values()),
            "completed": sum(c["completed"] for c in cells.values()),
            "attempts": sum(c["attempts"] for c in cells.values()),
            "writer_collapsed": sum(c["writer_collapsed"] for c in cells.values()),
            "consumer_collapsed": sum(c["consumer_collapsed"] for c in cells.values()),
            "underfilled": sorted(key for key, c in cells.items() if c["underfilled"]),
            "cells": cells,
        }
    return {"target": TARGET, "cap": CAP, "families": families}


def write_summary(out: Path, offset: int = 0) -> Path:
    path = Path(out) / SUMMARY_NAME
    _write_json(path, summarise(load_states(read_markers(out), offset)))
    return path


def render_status(out: Path, offset: int = 0) -> str:
    out = Path(out)
    markers = read_markers(out)
    summary = summarise(load_states(markers, offset))
    ablated = Counter(
        markers[path.parent.name]["family"]
        for path in out.glob(f"{ABLATION_DIR}/*/*.ablation.json")
        if path.parent.name in markers
    )
    lines = [f"{out}: target {TARGET} per cell, cap {CAP}"]
    for family, data in summary["families"].items():
        cells = len(data["cells"])
        lines.append(
            f"{family} ({data['model']}): cells closed {data['closed_cells']}/{cells}, "
            f"completed {data['completed']}/{TARGET * cells}, attempts {data['attempts']}, "
            f"writer collapse {data['writer_collapsed']}, consumer collapse {data['consumer_collapsed']}, "
            f"underfilled {', '.join(data['underfilled']) or 'none'}, "
            f"ablated {ablated[family]}/{len(ablation_targets(markers, family))}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _lock(out: Path):
    handle = (out / LOCK_NAME).open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise CollectionError(f"another collection is running in {out}") from None
    return handle


def main(argv: Sequence[str] | None = None) -> int:
    labels = [label for label, _ in COLLECTION_FAMILIES]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="directory collection is stored in")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--families", nargs="+", choices=labels, default=labels)
    parser.add_argument("--max-runs", type=int, default=None, help="stop after this many runs")
    parser.add_argument("--no-ablate", action="store_true", help="do not ablate a closed block")
    parser.add_argument("--ablate-only", action="store_true", help="only ablate what is stored")
    parser.add_argument("--status", action="store_true", help="show what is stored and exit")
    parser.add_argument(
        "--replicate-offset", type=int, default=0,
        help=f"0 for collection; at least {MIN_OFFSET} for a smoke run outside the data",
    )
    args = parser.parse_args(argv)
    try:
        check_offset(args.replicate_offset)
    except CollectionError as exc:
        parser.error(str(exc))
    out = args.out.expanduser()
    families = [label for label in labels if label in args.families]
    if args.status:
        print(render_status(out, args.replicate_offset))
        return 0

    out.mkdir(parents=True, exist_ok=True)
    log = out / LOG_NAME

    def emit(line: str) -> None:
        print(line, flush=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    client = OllamaClient(args.host)
    ablate_one = make_ablate_one(client)
    try:
        lock = _lock(out)
    except CollectionError as exc:
        print(exc, file=sys.stderr)
        return 2
    try:
        if args.ablate_only:
            for family in families:
                ablation_pass(family, out, ablate_one=ablate_one, emit=emit)
        else:
            collect(
                out, families, execute=lambda config, world: execute_run(config, world, client),
                ablate_one=ablate_one, emit=emit, ablate=not args.no_ablate,
                offset=args.replicate_offset, max_runs=args.max_runs,
            )
    except CollectionError as exc:
        emit(f"stopped: {exc}")
        return 2
    except KeyboardInterrupt:
        emit("interrupted; run the same command to resume")
        return 130
    finally:
        lock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
