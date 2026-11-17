# SPDX-License-Identifier: Apache-2.0

"""The check of the writer-fidelity measure on entries it was not calibrated on.

The matcher of ``harness/fidelity.py`` was revised on the entries of the pilot
and the check of the amended briefs until it agreed with a reading of each by
hand (§5, *Writer fidelity*). Agreement on the entries it was fitted to says
little, so before collection it is checked on entries it has not seen: the
writers of the three models of collection, in the reasoning modes the pilot
selected, on the twenty tasks of the four substantive categories, at both
writer levels — 120 writers.

**Only the writer runs.** No consumer is started (``execute_writer``). The
substantive worlds have not been run before, and what a consumer does on them
is what collection measures; this check reads entries and nothing else.

The runs take replicate index 903, above every pilot index and the check of the
amended briefs, so no seed of the check recurs in collection. They use the
settings §5 fixes for collection. A run that ends in a backend failure is
executed again up to three times, as in the pilot.

What is read by hand is registered with the check: every writer that finished
with entries and scored below 1, and twelve that scored 1, drawn with a fixed
seed, so that a matcher finding answers that are not there is looked for too.
``render`` lays out both, with the items the matcher did not find.

No external dependencies: standard library only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from executor import (  # noqa: E402
    ChatClient,
    ModelError,
    OllamaClient,
    RunConfig,
    Status,
    World,
    derive_seed,
    execute_writer,
    run_id_for,
    write_run,
)
from fidelity import coverage  # noqa: E402
from pilot import BACKEND_RETRIES, BRIEF_CHECK_MODELS, FAMILIES, PILOT_CONDITIONS, group_label  # noqa: E402
from tasks import TASKS  # noqa: E402
from worlds import world_path  # noqa: E402

__all__ = [
    "WRITER_CHECK_REPLICATE",
    "WRITER_CHECK_SETTINGS",
    "WRITER_CHECK_TASKS",
    "FULL_SAMPLE_SIZE",
    "FULL_SAMPLE_SEED",
    "writer_check_plan",
    "run_writer_check",
    "load_check",
    "full_sample",
    "render",
]

#: Above both pilot indices (900, 901) and the check of the amended briefs (902).
WRITER_CHECK_REPLICATE = 903

#: The settings §5 fixes for collection.
WRITER_CHECK_SETTINGS: dict[str, int] = {
    "num_ctx": 8192,
    "num_predict": 8192,
    "max_steps": 28,
    "tool_result_limit": 4000,
}

#: The twenty tasks of the four substantive categories, in the order of the task set.
WRITER_CHECK_TASKS: tuple[str, ...] = tuple(t.task_id for t in TASKS if not t.category.is_control)

#: How many writers that scored 1 are read by hand, and the seed they are drawn with.
FULL_SAMPLE_SIZE = 12
FULL_SAMPLE_SEED = 20261117


def writer_check_plan() -> list[RunConfig]:
    """Every writer of the check, in the order it is executed."""
    plan = []
    for label, think in BRIEF_CHECK_MODELS:
        family = FAMILIES[label]
        for task_id in WRITER_CHECK_TASKS:
            for writer_level, consumer_level in PILOT_CONDITIONS:
                plan.append(
                    RunConfig(
                        model=family.model,
                        model_family=family.label,
                        task_id=task_id,
                        writer_level=writer_level,
                        consumer_level=consumer_level,
                        replicate=WRITER_CHECK_REPLICATE,
                        seed=derive_seed(family.label, task_id, writer_level, consumer_level,
                                         WRITER_CHECK_REPLICATE),
                        temperature=None,
                        think=think,
                        **WRITER_CHECK_SETTINGS,
                    )
                )
    return plan


def _emit(line: str) -> None:
    print(line, flush=True)


def run_writer_check(
    plan: Sequence[RunConfig],
    client: ChatClient,
    out: Path,
    *,
    load: Callable[[str], World] = lambda task_id: World.from_json(world_path(task_id)),
    emit: Callable[[str], None] = _emit,
    retries: int = BACKEND_RETRIES,
    timer: Callable[[], float] = time.monotonic,
    clock: Callable[[], str] | None = None,
) -> int:
    """Execute every writer of ``plan`` not yet stored in ``out``; returns how many it executed."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    executed = 0
    for position, config in enumerate(plan, start=1):
        run_id = run_id_for(config)
        if (out / f"{run_id}.run.json").exists():
            emit(f"[{position}/{len(plan)}] {run_id} already stored")
            continue
        world = load(config.task_id)
        failures: list[str] = []
        while True:
            started = timer()
            result = execute_writer(config, world, client, run_id=run_id, clock=clock)
            elapsed = timer() - started
            if result.status != Status.MODEL_ERROR or len(failures) >= retries:
                break
            failures.append(result.error or "")
        write_run(result, out)
        executed += 1
        line = f"[{position}/{len(plan)}] {run_id} {result.status} {elapsed:.0f} s"
        if result.entries:
            _, hits = coverage(config.task, world, [e.content for e in result.entries])
            line += f"  F={sum(hits) / len(hits):.2f}"
        if failures:
            line += f"  ({len(failures)} backend failure(s))"
        emit(line)
    return executed


def load_check(
    out: Path, *, load: Callable[[str], World] = lambda task_id: World.from_json(world_path(task_id))
) -> list[dict[str, Any]]:
    """One row per stored writer of the check, with its fidelity recomputed from the record."""
    by_id = {t.task_id: t for t in TASKS}
    rows = []
    for path in sorted(Path(out).glob(f"*.r{WRITER_CHECK_REPLICATE}.run.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        config = record["config"]
        entries = [e["content"] for e in record["entries"]]
        row = {
            "run_id": record["run_id"],
            "group": group_label(config["model_family"], config["think"]),
            "writer_level": config["writer_level"],
            "task_id": config["task_id"],
            "status": record["status"],
            "entries": entries,
            "fidelity": None,
            "missing": [],
        }
        if record["status"] == Status.WRITER_DONE:
            items, hits = coverage(by_id[config["task_id"]], load(config["task_id"]), entries)
            row["fidelity"] = sum(hits) / len(items)
            row["missing"] = [" ".join((i.keys[0], *i.values)) for i, h in zip(items, hits) if not h]
        rows.append(row)
    return rows


def full_sample(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """The writers that scored 1 and are read by hand all the same."""
    full = sorted(row["run_id"] for row in rows if row["fidelity"] == 1)
    return sorted(random.Random(FULL_SAMPLE_SEED).sample(full, min(FULL_SAMPLE_SIZE, len(full))))


def render(rows: Sequence[Mapping[str, Any]], *, planned: int) -> str:
    groups = sorted({row["group"] for row in rows})
    lines = [
        f"# Check of the fidelity measure — {len(rows)} of {planned} planned writers",
        "",
        "| Measured | " + " | ".join(groups) + " |",
        "|---|" + "---|" * len(groups),
    ]

    def cell(group: str, level: int | None, what: str) -> str:
        mine = [r for r in rows if r["group"] == group and (level is None or r["writer_level"] == level)]
        done = [r for r in mine if r["fidelity"] is not None]
        if what == "runs":
            return str(len(mine))
        if what == "done":
            return str(len(done))
        if what == "full":
            return str(sum(r["fidelity"] == 1 for r in done))
        return f"{sum(r['fidelity'] for r in done) / len(done):.2f}" if done else "–"

    for level in (None, 0, 1):
        scope = "all" if level is None else f"writer at L{level}"
        for what, label in (("runs", "Writers"), ("done", "Finished with entries"),
                            ("full", "Fidelity 1"), ("mean", "Mean fidelity")):
            lines.append(f"| {label}, {scope} | " + " | ".join(cell(g, level, what) for g in groups) + " |")
    endings: dict[str, int] = {}
    for row in rows:
        endings[row["status"]] = endings.get(row["status"], 0) + 1
    lines += ["", "Endings: " + ", ".join(f"{k} {v}" for k, v in sorted(endings.items()))]

    sample = set(full_sample(rows))
    to_read = [r for r in rows if (r["fidelity"] is not None and r["fidelity"] < 1) or r["run_id"] in sample]
    lines += ["", f"## To be read by hand: {len(to_read)} writers "
              f"({sum(r['run_id'] in sample for r in to_read)} of them scored 1, drawn with seed {FULL_SAMPLE_SEED})"]
    for row in to_read:
        lines += ["", f"### {row['run_id']} — F = {row['fidelity']:.2f}"]
        if row["missing"]:
            lines.append("Not found: " + "; ".join(row["missing"]))
        for number, entry in enumerate(row["entries"], start=1):
            lines += ["", f"[{number}]", "```", entry, "```"]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the check of the writer-fidelity measure (§5).")
    parser.add_argument("--out", type=Path, required=True, help="directory the check is stored in")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--table-only", action="store_true", help="rebuild the table from what is stored")
    args = parser.parse_args(argv)
    plan = writer_check_plan()
    if not args.table_only:
        client = OllamaClient(args.host)
        try:
            for model in sorted({c.model for c in plan}):
                client.identity(model)
            run_writer_check(plan, client, args.out)
        except (ModelError, OSError) as exc:
            print(f"check stopped: {exc}", file=sys.stderr)
            return 1
    rows = load_check(args.out)
    if not rows:
        print(f"no record of the check in {args.out}", file=sys.stderr)
        return 1
    table = render(rows, planned=len(plan))
    (Path(args.out) / "writer-check.md").write_text(table, encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
