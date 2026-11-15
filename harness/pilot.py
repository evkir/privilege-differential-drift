# SPDX-License-Identifier: Apache-2.0

"""Pilot runner: the operating characteristics of the setup, and nothing else.

Carries out the pilot registered in PREREGISTRATION.md §5 and computes its
table and the values the registered rules give. For each family it runs the
five control-category tasks under the 0->2 and 1->2 conditions, once in each
reasoning mode the family has, at the settings §5 fixes for the pilot.

**Each completed run is followed by a re-run of its consumer on the unaltered
entries.** This is the D7(iii) re-run with nothing removed. Attribution rests
on a consumer that repeats itself when its state is left as it was: if it does
not, a change after an ablation says nothing about the entries removed. The
re-run is compared with the original twice — on the consumer's sequence of
invocations, tool and arguments, and on every reply in full.

**A backend failure is not an attempt.** A run that ends in a model error is
executed again, up to a fixed number of times, and is never stored: §5 has it
say nothing about the agent. The errors are kept in the run's pilot record.

**Memory comes from the driver.** The share of the model held on the GPU is
the server's figure; memory in use is read from `nvidia-smi`, since the server
under-reports it for some models (§5).

**What is not computed.** MIR, AIR and any count of the consumer's gate
classifications are not read here, and no category or condition is compared on
them: the table is built without opening that part of a run's record.

A pilot can be interrupted and resumed. A run is finished when its pilot
record `<run_id>.pilot.json` exists; finished runs are skipped.

No external dependencies: standard library only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from executor import (  # noqa: E402
    ChatClient,
    ModelError,
    OllamaClient,
    RunConfig,
    RunResult,
    Status,
    World,
    _parse_call,
    derive_seed,
    execute_consumer,
    execute_run,
    run_id_for,
    write_run,
)
from gate import Classification  # noqa: E402
from journal import EventKind, canonical_json  # noqa: E402
from tasks import TASKS, Category  # noqa: E402
from worlds import world_path  # noqa: E402

__all__ = [
    "PILOT_REPLICATE_BASE",
    "PILOT_SETTINGS",
    "PILOT_CONDITIONS",
    "BACKEND_RETRIES",
    "BACKEND_FAILED",
    "Family",
    "FAMILIES",
    "REGISTERED_FAMILIES",
    "PilotError",
    "pilot_plan",
    "group_label",
    "invocation_sequence",
    "replies_of",
    "reply_sequence",
    "parse_nvidia_smi",
    "make_probe",
    "run_one",
    "run_pilot",
    "load_pilot",
    "summarise",
    "rule_values",
    "render_table",
]

#: Pilot runs take replicate indices from here, which collection never reaches.
#: The first reasoning mode of a family runs at this index, the second at the next.
PILOT_REPLICATE_BASE = 900

#: The settings §5 fixes for the pilot.
PILOT_SETTINGS: dict[str, int] = {
    "num_ctx": 8192,
    "num_predict": 4096,
    "max_steps": 32,
    "tool_result_limit": 4000,
}

#: Both writer levels, and the consumer of the decisive comparison.
PILOT_CONDITIONS: tuple[tuple[int, int], ...] = ((0, 2), (1, 2))

#: How many times a run that ended in a backend failure is executed again.
BACKEND_RETRIES = 3

#: Status of a pilot record whose run never got past the backend.
BACKEND_FAILED = "backend_failed"


@dataclass(frozen=True)
class Family:
    """A model family of the design, or a candidate replacement for one."""

    label: str
    model: str
    think_modes: tuple[bool | None, ...]


FAMILIES: dict[str, Family] = {
    family.label: family
    for family in (
        Family("llama3.1", "llama3.1:8b", (None,)),
        Family("mistral-nemo", "mistral-nemo:12b", (None,)),
        Family("qwen3", "qwen3:14b", (False, True)),
        Family("granite4.1", "granite4.1:8b", (None,)),
        Family("gemma4", "gemma4:12b", (False, True)),
    )
}

#: The three families of §5; the other two are the candidates of the replacement rule.
REGISTERED_FAMILIES: tuple[str, ...] = ("llama3.1", "mistral-nemo", "qwen3")


class PilotError(Exception):
    """The pilot cannot go on: a fault of the setup, not an outcome of a run."""


def pilot_plan(families: Sequence[str] = REGISTERED_FAMILIES) -> list[RunConfig]:
    """Every run of the pilot for ``families``, in the order it is executed."""
    control = sorted(t.task_id for t in TASKS if t.category is Category.CONTROL)
    plan: list[RunConfig] = []
    for label in families:
        if label not in FAMILIES:
            raise PilotError(f"unknown family {label!r}; known: {sorted(FAMILIES)}")
        family = FAMILIES[label]
        for index, think in enumerate(family.think_modes):
            replicate = PILOT_REPLICATE_BASE + index
            for task_id in control:
                for writer_level, consumer_level in PILOT_CONDITIONS:
                    plan.append(
                        RunConfig(
                            model=family.model,
                            model_family=family.label,
                            task_id=task_id,
                            writer_level=writer_level,
                            consumer_level=consumer_level,
                            replicate=replicate,
                            seed=derive_seed(
                                family.label, task_id, writer_level, consumer_level, replicate
                            ),
                            temperature=None,
                            think=think,
                            **PILOT_SETTINGS,
                        )
                    )
    return plan


def group_label(family: str, think: bool | None) -> str:
    """The column a run is reported in: its family, and its reasoning mode if it has one."""
    if think is None:
        return family
    return f"{family}, think {'on' if think else 'off'}"


# ---------------------------------------------------------------------------
# Comparing a consumer with its re-run
# ---------------------------------------------------------------------------


def invocation_sequence(result: RunResult, agent: str) -> list[tuple[str, str]]:
    """What ``agent`` attempted, in order: each tool with its arguments in canonical form."""
    return [
        (event.payload["tool"], canonical_json(event.payload["arguments"]))
        for event in result.journal.of_kind(EventKind.TOOL_INVOCATION)
        if event.payload["agent"] == agent
    ]


def replies_of(run_id: str, transcript: Mapping[str, Any], agent: str) -> list[str]:
    """Every reply of ``agent`` in a transcript, in order, each in canonical form.

    A reply is its text, its reasoning text and its calls, each call as the
    tool and the arguments the executor read from it. The raw call is not
    compared: a server may attach an identifier of its own to it, which the
    model did not generate and which differs between two identical replies.
    """
    messages = transcript.get(agent, {}).get("messages", [])
    replies = [m for m in messages if m.get("role") == "assistant"]
    calls = [c for c in transcript.get("calls", []) if c["agent"] == agent]
    if len(replies) != len(calls):
        raise PilotError(f"{run_id}: {len(replies)} replies of the {agent} but {len(calls)} calls")
    sequence = []
    for reply, call in zip(replies, calls):
        parsed = [_parse_call(raw) for raw in reply.get("tool_calls", [])]
        sequence.append(
            json.dumps(
                [
                    reply.get("content", ""),
                    call.get("thinking", ""),
                    [[tool, canonical_json(arguments)] for tool, arguments in parsed],
                ],
                sort_keys=True,
                default=str,
            )
        )
    return sequence


def reply_sequence(result: RunResult, agent: str) -> list[str]:
    """Every reply of ``agent`` in a run: text, reasoning text and parsed calls, in order."""
    return replies_of(result.run_id, result.transcript, agent)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

_MODEL_PROCESS_MARKS = ("ollama", "llama")


def parse_nvidia_smi(apps_csv: str, gpu_csv: str) -> dict[str, Any]:
    """Driver figures, in MiB, from two `nvidia-smi` queries in `csv,noheader,nounits` form.

    ``apps_csv`` is `--query-compute-apps=pid,process_name,used_memory` and
    ``gpu_csv`` is `--query-gpu=memory.used`. ``gpu_model_mib`` is the memory of
    the processes that serve the model; it is None when none is listed or a
    figure is not reported.
    """
    processes: list[dict[str, Any]] = []
    for line in apps_csv.splitlines():
        if not line.strip():
            continue
        pid, rest = line.split(",", 1)
        name, used = (part.strip() for part in rest.rsplit(",", 1))
        pid = pid.strip()
        processes.append({"pid": pid, "name": name, "used_mib": int(used) if used.isdigit() else None})
    serving = [p for p in processes if any(mark in p["name"].lower() for mark in _MODEL_PROCESS_MARKS)]
    model_mib = None
    if serving and all(p["used_mib"] is not None for p in serving):
        model_mib = sum(p["used_mib"] for p in serving)
    used = [int(line.strip()) for line in gpu_csv.splitlines() if line.strip().isdigit()]
    return {
        "gpu_processes": processes,
        "gpu_model_mib": model_mib,
        "gpu_used_mib": sum(used) if used else None,
    }


def _nvidia_smi(query: str) -> str:
    try:
        done = subprocess.run(
            ["nvidia-smi", query, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PilotError(f"nvidia-smi {query}: {exc}") from None
    return done.stdout


def make_probe(client: OllamaClient) -> Callable[[], dict[str, Any]]:
    """Reads, while a model is loaded, the server's GPU share and the driver's memory figures."""

    def probe() -> dict[str, Any]:
        figures = parse_nvidia_smi(
            _nvidia_smi("--query-compute-apps=pid,process_name,used_memory"),
            _nvidia_smi("--query-gpu=memory.used"),
        )
        try:
            loaded = client.loaded()
        except ModelError as exc:
            raise PilotError(f"the server did not report what it holds: {exc}") from None
        shares = [m["size_vram"] / m["size"] for m in loaded if m.get("size")]
        figures["server_gpu_share"] = min(shares) if shares else None
        return figures

    return probe


# ---------------------------------------------------------------------------
# One pilot run
# ---------------------------------------------------------------------------


def _attempt(
    execute: Callable[[], RunResult], timer: Callable[[], float], retries: int
) -> tuple[RunResult, float, list[str]]:
    """Execute until the backend holds, at most ``retries`` times more; errors are returned."""
    failures: list[str] = []
    while True:
        started = timer()
        result = execute()
        elapsed = timer() - started
        if result.status != Status.MODEL_ERROR:
            return result, elapsed, failures
        failures.append(result.error or "")
        if len(failures) > retries:
            return result, elapsed, failures


def _record_path(out: Path, run_id: str) -> Path:
    return Path(out) / f"{run_id}.pilot.json"


def run_one(
    config: RunConfig,
    world: World,
    client: ChatClient,
    out: Path,
    *,
    probe: Callable[[], dict[str, Any]],
    retries: int = BACKEND_RETRIES,
    timer: Callable[[], float] = time.monotonic,
    clock: Callable[[], str] | None = None,
) -> dict[str, Any]:
    """Execute one pilot run and, if it completed, the re-run of its consumer; store both.

    Nothing is written until everything has been executed, and the pilot
    record is written last, so a run with a pilot record is a finished run.
    """
    out = Path(out)
    run_id = run_id_for(config)
    null_id = f"{run_id}.null"
    for name in (f"{run_id}.run.json", f"{run_id}.journal.json", f"{null_id}.run.json",
                 f"{null_id}.journal.json", f"{run_id}.pilot.json"):
        if (out / name).exists():
            raise PilotError(f"{out / name} already exists; a stored run is not executed again")

    result, elapsed, failures = _attempt(
        lambda: execute_run(config, world, client, clock=clock), timer, retries
    )
    record: dict[str, Any] = {
        "run_id": run_id,
        "family": config.model_family,
        "model": config.model,
        "think": config.think,
        "task_id": config.task_id,
        "writer_level": config.writer_level,
        "consumer_level": config.consumer_level,
        "status": result.status,
        "backend_failures": failures,
        "elapsed_seconds": elapsed,
        "probe": None,
        "null": None,
    }
    null: RunResult | None = None
    if result.status == Status.MODEL_ERROR:
        record["status"] = BACKEND_FAILED
    else:
        record["probe"] = probe()
    if result.status == Status.COMPLETED:
        null, null_elapsed, null_failures = _attempt(
            lambda: execute_consumer(
                config, world, client, result.entries,
                run_id=null_id, ablated_from=run_id, clock=clock,
            ),
            timer, retries,
        )
        record["null"] = {
            "run_id": null_id,
            "status": null.status,
            "backend_failures": null_failures,
            "elapsed_seconds": null_elapsed,
            "same_invocations": None,
            "same_replies": None,
        }
        if null.status == Status.MODEL_ERROR:
            record["null"]["status"] = BACKEND_FAILED
            null = None
        else:
            record["null"]["same_invocations"] = (
                invocation_sequence(null, "consumer") == invocation_sequence(result, "consumer")
            )
            record["null"]["same_replies"] = (
                reply_sequence(null, "consumer") == reply_sequence(result, "consumer")
            )

    if record["status"] != BACKEND_FAILED:
        write_run(result, out)
    if null is not None:
        write_run(null, out)
    out.mkdir(parents=True, exist_ok=True)
    _record_path(out, run_id).write_text(
        json.dumps(record, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
    )
    return record


def _progress_line(position: int, total: int, record: Mapping[str, Any]) -> str:
    line = f"[{position}/{total}] {record['run_id']} {record['status']} {record['elapsed_seconds']:.0f} s"
    null = record["null"]
    if null is not None:
        if null["status"] == BACKEND_FAILED:
            line += "  re-run: backend failed"
        else:
            same = "same" if null["same_invocations"] else "DIFFERENT"
            text = "same" if null["same_replies"] else "different"
            line += f"  re-run: invocations {same}, replies {text}"
    if record["backend_failures"]:
        line += f"  ({len(record['backend_failures'])} backend failure(s))"
    return line


def _emit(line: str) -> None:
    """Print a progress line at once, also when the output is a pipe."""
    print(line, flush=True)


def run_pilot(
    plan: Sequence[RunConfig],
    client: ChatClient,
    out: Path,
    *,
    probe: Callable[[], dict[str, Any]],
    load: Callable[[str], World] = lambda task_id: World.from_json(world_path(task_id)),
    emit: Callable[[str], None] = _emit,
    retries: int = BACKEND_RETRIES,
    timer: Callable[[], float] = time.monotonic,
    clock: Callable[[], str] | None = None,
) -> int:
    """Execute every run of ``plan`` that has no pilot record yet; returns how many it executed."""
    out = Path(out)
    executed = 0
    for position, config in enumerate(plan, start=1):
        run_id = run_id_for(config)
        if _record_path(out, run_id).exists():
            emit(f"[{position}/{len(plan)}] {run_id} already stored")
            continue
        record = run_one(
            config, load(config.task_id), client, out,
            probe=probe, retries=retries, timer=timer, clock=clock,
        )
        executed += 1
        emit(_progress_line(position, len(plan), record))
    return executed


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def load_pilot(out: Path) -> list[dict[str, Any]]:
    """Every pilot record in ``out`` with the stored record of its run under ``"run"``.

    Only the parts of a run's record the table uses are kept, and the
    consumer's gate classifications are not among them.

    Whether a re-run repeats every reply is computed here from the two stored
    transcripts, not taken from the pilot record, so every stored pilot is
    reported under the comparison of `replies_of` whenever it was run.
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(out).glob("*.pilot.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["run"] = None
        if row["status"] != BACKEND_FAILED:
            stored = json.loads((path.parent / f"{row['run_id']}.run.json").read_text(encoding="utf-8"))
            metrics = stored["metrics"]
            row["run"] = {
                "num_ctx": stored["config"]["num_ctx"],
                "wall_seconds": metrics["wall_seconds"],
                "prompt_tokens_peak": metrics["prompt_tokens_peak"],
                "longest_reply": max(
                    (c.get("eval_count", 0) for c in stored["transcript"]["calls"]), default=0
                ),
                "writer_scope_breach": bool(
                    metrics["writer"]["classifications"].get(Classification.WRITER_SCOPE_BREACH.value)
                ),
                "agents": {
                    agent: {
                        key: metrics[agent][key]
                        for key in ("model_calls", "tool_invocations", "multi_call_replies",
                                    "eval_tokens", "eval_seconds", "load_seconds")
                    }
                    for agent in ("writer", "consumer")
                },
            }
            null = row["null"]
            if null is not None and null["status"] != BACKEND_FAILED:
                again = json.loads(
                    (path.parent / f"{null['run_id']}.run.json").read_text(encoding="utf-8")
                )
                null["same_replies"] = replies_of(
                    again["run_id"], again["transcript"], "consumer"
                ) == replies_of(stored["run_id"], stored["transcript"], "consumer")
        rows.append(row)
    return rows


def _share(count: int, total: int) -> Fraction | None:
    return Fraction(count, total) if total else None


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def summarise(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """The measurements of §5 for each family and reasoning mode, in order of first appearance."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(group_label(row["family"], row["think"]), []).append(row)
    summary: dict[str, dict[str, Any]] = {}
    for label, members in groups.items():
        runs = [m for m in members if m["run"] is not None]
        completed = [m for m in runs if m["status"] == Status.COMPLETED]
        nulls = [m["null"] for m in completed if m["null"]["status"] != BACKEND_FAILED]
        endings: dict[str, int] = {}
        for m in runs:
            endings[m["status"]] = endings.get(m["status"], 0) + 1
        agents = {
            agent: [m["run"]["agents"][agent] for m in runs] for agent in ("writer", "consumer")
        }
        both = agents["writer"] + agents["consumer"]
        speeds = []
        for m in runs:
            pair = m["run"]["agents"].values()
            seconds = sum(a["eval_seconds"] for a in pair)
            if seconds:
                speeds.append(sum(a["eval_tokens"] for a in pair) / seconds)
        shares = [m["probe"]["server_gpu_share"] for m in runs if m["probe"]["server_gpu_share"] is not None]
        model_mib = [m["probe"]["gpu_model_mib"] for m in runs if m["probe"]["gpu_model_mib"] is not None]
        used_mib = [m["probe"]["gpu_used_mib"] for m in runs if m["probe"]["gpu_used_mib"] is not None]
        summary[label] = {
            "family": members[0]["family"],
            "think": members[0]["think"],
            "runs": len(runs),
            "backend_failed": len(members) - len(runs),
            "backend_failures": sum(len(m["backend_failures"]) for m in members),
            "endings": endings,
            "collapsed": len(runs) - len(completed),
            "collapse_share": _share(len(runs) - len(completed), len(runs)),
            "wall_median": _median([m["run"]["wall_seconds"] for m in runs]),
            "wall_max": max((m["run"]["wall_seconds"] for m in runs), default=None),
            "elapsed_median": _median([m["elapsed_seconds"] for m in runs]),
            "tokens_per_second": _median(speeds),
            "calls": {
                agent: (_median([a["model_calls"] for a in agents[agent]]),
                        max((a["model_calls"] for a in agents[agent]), default=None))
                for agent in agents
            },
            "invocations": {
                agent: (_median([a["tool_invocations"] for a in agents[agent]]),
                        max((a["tool_invocations"] for a in agents[agent]), default=None))
                for agent in agents
            },
            "scope_breach_share": _share(sum(m["run"]["writer_scope_breach"] for m in runs), len(runs)),
            "multi_call_share": _share(
                sum(a["multi_call_replies"] for a in both), sum(a["model_calls"] for a in both)
            ),
            "consumer_idle_share": _share(
                sum(m["run"]["agents"]["consumer"]["tool_invocations"] == 0 for m in completed),
                len(completed),
            ),
            "prompt_peak": max((m["run"]["prompt_tokens_peak"] for m in runs), default=0),
            "num_ctx": max((m["run"]["num_ctx"] for m in runs), default=None),
            "longest_reply": max((m["run"]["longest_reply"] for m in runs), default=0),
            "most_calls_completed": max(
                (a["model_calls"] for m in completed for a in m["run"]["agents"].values()), default=None
            ),
            "load_median": _median([a["load_seconds"] for a in both if a["load_seconds"] is not None]),
            "server_gpu_share_min": min(shares) if shares else None,
            "gpu_model_mib_max": max(model_mib) if model_mib else None,
            "gpu_used_mib_max": max(used_mib) if used_mib else None,
            "null_runs": len(nulls),
            "null_backend_failed": len(completed) - len(nulls),
            "null_same_invocations": sum(bool(n["same_invocations"]) for n in nulls),
            "null_same_replies": sum(bool(n["same_replies"]) for n in nulls),
            "null_elapsed_median": _median([n["elapsed_seconds"] for n in nulls]),
        }
    return summary


def rule_values(summary: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """What the rules registered in §5 give on this pilot. Shares are compared exactly."""
    by_family: dict[str, dict[bool | None, Mapping[str, Any]]] = {}
    for group in summary.values():
        by_family.setdefault(group["family"], {})[group["think"]] = group

    reasoning: dict[str, str] = {}
    replacement: dict[str, str] = {}
    for family, modes in by_family.items():
        selected = modes.get(None)
        if False in modes and True in modes:
            off, on = modes[False]["collapse_share"], modes[True]["collapse_share"]
            if off is None or on is None:
                reasoning[family] = "undetermined: a mode has no stored run"
            elif on < off and off - on > Fraction(1, 10):
                reasoning[family], selected = "on", modes[True]
            else:
                reasoning[family], selected = "off", modes[False]
        elif False in modes or True in modes:
            reasoning[family] = "undetermined: only one mode was run"
        if selected is None or selected["collapse_share"] is None:
            replacement[family] = "undetermined"
        elif selected["collapse_share"] > Fraction(1, 2):
            replacement[family] = f"replace (collapse share {float(selected['collapse_share']):.2f})"
        else:
            replacement[family] = f"keep (collapse share {float(selected['collapse_share']):.2f})"

    groups = list(summary.values())
    longest = max((g["longest_reply"] for g in groups), default=0)
    reply_limit = 1024
    while reply_limit < 2 * longest:
        reply_limit *= 2
    fits = all(3 * g["prompt_peak"] <= 2 * g["num_ctx"] for g in groups if g["num_ctx"])
    most_calls = [g["most_calls_completed"] for g in groups if g["most_calls_completed"] is not None]
    null_runs = sum(g["null_runs"] for g in groups)
    null_differing = null_runs - sum(g["null_same_invocations"] for g in groups)
    return {
        "reasoning_mode": reasoning,
        "family_replacement": replacement,
        "reply_length_limit": reply_limit,
        "longest_reply": longest,
        "context_size": 8192 if fits else 16384,
        "context_size_conditional": not fits,
        "step_limit": max(12, 2 * max(most_calls)) if most_calls else None,
        "most_calls_completed": max(most_calls) if most_calls else None,
        "null_runs": null_runs,
        "null_differing_invocations": null_differing,
        "collection_blocked": null_differing > 0,
    }


def _fmt(value: Any, spec: str = "", unit: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, Fraction):
        return f"{float(value):.2f}"
    return f"{value:{spec}}{unit}"


def render_table(summary: Mapping[str, Mapping[str, Any]], *, planned: int | None = None) -> str:
    """The pilot table and the rule values, as Markdown."""
    labels = list(summary)
    stored = sum(g["runs"] + g["backend_failed"] for g in summary.values())
    endings = sorted({status for g in summary.values() for status in g["endings"]})

    def row(name: str, cell: Callable[[Mapping[str, Any]], str]) -> str:
        return f"| {name} | " + " | ".join(cell(summary[label]) for label in labels) + " |"

    def pair(key: str, agent: str) -> Callable[[Mapping[str, Any]], str]:
        return lambda g: f"{_fmt(g[key][agent][0], 'g')} / {_fmt(g[key][agent][1])}"

    lines = [
        "# Pilot table",
        "",
        f"Runs with a pilot record: {stored}" + (f" of {planned} planned." if planned is not None else "."),
        "",
        "| Measured | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(labels),
        row("Runs", lambda g: _fmt(g["runs"])),
        row("Wall time per run, s (median / max)",
            lambda g: f"{_fmt(g['wall_median'], '.1f')} / {_fmt(g['wall_max'], '.1f')}"),
        row("Time per run with unloading, s (median)", lambda g: _fmt(g["elapsed_median"], ".1f")),
        row("Generation speed, tokens/s (median)", lambda g: _fmt(g["tokens_per_second"], ".1f")),
        row("Model load, s (median)", lambda g: _fmt(g["load_median"], ".1f")),
        row("Writer model calls (median / max)", pair("calls", "writer")),
        row("Consumer model calls (median / max)", pair("calls", "consumer")),
        row("Writer tool invocations (median / max)", pair("invocations", "writer")),
        row("Consumer tool invocations (median / max)", pair("invocations", "consumer")),
    ]
    for status in endings:
        lines.append(row(f"Ended `{status}`", lambda g, s=status: _fmt(_share(g["endings"].get(s, 0), g["runs"]))))
    lines += [
        row("Collapse share", lambda g: _fmt(g["collapse_share"])),
        row("Runs with a writer scope breach", lambda g: _fmt(g["scope_breach_share"])),
        row("Replies holding more than one call", lambda g: _fmt(g["multi_call_share"])),
        row("Completed runs in which the consumer made no call", lambda g: _fmt(g["consumer_idle_share"])),
        row("Peak prompt, share of context",
            lambda g: _fmt(g["prompt_peak"] / g["num_ctx"] if g["num_ctx"] else None, ".2f")),
        row("Longest reply, tokens", lambda g: _fmt(g["longest_reply"])),
        row("Most model calls by an agent in a completed run", lambda g: _fmt(g["most_calls_completed"])),
        row("Share of the model on the GPU, server (min)", lambda g: _fmt(g["server_gpu_share_min"], ".2f")),
        row("Memory of the model process, driver, MiB (max)", lambda g: _fmt(g["gpu_model_mib_max"])),
        row("Memory in use on the GPU, driver, MiB (max)", lambda g: _fmt(g["gpu_used_mib_max"])),
        row("Consumer re-runs on unaltered entries", lambda g: _fmt(g["null_runs"])),
        row("Re-runs repeating the sequence of invocations",
            lambda g: _fmt(_share(g["null_same_invocations"], g["null_runs"]))),
        row("Re-runs repeating every reply in full",
            lambda g: _fmt(_share(g["null_same_replies"], g["null_runs"]))),
        row("Time per re-run, s (median)", lambda g: _fmt(g["null_elapsed_median"], ".1f")),
        row("Backend failures retried", lambda g: _fmt(g["backend_failures"])),
        row("Runs and re-runs abandoned on backend failure",
            lambda g: _fmt(g["backend_failed"] + g["null_backend_failed"])),
    ]

    rules = rule_values(summary)
    lines += ["", "## Values the registered rules give", ""]
    for family, mode in rules["reasoning_mode"].items():
        lines.append(f"- Reasoning mode of {family}: {mode}")
    for family, verdict in rules["family_replacement"].items():
        lines.append(f"- Family replacement, {family}: {verdict}")
    lines.append(
        f"- Reply-length limit: {rules['reply_length_limit']} tokens "
        f"(longest reply {rules['longest_reply']})"
    )
    context = f"- Context size: {rules['context_size']} tokens"
    if rules["context_size_conditional"]:
        context += ", provided every model is still held entirely on the GPU at that size (not measured here)"
    lines.append(context)
    if rules["step_limit"] is None:
        lines.append("- Step limit: undetermined, no run completed")
    else:
        lines.append(
            f"- Step limit: {rules['step_limit']} "
            f"(most model calls by an agent in a completed run: {rules['most_calls_completed']})"
        )
    verdict = "collection does NOT start" if rules["collection_blocked"] else "does not block collection"
    lines.append(
        f"- Reproducibility of the consumer: {rules['null_differing_invocations']} of "
        f"{rules['null_runs']} re-runs differ in their invocations — {verdict}"
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the pilot of §5 against a local Ollama server.")
    parser.add_argument("--out", type=Path, required=True, help="directory the pilot is stored in")
    parser.add_argument(
        "--families", nargs="+", default=list(REGISTERED_FAMILIES), choices=sorted(FAMILIES),
        help="families to pilot (default: the three registered ones)",
    )
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--table-only", action="store_true", help="rebuild the table from what is stored")
    args = parser.parse_args(argv)

    plan = pilot_plan(args.families)
    if not args.table_only:
        client = OllamaClient(args.host)
        probe = make_probe(client)
        try:
            for path in sorted({world_path(c.task_id) for c in plan}):
                World.from_json(path)
            for model in sorted({c.model for c in plan}):
                client.identity(model)
            probe()
            run_pilot(plan, client, args.out, probe=probe)
        except (PilotError, ModelError, OSError) as exc:
            print(f"pilot stopped: {exc}", file=sys.stderr)
            return 1
    rows = load_pilot(args.out)
    if not rows:
        print(f"no pilot record in {args.out}", file=sys.stderr)
        return 1
    present = {row["family"] for row in rows} | set(args.families)
    planned = len(pilot_plan([label for label in FAMILIES if label in present]))
    table = render_table(summarise(rows), planned=planned)
    (args.out / "pilot-table.md").write_text(table, encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
