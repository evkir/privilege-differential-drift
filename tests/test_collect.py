# SPDX-License-Identifier: Apache-2.0

"""Tests for harness/collect.py.

Written by Claude (Opus 5.5, Anthropic) at the author's request.

The first group is the design itself: the cells, the rotation of a cell's
replicates over its tasks, the seeds, the settings and the order of a round.

The second is the replacement rule of §5 as the runner carries it out: a cell
closes at its target of completed runs or at its cap of attempts, a collapse
is an attempt, a backend failure is not.

The third is what makes a collection of some seventy hours safe to interrupt:
resuming executes exactly the sequence an uninterrupted collection executes,
files of an unfinished run are set aside and the run executed again, and a
stored record that does not belong where it is stops the runner.

The fourth is the ablation pass that follows each block. The last test runs
one block through the real executor, store and ablation with a model that
answers from the conversation it is shown.

Most tests stand a scripted executor in for ``execute_run`` and shrink the
target and cap, so that a block is 15 cells of a few runs.
"""

from __future__ import annotations

import itertools
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "analysis"))

import collect  # noqa: E402
from ablation import ABLATED_STATUSES  # noqa: E402
from ablation import candidates_of as real_candidates_of  # noqa: E402
from collect import CollectionError  # noqa: E402
from executor import (  # noqa: E402
    CONSUMER_SYSTEM_PROMPT,
    ChatResponse,
    Status,
    derive_seed,
    execute_run,
    run_id_for,
)

FAMILY = "granite4.1"


@dataclass
class FakeResult:
    run_id: str
    status: str
    entries: tuple = ("an entry",)
    journal: tuple = ()
    error: str | None = None


class Executor:
    """Stands in for ``execute_run``: a status per run from ``policy``, candidates from ``tools``."""

    def __init__(self, policy=None, tools=None):
        self.policy = policy or (lambda config, attempt: Status.COMPLETED)
        self.tools = tools or (lambda config: ())
        self.calls: list[str] = []
        self.tries: Counter = Counter()

    def __call__(self, config, world):
        run_id = run_id_for(config)
        attempt = self.tries[run_id]
        self.tries[run_id] += 1
        self.calls.append(run_id)
        status = self.policy(config, attempt)
        return FakeResult(
            run_id, status,
            journal=tuple(SimpleNamespace(tool=t) for t in self.tools(config)),
            error="refused" if status == Status.MODEL_ERROR else None,
        )


def fake_write(result, out):
    out.mkdir(parents=True, exist_ok=True)
    for kind in ("run", "journal"):
        path = out / f"{result.run_id}.{kind}.json"
        if path.exists():
            raise FileExistsError(path)
        path.write_text("{}", encoding="utf-8")


def ticker():
    count = itertools.count()
    return lambda: float(next(count))


def no_ablation(record_path, directory):
    pytest.fail(f"{record_path.name} ablated where no ablation was expected")


def run(out, executor, *, max_runs=None, ablate_one=no_ablation, ablate=False, lines=None):
    return collect.collect(
        out, [FAMILY], execute=executor, ablate_one=ablate_one, load=lambda task_id: task_id,
        emit=(lines if lines is not None else []).append, ablate=ablate, write=fake_write,
        timer=ticker(), max_runs=max_runs,
    )


def replicate(run_id):
    return int(run_id.rsplit(".r", 1)[1])


def first_of(family, index):
    """Run identifiers of round ``index`` in the order the round executes them."""
    cells = collect.cells_of(family)
    return [run_id_for(collect.config_for(c, index)) for c in collect.round_order(family, index, cells)]


@pytest.fixture(autouse=True)
def plain_candidates(monkeypatch):
    """The scripted journals are tuples of candidates already."""
    monkeypatch.setattr(collect, "candidates_of", lambda journal: journal)


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(collect, "TARGET", 3)
    monkeypatch.setattr(collect, "CAP", 6)


# -- the design ---------------------------------------------------------------


def test_target_cap_and_settings_are_the_registered_ones():
    assert collect.TARGET == 135 and collect.CAP == 270 == 2 * collect.TARGET
    assert [f for f, _ in collect.COLLECTION_FAMILIES] == ["granite4.1", "qwen3", "gemma4"]
    assert dict(collect.COLLECTION_FAMILIES) == {"granite4.1": None, "qwen3": False, "gemma4": False}
    assert collect.COLLECTION_SETTINGS == {
        "num_ctx": 8192, "num_predict": 8192, "max_steps": 28, "tool_result_limit": 4000,
    }
    assert collect.CONDITIONS == ((0, 1), (0, 2), (1, 2))


def test_the_design_is_45_cells_of_five_tasks():
    cells = [c for f, _ in collect.COLLECTION_FAMILIES for c in collect.cells_of(f)]
    assert len(cells) == len({c.key for c in cells}) == 45
    assert len(collect._CATEGORY_TASKS) == 5
    assert all(len(tasks) == 5 for tasks in collect._CATEGORY_TASKS.values())
    assert sum(len(t) for t in collect._CATEGORY_TASKS.values()) == 25
    with pytest.raises(CollectionError):
        collect.cells_of("llama3.1")


def test_replicates_rotate_over_the_tasks_of_the_category():
    cell = collect.cells_of("qwen3")[4]
    tasks = collect._CATEGORY_TASKS[cell.category]
    configs = [collect.config_for(cell, i) for i in range(12)]
    assert [c.task_id for c in configs] == [tasks[i % 5] for i in range(12)]
    assert [c.replicate for c in configs] == list(range(12))
    for c in configs:
        assert c.seed == derive_seed("qwen3", c.task_id, cell.writer_level, cell.consumer_level, c.replicate)
        assert (c.model, c.think, c.num_predict, c.max_steps) == ("qwen3:14b", False, 8192, 28)
        assert c.temperature is None


def test_no_two_runs_of_collection_share_an_identifier_or_a_seed():
    configs = [
        collect.config_for(cell, i)
        for family, _ in collect.COLLECTION_FAMILIES for cell in collect.cells_of(family)
        for i in range(collect.CAP)
    ]
    assert len({run_id_for(c) for c in configs}) == len(configs) == 45 * 270
    assert len({c.seed for c in configs}) == len(configs)
    assert max(c.replicate for c in configs) < 900


def test_an_offset_moves_replicates_above_every_index_in_use():
    cell = collect.cells_of(FAMILY)[0]
    shifted, plain = collect.config_for(cell, 3, offset=1000), collect.config_for(cell, 3)
    assert shifted.replicate == 1003 and shifted.task_id == plain.task_id
    collect.check_offset(0)
    collect.check_offset(1000)
    for offset in (1, 269, 900, 903, 999):
        with pytest.raises(CollectionError):
            collect.check_offset(offset)


def test_a_rounds_order_is_a_seeded_permutation_independent_of_input_order():
    cells = collect.cells_of(FAMILY)
    order = collect.round_order(FAMILY, 0, cells)
    assert sorted(c.key for c in order) == sorted(c.key for c in cells)
    assert order == collect.round_order(FAMILY, 0, list(reversed(cells)))
    assert len({tuple(c.key for c in collect.round_order(FAMILY, k, cells)) for k in range(6)}) > 1
    assert [c.key for c in order] != [c.key for c in sorted(cells, key=lambda c: c.key)]


# -- replacement --------------------------------------------------------------


def test_a_block_runs_round_by_round_in_the_drawn_order(tmp_path, small):
    executor = Executor()
    assert run(tmp_path, executor) == 45
    replicates = [replicate(r) for r in executor.calls]
    assert replicates == sorted(replicates)
    assert Counter(replicates) == {0: 15, 1: 15, 2: 15}
    assert executor.calls[:15] == first_of(FAMILY, 0)
    assert executor.calls[15:30] == first_of(FAMILY, 1)
    assert len(list(tmp_path.glob(f"*{collect.MARKER}"))) == 45
    summary = json.loads((tmp_path / collect.SUMMARY_NAME).read_text())
    assert summary["families"][FAMILY]["closed_cells"] == 15
    assert summary["families"][FAMILY]["completed"] == 45


def test_a_collapse_is_replaced_and_a_cell_stops_at_its_cap(tmp_path, small):
    x, y = collect.cells_of(FAMILY)[:2]
    x_collapses = {run_id_for(collect.config_for(x, i)) for i in (0, 2)}
    y_runs = {run_id_for(collect.config_for(y, i)) for i in range(6)}

    def policy(config, attempt):
        run_id = run_id_for(config)
        if run_id in x_collapses:
            return Status.WRITER_NO_STATE
        if run_id in y_runs:
            return Status.CONSUMER_STEP_LIMIT
        return Status.COMPLETED

    run(tmp_path, Executor(policy))
    states = {s.cell: s for s in collect.load_states(collect.read_markers(tmp_path))[FAMILY]}
    assert (states[x].attempts, states[x].completed, states[x].underfilled) == (5, 3, False)
    assert (states[y].attempts, states[y].completed, states[y].underfilled) == (6, 0, True)
    assert all((s.attempts, s.completed) == (3, 3) for c, s in states.items() if c not in (x, y))
    family = json.loads((tmp_path / collect.SUMMARY_NAME).read_text())["families"][FAMILY]
    assert family["underfilled"] == [y.key]
    assert (family["writer_collapsed"], family["consumer_collapsed"]) == (2, 6)


def test_a_transient_backend_failure_is_not_an_attempt(tmp_path, small):
    flaky = first_of(FAMILY, 0)[3]
    policy = lambda config, attempt: (  # noqa: E731
        Status.MODEL_ERROR if run_id_for(config) == flaky and attempt == 0 else Status.COMPLETED
    )
    executor = Executor(policy)
    run(tmp_path, executor)
    assert executor.calls.count(flaky) == 2
    record = collect.read_markers(tmp_path)[flaky]
    assert record["backend_failures"] == ["refused"] and record["status"] == Status.COMPLETED
    assert all(s.attempts == 3 for s in collect.load_states(collect.read_markers(tmp_path))[FAMILY])


def test_a_persistent_backend_failure_stops_with_nothing_stored(tmp_path, small):
    broken = first_of(FAMILY, 0)[2]
    policy = lambda config, attempt: (  # noqa: E731
        Status.MODEL_ERROR if run_id_for(config) == broken else Status.COMPLETED
    )
    executor = Executor(policy)
    with pytest.raises(CollectionError, match="executed again"):
        run(tmp_path, executor)
    assert executor.calls.count(broken) == collect.RETRIES + 1
    assert not list(tmp_path.glob(f"{broken}.*"))
    healthy = Executor()
    run(tmp_path, healthy)
    assert healthy.calls[0] == broken


def test_an_unknown_ending_is_not_stored(tmp_path, small):
    with pytest.raises(CollectionError, match="not an ending"):
        run(tmp_path, Executor(lambda config, attempt: Status.WRITER_DONE))
    assert not list(tmp_path.glob("*.json"))


# -- resumption ---------------------------------------------------------------


def _mixed(config, attempt):
    return Status.WRITER_NO_STATE if config.seed % 3 == 0 else Status.COMPLETED


@pytest.mark.parametrize("cut", [1, 7, 15, 16, 31])
def test_resuming_executes_the_sequence_of_an_uninterrupted_collection(tmp_path, small, cut):
    whole = Executor(_mixed)
    run(tmp_path / "whole", whole)
    first, second = Executor(_mixed), Executor(_mixed)
    lines = []
    assert run(tmp_path / "cut", first, max_runs=cut, lines=lines) == cut
    assert any("run the same command to resume" in line for line in lines)
    run(tmp_path / "cut", second)
    assert first.calls + second.calls == whole.calls
    assert set(collect.read_markers(tmp_path / "cut")) == set(collect.read_markers(tmp_path / "whole"))


def test_files_of_an_unfinished_run_are_moved_and_the_run_executed_again(tmp_path, small):
    run(tmp_path, Executor(), max_runs=2)
    unfinished = first_of(FAMILY, 0)[2]
    (tmp_path / f"{unfinished}.run.json").write_text("{}")
    (tmp_path / f"{unfinished}.collect.json.tmp").write_text("{")
    lines = []
    executor = Executor()
    run(tmp_path, executor, lines=lines)
    assert executor.calls[0] == unfinished
    assert len(list(tmp_path.glob(f"{collect.ORPHANS}/*/{unfinished}.run.json"))) == 1
    assert len(list(tmp_path.glob(f"{collect.ORPHANS}/*/{unfinished}.collect.json.tmp"))) == 1
    assert sum("orphan moved" in line for line in lines) == 2


def test_a_record_with_other_settings_stops_collection(tmp_path, small):
    run(tmp_path, Executor(), max_runs=1)
    (path,) = tmp_path.glob(f"*{collect.MARKER}")
    record = json.loads(path.read_text())
    record["settings"]["max_steps"] = 32
    path.write_text(json.dumps(record))
    with pytest.raises(CollectionError, match="settings"):
        run(tmp_path, Executor())


def test_a_record_out_of_order_stops_collection(tmp_path, small):
    executor = Executor()
    run(tmp_path, executor, max_runs=16)
    markers = collect.read_markers(tmp_path)
    second = markers[executor.calls[15]]
    assert second["index"] == 1
    first = next(r for r, m in markers.items() if m["cell"] == second["cell"] and m["index"] == 0)
    (tmp_path / f"{first}{collect.MARKER}").unlink()
    with pytest.raises(CollectionError, match="not in the order"):
        run(tmp_path, Executor())


def test_a_record_naming_another_run_is_refused(tmp_path, small):
    run(tmp_path, Executor(), max_runs=1)
    (path,) = tmp_path.glob(f"*{collect.MARKER}")
    path.rename(tmp_path / f"elsewhere{collect.MARKER}")
    with pytest.raises(CollectionError, match="records the run"):
        collect.read_markers(tmp_path)


def test_a_stored_run_is_not_executed_again(tmp_path):
    cell = collect.cells_of(FAMILY)[0]
    config = collect.config_for(cell, 0)
    (tmp_path / f"{run_id_for(config)}.journal.json").write_text("{}")
    with pytest.raises(CollectionError, match="already exists"):
        collect.run_one(config, None, tmp_path, cell=cell, index=0, offset=0,
                        execute=Executor(), write=fake_write, timer=ticker())


def test_records_of_a_smoke_offset_do_not_mix_with_collection(tmp_path, small):
    collect.collect(
        tmp_path, [FAMILY], execute=Executor(), ablate_one=no_ablation, load=lambda t: t,
        emit=[].append, ablate=False, write=fake_write, timer=ticker(), offset=1000, max_runs=3,
    )
    assert all(replicate(r) >= 1000 for r in collect.read_markers(tmp_path))
    with pytest.raises(CollectionError):
        collect.load_states(collect.read_markers(tmp_path), 0)


# -- ablation pass ------------------------------------------------------------


def _statuses(config, attempt):
    return {0: Status.CONSUMER_STEP_LIMIT, 1: Status.WRITER_NO_STATE}.get(config.seed % 5, Status.COMPLETED)


def _tools(config):
    return ("http.post",) if config.seed % 2 == 0 else ()


def fake_ablate(calls, failing=()):
    def ablate_one(record_path, directory):
        run_id = record_path.name[: -len(".run.json")]
        calls.append(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{run_id}.ablation.json").write_text("{}")
        return {"run_id": run_id, "entries": 1, "reruns": 2, "backend_failed": run_id in failing,
                "verdicts": {"http.post": "not_caused"}}
    return ablate_one


def test_the_targets_are_ablated_statuses_with_a_candidate(tmp_path, small):
    run(tmp_path, Executor(_statuses, _tools))
    markers = collect.read_markers(tmp_path)
    targets = collect.ablation_targets(markers, FAMILY)
    assert targets
    assert targets == sorted(r for r, m in markers.items() if m["status"] in ABLATED_STATUSES and m["candidates"])
    assert any(markers[r]["status"] == Status.CONSUMER_STEP_LIMIT for r in targets)
    assert not any(markers[r]["status"] == Status.WRITER_NO_STATE for r in targets)
    assert collect.ablation_targets(markers, "qwen3") == []


def test_the_pass_is_idempotent_and_sets_aside_an_unfinished_ablation(tmp_path, small):
    run(tmp_path, Executor(_statuses, _tools))
    targets = collect.ablation_targets(collect.read_markers(tmp_path), FAMILY)
    leftover = tmp_path / collect.ABLATION_DIR / targets[0]
    leftover.mkdir(parents=True)
    (leftover / "half.run.json").write_text("{}")
    calls, lines = [], []
    assert collect.ablation_pass(FAMILY, tmp_path, ablate_one=fake_ablate(calls), emit=lines.append) == len(targets)
    assert calls == targets
    assert len(list(tmp_path.glob(f"{collect.ORPHANS}/*/ablation-{targets[0]}/half.run.json"))) == 1
    again = []
    assert collect.ablation_pass(FAMILY, tmp_path, ablate_one=fake_ablate(again), emit=[].append) == 0
    assert again == []
    assert f"ablated {len(targets)}/{len(targets)}" in collect.render_status(tmp_path)


def test_a_lost_rerun_stops_the_pass_after_it_is_stored(tmp_path, small):
    run(tmp_path, Executor(_statuses, _tools))
    targets = collect.ablation_targets(collect.read_markers(tmp_path), FAMILY)
    calls = []
    with pytest.raises(CollectionError, match="backend"):
        collect.ablation_pass(FAMILY, tmp_path, ablate_one=fake_ablate(calls, {targets[0]}), emit=[].append)
    assert calls == targets[:1]
    rest = []
    collect.ablation_pass(FAMILY, tmp_path, ablate_one=fake_ablate(rest), emit=[].append)
    assert rest == targets[1:]


def test_a_block_is_ablated_only_once_it_is_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "TARGET", 1)
    monkeypatch.setattr(collect, "CAP", 2)
    executor = Executor(tools=lambda config: ("http.post",))
    calls = []
    run(tmp_path, executor, max_runs=5, ablate=True, ablate_one=fake_ablate(calls))
    assert calls == []
    run(tmp_path, executor, ablate=True, ablate_one=fake_ablate(calls))
    assert len(calls) == 15


# -- progress and command line ------------------------------------------------


def test_eta_falls_back_to_the_budget_and_reaches_zero(tmp_path, small):
    fresh = [collect.CellState(c) for c in collect.cells_of(FAMILY)]
    assert collect.family_eta(fresh, [], "granite4.1:8b") > 0
    run(tmp_path, Executor())
    closed = collect.load_states(collect.read_markers(tmp_path))[FAMILY]
    assert collect.family_eta(closed, [1.0] * 45, "granite4.1:8b") == 0
    assert collect._fmt_duration(3660) == "1h01m"


def test_status_of_an_empty_directory_and_a_bad_offset(tmp_path, capsys):
    assert collect.main(["--out", str(tmp_path / "none"), "--status"]) == 0
    assert "completed 0/2025" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        collect.main(["--out", str(tmp_path), "--status", "--replicate-offset", "5"])


# -- one block through the real executor -------------------------------------


class Model:
    """Writer records a note; consumer inserts a row, outside every -01 mandate, then stops."""

    def __init__(self):
        self.unloads = 0

    def unload(self):
        self.unloads += 1

    def identity(self, model):
        return {"backend": "fake", "version": "0", "model": model, "digest": "d" * 64}

    def chat(self, *, model, messages, tools, options, think):
        results = sum(1 for m in messages if m["role"] == "tool")
        if messages[0]["content"] != CONSUMER_SYSTEM_PROMPT:
            calls = (("state.append", {"content": "a note"}),) if results == 0 else ()
        else:
            calls = (("db.insert_row", {"table": "t", "row": {}}),) if results == 0 else ()
        return ChatResponse(
            content="" if calls else "Done.", thinking="",
            tool_calls=tuple({"function": {"name": n, "arguments": a}} for n, a in calls),
            done_reason="stop",
            stats={"eval_count": 10, "eval_duration": 1, "prompt_eval_count": 100,
                   "total_duration": 1, "load_duration": 1},
        )


def test_one_block_through_the_executor_store_and_ablation(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "TARGET", 1)
    monkeypatch.setattr(collect, "CAP", 2)
    monkeypatch.setattr(collect, "candidates_of", real_candidates_of)  # real journals here
    model = Model()
    clock = lambda: "2026-01-01T00:00:00+00:00"  # noqa: E731
    executed = collect.collect(
        tmp_path, [FAMILY], execute=lambda config, world: execute_run(config, world, model, clock=clock),
        ablate_one=collect.make_ablate_one(model, clock=clock, timer=ticker()), emit=[].append,
        timer=ticker(),
    )
    assert executed == 15
    markers = collect.read_markers(tmp_path)
    assert {m["status"] for m in markers.values()} == {Status.COMPLETED}
    assert all(m["candidates"] == ["db.insert_row"] for m in markers.values())
    assert all((tmp_path / f"{r}.run.json").exists() and (tmp_path / f"{r}.journal.json").exists()
               for r in markers)
    ablations = sorted(tmp_path.glob(f"{collect.ABLATION_DIR}/*/*.ablation.json"))
    assert len(ablations) == 15
    record = json.loads(ablations[0].read_text())
    assert record["entries"] == 1 and len(record["reruns"]) == 2
    assert [c["tool"] for c in record["candidates"]] == ["db.insert_row"]
    assert not record["any_confirmed"]
