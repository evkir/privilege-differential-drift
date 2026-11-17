# SPDX-License-Identifier: Apache-2.0

"""Tests for execute_writer and harness/writer_check.py.

Written by Claude (Opus 5.5, Anthropic) at the author's request.

The check reads what writers record on worlds no run has touched, and must not
observe a consumer there; the first group of tests holds the executor to that.
The rest hold the plan to what §5 registers and the record to what the matcher
says of it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "analysis"))

import writer_check as wc  # noqa: E402
from executor import (  # noqa: E402
    CONSUMER_SYSTEM_PROMPT,
    ChatResponse,
    ModelError,
    Status,
    derive_seed,
    execute_writer,
)
from journal import EventKind  # noqa: E402
from pilot import BRIEF_CHECK_REPLICATE, PILOT_REPLICATE_BASE  # noqa: E402
from tasks import TASKS, Category  # noqa: E402
from worlds import load_world  # noqa: E402

CLOCK = lambda: "2026-01-01T00:00:00+00:00"  # noqa: E731


def _reply(*calls, text="", done_reason="stop"):
    return ChatResponse(
        content=text, thinking="",
        tool_calls=tuple({"function": {"name": n, "arguments": a}} for n, a in calls),
        done_reason=done_reason,
        stats={"eval_count": 10, "eval_duration": 1, "prompt_eval_count": 100, "total_duration": 1},
    )


class Writer:
    """Records ``notes`` one per call, then stops; fails the test if shown a consumer prompt."""

    def __init__(self, notes=("a note",), errors=(), loop=False):
        self.notes, self.errors, self.loop = list(notes), list(errors), loop
        self.chats = 0

    def unload(self):
        pass

    def identity(self, model):
        return {"backend": "fake", "version": "0", "model": model, "digest": "d" * 64}

    def chat(self, *, model, messages, tools, options, think):
        assert messages[0]["content"] != CONSUMER_SYSTEM_PROMPT, "a consumer was started"
        self.chats += 1
        if self.errors:
            error = self.errors.pop(0)
            if error:
                raise ModelError(error)
        done = sum(1 for m in messages if m["role"] == "tool")
        if done < len(self.notes):
            return _reply(("state.append", {"content": self.notes[done]}))
        if self.loop:
            return _reply(("fs.list_dir", {"path": ""}))
        return _reply(text="Done.")


def _config(task_id="financial-04", writer_level=0, **overrides):
    plan = {(c.task_id, c.writer_level): c for c in wc.writer_check_plan() if c.model_family == "granite4.1"}
    config = plan[(task_id, writer_level)]
    if overrides:
        from dataclasses import replace
        config = replace(config, **overrides)
    return config


# -- the executor stops after the writer --------------------------------------------


def test_a_writer_run_records_the_entries_and_starts_no_consumer():
    result = execute_writer(_config(), load_world("financial-04"), Writer(["one", "two"]), clock=CLOCK)
    assert result.status == Status.WRITER_DONE
    assert [e.content for e in result.entries] == ["one", "two"]
    assert "consumer" not in result.transcript
    assert all(c["agent"] == "writer" for c in result.transcript["calls"])
    result.journal.verify()
    assert result.journal.closed and result.journal.events[-1].payload == {"status": Status.WRITER_DONE}
    assert not result.journal.of_kind(EventKind.STATE_READ)


@pytest.mark.parametrize(
    "writer, expected",
    [
        (Writer(notes=()), Status.WRITER_NO_STATE),
        (Writer(loop=True), Status.WRITER_STEP_LIMIT),
        (Writer(errors=["connection refused"]), Status.MODEL_ERROR),
    ],
    ids=["no-entry", "step-limit", "backend"],
)
def test_a_writer_run_ends_the_ways_a_writer_can(writer, expected):
    result = execute_writer(_config(max_steps=3), load_world("financial-04"), writer, clock=CLOCK)
    assert result.status == expected


def test_a_truncated_writer_reply_ends_the_writer_run():
    class Truncating(Writer):
        def chat(self, **kwargs):
            return _reply(("state.append", {"content": "n"}), done_reason="length")

    result = execute_writer(_config(), load_world("financial-04"), Truncating(), clock=CLOCK)
    assert result.status == Status.WRITER_TRUNCATED and not result.entries


# -- the plan ------------------------------------------------------------------------


def test_the_plan_is_three_models_twenty_tasks_two_writer_levels():
    plan = wc.writer_check_plan()
    assert len(plan) == 120
    assert {(c.model_family, c.think) for c in plan} == {("granite4.1", None), ("gemma4", False), ("qwen3", False)}
    assert {(c.writer_level, c.consumer_level) for c in plan} == {(0, 2), (1, 2)}
    assert len({c.task_id for c in plan}) == 20
    assert all(not next(t for t in TASKS if t.task_id == c.task_id).category.is_control for c in plan)


def test_the_plan_uses_its_own_index_and_the_collection_settings():
    plan = wc.writer_check_plan()
    assert wc.WRITER_CHECK_REPLICATE not in (PILOT_REPLICATE_BASE, PILOT_REPLICATE_BASE + 1, BRIEF_CHECK_REPLICATE)
    for c in plan:
        assert c.replicate == wc.WRITER_CHECK_REPLICATE
        assert c.seed == derive_seed(c.model_family, c.task_id, c.writer_level, c.consumer_level, c.replicate)
        assert (c.num_ctx, c.num_predict, c.max_steps, c.tool_result_limit) == (8192, 8192, 28, 4000)
        assert c.temperature is None
    assert len({c.seed for c in plan}) == len(plan)


def test_every_substantive_category_is_in_the_plan_and_the_control_is_not():
    categories = {next(t for t in TASKS if t.task_id == task_id).category for task_id in wc.WRITER_CHECK_TASKS}
    assert categories == set(Category) - {Category.CONTROL}


# -- running and reading the check ----------------------------------------------------


def _faithful(task_id):
    from fidelity import reference_answer
    task = next(t for t in TASKS if t.task_id == task_id)
    return [f"{i.keys[0]}: " + ", ".join(i.values) for i in reference_answer(task, load_world(task_id))]


def test_the_check_stores_each_writer_once_and_scores_it(tmp_path):
    plan = [_config("financial-04"), _config("source_code-05", writer_level=1)]
    lines = []

    class Faithful(Writer):
        def chat(self, *, model, messages, tools, options, think):
            task_id = "financial-04" if "reimbursement" in messages[1]["content"] else "source_code-05"
            self.notes = ["\n".join(_faithful(task_id))]
            return super().chat(model=model, messages=messages, tools=tools, options=options, think=think)

    assert wc.run_writer_check(plan, Faithful(), tmp_path, emit=lines.append, clock=CLOCK) == 2
    assert all(line.endswith("F=1.00") for line in lines)
    assert wc.run_writer_check(plan, Faithful(), tmp_path, emit=lines.append, clock=CLOCK) == 0
    assert lines[-1].endswith("already stored")
    rows = wc.load_check(tmp_path)
    assert [(r["task_id"], r["status"], r["fidelity"]) for r in rows] == [
        ("financial-04", Status.WRITER_DONE, 1.0), ("source_code-05", Status.WRITER_DONE, 1.0)]


def test_a_backend_failure_is_tried_again_and_then_stored(tmp_path):
    lines = []
    writer = Writer(errors=["reset", "reset"])
    wc.run_writer_check([_config()], writer, tmp_path, emit=lines.append, clock=CLOCK, retries=3)
    record = json.loads(next(tmp_path.glob("*.run.json")).read_text())
    assert record["status"] == Status.WRITER_DONE and "(2 backend failure(s))" in lines[0]
    lines.clear()
    wc.run_writer_check([_config(writer_level=1)], Writer(errors=["x"] * 5), tmp_path, emit=lines.append,
                        clock=CLOCK, retries=3)
    assert Status.MODEL_ERROR in lines[0]


def test_what_is_read_by_hand_is_every_writer_below_one_and_a_fixed_draw_of_the_rest():
    rows = [{"run_id": f"r{i:03d}", "group": "g", "writer_level": i % 2, "task_id": "t", "status": "writer_done",
             "entries": ["x"], "fidelity": 1.0 if i % 5 else 0.5, "missing": [] if i % 5 else ["k 1"]}
            for i in range(40)]
    sample = wc.full_sample(rows)
    assert len(sample) == wc.FULL_SAMPLE_SIZE and sample == wc.full_sample(list(reversed(rows)))
    assert all(next(r for r in rows if r["run_id"] == s)["fidelity"] == 1 for s in sample)
    text = wc.render(rows, planned=40)
    below = [r["run_id"] for r in rows if r["fidelity"] < 1]
    assert all(f"### {run_id} — F = 0.50" in text for run_id in below)
    assert all(f"### {run_id} — F = 1.00" in text for run_id in sample)
    assert text.count("### ") == len(below) + len(sample)


def test_a_writer_that_did_not_finish_is_counted_but_not_scored(tmp_path):
    wc.run_writer_check([_config(max_steps=3)], Writer(loop=True), tmp_path, emit=lambda line: None, clock=CLOCK)
    (row,) = wc.load_check(tmp_path)
    assert row["status"] == Status.WRITER_STEP_LIMIT and row["entries"] == ["a note"]
    assert row["fidelity"] is None
    assert "writer_step_limit 1" in wc.render([row], planned=1)
