# SPDX-License-Identifier: Apache-2.0

"""Tests for harness/ablation.py.

Written by Claude (Opus 5, Anthropic) at the author's request.

The decision of D7(iii) is tested first on its own, from what each re-run is
given as having invoked: the unresolved endings, an action that one entry
causes, one that either of two entries causes, one that needs both removed,
one the state does not cause, and a lattice that is not monotone, where a
search that assumed it would stop early.

The runner is then tested end to end against a client that answers from the
conversation it is shown, with a consumer whose actions depend on which notes
it is handed. The rule for "produces" must hold in both directions: the same
tool on other arguments is the same action, and a re-run that drops the tool
is a change. A re-run on the unaltered state that acts differently must leave
the candidate unresolved, and nothing lost to the backend may be stored.

No model is called.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "analysis"))

import pilot  # noqa: E402
from ablation import (  # noqa: E402
    ABLATED_STATUSES,
    BACKEND_FAILED,
    BACKEND_RETRIES,
    NOT_REPRODUCED,
    TOO_MANY_ENTRIES,
    AblationError,
    Candidate,
    ablate,
    candidates_of,
    evaluate,
    minimal_sets,
    removal_sets,
    rerun_id,
    store,
    tools_invoked,
)
from budget import MAX_EXHAUSTIVE_ENTRIES  # noqa: E402
from executor import (  # noqa: E402
    CONSUMER_SYSTEM_PROMPT,
    ChatResponse,
    ModelError,
    Status,
    World,
    execute_run,
)
from worlds import world_path  # noqa: E402

CLOCK = lambda: "2026-01-01T00:00:00+00:00"  # noqa: E731
OUTSIDE = "http.get"  # in the L2 opportunity set of every task, outside every mandate
WITHIN = "fs.write_file"


class Model:
    """Writer appends ``notes``; the consumer's calls are ``consumer(visible notes, start)``."""

    def __init__(self, notes, consumer, *, errors=()):
        self.notes = list(notes)
        self.consumer = consumer
        self.errors = list(errors)
        self.starts = 0

    def unload(self):
        pass

    def identity(self, model):
        return {"backend": "fake", "version": "0", "model": model, "digest": "d" * 64}

    def _reply(self, calls=(), text=""):
        return ChatResponse(
            content=text, thinking="",
            tool_calls=tuple({"function": {"name": n, "arguments": a}} for n, a in calls),
            done_reason="stop",
            stats={"eval_count": 10, "eval_duration": 1_000_000_000, "prompt_eval_count": 500,
                   "total_duration": 2_000_000_000, "load_duration": 1_000_000_000},
        )

    def chat(self, *, model, messages, tools, options, think):
        if self.errors:
            error = self.errors.pop(0)
            if error:
                raise ModelError(error)
        results = sum(1 for m in messages if m["role"] == "tool")
        if messages[0]["content"] != CONSUMER_SYSTEM_PROMPT:
            if results < len(self.notes):
                return self._reply([("state.append", {"content": self.notes[results]})])
            return self._reply(text="Done.")
        if len(messages) == 2:
            self.starts += 1
        opening = messages[1]["content"]
        visible = frozenset(n for n in self.notes if n in opening)
        plan = self.consumer(visible, self.starts)
        if results < len(plan):
            return self._reply([plan[results]])
        return self._reply(text="Done.")


def _config():
    config = pilot.pilot_plan(["qwen3"])[0]
    assert (config.writer_level, config.consumer_level) == (0, 2)
    return config


def _world(config):
    return World.from_json(world_path(config.task_id))


def _original(model):
    config = _config()
    world = _world(config)
    result = execute_run(config, world, model, clock=CLOCK)
    return config, world, result


def _ablate(model, config, world, result, **kwargs):
    return ablate(
        config, world, model, run_id=result.run_id, status=result.status,
        entries=result.entries, journal=result.journal, clock=CLOCK, **kwargs,
    )


# -- the lattice --------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 6, MAX_EXHAUSTIVE_ENTRIES])
def test_removal_sets_are_every_non_empty_subset_once(n):
    sets = removal_sets(n)
    assert len(sets) == 2**n - 1 == len(set(sets))
    assert [len(s) for s in sets] == sorted(len(s) for s in sets)
    assert all(s == tuple(sorted(s)) for s in sets)


def test_removal_sets_of_no_entries_are_refused():
    with pytest.raises(ValueError):
        removal_sets(0)


def test_minimal_sets_drop_every_superset():
    assert minimal_sets([(0,), (0, 1), (1, 2), (0, 1, 2)]) == ((0,), (1, 2))
    assert minimal_sets([]) == ()


def test_rerun_identifiers():
    assert rerun_id("q.control-01.02.r900", ()) == "q.control-01.02.r900.null"
    assert rerun_id("q.control-01.02.r900", (0, 2)) == "q.control-01.02.r900.without-0-2"


# -- deciding D7(iii) ---------------------------------------------------------

A = Candidate(OUTSIDE, 7)


def _observed(n, stops):
    """The tool is invoked on every removal except those for which ``stops`` is true."""
    obs = {(): frozenset({OUTSIDE, WITHIN})}
    for r in removal_sets(n):
        obs[r] = frozenset({WITHIN}) if stops(set(r)) else frozenset({OUTSIDE, WITHIN})
    return obs


def test_one_entry_causes_the_action():
    (c,) = evaluate([A], 3, _observed(3, lambda r: 1 in r))
    assert c.confirmed and c.unique_minimal
    assert c.minimal_sets == ((1,),)
    assert len(c.causal_sets) == 4


def test_either_of_two_entries_causes_the_action():
    (c,) = evaluate([A], 3, _observed(3, lambda r: bool(r & {0, 2})))
    assert c.confirmed and not c.unique_minimal
    assert c.minimal_sets == ((0,), (2,))


def test_the_action_stops_only_when_both_are_removed():
    (c,) = evaluate([A], 3, _observed(3, lambda r: {0, 1} <= r))
    assert c.minimal_sets == ((0, 1),)
    assert c.unique_minimal


def test_the_state_does_not_cause_the_action():
    (c,) = evaluate([A], 3, _observed(3, lambda r: False))
    assert c.unresolved is None
    assert not c.confirmed and c.causal_sets == () and not c.unique_minimal


def test_a_lattice_that_is_not_monotone_is_read_in_full():
    # removing {0} stops the action; removing {0, 1} does not; removing everything does
    (c,) = evaluate([A], 3, _observed(3, lambda r: r in ({0}, {0, 1, 2})))
    assert c.causal_sets == ((0,), (0, 1, 2))
    assert c.minimal_sets == ((0,),)


def test_a_candidate_the_unaltered_rerun_does_not_produce_is_unresolved():
    obs = _observed(2, lambda r: True)
    obs[()] = frozenset({WITHIN})
    (c,) = evaluate([A], 2, obs)
    assert c.unresolved == NOT_REPRODUCED and not c.confirmed


def test_too_many_entries_leaves_every_candidate_unresolved():
    outcomes = evaluate([A, Candidate("email.send", 9)], MAX_EXHAUSTIVE_ENTRIES + 1, {})
    assert [c.unresolved for c in outcomes] == [TOO_MANY_ENTRIES] * 2


def test_a_rerun_lost_to_the_backend_leaves_the_candidate_unresolved():
    obs = _observed(2, lambda r: 0 in r)
    obs[(1,)] = None
    assert evaluate([A], 2, obs)[0].unresolved == BACKEND_FAILED
    obs = _observed(2, lambda r: 0 in r)
    obs[()] = None
    assert evaluate([A], 2, obs)[0].unresolved == BACKEND_FAILED


def test_a_lattice_with_removals_not_run_is_refused():
    obs = _observed(2, lambda r: True)
    del obs[(0, 1)]
    with pytest.raises(AblationError):
        evaluate([A], 2, obs)
    with pytest.raises(AblationError):
        evaluate([A], 2, {})


# -- end to end ---------------------------------------------------------------


def _by(note):
    """A consumer that reaches outside the mandate only when ``note`` is in its state."""
    def consumer(visible, start):
        calls = [(WITHIN, {"path": "out.txt", "content": "x"})]
        if note in visible:
            calls.insert(0, (OUTSIDE, {"url": "https://example.org/" + str(len(visible))}))
        return calls
    return consumer


def test_candidates_are_the_consumer_tools_under_a_candidate_ruling():
    def consumer(visible, start):
        return [(OUTSIDE, {"url": "a"}), (WITHIN, {"path": "o", "content": "x"}),
                (OUTSIDE, {"url": "b"})]
    _, _, result = _original(Model(["one"], consumer))
    assert result.status == Status.COMPLETED
    (c,) = candidates_of(result.journal)
    assert c.tool == OUTSIDE
    assert tools_invoked(result.journal) == {OUTSIDE, WITHIN}


def test_an_entry_that_causes_the_action_is_found():
    model = Model(["alpha", "beta"], _by("beta"))
    config, world, result = _original(model)
    ablation, reruns = _ablate(model, config, world, result)
    (c,) = ablation.candidates
    assert c.confirmed and c.minimal_sets == ((1,),)
    assert ablation.any_confirmed and not ablation.all_unresolved
    assert [r["removed"] for r in ablation.reruns] == [[], [0], [1], [0, 1]]
    assert len(reruns) == 4
    assert all(r.ablated_from == result.run_id for r in reruns)


def test_the_same_tool_on_other_arguments_is_the_same_action():
    # with "alpha" removed the URL changes, the tool does not
    model = Model(["alpha", "beta"], _by("beta"))
    config, world, result = _original(model)
    ablation, _ = _ablate(model, config, world, result)
    assert (0,) not in ablation.candidates[0].causal_sets


def test_the_replay_signature_would_have_called_every_removal_causal():
    model = Model(["alpha", "beta"], _by("beta"))
    config, world, result = _original(model)
    _, reruns = _ablate(model, config, world, result)
    null = reruns[0]
    assert tools_invoked(null.journal) == tools_invoked(result.journal)
    assert null.journal.replay_signature() != result.journal.replay_signature()


def test_a_run_without_candidates_is_not_rerun():
    model = Model(["alpha"], lambda visible, start: [(WITHIN, {"path": "o", "content": "x"})])
    config, world, result = _original(model)
    ablation, reruns = _ablate(model, config, world, result)
    assert ablation.candidates == () and reruns == [] and ablation.reruns == []
    assert not ablation.all_unresolved


def test_an_unaltered_rerun_that_acts_differently_stops_the_ablation():
    def consumer(visible, start):
        calls = [(WITHIN, {"path": "o", "content": "x"})]
        return [(OUTSIDE, {"url": "a"})] + calls if start == 1 else calls
    model = Model(["alpha", "beta"], consumer)
    config, world, result = _original(model)
    ablation, reruns = _ablate(model, config, world, result)
    assert ablation.candidates[0].unresolved == NOT_REPRODUCED
    assert ablation.all_unresolved
    assert len(ablation.reruns) == 1 and len(reruns) == 1


def test_a_run_above_the_entry_limit_is_not_rerun():
    notes = [f"note {i:02d}" for i in range(MAX_EXHAUSTIVE_ENTRIES + 1)]
    model = Model(notes, _by("note 00"))
    config, world, result = _original(model)
    assert len(result.entries) == MAX_EXHAUSTIVE_ENTRIES + 1
    ablation, reruns = _ablate(model, config, world, result)
    assert [c.unresolved for c in ablation.candidates] == [TOO_MANY_ENTRIES]
    assert reruns == [] and ablation.reruns == []


def test_a_backend_failure_is_retried_and_not_counted():
    model = Model(["alpha"], _by("alpha"))
    config, world, result = _original(model)
    model.errors = ["boom"]
    ablation, reruns = _ablate(model, config, world, result)
    assert ablation.reruns[0]["backend_failures"] == ["boom"]
    assert ablation.candidates[0].confirmed
    assert len(reruns) == 2


def test_a_rerun_the_backend_keeps_failing_is_not_stored():
    model = Model(["alpha"], _by("alpha"))
    config, world, result = _original(model)
    model.errors = ["boom"] * (BACKEND_RETRIES + 1)
    ablation, reruns = _ablate(model, config, world, result)
    assert ablation.reruns[0]["status"] == BACKEND_FAILED
    assert len(ablation.reruns[0]["backend_failures"]) == BACKEND_RETRIES + 1
    assert ablation.candidates[0].unresolved == BACKEND_FAILED
    assert reruns == []


def test_a_run_the_consumer_did_not_act_in_is_refused():
    model = Model(["alpha"], _by("alpha"))
    config, world, result = _original(model)
    assert Status.WRITER_NO_STATE not in ABLATED_STATUSES
    with pytest.raises(AblationError):
        ablate(config, world, model, run_id=result.run_id, status=Status.WRITER_NO_STATE,
               entries=result.entries, journal=result.journal, clock=CLOCK)
    with pytest.raises(AblationError):
        ablate(config, world, model, run_id=result.run_id, status=result.status,
               entries=(), journal=result.journal, clock=CLOCK)


def test_store_writes_the_record_last_and_refuses_to_overwrite(tmp_path):
    model = Model(["alpha", "beta"], _by("beta"))
    config, world, result = _original(model)
    ablation, reruns = _ablate(model, config, world, result)
    path = store(ablation, reruns, tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["any_confirmed"] is True and record["all_unresolved"] is False
    assert record["candidates"][0]["minimal_sets"] == [[1]]
    assert record["candidates"][0]["unique_minimal"] is True
    assert len(list(tmp_path.glob("*.run.json"))) == 4
    with pytest.raises(AblationError):
        store(ablation, reruns, tmp_path)


def test_the_cli_ablates_a_stored_run(tmp_path, monkeypatch):
    import ablation as module
    from executor import write_run

    model = Model(["alpha", "beta"], _by("beta"))
    config, world, result = _original(model)
    _, record_path = write_run(result, tmp_path / "runs")
    monkeypatch.setattr(module, "OllamaClient", lambda host: model)
    assert module.main(["--record", str(record_path), "--out", str(tmp_path / "abl")]) == 0
    stored = json.loads((tmp_path / "abl" / f"{result.run_id}.ablation.json").read_text())
    assert stored["candidates"][0]["minimal_sets"] == [[1]]


def test_settings_agree_with_the_pilot_and_the_budget():
    assert BACKEND_RETRIES == pilot.BACKEND_RETRIES
    import budget
    assert MAX_EXHAUSTIVE_ENTRIES == budget.MAX_EXHAUSTIVE_ENTRIES
