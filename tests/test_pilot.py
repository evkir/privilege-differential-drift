# SPDX-License-Identifier: Apache-2.0

"""Tests for harness/pilot.py.

Written by Claude (Fable 5.2, Anthropic) at the author's request.

The pilot fixes parameters of the design by rules registered before it runs,
so what is tested first is that the runner executes the registered plan and
that the rules give what §5 says they give, at their boundaries.

The second group is the re-run of the consumer on unaltered entries. It is the
check attribution rests on, and it must be able to fail: a consumer that acts
differently the second time has to be reported as different, and one that only
words a reply differently has to be told apart from it.

The third is what the pilot must not do: store a run the backend failed on,
execute a stored run again, or let the consumer's gate classifications reach
the table.

No model is called. A client that answers from the conversation it is shown
stands in for one.
"""

from __future__ import annotations

import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "analysis"))

import pilot  # noqa: E402
from executor import CONSUMER_SYSTEM_PROMPT, ChatResponse, ModelError, Status, World  # noqa: E402
from pilot import (  # noqa: E402
    BACKEND_FAILED,
    FAMILIES,
    PILOT_SETTINGS,
    PilotError,
    load_pilot,
    parse_nvidia_smi,
    pilot_plan,
    render_table,
    rule_values,
    run_one,
    run_pilot,
    summarise,
)
from worlds import world_path  # noqa: E402

CLOCK = lambda: "2026-01-01T00:00:00+00:00"  # noqa: E731
PROBE = lambda: {"gpu_processes": [], "gpu_model_mib": 8428, "gpu_used_mib": 8600,  # noqa: E731
                 "server_gpu_share": 1.0}
DOCUMENT = (ROOT / "PREREGISTRATION.md").read_text(encoding="utf-8")


class Model:
    """Answers from the conversation it is shown; ``consumer`` picks its behaviour per start."""

    def __init__(self, *, writer="append", consumer=("write",), errors=(), call_ids=False):
        self.writer, self.consumer = writer, list(consumer)
        self.call_ids = call_ids
        self.errors = list(errors)
        self.consumer_starts = 0
        self.chats = 0
        self.unloads = 0

    def unload(self):
        self.unloads += 1

    def identity(self, model):
        return {"backend": "fake", "version": "0", "model": model, "digest": "d" * 64}

    def _reply(self, *calls, text=""):
        return ChatResponse(
            content=text, thinking="",
            tool_calls=tuple(
                {"function": {"name": n, "arguments": a},
                 **({"id": f"call_{self.chats}_{i}"} if self.call_ids else {})}
                for i, (n, a) in enumerate(calls)
            ),
            done_reason="stop",
            stats={"eval_count": 40, "eval_duration": 2_000_000_000, "prompt_eval_count": 1000,
                   "total_duration": 3_000_000_000, "load_duration": 5_000_000_000},
        )

    def chat(self, *, model, messages, tools, options, think):
        self.chats += 1
        if self.errors:
            error = self.errors.pop(0)
            if error:
                raise ModelError(error)
        results = sum(1 for m in messages if m["role"] == "tool")
        if messages[0]["content"] != CONSUMER_SYSTEM_PROMPT:
            if self.writer == "append" and results == 0:
                return self._reply(("state.append", {"content": "a note"}))
            if self.writer == "loop":
                return self._reply(("fs.list_dir", {"path": ""}))
            return self._reply(text="Done.")
        if len(messages) == 2:
            self.consumer_starts += 1
        behaviour = self.consumer[min(self.consumer_starts, len(self.consumer)) - 1]
        if results == 0 and behaviour != "idle":
            path = "other.txt" if behaviour == "write-elsewhere" else "out.txt"
            return self._reply(("fs.write_file", {"path": path, "content": "x"}))
        return self._reply(text="Done, at length." if behaviour == "wordy" else "Done.")


def _first(family="qwen3"):
    return pilot_plan([family])[0]


def _one(model, tmp_path, config=None, **kwargs):
    config = config or _first()
    return run_one(config, World.from_json(world_path(config.task_id)), model, tmp_path,
                   probe=PROBE, clock=CLOCK, **kwargs)


# -- the plan -----------------------------------------------------------------


def test_the_registered_pilot_is_forty_runs():
    plan = pilot_plan()
    assert len(plan) == 40
    assert [sum(c.model_family == f for c in plan) for f in ("llama3.1", "mistral-nemo", "qwen3")] == [10, 10, 20]
    assert {c.task_id for c in plan} == {f"control-0{n}" for n in range(1, 6)}
    assert {(c.writer_level, c.consumer_level) for c in plan} == {(0, 2), (1, 2)}
    assert all(c.temperature is None for c in plan)


def test_no_two_pilot_runs_share_an_identifier_or_a_seed():
    plan = pilot_plan(sorted(FAMILIES))
    assert len({pilot.run_id_for(c) for c in plan}) == len(plan) == 70
    assert len({c.seed for c in plan}) == len(plan)


def test_the_second_reasoning_mode_takes_the_next_replicate_index():
    plan = pilot_plan(["qwen3"])
    assert {(c.think, c.replicate) for c in plan} == {(False, 900), (True, 901)}
    assert {(c.think, c.replicate) for c in pilot_plan(["llama3.1"])} == {(None, 900)}
    assert all(c.replicate >= 900 for c in pilot_plan(sorted(FAMILIES)))


def test_the_candidates_are_piloted_as_section_5_says():
    assert len(pilot_plan(["granite4.1"])) == 10
    assert len(pilot_plan(["gemma4"])) == 20
    for tag in ("llama3.1:8b", "mistral-nemo:12b", "qwen3:14b", "granite4.1:8b", "gemma4:12b"):
        assert f"`{tag}`" in DOCUMENT and tag in {f.model for f in FAMILIES.values()}


def test_the_pilot_settings_are_the_ones_the_document_registers():
    assert PILOT_SETTINGS == {"num_ctx": 8192, "num_predict": 4096, "max_steps": 32, "tool_result_limit": 4000}
    assert ("context of 8,192 tokens, reply-length limit of 4,096 tokens, step limit of 32 per agent, "
            "tool results cut at 4,000 characters") in DOCUMENT
    assert "replicate indices from 900" in DOCUMENT and pilot.PILOT_REPLICATE_BASE == 900


def _recorded_pilot():
    """The groups of the pilot table recorded in §5, as far as the rules read them."""
    block = DOCUMENT.split("**Pilot result.**", 1)[1].split("**Statistical plan.**", 1)[0]
    rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in block.splitlines() if line.startswith("|")
    ]
    cells = {row[0]: row[1:] for row in rows}
    groups = {}
    for column, label in enumerate(cells["Measured"]):
        family, _, mode = label.partition(", think ")
        value = lambda name: cells[name][column]  # noqa: E731
        runs, nulls = int(value("Runs")), int(value("Consumer re-runs on unaltered entries"))
        most = value("Most model calls by an agent in a completed run")
        same = value("Re-runs repeating the sequence of invocations")
        groups[label] = {
            "family": family, "think": {"": None, "on": True, "off": False}[mode],
            "collapse_share": Fraction(round(float(value("Collapse share")) * runs), runs),
            "longest_reply": int(value("Longest reply, tokens")),
            "prompt_peak": float(value("Peak prompt, share of context")) * 8192, "num_ctx": 8192,
            "most_calls_completed": None if most == "n/a" else int(most),
            "null_runs": nulls,
            "null_same_invocations": 0 if same == "n/a" else round(float(same) * nulls),
        }
    return block, cells, groups


def test_the_recorded_rule_values_are_what_the_rules_give_on_the_recorded_table():
    block, cells, groups = _recorded_pilot()
    assert set(groups) == {pilot.group_label(f.label, t) for f in FAMILIES.values() for t in f.think_modes}
    assert sum(int(n) for n in cells["Runs"]) == len(pilot_plan(sorted(FAMILIES))) == 70
    rules = rule_values(groups)
    assert rules["reasoning_mode"] == {"gemma4": "off", "qwen3": "off"}
    assert {f for f, verdict in rules["family_replacement"].items() if verdict.startswith("replace")} == {
        "llama3.1", "mistral-nemo"}
    assert (rules["reply_length_limit"], rules["context_size"], rules["step_limit"]) == (8192, 8192, 28)
    assert (rules["null_runs"], rules["collection_blocked"]) == (47, False)
    for family, mode in rules["reasoning_mode"].items():
        assert f"- Reasoning mode of {family}: {mode}\n" in block
    for family, verdict in rules["family_replacement"].items():
        assert f"- Family replacement, {family}: {verdict}\n" in block
    for line in (
        "- Reply-length limit: 8192 tokens (longest reply 2535)",
        "- Context size: 8192 tokens",
        "- Step limit: 28 (most model calls by an agent in a completed run: 14)",
        "- Reproducibility of the consumer: 0 of 47 re-runs differ in their invocations — does not block collection",
        "a context of 8,192 tokens, a reply-length limit of 8,192 tokens and a step limit of 28 per agent",
        "`granite4.1:8b` (IBM), `gemma4:12b` (Google) and `qwen3:14b` (Alibaba, Qwen team), the last two with "
        "the reasoning mode off",
    ):
        assert line in block
    assert set(cells["Re-runs repeating every reply in full"]) == {"1.00", "n/a"}


def test_an_unknown_family_is_refused():
    with pytest.raises(PilotError):
        pilot_plan(["gpt"])


# -- one run and the re-run of its consumer -----------------------------------


def test_a_completed_run_is_stored_with_the_re_run_of_its_consumer(tmp_path):
    model = Model()
    record = _one(model, tmp_path)
    run_id = "qwen3.control-01.02.r900"
    assert record["run_id"] == run_id and record["status"] == Status.COMPLETED
    assert record["null"] == {
        "run_id": f"{run_id}.null", "status": Status.COMPLETED, "backend_failures": [],
        "elapsed_seconds": record["null"]["elapsed_seconds"],
        "same_invocations": True, "same_replies": True,
    }
    assert {p.name for p in tmp_path.iterdir()} == {
        f"{stem}.{kind}.json" for stem in (run_id, f"{run_id}.null") for kind in ("journal", "run")
    } | {f"{run_id}.pilot.json"}
    original = json.loads((tmp_path / f"{run_id}.run.json").read_text())
    null = json.loads((tmp_path / f"{run_id}.null.run.json").read_text())
    assert null["ablated_from"] == run_id and null["config"] == original["config"]
    assert null["entries"] == original["entries"] and null["world_digest"] == original["world_digest"]
    assert "writer" not in null["transcript"]
    assert model.unloads == 3  # writer, consumer, consumer of the re-run
    assert json.loads((tmp_path / f"{run_id}.pilot.json").read_text()) == record


def test_a_consumer_that_acts_differently_the_second_time_is_reported(tmp_path):
    record = _one(Model(consumer=("write", "write-elsewhere")), tmp_path)
    assert record["null"]["same_invocations"] is False and record["null"]["same_replies"] is False


def test_a_consumer_that_only_words_a_reply_differently_is_told_apart(tmp_path):
    record = _one(Model(consumer=("write", "wordy")), tmp_path)
    assert record["null"]["same_invocations"] is True and record["null"]["same_replies"] is False


def test_an_identifier_the_server_attaches_to_a_call_is_not_part_of_a_reply(tmp_path):
    record = _one(Model(call_ids=True), tmp_path)
    run_id = record["run_id"]
    raw = [
        [m["tool_calls"] for m in json.loads((tmp_path / f"{stem}.run.json").read_text())
         ["transcript"]["consumer"]["messages"] if m.get("tool_calls")]
        for stem in (run_id, f"{run_id}.null")
    ]
    assert raw[0] != raw[1] and raw[0][0][0]["function"] == raw[1][0][0]["function"]
    assert record["null"]["same_invocations"] is True and record["null"]["same_replies"] is True


def test_identifiers_do_not_hide_a_reply_that_differs(tmp_path):
    record = _one(Model(consumer=("write", "write-elsewhere"), call_ids=True), tmp_path)
    assert record["null"]["same_invocations"] is False and record["null"]["same_replies"] is False
    record = _one(Model(consumer=("write", "wordy"), call_ids=True), tmp_path / "wordy")
    assert record["null"]["same_invocations"] is True and record["null"]["same_replies"] is False


def test_replies_are_compared_by_text_reasoning_and_parsed_calls():
    def transcript(*, text="", thinking="", arguments='{"path": "a", "content": "x"}', extra=None):
        call = {"function": {"name": "fs.write_file", "arguments": arguments}, **(extra or {})}
        return {
            "consumer": {"messages": [{"role": "user", "content": "go"},
                                      {"role": "assistant", "content": text, "tool_calls": [call]}]},
            "calls": [{"agent": "consumer", "thinking": thinking}],
        }

    base = pilot.replies_of("r", transcript(), "consumer")
    assert base == pilot.replies_of("r", transcript(extra={"id": "call_x", "type": "function"}), "consumer")
    assert base == pilot.replies_of("r", transcript(arguments={"content": "x", "path": "a"}), "consumer")
    assert base != pilot.replies_of("r", transcript(arguments={"content": "y", "path": "a"}), "consumer")
    assert base != pilot.replies_of("r", transcript(text="Writing."), "consumer")
    assert base != pilot.replies_of("r", transcript(thinking="Let me write."), "consumer")
    broken = transcript()
    broken["calls"] = []
    with pytest.raises(PilotError):
        pilot.replies_of("r", broken, "consumer")


def test_a_collapsed_run_has_no_re_run(tmp_path):
    record = _one(Model(writer="silent"), tmp_path)
    assert record["status"] == Status.WRITER_NO_STATE and record["null"] is None
    assert not list(tmp_path.glob("*.null.*"))
    looping = _one(Model(writer="loop"), tmp_path, config=pilot_plan(["qwen3"])[1])
    assert looping["status"] == Status.WRITER_STEP_LIMIT and looping["null"] is None


def test_a_backend_failure_is_executed_again_and_never_stored(tmp_path):
    model = Model(errors=["connection reset", None, None, None, None, "timed out"])
    record = _one(model, tmp_path)
    assert record["status"] == Status.COMPLETED and record["backend_failures"] == ["connection reset"]
    assert record["null"]["backend_failures"] == ["timed out"] and record["null"]["same_invocations"] is True
    stored = [json.loads(p.read_text())["status"] for p in tmp_path.glob("*.run.json")]
    assert stored == [Status.COMPLETED, Status.COMPLETED]


def test_a_run_the_backend_keeps_failing_on_is_abandoned_without_a_stored_run(tmp_path):
    record = _one(Model(errors=["down"] * 10), tmp_path, retries=2)
    assert record["status"] == BACKEND_FAILED and record["backend_failures"] == ["down"] * 3
    assert record["probe"] is None and record["null"] is None
    assert [p.name for p in tmp_path.iterdir()] == ["qwen3.control-01.02.r900.pilot.json"]


def test_a_re_run_the_backend_keeps_failing_on_is_recorded_as_such(tmp_path):
    record = _one(Model(errors=[None, None, None, None, "down", "down"]), tmp_path, retries=1)
    assert record["status"] == Status.COMPLETED
    assert record["null"]["status"] == BACKEND_FAILED and record["null"]["same_invocations"] is None
    assert not list(tmp_path.glob("*.null.*"))


@pytest.mark.parametrize("leftover", ["run.json", "journal.json", "null.run.json", "pilot.json"])
def test_a_stored_run_is_not_executed_again(tmp_path, leftover):
    (tmp_path / f"qwen3.control-01.02.r900.{leftover}").write_text("{}")
    model = Model()
    with pytest.raises(PilotError, match="already exists"):
        _one(model, tmp_path)
    assert model.chats == 0


# -- the whole pilot -----------------------------------------------------------


def test_an_interrupted_pilot_resumes_where_it_stopped(tmp_path):
    plan = pilot_plan(["llama3.1"])
    lines: list[str] = []
    first = Model()
    assert run_pilot(plan[:4], first, tmp_path, probe=PROBE, clock=CLOCK, emit=lines.append) == 4
    second = Model()
    assert run_pilot(plan, second, tmp_path, probe=PROBE, clock=CLOCK, emit=lines.append) == 6
    assert sum("already stored" in line for line in lines) == 4
    third = Model()
    assert run_pilot(plan, third, tmp_path, probe=PROBE, clock=CLOCK, emit=lines.append) == 0
    assert third.chats == 0 and len(load_pilot(tmp_path)) == 10
    assert "re-run: invocations same, replies same" in lines[0]


def test_a_differing_re_run_stands_out_in_the_progress_line(tmp_path):
    lines: list[str] = []
    run_pilot(pilot_plan(["llama3.1"])[:1], Model(consumer=("write", "write-elsewhere")), tmp_path,
              probe=PROBE, clock=CLOCK, emit=lines.append)
    assert "invocations DIFFERENT" in lines[0]


# -- the table ------------------------------------------------------------------


def _stored(tmp_path, family, model_for):
    for index, config in enumerate(pilot_plan([family])):
        _one(model_for(index, config), tmp_path, config=config)
    return load_pilot(tmp_path)


def test_the_table_reports_what_section_5_lists(tmp_path):
    rows = _stored(tmp_path, "llama3.1", lambda i, c: Model(
        writer="silent" if i < 3 else "append", consumer=("idle",) if i == 3 else ("write",)))
    summary = summarise(rows)
    group = summary["llama3.1"]
    assert (group["runs"], group["collapsed"], group["collapse_share"]) == (10, 3, Fraction(3, 10))
    assert group["endings"] == {Status.WRITER_NO_STATE: 3, Status.COMPLETED: 7}
    assert group["consumer_idle_share"] == Fraction(1, 7)
    assert group["scope_breach_share"] == 0 and group["multi_call_share"] == 0
    assert group["calls"]["writer"] == (2, 2) and group["calls"]["consumer"][1] == 2
    assert group["tokens_per_second"] == pytest.approx(20.0)
    assert group["load_median"] == pytest.approx(5.0)
    assert (group["prompt_peak"], group["num_ctx"], group["longest_reply"]) == (1000, 8192, 40)
    assert (group["null_runs"], group["null_same_invocations"], group["null_same_replies"]) == (7, 7, 7)
    assert (group["gpu_model_mib_max"], group["server_gpu_share_min"]) == (8428, 1.0)
    table = render_table(summary, planned=10)
    for needle in ("Runs with a pilot record: 10 of 10 planned.", "| Collapse share | 0.30 |",
                   "| Ended `writer_no_state` | 0.30 |", "| Completed runs in which the consumer made no call | 0.14 |",
                   "| Memory of the model process, driver, MiB (max) | 8428 |",
                   "| Re-runs repeating the sequence of invocations | 1.00 |",
                   "- Family replacement, llama3.1: keep (collapse share 0.30)",
                   "- Step limit: 12 ", "- Reply-length limit: 1024 tokens", "- Context size: 8192 tokens",
                   "0 of 7 re-runs differ in their invocations — does not block collection"):
        assert needle in table, needle


def test_a_writer_scope_breach_is_counted_on_its_own_line(tmp_path):
    class Breaching(Model):
        def chat(self, *, model, messages, tools, options, think):
            first_of_writer = messages[0]["content"] != CONSUMER_SYSTEM_PROMPT and len(messages) == 2
            if first_of_writer:
                self.chats += 1
                return self._reply(("fs.write_file", {"path": "x.txt", "content": "x"}))
            return super().chat(model=model, messages=messages, tools=tools, options=options, think=think)

    one_to_two = [c for c in pilot_plan(["llama3.1"]) if c.writer_level == 1][0]
    _one(Breaching(writer="silent"), tmp_path, config=one_to_two)
    assert summarise(load_pilot(tmp_path))["llama3.1"]["scope_breach_share"] == 1


def test_the_consumers_gate_classifications_do_not_reach_the_table(tmp_path):
    _stored(tmp_path, "llama3.1", lambda i, c: Model())
    before = render_table(summarise(load_pilot(tmp_path)))
    for path in tmp_path.glob("*.run.json"):
        stored = json.loads(path.read_text())
        stored["metrics"]["consumer"]["classifications"] = {"candidate_inflation": 9, "escalation_attempt": 9}
        path.write_text(json.dumps(stored))
    assert render_table(summarise(load_pilot(tmp_path))) == before
    source = (ROOT / "harness" / "pilot.py").read_text(encoding="utf-8")
    assert not re.search(r"CANDIDATE_INFLATION|ESCALATION_ATTEMPT|candidate_inflation|escalation_attempt", source)
    assert not re.search(r"\bMIR\b|\bAIR\b", before)
    assert all(row["run"] is None or "classifications" not in json.dumps(row["run"]) for row in load_pilot(tmp_path))


def test_an_abandoned_run_is_counted_and_enters_no_share(tmp_path):
    plan = pilot_plan(["llama3.1"])
    _one(Model(errors=["down"] * 9), tmp_path, config=plan[0])
    _one(Model(), tmp_path, config=plan[1])
    group = summarise(load_pilot(tmp_path))["llama3.1"]
    assert (group["runs"], group["backend_failed"], group["backend_failures"]) == (1, 1, 4)
    assert group["collapse_share"] == 0


# -- the rules ------------------------------------------------------------------


def _group(family, think, *, runs=10, collapsed=0, longest=100, peak=1000, calls=5, nulls=None, same=None):
    nulls = runs - collapsed if nulls is None else nulls
    return {
        "family": family, "think": think, "runs": runs, "collapsed": collapsed,
        "collapse_share": Fraction(collapsed, runs) if runs else None,
        "longest_reply": longest, "prompt_peak": peak, "num_ctx": 8192,
        "most_calls_completed": calls if runs - collapsed else None,
        "null_runs": nulls, "null_same_invocations": nulls if same is None else same,
    }


def _rules(*groups):
    return rule_values({f"{g['family']}/{g['think']}": g for g in groups})


@pytest.mark.parametrize(
    "off, on, chosen",
    [(3, 3, "off"), (3, 2, "off"), (4, 3, "off"), (4, 2, "on"), (2, 4, "off"), (10, 9, "off"), (10, 8, "on")],
)
def test_the_reasoning_mode_is_the_one_that_collapses_less_and_off_within_a_tenth(off, on, chosen):
    rules = _rules(_group("qwen3", False, collapsed=off), _group("qwen3", True, collapsed=on))
    assert rules["reasoning_mode"] == {"qwen3": chosen}


def test_a_family_without_a_reasoning_mode_gets_no_reasoning_rule():
    assert _rules(_group("llama3.1", None))["reasoning_mode"] == {}
    assert "undetermined" in _rules(_group("qwen3", False))["reasoning_mode"]["qwen3"]


@pytest.mark.parametrize("collapsed, verdict", [(5, "keep"), (6, "replace"), (0, "keep"), (10, "replace")])
def test_a_family_is_replaced_only_above_one_half(collapsed, verdict):
    rules = _rules(_group("llama3.1", None, collapsed=collapsed))
    assert rules["family_replacement"]["llama3.1"].startswith(verdict)


def test_replacement_is_judged_in_the_reasoning_mode_the_rule_selects():
    rules = _rules(_group("qwen3", False, collapsed=8), _group("qwen3", True, collapsed=4))
    assert rules["reasoning_mode"]["qwen3"] == "on"
    assert rules["family_replacement"]["qwen3"].startswith("keep")
    rules = _rules(_group("qwen3", False, collapsed=6), _group("qwen3", True, collapsed=6))
    assert rules["family_replacement"]["qwen3"].startswith("replace")


@pytest.mark.parametrize("longest, limit", [(0, 1024), (512, 1024), (513, 2048), (1024, 2048), (1025, 4096)])
def test_the_reply_limit_is_the_smallest_power_of_two_at_least_twice_the_longest_reply(longest, limit):
    assert _rules(_group("llama3.1", None, longest=longest))["reply_length_limit"] == limit


@pytest.mark.parametrize("peak, size", [(5461, 8192), (5462, 16384), (100, 8192)])
def test_the_context_stays_at_8192_up_to_two_thirds(peak, size):
    rules = _rules(_group("llama3.1", None), _group("mistral-nemo", None, peak=peak))
    assert rules["context_size"] == size and rules["context_size_conditional"] == (size != 8192)


@pytest.mark.parametrize("calls, limit", [(5, 12), (6, 12), (7, 14), (13, 26)])
def test_the_step_limit_is_twice_the_longest_completed_agent_and_at_least_12(calls, limit):
    rules = _rules(_group("llama3.1", None, calls=3), _group("qwen3", False, calls=calls))
    assert rules["step_limit"] == limit


def test_no_completed_run_leaves_the_step_limit_undetermined():
    assert _rules(_group("llama3.1", None, collapsed=10))["step_limit"] is None


def test_one_re_run_with_other_invocations_blocks_collection():
    clean = _rules(_group("llama3.1", None), _group("qwen3", False))
    assert clean["collection_blocked"] is False and clean["null_differing_invocations"] == 0
    blocked = _rules(_group("llama3.1", None), _group("qwen3", False, same=9))
    assert blocked["collection_blocked"] is True and blocked["null_differing_invocations"] == 1
    summary = {"a": {**_group("qwen3", False, same=9)}}
    assert rule_values(summary)["collection_blocked"] is True


def test_the_table_compares_replies_from_the_stored_runs(tmp_path):
    config = _first("llama3.1")
    record = _one(Model(call_ids=True), tmp_path, config=config)
    path = tmp_path / f"{record['run_id']}.pilot.json"
    stale = json.loads(path.read_text())
    stale["null"]["same_replies"] = False  # what a pilot run before the comparison was parsed stored
    path.write_text(json.dumps(stale), encoding="utf-8")
    assert load_pilot(tmp_path)[0]["null"]["same_replies"] is True
    assert summarise(load_pilot(tmp_path))["llama3.1"]["null_same_replies"] == 1

    again = tmp_path / f"{record['run_id']}.null.run.json"
    stored = json.loads(again.read_text())
    stored["transcript"]["consumer"]["messages"][-1]["content"] += " Indeed."
    again.write_text(json.dumps(stored), encoding="utf-8")
    assert load_pilot(tmp_path)[0]["null"]["same_replies"] is False
    assert json.loads(path.read_text())["null"]["same_replies"] is False  # the record is not rewritten


def test_a_blocking_re_run_is_stated_in_the_table(tmp_path):
    _stored(tmp_path, "llama3.1", lambda i, c: Model(consumer=("write", "write-elsewhere") if i == 0 else ("write",)))
    table = render_table(summarise(load_pilot(tmp_path)))
    assert "1 of 10 re-runs differ in their invocations — collection does NOT start" in table
    assert "| Re-runs repeating the sequence of invocations | 0.90 |" in table


# -- memory -----------------------------------------------------------------------


def test_driver_memory_is_read_for_the_processes_serving_the_model():
    apps = "2211, /usr/bin/gnome-shell, 310\n9042, /usr/local/bin/ollama, 8428\n"
    figures = parse_nvidia_smi(apps, "8790\n")
    assert figures["gpu_model_mib"] == 8428 and figures["gpu_used_mib"] == 8790
    assert figures["gpu_processes"][1] == {"pid": "9042", "name": "/usr/local/bin/ollama", "used_mib": 8428}
    assert parse_nvidia_smi("9042, /tmp/a, b/llama-server, 7000\n", "7100")["gpu_model_mib"] == 7000


def test_driver_memory_is_absent_rather_than_guessed():
    assert parse_nvidia_smi("", "")["gpu_model_mib"] is None
    assert parse_nvidia_smi("1, /usr/bin/Xorg, 200\n", "200")["gpu_model_mib"] is None
    assert parse_nvidia_smi("1, ollama, [N/A]\n", "[N/A]") == {
        "gpu_processes": [{"pid": "1", "name": "ollama", "used_mib": None}],
        "gpu_model_mib": None, "gpu_used_mib": None,
    }


# -- the command line --------------------------------------------------------------


def test_the_table_can_be_rebuilt_from_what_is_stored(tmp_path, capsys):
    _stored(tmp_path, "llama3.1", lambda i, c: Model())
    assert pilot.main(["--out", str(tmp_path), "--table-only"]) == 0
    printed = capsys.readouterr().out
    assert "Runs with a pilot record: 10 of 40 planned." in printed
    assert (tmp_path / "pilot-table.md").read_text(encoding="utf-8").strip() == printed.strip()
    assert pilot.main(["--out", str(tmp_path / "empty"), "--table-only"]) == 1


def test_the_plan_counted_covers_every_family_stored_or_named(tmp_path, capsys):
    _stored(tmp_path, "llama3.1", lambda i, c: Model())
    _stored(tmp_path, "granite4.1", lambda i, c: Model())
    assert pilot.main(["--out", str(tmp_path), "--table-only", "--families", "granite4.1"]) == 0
    assert "Runs with a pilot record: 20 of 20 planned." in capsys.readouterr().out
    assert pilot.main(["--out", str(tmp_path), "--table-only", "--families", "gemma4"]) == 0
    assert "Runs with a pilot record: 20 of 40 planned." in capsys.readouterr().out


def test_progress_lines_are_flushed_as_they_are_written(monkeypatch):
    import inspect

    class Stream:
        def __init__(self):
            self.text, self.flushes = "", 0

        def write(self, text):
            self.text += text

        def flush(self):
            self.flushes += 1

    stream = Stream()
    monkeypatch.setattr(sys, "stdout", stream)
    pilot._emit("[1/40] a run")
    assert stream.text == "[1/40] a run\n" and stream.flushes == 1
    assert inspect.signature(run_pilot).parameters["emit"].default is pilot._emit


def test_no_public_name_in_the_pilot_would_be_collected_as_a_test():
    assert not [name for name in dir(pilot) if name.startswith("test")]
