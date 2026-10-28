# SPDX-License-Identifier: Apache-2.0

"""Tests for harness/journal.py.

Written by Claude (Opus 5, Anthropic) at the author's request.

The journal is the evidentiary base of the whole study: MIR is computed from it,
D7(iii) is decided by comparing two of them, and §5 undertakes to publish them.
These tests therefore check the properties that make it evidence rather than the
particular field names.

Three groups carry the weight. First, tamper detection is tested as a general
property over randomly generated journals and randomly chosen mutations, rather
than against a handful of hand-written edits — a chain that caught only the
alterations its author thought of would give false assurance. Second, the
exclusions are tested in both directions: the run id and timestamps must *not*
affect the replay signature, and the tool sequence must. Third, the append-only
rule is tested at the boundary, since a journal that could be extended after
close would let a run be edited after its result was known.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

from journal import (  # noqa: E402
    GENESIS_HASH,
    Event,
    EventKind,
    Journal,
    JournalIntegrityError,
    canonical_json,
    digest_event,
)


# -- helpers ---------------------------------------------------------------


def _fixed_clock():
    """A clock that never moves, so timestamps cannot accidentally carry signal."""
    return lambda: "2026-01-01T00:00:00+00:00"


def _journal(run_id: str = "run-0001", seed: int = 42) -> Journal:
    return Journal(run_id=run_id, seed=seed, _clock=_fixed_clock())


def _populate(j: Journal, *, tools: list[str] | None = None) -> Journal:
    tools = tools or ["fs.read_file", "fs.write_file", "email.send"]
    j.state_append(entry_id="e1", author="writer", author_level=0, content="observed X")
    j.state_read(reader="consumer", reader_level=2, entry_ids=["e1"])
    for n, tool in enumerate(tools):
        inv = j.tool_invocation(agent="consumer", agent_level=2, tool=tool)
        j.gate_decision(
            invocation_index=inv.index,
            outcome="permitted" if n < len(tools) - 1 else "denied",
            reason="test",
            classification="within_mandate",
        )
    return j


def _random_journal(rng: random.Random) -> Journal:
    j = Journal(run_id=f"run-{rng.randrange(10**6):06d}", seed=rng.randrange(10**9), _clock=_fixed_clock())
    for n in range(rng.randrange(1, 12)):
        pick = rng.random()
        if pick < 0.35:
            j.state_append(
                entry_id=f"e{n}", author="writer", author_level=rng.randrange(0, 2),
                content=f"entry {n} {rng.random()}",
            )
        elif pick < 0.6:
            j.state_read(reader="consumer", reader_level=2, entry_ids=[f"e{n}"])
        else:
            inv = j.tool_invocation(
                agent="consumer", agent_level=2,
                tool=rng.choice(["fs.read_file", "db.query_readonly", "http.get"]),
            )
            j.gate_decision(
                invocation_index=inv.index,
                outcome=rng.choice(["permitted", "denied"]),
                reason="random",
                classification=rng.choice(["within_mandate", "candidate_inflation"]),
            )
    return j


# -- construction ----------------------------------------------------------


def test_journal_opens_with_run_start():
    j = _journal()
    assert len(j) == 1
    assert j.events[0].kind is EventKind.RUN_START


def test_run_start_carries_the_seed():
    j = _journal(seed=1234)
    assert j.events[0].payload["seed"] == 1234


def test_run_start_does_not_carry_the_run_id():
    """The defect this guards against breaks every ablation comparison.

    D7(iii) compares an original run with an ablation re-run. Ablation runs have
    different run ids by construction, so a run id inside a hashed payload would
    make the two differ in the one field that is guaranteed to differ.
    """
    j = _journal(run_id="run-ABC")
    assert "run_id" not in j.events[0].payload
    assert "run-ABC" not in canonical_json(j.events[0].payload)


def test_first_event_links_to_genesis():
    assert _journal().events[0].prev_hash == GENESIS_HASH


def test_seed_must_be_an_int():
    with pytest.raises(TypeError):
        Journal(run_id="r", seed="42")  # type: ignore[arg-type]


def test_bool_is_not_an_acceptable_seed():
    """bool is a subclass of int; a True seed is a bug, not a seed."""
    with pytest.raises(TypeError):
        Journal(run_id="r", seed=True)  # type: ignore[arg-type]


def test_run_id_must_be_non_empty():
    with pytest.raises(ValueError):
        Journal(run_id="", seed=1)


# -- chain integrity -------------------------------------------------------


def test_populated_journal_verifies():
    _populate(_journal()).verify()


def test_each_event_links_to_its_predecessor():
    j = _populate(_journal())
    for earlier, later in zip(j.events, j.events[1:]):
        assert later.prev_hash == earlier.digest


def test_indices_are_dense_and_ordered():
    j = _populate(_journal())
    assert [e.index for e in j.events] == list(range(len(j)))


def test_head_is_the_last_digest():
    j = _populate(_journal())
    assert j.head == j.events[-1].digest


def test_tampering_with_content_is_detected():
    j = _populate(_journal())
    victim = j.events[1]
    j._events[1] = Event(
        index=victim.index, kind=victim.kind,
        payload={**victim.payload, "content": "rewritten"},
        prev_hash=victim.prev_hash, digest=victim.digest, timestamp=victim.timestamp,
    )
    with pytest.raises(JournalIntegrityError, match="content hashes to"):
        j.verify()


def test_reordering_two_events_is_detected():
    """Ordering is what the attribution procedure reads, so it must be committed to."""
    j = _populate(_journal())
    j._events[2], j._events[3] = j._events[3], j._events[2]
    with pytest.raises(JournalIntegrityError):
        j.verify()


def test_truncating_the_chain_leaves_it_verifiable_but_shorter():
    """Truncation alone is not detectable from the chain; the head digest is.

    A prefix of a valid chain is itself a valid chain. This is a real property
    of hash chains and is recorded here rather than papered over: detection of
    truncation relies on the head digest being published, not on `verify`.
    """
    j = _populate(_journal())
    head_before = j.head
    j._events.pop()
    j.verify()
    assert j.head != head_before


def test_recomputing_a_digest_after_an_edit_still_fails_on_linkage():
    """An attacker who recomputes one digest breaks the next event's link."""
    j = _populate(_journal())
    victim = j.events[1]
    edited_payload = {**victim.payload, "content": "rewritten"}
    j._events[1] = Event(
        index=victim.index, kind=victim.kind, payload=edited_payload,
        prev_hash=victim.prev_hash,
        digest=digest_event(victim.index, victim.prev_hash, victim.kind, edited_payload),
        timestamp=victim.timestamp,
    )
    with pytest.raises(JournalIntegrityError, match="links to"):
        j.verify()


@pytest.mark.parametrize("trial", range(40))
def test_any_single_mutation_is_detected(trial):
    """Property: mutating any one field of any one event fails verification."""
    rng = random.Random(trial)
    j = _random_journal(rng)
    if len(j) < 2:
        pytest.skip("journal too short to mutate meaningfully")
    position = rng.randrange(len(j))
    victim = j.events[position]
    field = rng.choice(["index", "prev_hash", "digest", "payload"])
    if field == "index":
        mutated = Event(victim.index + 1, victim.kind, victim.payload, victim.prev_hash, victim.digest, victim.timestamp)
    elif field == "prev_hash":
        mutated = Event(victim.index, victim.kind, victim.payload, "f" * 64, victim.digest, victim.timestamp)
    elif field == "digest":
        mutated = Event(victim.index, victim.kind, victim.payload, victim.prev_hash, "f" * 64, victim.timestamp)
    else:
        mutated = Event(victim.index, victim.kind, {**victim.payload, "injected": True}, victim.prev_hash, victim.digest, victim.timestamp)
    j._events[position] = mutated
    assert not j.is_valid()


@pytest.mark.parametrize("trial", range(20))
def test_untampered_random_journals_always_verify(trial):
    """The negative control for the mutation test above."""
    _random_journal(random.Random(1000 + trial)).verify()


# -- append-only -----------------------------------------------------------


def test_closing_records_run_end():
    j = _populate(_journal())
    j.close()
    assert j.events[-1].kind is EventKind.RUN_END
    assert j.closed


def test_nothing_can_be_appended_after_close():
    j = _populate(_journal())
    j.close()
    with pytest.raises(JournalIntegrityError, match="closed"):
        j.state_append(entry_id="late", author="w", author_level=0, content="x")


def test_cannot_close_twice():
    j = _journal()
    j.close()
    with pytest.raises(JournalIntegrityError):
        j.close()


def test_events_property_is_immutable():
    """Handing out the backing list would bypass the chain entirely."""
    j = _populate(_journal())
    assert isinstance(j.events, tuple)


# -- gate decisions --------------------------------------------------------


def test_gate_decision_must_reference_a_tool_invocation():
    j = _journal()
    j.state_append(entry_id="e1", author="w", author_level=0, content="x")
    with pytest.raises(ValueError, match="not a tool invocation"):
        j.gate_decision(invocation_index=1, outcome="permitted", reason="r")


def test_gate_decision_rejects_an_out_of_range_index():
    j = _journal()
    with pytest.raises(ValueError, match="no event at index"):
        j.gate_decision(invocation_index=99, outcome="permitted", reason="r")


def test_every_invocation_can_be_paired_with_its_decision():
    j = _populate(_journal())
    invocations = {e.index for e in j.of_kind(EventKind.TOOL_INVOCATION)}
    referenced = {e.payload["invocation_index"] for e in j.of_kind(EventKind.GATE_DECISION)}
    assert invocations == referenced


# -- replay signature ------------------------------------------------------


def test_replay_signature_ignores_run_id():
    a = _populate(_journal(run_id="run-A"))
    b = _populate(_journal(run_id="run-B"))
    assert a.replay_signature() == b.replay_signature()


def test_replay_signature_ignores_timestamps():
    a = Journal(run_id="r", seed=1, _clock=lambda: "2026-01-01T00:00:00+00:00")
    b = Journal(run_id="r", seed=1, _clock=lambda: "2027-06-30T12:34:56+00:00")
    assert _populate(a).replay_signature() == _populate(b).replay_signature()


def test_replay_signature_ignores_state_content():
    """Ablation removes state entries; that alone must not change the signature.

    Otherwise every ablation would register as a behaviour change and D7(iii)
    would be satisfied vacuously for every candidate.
    """
    a = _journal()
    a.state_append(entry_id="e1", author="w", author_level=0, content="original")
    b = _journal()
    b.state_append(entry_id="e1", author="w", author_level=0, content="completely different")
    for j in (a, b):
        inv = j.tool_invocation(agent="c", agent_level=2, tool="fs.read_file")
        j.gate_decision(invocation_index=inv.index, outcome="permitted", reason="r")
    assert a.replay_signature() == b.replay_signature()


def test_replay_signature_changes_when_a_tool_changes():
    a = _populate(_journal(), tools=["fs.read_file", "fs.write_file"])
    b = _populate(_journal(), tools=["fs.read_file", "email.send"])
    assert a.replay_signature() != b.replay_signature()


def test_replay_signature_changes_when_order_changes():
    a = _populate(_journal(), tools=["fs.read_file", "fs.write_file"])
    b = _populate(_journal(), tools=["fs.write_file", "fs.read_file"])
    assert a.replay_signature() != b.replay_signature()


def test_replay_signature_changes_when_an_outcome_changes():
    a = _journal()
    inv = a.tool_invocation(agent="c", agent_level=2, tool="http.get")
    a.gate_decision(invocation_index=inv.index, outcome="permitted", reason="r")
    b = _journal()
    inv = b.tool_invocation(agent="c", agent_level=2, tool="http.get")
    b.gate_decision(invocation_index=inv.index, outcome="denied", reason="r")
    assert a.replay_signature() != b.replay_signature()


def test_replay_signature_of_an_empty_run_is_stable():
    assert _journal().replay_signature() == _journal(run_id="other").replay_signature()


# -- serialisation ---------------------------------------------------------


def test_round_trip_preserves_the_chain():
    j = _populate(_journal())
    j.close()
    restored = Journal.from_json(j.to_json())
    assert restored.head == j.head
    assert len(restored) == len(j)
    assert restored.replay_signature() == j.replay_signature()


def test_loading_a_tampered_journal_raises():
    """Loading implies verification: a journal read from disk may have been edited."""
    j = _populate(_journal())
    raw = json.loads(j.to_json())
    raw["events"][1]["payload"]["content"] = "rewritten"
    with pytest.raises(JournalIntegrityError):
        Journal.from_json(json.dumps(raw))


def test_serialisation_is_canonical():
    j = _populate(_journal())
    assert j.to_json() == j.to_json()


def test_unserialisable_payload_is_rejected_at_write_time():
    j = _journal()
    with pytest.raises(TypeError):
        j.tool_invocation(agent="c", agent_level=2, tool="fs.read_file", arguments={"x": {1, 2}})


# -- canonical json --------------------------------------------------------


def test_canonical_json_is_key_order_independent():
    assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})


def test_digest_depends_on_position():
    payload = {"x": 1}
    assert digest_event(0, GENESIS_HASH, EventKind.RUN_START, payload) != digest_event(
        1, GENESIS_HASH, EventKind.RUN_START, payload
    )


def test_digest_depends_on_predecessor():
    payload = {"x": 1}
    assert digest_event(1, "a" * 64, EventKind.STATE_READ, payload) != digest_event(
        1, "b" * 64, EventKind.STATE_READ, payload
    )


def test_digest_depends_on_kind():
    payload = {"x": 1}
    assert digest_event(1, GENESIS_HASH, EventKind.STATE_READ, payload) != digest_event(
        1, GENESIS_HASH, EventKind.STATE_APPEND, payload
    )
