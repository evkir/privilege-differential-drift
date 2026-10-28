# SPDX-License-Identifier: Apache-2.0

"""Append-only, hash-chained event journal for a single experimental run.

Implements the logging requirement of PREREGISTRATION.md §5 ("Append-only,
hash-chained event log capturing every state write, every state read, every
tool invocation and every gate decision, with run seed. Logs are published with
results") and the append-only, hash-chained property D5 requires of shared
state.

Three design decisions are recorded here rather than left to prose.

**The chain covers ordering, not just content.** Each event's digest is taken
over its own payload *and* the digest of the event before it, so the journal
commits to a sequence rather than to a set. This matters because the causal
claim in D7(iii) is about what an agent did *after* reading particular state:
a journal that detected edits to individual entries but not their reordering
would leave the ordering free to be adjusted after the fact, and ordering is
precisely what the attribution procedure reads.

**The run identifier is not part of the chain.** D7(iii) is decided by
re-running a consumer against ``S \\ E`` under a fixed seed and comparing what
it does to the original run. Ablation runs are different runs and therefore
carry different run identifiers by construction. If the identifier entered any
hashed payload, every ablation comparison would differ in the one field
guaranteed to differ, and `replay_signature` would never match even for
behaviour that was in fact identical. The seed is hashed; the run id is carried
beside the chain as metadata.

**Wall-clock time is metadata too, for the same reason.** Timestamps are
recorded because the published logs should be readable as a trace, but they are
excluded from the digest: an ablation run executes at a different moment, and
replay comparison must not depend on that.

The practical consequence is a two-tier integrity model. `verify` answers "has
this journal been altered since it was written", and covers everything that is
hashed. `replay_signature` answers the narrower question the attribution
procedure actually asks — "did the agent do the same things in the same order"
— and is deliberately blind to run identity, timing, and the content of state
entries that ablation removes.

No external dependencies: standard library only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Iterator

__all__ = [
    "EventKind",
    "Event",
    "Journal",
    "JournalIntegrityError",
    "GENESIS_HASH",
    "canonical_json",
    "digest_event",
]

#: The predecessor digest of the first event. A fixed, recognisable constant
#: rather than an empty string, so that a truncated chain cannot be mistaken for
#: a short but well-formed one.
GENESIS_HASH: str = "0" * 64


class JournalIntegrityError(Exception):
    """Raised when a journal fails verification, or an append-only rule is broken."""


class EventKind(Enum):
    """Every kind of event §5 requires the journal to capture.

    ``STATE_APPEND`` and ``STATE_READ`` are the D5 shared-state channel.
    ``TOOL_INVOCATION`` is an attempted invocation, recorded whether or not the
    gate permitted it. ``GATE_DECISION`` is the gate's ruling on that attempt,
    recorded as its own event so that an attempt and its disposition cannot be
    separated in the log.
    """

    RUN_START = "run_start"
    RUN_END = "run_end"
    STATE_APPEND = "state_append"
    STATE_READ = "state_read"
    TOOL_INVOCATION = "tool_invocation"
    GATE_DECISION = "gate_decision"


def canonical_json(payload: Any) -> str:
    """Serialise a payload so that equal payloads always produce equal bytes.

    Keys are sorted and separators are fixed: without this, two journals
    recording the same events could hash differently because a dict happened to
    be built in a different order, and verification would report tampering where
    there was none.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_event(index: int, prev_hash: str, kind: EventKind, payload: Any) -> str:
    """The digest committing to one event and to its position in the chain.

    The index is included so that an event cannot be moved to another position
    without detection even if the surrounding digests were recomputed.
    """
    material = canonical_json(
        {"index": index, "prev": prev_hash, "kind": kind.value, "payload": payload}
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Event:
    """One journal entry.

    ``timestamp`` is recorded for readability and deliberately excluded from
    ``digest`` — see the module docstring on ablation comparison.
    """

    index: int
    kind: EventKind
    payload: dict[str, Any]
    prev_hash: str
    digest: str
    timestamp: str

    def recompute(self) -> str:
        return digest_event(self.index, self.prev_hash, self.kind, self.payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind.value,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
            "digest": self.digest,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Event":
        return cls(
            index=raw["index"],
            kind=EventKind(raw["kind"]),
            payload=raw["payload"],
            prev_hash=raw["prev_hash"],
            digest=raw["digest"],
            timestamp=raw["timestamp"],
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Journal:
    """The event log of one run.

    ``run_id`` and the clock are injectable so that tests are deterministic and
    so that an ablation run can be given its own identity without that identity
    reaching the chain.
    """

    run_id: str
    seed: int
    _events: list[Event] = field(default_factory=list, repr=False)
    _closed: bool = field(default=False, repr=False)
    _clock: Any = field(default=_now, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError(f"seed must be an int, got {type(self.seed).__name__}")
        if not self.run_id:
            raise ValueError("run_id must be a non-empty string")
        # The seed is hashed; the run id is not. See the module docstring.
        self._append(EventKind.RUN_START, {"seed": self.seed})

    # -- construction ----------------------------------------------------

    def _append(self, kind: EventKind, payload: dict[str, Any]) -> Event:
        if self._closed:
            raise JournalIntegrityError(
                "journal is closed; a run that has ended cannot record further events"
            )
        index = len(self._events)
        prev_hash = self._events[-1].digest if self._events else GENESIS_HASH
        # Round-trip the payload through canonical JSON at write time so that a
        # value which cannot be serialised is rejected here, where the caller can
        # see which event caused it, rather than at publication time.
        payload = json.loads(canonical_json(payload))
        event = Event(
            index=index,
            kind=kind,
            payload=payload,
            prev_hash=prev_hash,
            digest=digest_event(index, prev_hash, kind, payload),
            timestamp=self._clock(),
        )
        self._events.append(event)
        return event

    def state_append(self, *, entry_id: str, author: str, author_level: int, content: str) -> Event:
        """Record a write into shared state (D5)."""
        return self._append(
            EventKind.STATE_APPEND,
            {
                "entry_id": entry_id,
                "author": author,
                "author_level": author_level,
                "content_digest": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "content": content,
            },
        )

    def state_read(self, *, reader: str, reader_level: int, entry_ids: Iterable[str]) -> Event:
        """Record a read of shared state (D5).

        The entries read are recorded as an ordered list rather than a set:
        M4 counts hops between the write of a causal entry and the inflation
        event, which requires knowing what was read when.
        """
        return self._append(
            EventKind.STATE_READ,
            {"reader": reader, "reader_level": reader_level, "entry_ids": list(entry_ids)},
        )

    def tool_invocation(
        self, *, agent: str, agent_level: int, tool: str, arguments: dict[str, Any] | None = None
    ) -> Event:
        """Record an attempted tool invocation, permitted or not."""
        return self._append(
            EventKind.TOOL_INVOCATION,
            {
                "agent": agent,
                "agent_level": agent_level,
                "tool": tool,
                "arguments": arguments or {},
            },
        )

    def gate_decision(
        self,
        *,
        invocation_index: int,
        outcome: str,
        reason: str,
        classification: str | None = None,
    ) -> Event:
        """Record the gate's ruling on the invocation at ``invocation_index``."""
        if not 0 <= invocation_index < len(self._events):
            raise ValueError(f"no event at index {invocation_index}")
        if self._events[invocation_index].kind is not EventKind.TOOL_INVOCATION:
            raise ValueError(
                f"event {invocation_index} is {self._events[invocation_index].kind.value}, "
                "not a tool invocation"
            )
        return self._append(
            EventKind.GATE_DECISION,
            {
                "invocation_index": invocation_index,
                "outcome": outcome,
                "reason": reason,
                "classification": classification,
            },
        )

    def close(self, *, status: str = "completed") -> Event:
        """Record the end of the run. Nothing may be appended afterwards."""
        event = self._append(EventKind.RUN_END, {"status": status})
        self._closed = True
        return event

    # -- inspection ------------------------------------------------------

    @property
    def events(self) -> tuple[Event, ...]:
        """The chain, as an immutable snapshot.

        A tuple rather than the backing list: handing out the list would let a
        caller append or reorder events without going through `_append`, which
        is the only place the chain is maintained.
        """
        return tuple(self._events)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def head(self) -> str:
        """The digest of the most recent event — the journal's current commitment."""
        return self._events[-1].digest if self._events else GENESIS_HASH

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    def of_kind(self, kind: EventKind) -> tuple[Event, ...]:
        return tuple(e for e in self._events if e.kind is kind)

    def verify(self) -> None:
        """Check that the journal has not been altered since it was written.

        Raises `JournalIntegrityError` naming the first event that fails. Checks
        every property the chain commits to: position, linkage, and content.
        """
        expected_prev = GENESIS_HASH
        for position, event in enumerate(self._events):
            if event.index != position:
                raise JournalIntegrityError(
                    f"event at position {position} carries index {event.index}"
                )
            if event.prev_hash != expected_prev:
                raise JournalIntegrityError(
                    f"event {position} links to {event.prev_hash[:12]}…, "
                    f"expected {expected_prev[:12]}…"
                )
            recomputed = event.recompute()
            if recomputed != event.digest:
                raise JournalIntegrityError(
                    f"event {position} ({event.kind.value}) has digest "
                    f"{event.digest[:12]}…, content hashes to {recomputed[:12]}…"
                )
            expected_prev = event.digest
        if self._events and self._events[0].kind is not EventKind.RUN_START:
            raise JournalIntegrityError("journal does not begin with a run_start event")

    def is_valid(self) -> bool:
        try:
            self.verify()
        except JournalIntegrityError:
            return False
        return True

    def replay_signature(self) -> str:
        """A digest of what the agents *did*, blind to run identity and timing.

        This is the comparison D7(iii) needs. Re-running a consumer against
        ``S \\ E`` produces a different run id, different timestamps, and a
        different set of state entries — none of which bear on whether the agent
        took the same actions. The signature therefore covers the ordered
        sequence of tool invocations and their gate outcomes, and nothing else.

        Equal signatures mean the ablation did not change behaviour; a changed
        signature is what makes an entry set a candidate causal set.
        """
        steps: list[dict[str, Any]] = []
        outcomes = {
            e.payload["invocation_index"]: e.payload["outcome"]
            for e in self.of_kind(EventKind.GATE_DECISION)
        }
        for event in self._events:
            if event.kind is not EventKind.TOOL_INVOCATION:
                continue
            steps.append(
                {
                    "agent_level": event.payload["agent_level"],
                    "tool": event.payload["tool"],
                    "outcome": outcomes.get(event.index),
                }
            )
        return hashlib.sha256(canonical_json(steps).encode("utf-8")).hexdigest()

    # -- serialisation ---------------------------------------------------

    def to_json(self) -> str:
        """Serialise for publication alongside results, per §5."""
        return canonical_json(
            {
                "run_id": self.run_id,
                "seed": self.seed,
                "closed": self._closed,
                "head": self.head,
                "events": [e.to_dict() for e in self._events],
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> "Journal":
        """Load a published journal and verify it before returning it.

        Loading implies verification: a caller who reads a journal from disk is
        reading something that may have been altered in transit, and the only
        safe default is to refuse to hand back a chain that does not check out.
        """
        data = json.loads(raw)
        journal = cls.__new__(cls)
        journal.run_id = data["run_id"]
        journal.seed = data["seed"]
        journal._events = [Event.from_dict(e) for e in data["events"]]
        journal._closed = data["closed"]
        journal._clock = _now
        journal.verify()
        return journal
