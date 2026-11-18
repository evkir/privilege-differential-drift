# SPDX-License-Identifier: Apache-2.0

"""Time of collection and of ablation, from the figures of the pilot and the checks.

Implements the budget registered in PREREGISTRATION.md §5 (*Attribution
procedure*, *Budget*). Collection runs on one GPU, one agent at a time, so the
cost of the design is wall time, and this module states it before collection
begins: the time of the 6,075 runs with the replacement of collapsed runs, and
the time of the D7(iii) ablation for a range of shares of runs that carry a
candidate event.

The ablation is exhaustive. For a completed run in which the gate records at
least one candidate, the consumer is re-run once for every non-empty subset
E of the entries S, on S \\ E, and once on S unaltered: 2^n re-runs for n
entries. One set of re-runs serves every candidate of the run, since each
re-run is compared against every candidate invocation. A run with more than
``MAX_EXHAUSTIVE_ENTRIES`` entries is not ablated; its candidates are
unresolved under D7(iii) and are reported, not counted.

Inputs, all recorded in §5:

* time per run with unloading and time per consumer re-run — medians of the
  pilot table, for the model of each family in its selected reasoning mode;
* writer time, writer collapse and the number of entries a writer leaves — the
  check of the fidelity measure, 40 writers per model on the substantive
  worlds at the settings fixed for collection.

A collapsed attempt is costed at the writer's time: every collapse in the
pilot, the check of the amended briefs and the check of the fidelity measure
was a writer that stopped without an entry, after which no consumer starts.
A consumer re-run is costed at the median re-run on unaltered entries,
whatever is removed. The consumer of the 0→1 condition, at L1, was not
piloted and is costed as the L2 consumer.

Expected attempts per cell are exact under the registered replacement rule:
replicates run in index order until 135 complete or 270 have been attempted.

No external dependencies: standard library only.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from math import exp, lgamma, log
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from power import N_CATEGORIES, N_PER_CELL  # noqa: E402

N_GAP_CONDITIONS = 3
CELLS_PER_FAMILY = N_CATEGORIES * N_GAP_CONDITIONS
ATTEMPT_CAP = 270
MAX_EXHAUSTIVE_ENTRIES = 10
CANDIDATE_SHARES = (0.10, 0.20, 0.30, 0.50, 1.00)


@dataclass(frozen=True)
class FamilyFigures:
    """The measured inputs of one model, each with where it comes from in §5."""

    label: str
    run_seconds: float  # pilot table: time per run with unloading, median
    rerun_seconds: float  # pilot table: time per re-run, median
    writer_seconds: float  # fidelity check: writer time, median
    writers: int  # fidelity check: writers run
    writers_collapsed: int  # fidelity check: writers that left no entry
    entries: tuple[tuple[int, int], ...]  # fidelity check: (entries, writers)

    @property
    def collapse_share(self) -> float:
        return self.writers_collapsed / self.writers


FAMILIES: tuple[FamilyFigures, ...] = (
    FamilyFigures("gemma4:12b", 53.5, 23.4, 30.9, 40, 0, ((1, 40),)),
    FamilyFigures("granite4.1:8b", 27.5, 11.0, 15.6, 40, 4, ((1, 28), (3, 1), (5, 4), (6, 3))),
    FamilyFigures("qwen3:14b", 35.5, 14.1, 19.9, 40, 6, ((1, 26), (4, 2), (5, 6))),
)


def _binom_pmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 1.0 if k == 0 else 0.0
    if p >= 1.0:
        return 1.0 if k == n else 0.0
    return exp(
        lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + k * log(p) + (n - k) * log(1.0 - p)
    )


def _binom_cdf_below(target: int, n: int, p: float) -> float:
    """P(X < target) for X ~ Binomial(n, p)."""
    return min(1.0, sum(_binom_pmf(k, n, p) for k in range(min(target, n + 1))))


@dataclass(frozen=True)
class CellAttempts:
    attempts: float  # expected attempts executed
    completed: float  # expected completed runs
    underfill: float  # probability the cell stops at the cap short of the target

    @property
    def collapsed(self) -> float:
        return self.attempts - self.completed


def cell_attempts(collapse: float, target: int = N_PER_CELL, cap: int = ATTEMPT_CAP) -> CellAttempts:
    """Expected attempts and completed runs of one cell under the replacement rule.

    Attempts stop at the ``target``-th completion or at ``cap``, whichever
    comes first. E[attempts] = sum over a < cap of P(fewer than ``target``
    completions in the first a attempts).
    """
    if not 0.0 <= collapse < 1.0:
        raise ValueError(f"collapse share must be in [0, 1), got {collapse}")
    if not 0 < target <= cap:
        raise ValueError(f"need 0 < target <= cap, got target={target}, cap={cap}")
    success = 1.0 - collapse
    attempts = sum(_binom_cdf_below(target, a, success) for a in range(cap))
    completed = sum(min(k, target) * _binom_pmf(k, cap, success) for k in range(cap + 1))
    return CellAttempts(attempts, completed, _binom_cdf_below(target, cap, success))


def reruns_for(entries: int) -> int:
    """Consumer re-runs for a run with this many entries: 2^n, or 0 if not ablated.

    2^n − 1 re-runs on S \\ E, one per non-empty subset E, and one on S
    unaltered. Above ``MAX_EXHAUSTIVE_ENTRIES`` the run is not ablated.
    """
    if entries < 1:
        raise ValueError("a completed run has at least one entry")
    return 2**entries if entries <= MAX_EXHAUSTIVE_ENTRIES else 0


def mean_reruns(figures: FamilyFigures) -> float:
    total = sum(count for _, count in figures.entries)
    return sum(reruns_for(n) * count for n, count in figures.entries) / total


@dataclass(frozen=True)
class FamilyBudget:
    label: str
    collection_hours: float
    replacement_hours: float
    underfill: float
    mean_reruns: float
    ablation_hours: dict[float, float]  # candidate share -> hours


def family_budget(figures: FamilyFigures, shares: tuple[float, ...] = CANDIDATE_SHARES) -> FamilyBudget:
    cell = cell_attempts(figures.collapse_share)
    completed = CELLS_PER_FAMILY * cell.completed
    collapsed = CELLS_PER_FAMILY * cell.collapsed
    reruns = mean_reruns(figures)
    for share in shares:
        if not 0.0 <= share <= 1.0:
            raise ValueError(f"candidate share must be in [0, 1], got {share}")
    return FamilyBudget(
        label=figures.label,
        collection_hours=completed * figures.run_seconds / 3600.0,
        replacement_hours=collapsed * figures.writer_seconds / 3600.0,
        underfill=cell.underfill,
        mean_reruns=reruns,
        ablation_hours={
            share: completed * share * reruns * figures.rerun_seconds / 3600.0 for share in shares
        },
    )


def budget(shares: tuple[float, ...] = CANDIDATE_SHARES) -> list[FamilyBudget]:
    return [family_budget(f, shares) for f in FAMILIES]


def render_inputs() -> str:
    """The measured inputs as §5 records them."""
    head = "| Input | " + " | ".join(f"`{f.label}`" for f in FAMILIES) + " |"
    lines = [head, "|---" * (len(FAMILIES) + 1) + "|"]

    def line(name: str, values: list[str]) -> None:
        lines.append(f"| {name} | " + " | ".join(values) + " |")

    line("Time per run with unloading, s (median; pilot)", [f"{f.run_seconds:.1f}" for f in FAMILIES])
    line("Time per consumer re-run, s (median; pilot)", [f"{f.rerun_seconds:.1f}" for f in FAMILIES])
    line("Writer time, s (median; fidelity check)", [f"{f.writer_seconds:.1f}" for f in FAMILIES])
    line(
        "Writers that left no entry (fidelity check)",
        [f"{f.writers_collapsed} of {f.writers}" for f in FAMILIES],
    )
    line(
        "Entries left: count (writers) (fidelity check)",
        [", ".join(f"{n} ({c})" for n, c in f.entries) for f in FAMILIES],
    )
    return "\n".join(lines)


def render_table(shares: tuple[float, ...] = CANDIDATE_SHARES) -> str:
    """The budget table as §5 records it, hours of GPU time."""
    rows = budget(shares)
    head = "| Hours | " + " | ".join(f"`{r.label}`" for r in rows) + " | Total |"
    lines = [head, "|---" * (len(rows) + 2) + "|"]

    def line(name: str, values: list[float], fmt: str = ".1f") -> None:
        cells = [format(v, fmt) for v in values] + [format(sum(values), fmt)]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    line("Collection, completed runs", [r.collection_hours for r in rows])
    line("Collection, replaced collapses", [r.replacement_hours for r in rows])
    line("Collection, total", [r.collection_hours + r.replacement_hours for r in rows])
    lines.append(
        "| Consumer re-runs per ablated run (mean) | "
        + " | ".join(f"{r.mean_reruns:.2f}" for r in rows)
        + " | — |"
    )
    for share in shares:
        line(f"Ablation, candidate in {share:.0%} of completed runs", [r.ablation_hours[share] for r in rows])
    top = max(shares)
    line(
        f"Collection and ablation at {top:.0%}",
        [r.collection_hours + r.replacement_hours + r.ablation_hours[top] for r in rows],
    )
    return "\n".join(lines)


def main() -> None:
    print(render_inputs())
    print()
    print(render_table())
    print()
    for r in budget():
        print(f"{r.label}: probability a cell is underfilled {r.underfill:.2e}")
    print(f"re-runs for a run at the limit of {MAX_EXHAUSTIVE_ENTRIES} entries: "
          f"{reruns_for(MAX_EXHAUSTIVE_ENTRIES)}")


if __name__ == "__main__":
    main()
