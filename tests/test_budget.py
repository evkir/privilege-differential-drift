# SPDX-License-Identifier: Apache-2.0

"""Tests for analysis/budget.py.

Written by Claude (Opus 5, Anthropic) at the author's request.

The expected number of attempts per cell is an exact sum over binomial
probabilities, so it is checked against simulation of the replacement rule
itself under a fixed seed rather than against a restatement of the sum. The
count of re-runs is checked against an enumeration of subsets. The inputs are
held to the figures §5 records from the pilot and the check of the fidelity
measure, and the tables the module prints are required to appear in §5 as
they are, so a changed input or a changed rule cannot leave the registered
budget behind.
"""

from __future__ import annotations

import random
import re
import sys
from itertools import combinations
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from budget import (  # noqa: E402
    ATTEMPT_CAP,
    CANDIDATE_SHARES,
    CELLS_PER_FAMILY,
    FAMILIES,
    MAX_EXHAUSTIVE_ENTRIES,
    N_GAP_CONDITIONS,
    budget,
    cell_attempts,
    family_budget,
    mean_reruns,
    render_inputs,
    render_table,
    reruns_for,
)
from power import N_CATEGORIES, N_MODEL_FAMILIES, N_PER_CELL  # noqa: E402

DOC = Path(__file__).resolve().parents[1] / "PREREGISTRATION.md"


def _doc() -> str:
    return DOC.read_text(encoding="utf-8")


def _simulate_cell(collapse: float, rng: random.Random, target: int, cap: int) -> tuple[int, int]:
    attempts = completed = 0
    while completed < target and attempts < cap:
        attempts += 1
        if rng.random() >= collapse:
            completed += 1
    return attempts, completed


# --- the replacement rule --------------------------------------------------


def test_no_collapse_means_exactly_the_target() -> None:
    cell = cell_attempts(0.0)
    assert cell.attempts == pytest.approx(N_PER_CELL)
    assert cell.completed == pytest.approx(N_PER_CELL)
    assert cell.collapsed == pytest.approx(0.0)
    assert cell.underfill == 0.0


@pytest.mark.parametrize("collapse", [0.10, 0.15, 0.30])
def test_uncapped_cells_follow_the_negative_binomial_mean(collapse: float) -> None:
    cell = cell_attempts(collapse)
    assert cell.attempts == pytest.approx(N_PER_CELL / (1.0 - collapse), rel=1e-6)
    assert cell.completed == pytest.approx(N_PER_CELL, rel=1e-9)
    assert cell.underfill < 1e-6


@pytest.mark.parametrize("collapse", [0.15, 0.48, 0.55])
def test_expected_attempts_match_simulation_of_the_rule(collapse: float) -> None:
    rng = random.Random(20261118)
    reps = 4000
    sims = [_simulate_cell(collapse, rng, N_PER_CELL, ATTEMPT_CAP) for _ in range(reps)]
    attempts = sum(a for a, _ in sims) / reps
    completed = sum(c for _, c in sims) / reps
    underfill = sum(c < N_PER_CELL for _, c in sims) / reps
    cell = cell_attempts(collapse)
    assert attempts == pytest.approx(cell.attempts, abs=0.6)
    assert completed == pytest.approx(cell.completed, abs=0.6)
    assert underfill == pytest.approx(cell.underfill, abs=0.03)


def test_the_cap_binds_when_most_attempts_collapse() -> None:
    cell = cell_attempts(0.60)
    assert cell.attempts == pytest.approx(ATTEMPT_CAP, abs=0.01)
    assert cell.completed == pytest.approx(ATTEMPT_CAP * 0.40, abs=0.01)
    assert cell.underfill > 0.99


@pytest.mark.parametrize("collapse", [-0.1, 1.0, 1.5])
def test_collapse_share_outside_range_is_refused(collapse: float) -> None:
    with pytest.raises(ValueError):
        cell_attempts(collapse)


def test_target_above_cap_is_refused() -> None:
    with pytest.raises(ValueError):
        cell_attempts(0.1, target=300, cap=270)


# --- exhaustive ablation ---------------------------------------------------


@pytest.mark.parametrize("n", range(1, MAX_EXHAUSTIVE_ENTRIES + 1))
def test_reruns_are_every_non_empty_subset_plus_the_unaltered_state(n: int) -> None:
    entries = range(n)
    subsets = sum(1 for k in range(1, n + 1) for _ in combinations(entries, k))
    assert reruns_for(n) == subsets + 1


def test_a_run_above_the_limit_is_not_ablated() -> None:
    assert reruns_for(MAX_EXHAUSTIVE_ENTRIES) == 1024
    assert reruns_for(MAX_EXHAUSTIVE_ENTRIES + 1) == 0
    assert reruns_for(28) == 0


def test_a_run_without_entries_is_refused() -> None:
    with pytest.raises(ValueError):
        reruns_for(0)


def test_mean_reruns_by_hand() -> None:
    by_label = {f.label: mean_reruns(f) for f in FAMILIES}
    assert by_label["gemma4:12b"] == pytest.approx(2.0)
    assert by_label["granite4.1:8b"] == pytest.approx((28 * 2 + 8 + 4 * 32 + 3 * 64) / 36)
    assert by_label["qwen3:14b"] == pytest.approx((26 * 2 + 2 * 16 + 6 * 32) / 34)


def test_ablation_is_linear_in_the_candidate_share() -> None:
    for figures in FAMILIES:
        result = family_budget(figures, (0.0, 0.25, 0.5, 1.0))
        assert result.ablation_hours[0.0] == 0.0
        assert result.ablation_hours[0.5] == pytest.approx(2 * result.ablation_hours[0.25])
        assert result.ablation_hours[1.0] == pytest.approx(2 * result.ablation_hours[0.5])


def test_candidate_share_outside_range_is_refused() -> None:
    with pytest.raises(ValueError):
        family_budget(FAMILIES[0], (1.2,))


def test_collection_hours_cover_the_registered_design() -> None:
    assert CELLS_PER_FAMILY * N_PER_CELL * N_MODEL_FAMILIES == 6075
    for figures, result in zip(FAMILIES, budget()):
        runs = result.collection_hours * 3600.0 / figures.run_seconds
        assert runs == pytest.approx(CELLS_PER_FAMILY * N_PER_CELL, rel=1e-9)


# --- inputs against §5 -----------------------------------------------------


def test_design_constants_agree_with_power() -> None:
    assert CELLS_PER_FAMILY == N_CATEGORIES * N_GAP_CONDITIONS
    assert len(FAMILIES) == N_MODEL_FAMILIES


def test_attempt_cap_matches_section_5() -> None:
    assert re.findall(r"a cell stops at (\d+) attempts", _doc()) == [str(ATTEMPT_CAP)]


def test_entry_limit_matches_section_5() -> None:
    found = re.findall(r"with more than (\d+) entries is not ablated", _doc())
    assert found == [str(MAX_EXHAUSTIVE_ENTRIES)]


def test_writers_with_entries_match_the_fidelity_check_table() -> None:
    row = re.findall(r"^\| Finished with entries \| (\d+) \| (\d+) \| (\d+) \|$", _doc(), re.M)
    assert len(row) == 1
    finished = [sum(c for _, c in f.entries) for f in FAMILIES]
    assert finished == [int(x) for x in row[0]]
    assert finished == [f.writers - f.writers_collapsed for f in FAMILIES]


def _pilot_row(label: str) -> list[str]:
    rows = re.findall(rf"^\| {re.escape(label)} \| (.+) \|$", _doc(), re.M)
    assert rows, f"no row {label!r} in §5"
    return [cell.strip() for cell in rows[0].split("|")]


def test_run_and_rerun_times_match_the_pilot_table() -> None:
    # pilot columns: gemma4 off, gemma4 on, granite4.1, llama3.1, mistral-nemo, qwen3 off, qwen3 on
    selected = (0, 2, 5)
    runs = _pilot_row("Time per run with unloading, s (median)")
    reruns = _pilot_row("Time per re-run, s (median)")
    assert [float(runs[i]) for i in selected] == [f.run_seconds for f in FAMILIES]
    assert [float(reruns[i]) for i in selected] == [f.rerun_seconds for f in FAMILIES]


def test_section_5_records_the_inputs_as_the_module_prints_them() -> None:
    assert render_inputs() in _doc()


def test_section_5_records_the_budget_as_the_module_prints_it() -> None:
    assert render_table() in _doc()


def test_registered_shares_end_at_the_bound() -> None:
    assert max(CANDIDATE_SHARES) == 1.0
    assert list(CANDIDATE_SHARES) == sorted(CANDIDATE_SHARES)
