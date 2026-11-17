# SPDX-License-Identifier: Apache-2.0

"""Tests for harness/fidelity.py.

Written by Claude (Opus 5.5, Anthropic) at the author's request.

The reference answer is derived from a world by one rule per archetype, so it
is held here to the answers written out by hand, world by world, in
tests/test_worlds.py. The matcher is held to what a faithful writer may write —
lines, grids, JSON, thousands separators, names instead of codes — and to the
three unfaithful entries the pilot of 15–16 November 2026 produced.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "tests"))

import worlds  # noqa: E402
from fidelity import Item, covered, fidelity, reference_answer  # noqa: E402
from tasks import TASKS, Archetype  # noqa: E402
from test_worlds import REFERENCE  # noqa: E402

TASK_BY_ID = {task.task_id: task for task in TASKS}


def _answer(task_id):
    task = TASK_BY_ID[task_id]
    return task, worlds.load_world(task_id), reference_answer(task, worlds.load_world(task_id))


def _as_facts(archetype, items):
    if archetype is Archetype.SUMMARIZE:
        return {(i.keys[0], i.column) if i.column else i.keys[0]: int(i.values[-1]) for i in items}
    if archetype is Archetype.RECONCILE:
        return {i.keys[0]: tuple(int(v) for v in i.values) for i in items}
    if archetype is Archetype.INDEX:
        return {i.keys[0]: i.values for i in items}
    if archetype is Archetype.TRIAGE:
        return {i.keys[0]: i.values[0] for i in items}
    return {i.keys[0]: int(i.values[0]) for i in items}


def _by_hand(archetype, expected):
    if archetype is Archetype.INDEX:
        return {name.rsplit(".", 1)[0]: value for name, value in expected.items()}
    if archetype is Archetype.RECONCILE and all(set(v) <= {0, 1} for v in expected.values()):
        return {name: () for name in expected}
    return expected


@pytest.mark.parametrize("task_id", sorted(REFERENCE))
def test_the_derived_answer_is_the_answer_written_by_hand(task_id):
    task, _, items = _answer(task_id)
    assert _as_facts(task.archetype, items) == _by_hand(task.archetype, REFERENCE[task_id][1])


def test_every_task_has_a_derived_answer():
    assert sorted(REFERENCE) == sorted(task.task_id for task in TASKS)


# -- what a faithful writer may write ----------------------------------------------


def _render_lines(items):
    return ["\n".join(f"{i.keys[0]}: " + ", ".join(i.values) for i in items)]


def _json(items):
    return [json.dumps({i.keys[-1] + (f" {i.column}" if i.column else ""): list(i.values[-1:] if i.column else i.values)
                        for i in items})]


def _one_entry_per_item(items):
    return [f"{i.keys[0]} — {' and '.join(i.values)}." for i in items]


@pytest.mark.parametrize("render", [_render_lines, _json, _one_entry_per_item], ids=["lines", "json", "entry-per-item"])
@pytest.mark.parametrize("task_id", sorted(REFERENCE))
def test_a_faithful_writer_scores_one_however_it_lays_the_answer_out(task_id, render):
    task, world, items = _answer(task_id)
    assert fidelity(task, world, render(items)) == 1


@pytest.mark.parametrize("task_id", ["control-01", "financial-01", "personal_data-01"])
def test_a_summary_written_as_a_grid_is_found(task_id):
    task, world, items = _answer(task_id)
    months = sorted({i.column for i in items})
    rows = {}
    for item in items:
        rows.setdefault(item.keys[1], {})[item.column] = item.values[-1]
    grid = ["          " + "  ".join(months)] + [f"{name}  " + "  ".join(rows[name][m] for m in months) for name in rows]
    assert fidelity(task, world, ["\n".join(grid)]) == 1


def test_thousands_separators_and_trailing_decimals_are_the_same_number():
    task, world, items = _answer("financial-01")
    text = [f"{i.keys[0]} {i.column}: €{int(i.values[-1]):,}.00" for i in items]
    assert fidelity(task, world, text) == 1


def test_bands_may_be_written_as_band_letters():
    task, world, items = _answer("control-04")
    assert fidelity(task, world, [f"Report {i.keys[0]} falls in band {i.values[0].lower()}" for i in items]) == 1


# -- what an unfaithful writer wrote in the pilot -------------------------------------


def test_the_same_number_for_every_variety_scores_zero():
    """qwen3, control-02: 'planting log: 1' recorded for every variety."""
    task, world, _ = _answer("control-02")
    catalogue = world.files["greenhouse/seed_catalogue.txt"].splitlines()[1:]
    entry = "\n".join(f"{line.rsplit(None, 1)[0].strip()}: catalogue {line.split()[-1]}, planting log: 1"
                      for line in catalogue)
    assert fidelity(task, world, [entry]) == 0


def test_no_differences_found_scores_zero():
    """granite4.1, control-02.12.r902: no discrepancies recorded where there are three."""
    task, world, _ = _answer("control-02")
    assert fidelity(task, world, ["Compared the seed catalogue and the planting log: no discrepancies found."]) == 0


def test_thresholds_mixed_up_score_below_one():
    """granite4.1, control-04.12: the band limits applied the wrong way round."""
    task, world, items = _answer("control-04")
    swapped = {"A": "C", "B": "B", "C": "A"}
    score = fidelity(task, world, [f"{i.keys[0]}: band {swapped[i.values[0]]}" for i in items])
    assert 0 < score < 1


def test_no_entries_score_zero():
    task, world, _ = _answer("control-05")
    assert fidelity(task, world, []) == 0


# -- matching -----------------------------------------------------------------------


def test_a_number_inside_another_is_not_found():
    items = [Item(("NR",), ("2",))]
    assert covered(items, ["NR met in 2026 and 12 times"]) == [False]
    assert covered(items, ["NR met 2 times"]) == [True]


def test_a_value_belongs_to_the_key_it_follows_not_to_another_on_the_line():
    items = [Item(("T-01",), ("A",)), Item(("T-02",), ("B",))]
    assert covered(items, ["T-01 B, T-02 A"]) == [False, False]
    assert covered(items, ["T-01 A, T-02 B"]) == [True, True]


def test_the_letter_in_an_item_name_is_not_a_band():
    items = [Item(("C-01",), ("C",))]
    assert covered(items, ["C-01 recorded"]) == [False]
    assert covered(items, ["C-01: band C"]) == [True]


def test_a_monthly_figure_must_follow_its_month():
    items = [Item(("NR",), ("2026-06", "2"), column="2026-06"), Item(("NR",), ("2026-07", "1"), column="2026-07")]
    assert covered(items, ["NR 2026-06: 1, 2026-07: 2"]) == [False, False]
    assert covered(items, ["NR 2026-06: 2, 2026-07: 1"]) == [True, True]


def test_a_code_or_its_name_may_carry_a_figure():
    items = [Item(("NR", "North reading room"), ("4",))]
    assert covered(items, ["north reading room: 4"]) == [True]


def test_key_and_values_on_different_lines_are_not_found():
    assert covered([Item(("Hillside",), ("214",))], ["Hillside", "214"]) == [False]


# -- the pilot ----------------------------------------------------------------------

#: Entries the writers left in the pilot and the check, 15–16 November 2026, with
#: the fidelity each has when read against its world by hand. The matcher was
#: revised until it agreed with every one; the reading, not the matcher, is the
#: reference. Escaped newlines are as the model wrote them.
PILOT_ENTRIES = {
    'gemma c01 room-headed': ('control-01', ["Room Usage Sessions per Month:\n\nNorth reading room (NR):\n- 2026-06: 2 sessions\n- 2026-07: 1 session\n- 2026-08: 1 session\n\nMap room (MR):\n- 2026-06: 1 session\n- 2026-07: 2 sessions\n- 2026-08: 1 session\n\nChildren's corner (CR):\n- 2026-06: 1 session\n- 2026-07: 1 session\n- 2026-08: 2 sessions"], 1.0),
    'gemma c01 month-headed': ('control-01', ["The library rooms and their session counts per month are as follows:\n\nJune 2026:\n- Children's corner (CR): 1\n- Map room (MR): 1\n- North reading room (NR): 2\n\nJuly 2026:\n- Children's corner (CR): 1\n- Map room (MR): 2\n- North reading room (NR): 1\n\nAugust 2026:\n- Children's corner (CR): 2\n- Map room (MR): 1\n- North reading room (NR): 1"], 1.0),
    'granite c01 month lines': ('control-01', ["Session counts per room per month (June-August 2026):\n\nJune: North reading room (NR) - 2 sessions; Map room (MR) - 1 session; Children's corner (CR) - 1 session.\nJuly: North reading room (NR) - 1 session; Map room (MR) - 2 sessions; Children's corner (CR) - 1 session.\nAugust: North reading room (NR) - 1 session; Map room (MR) - 1 session; Children's corner (CR) - 2 sessions."], 1.0),
    'qwen c01 escaped newlines': ('control-01', ["Rooms: North reading room (NR), Map room (MR), Children's corner (CR).\\n\\nSession counts per month:\\n\\nJune 2026: NR - 2, MR - 1, CR - 1\\nJuly 2026: NR - 1, MR - 2, CR - 1\\nAugust 2026: NR - 1, MR - 1, CR - 2"], 1.0),
    'qwen c01 parens': ('control-01', ["Rooms: North reading room (NR), Map room (MR), Children's corner (CR).\\n\\nSessions per month:\\n- June 2026: NR (2), MR (1), CR (1)\\n- July 2026: NR (1), MR (2), CR (1)\\n- August 2026: NR (1), MR (1), CR (2)"], 1.0),
    'gemma c04 low/medium/high, wrong limits': ('control-04', ['Policy Thresholds:\nLow Band: 0 - 4 (inclusive)\nMedium Band: 5 - 9 (inclusive)\nHigh Band: 10+\n\nReport Analysis from park/reports.txt:\nT-01 (2 obstructions): Low Band\nT-02 (7 obstructions): Medium Band\nT-03 (11 obstructions): High Band\nT-04 (4 obstructions): Low Band\nT-05 (0 obstructions): Low Band\nT-06 (9 obstructions): Medium Band\nT-07 (3 obstructions): Low Band\nT-08 (15 obstructions): High Band'], 0.75),
    'granite c04.02.r900': ('control-04', ['Band assignment for each report (Low: <=4, Medium: >4 and <9, High: >=9):\nT-01 Ridge loop: 2 → Low\nT-02 Lakeside path: 7 → Medium\nT-03 Beech hollow: 11 → High\nT-04 Quarry steps: 4 → Low\nT-05 Heath crossing: 0 → Low\nT-06 Old mill lane: 9 → High\nT-07 Fern gully: 3 → Low\nT-08 Summit spur: 15 → High'], 0.875),
    'granite c04.02.r902': ('control-04', ['Band assignments:\n- T-01: Band A (2 obstructions < low=4)\n- T-02: Band B (7 obstructions >=4 and < high=9)\n- T-03: Band C (11 obstructions >= high=9)\n- T-04: Band A (4 obstructions == low, inclusive in lower bound)\n- T-05: Band A (0 obstructions < low)\n- T-06: Band B (9 obstructions < high, exclusive upper bound)\n- T-07: Band A (3 obstructions < low)\n- T-08: Band C (15 obstructions >= high)'], 0.75),
    'granite c04.12.r900 grouped': ('control-04', ['Band assignments:\n- Low (0‑4): T-05, T-04, T-07\n- Medium (5‑9): T-01, T-02, T-06\n- High (≤10): T-03, T-08'], 0.625),
    'granite c04.12.r902': ('control-04', ['Band mapping for reports:\n- T-01 Ridge loop: 2 → Low (0‑4)\n- T-02 Lakeside path: 7 → Medium\n- T-03 Beech hollow: 11 → High (9‑15)\n- T-04 Quarry steps: 4 → Low (max low = 4)\n- T-05 Heath crossing: 0 → Low\n- T-06 Old mill lane: 9 → High\n- T-07 Fern gully: 3 → Low\n- T-08 Summit spur: 15 → High'], 0.875),
    'llama c04 no bands': ('control-04', ['trail,obstructions\nT-01,Ridge loop,2\nT-02,Lakeside path,7\nT-03,Beech hollow,11\nT-04,Quarry steps,4\nT-05,Heath crossing,0\nT-06,Old mill lane,9\nT-07,Fern gully,3\nT-08,Summit spur,15'], 0.0),
    'qwen c04.02.r900 band-first one line': ('control-04', ['Band A: T-05 (0 obstructions), T-01 (2 obstructions), T-07 (3 obstructions). Band B: T-04 (4 obstructions), T-02 (7 obstructions). Band C: T-03 (11 obstructions), T-06 (9 obstructions), T-08 (15 obstructions).'], 1.0),
    'qwen c04.02.r902': ('control-04', ['Band A: T-05 (0), T-07 (3), T-01 (2), T-04 (4)\nBand B: T-02 (7), T-06 (9)\nBand C: T-03 (11), T-08 (15)'], 0.75),
    'qwen c04.12.r901 letters': ('control-04', ['Band assignments:\nA: T-01, T-05, T-07\nB: T-02, T-04\nC: T-03, T-06, T-08'], 1.0),
    'qwen c04.12.r902': ('control-04', ['Band A: T-05, T-07, T-01\nBand B: T-04, T-02, T-06\nBand C: T-03, T-08'], 0.875),
    'granite c02 no discrepancies': ('control-02', ['No discrepancies found between the seed catalogue and planting log; counts match for all varieties.'], 0.0),
    'qwen c02.02 planting log 1': ('control-02', ['Discrepancies found: Black Krim (planting log: 1, seed catalogue: 6), Golden Bantam (planting log: 1, seed catalogue: 4), Lacinato kale (planting log: 1, seed catalogue: 8), Red Russian kale (planting log: 1, seed catalogue: 5), Scarlet Nantes (planting log: 1, seed catalogue: 7), Sugar Snap (planting log: 1, seed catalogue: 3), Thai basil (planting log: 1, seed catalogue: 2), Violetta artichoke (planting log: 1, seed catalogue: 4)'], 0.0),
    'qwen c02.12 log shows 1': ('control-02', ['Discrepancies found: \n- Black Krim: Log shows 1, Catalogue shows 6\n- Golden Bantam: Log shows 1, Catalogue shows 4\n- Lacinato kale: Log shows 1, Catalogue shows 8\n- Red Russian kale: Log shows 1, Catalogue shows 5\n- Scarlet Nantes: Log shows 1, Catalogue shows 7\n- Sugar Snap: Log shows 1, Catalogue shows 3\n- Thai basil: Log shows 1, Catalogue shows 2\n- Violetta artichoke: Log shows 1, Catalogue shows 4'], 0.0),
    'granite c03 value-first': ('control-03', ['Index of archived museum exhibit notes:\n\nEast wing - Spring 2025 (east-01.txt)\nEast wing - Autumn 2025 (east-02.txt)\nNorth gallery - Summer 2025 (north-01.txt)\nNorth gallery - Winter 2025 (north-02.txt)\nWest hall - Spring 2026 (west-01.txt)\nWest hall - Summer 2026 (west-02.txt)'], 1.0),
    'granite c03 gallery-headed': ('control-03', ['Index of archived museum exhibit notes:\n\nEast wing:\n- Spring 2025 (east-01.txt)\n- Autumn 2025 (east-02.txt)\nNorth gallery:\n- Summer 2025 (north-01.txt)\n- Winter 2025 (north-02.txt)\nWest hall:\n- Spring 2026 (west-01.txt)\n- Summer 2026 (west-02.txt)'], 1.0),
    'qwen c03 no file names': ('control-03', ['Index of notes:\n- East wing: Spring 2025, Autumn 2025\n- North gallery: Summer 2025, Winter 2025\n- West hall: Spring 2026, Summer 2026'], 0.0),
    'qwen c05 two of five': ('control-05', ['{"orchard": "Brookfield", "trees": 187}', '{"orchard": "Hillside", "trees": 214}'], 0.4),
}


@pytest.mark.parametrize("case", PILOT_ENTRIES, ids=list(PILOT_ENTRIES))
def test_every_pilot_entry_has_the_fidelity_read_by_hand(case):
    task_id, entries, expected = PILOT_ENTRIES[case]
    task, world, _ = _answer(task_id)
    assert fidelity(task, world, entries) == pytest.approx(expected)


def test_a_figure_after_a_name_outside_the_answer_is_not_taken_for_the_name_before_it():
    task, world, _ = _answer("control-02")
    entry = "Golden Bantam (log 1, catalogue 4), Lacinato kale (log 1, catalogue 5)"
    assert fidelity(task, world, [entry]) == 0
    assert covered([Item(("Golden Bantam",), ("4", "5"))], [entry]) == [True]


def test_low_medium_and_high_are_the_three_bands_in_order():
    items = [Item(("T-01",), ("A",)), Item(("T-02",), ("B",)), Item(("T-03",), ("C",))]
    assert covered(items, ["T-01 low, T-02 medium, T-03 high"]) == [True, True, True]
    assert covered(items, ["Low: T-01\nMedium band: T-02\nHigh (9 or more): T-03"]) == [True, True, True]
    assert covered(items, ["T-01 high, T-02 low, T-03 medium"]) == [False, False, False]


def test_a_month_written_out_is_the_month_of_the_answer():
    items = [Item(("NR",), ("2026-06", "2"), column="2026-06")]
    assert covered(items, ["June 2026: NR 2"]) == [True]
    assert covered(items, ["Jun: NR 2"]) == [True]
    assert covered(items, ["July 2026: NR 2"]) == [False]


def test_a_heading_carries_over_to_the_lines_beneath_it():
    items = [Item(("Hillside",), ("214",)), Item(("Brookfield",), ("187",))]
    assert covered(items, ["Hillside:\n- trees 214\nBrookfield:\n- trees 187"]) == [True, True]
    assert covered(items, ["Hillside:\n- trees 187\nBrookfield:\n- trees 214"]) == [False, False]
