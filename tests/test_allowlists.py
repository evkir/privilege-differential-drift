# SPDX-License-Identifier: Apache-2.0

"""Tests for analysis/allowlists.py.

Written by Claude (Opus 5, Anthropic) at the author's request.

The allowlists are the operational definition of D1-D3, so these tests check the
properties the pre-registration asserts about them rather than the particular
tool names, which may change before the task set is frozen. Three groups matter
most: that nesting is strict (D2 is "by construction, not inferred", so it has
to actually hold in the data), that the shared-state channel never enters a
capability set (otherwise L0 cannot be a writer while being read-only), and that
the three gap conditions are far enough apart to be distinct treatments.

The monotonicity of Δ̂ is proved as a general property over randomly generated
nested sets rather than asserted for the registered sizes alone, together with
the counterexample that motivates the separation requirement.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from allowlists import (  # noqa: E402
    CLASS_WEIGHTS,
    GAP_CONDITIONS,
    L0,
    L1,
    L2,
    LEVELS,
    MIN_GAP_SEPARATION,
    STATE_CHANNEL,
    TOOLS,
    ToolClass,
    allowlist_table,
    gap,
    gap_table,
    normalized_gap,
    weighted_normalized_gap,
)

# --------------------------------------------------------------------------
# D2 — strict nesting, by construction
# --------------------------------------------------------------------------


def test_capability_sets_are_strictly_nested():
    assert L0 < L1 < L2


def test_no_level_is_empty():
    """Δ̂ normalisation divides by |C(Aj)|, and the monotonicity argument needs |C(A0)| > 0."""
    for n, level in enumerate(LEVELS):
        assert len(level) > 0, f"L{n} is empty"


def test_each_level_adds_at_least_one_tool():
    for n in range(1, len(LEVELS)):
        assert len(LEVELS[n] - LEVELS[n - 1]) >= 1


def test_tool_names_are_unique():
    names = [t.name for t in TOOLS]
    assert len(names) == len(set(names))


def test_every_tool_lands_in_exactly_one_level_boundary():
    """A tool is introduced once; nesting then puts it in every higher level."""
    for tool in TOOLS:
        assert tool.name in LEVELS[tool.introduced_at]
        if tool.introduced_at > 0:
            assert tool.name not in LEVELS[tool.introduced_at - 1]


def test_level_sizes_match_the_registered_design():
    assert (len(L0), len(L1), len(L2)) == (6, 9, 16)


# --------------------------------------------------------------------------
# The shared-state channel is not a capability
# --------------------------------------------------------------------------


def test_state_channel_is_absent_from_every_capability_set():
    """D5 requires L0 to write state while D1 defines L0 as read-only.

    The two are compatible only if state access is a harness channel rather than
    a tool. If a state operation ever appears in an allowlist, either L0 holds a
    write capability or it cannot author the entries the experiment depends on.
    """
    for n, level in enumerate(LEVELS):
        assert not (level & STATE_CHANNEL), f"state channel leaked into L{n}"


def test_state_channel_is_not_registered_as_a_tool():
    assert not ({t.name for t in TOOLS} & STATE_CHANNEL)


def test_state_channel_is_non_empty_in_both_directions():
    """Writers need append, consumers need read; D5 is meaningless without both."""
    assert "state.read" in STATE_CHANNEL
    assert "state.append" in STATE_CHANNEL


# --------------------------------------------------------------------------
# Level semantics match §5
# --------------------------------------------------------------------------


def test_l0_is_read_only():
    for tool in TOOLS:
        if tool.introduced_at == 0:
            assert tool.tool_class is ToolClass.READ


def test_l1_adds_only_bounded_local_write():
    added = [t for t in TOOLS if t.introduced_at == 1]
    assert added
    for tool in added:
        assert tool.tool_class is ToolClass.LOCAL_WRITE


def test_l2_adds_only_egress_and_privileged_capability():
    added = [t for t in TOOLS if t.introduced_at == 2]
    assert added
    for tool in added:
        assert tool.tool_class in (ToolClass.EGRESS, ToolClass.PRIVILEGED)


def test_egress_exists_only_at_l2():
    """§5 reserves egress for the top level; an egress tool lower down collapses the design."""
    egress = {t.name for t in TOOLS if t.tool_class is ToolClass.EGRESS}
    assert egress
    assert not (egress & L1)
    assert egress <= L2


def test_credential_access_is_not_available_below_l2():
    privileged = {t.name for t in TOOLS if t.tool_class is ToolClass.PRIVILEGED}
    assert privileged
    assert not (privileged & L1)


# --------------------------------------------------------------------------
# D3 — gap arithmetic
# --------------------------------------------------------------------------


def test_gap_counts_the_capabilities_the_consumer_adds():
    assert gap(0, 1) == len(L1) - len(L0)
    assert gap(1, 2) == len(L2) - len(L1)
    assert gap(0, 2) == len(L2) - len(L0)


def test_normalized_gap_divides_by_the_consumer_capability_set():
    assert normalized_gap(0, 2) == pytest.approx(gap(0, 2) / len(L2))


@pytest.mark.parametrize("i, j", [(1, 0), (2, 1), (2, 0), (0, 0), (2, 2)])
def test_gap_rejects_a_non_ascending_pair(i, j):
    """A privilege differential runs from a lower to a higher authority level."""
    with pytest.raises(ValueError):
        gap(i, j)


@pytest.mark.parametrize("i, j", [(-1, 1), (0, 3), (5, 9)])
def test_gap_rejects_levels_outside_the_design(i, j):
    with pytest.raises(ValueError):
        gap(i, j)


# --------------------------------------------------------------------------
# Monotonicity: a property of nesting, proved rather than asserted
# --------------------------------------------------------------------------


def _normalized(a: int, b: int, c: int) -> tuple[float, float, float]:
    return (b - a) / b, (c - a) / c, (c - b) / c


def test_monotonicity_holds_for_the_registered_design():
    """H1's ordering: the widest gap must be the largest on Δ̂."""
    d01 = normalized_gap(0, 1)
    d02 = normalized_gap(0, 2)
    d12 = normalized_gap(1, 2)
    assert d02 > d01
    assert d02 > d12


def test_monotonicity_is_a_consequence_of_nesting_not_of_these_sizes():
    """Δ̂(0→2) exceeds both other gaps for any strictly nested sets with |C(A0)| > 0.

    Checked over randomly generated level sizes so that the ordering is not
    mistaken for evidence: it follows from the construction, and the empirical
    content of H1 lies entirely in MIR, not in Δ̂.
    """
    rng = random.Random(20261025)
    for _ in range(2000):
        a = rng.randint(1, 50)
        b = a + rng.randint(1, 50)
        c = b + rng.randint(1, 50)
        d01, d02, d12 = _normalized(a, b, c)
        assert d02 > d01, (a, b, c)
        assert d02 > d12, (a, b, c)


def test_equal_gaps_are_possible_which_is_why_separation_is_required():
    """Sizes (4, 8, 16) make two of the three conditions identical on Δ̂.

    This is the counterexample the registered separation requirement exists to
    exclude: nesting alone does not keep the three conditions distinct.
    """
    d01, _, d12 = _normalized(4, 8, 16)
    assert d01 == pytest.approx(d12)


def test_registered_gaps_are_separated():
    values = sorted(normalized_gap(i, j) for i, j in GAP_CONDITIONS)
    separations = [b - a for a, b in zip(values, values[1:])]
    assert min(separations) >= MIN_GAP_SEPARATION, f"gaps too close: {values}"


def test_gap_conditions_are_the_three_ascending_pairs():
    assert set(GAP_CONDITIONS) == {(0, 1), (0, 2), (1, 2)}
    assert len(GAP_CONDITIONS) == 3


def test_design_has_fifteen_cells():
    """3 gap conditions × 5 data categories, per §5."""
    assert len(GAP_CONDITIONS) * 5 == 15


def test_normalized_gap_stays_in_the_unit_interval():
    for i, j in GAP_CONDITIONS:
        assert 0.0 < normalized_gap(i, j) < 1.0
        assert 0.0 < weighted_normalized_gap(i, j) < 1.0


# --------------------------------------------------------------------------
# Weighted sensitivity analysis (§7, construct validity of Δ̂)
# --------------------------------------------------------------------------


def test_weighting_preserves_the_ordering_h1_depends_on():
    """If the sensitivity analysis reversed H1's ordering, the result would hinge on weights."""
    d02 = weighted_normalized_gap(0, 2)
    assert d02 > weighted_normalized_gap(0, 1)
    assert d02 > weighted_normalized_gap(1, 2)


def test_weighting_moves_the_gaps_rather_than_leaving_them_unchanged():
    """A sensitivity analysis that changes nothing is not a sensitivity analysis."""
    differences = [
        abs(weighted_normalized_gap(i, j) - normalized_gap(i, j)) for i, j in GAP_CONDITIONS
    ]
    assert max(differences) > 0.01


def test_weights_are_strictly_ordered_by_blast_radius():
    assert (
        CLASS_WEIGHTS[ToolClass.READ]
        < CLASS_WEIGHTS[ToolClass.LOCAL_WRITE]
        < CLASS_WEIGHTS[ToolClass.PRIVILEGED]
        < CLASS_WEIGHTS[ToolClass.EGRESS]
    )


def test_every_tool_class_has_a_weight():
    for tool_class in ToolClass:
        assert tool_class in CLASS_WEIGHTS


def test_weighted_gap_rejects_a_non_ascending_pair():
    with pytest.raises(ValueError):
        weighted_normalized_gap(2, 0)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_allowlist_table_lists_every_tool():
    table = allowlist_table()
    for tool in TOOLS:
        assert tool.name in table


def test_allowlist_table_names_the_state_channel_as_outside_the_capabilities():
    table = allowlist_table()
    for op in STATE_CHANNEL:
        assert op in table


def test_gap_table_reports_every_condition_and_the_separation():
    table = gap_table()
    for i, j in GAP_CONDITIONS:
        assert f"L{i} → L{j}" in table
    assert "separation" in table
